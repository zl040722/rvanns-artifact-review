/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexHNSW.h>

#include <omp.h>
#include <cassert>
#include <cctype>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <algorithm>
#include <limits>
#include <memory>
#include <queue>
#include <random>
#include <unordered_set>

#include <sys/stat.h>
#include <sys/types.h>
#include <cstdint>

#include <faiss/FaissHook.h>
#include <faiss/Index2Layer.h>
#include <faiss/IndexCosine.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/ResultHandler.h>
#include <faiss/impl/platform_macros.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/random.h>
#include <faiss/utils/sorting.h>

extern "C" {

/* declare BLAS functions, see http://www.netlib.org/clapack/cblas/ */

int sgemm_(
        const char* transa,
        const char* transb,
        FINTEGER* m,
        FINTEGER* n,
        FINTEGER* k,
        const float* alpha,
        const float* a,
        FINTEGER* lda,
        const float* b,
        FINTEGER* ldb,
        float* beta,
        float* c,
        FINTEGER* ldc);
}

namespace faiss {

using MinimaxHeap = HNSW::MinimaxHeap;
using storage_idx_t = HNSW::storage_idx_t;
using NodeDistFarther = HNSW::NodeDistFarther;

HNSWStats hnsw_stats;

/**************************************************************
 * add / search blocks of descriptors
 **************************************************************/

constexpr size_t kCachelineSize = 64;

size_t round_up_to_cacheline(size_t n) {
    if (n == 0) {
        return kCachelineSize;
    }
    return (n + (kCachelineSize - 1)) & ~(kCachelineSize - 1);
}

namespace {

bool method_equals(const char* lhs, const char* rhs) {
    if (lhs == nullptr || rhs == nullptr) {
        return lhs == rhs;
    }
    while (*lhs != '\0' && *rhs != '\0') {
        if (std::tolower(static_cast<unsigned char>(*lhs)) !=
            std::tolower(static_cast<unsigned char>(*rhs))) {
            return false;
        }
        ++lhs;
        ++rhs;
    }
    return *lhs == '\0' && *rhs == '\0';
}

std::vector<std::vector<storage_idx_t>> build_undirected_level0_graph(
        const HNSW& hnsw,
        idx_t ntotal) {
    std::vector<std::vector<storage_idx_t>> graph(ntotal);
    for (storage_idx_t i = 0; i < ntotal; ++i) {
        size_t begin = 0;
        size_t end = 0;
        hnsw.neighbor_range(i, 0, &begin, &end);
        for (size_t j = begin; j < end; ++j) {
            const storage_idx_t neighbor = hnsw.neighbors[j];
            if (neighbor < 0) {
                break;
            }
            graph[i].push_back(neighbor);
            if (neighbor != i) {
                graph[neighbor].push_back(i);
            }
        }
    }
    for (storage_idx_t i = 0; i < ntotal; ++i) {
        std::vector<storage_idx_t>& neighbors = graph[i];
        std::sort(neighbors.begin(), neighbors.end());
        neighbors.erase(
                std::unique(neighbors.begin(), neighbors.end()),
                neighbors.end());
    }
    return graph;
}

storage_idx_t choose_low_degree_seed(
        const std::vector<std::vector<storage_idx_t>>& graph,
        const std::vector<uint8_t>& component_visited) {
    storage_idx_t best = -1;
    size_t best_degree = std::numeric_limits<size_t>::max();
    for (storage_idx_t i = 0; i < static_cast<storage_idx_t>(graph.size());
         ++i) {
        if (component_visited[i]) {
            continue;
        }
        const size_t degree = graph[i].size();
        if (best < 0 || degree < best_degree) {
            best = i;
            best_degree = degree;
        }
    }
    return best;
}

storage_idx_t choose_pseudo_peripheral_seed(
        const std::vector<std::vector<storage_idx_t>>& graph,
        storage_idx_t start,
        const std::vector<uint8_t>& component_visited) {
    if (start < 0) {
        return start;
    }

    const storage_idx_t n = static_cast<storage_idx_t>(graph.size());
    std::vector<int> distance(n, -1);
    std::queue<storage_idx_t> queue;
    int previous_eccentricity = -1;
    storage_idx_t current = start;

    while (true) {
        std::fill(distance.begin(), distance.end(), -1);
        while (!queue.empty()) {
            queue.pop();
        }

        queue.push(current);
        distance[current] = 0;
        int eccentricity = 0;
        std::vector<storage_idx_t> farthest;

        while (!queue.empty()) {
            const storage_idx_t node = queue.front();
            queue.pop();
            const int node_distance = distance[node];
            if (node_distance > eccentricity) {
                eccentricity = node_distance;
                farthest.clear();
            }
            if (node_distance == eccentricity) {
                farthest.push_back(node);
            }

            const std::vector<storage_idx_t>& neighbors = graph[node];
            for (size_t i = 0; i < neighbors.size(); ++i) {
                const storage_idx_t next = neighbors[i];
                if (component_visited[next] || distance[next] >= 0) {
                    continue;
                }
                distance[next] = node_distance + 1;
                queue.push(next);
            }
        }

        if (eccentricity <= previous_eccentricity || farthest.empty()) {
            return current;
        }

        storage_idx_t next_seed = farthest[0];
        size_t next_degree = graph[next_seed].size();
        for (size_t i = 1; i < farthest.size(); ++i) {
            const storage_idx_t candidate = farthest[i];
            const size_t degree = graph[candidate].size();
            if (degree < next_degree) {
                next_seed = candidate;
                next_degree = degree;
            }
        }

        if (next_seed == current) {
            return current;
        }

        previous_eccentricity = eccentricity;
        current = next_seed;
    }
}

struct RabbitCommunity {
    storage_idx_t parent;
    std::vector<storage_idx_t> members;
    double strength;
    bool active;
};

storage_idx_t rabbit_find_root(
        std::vector<RabbitCommunity>& communities,
        storage_idx_t node) {
    storage_idx_t root = node;
    while (communities[root].parent != root) {
        root = communities[root].parent;
    }
    while (communities[node].parent != node) {
        const storage_idx_t parent = communities[node].parent;
        communities[node].parent = root;
        node = parent;
    }
    return root;
}

void rabbit_emit_members(
        storage_idx_t root,
        const std::vector<RabbitCommunity>& communities,
        std::vector<uint8_t>& emitted,
        std::vector<idx_t>& perm) {
    const std::vector<storage_idx_t>& members = communities[root].members;
    for (size_t i = 0; i < members.size(); ++i) {
        const storage_idx_t member = members[i];
        if (emitted[member]) {
            continue;
        }
        emitted[member] = 1;
        perm.push_back(member);
    }
}

} // anonymous namespace

struct CacheAlignedBuffer {
    uint8_t* data = nullptr;
    size_t size = 0;

    ~CacheAlignedBuffer() {
        reset();
    }

    CacheAlignedBuffer() = default;

    CacheAlignedBuffer(const CacheAlignedBuffer&) = delete;
    CacheAlignedBuffer& operator=(const CacheAlignedBuffer&) = delete;

    CacheAlignedBuffer(CacheAlignedBuffer&& other) noexcept {
        data = other.data;
        size = other.size;
        other.data = nullptr;
        other.size = 0;
    }

    CacheAlignedBuffer& operator=(CacheAlignedBuffer&& other) noexcept {
        if (this == &other) {
            return *this;
        }
        reset();
        data = other.data;
        size = other.size;
        other.data = nullptr;
        other.size = 0;
        return *this;
    }

    void reset() {
        if (data) {
            posix_memalign_free(data);
            data = nullptr;
            size = 0;
        }
    }

    void resize(size_t new_size) {
        if (new_size == size) {
            return;
        }
        reset();
        if (new_size == 0) {
            return;
        }
        void* ptr = nullptr;
        int ret = posix_memalign(&ptr, kCachelineSize, new_size);
        if (ret != 0) {
            throw std::bad_alloc();
        }
        data = reinterpret_cast<uint8_t*>(ptr);
        size = new_size;
    }

    uint8_t* ptr() {
        return data;
    }

    const uint8_t* ptr() const {
        return data;
    }

    void copy_from(const CacheAlignedBuffer& other) {
        if (this == &other) {
            return;
        }
        resize(other.size);
        if (other.size > 0) {
            memcpy(data, other.data, other.size);
        }
    }

    size_t nbytes() const {
        return size;
    }
};

struct IndexHNSW::CacheAlignedCoLayout {
    CacheAlignedBuffer codes;
    CacheAlignedBuffer code_norms;
    size_t count = 0;
    size_t code_row_bytes = 0;
    size_t code_stride = 0;
    size_t code_norm_row_bytes = 0;
    size_t code_norm_stride = 0;
    bool enabled = false;

    CacheAlignedCoLayout() = default;
    CacheAlignedCoLayout(const CacheAlignedCoLayout& other) {
        *this = other;
    }

    CacheAlignedCoLayout(CacheAlignedCoLayout&&) = default;
    CacheAlignedCoLayout& operator=(CacheAlignedCoLayout&&) = default;

    CacheAlignedCoLayout& operator=(const CacheAlignedCoLayout& other) {
        if (this == &other) {
            return *this;
        }
        codes.copy_from(other.codes);
        code_norms.copy_from(other.code_norms);
        count = other.count;
        code_row_bytes = other.code_row_bytes;
        code_stride = other.code_stride;
        code_norm_row_bytes = other.code_norm_row_bytes;
        code_norm_stride = other.code_norm_stride;
        enabled = other.enabled;
        return *this;
    }

    void reset() {
        codes.reset();
        code_norms.reset();
        count = 0;
        code_row_bytes = 0;
        code_stride = 0;
        code_norm_row_bytes = 0;
        code_norm_stride = 0;
        enabled = false;
    }
};

class AlignedFlatDistanceComputer : public DistanceComputer {
    MetricType metric_type_;
    size_t d_;
    size_t stride_;
    size_t row_bytes_;
    const uint8_t* base_;
    const float* q_ = nullptr;

   public:
    AlignedFlatDistanceComputer(
            MetricType metric_type,
            size_t d,
            size_t stride,
            size_t row_bytes,
            const uint8_t* base)
            : metric_type_(metric_type),
              d_(d),
              stride_(stride),
              row_bytes_(row_bytes),
              base_(base) {
        FAISS_ASSERT(base_);
        FAISS_ASSERT(stride_ >= row_bytes_);
    }

    void set_query(const float* x) override {
        q_ = x;
    }

    const float* ptr(idx_t i) const {
        return reinterpret_cast<const float*>(base_ + stride_ * i);
    }

    float operator()(idx_t i) override {
        const float* xb = ptr(i);
        switch (metric_type_) {
            case METRIC_L2:
                return fvec_L2sqr(q_, xb, d_);
            case METRIC_INNER_PRODUCT:
                return fvec_inner_product(q_, xb, d_);
            default:
                FAISS_THROW_IF_NOT_MSG(
                        false,
                        "AlignedFlatDistanceComputer only supports L2/IP");
        }
    }

    void distances_batch_4(
            const idx_t idx0,
            const idx_t idx1,
            const idx_t idx2,
            const idx_t idx3,
            float& dis0,
            float& dis1,
            float& dis2,
            float& dis3) override {
        const float* y0 = ptr(idx0);
        const float* y1 = ptr(idx1);
        const float* y2 = ptr(idx2);
        const float* y3 = ptr(idx3);

        switch (metric_type_) {
            case METRIC_L2:
                fvec_L2sqr_batch_4(
                        q_, y0, y1, y2, y3, d_, dis0, dis1, dis2, dis3);
                return;
            case METRIC_INNER_PRODUCT:
                dis0 = fvec_inner_product(q_, y0, d_);
                dis1 = fvec_inner_product(q_, y1, d_);
                dis2 = fvec_inner_product(q_, y2, d_);
                dis3 = fvec_inner_product(q_, y3, d_);
                return;
            default:
                FAISS_THROW_IF_NOT_MSG(
                        false,
                        "AlignedFlatDistanceComputer only supports L2/IP");
        }
    }

    float symmetric_dis(idx_t i, idx_t j) override {
        const float* xi = ptr(i);
        const float* xj = ptr(j);
        switch (metric_type_) {
            case METRIC_L2:
                return fvec_L2sqr(xi, xj, d_);
            case METRIC_INNER_PRODUCT:
                return fvec_inner_product(xi, xj, d_);
            default:
                FAISS_THROW_IF_NOT_MSG(
                        false,
                        "AlignedFlatDistanceComputer only supports L2/IP");
        }
    }
};

namespace {

DistanceComputer* storage_distance_computer(const Index* storage) {
    if (is_similarity_metric(storage->metric_type)) {
        return new NegativeDistanceComputer(storage->get_distance_computer());
    } else {
        return storage->get_distance_computer();
    }
}

void hnsw_add_vertices(
        IndexHNSW& index_hnsw,
        size_t n0,
        size_t n,
        const float* x,
        bool verbose,
        bool preset_levels = false) {
    size_t d = index_hnsw.d;
    HNSW& hnsw = index_hnsw.hnsw;
    size_t ntotal = n0 + n;
    double t0 = getmillisecs();
    if (verbose) {
        printf("hnsw_add_vertices: adding %zd elements on top of %zd "
               "(preset_levels=%d)\n",
               n,
               n0,
               int(preset_levels));
    }

    if (n == 0) {
        return;
    }

    int max_level = hnsw.prepare_level_tab(n, preset_levels);

    if (verbose) {
        printf("  max_level = %d\n", max_level);
    }

    std::vector<omp_lock_t> locks(ntotal);
    for (int i = 0; i < ntotal; i++)
        omp_init_lock(&locks[i]);

    // add vectors from highest to lowest level
    std::vector<int> hist;
    std::vector<int> order(n);

    { // make buckets with vectors of the same level

        // build histogram
        for (int i = 0; i < n; i++) {
            storage_idx_t pt_id = i + n0;
            int pt_level = hnsw.levels[pt_id] - 1;
            while (pt_level >= hist.size())
                hist.push_back(0);
            hist[pt_level]++;
        }

        // accumulate
        std::vector<int> offsets(hist.size() + 1, 0);
        for (int i = 0; i < hist.size() - 1; i++) {
            offsets[i + 1] = offsets[i] + hist[i];
        }

        // bucket sort
        for (int i = 0; i < n; i++) {
            storage_idx_t pt_id = i + n0;
            int pt_level = hnsw.levels[pt_id] - 1;
            order[offsets[pt_level]++] = pt_id;
        }
    }

    idx_t check_period = InterruptCallback::get_period_hint(
            max_level * index_hnsw.d * hnsw.efConstruction);

    { // perform add
        RandomGenerator rng2(789);

        int i1 = n;

        for (int pt_level = hist.size() - 1;
             pt_level >= !index_hnsw.init_level0;
             pt_level--) {
            int i0 = i1 - hist[pt_level];

            if (verbose) {
                printf("Adding %d elements at level %d\n", i1 - i0, pt_level);
            }

            // random permutation to get rid of dataset order bias
            for (int j = i0; j < i1; j++)
                std::swap(order[j], order[j + rng2.rand_int(i1 - j)]);

            bool interrupt = false;

#pragma omp parallel if (i1 > i0 + 100)
            {
                VisitedTable vt(ntotal);

                std::unique_ptr<DistanceComputer> dis(
                        index_hnsw.make_distance_computer());
                int prev_display =
                        verbose && omp_get_thread_num() == 0 ? 0 : -1;
                size_t counter = 0;

                // here we should do schedule(dynamic) but this segfaults for
                // some versions of LLVM. The performance impact should not be
                // too large when (i1 - i0) / num_threads >> 1
#pragma omp for schedule(static)
                for (int i = i0; i < i1; i++) {
                    storage_idx_t pt_id = order[i];
                    dis->set_query(x + (pt_id - n0) * d);

                    // cannot break
                    if (interrupt) {
                        continue;
                    }

                    hnsw.add_with_locks(
                            *dis,
                            pt_level,
                            pt_id,
                            locks,
                            vt,
                            index_hnsw.keep_max_size_level0 && (pt_level == 0));

                    if (prev_display >= 0 && i - i0 > prev_display + 10000) {
                        prev_display = i - i0;
                        printf("  %d / %d\r", i - i0, i1 - i0);
                        fflush(stdout);
                    }
                    if (counter % check_period == 0) {
                        if (InterruptCallback::is_interrupted()) {
                            interrupt = true;
                        }
                    }
                    counter++;
                }
            }
            if (interrupt) {
                FAISS_THROW_MSG("computation interrupted");
            }
            i1 = i0;
        }
        if (index_hnsw.init_level0) {
            FAISS_ASSERT(i1 == 0);
        } else {
            FAISS_ASSERT((i1 - hist[0]) == 0);
        }
    }
    if (verbose) {
        printf("Done in %.3f ms\n", getmillisecs() - t0);
    }

    for (int i = 0; i < ntotal; i++) {
        omp_destroy_lock(&locks[i]);
    }
}

} // namespace

/**************************************************************
 * IndexHNSW implementation
 **************************************************************/

IndexHNSW::IndexHNSW(int d, int M, MetricType metric)
        : Index(d, metric), hnsw(M) {
    reorder_perm.clear(); // 确保初始化为空
}

IndexHNSW::IndexHNSW(Index* storage, int M)
        : Index(storage->d, storage->metric_type), hnsw(M), storage(storage) {
    reorder_perm.clear(); // 确保初始化为空
}

IndexHNSW::IndexHNSW(const IndexHNSW& other)
        : Index(other),
          hnsw(other.hnsw),
          own_fields(other.own_fields),
          storage(other.storage),
          init_level0(other.init_level0),
          keep_max_size_level0(other.keep_max_size_level0),
          reorder_perm(other.reorder_perm) {
    if (other.cache_aligned_soa_) {
        cache_aligned_soa_ = std::make_unique<CacheAlignedCoLayout>(
                *other.cache_aligned_soa_);
    }
    pending_cacheline_layout_build_ = other.pending_cacheline_layout_build_;
}

IndexHNSW& IndexHNSW::operator=(const IndexHNSW& other) {
    if (this == &other) {
        return *this;
    }
    Index::operator=(other);
    hnsw = other.hnsw;
    own_fields = other.own_fields;
    storage = other.storage;
    init_level0 = other.init_level0;
    keep_max_size_level0 = other.keep_max_size_level0;
    reorder_perm = other.reorder_perm;
    if (other.cache_aligned_soa_) {
        cache_aligned_soa_ = std::make_unique<CacheAlignedCoLayout>(
                *other.cache_aligned_soa_);
    } else {
        cache_aligned_soa_.reset();
    }
    pending_cacheline_layout_build_ = other.pending_cacheline_layout_build_;
    return *this;
}

IndexHNSW::~IndexHNSW() {
    if (own_fields) {
        delete storage;
    }
}

void IndexHNSW::train(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(
            storage,
            "Please use IndexHNSWFlat (or variants) instead of IndexHNSW directly");
    // hnsw structure does not require training
    storage->train(n, x);
    is_trained = true;
}

namespace {

template <class BlockResultHandler>
void hnsw_search(
        const IndexHNSW* index,
        idx_t n,
        const float* x,
        BlockResultHandler& bres,
        const SearchParameters* params_in) {
    FAISS_THROW_IF_NOT_MSG(
            index->storage,
            "No storage index, please use IndexHNSWFlat (or variants) "
            "instead of IndexHNSW directly");
    const SearchParametersHNSW* params = nullptr;
    const HNSW& hnsw = index->hnsw;

    int efSearch = hnsw.efSearch;
    if (params_in) {
        params = dynamic_cast<const SearchParametersHNSW*>(params_in);
        FAISS_THROW_IF_NOT_MSG(params, "params type invalid");
        efSearch = params->efSearch;
    }
    size_t n1 = 0, n2 = 0, ndis = 0, nhops = 0;

    idx_t check_period = InterruptCallback::get_period_hint(
            hnsw.max_level * index->d * efSearch);

    for (idx_t i0 = 0; i0 < n; i0 += check_period) {
        idx_t i1 = std::min(i0 + check_period, n);

#pragma omp parallel if (i1 - i0 > 1)
        {
            VisitedTable vt(index->ntotal);
            typename BlockResultHandler::SingleResultHandler res(bres);

            std::unique_ptr<DistanceComputer> dis(
                    index->make_distance_computer());

#pragma omp for reduction(+ : n1, n2, ndis, nhops) schedule(guided)
            for (idx_t i = i0; i < i1; i++) {
                res.begin(i);
                dis->set_query(x + i * index->d);

                HNSWStats stats = hnsw.search(*dis, res, vt, params);
                n1 += stats.n1;
                n2 += stats.n2;
                ndis += stats.ndis;
                nhops += stats.nhops;
                res.end();
            }
        }
        InterruptCallback::check();
    }

    hnsw_stats.combine({n1, n2, ndis, nhops});
}

} // anonymous namespace

void IndexHNSW::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params_in) const {
    FAISS_THROW_IF_NOT(k > 0);

    using RH = HeapBlockResultHandler<HNSW::C>;
    RH bres(n, distances, labels, k);

    hnsw_search(this, n, x, bres, params_in);

    if (is_similarity_metric(this->metric_type)) {
        // we need to revert the negated distances
        for (size_t i = 0; i < k * n; i++) {
            distances[i] = -distances[i];
        }
    }
}

void IndexHNSW::range_search(
        idx_t n,
        const float* x,
        float radius,
        RangeSearchResult* result,
        const SearchParameters* params) const {
    using RH = RangeSearchBlockResultHandler<HNSW::C>;
    RH bres(result, radius);

    hnsw_search(this, n, x, bres, params);

    if (is_similarity_metric(this->metric_type)) {
        // we need to revert the negated distances
        for (size_t i = 0; i < result->lims[result->nq]; i++) {
            result->distances[i] = -result->distances[i];
        }
    }
}

void IndexHNSW::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(
            storage,
            "Please use IndexHNSWFlat (or variants) instead of IndexHNSW directly");
    FAISS_THROW_IF_NOT(is_trained);
    FAISS_THROW_IF_NOT_MSG(
            reorder_perm.empty(),
            "Cannot add vectors after a post-build HNSW layout pass; reset or rebuild the index first");
    int n0 = ntotal;
    storage->add(n, x);
    ntotal = storage->ntotal;

    hnsw_add_vertices(*this, n0, n, x, verbose, hnsw.levels.size() == ntotal);
}

void IndexHNSW::reset() {
    hnsw.reset();
    storage->reset();
    ntotal = 0;
    reorder_perm.clear(); // 清除重排序映射
}

void IndexHNSW::reconstruct(idx_t key, float* recons) const {
    storage->reconstruct(key, recons);
}

void IndexHNSW::shrink_level_0_neighbors(int new_size) {
#pragma omp parallel
    {
        std::unique_ptr<DistanceComputer> dis(make_distance_computer());

#pragma omp for
        for (idx_t i = 0; i < ntotal; i++) {
            size_t begin, end;
            hnsw.neighbor_range(i, 0, &begin, &end);

            std::priority_queue<NodeDistFarther> initial_list;

            for (size_t j = begin; j < end; j++) {
                int v1 = hnsw.neighbors[j];
                if (v1 < 0)
                    break;
                initial_list.emplace(dis->symmetric_dis(i, v1), v1);

                // initial_list.emplace(qdis(v1), v1);
            }

            std::vector<NodeDistFarther> shrunk_list;
            HNSW::shrink_neighbor_list(
                    *dis, initial_list, shrunk_list, new_size);

            for (size_t j = begin; j < end; j++) {
                if (j - begin < shrunk_list.size())
                    hnsw.neighbors[j] = shrunk_list[j - begin].id;
                else
                    hnsw.neighbors[j] = -1;
            }
        }
    }
}

void IndexHNSW::search_level_0(
        idx_t n,
        const float* x,
        idx_t k,
        const storage_idx_t* nearest,
        const float* nearest_d,
        float* distances,
        idx_t* labels,
        int nprobe,
        int search_type,
        const SearchParameters* params_in) const {
    FAISS_THROW_IF_NOT(k > 0);
    FAISS_THROW_IF_NOT(nprobe > 0);

    const SearchParametersHNSW* params = nullptr;

    if (params_in) {
        params = dynamic_cast<const SearchParametersHNSW*>(params_in);
        FAISS_THROW_IF_NOT_MSG(params, "params type invalid");
    }

    storage_idx_t ntotal = hnsw.levels.size();

    using RH = HeapBlockResultHandler<HNSW::C>;
    RH bres(n, distances, labels, k);

#pragma omp parallel
    {
        std::unique_ptr<DistanceComputer> qdis(make_distance_computer());
        HNSWStats search_stats;
        VisitedTable vt(ntotal);
        RH::SingleResultHandler res(bres);

#pragma omp for
        for (idx_t i = 0; i < n; i++) {
            res.begin(i);
            qdis->set_query(x + i * d);

            hnsw.search_level_0(
                    *qdis.get(),
                    res,
                    nprobe,
                    nearest + i * nprobe,
                    nearest_d + i * nprobe,
                    search_type,
                    search_stats,
                    vt,
                    params);
            res.end();
            vt.advance();
        }
#pragma omp critical
        { hnsw_stats.combine(search_stats); }
    }
    if (is_similarity_metric(this->metric_type)) {
// we need to revert the negated distances
#pragma omp parallel for
        for (int64_t i = 0; i < k * n; i++) {
            distances[i] = -distances[i];
        }
    }
}

void IndexHNSW::init_level_0_from_knngraph(
        int k,
        const float* D,
        const idx_t* I) {
    int dest_size = hnsw.nb_neighbors(0);

#pragma omp parallel for
    for (idx_t i = 0; i < ntotal; i++) {
        DistanceComputer* qdis = make_distance_computer();
        std::vector<float> vec(d);
        storage->reconstruct(i, vec.data());
        qdis->set_query(vec.data());

        std::priority_queue<NodeDistFarther> initial_list;

        for (size_t j = 0; j < k; j++) {
            int v1 = I[i * k + j];
            if (v1 == i)
                continue;
            if (v1 < 0)
                break;
            initial_list.emplace(D[i * k + j], v1);
        }

        std::vector<NodeDistFarther> shrunk_list;
        HNSW::shrink_neighbor_list(*qdis, initial_list, shrunk_list, dest_size);

        size_t begin, end;
        hnsw.neighbor_range(i, 0, &begin, &end);

        for (size_t j = begin; j < end; j++) {
            if (j - begin < shrunk_list.size())
                hnsw.neighbors[j] = shrunk_list[j - begin].id;
            else
                hnsw.neighbors[j] = -1;
        }
    }
}

void IndexHNSW::init_level_0_from_entry_points(
        int n,
        const storage_idx_t* points,
        const storage_idx_t* nearests) {
    std::vector<omp_lock_t> locks(ntotal);
    for (int i = 0; i < ntotal; i++)
        omp_init_lock(&locks[i]);

#pragma omp parallel
    {
        VisitedTable vt(ntotal);

        std::unique_ptr<DistanceComputer> dis(make_distance_computer());
        std::vector<float> vec(storage->d);

#pragma omp for schedule(dynamic)
        for (int i = 0; i < n; i++) {
            storage_idx_t pt_id = points[i];
            storage_idx_t nearest = nearests[i];
            storage->reconstruct(pt_id, vec.data());
            dis->set_query(vec.data());

            hnsw.add_links_starting_from(
                    *dis, pt_id, nearest, (*dis)(nearest), 0, locks.data(), vt);

            if (verbose && i % 10000 == 0) {
                printf("  %d / %d\r", i, n);
                fflush(stdout);
            }
        }
    }
    if (verbose) {
        printf("\n");
    }

    for (int i = 0; i < ntotal; i++)
        omp_destroy_lock(&locks[i]);
}

void IndexHNSW::reorder_links() {
    int M = hnsw.nb_neighbors(0);

#pragma omp parallel
    {
        std::vector<float> distances(M);
        std::vector<size_t> order(M);
        std::vector<storage_idx_t> tmp(M);
        std::unique_ptr<DistanceComputer> dis(make_distance_computer());

#pragma omp for
        for (storage_idx_t i = 0; i < ntotal; i++) {
            size_t begin, end;
            hnsw.neighbor_range(i, 0, &begin, &end);

            for (size_t j = begin; j < end; j++) {
                storage_idx_t nj = hnsw.neighbors[j];
                if (nj < 0) {
                    end = j;
                    break;
                }
                distances[j - begin] = dis->symmetric_dis(i, nj);
                tmp[j - begin] = nj;
            }

            fvec_argsort(end - begin, distances.data(), order.data());
            for (size_t j = begin; j < end; j++) {
                hnsw.neighbors[j] = tmp[order[j - begin]];
            }
        }
    }
}

void IndexHNSW::optimize_neighbors_by_id() {
    if (verbose) {
        printf("开始优化neighbors顺序（并行版本）...\n");
    }
    int M = hnsw.nb_neighbors(0);
    const storage_idx_t ntotal_local = ntotal;

#pragma omp parallel
    {
        // 每个线程有自己的临时数组，避免竞争
        std::vector<storage_idx_t> tmp(M);

        // 并行处理每个节点
        // 不同节点的邻居范围互不重叠，可以安全并行
#pragma omp for schedule(static)
        for (storage_idx_t i = 0; i < ntotal_local; i++) {
            size_t begin, end;
            hnsw.neighbor_range(i, 0, &begin, &end);

            // 收集有效的neighbors到临时数组
            int valid_count = 0;
            for (size_t j = begin; j < end; j++) {
                storage_idx_t nj = hnsw.neighbors[j];
                if (nj < 0) {
                    break;
                }
                tmp[valid_count++] = nj;
            }

            // 对邻居ID排序
            std::sort(tmp.begin(), tmp.begin() + valid_count);

            // 写回排序后的neighbors
            // 安全：每个线程只写入自己节点的范围 [begin, end)
            // 不同线程处理不同节点，范围不重叠，无竞争条件
            for (size_t j = begin; j < end; j++) {
                if (j - begin < valid_count) {
                    hnsw.neighbors[j] = tmp[j - begin];
                } else {
                    hnsw.neighbors[j] = -1;
                }
            }
        }
    }

    if (verbose) {
        printf("优化neighbors顺序完成（并行版本）\n");
    }
}

void IndexHNSW::link_singletons() {
    printf("search for singletons\n");

    std::vector<bool> seen(ntotal);

    for (size_t i = 0; i < ntotal; i++) {
        size_t begin, end;
        hnsw.neighbor_range(i, 0, &begin, &end);
        for (size_t j = begin; j < end; j++) {
            storage_idx_t ni = hnsw.neighbors[j];
            if (ni >= 0)
                seen[ni] = true;
        }
    }

    int n_sing = 0, n_sing_l1 = 0;
    std::vector<storage_idx_t> singletons;
    for (storage_idx_t i = 0; i < ntotal; i++) {
        if (!seen[i]) {
            singletons.push_back(i);
            n_sing++;
            if (hnsw.levels[i] > 1)
                n_sing_l1++;
        }
    }

    printf("  Found %d / %" PRId64 " singletons (%d appear in a level above)\n",
           n_sing,
           ntotal,
           n_sing_l1);

    std::vector<float> recons(singletons.size() * d);
    for (int i = 0; i < singletons.size(); i++) {
        FAISS_ASSERT(!"not implemented");
    }
}

void IndexHNSW::permute_entries(const idx_t* perm) {
    auto flat_storage = dynamic_cast<IndexFlatCodes*>(storage);
    FAISS_THROW_IF_NOT_MSG(
            flat_storage, "don't know how to permute this index");
    invalidate_cacheline_layout();

    auto reorder_node_metadata = [&](std::vector<float>& values) {
        if (values.size() != (size_t)ntotal)
            return;
        std::vector<float> reordered(ntotal);
        for (idx_t i = 0; i < ntotal; i++) {
            reordered[i] = values[perm[i]];
        }
        values.swap(reordered);
    };

    reorder_node_metadata(flat_storage->code_norms);
    flat_storage->permute_entries(perm);
    hnsw.permute_entries(perm);

    // Check for Cosine variants and reorder their inverse_l2_norms
    if (auto* s = dynamic_cast<IndexFlatCosine*>(storage)) {
        reorder_node_metadata(s->inverse_norms_storage.inverse_l2_norms);
    } else if (auto* s = dynamic_cast<IndexScalarQuantizerCosine*>(storage)) {
        reorder_node_metadata(s->inverse_norms_storage.inverse_l2_norms);
    } else if (auto* s = dynamic_cast<IndexPQCosine*>(storage)) {
        reorder_node_metadata(s->inverse_norms_storage.inverse_l2_norms);
    } else if (
            auto* s = dynamic_cast<IndexProductResidualQuantizerCosine*>(
                    storage)) {
        reorder_node_metadata(s->inverse_norms_storage.inverse_l2_norms);
    }

    if (pending_cacheline_layout_build_) {
        apply_cacheline_co_layout_to_storage();
    }
}

void IndexHNSW::invalidate_cacheline_layout() {
    if (cache_aligned_soa_) {
        cache_aligned_soa_->reset();
    }
}

void IndexHNSW::apply_cacheline_co_layout_to_storage() {
    pending_cacheline_layout_build_ = false;
    auto* flat_storage = dynamic_cast<IndexFlatCodes*>(storage);
    if (!flat_storage || ntotal == 0 || flat_storage->code_size == 0) {
        return;
    }

    if (!cache_aligned_soa_) {
        cache_aligned_soa_ = std::make_unique<CacheAlignedCoLayout>();
    } else {
        cache_aligned_soa_->reset();
    }

    CacheAlignedCoLayout& layout = *cache_aligned_soa_;
    layout.count = ntotal;
    layout.code_row_bytes = flat_storage->code_size;
    layout.code_stride = round_up_to_cacheline(layout.code_row_bytes);
    layout.codes.resize(ntotal * layout.code_stride);

    const uint8_t* src_codes = flat_storage->codes.data();
    uint8_t* dst_codes = layout.codes.ptr();
    for (idx_t i = 0; i < ntotal; i++) {
        uint8_t* row_dst = dst_codes + layout.code_stride * i;
        memcpy(row_dst,
               src_codes + layout.code_row_bytes * i,
               layout.code_row_bytes);
        if (layout.code_stride > layout.code_row_bytes) {
            memset(row_dst + layout.code_row_bytes,
                   0,
                   layout.code_stride - layout.code_row_bytes);
        }
    }

    if (!flat_storage->code_norms.empty()) {
        layout.code_norm_row_bytes = sizeof(float);
        layout.code_norm_stride =
                round_up_to_cacheline(layout.code_norm_row_bytes);
        layout.code_norms.resize(ntotal * layout.code_norm_stride);

        const uint8_t* src_norms = reinterpret_cast<const uint8_t*>(
                flat_storage->code_norms.data());
        uint8_t* dst_norms = layout.code_norms.ptr();
        for (idx_t i = 0; i < ntotal; i++) {
            uint8_t* row_dst = dst_norms + layout.code_norm_stride * i;
            memcpy(row_dst,
                   src_norms + layout.code_norm_row_bytes * i,
                   layout.code_norm_row_bytes);
            if (layout.code_norm_stride > layout.code_norm_row_bytes) {
                memset(row_dst + layout.code_norm_row_bytes,
                       0,
                       layout.code_norm_stride - layout.code_norm_row_bytes);
            }
        }
    }

    layout.enabled = true;
}

IndexHNSW::CacheAlignedLayoutInfo IndexHNSW::get_cache_aligned_codes() const {
    CacheAlignedLayoutInfo info;
    if (cache_aligned_soa_ && cache_aligned_soa_->enabled &&
        cache_aligned_soa_->code_stride != 0) {
        info.enabled = true;
        info.count = cache_aligned_soa_->count;
        info.stride = cache_aligned_soa_->code_stride;
        info.row_bytes = cache_aligned_soa_->code_row_bytes;
        info.data = cache_aligned_soa_->codes.ptr();
    }
    return info;
}

IndexHNSW::CacheAlignedLayoutInfo IndexHNSW::get_cache_aligned_code_norms()
        const {
    CacheAlignedLayoutInfo info;
    if (cache_aligned_soa_ && cache_aligned_soa_->enabled &&
        cache_aligned_soa_->code_norm_stride != 0) {
        info.enabled = true;
        info.count = cache_aligned_soa_->count;
        info.stride = cache_aligned_soa_->code_norm_stride;
        info.row_bytes = cache_aligned_soa_->code_norm_row_bytes;
        info.data = cache_aligned_soa_->code_norms.ptr();
    }
    return info;
}
DistanceComputer* IndexHNSW::get_distance_computer() const {
    return storage->get_distance_computer();
}

DistanceComputer* IndexHNSW::make_distance_computer() const {
    if (cache_aligned_soa_ && cache_aligned_soa_->enabled && storage) {
        auto* flat_storage = dynamic_cast<IndexFlat*>(storage);
        if (flat_storage &&
            (flat_storage->metric_type == METRIC_L2 ||
             flat_storage->metric_type == METRIC_INNER_PRODUCT)) {
            auto info = get_cache_aligned_codes();
            if (info.enabled && info.data) {
                DistanceComputer* dc = new AlignedFlatDistanceComputer(
                        flat_storage->metric_type,
                        flat_storage->d,
                        info.stride,
                        info.row_bytes,
                        info.data);
                if (is_similarity_metric(flat_storage->metric_type)) {
                    return new NegativeDistanceComputer(dc);
                }
                return dc;
            }
        }
    }
    return storage_distance_computer(storage);
}

void IndexHNSW::reorder_graph_after_build(
        int level,
        const char* method,
        bool freeze) {
    FAISS_THROW_IF_NOT_MSG(
            reorder_perm.empty(),
            "The post-build HNSW layout pass may be applied only once; reset or rebuild the index first");
    if (ntotal == 0 || hnsw.entry_point == -1) {
        return;
    }
    FAISS_THROW_IF_NOT_MSG(level == 0, "Only level-0 reordering is supported");

    const idx_t N = ntotal;

    std::vector<idx_t> perm;
    perm.reserve(N);

    if (method_equals(method, "rorder")) {
        // ROrder uses the same static topology-derived global permutation as
        // GOrder, then additionally normalizes level-0 adjacency order.
        perm = generateGorderPermutation();
        this->permute_entries(perm.data());
        reorder_perm = perm;

        if (verbose) {
            printf("ROrder global permutation complete; normalizing level-0 adjacency order...\n");
        }
        optimize_neighbors_by_id();
    } else if (method_equals(method, "gorder")) {
        perm = generateGorderPermutation();
        this->permute_entries(perm.data());
        reorder_perm = perm;
    } else if (method_equals(method, "rcm")) {
        perm = generateRCMPermutation();
        this->permute_entries(perm.data());
        reorder_perm = perm;
    } else if (method_equals(method, "rabbitorder")) {
        perm = generateRabbitOrderPermutation();
        this->permute_entries(perm.data());
        reorder_perm = perm;
    } else if (method_equals(method, "bfs")) {
        perm = generateBFSPermutation(level);
        this->permute_entries(perm.data());
        reorder_perm = perm;
    } else {
        FAISS_THROW_FMT(
                "Unknown HNSW layout method: %s",
                method ? method : "(null)");
    }

    (void)freeze;
}

std::vector<idx_t> IndexHNSW::generateBFSPermutation(int level) const {
    const idx_t n = ntotal;
    std::vector<uint8_t> visited(n, 0);
    std::vector<idx_t> perm;
    perm.reserve(n);

    auto bfs_traversal = [&](storage_idx_t start) {
        if (visited[start]) {
            return;
        }

        std::queue<storage_idx_t> queue;
        queue.push(start);
        visited[start] = 1;
        while (!queue.empty()) {
            const storage_idx_t node = queue.front();
            queue.pop();
            perm.push_back(node);

            size_t begin = 0;
            size_t end = 0;
            hnsw.neighbor_range(node, level, &begin, &end);
            for (size_t j = begin; j < end; ++j) {
                const storage_idx_t neighbor = hnsw.neighbors[j];
                if (neighbor < 0) {
                    break;
                }
                if (!visited[neighbor]) {
                    visited[neighbor] = 1;
                    queue.push(neighbor);
                }
            }
        }
    };

    bfs_traversal(hnsw.entry_point);
    for (storage_idx_t i = 0; i < n; ++i) {
        if (!visited[i]) {
            bfs_traversal(i);
        }
    }

    return perm;
}

std::vector<idx_t> IndexHNSW::generateRCMPermutation() const {
    const idx_t n = ntotal;
    if (n == 0) {
        return {};
    }

    const std::vector<std::vector<storage_idx_t>> graph =
            build_undirected_level0_graph(hnsw, n);
    std::vector<uint8_t> component_visited(n, 0);
    std::vector<idx_t> perm;
    perm.reserve(n);

    std::vector<storage_idx_t> neighbor_buffer;
    std::vector<storage_idx_t> component_order;
    std::queue<storage_idx_t> queue;

    while (perm.size() < static_cast<size_t>(n)) {
        storage_idx_t seed = choose_low_degree_seed(graph, component_visited);
        if (seed < 0) {
            break;
        }
        seed = choose_pseudo_peripheral_seed(graph, seed, component_visited);

        while (!queue.empty()) {
            queue.pop();
        }
        queue.push(seed);
        component_visited[seed] = 1;
        component_order.clear();

        while (!queue.empty()) {
            const storage_idx_t node = queue.front();
            queue.pop();
            component_order.push_back(node);

            neighbor_buffer.clear();
            const std::vector<storage_idx_t>& neighbors = graph[node];
            for (size_t i = 0; i < neighbors.size(); ++i) {
                const storage_idx_t neighbor = neighbors[i];
                if (!component_visited[neighbor]) {
                    neighbor_buffer.push_back(neighbor);
                }
            }

            std::sort(
                    neighbor_buffer.begin(),
                    neighbor_buffer.end(),
                    [&](storage_idx_t lhs, storage_idx_t rhs) {
                        const size_t lhs_degree = graph[lhs].size();
                        const size_t rhs_degree = graph[rhs].size();
                        if (lhs_degree != rhs_degree) {
                            return lhs_degree < rhs_degree;
                        }
                        return lhs < rhs;
                    });

            for (size_t i = 0; i < neighbor_buffer.size(); ++i) {
                const storage_idx_t neighbor = neighbor_buffer[i];
                if (!component_visited[neighbor]) {
                    component_visited[neighbor] = 1;
                    queue.push(neighbor);
                }
            }
        }

        for (std::vector<storage_idx_t>::reverse_iterator it =
                     component_order.rbegin();
             it != component_order.rend();
             ++it) {
            perm.push_back(*it);
        }
    }

    return perm;
}

std::vector<idx_t> IndexHNSW::generateRabbitOrderPermutation() const {
    const idx_t n = ntotal;
    if (n == 0) {
        return {};
    }

    const std::vector<std::vector<storage_idx_t>> graph =
            build_undirected_level0_graph(hnsw, n);
    std::vector<storage_idx_t> order(n);
    for (storage_idx_t i = 0; i < n; ++i) {
        order[i] = i;
    }
    std::sort(
            order.begin(),
            order.end(),
            [&](storage_idx_t lhs, storage_idx_t rhs) {
                const size_t lhs_degree = graph[lhs].size();
                const size_t rhs_degree = graph[rhs].size();
                if (lhs_degree != rhs_degree) {
                    return lhs_degree < rhs_degree;
                }
                return lhs < rhs;
            });

    std::vector<RabbitCommunity> communities(n);
    double total_edge_weight = 0.0;
    for (storage_idx_t i = 0; i < n; ++i) {
        communities[i].parent = i;
        communities[i].members.push_back(i);
        communities[i].strength = static_cast<double>(graph[i].size());
        communities[i].active = true;
        total_edge_weight += communities[i].strength;
    }
    if (total_edge_weight <= 0.0) {
        total_edge_weight = 1.0;
    }

    std::vector<double> edge_weights(n, 0.0);
    std::vector<storage_idx_t> touched_roots;
    std::vector<uint8_t> root_processed(n, 0);

    for (size_t order_index = 0; order_index < order.size(); ++order_index) {
        const storage_idx_t node = order[order_index];
        const storage_idx_t source_root = rabbit_find_root(communities, node);
        if (!communities[source_root].active || root_processed[source_root]) {
            continue;
        }
        root_processed[source_root] = 1;

        touched_roots.clear();
        const std::vector<storage_idx_t>& members =
                communities[source_root].members;
        for (size_t i = 0; i < members.size(); ++i) {
            const storage_idx_t member = members[i];
            const std::vector<storage_idx_t>& neighbors = graph[member];
            for (size_t j = 0; j < neighbors.size(); ++j) {
                const storage_idx_t neighbor = neighbors[j];
                const storage_idx_t neighbor_root =
                        rabbit_find_root(communities, neighbor);
                if (neighbor_root == source_root ||
                    !communities[neighbor_root].active) {
                    continue;
                }
                if (edge_weights[neighbor_root] == 0.0) {
                    touched_roots.push_back(neighbor_root);
                }
                edge_weights[neighbor_root] += 1.0;
            }
        }

        storage_idx_t best_root = -1;
        double best_score = 0.0;
        for (size_t i = 0; i < touched_roots.size(); ++i) {
            const storage_idx_t target_root = touched_roots[i];
            const double edge_weight = edge_weights[target_root];
            const double score = edge_weight -
                    (communities[source_root].strength *
                     communities[target_root].strength) /
                            total_edge_weight;
            if (score > best_score ||
                (score == best_score && best_root >= 0 &&
                 target_root < best_root)) {
                best_score = score;
                best_root = target_root;
            }
        }

        if (best_root >= 0 && best_score > 0.0) {
            communities[source_root].parent = best_root;
            communities[best_root].members.insert(
                    communities[best_root].members.end(),
                    communities[source_root].members.begin(),
                    communities[source_root].members.end());
            communities[best_root].strength +=
                    communities[source_root].strength;
            communities[source_root].active = false;
            communities[source_root].members.clear();
            communities[source_root].strength = 0.0;
        }

        for (size_t i = 0; i < touched_roots.size(); ++i) {
            edge_weights[touched_roots[i]] = 0.0;
        }
    }

    std::vector<storage_idx_t> roots;
    roots.reserve(n);
    for (storage_idx_t i = 0; i < n; ++i) {
        if (rabbit_find_root(communities, i) == i && communities[i].active) {
            roots.push_back(i);
        }
    }

    std::sort(
            roots.begin(),
            roots.end(),
            [&](storage_idx_t lhs, storage_idx_t rhs) {
                const double lhs_strength = communities[lhs].strength;
                const double rhs_strength = communities[rhs].strength;
                if (lhs_strength != rhs_strength) {
                    return lhs_strength > rhs_strength;
                }
                const storage_idx_t lhs_min = communities[lhs].members.empty()
                        ? lhs
                        : *std::min_element(
                                  communities[lhs].members.begin(),
                                  communities[lhs].members.end());
                const storage_idx_t rhs_min = communities[rhs].members.empty()
                        ? rhs
                        : *std::min_element(
                                  communities[rhs].members.begin(),
                                  communities[rhs].members.end());
                return lhs_min < rhs_min;
            });

    std::vector<uint8_t> emitted(n, 0);
    std::vector<idx_t> perm;
    perm.reserve(n);
    for (size_t i = 0; i < roots.size(); ++i) {
        rabbit_emit_members(roots[i], communities, emitted, perm);
    }
    for (storage_idx_t i = 0; i < n; ++i) {
        if (!emitted[i]) {
            perm.push_back(i);
        }
    }

    return perm;
}

namespace {

/*
MIT License

Copyright (c) 2016, Hao Wei.

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

class RorderListElement {
   public:
    int key;
    int prev;
    int next;
};

class RorderHeadEnd {
   public:
    int first;
    int second;
    RorderHeadEnd() {
        first = -1;
        second = -1;
    }
};

class UnitHeap {
   public:
    std::vector<int> update;
    std::vector<RorderListElement> LinkedList;
    std::vector<RorderHeadEnd> Header;
    int top{0};
    int heapsize{0};

    explicit UnitHeap(int size)
            : update(size), LinkedList(size), heapsize(size) {
        if (size > 0) {
            Header.clear();
            Header.resize(std::max(size >> 4, 4));
            for (int i = 0; i < size; i++) {
                LinkedList[i].prev = i - 1;
                LinkedList[i].next = i + 1;
                LinkedList[i].key = 0;
                update[i] = 0;
            }
            LinkedList[size - 1].next = -1;
            Header[0].first = 0;
            Header[0].second = size - 1;
            top = 0;
        }
    }

    void DeleteElement(int index) {
        int prev = LinkedList[index].prev;
        int next = LinkedList[index].next;
        int key = LinkedList[index].key;

        if (prev >= 0)
            LinkedList[prev].next = next;
        if (next >= 0)
            LinkedList[next].prev = prev;

        if (Header[key].first == Header[key].second) {
            Header[key].first = Header[key].second = -1;
        } else if (Header[key].first == index) {
            Header[key].first = next;
        } else if (Header[key].second == index) {
            Header[key].second = prev;
        }

        if (top == index) {
            top = LinkedList[top].next;
        }
        LinkedList[index].prev = LinkedList[index].next = -1;
    }

    int ExtractMax() {
        int tmptop;
        do {
            tmptop = top;
            if (update[top] < 0)
                DecreaseTop();
        } while (top != tmptop);
        DeleteElement(tmptop);
        return tmptop;
    }

    void decrementKey(int index) {
        update[index]--;
    }

    void DecreaseTop() {
        const int tmptop = top;
        const int key = LinkedList[tmptop].key;
        const int next = LinkedList[tmptop].next;
        if (next < 0) {
            return;
        }
        int p = key;
        const int newkey = key + update[tmptop] - (update[tmptop] / 2);

        if (newkey < LinkedList[next].key) {
            int tmp = LinkedList[Header[p].second].next;
            while (tmp >= 0 && LinkedList[tmp].key >= newkey) {
                p = LinkedList[tmp].key;
                tmp = LinkedList[Header[p].second].next;
            }
            LinkedList[next].prev = -1;
            const int psecond = Header[p].second;
            int tailnext = LinkedList[psecond].next;
            LinkedList[top].prev = psecond;
            LinkedList[top].next = tailnext;
            LinkedList[psecond].next = tmptop;
            if (tailnext >= 0)
                LinkedList[tailnext].prev = tmptop;
            top = next;

            if (Header[key].first == Header[key].second)
                Header[key].first = Header[key].second = -1;
            else
                Header[key].first = next;

            LinkedList[tmptop].key = newkey;
            update[tmptop] /= 2; /**/
            Header[newkey].second = tmptop;
            if (Header[newkey].first < 0)
                Header[newkey].first = tmptop;
        }
    }

    void ReConstruct() {
        std::vector<int> tmp(heapsize);
        for (int i = 0; i < heapsize; i++)
            tmp[i] = i;

        std::sort(tmp.begin(), tmp.end(), [&](int a, int b) {
            return LinkedList[a].key > LinkedList[b].key;
        });

        int key = LinkedList[tmp[0]].key;
        LinkedList[tmp[0]].next = tmp[1];
        LinkedList[tmp[0]].prev = -1;
        LinkedList[tmp.back()].next = -1;
        LinkedList[tmp.back()].prev = tmp[tmp.size() - 2];
        Header = std::vector<RorderHeadEnd>(
                std::max(key + 1, (int)Header.size()));
        Header[key].first = tmp[0];
        for (int i = 1; i < tmp.size() - 1; i++) {
            int prev = tmp[i - 1];
            int v = tmp[i];
            int next = tmp[i + 1];
            LinkedList[v].prev = prev;
            LinkedList[v].next = next;

            int tmpkey = LinkedList[tmp[i]].key;
            if (tmpkey != key) {
                Header[key].second = tmp[i - 1];
                Header[tmpkey].first = tmp[i];
                key = tmpkey;
            }
        }
        if (key == LinkedList[tmp.back()].key) {
            Header[key].second = tmp.back();
        } else {
            Header[key].second = tmp[tmp.size() - 2];
            int lastone = tmp.back();
            int lastkey = LinkedList[lastone].key;
            Header[lastkey].first = Header[lastkey].second = lastone;
        }
        top = tmp[0];
    }

    void IncrementKey(int index) {
        int key = LinkedList[index].key;
        const int head = Header[key].first;
        const int prev = LinkedList[index].prev;
        const int next = LinkedList[index].next;

        if (head != index) {
            LinkedList[prev].next = next;
            if (next >= 0)
                LinkedList[next].prev = prev;

            int headprev = LinkedList[head].prev;
            LinkedList[index].prev = headprev;
            LinkedList[index].next = head;
            LinkedList[head].prev = index;
            if (headprev >= 0)
                LinkedList[headprev].next = index;
        }

        LinkedList[index].key++;
        if (Header[key].first == Header[key].second)
            Header[key].first = Header[key].second = -1;
        else if (Header[key].first == index)
            Header[key].first = next;
        else if (Header[key].second == index)
            Header[key].second = prev;

        key++;
        if (static_cast<size_t>(key) >= Header.size()) {
            Header.resize(static_cast<size_t>(key) + 5);
        }
        Header[key].second = index;
        if (Header[key].first < 0)
            Header[key].first = index;
        if (LinkedList[top].key < key)
            top = index;
    }
};

} // anonymous namespace

std::vector<faiss::idx_t> IndexHNSW::generateGorderPermutation() {
    const idx_t N = ntotal;
    const int window = 5;

    if (N == 0)
        return {};
    if (N == 1)
        return {0};

    std::vector<std::vector<int>> out_graph(N);
    std::vector<std::vector<int>> in_graph(N);
    out_graph.reserve(N);
    in_graph.reserve(N);
    for (int i = 0; i < N; i++) {
        size_t begin, end;
        hnsw.neighbor_range(i, 0, &begin, &end);
        for (size_t j = begin; j < end; j++) {
            int v = hnsw.neighbors[j];
            if (v < 0)
                break;
            out_graph[i].push_back(v);
            in_graph[v].push_back(i);
        }
        std::sort(out_graph[i].begin(), out_graph[i].end());
    }

    std::vector<int> indegree(N, 0), outdegree(N, 0);
    for (int i = 0; i < N; i++) {
        outdegree[i] = (int)out_graph[i].size();
        for (int v : out_graph[i])
            indegree[v]++;
    }

    UnitHeap unitheap((int)N);
    std::vector<bool> popvexist(N, false);
    std::vector<int> order;
    order.reserve(N);
    std::vector<int> zero;
    zero.reserve(N);
    const int hugevertex = (int)std::sqrt((double)N);

    for (int i = 0; i < N; i++) {
        unitheap.LinkedList[i].key = indegree[i];
        unitheap.update[i] = -indegree[i];
    }
    unitheap.ReConstruct();

    int tmpindex = 0, tmpweight = -1;
    for (int i = 0; i < N; i++) {
        if (indegree[i] > tmpweight) {
            tmpweight = indegree[i];
            tmpindex = i;
        } else if (indegree[i] + outdegree[i] == 0) {
            unitheap.update[i] = INT_MAX / 2;
            zero.push_back(i);
            unitheap.DeleteElement(i);
        }
    }

    order.push_back(tmpindex);
    unitheap.update[tmpindex] = INT_MAX / 2;
    unitheap.DeleteElement(tmpindex);

    for (int u : in_graph[tmpindex]) {
        if (outdegree[u] <= hugevertex) {
            if (unitheap.update[u] == 0)
                unitheap.IncrementKey(u);
            else
                unitheap.update[u]++;
            if (outdegree[u] > 1) {
                for (int w : out_graph[u]) {
                    if (unitheap.update[w] == 0)
                        unitheap.IncrementKey(w);
                    else
                        unitheap.update[w]++;
                }
            }
        }
    }
    if (outdegree[tmpindex] <= hugevertex) {
        for (int w : out_graph[tmpindex]) {
            if (unitheap.update[w] == 0)
                unitheap.IncrementKey(w);
            else
                unitheap.update[w]++;
        }
    }

    int count = 0;
    while ((size_t)count < N - 1 - zero.size()) {
        int v = unitheap.ExtractMax();
        count++;
        order.push_back(v);
        unitheap.update[v] = INT_MAX / 2;

        int popv = (count - window >= 0) ? order[count - window] : -1;
        if (popv >= 0) {
            if (outdegree[popv] <= hugevertex) {
                for (int w : out_graph[popv]) {
                    unitheap.update[w]--;
                }
            }
            for (int u : in_graph[popv]) {
                if (outdegree[u] <= hugevertex) {
                    unitheap.update[u]--;
                    if (outdegree[u] > 1) {
                        if (!std::binary_search(
                                    out_graph[u].begin(),
                                    out_graph[u].end(),
                                    v)) {
                            for (int w : out_graph[u]) {
                                unitheap.update[w]--;
                            }
                        } else {
                            popvexist[u] = true;
                        }
                    }
                }
            }
        }

        if (outdegree[v] <= hugevertex) {
            for (int w : out_graph[v]) {
                if (unitheap.update[w] == 0)
                    unitheap.IncrementKey(w);
                else
                    unitheap.update[w]++;
            }
        }
        for (int u : in_graph[v]) {
            if (outdegree[u] <= hugevertex) {
                if (unitheap.update[u] == 0)
                    unitheap.IncrementKey(u);
                else
                    unitheap.update[u]++;
                if (!popvexist[u]) {
                    if (outdegree[u] > 1) {
                        for (int w : out_graph[u]) {
                            if (unitheap.update[w] == 0)
                                unitheap.IncrementKey(w);
                            else
                                unitheap.update[w]++;
                        }
                    }
                } else {
                    popvexist[u] = false;
                }
            }
        }
    }

    if (!zero.empty())
        order.insert(order.end() - 1, zero.begin(), zero.end());

    std::vector<faiss::idx_t> perm;
    perm.reserve(N);
    for (int x : order)
        perm.push_back((faiss::idx_t)x);
    return perm;
}

/**************************************************************
 * IndexHNSWFlat implementation
 **************************************************************/

IndexHNSWFlat::IndexHNSWFlat() {
    is_trained = true;
}

IndexHNSWFlat::IndexHNSWFlat(int d, int M, MetricType metric)
        : IndexHNSW(
                  (metric == METRIC_L2) ? new IndexFlatL2(d)
                                        : new IndexFlat(d, metric),
                  M) {
    own_fields = true;
    is_trained = true;
}

/**************************************************************
 * IndexHNSWPQ implementation
 **************************************************************/

IndexHNSWPQ::IndexHNSWPQ() = default;

IndexHNSWPQ::IndexHNSWPQ(
        int d,
        int pq_m,
        int M,
        int pq_nbits,
        MetricType metric)
        : IndexHNSW(new IndexPQ(d, pq_m, pq_nbits, metric), M) {
    own_fields = true;
    is_trained = false;
}

void IndexHNSWPQ::train(idx_t n, const float* x) {
    IndexHNSW::train(n, x);
    (dynamic_cast<IndexPQ*>(storage))->pq.compute_sdc_table();
}

/**************************************************************
 * IndexHNSWSQ implementation
 **************************************************************/

IndexHNSWSQ::IndexHNSWSQ(
        int d,
        ScalarQuantizer::QuantizerType qtype,
        int M,
        MetricType metric)
        : IndexHNSW(new IndexScalarQuantizer(d, qtype, metric), M) {
    is_trained = this->storage->is_trained;
    own_fields = true;
}

IndexHNSWSQ::IndexHNSWSQ() = default;

/**************************************************************
 * IndexHNSW2Level implementation
 **************************************************************/

IndexHNSW2Level::IndexHNSW2Level(
        Index* quantizer,
        size_t nlist,
        int m_pq,
        int M)
        : IndexHNSW(new Index2Layer(quantizer, nlist, m_pq), M) {
    own_fields = true;
    is_trained = false;
}

IndexHNSW2Level::IndexHNSW2Level() = default;

namespace {

// same as search_from_candidates but uses v
// visno -> is in result list
// visno + 1 -> in result list + in candidates
int search_from_candidates_2(
        const HNSW& hnsw,
        DistanceComputer& qdis,
        int k,
        idx_t* I,
        float* D,
        MinimaxHeap& candidates,
        VisitedTable& vt,
        HNSWStats& stats,
        int level,
        int nres_in = 0) {
    int nres = nres_in;
    for (int i = 0; i < candidates.size(); i++) {
        idx_t v1 = candidates.ids[i];
        FAISS_ASSERT(v1 >= 0);
        vt.visited[v1] = vt.visno + 1;
    }

    int nstep = 0;

    while (candidates.size() > 0) {
        float d0 = 0;
        int v0 = candidates.pop_min(&d0);

        size_t begin, end;
        hnsw.neighbor_range(v0, level, &begin, &end);

        for (size_t j = begin; j < end; j++) {
            int v1 = hnsw.neighbors[j];
            if (v1 < 0)
                break;
            if (vt.visited[v1] == vt.visno + 1) {
                // nothing to do
            } else {
                float d = qdis(v1);
                candidates.push(v1, d);

                // never seen before --> add to heap
                if (vt.visited[v1] < vt.visno) {
                    if (nres < k) {
                        faiss::maxheap_push(++nres, D, I, d, v1);
                    } else if (d < D[0]) {
                        faiss::maxheap_replace_top(nres, D, I, d, v1);
                    }
                }
                vt.visited[v1] = vt.visno + 1;
            }
        }

        nstep++;
        if (nstep > hnsw.efSearch) {
            break;
        }
    }

    stats.n1++;
    if (candidates.size() == 0)
        stats.n2++;

    return nres;
}

} // namespace

void IndexHNSW2Level::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT(k > 0);
    FAISS_THROW_IF_NOT_MSG(
            !params, "search params not supported for this index");

    if (dynamic_cast<const Index2Layer*>(storage)) {
        IndexHNSW::search(n, x, k, distances, labels);

    } else { // "mixed" search
        size_t n1 = 0, n2 = 0, ndis = 0, nhops = 0;

        const IndexIVFPQ* index_ivfpq =
                dynamic_cast<const IndexIVFPQ*>(storage);

        int nprobe = index_ivfpq->nprobe;

        std::unique_ptr<idx_t[]> coarse_assign(new idx_t[n * nprobe]);
        std::unique_ptr<float[]> coarse_dis(new float[n * nprobe]);

        index_ivfpq->quantizer->search(
                n, x, nprobe, coarse_dis.get(), coarse_assign.get());

        index_ivfpq->search_preassigned(
                n,
                x,
                k,
                coarse_assign.get(),
                coarse_dis.get(),
                distances,
                labels,
                false);

#pragma omp parallel
        {
            VisitedTable vt(ntotal);
            std::unique_ptr<DistanceComputer> dis(make_distance_computer());

            int candidates_size = hnsw.upper_beam;
            MinimaxHeap candidates(candidates_size);

#pragma omp for reduction(+ : n1, n2, ndis, nhops)
            for (idx_t i = 0; i < n; i++) {
                idx_t* idxi = labels + i * k;
                float* simi = distances + i * k;
                dis->set_query(x + i * d);

                // mark all inverted list elements as visited

                for (int j = 0; j < nprobe; j++) {
                    idx_t key = coarse_assign[j + i * nprobe];
                    if (key < 0)
                        break;
                    size_t list_length = index_ivfpq->get_list_size(key);
                    const idx_t* ids = index_ivfpq->invlists->get_ids(key);

                    for (int jj = 0; jj < list_length; jj++) {
                        vt.set(ids[jj]);
                    }
                }

                candidates.clear();

                for (int j = 0; j < hnsw.upper_beam && j < k; j++) {
                    if (idxi[j] < 0)
                        break;
                    candidates.push(idxi[j], simi[j]);
                }

                // reorder from sorted to heap
                maxheap_heapify(k, simi, idxi, simi, idxi, k);

                HNSWStats search_stats;
                search_from_candidates_2(
                        hnsw,
                        *dis,
                        k,
                        idxi,
                        simi,
                        candidates,
                        vt,
                        search_stats,
                        0,
                        k);
                n1 += search_stats.n1;
                n2 += search_stats.n2;
                ndis += search_stats.ndis;
                nhops += search_stats.nhops;

                vt.advance();
                vt.advance();

                maxheap_reorder(k, simi, idxi);
            }
        }

        hnsw_stats.combine({n1, n2, ndis, nhops});
    }
}

void IndexHNSW2Level::flip_to_ivf() {
    Index2Layer* storage2l = dynamic_cast<Index2Layer*>(storage);

    FAISS_THROW_IF_NOT(storage2l);

    IndexIVFPQ* index_ivfpq = new IndexIVFPQ(
            storage2l->q1.quantizer,
            d,
            storage2l->q1.nlist,
            storage2l->pq.M,
            8);
    index_ivfpq->pq = storage2l->pq;
    index_ivfpq->is_trained = storage2l->is_trained;
    index_ivfpq->precompute_table();
    index_ivfpq->own_fields = storage2l->q1.own_fields;
    storage2l->transfer_to_IVFPQ(*index_ivfpq);
    index_ivfpq->make_direct_map(true);

    storage = index_ivfpq;
    delete storage2l;
}

/**************************************************************
 * IndexHNSWCagra implementation
 **************************************************************/

IndexHNSWCagra::IndexHNSWCagra() {
    is_trained = true;
}

IndexHNSWCagra::IndexHNSWCagra(int d, int M, MetricType metric)
        : IndexHNSW(
                  (metric == METRIC_L2)
                          ? static_cast<IndexFlat*>(new IndexFlatL2(d))
                          : static_cast<IndexFlat*>(new IndexFlatIP(d)),
                  M) {
    FAISS_THROW_IF_NOT_MSG(
            ((metric == METRIC_L2) || (metric == METRIC_INNER_PRODUCT)),
            "unsupported metric type for IndexHNSWCagra");
    own_fields = true;
    is_trained = true;
    init_level0 = true;
    keep_max_size_level0 = true;
}

void IndexHNSWCagra::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(
            !base_level_only,
            "Cannot add vectors when base_level_only is set to True");

    IndexHNSW::add(n, x);
}

void IndexHNSWCagra::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    if (!base_level_only) {
        IndexHNSW::search(n, x, k, distances, labels, params);
    } else {
        std::vector<storage_idx_t> nearest(n);
        std::vector<float> nearest_d(n);

#pragma omp for
        for (idx_t i = 0; i < n; i++) {
            std::unique_ptr<DistanceComputer> dis(make_distance_computer());
            dis->set_query(x + i * d);
            nearest[i] = -1;
            nearest_d[i] = std::numeric_limits<float>::max();

            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<idx_t> distrib(0, this->ntotal - 1);

            for (idx_t j = 0; j < num_base_level_search_entrypoints; j++) {
                auto idx = distrib(gen);
                auto distance = (*dis)(idx);
                if (distance < nearest_d[i]) {
                    nearest[i] = idx;
                    nearest_d[i] = distance;
                }
            }
            FAISS_THROW_IF_NOT_MSG(
                    nearest[i] >= 0, "Could not find a valid entrypoint.");
        }

        search_level_0(
                n,
                x,
                k,
                nearest.data(),
                nearest_d.data(),
                distances,
                labels,
                1, // n_probes
                1, // search_type
                params);
    }
}

} // namespace faiss
