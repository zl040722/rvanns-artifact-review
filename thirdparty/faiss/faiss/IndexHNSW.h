/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#pragma once

#include <memory>
#include <vector>

#include <faiss/IndexFlat.h>
#include <faiss/IndexPQ.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/impl/HNSW.h>
#include <faiss/utils/utils.h>

namespace faiss {

struct IndexHNSW;

/** The HNSW index is a normal random-access index with a HNSW
 * link structure built on top */

struct IndexHNSW : Index {
    typedef HNSW::storage_idx_t storage_idx_t;

    struct CacheAlignedLayoutInfo {
        bool enabled = false;
        size_t stride = 0;
        size_t row_bytes = 0;
        size_t count = 0;
        const uint8_t* data = nullptr;
    };

    // the link structure
    HNSW hnsw;

    // the sequential storage
    bool own_fields = false;
    Index* storage = nullptr;

    // When set to false, level 0 in the knn graph is not initialized.
    // This option is used by GpuIndexCagra::copyTo(IndexHNSWCagra*)
    // as level 0 knn graph is copied over from the index built by
    // GpuIndexCagra.
    bool init_level0 = true;

    // When set to true, all neighbors in level 0 are filled up
    // to the maximum size allowed (2 * M). This option is used by
    // IndexHHNSWCagra to create a full base layer graph that is
    // used when GpuIndexCagra::copyFrom(IndexHNSWCagra*) is invoked.
    bool keep_max_size_level0 = false;

    // ID mapping after reordering: perm[new_id] = old_id
    // This is used to convert new IDs back to original IDs for recall
    // calculation
    std::vector<idx_t> reorder_perm;

    explicit IndexHNSW(int d = 0, int M = 32, MetricType metric = METRIC_L2);
    explicit IndexHNSW(Index* storage, int M = 32);
    IndexHNSW(const IndexHNSW& other);
    IndexHNSW& operator=(const IndexHNSW& other);

    ~IndexHNSW() override;

    void add(idx_t n, const float* x) override;

    /// Trains the storage if needed
    void train(idx_t n, const float* x) override;

    /// entry point for search
    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    void range_search(
            idx_t n,
            const float* x,
            float radius,
            RangeSearchResult* result,
            const SearchParameters* params = nullptr) const override;

    void reconstruct(idx_t key, float* recons) const override;

    void reset() override;

    void shrink_level_0_neighbors(int size);

    /** Perform search only on level 0, given the starting points for
     * each vertex.
     *
     * @param search_type 1:perform one search per nprobe, 2: enqueue
     *                    all entry points
     */
    void search_level_0(
            idx_t n,
            const float* x,
            idx_t k,
            const storage_idx_t* nearest,
            const float* nearest_d,
            float* distances,
            idx_t* labels,
            int nprobe = 1,
            int search_type = 1,
            const SearchParameters* params = nullptr) const;

    /// alternative graph building
    void init_level_0_from_knngraph(int k, const float* D, const idx_t* I);

    /// alternative graph building
    void init_level_0_from_entry_points(
            int npt,
            const storage_idx_t* points,
            const storage_idx_t* nearests);

    // reorder links from nearest to farthest
    void reorder_links();

    // optimize neighbors by sorting them by ID for better cache locality
    void optimize_neighbors_by_id();

    void link_singletons();

    void permute_entries(const idx_t* perm);

    /**
     * Reorder nodes using graph traversal algorithms to improve spatial
     * locality. This should be called after all vectors are added but before
     * saving to disk.
     *
     * @param level Graph level to use for traversal (default: 0, only level-0
     * supported)
     * @param method Traversal method 
     * @param freeze Whether to freeze the index after reordering (default:
     * true)
     */
    void reorder_graph_after_build(
            int level = 0,
            const char* method = "bfs",
            bool freeze = true);

    /**
     * Get the ID mapping after reordering: perm[new_id] = old_id
     * Returns empty vector if no reordering has been performed.
     *
     * @return Permutation array where perm[new_id] = old_id
     */
    const std::vector<idx_t>& get_reorder_perm() const {
        return reorder_perm;
    }

    CacheAlignedLayoutInfo get_cache_aligned_codes() const;
    CacheAlignedLayoutInfo get_cache_aligned_code_norms() const;

    /**
     * Convert new ID back to original ID using the reordering permutation.
     * Returns the same ID if no reordering has been performed.
     *
     * @param new_id The new ID after reordering
     * @return The original ID before reordering
     */
    idx_t convert_to_original_id(idx_t new_id) const {
        if (reorder_perm.empty() || new_id < 0 ||
            new_id >= (idx_t)reorder_perm.size()) {
            return new_id;
        }
        return reorder_perm[new_id];
    }

    DistanceComputer* get_distance_computer() const override;
    DistanceComputer* make_distance_computer() const;

   private:
    // Generate Rorder permutation for graph reordering
    std::vector<idx_t> generateRorderPermutation();

    struct CacheAlignedCoLayout;
    std::unique_ptr<CacheAlignedCoLayout> cache_aligned_soa_;
    bool pending_cacheline_layout_build_ = false;

    void apply_cacheline_co_layout_to_storage();
    void invalidate_cacheline_layout();
};

/** Flat index topped with with a HNSW structure to access elements
 *  more efficiently.
 */

struct IndexHNSWFlat : IndexHNSW {
    IndexHNSWFlat();
    IndexHNSWFlat(int d, int M, MetricType metric = METRIC_L2);
};

/** PQ index topped with with a HNSW structure to access elements
 *  more efficiently.
 */
struct IndexHNSWPQ : IndexHNSW {
    IndexHNSWPQ();
    IndexHNSWPQ(
            int d,
            int pq_m,
            int M,
            int pq_nbits = 8,
            MetricType metric = METRIC_L2);
    void train(idx_t n, const float* x) override;
};

/** SQ index topped with with a HNSW structure to access elements
 *  more efficiently.
 */
struct IndexHNSWSQ : IndexHNSW {
    IndexHNSWSQ();
    IndexHNSWSQ(
            int d,
            ScalarQuantizer::QuantizerType qtype,
            int M,
            MetricType metric = METRIC_L2);
};

/** 2-level code structure with fast random access
 */
struct IndexHNSW2Level : IndexHNSW {
    IndexHNSW2Level();
    IndexHNSW2Level(Index* quantizer, size_t nlist, int m_pq, int M);

    void flip_to_ivf();

    /// entry point for search
    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;
};

struct IndexHNSWCagra : IndexHNSW {
    IndexHNSWCagra();
    IndexHNSWCagra(int d, int M, MetricType metric = METRIC_L2);

    /// When set to true, the index is immutable.
    /// This option is used to copy the knn graph from GpuIndexCagra
    /// to the base level of IndexHNSWCagra without adding upper levels.
    /// Doing so enables to search the HNSW index, but removes the
    /// ability to add vectors.
    bool base_level_only = false;

    /// When `base_level_only` is set to `True`, the search function
    /// searches only the base level knn graph of the HNSW index.
    /// This parameter selects the entry point by randomly selecting
    /// some points and using the best one.
    int num_base_level_search_entrypoints = 32;

    void add(idx_t n, const float* x) override;

    /// entry point for search
    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;
};

} // namespace faiss
