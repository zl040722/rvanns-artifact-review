// Unified four-path ablation driver used by the supplementary material:
//   S = Scalar+FP32, V = SIMD+FP32, M = SIMD+MPMI, F = MPMI+ROrder.
//
// The cached index always records its quantizer and physical layout in the
// filename.  F loads/builds the unordered MPMI cache and applies ROrder in
// memory on every run because the legacy Faiss index format does not persist
// IndexHNSW::reorder_perm.

#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/index_io.h>
#include <knowhere/comp/knowhere_config.h>

#include "simd/distances_ref.h"
#include "simd/hook.h"

#if defined(__riscv_vector)
#include <faiss/impl/ScalarQuantizerCodec_rvv.h>
#endif

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

enum class Ablation {
    S,
    V,
    M,
    F,
};

struct Cli {
    Ablation config = Ablation::F;
    std::string dataset = "CUSTOM";
    std::string base_file;
    std::string query_file;
    std::string ground_truth_file;
    std::string backend = "AUTO";
    std::string metric = "l2";
    std::string cache_dir = "vectors_out/index_cache";
    std::string output_csv = "vectors_out/results/mpmi_cq1.csv";
    int dim = 0;
    int M = 16;
    int ef_construction = 256;
    std::vector<int> ef_search{16, 32, 64, 128, 256, 512};
    int k = 100;
    int query_limit = 0;
    int warmup_queries = 1000;
    int warmup_rounds = 2;
    int rounds = 1;
    bool force_rebuild = false;
};

struct EffectiveConfig {
    std::string key;
    std::string name;
    std::string backend;
    std::string qtype;
    std::string reorder;
    std::string decode_lmul;
    std::string acc_lmul;
    std::string lmul_tag;
    bool hybrid = false;
    bool apply_rorder = false;
};

struct BuildStats {
    long long load_ms = 0;
    long long train_ms = 0;
    long long build_ms = 0;
    long long save_ms = 0;
    long long reorder_ms = 0;
    std::string cache_file;
};

std::string
upper_ascii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::toupper(c));
    });
    return value;
}

std::string
sanitize(std::string value) {
    for (char& c : value) {
        if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '-' ||
              c == '_')) {
            c = '_';
        }
    }
    return value;
}

uint64_t
fnv1a_64(const std::string& value) {
    uint64_t hash = UINT64_C(14695981039346656037);
    for (unsigned char byte : value) {
        hash ^= byte;
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

std::string
base_file_identity(const std::string& path) {
    std::error_code ec;
    std::filesystem::path resolved = std::filesystem::weakly_canonical(path, ec);
    if (ec) {
        ec.clear();
        resolved = std::filesystem::absolute(path, ec);
    }
    if (ec) {
        resolved = path;
        ec.clear();
    }

    const auto modified = std::filesystem::last_write_time(path, ec);
    const auto modified_ticks = ec ? 0 : modified.time_since_epoch().count();
    std::ostringstream identity;
    identity << resolved.generic_string() << '|' << modified_ticks;

    std::ostringstream compact;
    compact << std::hex << fnv1a_64(identity.str());
    return compact.str();
}

Ablation
parse_ablation(const std::string& value) {
    const std::string key = upper_ascii(value);
    if (key == "S") {
        return Ablation::S;
    }
    if (key == "V") {
        return Ablation::V;
    }
    if (key == "M") {
        return Ablation::M;
    }
    if (key == "F") {
        return Ablation::F;
    }
    throw std::runtime_error("--config must be one of S, V, M, or F");
}

std::vector<int>
parse_int_list(const std::string& value) {
    std::vector<int> result;
    std::stringstream input(value);
    std::string token;
    while (std::getline(input, token, ',')) {
        if (!token.empty()) {
            const int number = std::stoi(token);
            if (number <= 0) {
                throw std::runtime_error("efSearch values must be positive");
            }
            result.push_back(number);
        }
    }
    if (result.empty()) {
        throw std::runtime_error("--efs requires at least one value");
    }
    return result;
}

Cli
parse_cli(int argc, char** argv) {
    Cli cli;
    auto next = [&](int& i) -> std::string {
        if (i + 1 >= argc) {
            throw std::runtime_error(std::string("missing value for ") + argv[i]);
        }
        return argv[++i];
    };

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--config") {
            cli.config = parse_ablation(next(i));
        } else if (arg == "--dataset" || arg == "--name") {
            cli.dataset = next(i);
        } else if (arg == "--base") {
            cli.base_file = next(i);
        } else if (arg == "--query") {
            cli.query_file = next(i);
        } else if (arg == "--gt") {
            cli.ground_truth_file = next(i);
        } else if (arg == "--backend" || arg == "--simd") {
            cli.backend = next(i);
        } else if (arg == "--metric") {
            cli.metric = upper_ascii(next(i));
            if (cli.metric == "L2") {
                cli.metric = "l2";
            } else if (cli.metric == "IP" ||
                       cli.metric == "INNER_PRODUCT") {
                cli.metric = "ip";
            } else if (cli.metric == "COSINE") {
                cli.metric = "cosine";
            } else {
                throw std::runtime_error("--metric must be l2, ip, or cosine");
            }
        } else if (arg == "--cache-dir") {
            cli.cache_dir = next(i);
        } else if (arg == "--output-csv") {
            cli.output_csv = next(i);
        } else if (arg == "--dim") {
            cli.dim = std::stoi(next(i));
        } else if (arg == "--M") {
            cli.M = std::stoi(next(i));
        } else if (arg == "--efC") {
            cli.ef_construction = std::stoi(next(i));
        } else if (arg == "--efs") {
            cli.ef_search = parse_int_list(next(i));
        } else if (arg == "--k") {
            cli.k = std::stoi(next(i));
        } else if (arg == "--nq") {
            cli.query_limit = std::stoi(next(i));
        } else if (arg == "--warmup-queries") {
            cli.warmup_queries = std::stoi(next(i));
        } else if (arg == "--warmup-rounds") {
            cli.warmup_rounds = std::stoi(next(i));
        } else if (arg == "--rounds") {
            cli.rounds = std::stoi(next(i));
        } else if (arg == "--force-rebuild") {
            cli.force_rebuild = true;
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                    << "Usage: test_mpmi_cq1 --config S|V|M|F --base BASE.f32 --query QUERY.f32 "
                       "--dim D [--gt GT.ivecs] [--metric l2|ip|cosine] "
                       "[--backend AUTO|AVX512|AVX2|SSE4_2] "
                       "[--dataset NAME] [--M M] [--efC N] [--efs 16,32,...] [--k K] "
                       "[--nq N] [--warmup-queries N] [--warmup-rounds N] [--rounds N] "
                       "[--cache-dir DIR] [--output-csv FILE] [--force-rebuild]\n\n"
                    << "F always applies rorder; no command-line option can substitute another layout.\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }

    if (cli.base_file.empty() || cli.query_file.empty() || cli.dim <= 0) {
        throw std::runtime_error("--base, --query, and a positive --dim are required");
    }
    if (cli.M <= 0 || cli.ef_construction <= 0 || cli.k <= 0 ||
        cli.warmup_queries < 0 || cli.warmup_rounds < 0 || cli.rounds <= 0) {
        throw std::runtime_error("invalid non-positive benchmark parameter");
    }
    return cli;
}

knowhere::KnowhereConfig::SimdType
parse_backend(const std::string& value) {
    const std::string backend = upper_ascii(value);
    if (backend == "AUTO") {
        return knowhere::KnowhereConfig::SimdType::AUTO;
    }
    if (backend == "AVX512" || backend == "AVX-512") {
        return knowhere::KnowhereConfig::SimdType::AVX512;
    }
    if (backend == "AVX2") {
        return knowhere::KnowhereConfig::SimdType::AVX2;
    }
    if (backend == "SSE4_2" || backend == "SSE42") {
        return knowhere::KnowhereConfig::SimdType::SSE4_2;
    }
    if (backend == "GENERIC" || backend == "SCALAR") {
        return knowhere::KnowhereConfig::SimdType::GENERIC;
    }
    throw std::runtime_error("unknown SIMD backend: " + value);
}

EffectiveConfig
apply_config(const Cli& cli) {
    EffectiveConfig effective;
    switch (cli.config) {
        case Ablation::S:
            effective.key = "S";
            effective.name = "Scalar+FP32";
            (void)knowhere::KnowhereConfig::SetSimdType(
                    knowhere::KnowhereConfig::SimdType::GENERIC);
            // SetSimdType(GENERIC) is effective on x86, but non-x86 hook
            // initialization historically selects the native backend
            // unconditionally.  Pin both HNSW L2 call sites to the common
            // reference functions so S is genuinely scalar on every ISA.
            faiss::fvec_L2sqr = faiss::fvec_L2sqr_ref;
            faiss::fvec_L2sqr_batch_4 = faiss::fvec_L2sqr_batch_4_ref;
            faiss::fvec_inner_product = faiss::fvec_inner_product_ref;
            faiss::fvec_inner_product_batch_4 =
                    faiss::fvec_inner_product_batch_4_ref;
            effective.backend = "SCALAR-REF";
            effective.qtype = "FP32";
            effective.reorder = "unordered";
            break;
        case Ablation::V:
            effective.key = "V";
            effective.name = "SIMD+FP32";
            effective.backend = knowhere::KnowhereConfig::SetSimdType(
                    parse_backend(cli.backend));
            effective.qtype = "FP32";
            effective.reorder = "unordered";
            break;
        case Ablation::M:
            effective.key = "M";
            effective.name = "SIMD+MPMI";
            effective.backend = knowhere::KnowhereConfig::SetSimdType(
                    parse_backend(cli.backend));
            effective.qtype = "QT_HYBRID_FP8_16_32";
            effective.reorder = "unordered";
            effective.hybrid = true;
            break;
        case Ablation::F:
            effective.key = "F";
            effective.name = "MPMI+ROrder";
            effective.backend = knowhere::KnowhereConfig::SetSimdType(
                    parse_backend(cli.backend));
            effective.qtype = "QT_HYBRID_FP8_16_32";
            effective.reorder = "rorder";
            effective.hybrid = true;
            effective.apply_rorder = true;
            break;
    }
    if (effective.backend.empty()) {
        effective.backend = cli.config == Ablation::S ? "GENERIC" : upper_ascii(cli.backend);
    }
    if (cli.config != Ablation::S &&
        (upper_ascii(cli.backend) == "GENERIC" ||
         upper_ascii(cli.backend) == "SCALAR")) {
        throw std::runtime_error(
                effective.key + " is a SIMD path and cannot use a scalar backend");
    }

    if (effective.hybrid) {
#if defined(__riscv_vector)
        effective.decode_lmul = std::to_string(faiss::rvanns_rvv_decode_lmul());
        effective.acc_lmul = std::to_string(faiss::rvanns_rvv_acc_lmul());
        effective.lmul_tag = faiss::rvanns_rvv_lmul_tag();
#else
        effective.decode_lmul = "n/a";
        effective.acc_lmul = "n/a";
        effective.lmul_tag = "non-rvv";
#endif
    } else {
        effective.decode_lmul = "not-used";
        effective.acc_lmul = "not-used";
        effective.lmul_tag = "not-used";
    }
    return effective;
}

std::vector<float>
load_f32(const std::string& path, int dim, int& rows) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error("cannot open float data: " + path);
    }
    const std::streamsize bytes = input.tellg();
    const std::streamsize row_bytes =
            static_cast<std::streamsize>(dim) * sizeof(float);
    if (bytes <= 0 || bytes % row_bytes != 0) {
        throw std::runtime_error("file size is not a positive multiple of dim: " + path);
    }
    rows = static_cast<int>(bytes / row_bytes);
    std::vector<float> data(static_cast<size_t>(rows) * dim);
    input.seekg(0);
    input.read(reinterpret_cast<char*>(data.data()), bytes);
    if (!input) {
        throw std::runtime_error("short read from: " + path);
    }
    return data;
}

void
normalize_rows(std::vector<float>& data, int rows, int dim) {
    for (int row = 0; row < rows; ++row) {
        float* values = data.data() + static_cast<size_t>(row) * dim;
        double squared_norm = 0.0;
        for (int j = 0; j < dim; ++j) {
            squared_norm += static_cast<double>(values[j]) * values[j];
        }
        if (squared_norm == 0.0) {
            continue;
        }
        const float inverse_norm =
                static_cast<float>(1.0 / std::sqrt(squared_norm));
        for (int j = 0; j < dim; ++j) {
            values[j] *= inverse_norm;
        }
    }
}

faiss::MetricType
metric_type(const Cli& cli) {
    return cli.metric == "l2" ? faiss::METRIC_L2
                              : faiss::METRIC_INNER_PRODUCT;
}

std::vector<faiss::idx_t>
load_ivecs(const std::string& path, int nq, int k) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot open ground truth: " + path);
    }
    std::vector<faiss::idx_t> result(static_cast<size_t>(nq) * k, -1);
    for (int qi = 0; qi < nq; ++qi) {
        int width = 0;
        if (!input.read(reinterpret_cast<char*>(&width), sizeof(width))) {
            throw std::runtime_error("ground truth has fewer rows than queries");
        }
        if (width < k || width <= 0) {
            throw std::runtime_error("ground-truth row is narrower than k");
        }
        std::vector<int> row(static_cast<size_t>(width));
        if (!input.read(
                    reinterpret_cast<char*>(row.data()),
                    static_cast<std::streamsize>(row.size() * sizeof(int)))) {
            throw std::runtime_error("truncated ground-truth row");
        }
        for (int rank = 0; rank < k; ++rank) {
            result[static_cast<size_t>(qi) * k + rank] = row[rank];
        }
    }
    return result;
}

std::string
cache_path(const Cli& cli, const EffectiveConfig& effective, int nb) {
    std::filesystem::create_directories(cli.cache_dir);
    const std::string cache_qtype = effective.hybrid ? "hybrid" : "fp32";
    // F intentionally shares the unordered hybrid source cache with M and
    // applies ROrder after loading.  Both cache properties remain explicit.
    const std::string cache_layout = "unordered";
    std::error_code ec;
    const auto bytes = std::filesystem::file_size(cli.base_file, ec);
    const auto safe_bytes = ec ? 0 : bytes;
    std::ostringstream path;
    path << cli.cache_dir << '/' << sanitize(cli.dataset)
         << "_fmt_river2"
         << "_hnsw_qtype_" << cache_qtype
         << "_layout_" << cache_layout
         << "_metric_" << cli.metric
         << "_nb" << nb << "_dim" << cli.dim << "_M" << cli.M
         << "_efC" << cli.ef_construction << "_bytes" << safe_bytes
         << "_base" << base_file_identity(cli.base_file)
         << ".faiss";
    return path.str();
}

bool
valid_cached_index(
        faiss::Index* index,
        const EffectiveConfig& effective,
        faiss::MetricType metric,
        int dim,
        int nb) {
    if (!index || index->d != dim || index->ntotal != nb ||
        index->metric_type != metric) {
        return false;
    }
    if (!effective.hybrid) {
        return dynamic_cast<faiss::IndexHNSWFlat*>(index) != nullptr;
    }
    auto* hnsw_sq = dynamic_cast<faiss::IndexHNSWSQ*>(index);
    if (!hnsw_sq) {
        return false;
    }
    auto* storage = dynamic_cast<faiss::IndexScalarQuantizer*>(
            hnsw_sq->storage);
    return storage &&
            storage->sq.qtype ==
            faiss::ScalarQuantizer::QT_HYBRID_FP8_16_32;
}

std::unique_ptr<faiss::Index>
load_or_build(
        const Cli& cli,
        const EffectiveConfig& effective,
        const std::vector<float>& base,
        int nb,
        BuildStats& stats) {
    stats.cache_file = cache_path(cli, effective, nb);
    if (!cli.force_rebuild && std::filesystem::exists(stats.cache_file)) {
        const auto begin = std::chrono::steady_clock::now();
        std::unique_ptr<faiss::Index> loaded(
                faiss::read_index(stats.cache_file.c_str()));
        const auto end = std::chrono::steady_clock::now();
        stats.load_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                end - begin)
                                .count();
        if (valid_cached_index(
                    loaded.get(), effective, metric_type(cli), cli.dim, nb)) {
            return loaded;
        }
        std::cerr << "Ignoring incompatible cache: " << stats.cache_file << '\n';
    }

    std::unique_ptr<faiss::Index> index;
    if (effective.hybrid) {
        auto* hnsw = new faiss::IndexHNSWSQ(
                cli.dim,
                faiss::ScalarQuantizer::QT_HYBRID_FP8_16_32,
                cli.M,
                metric_type(cli));
        hnsw->hnsw.efConstruction = cli.ef_construction;
        index.reset(hnsw);
        const auto train_begin = std::chrono::steady_clock::now();
        index->train(nb, base.data());
        const auto train_end = std::chrono::steady_clock::now();
        stats.train_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                 train_end - train_begin)
                                 .count();
    } else {
        auto* hnsw = new faiss::IndexHNSWFlat(
                cli.dim, cli.M, metric_type(cli));
        hnsw->hnsw.efConstruction = cli.ef_construction;
        index.reset(hnsw);
    }

    const auto build_begin = std::chrono::steady_clock::now();
    index->add(nb, base.data());
    const auto build_end = std::chrono::steady_clock::now();
    stats.build_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                             build_end - build_begin)
                             .count();

    // Persist only the pre-layout index.  The filename explicitly records
    // qtype=hybrid/fp32 and layout=unordered.
    const auto save_begin = std::chrono::steady_clock::now();
    faiss::write_index(index.get(), stats.cache_file.c_str());
    const auto save_end = std::chrono::steady_clock::now();
    stats.save_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            save_end - save_begin)
                            .count();
    return index;
}

std::vector<faiss::idx_t>
apply_layout(
        faiss::Index* index,
        const EffectiveConfig& effective,
        BuildStats& stats) {
    if (!effective.apply_rorder) {
        return {};
    }
    auto* hnsw = dynamic_cast<faiss::IndexHNSW*>(index);
    if (!hnsw) {
        throw std::runtime_error("F requires an HNSW index");
    }
    const auto begin = std::chrono::steady_clock::now();
    hnsw->reorder_graph_after_build(0, "rorder", true);
    const auto end = std::chrono::steady_clock::now();
    stats.reorder_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                               end - begin)
                               .count();
    const std::vector<faiss::idx_t> permutation = hnsw->get_reorder_perm();
    if (permutation.size() != static_cast<size_t>(index->ntotal)) {
        throw std::runtime_error(
                "ROrder did not produce a complete perm[new_id]=old_id mapping");
    }
    return permutation;
}

std::vector<faiss::idx_t>
ground_truth(
        const Cli& cli,
        const std::vector<float>& base,
        int nb,
        const std::vector<float>& queries,
        int nq) {
    if (!cli.ground_truth_file.empty()) {
        return load_ivecs(cli.ground_truth_file, nq, cli.k);
    }
    std::unique_ptr<faiss::Index> exact;
    if (metric_type(cli) == faiss::METRIC_L2) {
        exact.reset(new faiss::IndexFlatL2(cli.dim));
    } else {
        exact.reset(new faiss::IndexFlatIP(cli.dim));
    }
    exact->add(nb, base.data());
    std::vector<float> distances(static_cast<size_t>(nq) * cli.k);
    std::vector<faiss::idx_t> labels(static_cast<size_t>(nq) * cli.k);
    exact->search(
            nq,
            queries.data(),
            cli.k,
            distances.data(),
            labels.data());
    return labels;
}

double
recall_at_k(
        const std::vector<faiss::idx_t>& labels,
        const std::vector<faiss::idx_t>& truth,
        int nq,
        int k,
        const std::vector<faiss::idx_t>& new_to_old) {
    double total = 0.0;
    for (int qi = 0; qi < nq; ++qi) {
        int hits = 0;
        const auto truth_begin = truth.begin() + static_cast<size_t>(qi) * k;
        for (int rank = 0; rank < k; ++rank) {
            faiss::idx_t predicted = labels[static_cast<size_t>(qi) * k + rank];
            if (!new_to_old.empty() && predicted >= 0) {
                if (static_cast<size_t>(predicted) >= new_to_old.size()) {
                    throw std::runtime_error("search returned ID outside ROrder mapping");
                }
                predicted = new_to_old[static_cast<size_t>(predicted)];
            }
            if (std::find(truth_begin, truth_begin + k, predicted) !=
                truth_begin + k) {
                ++hits;
            }
        }
        total += static_cast<double>(hits) / k;
    }
    return total / nq;
}

void
append_csv(
        const Cli& cli,
        const EffectiveConfig& effective,
        const BuildStats& stats,
        int nb,
        int nq,
        int ef_search,
        double qps,
        double recall) {
    if (cli.output_csv.empty()) {
        return;
    }
    const std::filesystem::path output_path(cli.output_csv);
    if (!output_path.parent_path().empty()) {
        std::filesystem::create_directories(output_path.parent_path());
    }
    const bool write_header = !std::filesystem::exists(output_path) ||
            std::filesystem::file_size(output_path) == 0;
    std::ofstream output(output_path, std::ios::app);
    if (!output) {
        throw std::runtime_error("cannot open output CSV: " + cli.output_csv);
    }
    if (write_header) {
        output << "dataset,metric,config,config_name,effective_backend,qtype,effective_reorder,"
                  "cache_layout,decode_lmul,acc_lmul,lmul_tag,nb,nq,dim,M,efC,efS,k,rounds,"
                  "qps,recall,load_ms,train_ms,build_ms,save_ms,reorder_ms,cache_file\n";
    }
    output << sanitize(cli.dataset) << ',' << cli.metric << ',' << effective.key << ','
           << effective.name << ',' << effective.backend << ',' << effective.qtype
           << ',' << effective.reorder << ",unordered," << effective.decode_lmul
           << ',' << effective.acc_lmul << ',' << effective.lmul_tag << ',' << nb
           << ',' << nq << ',' << cli.dim << ',' << cli.M << ','
           << cli.ef_construction << ',' << ef_search << ',' << cli.k << ','
           << cli.rounds << ',' << std::fixed << std::setprecision(3) << qps
           << ',' << std::setprecision(6) << recall << ',' << stats.load_ms << ','
           << stats.train_ms << ',' << stats.build_ms << ',' << stats.save_ms
           << ',' << stats.reorder_ms << ',' << stats.cache_file << '\n';
}

void
warmup(
        faiss::Index* index,
        const std::vector<float>& queries,
        int dim,
        int nq,
        int k,
        int warmup_queries,
        int rounds) {
    const int count = std::min(nq, warmup_queries);
    if (count <= 0 || rounds <= 0) {
        return;
    }
    std::vector<float> distances(static_cast<size_t>(count) * k);
    std::vector<faiss::idx_t> labels(static_cast<size_t>(count) * k);
    for (int round = 0; round < rounds; ++round) {
        index->search(
                count,
                queries.data(),
                k,
                distances.data(),
                labels.data());
    }
    (void)dim;
}

} // namespace

int
main(int argc, char** argv) {
    try {
        const Cli cli = parse_cli(argc, argv);

        int nb = 0;
        int nq_available = 0;
        std::vector<float> base = load_f32(cli.base_file, cli.dim, nb);
        std::vector<float> queries =
                load_f32(cli.query_file, cli.dim, nq_available);
        if (cli.metric == "cosine") {
            normalize_rows(base, nb, cli.dim);
            normalize_rows(queries, nq_available, cli.dim);
        }
        const int nq = cli.query_limit > 0
                ? std::min(cli.query_limit, nq_available)
                : nq_available;
        queries.resize(static_cast<size_t>(nq) * cli.dim);
        if (cli.k > nb) {
            throw std::runtime_error("k must not exceed the number of base vectors");
        }

        // Generate optional on-the-fly ground truth through one deterministic
        // scalar backend before selecting S/V/M/F. This keeps the reference
        // labels identical across configurations even if their FP32 reduction
        // order differs. Paper-facing runs should still provide the shared
        // dataset ground-truth file via --gt.
        faiss::fvec_L2sqr = faiss::fvec_L2sqr_ref;
        faiss::fvec_L2sqr_batch_4 = faiss::fvec_L2sqr_batch_4_ref;
        faiss::fvec_inner_product = faiss::fvec_inner_product_ref;
        faiss::fvec_inner_product_batch_4 =
                faiss::fvec_inner_product_batch_4_ref;
        const std::vector<faiss::idx_t> truth =
                ground_truth(cli, base, nb, queries, nq);

        const EffectiveConfig effective = apply_config(cli);

        BuildStats stats;
        std::unique_ptr<faiss::Index> index =
                load_or_build(cli, effective, base, nb, stats);
        const std::vector<faiss::idx_t> new_to_old =
                apply_layout(index.get(), effective, stats);

        std::cout << "EFFECTIVE config=" << effective.key
                  << " name=" << effective.name
                  << " metric=" << cli.metric
                  << " backend=" << effective.backend
                  << " qtype=" << effective.qtype
                  << " reorder=" << effective.reorder
                  << " cache_layout=unordered"
                  << " decode_lmul=" << effective.decode_lmul
                  << " acc_lmul=" << effective.acc_lmul
                  << " lmul_tag=" << effective.lmul_tag
                  << " cache=" << stats.cache_file << '\n';

        auto* hnsw = dynamic_cast<faiss::IndexHNSW*>(index.get());
        if (!hnsw) {
            throw std::runtime_error("cached/built index is not HNSW");
        }
        for (int ef_search : cli.ef_search) {
            hnsw->hnsw.efSearch = ef_search;
            warmup(
                    index.get(),
                    queries,
                    cli.dim,
                    nq,
                    cli.k,
                    cli.warmup_queries,
                    cli.warmup_rounds);

            std::vector<float> distances(static_cast<size_t>(nq) * cli.k);
            std::vector<faiss::idx_t> labels(static_cast<size_t>(nq) * cli.k);
            const auto begin = std::chrono::steady_clock::now();
            for (int round = 0; round < cli.rounds; ++round) {
                index->search(
                        nq,
                        queries.data(),
                        cli.k,
                        distances.data(),
                        labels.data());
            }
            const auto end = std::chrono::steady_clock::now();
            const double seconds =
                    std::chrono::duration<double>(end - begin).count();
            const double qps = static_cast<double>(nq) * cli.rounds /
                    std::max(seconds, std::numeric_limits<double>::min());
            const double recall = recall_at_k(
                    labels, truth, nq, cli.k, new_to_old);
            append_csv(
                    cli,
                    effective,
                    stats,
                    nb,
                    nq,
                    ef_search,
                    qps,
                    recall);
            std::cout << std::fixed << std::setprecision(6)
                      << "RESULT config=" << effective.key
                      << " efS=" << ef_search << " qps=" << qps
                      << " recall=" << recall
                      << " load_ms=" << stats.load_ms
                      << " train_ms=" << stats.train_ms
                      << " build_ms=" << stats.build_ms
                      << " reorder_ms=" << stats.reorder_ms << '\n';
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "CQ1 driver failure: " << error.what() << '\n';
        return 1;
    }
}
