// Small, deterministic contract and performance check for the graph-layout
// methods used in the ROrder evaluation.  This test deliberately uses a
// synthetic HNSW-Flat index so it isolates physical layout from MPMI.

#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/impl/HNSW.h>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {

const char* const kMethods[] = {
        "unordered", "bfs", "rcm", "rabbitorder", "gorder", "rorder"};

struct Cli {
    std::string method = "all";
    std::string output_csv;
    int dim = 37;
    int nb = 512;
    int nq = 64;
    int M = 16;
    int ef_construction = 80;
    int ef_search = 64;
    int k = 10;
    int rounds = 3;
};

struct Result {
    std::string method;
    double qps = 0.0;
    double recall = 0.0;
    long long build_us = 0;
    long long reorder_us = 0;
    bool edge_set_preserved = false;
    bool permutation_valid = false;
    bool level0_sorted = false;
};

std::string
lower_ascii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

bool
is_supported_method(const std::string& method) {
    for (const char* candidate : kMethods) {
        if (method == candidate) {
            return true;
        }
    }
    return false;
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
        if (arg == "--method") {
            cli.method = lower_ascii(next(i));
        } else if (arg == "--output-csv") {
            cli.output_csv = next(i);
        } else if (arg == "--dim") {
            cli.dim = std::stoi(next(i));
        } else if (arg == "--nb") {
            cli.nb = std::stoi(next(i));
        } else if (arg == "--nq") {
            cli.nq = std::stoi(next(i));
        } else if (arg == "--M") {
            cli.M = std::stoi(next(i));
        } else if (arg == "--efC") {
            cli.ef_construction = std::stoi(next(i));
        } else if (arg == "--efS") {
            cli.ef_search = std::stoi(next(i));
        } else if (arg == "--k") {
            cli.k = std::stoi(next(i));
        } else if (arg == "--rounds") {
            cli.rounds = std::stoi(next(i));
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                    << "Usage: test_hnsw_reorder_micro [--method all|unordered|bfs|rcm|rabbitorder|gorder|rorder] "
                       "[--nb N] [--nq N] [--dim D] [--M M] [--efC N] [--efS N] [--k K] "
                       "[--rounds N] [--output-csv FILE]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    if (cli.method != "all" && !is_supported_method(cli.method)) {
        throw std::runtime_error("unsupported layout method: " + cli.method);
    }
    if (cli.dim <= 0 || cli.nb <= 0 || cli.nq <= 0 || cli.M <= 0 ||
        cli.k <= 0 || cli.k > cli.nb || cli.rounds <= 0) {
        throw std::runtime_error("all sizes must be positive and k must not exceed nb");
    }
    return cli;
}

std::vector<float>
make_vectors(int count, int dim, float shift) {
    std::vector<float> values(static_cast<size_t>(count) * dim);
    for (int row = 0; row < count; ++row) {
        for (int j = 0; j < dim; ++j) {
            const float x = static_cast<float>((row + 1) * (j + 3));
            values[static_cast<size_t>(row) * dim + j] =
                    std::sin(0.017f * x + shift) +
                    0.3f * std::cos(0.007f * x - shift) +
                    0.0001f * static_cast<float>(row);
        }
    }
    return values;
}

using Edge = std::tuple<int, faiss::idx_t, faiss::idx_t>;

std::multiset<Edge>
all_level_edges(
        const faiss::IndexHNSW& index,
        const std::vector<faiss::idx_t>& new_to_old) {
    const auto map_id = [&](faiss::idx_t id) -> faiss::idx_t {
        if (new_to_old.empty()) {
            return id;
        }
        if (id < 0 || static_cast<size_t>(id) >= new_to_old.size()) {
            throw std::runtime_error("neighbor ID is outside the permutation");
        }
        return new_to_old[static_cast<size_t>(id)];
    };

    std::multiset<Edge> edges;
    for (faiss::idx_t node = 0; node < index.ntotal; ++node) {
        const int level_count = index.hnsw.levels[node];
        for (int level = 0; level < level_count; ++level) {
            size_t begin = 0;
            size_t end = 0;
            index.hnsw.neighbor_range(node, level, &begin, &end);
            for (size_t pos = begin; pos < end; ++pos) {
                const faiss::HNSW::storage_idx_t neighbor =
                        index.hnsw.neighbors[pos];
                if (neighbor < 0) {
                    break;
                }
                edges.emplace(level, map_id(node), map_id(neighbor));
            }
        }
    }
    return edges;
}

void
verify_payload_permutation(
        const faiss::IndexHNSW& index,
        const std::vector<float>& original,
        int dim,
        const std::vector<faiss::idx_t>& new_to_old) {
    std::vector<float> reconstructed(dim, 0.0f);
    for (faiss::idx_t new_id = 0; new_id < index.ntotal; ++new_id) {
        const faiss::idx_t old_id = new_to_old.empty()
                ? new_id
                : new_to_old[static_cast<size_t>(new_id)];
        if (old_id < 0 || old_id >= index.ntotal) {
            throw std::runtime_error("payload permutation contains an invalid ID");
        }
        index.reconstruct(new_id, reconstructed.data());
        const float* expected =
                original.data() + static_cast<size_t>(old_id) * dim;
        for (int j = 0; j < dim; ++j) {
            if (reconstructed[j] != expected[j]) {
                throw std::runtime_error(
                        "payload[new_id] does not match payload[perm[new_id]]");
            }
        }
    }
}

void
verify_node_metadata_permutation(
        const faiss::IndexHNSW& index,
        const std::vector<faiss::idx_t>& new_to_old) {
    const auto* flat_storage =
            dynamic_cast<const faiss::IndexFlat*>(index.storage);
    if (!flat_storage ||
        flat_storage->code_norms.size() != static_cast<size_t>(index.ntotal)) {
        throw std::runtime_error("node metadata test fixture is incomplete");
    }
    for (faiss::idx_t new_id = 0; new_id < index.ntotal; ++new_id) {
        const faiss::idx_t old_id = new_to_old.empty()
                ? new_id
                : new_to_old[static_cast<size_t>(new_id)];
        const float expected = static_cast<float>(old_id) + 0.25f;
        if (flat_storage->code_norms[static_cast<size_t>(new_id)] != expected) {
            throw std::runtime_error(
                    "node metadata[new_id] does not match metadata[perm[new_id]]");
        }
    }
}

bool
valid_permutation(
        const std::vector<faiss::idx_t>& permutation,
        size_t expected_size) {
    if (permutation.size() != expected_size) {
        return false;
    }
    std::vector<unsigned char> seen(expected_size, 0);
    for (faiss::idx_t value : permutation) {
        if (value < 0 || static_cast<size_t>(value) >= expected_size ||
            seen[static_cast<size_t>(value)] != 0) {
            return false;
        }
        seen[static_cast<size_t>(value)] = 1;
    }
    return true;
}

bool
level0_neighbors_sorted(const faiss::IndexHNSW& index) {
    for (faiss::idx_t node = 0; node < index.ntotal; ++node) {
        size_t begin = 0;
        size_t end = 0;
        index.hnsw.neighbor_range(node, 0, &begin, &end);
        faiss::HNSW::storage_idx_t previous = -1;
        for (size_t pos = begin; pos < end; ++pos) {
            const faiss::HNSW::storage_idx_t neighbor =
                    index.hnsw.neighbors[pos];
            if (neighbor < 0) {
                continue;
            }
            if (previous >= 0 && neighbor < previous) {
                return false;
            }
            previous = neighbor;
        }
    }
    return true;
}

double
recall_at_k(
        const std::vector<faiss::idx_t>& result,
        const std::vector<faiss::idx_t>& ground_truth,
        int nq,
        int k,
        const std::vector<faiss::idx_t>& new_to_old) {
    double total = 0.0;
    for (int qi = 0; qi < nq; ++qi) {
        int hits = 0;
        for (int rank = 0; rank < k; ++rank) {
            faiss::idx_t predicted = result[static_cast<size_t>(qi) * k + rank];
            if (!new_to_old.empty() && predicted >= 0) {
                if (static_cast<size_t>(predicted) >= new_to_old.size()) {
                    throw std::runtime_error(
                            "search returned an ID outside the permutation");
                }
                predicted = new_to_old[static_cast<size_t>(predicted)];
            }
            const auto gt_begin = ground_truth.begin() + static_cast<size_t>(qi) * k;
            if (std::find(gt_begin, gt_begin + k, predicted) != gt_begin + k) {
                ++hits;
            }
        }
        total += static_cast<double>(hits) / k;
    }
    return total / nq;
}

Result
run_method(
        const Cli& cli,
        const std::string& method,
        const std::vector<float>& base,
        const std::vector<float>& queries,
        const std::vector<faiss::idx_t>& ground_truth) {
    Result result;
    result.method = method;

    faiss::IndexHNSWFlat index(cli.dim, cli.M, faiss::METRIC_L2);
    index.hnsw.efConstruction = cli.ef_construction;
    const auto build_begin = std::chrono::steady_clock::now();
    index.add(cli.nb, base.data());
    const auto build_end = std::chrono::steady_clock::now();
    result.build_us = std::chrono::duration_cast<std::chrono::microseconds>(
                              build_end - build_begin)
                              .count();

    auto* flat_storage = dynamic_cast<faiss::IndexFlat*>(index.storage);
    if (!flat_storage) {
        throw std::runtime_error("layout test requires IndexFlat storage");
    }
    flat_storage->code_norms.resize(static_cast<size_t>(cli.nb));
    for (int old_id = 0; old_id < cli.nb; ++old_id) {
        flat_storage->code_norms[static_cast<size_t>(old_id)] =
                static_cast<float>(old_id) + 0.25f;
    }

    const std::multiset<Edge> before = all_level_edges(index, {});
    std::vector<faiss::idx_t> permutation;
    if (method != "unordered") {
        const auto reorder_begin = std::chrono::steady_clock::now();
        index.reorder_graph_after_build(0, method.c_str(), true);
        const auto reorder_end = std::chrono::steady_clock::now();
        result.reorder_us = std::chrono::duration_cast<std::chrono::microseconds>(
                                    reorder_end - reorder_begin)
                                    .count();
        permutation = index.get_reorder_perm();
        result.permutation_valid =
                valid_permutation(permutation, static_cast<size_t>(cli.nb));
        if (!result.permutation_valid) {
            throw std::runtime_error(method + ": invalid perm[new_id]=old_id mapping");
        }

        bool repeated_layout_rejected = false;
        try {
            index.reorder_graph_after_build(0, method.c_str(), true);
        } catch (const std::exception&) {
            repeated_layout_rejected = true;
        }
        if (!repeated_layout_rejected) {
            throw std::runtime_error(
                    method + ": repeated post-build layout pass was accepted");
        }

        bool post_layout_add_rejected = false;
        try {
            index.add(1, base.data());
        } catch (const std::exception&) {
            post_layout_add_rejected = true;
        }
        if (!post_layout_add_rejected) {
            throw std::runtime_error(
                    method + ": vector addition after the layout pass was accepted");
        }
    } else {
        result.permutation_valid = index.get_reorder_perm().empty();
    }

    const std::multiset<Edge> after = all_level_edges(index, permutation);
    result.edge_set_preserved = before == after;
    if (!result.edge_set_preserved) {
        throw std::runtime_error(method + ": HNSW edge set changed");
    }
    verify_payload_permutation(index, base, cli.dim, permutation);
    verify_node_metadata_permutation(index, permutation);

    result.level0_sorted = level0_neighbors_sorted(index);
    if (method == "rorder" && !result.level0_sorted) {
        throw std::runtime_error(
                "rorder: level-0 adjacency lists are not normalized by remapped ID");
    }

    index.hnsw.efSearch = cli.ef_search;
    std::vector<float> distances(static_cast<size_t>(cli.nq) * cli.k);
    std::vector<faiss::idx_t> labels(static_cast<size_t>(cli.nq) * cli.k);
    index.search(cli.nq, queries.data(), cli.k, distances.data(), labels.data());

    const auto search_begin = std::chrono::steady_clock::now();
    for (int round = 0; round < cli.rounds; ++round) {
        index.search(
                cli.nq,
                queries.data(),
                cli.k,
                distances.data(),
                labels.data());
    }
    const auto search_end = std::chrono::steady_clock::now();
    const double seconds =
            std::chrono::duration<double>(search_end - search_begin).count();
    result.qps = static_cast<double>(cli.nq) * cli.rounds /
            std::max(seconds, 1.0e-12);
    result.recall = recall_at_k(
            labels, ground_truth, cli.nq, cli.k, permutation);
    if (!(result.recall >= 0.0 && result.recall <= 1.0)) {
        throw std::runtime_error(method + ": invalid recall");
    }
    return result;
}

void
append_csv(const Cli& cli, const Result& result) {
    if (cli.output_csv.empty()) {
        return;
    }
    std::ifstream existing(cli.output_csv);
    const bool write_header =
            !existing.good() || existing.peek() == std::char_traits<char>::eof();
    existing.close();
    std::ofstream output(cli.output_csv, std::ios::app);
    if (!output) {
        throw std::runtime_error("cannot open CSV: " + cli.output_csv);
    }
    if (write_header) {
        output << "method,nb,nq,dim,M,efC,efS,k,rounds,qps,recall,build_us,reorder_us,"
                  "edge_set_preserved,permutation_valid,level0_sorted\n";
    }
    output << result.method << ',' << cli.nb << ',' << cli.nq << ',' << cli.dim
           << ',' << cli.M << ',' << cli.ef_construction << ',' << cli.ef_search
           << ',' << cli.k << ',' << cli.rounds << ',' << std::fixed
           << std::setprecision(3) << result.qps << ',' << std::setprecision(6)
           << result.recall << ',' << result.build_us << ',' << result.reorder_us
           << ',' << result.edge_set_preserved << ',' << result.permutation_valid
           << ',' << result.level0_sorted << '\n';
}

} // namespace

int
main(int argc, char** argv) {
    try {
        const Cli cli = parse_cli(argc, argv);
        const std::vector<float> base = make_vectors(cli.nb, cli.dim, 0.0f);
        const std::vector<float> queries = make_vectors(cli.nq, cli.dim, 0.31f);

        faiss::IndexFlatL2 exact(cli.dim);
        exact.add(cli.nb, base.data());
        std::vector<float> gt_distances(static_cast<size_t>(cli.nq) * cli.k);
        std::vector<faiss::idx_t> ground_truth(
                static_cast<size_t>(cli.nq) * cli.k);
        exact.search(
                cli.nq,
                queries.data(),
                cli.k,
                gt_distances.data(),
                ground_truth.data());

        std::vector<std::string> methods;
        if (cli.method == "all") {
            methods.assign(std::begin(kMethods), std::end(kMethods));
        } else {
            methods.push_back(cli.method);
        }

        for (const std::string& method : methods) {
            const Result result =
                    run_method(cli, method, base, queries, ground_truth);
            append_csv(cli, result);
            std::cout << std::fixed << std::setprecision(6)
                      << "RESULT method=" << result.method
                      << " qps=" << result.qps
                      << " recall=" << result.recall
                      << " build_us=" << result.build_us
                      << " reorder_us=" << result.reorder_us
                      << " edge_set_preserved=" << result.edge_set_preserved
                      << " permutation_valid=" << result.permutation_valid
                      << " level0_sorted=" << result.level0_sorted << '\n';
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "reorder microbenchmark failure: " << error.what() << '\n';
        return 1;
    }
}
