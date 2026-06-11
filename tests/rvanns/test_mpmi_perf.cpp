// HNSW 批查询 Perf测试 - efSearch=512

#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/impl/HNSW.h>
#include <faiss/index_io.h>
#include <omp.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// -------------------- Perf 辅助函数 --------------------
struct PerfEventGroup {
    std::string name;
    std::string events;
};

static std::vector<PerfEventGroup>
get_perf_event_groups() {
    return {{"CPU Cycles", "cpu-cycles"},
            {"Instructions", "instructions"},
            {"Cache References", "cache-references"},
            {"Cache Misses", "cache-misses"},
            {"Branch Instructions", "branch-instructions"},
            {"Branch Misses", "branch-misses"},
            {"Stalled Cycles Frontend", "stalled-cycles-frontend"},
            {"Stalled Cycles Backend", "stalled-cycles-backend"},
            {"L1 DCache Loads", "L1-dcache-loads"},
            {"L1 DCache Load Misses", "L1-dcache-load-misses"},
            {"LLC Loads", "LLC-loads"},
            {"LLC Load Misses", "LLC-load-misses"},
            {"dTLB-load-misses", "dTLB-load-misses"}};
}

static pid_t
start_perf_stat(pid_t target_pid, const std::string& events, const std::string& output_file) {
    pid_t perf_pid = fork();
    if (perf_pid == 0) {
        freopen(output_file.c_str(), "w", stderr);
        execlp("perf", "perf", "stat", "-e", events.c_str(), "-p", std::to_string(target_pid).c_str(), nullptr);
        _exit(127);
    }
    return perf_pid;
}

static void
stop_perf_stat(pid_t perf_pid) {
    if (perf_pid > 0) {
        kill(perf_pid, SIGINT);
        int status;
        waitpid(perf_pid, &status, 0);
        usleep(500000);
    }
}

// -------------------- FS helpers --------------------
static bool
file_exists(const std::string& filename) {
    struct stat st {};
    return ::stat(filename.c_str(), &st) == 0;
}
static time_t
file_mtime(const std::string& filename) {
    struct stat st {};
    if (::stat(filename.c_str(), &st) == 0)
        return st.st_mtime;
    return 0;
}
static off_t
file_size(const std::string& filename) {
    struct stat st {};
    if (::stat(filename.c_str(), &st) == 0)
        return st.st_size;
    return 0;
}
static std::string
basename_of(const std::string& path) {
    auto p = path.find_last_of("/\\");
    return (p == std::string::npos) ? path : path.substr(p + 1);
}
static void
ensure_dir(const std::string& path) {
    std::ostringstream cmd;
    cmd << "mkdir -p '" << path << "' 2>/dev/null";
    (void)::system(cmd.str().c_str());
}

// -------------------- 路径 helpers --------------------
static std::string
gt_cache_dir() {
    return "vectors_out/gt_cache";
}
static std::string
index_cache_dir() {
    return "vectors_out/index_cache";
}
static std::string
result_dir() {
    return "vectors_out/results";
}

static std::string
sanitize_tag(const std::string& s) {
    std::string t = s;
    for (auto& c : t)
        if (c == '/' || c == '\\' || c == ' ' || c == ':')
            c = '_';
    return t;
}

static std::string
generate_gt_cache_filename(const std::string& name, int nb, int nq, int k) {
    ensure_dir(gt_cache_dir());
    std::ostringstream oss;
    oss << gt_cache_dir() << "/" << name << "_nb" << nb << "_nq" << nq << "_k" << k << "_flat_gt.ivecs";
    return oss.str();
}

// -------------------- I/O --------------------
static std::vector<float>
load_binary_data(const std::string& filename, int& num_vectors, int dim) {
    std::ifstream file(filename, std::ios::binary);
    if (!file)
        throw std::runtime_error("无法打开数据文件: " + filename);
    file.seekg(0, std::ios::end);
    size_t file_sz = (size_t)file.tellg();
    file.seekg(0, std::ios::beg);
    if (file_sz % (dim * sizeof(float)) != 0) {
        std::ostringstream oss;
        oss << "文件大小与维度不匹配: " << filename << " (size=" << file_sz << ", dim=" << dim << ")";
        throw std::runtime_error(oss.str());
    }
    num_vectors = (int)(file_sz / (dim * sizeof(float)));
    std::vector<float> data((size_t)num_vectors * dim);
    file.read(reinterpret_cast<char*>(data.data()), data.size() * sizeof(float));
    return data;
}

static std::vector<std::vector<int>>
load_ivecs(const std::string& filename, int k_limit, int& nq_out, int* d_first_row_out = nullptr) {
    std::ifstream f(filename, std::ios::binary);
    if (!f) {
        nq_out = 0;
        return {};
    }
    std::vector<std::vector<int>> rows;
    int first_d = -1;
    while (true) {
        int d = 0;
        if (!f.read(reinterpret_cast<char*>(&d), sizeof(int)))
            break;
        if (first_d < 0)
            first_d = d;
        std::vector<int> row(d);
        if (!f.read(reinterpret_cast<char*>(row.data()), d * sizeof(int)))
            break;
        if (k_limit > 0 && d > k_limit)
            row.resize(k_limit);
        rows.emplace_back(std::move(row));
    }
    nq_out = (int)rows.size();
    if (d_first_row_out)
        *d_first_row_out = first_d;
    return rows;
}

static bool
save_ivecs(const std::string& filename, const std::vector<std::vector<int>>& rows) {
    ensure_dir(gt_cache_dir());
    std::ofstream f(filename, std::ios::binary);
    if (!f)
        return false;
    for (const auto& r : rows) {
        int d = (int)r.size();
        f.write(reinterpret_cast<const char*>(&d), sizeof(int));
        f.write(reinterpret_cast<const char*>(r.data()), d * sizeof(int));
    }
    return true;
}

// -------------------- GT --------------------
static std::vector<std::vector<int>>
compute_flat_groundtruth(const std::vector<float>& base_vectors, const std::vector<float>& query_vectors, int dim,
                         int nb, int nq, int k, faiss::MetricType metric) {
    std::cout << "[GT] 使用 IndexFlat 现算 GT..." << std::flush;
    auto t0 = std::chrono::high_resolution_clock::now();

    faiss::Index* flat = nullptr;
    if (metric == faiss::METRIC_L2) {
        flat = new faiss::IndexFlatL2(dim);
    } else if (metric == faiss::METRIC_INNER_PRODUCT) {
        flat = new faiss::IndexFlatIP(dim);
    } else {
        throw std::runtime_error("不支持的 metric 类型");
    }

    flat->add(nb, base_vectors.data());

    std::vector<std::vector<int>> gts(nq);
    std::vector<float> D(k);
    std::vector<faiss::idx_t> I(k);
    for (int i = 0; i < nq; ++i) {
        const float* q = query_vectors.data() + (size_t)i * dim;
        flat->search(1, q, k, D.data(), I.data());
        gts[i].assign(I.begin(), I.end());
    }

    delete flat;

    auto t1 = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::cout << " 完成 (" << ms << " ms)\n";
    return gts;
}

static std::vector<std::vector<int>>
get_groundtruth(const std::vector<float>& base_vectors, const std::vector<float>& query_vectors, int dim, int nb,
                int nq, int k, const std::string& dataset_name, const std::string& gt_file_config,
                faiss::MetricType metric, long& gt_ms_out) {
    auto t0 = std::chrono::high_resolution_clock::now();

    if (!gt_file_config.empty() && file_exists(gt_file_config)) {
        int off_nq = 0, off_k = -1;
        auto all = load_ivecs(gt_file_config, 0, off_nq, &off_k);
        if (all.empty())
            throw std::runtime_error("官方GT读取失败或为空: " + gt_file_config);
        if (k > off_k)
            throw std::runtime_error("GT列数不足：要求k=" + std::to_string(k) + "，实际=" + std::to_string(off_k));
        if (nq > off_nq)
            throw std::runtime_error("GT行数不足：要求nq=" + std::to_string(nq) + "，实际=" + std::to_string(off_nq));
        std::vector<std::vector<int>> rows(nq);
        for (int i = 0; i < nq; ++i) rows[i].assign(all[i].begin(), all[i].begin() + k);
        auto t1 = std::chrono::high_resolution_clock::now();
        gt_ms_out = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        std::cout << "使用官方GT(裁剪) (耗时 " << gt_ms_out << " ms)\n";
        std::cout << "GT路径: " << gt_file_config << "\n";
        return rows;
    }

    auto rows = compute_flat_groundtruth(base_vectors, query_vectors, dim, nb, nq, k, metric);
    auto cache_file = generate_gt_cache_filename(dataset_name, nb, nq, k);
    (void)save_ivecs(cache_file, rows);
    auto t1 = std::chrono::high_resolution_clock::now();
    gt_ms_out = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::cout << "已计算并缓存 GT: " << cache_file << " (总耗时 " << gt_ms_out << " ms)\n";
    return rows;
}

// -------------------- metrics --------------------
static double
recall_at_k_one(const faiss::idx_t* pred_begin, const std::vector<int>& gt, int k) {
    if (k <= 0)
        return 0.0;
    int hit = 0;
    for (int i = 0; i < k; ++i) {
        faiss::idx_t p = pred_begin[i];
        if (p < 0)
            continue;
        for (int j = 0; j < k; ++j) {
            if (p == (faiss::idx_t)gt[j]) {
                ++hit;
                break;
            }
        }
    }
    return double(hit) / k;
}

struct RunOutcome {
    bool ok = false;
    double qps = 0.0;
    double avg_recall = 0.0;
    long build_ms = 0, gt_ms = 0, search_ms = 0, load_ms = 0;
};

// -------------------- 预热 --------------------
static void
warmup_index(faiss::Index* index, const std::vector<float>& query_vectors, int dim, int nq, int k, int warmup_rounds,
             int warmup_q) {
    std::vector<float> D(k);
    std::vector<faiss::idx_t> I(k);
    int take = std::min(nq, warmup_q);
    if (auto* h = dynamic_cast<faiss::IndexHNSW*>(index))
        h->hnsw.efSearch = 128;

    for (int r = 0; r < warmup_rounds; ++r) {
        for (int i = 0; i < take; ++i) {
            const float* q = query_vectors.data() + (size_t)i * dim;
            index->search(1, q, k, D.data(), I.data());
        }
    }
}

// -------------------- 批查询测试（带perf）--------------------
static RunOutcome
measure_batch_with_perf(faiss::Index* index, int efSearch, const std::vector<float>& query_vectors, int dim, int nq,
                        int k, const std::vector<std::vector<int>>& ground_truths) {
    RunOutcome out;
    out.ok = false;
    omp_set_num_threads(16);

    if (auto* h = dynamic_cast<faiss::IndexHNSW*>(index)) {
        h->hnsw.efSearch = efSearch;
    }

    std::vector<float> D((size_t)nq * k);
    std::vector<faiss::idx_t> I((size_t)nq * k);

    // ========== Perf 分析 ==========
    auto event_groups = get_perf_event_groups();
    pid_t my_pid = getpid();
    std::cout << "\n[Perf] 开始性能分析 (共 " << event_groups.size() << " 组)...\n";
    std::cout << "  模式：batch search(nq=" << nq << ")\n";
    std::cout << "  参数：efSearch=" << efSearch << "\n\n";

    std::chrono::duration<double> actual_search_duration;

    for (size_t i = 0; i < event_groups.size(); i++) {
        std::string perf_output = "perf_hnsw_batch512_group" + std::to_string(i + 1) + ".txt";
        std::cout << "[Perf " << (i + 1) << "/" << event_groups.size() << "] " << event_groups[i].name << " ..."
                  << std::flush;

        pid_t perf_pid = start_perf_stat(my_pid, event_groups[i].events, perf_output);
        usleep(1000000);

        auto search_start = std::chrono::high_resolution_clock::now();
        index->search(nq, query_vectors.data(), k, D.data(), I.data());
        auto search_end = std::chrono::high_resolution_clock::now();

        if (i == 0) {
            actual_search_duration = search_end - search_start;
        }

        usleep(1000000);
        stop_perf_stat(perf_pid);
        usleep(50000);
        std::cout << " 完成\n";

        off_t sz = file_size(perf_output);
        if (sz < 10) {
            std::cerr << "  警告: " << perf_output << " 文件太小 (" << sz << " 字节)\n";
            usleep(500000);
        }
    }

    // 计算统计数据
    out.search_ms = std::chrono::duration_cast<std::chrono::milliseconds>(actual_search_duration).count();
    out.qps = (double)nq / (actual_search_duration.count() + 1e-12);

    // 计算Recall
    double total_recall = 0.0;
    for (int i = 0; i < nq; ++i) {
        const faiss::idx_t* pred = &I[(size_t)i * k];
        total_recall += recall_at_k_one(pred, ground_truths[i], k);
    }
    out.avg_recall = total_recall / nq;

    // 打印Perf汇总
    std::cout << "\n========== Perf 统计汇总 ==========\n\n";
    for (size_t i = 0; i < event_groups.size(); i++) {
        std::string perf_output = "perf_hnsw_batch512_group" + std::to_string(i + 1) + ".txt";
        std::cout << "--- [组" << (i + 1) << "] " << event_groups[i].name << " ---\n";
        std::ifstream pf(perf_output);
        if (pf && file_size(perf_output) > 10) {
            std::string line;
            while (std::getline(pf, line)) {
                std::cout << line << "\n";
            }
        } else {
            std::cout << "(文件为空 - 可能perf未成功收集数据)\n";
        }
        std::cout << "\n";
    }
    std::cout << "=============================================\n\n";

    out.ok = true;
    return out;
}

// -------------------- CSV --------------------
static void
append_result_csv(const std::string& dataset_name, int nb, int nq, int dim, int M, int efC, int efS, int k,
                  int warmup_rounds, int warmup_q, const RunOutcome& out) {
    ensure_dir(result_dir());
    std::string path = result_dir() + "/hnsw_batch512_perf.csv";
    bool need_header = !file_exists(path);
    std::ofstream f(path, std::ios::app);
    if (!f)
        return;
    if (need_header) {
        f << "dataset,nb,nq,dim,M,efC,efS,k,warmup_rounds,warmup_q,"
          << "qps,recall,build_ms,load_ms,gt_ms,search_ms\n";
    }
    f << dataset_name << "," << nb << "," << nq << "," << dim << "," << M << "," << efC << "," << efS << "," << k << ","
      << warmup_rounds << "," << warmup_q << "," << std::fixed << std::setprecision(2) << out.qps << "," << std::fixed
      << std::setprecision(3) << out.avg_recall << "," << out.build_ms << "," << out.load_ms << "," << out.gt_ms << ","
      << out.search_ms << "\n";
}

// -------------------- CLI --------------------
struct Cli {
    std::string dataset;
    std::string name;
    std::string base_file;
    std::string query_file;
    std::string gt_file;
    std::string index_file;  // 直接指定索引文件
    int dim = 0;

    int M = 16;
    int efC = 256;
    int k = 100;
    int warmup_rounds = 2;
    int warmup_q = 1000;
    std::string metric = "l2";  // l2 or ip
};

static Cli
parse_cli(int argc, char** argv) {
    Cli c;
    for (int i = 1; i < argc; i++) {
        std::string a(argv[i]);
        auto next = [&]() -> std::string {
            if (i + 1 >= argc)
                throw std::runtime_error("缺少参数值: " + a);
            return std::string(argv[++i]);
        };
        if (a == "--dataset")
            c.dataset = next();
        else if (a == "--name")
            c.name = next();
        else if (a == "--base")
            c.base_file = next();
        else if (a == "--query")
            c.query_file = next();
        else if (a == "--gt")
            c.gt_file = next();
        else if (a == "--index")
            c.index_file = next();
        else if (a == "--dim")
            c.dim = std::stoi(next());
        else if (a == "--M")
            c.M = std::stoi(next());
        else if (a == "--efC")
            c.efC = std::stoi(next());
        else if (a == "--k")
            c.k = std::stoi(next());
        else if (a == "--warmup_rounds")
            c.warmup_rounds = std::stoi(next());
        else if (a == "--warmup_q")
            c.warmup_q = std::stoi(next());
        else if (a == "--metric")
            c.metric = next();
        else
            throw std::runtime_error("未知参数: " + a);
    }
    return c;
}

// -------------------- 预置数据集 --------------------
struct DatasetDesc {
    std::string name, base_file, query_file, gt_file, index_file, metric;
    int dim;
};

static std::vector<DatasetDesc>
all_presets() {
    return {{"SIFT1M", "vectors_out/sift1m_base.bin", "vectors_out/sift1m_query.bin",
             "vectors_out/sift1m_groundtruth.ivecs",
             "vectors_out/index_cache/SIFT1M_hnsw_nb1000000_dim128_M16_efC256_l2_sift1m_base.bin_512000000.faiss", "l2",
             128},
            {"GIST1M", "vectors_out/gist1m_base.bin", "vectors_out/gist1m_query.bin",
             "vectors_out/gist1m_groundtruth.ivecs",
             "vectors_out/index_cache/GIST1M_hnsw_nb1000000_dim960_M16_efC256_l2_gist1m_base.bin_3840000000.faiss",
             "l2", 960},
            {"Cohere10M-IP", "vectors_out/unit_cohere_10m_base.bin", "vectors_out/unit_cohere_10m_query.bin",
             "vectors_out/cohere_10m_groundtruth.ivecs",
             "vectors_out/index_cache/"
             "Cohere10M-IP-GTinternal-20251109_hnsw_nb10000000_dim768_M16_efC256_ip_unit_cohere_10m_base.bin_"
             "30720000000.faiss",
             "ip", 768}};
}

// -------------------- 主测试函数 --------------------
static void
run_one_dataset(const std::string& name, const std::string& base_file, const std::string& query_file,
                const std::string& gt_file, const std::string& index_file, const std::string& metric_str, int dim,
                const Cli& cli) {
    std::cout << "\n==== 数据集: " << name << " ====\n";
    std::cout << "索引文件: " << index_file << "\n";
    std::cout << "度量类型: " << metric_str << "\n\n";

    if (!file_exists(base_file) || !file_exists(query_file)) {
        throw std::runtime_error("数据文件不存在（" + base_file + " 或 " + query_file + "）");
    }

    if (!file_exists(index_file)) {
        throw std::runtime_error("索引文件不存在: " + index_file);
    }

    faiss::MetricType metric = (metric_str == "ip") ? faiss::METRIC_INNER_PRODUCT : faiss::METRIC_L2;

    int nb = 0, nq = 0;
    auto base = load_binary_data(base_file, nb, dim);
    auto query = load_binary_data(query_file, nq, dim);

    // 加载索引
    std::cout << "加载索引..." << std::flush;
    auto t0 = std::chrono::high_resolution_clock::now();
    std::unique_ptr<faiss::Index> index(faiss::read_index(index_file.c_str()));
    auto t1 = std::chrono::high_resolution_clock::now();
    long load_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::cout << " 完成 (" << load_ms << " ms)\n";
    std::cout << "索引类型: " << (dynamic_cast<faiss::IndexHNSW*>(index.get()) ? "HNSW" : "Unknown") << "\n";
    std::cout << "向量数量: " << index->ntotal << ", 维度: " << index->d << "\n";

    long gt_ms = 0;
    auto gts = get_groundtruth(base, query, dim, nb, nq, cli.k, name, gt_file, metric, gt_ms);

    warmup_index(index.get(), query, dim, nq, cli.k, cli.warmup_rounds, std::min(cli.warmup_q, nq));

    // 测试 efSearch=512
    const int efSearch = 512;
    std::cout << "\n========== 开始测试 efSearch=" << efSearch << " ==========\n";

    auto out = measure_batch_with_perf(index.get(), efSearch, query, dim, nq, cli.k, gts);

    out.build_ms = 0;  // 使用预构建索引,无构建时间
    out.load_ms = load_ms;
    out.gt_ms = gt_ms;

    // 输出结果
    std::cout << "\n========== 测试结果 ==========\n";
    std::cout << "QPS (批查询): " << std::fixed << std::setprecision(2) << out.qps << "\n";
    std::cout << "Recall@" << cli.k << ": " << std::fixed << std::setprecision(3) << out.avg_recall << "\n";
    std::cout << "==============================\n\n";

    // 写CSV
    append_result_csv(name, nb, nq, dim, cli.M, cli.efC, efSearch, cli.k, cli.warmup_rounds, std::min(cli.warmup_q, nq),
                      out);
}

// -------------------- main --------------------
int
main(int argc, char** argv) {
    try {
        auto cli = parse_cli(argc, argv);
        std::cout << "HNSW 批查询 Perf测试\n";
        std::cout << "固定参数: efSearch=512, 批查询模式\n";
        std::cout << "*** 适用于预构建的 HNSW 索引 ***\n\n";

        if (cli.dataset == "CUSTOM") {
            if (cli.name.empty())
                throw std::runtime_error("CUSTOM 需要 --name");
            if (cli.base_file.empty() || cli.query_file.empty() || cli.dim <= 0)
                throw std::runtime_error("CUSTOM 需要 --base, --query, --dim");
            if (cli.index_file.empty())
                throw std::runtime_error("CUSTOM 需要 --index 指定索引文件路径");
            run_one_dataset(cli.name, cli.base_file, cli.query_file, cli.gt_file, cli.index_file, cli.metric, cli.dim,
                            cli);
            std::cout << "\n结果已追加到: " << result_dir() << "/hnsw_batch512_perf.csv\n";
            return 0;
        }

        if (cli.dataset == "ALL") {
            auto presets = all_presets();
            for (const auto& ds : presets) {
                run_one_dataset(ds.name, ds.base_file, ds.query_file, ds.gt_file, ds.index_file, ds.metric, ds.dim,
                                cli);
            }
            std::cout << "\n结果已追加到: " << result_dir() << "/hnsw_batch512_perf.csv\n";
            return 0;
        }

        auto presets = all_presets();
        if (cli.dataset.empty()) {
            std::cout << "\n数据集:\n";
            for (size_t i = 0; i < presets.size(); ++i) std::cout << (i + 1) << ". " << presets[i].name << "\n";
            std::cout << "选择: ";
            int choice = 0;
            std::cin >> choice;
            if (choice < 1 || choice > (int)presets.size()) {
                std::cerr << "无效选择\n";
                return 1;
            }
            const auto& ds = presets[choice - 1];
            run_one_dataset(ds.name, ds.base_file, ds.query_file, ds.gt_file, ds.index_file, ds.metric, ds.dim, cli);
            std::cout << "\n结果已追加到: " << result_dir() << "/hnsw_batch512_perf.csv\n";
            return 0;
        } else {
            auto it = std::find_if(presets.begin(), presets.end(),
                                   [&](const DatasetDesc& d) { return d.name == cli.dataset; });
            if (it == presets.end()) {
                std::ostringstream oss;
                oss << "未知 --dataset: " << cli.dataset << "（可选：SIFT1M|GIST1M|Cohere10M-IP|ALL|CUSTOM）";
                throw std::runtime_error(oss.str());
            }
            const auto& ds = *it;
            run_one_dataset(ds.name, ds.base_file, ds.query_file, ds.gt_file, ds.index_file, ds.metric, ds.dim, cli);
            std::cout << "\n结果已追加到: " << result_dir() << "/hnsw_batch512_perf.csv\n";
            return 0;
        }
    } catch (const std::exception& e) {
        std::cerr << "\n[ERROR] " << e.what() << "\n";
        return 1;
    }
}
