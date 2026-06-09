// test_hnsw_fix_IP_L2.cpp
// HNSW（faiss::IndexHNSWFlat，度量为 Inner Product），一次建索引，循环 efSearch，支持官方GT或Flat回退。
// 与 SQ 版本一致：数据在内存中先做 L2 单位化 => 用 IP 等价于余弦相似。
// 目录约定：vectors_out/{gt_cache,index_cache,results} 自动创建。
// 预置数据集：SIFT1M(128), GIST1M(960), GloVe1M(100) —— 路径与您当前仓库一致。

#include <faiss/IndexHNSW.h>
#include <faiss/IndexFlat.h>
#include <faiss/impl/HNSW.h>
#include <faiss/index_io.h>

#include <algorithm>
#include <chrono>
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
#include <unistd.h>
#include <string>
#include <sys/stat.h>
#include <vector>
#include <cmath>
#include <omp.h>
// -------------------- FS helpers --------------------
static bool file_exists(const std::string& filename) {
    struct stat st{}; return ::stat(filename.c_str(), &st) == 0;
}
static time_t file_mtime(const std::string& filename) {
    struct stat st{}; if (::stat(filename.c_str(), &st) == 0) return st.st_mtime; return 0;
}
static off_t file_size(const std::string& filename) {
    struct stat st{}; if (::stat(filename.c_str(), &st) == 0) return st.st_size; return 0;
}
static std::string basename_of(const std::string& path) {
    auto p = path.find_last_of("/\\"); return (p == std::string::npos) ? path : path.substr(p + 1);
}
static void ensure_dir(const std::string& path) {
    std::ostringstream cmd; cmd << "mkdir -p '" << path << "' 2>/dev/null"; (void)::system(cmd.str().c_str());
}

// -------------------- 路径 helpers --------------------
static std::string gt_cache_dir()    { return "vectors_out/gt_cache";    }
static std::string index_cache_dir() { return "vectors_out/index_cache"; }
static std::string result_dir()      { return "vectors_out/results";     }

static std::string sanitize_tag(const std::string& s) {
    std::string t = s;
    for (auto& c : t) if (c=='/' || c=='\\' || c==' ' || c==':') c = '_';
    return t;
}

static std::string generate_gt_cache_filename(const std::string& name, int nb, int nq, int k) {
    ensure_dir(gt_cache_dir());
    std::ostringstream oss;
    oss << gt_cache_dir() << "/" << name
        << "_nb" << nb << "_nq" << nq << "_k" << k << "_flat_gt.ivecs";
    return oss.str();
}

static std::string generate_index_cache_filename_hnsw(
    const std::string& name, int dim, int nb, int M, int efC,
    const std::string& metric, const std::string& base_path)
{
    ensure_dir(index_cache_dir());
    std::ostringstream oss;
    std::string base_tag = sanitize_tag(basename_of(base_path)) + "_" +
                           std::to_string((long long)file_size(base_path));
    oss << index_cache_dir() << "/" << name
        << "_hnsw_nb" << nb
        << "_dim" << dim
        << "_M" << M
        << "_efC" << efC
        << "_" << metric
        << "_" << base_tag
        << ".faiss";
    return oss.str();
}

// -------------------- I/O --------------------
static std::vector<float> load_binary_data(const std::string& filename, int& num_vectors, int dim) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) throw std::runtime_error("无法打开数据文件: " + filename);
    file.seekg(0, std::ios::end);
    size_t file_sz = (size_t)file.tellg();
    file.seekg(0, std::ios::beg);
    if (file_sz % (dim * sizeof(float)) != 0) {
        std::ostringstream oss; oss << "文件大小与维度不匹配: " << filename
                                    << " (size=" << file_sz << ", dim=" << dim << ")";
        throw std::runtime_error(oss.str());
    }
    num_vectors = (int)(file_sz / (dim * sizeof(float)));
    std::vector<float> data((size_t)num_vectors * dim);
    file.read(reinterpret_cast<char*>(data.data()), data.size() * sizeof(float));
    return data;
}

static std::vector<std::vector<int>> load_ivecs(const std::string& filename, int k_limit, int& nq_out, int* d_first_row_out=nullptr) {
    std::ifstream f(filename, std::ios::binary);
    if (!f) { nq_out = 0; return {}; }
    std::vector<std::vector<int>> rows;
    int first_d = -1;
    while (true) {
        int d = 0;
        if (!f.read(reinterpret_cast<char*>(&d), sizeof(int))) break;
        if (first_d < 0) first_d = d;
        std::vector<int> row(d);
        if (!f.read(reinterpret_cast<char*>(row.data()), d * sizeof(int))) break;
        if (k_limit > 0 && d > k_limit) row.resize(k_limit);
        rows.emplace_back(std::move(row));
    }
    nq_out = (int)rows.size();
    if (d_first_row_out) *d_first_row_out = first_d;
    return rows;
}

static bool save_ivecs(const std::string& filename, const std::vector<std::vector<int>>& rows) {
    ensure_dir(gt_cache_dir());
    std::ofstream f(filename, std::ios::binary);
    if (!f) return false;
    for (const auto& r : rows) {
        int d = (int)r.size();
        f.write(reinterpret_cast<const char*>(&d), sizeof(int));
        f.write(reinterpret_cast<const char*>(r.data()), d * sizeof(int));
    }
    return true;
}

// -------------------- 单位化（与 SQ 版一致） --------------------
static void l2_normalize_inplace(std::vector<float>& data, int n, int d) {
    for (int i = 0; i < n; ++i) {
        float* row = data.data() + (size_t)i * d;
        double s = 0.0;
        for (int j = 0; j < d; ++j) s += (double)row[j] * (double)row[j];
        if (s > 0.0) {
            float inv = 1.0f / (float)std::sqrt(s);
            for (int j = 0; j < d; ++j) row[j] *= inv;
        }
    }
}

// -------------------- GT --------------------
static std::vector<std::vector<int>> compute_flat_groundtruth(
    const std::vector<float>& base_vectors,
    const std::vector<float>& query_vectors,
    int dim, int nb, int nq, int k)
{
    std::cout << "[GT] 使用 IndexFlatIP 现算 GT..." << std::flush;
    auto t0 = std::chrono::high_resolution_clock::now();

    faiss::IndexFlatL2 flat(dim);
    flat.add(nb, base_vectors.data());

    std::vector<std::vector<int>> gts(nq);
    std::vector<float> D(k);
    std::vector<faiss::idx_t> I(k);
    for (int i = 0; i < nq; ++i) {
        const float* q = query_vectors.data() + (size_t)i * dim;
        flat.search(1, q, k, D.data(), I.data());
        gts[i].assign(I.begin(), I.end());
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::cout << " 完成 (" << ms << " ms)\n";
    return gts;
}

static std::vector<std::vector<int>> get_groundtruth(
    const std::vector<float>& base_vectors,
    const std::vector<float>& query_vectors,
    int dim, int nb, int nq, int k,
    const std::string& dataset_name,
    const std::string& gt_file_config, // 若非空且存在就强制用
    long& gt_ms_out)
{
    auto t0 = std::chrono::high_resolution_clock::now();

    if (!gt_file_config.empty() && file_exists(gt_file_config)) {
        int off_nq=0, off_k=-1;
        auto all = load_ivecs(gt_file_config, 0, off_nq, &off_k);
        if (all.empty()) throw std::runtime_error("官方GT读取失败或为空: " + gt_file_config);
        if (k > off_k)   throw std::runtime_error("GT列数不足：要求k=" + std::to_string(k) + "，实际=" + std::to_string(off_k));
        if (nq > off_nq) throw std::runtime_error("GT行数不足：要求nq=" + std::to_string(nq) + "，实际=" + std::to_string(off_nq));
        std::vector<std::vector<int>> rows(nq);
        for (int i=0;i<nq;++i) rows[i].assign(all[i].begin(), all[i].begin()+k);
        auto t1 = std::chrono::high_resolution_clock::now();
        gt_ms_out = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        std::cout << "使用官方GT(裁剪) (耗时 " << gt_ms_out << " ms)\n";
        std::cout << "GT路径: " << gt_file_config << "\n";
        return rows;
    }

    // Flat 回退
    auto rows = compute_flat_groundtruth(base_vectors, query_vectors, dim, nb, nq, k);
    auto cache_file = generate_gt_cache_filename(dataset_name, nb, nq, k);
    (void)save_ivecs(cache_file, rows);
    auto t1 = std::chrono::high_resolution_clock::now();
    gt_ms_out = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::cout << "已计算并缓存 GT: " << cache_file << " (总耗时 " << gt_ms_out << " ms)\n";
    return rows;
}

// -------------------- metrics --------------------
static double recall_at_k_one(const std::vector<faiss::idx_t>& pred, const std::vector<int>& gt, int k) {
    if ((int)pred.size() < k || (int)gt.size() < k) return 0.0;
    int hit = 0;
    for (int i = 0; i < k; i++) {
        faiss::idx_t p = pred[i];
        for (int j = 0; j < k; j++) {
            if (p == (faiss::idx_t)gt[j]) { hit++; break; }
        }
    }
    return double(hit) / k;
}
static double percentile_sorted(const std::vector<double>& sorted_latencies, double percentile) {
    if (sorted_latencies.empty()) return 0.0;
    double p = std::clamp(percentile, 0.0, 100.0) / 100.0;
    double pos = p * (sorted_latencies.size() - 1);
    size_t idx = (size_t)std::ceil(pos);
    if (idx >= sorted_latencies.size()) idx = sorted_latencies.size() - 1;
    return sorted_latencies[idx];
}

struct RunOutcome {
    bool ok = false;
    double qps = 0.0;
    double avg_recall = 0.0;
    long train_ms = 0, build_ms = 0, gt_ms = 0, search_ms = 0, load_ms = 0, save_ms = 0;
    double p95_latency_us = 0.0, p99_latency_us = 0.0, avg_latency_us = 0.0;
};

// -------------------- 索引构建/加载（HNSW-IP） --------------------
static std::unique_ptr<faiss::Index> build_or_load_hnswIP(
    const std::vector<float>& base,
    int dim, int nb,
    int M, int efConstruction,
    const std::string& dataset_name,
    const std::string& base_path_for_tag,
    long& load_ms_out, long& build_ms_out, long& save_ms_out, long& train_ms_out)
{
    const std::string metric = "l2";  // 与 SQ 版保持一致，避免混入“未单位化”的旧缓存
    std::string idx_cache = generate_index_cache_filename_hnsw(
        dataset_name, dim, nb, M, efConstruction, metric, base_path_for_tag);

    auto t0 = std::chrono::high_resolution_clock::now();
    std::unique_ptr<faiss::Index> index;
    if (file_exists(idx_cache)) {
        index.reset(faiss::read_index(idx_cache.c_str()));
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    load_ms_out = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

    if (index) {
        if ((int)index->d == dim && (int)index->ntotal == nb) {
            if (dynamic_cast<faiss::IndexHNSW*>(index.get())) {
                std::cout << "已加载缓存索引: " << idx_cache << " (" << load_ms_out << " ms), ntotal=" << index->ntotal << "\n";
                build_ms_out = 0; save_ms_out = 0; train_ms_out = 0;
                return index;
            }
        }
        index.reset();
    }

    std::cout << "开始构建 HNSW(IP) 索引 (nb=" << nb << ", dim=" << dim
              << ", M=" << M << ", efC=" << efConstruction << ")...\n";

    // 构造：HNSW-Flat + Inner Product
    auto* hnsw = new faiss::IndexHNSWFlat(dim, M, faiss::METRIC_L2);
    hnsw->hnsw.efConstruction = efConstruction;
    std::unique_ptr<faiss::Index> idx(hnsw);

    train_ms_out = 0; // HNSWFlat 无 train 步骤
    auto b0 = std::chrono::high_resolution_clock::now();
    idx->add(nb, base.data());
    auto b1 = std::chrono::high_resolution_clock::now();
    build_ms_out = std::chrono::duration_cast<std::chrono::milliseconds>(b1 - b0).count();
    std::cout << "构建完成: " << build_ms_out << " ms\n";

    auto s0 = std::chrono::high_resolution_clock::now();
    faiss::write_index(idx.get(), idx_cache.c_str());
    auto s1 = std::chrono::high_resolution_clock::now();
    save_ms_out = std::chrono::duration_cast<std::chrono::milliseconds>(s1 - s0).count();
    std::cout << "已缓存索引至: " << idx_cache << " (" << save_ms_out << " ms)\n";

    return idx;
}

// -------------------- 预热 & 评测 --------------------
static void warmup_index(
    faiss::Index* index,
    const std::vector<float>& query_vectors,
    int dim, int nq, int k,
    int warmup_rounds, int warmup_q)
{
    std::vector<float> D(k);
    std::vector<faiss::idx_t> I(k);
    int take = std::min(nq, warmup_q);
    // 固定用 efSearch=128 预热
    if (auto* h = dynamic_cast<faiss::IndexHNSW*>(index)) h->hnsw.efSearch = 128;

    for (int r = 0; r < warmup_rounds; ++r) {
        for (int i = 0; i < take; ++i) {
            const float* q = query_vectors.data() + (size_t)i * dim;
            index->search(1, q, k, D.data(), I.data());
        }
    }
}

static RunOutcome measure_once(
    faiss::Index* index,
    int efSearch,
    const std::vector<float>& query_vectors,
    int dim, int nq, int k,
    const std::vector<std::vector<int>>& ground_truths,
    bool batch_mode)   // true: 整批；false: 单条
{
    RunOutcome out; out.ok = false;
    omp_set_num_threads(16);
    if (auto *h = dynamic_cast<faiss::IndexHNSW*>(index)) {
        h->hnsw.efSearch = efSearch;
    }

    if (batch_mode) {
        // 整批 search(nq)
        std::vector<float> D((size_t)nq * k);
        std::vector<faiss::idx_t> I((size_t)nq * k);

        auto t0 = std::chrono::high_resolution_clock::now();
        index->search(nq, query_vectors.data(), k, D.data(), I.data());
        auto t1 = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> dt_sec = t1 - t0;
        out.search_ms = (long)std::llround(std::chrono::duration<double, std::milli>(dt_sec).count());
        out.qps = (double)nq / (dt_sec.count() + 1e-12);

        // Recall
        double total_recall = 0.0;
        std::vector<faiss::idx_t> pred(k);
        for (int i = 0; i < nq; ++i) {
            std::memcpy(pred.data(), &I[(size_t)i * k], (size_t)k * sizeof(faiss::idx_t));
            total_recall += recall_at_k_one(pred, ground_truths[i], k);
        }
        out.avg_recall = total_recall / nq;

        // 延迟不可用
        out.avg_latency_us = out.p95_latency_us = out.p99_latency_us = -1.0;

        out.ok = true;
        return out;
    } else {
        // 单条 search(1) + 延迟分布
        std::vector<float> D(k);
        std::vector<faiss::idx_t> I(k);
        std::vector<double> latencies_us;

        auto search_start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < nq; ++i) {
            const float* q = query_vectors.data() + (size_t)i * dim;
            auto q0 = std::chrono::high_resolution_clock::now();
            index->search(1, q, k, D.data(), I.data());
            auto q1 = std::chrono::high_resolution_clock::now();
            double us = std::chrono::duration_cast<std::chrono::nanoseconds>(q1 - q0).count()/1000.0;
            latencies_us.push_back(us);
        }
        auto search_end = std::chrono::high_resolution_clock::now();
        out.search_ms = std::chrono::duration_cast<std::chrono::milliseconds>(search_end - search_start).count();
        out.qps = (double)nq / (out.search_ms/1000.0 + 1e-9);

        out.avg_latency_us = std::accumulate(latencies_us.begin(), latencies_us.end(), 0.0) / latencies_us.size();
        std::vector<double> sorted_lat = latencies_us;
        std::sort(sorted_lat.begin(), sorted_lat.end());
        out.p95_latency_us = percentile_sorted(sorted_lat, 95.0);
        out.p99_latency_us = percentile_sorted(sorted_lat, 99.0);

        // Recall（独立测，避免计时干扰）
        std::vector<double> recalls;
        for (int i = 0; i < nq; ++i) {
            const float* q = query_vectors.data() + (size_t)i * dim;
            index->search(1, q, k, D.data(), I.data());
            recalls.push_back(recall_at_k_one(I, ground_truths[i], k));
        }
        out.avg_recall = std::accumulate(recalls.begin(), recalls.end(), 0.0) / recalls.size();
        out.ok = true;
        return out;
    }
}

// -------------------- CSV --------------------
static void append_result_csv(
    const std::string& dataset_name, int nb, int nq, int dim,
    int M, int efC, int efS, int k, int warmup_rounds, int warmup_q,
    const RunOutcome& out)
{
    ensure_dir(result_dir());
    std::string path = result_dir() + "/hnsw_efS_sweep.csv";
    bool need_header = !file_exists(path);
    std::ofstream f(path, std::ios::app);
    if (!f) return;
    if (need_header) {
        f << "dataset,nb,nq,dim,M,efC,efS,k,warmup_rounds,warmup_q,qps,avg_us,p95_us,p99_us,recall,build_ms,load_ms,save_ms,gt_ms,train_ms,search_ms\n";
    }
    f << dataset_name << "," << nb << "," << nq << "," << dim << ","
      << M << "," << efC << "," << efS << "," << k << ","
      << warmup_rounds << "," << warmup_q << ","
      << std::fixed << std::setprecision(2) << out.qps << ","
      << std::fixed << std::setprecision(2) << out.avg_latency_us << ","
      << std::fixed << std::setprecision(2) << out.p95_latency_us << ","
      << std::fixed << std::setprecision(2) << out.p99_latency_us << ","
      << std::fixed << std::setprecision(3) << out.avg_recall << ","
      << out.build_ms << "," << out.load_ms << "," << out.save_ms << ","
      << out.gt_ms << "," << out.train_ms << "," << out.search_ms << "\n";
}

// -------------------- CLI --------------------
struct Cli {
    std::string dataset;       // SIFT1M | GIST1M | GloVe1M | ALL | CUSTOM | ""(交互)
    std::string name;          // CUSTOM 的显示名
    std::string base_file;
    std::string query_file;
    std::string gt_file;       // 可空 => Flat 回退
    int dim = 0;

    std::vector<int> efS_list{16,32,64,128,256,512};
    int M = 16;
    int efC = 256;
    int k = 100;
    int warmup_rounds = 2;
    int warmup_q = 1000;

    bool batch_only = false;
    bool single_only = false;
};

static std::vector<int> parse_efs(const std::string& s) {
    std::vector<int> v; std::stringstream ss(s); std::string tok;
    while (std::getline(ss, tok, ',')) if (!tok.empty()) v.push_back(std::stoi(tok));
    return v.empty() ? std::vector<int>{16,32,64,128,256,512} : v;
}

static Cli parse_cli(int argc, char** argv) {
    Cli c;
    for (int i=1;i<argc;i++) {
        std::string a(argv[i]);
        auto next = [&]() -> std::string {
            if (i+1>=argc) throw std::runtime_error("缺少参数值: " + a);
            return std::string(argv[++i]);
        };
        if (a=="--dataset") c.dataset = next();
        else if (a=="--name") c.name = next();
        else if (a=="--base") c.base_file = next();
        else if (a=="--query") c.query_file = next();
        else if (a=="--gt") c.gt_file = next();
        else if (a=="--dim") c.dim = std::stoi(next());
        else if (a=="--efs") c.efS_list = parse_efs(next());
        else if (a=="--M") c.M = std::stoi(next());
        else if (a=="--efC") c.efC = std::stoi(next());
        else if (a=="--k") c.k = std::stoi(next());
        else if (a=="--warmup_rounds") c.warmup_rounds = std::stoi(next());
        else if (a=="--warmup_q") c.warmup_q = std::stoi(next());
        else if (a=="--batch") c.batch_only = true;
        else if (a=="--single") c.single_only = true;
        else throw std::runtime_error("未知参数: " + a);
    }
    return c;
}

// -------------------- 预置数据集 --------------------
struct DatasetDesc { std::string name, base_file, query_file, gt_file; int dim; };
static std::vector<DatasetDesc> all_presets() {
    return {
        {"SIFT1M",  "vectors_out/sift1m_base.bin",  "vectors_out/sift1m_query.bin",  "vectors_out/sift1m_groundtruth.ivecs", 128},
        {"GIST1M",  "vectors_out/gist1m_base.bin",  "vectors_out/gist1m_query.bin",  "vectors_out/gist1m_groundtruth.ivecs", 960},
        {"GloVe1M", "vectors_out/glove1m_base.bin", "vectors_out/glove1m_query.bin", "vectors_out/glove1m_groundtruth.ivecs", 100}
    };
}

// -------------------- 跑一个数据集 --------------------
static void run_one_dataset(const std::string& name,
                            const std::string& base_file,
                            const std::string& query_file,
                            const std::string& gt_file,
                            int dim,
                            const Cli& cli)
{
    std::cout << "\n==== 数据集: " << name << " ====\n";
    if (!file_exists(base_file) || !file_exists(query_file)) {
        throw std::runtime_error("数据文件不存在（" + base_file + " 或 " + query_file + "）");
    }

    int nb=0, nq=0;
    auto base  = load_binary_data(base_file,  nb, dim);
    auto query = load_binary_data(query_file, nq, dim);

    // 就地 L2 单位化（只改内存，不改磁盘）
    //l2_normalize_inplace(base,  nb, dim);
    //l2_normalize_inplace(query, nq, dim);

    long load_ms=0, build_ms=0, save_ms=0, train_ms=0;
    auto index = build_or_load_hnswIP(base, dim, nb, cli.M, cli.efC,
                                      name, base_file,
                                      load_ms, build_ms, save_ms, train_ms);

    long gt_ms=0;
    auto gts = get_groundtruth(base, query, dim, nb, nq, cli.k, name, gt_file, gt_ms);

    warmup_index(index.get(), query, dim, nq, cli.k, cli.warmup_rounds, std::min(cli.warmup_q, nq));

    for (int efS : cli.efS_list) {
        const bool do_batch  = !cli.single_only;
        const bool do_single = !cli.batch_only;

        RunOutcome out_batch, out_single;
        if (do_batch)  out_batch  = measure_once(index.get(), efS, query, dim, nq, cli.k, gts, /*batch=*/true);
        if (do_single) out_single = measure_once(index.get(), efS, query, dim, nq, cli.k, gts, /*batch=*/false);

        auto patch = [&](RunOutcome& o){
            o.build_ms = build_ms; o.load_ms = load_ms; o.save_ms = save_ms; o.gt_ms = gt_ms; o.train_ms = train_ms;
        };
        if (do_batch)  patch(out_batch);
        if (do_single) patch(out_single);

        std::cout << "[efS=" << efS << "] ";
        if (do_single) {
            std::cout << "QPS_single=" << std::fixed << std::setprecision(2) << out_single.qps
                      << ", Recall_single@" << cli.k << "=" << std::fixed << std::setprecision(3) << out_single.avg_recall
                      << ", avg=" << out_single.avg_latency_us << "us, p95=" << out_single.p95_latency_us
                      << "us, p99=" << out_single.p99_latency_us << "us";
        }
        if (do_batch) {
            if (do_single) std::cout << " | ";
            std::cout << "QPS_batch=" << std::fixed << std::setprecision(2) << out_batch.qps
                      << ", Recall_batch@" << cli.k << "=" << std::fixed << std::setprecision(3) << out_batch.avg_recall
                      << " (batch，延迟N/A)";
        }
        std::cout << "\n";

        if (do_single) {
            append_result_csv(name, nb, nq, dim, cli.M, cli.efC, efS, cli.k,
                              cli.warmup_rounds, std::min(cli.warmup_q, nq), out_single);
        } else if (do_batch) {
            append_result_csv(name, nb, nq, dim, cli.M, cli.efC, efS, cli.k,
                              cli.warmup_rounds, std::min(cli.warmup_q, nq), out_batch);
        }

        // 小憩避免噪声
        sleep(1);
    }
}

// -------------------- main --------------------
int main(int argc, char** argv) {
    try {
        auto cli = parse_cli(argc, argv);
        std::cout << "HNSW(IP, 单位化) 测试（一次建索引，官方GT优先/可回退，循环efSearch）\n";

        if (cli.dataset == "CUSTOM") {
            if (cli.name.empty()) throw std::runtime_error("CUSTOM 需要 --name");
            if (cli.base_file.empty() || cli.query_file.empty() || cli.dim <= 0)
                throw std::runtime_error("CUSTOM 需要 --base, --query, --dim");
            run_one_dataset(cli.name, cli.base_file, cli.query_file, cli.gt_file, cli.dim, cli);
            std::cout << "\n结果已追加到: " << result_dir() << "/hnsw_efS_sweep.csv\n";
            return 0;
        }

        if (cli.dataset == "ALL") {
            auto presets = all_presets();
            for (const auto& ds : presets) {
                run_one_dataset(ds.name, ds.base_file, ds.query_file, ds.gt_file, ds.dim, cli);
            }
            std::cout << "\n结果已追加到: " << result_dir() << "/hnsw_efS_sweep.csv\n";
            return 0;
        }

        auto presets = all_presets();
        if (cli.dataset.empty()) {
            std::cout << "\n数据集:\n";
            for (size_t i=0;i<presets.size();++i) std::cout << (i+1) << ". " << presets[i].name << "\n";
            std::cout << "选择: ";
            int choice=0; std::cin >> choice;
            if (choice<1 || choice>(int)presets.size()) { std::cerr << "无效选择\n"; return 1; }
            const auto& ds = presets[choice-1];
            run_one_dataset(ds.name, ds.base_file, ds.query_file, ds.gt_file, ds.dim, cli);
            std::cout << "\n结果已追加到: " << result_dir() << "/hnsw_efS_sweep.csv\n";
            return 0;
        } else {
            auto it = std::find_if(presets.begin(), presets.end(),
                                   [&](const DatasetDesc& d){ return d.name == cli.dataset; });
            if (it == presets.end()) {
                std::ostringstream oss; oss << "未知 --dataset: " << cli.dataset
                    << "（可选：SIFT1M|GIST1M|GloVe1M|ALL|CUSTOM）";
                throw std::runtime_error(oss.str());
            }
            const auto& ds = *it;
            run_one_dataset(ds.name, ds.base_file, ds.query_file, ds.gt_file, ds.dim, cli);
            std::cout << "\n结果已追加到: " << result_dir() << "/hnsw_efS_sweep.csv\n";
            return 0;
        }
    } catch (const std::exception& e) {
        std::cerr << "\n[ERROR] " << e.what() << "\n";
        return 1;
    }
}

