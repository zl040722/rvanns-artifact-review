#include <faiss/IndexHNSW.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/impl/HNSW.h>
#include <faiss/index_io.h>
#include <omp.h>

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

static bool file_exists(const std::string &filename)
{
  struct stat st{};
  return ::stat(filename.c_str(), &st) == 0;
}
static time_t file_mtime(const std::string &filename)
{
  struct stat st{};
  if (::stat(filename.c_str(), &st) == 0)
    return st.st_mtime;
  return 0;
}
static off_t file_size(const std::string &filename)
{
  struct stat st{};
  if (::stat(filename.c_str(), &st) == 0)
    return st.st_size;
  return 0;
}
static std::string basename_of(const std::string &path)
{
  auto p = path.find_last_of("/\\");
  return (p == std::string::npos) ? path : path.substr(p + 1);
}
static void ensure_dir(const std::string &path)
{
  std::ostringstream cmd;
  cmd << "mkdir -p '" << path << "' 2>/dev/null";
  (void)::system(cmd.str().c_str());
}

static std::string gt_cache_dir() { return "vectors_out/gt_cache"; }
static std::string index_cache_dir() { return "vectors_out/index_cache"; }
static std::string result_dir() { return "vectors_out/results"; }

static std::string sanitize_tag(const std::string &s)
{
  std::string t = s;
  for (auto &c : t)
    if (c == '/' || c == '\\' || c == ' ' || c == ':')
      c = '_';
  return t;
}

static std::string generate_gt_cache_filename(const std::string &name, int nb, int nq, int k)
{
  ensure_dir(gt_cache_dir());
  std::ostringstream oss;
  oss << gt_cache_dir() << "/" << name
      << "_nb" << nb << "_nq" << nq << "_k" << k << "_flat_gt.ivecs";
  return oss.str();
}

static std::string generate_index_cache_filename_hnswsq(
    const std::string &name, int dim, int nb, int M, int efC,
    const std::string &metric, const std::string &base_path)
{
  ensure_dir(index_cache_dir());
  std::ostringstream oss;
  std::string base_tag = sanitize_tag(basename_of(base_path)) + "_" +
                         std::to_string((long long)file_size(base_path));
  oss << index_cache_dir() << "/" << name
      << "_hnswsq_nb" << nb
      << "_dim" << dim
      << "_M" << M
      << "_efC" << efC
      << "_" << metric
      << "_" << base_tag
      << ".faiss";
  return oss.str();
}

static std::vector<float> load_binary_data(const std::string &filename, int &num_vectors, int dim)
{
  std::ifstream file(filename, std::ios::binary);
  if (!file)
    throw std::runtime_error("无法打开数据文件: " + filename);
  file.seekg(0, std::ios::end);
  size_t file_sz = (size_t)file.tellg();
  file.seekg(0, std::ios::beg);
  if (file_sz % (dim * sizeof(float)) != 0)
  {
    std::ostringstream oss;
    oss << "文件大小与维度不匹配: " << filename
        << " (size=" << file_sz << ", dim=" << dim << ")";
    throw std::runtime_error(oss.str());
  }
  num_vectors = (int)(file_sz / (dim * sizeof(float)));
  std::vector<float> data((size_t)num_vectors * dim);
  file.read(reinterpret_cast<char *>(data.data()), data.size() * sizeof(float));
  return data;
}

static std::vector<std::vector<int>> load_ivecs(const std::string &filename, int k_limit, int &nq_out, int *d_first_row_out = nullptr)
{
  std::ifstream f(filename, std::ios::binary);
  if (!f)
  {
    nq_out = 0;
    return {};
  }
  std::vector<std::vector<int>> rows;
  int first_d = -1;
  while (true)
  {
    int d = 0;
    if (!f.read(reinterpret_cast<char *>(&d), sizeof(int)))
      break;
    if (first_d < 0)
      first_d = d;
    std::vector<int> row(d);
    if (!f.read(reinterpret_cast<char *>(row.data()), d * sizeof(int)))
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

static bool save_ivecs(const std::string &filename, const std::vector<std::vector<int>> &rows)
{
  ensure_dir(gt_cache_dir());
  std::ofstream f(filename, std::ios::binary);
  if (!f)
    return false;
  for (const auto &r : rows)
  {
    int d = (int)r.size();
    f.write(reinterpret_cast<const char *>(&d), sizeof(int));
    f.write(reinterpret_cast<const char *>(r.data()), d * sizeof(int));
  }
  return true;
}

// -------------------- GT --------------------
static std::vector<std::vector<int>> compute_flat_groundtruth(
    const std::vector<float> &base_vectors,
    const std::vector<float> &query_vectors,
    int dim, int nb, int nq, int k)
{
  std::cout << "[GT] 使用 IndexFlatL2 现算 GT..." << std::flush;
  auto t0 = std::chrono::high_resolution_clock::now();

  faiss::IndexFlatL2 flat(dim);
  flat.add(nb, base_vectors.data());

  std::vector<std::vector<int>> gts(nq);
  std::vector<float> D(k);
  std::vector<faiss::idx_t> I(k);
  for (int i = 0; i < nq; ++i)
  {
    const float *q = query_vectors.data() + (size_t)i * dim;
    flat.search(1, q, k, D.data(), I.data());
    gts[i].assign(I.begin(), I.end());
  }

  auto t1 = std::chrono::high_resolution_clock::now();
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
  std::cout << " 完成 (" << ms << " ms)\n";
  return gts;
}

static std::vector<std::vector<int>> get_groundtruth(
    const std::vector<float> &base_vectors,
    const std::vector<float> &query_vectors,
    int dim, int nb, int nq, int k,
    const std::string &dataset_name,
    const std::string &gt_file_config, // 若非空且存在就强制用
    long &gt_ms_out)
{
  auto t0 = std::chrono::high_resolution_clock::now();

  if (!gt_file_config.empty() && file_exists(gt_file_config))
  {
    int off_nq = 0, off_k = -1;
    auto all = load_ivecs(gt_file_config, 0, off_nq, &off_k);
    if (all.empty())
      throw std::runtime_error("官方GT读取失败或为空: " + gt_file_config);
    if (k > off_k)
      throw std::runtime_error("GT列数不足：要求k=" + std::to_string(k) + "，实际=" + std::to_string(off_k));
    if (nq > off_nq)
      throw std::runtime_error("GT行数不足：要求nq=" + std::to_string(nq) + "，实际=" + std::to_string(off_nq));
    std::vector<std::vector<int>> rows(nq);
    for (int i = 0; i < nq; ++i)
      rows[i].assign(all[i].begin(), all[i].begin() + k);
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

// -------------------- metrics & mapping --------------------
static inline faiss::idx_t map_id_if_needed(
    faiss::idx_t id,
    const faiss::IndexHNSWSQ *hnsw,
    bool use_map)
{
  if (!use_map || id < 0 || !hnsw)
    return id;
  try
  {
    return hnsw->convert_to_original_id(id);
  }
  catch (...)
  {
    return -1;
  }
}

static double recall_at_k_one_maybe_map(
    const faiss::idx_t *pred_begin,
    const std::vector<int> &gt,
    int k,
    const faiss::IndexHNSWSQ *hnsw,
    bool use_map)
{
  if (k <= 0)
    return 0.0;
  int hit = 0;
  for (int i = 0; i < k; ++i)
  {
    faiss::idx_t p = map_id_if_needed(pred_begin[i], hnsw, use_map);
    if (p < 0)
      continue;
    for (int j = 0; j < k; ++j)
    {
      if (p == (faiss::idx_t)gt[j])
      {
        ++hit;
        break;
      }
    }
  }
  return double(hit) / k;
}

static double percentile_sorted(const std::vector<double> &sorted_latencies, double percentile)
{
  if (sorted_latencies.empty())
    return 0.0;
  double p = std::clamp(percentile, 0.0, 100.0) / 100.0;
  double pos = p * (sorted_latencies.size() - 1);
  size_t idx = (size_t)std::ceil(pos);
  if (idx >= sorted_latencies.size())
    idx = sorted_latencies.size() - 1;
  return sorted_latencies[idx];
}

struct RunOutcome
{
  bool ok = false;
  double qps = 0.0;
  double avg_recall = 0.0;
  long train_ms = 0, build_ms = 0, gt_ms = 0, search_ms = 0, load_ms = 0, save_ms = 0, reorder_ms = 0;
  double p95_latency_us = 0.0, p99_latency_us = 0.0, avg_latency_us = 0.0;
};

// -------------------- 索引构建/加载（不含重排）--------------------
static std::unique_ptr<faiss::Index> build_or_load_hnswSQ(
    const std::vector<float> &base,
    int dim, int nb,
    int M, int efConstruction,
    faiss::ScalarQuantizer::QuantizerType qtype,
    const std::string &dataset_name,
    const std::string &base_path_for_tag,
    bool force_rebuild,
    long &load_ms_out, long &build_ms_out, long &save_ms_out, long &train_ms_out)
{
  const std::string metric = "l2";
  std::string idx_cache = generate_index_cache_filename_hnswsq(
      dataset_name, dim, nb, M, efConstruction, metric, base_path_for_tag);

  load_ms_out = build_ms_out = save_ms_out = train_ms_out = 0;

  // 读取缓存（除非强制重建）
  std::unique_ptr<faiss::Index> index;
  auto t0 = std::chrono::high_resolution_clock::now();
  if (!force_rebuild && file_exists(idx_cache))
  {
    index.reset(faiss::read_index(idx_cache.c_str()));
  }
  auto t1 = std::chrono::high_resolution_clock::now();
  load_ms_out = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

  if (index)
  {
    if ((int)index->d == dim && (int)index->ntotal == nb && dynamic_cast<faiss::IndexHNSWSQ *>(index.get()))
    {
      std::cout << "已加载缓存索引: " << idx_cache << " (" << load_ms_out << " ms), ntotal=" << index->ntotal << "\n";
      return index;
    }
    index.reset();
  }

  std::cout << "开始构建 MPMI 索引 (nb=" << nb << ", dim=" << dim
            << ", M=" << M << ", efC=" << efConstruction << ", SQ=8bit)...\n";

  // 构建
  auto *hnsw = new faiss::IndexHNSWSQ(dim, qtype, M, faiss::METRIC_L2);
  hnsw->hnsw.efConstruction = efConstruction;
  std::unique_ptr<faiss::Index> idx(hnsw);

  auto tr0 = std::chrono::high_resolution_clock::now();
  idx->train(nb, base.data());
  auto tr1 = std::chrono::high_resolution_clock::now();
  train_ms_out = std::chrono::duration_cast<std::chrono::milliseconds>(tr1 - tr0).count();

  auto b0 = std::chrono::high_resolution_clock::now();
  idx->add(nb, base.data());
  auto b1 = std::chrono::high_resolution_clock::now();
  build_ms_out = std::chrono::duration_cast<std::chrono::milliseconds>(b1 - b0).count();
  std::cout << "构建完成: " << build_ms_out << " ms (train " << train_ms_out << " ms)\n";

  // 缓存写盘
  auto s0 = std::chrono::high_resolution_clock::now();
  faiss::write_index(idx.get(), idx_cache.c_str());
  auto s1 = std::chrono::high_resolution_clock::now();
  save_ms_out = std::chrono::duration_cast<std::chrono::milliseconds>(s1 - s0).count();
  std::cout << "已缓存索引至: " << idx_cache << " (" << save_ms_out << " ms)\n";

  return idx;
}

// -------------------- 预热 & 评测 --------------------
static void warmup_index(
    faiss::Index *index,
    const std::vector<float> &query_vectors,
    int dim, int nq, int k,
    int warmup_rounds, int warmup_q)
{
  std::vector<float> D(k);
  std::vector<faiss::idx_t> I(k);
  int take = std::min(nq, warmup_q);
  // 固定用 efSearch=128 预热
  if (auto *h = dynamic_cast<faiss::IndexHNSW *>(index))
    h->hnsw.efSearch = 128;

  for (int r = 0; r < warmup_rounds; ++r)
  {
    for (int i = 0; i < take; ++i)
    {
      const float *q = query_vectors.data() + (size_t)i * dim;
      index->search(1, q, k, D.data(), I.data());
    }
  }
}

static RunOutcome measure_once(
    faiss::Index *index,
    int efSearch,
    const std::vector<float> &query_vectors,
    int dim, int nq, int k,
    const std::vector<std::vector<int>> &ground_truths,
    bool batch_mode,
    bool use_reorder_map)
{
  RunOutcome out;
  out.ok = false;
  omp_set_num_threads(16);

  if (auto *h = dynamic_cast<faiss::IndexHNSW *>(index))
  {
    h->hnsw.efSearch = efSearch;
  }

  const auto *hnsw_sq = dynamic_cast<faiss::IndexHNSWSQ *>(index);

  if (batch_mode)
  {
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
    for (int i = 0; i < nq; ++i)
    {
      const faiss::idx_t *pred = &I[(size_t)i * k];
      total_recall += recall_at_k_one_maybe_map(pred, ground_truths[i], k, hnsw_sq, use_reorder_map);
    }
    out.avg_recall = total_recall / nq;

    out.avg_latency_us = out.p95_latency_us = out.p99_latency_us = -1.0;

    out.ok = true;
    return out;
  }
  else
  {
    // // 单条 search(1) + 延迟分布
    // std::vector<float> D(k);
    // std::vector<faiss::idx_t> I(k);
    // std::vector<double> latencies_us;

    // auto search_start = std::chrono::high_resolution_clock::now();
    // for (int i = 0; i < nq; ++i)
    // {
    //   const float *q = query_vectors.data() + (size_t)i * dim;
    //   auto q0 = std::chrono::high_resolution_clock::now();
    //   index->search(1, q, k, D.data(), I.data());
    //   auto q1 = std::chrono::high_resolution_clock::now();
    //   double us = std::chrono::duration_cast<std::chrono::nanoseconds>(q1 - q0).count() / 1000.0;
    //   latencies_us.push_back(us);
    // }
    // auto search_end = std::chrono::high_resolution_clock::now();
    // out.search_ms = std::chrono::duration_cast<std::chrono::milliseconds>(search_end - search_start).count();
    // out.qps = (double)nq / (out.search_ms / 1000.0 + 1e-9);

    // out.avg_latency_us = std::accumulate(latencies_us.begin(), latencies_us.end(), 0.0) / latencies_us.size();
    // std::vector<double> sorted_lat = latencies_us;
    // std::sort(sorted_lat.begin(), sorted_lat.end());
    // out.p95_latency_us = percentile_sorted(sorted_lat, 95.0);
    // out.p99_latency_us = percentile_sorted(sorted_lat, 99.0);

    // // Recall
    // std::vector<double> recalls;
    // for (int i = 0; i < nq; ++i)
    // {
    //   const float *q = query_vectors.data() + (size_t)i * dim;
    //   index->search(1, q, k, D.data(), I.data());
    //   recalls.push_back(recall_at_k_one_maybe_map(I.data(), ground_truths[i], k, hnsw_sq, use_reorder_map));
    // }
    // out.avg_recall = std::accumulate(recalls.begin(), recalls.end(), 0.0) / recalls.size();
    out.ok = true;
    return out;
  }
}

// -------------------- CSV --------------------
static void append_result_csv(
    const std::string &dataset_name, int nb, int nq, int dim,
    int M, int efC, int efS, int k, int warmup_rounds, int warmup_q,
    const std::string &reorder_algo,
    const RunOutcome &out)
{
  ensure_dir(result_dir());
  std::string path = result_dir() + "/hnsw_efs_rorder_sweep.csv";
  bool need_header = !file_exists(path);
  std::ofstream f(path, std::ios::app);
  if (!f)
    return;
  if (need_header)
  {
    f << "dataset,nb,nq,dim,M,efC,efS,k,warmup_rounds,warmup_q,reorder_algo,"
      << "qps,avg_us,p95_us,p99_us,recall,"
      << "build_ms,reorder_ms,reorder_vs_build,load_ms,save_ms,gt_ms,train_ms,search_ms\n";
  }
  double ratio = (out.build_ms > 0) ? (double)out.reorder_ms / (double)out.build_ms : -1.0;
  f << dataset_name << "," << nb << "," << nq << "," << dim << ","
    << M << "," << efC << "," << efS << "," << k << ","
    << warmup_rounds << "," << warmup_q << ","
    << (reorder_algo.empty() ? "none" : reorder_algo) << ","
    << std::fixed << std::setprecision(2) << out.qps << ","
    << std::fixed << std::setprecision(2) << out.avg_latency_us << ","
    << std::fixed << std::setprecision(2) << out.p95_latency_us << ","
    << std::fixed << std::setprecision(2) << out.p99_latency_us << ","
    << std::fixed << std::setprecision(3) << out.avg_recall << ","
    << out.build_ms << "," << out.reorder_ms << ","
    << std::fixed << std::setprecision(4) << ratio << ","
    << out.load_ms << "," << out.save_ms << ","
    << out.gt_ms << "," << out.train_ms << "," << out.search_ms << "\n";
}

// -------------------- CLI --------------------
struct Cli
{
  std::string dataset;
  std::string name;
  std::string base_file;
  std::string query_file;
  std::string gt_file;
  int dim = 0;

  std::vector<int> efS_list{16, 32, 64, 128, 256, 512};
  int M = 16;
  int efC = 256;
  int k = 100;
  int warmup_rounds = 2;
  int warmup_q = 1000;

  std::string reorder_algo = "gorder";
  bool use_reorder_map = true;
  bool force_rebuild = false;

  bool batch_only = false;
  bool single_only = false;
};

static std::vector<int> parse_efs(const std::string &s)
{
  std::vector<int> v;
  std::stringstream ss(s);
  std::string tok;
  while (std::getline(ss, tok, ','))
    if (!tok.empty())
      v.push_back(std::stoi(tok));
  return v.empty() ? std::vector<int>{16, 32, 64, 128, 256, 512} : v;
}

static Cli parse_cli(int argc, char **argv)
{
  Cli c;
  for (int i = 1; i < argc; i++)
  {
    std::string a(argv[i]);
    auto next = [&]() -> std::string
    {
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
    else if (a == "--dim")
      c.dim = std::stoi(next());
    else if (a == "--efs")
      c.efS_list = parse_efs(next());
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
    else if (a == "--batch")
      c.batch_only = true;
    else if (a == "--single")
      c.single_only = true;
    else if (a == "--reorder")
    {
      std::string algo = next();
      if (algo == "none")
        c.reorder_algo.clear();
      else
        c.reorder_algo = algo;
    }
    else if (a == "--reorder_map")
    {
      c.use_reorder_map = (std::stoi(next()) != 0);
    }
    else if (a == "--force_rebuild")
    {
      c.force_rebuild = true;
    }
    else
      throw std::runtime_error("未知参数: " + a);
  }
  return c;
}

struct DatasetDesc
{
  std::string name, base_file, query_file, gt_file;
  int dim;
};
static std::vector<DatasetDesc> all_presets()
{
  return {
      {"SIFT1M", "vectors_out/sift1m_base.bin", "vectors_out/sift1m_query.bin", "vectors_out/sift1m_groundtruth.ivecs", 128},
      {"GIST1M", "vectors_out/gist1m_base.bin", "vectors_out/gist1m_query.bin", "vectors_out/gist1m_groundtruth.ivecs", 960}};
}

static void run_one_dataset(const std::string &name,
                            const std::string &base_file,
                            const std::string &query_file,
                            const std::string &gt_file,
                            int dim,
                            const Cli &cli)
{
  std::cout << "\n==== 数据集: " << name << " ====\n";
  if (!file_exists(base_file) || !file_exists(query_file))
  {
    throw std::runtime_error("数据文件不存在（" + base_file + " 或 " + query_file + "）");
  }

  int nb = 0, nq = 0;
  auto base = load_binary_data(base_file, nb, dim);
  auto query = load_binary_data(query_file, nq, dim);

  // 构建/加载索引（不含重排）
  long load_ms = 0, build_ms = 0, save_ms = 0, train_ms = 0;
  auto index = build_or_load_hnswSQ(
      base, dim, nb, cli.M, cli.efC,
      faiss::ScalarQuantizer::QT_8bit,
      name, base_file,
      cli.force_rebuild,
      load_ms, build_ms, save_ms, train_ms);

  // 执行重排（如果指定）
  long reorder_ms = 0;
  if (!cli.reorder_algo.empty())
  {
    auto *hnsw = dynamic_cast<faiss::IndexHNSWSQ *>(index.get());
    if (!hnsw)
    {
      throw std::runtime_error("索引类型不是 IndexHNSWSQ，无法执行重排");
    }

    std::cout << "执行内存重排(" << cli.reorder_algo << ")..." << std::flush;
    auto r0 = std::chrono::high_resolution_clock::now();
    hnsw->reorder_graph_after_build(0, cli.reorder_algo.c_str(), true);
    auto r1 = std::chrono::high_resolution_clock::now();
    reorder_ms = std::chrono::duration_cast<std::chrono::milliseconds>(r1 - r0).count();
    std::cout << " 完成 (" << reorder_ms << " ms)\n";

    // 简单校验 perm 是否有效
    const auto &perm = hnsw->get_reorder_perm();
    if (perm.size() != (size_t)nb)
    {
      std::cerr << "  警告: reorder_perm 大小=" << perm.size()
                << " 与 nb=" << nb << " 不一致，可能无法做ID映射\n";
    }
  }

  long gt_ms = 0;
  auto gts = get_groundtruth(base, query, dim, nb, nq, cli.k, name, gt_file, gt_ms);

  warmup_index(index.get(), query, dim, nq, cli.k, cli.warmup_rounds, std::min(cli.warmup_q, nq));

  for (int efS : cli.efS_list)
  {
    const bool do_batch = !cli.single_only;
    const bool do_single = !cli.batch_only;

    RunOutcome out_batch, out_single;
    auto patch = [&](RunOutcome &o)
    {
      o.build_ms = build_ms;
      o.load_ms = load_ms;
      o.save_ms = save_ms;
      o.gt_ms = gt_ms;
      o.train_ms = train_ms;
      o.reorder_ms = reorder_ms;
    };

    if (do_batch)
    {
      out_batch = measure_once(index.get(), efS, query, dim, nq, cli.k, gts, /*batch=*/true,
                               /*use_reorder_map=*/!cli.reorder_algo.empty() && cli.use_reorder_map);
      patch(out_batch);
    }
    if (do_single)
    {
      out_single = measure_once(index.get(), efS, query, dim, nq, cli.k, gts, /*batch=*/false,
                                /*use_reorder_map=*/!cli.reorder_algo.empty() && cli.use_reorder_map);
      patch(out_single);
    }

    std::cout << "[efS=" << efS << "] ";
    // if (do_single)
    // {
    //   std::cout << "QPS_single=" << std::fixed << std::setprecision(2) << out_single.qps
    //             << ", Recall_single@" << cli.k << "=" << std::fixed << std::setprecision(3) << out_single.avg_recall
    //             << ", avg=" << out_single.avg_latency_us << "us, p95=" << out_single.p95_latency_us
    //             << "us, p99=" << out_single.p99_latency_us << "us";
    // }
    if (do_batch)
    {
      if (do_single)
        std::cout << "";
      std::cout << "QPS_batch=" << std::fixed << std::setprecision(2) << out_batch.qps
                << ", Recall_batch@" << cli.k << "=" << std::fixed << std::setprecision(3) << out_batch.avg_recall
                << " (batch，延迟N/A)";
    }
    std::cout << "\n";

    if (do_single)
    {
      append_result_csv(name, nb, nq, dim, cli.M, cli.efC, efS, cli.k,
                        cli.warmup_rounds, std::min(cli.warmup_q, nq), cli.reorder_algo, out_single);
    }
    else if (do_batch)
    {
      append_result_csv(name, nb, nq, dim, cli.M, cli.efC, efS, cli.k,
                        cli.warmup_rounds, std::min(cli.warmup_q, nq), cli.reorder_algo, out_batch);
    }

    sleep(1);
  }

  // 输出时间总结
  if (!cli.reorder_algo.empty())
  {
    double ratio = (build_ms > 0) ? (double)reorder_ms / (double)build_ms : -1.0;
    std::cout << "\n[时间总结] 构建用时: " << build_ms << " ms, 重排"
              << "用时: " << reorder_ms << " ms, 比例: " << std::fixed << std::setprecision(4) << ratio << "\n";
  }

  if (load_ms > 0 && build_ms == 0)
  {
    std::cout << "(提示) 本次使用了缓存索引，构建耗时未重新测量；如需测量请加 --force_rebuild\n";
  }
}

// -------------------- main --------------------
int main(int argc, char **argv)
{
  try
  {
    auto cli = parse_cli(argc, argv);
    std::cout << "MPMI 测试（索引缓存，重排每次执行，官方GT优先/可回退，循环efSearch）\n";

    if (cli.dataset == "CUSTOM")
    {
      if (cli.name.empty())
        throw std::runtime_error("CUSTOM 需要 --name");
      if (cli.base_file.empty() || cli.query_file.empty() || cli.dim <= 0)
        throw std::runtime_error("CUSTOM 需要 --base, --query, --dim");
      run_one_dataset(cli.name, cli.base_file, cli.query_file, cli.gt_file, cli.dim, cli);
      std::cout << "\n结果已追加到: " << result_dir() << "/hnsw_efs_rorder_sweep.csv\n";
      return 0;
    }

    if (cli.dataset == "ALL")
    {
      auto presets = all_presets();
      for (const auto &ds : presets)
      {
        run_one_dataset(ds.name, ds.base_file, ds.query_file, ds.gt_file, ds.dim, cli);
      }
      std::cout << "\n结果已追加到: " << result_dir() << "/hnsw_efs_rorder_sweep.csv\n";
      return 0;
    }

    auto presets = all_presets();
    if (cli.dataset.empty())
    {
      std::cout << "\n数据集:\n";
      for (size_t i = 0; i < presets.size(); ++i)
        std::cout << (i + 1) << ". " << presets[i].name << "\n";
      std::cout << "选择: ";
      int choice = 0;
      std::cin >> choice;
      if (choice < 1 || choice > (int)presets.size())
      {
        std::cerr << "无效选择\n";
        return 1;
      }
      const auto &ds = presets[choice - 1];
      run_one_dataset(ds.name, ds.base_file, ds.query_file, ds.gt_file, ds.dim, cli);
      std::cout << "\n结果已追加到: " << result_dir() << "/hnsw_efs_rorder_sweep.csv\n";
      return 0;
    }
    else
    {
      auto it = std::find_if(presets.begin(), presets.end(),
                             [&](const DatasetDesc &d)
                             { return d.name == cli.dataset; });
      if (it == presets.end())
      {
        std::ostringstream oss;
        oss << "未知 --dataset: " << cli.dataset
            << "（可选：SIFT1M|GIST1M|ALL|CUSTOM）";
        throw std::runtime_error(oss.str());
      }
      const auto &ds = *it;
      run_one_dataset(ds.name, ds.base_file, ds.query_file, ds.gt_file, ds.dim, cli);
      std::cout << "\n结果已追加到: " << result_dir() << "/hnsw_efs_rorder_sweep.csv\n";
      return 0;
    }
  }
  catch (const std::exception &e)
  {
    std::cerr << "\n[ERROR] " << e.what() << "\n";
    return 1;
  }
}

