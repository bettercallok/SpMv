// bench.cu — benchmark harness. For each matrix: convert to every format, check
// each kernel against the CPU reference, time it (warm-up + median of N
// launches), compare with cuSPARSE, print a table and append rows to a CSV.
//
//   spmv_bench [--kernels a,b] [--iters N] [--warmup N] [--peak-bw GB/s] [--csv PATH] MATRIX...
//   MATRIX: path.mtx | laplace2d:N (N^2 rows) | random:ROWS:PER_ROW
#include <cusparse.h>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "cuda_utils.h"
#include "formats.h"
#include "mmio.h"
#include "spmv_kernels.h"

#define CUSPARSE_CHECK(call)                                                                                  \
  do {                                                                                                        \
    if (cusparseStatus_t st_ = (call); st_ != CUSPARSE_STATUS_SUCCESS) {                                      \
      std::fprintf(stderr, "cuSPARSE error at %s:%d: %s\n", __FILE__, __LINE__, cusparseGetErrorString(st_)); \
      std::exit(EXIT_FAILURE);                                                                                \
    }                                                                                                         \
  } while (0)

namespace {

constexpr double kMaxEllPadding = 20.0;  // skip ELL when rows * max_row / nnz exceeds this
constexpr double kTolerance = 1e-5;

struct Options {
  std::vector<std::string> matrices, kernels;  // empty kernels = all
  int warmup = 5, iters = 100;
  double peak_bw = 0;  // GB/s; 0 = from device attributes
  std::string csv = "results/results.csv";
};

std::vector<std::string> split(const std::string& s, char sep) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  for (std::string item; std::getline(ss, item, sep);)
    if (!item.empty()) out.push_back(item);
  return out;
}

Options parse_args(int argc, char** argv) {
  Options o;
  bool ok = true;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    const bool more = i + 1 < argc;
    if (a == "--kernels" && more) o.kernels = split(argv[++i], ',');
    else if (a == "--iters" && more) o.iters = std::max(1, std::stoi(argv[++i]));
    else if (a == "--warmup" && more) o.warmup = std::max(0, std::stoi(argv[++i]));
    else if (a == "--peak-bw" && more) o.peak_bw = std::stod(argv[++i]);
    else if (a == "--csv" && more) o.csv = argv[++i];
    else if (a.rfind("--", 0) == 0) ok = false;
    else o.matrices.push_back(a);
  }
  if (!ok || o.matrices.empty()) {
    std::fprintf(stderr, "usage: %s [--kernels a,b] [--iters N] [--warmup N] [--peak-bw GB/s] [--csv PATH] MATRIX...\n"
                 "  MATRIX: path.mtx | laplace2d:N | random:ROWS:PER_ROW\n  kernels: csr_scalar csr_vector "
                 "csr_vector_auto ell coo hyb cusparse\n", argv[0]);
    std::exit(EXIT_FAILURE);
  }
  return o;
}

double now_ms() {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

CsrMatrix load_matrix(const std::string& spec, std::string& name) {
  const auto parts = split(spec, ':');
  if (parts.size() == 2 && parts[0] == "laplace2d") {
    name = "laplace2d_" + parts[1];
    return make_laplacian_2d(std::stoi(parts[1]));
  }
  if (parts.size() == 3 && parts[0] == "random") {
    name = "random_" + parts[1] + "_" + parts[2];
    return make_random(std::stoi(parts[1]), std::stoi(parts[2]), 1);
  }
  name = std::filesystem::path(spec).stem().string();
  return read_matrix_market(spec);
}

// Every device allocation for the current matrix; released by free_all().
std::vector<void*> g_allocs;

template <class T>
T* device_alloc(size_t n) {
  T* d = nullptr;
  CUDA_CHECK(cudaMalloc(&d, std::max<size_t>(n, 1) * sizeof(T)));
  g_allocs.push_back(d);
  return d;
}

template <class T>
T* to_device(const std::vector<T>& h) {
  T* d = device_alloc<T>(h.size());
  CUDA_CHECK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
  return d;
}

void free_all() {
  for (void* p : g_allocs) CUDA_CHECK(cudaFree(p));
  g_allocs.clear();
}

DeviceEll upload(const EllMatrix& E) { return {E.rows, E.cols, E.width, to_device(E.col_idx), to_device(E.vals)}; }
DeviceCoo upload(const CooMatrix& C) { return {C.rows, C.nnz(), to_device(C.row_idx), to_device(C.col_idx), to_device(C.vals)}; }

// Each launch gets its own event pair so we can report the median.
double median_ms(const std::function<void()>& launch, int warmup, int iters) {
  for (int i = 0; i < warmup; ++i) launch();
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  std::vector<float> ms(iters);
  for (auto& t : ms) {
    CUDA_CHECK(cudaEventRecord(start));
    launch();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&t, start, stop));
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  std::sort(ms.begin(), ms.end());
  return iters % 2 ? ms[iters / 2] : 0.5 * (ms[iters / 2 - 1] + ms[iters / 2]);
}

// Max over rows of |y - ref| / sum |a_ij x_j|; scaling by the absolute row sum
// keeps rows with cancellation from failing spuriously.
double max_rel_error(const std::vector<float>& y, const std::vector<double>& ref, const std::vector<double>& abs_ref) {
  double worst = 0;
  for (size_t i = 0; i < y.size(); ++i) {
    if (!std::isfinite(y[i])) return INFINITY;
    const double err = std::fabs(y[i] - ref[i]);
    if (err > 0) worst = std::max(worst, err / std::max(abs_ref[i], 1e-30));
  }
  return worst;
}

struct Case {
  std::string kernel;
  std::function<void()> launch;  // empty = skipped
  size_t matrix_bytes;           // "bytes of A" in this format (ELL includes padding)
  double conv_ms;
  std::string note;
};

void bench_matrix(const Options& opt, const std::string& gpu, double peak_bw, const std::string& spec) {
  std::string name;
  const CsrMatrix A = load_matrix(spec, name);
  const MatrixStats s = compute_stats(A);
  std::printf("\n%s: %d x %d, nnz %d, row length mean %.1f max %d, ELL padding %.1fx, HYB k=%d (%.0f%% in COO)\n",
              name.c_str(), A.rows, A.cols, A.nnz(), s.mean_row, s.max_row, s.ell_padding, s.hyb_k,
              100 * s.hyb_coo_fraction);
  double t0 = 0;
  auto wants = [&](const char* k) {
    return opt.kernels.empty() || std::find(opt.kernels.begin(), opt.kernels.end(), k) != opt.kernels.end();
  };

  std::vector<float> x(A.cols);
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto& v : x) v = dist(rng);
  std::vector<double> ref, abs_ref;
  spmv_cpu_csr(A, x, ref, &abs_ref);

  float* d_x = to_device(x);
  float* d_y = device_alloc<float>(A.rows);
  const DeviceCsr csr{A.rows, A.cols, A.nnz(), to_device(A.row_ptr), to_device(A.col_idx), to_device(A.vals)};
  const size_t csr_bytes = (A.rows + 1) * sizeof(int) + A.nnz() * (sizeof(int) + sizeof(float));
  const int auto_width = csr_vector_auto_width(s.mean_row);
  std::vector<Case> cases;

  // cuSPARSE baseline first, so every later row can report its speed ratio.
  cusparseHandle_t handle = nullptr;
  cusparseSpMatDescr_t mat = nullptr;
  cusparseDnVecDescr_t vx = nullptr, vy = nullptr;
  const float alpha = 1.0f, beta = 0.0f;
  if (wants("cusparse") && A.nnz() > 0) {
    t0 = now_ms();
    CUSPARSE_CHECK(cusparseCreate(&handle));
    CUSPARSE_CHECK(cusparseCreateCsr(&mat, A.rows, A.cols, A.nnz(), csr.row_ptr, csr.col_idx, csr.vals,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&vx, A.cols, d_x, CUDA_R_32F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&vy, A.rows, d_y, CUDA_R_32F));
    size_t buffer_size = 0;
    CUSPARSE_CHECK(cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mat, vx, &beta, vy,
                                           CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &buffer_size));
    void* buffer = device_alloc<char>(buffer_size);
    cases.push_back({"cusparse", [&, buffer] {
                       CUSPARSE_CHECK(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mat, vx, &beta,
                                                   vy, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, buffer));
                     }, csr_bytes, now_ms() - t0, ""});
  }
  if (wants("csr_scalar")) cases.push_back({"csr_scalar", [=] { spmv_csr_scalar(csr, d_x, d_y); }, csr_bytes, 0, ""});
  if (wants("csr_vector")) cases.push_back({"csr_vector", [=] { spmv_csr_vector(csr, d_x, d_y, 32); }, csr_bytes, 0, ""});
  if (wants("csr_vector_auto"))
    cases.push_back({"csr_vector_auto", [=] { spmv_csr_vector(csr, d_x, d_y, auto_width); }, csr_bytes, 0,
                     "width=" + std::to_string(auto_width)});
  double conv = 0;  // host conversion time of the last timed() call
  auto timed = [&](auto convert) { t0 = now_ms(); auto m = convert(A); conv = now_ms() - t0; return m; };
  if (wants("ell") && s.ell_padding > kMaxEllPadding) {
    cases.push_back({"ell", nullptr, 0, 0, "padding too large"});
  } else if (wants("ell")) {
    const EllMatrix ell = timed([](const CsrMatrix& M) { return csr_to_ell(M); });
    const DeviceEll d = upload(ell);
    cases.push_back({"ell", [=] { spmv_ell(d, d_x, d_y); }, ell.col_idx.size() * 8, conv, ""});
  }
  if (wants("coo")) {
    const CooMatrix coo = timed(csr_to_coo);
    const DeviceCoo d = upload(coo);
    cases.push_back({"coo", [=] { spmv_coo(d, d_x, d_y, false); }, coo.vals.size() * 12, conv, ""});
  }
  if (wants("hyb")) {
    const HybMatrix hyb = timed(csr_to_hyb);
    const DeviceHyb d{upload(hyb.ell), upload(hyb.coo)};
    cases.push_back({"hyb", [=] { spmv_hyb(d, d_x, d_y); }, hyb.ell.col_idx.size() * 8 + hyb.coo.vals.size() * 12,
                     conv, "k=" + std::to_string(s.hyb_k)});
  }

  std::ofstream csv;
  if (!opt.csv.empty()) {
    const std::filesystem::path path(opt.csv);
    if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
    const bool header = !std::filesystem::exists(path);
    csv.open(path, std::ios::app);
    if (header)
      csv << "gpu,matrix,rows,nnz,kernel,status,median_ms,gflops,gbps,pct_peak,intensity,conv_ms,max_rel_err,"
             "speedup_vs_cusparse,peak_bw_gbps,note\n";
  }
  std::printf("  %-16s %-7s %10s %8s %8s %7s %8s %8s %8s\n", "kernel", "status", "median ms", "GFLOP/s", "GB/s",
              "%peak", "vs cuSP", "conv ms", "max err");

  // Run each case: check once against the reference, then time it.
  const size_t xy_bytes = (static_cast<size_t>(A.cols) + A.rows) * sizeof(float);
  std::vector<float> y(A.rows);
  double cusparse_ms = 0;
  for (const auto& c : cases) {
    std::string status = "skipped";
    double ms = 0, err = 0, gflops = 0, gbps = 0, pct = 0, intensity = 0, vs = 0;
    if (c.launch) {
      CUDA_CHECK(cudaMemset(d_y, 0xFF, A.rows * sizeof(float)));  // NaN, so missed rows fail
      c.launch();
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaMemcpy(y.data(), d_y, A.rows * sizeof(float), cudaMemcpyDeviceToHost));
      err = max_rel_error(y, ref, abs_ref);
      status = err <= kTolerance ? "ok" : "FAIL";
      ms = median_ms(c.launch, opt.warmup, opt.iters);
      if (c.kernel == "cusparse" && status == "ok") cusparse_ms = ms;
      const double bytes = c.matrix_bytes + xy_bytes, sec = ms * 1e-3;
      gflops = 2.0 * A.nnz() / (sec * 1e9);
      gbps = bytes / (sec * 1e9);
      pct = peak_bw > 0 ? 100 * gbps / peak_bw : 0;
      intensity = 2.0 * A.nnz() / bytes;
      vs = cusparse_ms > 0 ? cusparse_ms / ms : 0;
    }
    std::printf("  %-16s %-7s %10.4f %8.2f %8.1f %6.1f%% %7.2fx %8.1f %8.1e %s\n", c.kernel.c_str(), status.c_str(),
                ms, gflops, gbps, pct, vs, c.conv_ms, err, c.note.c_str());
    if (csv.is_open())
      csv << '"' << gpu << "\"," << name << ',' << A.rows << ',' << A.nnz() << ',' << c.kernel << ',' << status << ','
          << ms << ',' << gflops << ',' << gbps << ',' << pct << ',' << intensity << ',' << c.conv_ms << ',' << err
          << ',' << vs << ',' << peak_bw << ",\"" << c.note << "\"\n";
  }

  if (handle) {
    CUSPARSE_CHECK(cusparseDestroyDnVec(vx));
    CUSPARSE_CHECK(cusparseDestroyDnVec(vy));
    CUSPARSE_CHECK(cusparseDestroySpMat(mat));
    CUSPARSE_CHECK(cusparseDestroy(handle));
  }
  free_all();
}

}  // namespace

int main(int argc, char** argv) {
  const Options opt = parse_args(argc, argv);
  cudaDeviceProp prop{};
  int dev = 0, mem_clock_khz = 0, bus_bits = 0, runtime = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
  CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
  cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, dev);
  cudaDeviceGetAttribute(&bus_bits, cudaDevAttrGlobalMemoryBusWidth, dev);
  cudaGetLastError();  // ignore an unsupported attribute
  // Peak = 2 transfers per clock (DDR) * memory clock * bus width, unless --peak-bw is given.
  const double peak_bw = opt.peak_bw > 0 ? opt.peak_bw : 2.0 * mem_clock_khz * 1e3 * (bus_bits / 8.0) / 1e9;
  std::printf("GPU: %s (sm_%d%d), CUDA runtime %d, peak DRAM %.1f GB/s; %d warm-ups + median of %d launches\n",
              prop.name, prop.major, prop.minor, runtime, peak_bw, opt.warmup, opt.iters);

  int failures = 0;
  for (const auto& spec : opt.matrices) {
    try {
      bench_matrix(opt, prop.name, peak_bw, spec);
    } catch (const std::exception& e) {
      std::fprintf(stderr, "error on %s: %s\n", spec.c_str(), e.what());
      free_all();
      ++failures;
    }
  }
  return failures ? EXIT_FAILURE : 0;
}
