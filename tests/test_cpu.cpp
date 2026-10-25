// test_cpu.cpp — reader and CPU reference on hand-checked matrices; every
// converted format must give the same y as CSR.
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "formats.h"
#include "mmio.h"

static int failures = 0;

static void check(bool ok, const std::string& what) {
  std::printf("%s %s\n", ok ? "  ok  " : "  FAIL", what.c_str());
  failures += !ok;
}

static bool near(const std::vector<double>& a, const std::vector<double>& b) {
  bool ok = a.size() == b.size();
  for (size_t i = 0; ok && i < a.size(); ++i) ok = std::fabs(a[i] - b[i]) <= 1e-9 * (1.0 + std::fabs(b[i]));
  return ok;
}

static void check_formats_agree(const CsrMatrix& A, const std::string& name) {
  std::vector<float> x(A.cols);
  for (int i = 0; i < A.cols; ++i) x[i] = 0.5f + 0.25f * static_cast<float>(i % 7);
  std::vector<double> ref, ell, coo, hyb;
  spmv_cpu_csr(A, x, ref);
  spmv_cpu_ell(csr_to_ell(A), x, ell);
  spmv_cpu_coo(csr_to_coo(A), x, coo);
  const HybMatrix H = csr_to_hyb(A);
  spmv_cpu_ell(H.ell, x, hyb);
  spmv_cpu_coo(H.coo, x, hyb);
  check(near(ell, ref) && near(coo, ref) && near(hyb, ref), name + ": ELL, COO and HYB match CSR");
}

int main() {
  const std::string dir = TEST_DATA_DIR;
  // A = [1 0 2 0; 0 3 0 0; 4 0 5 6; 0 0 0 0], entries stored out of order.
  const CsrMatrix A = read_matrix_market(dir + "/small.mtx");
  check(A.row_ptr == std::vector<int>({0, 2, 3, 6, 6}) && A.col_idx == std::vector<int>({0, 2, 1, 0, 2, 3}),
        "small: CSR arrays, columns sorted within rows");
  std::vector<double> y;
  spmv_cpu_csr(A, {1, 2, 3, 4}, y);
  check(near(y, {7, 6, 43, 0}), "small: A*[1 2 3 4] = [7 6 43 0]");
  check(hyb_choose_k(A) == 2, "small: HYB k = 2 (row lengths 2,1,3,0)");
  const EllMatrix E = csr_to_ell(A);
  check(E.width == 3 && E.col_idx[1 * 4 + 1] == -1 && E.col_idx[2 * 4 + 2] == 3, "small: ELL column-major, -1 pad");
  check_formats_agree(A, "small");

  const CsrMatrix S = read_matrix_market(dir + "/sym.mtx");
  spmv_cpu_csr(S, {1, 1, 1}, y);
  check(S.nnz() == 6 && near(y, {1, -2, 1}), "sym: expanded, A*[1 1 1] = [1 -2 1]");
  check(coo_to_csr(2, 2, {0, 0, 1}, {1, 1, 0}, {1.5f, 2.5f, 3.0f}).vals[0] == 4.0f, "duplicates are summed");

  const CsrMatrix L = make_laplacian_2d(4);
  spmv_cpu_csr(L, std::vector<float>(16, 1.0f), y);
  check(L.nnz() == 64 && y[5] == 0.0 && y[0] == 2.0, "laplace2d(4): 64 nnz, interior 0, corner 2");
  check_formats_agree(L, "laplace2d");
  check_formats_agree(make_random(200, 7, 1), "random");

  // One long row among short ones: HYB must split it between ELL and COO.
  std::vector<int> ri, ci;
  std::vector<float> v;
  for (int r = 0; r < 100; ++r)
    for (int k = 0; k < (r == 0 ? 90 : 1 + r % 3); ++k) ri.push_back(r), ci.push_back((r + 7 * k) % 100), v.push_back(1);
  const CsrMatrix P = coo_to_csr(100, 100, ri, ci, v);
  check(hyb_choose_k(P) > 0 && hyb_choose_k(P) < 90, "power-law: 0 < k < max row");
  check_formats_agree(P, "power-law");

  std::printf(failures ? "\n%d check(s) FAILED\n" : "\nall checks passed\n", failures);
  return failures ? 1 : 0;
}
