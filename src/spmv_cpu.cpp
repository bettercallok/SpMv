// spmv_cpu.cpp — CPU reference SpMV (double accumulation).
#include <cmath>

#include "formats.h"

void spmv_cpu_csr(const CsrMatrix& A, const std::vector<float>& x, std::vector<double>& y,
                  std::vector<double>* abs_y) {
  y.assign(A.rows, 0.0);
  if (abs_y) abs_y->assign(A.rows, 0.0);
  for (int r = 0; r < A.rows; ++r) {
    for (int p = A.row_ptr[r]; p < A.row_ptr[r + 1]; ++p) {
      const double t = static_cast<double>(A.vals[p]) * x[A.col_idx[p]];
      y[r] += t;
      if (abs_y) (*abs_y)[r] += std::fabs(t);
    }
  }
}

void spmv_cpu_ell(const EllMatrix& E, const std::vector<float>& x, std::vector<double>& y) {
  y.assign(E.rows, 0.0);
  for (int r = 0; r < E.rows; ++r) {
    for (int k = 0; k < E.width; ++k) {
      const size_t idx = static_cast<size_t>(k) * E.rows + r;
      if (E.col_idx[idx] < 0) break;  // padding only follows real entries
      y[r] += static_cast<double>(E.vals[idx]) * x[E.col_idx[idx]];
    }
  }
}

void spmv_cpu_coo(const CooMatrix& C, const std::vector<float>& x, std::vector<double>& y) {
  y.resize(C.rows, 0.0);
  for (int i = 0; i < C.nnz(); ++i) y[C.row_idx[i]] += static_cast<double>(C.vals[i]) * x[C.col_idx[i]];
}
