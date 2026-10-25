// formats.h — host sparse formats, converters, generators and CPU reference.
#pragma once

#include <vector>

struct CsrMatrix {  // column indices sorted and unique within each row
  int rows = 0, cols = 0;
  std::vector<int> row_ptr, col_idx;
  std::vector<float> vals;
  int nnz() const { return static_cast<int>(vals.size()); }
};

// Column-major: entry k of row r is at [k * rows + r]. Padding has col = -1
// and always comes after a row's real entries.
struct EllMatrix {
  int rows = 0, cols = 0, width = 0, nnz = 0;
  std::vector<int> col_idx;
  std::vector<float> vals;
};

struct CooMatrix {  // sorted by row
  int rows = 0, cols = 0;
  std::vector<int> row_idx, col_idx;
  std::vector<float> vals;
  int nnz() const { return static_cast<int>(vals.size()); }
};

struct HybMatrix { EllMatrix ell; CooMatrix coo; };  // first k entries per row in ELL, rest in COO

struct MatrixStats {
  double mean_row = 0, ell_padding = 0, hyb_coo_fraction = 0;
  int max_row = 0, hyb_k = 0;
};

// Builds CSR from 0-based triplets in any order; duplicates are summed.
CsrMatrix coo_to_csr(int rows, int cols, const std::vector<int>& ri, const std::vector<int>& ci,
                     const std::vector<float>& v);
EllMatrix csr_to_ell(const CsrMatrix& A, int width = -1);  // -1 = max row length; longer rows are cut
CooMatrix csr_to_coo(const CsrMatrix& A);
int hyb_choose_k(const CsrMatrix& A);  // largest row length that at least a third of the rows reach
HybMatrix csr_to_hyb(const CsrMatrix& A);
MatrixStats compute_stats(const CsrMatrix& A);

CsrMatrix make_laplacian_2d(int grid_n);  // 5-point stencil, grid_n^2 rows
CsrMatrix make_random(int rows, int nnz_per_row, unsigned seed);

// CPU reference in double. abs_y (optional) gets sum |a_ij * x_j| per row for the error check.
void spmv_cpu_csr(const CsrMatrix& A, const std::vector<float>& x, std::vector<double>& y,
                  std::vector<double>* abs_y = nullptr);
void spmv_cpu_ell(const EllMatrix& E, const std::vector<float>& x, std::vector<double>& y);  // y = E*x
void spmv_cpu_coo(const CooMatrix& C, const std::vector<float>& x, std::vector<double>& y);  // y += C*x
