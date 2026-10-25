// generate.cpp — synthetic test matrices.
#include <algorithm>
#include <random>

#include "formats.h"

CsrMatrix make_laplacian_2d(int n) {
  CsrMatrix A;
  A.rows = A.cols = n * n;
  A.row_ptr.push_back(0);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      const int r = i * n + j;
      // Neighbours in increasing column order: up, left, self, right, down.
      const int cols[] = {i > 0 ? r - n : -1, j > 0 ? r - 1 : -1, r,
                          j < n - 1 ? r + 1 : -1, i < n - 1 ? r + n : -1};
      for (int c : cols) {
        if (c < 0) continue;
        A.col_idx.push_back(c);
        A.vals.push_back(c == r ? 4.0f : -1.0f);
      }
      A.row_ptr.push_back(static_cast<int>(A.col_idx.size()));
    }
  }
  return A;
}

CsrMatrix make_random(int rows, int nnz_per_row, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> col_dist(0, rows - 1);
  std::uniform_real_distribution<float> val_dist(-1.0f, 1.0f);
  CsrMatrix A;
  A.rows = A.cols = rows;
  A.row_ptr.push_back(0);
  std::vector<int> row_cols;
  for (int r = 0; r < rows; ++r) {
    row_cols.clear();
    while (static_cast<int>(row_cols.size()) < nnz_per_row) {  // draw until the columns are distinct
      while (static_cast<int>(row_cols.size()) < nnz_per_row) row_cols.push_back(col_dist(rng));
      std::sort(row_cols.begin(), row_cols.end());
      row_cols.erase(std::unique(row_cols.begin(), row_cols.end()), row_cols.end());
    }
    for (int c : row_cols) {
      A.col_idx.push_back(c);
      A.vals.push_back(val_dist(rng));
    }
    A.row_ptr.push_back(static_cast<int>(A.col_idx.size()));
  }
  return A;
}
