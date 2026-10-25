// convert.cpp — CSR construction and CSR -> ELL / COO / HYB conversion.
#include <algorithm>
#include <utility>

#include "formats.h"

namespace {

int row_len(const CsrMatrix& A, int r) { return A.row_ptr[r + 1] - A.row_ptr[r]; }

int max_row_length(const CsrMatrix& A) {
  int m = 0;
  for (int r = 0; r < A.rows; ++r) m = std::max(m, row_len(A, r));
  return m;
}

}  // namespace

CsrMatrix coo_to_csr(int rows, int cols, const std::vector<int>& ri,
                     const std::vector<int>& ci, const std::vector<float>& v) {
  // Bucket entries by row (counting sort), then sort each row by column.
  std::vector<int> start(rows + 1, 0);
  for (int r : ri) start[r + 1]++;
  for (int r = 0; r < rows; ++r) start[r + 1] += start[r];
  std::vector<std::pair<int, float>> bucketed(v.size());
  std::vector<int> pos(start.begin(), start.end() - 1);
  for (size_t i = 0; i < v.size(); ++i) bucketed[pos[ri[i]]++] = {ci[i], v[i]};

  CsrMatrix A{rows, cols, std::vector<int>(rows + 1, 0), {}, {}};
  for (int r = 0; r < rows; ++r) {
    auto first = bucketed.begin() + start[r], last = bucketed.begin() + start[r + 1];
    std::sort(first, last, [](const auto& a, const auto& b) { return a.first < b.first; });
    const size_t row_begin = A.col_idx.size();
    for (auto it = first; it != last; ++it) {
      if (A.col_idx.size() > row_begin && A.col_idx.back() == it->first) {
        A.vals.back() += it->second;  // duplicate entry
        continue;
      }
      A.col_idx.push_back(it->first);
      A.vals.push_back(it->second);
    }
    A.row_ptr[r + 1] = static_cast<int>(A.col_idx.size());
  }
  return A;
}

EllMatrix csr_to_ell(const CsrMatrix& A, int width) {
  EllMatrix E{A.rows, A.cols, width < 0 ? max_row_length(A) : width, 0, {}, {}};
  E.col_idx.assign(static_cast<size_t>(E.width) * A.rows, -1);
  E.vals.assign(E.col_idx.size(), 0.0f);
  for (int r = 0; r < A.rows; ++r) {
    const int len = std::min(row_len(A, r), E.width);
    for (int k = 0; k < len; ++k) {
      const size_t idx = static_cast<size_t>(k) * A.rows + r;
      E.col_idx[idx] = A.col_idx[A.row_ptr[r] + k];
      E.vals[idx] = A.vals[A.row_ptr[r] + k];
    }
    E.nnz += len;
  }
  return E;
}

CooMatrix csr_to_coo(const CsrMatrix& A) {
  CooMatrix C{A.rows, A.cols, std::vector<int>(A.nnz()), A.col_idx, A.vals};
  for (int r = 0; r < A.rows; ++r)
    std::fill(C.row_idx.begin() + A.row_ptr[r], C.row_idx.begin() + A.row_ptr[r + 1], r);
  return C;
}

int hyb_choose_k(const CsrMatrix& A) {
  std::vector<long long> hist(max_row_length(A) + 1, 0);
  for (int r = 0; r < A.rows; ++r) hist[row_len(A, r)]++;
  long long rows_at_least = 0;  // rows with length >= len
  for (int len = static_cast<int>(hist.size()) - 1; len >= 1; --len) {
    rows_at_least += hist[len];
    if (3 * rows_at_least >= A.rows) return len;
  }
  return 0;
}

HybMatrix csr_to_hyb(const CsrMatrix& A) {
  const int k = hyb_choose_k(A);
  HybMatrix H{csr_to_ell(A, k), CooMatrix{A.rows, A.cols, {}, {}, {}}};
  for (int r = 0; r < A.rows; ++r)
    for (int p = A.row_ptr[r] + k; p < A.row_ptr[r + 1]; ++p) {  // overflow past k goes to COO
      H.coo.row_idx.push_back(r);
      H.coo.col_idx.push_back(A.col_idx[p]);
      H.coo.vals.push_back(A.vals[p]);
    }
  return H;
}

MatrixStats compute_stats(const CsrMatrix& A) {
  MatrixStats s;
  if (A.rows == 0 || A.nnz() == 0) return s;
  s.max_row = max_row_length(A);
  s.mean_row = static_cast<double>(A.nnz()) / A.rows;
  s.ell_padding = static_cast<double>(A.rows) * s.max_row / A.nnz();
  s.hyb_k = hyb_choose_k(A);
  long long in_coo = 0;
  for (int r = 0; r < A.rows; ++r) in_coo += std::max(0, row_len(A, r) - s.hyb_k);
  s.hyb_coo_fraction = static_cast<double>(in_coo) / A.nnz();
  return s;
}
