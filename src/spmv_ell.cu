// spmv_ell.cu — ELL: one thread per row over column-major padded storage.
#include "cuda_utils.h"
#include "spmv_kernels.h"

namespace {

// Entry k of row r is at [k * rows + r], so at each step k the threads of a
// warp (consecutive rows) read consecutive addresses.
__global__ void ell_kernel(int rows, int width, const int* __restrict__ col_idx,
                           const float* __restrict__ vals, const float* __restrict__ x,
                           float* __restrict__ y) {
  for (int row = blockIdx.x * blockDim.x + threadIdx.x; row < rows; row += blockDim.x * gridDim.x) {
    float sum = 0.0f;
    for (int k = 0; k < width; ++k) {
      const size_t idx = static_cast<size_t>(k) * rows + row;
      const int c = col_idx[idx];
      if (c < 0) break;  // padding (col = -1) only follows a row's real entries
      sum += vals[idx] * LOAD_X(x, c);
    }
    y[row] = sum;
  }
}

}  // namespace

void spmv_ell(const DeviceEll& E, const float* x, float* y) {
  if (E.rows == 0) return;
  ell_kernel<<<grid_size(E.rows, kBlockSize), kBlockSize>>>(E.rows, E.width, E.col_idx, E.vals, x, y);
}
