// spmv_csr.cu — CSR-scalar (thread per row) and CSR-vector (lanes per row).
#include "cuda_utils.h"
#include "spmv_kernels.h"

namespace {

// Neighbouring threads walk different rows, so loads are strided (uncoalesced)
// and one long row stalls its whole warp.
__global__ void csr_scalar_kernel(int rows, const int* __restrict__ row_ptr,
                                  const int* __restrict__ col_idx, const float* __restrict__ vals,
                                  const float* __restrict__ x, float* __restrict__ y) {
  for (int row = blockIdx.x * blockDim.x + threadIdx.x; row < rows; row += blockDim.x * gridDim.x) {
    float sum = 0.0f;
    for (int p = row_ptr[row]; p < row_ptr[row + 1]; ++p) sum += vals[p] * LOAD_X(x, col_idx[p]);
    y[row] = sum;
  }
}

// Width lanes share a row and read it contiguously (coalesced); the partial
// sums are combined with __shfl_down_sync.
template <int Width>
__global__ void csr_vector_kernel(int rows, const int* __restrict__ row_ptr,
                                  const int* __restrict__ col_idx, const float* __restrict__ vals,
                                  const float* __restrict__ x, float* __restrict__ y) {
  constexpr int kRowsPerWarp = kWarpSize / Width;
  const int lane = threadIdx.x % Width;
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) / kWarpSize;
  const int num_warps = gridDim.x * blockDim.x / kWarpSize;
  // The loop bound is the same for the whole warp, so all 32 lanes reach the shuffles.
  for (int base = warp * kRowsPerWarp; base < rows; base += num_warps * kRowsPerWarp) {
    const int row = base + (threadIdx.x % kWarpSize) / Width;
    float sum = 0.0f;
    if (row < rows)
      for (int p = row_ptr[row] + lane; p < row_ptr[row + 1]; p += Width)
        sum += vals[p] * LOAD_X(x, col_idx[p]);
    for (int offset = Width / 2; offset > 0; offset >>= 1)
      sum += __shfl_down_sync(kFullMask, sum, offset, Width);
    if (lane == 0 && row < rows) y[row] = sum;
  }
}

}  // namespace

void spmv_csr_scalar(const DeviceCsr& A, const float* x, float* y) {
  if (A.rows == 0) return;
  csr_scalar_kernel<<<grid_size(A.rows, kBlockSize), kBlockSize>>>(A.rows, A.row_ptr, A.col_idx,
                                                                    A.vals, x, y);
}

void spmv_csr_vector(const DeviceCsr& A, const float* x, float* y, int width) {
  if (A.rows == 0) return;
  auto kernel = width == 2 ? csr_vector_kernel<2> : width == 4 ? csr_vector_kernel<4>
              : width == 8 ? csr_vector_kernel<8> : width == 16 ? csr_vector_kernel<16> : csr_vector_kernel<32>;
  const int grid = grid_size(static_cast<long long>(A.rows) * width, kBlockSize);
  kernel<<<grid, kBlockSize>>>(A.rows, A.row_ptr, A.col_idx, A.vals, x, y);
}

int csr_vector_auto_width(double mean_row) {
  int width = 2;
  while (width < kWarpSize && width < mean_row) width *= 2;
  return width;
}
