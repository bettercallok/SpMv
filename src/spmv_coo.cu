// spmv_coo.cu — COO: one thread per nonzero, atomicAdd into y.
#include "cuda_utils.h"
#include "spmv_kernels.h"

namespace {

// One atomic per nonzero: perfect load balance on any row shape, but every
// entry of a dense row hits the same y entry (atomic contention).
__global__ void coo_kernel(int nnz, const int* __restrict__ row_idx, const int* __restrict__ col_idx,
                           const float* __restrict__ vals, const float* __restrict__ x, float* __restrict__ y) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nnz; i += blockDim.x * gridDim.x)
    atomicAdd(&y[row_idx[i]], vals[i] * LOAD_X(x, col_idx[i]));
}

}  // namespace

void spmv_coo(const DeviceCoo& C, const float* x, float* y, bool accumulate) {
  if (!accumulate && C.rows > 0) CUDA_CHECK(cudaMemsetAsync(y, 0, C.rows * sizeof(float)));
  if (C.nnz == 0) return;
  coo_kernel<<<grid_size(C.nnz, kBlockSize), kBlockSize>>>(C.nnz, C.row_idx, C.col_idx, C.vals, x, y);
}
