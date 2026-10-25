// spmv_kernels.h — device copies of the formats and the launch wrappers.
// Every wrapper computes y = A*x on the default stream (x, y are device pointers).
#pragma once

struct DeviceCsr { int rows = 0, cols = 0, nnz = 0; int *row_ptr = nullptr, *col_idx = nullptr; float* vals = nullptr; };
struct DeviceEll { int rows = 0, cols = 0, width = 0; int* col_idx = nullptr; float* vals = nullptr; };
struct DeviceCoo { int rows = 0, nnz = 0; int *row_idx = nullptr, *col_idx = nullptr; float* vals = nullptr; };
struct DeviceHyb { DeviceEll ell; DeviceCoo coo; };

constexpr int kBlockSize = 256;  // must be a multiple of 32

// 1. One thread per row.
void spmv_csr_scalar(const DeviceCsr& A, const float* x, float* y);
// 2. `width` lanes per row (2, 4, 8, 16 or 32) + shuffle reduction; 32 = warp per row.
void spmv_csr_vector(const DeviceCsr& A, const float* x, float* y, int width);
int csr_vector_auto_width(double mean_row);  // smallest power of two >= mean, max 32
// 3. One thread per row over column-major ELL. Overwrites y.
void spmv_ell(const DeviceEll& E, const float* x, float* y);
// 4. One thread per nonzero with atomicAdd. accumulate = false zeroes y first.
void spmv_coo(const DeviceCoo& C, const float* x, float* y, bool accumulate);
// 5. ELL for the first k entries per row, then COO for the rest.
void spmv_hyb(const DeviceHyb& H, const float* x, float* y);
