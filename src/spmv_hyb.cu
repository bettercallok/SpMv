// spmv_hyb.cu — HYB: ELL for the regular part, COO for the overflow.
#include "spmv_kernels.h"

void spmv_hyb(const DeviceHyb& H, const float* x, float* y) {
  // ELL writes every y[row], so COO can accumulate on top (same stream, so in order).
  spmv_ell(H.ell, x, y);
  spmv_coo(H.coo, x, y, /*accumulate=*/true);
}
