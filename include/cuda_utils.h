// cuda_utils.h — error checking and small helpers shared by the kernels.
#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                                              \
  do {                                                                                                \
    if (cudaError_t err_ = (call); err_ != cudaSuccess) {                                             \
      std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err_)); \
      std::exit(EXIT_FAILURE);                                                                        \
    }                                                                                                 \
  } while (0)

constexpr int kWarpSize = 32;
constexpr unsigned kFullMask = 0xffffffffu;

// Blocks for a grid-stride loop over n items, capped at 32 blocks per SM.
inline int grid_size(long long n, int block) {
  static const int max_blocks = [] {
    int dev = 0, sms = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    return 32 * sms;
  }();
  return static_cast<int>(std::clamp<long long>((n + block - 1) / block, 1, max_blocks));
}

// Reads of x. Build with -DSPMV_USE_LDG=ON to send them through the read-only cache.
#ifdef SPMV_USE_LDG
#define LOAD_X(x, i) __ldg((x) + (i))
#else
#define LOAD_X(x, i) (x)[i]
#endif
