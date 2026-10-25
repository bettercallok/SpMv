# SpMV on CUDA: five kernels, one benchmark

CSR-scalar, CSR-vector, ELL, COO and HYB SpMV kernels benchmarked on SuiteSparse
matrices against cuSPARSE, following Bell & Garland (NVIDIA Research, 2008).

> **Headline:** best kernel reaches **X% of peak DRAM bandwidth** and runs within
> **Y× of cuSPARSE**. GPU _TBD_ · CUDA _TBD_ · driver _TBD_

## Build and run

```bash
cmake -S . -B build && cmake --build build -j
scripts/get_matrices.sh
./build/spmv_bench matrices/*.mtx laplace2d:1000 random:1000000:16
```

Everything at once: `pip install -r requirements.txt && scripts/run_all.sh`. Nsight Compute:
`scripts/profile.sh`. `__ldg` before/after: rebuild with `-DSPMV_USE_LDG=ON`. No GPU:
`-DSPMV_CPU_ONLY=ON` builds only the reader, converters and CPU tests.

## Kernels

| CSV name | mapping |
|---|---|
| `csr_scalar` | 1 thread per row, grid-stride loop |
| `csr_vector` / `csr_vector_auto` | 32 (or 2–32, from mean row length) lanes per row, shuffle reduction |
| `ell` | 1 thread per row, column-major, -1 padding; skipped if padding > 20× |
| `coo` | 1 thread per nonzero, `atomicAdd` into y |
| `hyb` | ELL for the first k per row + COO for the rest |
| `cusparse` | `cusparseSpMV` on CSR (baseline) |

## Method

5 warm-ups, then the median of 100 `cudaEvent`-timed launches; copies are excluded and
conversion is reported as `conv_ms`. y is pre-filled with NaN and each row must satisfy
`|y - ref| / Σ|a_ij·x_j| ≤ 1e-5` against a double CPU reference. `gflops` = 2·nnz/t,
`gbps` = (bytes of A in that format + x + y)/t, `pct_peak` uses `--peak-bw` or the device's
clock × bus width, `speedup_vs_cusparse` = t_cusparse/t.

## Results and findings

_TBD:_ `results/gbps.png`, `results/roofline.png`, `results/matrix_stats.md`, one Nsight-backed
finding per format, and a link to the report PDF.
