#!/usr/bin/env bash
# Nsight Compute metrics for each kernel on a few representative matrices; one CSV
# per (matrix, kernel) in results/ncu/. Usage: scripts/profile.sh [matrix ...]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/results/ncu"
mkdir -p "$OUT"
[[ $# -gt 0 ]] && MATRICES=("$@") || MATRICES=("$ROOT/matrices/cant.mtx" "$ROOT/matrices/webbase-1M.mtx" laplace2d:1000)

# DRAM throughput, coalescing (sectors per request), active threads per instruction
# (/32 = warp execution efficiency), achieved occupancy, L2 hit rate (reuse of x).
METRICS=dram__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,smsp__thread_inst_executed_per_inst_executed.ratio,\
sm__warps_active.avg.pct_of_peak_sustained_active,lts__t_sector_hit_rate.pct

for m in "${MATRICES[@]}"; do
  name="$(basename "${m%.mtx}")"; name="${name//:/_}"
  for k in csr_scalar csr_vector ell coo hyb cusparse; do
    echo "== $name / $k"
    ncu --metrics "$METRICS" --csv --log-file "$OUT/${name}_${k}.csv" \
      "$ROOT/build/spmv_bench" --kernels "$k" --warmup 0 --iters 1 --csv "" "$m" > /dev/null \
      || echo "   ncu failed for $name / $k"
  done
done
echo "CSVs in $OUT"
