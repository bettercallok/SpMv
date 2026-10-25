#!/usr/bin/env bash
# One command for every number and chart: build, test, download, stats, benchmark, plots.
# Extra arguments go to spmv_bench (e.g. --peak-bw 320).
set -euo pipefail
cd "$(dirname "$0")/.."
cmake -S . -B build && cmake --build build -j
(cd build && ctest --output-on-failure)
scripts/get_matrices.sh matrices
mkdir -p results
python3 scripts/matrix_stats.py matrices/*.mtx | tee results/matrix_stats.md
rm -f results/results.csv
./build/spmv_bench --csv results/results.csv "$@" matrices/*.mtx laplace2d:1000 random:1000000:16 \
  | tee results/bench_log.txt
python3 scripts/plot.py results/results.csv results
