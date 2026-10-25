#!/usr/bin/env python3
"""GB/s bar chart and roofline plot from results.csv (needs matplotlib).
Usage: plot.py [results/results.csv] [results]"""
import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

path = sys.argv[1] if len(sys.argv) > 1 else "results/results.csv"
out = sys.argv[2] if len(sys.argv) > 2 else "results"
latest = {}  # the last run of each (matrix, kernel) wins
with open(path, newline="") as f:
    for row in csv.DictReader(f):
        if row["status"] == "ok":
            latest[(row["matrix"], row["kernel"])] = row
rows = list(latest.values())
if not rows:
    sys.exit(f"no 'ok' rows in {path}")
matrices = list(dict.fromkeys(r["matrix"] for r in rows))
kernels = list(dict.fromkeys(r["kernel"] for r in rows))
color = {k: plt.get_cmap("tab10")(i % 10) for i, k in enumerate(kernels)}
peak = float(rows[0]["peak_bw_gbps"])
os.makedirs(out, exist_ok=True)

# Effective bandwidth per kernel per matrix, with the peak DRAM line.
fig, ax = plt.subplots(figsize=(max(8, 0.35 * len(matrices) * len(kernels)), 5))
w = 0.8 / len(kernels)
for ki, k in enumerate(kernels):
    pts = [(mi, float(latest[(m, k)]["gbps"])) for mi, m in enumerate(matrices) if (m, k) in latest]
    ax.bar([mi + (ki - (len(kernels) - 1) / 2) * w for mi, _ in pts], [g for _, g in pts], w, label=k, color=color[k])
ax.axhline(peak, color="black", linestyle="--", linewidth=1, label="peak DRAM")
ax.set_xticks(range(len(matrices)), matrices, rotation=30, ha="right")
ax.set(ylabel="effective GB/s", title="Effective bandwidth per kernel")
ax.legend(fontsize=8, ncol=2)
fig.tight_layout()
fig.savefig(os.path.join(out, "gbps.png"), dpi=150)

# Roofline: every SpMV point should sit under the sloped bandwidth line.
fig, ax = plt.subplots(figsize=(8, 6))
xs = [float(r["intensity"]) for r in rows]
lo, hi = min(xs) / 2, max(xs) * 4
ax.plot([lo, hi], [peak * lo, peak * hi], "k-", label=f"DRAM roof {peak:.0f} GB/s")
for k in kernels:
    pts = [r for r in rows if r["kernel"] == k]
    ax.scatter([float(r["intensity"]) for r in pts], [float(r["gflops"]) for r in pts], label=k, color=color[k], s=30)
ax.set(xscale="log", yscale="log", xlabel="arithmetic intensity (flop / byte)", ylabel="achieved GFLOP/s",
       title="SpMV roofline (one point per kernel per matrix)")
ax.legend(fontsize=8)
ax.grid(which="both", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(out, "roofline.png"), dpi=150)
print(f"wrote {out}/gbps.png and {out}/roofline.png")
