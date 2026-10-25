#!/usr/bin/env python3
"""Summarise benchmark CSVs into the Markdown tables used in the README.

    python3 scripts/analyze.py [results/results.csv] [results/results_ldg.csv] [results/bench_log.txt]

Standard library only. Prints: per-matrix best kernel, headline numbers, the
"useful" bandwidth of every kernel (CSR byte count, so padding is not counted),
long-row matrices, csr_vector vs csr_vector_auto, __ldg before/after (with
cuSPARSE as a run-to-run noise control, since the build flag does not touch it),
and conversion cost.
"""
import csv
import re
import statistics
import sys

SYNTHETIC = ("laplace2d_", "random_", "powerlaw_")


def load(path):
    with open(path, newline="") as f:
        return {(r["matrix"], r["kernel"]): r for r in csv.DictReader(f)}


def load_log(path):
    """Per-matrix shape stats from the header lines of bench_log.txt."""
    pat = re.compile(r"^(\S+): (\d+) x (\d+), nnz (\d+), row length mean ([\d.]+) max (\d+), "
                     r"ELL padding ([\d.]+)x, HYB k=(\d+) \((\d+)% in COO\)")
    stats = {}
    try:
        with open(path) as f:
            for line in f:
                m = pat.match(line)
                if m:
                    stats[m[1]] = {"cols": int(m[3]), "mean": float(m[5]), "max": int(m[6]),
                                   "pad": float(m[7]), "k": int(m[8]), "coo_pct": int(m[9])}
    except FileNotFoundError:
        pass
    return stats


def f(r, key):
    return float(r[key])


def ok(r):
    return r is not None and r["status"] == "ok"


def useful_pct(r, cols, peak):
    """% of peak using CSR bytes (minimum traffic) for every kernel."""
    rows, nnz = int(r["rows"]), int(r["nnz"])
    useful = 4 * (rows + 1) + 8 * nnz + 4 * cols + 4 * rows
    return 100 * useful / (f(r, "median_ms") * 1e-3 * 1e9) / peak


def table(header, rows):
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for row in rows:
        print("| " + " | ".join(str(c) for c in row) + " |")
    print()


def main():
    base_path = sys.argv[1] if len(sys.argv) > 1 else "results/results.csv"
    ldg_path = sys.argv[2] if len(sys.argv) > 2 else "results/results_ldg.csv"
    log_path = sys.argv[3] if len(sys.argv) > 3 else "results/bench_log.txt"
    base = load(base_path)
    stats = load_log(log_path)
    matrices = list(dict.fromkeys(m for m, _ in base))
    kernels = list(dict.fromkeys(k for _, k in base))
    peak = f(next(iter(base.values())), "peak_bw_gbps")
    real = [m for m in matrices if not m.startswith(SYNTHETIC)]
    synth = [m for m in matrices if m.startswith(SYNTHETIC)]
    cols = {m: stats.get(m, {}).get("cols", int(base[(m, "cusparse")]["rows"])) for m in matrices}

    # ---- per-matrix best kernel (fastest of ours, cuSPARSE excluded)
    print(f"## Per-matrix best kernel (peak {peak:.0f} GB/s)\n")
    best = {}
    out = []
    for m in real + synth:
        mine = [base[(m, k)] for k in kernels if k != "cusparse" and ok(base.get((m, k)))]
        b = min(mine, key=lambda r: f(r, "median_ms"))
        ties = [r["kernel"] for r in mine
                if r is not b and f(r, "median_ms") <= 1.01 * f(b, "median_ms")]
        best[m] = b
        cs = base[(m, "cusparse")]
        s = stats.get(m, {})
        out.append([m, "synthetic" if m in synth else "real", s.get("mean", ""), s.get("max", ""),
                    b["kernel"] + (f" (tie: {', '.join(ties)})" if ties else ""),
                    f"{f(b, 'median_ms'):.4f}", f"{f(b, 'pct_peak'):.1f}", f"{useful_pct(b, cols[m], peak):.1f}",
                    f"{f(b, 'speedup_vs_cusparse'):.2f}x", f"{f(cs, 'median_ms'):.4f}", f"{f(cs, 'pct_peak'):.1f}"])
    table(["matrix", "type", "mean row", "max row", "best kernel", "ms", "% peak", "useful % peak", "vs cuSPARSE",
           "cuSPARSE ms", "cuSPARSE % peak"], out)

    # ---- headline
    sp = lambda ms: [f(best[m], "speedup_vs_cusparse") for m in ms]
    print("## Headline\n")
    print(f"- real matrices: best kernel at {min(sp(real)):.2f}x-{max(sp(real)):.2f}x cuSPARSE; "
          f"faster than cuSPARSE on: "
          + ", ".join(f"{m} ({best[m]['kernel']} {f(best[m], 'speedup_vs_cusparse'):.2f}x)"
                      for m in real if f(best[m], "speedup_vs_cusparse") > 1))
    print(f"- synthetic: " + ", ".join(f"{m} best {best[m]['kernel']} {f(best[m], 'speedup_vs_cusparse'):.2f}x"
                                       for m in synth))
    beats = [(m, k, f(r, "speedup_vs_cusparse")) for (m, k), r in base.items()
             if k != "cusparse" and ok(r) and f(r, "speedup_vs_cusparse") > 1]
    print("- every (matrix, kernel) faster than cuSPARSE: "
          + ", ".join(f"{m}/{k} {s:.2f}x" for m, k, s in sorted(beats)))
    mine_ok = [r for (m, k), r in base.items() if k != "cusparse" and ok(r)]
    top = max(mine_ok, key=lambda r: f(r, "pct_peak"))
    print(f"- highest reported % of peak (format bytes): {top['matrix']}/{top['kernel']} "
          f"{f(top, 'pct_peak'):.1f}% ({f(top, 'gbps'):.1f} GB/s)")
    top_u = max(mine_ok, key=lambda r: useful_pct(r, cols[r["matrix"]], peak))
    print(f"- highest useful % of peak (CSR bytes): {top_u['matrix']}/{top_u['kernel']} "
          f"{useful_pct(top_u, cols[top_u['matrix']], peak):.1f}% "
          f"({useful_pct(top_u, cols[top_u['matrix']], peak) * peak / 100:.1f} GB/s)")
    cs_all = [base[(m, "cusparse")] for m in matrices]
    print(f"- cuSPARSE % of peak: {min(f(r, 'pct_peak') for r in cs_all):.1f}-"
          f"{max(f(r, 'pct_peak') for r in cs_all):.1f}%\n")

    # ---- ELL: reported vs useful bandwidth
    print("## ELL: reported vs useful bandwidth\n")
    out = []
    for m in matrices:
        r = base.get((m, "ell"))
        if ok(r):
            out.append([m, stats.get(m, {}).get("pad", ""), f"{f(r, 'pct_peak'):.1f}",
                        f"{useful_pct(r, cols[m], peak):.1f}", f"{f(r, 'speedup_vs_cusparse'):.2f}x"])
        elif r is not None:
            out.append([m, stats.get(m, {}).get("pad", ""), r["status"], r["note"], ""])
    table(["matrix", "ELL padding", "reported % peak", "useful % peak", "vs cuSPARSE"], out)

    # ---- long rows
    print("## Matrices with very long rows (max row > 10,000)\n")
    out = []
    for m in matrices:
        if stats.get(m, {}).get("max", 0) > 10000:
            sv = lambda k: f"{f(base[(m, k)], 'speedup_vs_cusparse'):.2f}x" if ok(base.get((m, k))) else "-"
            out.append([m, stats[m]["mean"], stats[m]["max"], sv("csr_scalar"), sv("csr_vector"),
                        sv("csr_vector_auto"), sv("coo"), sv("hyb"), f"{f(base[(m, 'cusparse')], 'pct_peak'):.1f}"])
    table(["matrix", "mean row", "max row", "csr_scalar", "csr_vector", "csr_vector_auto", "coo", "hyb",
           "cuSPARSE % peak"], out)

    # ---- csr_vector vs auto
    print("## csr_vector (width 32) vs csr_vector_auto\n")
    out = []
    for m in matrices:
        v, a = base.get((m, "csr_vector")), base.get((m, "csr_vector_auto"))
        if ok(v) and ok(a):
            out.append([m, stats.get(m, {}).get("mean", ""), stats.get(m, {}).get("max", ""), a["note"],
                        f"{f(v, 'median_ms'):.4f}", f"{f(a, 'median_ms'):.4f}",
                        f"{f(v, 'median_ms') / f(a, 'median_ms'):.2f}x"])
    table(["matrix", "mean row", "max row", "auto width", "width 32 ms", "auto ms", "speedup of auto"], out)

    # ---- __ldg
    try:
        ldg = load(ldg_path)
    except FileNotFoundError:
        ldg = {}
    if ldg:
        print("## __ldg build vs default build (ratio = default time / ldg time; > 1 means ldg faster)\n")
        ratios = {}
        for (m, k), r in base.items():
            q = ldg.get((m, k))
            if ok(r) and ok(q):
                ratios[(m, k)] = f(r, "median_ms") / f(q, "median_ms")
        out = []
        for k in kernels:
            vals = [v for (m, kk), v in ratios.items() if kk == k]
            if vals:
                lo = min(((v, m) for (m, kk), v in ratios.items() if kk == k))
                hi = max(((v, m) for (m, kk), v in ratios.items() if kk == k))
                out.append([k + (" (control: not affected by the flag)" if k == "cusparse" else ""),
                            f"{statistics.median(vals):.2f}x", f"{lo[0]:.2f}x ({lo[1]})", f"{hi[0]:.2f}x ({hi[1]})"])
        table(["kernel", "median", "worst", "best"], out)
        print("Largest changes (|ratio - 1| > 15%):\n")
        out = [[m, k, f"{f(base[(m, k)], 'median_ms'):.4f}", f"{f(ldg[(m, k)], 'median_ms'):.4f}", f"{v:.2f}x"]
               for (m, k), v in sorted(ratios.items(), key=lambda kv: kv[1]) if abs(v - 1) > 0.15]
        table(["matrix", "kernel", "default ms", "ldg ms", "ratio"], out)

    # ---- conversion cost
    print("## Host conversion cost vs kernel time\n")
    out = []
    for k in ("ell", "coo", "hyb"):
        rs = [r for (m, kk), r in base.items() if kk == k and ok(r)]
        if rs:
            worst = max(rs, key=lambda r: f(r, "conv_ms"))
            out.append([k, f"{max(f(r, 'conv_ms') for r in rs):.1f} ({worst['matrix']})",
                        f"{statistics.median(f(r, 'conv_ms') for r in rs):.1f}",
                        f"{statistics.median(f(r, 'conv_ms') / f(r, 'median_ms') for r in rs):.0f}x"])
    table(["format", "max conv ms", "median conv ms", "median conv / kernel time"], out)
    times = [f(best[m], "median_ms") for m in matrices]
    print(f"Best-kernel times range {min(times):.4f}-{max(times):.4f} ms.")


if __name__ == "__main__":
    main()
