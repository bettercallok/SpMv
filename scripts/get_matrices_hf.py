#!/usr/bin/env python3
"""Downloads SuiteSparse matrices from the Hugging Face mirror
Csed-dev/matrixpfn-suitesparse and generates the synthetic powerlaw_1M matrix.

Use this where sparse.tamu.edu is unreachable (it returned 403 on Kaggle); elsewhere
scripts/get_matrices.sh downloads the Bell & Garland set directly.

    pip install requests numpy pandas pyarrow
    python3 scripts/get_matrices_hf.py [--dest matrices] [--count 9] [--min-nnz 500000] [--no-powerlaw]

Selection (same as the Kaggle run): keep matrices with at least --min-nnz nonzeros,
split them into --count groups by mean row length, take the largest of each group.
Only the chosen .mtx files are streamed out of the ~5.7 GB archive.
"""
import argparse
import glob
import io
import os
import shutil
import tarfile

import numpy as np
import pandas as pd
import requests

HF = "https://huggingface.co/datasets/Csed-dev/matrixpfn-suitesparse/resolve/main"


def pick_matrices(count, min_nnz):
    man = pd.read_parquet(io.BytesIO(requests.get(f"{HF}/manifest.parquet", timeout=60).content))
    cols = {c.lower(): c for c in man.columns}

    def col(*names):  # the manifest's column names are matched loosely
        for n in names:
            for k, c in cols.items():
                if n in k:
                    return c
        raise KeyError(f"none of {names} in {list(man.columns)}")

    c_name, c_rows, c_nnz = col("name"), col("row", "n_rows", "m"), col("nnz", "nonzero")
    kind = next((cols[k] for k in cols if "domain" in k or "kind" in k), None)
    man["mean_row"] = man[c_nnz] / man[c_rows]
    big = man[man[c_nnz] >= min_nnz].copy()
    big["bucket"] = pd.qcut(big.mean_row.rank(method="first"), count, labels=False)
    pick = big.sort_values(c_nnz, ascending=False).groupby("bucket").head(1).sort_values("mean_row")
    show = [c_name, c_rows, c_nnz, "mean_row"] + ([kind] if kind else [])
    print(pick[show].round(1).to_string(index=False))
    return set(pick[c_name].astype(str))


def download(want, dest):
    have = {os.path.basename(p)[:-4] for p in glob.glob(f"{dest}/*.mtx")}
    todo = want - have
    if not todo:
        print("all chosen matrices already present")
        return
    with requests.get(f"{HF}/suitesparse.tar.gz", stream=True, timeout=120) as r:
        r.raise_for_status()
        r.raw.decode_content = True
        with tarfile.open(fileobj=r.raw, mode="r|gz") as tar:  # stream, never extract the whole archive
            for m in tar:
                base = os.path.basename(m.name)
                if m.isfile() and base.endswith(".mtx") and base[:-4] in todo:
                    with open(f"{dest}/{base}", "wb") as f:
                        shutil.copyfileobj(tar.extractfile(m), f)
                    todo.discard(base[:-4])
                    print("got", base, f"({len(todo)} left)")
                    if not todo:
                        break
    print("missing:", todo or "none")


def make_powerlaw(path, n=1_000_000, seed=7):
    """Web-graph-like matrix: Pareto row lengths (x2, capped at 5000), random columns.
    Duplicate (row, col) pairs are possible; the benchmark's reader sums them."""
    rng = np.random.default_rng(seed)
    lens = np.minimum(np.floor(rng.pareto(1.6, n) + 1).astype(np.int64) * 2, 5000)
    rows = np.repeat(np.arange(1, n + 1), lens)
    cols = rng.integers(1, n + 1, rows.size)
    vals = rng.uniform(-1, 1, rows.size)
    with open(path, "w") as f:
        f.write(f"%%MatrixMarket matrix coordinate real general\n{n} {n} {rows.size}\n")
        np.savetxt(f, np.column_stack([rows, cols, vals]), fmt="%d %d %.6f")
    print(f"powerlaw_1M: nnz={rows.size:,}, mean row={lens.mean():.1f}, max row={lens.max()}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", default=os.path.join(os.path.dirname(__file__), "..", "matrices"))
    ap.add_argument("--count", type=int, default=9, help="number of real matrices (default 9)")
    ap.add_argument("--min-nnz", type=int, default=500_000, help="smallest nnz considered (default 500000)")
    ap.add_argument("--no-powerlaw", action="store_true", help="skip generating powerlaw_1M.mtx")
    args = ap.parse_args()

    os.makedirs(args.dest, exist_ok=True)
    download(pick_matrices(args.count, args.min_nnz), args.dest)
    pl = os.path.join(args.dest, "powerlaw_1M.mtx")
    if not args.no_powerlaw and not os.path.exists(pl):
        make_powerlaw(pl)
    print(len(glob.glob(f"{args.dest}/*.mtx")), "matrices in", os.path.abspath(args.dest))


if __name__ == "__main__":
    main()
