#!/usr/bin/env python3
"""Markdown table of rows, nnz, mean/max nonzeros per row, ELL padding ratio and HYB k
for each .mtx file (standard library only).  Usage: matrix_stats.py matrices/*.mtx"""
import os
import sys

print("| matrix | rows | nnz | mean/row | max/row | ELL padding | HYB k | HYB COO % |")
print("|---|---|---|---|---|---|---|---|")
for path in sys.argv[1:]:
    with open(path) as f:
        symmetric = f.readline().lower().split()[4] == "symmetric"
        line = f.readline()
        while line.startswith("%"):
            line = f.readline()
        rows, _, entries = (int(t) for t in line.split()[:3])
        n = [0] * rows  # nonzeros per row
        for _ in range(entries):
            r, c = (int(t) - 1 for t in f.readline().split()[:2])
            n[r] += 1
            if symmetric and r != c:
                n[c] += 1
    nnz, mx = sum(n), max(n)
    hist = [0] * (mx + 1)
    for length in n:
        hist[length] += 1
    k, at_least = 0, 0
    for length in range(mx, 0, -1):  # same rule as hyb_choose_k in convert.cpp
        at_least += hist[length]
        if 3 * at_least >= rows:
            k = length
            break
    coo = sum(max(0, length - k) for length in n)
    name = os.path.splitext(os.path.basename(path))[0]
    print(f"| {name} | {rows} | {nnz} | {nnz / rows:.1f} | {mx} | {rows * mx / nnz:.1f} | {k} | {100 * coo / nnz:.1f} |")
