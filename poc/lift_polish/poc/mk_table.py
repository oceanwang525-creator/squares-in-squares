#!/usr/bin/env python3
"""Render the per-n markdown table for the README from `verify_final_verify.json`.

usage: mk_table.py VERIFY.json [gap_vs_frontier.csv]
Columns: n, C1 side, best side, dL literal / pandas (vs C1 and vs fresh main),
residual to the all-team frontier, score gain.
"""
from __future__ import annotations

import csv
import json
import math
import sys

GAP = "poc/koboshchan_coordinator/data/gap_vs_frontier.csv"


def main() -> None:
    rows = json.load(open(sys.argv[1]))
    gap_path = sys.argv[2] if len(sys.argv) > 2 else GAP
    frontier = {int(r["n"]): float(r["frontier"]) for r in csv.DictReader(open(gap_path))}
    total_c1 = 0.0
    total_main = 0.0
    print("| n | C1 side | best side (literal / pandas) | dL vs C1 (lit / pan) | "
          "dL vs main (lit / pan) | residual to frontier | score gain |")
    print("|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        n = r["n"]
        d1 = float(r["dL_c1_literal"])
        d2 = float(r["dL_c1_pandas"])
        m1 = float(r["dL_fresh_literal"])
        m2 = float(r["dL_fresh_pandas"])
        gain = min(d1, d2) / math.sqrt(n)
        total_c1 += -gain
        total_main += -min(m1, m2) / math.sqrt(n)
        print(f"| {n} | {float(r['c1']):.12f} | {float(r['literal']):.12f} / "
              f"{float(r['pandas']):.12f} | {d1:+.6e} / {d2:+.6e} | "
              f"{m1:+.6e} / {m2:+.6e} | {float(r['literal']) - frontier[n]:.6e} | "
              f"{-gain:+.4e} |")
    print(f"\nsum of score gains vs C1: {-total_c1:.6e}   "
          f"vs fresh main: {-total_main:.6e}")


if __name__ == "__main__":
    main()
