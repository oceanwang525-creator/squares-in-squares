#!/usr/bin/env python3
"""Per-n round table + frontier residuals for the README.

Reads the 50-digit side tables produced by `poc/faultline_dp2/poc/mkreport.py`
for each round, the verified `final.csv` sides, and the frontier column of
`poc/koboshchan_coordinator/data/gap_vs_frontier.csv`.

usage: mk_table.py
"""
from __future__ import annotations

import csv
import json

import mpmath as mp

D = "poc/lift_polish2/data"
ROUNDS = [("main", f"{D}/main_report.json"), ("start", f"{D}/start_report.json"),
          ("r1", f"{D}/polished1_report.json"), ("r2", f"{D}/polished2_report.json"),
          ("r3", f"{D}/polished3_report.json"), ("r4", f"{D}/polished4_report.json")]
NS = [442, 447, 485, 492, 530, 577, 740, 743, 798]


def load(path):
    out = {}
    for r in json.load(open(path))["results"]:
        if r.get("L") is not None:
            out[int(r["n"])] = mp.mpf(r["L"])
    return out


def main() -> None:
    tabs = {tag: load(p) for tag, p in ROUNDS}
    front = {}
    with open("poc/koboshchan_coordinator/data/gap_vs_frontier.csv", newline="",
              encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            front[int(row["n"])] = mp.mpf(row["frontier"])
    ver = {int(r["n"]): r for r in json.load(open("poc/lift_polish2/final_verify.json"))}
    hdr = ("| n | fresh main | start (D1/D2) | round 1 | round 2 | round 3 | round 4 "
           "| final | dL vs main (lit = pan) | dL vs start | frontier | residual |")
    print(hdr)
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    tot_main = mp.mpf(0)
    tot_start = mp.mpf(0)
    for n in NS:
        v = ver[n]
        L = mp.mpf(v["literal"])
        d_main = mp.mpf(v["dL_fresh_literal"])
        d_start = mp.mpf(v["dL_start_literal"]) if v.get("dL_start_literal") else mp.mpf(0)
        tot_main += d_main / mp.sqrt(n)
        tot_start += d_start / mp.sqrt(n)
        cells = [mp.nstr(tabs[t][n], 13) for t, _ in ROUNDS]
        print(f"| {n} | " + " | ".join(cells) + f" | {mp.nstr(L, 13)} | "
              f"{mp.nstr(d_main, 6)} | {mp.nstr(d_start, 6)} | "
              f"{mp.nstr(front[n], 10)} | {mp.nstr(L - front[n], 6)} |")
    print(f"| **total** | | | | | | | | **{mp.nstr(tot_main, 8)}** | "
          f"**{mp.nstr(tot_start, 8)}** | | |")


if __name__ == "__main__":
    main()
