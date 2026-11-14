#!/usr/bin/env python3
"""Reconstruct the C1 winning groups as an incumbent-format CSV.

Reads `poc/pocket_lift_scan/data/built/n{...}.json` (square_id, x, y, angle in
radians, 50-digit decimal strings) and writes an incumbent-schema CSV
(`id,n,square_id,x,y,angle`), which is what `alln_fixed_lp`, `lip_slp` and
`swarm_lpslack` all expect.  Feed the result to
`poc/faultline_dp2/poc/mkreport.py` to get the official 50-digit side per n.

usage: mk_c1.py OUT.csv [n ...]
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
BUILT = REPO / "poc/pocket_lift_scan/data/built"
ALL = [442, 447, 485, 530, 577, 743, 798]


def load(n: int):
    return json.loads((BUILT / f"n{n}.json").read_text(encoding="utf-8"))


def main() -> None:
    out = Path(sys.argv[1])
    ns = [int(x) for x in sys.argv[2:]] or ALL
    gid = 0
    rows = 0
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["id", "n", "square_id", "x", "y", "angle"])
        for n in ns:
            group = sorted(load(n), key=lambda r: int(r["square_id"]))
            assert len(group) == n, (n, len(group))
            for r in group:
                w.writerow([gid, n, r["square_id"], r["x"], r["y"], r["angle"]])
                gid += 1
                rows += 1
    print(f"wrote {out} ({len(ns)} groups, {rows} rows): {ns}")


if __name__ == "__main__":
    main()
