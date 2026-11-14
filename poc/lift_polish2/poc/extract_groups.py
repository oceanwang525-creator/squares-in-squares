#!/usr/bin/env python3
"""Extract the current best group for a list of n from a submission CSV.

usage: extract_groups.py MERGED.csv N1,N2,... OUT.csv
Writes an incumbent-schema CSV (id,n,square_id,x,y,angle) with only those n.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path


def main() -> None:
    src = Path(sys.argv[1])
    targets = {int(x) for x in sys.argv[2].split(",") if x.strip()}
    dst = Path(sys.argv[3])
    n_out = 0
    with src.open(newline="", encoding="utf-8") as fh, \
            dst.open("w", newline="", encoding="utf-8") as fo:
        w = csv.writer(fo, lineterminator="\n")
        w.writerow(["id", "n", "square_id", "x", "y", "angle"])
        for row in csv.DictReader(fh):
            if int(row["n"]) in targets:
                w.writerow([n_out, row["n"], row["square_id"], row["x"], row["y"],
                            row["angle"]])
                n_out += 1
    print(f"wrote {dst}: {n_out} rows for n={sorted(targets)}")


if __name__ == "__main__":
    main()
