#!/usr/bin/env python3
"""Build a donor-row CSV: fresh main with the polished rows substituted in.

`poc/pocket_lift_scan/poc/build_cert.py` copies every pocket from the decimal
strings of its donor row, looked up by `src` in one CSV.  Once the donors are
themselves polished here, the donor table must be main **plus** the polished
rows, with the polished version winning for the `n` it covers.

usage: mk_donors.py POLISHED.csv OUT.csv [--main path/to/merged.csv]
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

MAIN = "poc/pell_corrugation/data/border-scan/merged.csv"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("polished")
    ap.add_argument("out")
    ap.add_argument("--main", default=MAIN)
    args = ap.parse_args()

    mine: dict[int, list[dict]] = {}
    with open(args.polished, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            mine.setdefault(int(row["n"]), []).append(row)

    gid = 0
    kept = 0
    with Path(args.out).open("w", newline="", encoding="utf-8") as fo:
        w = csv.writer(fo, lineterminator="\n")
        w.writerow(["id", "n", "square_id", "x", "y", "angle"])
        with open(args.main, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if int(row["n"]) in mine:
                    continue
                w.writerow([gid, row["n"], row["square_id"], row["x"], row["y"],
                            row["angle"]])
                gid += 1
                kept += 1
        for n in sorted(mine):
            for row in sorted(mine[n], key=lambda r: int(r["square_id"])):
                w.writerow([gid, n, row["square_id"], row["x"], row["y"], row["angle"]])
                gid += 1
    print(f"wrote {args.out}: {kept} main rows + {len(mine)} replaced groups "
          f"{sorted(mine)} = {gid} rows")


if __name__ == "__main__":
    main()
