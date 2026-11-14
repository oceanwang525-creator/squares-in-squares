#!/usr/bin/env python3
"""Write the deliverable `final.csv` (header `id,x,y,deg`, `s`-prefixed decimals).

Rows come from an incumbent-schema CSV (`id,n,square_id,x,y,angle`, radians) —
normally the output of `assemble.py`, i.e. the best certified candidate per n.
Degrees are produced with mpmath at 75 significant digits, as the coordinator
HOWTO prescribes, and every decimal field is `s`-prefixed so pandas keeps it as
text.

usage: mk_final.py IN.csv OUT.csv
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import mpmath as mp


def main() -> None:
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    groups: dict[int, list[dict[str, str]]] = {}
    with src.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            groups.setdefault(int(row["n"]), []).append(row)
    with dst.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["id", "x", "y", "deg"])
        with mp.workdps(95):
            for n in sorted(groups):
                rows = sorted(groups[n], key=lambda r: int(r["square_id"]))
                assert len(rows) == n, (n, len(rows))
                for r in rows:
                    deg = mp.nstr(mp.mpf(r["angle"]) * 180 / mp.pi, 75)
                    w.writerow([f"{n:03}_{r['square_id']}", "s" + r["x"], "s" + r["y"],
                                "s" + deg])
    print(f"wrote {dst}: {len(groups)} groups {sorted(groups)}, "
          f"{sum(len(v) for v in groups.values())} rows")


if __name__ == "__main__":
    main()
