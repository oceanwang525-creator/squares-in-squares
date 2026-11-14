#!/usr/bin/env python3
"""Re-certify `final.csv` from its own bytes against TWO references.

For every group in `final.csv` this reads the file back (degrees, `s`-prefixed
decimals), converts `deg -> rad` at 80 dps and runs the unchanged official
metric in

  * **literal** mode  - the decimal strings as written, and
  * **pandas** mode   - a real `pandas.read_csv` / `to_csv` round trip,

then compares both sides with

  * **fresh main** - the official 50-digit side table of
    `poc/pell_corrugation/data/border-scan/merged.csv` (score 22.153247131307513),
  * **C1** - the official side of the `poc/pocket_lift_scan` lift being polished.

A claim needs `dL <= -1e-9` against *both* references in *both* modes.

usage: verify_final.py [final.csv]
"""
from __future__ import annotations

import collections
import csv
import json
import sys

import mpmath as mp

sys.path.insert(0, "poc/baseline/poc/src")
from squares.core import certify, kaggle_roundtrip  # noqa: E402

FRESH_MAIN = "poc/pocket_lift_scan/data/report.json"
C1 = "poc/lift_polish/data/c1_report.json"
LIMIT = mp.mpf("-1e-9")


def load_L(path):
    out = {}
    for r in json.load(open(path))["results"]:
        if r["L"] is not None:
            out[int(r["n"])] = mp.mpf(r["L"])
    return out


def main() -> None:
    final = sys.argv[1] if len(sys.argv) > 1 else "poc/lift_polish/final.csv"
    main_L = load_L(FRESH_MAIN)
    c1_L = load_L(C1)
    rows = collections.defaultdict(list)
    with open(final, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            gid, sid = r["id"].split("_")
            rows[int(gid)].append((sid, r["x"].lstrip("s"), r["y"].lstrip("s"),
                                   r["deg"].lstrip("s")))
    print(f"{final}: {len(rows)} groups, {sum(len(v) for v in rows.values())} rows")
    total_main = mp.mpf(0)
    total_c1 = mp.mpf(0)
    table = []
    ok_all = bool(rows)
    for n in sorted(rows):
        if len(rows[n]) != n:
            print(f"  n={n}: WRONG COUNT {len(rows[n])}")
            ok_all = False
            continue
        with mp.workdps(80):
            r = [dict(square_id=sid, x=x, y=y,
                      angle=mp.nstr(mp.mpf(d) * mp.pi / 180, 60))
                 for sid, x, y, d in sorted(rows[n], key=lambda t: int(t[0]))]
        L1 = mp.mpf(certify(n, r))
        L2 = mp.mpf(certify(n, kaggle_roundtrip({n: r})[n]))
        d_main = [L1 - main_L[n], L2 - main_L[n]]
        d_c1 = [L1 - c1_L[n], L2 - c1_L[n]]
        win_main = all(d < LIMIT for d in d_main)
        win_c1 = all(d < LIMIT for d in d_c1)
        ok_all &= win_main and win_c1
        total_main += min(d_main) / mp.sqrt(n)
        total_c1 += min(d_c1) / mp.sqrt(n)
        table.append(dict(n=n, literal=str(L1), pandas=str(L2),
                          fresh_main=str(main_L[n]), c1=str(c1_L[n]),
                          dL_fresh_literal=str(d_main[0]), dL_fresh_pandas=str(d_main[1]),
                          dL_c1_literal=str(d_c1[0]), dL_c1_pandas=str(d_c1[1]),
                          win_fresh=win_main, win_c1=win_c1))
        print(f"  n={n}: L lit {mp.nstr(L1, 24)} / pan {mp.nstr(L2, 24)}")
        print(f"        vs fresh main {mp.nstr(main_L[n], 20)}: dL "
              f"{mp.nstr(d_main[0], 10)} / {mp.nstr(d_main[1], 10)} "
              f"{'STRICT' if win_main else 'FAIL'}")
        print(f"        vs C1        {mp.nstr(c1_L[n], 20)}: dL "
              f"{mp.nstr(d_c1[0], 10)} / {mp.nstr(d_c1[1], 10)} "
              f"{'STRICT' if win_c1 else 'FAIL'}")
    print(f"score gain vs fresh main : {mp.nstr(total_main, 10)}")
    print(f"score gain vs C1         : {mp.nstr(total_c1, 10)}")
    print("ALL STRICT WINS (both refs, both modes)" if ok_all else "NOT ALL STRICT")
    json.dump(table, open(str(final).replace(".csv", "_verify.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
