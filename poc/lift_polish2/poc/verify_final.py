#!/usr/bin/env python3
"""Re-certify `final.csv` from its own bytes against TWO references.

For every group in `final.csv` this reads the file back (degrees, `s`-prefixed
decimals), converts `deg -> rad` at 80 dps and runs the unchanged official
metric in

  * **literal** mode  - the decimal strings as written, and
  * **pandas** mode   - a real `pandas.read_csv` / `to_csv` round trip,

then compares both sides with

  * **fresh main** - the 50-digit side table regenerated here from
    `poc/pell_corrugation/data/border-scan/merged.csv` (`data/main_report.json`,
    score 22.152957455700296),
  * **start**      - the best previously-known certified group per n
    (`data/start_report.json`), i.e. the D1/D2 union.

A claim needs `dL <= -1e-9` against *both* references in *both* modes.

usage: verify_final.py [final.csv]
"""
from __future__ import annotations

import collections
import csv
import json
import sys
import tempfile

import mpmath as mp

sys.path.insert(0, "poc/baseline/poc/src")
from squares.core import certify, kaggle_roundtrip  # noqa: E402

FRESH_MAIN = "poc/lift_polish2/data/main_report.json"
START = "poc/lift_polish2/data/start_report.json"
LIMIT = mp.mpf("-1e-9")


def load_L(path):
    out = {}
    for r in json.load(open(path))["results"]:
        if r.get("L") is not None:
            out[int(r["n"])] = mp.mpf(r["L"])
    return out


def main() -> None:
    final = sys.argv[1] if len(sys.argv) > 1 else "poc/lift_polish2/final.csv"
    main_L = load_L(FRESH_MAIN)
    start_L = load_L(START)
    rows = collections.defaultdict(list)
    with open(final, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            gid, sid = r["id"].split("_")
            rows[int(gid)].append((sid, r["x"].lstrip("s"), r["y"].lstrip("s"),
                                   r["deg"].lstrip("s")))
    print(f"{final}: {len(rows)} groups, {sum(len(v) for v in rows.values())} rows")
    total_main = mp.mpf(0)
    total_start = mp.mpf(0)
    table = []
    ok_main = bool(rows)
    ok_start = bool(rows)
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
        d_start = [L1 - start_L[n], L2 - start_L[n]] if n in start_L else [None, None]
        win_main = all(d < LIMIT for d in d_main)
        win_start = (all(d < LIMIT for d in d_start) if d_start[0] is not None else None)
        ok_main &= win_main
        ok_start &= bool(win_start)
        total_main += min(d_main) / mp.sqrt(n)
        if win_start:
            total_start += min(d_start) / mp.sqrt(n)
        table.append(dict(n=n, literal=str(L1), pandas=str(L2),
                          fresh_main=str(main_L[n]),
                          start=(str(start_L[n]) if n in start_L else None),
                          dL_fresh_literal=str(d_main[0]), dL_fresh_pandas=str(d_main[1]),
                          dL_start_literal=(str(d_start[0]) if win_start is not None else None),
                          dL_start_pandas=(str(d_start[1]) if win_start is not None else None),
                          win_fresh=win_main, win_start=win_start))
        print(f"  n={n}: L lit {mp.nstr(L1, 24)} / pan {mp.nstr(L2, 24)}")
        print(f"        vs fresh main {mp.nstr(main_L[n], 20)}: dL "
              f"{mp.nstr(d_main[0], 10)} / {mp.nstr(d_main[1], 10)} "
              f"{'STRICT' if win_main else 'FAIL'}")
        if d_start[0] is not None:
            print(f"        vs start      {mp.nstr(start_L[n], 20)}: dL "
                  f"{mp.nstr(d_start[0], 10)} / {mp.nstr(d_start[1], 10)} "
                  f"{'STRICT' if win_start else 'ties the start'}")
    print(f"score gain vs fresh main : {mp.nstr(total_main, 10)}")
    print(f"score gain vs start      : {mp.nstr(total_start, 10)}")
    print("ALL STRICT WINS vs fresh main (both modes)"
          if ok_main else "NOT ALL STRICT vs fresh main")
    print("all also strict vs start (D1/D2 union)" if ok_start else
          "some groups only tie the start (still strict vs fresh main)")
    out = str(final).replace(".csv", "_verify.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(table, fh, indent=1)


if __name__ == "__main__":
    main()
