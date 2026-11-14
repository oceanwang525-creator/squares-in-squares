"""Driver: does a SMALL-`m` corner gadget achieve the target excess `c`?

For every (D, m) in the requested grid, search for a packing of `m^2 + D` unit
squares with side <= m + c_target, i.e. `L - m <= c_target`.  Such a packing is
exactly the corner gadget needed by the lift of `lift.py`:

    n = K^2 + D = (K^2 - m^2) + (m^2 + D),      L(K^2+D) <= K + (L_gadget - m)

for *every* K >= m.  Runs one worker process per (D, m) so the whole grid is
evaluated in one wall-clock pass at `--jobs` processes.

    python quicktest.py --ms 8,9,10 --Ds 6,7,8 --seconds 90 --jobs 9
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import overlap as ov  # noqa: E402
import search as S  # noqa: E402

DATA = HERE.parent / "data"
DONORS = DATA / "donors_small_n.csv"


def incumbent(n):
    rows = [r for r in csv.DictReader(open(DONORS)) if int(r["n"]) == n]
    if len(rows) != n:
        return None, None
    rows.sort(key=lambda r: int(r["square_id"]))
    q = ov.normalise(np.array([(float(r["x"]), float(r["y"]), float(r["angle"]))
                               for r in rows]))
    return q, ov.side(q)


def work(job):
    D, m, c_target, seconds, seed = job
    n = m * m + D
    target = m + c_target
    q0, L0 = incumbent(n)
    t0 = time.time()
    side, rows, stats = S.search_one(
        n, L0 if L0 else m * 1.4, time.time() + seconds, target=target,
        incumbent=q0, seed=seed, verbose=False)
    out = dict(D=D, m=m, n=n, c_target=c_target, target_side=target,
               incumbent_side=L0, best_side=side,
               best_c=(side - m) if np.isfinite(side) else None,
               hit=bool(np.isfinite(side) and side <= target),
               seconds=round(time.time() - t0, 1), **stats)
    if rows is not None and np.isfinite(side):
        with open(DATA / f"gadget_D{D}_m{m}.csv", "w", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(["n", "square_id", "x", "y", "angle"])
            for i, r in enumerate(rows):
                w.writerow([n, i, r["x"], r["y"], r["angle"]])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ms", default="8,9,10")
    ap.add_argument("--Ds", default="6,7,8")
    ap.add_argument("--c", type=float, default=0.6058845085404,
                    help="target excess c (frontier D=7 value at K=28)")
    ap.add_argument("--seconds", type=float, default=90.0)
    ap.add_argument("--jobs", type=int, default=9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(DATA / "quicktest.json"))
    a = ap.parse_args()

    jobs = [(D, m, a.c, a.seconds, a.seed + 17 * i)
            for i, (D, m) in enumerate(
                [(D, m) for D in [int(x) for x in a.Ds.split(",")]
                 for m in [int(x) for x in a.ms.split(",")]])]
    t0 = time.time()
    with mp.get_context("spawn").Pool(min(a.jobs, len(jobs))) as pool:
        res = pool.map(work, jobs)
    res.sort(key=lambda r: (r["D"], r["m"]))
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"{'D':>2} {'m':>3} {'n':>4} {'incumbent':>11} {'c_inc':>8} "
          f"{'best':>11} {'c_best':>8} {'target':>9} {'c_tgt':>7} hit starts")
    for r in res:
        inc = r["incumbent_side"]
        print(f"{r['D']:>2} {r['m']:>3} {r['n']:>4} "
              f"{inc if inc else float('nan'):>11.6f} "
              f"{(inc - r['m']) if inc else float('nan'):>8.6f} "
              f"{r['best_side']:>11.6f} {r['best_c']:>8.6f} "
              f"{r['target_side']:>9.4f} {r['c_target']:>7.4f} "
              f"{'YES' if r['hit'] else 'no':>3} {r['starts']:>6}")
    print(f"total {time.time() - t0:.0f}s -> {a.out}")


if __name__ == "__main__":
    main()
