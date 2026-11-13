#!/usr/bin/env python3
"""Exact upright re-fill around the tilted squares of an existing packing.

The tilted squares T of donor packing n are kept as a rigid set, placed in a box
of side s, and the upright squares are re-chosen as a maximum independent set
over corner-lattice and staircase candidates (the `fillfast` module of the
public evand packer, imported from a local checkout, not vendored here).

  refill.py capacity FULL.csv LO HI JOBS FILLFAST_DIR > capacity.csv
      own-box capacity: T(n) in the box of side L(n); count vs n.
  refill.py transplant FULL.csv LO HI JOBS FILLFAST_DIR SPAN > transplant.csv
      T(n) placed in the box of every target m in (n, n+SPAN] at side
      L(m) - 2e-4, for the 4 corner anchorings; count vs m.

A count >= target is only a *candidate*: fillfast tolerates 1e-6 overlaps, so a
hit still has to be LP-polished and certified by the official metric.
"""
import math
import multiprocessing as mp
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
import pandas as pd

UPRIGHT_TOL_DEG = 1e-7
SHRINK = 2e-4   # fillfast tolerates 1e-6 per contact; a chain of 25 contacts hides 2.5e-5, so test well below that
G = {}


def load(path):
    d = pd.read_csv(path)
    out = {}
    for n, g in d.groupby("n"):
        x, y, a = g.x.values.copy(), g.y.values.copy(), g.angle.values
        e = 0.5 * (np.abs(np.cos(a)) + np.abs(np.sin(a)))
        x0, y0 = (x - e).min(), (y - e).min()
        L = max((x + e).max() - x0, (y + e).max() - y0)
        deg = np.degrees(a) % 90.0
        tilt = np.minimum(deg, 90 - deg) > UPRIGHT_TOL_DEG
        out[int(n)] = (L, np.c_[x - x0, y - y0, deg][tilt])
    return out


def count(s, T):
    import fillfast
    if len(T) == 0:
        return int(math.floor(s + 1e-9)) ** 2
    return len(fillfast.fill(s, T, mode="exact", time_limit=20))


def cap_task(n):
    L, T = G["P"][n]
    return n, L, len(T), count(L + 1e-9, T)


def tr_task(args):
    n, m = args
    L, T = G["P"][n]
    s = G["P"][m][0] - SHRINK
    if len(T) == 0 or s < L:
        return None
    best = (-1, None)
    d = s - L
    for ax in (0.0, d):
        for ay in (0.0, d):
            c = count(s, T + np.array([ax, ay, 0.0]))
            if c > best[0]:
                best = (c, (ax > 0, ay > 0))
    return n, m, s, best[0], best[1]


def main():
    mode, path, lo, hi, jobs, ffdir = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
    sys.path.insert(0, ffdir)
    G["P"] = load(path)
    with mp.get_context("fork").Pool(jobs) as pool:
        if mode == "capacity":
            print("n,L,tilted,count,surplus")
            for n, L, t, c in pool.imap(cap_task, range(lo, hi + 1), chunksize=1):
                print(f"{n},{L:.9f},{t},{c},{c - n}", flush=True)
        else:
            span = int(sys.argv[7])
            tasks = [(n, m) for n in range(lo, hi + 1) for m in range(n + 1, min(hi, n + span) + 1)
                     if G["P"][m][0] > G["P"][n][0] + 1e-6]
            print("donor,target,s,count,surplus,anchor")
            for r in pool.imap_unordered(tr_task, tasks, chunksize=4):
                if r:
                    print(f"{r[0]},{r[1]},{r[2]:.9f},{r[3]},{r[3] - r[1]},{r[4]}", flush=True)


if __name__ == "__main__":
    main()
