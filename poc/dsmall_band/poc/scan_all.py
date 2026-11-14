"""Run the dsmall_band families over every target side, leaner and JSON-safe.

v1 is bounded analytically (a single-lattice upright grid can never exceed
ncol**2), so it is only scanned on a reduced grid for the record; v2 and v3 get
the search budget.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from construct import build  # noqa: E402
from construct2 import build2  # noqa: E402
from construct3 import build3  # noqa: E402

TARGETS = [
    (901, 30.5021527079077, 1.0),
    (842, 29.5021527079033, 1.0),
    (962, 31.5021527079108, 1.0),
    (902, 30.5140787741393, 1.0),
    (963, 31.5140787741295, 1.0),
    (843, 29.5182442831318, 1.0),
    (328, 18.5995354648065, 0.5),
    (365, 19.5988242989228, 0.5),
]
M_LIST = [8, 12, 16, 20, 24]
W_LIST = [1, 2, 3, 6, 12, 17]


def t_v1(a):
    L, th, m, w, bx, by = a
    c, _ = build(math.radians(th), m, w, bx, by, L, caps=False)
    return (int(c), float(th), m, w, float(bx), float(by), 0, 0)


def t_v2(a):
    L, th, m, w, bx, by = a
    ncol = int(math.floor(L + 1e-9))
    best = (0, 0)
    for k in (0, 1, 2):
        t = ncol - 2 - k
        if t < 1:
            continue
        c, _ = build2(math.radians(th), m, w, bx, by, L, t, t, caps=False)
        if c > best[0]:
            best = (int(c), t)
    return (best[0], float(th), m, w, float(bx), float(by), best[1], best[1])


def t_v3(a):
    L, th, m, w, bx, by = a
    ncol = int(math.floor(L + 1e-9))
    best = (0, 0, 0)
    for ic in (0, ncol // 2, ncol - 2):
        for jc in (0, ncol // 2, ncol - 2):
            c, _ = build3(math.radians(th), m, w, bx, by, L, ic, jc, caps=False)
            if c > best[0]:
                best = (int(c), ic, jc)
    return (best[0], float(th), m, w, float(bx), float(by), best[1], best[2])


def run(n, L, step, jobs=30, budgets=(("v1", t_v1), ("v2", t_v2), ("v3", t_v3))):
    pos = np.arange(0.0, L + 1e-9, step)
    thetas = np.arange(6.0, 86.0 + 1e-9, 6.0)
    tasks = [(float(L), float(th), int(m), int(w), float(bx), float(by))
             for th in thetas for m in M_LIST for w in W_LIST
             for bx in pos for by in pos]
    out = {}
    for name, fn in budgets:
        t0 = time.time()
        best = []
        with Pool(jobs) as pool:
            for r in pool.imap_unordered(fn, tasks, chunksize=32):
                if len(best) < 5 or r[0] > best[-1][0]:
                    best.append(r)
                    best.sort(key=lambda z: -z[0])
                    best = best[:5]
        out[name] = dict(seconds=round(time.time() - t0, 1),
                         best=[dict(count=r[0], theta_deg=r[1], m=r[2], w=r[3],
                                    bx=r[4], by=r[5], p1=r[6], p2=r[7])
                               for r in best])
        print(f"  {name}: {out[name]['best'][0]['count']} "
              f"(theta={out[name]['best'][0]['theta_deg']} "
              f"m={out[name]['best'][0]['m']} w={out[name]['best'][0]['w']}) "
              f"{out[name]['seconds']}s", flush=True)
    return dict(n=n, L=L, step=step, tasks=len(tasks), families=out)


def main():
    res = []
    for n, L, step in TARGETS:
        print(f"n={n} L={L} need={n}", flush=True)
        res.append(run(n, L, step))
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "summary.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("wrote docs/summary.json")


if __name__ == "__main__":
    main()
