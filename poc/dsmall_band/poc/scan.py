"""Scan the dsmall_band family at a fixed box side L and report the best count."""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from construct import build  # noqa: E402

M_LIST = [8, 12, 16, 20, 24, 28]
W_LIST = [1, 2, 3, 4, 6, 8, 12, 17]


def task(args):
    L, theta_deg, m, w, bx, by, caps = args
    cnt, P = build(math.radians(theta_deg), m, w, bx, by, L, caps=caps)
    return cnt, theta_deg, m, w, bx, by


def scan(L, thetas, ms, ws, step, jobs, keep_n=5, caps=False):
    pos = np.arange(0.0, L + 1e-9, step)
    tasks = [(L, th, m, w, bx, by, caps)
             for th in thetas for m in ms for w in ws
             for bx in pos for by in pos]
    t0 = time.time()
    best = []
    with Pool(jobs) as pool:
        for i, res in enumerate(pool.imap_unordered(task, tasks, chunksize=64)):
            cnt = res[0]
            if len(best) < keep_n or cnt > best[0][0]:
                best.append(res)
                best.sort(key=lambda r: -r[0])
                best = best[:keep_n]
    dt = time.time() - t0
    return best, len(tasks), dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=float, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--tstep", type=float, default=4.0)
    ap.add_argument("--tlo", type=float, default=4.0)
    ap.add_argument("--thi", type=float, default=86.0)
    ap.add_argument("--step", type=float, default=1.0)
    ap.add_argument("--jobs", type=int, default=30)
    ap.add_argument("--ms", type=str, default="")
    ap.add_argument("--ws", type=str, default="")
    ap.add_argument("--tag", type=str, default="")
    a = ap.parse_args()
    ms = [int(x) for x in a.ms.split(",")] if a.ms else M_LIST
    ws = [int(x) for x in a.ws.split(",")] if a.ws else W_LIST
    thetas = list(np.arange(a.tlo, a.thi + 1e-9, a.tstep))
    best, ntask, dt = scan(a.L, thetas, ms, ws, a.step, a.jobs)
    print(f"n={a.n} L={a.L!r} tasks={ntask} time={dt:.1f}s  needle={a.n}")
    for r in best:
        print("  count=%d theta=%.3f m=%d w=%d bx=%.2f by=%.2f" % r)
    out = dict(n=a.n, L=a.L, tasks=ntask, seconds=dt, tag=a.tag,
               best=[dict(count=r[0], theta_deg=r[1], m=r[2], w=r[3],
                          bx=r[4], by=r[5]) for r in best])
    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs"),
                exist_ok=True)
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs",
                        f"scan_{a.n}{('_' + a.tag) if a.tag else ''}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", os.path.normpath(path))


if __name__ == "__main__":
    main()
