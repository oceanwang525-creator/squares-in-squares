"""Scan the v2 family (two corner-anchored staircase slabs + free-angle block)."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from construct2 import build2  # noqa: E402

NEEDLE = {}


def task(args):
    L, theta_deg, m, w, bx, by, t, tp, caps = args
    cnt, P = build2(math.radians(theta_deg), m, w, bx, by, L, t, tp, caps=caps)
    return cnt, theta_deg, m, w, bx, by, t, tp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=float, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--tstep", type=float, default=4.0)
    ap.add_argument("--step", type=float, default=1.5)
    ap.add_argument("--gapk", type=str, default="0,1,2",
                    help="channel gaps k with t=tp=ncol-2-k")
    ap.add_argument("--ms", type=str, default="12,16,17,20,24")
    ap.add_argument("--ws", type=str, default="1,2,3,6,12,17")
    ap.add_argument("--jobs", type=int, default=30)
    ap.add_argument("--caps", action="store_true")
    ap.add_argument("--tag", type=str, default="")
    a = ap.parse_args()
    ncol = int(math.floor(a.L + 1e-9))
    thetas = list(np.arange(4.0, 86.0 + 1e-9, a.tstep))
    ms = [int(x) for x in a.ms.split(",")]
    ws = [int(x) for x in a.ws.split(",")]
    gaps = [int(x) for x in a.gapk.split(",")]
    pos = np.arange(0.0, a.L + 1e-9, a.step)
    tasks = []
    for th in thetas:
        for k in gaps:
            t = ncol - 2 - k
            if t < 1:
                continue
            for m in ms:
                for w in ws:
                    for bx in pos:
                        for by in pos:
                            tasks.append((a.L, th, m, w, bx, by, t, t, a.caps))
    t0 = time.time()
    best = []
    with Pool(a.jobs) as pool:
        for res in pool.imap_unordered(task, tasks, chunksize=32):
            cnt = res[0]
            if len(best) < 8 or cnt > best[-1][0]:
                best.append(res)
                best.sort(key=lambda r: -r[0])
                best = best[:8]
    dt = time.time() - t0
    print(f"n={a.n} L={a.L!r} tasks={len(tasks)} time={dt:.1f}s needle={a.n}")
    for r in best:
        print("  count=%d theta=%.2f m=%d w=%d bx=%.2f by=%.2f t=%d tp=%d" % r)
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs")
    os.makedirs(d, exist_ok=True)
    out = dict(n=a.n, L=a.L, tasks=len(tasks), seconds=dt, tag=a.tag,
               best=[dict(count=r[0], theta_deg=r[1], m=r[2], w=r[3], bx=r[4],
                          by=r[5], t=r[6], tp=r[7]) for r in best])
    with open(os.path.join(d, f"scan2_{a.n}{('_' + a.tag) if a.tag else ''}.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
