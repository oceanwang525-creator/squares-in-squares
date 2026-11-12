#!/usr/bin/env python3
"""Exact set-packing LNS for n=911 in the frontier box.

Generate candidate unit squares on a 0.5 grid and 8 orientations. Build the
conflict graph. Seed with the 30x30 upright grid (900). Then repeatedly solve
an exact maximum-independent-set MILP in a random window, keeping the fixed
selection outside, and accept improvements (or occasional equal-count
rearrangements for diversification). Any selection of >=911 is certified with
the official metric.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from decimal import Decimal

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree

ROOT = "/mnt/e/squares-in-squares"
sys.path.insert(0, ROOT + "/poc/baseline/poc/src")
from squares import core  # noqa: E402
from squares import _official_metric as metric  # noqa: E402

metric.KAGGLE_CORES = 1
L = 30.6609125615235
H = 0.5
ANGLES = [math.radians(a) for a in (0, 15, 25, 35, 45, 55, 65, 75)]


def gen_candidates():
    cands = []
    for th in ANGLES:
        e = 0.5 * (abs(math.cos(th)) + abs(math.sin(th)))
        xs = np.arange(e, L - e + 1e-9, H)
        for x in xs:
            for y in xs:
                cands.append((x, y, th))
    return np.array(cands, dtype=float)


def conflicts(C):
    tree = cKDTree(C[:, :2])
    pairs = tree.query_pairs(1.45, output_type="ndarray")
    i = pairs[:, 0]; j = pairs[:, 1]
    dx = C[j, 0] - C[i, 0]; dy = C[j, 1] - C[i, 1]
    ti = C[i, 2]; tj = C[j, 2]
    ci = np.cos(ti); si = np.sin(ti)
    cj = np.cos(tj); sj = np.sin(tj)
    dt = tj - ti
    rad = 0.5 * (1.0 + np.abs(np.cos(dt)) + np.abs(np.sin(dt)))
    p = np.abs(ci * dx + si * dy)
    q = np.abs(-si * dx + ci * dy); p = np.maximum(p, q)
    q = np.abs(cj * dx + sj * dy); p = np.maximum(p, q)
    q = np.abs(-sj * dx + cj * dy); p = np.maximum(p, q)
    ov = p < rad - 1e-9
    return pairs[ov]


def csr(n, edges):
    src = np.concatenate([edges[:, 0], edges[:, 1]])
    dst = np.concatenate([edges[:, 1], edges[:, 0]])
    order = np.argsort(src, kind="stable")
    src = src[order]; dst = dst[order]
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.add.at(indptr, src + 1, 1)
    indptr = np.cumsum(indptr)
    return indptr, dst


def solve_window(selected, win_idx, edges, indptr, indices):
    nw = len(win_idx)
    m = np.full(len(selected), -1, dtype=np.int64)
    m[win_idx] = np.arange(nw)
    # drop candidates conflicting with fixed selected squares
    keep = np.ones(nw, dtype=bool)
    for k, c in enumerate(win_idx):
        for p in range(indptr[c], indptr[c + 1]):
            q = indices[p]
            if selected[q] and m[q] < 0:
                keep[k] = False
                break
    win_idx2 = win_idx[keep]
    nw2 = len(win_idx2)
    if nw2 == 0:
        return None
    m2 = np.full(len(selected), -1, dtype=np.int64)
    m2[win_idx2] = np.arange(nw2)
    mask = (m2[edges[:, 0]] >= 0) & (m2[edges[:, 1]] >= 0)
    e2 = edges[mask]
    if len(e2):
        rows = np.arange(len(e2)); cols0 = m2[e2[:, 0]]; cols1 = m2[e2[:, 1]]
        A = coo_matrix((np.ones(2 * len(e2)),
                        (np.concatenate([rows, rows]),
                         np.concatenate([cols0, cols1]))),
                       shape=(len(e2), nw2)).tocsr()
        cons = LinearConstraint(A, -np.inf, 1.0)
    else:
        cons = ()
    res = milp(c=-np.ones(nw2), constraints=cons,
               integrality=np.ones(nw2), bounds=Bounds(0, 1),
               options={"time_limit": 10.0, "presolve": True})
    if not res.success:
        return None
    pick = res.x > 0.5
    return win_idx2[pick]


def main() -> None:
    t0 = time.time()
    C = gen_candidates()
    print("candidates", len(C), flush=True)
    E = conflicts(C)
    print("edges", len(E), f"{time.time()-t0:.1f}s", flush=True)
    indptr, indices = csr(len(C), E)
    # seed: 30x30 upright grid
    sel = np.zeros(len(C), dtype=bool)
    # index candidates by (x,y,angle)
    key = {}
    for k in range(len(C)):
        key[(round(C[k, 0], 6), round(C[k, 1], 6), round(C[k, 2], 9))] = k
    for i in range(30):
        for j in range(30):
            k = key[(i + 0.5, j + 0.5, 0.0)]
            sel[k] = True
    print("seed count", int(sel.sum()), flush=True)
    best_count = int(sel.sum())
    best_sel = sel.copy()
    rng = np.random.default_rng(99)
    W = 8.0
    for it in range(150):
        if rng.random() < 0.5:
            x0 = rng.choice([0.0, L - W])
        else:
            x0 = rng.uniform(0.0, L - W)
        if rng.random() < 0.5:
            y0 = rng.choice([0.0, L - W])
        else:
            y0 = rng.uniform(0.0, L - W)
        inside = np.where((C[:, 0] >= x0) & (C[:, 0] <= x0 + W) &
                          (C[:, 1] >= y0) & (C[:, 1] <= y0 + W))[0]
        if len(inside) == 0:
            continue
        cur = int(sel[inside].sum())
        new = solve_window(sel, inside, E, indptr, indices)
        if new is None:
            continue
        later = int(sel[inside].sum()) - cur
        gain = len(new) - cur
        if gain > 0 or (gain == 0 and rng.random() < 0.15):
            sel[inside] = False
            sel[new] = True
            cnt = int(sel.sum())
            if cnt > best_count:
                best_count = cnt
                best_sel = sel.copy()
                print(f"it {it} window ({x0:.2f},{y0:.2f}) count {cnt} "
                      f"{time.time()-t0:.1f}s", flush=True)
                if cnt >= 911:
                    break
    print("best count", best_count, flush=True)
    if best_count >= 911:
        rows = [dict(square_id=str(k), x=repr(float(C[c, 0])),
                     y=repr(float(C[c, 1])), angle=repr(float(C[c, 2])))
                for k, c in enumerate(np.where(best_sel)[0])]
        rows = rows[:911]
        for eps in (0.0, 1e-12, 1e-11, 1e-10, 1e-9):
            rr = rows
            if eps > 0:
                a = np.array([[float(r["x"]), float(r["y"]),
                               float(r["angle"])] for r in rows])
                e = 0.5 * (np.abs(np.cos(a[:, 2])) + np.abs(np.sin(a[:, 2])))
                cx = ((a[:, 0] - e).min() + (a[:, 0] + e).max()) / 2
                cy = ((a[:, 1] - e).min() + (a[:, 1] + e).max()) / 2
                a[:, 0] = cx + (a[:, 0] - cx) * (1 + eps)
                a[:, 1] = cy + (a[:, 1] - cy) * (1 + eps)
                rr = [dict(square_id=str(i), x=repr(float(a[i, 0])),
                           y=repr(float(a[i, 1])), angle=repr(float(a[i, 2])))
                      for i in range(911)]
            try:
                lit = Decimal(core.certify(911, rr))
                tr = core.kaggle_roundtrip({911: rr})
                pan = Decimal(core.certify(911, tr[911]))
                ours = Decimal(core.certify(911, [r for r in
                    __import__("csv").DictReader(open(
                        ROOT + "/poc/pell_corrugation/data/border-scan/merged.csv"))
                    if int(r["n"]) == 911]))
                print("CERT", lit, pan, "ours", ours, flush=True)
                if lit < ours and pan < ours:
                    json.dump({"911": dict(old_L=str(ours), new_L=str(max(lit, pan)),
                                           literal_L=str(lit), pandas_L=str(pan),
                                           rows=rr)},
                              open("/tmp/opencode/setpack_win.json", "w"), indent=1)
                    print("WIN", flush=True)
                break
            except Exception as ex:
                print("cert eps", eps, "fail", str(ex)[:80], flush=True)


if __name__ == "__main__":
    main()
