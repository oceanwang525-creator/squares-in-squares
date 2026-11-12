#!/usr/bin/env python3
"""Margin-tight fixed-angle LP polish (delta form, dual simplex).

`poc/alln_fixed_lp` keeps a 1e-9 safety margin on every wall and pair row, so a
saturated packing comes back ~1.5e-8..2.7e-8 *above* the incumbent and any
slack smaller than that is invisible.  Here the LP is written in displacement
variables around the current coordinates (right-hand sides are the actual gaps,
so they are resolved to ~1e-15), the pair margin is 5e-13, walls carry no
margin, and the result is re-measured with a float SAT before the official
gate.  Angles are never changed.

usage: tight_lp.py FULL.csv LO HI JOBS OUT_ROWS.csv REPORT.json [ROUNDS] [TRUST]
Every emitted group is strictly below the input group in BOTH official
transports (literal strings and pandas round trip).
"""
import csv
import json
import multiprocessing as mp
import os
import sys
from decimal import Decimal
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rows as R  # noqa: E402

MARGIN = 5e-13
SCALE = 1e-7      # LP variables are displacements in units of SCALE (keeps HiGHS well conditioned)
RADIUS = 1.6


def pair_axes(c, th):
    """For every nearby pair: best separating axis, signed toward j, and its gap."""
    u = np.c_[np.cos(th), np.sin(th)]
    v = np.c_[-np.sin(th), np.cos(th)]
    pr = np.array(sorted(cKDTree(c).query_pairs(RADIUS)))
    i, j = pr[:, 0], pr[:, 1]
    d = c[j] - c[i]
    best_gap = np.full(len(pr), -np.inf)
    best_ax = np.zeros((len(pr), 2))
    for ax in (u[i], v[i], u[j], v[j]):
        proj = np.einsum("ij,ij->i", ax, d)
        rad = 0.5 * (np.abs(np.einsum("ij,ij->i", ax, u[i])) + np.abs(np.einsum("ij,ij->i", ax, v[i]))
                     + np.abs(np.einsum("ij,ij->i", ax, u[j])) + np.abs(np.einsum("ij,ij->i", ax, v[j])))
        gap = np.abs(proj) - rad
        better = gap > best_gap
        best_gap[better] = gap[better]
        best_ax[better] = (ax * np.sign(proj)[:, None])[better]
    return i, j, best_ax, best_gap


def side(c, e):
    return max((c[:, 0] + e).max() - (c[:, 0] - e).min(), (c[:, 1] + e).max() - (c[:, 1] - e).min())


def lp_round(c, th, e, trust, repair=False):
    n = len(c)
    i, j, a, gap = pair_axes(c, th)
    k = np.arange(len(i))
    # a.(d_i - d_j) <= gap - MARGIN
    rr = np.r_[k, k, k, k]
    cc = np.r_[2 * i, 2 * i + 1, 2 * j, 2 * j + 1]
    vv = np.r_[a[:, 0], a[:, 1], -a[:, 0], -a[:, 1]]
    # pairs already closer than MARGIN (touching contacts) are frozen: no closer along their axis
    rhs = np.maximum(gap - MARGIN, 0.0)
    if repair:                                 # overlapping pairs must separate by their penetration
        rhs = np.where(gap < 0, gap - MARGIN, rhs)
    rhs = list(rhs)
    L = side(c, e)
    row = len(k)
    R_, C_, V_ = [rr], [cc], [vv]
    idx = np.arange(n)
    for axis in (0, 1):
        lo0 = (c[:, axis] - e).min()
        o = 2 * n + axis                       # box-origin shift variable
        # -(d) + o <= c - e - lo0
        R_ += [row + idx, row + idx]; C_ += [2 * idx + axis, np.full(n, o)]; V_ += [-np.ones(n), np.ones(n)]
        rhs += list(c[:, axis] - e - lo0); row += n
        # d - o - dL <= lo0 + L - c - e
        R_ += [row + idx, row + idx, row + idx]
        C_ += [2 * idx + axis, np.full(n, o), np.full(n, 2 * n + 2)]
        V_ += [np.ones(n), -np.ones(n), -np.ones(n)]
        rhs += list(lo0 + L - c[:, axis] - e); row += n
    A = coo_matrix((np.concatenate(V_), (np.concatenate(R_), np.concatenate(C_))), shape=(row, 2 * n + 3)).tocsr()
    obj = np.zeros(2 * n + 3); obj[-1] = 1
    t = trust / SCALE
    res = linprog(obj, A_ub=A, b_ub=np.array(rhs) / SCALE, bounds=[(-t, t)] * (2 * n) + [(None, None)] * 2 + [(None, None if repair else 0)],
                  method="highs-ds", options={"time_limit": 120})
    if not res.success:
        return None
    return c + SCALE * res.x[:2 * n].reshape(n, 2), SCALE * float(res.x[-1])


def work(task):
    n, grp, rounds, trust = task
    c0 = np.array([[float(r["x"]), float(r["y"])] for r in grp])
    th = np.array([float(r["angle"]) for r in grp])
    e = 0.5 * (np.abs(np.cos(th)) + np.abs(np.sin(th)))
    base = side(c0, e)
    c, cur = c0, base
    for _ in range(rounds):
        out = lp_round(c, th, e, trust)
        if out is None:
            break
        q, _pred = out
        if pair_axes(q, th)[3].min() < -1e-12:           # LP left a real overlap: stop here
            break
        s = side(q, e)
        if s >= cur - 1e-14:
            break
        c, cur = q, s
    info = dict(n=n, base=base, lp=cur, delta=cur - base)
    if cur >= base - 1e-13:
        info["status"] = "no-gain"
        return n, info, None
    old = R.both(n, grp)
    mid = 0.5 * (c.min(0) + c.max(0))
    new = None
    # touching contacts that moved can pick up ~1e-15 float overlaps; if the exact
    # metric rejects the raw result, retry with a minimal uniform dilation.
    for dil in (0.0, 2e-14, 2e-13, 2e-12):
        cd = mid + (c - mid) * (1.0 + dil)
        cand = [dict(n=str(n), square_id=r["square_id"], x=repr(float(cd[t, 0])), y=repr(float(cd[t, 1])),
                     angle=r["angle"]) for t, r in enumerate(grp)]
        new = R.both(n, cand)
        if new is not None:
            info["dilation"] = dil
            break
    if new is None:
        info["status"] = "official-reject"
        return n, info, None
    if not (new[0] < old[0] and new[1] < old[1]):
        info["status"] = "not-strict"
        return n, info, None
    info.update(status="WIN", old_literal=str(old[0]), new_literal=str(new[0]),
                old_pandas=str(old[1]), new_pandas=str(new[1]))
    return n, info, cand


def main():
    path, lo, hi, jobs, out_rows, out_rep = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], sys.argv[6]
    rounds = int(sys.argv[7]) if len(sys.argv) > 7 else 12
    trust = float(sys.argv[8]) if len(sys.argv) > 8 else 1e-5
    groups = R.read_groups(path, set(range(lo, hi + 1)))
    wins, rep = {}, []
    with mp.get_context("fork").Pool(jobs) as pool:
        for n, info, cand in pool.imap_unordered(work, [(n, groups[n], rounds, trust) for n in sorted(groups, reverse=True)]):
            rep.append(info)
            if cand:
                wins[n] = cand
                print(f"WIN n={n} dL={info['delta']:.3e}", flush=True)
    rep.sort(key=lambda r: r["n"])
    R.write_rows(out_rows, wins)
    json.dump(rep, open(out_rep, "w"), indent=1)
    gain = sum((Decimal(r["old_literal"]) - Decimal(r["new_literal"])) / Decimal(r["n"]).sqrt() for r in rep if r["status"] == "WIN")
    from collections import Counter
    print(Counter(r["status"] for r in rep), "score gain", gain)


if __name__ == "__main__":
    main()
