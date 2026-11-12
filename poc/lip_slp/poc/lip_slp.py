#!/usr/bin/env python3
"""Lipschitz-safe free-angle SLP over the whole portfolio.

The earlier ``free_angle_slp`` used a tangent linearisation of the support
function at its kinks, which let angle pairs move in first-order-overlapping
directions.  Here every pair row bounds the support growth by
``LIP*(t_i+t_j)`` with ``t_i >= |dtheta_i|``, and the wall rows use the same
bound, so any accepted LP step is exactly non-overlapping by construction.

Each worker runs one ``n``: iterative LP on (dx, dy, dtheta, t, dL), objective
``min dL``, followed by the unchanged official certification.  Results are
returned in sorted ``n`` order.
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import sys
import time
from decimal import Decimal
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = "1"

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "poc" / "baseline" / "poc" / "src"))
from squares import core  # noqa: E402

PAIR = 1.65
MARGIN = 1e-9
LIP = 0.7071067811865476


def extents(theta):
    return 0.5 * (np.abs(np.cos(theta)) + np.abs(np.sin(theta)))


def exact_side(centers, theta):
    e = extents(theta)
    return float(max((centers[:, 0] + e).max() - (centers[:, 0] - e).min(),
                     (centers[:, 1] + e).max() - (centers[:, 1] - e).min()))


def build(theta, centers, L0, trust_angle, trust_move):
    n = len(theta)
    u = np.column_stack([np.cos(theta), np.sin(theta)])
    v = np.column_stack([-np.sin(theta), np.cos(theta)])
    pairs = list(cKDTree(centers).query_pairs(PAIR))
    rows, cols, vals, rhs = [], [], [], []

    def add(columns, values, b):
        row = len(rhs)
        rows.extend([row] * len(columns))
        cols.extend(columns)
        vals.extend(values)
        rhs.append(b)

    def ix(i):
        return 2 * i

    def iy(i):
        return 2 * i + 1

    def it(i):
        return 2 * n + i

    def itl(i):
        return 3 * n + i

    iL = 4 * n
    for i, j in pairs:
        axes = np.array([u[i], v[i], u[j], v[j]])
        delta = centers[j] - centers[i]
        proj = axes @ delta
        rad = 0.5 * (np.abs(axes @ u[i]) + np.abs(axes @ v[i])
                     + np.abs(axes @ u[j]) + np.abs(axes @ v[j]))
        alpha = np.abs(proj) - rad
        k = int(np.argmax(alpha))
        direction = 1.0 if proj[k] >= 0 else -1.0
        a = axes[k] * direction
        slack = direction * proj[k] - rad[k] - MARGIN
        add([ix(i), iy(i), ix(j), iy(j), itl(i), itl(j)],
            [a[0], a[1], -a[0], -a[1], LIP, LIP], slack)

    e = extents(theta)
    for i in range(n):
        add([itl(i), it(i)], [1, -1], 0.0)
        add([itl(i), it(i)], [1, 1], 0.0)
        add([ix(i), itl(i), iL], [1, LIP, -1],
            L0 - centers[i, 0] - e[i] - MARGIN)
        add([ix(i), itl(i)], [-1, LIP],
            centers[i, 0] - e[i] - MARGIN)
        add([iy(i), itl(i), iL], [1, LIP, -1],
            L0 - centers[i, 1] - e[i] - MARGIN)
        add([iy(i), itl(i)], [-1, LIP],
            centers[i, 1] - e[i] - MARGIN)

    A = coo_matrix((vals, (rows, cols)), shape=(len(rhs), 4 * n + 1)).tocsr()
    bounds = []
    for _ in range(n):
        bounds.append((-trust_move, trust_move))
        bounds.append((-trust_move, trust_move))
    for _ in range(n):
        bounds.append((-trust_angle, trust_angle))
    for _ in range(n):
        bounds.append((0.0, trust_angle))
    bounds.append((-0.05, 0.05))
    return A, np.array(rhs), bounds


def run_group(task):
    n, rows, ours_literal, rounds = task
    ours = Decimal(ours_literal)
    centers = np.array([[float(r["x"]), float(r["y"])] for r in rows])
    theta = np.array([float(r["angle"]) for r in rows])
    L = exact_side(centers, theta)
    best = Decimal(core.certify(n, rows))
    ta, tm = 0.002, 0.008
    for _ in range(rounds):
        A, b, bounds = build(theta, centers, L, ta, tm)
        c = np.zeros(4 * n + 1)
        c[-1] = 1.0
        res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="highs",
                      options={"dual_feasibility_tolerance": 1e-9,
                               "primal_feasibility_tolerance": 1e-9,
                               "time_limit": 120.0})
        if not res.success:
            ta *= 0.5
            tm *= 0.5
            if ta < 1e-7:
                break
            continue
        x = res.x
        nc = centers + np.column_stack([x[0:2 * n:2], x[1:2 * n:2]])
        nt = theta + x[2 * n:3 * n]
        cand = [dict(square_id=str(i), x=repr(float(nc[i, 0])),
                     y=repr(float(nc[i, 1])), angle=repr(float(nt[i])))
                for i in range(n)]
        try:
            literal = Decimal(core.certify(n, cand))
        except Exception:  # noqa: BLE001
            ta *= 0.5
            tm *= 0.5
            if ta < 1e-7:
                break
            continue
        centers, theta = nc, nt
        L = exact_side(centers, theta)
        if literal < best:
            best = literal
        moved = np.abs(x).max()
        if moved < 1e-10:
            break
        ta = min(0.002, max(ta * 1.3, 1e-6))
        tm = min(0.008, max(tm * 1.3, 1e-6))
    win = None
    if best < ours:
        cand = [dict(square_id=str(i), x=repr(float(centers[i, 0])),
                     y=repr(float(centers[i, 1])), angle=repr(float(theta[i])))
                for i in range(n)]
        try:
            literal = Decimal(core.certify(n, cand))
            transported = core.kaggle_roundtrip({n: cand})
            pandas_L = Decimal(core.certify(n, transported[n]))
        except Exception as error:  # noqa: BLE001
            return n, dict(n=n, status="final-fail", error=str(error)[:120]), None
        if literal < ours and pandas_L < ours:
            win = dict(old_L=str(ours), new_L=str(max(literal, pandas_L)),
                       literal_L=str(literal), pandas_L=str(pandas_L),
                       rows=cand)
    return n, dict(n=n, base_side=L, status="win" if win else "no-gain"), win


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--incumbent", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--wins", required=True)
    parser.add_argument("--n-list", default="")
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--jobs", type=int, default=32)
    args = parser.parse_args()

    report = json.load(open(args.report, encoding="utf-8"))
    ours = {int(r["n"]): Decimal(r["L"]) for r in report["results"]
            if r["L"] is not None}
    groups: dict[int, list[dict]] = {}
    with open(args.incumbent, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            groups.setdefault(int(row["n"]), []).append(row)
    targets = [int(x) for x in args.n_list.split(",") if x.strip()]
    if not targets:
        targets = list(range(1, 1001))
    tasks = [(n, groups[n], str(ours[n]), args.rounds)
             for n in sorted(targets, reverse=True)]

    wins: dict[int, dict] = {}
    log: list[dict] = []
    done = 0
    t0 = time.time()
    jobs = max(1, min(args.jobs, os.cpu_count() or 1))
    with mp.get_context("fork").Pool(jobs) as pool:
        for n, entry, win in pool.imap_unordered(run_group, tasks, chunksize=1):
            log.append(entry)
            done += 1
            if win is not None:
                wins[n] = win
                print(f"WIN n={n} {win['old_L']} -> {win['new_L']}", flush=True)
            if done % 200 == 0 or done == len(tasks):
                rate = done / max(time.time() - t0, 1e-9) * 60.0
                print(f"{done}/{len(tasks)} groups, {rate:.0f}/min, "
                      f"wins={len(wins)}", flush=True)

    log.sort(key=lambda e: e["n"])
    ordered = {n: wins[n] for n in sorted(wins)}
    json.dump({str(k): v for k, v in ordered.items()},
              open(args.wins, "w", encoding="utf-8"), indent=1)
    json.dump(log, open(args.output, "w", encoding="utf-8"), indent=1)
    reduction = sum(
        (Decimal(v["old_L"]) - Decimal(v["new_L"])) / Decimal(n).sqrt()
        for n, v in ordered.items())
    print(json.dumps(dict(scanned=len(targets), wins=len(ordered),
                          reduction=str(reduction), jobs=jobs,
                          seconds=round(time.time() - t0, 1)), indent=1))


if __name__ == "__main__":
    mp.freeze_support()
    main()
