"""Joint centre + angle SLP for a single packing.

Variables are all centre coordinates, all square rotations, and the container
side. Every nearby pair contributes one active separating axis, and all
extents/radii are linearised around the current angles, so one LP step can move
and rotate every square at once. The trust region on the angles is shrunk until
the LP is feasible, and the pair set is rebuilt after each accepted step.
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

os.environ.setdefault("OMP_NUM_THREADS", "1")

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = "1"

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "poc/baseline/poc/src"))
from squares import core  # noqa: E402

PAIR_RADIUS = 1.6
MARGIN = 1e-9


def extent(theta):
    return 0.5 * (np.abs(np.cos(theta)) + np.abs(np.sin(theta)))


def extent_grad(theta):
    return 0.5 * (-np.sign(np.cos(theta)) * np.sin(theta)
                  + np.sign(np.sin(theta)) * np.cos(theta))


def axis_radius(theta, phi):
    """Radius of a unit square rotated by theta projected on axis phi."""
    c = np.abs(np.cos(theta - phi))
    s = np.abs(np.sin(theta - phi))
    return 0.5 * (c + s)


def axis_radius_grad(theta, phi):
    d = theta - phi
    return 0.5 * (-np.sign(np.cos(d)) * np.sin(d)
                  + np.sign(np.sin(d)) * np.cos(d))


def build_lp(theta, centers, trust_angle, trust_move):
    n = len(theta)
    offsets = [np.array([np.cos(t), np.sin(t)]) for t in theta]
    normals = [np.array([-np.sin(t), np.cos(t)]) for t in theta]
    pairs = list(cKDTree(centers).query_pairs(PAIR_RADIUS))
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    rhs: list[float] = []

    def add(columns, values, b):
        row = len(rhs)
        rows.extend([row] * len(columns))
        cols.extend(columns)
        vals.extend(values)
        rhs.append(b)

    ix = lambda i: 2 * i
    iy = lambda i: 2 * i + 1
    it = lambda i: 2 * n + i
    iL = 3 * n

    for i, j in pairs:
        axes = np.array([offsets[i], normals[i], offsets[j], normals[j]])
        phi = np.arctan2(axes[:, 1], axes[:, 0])
        delta = centers[j] - centers[i]
        proj = axes @ delta
        rad = np.array([axis_radius(theta[i], p) + axis_radius(theta[j], p)
                        for p in phi])
        alpha = np.abs(proj) - rad
        k = int(np.argmax(alpha))
        direction = 1.0 if proj[k] >= 0 else -1.0
        a = axes[k] * direction
        di = direction * axis_radius_grad(theta[i], phi[k])
        dj = direction * axis_radius_grad(theta[j], phi[k])
        slack = direction * proj[k] - rad[k] - MARGIN
        add([ix(j), iy(j), ix(i), iy(i), it(i), it(j)],
            [a[0], a[1], -a[0], -a[1], -di, -dj], -slack)

    ext = extent(theta)
    grad = extent_grad(theta)
    for i in range(n):
        add([ix(i)], [-1], -ext[i] - MARGIN)
        add([ix(i), it(i)], [-1, grad[i]], -ext[i] - MARGIN)
        add([ix(i)], [1], -ext[i] - MARGIN)
        add([ix(i), it(i), iL], [1, grad[i], -1], -ext[i] - MARGIN)
        add([iy(i)], [-1], -ext[i] - MARGIN)
        add([iy(i), it(i)], [-1, grad[i]], -ext[i] - MARGIN)
        add([iy(i)], [1], -ext[i] - MARGIN)
        add([iy(i), it(i), iL], [1, grad[i], -1], -ext[i] - MARGIN)

    A = coo_matrix((vals, (rows, cols)), shape=(len(rhs), 3 * n + 1)).tocsr()
    c = np.zeros(3 * n + 1)
    c[-1] = 1.0
    bounds = []
    for i in range(n):
        bounds.append((float(centers[i, 0]) - trust_move,
                       float(centers[i, 0]) + trust_move))
        bounds.append((float(centers[i, 1]) - trust_move,
                       float(centers[i, 1]) + trust_move))
    for i in range(n):
        bounds.append((float(theta[i]) - trust_angle,
                       float(theta[i]) + trust_angle))
    bounds.append((0.0, None))
    return A, np.array(rhs), c, bounds, (ix, iy, it, iL)


def step(theta, centers, L, trust_angle, trust_move, time_limit):
    A, b, c, bounds, idx = build_lp(theta, centers, trust_angle, trust_move)
    n = len(theta)
    res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="highs",
                  options={"dual_feasibility_tolerance": 1e-9,
                           "primal_feasibility_tolerance": 1e-9,
                           "time_limit": time_limit})
    if not res.success:
        return None
    x = res.x
    new_centers = np.column_stack([x[0:2 * n:2], x[1:2 * n:2]])
    new_theta = x[2 * n:3 * n]
    return float(x[-1]), new_centers, new_theta


def exact_side(centers, theta):
    ext = extent(theta)
    return float(max((centers[:, 0] + ext).max() - (centers[:, 0] - ext).min(),
                     (centers[:, 1] + ext).max() - (centers[:, 1] - ext).min()))


def refine(theta, centers, L, rounds, time_limit):
    trust_angle = 0.01
    trust_move = 0.05
    best = (L, centers.copy(), theta.copy())
    for _ in range(rounds):
        out = None
        for _ in range(6):
            out = step(theta, centers, L, trust_angle, trust_move, time_limit)
            if out is not None:
                break
            trust_angle *= 0.5
            trust_move *= 0.5
        if out is None:
            break
        new_L, new_centers, new_theta = out
        moved = np.abs(new_centers - centers).max()
        rotated = np.abs(new_theta - theta).max()
        centers, theta = new_centers, new_theta
        side = exact_side(centers, theta)
        if side < best[0] - 1e-13:
            best = (side, centers.copy(), theta.copy())
        if moved < 1e-10 and rotated < 1e-10:
            break
        trust_angle = min(0.01, max(rotated * 2, 1e-5))
        trust_move = min(0.05, max(moved * 2, 1e-5))
    return best


def run_group(task):
    """Worker: SLP + exact literal/pandas certification for one ``n``."""
    n, rows, ours_literal, rounds, time_limit = task
    ours = Decimal(ours_literal)
    theta = np.array([float(r["angle"]) for r in rows])
    centers = np.array([[float(r["x"]), float(r["y"])] for r in rows])
    base = exact_side(centers, theta)
    entry = dict(n=n, base=base)
    side, new_centers, new_theta = refine(theta, centers, base, rounds, time_limit)
    entry["slp_side"] = side
    if side >= float(ours) - 1e-13:
        entry["status"] = "no-gain"
        return n, entry, None
    cand = [dict(square_id=str(i), x=repr(float(new_centers[i, 0])),
                 y=repr(float(new_centers[i, 1])),
                 angle=repr(float(new_theta[i]))) for i in range(n)]
    try:
        literal = Decimal(core.certify(n, cand))
        transported = core.kaggle_roundtrip({n: cand})
        pandas_l = Decimal(core.certify(n, transported[n]))
    except Exception as error:  # noqa: BLE001
        entry["status"] = "certify-fail"
        entry["error"] = str(error)[:140]
        return n, entry, None
    entry.update(literal=str(literal), pandas=str(pandas_l))
    if literal < ours and pandas_l < ours:
        entry["status"] = "WIN"
        win = dict(old_L=str(ours), new_L=str(max(literal, pandas_l)),
                   literal_L=str(literal), pandas_L=str(pandas_l), rows=cand)
        return n, entry, win
    entry["status"] = "not-strict"
    return n, entry, None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--incumbent", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--n-list", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--wins", required=True)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--time-limit", type=float, default=60.0)
    parser.add_argument("--jobs", type=int, default=32)
    args = parser.parse_args()

    ours = {int(r["n"]): Decimal(r["L"])
            for r in json.load(open(args.report, encoding="utf-8"))["results"]
            if r["L"] is not None}
    groups: dict[int, list[dict]] = {}
    with open(args.incumbent, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            groups.setdefault(int(row["n"]), []).append(row)

    targets = [int(x) for x in args.n_list.split(",") if x.strip()]
    if not targets:
        targets = list(range(1, 1001))
    tasks = [(n, groups[n], str(ours[n]), args.rounds, args.time_limit)
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
                print(f"n={n} WIN {float(Decimal(win['new_L'])):.9f} "
                      f"from {float(Decimal(win['old_L'])):.9f}", flush=True)
            if done % 100 == 0 or done == len(tasks):
                rate = done / max(time.time() - t0, 1e-9) * 60.0
                print(f"{done}/{len(tasks)} groups, {rate:.0f} groups/min, "
                      f"wins={len(wins)}", flush=True)

    log.sort(key=lambda e: e["n"])
    ordered_wins = {n: wins[n] for n in sorted(wins)}
    json.dump({str(k): v for k, v in ordered_wins.items()},
              open(args.wins, "w", encoding="utf-8"), indent=1)
    json.dump(log, open(args.output, "w", encoding="utf-8"), indent=1)
    reduction = sum((Decimal(v["old_L"]) - Decimal(v["new_L"]))
                    / Decimal(n).sqrt() for n, v in ordered_wins.items())
    print(json.dumps(dict(scanned=len(targets), wins=len(ordered_wins),
                          jobs=jobs, seconds=round(time.time() - t0, 1),
                          reduction=str(reduction)), indent=1))


if __name__ == "__main__":
    mp.freeze_support()
    main()
