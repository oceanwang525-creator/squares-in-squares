"""Parallel fixed-angle box-side LP over every incumbent group.

For a packing with fixed square angles, the box side is a linear program in the
centers and L once one separating axis per nearby pair is fixed. The LP is
solved with no trust region, so it can move squares arbitrarily far along
directions the original contacts leave free.

Parallelism
- one process per ``n`` (``multiprocessing`` pool, default 32 workers);
- BLAS/OpenMP thread counts pinned to 1 before numpy/scipy import, so workers
  do not oversubscribe the 32 hardware threads;
- ``imap_unordered`` with the largest ``n`` first, so slow groups start early;
- results are merged in sorted ``n`` order, which keeps the output
  deterministic.

Every LP solution with L' < L is emitted and re-checked with the unchanged
official certifier; only strictly certified winners are kept.
"""
from __future__ import annotations

import os

# Pin BLAS/OpenMP before numpy or scipy is imported. This runs in the parent
# and is inherited by every forked worker.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = "1"

import argparse
import csv
import json
import multiprocessing as mp
import sys
import time
from decimal import Decimal
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "poc/baseline/poc/src"))
from squares import core  # noqa: E402

PAIR_RADIUS = 1.6
MARGIN = 1e-9


def build_lp(angles, centers):
    n = len(angles)
    u = np.column_stack([np.cos(angles), np.sin(angles)])
    v = np.column_stack([-np.sin(angles), np.cos(angles)])
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

    for i, j in pairs:
        axes = np.array([u[i], v[i], u[j], v[j]])
        delta = centers[j] - centers[i]
        proj = axes @ delta
        radii = 0.5 * (
            np.abs(axes @ u[i])
            + np.abs(axes @ v[i])
            + np.abs(axes @ u[j])
            + np.abs(axes @ v[j])
        )
        k = int(np.argmax(np.abs(proj) - radii))
        a = axes[k] * (1 if proj[k] >= 0 else -1)
        add([2 * i, 2 * i + 1, 2 * j, 2 * j + 1],
            [a[0], a[1], -a[0], -a[1]], -radii[k] - MARGIN)

    ext = 0.5 * (np.abs(np.cos(angles)) + np.abs(np.sin(angles)))
    for i in range(n):
        for k in range(2):
            add([2 * i + k], [-1], -ext[i] - MARGIN)
            add([2 * i + k, 2 * n], [1, -1], -ext[i] - MARGIN)

    A = coo_matrix((vals, (rows, cols)), shape=(len(rhs), 2 * n + 1)).tocsr()
    c = np.zeros(2 * n + 1)
    c[-1] = 1.0
    return A, np.array(rhs), c


def solve_lp(angles, centers, time_limit=60.0):
    A, b, c = build_lp(angles, centers)
    n = len(angles)
    res = linprog(c, A_ub=A, b_ub=b, bounds=[(None, None)] * (2 * n) + [(0, None)],
                  method="highs",
                  options={"dual_feasibility_tolerance": 1e-9,
                           "primal_feasibility_tolerance": 1e-9,
                           "time_limit": time_limit})
    if not res.success:
        return None
    return float(res.fun), np.asarray(res.x[:-1]).reshape(n, 2)


def exact_side(n, centers, angles):
    ext = 0.5 * (np.abs(np.cos(angles)) + np.abs(np.sin(angles)))
    return float(max((centers[:, 0] + ext).max() - (centers[:, 0] - ext).min(),
                     (centers[:, 1] + ext).max() - (centers[:, 1] - ext).min()))


def refine(n, centers, angles, rounds, time_limit):
    cur = centers
    best = None
    for _ in range(rounds):
        out = solve_lp(angles, cur, time_limit=time_limit)
        if out is None:
            break
        side, proposed = out
        if best is None or side < best[0] - 1e-12:
            best = (side, proposed)
        shift = np.abs(proposed - cur).max()
        cur = proposed
        if shift < 1e-10:
            break
    return best


def to_rows(n, centers, angles):
    return [dict(square_id=str(i), x=repr(float(centers[i, 0])),
                 y=repr(float(centers[i, 1])), angle=repr(float(angles[i])))
            for i in range(n)]


def run_group(task):
    """Worker: LP + exact literal/pandas certification for one ``n``."""
    n, rows, ours_literal, rounds, time_limit = task
    ours = Decimal(ours_literal)
    angles = np.array([float(r["angle"]) for r in rows])
    centers = np.array([[float(r["x"]), float(r["y"])] for r in rows])
    entry = dict(n=n, base_side=exact_side(n, centers, angles))
    out = refine(n, centers, angles, rounds, time_limit)
    if out is None:
        entry["status"] = "lp-infeasible"
        return n, entry, None
    side, proposed = out
    entry["lp_side"] = side
    if side >= float(ours) - 1e-12:
        entry["status"] = "no-gain"
        return n, entry, None
    cand = to_rows(n, proposed, angles)
    try:
        literal = Decimal(core.certify(n, cand))
    except Exception as error:  # noqa: BLE001
        entry["status"] = "certify-fail"
        entry["error"] = str(error)[:160]
        return n, entry, None
    entry["literal"] = str(literal)
    if literal >= ours:
        entry["status"] = "not-strict-literal"
        return n, entry, None
    try:
        transported = core.kaggle_roundtrip({n: cand})
        pandas_l = Decimal(core.certify(n, transported[n]))
    except Exception as error:  # noqa: BLE001
        entry["status"] = "pandas-fail"
        entry["error"] = str(error)[:160]
        return n, entry, None
    entry["pandas"] = str(pandas_l)
    if pandas_l >= ours:
        entry["status"] = "not-strict-pandas"
        return n, entry, None
    entry["status"] = "WIN"
    win = dict(old_L=str(ours), new_L=str(max(literal, pandas_l)),
               literal_L=str(literal), pandas_L=str(pandas_l),
               lp_side=side, rows=cand)
    return n, entry, win


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--incumbent", required=True)
    parser.add_argument("--report", required=True,
                        help="official literal report for the same incumbent")
    parser.add_argument("--output", required=True)
    parser.add_argument("--wins", required=True)
    parser.add_argument("--n-list", default="")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--time-limit", type=float, default=60.0)
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
    # Expensive groups first so they do not tail the schedule.
    tasks = [(n, groups[n], str(ours[n]), args.rounds, args.time_limit)
             for n in sorted(targets, reverse=True)]

    wins: dict[int, dict] = {}
    results: list[dict] = []
    t0 = time.time()
    done = 0
    jobs = max(1, min(args.jobs, os.cpu_count() or 1))
    with mp.get_context("fork").Pool(jobs) as pool:
        for n, entry, win in pool.imap_unordered(run_group, tasks, chunksize=1):
            results.append(entry)
            done += 1
            if win is not None:
                wins[n] = win
                print(f"WIN n={n} old={float(Decimal(win['old_L'])):.9f} "
                      f"new={float(Decimal(win['new_L'])):.9f}", flush=True)
            if done % 100 == 0 or done == len(tasks):
                rate = done / max(time.time() - t0, 1e-9) * 60.0
                print(f"{done}/{len(tasks)} groups, {rate:.0f} groups/min, "
                      f"wins={len(wins)}", flush=True)

    results.sort(key=lambda e: e["n"])
    ordered_wins = {n: wins[n] for n in sorted(wins)}
    with open(args.wins, "w", encoding="utf-8") as handle:
        json.dump({str(k): v for k, v in ordered_wins.items()}, handle, indent=1)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=1)
    reduction = sum(
        (Decimal(v["old_L"]) - Decimal(v["new_L"])) / Decimal(n).sqrt()
        for n, v in ordered_wins.items())
    print(json.dumps(dict(scanned=len(targets), wins=len(ordered_wins),
                          reduction=str(reduction), jobs=jobs,
                          seconds=round(time.time() - t0, 1),
                          groups_per_min=round(len(targets) / max(time.time() - t0, 1e-9) * 60.0, 1)),
                     indent=1))


if __name__ == "__main__":
    mp.freeze_support()
    main()
