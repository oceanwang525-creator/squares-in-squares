"""Global fixed-angle box-side LP over every incumbent group.

For a packing with fixed square angles, the box side is a linear program in the
centers and L once one separating axis per nearby pair is fixed. The existing
repair tools only re-solved candidates inside a small trust region. Here we
solve with no trust region, so the LP can move squares arbitrarily far along
directions the original contacts leave free.

Any LP solution with L' < L is emitted and then re-checked with the unchanged
official certifier; only strictly certified winners are kept. The pair set and
active axis are rebuilt from the LP solution and re-solved a few times (SLP).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from decimal import Decimal
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

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


def pandas_side(n, rows):
    transported = core.kaggle_roundtrip({n: rows})
    return Decimal(core.certify(n, transported[n]))


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
    wins: dict[int, dict] = {}
    results = []
    t0 = time.time()
    for k, n in enumerate(targets):
        rows = groups[n]
        angles = np.array([float(r["angle"]) for r in rows])
        centers = np.array([[float(r["x"]), float(r["y"])] for r in rows])
        entry = dict(n=n, base_side=exact_side(n, centers, angles))
        out = refine(n, centers, angles, args.rounds, args.time_limit)
        if out is None:
            entry["status"] = "lp-infeasible"
            results.append(entry)
            continue
        side, proposed = out
        entry["lp_side"] = side
        if side >= float(ours[n]) - 1e-12:
            entry["status"] = "no-gain"
            results.append(entry)
            continue
        cand = to_rows(n, proposed, angles)
        try:
            literal = Decimal(core.certify(n, cand))
        except Exception as error:  # noqa: BLE001
            entry["status"] = "certify-fail"
            entry["error"] = str(error)[:160]
            results.append(entry)
            continue
        entry["literal"] = str(literal)
        if literal >= ours[n]:
            entry["status"] = "not-strict-literal"
            results.append(entry)
            continue
        try:
            transported = pandas_side(n, cand)
        except Exception as error:  # noqa: BLE001
            entry["status"] = "pandas-fail"
            entry["error"] = str(error)[:160]
            results.append(entry)
            continue
        entry["pandas"] = str(transported)
        if transported >= ours[n]:
            entry["status"] = "not-strict-pandas"
            results.append(entry)
            continue
        entry["status"] = "WIN"
        wins[n] = dict(old_L=str(ours[n]), new_L=str(max(literal, transported)),
                       literal_L=str(literal), pandas_L=str(transported),
                       lp_side=side, rows=cand)
        results.append(entry)
        print(f"WIN n={n} old={float(ours[n]):.9f} new={float(max(literal, transported)):.9f}",
              flush=True)
        if (k + 1) % 10 == 0:
            print(f"{k + 1}/{len(targets)} t={time.time() - t0:.0f}s wins={len(wins)}",
                  flush=True)

    with open(args.wins, "w", encoding="utf-8") as handle:
        json.dump({str(k): v for k, v in wins.items()}, handle, indent=1)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=1)
    reduction = sum(
        (Decimal(v["old_L"]) - Decimal(v["new_L"])) / Decimal(n).sqrt()
        for n, v in wins.items())
    print(json.dumps(dict(scanned=len(targets), wins=len(wins),
                          reduction=str(reduction),
                          seconds=round(time.time() - t0, 1)), indent=1))


if __name__ == "__main__":
    main()
