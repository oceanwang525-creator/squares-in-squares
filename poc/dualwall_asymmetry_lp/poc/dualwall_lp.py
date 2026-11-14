#!/usr/bin/env python3
"""Fixed-orientation, wall-anchored block LP for a few incumbent groups.

Each square belongs to an orientation group and a spatial anchor block.  A
block has one rigid translation.  The model represents left/right and
bottom/top boundary extents independently while minimizing max(width, height).
The incumbent rows and official evaluator are used unchanged.
"""
from __future__ import annotations

import os
for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(var, "1")

import argparse
import csv
import json
import math
import time
from decimal import Decimal
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[3]
sys_path = ROOT / "poc/baseline/poc/src"
import sys  # noqa: E402
sys.path.insert(0, str(sys_path))
from squares.core import certify, kaggle_roundtrip  # noqa: E402

PAIR_RADIUS = 1.6
MARGIN = 1e-9


def orientation_key(angle: float) -> int:
    """Canonical square orientation; 0 and 90 degrees are the same group."""
    quadrant = round(math.degrees(angle) / 90.0)
    degrees = math.degrees(angle) - 90.0 * quadrant
    return int(round(degrees * 1_000_000))


def blocks_for(rows: list[dict[str, str]], mode: str) -> list[int]:
    centers = np.array([[float(r["x"]), float(r["y"])] for r in rows])
    angles = np.array([float(r["angle"]) for r in rows])
    ext = 0.5 * (np.abs(np.cos(angles)) + np.abs(np.sin(angles)))
    span_x = (centers[:, 0] + ext).max() - (centers[:, 0] - ext).min()
    span_y = (centers[:, 1] + ext).max() - (centers[:, 1] - ext).min()
    scale = max(span_x, span_y)
    origin = np.array([(centers[:, 0] - ext).min() + (centers[:, 0] + ext).max(),
                       (centers[:, 1] - ext).min() + (centers[:, 1] + ext).max()]) / 2
    groups = [orientation_key(a) for a in angles]
    if mode == "orientation":
        return groups
    if mode.startswith("grid-"):
        k = int(mode[5:])
        return [g * 1000 + int(np.floor((c[0] - origin[0]) / scale * k + k / 2)) * 10
                + int(np.floor((c[1] - origin[1]) / scale * k + k / 2))
                for g, c in zip(groups, centers)]
    if mode == "anchor":
        out = []
        for g, c in zip(groups, centers):
            d = [(c[0] - origin[0]) / scale, (origin[0] - c[0]) / scale,
                 (c[1] - origin[1]) / scale, (origin[1] - c[1]) / scale]
            out.append(g * 1000 + int(np.argmin(d)))
        return out
    raise ValueError(mode)


def solve(angles: np.ndarray, centers: np.ndarray, block_ids: list[int],
          trust: float | None = None) -> tuple[float, np.ndarray] | None:
    n = len(angles)
    unique = sorted(set(block_ids))
    block_index = {b: i for i, b in enumerate(unique)}
    bidx = np.array([block_index[b] for b in block_ids])
    nb = len(unique)
    u = np.column_stack([np.cos(angles), np.sin(angles)])
    v = np.column_stack([-np.sin(angles), np.cos(angles)])
    pairs = list(cKDTree(centers).query_pairs(PAIR_RADIUS))
    rows, cols, vals, rhs = [], [], [], []

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
        radii = 0.5 * (np.abs(axes @ u[i]) + np.abs(axes @ v[i]) +
                       np.abs(axes @ u[j]) + np.abs(axes @ v[j]))
        k = int(np.argmax(np.abs(proj) - radii))
        sign = 1 if proj[k] >= 0 else -1
        a = axes[k] * sign
        columns, values = [], []
        # a.(s_i-s_j) <= slack; s_i and s_j are rigid block shifts.
        for square, coefficient in ((i, a[0]), (j, -a[0])):
            columns.append(2 * bidx[square]); values.append(coefficient)
        for square, coefficient in ((i, a[1]), (j, -a[1])):
            columns.append(2 * bidx[square] + 1); values.append(coefficient)
        add(columns, values, float(abs(proj[k])) - radii[k] - MARGIN)

    # Four boundary extents.  Objective is max(width,height) via a common L.
    for b in range(nb):
        members = np.flatnonzero(bidx == b)
        x_ext = 0.5 * (np.abs(np.cos(angles[members])) +
                       np.abs(np.sin(angles[members])))
        y_ext = x_ext
        # Boundary inequalities are represented against each member center.
        for idx, ee in zip(members, x_ext):
            add([2 * b, 2 * nb], [-1, 1], float(centers[idx, 0]) - ee - MARGIN)
            add([2 * b, 2 * nb + 1], [1, -1], -float(centers[idx, 0]) - ee - MARGIN)
        for idx, ee in zip(members, y_ext):
            add([2 * b + 1, 2 * nb + 2], [-1, 1], float(centers[idx, 1]) - ee - MARGIN)
            add([2 * b + 1, 2 * nb + 3], [1, -1], -float(centers[idx, 1]) - ee - MARGIN)

    # L >= width and L >= height.
    add([2 * nb, 2 * nb + 1, 2 * nb + 4], [-1, 1, -1], 0)
    add([2 * nb + 2, 2 * nb + 3, 2 * nb + 4], [-1, 1, -1], 0)
    A = coo_matrix((vals, (rows, cols)), shape=(len(rhs), 2 * nb + 5)).tocsr()
    c = np.zeros(2 * nb + 5)
    c[-1] = 1
    bounds = []
    for b in range(nb):
        if trust is None:
            bounds.extend([(None, None), (None, None)])
        else:
            bounds.extend([(-trust, trust), (-trust, trust)])
    bounds.extend([(None, None)] * 4)
    bounds.append((0, None))
    result = linprog(c, A_ub=A, b_ub=np.array(rhs), bounds=bounds,
                     method="highs", options={"dual_feasibility_tolerance": 1e-9,
                                              "primal_feasibility_tolerance": 1e-9,
                                              "time_limit": 30})
    if not result.success:
        return None
    shifts = np.asarray(result.x[:2 * nb]).reshape(nb, 2)
    proposed = centers + shifts[bidx]
    # Independent boundary extents can only affect the model, not the emitted
    # packing: certification measures the actual coordinate extents.
    width = (proposed[:, 0] + 0.5 * (np.abs(np.cos(angles)) +
            np.abs(np.sin(angles)))).max() - (proposed[:, 0] - 0.5 *
            (np.abs(np.cos(angles)) + np.abs(np.sin(angles)))).min()
    height = (proposed[:, 1] + 0.5 * (np.abs(np.cos(angles)) +
             np.abs(np.sin(angles)))).max() - (proposed[:, 1] - 0.5 *
             (np.abs(np.cos(angles)) + np.abs(np.sin(angles)))).min()
    return max(float(width), float(height)), proposed


def refine(rows: list[dict[str, str]], mode: str, rounds: int,
           trust: float | None):
    angles = np.array([float(r["angle"]) for r in rows])
    centers = np.array([[float(r["x"]), float(r["y"])] for r in rows])
    block_ids = blocks_for(rows, mode)
    best = None
    history = []
    for round_no in range(rounds):
        out = solve(angles, centers, block_ids, trust)
        if out is None:
            history.append({"round": round_no + 1, "status": "infeasible"})
            break
        side, proposed = out
        history.append({"round": round_no + 1, "side": side,
                        "shift": float(np.abs(proposed - centers).max())})
        if best is None or side < best[0] - 1e-12:
            best = (side, proposed)
        if np.abs(proposed - centers).max() < 1e-10:
            break
        centers = proposed
    return block_ids, best, history


def to_rows(rows: list[dict[str, str]], proposed: np.ndarray) -> list[dict[str, str]]:
    return [dict(square_id=r["square_id"], x=repr(float(proposed[i, 0])),
                 y=repr(float(proposed[i, 1])), angle=repr(float(r["angle"])))
            for i, r in enumerate(rows)]


def evaluate(n: int, old: Decimal, rows: list[dict[str, str]],
             proposed: np.ndarray, mode: str, history: list[dict]):
    entry = dict(n=n, mode=mode, blocks=len(set(blocks_for(rows, mode))),
                 old_L=str(old), history=history)
    if proposed is None:
        entry["status"] = "lp-infeasible"
        return entry, None
    cand = to_rows(rows, proposed)
    try:
        literal = Decimal(certify(n, cand))
    except Exception as error:
        entry.update(status="certify-fail", error=str(error)[:180])
        return entry, None
    entry["literal_L"] = str(literal)
    if literal >= old:
        entry["status"] = "literal-tie-or-worse"
        return entry, None
    try:
        transported = kaggle_roundtrip({n: cand})
        pandas_l = Decimal(certify(n, transported[n]))
    except Exception as error:
        entry.update(status="pandas-fail", error=str(error)[:180])
        return entry, None
    entry["pandas_L"] = str(pandas_l)
    if pandas_l >= old:
        entry["status"] = "pandas-tie-or-worse"
        return entry, None
    entry["status"] = "WIN"
    return entry, dict(n=n, old_L=str(old), new_L=str(max(literal, pandas_l)),
                       literal_L=str(literal), pandas_L=str(pandas_l), rows=cand)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--incumbent", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--wins", required=True)
    parser.add_argument("--n-list", required=True)
    parser.add_argument("--modes", default="orientation,anchor,grid-2,grid-3")
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--trust", default="none",
                        help="block translation bound, or none")
    args = parser.parse_args()
    groups: dict[int, list[dict[str, str]]] = {}
    with open(args.incumbent, newline="") as handle:
        for row in csv.DictReader(handle):
            groups.setdefault(int(row["n"]), []).append(row)
    targets = [int(x) for x in args.n_list.split(",") if x]
    modes = args.modes.split(",")
    trust = None if args.trust == "none" else float(args.trust)
    results, wins = [], {}
    started = time.time()
    for n in targets:
        rows = groups[n]
        old = Decimal(certify(n, rows))
        for mode in modes:
            block_ids, out, history = refine(rows, mode, args.rounds, trust)
            proposed = out[1] if out else None
            entry, win = evaluate(n, old, rows, proposed, mode, history)
            results.append(entry)
            print(json.dumps({k: entry[k] for k in ("n", "mode", "status",
                 "literal_L", "pandas_L") if k in entry}), flush=True)
            if win is not None:
                known = wins.get(n)
                if known is None or Decimal(win["new_L"]) < Decimal(known["new_L"]):
                    wins[n] = win
                    print(f"WIN n={n} mode={mode}", flush=True)
    report = dict(targets=targets, modes=modes, rounds=args.rounds,
                  trust=args.trust, seconds=time.time() - started,
                  results=results, wins=wins)
    Path(args.output).write_text(json.dumps(report, indent=1) + "\n")
    Path(args.wins).write_text(json.dumps(wins, indent=1) + "\n")
    print(json.dumps({"targets": len(targets), "tasks": len(results),
                      "wins": len(wins), "seconds": report["seconds"]}, indent=1))


if __name__ == "__main__":
    main()
