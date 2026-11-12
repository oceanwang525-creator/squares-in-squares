#!/usr/bin/env python3
"""Repack the two independent arm rectangles of a corner-block packing.

The incumbent is split into a bottom-left upright block (a x a), a top-right
upright block (g x g), and two rectangular arms of size w x (L-a) where
w = L-g.  Shrinking the box by d only needs the top-right block to translate
by (-d,-d) and the two arms to repack in the narrower rectangles; the
bottom-left block is untouched.

The arms are packed with a fixed-angle sequential LP (one separating axis per
nearby pair, rebuilt every round) under rectangular wall constraints.  The
assembled packing is then certified with the unchanged official metric, both
literal and through actual pandas transport.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
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

PAIR_RADIUS = 1.65
MARGIN = 1e-9
LIP = 0.7071067811865476


def extents(angles):
    return 0.5 * (np.abs(np.cos(angles)) + np.abs(np.sin(angles)))


def build_arm_lp(angles, centers, walls):
    """LP for fixed angles inside ``walls = (x0, x1, y0, y1)``.

    ``walls`` are inclusive bounds on the squares' axis-aligned extents.
    """
    n = len(angles)
    u = np.column_stack([np.cos(angles), np.sin(angles)])
    v = np.column_stack([-np.sin(angles), np.cos(angles)])
    pairs = list(cKDTree(centers).query_pairs(PAIR_RADIUS + 2 * extents(angles).max()))
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
        radii = 0.5 * (np.abs(axes @ u[i]) + np.abs(axes @ v[i])
                       + np.abs(axes @ u[j]) + np.abs(axes @ v[j]))
        k = int(np.argmax(np.abs(proj) - radii))
        a = axes[k] * (1 if proj[k] >= 0 else -1)
        add([2 * i, 2 * i + 1, 2 * j, 2 * j + 1],
            [a[0], a[1], -a[0], -a[1]], -radii[k] - MARGIN)

    e = extents(angles)
    x0, x1, y0, y1 = walls
    for i in range(n):
        # x - e >= x0 ; x + e <= x1
        add([2 * i], [1], x1 - e[i] - MARGIN)
        add([2 * i], [-1], -x0 - e[i] - MARGIN)
        add([2 * i + 1], [1], y1 - e[i] - MARGIN)
        add([2 * i + 1], [-1], -y0 - e[i] - MARGIN)
    A = coo_matrix((vals, (rows, cols)), shape=(len(rhs), 2 * n)).tocsr()
    return A, np.array(rhs)


def axis_radius(theta, phi):
    c = np.abs(np.cos(theta - phi))
    s = np.abs(np.sin(theta - phi))
    return 0.5 * (c + s)


def axis_radius_grad(theta, phi):
    d = theta - phi
    return 0.5 * (-np.sign(np.cos(d)) * np.sin(d)
                  + np.sign(np.sin(d)) * np.cos(d))


def extent_grad(theta):
    return 0.5 * (-np.sign(np.cos(theta)) * np.sin(theta)
                  + np.sign(np.sin(theta)) * np.cos(theta))


def build_arm_slp(theta, centers, walls, trust_angle, trust_move):
    """Feasibility SLP in (dx, dy, dtheta) with fixed rectangular walls."""
    n = len(theta)
    offsets = [np.array([np.cos(t), np.sin(t)]) for t in theta]
    normals = [np.array([-np.sin(t), np.cos(t)]) for t in theta]
    pairs = list(cKDTree(centers).query_pairs(PAIR_RADIUS))
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
        # rad_k = R_i(theta_i, phi_k) + R_j(theta_j, phi_k); no direction factor.
        di = axis_radius_grad(theta[i], phi[k])
        dj = axis_radius_grad(theta[j], phi[k])
        slack = direction * proj[k] - rad[k] - MARGIN
        # a.(dci) - a.(dcj) + di*dthi + dj*dthj <= slack
        add([ix(i), iy(i), ix(j), iy(j), it(i), it(j)],
            [a[0], a[1], -a[0], -a[1], di, dj], slack)

    e0 = extents(theta)
    grad = extent_grad(theta)
    x0, x1, y0, y1 = walls
    for i in range(n):
        # upper x: dx + grad*dth <= x1 - e0 - x0 - margin
        add([ix(i), it(i)], [1, grad[i]], x1 - e0[i] - centers[i, 0] - MARGIN)
        # lower x: -dx + grad*dth <= x0 - e0 - x0_wall? derive below
        add([ix(i), it(i)], [-1, grad[i]], centers[i, 0] - e0[i] - x0 - MARGIN)
        # upper y
        add([iy(i), it(i)], [1, grad[i]], y1 - e0[i] - centers[i, 1] - MARGIN)
        # lower y
        add([iy(i), it(i)], [-1, grad[i]], centers[i, 1] - e0[i] - y0 - MARGIN)

    A = coo_matrix((vals, (rows, cols)), shape=(len(rhs), 3 * n)).tocsr()
    bounds = []
    for i in range(n):
        bounds.append((-trust_move, trust_move))
        bounds.append((-trust_move, trust_move))
    for i in range(n):
        bounds.append((-trust_angle, trust_angle))
    return A, np.array(rhs), bounds


def solve_arm_slp(theta, centers, walls, rounds=40, time_limit=120.0):
    """Trust-region feasibility SLP; exact official certify is the oracle."""
    theta = theta.copy()
    centers = centers.copy()
    trust_angle = 0.02
    trust_move = 0.04
    best = None
    for _ in range(rounds):
        A, b, bounds = build_arm_slp(theta, centers, walls,
                                     trust_angle, trust_move)
        res = linprog(np.zeros(3 * len(theta)), A_ub=A, b_ub=b,
                      bounds=bounds, method="highs",
                      options={"dual_feasibility_tolerance": 1e-9,
                               "primal_feasibility_tolerance": 1e-9,
                               "time_limit": time_limit})
        if not res.success:
            trust_angle *= 0.5
            trust_move *= 0.5
            if trust_angle < 1e-7:
                break
            continue
        x = res.x
        nxt_c = np.column_stack([centers[:, 0] + x[0:2 * len(theta):2],
                                 centers[:, 1] + x[1:2 * len(theta):2]])
        nxt_t = theta + x[2 * len(theta):]
        cand = rows_from(0, nxt_c, nxt_t)
        e = extents(nxt_t)
        ok = ((nxt_c[:, 0] - e).min() >= walls[0] - 1e-12
              and (nxt_c[:, 0] + e).max() <= walls[1] + 1e-12
              and (nxt_c[:, 1] - e).min() >= walls[2] - 1e-12
              and (nxt_c[:, 1] + e).max() <= walls[3] + 1e-12)
        if ok:
            try:
                core.certify(len(theta), cand)
            except Exception:  # noqa: BLE001
                ok = False
        if ok:
            moved = max(np.abs(nxt_c - centers).max(),
                        np.abs(nxt_t - theta).max())
            centers, theta = nxt_c, nxt_t
            best = (centers.copy(), theta.copy())
            trust_angle = min(0.02, max(trust_angle * 1.5, 1e-5))
            trust_move = min(0.04, max(trust_move * 1.5, 1e-5))
            if moved < 1e-9:
                break
        else:
            trust_angle *= 0.5
            trust_move *= 0.5
            if trust_angle < 1e-7:
                break
    return best


def solve_arm(angles, centers, walls, rounds=12):
    cur = centers.copy()
    lo = np.array([walls[0], walls[2]])
    hi = np.array([walls[1], walls[3]])
    e = extents(angles)[:, None]
    cur = np.minimum(np.maximum(cur, lo + e), hi - e)
    for _ in range(rounds):
        A, b = build_arm_lp(angles, cur, walls)
        res = linprog(np.zeros(2 * len(angles)), A_ub=A, b_ub=b,
                      bounds=[(None, None)] * (2 * len(angles)), method="highs",
                      options={"dual_feasibility_tolerance": 1e-9,
                               "primal_feasibility_tolerance": 1e-9,
                               "time_limit": 120.0})
        if not res.success:
            return None
        nxt = np.asarray(res.x).reshape(-1, 2)
        shift = np.abs(nxt - cur).max()
        cur = nxt
        if shift < 1e-11:
            break
    # Verify exact feasibility inside the walls.
    e = extents(angles)
    if (cur[:, 0] - e).min() < walls[0] - 1e-12:
        return None
    if (cur[:, 0] + e).max() > walls[1] + 1e-12:
        return None
    if (cur[:, 1] - e).min() < walls[2] - 1e-12:
        return None
    if (cur[:, 1] + e).max() > walls[3] + 1e-12:
        return None
    return cur


def rows_from(n, centers, angles, start=0):
    return [dict(square_id=str(start + i), x=repr(float(centers[i, 0])),
                 y=repr(float(centers[i, 1])), angle=repr(float(angles[i])))
            for i in range(len(angles))]


def read_group(path, n):
    rows = []
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(row["n"]) == n:
                rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--incumbent", default="poc/pell_corrugation/data/border-scan/merged.csv")
    parser.add_argument("--report", default="poc/pell_corrugation/data/border-scan/merged-literal-official.json")
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--a", type=int, required=True, help="bottom-left block size")
    parser.add_argument("--g", type=int, required=True, help="top-right block size")
    parser.add_argument("--target-L", type=float, required=True)
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--output", required=True)
    parser.add_argument("--wins", required=True)
    args = parser.parse_args()

    rows = read_group(args.incumbent, args.n)
    if len(rows) != args.n:
        raise SystemExit(f"group {args.n} has {len(rows)} rows")
    x = np.array([float(r["x"]) for r in rows])
    y = np.array([float(r["y"]) for r in rows])
    angle = np.array([float(r["angle"]) for r in rows])
    ours = Decimal(core.certify(args.n, rows))
    e = extents(angle)
    L_cur = float(max((x + e).max() - (x - e).min(),
                      (y + e).max() - (y - e).min()))
    w_cur = L_cur - args.g
    a = args.a
    g = args.g
    W = args.target_L - g
    H = args.target_L - a
    print(f"n={args.n} L_cur={L_cur:.12f} a={a} g={g} w_cur={w_cur:.12f} "
          f"target W={W:.12f} H={H:.12f}")

    # Classify against the *current* split.
    region = np.full(len(rows), -1, dtype=int)
    for i in range(len(rows)):
        if x[i] < w_cur and y[i] >= a:
            region[i] = 0  # TL arm
        elif y[i] < w_cur and x[i] >= a:
            region[i] = 1  # BR arm
        elif x[i] >= w_cur and y[i] >= w_cur:
            region[i] = 2  # TR block
        elif x[i] < a and y[i] < a:
            region[i] = 3  # BL block
    counts = [int((region == k).sum()) for k in range(4)]
    print("region counts TL,BR,TR,BL =", counts)
    if min(counts) <= 0 or (region < 0).any():
        raise SystemExit("classification failed")

    delta = W - w_cur
    pieces = {}

    # TL arm: global walls x in [0,W], y in [a,L].
    m0 = region == 0
    tl0 = np.column_stack([x[m0], y[m0]])
    tl_slp = solve_arm_slp(angle[m0], tl0, (0.0, W, a, args.target_L))
    if tl_slp is not None:
        tl, tl_theta = tl_slp
        print("TL arm repacked (free-angle SLP)")
    else:
        tl = solve_arm(angle[m0], tl0, (0.0, W, a, args.target_L),
                       rounds=args.rounds)
        tl_theta = angle[m0]
        if tl is not None:
            print("TL arm repacked (fixed-angle LP)")
    if tl is None:
        print("TL arm infeasible at target")
        json.dump(dict(status="arm-infeasible", arm="TL"), open(args.wins, "w"))
        return

    # BR arm: global walls x in [a,L], y in [0,W].
    m1 = region == 1
    br0 = np.column_stack([x[m1], y[m1]])
    br_slp = solve_arm_slp(angle[m1], br0, (a, args.target_L, 0.0, W))
    if br_slp is not None:
        br, br_theta = br_slp
        print("BR arm repacked (free-angle SLP)")
    else:
        br = solve_arm(angle[m1], br0, (a, args.target_L, 0.0, W),
                       rounds=args.rounds)
        br_theta = angle[m1]
        if br is not None:
            print("BR arm repacked (fixed-angle LP)")
    if br is None:
        print("BR arm infeasible at target")
        json.dump(dict(status="arm-infeasible", arm="BR"), open(args.wins, "w"))
        return

    # TR block: rigid translation by delta in x and y.
    m2 = region == 2
    tr = np.column_stack([x[m2] + delta, y[m2] + delta])
    # BL block: untouched.
    m3 = region == 3
    bl = np.column_stack([x[m3], y[m3]])

    centers = np.vstack([tl, br, tr, bl])
    angles = np.concatenate([tl_theta, br_theta, angle[m2], angle[m3]])
    cand = rows_from(args.n, centers, angles)
    try:
        literal = Decimal(core.certify(args.n, cand))
        transported = core.kaggle_roundtrip({args.n: cand})
        pandas_L = Decimal(core.certify(args.n, transported[args.n]))
    except Exception as error:  # noqa: BLE001
        print("assembly certify failed:", error)
        json.dump(dict(status="assemble-fail", error=str(error)[:200]),
                  open(args.wins, "w"))
        return
    print(f"assembled literal={literal} pandas={pandas_L} ours={ours}")
    payload = dict(status="ok", n=args.n, old_L=str(ours),
                   new_L=str(max(literal, pandas_L)),
                   literal_L=str(literal), pandas_L=str(pandas_L),
                   a=a, g=g, rows=cand)
    if literal < ours and pandas_L < ours:
        print("WIN")
        json.dump({str(args.n): payload}, open(args.wins, "w"), indent=1)
        # write full merged file in sorted n order
        groups = {}
        with open(args.incumbent, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                groups.setdefault(int(row["n"]), []).append(row)
        groups[args.n] = [dict(n=str(args.n), square_id=r["square_id"],
                               x=r["x"], y=r["y"], angle=r["angle"]) for r in cand]
        with open(args.output, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["id", "n", "square_id", "x", "y", "angle"])
            index = 0
            for n in range(1, 1001):
                for row in groups[n]:
                    writer.writerow([index, n, row["square_id"], row["x"],
                                     row["y"], row["angle"]])
                    index += 1
    else:
        json.dump(dict(status="no-win", **{k: payload[k] for k in
                                           ("n", "old_L", "new_L")}),
                  open(args.wins, "w"), indent=1)


if __name__ == "__main__":
    main()


def build_arm_lip(theta, centers, walls, trust_angle, trust_move):
    """Lipschitz-safe feasibility LP for a fixed rectangle.

    Every pair row bounds the support growth by ``LIP*(t_i+t_j)`` with
    ``t_i >= |dtheta_i|``, so any accepted LP step is exactly overlap-free.
    Variables are (dx, dy, dtheta, t) per square.
    """
    n = len(theta)
    u = np.column_stack([np.cos(theta), np.sin(theta)])
    v = np.column_stack([-np.sin(theta), np.cos(theta)])
    pairs = list(cKDTree(centers).query_pairs(PAIR_RADIUS))
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
    x0, x1, y0, y1 = walls
    for i in range(n):
        add([itl(i), it(i)], [1, -1], 0.0)
        add([itl(i), it(i)], [1, 1], 0.0)
        add([ix(i), itl(i)], [1, LIP], x1 - e[i] - centers[i, 0] - MARGIN)
        add([ix(i), itl(i)], [-1, LIP], centers[i, 0] - e[i] - x0 - MARGIN)
        add([iy(i), itl(i)], [1, LIP], y1 - e[i] - centers[i, 1] - MARGIN)
        add([iy(i), itl(i)], [-1, LIP], centers[i, 1] - e[i] - y0 - MARGIN)

    A = coo_matrix((vals, (rows, cols)), shape=(len(rhs), 4 * n)).tocsr()
    bounds = []
    for _ in range(n):
        bounds.append((-trust_move, trust_move))
        bounds.append((-trust_move, trust_move))
    for _ in range(n):
        bounds.append((-trust_angle, trust_angle))
    for _ in range(n):
        bounds.append((0.0, trust_angle))
    return A, np.array(rhs), bounds


def solve_arm_lip(theta, centers, walls, rounds=200, time_limit=120.0):
    """Iterated Lipschitz-safe LP; returns a certified-in-rectangle arm."""
    theta = theta.copy()
    centers = centers.copy()
    ta, tm = 0.01, 0.03
    for _ in range(rounds):
        A, b, bounds = build_arm_lip(theta, centers, walls, ta, tm)
        res = linprog(np.zeros(4 * len(theta)), A_ub=A, b_ub=b,
                      bounds=bounds, method="highs",
                      options={"dual_feasibility_tolerance": 1e-9,
                               "primal_feasibility_tolerance": 1e-9,
                               "time_limit": time_limit})
        if not res.success:
            ta *= 0.5
            tm *= 0.5
            if ta < 1e-7:
                return None
            continue
        x = res.x
        nxt_c = centers + np.column_stack([x[0:2 * len(theta):2],
                                           x[1:2 * len(theta):2]])
        nxt_t = theta + x[2 * len(theta):3 * len(theta)]
        cand = rows_from(0, nxt_c, nxt_t)
        e = extents(nxt_t)
        ok = ((nxt_c[:, 0] - e).min() >= walls[0] - 1e-12
              and (nxt_c[:, 0] + e).max() <= walls[1] + 1e-12
              and (nxt_c[:, 1] - e).min() >= walls[2] - 1e-12
              and (nxt_c[:, 1] + e).max() <= walls[3] + 1e-12)
        if not ok:
            ta *= 0.5
            tm *= 0.5
            continue
        try:
            core.certify(len(theta), cand)
        except Exception:  # noqa: BLE001
            ta *= 0.5
            tm *= 0.5
            continue
        centers, theta = nxt_c, nxt_t
        moved = np.abs(x).max()
        if moved < 1e-10:
            return centers, theta
        ta = min(0.01, max(ta * 1.2, 1e-6))
        tm = min(0.03, max(tm * 1.2, 1e-6))
    try:
        core.certify(len(theta), rows_from(0, centers, theta))
    except Exception:  # noqa: BLE001
        return None
    return centers, theta
