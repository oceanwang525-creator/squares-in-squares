"""Fixed-angle box LP and Lipschitz-safe free-angle SLP, copied verbatim in
spirit from the repository's proven tools so this POC is self-contained:

* fixed-angle LP  : ``poc/alln_fixed_lp/poc/alln_fixed_lp.py`` (``build_lp``,
  ``solve_lp``, ``refine``) -- no trust region, one separating axis per nearby
  pair, rebuilt every round.
* free-angle SLP  : ``poc/lip_slp/poc/lip_slp.py`` (``build``) -- support growth
  bounded by ``LIP*(t_i+t_j)`` so every accepted step is overlap-free.

Nothing here is modified relative to those tools except that the entry points
accept plain ``(centers, angles)`` arrays and return them.
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = "1"

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree

PAIR_RADIUS = 1.6
MARGIN = 1e-9
LIP = 0.7071067811865476


def extents(angles):
    return 0.5 * (np.abs(np.cos(angles)) + np.abs(np.sin(angles)))


def exact_side(centers, angles):
    e = extents(angles)
    return float(max((centers[:, 0] + e).max() - (centers[:, 0] - e).min(),
                     (centers[:, 1] + e).max() - (centers[:, 1] - e).min()))


# --------------------------------------------------------------------------
# fixed-angle LP (centres only)
# --------------------------------------------------------------------------
def build_lp(angles, centers):
    n = len(angles)
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
        radii = 0.5 * (np.abs(axes @ u[i]) + np.abs(axes @ v[i])
                       + np.abs(axes @ u[j]) + np.abs(axes @ v[j]))
        alpha = np.abs(proj) - radii
        k = int(np.argmax(alpha))
        a = axes[k] * (1 if proj[k] >= 0 else -1)
        add([2 * i, 2 * i + 1, 2 * j, 2 * j + 1],
            [a[0], a[1], -a[0], -a[1]], -radii[k] - MARGIN)

    ext = extents(angles)
    for i in range(n):
        for k in range(2):
            add([2 * i + k], [-1], -ext[i] - MARGIN)
            add([2 * i + k, 2 * n], [1, -1], -ext[i] - MARGIN)

    A = coo_matrix((vals, (rows, cols)), shape=(len(rhs), 2 * n + 1)).tocsr()
    c = np.zeros(2 * n + 1)
    c[-1] = 1.0
    return A, np.array(rhs), c


def solve_lp(angles, centers, time_limit=120.0):
    A, b, c = build_lp(angles, centers)
    n = len(angles)
    res = linprog(c, A_ub=A, b_ub=b,
                  bounds=[(None, None)] * (2 * n) + [(0, None)],
                  method="highs",
                  options={"dual_feasibility_tolerance": 1e-9,
                           "primal_feasibility_tolerance": 1e-9,
                           "time_limit": time_limit})
    if not res.success:
        return None
    return float(res.fun), np.asarray(res.x[:-1]).reshape(n, 2)


def fixed_lp_refine(angles, centers, rounds=8, time_limit=120.0):
    """Returns (best_lp_side, centers_of_best) or None if the LP is infeasible."""
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


# --------------------------------------------------------------------------
# Lipschitz-safe free-angle SLP
# --------------------------------------------------------------------------
def build_slp(theta, centers, L0, trust_angle, trust_move):
    n = len(theta)
    u = np.column_stack([np.cos(theta), np.sin(theta)])
    v = np.column_stack([-np.sin(theta), np.cos(theta)])
    pairs = list(cKDTree(centers).query_pairs(1.65))
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
        k = int(np.argmax(np.abs(proj) - rad))
        direction = 1.0 if proj[k] >= 0 else -1.0
        a = axes[k] * direction
        slack = direction * proj[k] - rad[k] - MARGIN
        add([ix(i), iy(i), ix(j), iy(j), itl(i), itl(j)],
            [a[0], a[1], -a[0], -a[1], LIP, LIP], slack)

    e = extents(theta)
    for i in range(n):
        add([itl(i), it(i)], [1, -1], 0.0)
        add([itl(i), it(i)], [1, 1], 0.0)
        add([ix(i), itl(i), iL], [1, LIP, -1], L0 - centers[i, 0] - e[i] - MARGIN)
        add([ix(i), itl(i)], [-1, LIP], centers[i, 0] - e[i] - MARGIN)
        add([iy(i), itl(i), iL], [1, LIP, -1], L0 - centers[i, 1] - e[i] - MARGIN)
        add([iy(i), itl(i)], [-1, LIP], centers[i, 1] - e[i] - MARGIN)

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


def slp_step(theta, centers, L, ta, tm):
    n = len(theta)
    A, b, bounds = build_slp(theta, centers, L, ta, tm)
    c = np.zeros(4 * n + 1)
    c[-1] = 1.0
    res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="highs",
                  options={"dual_feasibility_tolerance": 1e-9,
                           "primal_feasibility_tolerance": 1e-9,
                           "time_limit": 120.0})
    if not res.success:
        return None
    x = res.x
    nc = centers + np.column_stack([x[0:2 * n:2], x[1:2 * n:2]])
    nt = theta + x[2 * n:3 * n]
    return nc, nt, float(np.abs(x).max())
