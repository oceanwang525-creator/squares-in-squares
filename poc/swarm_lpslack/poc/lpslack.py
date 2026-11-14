#!/usr/bin/env python3
"""H75: Alternating Soft-Angle Dual-Contact Release SLP + Unbounded Fixed-Angle LP + Uniform Shrink."""
from __future__ import annotations

import os

for _var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_var] = "1"

import argparse
import csv
import io
import json
import math
import multiprocessing as mp
import sys
import time
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "poc" / "baseline" / "poc" / "src"))
sys.path.insert(0, str(REPO / "poc" / "auto_union" / "poc"))
from squares import core  # noqa: E402
import union as auto_union  # noqa: E402

PAIR_RADIUS = 1.65
SHRINK_RADIUS = 2.0


def exact_side(centers: np.ndarray, angles: np.ndarray) -> float:
    ext = 0.5 * (np.abs(np.cos(angles)) + np.abs(np.sin(angles)))
    return float(
        max(
            (centers[:, 0] + ext).max() - (centers[:, 0] - ext).min(),
            (centers[:, 1] + ext).max() - (centers[:, 1] - ext).min(),
        )
    )


def sat_gaps(centers: np.ndarray, angles: np.ndarray, radius: float = SHRINK_RADIUS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (pairs, gaps, distances) for all pairs within radius."""
    pairs = cKDTree(centers).query_pairs(radius, output_type="ndarray")
    if len(pairs) == 0:
        return pairs, np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    i, j = pairs.T
    delta = centers[j] - centers[i]
    ai, aj = angles[i], angles[j]
    ca, sa = np.cos(ai), np.sin(ai)
    cb, sb = np.cos(aj), np.sin(aj)
    axes_x = np.stack((ca, -sa, cb, -sb), axis=1)
    axes_y = np.stack((sa, ca, sb, cb), axis=1)
    proj = delta[:, 0:1] * axes_x + delta[:, 1:2] * axes_y
    cosine = ca * cb + sa * sb
    sine = ca * sb - sa * cb
    rad = 0.5 * (1.0 + np.abs(cosine) + np.abs(sine))
    gaps = (np.abs(proj) - rad[:, None]).max(axis=1)
    dists = np.hypot(delta[:, 0], delta[:, 1])
    return pairs, gaps, dists


def min_sat_gap(centers: np.ndarray, angles: np.ndarray) -> float:
    _, gaps, _ = sat_gaps(centers, angles, radius=PAIR_RADIUS)
    return float(gaps.min()) if len(gaps) else math.inf


def build_fixed_lp(
    angles: np.ndarray,
    centers: np.ndarray,
    margin: float = 1e-10,
    box_margin: float = 0.0,
    rng: np.random.Generator | None = None,
    tie_tol: float = 0.0,
    pair_radius: float = PAIR_RADIUS,
):
    n = len(angles)
    u = np.column_stack([np.cos(angles), np.sin(angles)])
    v = np.column_stack([-np.sin(angles), np.cos(angles)])
    pairs = cKDTree(centers).query_pairs(pair_radius, output_type="ndarray")

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

    for idx_p in range(len(pairs)):
        i, j = int(pairs[idx_p, 0]), int(pairs[idx_p, 1])
        axes = np.array([u[i], v[i], u[j], v[j]])
        delta = centers[j] - centers[i]
        proj = axes @ delta
        radii = 0.5 * (
            np.abs(axes @ u[i])
            + np.abs(axes @ v[i])
            + np.abs(axes @ u[j])
            + np.abs(axes @ v[j])
        )
        alpha = np.abs(proj) - radii
        if rng is not None and tie_tol > 0.0:
            near = np.flatnonzero(alpha >= alpha.max() - tie_tol)
            k = int(near[rng.integers(len(near))]) if len(near) > 1 else int(np.argmax(alpha))
        else:
            k = int(np.argmax(alpha))
        a = axes[k] * (1.0 if proj[k] >= 0.0 else -1.0)
        add(
            [2 * i, 2 * i + 1, 2 * j, 2 * j + 1],
            [float(a[0]), float(a[1]), float(-a[0]), float(-a[1])],
            float(-radii[k] - margin),
        )

    ext = 0.5 * (np.abs(np.cos(angles)) + np.abs(np.sin(angles)))
    for i in range(n):
        e = float(ext[i])
        for coord in range(2):
            add([2 * i + coord], [-1.0], -e - box_margin)
            add([2 * i + coord, 2 * n], [1.0, -1.0], -e - box_margin)

    A = coo_matrix((vals, (rows, cols)), shape=(len(rhs), 2 * n + 1)).tocsr()
    c = np.zeros(2 * n + 1)
    c[-1] = 1.0
    return A, np.array(rhs, dtype=np.float64), c, pairs


def solve_fixed_lp(
    angles: np.ndarray,
    centers: np.ndarray,
    margin: float = 1e-10,
    box_margin: float = 0.0,
    time_limit: float = 30.0,
    rng: np.random.Generator | None = None,
    tie_tol: float = 0.0,
    trust_move: float | None = None,
):
    n = len(angles)
    A, b, c, pairs = build_fixed_lp(
        angles, centers, margin=margin, box_margin=box_margin, rng=rng, tie_tol=tie_tol
    )
    if trust_move is None:
        bounds = [(None, None)] * (2 * n) + [(0.0, None)]
    else:
        bounds = []
        for i in range(n):
            bounds.append((float(centers[i, 0]) - trust_move, float(centers[i, 0]) + trust_move))
            bounds.append((float(centers[i, 1]) - trust_move, float(centers[i, 1]) + trust_move))
        bounds.append((0.0, None))
    res = linprog(
        c,
        A_ub=A,
        b_ub=b,
        bounds=bounds,
        method="highs",
        options={
            "dual_feasibility_tolerance": 1e-9,
            "primal_feasibility_tolerance": 1e-9,
            "time_limit": time_limit,
        },
    )
    if not res.success:
        return None
    new_centers = np.asarray(res.x[:-1], dtype=np.float64).reshape(n, 2)
    marginals = np.asarray(res.ineqlin.marginals, dtype=np.float64)
    return float(res.fun), new_centers, pairs, marginals


def fixed_lp_loop(
    angles: np.ndarray,
    centers: np.ndarray,
    rounds: int = 12,
    margin: float = 1e-10,
    time_limit: float = 30.0,
    rng: np.random.Generator | None = None,
    tie_tol: float = 0.0,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray] | None:
    cur = centers.copy()
    best_side = exact_side(cur, angles) if min_sat_gap(cur, angles) > 0.0 else math.inf
    best_centers = cur.copy()
    last_pairs = np.empty((0, 2), dtype=np.int64)
    last_marginals = np.empty(0, dtype=np.float64)

    for _ in range(rounds):
        out = solve_fixed_lp(
            angles, cur, margin=margin, box_margin=0.0, time_limit=time_limit, rng=rng, tie_tol=tie_tol
        )
        if out is None:
            break
        _, proposed, pairs, marginals = out
        last_pairs, last_marginals = pairs, marginals
        gap = min_sat_gap(proposed, angles)
        if gap <= 0.0:
            # Tiny precision overlap from margin -> nudge apart uniformly
            pairs_g, gaps_g, dists_g = sat_gaps(proposed, angles, radius=PAIR_RADIUS)
            if len(gaps_g) and gaps_g.min() <= 0.0:
                eps = float((-gaps_g / np.maximum(dists_g, 1e-9)).max()) + 1e-11
                if eps < 1e-6:
                    c0 = proposed.mean(axis=0)
                    proposed = c0 + (1.0 + eps) * (proposed - c0)
                    gap = min_sat_gap(proposed, angles)
        side = exact_side(proposed, angles)
        shift = float(np.abs(proposed - cur).max())
        if gap > 0.0 and side < best_side - 1e-13:
            best_side = side
            best_centers = proposed.copy()
        cur = proposed
        if shift < 1e-11:
            break
    return best_side, best_centers, last_pairs, last_marginals


def _active_branches(angle: float, branch_tol: float = 0.05) -> list[tuple[int, int]]:
    """Return (s1, s2) in {+-1}^2 branches near max of |cos| + |sin|."""
    c, s = math.cos(angle), math.sin(angle)
    best = abs(c) + abs(s)
    out = []
    for s1 in (1, -1):
        for s2 in (1, -1):
            if s1 * c + s2 * s >= best - branch_tol:
                out.append((s1, s2))
    return out


def build_soft_angle_slp(
    angles: np.ndarray,
    centers: np.ndarray,
    free_mask: np.ndarray,
    tr_th: float,
    tr_xy: float | None,
    margin: float = 1e-8,
    rng: np.random.Generator | None = None,
    tie_tol: float = 0.0,
):
    """Joint (x, y, dtheta, L) SLP with co-rotating separating axes and 4-sign kink branches."""
    n = len(angles)
    u = np.column_stack([np.cos(angles), np.sin(angles)])
    v = np.column_stack([-np.sin(angles), np.cos(angles)])
    pairs = cKDTree(centers).query_pairs(PAIR_RADIUS, output_type="ndarray")

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

    # Variables: x_i (2*i), y_i (2*i+1), dtheta_i (2*n+i), L (3*n)
    for idx_p in range(len(pairs)):
        i, j = int(pairs[idx_p, 0]), int(pairs[idx_p, 1])
        axes = np.array([u[i], v[i], u[j], v[j]])
        phis = [
            float(angles[i]),
            float(angles[i] + 0.5 * math.pi),
            float(angles[j]),
            float(angles[j] + 0.5 * math.pi),
        ]
        delta = centers[j] - centers[i]
        proj = axes @ delta
        radii = 0.5 * (
            np.abs(axes @ u[i])
            + np.abs(axes @ v[i])
            + np.abs(axes @ u[j])
            + np.abs(axes @ v[j])
        )
        alpha = np.abs(proj) - radii
        if rng is not None and tie_tol > 0.0:
            near = np.flatnonzero(alpha >= alpha.max() - tie_tol)
            k = int(near[rng.integers(len(near))]) if len(near) > 1 else int(np.argmax(alpha))
        else:
            k = int(np.argmax(alpha))

        sgn = 1.0 if proj[k] >= 0.0 else -1.0
        a0 = axes[k] * sgn
        a_perp = np.array([-a0[1], a0[0]])
        phi_0 = phis[k] + (0.0 if sgn > 0 else math.pi)
        owner = i if k < 2 else j
        other = j if k < 2 else i

        rot_lever = float(np.dot(a_perp, delta))
        d0 = float(angles[other]) - phi_0

        for s1, s2 in _active_branches(d0, branch_tol=0.08):
            val0 = 0.5 * (s1 * math.cos(d0) + s2 * math.sin(d0))
            grad0 = 0.5 * (-s1 * math.sin(d0) + s2 * math.cos(d0))
            # a0.(c_j' - c_i') + (a_perp.delta + grad0)*dtheta_owner - grad0*dtheta_other >= 0.5 + val0 + margin
            col_list = [2 * i, 2 * i + 1, 2 * j, 2 * j + 1]
            val_list = [float(a0[0]), float(a0[1]), float(-a0[0]), float(-a0[1])]
            if free_mask[owner]:
                col_list.append(2 * n + owner)
                val_list.append(-(rot_lever + grad0))
            if free_mask[other]:
                col_list.append(2 * n + other)
                val_list.append(grad0)
            add(col_list, val_list, -(0.5 + val0 + margin))

    for i in range(n):
        th = float(angles[i])
        for s1, s2 in _active_branches(th, branch_tol=0.08):
            val0 = 0.5 * (s1 * math.cos(th) + s2 * math.sin(th))
            grad0 = 0.5 * (-s1 * math.sin(th) + s2 * math.cos(th))
            if free_mask[i]:
                add([2 * i, 2 * n + i], [-1.0, grad0], -val0)
                add([2 * i, 2 * n + i, 3 * n], [1.0, grad0, -1.0], -val0)
                add([2 * i + 1, 2 * n + i], [-1.0, grad0], -val0)
                add([2 * i + 1, 2 * n + i, 3 * n], [1.0, grad0, -1.0], -val0)
            else:
                add([2 * i], [-1.0], -val0)
                add([2 * i, 3 * n], [1.0, -1.0], -val0)
                add([2 * i + 1], [-1.0], -val0)
                add([2 * i + 1, 3 * n], [1.0, -1.0], -val0)

    A = coo_matrix((vals, (rows, cols)), shape=(len(rhs), 3 * n + 1)).tocsr()
    c = np.zeros(3 * n + 1)
    c[-1] = 1.0
    bounds = []
    for i in range(n):
        if tr_xy is None:
            bounds.append((None, None))
            bounds.append((None, None))
        else:
            bounds.append((float(centers[i, 0]) - tr_xy, float(centers[i, 0]) + tr_xy))
            bounds.append((float(centers[i, 1]) - tr_xy, float(centers[i, 1]) + tr_xy))
    for i in range(n):
        if free_mask[i]:
            bounds.append((-tr_th, tr_th))
        else:
            bounds.append((0.0, 0.0))
    bounds.append((0.0, None))
    return A, np.array(rhs, dtype=np.float64), c, bounds


def solve_soft_angle_slp(
    angles: np.ndarray,
    centers: np.ndarray,
    free_mask: np.ndarray,
    tr_th: float,
    tr_xy: float | None,
    margin: float = 1e-8,
    time_limit: float = 30.0,
    rng: np.random.Generator | None = None,
    tie_tol: float = 0.0,
):
    n = len(angles)
    A, b, c, bounds = build_soft_angle_slp(
        angles, centers, free_mask, tr_th, tr_xy, margin=margin, rng=rng, tie_tol=tie_tol
    )
    res = linprog(
        c,
        A_ub=A,
        b_ub=b,
        bounds=bounds,
        method="highs",
        options={
            "dual_feasibility_tolerance": 1e-9,
            "primal_feasibility_tolerance": 1e-9,
            "time_limit": time_limit,
        },
    )
    if not res.success:
        return None
    x = np.asarray(res.x, dtype=np.float64)
    new_centers = x[: 2 * n].reshape(n, 2)
    dtheta = x[2 * n : 3 * n]
    new_angles = angles + dtheta
    return float(x[-1]), new_centers, new_angles, float(np.abs(dtheta).max())


def select_free_mask(
    angles: np.ndarray,
    mode: str,
    pairs: np.ndarray | None = None,
    marginals: np.ndarray | None = None,
    top_k: int = 120,
) -> np.ndarray:
    n = len(angles)
    deg90 = np.abs(np.sin(2.0 * angles))
    non90 = deg90 > 1e-4

    if mode == "non90":
        return non90
    if mode == "all":
        return np.ones(n, dtype=bool)
    if mode in ("dual_non90", "dual_all"):
        if pairs is None or marginals is None or len(pairs) == 0:
            return non90 if mode == "dual_non90" else np.ones(n, dtype=bool)
        weights = np.zeros(n, dtype=np.float64)
        npair = len(pairs)
        pair_duals = np.abs(marginals[:npair])
        for idx_p in np.flatnonzero(pair_duals > 1e-12):
            w = pair_duals[idx_p]
            weights[int(pairs[idx_p, 0])] += w
            weights[int(pairs[idx_p, 1])] += w
        # Also include box duals
        box_duals = np.abs(marginals[npair:])
        for i in range(n):
            base = 4 * i
            if base + 3 < len(box_duals):
                weights[i] += box_duals[base : base + 4].sum()
        if mode == "dual_non90":
            weights[~non90] = 0.0
        active = np.flatnonzero(weights > 1e-12)
        if len(active) == 0:
            return non90 if mode == "dual_non90" else np.ones(n, dtype=bool)
        if len(active) > top_k:
            order = active[np.argsort(-weights[active])[:top_k]]
        else:
            order = active
        mask = np.zeros(n, dtype=bool)
        mask[order] = True
        return mask
    raise ValueError(f"Unknown mode {mode}")


def rows_from_arrays(centers: np.ndarray, angles: np.ndarray) -> list[dict[str, str]]:
    return [
        {
            "square_id": str(i),
            "x": repr(float(centers[i, 0])),
            "y": repr(float(centers[i, 1])),
            "angle": repr(float(angles[i])),
        }
        for i in range(len(angles))
    ]


def bbox_center_decimal(rows: list[dict[str, str]]) -> tuple[Decimal, Decimal]:
    xmin = xmax = ymin = ymax = None
    for row in rows:
        angle = float(row["angle"])
        half = 0.5 * (abs(math.cos(angle)) + abs(math.sin(angle)))
        x, y = float(row["x"]), float(row["y"])
        xmin = x - half if xmin is None else min(xmin, x - half)
        xmax = x + half if xmax is None else max(xmax, x + half)
        ymin = y - half if ymin is None else min(ymin, y - half)
        ymax = y + half if ymax is None else max(ymax, y + half)
    return Decimal(repr((xmin + xmax) / 2.0)), Decimal(repr((ymin + ymax) / 2.0))


def uniform_shrink_rows(rows: list[dict[str, str]], ratio: float, factor: float) -> list[dict[str, str]]:
    cx, cy = bbox_center_decimal(rows)
    with localcontext() as ctx:
        ctx.prec = 80
        scale = Decimal(1) - Decimal(repr(factor * ratio))
        quantum = Decimal("1e-55")
        out = []
        for row in rows:
            x = cx + scale * (Decimal(row["x"]) - cx)
            y = cy + scale * (Decimal(row["y"]) - cy)
            out.append(
                {
                    "square_id": row["square_id"],
                    "x": format(x.quantize(quantum), "f"),
                    "y": format(y.quantize(quantum), "f"),
                    "angle": row["angle"],
                }
            )
        return out


def certify_pair(n: int, rows: list[dict[str, str]]) -> tuple[Decimal, Decimal] | None:
    try:
        lit = Decimal(core.certify(n, rows))
        transported = core.kaggle_roundtrip({n: rows})[n]
        pan = Decimal(core.certify(n, transported))
    except Exception:  # noqa: BLE001
        return None
    return lit, pan


def try_shrink_and_certify(
    n: int, centers: np.ndarray, angles: np.ndarray, old_L: Decimal
) -> tuple[Decimal, Decimal, list[dict[str, str]]] | None:
    """Try uniform separating-axis shrink (and raw candidate); return best strict certified win."""
    rows = rows_from_arrays(centers, angles)
    _, gaps, dists = sat_gaps(centers, angles, radius=SHRINK_RADIUS)
    candidates: list[list[dict[str, str]]] = []
    if len(gaps) > 0:
        min_gap = float(gaps.min())
        if min_gap > 1e-12:
            ratio = float((gaps / np.maximum(dists, 1e-9)).min())
            max_dist = float(dists.max()) if len(dists) else 1.0
            # Keep >= 5e-13 residual gap so 50-dps interval arithmetic & float64 pandas pass fast.
            safe_ratio = min(ratio * 0.95, max(0.0, ratio - 5e-13 / max(max_dist, 1.0)))
            if safe_ratio > 1e-13:
                for factor in (0.98, 0.80, 0.40):
                    candidates.append(uniform_shrink_rows(rows, safe_ratio, factor))
    candidates.append(rows)

    for cand in candidates:
        res = certify_pair(n, cand)
        if res is None:
            continue
        lit, pan = res
        if lit < old_L and pan < old_L:
            return lit, pan, cand
    return None


def optimize_group(
    n: int,
    base_rows: list[dict[str, str]],
    old_L_str: str,
    outer_iters: int = 8,
    fixed_rounds: int = 10,
    slp_modes: tuple[str, ...] = ("dual_non90", "non90", "all"),
    tr_th_schedule: tuple[float, ...] = (5e-4, 2e-4, 5e-5),
    time_limit: float = 15.0,
) -> dict:
    t0 = time.time()
    old_L = Decimal(old_L_str)
    centers0 = np.array([[float(r["x"]), float(r["y"])] for r in base_rows], dtype=np.float64)
    angles0 = np.array([float(r["angle"]) for r in base_rows], dtype=np.float64)

    best_centers = centers0.copy()
    best_angles = angles0.copy()
    best_L_float = exact_side(best_centers, best_angles)
    history: list[str] = []

    # Step 0: pure fixed-angle LP baseline
    for margin in (1e-10, 1e-9):
        out = fixed_lp_loop(angles0, centers0, rounds=12, margin=margin, time_limit=time_limit)
        if out is not None:
            s_fx, c_fx, _, _ = out
            if min_sat_gap(c_fx, angles0) > 1e-12 and s_fx < best_L_float - 1e-12:
                best_L_float = s_fx
                best_centers = c_fx.copy()
                best_angles = angles0.copy()
                history.append(f"fixed0(m={margin:.0e}):{s_fx:.12f}")

    # Step 1: Alternating Soft-Angle Dual-Contact Release SLP + Fixed-Angle LP
    for mode in slp_modes:
        for tr_th0 in tr_th_schedule:
            cur_c = best_centers.copy()
            cur_a = best_angles.copy()
            cur_pairs, cur_marg = None, None
            fx_init = solve_fixed_lp(cur_a, cur_c, margin=1e-10, time_limit=time_limit)
            if fx_init is not None:
                _, _, cur_pairs, cur_marg = fx_init

            tr_th = tr_th0
            tr_xy = min(0.02, max(2e-3, 10.0 * tr_th))
            for it in range(outer_iters):
                free_mask = select_free_mask(cur_a, mode, cur_pairs, cur_marg, top_k=150)
                if not np.any(free_mask):
                    break
                slp_ok = False
                for _ in range(2):
                    curv_pad = 1e-8 + 2.0 * tr_th * tr_xy
                    slp_out = solve_soft_angle_slp(
                        cur_a, cur_c, free_mask, tr_th=tr_th, tr_xy=tr_xy, margin=curv_pad, time_limit=time_limit
                    )
                    if slp_out is None:
                        slp_out = solve_soft_angle_slp(
                            cur_a, cur_c, free_mask, tr_th=tr_th, tr_xy=None, margin=1e-8, time_limit=time_limit
                        )
                    if slp_out is None:
                        break
                    _, prop_c, prop_a, max_dth = slp_out
                    cur_c, cur_a = prop_c, prop_a
                    slp_ok = True
                    if max_dth < 1e-9:
                        break
                if not slp_ok:
                    tr_th *= 0.5
                    tr_xy *= 0.5
                    continue

                # Follow immediately with unbounded fixed-angle LP rounds
                fx_out = fixed_lp_loop(cur_a, cur_c, rounds=fixed_rounds, margin=1e-10, time_limit=time_limit)
                if fx_out is None:
                    fx_out = fixed_lp_loop(cur_a, cur_c, rounds=fixed_rounds, margin=1e-9, time_limit=time_limit)
                if fx_out is None:
                    tr_th *= 0.5
                    continue
                s_fx, c_fx, cur_pairs, cur_marg = fx_out
                cur_c = c_fx
                gap_fx = min_sat_gap(cur_c, cur_a)

                if gap_fx > 1e-12 and s_fx < best_L_float - 1e-12:
                    best_L_float = s_fx
                    best_centers = cur_c.copy()
                    best_angles = cur_a.copy()
                    history.append(f"{mode}(tr={tr_th:.1e},it={it}):{best_L_float:.12f}")
                else:
                    tr_th *= 0.5
                    tr_xy = min(0.02, max(1e-3, 10.0 * tr_th))
                    if tr_th < 1e-6:
                        break

    # Step 2: Final uniform separating-axis shrink + official literal & pandas certification
    best_cert = None
    if best_L_float < float(old_L) - 1e-12 or min_sat_gap(best_centers, best_angles) > 1e-11:
        best_cert = try_shrink_and_certify(n, best_centers, best_angles, old_L)

    elapsed = round(time.time() - t0, 2)
    if best_cert is not None:
        lit, pan, cand = best_cert
        dL = old_L - max(lit, pan)
        return {
            "n": n,
            "status": "WIN" if dL >= Decimal("1e-9") else "TINY_WIN",
            "old_L": str(old_L),
            "new_L": str(max(lit, pan)),
            "literal_L": str(lit),
            "pandas_L": str(pan),
            "dL": str(dL),
            "elapsed_s": elapsed,
            "history": history,
            "rows": cand,
        }
    return {
        "n": n,
        "status": "NO_WIN",
        "old_L": str(old_L),
        "best_float_L": best_L_float,
        "elapsed_s": elapsed,
        "history": history,
    }


def _certify_candidate(item: tuple[int, int, list[dict[str, str]]]) -> tuple[int, int, str | None]:
    n, idx, rows = item
    res = certify_pair(n, rows)
    if res is None:
        return n, idx, None
    return n, idx, str(max(res[0], res[1]))


def load_targets_from_portfolio(targets: set[int], jobs: int = 16) -> dict[int, tuple[list[dict[str, str]], Decimal]]:
    """Load target n from origin/main portfolio (merged.csv + all poc/*/final.csv) fast."""
    import pandas as pd

    candidates: dict[int, list[list[dict[str, str]]]] = {n: [] for n in targets}
    merged_path = REPO / "poc" / "pell_corrugation" / "data" / "border-scan" / "merged.csv"
    with merged_path.open(newline="", encoding="utf-8") as fh:
        tmp: dict[int, list[dict[str, str]]] = {}
        for row in csv.DictReader(fh):
            n = int(row["n"])
            if n in targets:
                tmp.setdefault(n, []).append(
                    {"square_id": row["square_id"], "x": row["x"], "y": row["y"], "angle": row["angle"]}
                )
        for n, rows in tmp.items():
            candidates[n].append(rows)

    for p in sorted((REPO / "poc").glob("*/final.csv")):
        if "swarm_lpslack" in p.parts:
            continue
        text = p.read_text(encoding="utf-8")
        lines = text.splitlines()
        if len(lines) <= 1:
            continue
        prefixes = tuple(f"{n:03}_" if n < 1000 else "1000_" for n in targets)
        sub_lines = [lines[0]] + [ln for ln in lines[1:] if ln.startswith(prefixes)]
        if len(sub_lines) <= 1:
            continue
        sub_text = "\n".join(sub_lines) + "\n"
        a = auto_union.parse_text(sub_text)
        b = auto_union.parse_text(pd.read_csv(io.StringIO(sub_text)).to_csv(index=False))
        for n in a:
            if n in targets and len(a[n]) == n and len(b.get(n, [])) == n:
                candidates[n].append(a[n])

    cert_tasks = [(n, idx, rows) for n in sorted(targets) for idx, rows in enumerate(candidates[n])]
    best_idx: dict[int, tuple[int, Decimal]] = {}
    with mp.get_context("fork").Pool(min(jobs, max(1, len(cert_tasks)))) as pool:
        for n, idx, val in pool.imap_unordered(_certify_candidate, cert_tasks, chunksize=1):
            if val is None:
                continue
            cur_L = Decimal(val)
            if n not in best_idx or cur_L < best_idx[n][1]:
                best_idx[n] = (idx, cur_L)

    best: dict[int, tuple[list[dict[str, str]], Decimal]] = {}
    for n in sorted(targets):
        if n not in best_idx:
            raise RuntimeError(f"No valid incumbent found for n={n}")
        idx, best_L = best_idx[n]
        best[n] = (candidates[n][idx], best_L)
    return best


def _run_task(args):
    n, base_rows, old_L_str, outer_iters, fixed_rounds, tr_sched = args
    return optimize_group(
        n,
        base_rows,
        old_L_str,
        outer_iters=outer_iters,
        fixed_rounds=fixed_rounds,
        tr_th_schedule=tr_sched,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-list", default="911,740,861,796,853,909")
    parser.add_argument("--jobs", type=int, default=24)
    parser.add_argument("--outer-iters", type=int, default=8)
    parser.add_argument("--fixed-rounds", type=int, default=8)
    parser.add_argument("--tr-sched", default="5e-4,5e-5")
    parser.add_argument("--out", default=str(REPO / "poc/swarm_lpslack/data/scale_results.json"))
    parser.add_argument("--jsonl", default=str(REPO / "poc/swarm_lpslack/data/scale_results.jsonl"))
    args = parser.parse_args()

    targets = sorted({int(x) for x in args.n_list.split(",") if x.strip()}, reverse=True)
    tr_sched = tuple(float(x) for x in args.tr_sched.split(",") if x.strip())
    portfolio = load_targets_from_portfolio(set(targets), jobs=args.jobs)
    tasks = [
        (n, portfolio[n][0], str(portfolio[n][1]), args.outer_iters, args.fixed_rounds, tr_sched)
        for n in targets
    ]

    jsonl_path = Path(args.jsonl)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path.write_text("", encoding="utf-8")

    results = []
    with mp.get_context("fork").Pool(min(args.jobs, len(tasks))) as pool:
        for res in pool.imap_unordered(_run_task, tasks, chunksize=1):
            results.append(res)
            with jsonl_path.open("a", encoding="utf-8") as jf:
                jf.write(json.dumps(res) + "\n")
            dL = res.get("dL", "0")
            print(
                f"[{len(results)}/{len(tasks)}] n={res['n']} status={res['status']} "
                f"old={res['old_L'][:16]} new={res.get('new_L', str(res.get('best_float_L', '')))[:16]} "
                f"dL={dL} t={res['elapsed_s']}s",
                flush=True,
            )

    results.sort(key=lambda r: r["n"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
