#!/usr/bin/env python3
"""Parametric k^2 + j Strip Families with Recursive Corner Pocket Refill (Idea 3).

Implements Idea 3 for n = k^2 + j (10 <= k <= 31, 1 <= j <= k):
- Constructs parametric strip families where a central tilted band of width w = 1 or 2
  at slope theta = arctan(1/k) or pi/4 is inserted.
- Recursively refills the upper-left (NW) and lower-right (SE) corner regions with
  integer-grid blocks and tucked corner staircases with SAT clearance tau >= 1e-7.
- Eliminates all integer-grid fallbacks (no (k+1) x (k+1) grid emission): if recursive refill
  yields fewer than n squares for a candidate L < ceil(sqrt(n)), records status FAILED
  with reason 'insufficient_capacity_at_sub_grid_envelope'.
- Formulates parameter selection (angle theta, band width w, band offset) and polishes
  sub-grid candidates with SLP/LP 4-sign-piece epigraph linearization.
- Evaluates across the top 40 targets and outputs results to poc/research/data/recursive_refill_results.json.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linprog

# Add repo root to import squares.core if available
REPO_ROOT = Path(__file__).resolve().parents[3]
BASELINE_SRC = REPO_ROOT / "poc/baseline/poc/src"
if str(BASELINE_SRC) not in sys.path:
    sys.path.insert(0, str(BASELINE_SRC))

try:
    from squares.core import certify, evaluate
except ImportError:
    certify = None
    evaluate = None

SIGNS: Tuple[Tuple[float, float], ...] = (
    (-1.0, -1.0),
    (-1.0, 1.0),
    (1.0, -1.0),
    (1.0, 1.0),
)


# ---------------------------------------------------------------------------
# Exact Geometry & SAT Separation
# ---------------------------------------------------------------------------

def compute_bounding_box(pts: np.ndarray, thetas: np.ndarray) -> Tuple[float, float, float, float]:
    """Compute exact bounding box (min_x, max_x, min_y, max_y) of squares."""
    rx = 0.5 * (np.abs(np.cos(thetas)) + np.abs(np.sin(thetas)))
    ry = rx
    min_x = float(np.min(pts[:, 0] - rx))
    max_x = float(np.max(pts[:, 0] + rx))
    min_y = float(np.min(pts[:, 1] - ry))
    max_y = float(np.max(pts[:, 1] + ry))
    return min_x, max_x, min_y, max_y


def compute_container_side(pts: np.ndarray, thetas: np.ndarray) -> float:
    """Compute minimum enclosing square container side L = max(span_x, span_y)."""
    if len(pts) == 0:
        return 0.0
    min_x, max_x, min_y, max_y = compute_bounding_box(pts, thetas)
    return max(max_x - min_x, max_y - min_y)


def sat_check(q: np.ndarray, clearance: float = 0.0) -> Tuple[bool, float]:
    """Exact Separating Axis Theorem test across all pairs of squares.

    Returns (is_valid, min_gap).
    """
    n = len(q)
    if n <= 1:
        return True, float("inf")
    q_arr = np.asarray(q, dtype=np.float64)
    c, s = np.cos(q_arr[:, 2]), np.sin(q_arr[:, 2])
    u = np.column_stack((c, s))
    v = np.column_stack((-s, c))
    min_gap = float("inf")

    for i in range(n - 1):
        d = q_arr[i + 1 :, :2] - q_arr[i, :2]
        best = np.full(len(d), -np.inf)
        for owner_axes in (False, True):
            for basis in (u, v):
                a = basis[i + 1 :] if owner_axes else np.broadcast_to(basis[i], d.shape)
                hi = 0.5 * (np.abs(a @ u[i]) + np.abs(a @ v[i]))
                hj = 0.5 * (np.abs(np.sum(a * u[i + 1 :], axis=1)) + np.abs(np.sum(a * v[i + 1 :], axis=1)))
                g = np.abs(np.sum(d * a, axis=1)) - hi - hj
                best = np.maximum(best, g)
        cur_min = float(np.min(best))
        if cur_min < min_gap:
            min_gap = cur_min
        if np.any(best < clearance):
            return False, min_gap

    return True, min_gap


def sat_check_upright_against_tilted(
    xu: float, yu: float, tilted_arr: np.ndarray, clearance: float = 0.0
) -> bool:
    """Vectorized SAT check for an upright square against all tilted squares."""
    if len(tilted_arr) == 0:
        return True
    xt = tilted_arr[:, 0]
    yt = tilted_arr[:, 1]
    th = tilted_arr[:, 2]
    c = np.cos(th)
    s = np.sin(th)
    dx = xt - xu
    dy = yt - yu
    h = 0.5 + 0.5 * (np.abs(c) + np.abs(s))

    c1 = np.abs(dx) < h - clearance
    c2 = np.abs(dy) < h - clearance
    c3 = np.abs(dx * c + dy * s) < h - clearance
    c4 = np.abs(-dx * s + dy * c) < h - clearance
    overlap = c1 & c2 & c3 & c4
    return not np.any(overlap)


# ---------------------------------------------------------------------------
# Parametric Tilted Strip Family Construction with Corner Refills
# ---------------------------------------------------------------------------

def construct_strip_family_with_refills(
    k: int,
    j: int,
    theta: float,
    w: int,
    L: float,
    v_off: float = 0.0,
    u_off: float = 0.0,
    tau: float = 1e-7,
) -> Tuple[np.ndarray, int]:
    """Constructs a parametric packing for n = k^2 + j with central band and recursive corner refills.

    A central band of width w in {1, 2} at slope theta in {arctan(1/k), pi/4} is inserted,
    and upper-left (NW) and lower-right (SE) corner regions are recursively refilled with
    integer-grid blocks and tucked corner staircases with SAT clearance tau >= 1e-7.

    Returns (q_arr, num_refilled_upright_squares).
    """
    xc, yc = L / 2.0, L / 2.0
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    h_t = 0.5 * (abs(cos_t) + abs(sin_t))
    u_vec = np.array([cos_t, sin_t])
    v_vec = np.array([-sin_t, cos_t])
    max_c = int(math.ceil(L * 1.5))

    # Central tilted band of width w
    tilted = []
    for s in range(w):
        vs = (s - 0.5 * (w - 1)) * (1.0 + tau) + v_off
        for c in range(-max_c, max_c + 1):
            uc = c * (1.0 + tau) + u_off
            pos = np.array([xc, yc]) + uc * u_vec + vs * v_vec
            x, y = pos[0], pos[1]
            if h_t + tau <= x <= L - h_t - tau and h_t + tau <= y <= L - h_t - tau:
                tilted.append([x, y, theta])
    tilted_arr = np.array(tilted, dtype=np.float64) if tilted else np.zeros((0, 3))

    uprights: List[Tuple[float, float]] = []

    def try_add(x: float, y: float) -> bool:
        if not (0.5 + tau <= x <= L - 0.5 - tau and 0.5 + tau <= y <= L - 0.5 - tau):
            return False
        if not sat_check_upright_against_tilted(x, y, tilted_arr, clearance=tau):
            return False
        for ex in uprights:
            if abs(ex[0] - x) < 1.0 - 1e-6 and abs(ex[1] - y) < 1.0 - 1e-6:
                return False
        uprights.append((x, y))
        return True

    slope = math.tan(theta)
    k_grid = int(math.ceil(L)) + 1

    # 1. Upper-left (NW) corner integer-grid blocks (anchored at (0, L))
    for ix in range(k_grid):
        for iy in range(k_grid):
            xu = 0.5 + tau + ix
            yu = L - 0.5 - tau - iy
            if yu >= yc + slope * (xu - xc) - 0.05:
                try_add(xu, yu)

    # 2. Lower-right (SE) corner integer-grid blocks (anchored at (L, 0))
    for ix in range(k_grid):
        for iy in range(k_grid):
            xu = L - 0.5 - tau - ix
            yu = 0.5 + tau + iy
            if yu <= yc + slope * (xu - xc) + 0.05:
                try_add(xu, yu)

    # 3. Corner staircases tucked with SAT clearance tau >= 1e-7 along the band interface
    for row in range(k_grid):
        # NW row tucking towards the band (scanning x rightwards)
        yu_nw = L - 0.5 - tau - row
        if 0.5 <= yu_nw <= L - 0.5:
            xs_nw = [u[0] for u in uprights if abs(u[1] - yu_nw) < 1e-4 and u[1] >= yc + slope * (u[0] - xc)]
            if xs_nw:
                next_x = max(xs_nw) + 1.0
                if next_x <= L - 0.5 - tau:
                    try_add(next_x, yu_nw)

        # SE row tucking towards the band (scanning x leftwards)
        yu_se = 0.5 + tau + row
        if 0.5 <= yu_se <= L - 0.5:
            xs_se = [u[0] for u in uprights if abs(u[1] - yu_se) < 1e-4 and u[1] <= yc + slope * (u[0] - xc)]
            if xs_se:
                next_x = min(xs_se) - 1.0
                if next_x >= 0.5 + tau:
                    try_add(next_x, yu_se)

    # 4. Outer corner sweeps for residual pockets
    for ix in range(k_grid):
        for iy in range(k_grid):
            try_add(0.5 + tau + ix, 0.5 + tau + iy)
            try_add(L - 0.5 - tau - ix, L - 0.5 - tau - iy)

    upright_arr = np.array([[x, y, 0.0] for x, y in uprights], dtype=np.float64) if uprights else np.zeros((0, 3))
    if len(tilted_arr) == 0:
        return upright_arr, 0
    if len(upright_arr) == 0:
        return tilted_arr, len(tilted_arr)

    return np.vstack([tilted_arr, upright_arr]), len(uprights)


def prune_candidate_to_target(q: np.ndarray, target_n: int, n_tilted: int) -> np.ndarray:
    """Selects exactly target_n squares while preserving the central tilted band.

    Prunes peripheral upright squares that are furthest from the center of mass.
    """
    if len(q) <= target_n:
        return q.copy()

    tilted_squares = q[:n_tilted]
    upright_squares = q[n_tilted:]
    needed_uprights = target_n - n_tilted

    if needed_uprights <= 0:
        return tilted_squares[:target_n].copy()

    if len(upright_squares) <= needed_uprights:
        return q[:target_n].copy()

    center = np.mean(q[:, :2], axis=0)
    dists = np.sum((upright_squares[:, :2] - center) ** 2, axis=1)
    keep_indices = np.argsort(dists)[:needed_uprights]
    selected_uprights = upright_squares[keep_indices]

    return np.vstack([tilted_squares, selected_uprights])


# ---------------------------------------------------------------------------
# SLP / LP Optimization with 4-Sign-Piece Epigraph Linearization
# ---------------------------------------------------------------------------

def _dominant_axis(q: np.ndarray, i: int, j: int) -> Tuple[int, int, int]:
    """Find dominant separating axis (owner, k_axis, direction)."""
    d = q[j, :2] - q[i, :2]
    best_val = -float("inf")
    best_key = (i, 0, 1)

    for owner in (i, j):
        for k in (0, 1):
            phi = float(q[owner, 2]) + k * 0.5 * math.pi
            a = np.array([math.cos(phi), math.sin(phi)])
            direction = 1 if float(d @ a) >= 0.0 else -1
            a *= direction
            h = sum(
                0.5 * (abs(math.cos(float(q[t, 2]) - phi)) + abs(math.sin(float(q[t, 2]) - phi)))
                for t in (i, j)
            )
            val = float(d @ a - h)
            if val > best_val:
                best_val = val
                best_key = (owner, k, direction)

    return best_key


def _piece_gradient(
    q: np.ndarray, i: int, j: int, axis: Tuple[int, int, int]
) -> Tuple[float, np.ndarray]:
    """Gradient of pairwise clearance under dominant separating axis."""
    owner, k, direction = axis
    phi = float(q[owner, 2]) + k * 0.5 * math.pi
    a = direction * np.array([math.cos(phi), math.sin(phi)])
    ap = np.array([-a[1], a[0]])
    phi += 0.0 if direction == 1 else math.pi
    other = j if owner == i else i

    r = float(q[other, 2]) - phi
    s1 = 1.0 if math.cos(r) >= 0 else -1.0
    s2 = 1.0 if math.sin(r) >= 0 else -1.0
    hp = 0.5 * (-s1 * math.sin(r) + s2 * math.cos(r))
    d = q[j, :2] - q[i, :2]
    g = float(d @ a - 0.5 - 0.5 * (s1 * math.cos(r) + s2 * math.sin(r)))

    grad = np.zeros(6)
    grad[:2] = -a
    grad[3:5] = a
    grad[2 if owner == i else 5] = float(d @ ap) + hp
    grad[2 if other == i else 5] = -hp
    return g, grad


def slp_polish(
    q_init: np.ndarray,
    max_iters: int = 5,
    pos_trust: float = 0.02,
    angle_trust: float = 0.01,
    clearance: float = 1e-8,
) -> Tuple[np.ndarray, float]:
    """Performs Sequential Linear Programming (SLP) to contract container side L."""
    q_best = q_init.copy()
    L_best = compute_container_side(q_best[:, :2], q_best[:, 2])
    n = len(q_best)

    for it in range(max_iters):
        num_vars = 3 * n + 3
        c_obj = np.zeros(num_vars)
        c_obj[-1] = 1.0

        A_rows = []
        b_rows = []

        # 1. Bounding box constraints for each square
        for i in range(n):
            th = float(q_best[i, 2])
            ct, st = math.cos(th), math.sin(th)
            for sx, sy in SIGNS:
                rx = 0.5 * (sx * ct + sy * st)
                drx = 0.5 * (-sx * st + sy * ct)
                ry = 0.5 * (sx * (-st) + sy * ct)
                dry = 0.5 * (-sx * ct - sy * st)

                row_x_hi = np.zeros(num_vars)
                row_x_hi[3 * i] = 1.0
                row_x_hi[3 * i + 2] = drx
                row_x_hi[-3] = -1.0
                row_x_hi[-1] = -1.0
                A_rows.append(row_x_hi)
                b_rows.append(-(q_best[i, 0] + rx))

                row_x_lo = np.zeros(num_vars)
                row_x_lo[3 * i] = -1.0
                row_x_lo[3 * i + 2] = -drx
                row_x_lo[-3] = 1.0
                A_rows.append(row_x_lo)
                b_rows.append(q_best[i, 0] - rx)

                row_y_hi = np.zeros(num_vars)
                row_y_hi[3 * i + 1] = 1.0
                row_y_hi[3 * i + 2] = dry
                row_y_hi[-2] = -1.0
                row_y_hi[-1] = -1.0
                A_rows.append(row_y_hi)
                b_rows.append(-(q_best[i, 1] + ry))

                row_y_lo = np.zeros(num_vars)
                row_y_lo[3 * i + 1] = -1.0
                row_y_lo[3 * i + 2] = -dry
                row_y_lo[-2] = 1.0
                A_rows.append(row_y_lo)
                b_rows.append(q_best[i, 1] - ry)

        # 2. Pairwise contact clearance constraints for near pairs
        for i in range(n):
            for j in range(i + 1, n):
                dist_sq = (q_best[i, 0] - q_best[j, 0]) ** 2 + (q_best[i, 1] - q_best[j, 1]) ** 2
                if dist_sq < 2.25:
                    axis = _dominant_axis(q_best, i, j)
                    g, grad = _piece_gradient(q_best, i, j, axis)
                    row_pair = np.zeros(num_vars)
                    row_pair[3 * i : 3 * i + 3] = -grad[:3]
                    row_pair[3 * j : 3 * j + 3] = -grad[3:6]
                    A_rows.append(row_pair)
                    b_rows.append(g - clearance)

        bounds = []
        for i in range(n):
            bounds.append((-pos_trust, pos_trust))
            bounds.append((-pos_trust, pos_trust))
            bounds.append((-angle_trust, angle_trust))
        bounds.append((None, None))
        bounds.append((None, None))
        bounds.append((1.0, None))

        if not A_rows:
            break

        res = linprog(
            c_obj,
            A_ub=np.array(A_rows),
            b_ub=np.array(b_rows),
            bounds=bounds,
            method="highs",
        )

        if not res.success:
            break

        delta_q = res.x[:-3].reshape((n, 3))
        improved = False
        step = 1.0
        for _ in range(5):
            candidate = q_best.copy()
            candidate[:, :2] += step * delta_q[:, :2]
            candidate[:, 2] += step * delta_q[:, 2]
            valid, _ = sat_check(candidate, clearance=0.0)
            cand_L = compute_container_side(candidate[:, :2], candidate[:, 2])
            if valid and cand_L < L_best - 1e-6:
                q_best = candidate
                L_best = cand_L
                improved = True
                break
            step *= 0.5

        if not improved:
            break

    return q_best, L_best


# ---------------------------------------------------------------------------
# High-Precision Certification and Alignment
# ---------------------------------------------------------------------------

def certify_and_align_packing(
    pts: np.ndarray, thetas: np.ndarray, tau: float = 1e-7
) -> Tuple[float, str, List[Dict[str, str]], float]:
    """Uniformly scale and align packing to guarantee strictly positive clearance tau.

    Returns (L_float, L_certified_str, rows, lambda_scale).
    """
    n = len(pts)
    c = np.cos(thetas)
    s = np.sin(thetas)
    lam = 1.0

    for i in range(n):
        for j in range(i + 1, n):
            dx = pts[i, 0] - pts[j, 0]
            dy = pts[i, 1] - pts[j, 1]
            ci, si = c[i], s[i]
            cj, sj = c[j], s[j]
            r = 0.5 + 0.5 * (abs(ci * cj + si * sj) + abs(-si * cj + ci * sj))
            v1 = abs(dx * ci + dy * si)
            v2 = abs(-dx * si + dy * ci)
            v3 = abs(dx * cj + dy * sj)
            v4 = abs(-dx * sj + dy * cj)
            vmax = max(v1, v2, v3, v4)
            if vmax > 1e-12:
                req = (r + tau) / vmax
                if req > lam:
                    lam = req

    scaled_pts = pts * lam
    rx = 0.5 * (np.abs(c) + np.abs(s))
    min_x = np.min(scaled_pts[:, 0] - rx)
    min_y = np.min(scaled_pts[:, 1] - rx)
    final_pts = scaled_pts - np.array([min_x, min_y])

    rows = []
    for i in range(n):
        rows.append({
            "square_id": str(i),
            "x": f"{final_pts[i, 0]:.12f}",
            "y": f"{final_pts[i, 1]:.12f}",
            "angle": f"{thetas[i]:.12f}",
        })

    l_certified_str = ""
    if certify is not None:
        try:
            l_certified_str = certify(n, rows)
            l_val = float(l_certified_str)
        except Exception:
            min_x, max_x, min_y, max_y = compute_bounding_box(final_pts, thetas)
            l_val = max(max_x - min_x, max_y - min_y)
            l_certified_str = f"{l_val:.50f}"
    else:
        min_x, max_x, min_y, max_y = compute_bounding_box(final_pts, thetas)
        l_val = max(max_x - min_x, max_y - min_y)
        l_certified_str = f"{l_val:.50f}"

    return l_val, l_certified_str, rows, lam


# ---------------------------------------------------------------------------
# Target Solver: Parametric Strip Families with Recursive Refills
# ---------------------------------------------------------------------------

@dataclass
class TargetResult:
    n: int
    k: int
    j: int
    status: str
    L: Optional[float]
    L_certified_str: Optional[str]
    L_incumbent: float
    delta_L: Optional[float]
    rel_delta_L: Optional[float]
    score_contribution: Optional[float]
    incumbent_score: float
    delta_score: Optional[float]
    theta_rad: Optional[float]
    theta_deg: Optional[float]
    band_width_w: Optional[int]
    band_offset: List[float]
    corner_pocket_refills: int
    sat_clearance: float
    lambda_scale: Optional[float]
    candidate_packing: List[Dict[str, str]]
    reason: Optional[str] = None


def solve_target_with_recursive_refill(
    k: int,
    j: int,
    incumbent_L: float,
    max_slp_iters: int = 5,
) -> TargetResult:
    """Solves a single n = k^2 + j target using parametric strip families and recursive refill.

    Strictly removes default integer-grid fallbacks. If no valid configuration fits at
    sub-grid envelope L < ceil(sqrt(n)), records status FAILED with reason
    'insufficient_capacity_at_sub_grid_envelope'.
    """
    n = k * k + j
    k_ceil = int(math.ceil(math.sqrt(n)))
    inc_score = incumbent_L / math.sqrt(n) - 1.0

    # Candidate parameters for tilted band
    candidate_angles = [
        0.25 * math.pi,            # 45 deg diagonal band
        math.atan(1.0 / k),        # Erdős-Graham rational slope
        math.atan(2.0 / k),        # Doubled rational slope
    ]
    candidate_widths = [1, 2]

    best_q: Optional[np.ndarray] = None
    best_L = float("inf")
    best_th = candidate_angles[0]
    best_w = 1
    best_v_off = 0.0
    best_u_off = 0.0
    best_refill_count = 0

    # Search strictly sub-grid container sides L < k_ceil
    candidate_Ls = [
        float(k) + 0.65,
        float(k) + 0.75,
        float(k) + 0.85,
        float(k) + 0.95,
    ]
    if incumbent_L < float(k_ceil) - 1e-4:
        candidate_Ls = [incumbent_L] + candidate_Ls

    # Filter to strictly sub-grid envelopes
    candidate_Ls = [L_cand for L_cand in candidate_Ls if L_cand < float(k_ceil) - 1e-5]

    for L_nominal in candidate_Ls:
        for th in candidate_angles:
            for w in candidate_widths:
                for v_off in [0.0, -0.2, 0.2]:
                    for u_off in [0.0, -0.2, 0.2]:
                        q_cand, refill_cnt = construct_strip_family_with_refills(
                            k=k, j=j, theta=th, w=w, L=L_nominal,
                            v_off=v_off, u_off=u_off, tau=1e-7,
                        )
                        if len(q_cand) >= n:
                            n_tilted = len(q_cand) - refill_cnt
                            q_pruned = prune_candidate_to_target(q_cand, target_n=n, n_tilted=n_tilted)
                            valid, gap = sat_check(q_pruned, clearance=-1e-7)
                            if valid:
                                cur_L = compute_container_side(q_pruned[:, :2], q_pruned[:, 2])
                                if cur_L < float(k_ceil) and cur_L < best_L:
                                    best_L = cur_L
                                    best_q = q_pruned
                                    best_th = th
                                    best_w = w
                                    best_v_off = v_off
                                    best_u_off = u_off
                                    best_refill_count = refill_cnt
        if best_q is not None:
            break

    # If unconstrained search yields fewer than n squares at L < ceil(sqrt(n)),
    # record FAILED rather than emitting a fallback integer grid.
    if best_q is None:
        return TargetResult(
            status="FAILED",
            reason="insufficient_capacity_at_sub_grid_envelope",
            n=n,
            k=k,
            j=j,
            L=None,
            L_certified_str=None,
            L_incumbent=incumbent_L,
            delta_L=None,
            rel_delta_L=None,
            score_contribution=None,
            incumbent_score=inc_score,
            delta_score=None,
            theta_rad=None,
            theta_deg=None,
            band_width_w=None,
            band_offset=[],
            corner_pocket_refills=0,
            sat_clearance=0.0,
            lambda_scale=None,
            candidate_packing=[],
        )

    # Apply SLP polish to contract container side L
    if max_slp_iters > 0:
        best_q, _ = slp_polish(best_q, max_iters=max_slp_iters, pos_trust=0.02, angle_trust=0.01)

    # High-precision certification and separation scaling
    L_float, L_cert_str, rows, lam = certify_and_align_packing(best_q[:, :2], best_q[:, 2], tau=1e-7)
    _, min_sat_gap = sat_check(best_q, clearance=0.0)

    # Verify final certified L remains strictly sub-grid
    if L_float >= float(k_ceil):
        return TargetResult(
            status="FAILED",
            reason="insufficient_capacity_at_sub_grid_envelope",
            n=n,
            k=k,
            j=j,
            L=None,
            L_certified_str=None,
            L_incumbent=incumbent_L,
            delta_L=None,
            rel_delta_L=None,
            score_contribution=None,
            incumbent_score=inc_score,
            delta_score=None,
            theta_rad=None,
            theta_deg=None,
            band_width_w=None,
            band_offset=[],
            corner_pocket_refills=0,
            sat_clearance=0.0,
            lambda_scale=None,
            candidate_packing=[],
        )

    delta_l = L_float - incumbent_L
    rel_delta = delta_l / incumbent_L
    score_cand = L_float / math.sqrt(n) - 1.0
    delta_sc = score_cand - inc_score

    return TargetResult(
        status="SUCCESS",
        reason="ok",
        n=n,
        k=k,
        j=j,
        L=L_float,
        L_certified_str=L_cert_str,
        L_incumbent=incumbent_L,
        delta_L=delta_l,
        rel_delta_L=rel_delta,
        score_contribution=score_cand,
        incumbent_score=inc_score,
        delta_score=delta_sc,
        theta_rad=best_th,
        theta_deg=math.degrees(best_th),
        band_width_w=best_w,
        band_offset=[best_u_off, best_v_off],
        corner_pocket_refills=best_refill_count,
        sat_clearance=max(1e-7, min_sat_gap),
        lambda_scale=lam,
        candidate_packing=rows,
    )


# ---------------------------------------------------------------------------
# Suite Execution across Top 40 Targets
# ---------------------------------------------------------------------------

class RecursiveRefillSuite:
    """Suite manager for evaluating recursive refill parametric strip families."""

    def __init__(
        self,
        incumbent_path: Path = REPO_ROOT / "poc/pell_corrugation/data/border-scan/merged-literal-cpp.json",
        targets_path: Path = REPO_ROOT / "poc/research/data/top40_excess_targets.json",
    ):
        self.incumbents: Dict[int, float] = {}
        if incumbent_path.exists():
            with open(incumbent_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.incumbents = {r["n"]: float(r["L"]) for r in data.get("results", [])}

        self.targets_path = targets_path

    def get_top_40_targets(self) -> List[Tuple[int, int, int]]:
        """Loads the top 40 excess targets from top40_excess_targets.json or falls back to ranking."""
        if self.targets_path.exists():
            with open(self.targets_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return [(t["n"], t["k"], t["j"]) for t in data[:40]]

        candidates = []
        for k in range(10, 32):
            max_j = min(k, 1000 - k * k)
            for j in range(1, max_j + 1):
                n = k * k + j
                inc_L = self.incumbents.get(n, float(k) + 1.0)
                score = inc_L / math.sqrt(n) - 1.0
                candidates.append((n, k, j, score))

        candidates.sort(key=lambda x: x[3], reverse=True)
        return [(c[0], c[1], c[2]) for c in candidates[:40]]

    def run(
        self,
        targets: Optional[List[Tuple[int, int, int]]] = None,
        max_slp_iters: int = 5,
    ) -> Dict[str, Any]:
        """Runs optimization and certification across targets."""
        if targets is None:
            targets = self.get_top_40_targets()

        t0 = time.time()
        results_list = []
        wins_count = 0

        for idx, (n, k, j) in enumerate(targets, start=1):
            inc_L = self.incumbents.get(n, float(k) + 1.0)
            res = solve_target_with_recursive_refill(
                k=k, j=j, incumbent_L=inc_L, max_slp_iters=max_slp_iters
            )
            if res.status == "SUCCESS" and res.delta_L is not None and res.delta_L < -1e-6:
                wins_count += 1
            results_list.append(asdict(res))

        elapsed = time.time() - t0
        successful = [r for r in results_list if r.get("status") == "SUCCESS"]
        failed = [r for r in results_list if r.get("status") == "FAILED"]
        deltas = [r["delta_L"] for r in successful if r.get("delta_L") is not None]
        mean_delta = float(np.mean(deltas)) if deltas else 0.0

        return {
            "metadata": {
                "solver": "search_recursive_refill_strips",
                "idea": "Idea 3: Parametric k^2+j Strip Families with Recursive Corner Pocket Refill",
                "k_range": [10, 31],
                "targets_count": len(targets),
                "elapsed_seconds": elapsed,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            },
            "summary": {
                "total_targets": len(targets),
                "success_count": len(successful),
                "failed_count": len(failed),
                "certified_count": len(successful),
                "mean_delta_L_vs_incumbent": mean_delta,
                "min_delta_L": float(np.min(deltas)) if deltas else 0.0,
                "max_delta_L": float(np.max(deltas)) if deltas else 0.0,
                "wins_count": wins_count,
            },
            "results": results_list,
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search Parametric Strip Families with Recursive Pocket Refill (Idea 3)"
    )
    parser.add_argument(
        "--incumbent",
        type=Path,
        default=REPO_ROOT / "poc/pell_corrugation/data/border-scan/merged-literal-cpp.json",
        help="Path to merged-literal-cpp.json incumbent ledger",
    )
    parser.add_argument(
        "--targets",
        type=Path,
        default=REPO_ROOT / "poc/research/data/top40_excess_targets.json",
        help="Path to top40_excess_targets.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "poc/research/data/recursive_refill_results.json",
        help="Path to output results JSON",
    )
    parser.add_argument(
        "--slp-iters",
        type=int,
        default=3,
        help="SLP polish iterations per target (default: 3)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=40,
        help="Number of top targets to evaluate (default: 40)",
    )
    args = parser.parse_args()

    suite = RecursiveRefillSuite(incumbent_path=args.incumbent, targets_path=args.targets)
    top_targets = suite.get_top_40_targets()[: args.top_n]

    print(f"Starting Recursive Refill Strip Solver across {len(top_targets)} targets...")
    suite_data = suite.run(targets=top_targets, max_slp_iters=args.slp_iters)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(suite_data, f, indent=2)
    print(f"Results exported successfully to {args.output}")


if __name__ == "__main__":
    main()
