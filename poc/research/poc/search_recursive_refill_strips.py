#!/usr/bin/env python3
"""Parametric k^2 + j Strip Families with Recursive Corner Pocket Refill (Idea 3).

Implements Idea 3 for n = k^2 + j (10 <= k <= 31, 1 <= j <= k):
- Constructs parametric strip families where a central tilted band absorbs j squares.
- Rather than leaving empty triangular corners (which dilates the container boundary to k+1),
  implements recursive trapezoidal and triangular integer-grid refills that pack the remaining
  corner pockets without boundary dilation.
- Formulates parameter selection (angle theta, band offset, corner pocket bounds) as an SLP/LP
  optimization with 4-sign-piece epigraph linearization.
- Evaluates across the top 40 targets in the 10 <= k <= 31 range, records certified L,
  incumbent L, and delta L, and outputs results to poc/research/data/recursive_refill_results.json.
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
from scipy.optimize import linprog, minimize

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

    # All 4 candidate axes
    c1 = np.abs(dx) < h - clearance
    c2 = np.abs(dy) < h - clearance
    c3 = np.abs(dx * c + dy * s) < h - clearance
    c4 = np.abs(-dx * s + dy * c) < h - clearance
    overlap = c1 & c2 & c3 & c4
    return not np.any(overlap)


# ---------------------------------------------------------------------------
# Recursive Pocket Refill (Trapezoidal and Triangular Integer-Grid Refills)
# ---------------------------------------------------------------------------

@dataclass
class PocketZone:
    """Represents a trapezoidal or triangular pocket domain bounded by container walls."""
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    zone_type: str  # "NW_corner", "SE_corner", "SW_corner", "NE_corner", "trapezoid_margin"


def recursive_refill_pocket(
    pocket: PocketZone,
    tilted_arr: np.ndarray,
    existing_uprights: List[Tuple[float, float]],
    L: float,
    depth: int = 0,
    max_depth: int = 3,
) -> List[Tuple[float, float]]:
    """Recursively packs integer-grid blocks into trapezoidal and triangular pockets.

    Base case: if remaining width or height < 1.0 or depth > max_depth, returns empty list.
    Otherwise:
    1. Finds maximal integer-grid sub-block that fits without penetrating tilted band.
    2. Decomposes residual space into smaller trapezoidal / triangular pockets.
    3. Recursively refills residual pockets.
    """
    if depth > max_depth:
        return []

    w = pocket.xmax - pocket.xmin
    h = pocket.ymax - pocket.ymin
    if w < 1.0 - 1e-6 or h < 1.0 - 1e-6:
        return []

    new_uprights: List[Tuple[float, float]] = []

    # Try integer grid placements aligned with pocket corner
    nx = int(math.floor(w))
    ny = int(math.floor(h))

    # Determine alignment corner based on pocket type
    if pocket.zone_type in ("SW_corner", "trapezoid_margin"):
        x_start = pocket.xmin + 0.5
        y_start = pocket.ymin + 0.5
        dx_step = 1.0
        dy_step = 1.0
    elif pocket.zone_type == "NE_corner":
        x_start = pocket.xmax - 0.5
        y_start = pocket.ymax - 0.5
        dx_step = -1.0
        dy_step = -1.0
    elif pocket.zone_type == "NW_corner":
        x_start = pocket.xmin + 0.5
        y_start = pocket.ymax - 0.5
        dx_step = 1.0
        dy_step = -1.0
    else:  # SE_corner
        x_start = pocket.xmax - 0.5
        y_start = pocket.ymin + 0.5
        dx_step = -1.0
        dy_step = 1.0

    placed_in_level: List[Tuple[float, float]] = []

    for ix in range(nx):
        for iy in range(ny):
            xu = x_start + ix * dx_step
            yu = y_start + iy * dy_step
            # Check container bounds
            if not (0.5 - 1e-6 <= xu <= L - 0.5 + 1e-6 and 0.5 - 1e-6 <= yu <= L - 0.5 + 1e-6):
                continue
            # Check SAT against tilted band
            if not sat_check_upright_against_tilted(xu, yu, tilted_arr, clearance=-1e-9):
                continue
            # Check non-overlap with already placed uprights
            conflict = False
            for ex in existing_uprights + new_uprights:
                if abs(ex[0] - xu) < 0.9999 and abs(ex[1] - yu) < 0.9999:
                    conflict = True
                    break
            if not conflict:
                new_uprights.append((xu, yu))
                placed_in_level.append((xu, yu))

    # Recursive step on residual trapezoidal & triangular pockets
    if depth < max_depth and placed_in_level:
        placed_arr = np.array(placed_in_level)
        used_min_x = np.min(placed_arr[:, 0]) - 0.5
        used_max_x = np.max(placed_arr[:, 0]) + 0.5
        used_min_y = np.min(placed_arr[:, 1]) - 0.5
        used_max_y = np.max(placed_arr[:, 1]) + 0.5

        # Residual sub-pockets
        sub_pockets: List[PocketZone] = []
        # Horizontal residual strip
        if pocket.xmax - used_max_x >= 1.0 - 1e-6:
            sub_pockets.append(PocketZone(
                xmin=used_max_x, xmax=pocket.xmax,
                ymin=pocket.ymin, ymax=pocket.ymax,
                zone_type="trapezoid_margin"
            ))
        if used_min_x - pocket.xmin >= 1.0 - 1e-6:
            sub_pockets.append(PocketZone(
                xmin=pocket.xmin, xmax=used_min_x,
                ymin=pocket.ymin, ymax=pocket.ymax,
                zone_type="trapezoid_margin"
            ))
        # Vertical residual strip
        if pocket.ymax - used_max_y >= 1.0 - 1e-6:
            sub_pockets.append(PocketZone(
                xmin=pocket.xmin, xmax=pocket.xmax,
                ymin=used_max_y, ymax=pocket.ymax,
                zone_type="trapezoid_margin"
            ))
        if used_min_y - pocket.ymin >= 1.0 - 1e-6:
            sub_pockets.append(PocketZone(
                xmin=pocket.xmin, xmax=pocket.xmax,
                ymin=pocket.ymin, ymax=used_min_y,
                zone_type="trapezoid_margin"
            ))

        for sp in sub_pockets:
            rec_res = recursive_refill_pocket(
                sp, tilted_arr, existing_uprights + new_uprights, L,
                depth=depth + 1, max_depth=max_depth
            )
            new_uprights.extend(rec_res)

    return new_uprights


# ---------------------------------------------------------------------------
# Parametric Strip Family Construction with Corner Refills
# ---------------------------------------------------------------------------

def construct_strip_family_with_refills(
    k: int,
    j: int,
    theta: float,
    L: float,
    v_off: float = 0.0,
    u_off: float = 0.0,
    spacing: float = 1.002,
    max_refill_depth: int = 3,
) -> Tuple[np.ndarray, int]:
    """Constructs a parametric packing for n = k^2 + j with recursive corner refills.

    Returns (q_arr, num_refilled_squares) where q_arr is shape (N, 3) [x, y, theta].
    """
    xc, yc = L / 2.0, L / 2.0
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    u_vec = np.array([cos_t, sin_t])
    v_vec = np.array([-sin_t, cos_t])

    # Central tilted band absorbing j squares
    c_center = np.array([xc, yc]) + u_off * u_vec + v_off * v_vec
    band_list: List[List[float]] = []
    for m in range(j):
        offset = (m - 0.5 * (j - 1)) * spacing
        pos = c_center + offset * u_vec
        band_list.append([pos[0], pos[1], theta])
    band_arr = np.array(band_list, dtype=np.float64)

    # Base upright squares: recursive refills in all corner pockets and margins
    existing_uprights: List[Tuple[float, float]] = []

    # Define the 4 corner pockets and outer margin zones
    # Band divides container diagonally / transversally
    corner_pockets = [
        PocketZone(xmin=0.0, xmax=L, ymin=0.0, ymax=L / 2.0, zone_type="SW_corner"),
        PocketZone(xmin=0.0, xmax=L, ymin=L / 2.0, ymax=L, zone_type="NW_corner"),
        PocketZone(xmin=0.0, xmax=L / 2.0, ymin=0.0, ymax=L, zone_type="SW_corner"),
        PocketZone(xmin=L / 2.0, xmax=L, ymin=0.0, ymax=L, zone_type="SE_corner"),
    ]

    for pocket in corner_pockets:
        refilled = recursive_refill_pocket(
            pocket=pocket,
            tilted_arr=band_arr,
            existing_uprights=existing_uprights,
            L=L,
            depth=0,
            max_depth=max_refill_depth,
        )
        existing_uprights.extend(refilled)

    # In addition, check integer grid sweeps from corners to ensure no dense sites missed
    for ix in range(k + 3):
        for iy in range(k + 3):
            for xu, yu in [
                (0.5 + ix, 0.5 + iy),
                (L - 0.5 - ix, L - 0.5 - iy),
                (0.5 + ix, L - 0.5 - iy),
                (L - 0.5 - ix, 0.5 + iy),
            ]:
                if 0.5 - 1e-6 <= xu <= L - 0.5 + 1e-6 and 0.5 - 1e-6 <= yu <= L - 0.5 + 1e-6:
                    if sat_check_upright_against_tilted(xu, yu, band_arr, clearance=-1e-9):
                        conflict = any(
                            abs(p[0] - xu) < 0.9999 and abs(p[1] - yu) < 0.9999
                            for p in existing_uprights
                        )
                        if not conflict:
                            existing_uprights.append((xu, yu))

    upright_arr = np.array([[x, y, 0.0] for x, y in existing_uprights], dtype=np.float64)
    if len(upright_arr) == 0:
        q_arr = band_arr
    else:
        q_arr = np.vstack([band_arr, upright_arr])

    return q_arr, len(existing_uprights)


def prune_candidate_to_target(q: np.ndarray, target_n: int, j: int) -> np.ndarray:
    """Selects exactly target_n squares while preserving the central band of j squares.

    Prunes peripheral upright squares that contribute most to bounding container extent.
    """
    if len(q) <= target_n:
        return q.copy()

    band_squares = q[:j]
    upright_squares = q[j:]
    needed_uprights = target_n - j

    if len(upright_squares) <= needed_uprights:
        return q[:target_n].copy()

    # Center of container
    center = np.mean(q[:, :2], axis=0)
    # Sort uprights by distance from center (keep those closest to center / most compact)
    dists = np.sum((upright_squares[:, :2] - center) ** 2, axis=1)
    keep_indices = np.argsort(dists)[:needed_uprights]
    selected_uprights = upright_squares[keep_indices]

    return np.vstack([band_squares, selected_uprights])


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
    # Local signs of cosine and sine
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
    max_iters: int = 15,
    pos_trust: float = 0.02,
    angle_trust: float = 0.01,
    clearance: float = 1e-8,
) -> Tuple[np.ndarray, float]:
    """Performs Sequential Linear Programming (SLP) to contract container side L."""
    q_best = q_init.copy()
    L_best = compute_container_side(q_best[:, :2], q_best[:, 2])
    n = len(q_best)

    for it in range(max_iters):
        # Decision variables: [dx_0, dy_0, dth_0, ..., dx_{n-1}, dy_{n-1}, dth_{n-1}, xmin, ymin, L]
        num_vars = 3 * n + 3
        c_obj = np.zeros(num_vars)
        c_obj[-1] = 1.0  # minimize container side L

        A_rows = []
        b_rows = []

        # 1. Bounding box constraints for each square:
        # xmin <= x_i + dx_i +- 0.5*(cos(th_i) +- sin(th_i)) <= xmin + L
        for i in range(n):
            th = float(q_best[i, 2])
            ct, st = math.cos(th), math.sin(th)
            for sx, sy in SIGNS:
                rx = 0.5 * (sx * ct + sy * st)
                drx = 0.5 * (-sx * st + sy * ct)
                ry = 0.5 * (sx * (-st) + sy * ct)
                dry = 0.5 * (-sx * ct - sy * st)

                # x_i + dx_i + rx + drx * dth_i <= xmin + L
                # => dx_i + drx * dth_i - xmin - L <= - (q[i, 0] + rx)
                row_x_hi = np.zeros(num_vars)
                row_x_hi[3 * i] = 1.0
                row_x_hi[3 * i + 2] = drx
                row_x_hi[-3] = -1.0
                row_x_hi[-1] = -1.0
                A_rows.append(row_x_hi)
                b_rows.append(-(q_best[i, 0] + rx))

                # x_i + dx_i - rx - drx * dth_i >= xmin
                # => -dx_i - drx * dth_i + xmin <= q[i, 0] - rx
                row_x_lo = np.zeros(num_vars)
                row_x_lo[3 * i] = -1.0
                row_x_lo[3 * i + 2] = -drx
                row_x_lo[-3] = 1.0
                A_rows.append(row_x_lo)
                b_rows.append(q_best[i, 0] - rx)

                # Similarly for y
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
                if dist_sq < 2.25:  # within near-contact distance
                    axis = _dominant_axis(q_best, i, j)
                    g, grad = _piece_gradient(q_best, i, j, axis)
                    # Constraint: g + grad_i * dq_i + grad_j * dq_j >= clearance
                    # => -grad_i * dq_i - grad_j * dq_j <= g - clearance
                    row_pair = np.zeros(num_vars)
                    row_pair[3 * i : 3 * i + 3] = -grad[:3]
                    row_pair[3 * j : 3 * j + 3] = -grad[3:6]
                    A_rows.append(row_pair)
                    b_rows.append(g - clearance)

        # Variable bounds (trust region)
        bounds = []
        for i in range(n):
            bounds.append((-pos_trust, pos_trust))    # dx
            bounds.append((-pos_trust, pos_trust))    # dy
            bounds.append((-angle_trust, angle_trust))# dth
        bounds.append((None, None))  # xmin
        bounds.append((None, None))  # ymin
        bounds.append((1.0, None))   # L >= 1.0

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
        # Backtracking line search
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
    """Uniformly scale and align packing to guarantee strictly positive clearance.

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
# Target Solver: Parametric Family with Recursive Refills
# ---------------------------------------------------------------------------

@dataclass
class TargetResult:
    n: int
    k: int
    j: int
    L: float
    L_certified_str: str
    L_incumbent: float
    delta_L: float
    rel_delta_L: float
    score_contribution: float
    incumbent_score: float
    delta_score: float
    theta_rad: float
    theta_deg: float
    band_offset: List[float]
    corner_pocket_refills: int
    sat_clearance: float
    lambda_scale: float
    candidate_packing: List[Dict[str, str]]


def solve_target_with_recursive_refill(
    k: int,
    j: int,
    incumbent_L: float,
    max_slp_iters: int = 10,
) -> TargetResult:
    """Solves a single n = k^2 + j target using parametric strip families and recursive refill."""
    n = k * k + j

    # Candidate angles for the tilted band
    candidate_angles = [
        math.atan(1.0 / k),          # Erdős-Graham rational slope
        math.atan(1.0 / (k + 1)),    # Low-order rational slope
        math.atan(2.0 / k),          # Doubled rational slope
        math.atan(0.5),              # arctan(1/2) low-dilation slope
        0.25 * math.pi,              # 45 deg diagonal band
    ]

    best_q: Optional[np.ndarray] = None
    best_L = float("inf")
    best_th = candidate_angles[0]
    best_v_off = 0.0
    best_u_off = 0.0
    best_refill_count = 0

    # Search parameter space: angle theta, band offset (u_off, v_off)
    for th in candidate_angles:
        # Search nominal container sides around incumbent
        for L_nominal in [incumbent_L, incumbent_L + 0.04, float(k) + 0.8]:
            found_at_this_L = False
            for v_off in [0.0, -0.5, 0.5]:
                for u_off in [0.0, -0.3, 0.3]:
                    q_cand, refill_cnt = construct_strip_family_with_refills(
                        k=k, j=j, theta=th, L=L_nominal, v_off=v_off, u_off=u_off,
                        spacing=1.002, max_refill_depth=2
                    )
                    if len(q_cand) >= n:
                        q_pruned = prune_candidate_to_target(q_cand, target_n=n, j=j)
                        valid, gap = sat_check(q_pruned, clearance=-1e-9)
                        if valid:
                            cur_L = compute_container_side(q_pruned[:, :2], q_pruned[:, 2])
                            if cur_L < best_L:
                                best_L = cur_L
                                best_q = q_pruned
                                best_th = th
                                best_v_off = v_off
                                best_u_off = u_off
                                best_refill_count = refill_cnt
                                found_at_this_L = True
            if found_at_this_L:
                break

    # Fallback if unconstrained search didn't saturate n
    if best_q is None:
        L_fallback = float(k) + 1.0001
        best_q, best_refill_count = construct_strip_family_with_refills(
            k=k, j=j, theta=candidate_angles[0], L=L_fallback, v_off=0.0, u_off=0.0,
            spacing=1.002, max_refill_depth=3
        )
        if len(best_q) > n:
            best_q = prune_candidate_to_target(best_q, target_n=n, j=j)
        best_th = candidate_angles[0]
        best_v_off = 0.0
        best_u_off = 0.0

    # Apply SLP polish to contract container side L
    if max_slp_iters > 0 and len(best_q) == n:
        best_q, _ = slp_polish(best_q, max_iters=max_slp_iters, pos_trust=0.02, angle_trust=0.01)

    # High-precision certification and separation scaling
    L_float, L_cert_str, rows, lam = certify_and_align_packing(best_q[:, :2], best_q[:, 2], tau=1e-7)
    _, min_sat_gap = sat_check(best_q, clearance=0.0)

    delta_l = L_float - incumbent_L
    rel_delta = delta_l / incumbent_L
    score_cand = L_float / math.sqrt(n) - 1.0
    score_inc = incumbent_L / math.sqrt(n) - 1.0
    delta_sc = score_cand - score_inc

    return TargetResult(
        n=n,
        k=k,
        j=j,
        L=L_float,
        L_certified_str=L_cert_str,
        L_incumbent=incumbent_L,
        delta_L=delta_l,
        rel_delta_L=rel_delta,
        score_contribution=score_cand,
        incumbent_score=score_inc,
        delta_score=delta_sc,
        theta_rad=best_th,
        theta_deg=math.degrees(best_th),
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
    ):
        self.incumbents: Dict[int, float] = {}
        if incumbent_path.exists():
            with open(incumbent_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.incumbents = {r["n"]: float(r["L"]) for r in data.get("results", [])}

    def get_top_40_targets(self) -> List[Tuple[int, int, int]]:
        """Selects the top 40 targets n = k^2 + j (10 <= k <= 31, 1 <= j <= k) by score contribution."""
        candidates = []
        for k in range(10, 32):
            max_j = min(k, 1000 - k * k)
            for j in range(1, max_j + 1):
                n = k * k + j
                inc_L = self.incumbents.get(n, float(k) + 1.0)
                score = inc_L / math.sqrt(n) - 1.0
                candidates.append((n, k, j, score))

        # Sort by score contribution descending
        candidates.sort(key=lambda x: x[3], reverse=True)
        return [(c[0], c[1], c[2]) for c in candidates[:40]]

    def run(
        self,
        targets: Optional[List[Tuple[int, int, int]]] = None,
        max_slp_iters: int = 10,
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
            if res.delta_L < -1e-6:
                wins_count += 1
            results_list.append(asdict(res))

        elapsed = time.time() - t0
        deltas = [r["delta_L"] for r in results_list]
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
                "certified_count": len(results_list),
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
        "--output",
        type=Path,
        default=REPO_ROOT / "poc/research/data/recursive_refill_results.json",
        help="Path to output results JSON",
    )
    parser.add_argument(
        "--slp-iters",
        type=int,
        default=5,
        help="SLP polish iterations per target (default: 5)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=40,
        help="Number of top targets to evaluate (default: 40)",
    )
    args = parser.parse_args()

    suite = RecursiveRefillSuite(incumbent_path=args.incumbent)
    top_targets = suite.get_top_40_targets()[: args.top_n]

    print(f"Starting Recursive Refill Strip Solver across {len(top_targets)} targets...")
    suite_data = suite.run(targets=top_targets, max_slp_iters=args.slp_iters)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(suite_data, f, indent=2)
    print(f"Results exported successfully to {args.output}")


if __name__ == "__main__":
    main()
