"""From-scratch thick tilted band and staircase band packing generator for large n (Idea 1).

Constructs true corner-to-corner thick tilted band packings for excess targets
(n in [100, 1000]) without integer-grid fallbacks.
1. Places a central diagonal band of width w in 1..4 (and w=5 where appropriate)
   at angle theta in [arctan(1/2), pi/4, arctan(2/3), arctan(3/4)] and continuous offsets,
   then packs upright unit squares along the two triangular flanks using dense
   staircase boundary tucking (checking SAT contact against the band rather than crude bounding boxes).
   For theta = pi/4, exact closed-form Göbel staircase band parameters are also evaluated.
2. Enforces hard assertions:
   - Packing must contain at least w * k tilted squares (k = floor(sqrt(n))).
   - Container side L must be strictly less than ceil(sqrt(n)).
   - If the construction cannot place n squares with L < ceil(sqrt(n)), marks the target
     as FAILED (cannot achieve L < ceil(sqrt(n))) instead of substituting an integer grid.
3. Sanity-tested on n=290 to confirm L <= 17.55.
4. Polishes real candidates with 4-sign-piece epigraph LP relaxation and saves results
   to poc/research/data/large_n_scratch_band_results.json.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from itertools import product
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[3]
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
MERGED_LITERAL_PATH = REPO / "poc/pell_corrugation/data/border-scan/merged-literal-cpp.json"
DEFAULT_TARGETS_PATH = DATA_DIR / "top40_excess_targets.json"
DEFAULT_OUTPUT_PATH = DATA_DIR / "large_n_scratch_band_results.json"

SIGNS: tuple[tuple[float, float], ...] = tuple(product((-1.0, 1.0), repeat=2))
S2 = math.sqrt(2.0)
INV_S2 = 1.0 / S2

STANDARD_THETAS: list[float] = [
    math.atan(0.5),          # arctan(1/2) ~ 26.565 deg
    math.pi / 4.0,           # pi/4 = 45.000 deg
    math.atan(2.0 / 3.0),    # arctan(2/3) ~ 33.690 deg
    math.atan(3.0 / 4.0),    # arctan(3/4) ~ 36.870 deg
]


# ---------------------------------------------------------------------------
# Geometry & Separating Axis Theorem (SAT)
# ---------------------------------------------------------------------------

def extent(q: np.ndarray) -> float:
    """Compute unconstrained square container side L for packing q."""
    if len(q) == 0:
        return 0.0
    c = np.abs(np.cos(q[:, 2]))
    s = np.abs(np.sin(q[:, 2]))
    h = 0.5 * (c + s)
    lo = np.min(q[:, :2] - h[:, None], axis=0)
    hi = np.max(q[:, :2] + h[:, None], axis=0)
    return float(np.max(hi - lo))


def normalize_box(q: np.ndarray) -> np.ndarray:
    """Translate packing so minimum bounding corner is at origin (0, 0)."""
    if len(q) == 0:
        return q.copy()
    out = q.copy()
    c = np.abs(np.cos(out[:, 2]))
    s = np.abs(np.sin(out[:, 2]))
    h = 0.5 * (c + s)
    lo = np.min(out[:, :2] - h[:, None], axis=0)
    out[:, :2] -= lo
    return out


def sat_check(q: np.ndarray, clearance: float = -1e-7) -> tuple[bool, float]:
    """Exact Separating Axis Theorem test across all pairs of squares."""
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
        min_gap = min(min_gap, float(np.min(best)))
        if np.any(best < clearance):
            return False, min_gap

    return True, min_gap


def sat_check_upright_against_tilted(
    xu: float, yu: float, tilted_arr: np.ndarray, clearance: float = -1e-7
) -> bool:
    """Vectorized exact SAT check for an upright unit square against all tilted squares."""
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
# 4-Sign-Piece Epigraph LP Relaxation Polisher
# ---------------------------------------------------------------------------

def chosen_axis(q: np.ndarray, i: int, j: int, previous=None, hysteresis: float = 1e-9):
    """Find dominant separating axis between square i and square j."""
    d = q[j, :2] - q[i, :2]
    choices = []
    for owner in (i, j):
        for k in (0, 1):
            phi = float(q[owner, 2]) + k * math.pi / 2.0
            a = np.array([math.cos(phi), math.sin(phi)])
            direction = 1 if float(d @ a) >= 0.0 else -1
            a *= direction
            h = sum(
                0.5 * (abs(math.cos(float(q[t, 2]) - phi)) + abs(math.sin(float(q[t, 2]) - phi)))
                for t in (i, j)
            )
            choices.append((float(d @ a - h), (owner, k, direction)))
    best = max(choices, key=lambda x: x[0])
    if previous is not None:
        for value, key in choices:
            if key == previous and value >= best[0] - hysteresis:
                return key
    return best[1]


def piece_gap_gradient(q: np.ndarray, i: int, j: int, axis, signs: tuple[float, float]):
    """Gradient of pairwise gap under 4-sign-piece epigraph linearization."""
    owner, k, direction = axis
    phi = float(q[owner, 2]) + k * math.pi / 2.0
    a = direction * np.array([math.cos(phi), math.sin(phi)])
    ap = np.array([-a[1], a[0]])
    phi += 0.0 if direction == 1 else math.pi
    other = j if owner == i else i
    r = float(q[other, 2]) - phi
    s1, s2 = signs
    hp = 0.5 * (-s1 * math.sin(r) + s2 * math.cos(r))
    d = q[j, :2] - q[i, :2]
    g = float(d @ a - 0.5 - 0.5 * (s1 * math.cos(r) + s2 * math.sin(r)))
    grad = np.zeros(6)
    grad[:2] = -a
    grad[3:5] = a
    grad[2 if owner == i else 5] = float(d @ ap) + hp
    grad[2 if other == i else 5] = -hp
    return g, grad


def _lp_step(
    q: np.ndarray,
    upper_L: float,
    position_trust: float,
    angle_trust: float,
    clearance: float,
    axes: dict,
) -> tuple[np.ndarray | None, str]:
    """Single LP step with 4-sign-piece epigraph linearization to minimize container side L."""
    from scipy.sparse import coo_matrix

    n = len(q)
    free = angle_trust > 0.0
    stride = 3 if free else 2
    nv = stride * n + 3  # offsets (dx, dy, [dtheta]) + xmin, ymin, L
    rr: list[int] = []
    cc: list[int] = []
    vv: list[float] = []
    rhs: list[float] = []

    def row(coeff: dict[int, float], b: float) -> None:
        r = len(rhs)
        rhs.append(float(b))
        for col, val in coeff.items():
            if val:
                rr.append(r)
                cc.append(col)
                vv.append(float(val))

    # Bounding box constraints for each square
    for i in range(n):
        t = float(q[i, 2])
        c, s = math.cos(t), math.sin(t)
        pieces = (
            [(0.5 * (a * c + b * s), 0.5 * (-a * s + b * c)) for a, b in SIGNS]
            if free
            else [(0.5 * (abs(c) + abs(s)), 0.0)]
        )
        for h, dh in pieces:
            for dim in (0, 1):
                lower = {stride * i + dim: -1.0, stride * n + dim: 1.0}
                upperrow = {stride * i + dim: 1.0, stride * n + dim: -1.0, nv - 1: -1.0}
                if free:
                    lower[stride * i + 2] = dh
                    upperrow[stride * i + 2] = dh
                row(lower, float(q[i, dim]) - h - clearance)
                row(upperrow, -float(q[i, dim]) - h - clearance)

    # Pairwise collision separation constraints
    radius = S2 + 2.0 * S2 * position_trust + 1e-6
    pairs = sorted(cKDTree(q[:, :2]).query_pairs(radius))
    for i, j in pairs:
        key = (i, j)
        axis = chosen_axis(q, i, j, axes.get(key))
        axes[key] = axis
        if free:
            for signs in SIGNS:
                g, grad = piece_gap_gradient(q, i, j, axis, signs)
                coeff = {}
                for k_dim in range(3):
                    coeff[stride * i + k_dim] = -grad[k_dim]
                    coeff[stride * j + k_dim] = -grad[3 + k_dim]
                row(coeff, g - clearance)
        else:
            owner, k_ax, direction = axis
            phi = float(q[owner, 2]) + k_ax * math.pi / 2.0
            a = direction * np.array([math.cos(phi), math.sin(phi)])
            h = sum(
                0.5 * (abs(math.cos(float(q[t, 2]) - phi)) + abs(math.sin(float(q[t, 2]) - phi)))
                for t in (i, j)
            )
            row(
                {stride * i: a[0], stride * i + 1: a[1], stride * j: -a[0], stride * j + 1: -a[1]},
                float((q[j, :2] - q[i, :2]) @ a - h - clearance),
            )

    bounds = []
    for i in range(n):
        bounds.extend([(-position_trust, position_trust)] * 2)
        if free:
            bounds.append((-angle_trust, angle_trust))

    bounds.extend([(None, None), (None, None), (1.0, upper_L)])
    objective = np.zeros(nv)
    objective[-1] = 1.0

    A = coo_matrix((vv, (rr, cc)), shape=(len(rhs), nv)).tocsr()
    res = linprog(
        objective,
        A_ub=A,
        b_ub=rhs,
        bounds=bounds,
        method="highs",
        options={
            "primal_feasibility_tolerance": 1e-9,
            "dual_feasibility_tolerance": 1e-9,
            "time_limit": 5.0,
        },
    )
    if not res.success or not np.isfinite(res.x).all():
        return None, str(res.message)

    out = q.copy()
    out[:, :2] += res.x[: stride * n].reshape(n, stride)[:, :2]
    if free:
        out[:, 2] += res.x[: stride * n].reshape(n, stride)[:, 2]
    return out, str(res.message)


def polish_lp(
    q: np.ndarray,
    max_iters: int = 15,
    position_trust: float = 0.10,
    angle_trust: float = 0.010,
    clearance: float = 1e-8,
) -> tuple[np.ndarray, float]:
    """Sequential LP polishing using 4-sign-piece epigraph relaxation."""
    best_q = normalize_box(q)
    best_L = extent(best_q)
    axes: dict[tuple[int, int], Any] = {}

    for it in range(max_iters):
        old_L = best_L
        # Position LP
        fixed, _ = _lp_step(best_q, best_L + 1e-4, position_trust, 0.0, clearance, axes)
        if fixed is not None:
            valid, _ = sat_check(fixed, clearance=-1e-7)
            cur_L = extent(fixed)
            if valid and cur_L < best_L:
                best_q = normalize_box(fixed)
                best_L = cur_L

        # Angle + position LP
        proposal, _ = _lp_step(best_q, best_L + 1e-4, position_trust, angle_trust, clearance, axes)
        if proposal is not None:
            origin = best_q.copy()
            for bt in range(5):
                alpha = 2.0 ** (-bt)
                seed = origin + alpha * (proposal - origin)
                valid, _ = sat_check(seed, clearance=-1e-7)
                cur_L = extent(seed)
                if valid and cur_L < best_L:
                    best_q = normalize_box(seed)
                    best_L = cur_L
                    break

        if old_L - best_L < 1e-8 and it >= 3:
            break

    return best_q, best_L


# ---------------------------------------------------------------------------
# True Tilted Band Placement & Dense Staircase Boundary Tucking
# ---------------------------------------------------------------------------

def generate_scratch_band(
    L: float,
    theta: float,
    w: int,
    v_off: float = 0.0,
    u_off: float = 0.0,
    trim: int = 0,
) -> tuple[np.ndarray, int]:
    """Place central diagonal band of width w at angle theta and tuck upright squares on flanks."""
    xc, yc = L / 2.0, L / 2.0
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    h_t = 0.5 * (abs(cos_t) + abs(sin_t))
    max_c = int(math.ceil(L * 1.5))

    tilted: list[list[float]] = []
    for s_idx in range(w):
        vs = (s_idx - (w - 1) / 2.0) * 1.0 + v_off
        valid_c: list[int] = []
        for c in range(-max_c, max_c + 1):
            uc = c * 1.0 + u_off
            x = xc + uc * cos_t - vs * sin_t
            y = yc + uc * sin_t + vs * cos_t
            if h_t - 1e-6 <= x <= L - h_t + 1e-6 and h_t - 1e-6 <= y <= L - h_t + 1e-6:
                valid_c.append(c)
        if trim > 0 and len(valid_c) > 2 * trim:
            valid_c = valid_c[trim : -trim]
        for c in valid_c:
            uc = c * 1.0 + u_off
            x = xc + uc * cos_t - vs * sin_t
            y = yc + uc * sin_t + vs * cos_t
            tilted.append([x, y, theta])

    if not tilted:
        return np.zeros((0, 3)), 0
    tilted_arr = np.array(tilted, dtype=np.float64)
    n_tilted = len(tilted)

    k_L = int(math.ceil(L))

    # Flank 1: North-West triangular flank (above the band: v > v_off)
    nw: list[list[float]] = []
    for i in range(k_L):
        xu = 0.5 + i
        if xu > L - 0.5:
            break
        for j in range(k_L):
            yu = L - 0.5 - j
            if yu < 0.5:
                break
            v = -(xu - xc) * sin_t + (yu - yc) * cos_t
            if v > v_off and sat_check_upright_against_tilted(xu, yu, tilted_arr):
                nw.append([xu, yu, 0.0])

    # Flank 2: South-East triangular flank (below the band: v < v_off)
    se: list[list[float]] = []
    for i in range(k_L):
        xu = L - 0.5 - i
        if xu < 0.5:
            break
        for j in range(k_L):
            yu = 0.5 + j
            if yu > L - 0.5:
                break
            v = -(xu - xc) * sin_t + (yu - yc) * cos_t
            if v < v_off and sat_check_upright_against_tilted(xu, yu, tilted_arr):
                se.append([xu, yu, 0.0])

    # Combine flanks and eliminate any mutual overlap
    upright: list[list[float]] = list(nw)
    for p in se:
        col = False
        for ex in upright:
            if abs(ex[0] - p[0]) < 1.0 - 1e-6 and abs(ex[1] - p[1]) < 1.0 - 1e-6:
                col = True
                break
        if not col:
            upright.append(p)

    # Dense staircase boundary tucking: corner & serration tucks
    for pt in [
        (0.5, 0.5), (L - 0.5, L - 0.5),
        (1.5, 0.5), (0.5, 1.5),
        (L - 1.5, L - 0.5), (L - 0.5, L - 1.5),
        (1.5, 1.5), (L - 1.5, L - 1.5),
    ]:
        if 0.5 <= pt[0] <= L - 0.5 and 0.5 <= pt[1] <= L - 0.5:
            col = False
            for ex in upright:
                if abs(ex[0] - pt[0]) < 1.0 - 1e-6 and abs(ex[1] - pt[1]) < 1.0 - 1e-6:
                    col = True
                    break
            if not col and sat_check_upright_against_tilted(pt[0], pt[1], tilted_arr):
                upright.append([pt[0], pt[1], 0.0])

    upright_arr = np.array(upright, dtype=np.float64) if upright else np.zeros((0, 3))
    total_arr = np.vstack([tilted_arr, upright_arr]) if len(upright_arr) > 0 else tilted_arr
    return total_arr, n_tilted


def generate_gobel_staircase(
    target_n: int,
    L: float,
    w: int,
    a1: int,
    a2: int,
    b1: int,
    b2: int,
) -> tuple[np.ndarray | None, int]:
    """Exact analytical Göbel 4-corner staircase band layout for theta = pi/4."""
    corners: list[list[float]] = []
    lim = int(math.floor(L - 1.0))
    for i in range(min(lim + 1, a1)):
        for j in range(min(lim + 1, a1 - i)):
            corners.append([0.5 + i, 0.5 + j, 0.0])
    for i in range(min(lim + 1, a2)):
        for j in range(min(lim + 1, a2 - i)):
            corners.append([L - 0.5 - i, L - 0.5 - j, 0.0])
    cx1 = max(0, int(math.floor(L - a2)))
    cy1 = max(0, int(math.floor(L - a1)))
    cx2, cy2 = cy1, cx1
    for i in range(min(cx1, b1)):
        for j in range(min(cy1, b1 - i)):
            corners.append([0.5 + i, L - 0.5 - j, 0.0])
    for i in range(min(cx2, b2)):
        for j in range(min(cy2, b2 - i)):
            corners.append([L - 0.5 - i, 0.5 + j, 0.0])

    u_start = (a1 + 1.0) * INV_S2 + 0.5
    tilted: list[list[float]] = []
    for k in range(w):
        u = u_start + k * 1.0
        v_lo = -min(u - 1.0, S2 * L - u - 1.0)
        v_hi = min(u - 1.0, S2 * L - u - 1.0)
        if b1 > 0:
            v_hi = min(v_hi, (L - b1 - 1.0) * INV_S2 - 0.5)
        if b2 > 0:
            v_lo = max(v_lo, -(L - b2 - 1.0) * INV_S2 + 0.5)
        if v_hi >= v_lo:
            cnt = int(math.floor((v_hi - v_lo) / 1.0)) + 1
            span = (cnt - 1) * 1.0
            v0 = 0.5 * (v_lo + v_hi) - 0.5 * span
            for m in range(cnt):
                v = v0 + m * 1.0
                x = (u - v) * INV_S2
                y = (u + v) * INV_S2
                tilted.append([x, y, math.pi / 4.0])

    tot = len(corners) + len(tilted)
    if tot < target_n:
        return None, 0
    q = np.array(corners + tilted, dtype=np.float64)
    return q, len(tilted)


def prune_to_n(q: np.ndarray, target_n: int, min_tilted: int = 0) -> np.ndarray:
    """Select exactly target_n squares that minimize bounding extent, preserving at least min_tilted tilted squares."""
    if len(q) <= target_n:
        return normalize_box(q)

    current = q.copy()
    while len(current) > target_n:
        center = np.mean(current[:, :2], axis=0)
        dist_sq = np.sum((current[:, :2] - center) ** 2, axis=1)
        outer_candidates = np.argsort(-dist_sq)

        tilted_count = int(np.sum(np.abs(current[:, 2]) > 1e-4))
        best_idx = None
        best_L = float("inf")
        c = np.abs(np.cos(current[:, 2]))
        s = np.abs(np.sin(current[:, 2]))
        h = 0.5 * (c + s)
        lo = current[:, :2] - h[:, None]
        hi = current[:, :2] + h[:, None]

        for idx in outer_candidates[: min(15, len(current))]:
            is_tilted = abs(current[idx, 2]) > 1e-4
            if is_tilted and tilted_count <= min_tilted:
                continue

            mask = np.ones(len(current), dtype=bool)
            mask[idx] = False
            cur_lo = np.min(lo[mask], axis=0)
            cur_hi = np.max(hi[mask], axis=0)
            cur_extent = float(np.max(cur_hi - cur_lo))
            if cur_extent < best_L:
                best_L = cur_extent
                best_idx = idx

        if best_idx is None:
            for idx in outer_candidates:
                is_tilted = abs(current[idx, 2]) > 1e-4
                if not (is_tilted and tilted_count <= min_tilted):
                    best_idx = idx
                    break
        if best_idx is None:
            best_idx = outer_candidates[0]

        current = np.delete(current, best_idx, axis=0)

    return normalize_box(current)


# ---------------------------------------------------------------------------
# Search and Polishing Engine
# ---------------------------------------------------------------------------

@dataclass
class TargetResult:
    target_n: int
    incumbent_L: float
    unpolished_L: float
    best_L: float
    delta_L: float
    within_1e3: bool
    family: str
    optimal_parameters: dict[str, Any]
    layout: list[dict[str, float]] = field(default_factory=list)
    status: str = "SUCCESS"


def search_target(
    target_n: int,
    incumbent_L: float,
    theta_grid: Sequence[float] = STANDARD_THETAS,
    widths: Sequence[int] = (1, 2, 3, 4, 5),
    offsets: Sequence[float] = (-0.2, 0.0, 0.2),
) -> TargetResult:
    """Search tilted band layouts for target n without integer-grid fallbacks."""
    k = int(math.floor(math.sqrt(target_n)))
    k_ceil = int(math.ceil(math.sqrt(target_n)))

    best_candidate: np.ndarray | None = None
    best_unpolished_L = float("inf")
    best_params: dict[str, Any] = {}
    best_w = 0

    # 1. Exact Göbel 4-corner staircase band search (theta = pi/4)
    for w in widths:
        min_tilted_req = w * k
        for a1 in range(max(1, k - 5), k + 5):
            for a2 in range(max(1, k - 6), a1 + 1):
                L_cand = 0.5 * (a1 + a2) + 1.0 + w * INV_S2
                if L_cand >= k_ceil or L_cand >= best_unpolished_L:
                    continue
                for b1 in range(0, 9):
                    for b2 in range(0, b1 + 1):
                        q_raw, n_t_raw = generate_gobel_staircase(target_n, L_cand, w, a1, a2, b1, b2)
                        if q_raw is not None and len(q_raw) >= target_n and n_t_raw >= min_tilted_req:
                            pruned = prune_to_n(q_raw, target_n, min_tilted=min_tilted_req)
                            valid, _ = sat_check(pruned, clearance=-1e-7)
                            if valid:
                                L_ext = extent(pruned)
                                n_t_pruned = int(np.sum(np.abs(pruned[:, 2]) > 1e-4))
                                if n_t_pruned >= min_tilted_req and L_ext < best_unpolished_L and L_ext < k_ceil:
                                    best_unpolished_L = L_ext
                                    best_candidate = pruned.copy()
                                    best_w = w
                                    best_params = {
                                        "theta": math.pi / 4.0,
                                        "theta_deg": 45.0,
                                        "w": w,
                                        "a1": a1,
                                        "a2": a2,
                                        "b1": b1,
                                        "b2": b2,
                                        "family": "gobel_staircase_band",
                                        "n_tilted": n_t_pruned,
                                        "n_upright": len(pruned) - n_t_pruned,
                                    }

    # 2. Continuous diagonal band + staircase tucking search across requested thetas
    trial_Ls = [incumbent_L, float(k_ceil) - 0.05, float(k_ceil) - 0.20]
    for L_trial in trial_Ls:
        if L_trial >= k_ceil or L_trial >= best_unpolished_L:
            continue
        for w in [w_val for w_val in widths if w_val <= 4]:
            min_tilted_req = w * k
            for theta in theta_grid:
                for v_off in offsets:
                    for u_off in (-0.1, 0.0, 0.1):
                        for trim in (0, 1, 2):
                            q_raw, n_t_raw = generate_scratch_band(
                                L=L_trial,
                                theta=theta,
                                w=w,
                                v_off=v_off,
                                u_off=u_off,
                                trim=trim,
                            )
                            if len(q_raw) >= target_n and n_t_raw >= min_tilted_req:
                                pruned = prune_to_n(q_raw, target_n, min_tilted=min_tilted_req)
                                valid, _ = sat_check(pruned, clearance=-1e-7)
                                if valid:
                                    L_ext = extent(pruned)
                                    n_t_pruned = int(np.sum(np.abs(pruned[:, 2]) > 1e-4))
                                    if n_t_pruned >= min_tilted_req and L_ext < best_unpolished_L and L_ext < k_ceil:
                                        best_unpolished_L = L_ext
                                        best_candidate = pruned.copy()
                                        best_w = w
                                        best_params = {
                                            "theta": theta,
                                            "theta_deg": float(math.degrees(theta)),
                                            "w": w,
                                            "v_off": v_off,
                                            "u_off": u_off,
                                            "trim": trim,
                                            "family": "tilted_staircase_band",
                                            "n_tilted": n_t_pruned,
                                            "n_upright": len(pruned) - n_t_pruned,
                                        }

    # Hard assertion check: If no candidate achieved L < ceil(sqrt(n)), mark as FAILED
    if best_candidate is None or best_unpolished_L >= k_ceil:
        return TargetResult(
            target_n=target_n,
            incumbent_L=incumbent_L,
            unpolished_L=float("inf"),
            best_L=float("inf"),
            delta_L=float("inf"),
            within_1e3=False,
            family="none",
            optimal_parameters={
                "status": "FAILED",
                "reason": f"cannot achieve L < ceil(sqrt(n)) ({k_ceil})",
            },
            layout=[],
            status=f"FAILED (cannot achieve L < ceil(sqrt(n)))",
        )

    # Validate assertions on unpolished candidate
    n_t_cand = int(np.sum(np.abs(best_candidate[:, 2]) > 1e-4))
    assert n_t_cand >= best_w * k, (
        f"Target {target_n}: candidate has {n_t_cand} tilted squares, expected >= {best_w * k}"
    )
    assert best_unpolished_L < k_ceil, (
        f"Target {target_n}: candidate L={best_unpolished_L} >= ceil(sqrt(n))={k_ceil}"
    )

    # LP Polishing using 4-sign-piece epigraph relaxation
    polished_q, polished_L = polish_lp(best_candidate, max_iters=15)
    valid, _ = sat_check(polished_q, clearance=-1e-7)
    if not valid or polished_L >= best_unpolished_L:
        polished_q = best_candidate
        polished_L = best_unpolished_L

    # Assertions on final polished packing
    n_t_final = int(np.sum(np.abs(polished_q[:, 2]) > 1e-4))
    assert n_t_final >= best_w * k, (
        f"Target {target_n}: final packing has {n_t_final} tilted squares, expected >= {best_w * k}"
    )
    assert polished_L < k_ceil, (
        f"Target {target_n}: final polished L={polished_L} >= ceil(sqrt(n))={k_ceil}"
    )

    delta = polished_L - incumbent_L
    within_1e3 = delta <= 1e-3

    best_params["n_tilted"] = n_t_final
    best_params["n_upright"] = len(polished_q) - n_t_final

    layout = [
        {
            "id": i,
            "x": float(polished_q[i, 0]),
            "y": float(polished_q[i, 1]),
            "angle": float(polished_q[i, 2]),
        }
        for i in range(len(polished_q))
    ]

    return TargetResult(
        target_n=target_n,
        incumbent_L=incumbent_L,
        unpolished_L=best_unpolished_L,
        best_L=polished_L,
        delta_L=delta,
        within_1e3=within_1e3,
        family=best_params["family"],
        optimal_parameters=best_params,
        layout=layout,
        status="SUCCESS",
    )


# ---------------------------------------------------------------------------
# Target Loading & Main Execution
# ---------------------------------------------------------------------------

def load_or_compute_top40(
    targets_path: Path = DEFAULT_TARGETS_PATH,
    merged_path: Path = MERGED_LITERAL_PATH,
) -> list[dict[str, Any]]:
    """Load top 40 targets from targets_path or compute top 40 in [100, 1000] by score excess."""
    if targets_path.exists():
        with open(targets_path) as f:
            targets = json.load(f)
        if len(targets) >= 40:
            return targets[:40]

    if not merged_path.exists():
        raise FileNotFoundError(f"Missing {merged_path}")
    with open(merged_path) as f:
        merged_data = json.load(f)

    results = [r for r in merged_data["results"] if 100 <= r["n"] <= 1000]
    sorted_targets = sorted(
        results,
        key=lambda r: float(r["L"]) / math.sqrt(r["n"]) - 1.0,
        reverse=True,
    )
    top40 = []
    for r in sorted_targets[:40]:
        n_val = int(r["n"])
        k_val = int(math.isqrt(n_val))
        j_val = n_val - k_val * k_val
        l_val = float(r["L"])
        score_val = l_val / math.sqrt(n_val) - 1.0
        top40.append({
            "n": n_val,
            "k": k_val,
            "j": j_val,
            "L": l_val,
            "score": score_val,
        })

    targets_path.parent.mkdir(parents=True, exist_ok=True)
    with open(targets_path, "w") as f:
        json.dump(top40, f, indent=2)

    return top40


def run_all(
    targets_file: Path | None = None,
    output_file: Path | None = None,
    max_targets: int | None = None,
) -> dict[str, Any]:
    """Run tilted band and staircase search across top 40 targets with LP polishing."""
    t0 = time.time()
    t_path = targets_file or DEFAULT_TARGETS_PATH
    out_path = output_file or DEFAULT_OUTPUT_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)

    targets_list = load_or_compute_top40(t_path)
    if max_targets is not None:
        targets_list = targets_list[:max_targets]

    # Sanity-test on n=290 to confirm L <= 17.55
    print("Running sanity test on n=290...")
    res_290 = search_target(
        target_n=290,
        incumbent_L=17.53553390593285,
        theta_grid=STANDARD_THETAS,
        widths=[1, 2, 3, 4, 5],
        offsets=[-0.2, 0.0, 0.2],
    )
    assert res_290.status == "SUCCESS", f"Sanity test on n=290 failed: status={res_290.status}"
    assert res_290.best_L <= 17.55 + 1e-6, f"Sanity test on n=290 failed: L={res_290.best_L} > 17.55"
    assert res_290.optimal_parameters["n_tilted"] >= res_290.optimal_parameters["w"] * 17
    print(f"Sanity test PASSED: n=290 L={res_290.best_L:.6f} <= 17.55 (n_tilted={res_290.optimal_parameters['n_tilted']})")

    all_results: dict[str, Any] = {}
    success_count = 0
    failed_count = 0

    for idx, target in enumerate(targets_list, 1):
        target_n = int(target["n"])
        inc_L = float(target["L"])

        t_start = time.time()
        res = search_target(
            target_n=target_n,
            incumbent_L=inc_L,
            theta_grid=STANDARD_THETAS,
            widths=[1, 2, 3, 4, 5],
            offsets=[-0.2, 0.0, 0.2],
        )
        t_elapsed = time.time() - t_start

        if res.status == "SUCCESS":
            success_count += 1
            all_results[str(target_n)] = {
                "target_n": res.target_n,
                "status": "SUCCESS",
                "incumbent_L": res.incumbent_L,
                "unpolished_L": res.unpolished_L,
                "best_L": res.best_L,
                "delta_L": res.delta_L,
                "within_1e3": res.within_1e3,
                "family": res.family,
                "optimal_parameters": res.optimal_parameters,
                "layout": res.layout,
            }
            print(f"[{idx:2d}/{len(targets_list)}] n={target_n:3d}: SUCCESS L={res.best_L:.6f} (inc={inc_L:.6f}, delta={res.delta_L:+.6f}) in {t_elapsed:.2f}s")
        else:
            failed_count += 1
            all_results[str(target_n)] = {
                "target_n": res.target_n,
                "status": res.status,
                "incumbent_L": res.incumbent_L,
                "unpolished_L": None,
                "best_L": None,
                "delta_L": None,
                "within_1e3": False,
                "family": "none",
                "optimal_parameters": res.optimal_parameters,
                "layout": [],
            }
            print(f"[{idx:2d}/{len(targets_list)}] n={target_n:3d}: {res.status} in {t_elapsed:.2f}s")

    elapsed = time.time() - t0
    within_count = sum(1 for r in all_results.values() if r.get("within_1e3"))
    improved_count = sum(1 for r in all_results.values() if r.get("delta_L") is not None and r["delta_L"] < -1e-9)

    summary = {
        "elapsed_seconds": elapsed,
        "total_targets": len(targets_list),
        "success_count": success_count,
        "failed_count": failed_count,
        "within_1e3_count": within_count,
        "improved_count": improved_count,
        "targets": all_results,
    }

    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nExecution finished in {elapsed:.2f}s. Saved to {out_path}.")
    print(f"Success: {success_count}/{len(targets_list)}, Failed: {failed_count}/{len(targets_list)}, Within 1e-3: {within_count}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Large-N Scratch Band Search (Idea 1)")
    parser.add_argument("--targets-file", type=Path, default=None, help="Path to top40 targets json")
    parser.add_argument("--output-file", type=Path, default=None, help="Path to output json")
    parser.add_argument("--max-targets", type=int, default=None, help="Limit number of targets")
    args = parser.parse_args()

    run_all(
        targets_file=args.targets_file,
        output_file=args.output_file,
        max_targets=args.max_targets,
    )
