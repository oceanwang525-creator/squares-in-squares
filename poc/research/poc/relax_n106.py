"""Multi-seed basin hopping and mixed relaxation for target n=106.

Target incumbent L = 10.822908, candidate scratch band L = 10.828427.
1. Loads Seed 1 (Idea 1 scratch band) and Seed 2 (incumbent).
2. Implements basin hopping: identifies 5-15 squares with tightest SAT contact margins,
   applies random coordinate kicks and angular perturbations, runs L-BFGS with smooth
   overlap penalty, and contracts with 4-sign-piece epigraph LP.
3. Tests mixed structures: hybridizes variable-slope band with exact rational slopes
   (theta = arctan(1/2) and arctan(2/3)) and corner notch tucking.
4. Runs across at least 30 random kick seeds per base seed, tracks best L per seed and method,
   and writes results to poc/research/data/relax_n106_results.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linprog, minimize
from scipy.spatial import cKDTree

# ---------------------------------------------------------------------------
# Path & Environment Setup
# ---------------------------------------------------------------------------

REPO = Path(__file__).resolve().parents[3]
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SCRATCH_RESULTS_PATH = DATA_DIR / "large_n_scratch_band_results.json"
MERGED_CSV_PATH = REPO / "poc/pell_corrugation/data/border-scan/merged.csv"
OUTPUT_PATH = DATA_DIR / "relax_n106_results.json"

TARGET_N = 106
INCUMBENT_L = 10.822908044132857
SIGNS: tuple[tuple[float, float], ...] = (
    (-1.0, -1.0),
    (-1.0, 1.0),
    (1.0, -1.0),
    (1.0, 1.0),
)
S2 = math.sqrt(2.0)


# ---------------------------------------------------------------------------
# Geometric Primitives & Separation Tests
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


def compute_sat_margins(q: np.ndarray) -> np.ndarray:
    """Compute per-square minimum contact margin to other squares and container walls."""
    n = len(q)
    q_arr = np.asarray(q, dtype=np.float64)
    c, s = np.cos(q_arr[:, 2]), np.sin(q_arr[:, 2])
    u = np.column_stack((c, s))
    v = np.column_stack((-s, c))
    margins = np.full(n, np.inf)

    # Square-to-square margins
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
        for idx_offset, gap in enumerate(best):
            j = i + 1 + idx_offset
            if gap < margins[i]:
                margins[i] = gap
            if gap < margins[j]:
                margins[j] = gap

    # Wall margins
    L = extent(q_arr)
    h = 0.5 * (np.abs(c) + np.abs(s))
    left = q_arr[:, 0] - h
    right = L - (q_arr[:, 0] + h)
    bottom = q_arr[:, 1] - h
    top = L - (q_arr[:, 1] + h)
    wall_margins = np.minimum.reduce([left, right, bottom, top])
    total_margins = np.minimum(margins, wall_margins)
    return total_margins


# ---------------------------------------------------------------------------
# 4-Sign-Piece Epigraph LP Relaxation
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
        # Step 1: Fixed-angle position LP
        fixed, _ = _lp_step(best_q, best_L + 1e-4, position_trust, 0.0, clearance, axes)
        if fixed is not None:
            valid, _ = sat_check(fixed, clearance=-1e-7)
            cur_L = extent(fixed)
            if valid and cur_L < best_L:
                best_q = normalize_box(fixed)
                best_L = cur_L

        # Step 2: Joint angle and position LP with backtracking line search
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


def restore_feasibility(q_arr: np.ndarray) -> np.ndarray | None:
    """Restore strict SAT feasibility via mild center dilation if numerical noise creates overlap."""
    valid, min_g = sat_check(q_arr, clearance=-1e-7)
    if valid:
        return q_arr.copy()
    center = np.mean(q_arr[:, :2], axis=0)
    for d in [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3]:
        q_dil = q_arr.copy()
        q_dil[:, :2] = center + (1.0 + d) * (q_arr[:, :2] - center)
        v, g = sat_check(q_dil, clearance=-1e-7)
        if v:
            return q_dil
    return None


# ---------------------------------------------------------------------------
# Smooth Overlap Penalty & L-BFGS Relaxation
# ---------------------------------------------------------------------------

def fast_subspace_overlap_obj(
    z_sub: np.ndarray,
    q_full: np.ndarray,
    active_idx: np.ndarray,
    side: float,
    i_arr: np.ndarray,
    j_arr: np.ndarray,
    clearance_margin: float = 1e-5,
) -> float:
    """Vectorized smooth overlap and wall violation penalty for active squares."""
    q_curr = q_full.copy()
    q_curr[active_idx] = z_sub.reshape(len(active_idx), 3)
    xy, angle = q_curr[:, :2], q_curr[:, 2]
    c, s = np.cos(angle), np.sin(angle)
    u, v = np.stack([c, s], axis=1), np.stack([-s, c], axis=1)

    if len(i_arr) > 0:
        delta = xy[j_arr] - xy[i_arr]
        axes = np.stack([u[i_arr], v[i_arr], u[j_arr], v[j_arr]], axis=1)
        ri = 0.5 * (
            abs(np.einsum("pkd,pd->pk", axes, u[i_arr]))
            + abs(np.einsum("pkd,pd->pk", axes, v[i_arr]))
        )
        rj = 0.5 * (
            abs(np.einsum("pkd,pd->pk", axes, u[j_arr]))
            + abs(np.einsum("pkd,pd->pk", axes, v[j_arr]))
        )
        gap = abs(np.einsum("pkd,pd->pk", axes, delta)) - (ri + rj)
        overlap = np.maximum(0.0, clearance_margin - np.max(gap, axis=1))
        pair_penalty = float(np.sum(overlap ** 2))
    else:
        pair_penalty = 0.0

    # Boundary wall penalties
    extent_val = 0.5 * (abs(c[active_idx]) + abs(s[active_idx]))
    walls = np.maximum(
        0.0,
        abs(xy[active_idx] - side / 2.0) + extent_val[:, None] - side / 2.0,
    )
    wall_penalty = float(np.sum(walls ** 2))

    return 1000.0 * pair_penalty + 10.0 * wall_penalty


def lbfgs_smooth_relax(
    q: np.ndarray,
    active_idx: np.ndarray,
    target_side: float,
    maxiter: int = 25,
) -> np.ndarray:
    """Run L-BFGS to resolve overlaps on perturbed active squares."""
    tree = cKDTree(q[:, :2])
    pairs = set()
    for idx in active_idx:
        nbrs = tree.query_ball_point(q[idx, :2], 2.5)
        for nb in nbrs:
            if nb != idx:
                pairs.add((min(idx, nb), max(idx, nb)))

    if pairs:
        pair_list = np.array(list(pairs), dtype=int)
        i_arr, j_arr = pair_list[:, 0], pair_list[:, 1]
    else:
        i_arr, j_arr = np.zeros(0, dtype=int), np.zeros(0, dtype=int)

    k = len(active_idx)
    z0 = q[active_idx].ravel()

    bounds = []
    for _ in range(k):
        bounds.extend([(0.0, target_side + 1.0), (0.0, target_side + 1.0), (-math.pi, math.pi)])

    res = minimize(
        fast_subspace_overlap_obj,
        z0,
        args=(q, active_idx, target_side, i_arr, j_arr, 1e-5),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-6},
    )

    out = q.copy()
    out[active_idx] = res.x.reshape(k, 3)
    return out


# ---------------------------------------------------------------------------
# Seed Loaders & Mixed Structure Builders
# ---------------------------------------------------------------------------

def load_seed1() -> np.ndarray:
    """Load Seed 1: Idea 1 scratch band from large_n_scratch_band_results.json."""
    if not SCRATCH_RESULTS_PATH.exists():
        raise FileNotFoundError(f"Scratch results file not found at {SCRATCH_RESULTS_PATH}")
    with open(SCRATCH_RESULTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    t = data["targets"][str(TARGET_N)]
    layout = [[sq["x"], sq["y"], sq["angle"]] for sq in t["layout"]]
    arr = np.array(layout, dtype=np.float64)
    return normalize_box(arr)


def load_seed2() -> np.ndarray:
    """Load Seed 2: Incumbent from poc/pell_corrugation/data/border-scan/merged.csv."""
    if not MERGED_CSV_PATH.exists():
        raise FileNotFoundError(f"Merged CSV not found at {MERGED_CSV_PATH}")
    rows: list[list[float]] = []
    with open(MERGED_CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r["n"] == str(TARGET_N):
                rows.append([float(r["x"]), float(r["y"]), float(r["angle"])])
    if len(rows) != TARGET_N:
        raise ValueError(f"Expected {TARGET_N} squares in merged.csv for n={TARGET_N}, found {len(rows)}")
    arr = np.array(rows, dtype=np.float64)
    return normalize_box(arr)


def build_rational_hybrid(
    q_base: np.ndarray,
    target_theta: float,
    angle_threshold: float = 0.5,
) -> np.ndarray:
    """Hybridize variable-slope band with an exact rational slope (arctan(1/2) or arctan(2/3))."""
    q_hyb = q_base.copy()
    n = len(q_hyb)
    tilted_mask = np.abs(q_hyb[:, 2]) > 1e-4

    for i in range(n):
        if tilted_mask[i]:
            if target_theta < 0.5 and q_hyb[i, 2] < angle_threshold:
                q_hyb[i, 2] = target_theta
            elif target_theta >= 0.5 and q_hyb[i, 2] >= angle_threshold:
                q_hyb[i, 2] = target_theta

    active = np.where(tilted_mask)[0]
    q_relaxed = lbfgs_smooth_relax(q_hyb, active, extent(q_hyb) + 0.05, maxiter=30)
    q_restored = restore_feasibility(q_relaxed)
    if q_restored is not None:
        q_pol, _ = polish_lp(q_restored, max_iters=10)
        return q_pol
    return q_hyb


def build_dual_rational_hybrid(q_base: np.ndarray) -> np.ndarray:
    """Hybridize both band segments with exact rational slopes: arctan(1/2) and arctan(2/3)."""
    q_hyb = q_base.copy()
    n = len(q_hyb)
    tilted_mask = np.abs(q_hyb[:, 2]) > 1e-4
    th1 = math.atan(0.5)        # ~0.463648
    th2 = math.atan(2.0 / 3.0)  # ~0.588003

    for i in range(n):
        if tilted_mask[i]:
            q_hyb[i, 2] = th1 if q_hyb[i, 2] < 0.5 else th2

    active = np.where(tilted_mask)[0]
    q_relaxed = lbfgs_smooth_relax(q_hyb, active, extent(q_hyb) + 0.05, maxiter=35)
    q_restored = restore_feasibility(q_relaxed)
    if q_restored is not None:
        q_pol, _ = polish_lp(q_restored, max_iters=10)
        return q_pol
    return q_hyb


def build_corner_notch_tucking_hybrid(q_base: np.ndarray) -> np.ndarray:
    """Tuck squares into corner notches along inner boundary of border blocks."""
    q_tuck = q_base.copy()
    L = extent(q_tuck)
    c = np.abs(np.cos(q_tuck[:, 2]))
    s = np.abs(np.sin(q_tuck[:, 2]))
    h = 0.5 * (c + s)
    hi = q_tuck[:, :2] + h[:, None]
    lo = q_tuck[:, :2] - h[:, None]

    best_drop: list[tuple[int, float]] = []
    for idx in range(len(q_tuck)):
        mask = np.ones(len(q_tuck), dtype=bool)
        mask[idx] = False
        cur_extent = float(np.max(np.max(hi[mask], axis=0) - np.min(lo[mask], axis=0)))
        best_drop.append((idx, cur_extent))
    best_drop.sort(key=lambda x: x[1])

    corner_candidates = [
        [0.5, 0.5, 0.0],
        [L - 0.5, 0.5, 0.0],
        [0.5, L - 0.5, 0.0],
        [L - 0.5, L - 0.5, 0.0],
        [1.5, 0.5, 0.0],
        [0.5, 1.5, 0.0],
        [L - 1.5, 0.5, 0.0],
        [0.5, L - 1.5, 0.0],
        [L - 0.5, 1.5, 0.0],
        [L - 1.5, L - 0.5, 0.0],
    ]

    for candidate in corner_candidates:
        for drop_idx, _ in best_drop[:5]:
            q_test = q_tuck.copy()
            q_test[drop_idx] = candidate
            active = np.array([drop_idx])
            q_relaxed = lbfgs_smooth_relax(q_test, active, L, maxiter=20)
            q_restored = restore_feasibility(q_relaxed)
            if q_restored is not None:
                q_pol, L_pol = polish_lp(q_restored, max_iters=5)
                valid, min_g = sat_check(q_pol)
                if valid and L_pol < L:
                    return q_pol

    perm_idx = np.array([drop_idx for drop_idx, _ in best_drop[:8]])
    center = np.mean(q_tuck[:, :2], axis=0)
    q_tuck[perm_idx, :2] += 0.01 * (center - q_tuck[perm_idx, :2])
    q_relaxed = lbfgs_smooth_relax(q_tuck, perm_idx, L, maxiter=25)
    q_restored = restore_feasibility(q_relaxed)
    if q_restored is not None:
        q_pol, _ = polish_lp(q_restored, max_iters=10)
        return q_pol
    return q_base


# ---------------------------------------------------------------------------
# Basin Hopping Engine
# ---------------------------------------------------------------------------

def run_basin_hopping_step(
    base_q: np.ndarray,
    seed: int,
    k_min: int = 5,
    k_max: int = 15,
    pos_kick_scale: float = 0.025,
    ang_kick_scale: float = 0.015,
) -> tuple[np.ndarray, float, bool, float, int]:
    """Execute one basin-hopping perturbation and contraction step."""
    rng = np.random.default_rng(seed)
    n = len(base_q)
    cur_L = extent(base_q)

    # 1. Identify 5-15 squares with tightest SAT contact margins
    margins = compute_sat_margins(base_q)
    tight_ranking = np.argsort(margins)
    k = int(rng.integers(k_min, k_max + 1))
    candidate_pool = tight_ranking[: min(30, n)]
    kick_idx = rng.choice(candidate_pool, size=k, replace=False)

    # 2. Apply random coordinate kicks and small angular perturbations
    q_kicked = base_q.copy()
    q_kicked[kick_idx, :2] += rng.uniform(-pos_kick_scale, pos_kick_scale, (k, 2))
    q_kicked[kick_idx, 2] += rng.uniform(-ang_kick_scale, ang_kick_scale, k)

    # 3. Run L-BFGS with smooth overlap penalty
    q_relaxed = lbfgs_smooth_relax(q_kicked, kick_idx, target_side=cur_L + 0.01, maxiter=25)

    # 4. Feasibility restoration
    q_restored = restore_feasibility(q_relaxed)
    if q_restored is None:
        q_restored = base_q.copy()

    # 5. Contract with 4-sign-piece epigraph LP
    q_contracted, L_contracted = polish_lp(
        q_restored,
        max_iters=10,
        position_trust=0.08,
        angle_trust=0.008,
        clearance=1e-8,
    )

    valid, min_g = sat_check(q_contracted)
    return q_contracted, float(L_contracted), bool(valid), float(min_g), k


# ---------------------------------------------------------------------------
# Multi-Seed Execution & Main Entry Point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Multi-seed basin hopping for n=106")
    parser.add_argument("--seeds-per-method", type=int, default=30, help="Random kick seeds per base seed")
    parser.add_argument("--save-winning", action="store_true", default=True, help="Save winning packing if L < incumbent")
    args = parser.parse_args()

    print(f"=================================================================")
    print(f"Target n=106 Multi-Seed Basin Hopping & Mixed Relaxation")
    print(f"Target Incumbent L: {INCUMBENT_L:.6f}")
    print(f"Seeds per base configuration: {args.seeds_per_method}")
    print(f"=================================================================")

    t0_all = time.time()

    # 1. Load Seed 1 and Seed 2
    q_seed1 = load_seed1()
    L_seed1 = extent(q_seed1)
    v_seed1, g_seed1 = sat_check(q_seed1)
    print(f"Seed 1 (Idea 1 Scratch Band): L={L_seed1:.6f}, valid={v_seed1}, min_gap={g_seed1:.2e}")

    q_seed2 = load_seed2()
    L_seed2 = extent(q_seed2)
    v_seed2, g_seed2 = sat_check(q_seed2)
    print(f"Seed 2 (Incumbent Variable-Slope): L={L_seed2:.6f}, valid={v_seed2}, min_gap={g_seed2:.2e}")

    # 2. Build Mixed Structures
    print("\nConstructing Mixed Structures...")
    th_1_2 = math.atan(0.5)
    th_2_3 = math.atan(2.0 / 3.0)

    q_hyb_1_2 = build_rational_hybrid(q_seed2, target_theta=th_1_2)
    L_hyb_1_2 = extent(q_hyb_1_2)
    v_hyb_1_2, g_hyb_1_2 = sat_check(q_hyb_1_2)
    print(f"Mixed 1 (Hybrid theta=arctan(1/2)): L={L_hyb_1_2:.6f}, valid={v_hyb_1_2}, min_gap={g_hyb_1_2:.2e}")

    q_hyb_2_3 = build_rational_hybrid(q_seed2, target_theta=th_2_3)
    L_hyb_2_3 = extent(q_hyb_2_3)
    v_hyb_2_3, g_hyb_2_3 = sat_check(q_hyb_2_3)
    print(f"Mixed 2 (Hybrid theta=arctan(2/3)): L={L_hyb_2_3:.6f}, valid={v_hyb_2_3}, min_gap={g_hyb_2_3:.2e}")

    q_hyb_dual = build_dual_rational_hybrid(q_seed2)
    L_hyb_dual = extent(q_hyb_dual)
    v_hyb_dual, g_hyb_dual = sat_check(q_hyb_dual)
    print(f"Mixed 3 (Dual Rational Hybrid): L={L_hyb_dual:.6f}, valid={v_hyb_dual}, min_gap={g_hyb_dual:.2e}")

    q_corner_tuck = build_corner_notch_tucking_hybrid(q_seed2)
    L_corner_tuck = extent(q_corner_tuck)
    v_corner_tuck, g_corner_tuck = sat_check(q_corner_tuck)
    print(f"Mixed 4 (Corner Notch Tucking): L={L_corner_tuck:.6f}, valid={v_corner_tuck}, min_gap={g_corner_tuck:.2e}")

    configurations: list[dict[str, Any]] = [
        {"name": "Seed_1_Scratch_Band", "base_q": q_seed1, "initial_L": L_seed1},
        {"name": "Seed_2_Incumbent_Variable_Slope", "base_q": q_seed2, "initial_L": L_seed2},
        {"name": "Mixed_Hybrid_arctan_1_2", "base_q": q_hyb_1_2, "initial_L": L_hyb_1_2},
        {"name": "Mixed_Hybrid_arctan_2_3", "base_q": q_hyb_2_3, "initial_L": L_hyb_2_3},
        {"name": "Mixed_Dual_Rational", "base_q": q_hyb_dual, "initial_L": L_hyb_dual},
        {"name": "Mixed_Corner_Notch_Tucking", "base_q": q_corner_tuck, "initial_L": L_corner_tuck},
    ]

    best_overall_L = min(L_seed1, L_seed2)
    best_overall_layout: np.ndarray | None = q_seed2.copy() if L_seed2 <= L_seed1 else q_seed1.copy()
    winning_achieved = False

    results_data: dict[str, Any] = {
        "target_n": TARGET_N,
        "target_incumbent_L": INCUMBENT_L,
        "scratch_band_candidate_L": L_seed1,
        "gap_initial": L_seed1 - INCUMBENT_L,
        "num_seeds_per_method": args.seeds_per_method,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "methods": {},
    }

    # 3. Run Basin Hopping Across Seeds
    for cfg in configurations:
        name = cfg["name"]
        base_q = cfg["base_q"]
        init_L = cfg["initial_L"]
        print(f"\n--- Running Basin Hopping for {name} ({args.seeds_per_method} seeds) ---")

        best_method_L = init_L
        best_method_layout = base_q.copy()
        trials_log: list[dict[str, Any]] = []

        t0_method = time.time()
        for s_idx in range(args.seeds_per_method):
            seed_val = 106000 + hash(name) % 10000 + s_idx
            q_out, L_out, valid, min_g, k_kicked = run_basin_hopping_step(
                base_q,
                seed=seed_val,
                k_min=5,
                k_max=15,
            )

            if valid and L_out < best_method_L:
                best_method_L = L_out
                best_method_layout = q_out.copy()

            if valid and L_out < best_overall_L:
                best_overall_L = L_out
                best_overall_layout = q_out.copy()
                if best_overall_L < INCUMBENT_L - 1e-7:
                    winning_achieved = True
                    print(f"  >>> NEW RECORD PACKING! L={best_overall_L:.8f} (beat incumbent {INCUMBENT_L:.6f})")

            trials_log.append({
                "trial": s_idx + 1,
                "kick_seed": seed_val,
                "squares_kicked": k_kicked,
                "L": L_out,
                "valid": valid,
                "min_gap": min_g,
                "delta_to_incumbent": L_out - INCUMBENT_L,
            })

            if (s_idx + 1) % 10 == 0 or s_idx == args.seeds_per_method - 1:
                print(f"  Completed {s_idx + 1:2d}/{args.seeds_per_method}: best L so far = {best_method_L:.6f}")

        elapsed_method = time.time() - t0_method
        cfg_summary = {
            "initial_L": init_L,
            "best_L": best_method_L,
            "improvement": init_L - best_method_L,
            "delta_to_incumbent": best_method_L - INCUMBENT_L,
            "elapsed_seconds": round(elapsed_method, 2),
            "trials": trials_log,
        }
        results_data["methods"][name] = cfg_summary
        print(f"Done {name}: best_L={best_method_L:.6f}, elapsed={elapsed_method:.1f}s")

    # 4. Save Results
    total_elapsed = time.time() - t0_all
    results_data["best_overall_L"] = best_overall_L
    results_data["improvement_over_incumbent"] = INCUMBENT_L - best_overall_L
    results_data["winning_achieved"] = winning_achieved
    results_data["total_elapsed_seconds"] = round(total_elapsed, 2)

    if best_overall_layout is not None:
        layout_dict = [
            {"id": i, "x": float(best_overall_layout[i, 0]), "y": float(best_overall_layout[i, 1]), "angle": float(best_overall_layout[i, 2])}
            for i in range(len(best_overall_layout))
        ]
        if winning_achieved:
            results_data["winning_packing_layout"] = layout_dict
        else:
            results_data["best_packing_layout"] = layout_dict

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2)
    print(f"\nResults successfully written to {OUTPUT_PATH}")
    print(f"Total time elapsed: {total_elapsed:.1f}s")
    print(f"Best overall L: {best_overall_L:.6f} (Incumbent: {INCUMBENT_L:.6f})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
