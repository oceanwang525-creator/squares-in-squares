#!/usr/bin/env python3
"""Multi-seed basin hopping and mixed relaxation for target n=295.

Executes multi-seed relaxation for n=295 comparing:
  - Seed 1: Idea 1 scratch band (large_n_scratch_band_results.json, L = 17.707107)
  - Seed 2: Incumbent packing (pell_corrugation/data/border-scan/merged.csv, L = 17.704233)
  - Mixed Structure 1: Central 45-deg diagonal band with tilted corner blocks
  - Mixed Structure 2: Central 45-deg diagonal band with dual-angle boundary staircases

For each base configuration and across 30 random kick seeds:
  1. Identifies 5-15 squares with tightest SAT contact margins.
  2. Perturbs them with coordinate kicks (sigma_x = 0.05) and angular kicks (sigma_theta = 0.03).
  3. Relaxes overlaps via L-BFGS with smooth quadratic overlap penalty.
  4. Polishes container side L using sequential 4-sign-piece epigraph LP relaxation.
Logs all trials and results to poc/research/data/relax_n295_results.json.
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Tuple

import numpy as np
from scipy.optimize import linprog, minimize
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "poc/baseline/poc/src"))
from squares.core import certify, evaluate

DATA_DIR = REPO_ROOT / "poc/research/data"
BORDER_SCAN_DIR = REPO_ROOT / "poc/pell_corrugation/data/border-scan"
RESULTS_PATH = DATA_DIR / "relax_n295_results.json"

TARGET_N = 295
TARGET_INCUMBENT_L = 17.70423279073607
SCRATCH_BAND_L = 17.707106781186546

SIGNS = ((-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0))
SQRT2 = math.sqrt(2.0)


# ---------------------------------------------------------------------------
# Geometry & Separating Axis Theorem (SAT)
# ---------------------------------------------------------------------------

def extent(q: np.ndarray) -> float:
    """Compute bounding container side L for packing q."""
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


def sat_check(q: np.ndarray, clearance: float = -1e-7) -> Tuple[bool, float]:
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
                hj = 0.5 * (
                    np.abs(np.sum(a * u[i + 1 :], axis=1))
                    + np.abs(np.sum(a * v[i + 1 :], axis=1))
                )
                g = np.abs(np.sum(d * a, axis=1)) - hi - hj
                best = np.maximum(best, g)
        min_gap = min(min_gap, float(np.min(best)))
        if np.any(best < clearance):
            return False, min_gap

    return True, min_gap


def compute_sat_margins(q: np.ndarray, radius: float = 1.7) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-square minimum SAT contact margins over spatial neighbor pairs."""
    n = len(q)
    tree = cKDTree(q[:, :2])
    pairs = np.array(list(tree.query_pairs(radius)), dtype=np.int32)
    if len(pairs) == 0:
        return np.full(n, np.inf), np.arange(n), pairs

    i_idx, j_idx = pairs[:, 0], pairs[:, 1]
    xy, th = q[:, :2], q[:, 2]
    c, s = np.cos(th), np.sin(th)
    u, v = np.column_stack((c, s)), np.column_stack((-s, c))
    d = xy[j_idx] - xy[i_idx]

    axes = np.stack([u[i_idx], v[i_idx], u[j_idx], v[j_idx]], axis=1)  # (P, 4, 2)
    ri = 0.5 * (
        np.abs(np.einsum("pkd,pd->pk", axes, u[i_idx]))
        + np.abs(np.einsum("pkd,pd->pk", axes, v[i_idx]))
    )
    rj = 0.5 * (
        np.abs(np.einsum("pkd,pd->pk", axes, u[j_idx]))
        + np.abs(np.einsum("pkd,pd->pk", axes, v[j_idx]))
    )
    proj = np.abs(np.einsum("pkd,pd->pk", axes, d))
    gaps = np.max(proj - ri - rj, axis=1)

    margins = np.full(n, np.inf)
    for k, (i, j) in enumerate(pairs):
        g = gaps[k]
        if g < margins[i]:
            margins[i] = g
        if g < margins[j]:
            margins[j] = g

    tight_order = np.argsort(margins)
    return margins, tight_order, pairs


# ---------------------------------------------------------------------------
# 4-Sign-Piece Epigraph LP Polisher
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
) -> Tuple[np.ndarray | None, str]:
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
    radius = SQRT2 + 2.0 * SQRT2 * position_trust + 1e-4
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
    max_iters: int = 12,
    position_trust: float = 0.05,
    angle_trust: float = 0.005,
    clearance: float = 1e-8,
) -> Tuple[np.ndarray, float, bool, float]:
    """Sequential LP polishing using 4-sign-piece epigraph relaxation."""
    best_q = normalize_box(q)
    best_L = extent(best_q)
    axes: dict[tuple[int, int], Any] = {}

    # Initial repair step to resolve any slight overlap
    repaired, _ = _lp_step(best_q, best_L + 0.20, position_trust, 0.0, clearance, axes)
    if repaired is not None:
        valid, min_g = sat_check(repaired, clearance=-1e-7)
        if valid:
            best_q = normalize_box(repaired)
            best_L = extent(best_q)

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
            for bt in range(4):
                alpha = 2.0 ** (-bt)
                seed = origin + alpha * (proposal - origin)
                valid, _ = sat_check(seed, clearance=-1e-7)
                cur_L = extent(seed)
                if valid and cur_L < best_L:
                    best_q = normalize_box(seed)
                    best_L = cur_L
                    break

        if old_L - best_L < 1e-8 and it >= 2:
            break

    is_valid, final_min_gap = sat_check(best_q, clearance=-1e-7)
    return best_q, best_L, is_valid, final_min_gap


# ---------------------------------------------------------------------------
# L-BFGS Relaxation with Smooth Quadratic Overlap Penalty
# ---------------------------------------------------------------------------

def lbfgs_relax_kicked_squares(
    q: np.ndarray,
    kicked_indices: np.ndarray,
    target_L: float,
    maxiter: int = 50,
) -> Tuple[np.ndarray, float]:
    """Run L-BFGS to relieve overlap and wall penalties on kicked squares."""
    n = len(q)
    k_len = len(kicked_indices)
    k_set = set(kicked_indices)

    # KD-Tree to find neighbor pairs involving at least one kicked square
    tree = cKDTree(q[:, :2])
    all_pairs = np.array(list(tree.query_pairs(1.7)), dtype=np.int32)
    if len(all_pairs) == 0:
        return q.copy(), 0.0

    mask = np.isin(all_pairs[:, 0], kicked_indices) | np.isin(all_pairs[:, 1], kicked_indices)
    active_pairs = all_pairs[mask]
    if len(active_pairs) == 0:
        active_pairs = all_pairs

    i_idx, j_idx = active_pairs[:, 0], active_pairs[:, 1]
    base_q = q.copy()

    def penalty_fn(z_sub: np.ndarray) -> float:
        arr = base_q.copy()
        arr[kicked_indices] = z_sub.reshape(k_len, 3)
        xy, th = arr[:, :2], arr[:, 2]
        c, s = np.cos(th), np.sin(th)
        u, v = np.column_stack((c, s)), np.column_stack((-s, c))
        d = xy[j_idx] - xy[i_idx]

        axes = np.stack([u[i_idx], v[i_idx], u[j_idx], v[j_idx]], axis=1)
        ri = 0.5 * (
            np.abs(np.einsum("pkd,pd->pk", axes, u[i_idx]))
            + np.abs(np.einsum("pkd,pd->pk", axes, v[i_idx]))
        )
        rj = 0.5 * (
            np.abs(np.einsum("pkd,pd->pk", axes, u[j_idx]))
            + np.abs(np.einsum("pkd,pd->pk", axes, v[j_idx]))
        )
        proj = np.abs(np.einsum("pkd,pd->pk", axes, d))
        depth = np.maximum(0.0, np.min(ri + rj - proj, axis=1))

        # Container boundary wall penalties for kicked squares
        h_sub = 0.5 * (np.abs(c[kicked_indices]) + np.abs(s[kicked_indices]))
        xy_sub = xy[kicked_indices]
        wall_pen = (
            np.maximum(0.0, h_sub - xy_sub[:, 0]) ** 2
            + np.maximum(0.0, xy_sub[:, 0] + h_sub - target_L) ** 2
            + np.maximum(0.0, h_sub - xy_sub[:, 1]) ** 2
            + np.maximum(0.0, xy_sub[:, 1] + h_sub - target_L) ** 2
        )
        return float(np.sum(depth**2) + 10.0 * np.sum(wall_pen))

    z0 = base_q[kicked_indices].ravel()
    res = minimize(
        penalty_fn,
        z0,
        method="L-BFGS-B",
        options={"maxiter": maxiter, "ftol": 1e-6},
    )
    relaxed_q = base_q.copy()
    relaxed_q[kicked_indices] = res.x.reshape(k_len, 3)
    return relaxed_q, float(res.fun)


# ---------------------------------------------------------------------------
# Base Seeds & Mixed Structure Generators
# ---------------------------------------------------------------------------

def load_seed1() -> Tuple[np.ndarray, float]:
    """Load Seed 1: Idea 1 scratch band from large_n_scratch_band_results.json."""
    path = DATA_DIR / "large_n_scratch_band_results.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    t295 = data["targets"]["295"]["layout"]
    q = np.array([[sq["x"], sq["y"], sq["angle"]] for sq in t295], dtype=float)
    return normalize_box(q), extent(q)


def load_seed2() -> Tuple[np.ndarray, float]:
    """Load Seed 2: Incumbent packing from pell_corrugation/data/border-scan/merged.csv."""
    path = BORDER_SCAN_DIR / "merged.csv"
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        rows_295 = [r for r in reader if len(r) > 1 and r[1] == str(TARGET_N)]
    if len(rows_295) != TARGET_N:
        raise ValueError(f"Expected {TARGET_N} rows for n={TARGET_N}, found {len(rows_295)}")
    q = np.array([[float(r[3]), float(r[4]), float(r[5])] for r in rows_295], dtype=float)
    return normalize_box(q), extent(q)


def generate_mixed_structure_corner_blocks(q_base: np.ndarray) -> Tuple[np.ndarray, float]:
    """Mixed Structure 1: Central 45-deg diagonal band with tilted corner blocks."""
    q = normalize_box(q_base.copy())
    L = extent(q)
    # Identify SW corner upright squares (x <= 1.5, y <= 1.5)
    sw_mask = (q[:, 0] <= 1.55) & (q[:, 1] <= 1.55) & (np.abs(q[:, 2]) < 1e-4)
    sw_idx = np.where(sw_mask)[0]
    if len(sw_idx) > 0:
        # Tilt SW corner block to pi/4
        q[sw_idx, 2] = math.pi / 4.0

    # Also tilt NE corner upright block if present
    ne_mask = (q[:, 0] >= L - 1.55) & (q[:, 1] >= L - 1.55) & (np.abs(q[:, 2]) < 1e-4)
    ne_idx = np.where(ne_mask)[0]
    if len(ne_idx) > 0:
        q[ne_idx, 2] = math.pi / 4.0

    # Polish with LP to make strictly feasible
    q_pol, L_pol, _, _ = polish_lp(q, max_iters=5, position_trust=0.05, angle_trust=0.005)
    return q_pol, L_pol


def generate_mixed_structure_dual_angle(q_base: np.ndarray) -> Tuple[np.ndarray, float]:
    """Mixed Structure 2: Central 45-deg diagonal band with dual-angle boundary staircases."""
    q = normalize_box(q_base.copy())
    tilted_mask = np.abs(q[:, 2]) > 0.1
    upright_mask = ~tilted_mask

    if np.any(tilted_mask) and np.any(upright_mask):
        tree = cKDTree(q[tilted_mask, :2])
        dists, _ = tree.query(q[upright_mask, :2])
        upright_indices = np.where(upright_mask)[0]
        # Identify interface teeth squares within 1.45 units of the band
        boundary_teeth = upright_indices[dists < 1.45]
        # Tilt alternate boundary staircase squares to secondary angle pi/8
        q[boundary_teeth[::2], 2] = math.pi / 8.0

    # Polish with LP to establish feasible contact network
    q_pol, L_pol, _, _ = polish_lp(q, max_iters=5, position_trust=0.05, angle_trust=0.005)
    return q_pol, L_pol


# ---------------------------------------------------------------------------
# Basin Hopping & Relaxation Runner
# ---------------------------------------------------------------------------

def run_basin_hopping_trial(
    base_q: np.ndarray,
    base_name: str,
    kick_seed: int,
    sigma_x: float = 0.05,
    sigma_theta: float = 0.03,
) -> dict[str, Any]:
    """Execute a single basin hopping trial on a base packing."""
    rng = np.random.default_rng(kick_seed)
    q_start = normalize_box(base_q.copy())
    L_start = extent(q_start)

    # 1. Identify 5-15 squares with tightest SAT contact margins
    margins, tight_order, pairs = compute_sat_margins(q_start)
    k_kicked = int(rng.integers(5, 16))
    kicked_indices = tight_order[:k_kicked]

    # 2. Apply random coordinate kicks and small angular perturbations
    q_kicked = q_start.copy()
    q_kicked[kicked_indices, 0] += rng.normal(0.0, sigma_x, size=k_kicked)
    q_kicked[kicked_indices, 1] += rng.normal(0.0, sigma_x, size=k_kicked)
    q_kicked[kicked_indices, 2] += rng.normal(0.0, sigma_theta, size=k_kicked)

    # 3. L-BFGS with smooth quadratic overlap penalty
    q_relaxed, lbfgs_loss = lbfgs_relax_kicked_squares(
        q_kicked, kicked_indices, target_L=L_start, maxiter=40
    )
    L_relaxed = extent(q_relaxed)

    # 4. Polish with 4-sign-piece epigraph LP
    q_polished, L_polished, is_valid, min_gap = polish_lp(
        q_relaxed, max_iters=10, position_trust=0.05, angle_trust=0.005, clearance=1e-8
    )

    improved = is_valid and (L_polished < TARGET_INCUMBENT_L - 1e-8)

    return {
        "base_seed": base_name,
        "kick_seed": kick_seed,
        "k_kicked": k_kicked,
        "kicked_indices": kicked_indices.tolist(),
        "L_start": float(L_start),
        "lbfgs_loss": float(lbfgs_loss),
        "L_relaxed": float(L_relaxed),
        "L_polished": float(L_polished),
        "delta_vs_incumbent": float(L_polished - TARGET_INCUMBENT_L),
        "is_valid_sat": bool(is_valid),
        "sat_min_margin": float(min_gap),
        "improved": bool(improved),
        "layout": [
            {
                "id": int(i),
                "x": float(q_polished[i, 0]),
                "y": float(q_polished[i, 1]),
                "angle": float(q_polished[i, 2]),
            }
            for i in range(len(q_polished))
        ],
    }


def main():
    print("=" * 70)
    print("MULTI-SEED BASIN HOPPING & MIXED RELAXATION FOR N=295")
    print("=" * 70)
    print(f"Target incumbent L : {TARGET_INCUMBENT_L:.6f}")
    print(f"Scratch band L     : {SCRATCH_BAND_L:.6f} (gap +{SCRATCH_BAND_L - TARGET_INCUMBENT_L:.6f})")

    start_time = time.time()

    # 1. Load Base Seeds
    print("\n[1/4] Loading base seeds...")
    q_seed1, L_seed1 = load_seed1()
    print(f"  Loaded Seed 1 (Idea 1 scratch band): L = {L_seed1:.6f}, N = {len(q_seed1)}")

    q_seed2, L_seed2 = load_seed2()
    print(f"  Loaded Seed 2 (Border-scan incumbent): L = {L_seed2:.6f}, N = {len(q_seed2)}")

    # 2. Generate Mixed Structures
    print("\n[2/4] Constructing mixed structures...")
    q_mix1, L_mix1 = generate_mixed_structure_corner_blocks(q_seed1)
    print(f"  Mixed Structure 1 (Tilted corner blocks): L = {L_mix1:.6f}")

    q_mix2, L_mix2 = generate_mixed_structure_dual_angle(q_seed1)
    print(f"  Mixed Structure 2 (Dual-angle boundary staircases): L = {L_mix2:.6f}")

    base_configs = [
        ("seed1_scratch_band", q_seed1),
        ("seed2_incumbent", q_seed2),
        ("mixed_corner_blocks", q_mix1),
        ("mixed_dual_angle", q_mix2),
    ]

    # 3. Multi-Seed Basin Hopping Execution (30 seeds per base)
    n_seeds_per_base = 30
    total_trials = len(base_configs) * n_seeds_per_base
    print(f"\n[3/4] Running basin hopping across {total_trials} total trials ({n_seeds_per_base} per base)...")

    all_trials: list[dict[str, Any]] = []
    best_overall_L = float("inf")
    best_overall_seed = ""
    best_overall_layout: list[dict[str, Any]] = []
    best_per_base: dict[str, float] = {}

    trial_counter = 0
    for base_name, base_q in base_configs:
        best_base_L = float("inf")
        print(f"\n  Starting base: {base_name} (initial L = {extent(base_q):.6f})")
        for s in range(n_seeds_per_base):
            trial_counter += 1
            res = run_basin_hopping_trial(base_q, base_name, kick_seed=1000 + s)
            trial_record = {k: v for k, v in res.items() if k != "layout"}
            all_trials.append(trial_record)

            cur_L = res["L_polished"]
            if res["is_valid_sat"] and cur_L < best_base_L:
                best_base_L = cur_L
            if res["is_valid_sat"] and cur_L < best_overall_L:
                best_overall_L = cur_L
                best_overall_seed = f"{base_name}_seed_{1000 + s}"
                best_overall_layout = res["layout"]

            if (s + 1) % 10 == 0:
                print(f"    Completed trial {s + 1:2d}/{n_seeds_per_base} | Best L so far: {best_base_L:.6f}")

        best_per_base[base_name] = best_base_L

    elapsed = time.time() - start_time
    print(f"\n[4/4] Basin hopping and mixed relaxation completed in {elapsed:.1f} s.")

    # Check for strict improvement
    strict_improvement = best_overall_L < TARGET_INCUMBENT_L - 1e-8
    cert_details = None
    if strict_improvement:
        print("\n" + "*" * 70)
        print(f"STRICT IMPROVEMENT FOUND: L = {best_overall_L:.8f} < {TARGET_INCUMBENT_L:.8f}")
        print("*" * 70)
        # Format for official decimal certifier
        packing_rows = [
            {
                "square_id": str(sq["id"]),
                "x": f"{sq['x']:.16f}",
                "y": f"{sq['y']:.16f}",
                "angle": f"{sq['angle']:.16f}",
            }
            for sq in best_overall_layout
        ]
        n_eval, _, side_str, reason = evaluate(TARGET_N, packing_rows)
        cert_details = {
            "n": n_eval,
            "certified_L": side_str,
            "reason": reason,
            "delta_vs_incumbent": float(best_overall_L - TARGET_INCUMBENT_L),
        }
        print(f"  Official Certifier: {reason}, Certified side: {side_str}")

    # Assemble Output JSON
    output_data = {
        "target_n": TARGET_N,
        "target_incumbent_L": float(TARGET_INCUMBENT_L),
        "candidate_scratch_band_L": float(SCRATCH_BAND_L),
        "initial_gap": float(SCRATCH_BAND_L - TARGET_INCUMBENT_L),
        "elapsed_seconds": float(elapsed),
        "total_trials": len(all_trials),
        "seeds_per_base": n_seeds_per_base,
        "best_per_base": {k: float(v) for k, v in best_per_base.items()},
        "best_overall_L": float(best_overall_L),
        "best_overall_seed": best_overall_seed,
        "strict_improvement_found": bool(strict_improvement),
        "certification_details": cert_details,
        "best_layout": best_overall_layout if strict_improvement else None,
        "trials": all_trials,
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults successfully written to {RESULTS_PATH}")

    # Summary table
    print("\n" + "=" * 70)
    print("FINAL SUMMARY REPORT")
    print("=" * 70)
    for base_name, b_L in best_per_base.items():
        delta = b_L - TARGET_INCUMBENT_L
        print(f"  {base_name:24s}: Best L = {b_L:.6f} (gap: {delta:+.6f})")
    print("-" * 70)
    print(f"  Overall Best L          : {best_overall_L:.6f}")
    print(f"  Target Incumbent L      : {TARGET_INCUMBENT_L:.6f}")
    print(f"  Strict Improvement Found: {strict_improvement}")
    print("=" * 70)


if __name__ == "__main__":
    main()
