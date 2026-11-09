"""Multi-seed basin hopping and mixed relaxation for target n=151.

Evaluates:
1. Seed 1: Idea 1 scratch band from poc/research/data/large_n_scratch_band_results.json
2. Seed 2: Incumbent packing from poc/pell_corrugation/data/border-scan/merged.csv
3. Basin hopping: Identifies 5-15 critical contact squares, applies Gaussian kicks to
   positions and angles, resolves contact/overlap stress via smooth penalty L-BFGS, and
   polishes container side L with 4-sign-piece epigraph LP.
4. Mixed structures: Combines the incumbent dual-angle band (theta ~ 24.3 deg and 45 deg)
   with continuous band offsets and boundary corner refilling.
5. Runs across at least 30 random kick seeds per base seed, tracks best L per seed and
   method, and writes results to poc/research/data/relax_n151_results.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linprog, minimize
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parents[3]
RESEARCH_POC = REPO_ROOT / "poc/research/poc"
if str(RESEARCH_POC) not in sys.path:
    sys.path.insert(0, str(RESEARCH_POC))

# 4-sign piece definitions for epigraph linearization
SIGNS = [(-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)]
S2 = math.sqrt(2.0)


def extent(q: np.ndarray) -> float:
    """Compute minimal square container side length enclosing all unit squares."""
    c = np.abs(np.cos(q[:, 2]))
    s = np.abs(np.sin(q[:, 2]))
    r = 0.5 * (c + s)
    xmin = np.min(q[:, 0] - r)
    xmax = np.max(q[:, 0] + r)
    ymin = np.min(q[:, 1] - r)
    ymax = np.max(q[:, 1] + r)
    return float(max(xmax - xmin, ymax - ymin))


def normalize_box(q: np.ndarray, margin: float = 0.0) -> np.ndarray:
    """Translate configuration so its bounding box is anchored at (margin, margin)."""
    out = q.copy()
    c = np.abs(np.cos(out[:, 2]))
    s = np.abs(np.sin(out[:, 2]))
    r = 0.5 * (c + s)
    xmin = np.min(out[:, 0] - r)
    ymin = np.min(out[:, 1] - r)
    out[:, 0] -= xmin - margin
    out[:, 1] -= ymin - margin
    return out


def sat_check(q: np.ndarray, clearance: float = 0.0) -> Tuple[bool, float]:
    """Exhaustive all-pairs SAT separation check."""
    n = len(q)
    if n <= 1:
        return True, float("inf")
    c = np.cos(q[:, 2])
    s = np.sin(q[:, 2])
    min_gap = float("inf")

    for i in range(n):
        dx = q[i + 1 :, 0] - q[i, 0]
        dy = q[i + 1 :, 1] - q[i, 1]
        dist2 = dx * dx + dy * dy
        cand = np.where(dist2 < 8.01)[0]
        if len(cand) == 0:
            continue
        cand_j = i + 1 + cand
        dx_c = dx[cand]
        dy_c = dy[cand]

        best = np.full(len(cand), -float("inf"))
        for phi in (q[i, 2], q[i, 2] + math.pi / 2.0):
            cp, sp = math.cos(phi), math.sin(phi)
            proj = np.abs(dx_c * cp + dy_c * sp)
            h_i = 0.5 * (abs(math.cos(q[i, 2] - phi)) + abs(math.sin(q[i, 2] - phi)))
            h_j = 0.5 * (np.abs(np.cos(q[cand_j, 2] - phi)) + np.abs(np.sin(q[cand_j, 2] - phi)))
            best = np.maximum(best, proj - (h_i + h_j))

        for offset in (0.0, math.pi / 2.0):
            phi_j = q[cand_j, 2] + offset
            cp = np.cos(phi_j)
            sp = np.sin(phi_j)
            proj = np.abs(dx_c * cp + dy_c * sp)
            h_i = 0.5 * (np.abs(np.cos(q[i, 2] - phi_j)) + np.abs(np.sin(q[i, 2] - phi_j)))
            h_j = 0.5 * (np.abs(np.cos(q[cand_j, 2] - phi_j)) + np.abs(np.sin(q[cand_j, 2] - phi_j)))
            best = np.maximum(best, proj - (h_i + h_j))

        cur_min = float(np.min(best))
        if cur_min < min_gap:
            min_gap = cur_min
        if np.any(best < clearance):
            return False, min_gap

    return True, min_gap


# ---------------------------------------------------------------------------
# 4-Sign-Piece Epigraph Linearization and LP Step
# ---------------------------------------------------------------------------

def chosen_axis(q: np.ndarray, i: int, j: int, previous=None, hysteresis: float = 1e-9):
    choices = []
    for owner in (i, j):
        for k in (0, 1):
            direction = 1 if owner == i else -1
            phi = float(q[owner, 2]) + k * math.pi / 2.0
            a = direction * np.array([math.cos(phi), math.sin(phi)])
            d = q[j, :2] - q[i, :2]
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
) -> Tuple[Optional[np.ndarray], str]:
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

    # Bounding box epigraph constraints for each square
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
    tree = cKDTree(q[:, :2])
    pairs = sorted(tree.query_pairs(radius))

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
            phi += 0.0 if direction == 1 else math.pi
            other = j if owner == i else i
            r = float(q[other, 2]) - phi
            h_j = 0.5 * (abs(math.cos(r)) + abs(math.sin(r)))
            d = q[j, :2] - q[i, :2]
            g = float(d @ a - 0.5 - h_j)
            row({stride * i: a[0], stride * i + 1: a[1], stride * j: -a[0], stride * j + 1: -a[1]}, g - clearance)

    bounds = []
    for _ in range(n):
        bounds.extend([(-position_trust, position_trust), (-position_trust, position_trust)])
        if free:
            bounds.append((-angle_trust, angle_trust))

    bounds.extend([(None, None), (None, None), (1.0, upper_L)])
    objective = np.zeros(nv)
    objective[-1] = 1.0

    a_mat = coo_matrix((vv, (rr, cc)), shape=(len(rhs), nv)).tocsr()
    res = linprog(
        objective,
        A_ub=a_mat,
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
    position_trust: float = 0.08,
    angle_trust: float = 0.008,
    clearance: float = 1e-8,
) -> Tuple[np.ndarray, float]:
    """Sequential LP polishing using 4-sign-piece epigraph relaxation."""
    best_q = normalize_box(q)
    best_L = extent(best_q)
    axes: dict = {}

    for _ in range(max_iters):
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

        if abs(best_L - old_L) < 1e-7:
            break

    return best_q, best_L


# ---------------------------------------------------------------------------
# Critical Contact Identification & Smooth Penalty L-BFGS
# ---------------------------------------------------------------------------

def identify_critical_contact_squares(
    q: np.ndarray,
    k_min: int = 5,
    k_max: int = 15,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Identify 5-15 critical contact squares based on active wall and pairwise contacts."""
    n = len(q)
    c = np.cos(q[:, 2])
    s = np.sin(q[:, 2])
    r = 0.5 * (np.abs(c) + np.abs(s))
    xmin, xmax = np.min(q[:, 0] - r), np.max(q[:, 0] + r)
    ymin, ymax = np.min(q[:, 1] - r), np.max(q[:, 1] + r)

    wall_dist = np.minimum(
        np.minimum(q[:, 0] - r - xmin, xmax - (q[:, 0] + r)),
        np.minimum(q[:, 1] - r - ymin, ymax - (q[:, 1] + r)),
    )
    scores = (wall_dist < 0.05).astype(float) * 2.0

    for i in range(n):
        dx = q[:, 0] - q[i, 0]
        dy = q[:, 1] - q[i, 1]
        dist2 = dx * dx + dy * dy
        near = np.where((dist2 > 1e-6) & (dist2 < 8.01))[0]
        for j in near:
            phi_i = [q[i, 2], q[i, 2] + math.pi / 2.0]
            phi_j = [q[j, 2], q[j, 2] + math.pi / 2.0]
            gaps = []
            for phi in phi_i + phi_j:
                cp, sp = math.cos(phi), math.sin(phi)
                proj = abs(dx[j] * cp + dy[j] * sp)
                hi = 0.5 * (abs(math.cos(q[i, 2] - phi)) + abs(math.sin(q[i, 2] - phi)))
                hj = 0.5 * (abs(math.cos(q[j, 2] - phi)) + abs(math.sin(q[j, 2] - phi)))
                gaps.append(proj - (hi + hj))
            if max(gaps) < 0.05:
                scores[i] += 1.0

    ranked = np.argsort(-scores)
    k = rng.integers(k_min, k_max + 1) if rng is not None else 10
    return ranked[:k]


def build_active_pairs(q: np.ndarray, active_indices: np.ndarray) -> np.ndarray:
    """Precompute pairs involving active squares within collision radius."""
    tree = cKDTree(q[:, :2])
    pairs = set()
    for a in active_indices:
        for nb in tree.query_ball_point(q[a, :2], r=2.83):
            if nb != a:
                pairs.add(tuple(sorted((int(a), int(nb)))))
    if not pairs:
        return np.zeros((0, 2), dtype=int)
    return np.array(list(pairs), dtype=int)


def smooth_penalty_objective(
    vars_flat: np.ndarray,
    q_fixed: np.ndarray,
    active_indices: np.ndarray,
    pairs: np.ndarray,
    side: float,
) -> float:
    """Fast SAT penetration depth and container bounding penalty."""
    q_curr = q_fixed.copy()
    q_curr[active_indices] = vars_flat.reshape(-1, 3)
    xy = q_curr[:, :2]
    angle = q_curr[:, 2]
    c, s = np.cos(angle), np.sin(angle)
    u = np.stack([c, s], axis=1)
    v = np.stack([-s, c], axis=1)

    if len(pairs) == 0:
        depth_sq = 0.0
    else:
        i, j = pairs[:, 0], pairs[:, 1]
        delta = xy[j] - xy[i]
        u_i, v_i = u[i], v[i]
        u_j, v_j = u[j], v[j]
        axes = np.stack([u_i, v_i, u_j, v_j], axis=1)
        ri = 0.5 * (np.abs(np.einsum("pkd,pd->pk", axes, u_i)) + np.abs(np.einsum("pkd,pd->pk", axes, v_i)))
        rj = 0.5 * (np.abs(np.einsum("pkd,pd->pk", axes, u_j)) + np.abs(np.einsum("pkd,pd->pk", axes, v_j)))
        proj = np.abs(np.einsum("pkd,pd->pk", axes, delta))
        depth = np.maximum(0.0, np.min(ri + rj - proj, axis=1))
        depth_sq = float(np.sum(depth**2))

    ext = 0.5 * (np.abs(c[active_indices]) + np.abs(s[active_indices]))
    walls = np.maximum(0.0, np.abs(xy[active_indices] - side / 2.0) + ext[:, None] - side / 2.0)
    return depth_sq + float(np.sum(walls**2))


# ---------------------------------------------------------------------------
# Basin Hopping & Mixed Relaxation Pipelines
# ---------------------------------------------------------------------------

def run_basin_hopping_iteration(
    base_q: np.ndarray,
    seed_idx: int,
    k_min: int = 5,
    k_max: int = 15,
    pos_sigma: float = 0.012,
    ang_sigma: float = 0.008,
    maxiter_lbfgs: int = 25,
) -> Dict[str, Any]:
    """Execute a single basin-hopping perturbation, L-BFGS relaxation, and LP polish."""
    rng = np.random.default_rng(seed_idx + 1000)
    q_norm = normalize_box(base_q)
    side = extent(q_norm)

    # 1. Identify 5-15 critical contact squares
    crit_squares = identify_critical_contact_squares(q_norm, k_min=k_min, k_max=k_max, rng=rng)

    # 2. Apply Gaussian kick to positions and angles
    q_kicked = q_norm.copy()
    k = len(crit_squares)
    q_kicked[crit_squares, :2] += rng.normal(0.0, pos_sigma, size=(k, 2))
    q_kicked[crit_squares, 2] += rng.normal(0.0, ang_sigma, size=k)

    # 3. Form active cluster (kicked squares + immediate neighbors)
    tree = cKDTree(q_kicked[:, :2])
    active_set = set(crit_squares)
    for c in crit_squares:
        for nb in tree.query_ball_point(q_kicked[c, :2], r=2.2):
            active_set.add(nb)
    active = np.array(sorted(active_set))
    pairs = build_active_pairs(q_kicked, active)

    # 4. Smooth penalty L-BFGS relaxation
    res = minimize(
        smooth_penalty_objective,
        q_kicked[active].ravel(),
        args=(q_kicked, active, pairs, side + 0.001),
        method="L-BFGS-B",
        options=dict(maxiter=maxiter_lbfgs),
    )
    q_relaxed = q_kicked.copy()
    q_relaxed[active] = res.x.reshape(-1, 3)
    q_relaxed = normalize_box(q_relaxed)

    # 5. 4-sign-piece epigraph LP polish
    q_pol, l_pol = polish_lp(q_relaxed, max_iters=12)
    valid, min_gap = sat_check(q_pol, clearance=-1e-7)

    return {
        "kick_seed": seed_idx,
        "num_critical_squares": int(k),
        "critical_squares": [int(x) for x in crit_squares],
        "active_cluster_size": int(len(active)),
        "lbfgs_fun": float(res.fun),
        "lbfgs_nit": int(res.nit),
        "polished_L": float(l_pol),
        "sat_valid": bool(valid),
        "min_gap": float(min_gap),
        "q_polished": q_pol if valid else None,
    }


def run_mixed_structure_iteration(
    incumbent_q: np.ndarray,
    seed_idx: int,
    offset_scale: float = 0.008,
) -> Dict[str, Any]:
    """Test mixed structures: incumbent dual-angle band with continuous offsets and corner refilling."""
    rng = np.random.default_rng(seed_idx + 2000)
    q_norm = normalize_box(incumbent_q)
    side = extent(q_norm)

    angles_deg = np.degrees(q_norm[:, 2]) % 90
    tilted_mask = angles_deg > 5.0
    band_indices = np.where(tilted_mask)[0]

    # Continuous band offset
    dx = float(rng.uniform(-offset_scale, offset_scale))
    dy = float(rng.uniform(-offset_scale, offset_scale))
    dth = float(rng.uniform(-0.003, 0.003))

    q_cand = q_norm.copy()
    q_cand[band_indices, 0] += dx
    q_cand[band_indices, 1] += dy
    q_cand[band_indices, 2] += dth

    # Active cluster: tilted band + boundary uprights
    tree = cKDTree(q_cand[:, :2])
    active_set = set(band_indices)
    for b in band_indices:
        for nb in tree.query_ball_point(q_cand[b, :2], r=2.2):
            active_set.add(nb)
    active = np.array(sorted(active_set))
    pairs = build_active_pairs(q_cand, active)

    # Smooth penalty L-BFGS to relieve interface overlap
    res = minimize(
        smooth_penalty_objective,
        q_cand[active].ravel(),
        args=(q_cand, active, pairs, side + 0.001),
        method="L-BFGS-B",
        options=dict(maxiter=25),
    )
    q_relaxed = q_cand.copy()
    q_relaxed[active] = res.x.reshape(-1, 3)
    q_relaxed = normalize_box(q_relaxed)

    # 4-sign-piece epigraph LP polish
    q_pol, l_pol = polish_lp(q_relaxed, max_iters=12)
    valid, min_gap = sat_check(q_pol, clearance=-1e-7)

    return {
        "variation_seed": seed_idx,
        "band_offset_dx": dx,
        "band_offset_dy": dy,
        "band_offset_dth": dth,
        "active_cluster_size": int(len(active)),
        "lbfgs_fun": float(res.fun),
        "lbfgs_nit": int(res.nit),
        "polished_L": float(l_pol),
        "sat_valid": bool(valid),
        "min_gap": float(min_gap),
        "q_polished": q_pol if valid else None,
    }


# ---------------------------------------------------------------------------
# Data Loading & Main Runner
# ---------------------------------------------------------------------------

def load_seeds(
    scratch_path: Path, incumbent_path: Path
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Load Seed 1 (scratch band) and Seed 2 (incumbent)."""
    # Seed 1
    with open(scratch_path, "r", encoding="utf-8") as f:
        scratch_data = json.load(f)
    t151 = scratch_data["targets"]["151"]
    q1 = np.array([[sq["x"], sq["y"], sq["angle"]] for sq in t151["layout"]])
    q1 = normalize_box(q1)
    l1 = extent(q1)

    # Seed 2
    q2_list = []
    with open(incumbent_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["n"] == "151":
                q2_list.append([float(row["x"]), float(row["y"]), float(row["angle"])])
    q2 = normalize_box(np.array(q2_list))
    l2 = extent(q2)

    return q1, q2, l1, l2


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-seed basin hopping and mixed relaxation for n=151")
    parser.add_argument("--num-kicks", type=int, default=30, help="Number of kick seeds per base seed (default: 30)")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "poc/research/data/relax_n151_results.json")
    parser.add_argument("--scratch-path", type=Path, default=REPO_ROOT / "poc/research/data/large_n_scratch_band_results.json")
    parser.add_argument("--incumbent-path", type=Path, default=REPO_ROOT / "poc/pell_corrugation/data/border-scan/merged.csv")
    args = parser.parse_args()

    t_start = time.time()
    print("=" * 70)
    print("Multi-Seed Basin Hopping and Mixed Relaxation for n=151")
    print("=" * 70)

    # 1. Load Seeds
    q_seed1, q_seed2, l_seed1, l_seed2 = load_seeds(args.scratch_path, args.incumbent_path)
    print(f"Loaded Seed 1 (Scratch Band): n={len(q_seed1)}, initial L={l_seed1:.8f}")
    print(f"Loaded Seed 2 (Incumbent):    n={len(q_seed2)}, initial L={l_seed2:.8f}")
    target_incumbent_L = 12.822875655532307
    print(f"Target Incumbent L: {target_incumbent_L:.8f}, Gap: {l_seed1 - target_incumbent_L:+.8f}")

    # Baseline direct LP polishes
    print("\n--- Baseline Direct LP Polishes ---")
    pol_q1, pol_l1 = polish_lp(q_seed1, max_iters=15)
    v1, g1 = sat_check(pol_q1, clearance=-1e-7)
    print(f"Seed 1 Direct LP Polish: L={pol_l1:.8f}, sat_valid={v1}, min_gap={g1:.2e}")

    pol_q2, pol_l2 = polish_lp(q_seed2, max_iters=15)
    v2, g2 = sat_check(pol_q2, clearance=-1e-7)
    print(f"Seed 2 Direct LP Polish: L={pol_l2:.8f}, sat_valid={v2}, min_gap={g2:.2e}")

    best_overall_L = min(pol_l1 if v1 else float("inf"), pol_l2 if v2 else float("inf"))
    best_overall_q = pol_q2 if pol_l2 <= pol_l1 else pol_q1
    best_overall_method = "direct_lp_seed2" if pol_l2 <= pol_l1 else "direct_lp_seed1"

    num_kicks = args.num_kicks

    # 2. Basin Hopping on Seed 1
    print(f"\n--- Method 1: Basin Hopping on Seed 1 ({num_kicks} kicks) ---")
    seed1_runs = []
    best_s1_L = pol_l1 if v1 else float("inf")
    for s in range(num_kicks):
        t0 = time.time()
        res = run_basin_hopping_iteration(q_seed1, seed_idx=s, k_min=5, k_max=15)
        dt = time.time() - t0
        q_pol = res.pop("q_polished")
        seed1_runs.append(res)
        if res["sat_valid"] and res["polished_L"] < best_s1_L:
            best_s1_L = res["polished_L"]
        if res["sat_valid"] and res["polished_L"] < best_overall_L:
            best_overall_L = res["polished_L"]
            best_overall_q = q_pol
            best_overall_method = f"basin_hopping_seed1_kick_{s}"
        print(f"  Kick {s:02d}: crit_sq={res['num_critical_squares']}, L={res['polished_L']:.8f}, valid={res['sat_valid']}, time={dt:.2f}s")

    print(f"Method 1 Best L: {best_s1_L:.8f}")

    # 3. Basin Hopping on Seed 2
    print(f"\n--- Method 2: Basin Hopping on Seed 2 ({num_kicks} kicks) ---")
    seed2_runs = []
    best_s2_L = pol_l2 if v2 else float("inf")
    for s in range(num_kicks):
        t0 = time.time()
        res = run_basin_hopping_iteration(q_seed2, seed_idx=s, k_min=5, k_max=15)
        dt = time.time() - t0
        q_pol = res.pop("q_polished")
        seed2_runs.append(res)
        if res["sat_valid"] and res["polished_L"] < best_s2_L:
            best_s2_L = res["polished_L"]
        if res["sat_valid"] and res["polished_L"] < best_overall_L:
            best_overall_L = res["polished_L"]
            best_overall_q = q_pol
            best_overall_method = f"basin_hopping_seed2_kick_{s}"
        print(f"  Kick {s:02d}: crit_sq={res['num_critical_squares']}, L={res['polished_L']:.8f}, valid={res['sat_valid']}, time={dt:.2f}s")

    print(f"Method 2 Best L: {best_s2_L:.8f}")

    # 4. Mixed Structures on Incumbent Dual-Angle Band
    print(f"\n--- Method 3: Mixed Structures on Dual-Angle Band ({num_kicks} variations) ---")
    mixed_runs = []
    best_mixed_L = float("inf")
    for s in range(num_kicks):
        t0 = time.time()
        res = run_mixed_structure_iteration(q_seed2, seed_idx=s)
        dt = time.time() - t0
        q_pol = res.pop("q_polished")
        mixed_runs.append(res)
        if res["sat_valid"] and res["polished_L"] < best_mixed_L:
            best_mixed_L = res["polished_L"]
        if res["sat_valid"] and res["polished_L"] < best_overall_L:
            best_overall_L = res["polished_L"]
            best_overall_q = q_pol
            best_overall_method = f"mixed_structure_var_{s}"
        print(f"  Var  {s:02d}: dx={res['band_offset_dx']:+.4f}, dy={res['band_offset_dy']:+.4f}, L={res['polished_L']:.8f}, valid={res['sat_valid']}, time={dt:.2f}s")

    print(f"Method 3 Best L: {best_mixed_L:.8f}")

    elapsed = time.time() - t_start
    beat_incumbent = bool(best_overall_L < target_incumbent_L - 1e-9)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Target Incumbent L:     {target_incumbent_L:.8f}")
    print(f"Seed 1 Baseline L:      {pol_l1:.8f}")
    print(f"Seed 2 Baseline L:      {pol_l2:.8f}")
    print(f"Method 1 (S1 Basin):    {best_s1_L:.8f}")
    print(f"Method 2 (S2 Basin):    {best_s2_L:.8f}")
    print(f"Method 3 (Mixed Dual):  {best_mixed_L:.8f}")
    print(f"Best Overall L:         {best_overall_L:.8f}")
    print(f"Best Overall Method:    {best_overall_method}")
    print(f"Beat Incumbent?         {beat_incumbent}")
    print(f"Total Elapsed Time:     {elapsed:.2f}s")

    # Serialize best layout
    best_layout = [
        {"id": i, "x": float(best_overall_q[i, 0]), "y": float(best_overall_q[i, 1]), "angle": float(best_overall_q[i, 2])}
        for i in range(len(best_overall_q))
    ]

    out_data = {
        "metadata": {
            "target_n": 151,
            "incumbent_L": target_incumbent_L,
            "candidate_scratch_band_L": l_seed1,
            "initial_gap": l_seed1 - target_incumbent_L,
            "num_kicks_per_seed": num_kicks,
            "total_runs": num_kicks * 3,
            "elapsed_seconds": elapsed,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        },
        "summary": {
            "target_incumbent_L": target_incumbent_L,
            "seed1_direct_lp_L": pol_l1,
            "seed2_direct_lp_L": pol_l2,
            "best_s1_basin_L": best_s1_L,
            "best_s2_basin_L": best_s2_L,
            "best_mixed_dual_L": best_mixed_L,
            "best_overall_L": best_overall_L,
            "best_overall_method": best_overall_method,
            "beat_incumbent": beat_incumbent,
        },
        "methods": {
            "seed1_direct_lp": {"L": pol_l1, "sat_valid": v1, "min_gap": g1},
            "seed2_direct_lp": {"L": pol_l2, "sat_valid": v2, "min_gap": g2},
            "basin_hopping_seed1": seed1_runs,
            "basin_hopping_seed2": seed2_runs,
            "mixed_relaxation": mixed_runs,
        },
        "best_packing": best_layout,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)
    print(f"\nWrote results to {args.output}")

    # If candidate beats incumbent, save candidate packing record
    if beat_incumbent:
        rec_path = REPO_ROOT / "poc/research/data/candidate_n151_record.json"
        with open(rec_path, "w", encoding="utf-8") as f:
            json.dump({
                "n": 151,
                "L": best_overall_L,
                "incumbent_L": target_incumbent_L,
                "improvement": target_incumbent_L - best_overall_L,
                "method": best_overall_method,
                "packing": best_layout,
            }, f, indent=2)
        print(f"Saved candidate packing to {rec_path}")


if __name__ == "__main__":
    main()
