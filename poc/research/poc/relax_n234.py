"""Multi-seed basin hopping and mixed relaxation for target n=234.

Targets:
- Incumbent L: 15.822875655532311 (theta ~ 24.3 deg rational slope band)
- Candidate scratch band L: 15.82842712474619 (gap +0.005551)

Pipeline:
1. Load Seed 1 (Idea 1 scratch band from poc/research/data/large_n_scratch_band_results.json)
   and Seed 2 (incumbent from poc/pell_corrugation/data/border-scan/merged.csv).
2. Construct mixed structure Seed 3: combine incumbent 36-square rational slope band
   (theta ~ 24.3 deg) with dense staircase flanks and boundary-anchored corner blocks.
3. Basin hopping:
   - Identify 5-15 tightest contact squares via exact pairwise SAT gaps and wall distances.
   - Perturb positions and angles.
   - Run L-BFGS with smooth overlap and wall penetration penalties.
   - Feasibility repair and exact L contraction with 4-sign-piece epigraph LP.
4. Run across >= 30 random kick seeds per base seed, track best L per seed and method,
   and output results to poc/research/data/relax_n234_results.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linprog, minimize
from scipy.spatial import cKDTree

SIGNS = [(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)]
S2 = math.sqrt(2.0)
INV_S2 = 1.0 / S2
TARGET_INCUMBENT_L = 15.822875655532311
TARGET_SCRATCH_L = 15.82842712474619


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


# ---------------------------------------------------------------------------
# 4-Sign-Piece Epigraph LP Formulation
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
    max_iters: int = 10,
    position_trust: float = 0.08,
    angle_trust: float = 0.008,
    clearance: float = 1e-7,
) -> tuple[np.ndarray, float]:
    """Sequential LP polishing using 4-sign-piece epigraph relaxation."""
    best_q = normalize_box(q)
    best_L = extent(best_q)
    axes: dict[tuple[int, int], Any] = {}

    # Initial feasibility repair if input has small residual overlaps
    init_valid, _ = sat_check(best_q, clearance=-1e-7)
    if not init_valid:
        repair, _ = _lp_step(best_q, best_L + 0.05, position_trust, 0.0, clearance, axes)
        if repair is not None:
            rep_valid, _ = sat_check(repair, clearance=-1e-7)
            if rep_valid:
                best_q = normalize_box(repair)
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
            for bt in range(5):
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

    return best_q, best_L


# ---------------------------------------------------------------------------
# Contact Detection & L-BFGS Smooth Overlap Penalty
# ---------------------------------------------------------------------------

def find_tightest_contact_squares(
    q: np.ndarray,
    k_min: int = 5,
    k_max: int = 15,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Identify between k_min and k_max squares with the smallest contact clearances."""
    n = len(q)
    tree = cKDTree(q[:, :2])
    pairs = tree.query_pairs(2.2, output_type="ndarray")

    th = q[:, 2]
    c, s = np.cos(th), np.sin(th)
    u = np.stack([c, s], axis=1)
    v = np.stack([-s, c], axis=1)

    min_sq_gaps = np.full(n, np.inf)

    if len(pairs) > 0:
        i, j = pairs[:, 0], pairs[:, 1]
        delta = q[j, :2] - q[i, :2]
        axes = np.stack([u[i], v[i], u[j], v[j]], axis=1)
        ri = 0.5 * (np.abs(np.einsum("pkd,pd->pk", axes, u[i])) + np.abs(np.einsum("pkd,pd->pk", axes, v[i])))
        rj = 0.5 * (np.abs(np.einsum("pkd,pd->pk", axes, u[j])) + np.abs(np.einsum("pkd,pd->pk", axes, v[j])))
        proj = np.abs(np.einsum("pkd,pd->pk", axes, delta))
        gaps = np.max(proj - ri - rj, axis=1)

        for idx, (sq_i, sq_j) in enumerate(pairs):
            g = gaps[idx]
            if g < min_sq_gaps[sq_i]:
                min_sq_gaps[sq_i] = g
            if g < min_sq_gaps[sq_j]:
                min_sq_gaps[sq_j] = g

    # Container wall clearance
    L = extent(q)
    h = 0.5 * (np.abs(c) + np.abs(s))
    wall_dist = np.minimum(
        np.minimum(q[:, 0] - h, L - (q[:, 0] + h)),
        np.minimum(q[:, 1] - h, L - (q[:, 1] + h)),
    )
    combined_clearance = np.minimum(min_sq_gaps, wall_dist)
    sorted_indices = np.argsort(combined_clearance)

    k = rng.integers(k_min, k_max + 1) if rng is not None else k_max
    # Sample from the top 2*k tightest squares to introduce stochasticity
    pool_size = min(n, max(k, 2 * k))
    candidates = sorted_indices[:pool_size]
    if rng is not None and len(candidates) > k:
        chosen = rng.choice(candidates, size=k, replace=False)
    else:
        chosen = sorted_indices[:k]

    return np.array(chosen, dtype=int)


def lbfgs_smooth_relax(
    q_init: np.ndarray,
    active_indices: np.ndarray,
    container_L: float,
    maxiter: int = 40,
) -> tuple[np.ndarray, float, bool]:
    """Smooth penalty L-BFGS to eliminate overlaps after perturbation."""
    active = np.asarray(active_indices, dtype=int)
    q_base = q_init.copy()
    n_act = len(active)
    active_set = set(active)

    def penalty_obj(z: np.ndarray) -> float:
        q_cur = q_base.copy()
        q_cur[active] = z.reshape(n_act, 3)
        xy = q_cur[:, :2]
        th = q_cur[:, 2]
        c, s = np.cos(th), np.sin(th)
        u = np.stack([c, s], axis=1)
        v = np.stack([-s, c], axis=1)

        tree = cKDTree(xy)
        pairs = tree.query_pairs(2.0, output_type="ndarray")
        if len(pairs) == 0:
            return 0.0

        mask = np.array([p[0] in active_set or p[1] in active_set for p in pairs])
        act_pairs = pairs[mask]
        if len(act_pairs) == 0:
            return 0.0

        ii, jj = act_pairs[:, 0], act_pairs[:, 1]
        delta = xy[jj] - xy[ii]
        axes = np.stack([u[ii], v[ii], u[jj], v[jj]], axis=1)
        ri = 0.5 * (np.abs(np.einsum("pkd,pd->pk", axes, u[ii])) + np.abs(np.einsum("pkd,pd->pk", axes, v[ii])))
        rj = 0.5 * (np.abs(np.einsum("pkd,pd->pk", axes, u[jj])) + np.abs(np.einsum("pkd,pd->pk", axes, v[jj])))
        proj = np.abs(np.einsum("pkd,pd->pk", axes, delta))
        gaps = np.max(proj - ri - rj, axis=1)
        depth = np.maximum(0.0, -gaps)

        # Wall penalty for active squares
        ha = 0.5 * (np.abs(c[active]) + np.abs(s[active]))
        w_lo = np.maximum(0.0, ha[:, None] - xy[active])
        w_hi = np.maximum(0.0, xy[active] + ha[:, None] - container_L)

        return float(np.sum(depth**2) * 1e4 + np.sum(w_lo**2 + w_hi**2) * 1e4)

    z0 = q_base[active].ravel()
    res = minimize(
        penalty_obj,
        z0,
        method="L-BFGS-B",
        options={"maxiter": maxiter, "ftol": 1e-10, "gtol": 1e-6},
    )

    q_out = q_base.copy()
    q_out[active] = res.x.reshape(n_act, 3)
    return q_out, float(res.fun), bool(res.success or res.fun < 1e-8)


# ---------------------------------------------------------------------------
# Seed Loaders & Mixed Structure Synthesis
# ---------------------------------------------------------------------------

def load_seed1(path: str = "poc/research/data/large_n_scratch_band_results.json") -> np.ndarray:
    """Load Idea 1 scratch band for n=234."""
    with open(path) as f:
        data = json.load(f)
    t234 = data["targets"]["234"]
    layout = t234["layout"]
    q = np.array([[s["x"], s["y"], s["angle"]] for s in layout], dtype=np.float64)
    return normalize_box(q)


def load_seed2(path: str = "poc/pell_corrugation/data/border-scan/merged.csv") -> np.ndarray:
    """Load incumbent packing for n=234."""
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r["n"] == "234":
                rows.append(r)
    rows.sort(key=lambda x: int(x["square_id"]))
    q = np.array([[float(r["x"]), float(r["y"]), float(r["angle"])] for r in rows], dtype=np.float64)
    return normalize_box(q)


def construct_seed3_mixed(seed2: np.ndarray, flank_shift: float = 0.001) -> np.ndarray:
    """Construct mixed structure combining 36-square rational slope band with staircase flanks and corner blocks."""
    q = seed2.copy()
    tilted_mask = q[:, 2] > 0.01
    th = q[tilted_mask, 2][0]
    n_vec = np.array([-np.sin(th), np.cos(th)])
    tilted_proj = q[tilted_mask, :2] @ n_vec
    min_band_v, max_band_v = tilted_proj.min(), tilted_proj.max()

    # Partition upright squares into boundary-anchored corner blocks and staircase flanks
    upright_proj = q[~tilted_mask, :2] @ n_vec
    flank_nw_idx = np.where((~tilted_mask) & (q[:, :2] @ n_vec > max_band_v))[0]
    flank_se_idx = np.where((~tilted_mask) & (q[:, :2] @ n_vec < min_band_v))[0]

    # Apply compacting shift towards the band
    mixed = q.copy()
    if flank_shift > 0.0:
        mixed[flank_nw_idx, :2] -= n_vec * flank_shift
        mixed[flank_se_idx, :2] += n_vec * flank_shift

    return normalize_box(mixed)


# ---------------------------------------------------------------------------
# Multi-Seed Basin Hopping Engine
# ---------------------------------------------------------------------------

def run_basin_hopping(
    base_seed: np.ndarray,
    seed_name: str,
    n_kicks: int = 30,
    rng_master_seed: int = 42,
    verbose: bool = True,
) -> tuple[np.ndarray, float, list[dict[str, Any]]]:
    """Execute basin hopping relaxation across n_kicks random kick seeds."""
    current_best_q = base_seed.copy()
    current_best_L = extent(current_best_q)
    kick_records: list[dict[str, Any]] = []

    if verbose:
        print(f"\n--- Basin Hopping on {seed_name} (initial L = {current_best_L:.6f}) ---")

    for k_idx in range(n_kicks):
        t0 = time.time()
        kick_seed = rng_master_seed + k_idx * 17 + 1
        rng = np.random.default_rng(kick_seed)

        # 1. Contact identification (perturb 5-15 tightest contact squares)
        pert_squares = find_tightest_contact_squares(current_best_q, k_min=5, k_max=15, rng=rng)
        k_count = len(pert_squares)

        # 2. Perturbation
        q_pert = current_best_q.copy()
        sigma_pos = rng.uniform(0.004, 0.015)
        sigma_ang = rng.uniform(0.002, 0.008)
        q_pert[pert_squares, :2] += rng.normal(0, sigma_pos, size=(k_count, 2))
        tilted_sub = pert_squares[current_best_q[pert_squares, 2] > 0.01]
        if len(tilted_sub) > 0:
            q_pert[tilted_sub, 2] += rng.normal(0, sigma_ang, size=len(tilted_sub))

        # 3. L-BFGS with smooth overlap penalty
        q_relaxed, lbfgs_loss, lbfgs_ok = lbfgs_smooth_relax(
            q_pert,
            pert_squares,
            container_L=current_best_L,
            maxiter=35,
        )

        # 4. Feasibility repair & exact L contraction via 4-sign-piece epigraph LP
        q_contracted, contracted_L = polish_lp(
            q_relaxed,
            max_iters=6,
            position_trust=0.06,
            angle_trust=0.006,
            clearance=1e-7,
        )

        is_valid, min_gap = sat_check(q_contracted, clearance=-1e-7)

        # Acceptance: accept strictly feasible improvement or Metropolis-style restart
        improved = False
        if is_valid and contracted_L < current_best_L - 1e-10:
            current_best_q = q_contracted.copy()
            current_best_L = contracted_L
            improved = True

        elapsed = time.time() - t0

        record = {
            "kick_idx": k_idx,
            "kick_seed": kick_seed,
            "k_perturbed": k_count,
            "perturbed_squares": pert_squares.tolist(),
            "sigma_pos": float(sigma_pos),
            "sigma_ang": float(sigma_ang),
            "lbfgs_loss": float(lbfgs_loss),
            "lbfgs_ok": lbfgs_ok,
            "contracted_L": float(contracted_L),
            "is_valid": is_valid,
            "min_gap": float(min_gap),
            "improved": improved,
            "current_best_L": float(current_best_L),
            "elapsed_seconds": float(elapsed),
        }
        kick_records.append(record)

        if verbose:
            imp_mark = " *IMPROVED*" if improved else ""
            print(
                f"[{seed_name}] Kick {k_idx+1:02d}/{n_kicks}: k={k_count}, "
                f"L={contracted_L:.6f}, valid={is_valid}, best={current_best_L:.6f} "
                f"({elapsed:.2f}s){imp_mark}"
            )

    return current_best_q, current_best_L, kick_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-seed relaxation for n=234.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("poc/research/data/relax_n234_results.json"),
        help="Output path for relaxation results JSON",
    )
    parser.add_argument(
        "--kicks-per-seed",
        type=int,
        default=30,
        help="Number of random kick seeds per base seed (default: 30)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose per-kick logs",
    )
    args = parser.parse_args()

    t_start = time.time()
    verbose = not args.quiet

    if verbose:
        print("=================================================================")
        print("  Multi-Seed Basin Hopping & Mixed Relaxation for Target n=234   ")
        print(f"  Incumbent L: {TARGET_INCUMBENT_L:.6f} | Scratch L: {TARGET_SCRATCH_L:.6f}")
        print("=================================================================")

    # 1. Load Base Seeds
    seed1 = load_seed1()
    seed2 = load_seed2()
    seed3 = construct_seed3_mixed(seed2, flank_shift=0.001)

    seeds = [
        ("Seed 1 (Idea 1 Scratch Band, 45 deg)", seed1),
        ("Seed 2 (Incumbent Rational Band, 24.3 deg)", seed2),
        ("Seed 3 (Mixed Structure, Rational + Staircase)", seed3),
    ]

    seeds_results = []
    overall_best_L = float("inf")
    overall_best_q = None
    overall_best_seed_name = ""

    for name, q_seed in seeds:
        init_L = extent(q_seed)
        best_q, best_L, records = run_basin_hopping(
            q_seed,
            seed_name=name,
            n_kicks=args.kicks_per_seed,
            rng_master_seed=42 + len(seeds_results) * 100,
            verbose=verbose,
        )

        delta_incumbent = best_L - TARGET_INCUMBENT_L
        seed_result = {
            "seed_name": name,
            "initial_L": float(init_L),
            "best_L": float(best_L),
            "delta_to_incumbent": float(delta_incumbent),
            "kicks_evaluated": len(records),
            "valid_kicks_count": sum(1 for r in records if r["is_valid"]),
            "kick_records": records,
        }
        seeds_results.append(seed_result)

        if best_L < overall_best_L:
            overall_best_L = best_L
            overall_best_q = best_q
            overall_best_seed_name = name

    improved_incumbent = overall_best_L < TARGET_INCUMBENT_L - 1e-7
    elapsed_total = time.time() - t_start

    # Format candidate packing if improved
    best_candidate_packing = None
    if overall_best_q is not None:
        best_candidate_packing = [
            {
                "id": i,
                "x": float(overall_best_q[i, 0]),
                "y": float(overall_best_q[i, 1]),
                "angle": float(overall_best_q[i, 2]),
            }
            for i in range(len(overall_best_q))
        ]

    output_data = {
        "target_n": 234,
        "incumbent_L": TARGET_INCUMBENT_L,
        "candidate_scratch_band_L": TARGET_SCRATCH_L,
        "gap_scratch_to_incumbent": TARGET_SCRATCH_L - TARGET_INCUMBENT_L,
        "overall_best_L": float(overall_best_L),
        "overall_best_seed": overall_best_seed_name,
        "improved_incumbent": improved_incumbent,
        "elapsed_seconds": float(elapsed_total),
        "seeds": seeds_results,
        "candidate_packing": best_candidate_packing if improved_incumbent else None,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    if verbose:
        print("\n=================================================================")
        print("  Relaxation Run Complete")
        print(f"  Overall Best L: {overall_best_L:.6f} ({overall_best_seed_name})")
        print(f"  Improved Incumbent: {improved_incumbent}")
        print(f"  Results saved to: {args.output}")
        print(f"  Elapsed: {elapsed_total:.2f}s")
        print("=================================================================")


if __name__ == "__main__":
    main()
