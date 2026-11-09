#!/usr/bin/env python3
"""Structural Transplant and Defect Solver (Idea 2).

Transplants incumbent packings across neighboring puzzles with strictly superior
packing density or smaller normalized side (e.g., n-1, n+1, or k^2).
  - n-1 -> n: Detects boundary slack pockets and inserts an additional unit square.
  - n+1 -> n: Identifies and removes the least constrained square (defect generation).
  - k^2 -> n: Reconstructs / trims regular grid core lattices.
Solves contact and non-overlap constraints via 4-sign-piece Sequential Linear
Programming (SLP) over local translations and rotations.
Evaluates resulting candidates against merged-literal-cpp.json incumbents and
records exact L, delta L, and metrics into poc/research/data/transplant_results.json.
"""
from __future__ import annotations

import argparse
import csv
import io
from itertools import product
import json
import math
from pathlib import Path
import sys
import time
from dataclasses import asdict, dataclass
from decimal import Decimal

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MERGED_CPP = REPO_ROOT / "poc/pell_corrugation/data/border-scan/merged-literal-cpp.json"
DEFAULT_MERGED_CSV = REPO_ROOT / "poc/pell_corrugation/data/border-scan/merged.csv"
DEFAULT_RESULTS_JSON = REPO_ROOT / "poc/research/data/transplant_results.json"

sys.path.insert(0, str(REPO_ROOT / "poc/baseline/poc/src"))
try:
    from squares._official_metric import _parse_submission_csv
    from squares.core import certify
except ImportError:
    certify = None
    _parse_submission_csv = None

S2 = math.sqrt(2.0)
INV_S2 = 1.0 / S2
SIGNS = tuple(product((-1.0, 1.0), repeat=2))

TOP_OFFENDERS_DEFAULT = [
    908, 909, 910, 911,
    971, 972, 973,
    591, 592, 593,
    543, 544, 545,
]


@dataclass
class PolishResult:
    q: np.ndarray
    L: float
    iterations: int = 0
    accepted: int = 0
    min_clearance: float = 0.0
    valid: bool = False


def extent(q: np.ndarray) -> float:
    """Bounding box side length enclosing all rotated unit square corners."""
    c = np.abs(np.cos(q[:, 2]))
    s = np.abs(np.sin(q[:, 2]))
    h = 0.5 * (c + s)
    lo = np.min(q[:, :2] - h[:, None], axis=0)
    hi = np.max(q[:, :2] + h[:, None], axis=0)
    return float(np.max(hi - lo))


def get_bounds(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns minimum and maximum (x, y) coordinates of the packing extent."""
    c = np.abs(np.cos(q[:, 2]))
    s = np.abs(np.sin(q[:, 2]))
    h = 0.5 * (c + s)
    lo = np.min(q[:, :2] - h[:, None], axis=0)
    hi = np.max(q[:, :2] + h[:, None], axis=0)
    return lo, hi


def normalize_box(q: np.ndarray, margin: float = 0.0) -> np.ndarray:
    """Translates packing so bottom-left extent starts at margin."""
    q_out = q.copy()
    lo, _ = get_bounds(q_out)
    q_out[:, :2] -= lo - margin
    return q_out


def sat_check(q: np.ndarray, clearance: float = 0.0) -> tuple[bool, float]:
    """Separating Axis Theorem exact minimum clearance between all pairs."""
    if len(q) <= 1:
        return True, float("inf")
    q_arr = np.asarray(q, dtype=np.longdouble)
    c, s = np.cos(q_arr[:, 2]), np.sin(q_arr[:, 2])
    u = np.column_stack((c, s))
    v = np.column_stack((-s, c))
    min_clearance = np.longdouble(np.inf)
    for i in range(len(q_arr) - 1):
        d = q_arr[i + 1 :, :2] - q_arr[i, :2]
        best = np.full(len(d), -np.inf, dtype=np.longdouble)
        for owner_axes in (False, True):
            for basis in (u, v):
                a = basis[i + 1 :] if owner_axes else np.broadcast_to(basis[i], d.shape)
                hi = 0.5 * (np.abs(a @ u[i]) + np.abs(a @ v[i]))
                hj = 0.5 * (np.abs(np.sum(a * u[i + 1 :], axis=1)) + np.abs(np.sum(a * v[i + 1 :], axis=1)))
                g = np.abs(np.sum(d * a, axis=1)) - hi - hj
                best = np.maximum(best, g)
        m = float(np.min(best))
        if m < min_clearance:
            min_clearance = m
        if m < clearance:
            return False, float(min_clearance)
    return True, float(min_clearance)


def chosen_axis(q: np.ndarray, i: int, j: int, previous=None, hysteresis: float = 1e-9):
    """Identifies the separating axis with maximum gap, with hysteresis."""
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
    """Computes exact gap and gradient for one of the 4 linear pieces of the support function."""
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
    upper: float,
    position_trust: float,
    angle_trust: float,
    clearance: float,
    axes: dict,
) -> tuple[np.ndarray | None, str]:
    """Single 4-sign-piece LP step solving translation and rotation under trust region."""
    from scipy.sparse import coo_matrix

    n = len(q)
    free = angle_trust > 0.0
    stride = 3 if free else 2
    nv = stride * n + 3  # displacements + box_min_x, box_min_y + L
    rr: list[int] = []
    cc: list[int] = []
    vv: list[float] = []
    rhs: list[float] = []

    def add_row(coeff: dict[int, float], b: float) -> None:
        r = len(rhs)
        rhs.append(float(b))
        for col, val in coeff.items():
            if val:
                rr.append(r)
                cc.append(col)
                vv.append(float(val))

    # Boundary box containment constraints
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
                add_row(lower, float(q[i, dim]) - h - clearance)
                add_row(upperrow, -float(q[i, dim]) - h - clearance)

    # Pairwise non-overlap constraints
    radius = S2 + 2.0 * S2 * position_trust + 1e-8
    pairs = sorted(cKDTree(q[:, :2]).query_pairs(radius))
    for i, j in pairs:
        key = (i, j)
        axis = chosen_axis(q, i, j, axes.get(key))
        axes[key] = axis
        if free:
            for signs in SIGNS:
                g, grad = piece_gap_gradient(q, i, j, axis, signs)
                coeff = {}
                for k in range(3):
                    coeff[stride * i + k] = -grad[k]
                    coeff[stride * j + k] = -grad[3 + k]
                add_row(coeff, g - clearance)
        else:
            owner, k, direction = axis
            phi = float(q[owner, 2]) + k * math.pi / 2.0
            a = direction * np.array([math.cos(phi), math.sin(phi)])
            h = sum(
                0.5 * (abs(math.cos(float(q[t, 2]) - phi)) + abs(math.sin(float(q[t, 2]) - phi)))
                for t in (i, j)
            )
            add_row(
                {stride * i: a[0], stride * i + 1: a[1], stride * j: -a[0], stride * j + 1: -a[1]},
                float((q[j, :2] - q[i, :2]) @ a - h - clearance),
            )

    bounds = []
    for _ in range(n):
        bounds.extend([(-position_trust, position_trust)] * 2)
        if free:
            bounds.append((-angle_trust, angle_trust))
    bounds.extend([(None, None), (None, None), (1.0, upper)])

    objective = np.zeros(nv)
    objective[-1] = 1.0
    A = coo_matrix((vv, (rr, cc)), shape=(len(rhs), nv)).tocsr()
    res = linprog(
        objective,
        A_ub=A,
        b_ub=rhs,
        bounds=bounds,
        method="highs",
        options={"primal_feasibility_tolerance": 1e-9, "dual_feasibility_tolerance": 1e-9, "time_limit": 5.0},
    )
    if not res.success or not np.isfinite(res.x).all():
        return None, str(res.message)

    out = q.copy()
    out[:, :2] += res.x[: stride * n].reshape(n, stride)[:, :2]
    if free:
        out[:, 2] += res.x[: stride * n].reshape(n, stride)[:, 2]
    return out, str(res.message)


def solve_slp(
    q: np.ndarray,
    L_init: float | None = None,
    iterations: int = 30,
    position_trust: float = 0.08,
    angle_trust: float = 0.008,
    clearance: float = 5e-9,
) -> PolishResult:
    """Solves local translation and rotation via Sequential Linear Programming (SLP)."""
    q_norm = normalize_box(q, margin=clearance)
    if L_init is None:
        L_init = extent(q_norm)
    axes: dict[tuple[int, int], list[tuple[float, float, int]]] = {}

    # Initial repair pass with pure translation
    repaired, _ = _lp_step(q_norm, L_init + 0.05, position_trust, 0.0, clearance, axes)
    current_q = repaired if repaired is not None else q_norm
    best_L = extent(current_q)
    best_q = current_q.copy()
    accepted_steps = 0

    for it in range(iterations):
        prev_L = best_L
        # Step 1: Fixed-angle translation
        fixed, _ = _lp_step(best_q, best_L + 1e-4, position_trust, 0.0, clearance, axes)
        if fixed is not None:
            is_valid, clr = sat_check(fixed, clearance=0.0)
            cur_ext = extent(fixed)
            if is_valid and cur_ext < best_L:
                best_q = fixed
                best_L = cur_ext
                accepted_steps += 1

        # Step 2: Joint translation + rotation step with line search
        if angle_trust > 0:
            proposal, _ = _lp_step(best_q, best_L + 1e-4, position_trust, angle_trust, clearance, axes)
            if proposal is not None:
                origin = best_q.copy()
                for bt in range(8):
                    alpha = 2.0 ** (-bt)
                    cand = origin + alpha * (proposal - origin)
                    rep, _ = _lp_step(cand, best_L + 1e-3, position_trust, 0.0, clearance, {})
                    test_q = rep if rep is not None else cand
                    is_valid, clr = sat_check(test_q, clearance=0.0)
                    cur_ext = extent(test_q)
                    if is_valid and cur_ext < best_L:
                        best_q = test_q
                        best_L = cur_ext
                        accepted_steps += 1
                        break

        if prev_L - best_L < 1e-10 and it >= 5:
            break

    is_valid, min_clr = sat_check(best_q, clearance=0.0)
    return PolishResult(
        q=best_q,
        L=best_L,
        iterations=it + 1,
        accepted=accepted_steps,
        min_clearance=min_clr,
        valid=is_valid,
    )


def load_incumbents(csv_path: Path, target_ns: set[int]) -> dict[int, np.ndarray]:
    """Loads square coordinates for requested puzzle sizes from merged.csv."""
    rows_by_n: dict[int, list[tuple[int, float, float, float]]] = {n: [] for n in target_ns}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n_val = int(row["n"])
            if n_val in target_ns:
                rows_by_n[n_val].append((
                    int(row["square_id"]),
                    float(row["x"]),
                    float(row["y"]),
                    float(row["angle"]),
                ))

    packings: dict[int, np.ndarray] = {}
    for n_val, sqs in rows_by_n.items():
        if len(sqs) == n_val:
            sqs.sort(key=lambda s: s[0])
            packings[n_val] = np.array([[x, y, a] for _, x, y, a in sqs], dtype=float)
    return packings


def load_merged_literal_meta(json_path: Path) -> dict[int, dict]:
    """Loads official/CPP evaluation metadata (exact L, score) for each n."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    meta: dict[int, dict] = {}
    for item in data.get("results", []):
        meta[int(item["n"])] = item
    return meta


def identify_boundary_slack_pockets(
    q: np.ndarray,
    num_samples_per_edge: int = 25,
) -> list[tuple[float, np.ndarray]]:
    """Identifies boundary slack pockets by probing container perimeter and corners.

    Returns ranked list of (max_penetration, candidate_square) where max_penetration <= 0
    indicates an open slack pocket with positive clearance.
    """
    lo, hi = get_bounds(q)
    W, H = hi[0] - lo[0], hi[1] - lo[1]
    L = max(W, H)
    candidates: list[np.ndarray] = []

    # Test upright and tilted orientations
    angles = (0.0, math.pi / 4.0, math.pi / 8.0, 3.0 * math.pi / 8.0)
    for a in angles:
        c, s = abs(math.cos(a)), abs(math.sin(a))
        h = 0.5 * (c + s)

        # Bottom and top boundary margins
        xs = np.linspace(lo[0] + h, lo[0] + L - h, num_samples_per_edge)
        for x in xs:
            candidates.append(np.array([x, lo[1] + h, a]))
            candidates.append(np.array([x, lo[1] + L - h, a]))

        # Left and right boundary margins
        ys = np.linspace(lo[1] + h, lo[1] + L - h, num_samples_per_edge)
        for y in ys:
            candidates.append(np.array([lo[0] + h, y, a]))
            candidates.append(np.array([lo[0] + L - h, y, a]))

        # Exact corners
        for cx in (lo[0] + h, lo[0] + L - h):
            for cy in (lo[1] + h, lo[1] + L - h):
                candidates.append(np.array([cx, cy, a]))

    scored_pockets: list[tuple[float, np.ndarray]] = []
    tree = cKDTree(q[:, :2])
    for cand in candidates:
        # Check against nearby existing squares
        near = tree.query_ball_point(cand[:2], S2 + 1e-4)
        if not near:
            scored_pockets.append((0.0, cand))
            continue

        max_penetration = -np.inf
        cand_u = np.array([math.cos(cand[2]), math.sin(cand[2])])
        cand_v = np.array([-cand_u[1], cand_u[0]])
        for j in near:
            other = q[j]
            d = other[:2] - cand[:2]
            oth_u = np.array([math.cos(other[2]), math.sin(other[2])])
            oth_v = np.array([-oth_u[1], oth_u[0]])
            pair_gap = -np.inf
            for ax in (cand_u, cand_v, oth_u, oth_v):
                hc = 0.5 * (abs(ax @ cand_u) + abs(ax @ cand_v))
                ho = 0.5 * (abs(ax @ oth_u) + abs(ax @ oth_v))
                g = abs(d @ ax) - hc - ho
                pair_gap = max(pair_gap, g)
            penetration = -pair_gap
            if penetration > max_penetration:
                max_penetration = penetration

        scored_pockets.append((float(max_penetration), cand))

    # Sort by minimum penetration into existing squares
    scored_pockets.sort(key=lambda item: item[0])
    return scored_pockets


def transplant_insert_slack_square(q_donor: np.ndarray) -> np.ndarray:
    """Inserts an additional square into the best boundary slack pocket of donor n-1."""
    pockets = identify_boundary_slack_pockets(q_donor)
    best_pen, best_square = pockets[0]

    # Append square to donor packing
    q_augmented = np.vstack([q_donor, best_square.reshape(1, 3)])
    if best_pen > 0:
        # Dilate slightly if there is initial overlap to initialize feasible SLP
        lo, _ = get_bounds(q_augmented)
        q_augmented[:, :2] += (q_augmented[:, :2] - lo) * (best_pen * 0.1)

    return q_augmented


def identify_least_constrained_square(q: np.ndarray) -> int:
    """Identifies the least constrained square in donor n+1 to create a defect vacancy."""
    n = len(q)
    lo, hi = get_bounds(q)
    tree = cKDTree(q[:, :2])

    constraint_scores = np.zeros(n)
    for i in range(n):
        # 1. Contact degree (number of neighbors within contact threshold)
        near = tree.query_ball_point(q[i, :2], 1.45)
        near = [j for j in near if j != i]
        contacts = 0
        min_gap = np.inf
        for j in near:
            d = q[j, :2] - q[i, :2]
            c_i, s_i = math.cos(q[i, 2]), math.sin(q[i, 2])
            u_i = np.array([c_i, s_i])
            v_i = np.array([-s_i, c_i])
            c_j, s_j = math.cos(q[j, 2]), math.sin(q[j, 2])
            u_j = np.array([c_j, s_j])
            v_j = np.array([-u_j[1], u_j[0]])

            best_g = -np.inf
            for ax in (u_i, v_i, u_j, v_j):
                hi_ax = 0.5 * (abs(ax @ u_i) + abs(ax @ v_i))
                hj_ax = 0.5 * (abs(ax @ u_j) + abs(ax @ v_j))
                g = abs(d @ ax) - hi_ax - hj_ax
                best_g = max(best_g, g)
            if best_g < 0.05:
                contacts += 1
            if best_g < min_gap:
                min_gap = best_g

        # 2. Wall proximity: squares directly touching boundary wall
        t = q[i, 2]
        h = 0.5 * (abs(math.cos(t)) + abs(math.sin(t)))
        dist_to_walls = min(
            q[i, 0] - h - lo[0],
            hi[0] - (q[i, 0] + h),
            q[i, 1] - h - lo[1],
            hi[1] - (q[i, 1] + h),
        )

        # Least constrained has fewer contacts and high wall clearance/exterior position
        constraint_scores[i] = contacts * 2.0 + min(dist_to_walls, 1.0) - max(0.0, min_gap)

    least_constrained_idx = int(np.argmin(constraint_scores))
    return least_constrained_idx


def transplant_remove_least_constrained(q_donor: np.ndarray) -> np.ndarray:
    """Removes the least constrained square from donor n+1 and shifts neighbors."""
    idx = identify_least_constrained_square(q_donor)
    q_out = np.delete(q_donor, idx, axis=0)
    return q_out


def build_k2_transplant(target_n: int, k: int) -> np.ndarray:
    """Builds an integer grid packing of size k x k and removes k^2 - n squares."""
    grid_coords = []
    for i in range(k):
        for j in range(k):
            grid_coords.append([i + 0.5, j + 0.5, 0.0])
    q_grid = np.array(grid_coords, dtype=float)

    to_remove = k * k - target_n
    if to_remove > 0:
        # Iteratively remove least constrained corner / boundary squares
        for _ in range(to_remove):
            idx = identify_least_constrained_square(q_grid)
            q_grid = np.delete(q_grid, idx, axis=0)

    return q_grid


def evaluate_target_transplant(
    target_n: int,
    incumbents: dict[int, np.ndarray],
    meta: dict[int, dict],
    slp_iterations: int = 25,
) -> dict:
    """Evaluates transplant candidates from neighboring donors for a specific target n."""
    incumbent_meta = meta.get(target_n, {})
    incumbent_L_str = incumbent_meta.get("L", "")
    incumbent_score = incumbent_meta.get("score", float("nan"))
    incumbent_L = float(incumbent_L_str) if incumbent_L_str else (extent(incumbents[target_n]) if target_n in incumbents else float("nan"))
    norm_side_target = incumbent_L / math.sqrt(target_n)

    candidates_evaluated = []
    best_candidate: PolishResult | None = None
    best_donor_info: dict = {}

    # Check neighboring donors
    neighbor_candidates = []
    # 1. Donor n - 1 (insert unit square into boundary slack pocket)
    if target_n - 1 in incumbents:
        L_prev = float(meta.get(target_n - 1, {}).get("L", extent(incumbents[target_n - 1])))
        norm_side_prev = L_prev / math.sqrt(target_n - 1)
        neighbor_candidates.append((target_n - 1, "n-1", norm_side_prev))

    # 2. Donor n + 1 (remove least constrained square)
    if target_n + 1 in incumbents:
        L_next = float(meta.get(target_n + 1, {}).get("L", extent(incumbents[target_n + 1])))
        norm_side_next = L_next / math.sqrt(target_n + 1)
        neighbor_candidates.append((target_n + 1, "n+1", norm_side_next))

    # 3. Donor k^2
    k_ceil = math.isqrt(target_n - 1) + 1
    k2 = k_ceil * k_ceil
    norm_side_k2 = 1.0  # k / sqrt(k^2) = 1.0
    if norm_side_k2 <= norm_side_target:
        neighbor_candidates.append((k2, "k^2", norm_side_k2))

    # Filter to donors with strictly superior packing density / smaller normalized side
    valid_donors = [c for c in neighbor_candidates if c[2] < norm_side_target + 1e-9]
    valid_donors.sort(key=lambda c: c[2])

    for donor_n, donor_type, norm_side in valid_donors:
        if donor_type == "n-1":
            q_cand = transplant_insert_slack_square(incumbents[donor_n])
        elif donor_type == "n+1":
            q_cand = transplant_remove_least_constrained(incumbents[donor_n])
        elif donor_type == "k^2":
            q_cand = build_k2_transplant(target_n, k_ceil)
        else:
            continue

        res = solve_slp(
            q_cand,
            L_init=extent(q_cand),
            iterations=slp_iterations,
            position_trust=0.08,
            angle_trust=0.008,
            clearance=5e-9,
        )

        cand_score = (res.L / math.sqrt(target_n)) - 1.0
        delta_L = res.L - incumbent_L
        delta_score = cand_score - incumbent_score

        eval_entry = {
            "donor_n": donor_n,
            "donor_type": donor_type,
            "donor_normalized_side": norm_side,
            "candidate_L": res.L,
            "candidate_score": cand_score,
            "delta_L": delta_L,
            "delta_score": delta_score,
            "min_clearance": res.min_clearance,
            "valid": res.valid,
            "accepted_slp_steps": res.accepted,
        }
        candidates_evaluated.append(eval_entry)

        if res.valid:
            if best_candidate is None or res.L < best_candidate.L:
                best_candidate = res
                best_donor_info = eval_entry

    improved = best_candidate is not None and best_candidate.L < incumbent_L - 1e-8
    exact_L_str = format(best_candidate.L, ".50f") if best_candidate else incumbent_L_str
    delta_exact_L = (
        str(Decimal(str(best_candidate.L)) - Decimal(incumbent_L_str))
        if best_candidate and incumbent_L_str
        else "0"
    )

    result_summary = {
        "target_n": target_n,
        "incumbent_L": incumbent_L_str,
        "incumbent_score": incumbent_score,
        "best_candidate_L": exact_L_str if best_candidate else None,
        "best_candidate_L_float": best_candidate.L if best_candidate else None,
        "delta_L": delta_exact_L,
        "delta_L_float": best_candidate.L - incumbent_L if best_candidate else 0.0,
        "improved": improved,
        "valid": best_candidate.valid if best_candidate else False,
        "min_clearance": best_candidate.min_clearance if best_candidate else None,
        "best_donor": best_donor_info,
        "all_candidates": candidates_evaluated,
    }
    return result_summary


def run_transplant_solver(
    targets: list[int] | None = None,
    merged_cpp_path: Path = DEFAULT_MERGED_CPP,
    merged_csv_path: Path = DEFAULT_MERGED_CSV,
    results_out_path: Path = DEFAULT_RESULTS_JSON,
    slp_iterations: int = 25,
) -> dict:
    """Executes structural transplant and defect solver across target puzzles."""
    if targets is None:
        targets = list(TOP_OFFENDERS_DEFAULT)

    # Required donor set includes targets plus neighbors
    required_ns = set(targets)
    for t in targets:
        required_ns.add(t - 1)
        required_ns.add(t + 1)

    meta = load_merged_literal_meta(merged_cpp_path)
    incumbents = load_incumbents(merged_csv_path, required_ns)

    results: dict[str, dict] = {}
    total_improved = 0
    for target_n in sorted(targets):
        res = evaluate_target_transplant(
            target_n=target_n,
            incumbents=incumbents,
            meta=meta,
            slp_iterations=slp_iterations,
        )
        results[str(target_n)] = res
        if res.get("improved"):
            total_improved += 1

    output_payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "solver": "structural_transplant_and_defect_solver_idea2",
        "total_targets_evaluated": len(targets),
        "total_improved": total_improved,
        "results": results,
    }

    results_out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_out_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    return output_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Idea 2: Structural Transplant & Defect Solver")
    parser.add_argument("--targets", nargs="+", type=int, default=TOP_OFFENDERS_DEFAULT)
    parser.add_argument("--merged-cpp", type=Path, default=DEFAULT_MERGED_CPP)
    parser.add_argument("--merged-csv", type=Path, default=DEFAULT_MERGED_CSV)
    parser.add_argument("--out", type=Path, default=DEFAULT_RESULTS_JSON)
    parser.add_argument("--iterations", type=int, default=25)
    args = parser.parse_args()

    run_transplant_solver(
        targets=args.targets,
        merged_cpp_path=args.merged_cpp,
        merged_csv_path=args.merged_csv,
        results_out_path=args.out,
        slp_iterations=args.iterations,
    )


if __name__ == "__main__":
    main()
