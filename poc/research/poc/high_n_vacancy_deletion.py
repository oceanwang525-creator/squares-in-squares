#!/usr/bin/env python3
"""Vacancy Deletion and Defect Relaxation Solver for High n in [600, 1000].

Solves high-n square container packings using vacancy deletion and defect
propagation patterns from poc/vacancy_deletion/.

Algorithm:
1. Loads incumbent packings from poc/pell_corrugation/data/border-scan/merged.csv.
2. For each target n in [600, 1000]:
   - Considers candidate donor packings at n + 1 and n + 2.
   - Computes pairwise SAT clearances and container boundary distances for every
     square in the donor packing to measure mechanical constraint/slack.
   - Selects the 1 to 5 squares with the largest slack or least contribution to
     container perimeter boundaries.
   - Removes the chosen squares to produce an n-square candidate.
   - Applies 4-sign-piece epigraph LP relaxation to contract container side length L
     while enforcing non-overlap.
3. Evaluates whether candidate L beats incumbent L (delta L < 0).
4. Streams per-n progress and saves all candidate packings and metrics to
   poc/research/data/high_n_vacancy_results.json.
"""
from __future__ import annotations

import argparse
import csv
import io
from itertools import combinations, product
import json
import math
from pathlib import Path
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[3]
DEFAULT_MERGED_CSV = REPO / "poc/pell_corrugation/data/border-scan/merged.csv"
DEFAULT_OUTPUT_JSON = REPO / "poc/research/data/high_n_vacancy_results.json"

S2 = math.sqrt(2.0)
INV_S2 = 1.0 / S2
SIGNS = tuple(product((-1.0, 1.0), repeat=2))


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
    """Returns minimum and maximum (x, y) coordinates of packing extent."""
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
    """Identifies the separating axis with maximum gap, maintaining hysteresis."""
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

    # Boundary box containment constraints under 4-sign-piece epigraph
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


def polish_packing(
    q: np.ndarray,
    L_init: float | None = None,
    iterations: int = 20,
    position_trust: float = 0.08,
    angle_trust: float = 0.008,
    clearance: float = 5e-9,
) -> PolishResult:
    """Polishes candidate packing with active container shrink LP."""
    q_norm = normalize_box(q, margin=clearance)
    if L_init is None:
        L_init = extent(q_norm)
    axes: dict[tuple[int, int], Any] = {}

    # Initial repair pass with pure translation
    repaired, _ = _lp_step(q_norm, L_init + 0.05, position_trust, 0.0, clearance, axes)
    current_q = repaired if repaired is not None else q_norm
    best_L = extent(current_q)
    best_q = current_q.copy()
    accepted_steps = 0

    for it in range(iterations):
        prev_L = best_L
        # Step 1: Fixed-angle translation step
        fixed, _ = _lp_step(best_q, best_L + 1e-4, position_trust, 0.0, clearance, axes)
        if fixed is not None:
            is_valid, _ = sat_check(fixed, clearance=0.0)
            cur_ext = extent(fixed)
            if is_valid and cur_ext < best_L:
                best_q = fixed
                best_L = cur_ext
                accepted_steps += 1

        # Step 2: Joint translation + rotation proposal with line search
        if angle_trust > 0:
            proposal, _ = _lp_step(best_q, best_L + 1e-4, position_trust, angle_trust, clearance, axes)
            if proposal is not None:
                origin = best_q.copy()
                for bt in range(6):
                    alpha = 2.0 ** (-bt)
                    cand = origin + alpha * (proposal - origin)
                    rep, _ = _lp_step(cand, best_L + 1e-3, position_trust, 0.0, clearance, {})
                    test_q = rep if rep is not None else cand
                    is_valid, _ = sat_check(test_q, clearance=0.0)
                    cur_ext = extent(test_q)
                    if is_valid and cur_ext < best_L:
                        best_q = test_q
                        best_L = cur_ext
                        accepted_steps += 1
                        break

        if prev_L - best_L < 1e-9 and it >= 4:
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


def compute_mechanical_slack(q: np.ndarray) -> dict[str, np.ndarray]:
    """Computes pairwise SAT clearances and container boundary distances for every square.

    Returns:
        Dictionary with:
          - boundary_distances: distance to closest container boundary wall
          - min_sat_clearance: minimum pairwise SAT clearance to neighboring squares
          - contact_count: number of near-contact neighbors (gap < 0.05)
          - slack_score: combined slack metric (higher means more unconstrained / interior)
    """
    n = len(q)
    lo, hi = get_bounds(q)
    tree = cKDTree(q[:, :2])

    boundary_distances = np.zeros(n, dtype=float)
    min_sat_clearances = np.full(n, np.inf, dtype=float)
    contact_counts = np.zeros(n, dtype=int)

    # 1. Container boundary distances
    for i in range(n):
        t = q[i, 2]
        h = 0.5 * (abs(math.cos(t)) + abs(math.sin(t)))
        dist_left = q[i, 0] - h - lo[0]
        dist_right = hi[0] - (q[i, 0] + h)
        dist_bottom = q[i, 1] - h - lo[1]
        dist_top = hi[1] - (q[i, 1] + h)
        boundary_distances[i] = min(dist_left, dist_right, dist_bottom, dist_top)

    # 2. Pairwise SAT clearances
    for i in range(n):
        near = tree.query_ball_point(q[i, :2], 1.55)
        near = [j for j in near if j != i]
        if not near:
            min_sat_clearances[i] = 1.0
            continue

        c_i, s_i = math.cos(q[i, 2]), math.sin(q[i, 2])
        u_i = np.array([c_i, s_i])
        v_i = np.array([-s_i, c_i])

        for j in near:
            d = q[j, :2] - q[i, :2]
            c_j, s_j = math.cos(q[j, 2]), math.sin(q[j, 2])
            u_j = np.array([c_j, s_j])
            v_j = np.array([-u_j[1], u_j[0]])

            best_g = -np.inf
            for ax in (u_i, v_i, u_j, v_j):
                hi_ax = 0.5 * (abs(ax @ u_i) + abs(ax @ v_i))
                hj_ax = 0.5 * (abs(ax @ u_j) + abs(ax @ v_j))
                g = abs(d @ ax) - hi_ax - hj_ax
                best_g = max(best_g, g)

            if best_g < min_sat_clearances[i]:
                min_sat_clearances[i] = best_g
            if best_g < 0.05:
                contact_counts[i] += 1

    # Invert min_sat_clearances inf if isolated
    min_sat_clearances = np.where(np.isinf(min_sat_clearances), 1.0, min_sat_clearances)

    # Slack score: higher = larger slack / least contribution to perimeter boundaries
    # boundary_distance: interior squares contribute 0 to perimeter
    # min_sat_clearance: looser squares have larger clearance
    # contact_count: fewer contacts indicates less mechanical jam
    slack_score = (
        boundary_distances * 1.5
        + min_sat_clearances * 2.0
        - contact_counts * 0.3
    )

    return {
        "boundary_distances": boundary_distances,
        "min_sat_clearances": min_sat_clearances,
        "contact_counts": contact_counts,
        "slack_score": slack_score,
    }


def select_candidate_squares(
    q: np.ndarray,
    top_k: int = 5,
) -> list[int]:
    """Selects 1 to 5 squares with largest slack or least contribution to container perimeter boundaries."""
    metrics = compute_mechanical_slack(q)
    slack_score = metrics["slack_score"]
    boundary_distances = metrics["boundary_distances"]
    min_sat_clearances = metrics["min_sat_clearances"]

    # Rank indices by:
    # 1. High slack score (composite mechanical slack)
    # 2. High boundary distance (deep interior squares)
    ranked_by_slack = list(np.argsort(slack_score)[::-1])
    ranked_by_boundary = list(np.argsort(boundary_distances)[::-1])
    ranked_by_sat = list(np.argsort(min_sat_clearances)[::-1])

    # Merge top selections uniquely up to top_k
    candidates: list[int] = []
    for idx_list in (ranked_by_slack, ranked_by_boundary, ranked_by_sat):
        for idx in idx_list:
            if idx not in candidates:
                candidates.append(int(idx))
            if len(candidates) >= top_k:
                break
        if len(candidates) >= top_k:
            break

    return candidates[:top_k]


def load_incumbent_packings(
    csv_path: Path,
    ns: Iterable[int],
) -> dict[int, np.ndarray]:
    """Loads square configurations for requested n values from merged.csv."""
    needed = set(ns)
    df = pd.read_csv(csv_path)
    df_needed = df[df["n"].isin(needed)].sort_values(["n", "square_id"])
    packings: dict[int, np.ndarray] = {}
    for n_val, grp in df_needed.groupby("n"):
        packings[int(n_val)] = grp[["x", "y", "angle"]].to_numpy(dtype=float)
    return packings


def validate_tilted_packing(
    n: int,
    L: float,
    angles: Sequence[float],
    tolerance: float = 1e-5,
) -> tuple[bool, str, str]:
    """Hard validation check:
    Rejects any packing where container side L equals ceil(sqrt(n)) to within 1e-5
    or has zero tilted squares (all angles == 0).

    Returns:
        (passed: bool, status: str, reason: str)
    """
    grid_L = float(math.ceil(math.sqrt(n)))
    if abs(L - grid_L) <= tolerance:
        return False, "FAILED", "integer_grid_fallback"

    tilted_count = 0
    for a in angles:
        norm_a = abs(float(a)) % (math.pi / 2.0)
        norm_a = min(norm_a, math.pi / 2.0 - norm_a)
        if norm_a > tolerance:
            tilted_count += 1

    if tilted_count == 0:
        return False, "FAILED", "integer_grid_fallback"

    return True, "REAL_CONSTRUCTION", "ok"


def evaluate_vacancy_target(
    target_n: int,
    packings: dict[int, np.ndarray],
    top_k: int = 5,
    iterations: int = 15,
) -> dict[str, Any]:
    """Evaluates vacancy deletion and defect relaxation for a target n in [600, 1000]."""
    t_start = time.time()
    incumbent_q = packings.get(target_n)
    if incumbent_q is None:
        raise ValueError(f"Incumbent packing for target n={target_n} not loaded.")

    incumbent_L = extent(incumbent_q)
    best_candidate_q: np.ndarray | None = None
    best_candidate_L = float("inf")
    best_donor_n = 0
    best_deleted_squares: list[int] = []
    best_polish_res: PolishResult | None = None

    # Consider candidate donor packings at n + 1 and n + 2
    for offset in (1, 2):
        donor_n = target_n + offset
        donor_q = packings.get(donor_n)
        if donor_q is None:
            continue

        donor_candidates = select_candidate_squares(donor_q, top_k=top_k)

        # Generate candidates to delete
        deletion_subsets: list[tuple[int, ...]] = []
        if offset == 1:
            # Delete 1 square among top candidates
            deletion_subsets = [(idx,) for idx in donor_candidates]
        elif offset == 2:
            # Delete 2 squares among top candidate combinations
            deletion_subsets = list(combinations(donor_candidates, 2))
            # Limit candidate pairs to evaluate top 5 pairs to stay focused
            deletion_subsets = deletion_subsets[:5]

        for del_indices in deletion_subsets:
            # Remove chosen squares to produce an n-square candidate
            cand_q = np.delete(donor_q, list(del_indices), axis=0)

            # Apply 4-sign-piece epigraph LP relaxation
            init_L = extent(cand_q)
            res = polish_packing(
                cand_q,
                L_init=init_L,
                iterations=iterations,
                position_trust=0.08,
                angle_trust=0.008,
                clearance=5e-9,
            )

            # Verify strictly positive SAT clearance and contract side length via 4-sign-piece epigraph LP
            if res.valid and res.min_clearance > 0.0:
                if res.L < best_candidate_L:
                    best_candidate_L = res.L
                    best_candidate_q = res.q
                    best_donor_n = donor_n
                    best_deleted_squares = [int(i) for i in del_indices]
                    best_polish_res = res

    elapsed = time.time() - t_start

    # Hard validation check
    if best_candidate_q is None:
        status = "FAILED"
        reason = "no_candidate_found"
        tilted_count = 0
        best_L = float("nan")
        delta_L = float("nan")
        improved = False
    else:
        best_L = best_candidate_L
        delta_L = best_candidate_L - incumbent_L
        improved = bool(delta_L < -1e-9)
        passed, val_status, val_reason = validate_tilted_packing(
            target_n,
            best_candidate_L,
            best_candidate_q[:, 2],
        )
        if not passed:
            status = "FAILED"
            reason = val_reason
        elif best_polish_res is None or not (best_polish_res.valid and best_polish_res.min_clearance > 0.0):
            status = "FAILED"
            reason = "non_positive_clearance"
        else:
            status = "REAL_CONSTRUCTION"
            reason = "ok"

        # Count tilted squares
        tilted_count = 0
        for a in best_candidate_q[:, 2]:
            norm_a = abs(float(a)) % (math.pi / 2.0)
            norm_a = min(norm_a, math.pi / 2.0 - norm_a)
            if norm_a > 1e-5:
                tilted_count += 1

    # Format layout if candidate was produced
    layout = []
    if best_candidate_q is not None:
        for i, row in enumerate(best_candidate_q):
            layout.append({
                "id": i,
                "x": float(row[0]),
                "y": float(row[1]),
                "angle": float(row[2]),
            })

    return {
        "target_n": target_n,
        "n": target_n,
        "best_L": best_L,
        "incumbent_L": incumbent_L,
        "delta_L": delta_L,
        "improved": improved,
        "donor_n": best_donor_n,
        "deleted_square_ids": best_deleted_squares,
        "deleted_squares": best_deleted_squares,
        "status": status,
        "reason": reason,
        "runtime": elapsed,
        "min_clearance": best_polish_res.min_clearance if best_polish_res else float("nan"),
        "tilted_squares": tilted_count,
        "iterations": best_polish_res.iterations if best_polish_res else 0,
        "accepted_steps": best_polish_res.accepted if best_polish_res else 0,
        "layout": layout,
    }


def _eval_target_task(args: tuple[int, dict[int, np.ndarray], int, int]) -> dict[str, Any]:
    target_n, packings_subset, top_k, iterations = args
    return evaluate_vacancy_target(
        target_n=target_n,
        packings=packings_subset,
        top_k=top_k,
        iterations=iterations,
    )


def run_high_n_vacancy_deletion(
    start_n: int = 600,
    end_n: int = 1000,
    n_list: Sequence[int] | None = None,
    csv_path: Path = DEFAULT_MERGED_CSV,
    output_path: Path = DEFAULT_OUTPUT_JSON,
    top_k: int = 5,
    iterations: int = 15,
    workers: int = 1,
) -> dict[str, Any]:
    """Runs vacancy deletion solver across specified n targets, streaming per-n progress."""
    if n_list is not None and len(n_list) > 0:
        seen: set[int] = set()
        targets = [x for x in n_list if not (x in seen or seen.add(x))]
    else:
        targets = list(range(start_n, end_n + 1))

    # All n values required (targets plus donors n+1, n+2)
    needed_ns = set(targets)
    for t in targets:
        needed_ns.add(t + 1)
        needed_ns.add(t + 2)

    print(f"Loading packings from {csv_path} for {len(needed_ns)} distinct puzzle sizes...")
    packings = load_incumbent_packings(csv_path, needed_ns)
    print(f"Successfully loaded {len(packings)} packings. Starting vacancy deletion solver...")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "start_n": targets[0] if targets else start_n,
        "end_n": targets[-1] if targets else end_n,
        "total_targets": len(targets),
        "improved_count": 0,
        "targets": {},
    }

    t0 = time.time()
    if workers > 1 and len(targets) > 1:
        import concurrent.futures

        tasks = [
            (
                t,
                {k: packings[k] for k in (t, t + 1, t + 2) if k in packings},
                top_k,
                iterations,
            )
            for t in targets
        ]
        with concurrent.futures.ProcessPoolExecutor(max_workers=min(workers, len(targets))) as executor:
            future_to_n = {executor.submit(_eval_target_task, task): task[0] for task in tasks}
            for idx, future in enumerate(concurrent.futures.as_completed(future_to_n)):
                record = future.result()
                target_n = record["target_n"]
                results["targets"][str(target_n)] = record
                if record.get("improved"):
                    results["improved_count"] += 1
                delta_str = f"{record['delta_L']:+.6e}" if not math.isnan(record["delta_L"]) else "N/A"
                cand_str = f"{record['best_L']:.6f}" if not math.isnan(record["best_L"]) else "N/A"
                print(
                    f"[{idx + 1:3d}/{len(targets):3d}] n={target_n:4d} | "
                    f"Incumbent L={record['incumbent_L']:.6f} | "
                    f"Cand L={cand_str} | "
                    f"dL={delta_str} | "
                    f"Donor: n+{record['donor_n'] - target_n if record['donor_n'] else 0} | "
                    f"Del={record['deleted_square_ids']} | "
                    f"Status: {record['status']} ({record['runtime']:.2f}s)",
                    flush=True,
                )
    else:
        for idx, target_n in enumerate(targets):
            t_target_start = time.time()
            record = evaluate_vacancy_target(
                target_n=target_n,
                packings=packings,
                top_k=top_k,
                iterations=iterations,
            )
            t_target_elapsed = time.time() - t_target_start

            if record["improved"]:
                results["improved_count"] += 1

            results["targets"][str(target_n)] = record

            # Stream per-n progress to stdout
            delta_str = f"{record['delta_L']:+.6e}" if not math.isnan(record["delta_L"]) else "N/A"
            cand_str = f"{record['best_L']:.6f}" if not math.isnan(record["best_L"]) else "N/A"
            print(
                f"[{idx + 1:3d}/{len(targets):3d}] n={target_n:4d} | "
                f"Incumbent L={record['incumbent_L']:.6f} | "
                f"Cand L={cand_str} | "
                f"dL={delta_str} | "
                f"Donor: n+{record['donor_n'] - target_n if record['donor_n'] else 0} | "
                f"Del={record['deleted_square_ids']} | "
                f"Status: {record['status']} ({t_target_elapsed:.2f}s)",
                flush=True,
            )

            # Save incrementally every 10 targets
            if (idx + 1) % 10 == 0 or (idx + 1) == len(targets):
                results["elapsed_seconds"] = time.time() - t0
                results["results"] = [results["targets"][str(t)] for t in targets if str(t) in results["targets"]]
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2)

    results["elapsed_seconds"] = time.time() - t0
    results["results"] = [results["targets"][str(t)] for t in targets if str(t) in results["targets"]]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(
        f"Completed {len(targets)} targets in {results['elapsed_seconds']:.2f}s. "
        f"Improved count: {results['improved_count']}. Results saved to {output_path}."
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Vacancy deletion and defect relaxation solver for high n in [600, 1000]."
    )
    parser.add_argument("--start-n", type=int, default=600, help="First target n (default: 600)")
    parser.add_argument("--end-n", type=int, default=1000, help="Last target n (default: 1000)")
    parser.add_argument(
        "--n-list",
        type=int,
        nargs="+",
        default=None,
        help="Optional explicit list of target n values to evaluate",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=DEFAULT_MERGED_CSV,
        help="Path to merged.csv incumbent dataset",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
        help="Path to save results JSON",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of candidate vacancy squares to rank/select (1..5, default: 5)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=15,
        help="Number of LP relaxation iterations per candidate (default: 15)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel worker processes (default: 1)",
    )

    args = parser.parse_args()
    run_high_n_vacancy_deletion(
        start_n=args.start_n,
        end_n=args.end_n,
        n_list=args.n_list,
        csv_path=args.csv_path,
        output_path=args.output_path,
        top_k=args.top_k,
        iterations=args.iterations,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
