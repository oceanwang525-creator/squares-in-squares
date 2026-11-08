"""Continuous variable-slope band optimizer around theta = 24.10 degrees.

Implements the 4-sign-piece epigraph LP polisher for variable-slope square packings
featuring tilted bands and pockets (dominant theta approx 24.1004 deg / 0.420631 rad,
matching rational lattice slope 17:38 and 1:2 rational staircases).

The LP polisher resolves the V-shaped non-differentiable SAT support kink at relative
angle r = 0 by enforcing the 4-piece piecewise-linear epigraph, allowing continuous
angle splitting and elastic column relaxation against orthogonal upright boundaries.
"""
from __future__ import annotations

import csv
import io
import json
import math
import sys
import time
from dataclasses import dataclass
from decimal import Decimal
from itertools import product
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parents[1] / "data"
DOCS = Path(__file__).resolve().parents[1] / "docs"
REPO = Path(__file__).resolve().parents[4] if len(Path(__file__).resolve().parents) > 4 else Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "poc/baseline/poc/src"))
from squares import _official_metric as metric
from squares.core import certify

S2 = math.sqrt(2.0)
INV_S2 = 1.0 / S2

# Dominant variable-slope angle for n=908 public incumbent and rational lattice frames:
# arctan(17/38) = 0.4206317 rad approx 24.1005 deg
THETA_24_10_RAD = 0.4206311914973453
THETA_24_10_DEG = 24.100448

SIGNS = tuple(product((-1.0, 1.0), repeat=2))


@dataclass
class OptimizerResult:
    q: np.ndarray
    L: float
    iterations: int
    accepted_steps: int
    feasible: bool
    clearance: float
    message: str = "ok"


def extent(q: np.ndarray) -> float:
    """Compute the maximum bounding extent max(hi - lo) across coordinate axes."""
    c = np.abs(np.cos(q[:, 2]))
    s = np.abs(np.sin(q[:, 2]))
    h = 0.5 * (c + s)
    lo = np.min(q[:, :2] - h[:, None], axis=0)
    hi = np.max(q[:, :2] + h[:, None], axis=0)
    return float(np.max(hi - lo))


def normalize_box(q: np.ndarray, margin: float = 0.0) -> np.ndarray:
    """Translate square centers so minimum corner extent starts at margin."""
    q = q.copy()
    h = 0.5 * (np.abs(np.cos(q[:, 2])) + np.abs(np.sin(q[:, 2])))
    lo = np.min(q[:, :2] - h[:, None], axis=0)
    q[:, :2] -= lo - margin
    return q


def sat_check(q: np.ndarray, clearance: float = 0.0) -> tuple[bool, float]:
    """Check non-overlap between all pairs of squares under the Separating Axis Theorem."""
    q = np.asarray(q, dtype=np.longdouble)
    c, s = np.cos(q[:, 2]), np.sin(q[:, 2])
    u = np.column_stack((c, s))
    v = np.column_stack((-s, c))
    minimum = np.longdouble(np.inf)
    for i in range(len(q) - 1):
        d = q[i + 1 :, :2] - q[i, :2]
        best = np.full(len(d), -np.inf, dtype=np.longdouble)
        for owner_axes in (False, True):
            for basis in (u, v):
                a = basis[i + 1 :] if owner_axes else np.broadcast_to(basis[i], d.shape)
                hi = 0.5 * (np.abs(a @ u[i]) + np.abs(a @ v[i]))
                hj = 0.5 * (np.abs(np.sum(a * u[i + 1 :], axis=1)) + np.abs(np.sum(a * v[i + 1 :], axis=1)))
                g = np.abs(np.sum(d * a, axis=1)) - hi - hj
                best = np.maximum(best, g)
        minimum = min(minimum, np.min(best))
        if np.any(best < clearance):
            return False, float(minimum)
    return True, float(minimum)


def chosen_axis(q: np.ndarray, i: int, j: int, previous=None, hysteresis: float = 1e-9):
    """Determine the optimal separating face-normal axis between square i and square j."""
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
    """Compute linearized gap and gradient for one sign piece (s1, s2) of the SAT support."""
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


def solve_epigraph_lp(
    q: np.ndarray,
    upper_L: float,
    free_mask: np.ndarray,
    position_trust: float = 0.15,
    angle_trust: float = 0.02,
    clearance: float = 5e-9,
    axes: dict | None = None,
) -> tuple[np.ndarray | None, str]:
    """Solve one step of the 4-sign-piece epigraph LP polisher.

    Supports heterogeneous freedom: squares where free_mask[i] is True have 3 degrees
    of freedom (dx, dy, dtheta); upright or frozen squares where free_mask[i] is False
    have 2 degrees of freedom (dx, dy) with dtheta identically constrained to 0.
    """
    from scipy.sparse import coo_matrix

    n = len(q)
    if axes is None:
        axes = {}

    # Map each square i to its variable offsets
    offsets: list[int] = []
    curr = 0
    for i in range(n):
        offsets.append(curr)
        curr += 3 if free_mask[i] else 2
    n_square_vars = curr
    # Additional global variables: x_min, y_min, L
    nv = n_square_vars + 3
    col_xmin = n_square_vars
    col_ymin = n_square_vars + 1
    col_L = n_square_vars + 2

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

    # Boundary box constraints for each square i
    for i in range(n):
        t = float(q[i, 2])
        c, s = math.cos(t), math.sin(t)
        off = offsets[i]
        is_free = bool(free_mask[i])

        pieces = (
            [(0.5 * (a * c + b * s), 0.5 * (-a * s + b * c)) for a, b in SIGNS]
            if is_free
            else [(0.5 * (abs(c) + abs(s)), 0.0)]
        )

        for h, dh in pieces:
            # Lower bound: q[i, dim] + dq[i, dim] - h - dh * dtheta >= min_bound + clearance
            # Upper bound: q[i, dim] + dq[i, dim] + h + dh * dtheta <= min_bound + L - clearance
            for dim in (0, 1):
                col_coord = off + dim
                col_min = col_xmin if dim == 0 else col_ymin

                lower_coeff = {col_coord: -1.0, col_min: 1.0}
                upper_coeff = {col_coord: 1.0, col_min: -1.0, col_L: -1.0}
                if is_free and abs(dh) > 1e-12:
                    lower_coeff[off + 2] = dh
                    upper_coeff[off + 2] = dh

                row(lower_coeff, float(q[i, dim]) - h - clearance)
                row(upper_coeff, -float(q[i, dim]) - h - clearance)

    # Pairwise non-overlap contact constraints via spatial neighbor query
    radius = S2 + 2.0 * S2 * position_trust + 1e-9
    tree = cKDTree(q[:, :2])
    pairs = sorted(tree.query_pairs(radius))

    for i, j in pairs:
        key = (i, j)
        axis = chosen_axis(q, i, j, axes.get(key))
        axes[key] = axis
        off_i = offsets[i]
        off_j = offsets[j]
        free_i = bool(free_mask[i])
        free_j = bool(free_mask[j])

        if free_i or free_j:
            for signs in SIGNS:
                g, grad = piece_gap_gradient(q, i, j, axis, signs)
                coeff = {
                    off_i: -grad[0],
                    off_i + 1: -grad[1],
                    off_j: -grad[3],
                    off_j + 1: -grad[4],
                }
                if free_i:
                    coeff[off_i + 2] = -grad[2]
                if free_j:
                    coeff[off_j + 2] = -grad[5]
                row(coeff, g - clearance)
        else:
            owner, k, direction = axis
            phi = float(q[owner, 2]) + k * math.pi / 2.0
            a = direction * np.array([math.cos(phi), math.sin(phi)])
            h = sum(
                0.5 * (abs(math.cos(float(q[t, 2]) - phi)) + abs(math.sin(float(q[t, 2]) - phi)))
                for t in (i, j)
            )
            g = float((q[j, :2] - q[i, :2]) @ a - h)
            row(
                {
                    off_i: a[0],
                    off_i + 1: a[1],
                    off_j: -a[0],
                    off_j + 1: -a[1],
                },
                g - clearance,
            )

    # Variable bounds: trust regions on displacements
    bounds = []
    for i in range(n):
        bounds.extend([(-position_trust, position_trust)] * 2)
        if free_mask[i]:
            bounds.append((-angle_trust, angle_trust))
    bounds.extend([(None, None), (None, None), (1.0, upper_L)])

    objective = np.zeros(nv)
    objective[col_L] = 1.0

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
            "time_limit": 6.0,
        },
    )

    if not res.success or not np.isfinite(res.x).all():
        return None, str(res.message)

    out = q.copy()
    sol = res.x
    for i in range(n):
        off = offsets[i]
        out[i, 0] += sol[off]
        out[i, 1] += sol[off + 1]
        if free_mask[i]:
            out[i, 2] += sol[off + 2]

    return out, str(res.message)


def optimize_variable_slope_band(
    q_init: np.ndarray,
    free_mask: np.ndarray | None = None,
    iterations: int = 40,
    position_trust: float = 0.15,
    angle_trust: float = 0.015,
    clearance: float = 5e-9,
    target_dominant_theta: float = THETA_24_10_RAD,
) -> OptimizerResult:
    """Run continuous variable-slope optimization around theta = 24.10 degrees.

    Iterates between:
      1. Fixed-angle LP relaxation for rapid primal feasibility recovery
      2. Free-angle 4-sign-piece epigraph LP proposals for continuous angle adjustments
      3. Geometric backtracking line searches certified against non-linear SAT clearance
    """
    q = np.array(q_init, dtype=float).copy()
    n = len(q)

    # If no free_mask provided, detect tilted squares near target_dominant_theta
    if free_mask is None:
        free_mask = np.zeros(n, dtype=bool)
        for i in range(n):
            th = abs(q[i, 2])
            # Consider free if tilted beyond upright tolerance (> 0.05 rad)
            if th > 0.05 and abs(th - target_dominant_theta) < 0.25:
                free_mask[i] = True

    axes: dict[tuple[int, int], list[tuple[float, float, int]]] = {}
    L_curr = extent(q)

    # Initial repair pass (fixed angle) to guarantee strict feasibility
    fixed_mask_all = np.zeros(n, dtype=bool)
    repaired, msg = solve_epigraph_lp(
        q, L_curr + 0.15, fixed_mask_all, position_trust, 0.0, clearance, axes
    )
    if repaired is not None and sat_check(repaired, clearance)[0]:
        q = repaired
        L_curr = extent(q)

    best_q = q.copy()
    best_L = L_curr
    accepted_steps = 0

    for it in range(iterations):
        old_L = best_L

        # Stage A: Positional relaxation at current angles
        fixed_step, _ = solve_epigraph_lp(
            best_q, best_L + 1e-4, fixed_mask_all, position_trust, 0.0, clearance, axes
        )
        if fixed_step is not None:
            sat_ok, sat_min = sat_check(fixed_step, clearance)
            ext = extent(fixed_step)
            if sat_ok and ext < best_L:
                best_q = fixed_step
                best_L = ext
                accepted_steps += 1

        # Stage B: Free-angle 4-sign-piece epigraph step
        proposal, _ = solve_epigraph_lp(
            best_q, best_L + 1e-4, free_mask, position_trust, angle_trust, clearance, axes
        )
        if proposal is not None:
            origin = best_q.copy()
            step_accepted = False
            for bt in range(12):
                alpha = 2.0 ** (-bt)
                candidate = origin + alpha * (proposal - origin)
                # Geometric projection / repair pass
                rep, _ = solve_epigraph_lp(
                    candidate, best_L + 1e-3, fixed_mask_all, position_trust, 0.0, clearance, {}
                )
                if rep is not None:
                    sat_ok, _ = sat_check(rep, clearance)
                    ext = extent(rep)
                    if sat_ok and ext < best_L:
                        best_q = rep
                        best_L = ext
                        accepted_steps += 1
                        step_accepted = True
                        break
            if not step_accepted:
                # Direct check on candidate if repair was unnecessary
                sat_ok, _ = sat_check(proposal, clearance)
                ext = extent(proposal)
                if sat_ok and ext < best_L:
                    best_q = proposal
                    best_L = ext
                    accepted_steps += 1

        # Early stopping if progress stalls below numerical threshold
        if (old_L - best_L) < 1e-10 and it >= 5:
            break

    is_feasible, min_gap = sat_check(best_q, clearance)
    return OptimizerResult(
        q=best_q,
        L=best_L,
        iterations=it + 1,
        accepted_steps=accepted_steps,
        feasible=is_feasible,
        clearance=min_gap,
    )


def build_variable_slope_pocket(
    n_upright: int = 708,
    column_counts: Sequence[int] = (5, 7, 8, 8, 9, 8, 8, 9, 9, 8, 8, 8, 5),
    theta_0: float = THETA_24_10_RAD,
    u_pitch: float = 1.008,
    v_pitch: float = 0.998,
    nw_center: tuple[float, float] = (8.34, 24.91),
    offset_se: tuple[float, float] = (-3.0, 2.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Construct an initial synthetic seed with decoupled variable-slope pockets at theta_0.

    Matches the n=908 public incumbent topology:
      - 708 upright squares on Cartesian half-integer grid (with cavity cutouts)
      - NW tilted block of sum(column_counts) = 100 squares rotated near theta_0 = 24.10 deg
      - SE tilted block of 100 squares satisfying transposition symmetry (y - 3, x + 2, -theta)
    """
    total_tilted = sum(column_counts) * 2
    total_n = n_upright + total_tilted

    tilted_nw: list[list[float]] = []
    cos_th = math.cos(theta_0)
    sin_th = math.sin(theta_0)

    # Tangent and normal vectors along band slope
    t_hat = np.array([cos_th, sin_th])
    n_hat = np.array([-sin_th, cos_th])

    n_cols = len(column_counts)
    u_mid = (n_cols - 1) * 0.5 * u_pitch

    for c, count in enumerate(column_counts):
        u_col = (c * u_pitch) - u_mid
        v_mid = (count - 1) * 0.5 * v_pitch
        for s in range(count):
            v_row = (s * v_pitch) - v_mid
            pos = np.array(nw_center) + u_col * t_hat + v_row * n_hat
            tilted_nw.append([float(pos[0]), float(pos[1]), theta_0])

    tilted_nw_arr = np.array(tilted_nw)

    # Transposition symmetry for SE block: (y - 3, x + 2, -theta)
    tilted_se_arr = np.zeros_like(tilted_nw_arr)
    tilted_se_arr[:, 0] = tilted_nw_arr[:, 1] + offset_se[0]
    tilted_se_arr[:, 1] = tilted_nw_arr[:, 0] + offset_se[1]
    tilted_se_arr[:, 2] = -tilted_nw_arr[:, 2]

    # Combine tilted blocks
    all_tilted = np.vstack([tilted_nw_arr, tilted_se_arr])

    # Upright squares: grid of side K with cavity avoidance
    k = int(math.ceil(math.sqrt(total_n)))
    upright: list[list[float]] = []
    tilted_tree = cKDTree(all_tilted[:, :2])

    for i in range(k):
        for j in range(k):
            pt = np.array([i + 0.5, j + 0.5])
            # Check proximity to any tilted square
            dists, _ = tilted_tree.query(pt, k=1)
            # Upright square half-extent is 0.5; rotated square extent is approx 0.67
            if dists > 1.15:
                upright.append([pt[0], pt[1], 0.0])
            if len(upright) == n_upright:
                break
        if len(upright) == n_upright:
            break

    # If cavity cut left upright count short, pad along safe outer borders
    while len(upright) < n_upright:
        idx = len(upright)
        upright.append([float(k + (idx % 2) + 0.5), float(idx // 2 + 0.5), 0.0])

    upright_arr = np.array(upright)
    q = np.vstack([upright_arr, all_tilted])

    # Mark tilted squares as free
    free_mask = np.zeros(len(q), dtype=bool)
    free_mask[len(upright_arr) :] = True

    return q, free_mask


def rows(q: np.ndarray, dilate: float = 1.000000002) -> list[dict[str, str]]:
    """Format packing coordinates into decimal strings for submission CSV."""
    q_norm = normalize_box(q, margin=1e-8)
    L = extent(q_norm) * dilate
    res: list[dict[str, str]] = []
    for i in range(len(q_norm)):
        res.append({
            "square_id": str(i),
            "x": f"{Decimal(str(q_norm[i, 0])):.17f}",
            "y": f"{Decimal(str(q_norm[i, 1])):.17f}",
            "angle": f"{Decimal(str(q_norm[i, 2])):.17f}",
        })
    return res


def csv_text(candidates: dict[int, list[dict[str, str]]]) -> str:
    """Format puzzle dictionary into standard submission CSV string."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["id", "n", "square_id", "x", "y", "angle"])
    row_id = 0
    for n in sorted(candidates):
        for entry in candidates[n]:
            writer.writerow([
                row_id,
                n,
                entry["square_id"],
                entry["x"],
                entry["y"],
                entry["angle"],
            ])
            row_id += 1
    return buf.getvalue()


def certify_transport(
    text: str,
    temp_csv_path: Path | None = None,
) -> dict[str, dict[str, dict[str, str | bool]]]:
    """Certify literal CSV string and actual pandas 3.1.0 round-trip with official metric."""
    puzzles_literal = metric._parse_submission_csv(text)
    out: dict[str, dict[str, dict[str, str | bool]]] = {}

    if temp_csv_path is not None:
        temp_csv_path.parent.mkdir(parents=True, exist_ok=True)
        temp_csv_path.write_text(text)
        df = pd.read_csv(temp_csv_path)
        puzzles_pandas = {}
        for n_val, group in df.groupby("n"):
            puzzles_pandas[int(n_val)] = [
                {
                    "square_id": str(r["square_id"]),
                    "x": str(r["x"]),
                    "y": str(r["y"]),
                    "angle": str(r["angle"]),
                }
                for _, r in group.iterrows()
            ]
    else:
        puzzles_pandas = puzzles_literal

    for n, puz in puzzles_literal.items():
        _, _, side_lit, reason_lit = metric._evaluate_puzzle_worker((n, puz))
        _, _, side_pan, reason_pan = metric._evaluate_puzzle_worker((n, puzzles_pandas[n]))
        out[str(n)] = {
            "literal": {"ok": reason_lit == "ok", "reason": reason_lit, "L": side_lit or ""},
            "actual_pandas": {"ok": reason_pan == "ok", "reason": reason_pan, "L": side_pan or ""},
        }
    return out
