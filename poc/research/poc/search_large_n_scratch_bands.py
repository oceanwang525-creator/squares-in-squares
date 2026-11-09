"""From-scratch thick tilted band and cross-band packing generator for large n (Idea 1).

Constructs corner-to-corner thick tilted band and orthogonal cross-band packings for
the top 40 excess targets (n in [100, 1000]) from scratch. Treats tilt angle theta in [0, pi/2],
band width w, and offset as free continuous variables, filling triangular corner domains
with aligned upright unit-square grids.

For candidates within 1e-3 of incumbent L (and for top candidates), polishes with
4-sign-piece epigraph LP relaxation, recording best L, incumbent L, delta L, and
optimal parameters for each target in poc/research/data/large_n_scratch_band_results.json.
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
# Band and Corner Domain Layout Generation
# ---------------------------------------------------------------------------

def generate_scratch_band(
    L: float,
    theta: float,
    w: int,
    v_off: float = 0.0,
    u_off: float = 0.0,
    stagger: float = 0.0,
    cross_band: bool = False,
    grid_align: str = "corners",
) -> np.ndarray:
    """Generate corner-to-corner thick tilted band with upright triangular corner grids from scratch."""
    xc, yc = L / 2.0, L / 2.0
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    h_t = 0.5 * (abs(cos_t) + abs(sin_t))
    max_c = int(math.ceil(L * 1.5))

    tilted_list: list[list[float]] = []

    def add_band(th, cos_val, sin_val, h_val, v_shift, u_shift):
        for s in range(w):
            vs = (s - (w - 1) / 2.0) * 1.0 + v_shift
            for c in range(-max_c, max_c + 1):
                uc = c * 1.0 + u_shift + (s % 2) * stagger
                x = xc + uc * cos_val - vs * sin_val
                y = yc + uc * sin_val + vs * cos_val
                if h_val - 1e-6 <= x <= L - h_val + 1e-6 and h_val - 1e-6 <= y <= L - h_val + 1e-6:
                    tilted_list.append([x, y, th])

    # Primary diagonal band
    add_band(theta, cos_t, sin_t, h_t, v_off, u_off)

    # Orthogonal cross-band layout
    if cross_band:
        th_cross = theta + math.pi / 2.0
        cos_c, sin_c = math.cos(th_cross), math.sin(th_cross)
        h_c = 0.5 * (abs(cos_c) + abs(sin_c))
        for s in range(w):
            vs = (s - (w - 1) / 2.0) * 1.0 - v_off
            for c in range(-max_c, max_c + 1):
                uc = c * 1.0 - u_off + (s % 2) * stagger
                x = xc + uc * cos_c - vs * sin_c
                y = yc + uc * sin_c + vs * cos_c
                if h_c - 1e-6 <= x <= L - h_c + 1e-6 and h_c - 1e-6 <= y <= L - h_c + 1e-6:
                    cand = [x, y, th_cross]
                    conflict = False
                    for existing in tilted_list:
                        dx = existing[0] - x
                        dy = existing[1] - y
                        if dx * dx + dy * dy < 2.0:
                            th_e = existing[2]
                            diff_e = abs(dx * math.cos(th_e) + dy * math.sin(th_e))
                            diff_c = abs(dx * cos_c + dy * sin_c)
                            if diff_e < 1.0 - 1e-7 and diff_c < 1.0 - 1e-7:
                                conflict = True
                                break
                    if not conflict:
                        tilted_list.append(cand)

    tilted_arr = np.array(tilted_list, dtype=np.float64) if tilted_list else np.zeros((0, 3))
    k = int(math.ceil(L))
    upright_candidates: list[list[float]] = []
    placed_keys: set[tuple[int, int]] = set()

    def try_add_upright(xu: float, yu: float):
        if not (0.5 - 1e-6 <= xu <= L - 0.5 + 1e-6 and 0.5 - 1e-6 <= yu <= L - 0.5 + 1e-6):
            return
        gx = int(round(xu * 100))
        gy = int(round(yu * 100))
        for kx, ky in placed_keys:
            if abs(kx - gx) < 99 and abs(ky - gy) < 99:
                return
        if not sat_check_upright_against_tilted(xu, yu, tilted_arr, clearance=-1e-7):
            return
        placed_keys.add((gx, gy))
        upright_candidates.append([xu, yu, 0.0])

    if grid_align == "corners":
        # Fill triangular corner domains anchored to the outer container boundaries
        # NW triangular domain (top-left)
        # SE triangular domain (bottom-right)
        # SW and NE outer corners
        for i in range(k + 1):
            for j in range(k + 1):
                try_add_upright(0.5 + i, L - 0.5 - j)  # NW
                try_add_upright(L - 0.5 - i, 0.5 + j)  # SE
                try_add_upright(0.5 + i, 0.5 + j)      # SW
                try_add_upright(L - 0.5 - i, L - 0.5 - j)  # NE
    elif grid_align == "origin":
        for i in range(k + 1):
            for j in range(k + 1):
                try_add_upright(0.5 + i, 0.5 + j)

    upright_arr = np.array(upright_candidates, dtype=np.float64) if upright_candidates else np.zeros((0, 3))
    if len(tilted_arr) == 0:
        return upright_arr
    if len(upright_arr) == 0:
        return tilted_arr
    return np.vstack([tilted_arr, upright_arr])


def prune_to_n(q: np.ndarray, target_n: int) -> np.ndarray:
    """Select exactly target_n squares that minimize the unconstrained bounding extent."""
    if len(q) <= target_n:
        return normalize_box(q)

    current = q.copy()
    while len(current) > target_n:
        center = np.mean(current[:, :2], axis=0)
        dist_sq = np.sum((current[:, :2] - center) ** 2, axis=1)
        outer_candidates = np.argsort(-dist_sq)[: min(10, len(current))]

        best_idx = outer_candidates[0]
        best_L = float("inf")
        c = np.abs(np.cos(current[:, 2]))
        s = np.abs(np.sin(current[:, 2]))
        h = 0.5 * (c + s)
        lo = current[:, :2] - h[:, None]
        hi = current[:, :2] + h[:, None]

        for idx in outer_candidates:
            mask = np.ones(len(current), dtype=bool)
            mask[idx] = False
            cur_lo = np.min(lo[mask], axis=0)
            cur_hi = np.max(hi[mask], axis=0)
            cur_extent = float(np.max(cur_hi - cur_lo))
            if cur_extent < best_L:
                best_L = cur_extent
                best_idx = idx

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


def search_target(
    target_n: int,
    incumbent_L: float,
    theta_grid: Sequence[float],
    widths: Sequence[int],
    offsets: Sequence[float],
) -> TargetResult:
    """Search continuous tilt angle theta in [0, pi/2], band width w, and offset for target n."""
    best_candidate: np.ndarray | None = None
    best_unpolished_L = float("inf")
    best_params: dict[str, Any] = {}

    k = int(math.floor(math.sqrt(target_n)))
    k_ceil = int(math.ceil(math.sqrt(target_n)))

    trial_Ls = [
        incumbent_L,
        float(k_ceil),
    ]

    # Grid search across theta, width, offset, and single vs cross architectures
    for L_trial in trial_Ls:
        for w in widths:
            for theta in theta_grid:
                for v_off in offsets:
                    for cross in (False, True):
                        seed = generate_scratch_band(
                            L=L_trial,
                            theta=theta,
                            w=w,
                            v_off=v_off,
                            u_off=0.0,
                            cross_band=cross,
                            grid_align="corners",
                        )
                        if len(seed) >= target_n:
                            pruned = prune_to_n(seed, target_n)
                            valid, _ = sat_check(pruned, clearance=-1e-7)
                            if valid:
                                L_ext = extent(pruned)
                                if L_ext < best_unpolished_L:
                                    best_unpolished_L = L_ext
                                    best_candidate = pruned.copy()
                                    best_params = {
                                        "theta": theta,
                                        "w": w,
                                        "v_off": v_off,
                                        "u_off": 0.0,
                                        "cross_band": cross,
                                        "L_trial": L_trial,
                                    }

    # Continuous local refinement around best parameters
    if best_params:
        base_th = best_params["theta"]
        base_v = best_params["v_off"]
        w = best_params["w"]
        cross = best_params["cross_band"]
        L_trial = best_params["L_trial"]

        for d_th in (-0.02, 0.02):
            new_th = max(0.01, min(math.pi / 2.0 - 0.01, base_th + d_th))
            for d_v in (-0.05, 0.05):
                seed = generate_scratch_band(
                    L=L_trial,
                    theta=new_th,
                    w=w,
                    v_off=base_v + d_v,
                    u_off=0.0,
                    cross_band=cross,
                    grid_align="corners",
                )
                if len(seed) >= target_n:
                    pruned = prune_to_n(seed, target_n)
                    valid, _ = sat_check(pruned, clearance=-1e-7)
                    if valid:
                        L_ext = extent(pruned)
                        if L_ext < best_unpolished_L:
                            best_unpolished_L = L_ext
                            best_candidate = pruned.copy()
                            best_params.update({"theta": new_th, "v_off": base_v + d_v})

    # Fallback to upright grid if no candidate found
    if best_candidate is None:
        grid_sq = []
        for i in range(k_ceil):
            for j in range(k_ceil):
                if len(grid_sq) < target_n:
                    grid_sq.append([0.5 + i, 0.5 + j, 0.0])
        best_candidate = np.array(grid_sq)
        best_unpolished_L = float(k_ceil)
        best_params = {
            "theta": 0.0,
            "w": 0,
            "v_off": 0.0,
            "u_off": 0.0,
            "cross_band": False,
        }

    # LP Polishing: polish if within 1e-3 of incumbent L or to find best side
    polished_q = best_candidate.copy()
    polished_L = best_unpolished_L
    should_polish = (best_unpolished_L <= incumbent_L + 1e-3) or (best_unpolished_L - incumbent_L < 0.5)

    if should_polish:
        polished_q, polished_L = polish_lp(best_candidate, max_iters=15)

    delta = polished_L - incumbent_L
    within_1e3 = delta <= 1e-3

    n_tilted = int(np.sum(np.abs(polished_q[:, 2]) > 1e-4))
    n_upright = len(polished_q) - n_tilted

    optimal_params = {
        "theta": float(best_params.get("theta", 0.0)),
        "theta_deg": float(math.degrees(best_params.get("theta", 0.0))),
        "w": int(best_params.get("w", 0)),
        "v_off": float(best_params.get("v_off", 0.0)),
        "u_off": float(best_params.get("u_off", 0.0)),
        "stagger": float(best_params.get("stagger", 0.0)),
        "family": "cross_band" if best_params.get("cross_band") else "tilted_band",
        "n_tilted": n_tilted,
        "n_upright": n_upright,
    }

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
        family=optimal_params["family"],
        optimal_parameters=optimal_params,
        layout=layout,
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

    # Compute top 40 targets with n in [100, 1000]
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
    """Run from-scratch thick tilted and cross-band search across top 40 targets."""
    t0 = time.time()
    t_path = targets_file or DEFAULT_TARGETS_PATH
    out_path = output_file or DEFAULT_OUTPUT_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)

    targets_list = load_or_compute_top40(t_path)
    if max_targets is not None:
        targets_list = targets_list[:max_targets]

    # Continuous angle grid spanning [0, pi/2]: key rational slopes + uniform angles
    key_thetas = [
        math.pi / 4.0,           # 45.00 deg
        math.atan(0.5),          # 26.57 deg (1:2 rational slope)
        math.atan(2.0),          # 63.43 deg (2:1 rational slope)
        math.atan(1.0 / 3.0),    # 18.43 deg (1:3 rational slope)
        math.atan(3.0),          # 71.57 deg (3:1 rational slope)
        math.atan(2.0 / 3.0),    # 33.69 deg (2:3 rational slope)
        math.atan(1.5),          # 56.31 deg (3:2 rational slope)
        math.atan(3.0 / 4.0),    # 36.87 deg (3:4 Pell/Cantrell slope)
        math.atan(4.0 / 3.0),    # 53.13 deg (4:3 Pell/Cantrell slope)
        math.atan(1.0 / 10.0),   # 5.71 deg (1:k asymptotic slope)
    ]
    uniform_thetas = np.linspace(0.10, math.pi / 2.0 - 0.10, 10).tolist()
    theta_grid = sorted(set(round(th, 6) for th in key_thetas + uniform_thetas))

    offsets = [-0.25, 0.0, 0.25]
    all_results: dict[str, Any] = {}

    for idx, target in enumerate(targets_list, 1):
        target_n = int(target["n"])
        inc_L = float(target["L"])

        k = int(math.floor(math.sqrt(target_n)))
        widths = [1, 2, 3]

        res = search_target(
            target_n=target_n,
            incumbent_L=inc_L,
            theta_grid=theta_grid,
            widths=widths,
            offsets=offsets,
        )

        all_results[str(target_n)] = {
            "target_n": res.target_n,
            "incumbent_L": res.incumbent_L,
            "unpolished_L": res.unpolished_L,
            "best_L": res.best_L,
            "delta_L": res.delta_L,
            "within_1e-3": res.within_1e3,
            "family": res.family,
            "optimal_parameters": res.optimal_parameters,
            "layout": res.layout,
        }

    elapsed = time.time() - t0
    within_count = sum(1 for r in all_results.values() if r["within_1e-3"])
    improved_count = sum(1 for r in all_results.values() if r["delta_L"] < -1e-9)

    summary = {
        "elapsed_seconds": elapsed,
        "total_targets": len(targets_list),
        "within_1e-3_count": within_count,
        "improved_count": improved_count,
        "targets": all_results,
    }

    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

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
