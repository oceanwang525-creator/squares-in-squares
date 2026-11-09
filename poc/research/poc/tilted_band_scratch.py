"""Scratch tilted and cross-band packing generator (Idea 1 / pjmathematician construction).

Constructs corner-to-corner thick tilted band and cross-band packings for target n values
from scratch rather than perturbing incumbents. Treats tilt angle theta in [0, pi/2],
band width w, and offset as free continuous/discrete variables, filling upper-left and
lower-right triangular regions with aligned unit-square grids.

Evaluates unconstrained bounding side L, polishes candidates within 1e-3 of incumbent
with scipy LP/SLP (using 4-sign-piece epigraph linearization), and outputs all generated
layouts and deltas vs merged-literal-cpp.json to poc/research/data/scratch_band_results.json.
"""
from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[3]
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
MERGED_LITERAL_PATH = REPO / "poc/pell_corrugation/data/border-scan/merged-literal-cpp.json"

TARGET_NS: tuple[int, ...] = (
    5, 6, 7,
    10, 11, 12,
    17, 18, 19,
    26, 27, 28, 29,
    37, 38, 39,
    50, 51, 52, 53, 54, 55,
    65, 66, 67, 68, 69, 70,
)

SIGNS: tuple[tuple[float, float], ...] = tuple(product((-1.0, 1.0), repeat=2))
S2 = math.sqrt(2.0)


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


def sat_check(q: np.ndarray, clearance: float = 0.0) -> tuple[bool, float]:
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


def chosen_axis(q: np.ndarray, i: int, j: int, previous=None, hysteresis: float = 1e-9):
    """Find dominant separating axis between square i and j."""
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
    """4-sign-piece epigraph LP step to shrink container side L."""
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
    max_iters: int = 25,
    position_trust: float = 0.12,
    angle_trust: float = 0.012,
    clearance: float = 1e-8,
) -> tuple[np.ndarray, float]:
    """Sequential LP polishing using 4-sign-piece epigraph relaxation."""
    best_q = normalize_box(q)
    best_L = extent(best_q)
    axes: dict[tuple[int, int], Any] = {}

    for it in range(max_iters):
        old_L = best_L
        # Position LP
        fixed, _ = _lp_step(
            best_q, best_L + 1e-4, position_trust, 0.0, clearance, axes
        )
        if fixed is not None:
            valid, _ = sat_check(fixed, clearance=0.0)
            cur_L = extent(fixed)
            if valid and cur_L < best_L:
                best_q = normalize_box(fixed)
                best_L = cur_L

        # Angle + position LP
        proposal, _ = _lp_step(
            best_q, best_L + 1e-4, position_trust, angle_trust, clearance, axes
        )
        if proposal is not None:
            origin = best_q.copy()
            for bt in range(6):
                alpha = 2.0 ** (-bt)
                seed = origin + alpha * (proposal - origin)
                valid, _ = sat_check(seed, clearance=0.0)
                cur_L = extent(seed)
                if valid and cur_L < best_L:
                    best_q = normalize_box(seed)
                    best_L = cur_L
                    break

        if old_L - best_L < 1e-10 and it >= 5:
            break

    return best_q, best_L


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
    """Generate corner-to-corner thick tilted band with upright triangular grids from scratch."""
    xc, yc = L / 2.0, L / 2.0
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    h_t = 0.5 * (abs(cos_t) + abs(sin_t))
    max_c = int(math.ceil(L * 1.5))

    tilted_list: list[list[float]] = []

    def add_band(th, cos_val, sin_val, h_val, v_shift):
        for s in range(w):
            vs = (s - (w - 1) / 2.0) * 1.0 + v_shift
            for c in range(-max_c, max_c + 1):
                uc = c * 1.0 + u_off + (s % 2) * stagger
                x = xc + uc * cos_val - vs * sin_val
                y = yc + uc * sin_val + vs * cos_val
                if h_val - 1e-6 <= x <= L - h_val + 1e-6 and h_val - 1e-6 <= y <= L - h_val + 1e-6:
                    tilted_list.append([x, y, th])

    # Main tilted band
    add_band(theta, cos_t, sin_t, h_t, v_off)

    # Optional cross-band (orthogonal or reflected)
    if cross_band:
        th_cross = math.pi / 2.0 - theta if abs(theta - math.pi / 4.0) > 1e-4 else 3.0 * math.pi / 4.0
        cos_c, sin_c = math.cos(th_cross), math.sin(th_cross)
        h_c = 0.5 * (abs(cos_c) + abs(sin_c))
        # Add non-overlapping squares from cross band
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
                        # Simple distance gate and SAT check
                        dx = existing[0] - x
                        dy = existing[1] - y
                        if dx * dx + dy * dy < 2.0:
                            # SAT overlap test
                            cand_arr = np.array([cand])
                            ex_arr = np.array([existing])
                            if not sat_check(np.vstack([cand_arr, ex_arr]), clearance=-1e-9)[0]:
                                conflict = True
                                break
                    if not conflict:
                        tilted_list.append(cand)

    tilted_arr = np.array(tilted_list, dtype=np.float64) if tilted_list else np.zeros((0, 3))

    # Fill remaining upper-left, lower-right, and corner regions with aligned upright unit-square grids
    k = int(math.ceil(L))
    upright_candidates: list[list[float]] = []

    def try_add_upright(xu: float, yu: float):
        if not (0.5 - 1e-6 <= xu <= L - 0.5 + 1e-6 and 0.5 - 1e-6 <= yu <= L - 0.5 + 1e-6):
            return
        # Vectorized SAT against tilted band
        if not sat_check_upright_against_tilted(xu, yu, tilted_arr, clearance=-1e-9):
            return
        # Check against previously added upright squares
        for u in upright_candidates:
            if abs(u[0] - xu) < 1.0 - 1e-6 and abs(u[1] - yu) < 1.0 - 1e-6:
                return
        upright_candidates.append([xu, yu, 0.0])

    if grid_align == "corners":
        # Search from each of the 4 bounding corners inwards
        for i in range(k + 1):
            for j in range(k + 1):
                # NW triangular region
                try_add_upright(0.5 + i, L - 0.5 - j)
                # SE triangular region
                try_add_upright(L - 0.5 - i, 0.5 + j)
                # SW corner
                try_add_upright(0.5 + i, 0.5 + j)
                # NE corner
                try_add_upright(L - 0.5 - i, L - 0.5 - j)
    elif grid_align == "origin":
        for i in range(k + 1):
            for j in range(k + 1):
                try_add_upright(0.5 + i, 0.5 + j)
    elif grid_align == "center":
        x_base = xc - math.floor(xc)
        y_base = yc - math.floor(yc)
        for i in range(-k, k + 1):
            for j in range(-k, k + 1):
                try_add_upright(xc + i, yc + j)

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

    # Greedily prune squares that contribute most to bounding box extent
    current = q.copy()
    while len(current) > target_n:
        # Evaluate extent when dropping each candidate square
        best_idx = 0
        best_L = float("inf")
        c = np.abs(np.cos(current[:, 2]))
        s = np.abs(np.sin(current[:, 2]))
        h = 0.5 * (c + s)
        lo = current[:, :2] - h[:, None]
        hi = current[:, :2] + h[:, None]

        # Score squares by distance from center to prioritize removing outer perimeter squares
        center = np.mean(current[:, :2], axis=0)
        dist_sq = np.sum((current[:, :2] - center) ** 2, axis=1)
        outer_candidates = np.argsort(-dist_sq)[: min(10, len(current))]

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


@dataclass
class CandidateResult:
    target_n: int
    incumbent_L: float
    unpolished_L: float
    polished_L: float
    delta_vs_incumbent: float
    within_1e3: bool
    family: str
    theta: float
    w: int
    v_off: float
    u_off: float
    stagger: float
    n_tilted: int
    n_upright: int
    layout: list[dict[str, float]] = field(default_factory=list)


def search_target(
    target_n: int,
    incumbent_L: float,
    theta_grid: Sequence[float],
    widths: Sequence[int],
    offsets: Sequence[float],
    staggers: Sequence[float] = (0.0, 0.5),
) -> CandidateResult:
    """Grid-search continuous theta, band width w, and offset from scratch for target n."""
    best_candidate: np.ndarray | None = None
    best_unpolished_L = float("inf")
    best_params: dict[str, Any] = {}

    trial_Ls = [
        incumbent_L,
        incumbent_L - 0.002,
        incumbent_L + 0.005,
        incumbent_L + 0.015,
    ]

    for L_trial in trial_Ls:
        for w in widths:
            for theta in theta_grid:
                for v_off in offsets:
                    for u_off in (-0.5, 0.0, 0.5):
                        for stagger in staggers:
                            for cross_band in (False, True) if target_n in (5, 10, 17, 26, 37, 50, 65) else (False,):
                                for align_mode in ("corners", "origin"):
                                    seed = generate_scratch_band(
                                        L=L_trial,
                                        theta=theta,
                                        w=w,
                                        v_off=v_off,
                                        u_off=u_off,
                                        stagger=stagger,
                                        cross_band=cross_band,
                                        grid_align=align_mode,
                                    )
                                    if len(seed) >= target_n:
                                        pruned = prune_to_n(seed, target_n)
                                        # Verify SAT clearance
                                        valid, _ = sat_check(pruned, clearance=-1e-9)
                                        if valid:
                                            L_extent = extent(pruned)
                                            if L_extent < best_unpolished_L:
                                                best_unpolished_L = L_extent
                                                best_candidate = pruned.copy()
                                                best_params = {
                                                    "theta": theta,
                                                    "w": w,
                                                    "v_off": v_off,
                                                    "u_off": u_off,
                                                    "stagger": stagger,
                                                    "cross_band": cross_band,
                                                    "align_mode": align_mode,
                                                }

    # If no candidate found, fallback to pure upright grid with top-right squares
    if best_candidate is None:
        k = int(math.ceil(math.sqrt(target_n)))
        grid_sq = []
        for i in range(k):
            for j in range(k):
                if len(grid_sq) < target_n:
                    grid_sq.append([0.5 + i, 0.5 + j, 0.0])
        best_candidate = np.array(grid_sq)
        best_unpolished_L = float(k)
        best_params = {
            "theta": 0.0,
            "w": 0,
            "v_off": 0.0,
            "u_off": 0.0,
            "stagger": 0.0,
            "cross_band": False,
            "align_mode": "grid",
        }

    # Polish candidate if within 1e-3 of incumbent (or polish top candidate)
    polished_q = best_candidate.copy()
    polished_L = best_unpolished_L
    should_polish = (best_unpolished_L <= incumbent_L + 1e-3) or (best_unpolished_L - incumbent_L < 0.05)
    if should_polish:
        polished_q, polished_L = polish_lp(best_candidate, max_iters=20)

    delta = polished_L - incumbent_L
    within_1e3 = delta <= 1e-3

    n_tilted = int(np.sum(np.abs(polished_q[:, 2]) > 1e-4))
    n_upright = len(polished_q) - n_tilted

    layout = [
        {
            "id": i,
            "x": float(polished_q[i, 0]),
            "y": float(polished_q[i, 1]),
            "angle": float(polished_q[i, 2]),
        }
        for i in range(len(polished_q))
    ]

    return CandidateResult(
        target_n=target_n,
        incumbent_L=incumbent_L,
        unpolished_L=best_unpolished_L,
        polished_L=polished_L,
        delta_vs_incumbent=delta,
        within_1e3=within_1e3,
        family="cross_band" if best_params.get("cross_band") else "tilted_band",
        theta=float(best_params.get("theta", 0.0)),
        w=int(best_params.get("w", 0)),
        v_off=float(best_params.get("v_off", 0.0)),
        u_off=float(best_params.get("u_off", 0.0)),
        stagger=float(best_params.get("stagger", 0.0)),
        n_tilted=n_tilted,
        n_upright=n_upright,
        layout=layout,
    )


def run_all(output_file: Path | None = None) -> dict[str, Any]:
    """Execute grid search and LP polish across all requested target n values."""
    t0 = time.time()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = output_file or (DATA_DIR / "scratch_band_results.json")

    # Load incumbent L from merged-literal-cpp.json
    if not MERGED_LITERAL_PATH.exists():
        raise FileNotFoundError(f"Missing {MERGED_LITERAL_PATH}")
    with open(MERGED_LITERAL_PATH) as f:
        merged_data = json.load(f)
    incumbents: dict[int, float] = {r["n"]: float(r["L"]) for r in merged_data["results"]}

    # Define theta search grid: key rational/structural angles + fine uniform grid
    key_thetas = [
        math.pi / 4.0,           # 45.000 deg
        math.atan(0.5),          # 26.565 deg (1:2 rational slope)
        math.atan(2.0),          # 63.435 deg (2:1 rational slope)
        math.atan(1.0 / 3.0),    # 18.435 deg
        math.atan(3.0),          # 71.565 deg
        math.atan(2.0 / 3.0),    # 33.690 deg
        math.atan(1.5),          # 56.310 deg
        math.atan(3.0 / 4.0),    # 36.870 deg (Pell/Cantrell slope)
        math.atan(4.0 / 3.0),    # 53.130 deg
        0.4240,                  # n=18 Friedman empirical angle
        0.7013,                  # n=11 Friedman empirical angle
        0.6435,                  # n=50 empirical angle
    ]
    uniform_thetas = np.linspace(0.20, math.pi / 2.0 - 0.20, 20).tolist()
    theta_grid = sorted(set(round(th, 6) for th in key_thetas + uniform_thetas))

    offsets = [-0.5, -0.25, 0.0, 0.25, 0.5]
    all_results: dict[str, Any] = {}

    for n in TARGET_NS:
        inc_L = incumbents[n]
        # Adapt widths based on container size k
        k = int(math.floor(math.sqrt(n)))
        max_w = min(k, 4)
        widths = list(range(1, max_w + 1))

        res = search_target(
            target_n=n,
            incumbent_L=inc_L,
            theta_grid=theta_grid,
            widths=widths,
            offsets=offsets,
        )
        all_results[str(n)] = {
            "target_n": res.target_n,
            "incumbent_L": res.incumbent_L,
            "unpolished_L": res.unpolished_L,
            "polished_L": res.polished_L,
            "delta_vs_incumbent": res.delta_vs_incumbent,
            "within_1e-3": res.within_1e3,
            "family": res.family,
            "best_params": {
                "theta": res.theta,
                "theta_deg": math.degrees(res.theta),
                "w": res.w,
                "v_off": res.v_off,
                "u_off": res.u_off,
                "stagger": res.stagger,
                "n_tilted": res.n_tilted,
                "n_upright": res.n_upright,
            },
            "layout": res.layout,
        }

    elapsed = time.time() - t0
    within_count = sum(1 for r in all_results.values() if r["within_1e-3"])
    improved_count = sum(1 for r in all_results.values() if r["delta_vs_incumbent"] < -1e-9)

    summary_data = {
        "elapsed_seconds": elapsed,
        "total_targets": len(TARGET_NS),
        "within_1e-3_count": within_count,
        "improved_count": improved_count,
        "targets": all_results,
    }

    with open(out_path, "w") as f:
        json.dump(summary_data, f, indent=2)

    return summary_data


if __name__ == "__main__":
    out_file = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    run_all(out_file)
