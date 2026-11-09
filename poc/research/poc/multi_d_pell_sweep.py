"""Multi-D Pell Corrugated Cores Generator and Sweep Script.

Investigates alternative square-free Pell discriminants D in {3, 10, 27, 34, 51}
as analyzed in poc/research/docs/q1_corrugation_theorem.md and poc/multi_pell_cores/.

Pipeline:
1. For each discriminant D in {3, 10, 27, 34, 51}:
   - Solve Pell equations x^2 - D * y^2 = +-1 to find fundamental and secondary units.
   - Construct tilted band corrugated cores using width parameters w appropriate
     for container sizes L in [24.0, 32.0] (covering n in [600, 1000]).
   - Certify core non-overlap via SAT separating axes and polish boundaries with 4-sign LP.
2. For each target n in [600, 1000]:
   - Wrap viable multi-D cores with structured integer borders of thickness t in [1, 4]
     and corner residual packing.
   - Apply 4-sign-piece epigraph LP contraction to minimize container side length L.
3. Compare candidate L against incumbent L from poc/pell_corrugation/data/border-scan/merged.csv.
4. Output discovered cores, candidate packings, and per-n comparisons to
   poc/research/data/multi_d_pell_results.json, writing progress after each target.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from decimal import Decimal
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.spatial import cKDTree

# Setup repository paths
REPO = Path(__file__).resolve().parents[3]
SYS_SRC = REPO / "poc" / "baseline" / "poc" / "src"
if str(SYS_SRC) not in sys.path:
    sys.path.insert(0, str(SYS_SRC))
PELL_POC = REPO / "poc" / "pell_corrugation" / "poc"
if str(PELL_POC) not in sys.path:
    sys.path.insert(0, str(PELL_POC))

try:
    from squares.core import certify
except ImportError:
    certify = None

S2 = math.sqrt(2.0)
INV_S2 = 1.0 / S2
SIGNS = tuple(product((-1.0, 1.0), repeat=2))

TARGET_DISCRIMINANTS = (3, 10, 27, 34, 51)
DEFAULT_OUTPUT_JSON = REPO / "poc" / "research" / "data" / "multi_d_pell_results.json"
INCUMBENT_CSV = REPO / "poc" / "pell_corrugation" / "data" / "border-scan" / "merged.csv"
INCUMBENT_JSON = REPO / "poc" / "pell_corrugation" / "data" / "border-scan" / "merged-literal-official.json"


# ---------------------------------------------------------------------------
# 1. Pell Equation Solver
# ---------------------------------------------------------------------------

def solve_pell(D: int) -> dict[str, Any]:
    """Solve Pell equation x^2 - D * y^2 = +-1 via continued fractions.

    Finds fundamental unit, fundamental positive unit, and secondary unit.
    """
    a0 = int(math.isqrt(D))
    if a0 * a0 == D:
        raise ValueError(f"D={D} is a perfect square")

    m = 0
    d = 1
    a = a0
    cf_period: list[int] = []

    p_prev, p_curr = 1, a0
    q_prev, q_curr = 0, 1

    fund_x, fund_y, fund_norm = 0, 0, 0

    while True:
        m = d * a - m
        d = (D - m * m) // d
        a = (a0 + m) // d
        cf_period.append(a)

        p_next = a * p_curr + p_prev
        q_next = a * q_curr + q_prev
        p_prev, p_curr = p_curr, p_next
        q_prev, q_curr = q_curr, q_next

        norm = p_prev * p_prev - D * q_prev * q_prev
        if norm in (1, -1) and q_prev > 0:
            fund_x, fund_y, fund_norm = p_prev, q_prev, norm
            break

    # Positive fundamental unit
    if fund_norm == -1:
        pos_x = fund_x * fund_x + D * fund_y * fund_y
        pos_y = 2 * fund_x * fund_y
        # Secondary positive unit is pos_unit^2
        sec_x = pos_x * pos_x + D * pos_y * pos_y
        sec_y = 2 * pos_x * pos_y
        sec_norm = 1
    else:
        pos_x, pos_y = fund_x, fund_y
        # Secondary positive unit is pos_unit^2
        sec_x = pos_x * pos_x + D * pos_y * pos_y
        sec_y = 2 * pos_x * pos_y
        sec_norm = 1

    return {
        "D": D,
        "continued_fraction": [a0, cf_period],
        "fundamental_unit": {"x": int(fund_x), "y": int(fund_y), "norm": int(fund_norm)},
        "fundamental_positive_unit": {"x": int(pos_x), "y": int(pos_y), "norm": 1},
        "secondary_unit": {"x": int(sec_x), "y": int(sec_y), "norm": int(sec_norm)},
    }


# ---------------------------------------------------------------------------
# 2. Multi-D Core Geometry & Parameters
# ---------------------------------------------------------------------------

@dataclass
class CoreParameters:
    D: int
    w: int
    is_even: bool
    b1: int
    b2: int
    a1: int
    a2: int
    l: int
    L45: float
    k0: int
    n0: int
    slack: float
    w_frac: float
    clearance_ok: bool
    default_delta: float


def derive_core_parameters(D: int, w: int) -> CoreParameters:
    """Derive algebraic parameters for multi-D corrugated core according to Q1 theorem."""
    if w <= 0:
        raise ValueError(f"Width w={w} must be positive")

    is_even = (D % 2 == 0)
    floor_d_s2 = int(math.floor(D * INV_S2))
    slack = D * INV_S2 - floor_d_s2
    w_frac = (w * INV_S2) - math.floor(w * INV_S2)
    b = int(math.floor(w * INV_S2))

    if not is_even:
        # Symmetric odd D
        b1 = b
        b2 = b
        a1 = b + (D + 1) // 2
        a2 = b + (D - 1) // 2
        clearance_ok = (0.05 <= w_frac <= 0.42)
    else:
        # Asymmetric even D
        b1 = b
        b2 = b + 1
        a1 = b + (D // 2) + 1
        a2 = b + (D // 2)
        clearance_ok = (w_frac > 0.50)

    l = w + floor_d_s2
    L45 = 0.5 * (a1 + a2) + 1.0 + w * INV_S2
    k0 = int(math.floor(L45))

    n_a1 = a1 * (a1 + 1) // 2
    n_a2 = a2 * (a2 + 1) // 2
    n_b1 = b1 * (b1 + 1) // 2
    n_b2 = b2 * (b2 + 1) // 2
    n_band = w * l
    n0 = n_a1 + n_a2 + n_b1 + n_b2 + n_band

    # Scale nominal tilt proportionally to available longitudinal slack
    default_delta = min(0.0035, max(0.0015, slack * 0.12))

    return CoreParameters(
        D=D,
        w=w,
        is_even=is_even,
        b1=b1,
        b2=b2,
        a1=a1,
        a2=a2,
        l=l,
        L45=L45,
        k0=k0,
        n0=n0,
        slack=slack,
        w_frac=w_frac,
        clearance_ok=clearance_ok,
        default_delta=default_delta,
    )


# ---------------------------------------------------------------------------
# 3. Geometry Utilities, SAT Check & LP Polisher
# ---------------------------------------------------------------------------

def extent(q: np.ndarray) -> float:
    """Compute bounding box side length L = max(max(x) - min(x), max(y) - min(y))."""
    c = np.abs(np.cos(q[:, 2]))
    s = np.abs(np.sin(q[:, 2]))
    h = 0.5 * (c + s)
    lo = np.min(q[:, :2] - h[:, None], axis=0)
    hi = np.max(q[:, :2] + h[:, None], axis=0)
    return float(np.max(hi - lo))


def normalize_box(q: np.ndarray, margin: float = 0.0) -> np.ndarray:
    """Align lower bounds of all square footprints to margin."""
    q = q.copy()
    h = 0.5 * (np.abs(np.cos(q[:, 2])) + np.abs(np.sin(q[:, 2])))
    lo = np.min(q[:, :2] - h[:, None], axis=0)
    q[:, :2] -= lo - margin
    return q


def sat_check(q: np.ndarray, clearance: float = 0.0) -> tuple[bool, float]:
    """Rigorous Separating Axis Theorem check across all square pairs in long double precision."""
    q_arr = np.asarray(q, dtype=np.longdouble)
    c, s = np.cos(q_arr[:, 2]), np.sin(q_arr[:, 2])
    u = np.column_stack((c, s))
    v = np.column_stack((-s, c))
    minimum = np.longdouble(np.inf)
    n = len(q_arr)

    for i in range(n - 1):
        d = q_arr[i + 1 :, :2] - q_arr[i, :2]
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
    """Determine maximal separating axis for square pair (i, j)."""
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
    """Compute gap and gradient for one of 4 sign pieces of support function."""
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
    """Solve one LP step with 4-sign-piece epigraph linearization under HiGHS."""
    from scipy.sparse import coo_matrix

    n = len(q)
    free = angle_trust > 0.0
    stride = 3 if free else 2
    nv = stride * n + 3
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

    radius = S2 + 2.0 * S2 * position_trust + 1e-10
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
                row(coeff, g - clearance)
        else:
            owner, k, direction = axis
            phi = float(q[owner, 2]) + k * math.pi / 2.0
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
    L_init: float,
    iterations: int = 25,
    position_trust: float = 0.20,
    angle_trust: float = 0.015,
    clearance: float = 5e-9,
) -> tuple[np.ndarray, float]:
    """Polish a packing using 4-sign-piece epigraph trust-region LP."""
    axes: dict[tuple[int, int], list[tuple[float, float, int]]] = {}
    repaired, _ = _lp_step(q, L_init + 0.15, position_trust, 0.0, clearance, axes)
    if repaired is None:
        repaired = q.copy()

    best_q = repaired.copy()
    best_L = extent(best_q)

    for it in range(iterations):
        old_L = best_L
        fixed, _ = _lp_step(best_q, old_L + 1e-4, position_trust, 0.0, clearance, axes)
        if fixed is not None and extent(fixed) < best_L:
            ok, _ = sat_check(fixed)
            if ok:
                best_q = fixed
                best_L = extent(fixed)

        if angle_trust > 0:
            proposal, _ = _lp_step(best_q, best_L + 1e-4, position_trust, angle_trust, clearance, axes)
            if proposal is not None:
                origin = best_q.copy()
                for bt in range(6):
                    alpha = 2.0 ** (-bt)
                    seed = origin + alpha * (proposal - origin)
                    rep, _ = _lp_step(seed, best_L + 1e-3, position_trust, 0.0, clearance, {})
                    if rep is not None and extent(rep) < best_L:
                        ok, _ = sat_check(rep)
                        if ok:
                            best_q = rep
                            best_L = extent(rep)
                            break

        if old_L - best_L < 1e-8 and it >= 4:
            break

    return best_q, best_L


# ---------------------------------------------------------------------------
# 4. Multi-D Tilted Band Corrugated Core Construction
# ---------------------------------------------------------------------------

def construct_corrugated_core(params: CoreParameters, delta: float | None = None) -> tuple[np.ndarray, float]:
    """Construct tilted band corrugated core coordinates for given parameters."""
    w = params.w
    l = params.l
    a1, a2 = params.a1, params.a2
    b1, b2 = params.b1, params.b2
    L0 = params.L45

    if delta is None:
        delta = params.default_delta

    q: list[list[float]] = []
    gap_se_ne = max(0.0, (L0 - b2) - a1 - 0.05)
    gap_se_sw = max(0.0, (L0 - b1) - a1 - 0.05)

    # 1. SE flank staircase (a1 steps) at (L0, 0)
    for i in range(a1):
        for j in range(a1 - i):
            x, y = L0 - 0.5 - i, 0.5 + j * (1.0 + 1e-6)
            if i == a1 - 1 and j == 0 and gap_se_sw > 0:
                x -= min(0.19, 0.45 * gap_se_sw)
            if i == 0 and j == a1 - 1 and gap_se_ne > 0:
                x -= min(0.20, 0.45 * gap_se_ne)
            q.append([x, y, 0.0])

    # 2. NW flank staircase (a2 steps) at (0, L0)
    for i in range(a2):
        for j in range(a2 - i):
            x, y = 0.5 + i * (1.0 + 1e-6), L0 - 0.5 - j
            q.append([x, y, 0.0])

    # 3. SW end staircase (b1 steps) at (0, 0)
    for i in range(b1):
        for j in range(b1 - i):
            q.append([0.5 + i * (1.0 + 1e-6), 0.5 + j, 0.0])

    # 4. NE end staircase (b2 steps) at (L0, L0)
    for i in range(b2):
        for j in range(b2 - i):
            q.append([L0 - 0.5 - i * (1.0 + 1e-6), L0 - 0.5 - j, 0.0])

    # 5. Diagonal band column longitudinal and transverse coordinates
    u_teeth_se = np.array([(L0 + a1 - 1.0 - 2.0 * i) * INV_S2 for i in range(a1)])
    u_teeth_nw = np.array([(L0 - a2 + 1.0 + 2.0 * j) * INV_S2 for j in range(a2)])

    v_start = -0.5 * (w - 1.0) - 0.5 * (a1 - a2) * INV_S2
    u_lo = (b1 + 1.0) * INV_S2 + 0.5
    u_hi = S2 * L0 - (b2 + 1.0) * INV_S2 - 0.5
    u_cols = np.linspace(u_lo, u_hi, l) if l > 1 else np.array([0.5 * (u_lo + u_hi)])

    du_se_arr = np.array([uc - u_teeth_se[np.argmin(np.abs(uc - u_teeth_se))] for uc in u_cols])
    du_nw_arr = np.array([uc - u_teeth_nw[np.argmin(np.abs(uc - u_teeth_nw))] for uc in u_cols])

    # Pinch column detection
    pinch_c: list[int] = []
    pinch_dth: list[float] = []
    for c in range(l):
        if max(abs(du_se_arr[c]), abs(du_nw_arr[c])) < 0.50:
            pinch_c.append(c)
            pinch_dth.append(-abs(delta) if du_se_arr[c] > 0 else +abs(delta))

    if len(pinch_c) >= 2:
        dth_cols = np.interp(np.arange(l), pinch_c, pinch_dth)
    elif len(pinch_c) == 1:
        dth_cols = np.full(l, pinch_dth[0])
    else:
        c_best = int(np.argmin([max(abs(du_se_arr[c]), abs(du_nw_arr[c])) for c in range(l)]))
        dth_cols = np.full(l, -abs(delta) if du_se_arr[c_best] > 0 else +abs(delta))

    # Transverse positioning and square placement
    for c in range(l):
        uc = u_cols[c]
        du_se = du_se_arr[c]
        du_nw = du_nw_arr[c]
        dth = dth_cols[c]

        v_shift = np.zeros(w)
        if c == 0 and a1 > a2:
            v_shift[:] = -0.15
        elif c == l - 1 and a1 > a2:
            v_shift[:] = -0.17
        elif abs(dth) < 0.5 * abs(delta):
            if abs(du_se) >= 0.50:
                v_shift[:] = +0.85 * (abs(du_se) - 0.50)
            elif abs(du_nw) >= 0.50:
                v_shift[:] = -0.85 * (abs(du_nw) - 0.50)
        else:
            if abs(du_nw) >= 0.50:
                v_shift[0] = -0.05 * (abs(du_nw) - 0.50)
            elif abs(du_se) >= 0.50:
                v_shift[-1] = +0.05 * (abs(du_se) - 0.50)

        for s in range(w):
            v = v_start + s * 1.0 + v_shift[s]
            u = uc + (s - 0.5 * (w - 1)) * math.sin(dth)
            x = (u + v) * INV_S2
            y = (u - v) * INV_S2
            q.append([x, y, math.pi / 4.0 + dth])

    q_arr = np.array(q, dtype=float)
    return q_arr, L0


def build_and_polish_core(
    D: int,
    w: int,
    polish_iterations: int = 25,
) -> dict[str, Any] | None:
    """Build and certify a multi-D corrugated core."""
    params = derive_core_parameters(D, w)
    if not params.clearance_ok:
        return None

    q_raw, L0 = construct_corrugated_core(params)
    ok_raw, gap_raw = sat_check(q_raw)

    # Polish boundaries with 4-sign LP
    q_polished, L_polished = polish_packing(q_raw, L0, iterations=polish_iterations)
    ok_pol, gap_pol = sat_check(q_polished)

    if not ok_pol:
        # Fallback to normalized raw if polish failed separating axis
        if ok_raw:
            q_polished = normalize_box(q_raw)
            L_polished = extent(q_polished)
            gap_pol = gap_raw
            ok_pol = True
        else:
            return None

    return {
        "D": D,
        "w": w,
        "is_even": params.is_even,
        "n0": params.n0,
        "a": [params.a1, params.a2],
        "b": [params.b1, params.b2],
        "l": params.l,
        "L45": L0,
        "L_polished": L_polished,
        "gain": L_polished - L0,
        "slack": params.slack,
        "w_frac": params.w_frac,
        "sat_ok": bool(ok_pol),
        "sat_gap": float(gap_pol),
        "q": q_polished,
    }


# ---------------------------------------------------------------------------
# 5. Structured Border Wrapping & Corner Residual Placement
# ---------------------------------------------------------------------------

def wrap_core_with_border(
    q_core: np.ndarray,
    east: int = 0,
    north: int = 0,
    west: int = 0,
    south: int = 0,
    margin: float = 1e-7,
) -> tuple[np.ndarray, float]:
    """Wrap core with structured upright integer borders."""
    q_norm = normalize_box(q_core, margin=0.0)
    L_core = extent(q_norm)

    L_x = west + L_core + east + 2.0 * margin
    L_y = south + L_core + north + 2.0 * margin
    L_box = max(L_x, L_y)
    K = int(math.floor(L_box))

    q_placed_core = q_norm.copy()
    q_placed_core[:, 0] += west + margin
    q_placed_core[:, 1] += south + margin

    border_squares: list[list[float]] = []

    # 1. West border: full columns
    for i in range(west):
        x = 0.5 + i * (1.0 + 1e-8)
        for j in range(K):
            border_squares.append([x, 0.5 + j * (1.0 + 1e-8), 0.0])

    # 2. East border: full columns
    for i in range(east):
        x = L_box - 0.5 - i * (1.0 + 1e-8)
        for j in range(K):
            border_squares.append([x, 0.5 + j * (1.0 + 1e-8), 0.0])

    # 3. South border: horizontal rows spanning between west and east
    for j in range(south):
        y = 0.5 + j * (1.0 + 1e-8)
        for i in range(west, K - east):
            border_squares.append([0.5 + i * (1.0 + 1e-8), y, 0.0])

    # 4. North border: horizontal rows spanning between west and east
    for j in range(north):
        y = L_box - 0.5 - j * (1.0 + 1e-8)
        for i in range(west, K - east):
            border_squares.append([0.5 + i * (1.0 + 1e-8), y, 0.0])

    if border_squares:
        q_assembly = np.vstack([q_placed_core, np.array(border_squares, dtype=float)])
    else:
        q_assembly = q_placed_core

    return q_assembly, L_box


def find_residual_sites(
    q_base: np.ndarray,
    L_box: float,
    max_sites: int = 100,
    clearance: float = 1e-8,
) -> list[list[float]]:
    """Find valid candidate square positions in corner pockets and border notches."""
    K = int(math.floor(L_box))
    occupied = q_base[:, :2]

    candidates: list[tuple[float, float, float]] = []
    seen = set()

    # Search near the 4 corners (within distance 6 of edges)
    corner_zones = [
        (range(0, min(6, K)), range(0, min(6, K))),
        (range(max(0, K - 6), K), range(0, min(6, K))),
        (range(0, min(6, K)), range(max(0, K - 6), K)),
        (range(max(0, K - 6), K), range(max(0, K - 6), K)),
    ]

    for x_range, y_range in corner_zones:
        for ix in x_range:
            for iy in y_range:
                x = 0.5 + ix
                y = 0.5 + iy
                if not (0.5 <= x <= L_box - 0.5 and 0.5 <= y <= L_box - 0.5):
                    continue
                key = (round(x, 4), round(y, 4))
                if key in seen:
                    continue
                seen.add(key)

                dists = np.max(np.abs(occupied - np.array([x, y])), axis=1)
                if np.min(dists) < 0.999:
                    continue

                # Prioritize corner slots
                cdist = min(
                    x**2 + y**2,
                    (L_box - x)**2 + y**2,
                    x**2 + (L_box - y)**2,
                    (L_box - x)**2 + (L_box - y)**2,
                )
                candidates.append((cdist, x, y))

    candidates.sort(key=lambda item: item[0])

    valid_squares: list[list[float]] = []
    current_q = q_base.copy()

    for _, x, y in candidates:
        test_sq = np.array([[x, y, 0.0]])
        test_assembly = np.vstack([current_q, test_sq])
        ok, gap = sat_check(test_assembly, clearance=clearance)
        if ok and gap >= 0.0:
            valid_squares.append([x, y, 0.0])
            current_q = test_assembly
            if len(valid_squares) >= max_sites:
                break

    return valid_squares


# ---------------------------------------------------------------------------
# 6. Incumbents & Data Persistence
# ---------------------------------------------------------------------------

def load_incumbents(
    csv_path: Path = INCUMBENT_CSV,
    json_path: Path = INCUMBENT_JSON,
) -> dict[int, float]:
    """Load incumbent side length L for n in [600, 1000]."""
    incumbents: dict[int, float] = {}

    if json_path.exists():
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            for entry in data.get("results", []):
                n_val = int(entry["n"])
                if 600 <= n_val <= 1000:
                    incumbents[n_val] = float(entry["L"])
        except Exception:
            pass

    if len(incumbents) < 401 and csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            for n_val, group in df.groupby("n"):
                n_int = int(n_val)
                if 600 <= n_int <= 1000 and n_int not in incumbents:
                    q = group[["x", "y", "angle"]].values
                    incumbents[n_int] = extent(q)
        except Exception:
            pass

    return incumbents


def format_packing_rows(q: np.ndarray) -> list[dict[str, Any]]:
    """Format coordinates for results storage."""
    q_norm = normalize_box(q, margin=0.0)
    return [
        {
            "square_id": int(i),
            "x": float(x),
            "y": float(y),
            "angle": float(a),
        }
        for i, (x, y, a) in enumerate(q_norm)
    ]


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    """Write JSON data atomically to prevent corruption during progressive updates."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp_path.replace(path)


# ---------------------------------------------------------------------------
# 7. Multi-D Sweep Engine
# ---------------------------------------------------------------------------

def run_multi_d_sweep(
    discriminants: tuple[int, ...] = TARGET_DISCRIMINANTS,
    n_min: int = 600,
    n_max: int = 1000,
    explicit_targets: list[int] | None = None,
    output_path: Path = DEFAULT_OUTPUT_JSON,
    polish_iterations: int = 25,
    contraction_iterations: int = 10,
    max_targets: int | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Execute complete Multi-D Pell core generation, border wrapping, and target sweep."""
    t0 = time.time()

    # Step 1: Solve Pell equations
    if verbose:
        print("=" * 78)
        print("STEP 1: Solving Pell equations for D in", discriminants)
        print("=" * 78)

    pell_units: dict[str, Any] = {}
    for D in discriminants:
        sol = solve_pell(D)
        pell_units[str(D)] = sol
        if verbose:
            fu = sol["fundamental_unit"]
            fpu = sol["fundamental_positive_unit"]
            su = sol["secondary_unit"]
            print(f"D={D:>2}: fund=({fu['x']}, {fu['y']}, norm={fu['norm']:+d}), "
                  f"fund_pos=({fpu['x']}, {fpu['y']}), sec=({su['x']}, {su['y']})")

    # Step 2: Build viable corrugated cores
    if verbose:
        print("\n" + "=" * 78)
        print("STEP 2: Constructing and certifying multi-D corrugated cores")
        print("=" * 78)

    # Candidate widths covering L in [24.0, 32.0]
    candidate_w_by_D: dict[int, list[int]] = {
        3: [12, 13, 16, 20],
        10: [7, 8, 11, 14, 15, 18],
        27: [3, 6, 9, 10],
        34: [4, 7, 8],
        51: [2, 3],
    }

    discovered_cores: list[dict[str, Any]] = []
    viable_cores: list[dict[str, Any]] = []
    core_cache_path = Path("/tmp/multi_d_cores_cache.pkl")

    if core_cache_path.exists():
        try:
            import pickle
            with open(core_cache_path, "rb") as f:
                discovered_cores, viable_cores = pickle.load(f)
            if verbose:
                print(f"Loaded {len(viable_cores)} cached cores from {core_cache_path}")
        except Exception as e:
            if verbose:
                print(f"Cache load failed ({e}), recomputing cores...")
            discovered_cores.clear()
            viable_cores.clear()

    if not viable_cores:
        core_tasks = []
        for D in discriminants:
            widths = candidate_w_by_D.get(D, [])
            for w in widths:
                core_tasks.append((D, w))

        from concurrent.futures import ProcessPoolExecutor
        import os
        workers = min(8, os.cpu_count() or 4)
        if verbose:
            print(f"Polishing {len(core_tasks)} candidate cores with {workers} parallel workers...")

        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_task = {
                executor.submit(build_and_polish_core, D, w, polish_iterations): (D, w)
                for D, w in core_tasks
            }
            for fut in future_to_task:
                D, w = future_to_task[fut]
                try:
                    core = fut.result()
                    if core is not None:
                        discovered_cores.append({
                            "D": core["D"],
                            "w": core["w"],
                            "is_even": core["is_even"],
                            "n0": core["n0"],
                            "a": core["a"],
                            "b": core["b"],
                            "l": core["l"],
                            "L45": float(core["L45"]),
                            "L_polished": float(core["L_polished"]),
                            "gain": float(core["gain"]),
                            "slack": float(core["slack"]),
                            "w_frac": float(core["w_frac"]),
                            "sat_ok": core["sat_ok"],
                            "sat_gap": core["sat_gap"],
                        })
                        viable_cores.append(core)
                        if verbose:
                            print(f"Core D={D:>2}, w={w:>2} -> n0={core['n0']:>4}, "
                                  f"L45={core['L45']:>8.4f}, L_pol={core['L_polished']:>8.4f}, "
                                  f"SAT_gap={core['sat_gap']:.2e}")
                except Exception as e:
                    if verbose:
                        print(f"Core D={D}, w={w} failed: {e}")

        try:
            import pickle
            with open(core_cache_path, "wb") as f:
                pickle.dump((discovered_cores, viable_cores), f)
            if verbose:
                print(f"Cached {len(viable_cores)} certified cores to {core_cache_path}")
        except Exception:
            pass

    # Load incumbents
    incumbents = load_incumbents()
    if verbose:
        print(f"\nLoaded {len(incumbents)} incumbent targets from baseline data.")

    # Determine targets
    if explicit_targets is not None:
        targets = [n for n in explicit_targets if n_min <= n <= n_max]
    else:
        targets = list(range(n_min, n_max + 1))

    if max_targets is not None:
        targets = targets[:max_targets]

    # Initialize results container
    results_by_n: dict[str, Any] = {}
    master_data: dict[str, Any] = {
        "metadata": {
            "title": "Multi-D Pell Corrugated Cores Sweep (D in {3, 10, 27, 34, 51})",
            "discriminants": list(discriminants),
            "target_range": [n_min, n_max],
            "border_thickness_range": [1, 4],
            "incumbent_source": str(INCUMBENT_CSV.relative_to(REPO)),
            "total_targets_planned": len(targets),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        },
        "pell_units": pell_units,
        "discovered_cores": discovered_cores,
        "results_by_n": results_by_n,
        "summary": {
            "total_targets_evaluated": 0,
            "targets_with_candidate": 0,
            "targets_improving": 0,
            "min_delta_L": None,
            "mean_delta_L": None,
        },
    }

    # Step 3: Sweep over targets n in [n_min, n_max]
    if verbose:
        print("\n" + "=" * 78)
        print(f"STEP 3: Wrapping cores with borders t in [1, 4] for {len(targets)} targets")
        print("=" * 78)

    # Border allocations to try: L-borders and symmetric rings
    border_configs = [
        # thickness 0 (unbordered core)
        {"east": 0, "north": 0, "west": 0, "south": 0, "t": 0},
        # thickness 1 single strips
        {"east": 1, "north": 0, "west": 0, "south": 0, "t": 1},
        {"east": 0, "north": 1, "west": 0, "south": 0, "t": 1},
        {"east": 0, "north": 0, "west": 1, "south": 0, "t": 1},
        {"east": 0, "north": 0, "west": 0, "south": 1, "t": 1},
        # thickness 1
        {"east": 1, "north": 1, "west": 0, "south": 0, "t": 1},
        {"east": 0, "north": 1, "west": 1, "south": 0, "t": 1},
        {"east": 1, "north": 0, "west": 0, "south": 1, "t": 1},
        {"east": 0, "north": 0, "west": 1, "south": 1, "t": 1},
        {"east": 1, "north": 1, "west": 1, "south": 1, "t": 1},
        # thickness 2
        {"east": 2, "north": 2, "west": 0, "south": 0, "t": 2},
        {"east": 0, "north": 2, "west": 2, "south": 0, "t": 2},
        {"east": 2, "north": 2, "west": 2, "south": 2, "t": 2},
        # thickness 3
        {"east": 3, "north": 3, "west": 0, "south": 0, "t": 3},
        # thickness 4
        {"east": 4, "north": 4, "west": 0, "south": 0, "t": 4},
    ]

    for target_idx, n in enumerate(targets):
        t_target_start = time.time()
        L_incumbent = incumbents.get(n, math.ceil(math.sqrt(n)))

        best_cand: dict[str, Any] | None = None
        best_L = float("inf")

        # Evaluate candidate cores
        for core in viable_cores:
            n0 = core["n0"]
            if n0 > n:
                continue

            q_core = core["q"]

            for bcfg in border_configs:
                t = bcfg["t"]
                # Wrap core with border
                q_border, L_box = wrap_core_with_border(
                    q_core,
                    east=bcfg["east"],
                    north=bcfg["north"],
                    west=bcfg["west"],
                    south=bcfg["south"],
                )
                n_assembly = len(q_border)
                if n_assembly > n:
                    continue

                residuals_needed = n - n_assembly
                if residuals_needed > 35:
                    # Too many residuals needed for clean packing
                    continue

                # Place residuals if any needed
                if residuals_needed > 0:
                    res_squares = find_residual_sites(q_border, L_box, max_sites=residuals_needed)
                    if len(res_squares) < residuals_needed:
                        continue
                    q_candidate = np.vstack([q_border, np.array(res_squares, dtype=float)])
                else:
                    q_candidate = q_border

                assert len(q_candidate) == n
                ok_cand, gap_cand = sat_check(q_candidate)
                if not ok_cand:
                    continue

                # Epigraph LP contraction step
                L_cand_init = extent(q_candidate)
                if L_cand_init >= best_L:
                    continue

                q_contracted, L_contracted = polish_packing(
                    q_candidate,
                    L_cand_init,
                    iterations=contraction_iterations,
                    angle_trust=0.0,
                )
                ok_final, gap_final = sat_check(q_contracted)
                if not ok_final:
                    q_contracted = q_candidate
                    L_contracted = L_cand_init
                    gap_final = gap_cand

                if L_contracted < best_L:
                    best_L = L_contracted
                    best_cand = {
                        "n": n,
                        "L_incumbent": float(L_incumbent),
                        "L_candidate": float(L_contracted),
                        "delta_L": float(L_contracted - L_incumbent),
                        "core_D": core["D"],
                        "core_w": core["w"],
                        "border_thickness": t,
                        "border_allocation": {
                            "east": bcfg["east"],
                            "north": bcfg["north"],
                            "west": bcfg["west"],
                            "south": bcfg["south"],
                        },
                        "residuals": residuals_needed,
                        "sat_ok": True,
                        "sat_gap": float(gap_final),
                        "candidate_packing": format_packing_rows(q_contracted),
                    }

        target_elapsed = time.time() - t_target_start

        # Record target results with hard validation against integer grid fallback
        if best_cand is not None:
            cand_L = float(best_cand["L_candidate"])
            grid_L = float(math.ceil(math.sqrt(n)))
            is_grid_L = abs(cand_L - grid_L) <= 1e-5
            angles = [
                row["angle"] if isinstance(row, dict) else row[2]
                for row in best_cand["candidate_packing"]
            ]
            tilted_count = sum(
                1
                for a in angles
                if min(
                    abs(float(a)) % (math.pi / 2.0),
                    math.pi / 2.0 - (abs(float(a)) % (math.pi / 2.0)),
                )
                > 1e-5
            )

            if is_grid_L or tilted_count == 0:
                results_by_n[str(n)] = {
                    "target n": n,
                    "target_n": n,
                    "n": n,
                    "best L": None,
                    "best_L": None,
                    "incumbent L": float(L_incumbent),
                    "incumbent_L": float(L_incumbent),
                    "delta L": None,
                    "delta_L": None,
                    "discriminant D": best_cand["core_D"],
                    "discriminant_D": best_cand["core_D"],
                    "status": "FAILED",
                    "reason": "integer_grid_fallback",
                    "runtime": target_elapsed,
                    "tilted_squares": tilted_count,
                    "grid_L": grid_L,
                }
                status_str = f"FAILED (integer_grid_fallback): L={cand_L:.6f}, tilted={tilted_count}"
            else:
                # Rigorous SAT separating axis verification
                q_verify = np.array(
                    [
                        [row["x"], row["y"], row["angle"]]
                        if isinstance(row, dict)
                        else row
                        for row in best_cand["candidate_packing"]
                    ],
                    dtype=float,
                )
                sat_ok, sat_gap = sat_check(q_verify)
                if not sat_ok or sat_gap < 0:
                    results_by_n[str(n)] = {
                        "target n": n,
                        "target_n": n,
                        "n": n,
                        "best L": None,
                        "best_L": None,
                        "incumbent L": float(L_incumbent),
                        "incumbent_L": float(L_incumbent),
                        "delta L": None,
                        "delta_L": None,
                        "discriminant D": best_cand["core_D"],
                        "discriminant_D": best_cand["core_D"],
                        "status": "FAILED",
                        "reason": "sat_check_failed",
                        "runtime": target_elapsed,
                        "sat_gap": float(sat_gap),
                    }
                    status_str = f"FAILED (sat_check_failed): gap={sat_gap:.2e}"
                else:
                    results_by_n[str(n)] = {
                        "target n": n,
                        "target_n": n,
                        "n": n,
                        "best L": cand_L,
                        "best_L": cand_L,
                        "incumbent L": float(L_incumbent),
                        "incumbent_L": float(L_incumbent),
                        "delta L": float(best_cand["delta_L"]),
                        "delta_L": float(best_cand["delta_L"]),
                        "discriminant D": best_cand["core_D"],
                        "discriminant_D": best_cand["core_D"],
                        "status": "REAL_CONSTRUCTION",
                        "reason": "ok",
                        "runtime": target_elapsed,
                        "core_D": best_cand["core_D"],
                        "core_w": best_cand["core_w"],
                        "border_thickness": best_cand["border_thickness"],
                        "border_allocation": best_cand["border_allocation"],
                        "residuals": best_cand["residuals"],
                        "sat_ok": True,
                        "sat_gap": float(sat_gap),
                        "tilted_squares": tilted_count,
                        "candidate_packing": best_cand["candidate_packing"],
                    }
                    status_str = f"REAL_CONSTRUCTION: L_cand={cand_L:.6f} vs L_inc={L_incumbent:.6f} (dL={best_cand['delta_L']:+.6f}, D={best_cand['core_D']})"
        else:
            results_by_n[str(n)] = {
                "target n": n,
                "target_n": n,
                "n": n,
                "best L": None,
                "best_L": None,
                "incumbent L": float(L_incumbent),
                "incumbent_L": float(L_incumbent),
                "delta L": None,
                "delta_L": None,
                "discriminant D": None,
                "discriminant_D": None,
                "status": "FAILED",
                "reason": "no_feasible_multi_d_candidate",
                "runtime": target_elapsed,
            }
            status_str = f"FAILED (no candidate): L_inc={L_incumbent:.6f}"

        # Update summary statistics and convenience mappings
        master_data["results_by_n"] = results_by_n
        master_data["targets"] = results_by_n
        master_data[str(n)] = results_by_n[str(n)]
        master_data["smoke_test_metrics"] = [
            {
                "target n": v["target_n"],
                "target_n": v["target_n"],
                "best L": v.get("best_L"),
                "best_L": v.get("best_L"),
                "incumbent L": v["incumbent_L"],
                "incumbent_L": v["incumbent_L"],
                "delta L": v.get("delta_L"),
                "delta_L": v.get("delta_L"),
                "discriminant D": v.get("discriminant_D"),
                "discriminant_D": v.get("discriminant_D"),
                "status": v["status"],
                "runtime": v["runtime"],
            }
            for v in results_by_n.values()
        ]

        valid_deltas = [
            v["delta_L"]
            for v in results_by_n.values()
            if v.get("delta_L") is not None and v.get("status") == "REAL_CONSTRUCTION"
        ]
        master_data["summary"]["total_targets_evaluated"] = target_idx + 1
        master_data["summary"]["targets_with_candidate"] = len(valid_deltas)
        master_data["summary"]["targets_improving"] = sum(1 for d in valid_deltas if d < 0.0)
        master_data["summary"]["min_delta_L"] = min(valid_deltas) if valid_deltas else None
        master_data["summary"]["mean_delta_L"] = float(np.mean(valid_deltas)) if valid_deltas else None

        # Write progress atomically after each target
        atomic_write_json(output_path, master_data)

        if verbose:
            print(f"[{target_idx + 1:>3}/{len(targets)}] n={n:>4}: {status_str} ({target_elapsed:.2f}s)")

    total_time = time.time() - t0
    master_data["metadata"]["elapsed_seconds"] = total_time
    atomic_write_json(output_path, master_data)

    if verbose:
        print("\n" + "=" * 78)
        print(f"SWEEP COMPLETE in {total_time:.1f}s")
        print(f"Evaluated: {len(targets)} targets")
        print(f"Candidates found: {master_data['summary']['targets_with_candidate']}")
        print(f"Output saved to: {output_path}")
        print("=" * 78)

    return master_data


# ---------------------------------------------------------------------------
# 8. Command Line Interface
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-D Pell Corrugated Cores Generator & Sweep (D in {3, 10, 27, 34, 51})"
    )
    parser.add_argument(
        "--D",
        type=int,
        nargs="+",
        default=list(TARGET_DISCRIMINANTS),
        help="Discriminants D to evaluate (default: 3 10 27 34 51)",
    )
    parser.add_argument(
        "--n-min",
        type=int,
        default=600,
        help="Minimum target n (default: 600)",
    )
    parser.add_argument(
        "--n-max",
        type=int,
        default=1000,
        help="Maximum target n (default: 1000)",
    )
    parser.add_argument(
        "--targets",
        type=int,
        nargs="+",
        default=None,
        help="Explicit target n values to sweep",
    )
    parser.add_argument(
        "--max-targets",
        type=int,
        default=None,
        help="Limit on number of targets to sweep",
    )
    parser.add_argument(
        "--polish-iterations",
        type=int,
        default=25,
        help="LP polish iterations for core generation (default: 25)",
    )
    parser.add_argument(
        "--contraction-iterations",
        type=int,
        default=10,
        help="LP contraction iterations for border wrapping (default: 10)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_JSON),
        help=f"Output JSON path (default: {DEFAULT_OUTPUT_JSON})",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose progress logging",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_multi_d_sweep(
        discriminants=tuple(args.D),
        n_min=args.n_min,
        n_max=args.n_max,
        explicit_targets=args.targets,
        output_path=Path(args.output),
        polish_iterations=args.polish_iterations,
        contraction_iterations=args.contraction_iterations,
        max_targets=args.max_targets,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
