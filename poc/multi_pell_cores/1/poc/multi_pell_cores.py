"""Multi-Pell Corrugated Cores POC 1: Symmetric odd-D corrugated cores (D=3, 27, 51).

100% synthetic CPU construction extending out-of-phase pinch-column corrugation
from Pell-17 (D=17) to the full family of symmetric odd-D cores:
  - D = 3  (Diophantine 3^2 - 2*2^2 = +1, slack = 0.12132, l = w + 2)
  - D = 27 (Hybrid 10 + 17, slack = 0.09188, l = w + 19)
  - D = 51 (Triple Pell 3 x 17, slack = 0.06245, l = w + 36)

All candidate configurations enforce:
  1. Out-of-phase staircase parity: a1 + a2 = 2b + D (odd, since D is odd).
  2. Symmetric end staircases: b1 = b2 = b = floor(w / sqrt(2)).
  3. Flank staircases: a1 = b + (D + 1) // 2, a2 = b + (D - 1) // 2.
  4. 4-sign-piece epigraph LP polisher to handle support kinks at r = 0.
"""
from __future__ import annotations

import argparse
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
SIGNS = tuple(product((-1.0, 1.0), repeat=2))

# Supported symmetric odd-D families
SUPPORTED_ODD_D = (3, 17, 27, 51)

# Tabulated sweet-spot widths satisfying {w / sqrt(2)} in (0.05, 0.42) with n0 <= 1000
SWEET_SPOT_WIDTHS: dict[int, tuple[int, ...]] = {
    3: (2, 3, 6, 9, 10, 13, 16, 20),
    17: (2, 3, 6, 9, 10, 13, 16),
    27: (2, 3, 6, 9, 10),
    51: (2, 3),
}


@dataclass
class CoreParameters:
    D: int
    w: int
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


@dataclass
class PolishResult:
    q: np.ndarray
    L: float
    accepted: int = 0


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


def get_symmetric_odd_d_params(D: int, w: int) -> CoreParameters:
    """Derive algebraic parameters for a symmetric odd-D staircase-band core."""
    if D % 2 == 0:
        raise ValueError(f"D={D} must be odd for symmetric out-of-phase staircases")
    if D <= 0:
        raise ValueError(f"D={D} must be positive")
    if w <= 0:
        raise ValueError(f"Width w={w} must be positive")

    b = int(math.floor(w * INV_S2))
    a1 = b + (D + 1) // 2
    a2 = b + (D - 1) // 2
    l_strip = w + int(math.floor(D * INV_S2))
    L45 = 0.5 * (a1 + a2) + 1.0 + w * INV_S2
    k0 = int(math.floor(L45))

    # Square counts: triangular staircases + diagonal band
    t_a1 = a1 * (a1 + 1) // 2
    t_a2 = a2 * (a2 + 1) // 2
    t_b = 2 * (b * (b + 1) // 2)
    t_band = w * l_strip
    n0 = t_a1 + t_a2 + t_b + t_band

    slack = D * INV_S2 - math.floor(D * INV_S2)
    w_frac = (w * INV_S2) - math.floor(w * INV_S2)

    return CoreParameters(
        D=D,
        w=w,
        b1=b,
        b2=b,
        a1=a1,
        a2=a2,
        l=l_strip,
        L45=L45,
        k0=k0,
        n0=n0,
        slack=slack,
        w_frac=w_frac,
    )


def sat_check(q: np.ndarray, clearance: float = 0.0) -> tuple[bool, float]:
    """Rigorous separating axis theorem check across all square pairs in long double precision."""
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
    """Compute gap and gradient for one of the 4 sign pieces of the support function."""
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


def _lp(
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


def polish_band(
    q: np.ndarray,
    L_init: float,
    iterations: int = 35,
    position_trust: float = 0.20,
    angle_trust: float = 0.015,
    clearance: float = 5e-9,
) -> PolishResult | None:
    """Iteratively polish a corrugated band using 4-sign-piece epigraph trust-region LP."""
    axes: dict[tuple[int, int], list[tuple[float, float, int]]] = {}
    repaired, _ = _lp(q, L_init + 0.15, position_trust, 0.0, clearance, axes)
    if repaired is None:
        return None
    best = PolishResult(repaired.copy(), extent(repaired))

    for it in range(iterations):
        old = best.L
        fixed, _ = _lp(best.q, old + 1e-4, position_trust, 0.0, clearance, axes)
        if fixed is not None and sat_check(fixed)[0] and extent(fixed) < best.L:
            best.q = fixed
            best.L = extent(fixed)
            best.accepted += 1
        proposal, _ = _lp(best.q, best.L + 1e-4, position_trust, angle_trust, clearance, axes)
        if proposal is not None:
            origin = best.q.copy()
            for bt in range(12):
                alpha = 2.0 ** (-bt)
                seed = origin + alpha * (proposal - origin)
                rep, _ = _lp(seed, best.L + 1e-3, position_trust, 0.0, clearance, {})
                if rep is not None and sat_check(rep)[0] and extent(rep) < best.L:
                    best.q = rep
                    best.L = extent(rep)
                    best.accepted += 1
                    break
        if old - best.L < 1e-11 and it >= 5:
            break
    return best


def build_corrugated_band(
    w: int,
    l: int,
    a1: int,
    a2: int,
    b1: int,
    b2: int,
    delta: float = 0.002935,
) -> tuple[np.ndarray, float]:
    """Construct an out-of-phase pinch-column corrugated band layout.

    Applies the 4 pinch-column rules:
      1. Pinch-column angle interpolation: pinch columns (|du_se|, |du_nw| < 0.50) set sign;
         interpolation prevents abrupt inter-column shearing.
      2. Tilted-block normal alignment & shallow notch nudge.
      3. Corner staircase tie-breaking (1e-6 grid stretch).
      4. Corner tooth slide into container clearance gaps.
    """
    L0 = 0.5 * (a1 + a2) + 1.0 + w * INV_S2
    q: list[list[float]] = []
    gap_se_ne = max(0.0, (L0 - b2) - a1 - 0.05)
    gap_se_sw = max(0.0, (L0 - b1) - a1 - 0.05)

    for i in range(a1):
        for j in range(a1 - i):
            x, y = L0 - 0.5 - i, 0.5 + j * (1.0 + 1e-6)
            if i == a1 - 1 and j == 0 and gap_se_sw > 0:
                x -= min(0.19, 0.45 * gap_se_sw)
            if i == 0 and j == a1 - 1 and gap_se_ne > 0:
                x -= min(0.20, 0.45 * gap_se_ne)
            q.append([x, y, 0.0])

    for i in range(a2):
        for j in range(a2 - i):
            x, y = 0.5 + i * (1.0 + 1e-6), L0 - 0.5 - j
            q.append([x, y, 0.0])

    for i in range(b1):
        for j in range(b1 - i):
            q.append([0.5 + i * (1.0 + 1e-6), 0.5 + j, 0.0])

    for i in range(b2):
        for j in range(b2 - i):
            q.append([L0 - 0.5 - i * (1.0 + 1e-6), L0 - 0.5 - j, 0.0])

    u_teeth_se = np.array([(L0 + a1 - 1.0 - 2.0 * i) * INV_S2 for i in range(a1)])
    u_teeth_nw = np.array([(L0 - a2 + 1.0 + 2.0 * j) * INV_S2 for j in range(a2)])

    v_start = -0.5 * (w - 1.0) - 0.5 * (a1 - a2) * INV_S2
    u_lo = (b1 + 1.0) * INV_S2 + 0.5
    u_hi = S2 * L0 - (b2 + 1.0) * INV_S2 - 0.5
    u_cols = np.linspace(u_lo, u_hi, l) if l > 1 else np.array([0.5 * (u_lo + u_hi)])

    du_se_arr = np.array([uc - u_teeth_se[np.argmin(np.abs(uc - u_teeth_se))] for uc in u_cols])
    du_nw_arr = np.array([uc - u_teeth_nw[np.argmin(np.abs(uc - u_teeth_nw))] for uc in u_cols])

    pinch_c: list[int] = []
    pinch_dth: list[float] = []
    for c in range(l):
        if max(abs(du_se_arr[c]), abs(du_nw_arr[c])) < 0.50:
            pinch_c.append(c)
            pinch_dth.append(-abs(delta) if du_se_arr[c] > 0 else +abs(delta))

    if not pinch_c:
        # Fallback to column closest to pinch condition
        c_best = int(np.argmin([max(abs(du_se_arr[c]), abs(du_nw_arr[c])) for c in range(l)]))
        pinch_c.append(c_best)
        pinch_dth.append(-abs(delta) if du_se_arr[c_best] > 0 else +abs(delta))

    dth_cols = np.interp(np.arange(l), pinch_c, pinch_dth)

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

    return np.array(q, dtype=float), L0


def build_symmetric_odd_d_core(
    D: int,
    w: int,
    delta: float | None = None,
    iterations: int = 35,
    clearance: float = 5e-9,
) -> tuple[PolishResult, float, dict]:
    """Build and polish a symmetric odd-D out-of-phase corrugated core."""
    params = get_symmetric_odd_d_params(D, w)

    if delta is None:
        # Scale nominal tilt proportionally to available longitudinal slack
        # For D=17 (slack=0.02082), delta=0.002935.
        # Guard upper bound to avoid excess shear.
        delta = min(0.0035, max(0.0020, params.slack * 0.14))

    q_init, L0 = build_corrugated_band(
        w=params.w,
        l=params.l,
        a1=params.a1,
        a2=params.a2,
        b1=params.b1,
        b2=params.b2,
        delta=delta,
    )
    res = polish_band(q_init, L0, iterations=iterations, clearance=clearance)
    if res is None:
        raise RuntimeError(f"Odd-D core polish failed for D={D}, w={w}")

    meta = dict(
        family=f"symmetric_odd_d_corrugated_core",
        D=D,
        w=w,
        l=params.l,
        a=[params.a1, params.a2],
        b=[params.b1, params.b2],
        slack_45=params.slack,
        w_frac=params.w_frac,
        k0=params.k0,
        n0=len(res.q),
        L45=L0,
        L_corrugated=res.L,
        corrugation_gain=res.L - L0,
        delta_tilt=delta,
    )
    return res, L0, meta


# Specialized builders for POC 1 families
def build_d3_core(w: int, iterations: int = 35, clearance: float = 5e-9) -> tuple[PolishResult, float, dict]:
    """Build and polish a D=3 (Small-Odd Diophantine-3) corrugated core."""
    return build_symmetric_odd_d_core(3, w, delta=0.0035, iterations=iterations, clearance=clearance)


def build_d27_core(w: int, iterations: int = 35, clearance: float = 5e-9) -> tuple[PolishResult, float, dict]:
    """Build and polish a D=27 (Symmetric Hybrid) corrugated core."""
    return build_symmetric_odd_d_core(27, w, delta=0.0030, iterations=iterations, clearance=clearance)


def build_d51_core(w: int, iterations: int = 35, clearance: float = 5e-9) -> tuple[PolishResult, float, dict]:
    """Build and polish a D=51 (Triple-Pell) corrugated core."""
    return build_symmetric_odd_d_core(51, w, delta=0.0028, iterations=iterations, clearance=clearance)


def add_l_border(q_core: np.ndarray, t: int, k_total: int, margin: float = 1e-7) -> np.ndarray:
    """Wrap a normalized core packing q_core with a t-layer upright L-border on N and E edges."""
    q_core = normalize_box(q_core, margin=0.0)
    L_core = extent(q_core)
    L_total = L_core + t + 2.0 * margin
    k_core = k_total - t
    border: list[list[float]] = []
    for i in range(t):
        for j in range(k_total):
            border.append([L_total - 0.5 - i * (1.0 + 1e-8), 0.5 + j * (1.0 + 1e-8), 0.0])
    for i in range(k_core):
        for j in range(t):
            border.append([0.5 + i * (1.0 + 1e-8), L_total - 0.5 - j * (1.0 + 1e-8), 0.0])
    return np.vstack([q_core, np.array(border, dtype=float)])


def rows(q: np.ndarray, dilate: float = 1.0) -> list[dict[str, str]]:
    """Format coordinates for official metric submission."""
    q = q.copy()
    q[:, :2] -= q[:, :2].mean(0)
    if dilate != 1.0:
        q[:, :2] *= dilate
    return [
        dict(
            square_id=str(i),
            x=format(x, ".17g"),
            y=format(y, ".17g"),
            angle=format(a, ".17g"),
        )
        for i, (x, y, a) in enumerate(q)
    ]


def csv_text(puzzles: dict[int, list[dict[str, str]]]) -> str:
    """Serialize dictionary of puzzles into standard submission CSV format."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["n", "square_id", "x", "y", "angle"])
    for n in sorted(puzzles):
        for r in puzzles[n]:
            writer.writerow([n, r["square_id"], r["x"], r["y"], r["angle"]])
    return buf.getvalue()


def certify_transport(csv_string: str) -> dict[str, dict]:
    """Run official interval certifier on literal CSV and pandas 3.1.0 round-trip."""
    res_lit = certify(csv_string)
    df = pd.read_csv(io.StringIO(csv_string))
    pan_csv = df.to_csv(index=False)
    res_pan = certify(pan_csv)
    return {
        str(k): {
            "literal": dict(ok=v.ok, L=str(v.L), error=v.error),
            "actual_pandas": dict(ok=res_pan[k].ok, L=str(res_pan[k].L), error=res_pan[k].error),
        }
        for k, v in res_lit.items()
    }


def main():
    parser = argparse.ArgumentParser(description="Multi-Pell Symmetric Odd-D Corrugated Cores (D=3, 27, 51)")
    parser.add_argument("--D", type=int, choices=[3, 17, 27, 51], default=51, help="Odd-D value")
    parser.add_argument("--w", type=int, default=3, help="Band width w")
    parser.add_argument("--iterations", type=int, default=35, help="LP polish iterations")
    parser.add_argument("--all-sweet-spots", action="store_true", help="Print table of all sweet spots")
    args = parser.parse_args()

    if args.all_sweet_spots:
        print(f"{'D':>3} {'w':>3} {'b':>3} {'(a1, a2)':>10} {'l':>4} {'k0':>4} {'n0':>5} {'L45':>14} {'slack':>8} {'{w/sqrt2}':>9}")
        print("-" * 75)
        for d in (3, 27, 51):
            for w in SWEET_SPOT_WIDTHS[d]:
                p = get_symmetric_odd_d_params(d, w)
                print(f"{d:>3} {w:>3} {p.b1:>3} {f'({p.a1}, {p.a2})':>10} {p.l:>4} {p.k0:>4} {p.n0:>5} {p.L45:>14.9f} {p.slack:>8.5f} {p.w_frac:>9.4f}")
        return

    print(f"Building core for D={args.D}, w={args.w}...")
    res, L0, meta = build_symmetric_odd_d_core(args.D, args.w, iterations=args.iterations)
    print(f"Done: n0={meta['n0']}, L45={L0:.9f}, L_corr={res.L:.9f}, Delta={meta['corrugation_gain']:.9f}")


if __name__ == "__main__":
    main()
