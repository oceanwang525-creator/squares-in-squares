"""Pell-17 out-of-phase pinch-column corrugated staircase-band + L-border generator.

100% synthetic CPU construction for large-n square packings.
All emitted packings are certified with the unchanged official interval metric
on both literal CSV and actual pandas 3.1.0 read/write round-trip transport.
No incumbent or baseline coordinates are read or replayed in any generated packing.
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
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parents[1] / "data"
DOCS = Path(__file__).resolve().parents[1] / "docs"
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "poc/baseline/poc/src"))
from squares import _official_metric as metric
from squares.core import certify

S2 = math.sqrt(2.0)
INV_S2 = 1.0 / S2

# Widths w in [1, 20] where {w/sqrt(2)} in (0.05, 0.42), so b = floor(w/sqrt(2))
# backs the band ends without hitting the outer box walls x=0, y=0, x=L0, y=L0.
PELL17_WIDTHS = (2, 3, 6, 9, 10, 13, 16)

# Targeted public-table reference side lengths (from seed_cli.py & baseline-targets.csv)
PUBLIC_TARGET_INCUMBENTS = {
    908: "30.63702955961203500000000000000000000000000000000000",
    909: "30.65685424949258200000000000000000000000000000000000",
    910: "30.69096012064609301099087714033620383408805471054880",
    911: "30.70710678118672700000000000000000000000000000000000",
    971: "31.67877987534328000000000000000000000000000000000000",
    972: "31.69238815542838555961566082107349804168366869287818",
    973: "31.70710678118685000000000000000000000000000000000000",
}

# CUDA-loop portfolio baseline side lengths (rtx4070s-20261101/report.json)
PORTFOLIO_BASELINE = {
    543: "24.00000000000000000000000000000000000000000000000000",
    544: "24.00000000000000000000000000000000000000000000000000",
    545: "24.00000000000000000000000000000000000000000000000000",
    591: "25.00000000000000000000000000000000000000000000000000",
    592: "25.00000000000000000000000000000000000000000000000000",
    593: "25.00000000000000000000000000000000000000000000000000",
    908: "30.70710678118654800000000000000000000000000000000000",
    909: "30.70710678118654800000000000000000000000000000000000",
    910: "30.70710678118654800000000000000000000000000000000000",
    911: "30.70710678118654800000000000000000000000000000000000",
    971: "31.70710678118654800000000000000000000000000000000000",
    972: "31.70710678118654800000000000000000000000000000000000",
    973: "31.70710678118654800000000000000000000000000000000000",
}

ALL_BRIEF_TARGETS = (543, 544, 545, 591, 592, 593, 908, 909, 910, 911, 971, 972, 973)


@dataclass
class PolishResult:
    q: np.ndarray
    L: float
    accepted: int = 0


def extent(q: np.ndarray) -> float:
    c = np.abs(np.cos(q[:, 2]))
    s = np.abs(np.sin(q[:, 2]))
    h = 0.5 * (c + s)
    lo = np.min(q[:, :2] - h[:, None], axis=0)
    hi = np.max(q[:, :2] + h[:, None], axis=0)
    return float(np.max(hi - lo))


def normalize_box(q: np.ndarray, margin: float = 0.0) -> np.ndarray:
    q = q.copy()
    h = 0.5 * (np.abs(np.cos(q[:, 2])) + np.abs(np.sin(q[:, 2])))
    lo = np.min(q[:, :2] - h[:, None], axis=0)
    q[:, :2] -= lo - margin
    return q


from itertools import product

SIGNS = tuple(product((-1.0, 1.0), repeat=2))


def sat_check(q: np.ndarray, clearance: float = 0.0) -> tuple[bool, float]:
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
    """Construct a 100% synthetic odd-(a1+a2) out-of-phase corrugated staircase-band packing.

    Implements the 4 Pinch-Column Corrugation rules:
      1. Pinch-Column Angle Interpolation: only pinch columns max(|du_se|, |du_nw|) < 0.50
         dictate +-delta; non-pinch columns linearly interpolate delta_c between adjacent
         pinch columns, acting as uniform tilted blocks for shallow notches and zero-angle
         hinge buffers + transverse notch steppers for deep notches.
      2. Tilted-Block Normal Alignment & End-Only Shallow Notch Nudge.
      3. Corner Staircase Tie-Breaking (1e-6 grid stretch so SE slides in x and NW in y).
      4. Outermost Flank Tooth Slide into corner gaps.
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
    u_cols = np.linspace(u_lo, u_hi, l)

    du_se_arr = np.array([uc - u_teeth_se[np.argmin(np.abs(uc - u_teeth_se))] for uc in u_cols])
    du_nw_arr = np.array([uc - u_teeth_nw[np.argmin(np.abs(uc - u_teeth_nw))] for uc in u_cols])

    pinch_c: list[int] = []
    pinch_dth: list[float] = []
    for c in range(l):
        if max(abs(du_se_arr[c]), abs(du_nw_arr[c])) < 0.50:
            pinch_c.append(c)
            pinch_dth.append(-abs(delta) if du_se_arr[c] > 0 else +abs(delta))
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


def build_pell17_core(w: int, iterations: int = 35, clearance: float = 5e-9) -> tuple[PolishResult, float, dict]:
    """Build and polish a Pell-17 (D=17) out-of-phase corrugated core for width w."""
    b = int(math.floor(w * INV_S2))
    a1, a2 = b + 9, b + 8
    l = w + 12
    q_init, L0 = build_corrugated_band(w, l, a1, a2, b, b, delta=0.002935)
    res = polish_band(q_init, L0, iterations=iterations, clearance=clearance)
    if res is None:
        raise RuntimeError(f"Pell-17 core polish failed for w={w}")
    meta = dict(
        family="pell17_corrugated_band",
        w=w,
        l=l,
        a=[a1, a2],
        b=[b, b],
        D=17,
        slack_45=17.0 * INV_S2 - 12.0,
        k0=int(math.floor(L0)),
        n0=len(res.q),
        L45=L0,
        L_corrugated=res.L,
        corrugation_gain=res.L - L0,
    )
    return res, L0, meta


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


def build_staircase_band_45(
    w: int,
    l: int,
    a1: int,
    a2: int,
    b1: int,
    b2: int,
    drop_band: int = 0,
) -> tuple[np.ndarray, float]:
    """Construct an exact wall-constrained 45-degree staircase-band packing."""
    L0 = 0.5 * (a1 + a2) + 1.0 + w * INV_S2
    q: list[list[float]] = []
    for i in range(a1):
        for j in range(a1 - i):
            if i == a1 - 1 and j == 0 and a1 + b1 > L0 + 1e-9:
                continue
            if i == 0 and j == a1 - 1 and a1 + b2 > L0 + 1e-9:
                continue
            q.append([L0 - 0.5 - i, 0.5 + j, 0.0])
    for i in range(a2):
        for j in range(a2 - i):
            if i == 0 and j == a2 - 1 and a2 + b1 > L0 + 1e-9:
                continue
            if i == a2 - 1 and j == 0 and a2 + b2 > L0 + 1e-9:
                continue
            q.append([0.5 + i, L0 - 0.5 - j, 0.0])
    for i in range(b1):
        for j in range(b1 - i):
            q.append([0.5 + i, 0.5 + j, 0.0])
    for i in range(b2):
        for j in range(b2 - i):
            q.append([L0 - 0.5 - i, L0 - 0.5 - j, 0.0])

    v_start = -0.5 * (w - 1.0) - 0.5 * (a1 - a2) * INV_S2
    u_wall = 0.5 * (w - 1.0) + 0.5 * abs(a1 - a2) * INV_S2 + INV_S2
    u_lo = max((b1 + 1.0) * INV_S2 + 0.5, u_wall)
    u_hi = S2 * L0 - max((b2 + 1.0) * INV_S2 + 0.5, u_wall)
    u_cols = np.linspace(u_lo, u_hi, l) if l > 1 else np.array([0.5 * (u_lo + u_hi)])

    band: list[list[float]] = []
    for c in range(l):
        uc = u_cols[c]
        for s in range(w):
            v = v_start + s * 1.0
            x = (uc + v) * INV_S2
            y = (uc - v) * INV_S2
            band.append([x, y, math.pi / 4.0])
    if drop_band > 0:
        band = band[:-drop_band]
    q.extend(band)
    return np.array(q, dtype=float), L0


def add_l_border_exact(q_core: np.ndarray, L_core: float, t: int, k_total: int) -> np.ndarray:
    """Wrap an exact 45-degree staircase-band core with a t-layer L-border before global dilation."""
    L_total = L_core + t
    k_core = k_total - t
    border: list[list[float]] = []
    for i in range(t):
        for j in range(k_total):
            border.append([L_total - 0.5 - i, 0.5 + j, 0.0])
    for i in range(k_core):
        for j in range(t):
            border.append([0.5 + i, L_total - 0.5 - j, 0.0])
    return np.vstack([q_core, np.array(border, dtype=float)])


def rows(q: np.ndarray, dilate: float = 1.0) -> list[dict[str, str]]:
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
    f = io.StringIO()
    w = csv.DictWriter(f, fieldnames=("id", "n", "square_id", "x", "y", "angle"))
    w.writeheader()
    idx = 0
    for n, rs in sorted(puzzles.items()):
        for r in rs:
            w.writerow(dict(id=idx, n=n, **{k: r[k] for k in ("square_id", "x", "y", "angle")}))
            idx += 1
    return f.getvalue()


def certify_transport(text: str, path: Path | None = None) -> dict[str, dict]:
    literal = metric._parse_submission_csv(text)
    transported_text = pd.read_csv(io.StringIO(text)).to_csv(index=False)
    transported = metric._parse_submission_csv(transported_text)
    if path is not None:
        path.write_text(transported_text)
    result: dict[str, dict] = {}
    for n in literal:
        entry: dict[str, dict] = {}
        for mode, puzzles in (("literal", literal), ("actual_pandas", transported)):
            try:
                entry[mode] = dict(ok=True, L=str(certify(n, puzzles[n])))
            except ValueError as e:
                entry[mode] = dict(ok=False, error=str(e))
        result[str(n)] = entry
    return result


def generate_all_targets(targets: list[int] | tuple[int, ...] = ALL_BRIEF_TARGETS):
    wanted = set(targets)
    candidates: dict[int, list[dict[str, str]]] = {}
    records: dict[int, dict] = {}
    pell17_family_summary: list[dict] = []

    # 1. Build and verify the Pell-17 corrugated band cores needed for our targets (w=9 and w=13)
    # plus a fast sweep of the Pell-17 family widths.
    core_cache: dict[int, tuple[PolishResult, float, dict]] = {}
    for w in (9, 13):
        res_core, L0, meta = build_pell17_core(w, iterations=35, clearance=5e-9)
        core_cache[w] = (res_core, L0, meta)
        pell17_family_summary.append(meta)

    # 2. Wrap Pell-17 (w=13, b=9, k0=27, n0=739) with t=3 (n=910) and t=4 (n=971) L-borders
    res_739, L0_739, _ = core_cache[13]
    for t, k_tot, n_tot in ((3, 30, 910), (4, 31, 971)):
        if n_tot in wanted:
            q_wrapped = add_l_border(res_739.q, t=t, k_total=k_tot, margin=1e-7)
            res_w = polish_band(
                q_wrapped,
                extent(q_wrapped),
                iterations=20,
                position_trust=0.05,
                angle_trust=0.005,
                clearance=5e-9,
            )
            assert res_w is not None and len(res_w.q) == n_tot
            candidates[n_tot] = rows(res_w.q, dilate=1.0)
            records[n_tot] = dict(
                kind="pell17_corrugated_band_l_border",
                k_total=k_tot,
                k_core=27,
                t_border=t,
                w=13,
                l=25,
                a=[18, 17],
                b=[9, 9],
                D=17,
                L45=L0_739 + t,
                L_polished=res_w.L,
                corrugation_gain_vs_45=res_w.L - (L0_739 + t),
                count=len(res_w.q),
            )

    # 3. Also evaluate Pell-17 (w=9, b=6, k0=21, n0=456) wrapped with t=2 (n=544) and t=3 (n=591)
    res_456, L0_456, _ = core_cache[9]
    pell17_wrapped_544_591: dict[int, dict] = {}
    for t, k_tot, n_tot in ((2, 23, 544), (3, 24, 591)):
        q_wrapped = add_l_border(res_456.q, t=t, k_total=k_tot, margin=1e-7)
        res_w = polish_band(
            q_wrapped,
            extent(q_wrapped),
            iterations=15,
            position_trust=0.05,
            angle_trust=0.005,
            clearance=5e-9,
        )
        assert res_w is not None and len(res_w.q) == n_tot
        pell17_wrapped_544_591[n_tot] = dict(
            kind="pell17_corrugated_band_l_border",
            k_total=k_tot,
            k_core=21,
            t_border=t,
            w=9,
            l=21,
            a=[15, 14],
            b=[6, 6],
            D=17,
            L45=L0_456 + t,
            L_polished=res_w.L,
            corrugation_gain_vs_45=res_w.L - (L0_456 + t),
        )

    # 4. Exact wall-constrained 45-degree staircase-band + L-border packings for the remaining targets
    specs_45 = {
        543: (0, 23, 16, 16, 12, 11, 11, 12, 0),
        544: (0, 23, 4, 28, 20, 20, 3, 3, 0),
        545: (3, 23, 14, 14, 10, 10, 10, 10, 0),
        591: (1, 24, 4, 28, 20, 20, 3, 3, 0),
        592: (0, 24, 14, 21, 14, 14, 9, 9, 2),
        593: (0, 24, 14, 21, 14, 14, 9, 9, 1),
        908: (1, 30, 20, 20, 15, 14, 14, 15, 0),
        909: (1, 30, 8, 32, 23, 23, 6, 6, 0),
        911: (0, 30, 1, 42, 29, 29, 0, 0, 1),
        972: (0, 31, 1, 43, 30, 30, 0, 0, 1),
        973: (0, 31, 1, 43, 30, 30, 0, 0, 0),
    }

    for n_tot, (t, k_tot, w, l, a1, a2, b1, b2, drop_band) in sorted(specs_45.items()):
        if n_tot not in wanted:
            continue
        q, L0 = build_staircase_band_45(w, l, a1, a2, b1, b2, drop_band=drop_band)
        if t > 0:
            q = add_l_border_exact(q, L0, t=t, k_total=k_tot)
        assert len(q) == n_tot
        candidates[n_tot] = rows(q, dilate=1.000000002)
        rec = dict(
            kind="staircase_band_l_border",
            k_total=k_tot,
            k_core=k_tot - t,
            t_border=t,
            w=w,
            l=l,
            a=[a1, a2],
            b=[b1, b2],
            D=(a1 + a2) - (b1 + b2),
            L45=L0 + t,
            count=len(q),
        )
        if n_tot in pell17_wrapped_544_591:
            rec["pell17_corrugated_alternative"] = pell17_wrapped_544_591[n_tot]
        records[n_tot] = rec

    return candidates, records, pell17_family_summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate and certify Pell-17 corrugated band + L-border packings.")
    ap.add_argument("--out", type=Path, default=HERE)
    ap.add_argument("--targets", nargs="+", type=int, default=list(ALL_BRIEF_TARGETS))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)

    start = time.monotonic()
    candidates, records, pell17_family_summary = generate_all_targets(args.targets)
    text = csv_text(candidates)
    checks = certify_transport(text, args.out / "output-pandas.csv")
    (args.out / "output.csv").write_text(text)

    comparisons: dict[str, dict] = {}
    for n in sorted(args.targets):
        c = checks[str(n)]
        assert c["literal"]["ok"] and c["actual_pandas"]["ok"], f"Certification failed for n={n}: {c}"
        L_lit = Decimal(c["literal"]["L"])
        L_pan = Decimal(c["actual_pandas"]["L"])

        port_ref = Decimal(PORTFOLIO_BASELINE[n])
        delta_portfolio = {
            "literal": str(L_lit - port_ref),
            "actual_pandas": str(L_pan - port_ref),
        }
        improves_portfolio = (L_lit - port_ref < Decimal("-1e-8")) and (L_pan - port_ref < Decimal("-1e-8"))

        pub_str = PUBLIC_TARGET_INCUMBENTS.get(n)
        delta_public = None
        improves_public = False
        if pub_str is not None:
            pub_ref = Decimal(pub_str)
            delta_public = {
                "literal": str(L_lit - pub_ref),
                "actual_pandas": str(L_pan - pub_ref),
            }
            improves_public = (L_lit - pub_ref < Decimal("-1e-8")) and (L_pan - pub_ref < Decimal("-1e-8"))

        comparisons[str(n)] = dict(
            construction=records[n],
            certificates=c,
            portfolio_baseline_L=PORTFOLIO_BASELINE[n],
            delta_vs_portfolio_baseline=delta_portfolio,
            improves_portfolio_baseline=improves_portfolio,
            public_target_incumbent_L=pub_str,
            delta_vs_public_incumbent=delta_public,
            improves_public_incumbent=improves_public,
        )

    elapsed = time.monotonic() - start
    result = dict(
        elapsed_seconds=elapsed,
        pandas_version=pd.__version__,
        pell17_family_cores=pell17_family_summary,
        comparisons=comparisons,
        portfolio_baseline_wins=sum(1 for v in comparisons.values() if v["improves_portfolio_baseline"]),
        public_incumbent_wins=sum(1 for v in comparisons.values() if v["improves_public_incumbent"]),
    )
    (args.out / "results.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(dict(
        elapsed_seconds=elapsed,
        portfolio_baseline_wins=result["portfolio_baseline_wins"],
        public_incumbent_wins=result["public_incumbent_wins"],
        n910=comparisons.get("910"),
    ), indent=2))


if __name__ == "__main__":
    main()
