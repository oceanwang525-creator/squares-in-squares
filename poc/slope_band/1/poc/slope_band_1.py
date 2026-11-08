"""Exact 1:2 rational slope band generator (theta ≈ 26.565°).

Synthetic CPU packing generator along lattice slope 1:2 (theta = arctan(1/2)).
Transverse inflation extent is h(theta) = 3/sqrt(5) ≈ 1.341641, reducing dilation
waste by sqrt(2) - 3/sqrt(5) ≈ 0.072573 per square compared to 45° bands.
Flank upright staircases step by (2, 1) to match the band normal and tangent.
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
from typing import Any

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[4] if len(Path(__file__).resolve().parents) > 4 else Path(__file__).resolve().parents[3]
BASELINE_SRC = REPO / "poc/baseline/poc/src"
if BASELINE_SRC.exists():
    sys.path.insert(0, str(BASELINE_SRC))

try:
    from squares import _official_metric as metric
    from squares.core import certify
except ImportError:
    metric = None
    certify = None

THETA_1_2: float = math.atan(0.5)
SQRT5: float = math.sqrt(5.0)
INV_SQRT5: float = 1.0 / SQRT5
COS_THETA: float = 2.0 * INV_SQRT5
SIN_THETA: float = 1.0 * INV_SQRT5
TANGENT: np.ndarray = np.array([COS_THETA, SIN_THETA])
NORMAL: np.ndarray = np.array([-SIN_THETA, COS_THETA])

S2: float = math.sqrt(2.0)
H_45: float = S2
H_THETA: float = 3.0 * INV_SQRT5
DELTA_L_ADVANTAGE: float = H_45 - H_THETA

CONVERGENTS_SQRT5: tuple[tuple[int, int, float], ...] = (
    (2, 1, 2.0 - SQRT5),
    (9, 4, 9.0 - 4.0 * SQRT5),
    (38, 17, 38.0 - 17.0 * SQRT5),
    (161, 72, 161.0 - 72.0 * SQRT5),
)

SIGNS: tuple[tuple[float, float], ...] = tuple(product((-1.0, 1.0), repeat=2))

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


def _lp(
    q: np.ndarray,
    upper: float,
    position_trust: float,
    angle_trust: float,
    clearance: float,
    axes: dict,
    rigid_upright: bool = False,
) -> tuple[np.ndarray | None, str]:
    """4-sign-piece epigraph LP relaxation.

    When rigid_upright is True, upright squares (angle == 0) are frozen as rigid
    obstacles, and only tilted squares are permitted to shift and rotate.
    """
    from scipy.sparse import coo_matrix

    n = len(q)
    free = angle_trust > 0.0
    stride = 3 if free else 2
    nv = stride * n + 3  # offsets + xmin, ymin, L
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
    for i in range(n):
        is_upright = abs(float(q[i, 2])) < 1e-4
        if rigid_upright and is_upright:
            bounds.extend([(0.0, 0.0), (0.0, 0.0)])
            if free:
                bounds.append((0.0, 0.0))
        else:
            bounds.extend([(-position_trust, position_trust)] * 2)
            if free:
                bounds.append((-angle_trust, angle_trust))

    bounds.extend([(None, None), (None, None), (1.0, upper)])
    objective = np.zeros(nv)
    objective[-1] = 1.0  # minimize container size L
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
    iterations: int = 25,
    position_trust: float = 0.15,
    angle_trust: float = 0.015,
    clearance: float = 5e-9,
    rigid_upright: bool = False,
) -> PolishResult | None:
    """Iteratively polish packing using 4-sign-piece epigraph LP."""
    axes: dict[tuple[int, int], list[tuple[float, float, int]]] = {}
    repaired, _ = _lp(q, L_init + 0.20, position_trust, 0.0, clearance, axes, rigid_upright=rigid_upright)
    if repaired is None:
        return None
    best = PolishResult(repaired.copy(), extent(repaired))

    for it in range(iterations):
        old = best.L
        fixed, _ = _lp(best.q, old + 1e-4, position_trust, 0.0, clearance, axes, rigid_upright=rigid_upright)
        if fixed is not None and sat_check(fixed)[0] and extent(fixed) < best.L:
            best.q = fixed
            best.L = extent(fixed)
            best.accepted += 1
        proposal, _ = _lp(best.q, best.L + 1e-4, position_trust, angle_trust, clearance, axes, rigid_upright=rigid_upright)
        if proposal is not None:
            origin = best.q.copy()
            for bt in range(10):
                alpha = 2.0 ** (-bt)
                seed = origin + alpha * (proposal - origin)
                rep, _ = _lp(seed, best.L + 1e-3, position_trust, 0.0, clearance, {}, rigid_upright=rigid_upright)
                if rep is not None and sat_check(rep)[0] and extent(rep) < best.L:
                    best.q = rep
                    best.L = extent(rep)
                    best.accepted += 1
                    break
        if old - best.L < 1e-11 and it >= 5:
            break
    return best


def generate_slope_band_unpolished(
    L0: float,
    w: int,
    l: int,
    offset: tuple[float, float] = (0.0, 0.0),
    margin: float = 0.02,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Synthesize an unpolished 1:2 rational slope band with periodic 2x1 teeth.

    Geometry:
      - Band axis: tangent t = (2/sqrt(5), 1/sqrt(5)), normal n = (-1/sqrt(5), 2/sqrt(5)).
      - Tilted band: w strips x l columns rotated at theta = arctan(1/2).
      - Flank staircases: integer grid cells stepping by (2, 1) matching normal distance.
      - Asymmetric corner buffers: 26.56° on horizontal walls, 63.43° on vertical walls.
    """
    xc = L0 / 2.0 + offset[0]
    yc = L0 / 2.0 + offset[1]

    # Generate tilted band squares
    tilted_squares: list[list[float]] = []
    for c in range(l):
        uc = (c - (l - 1) / 2.0) * 1.0
        for s in range(w):
            vs = (s - (w - 1) / 2.0) * 1.0
            # Rotate by theta = arctan(0.5): x = xc + u*cos - v*sin, y = yc + u*sin + v*cos
            x = xc + uc * COS_THETA - vs * SIN_THETA
            y = yc + uc * SIN_THETA + vs * COS_THETA
            tilted_squares.append([x, y, THETA_1_2])

    tilted_arr = np.array(tilted_squares, dtype=float)

    # Flank tooth boundaries in 2:1 integer Cartesian relation
    # Face normal is n = (-1/sqrt(5), 2/sqrt(5)), so 2y - x is constant along band faces.
    # Half-width along n of rotated square is 0.5; half-width of upright square along n is 1.5/sqrt(5).
    # Normal clearance condition for SE flank: x - 2y >= xc - 2yc + 0.5*(w*sqrt(5) + 3).
    # Normal clearance condition for NW flank: 2y - x >= 2yc - xc + 0.5*(w*sqrt(5) + 3).
    k_grid = int(math.floor(L0))
    upright_candidates: list[list[float]] = []
    square_roles: list[str] = []

    for i in range(k_grid):
        for j in range(k_grid):
            x = i + 0.5
            y = j + 0.5
            # Check bounding box inside container
            if x < 0.5 or x > L0 - 0.5 or y < 0.5 or y > L0 - 0.5:
                continue

            # Band coordinates
            dx = x - xc
            dy = y - yc
            u = dx * COS_THETA + dy * SIN_THETA
            v = -dx * SIN_THETA + dy * COS_THETA

            # SAT test against tilted band bounding envelope
            # Band extent: |u| <= l/2 + 0.5, |v| <= w/2 + 0.5
            in_flank_se = v <= -0.5 * w - 0.5 * H_THETA - margin
            in_flank_nw = v >= +0.5 * w + 0.5 * H_THETA + margin
            in_corner_sw = u <= -0.5 * l - 0.5 * H_THETA - margin
            in_corner_ne = u >= +0.5 * l + 0.5 * H_THETA + margin

            if in_flank_se or in_flank_nw or in_corner_sw or in_corner_ne:
                # Fast SAT verification against actual tilted squares
                cand = np.array([[x, y, 0.0]])
                if sat_check(np.vstack([tilted_arr, cand]), clearance=-1e-9)[0]:
                    upright_candidates.append([x, y, 0.0])
                    if in_flank_se:
                        square_roles.append("flank_se_tooth")
                    elif in_flank_nw:
                        square_roles.append("flank_nw_tooth")
                    elif in_corner_sw:
                        square_roles.append("corner_sw_buffer")
                    else:
                        square_roles.append("corner_ne_buffer")

    upright_arr = np.array(upright_candidates, dtype=float) if upright_candidates else np.zeros((0, 3))
    all_squares = np.vstack([tilted_arr, upright_arr]) if len(upright_arr) else tilted_arr

    meta = {
        "L0": L0,
        "w": w,
        "l": l,
        "theta": THETA_1_2,
        "theta_deg": math.degrees(THETA_1_2),
        "n_total": len(all_squares),
        "n_tilted": len(tilted_arr),
        "n_upright": len(upright_arr),
        "roles_summary": {
            "tilted_band": len(tilted_arr),
            "flank_se_teeth": square_roles.count("flank_se_tooth"),
            "flank_nw_teeth": square_roles.count("flank_nw_tooth"),
            "corner_sw": square_roles.count("corner_sw_buffer"),
            "corner_ne": square_roles.count("corner_ne_buffer"),
        },
    }
    return all_squares, meta


def build_slope_band_prototype(
    L0: float = 15.0,
    w: int = 4,
    l: int = 9,
    polish: bool = False,
    rigid_upright: bool = True,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    """Build the prototype 1:2 rational slope band at L0 ≈ 15 (n ≈ 200).

    Uses convergent 9/4 (9 tilted squares match 4 integer 2x1 tooth blocks).
    """
    q_init, meta = generate_slope_band_unpolished(L0=L0, w=w, l=l, margin=0.01)
    L_curr = extent(q_init)

    if polish:
        res = polish_band(
            q_init,
            L_init=L_curr,
            iterations=25,
            position_trust=0.10,
            angle_trust=0.015,
            clearance=5e-9,
            rigid_upright=rigid_upright,
        )
        if res is not None:
            q_out = normalize_box(res.q)
            meta["polished"] = True
            meta["L_polished"] = res.L
            meta["polish_accepted"] = res.accepted
            return q_out, res.L, meta

    meta["polished"] = False
    return normalize_box(q_init), L_curr, meta


def build_slope_band_908(
    L0: float = 30.64,
    w: int = 10,
    l: int = 38,
    polish: bool = False,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    """Build the target n ≈ 908 slope band matching convergent 38/17.

    38 tilted squares (period 1.0) match 17 tooth blocks of 2x1 upright squares
    (period sqrt(5) ≈ 2.236068) with longitudinal vernier mismatch |38 - 17*sqrt(5)| ≈ 0.01315.
    """
    q_init, meta = generate_slope_band_unpolished(L0=L0, w=w, l=l, margin=0.02)
    L_curr = extent(q_init)

    if polish:
        res = polish_band(
            q_init,
            L_init=L_curr,
            iterations=20,
            position_trust=0.10,
            angle_trust=0.015,
            clearance=5e-9,
            rigid_upright=False,
        )
        if res is not None:
            q_out = normalize_box(res.q)
            meta["polished"] = True
            meta["L_polished"] = res.L
            return q_out, res.L, meta

    meta["polished"] = False
    return normalize_box(q_init), L_curr, meta


def falsification_test(
    L0: float = 15.0,
    w: int = 4,
    l: int = 9,
    position_trust: float = 0.10,
) -> dict[str, Any]:
    """Execute the exact fast falsification test from q5_beyond_atlas_and_plan.md.

    Gate:
      If minimum container dilation Delta L required to resolve vernier overlaps
      exceeds Delta L_adv = sqrt(2) - 3/sqrt(5) ≈ 0.072573 (the geometric inflation advantage),
      the tooth mismatch wipes out the slope advantage, falsifying the mechanism.
    """
    q_unpolished, meta = generate_slope_band_unpolished(L0=L0, w=w, l=l, margin=0.01)
    L_initial = extent(q_unpolished)

    # Formulate LP against rigid upright teeth with position trust region 0.10
    axes: dict[tuple[int, int], list[tuple[float, float, int]]] = {}
    repaired, msg = _lp(
        q_unpolished,
        upper=L0 + 0.50,
        position_trust=position_trust,
        angle_trust=0.0,
        clearance=1e-8,
        axes=axes,
        rigid_upright=True,
    )

    if repaired is None:
        return {
            "passed": False,
            "reason": f"LP infeasible: {msg}",
            "dilation": float("inf"),
            "threshold": DELTA_L_ADVANTAGE,
            "meta": meta,
        }

    L_feasible = extent(repaired)
    dilation = max(0.0, L_feasible - L0)
    passed = dilation <= DELTA_L_ADVANTAGE

    return {
        "passed": passed,
        "dilation": dilation,
        "inflation_advantage": DELTA_L_ADVANTAGE,
        "margin": DELTA_L_ADVANTAGE - dilation,
        "L0": L0,
        "L_feasible": L_feasible,
        "L_initial": L_initial,
        "n_total": meta["n_total"],
        "n_tilted": meta["n_tilted"],
        "n_upright": meta["n_upright"],
        "convergents_match": "9/4",
        "vernier_mismatch": abs(9.0 - 4.0 * SQRT5),
    }


def csv_text(puzzles: dict[int, list[dict[str, str]]]) -> str:
    """Serialize puzzles to official metric submission CSV format."""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=["n", "square_id", "x", "y", "angle"])
    writer.writeheader()
    for n in sorted(puzzles.keys()):
        for row in puzzles[n]:
            writer.writerow(dict(n=str(n), **row))
    return buf.getvalue()


def rows_from_array(q: np.ndarray) -> list[dict[str, str]]:
    """Convert numpy coordinate array to row dicts with exact 50-digit decimal strings."""
    out = []
    for i, sq in enumerate(q):
        out.append(
            dict(
                square_id=str(i),
                x=f"{Decimal(str(sq[0])):.50f}",
                y=f"{Decimal(str(sq[1])):.50f}",
                angle=f"{Decimal(str(sq[2])):.50f}",
            )
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Slope band (POC 1) 1:2 rational slope band generator.")
    ap.add_argument("--falsification-test", action="store_true", help="Run the exact fast falsification test.")
    ap.add_argument("--prototype", action="store_true", help="Build L0=15, n≈200 prototype.")
    ap.add_argument("--target-908", action="store_true", help="Build L0=30.64, n≈908 configuration.")
    ap.add_argument("--polish", action="store_true", help="Run 4-sign-piece epigraph LP polish.")
    ap.add_argument("--out", type=Path, default=None, help="Output directory for generated artifacts.")
    args = ap.parse_args()

    if args.falsification_test:
        res = falsification_test()
        print(json.dumps(res, indent=2))
        return

    if args.prototype or not (args.target_908 or args.falsification_test):
        q, L, meta = build_slope_band_prototype(polish=args.polish)
        print(f"Prototype generated: n={len(q)} (tilted={meta['n_tilted']}, upright={meta['n_upright']}), L={L:.6f}")
        print(f"SAT check: {sat_check(q)[0]}")
        if args.out:
            args.out.mkdir(parents=True, exist_ok=True)
            text = csv_text({len(q): rows_from_array(q)})
            (args.out / f"slope_band_proto_n{len(q)}.csv").write_text(text)

    if args.target_908:
        q, L, meta = build_slope_band_908(polish=args.polish)
        print(f"Target 908 generated: n={len(q)} (tilted={meta['n_tilted']}, upright={meta['n_upright']}), L={L:.6f}")
        print(f"SAT check: {sat_check(q)[0]}")
        if args.out:
            args.out.mkdir(parents=True, exist_ok=True)
            text = csv_text({len(q): rows_from_array(q)})
            (args.out / f"slope_band_908_n{len(q)}.csv").write_text(text)


if __name__ == "__main__":
    main()
