"""Shared geometry, Numba SAT energy/gradient, inflation, LP tighten, and official verification."""
from __future__ import annotations

import math
import os
import sys
from decimal import Decimal
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numba as nb
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

REPO_ROOT = Path(__file__).resolve().parents[3]
BASELINE_SRC = REPO_ROOT / "poc" / "baseline" / "poc" / "src"
if str(BASELINE_SRC) not in sys.path:
    sys.path.insert(0, str(BASELINE_SRC))

import squares.core as sc  # noqa: E402


@nb.njit(cache=True)
def sat_max_overlap_and_bbox(x: np.ndarray, y: np.ndarray, th: np.ndarray, side: float = 1.0) -> tuple[float, float]:
    """Return (max_penetration, bbox_L) for squares of side length `side`."""
    n = len(x)
    c = np.cos(th)
    s = np.sin(th)
    half = 0.5 * side
    hx = half * (np.abs(c) + np.abs(s))
    minx = 1e18
    maxx = -1e18
    miny = 1e18
    maxy = -1e18
    for i in range(n):
        if x[i] - hx[i] < minx:
            minx = x[i] - hx[i]
        if x[i] + hx[i] > maxx:
            maxx = x[i] + hx[i]
        if y[i] - hx[i] < miny:
            miny = y[i] - hx[i]
        if y[i] + hx[i] > maxy:
            maxy = y[i] + hx[i]
    L = max(maxx - minx, maxy - miny)

    max_pen = 0.0
    for i in range(n):
        xi, yi, ci, si, hxi = x[i], y[i], c[i], s[i], hx[i]
        for j in range(i + 1, n):
            dx = x[j] - xi
            if abs(dx) >= hxi + hx[j]:
                continue
            dy = y[j] - yi
            if abs(dy) >= hxi + hx[j]:
                continue
            cj, sj = c[j], s[j]
            rad = half * (1.0 + abs(ci * cj + si * sj) + abs(ci * sj - si * cj))
            d1 = abs(dx * ci + dy * si)
            pen1 = rad - d1
            if pen1 <= 0.0:
                continue
            d2 = abs(-dx * si + dy * ci)
            pen2 = rad - d2
            if pen2 <= 0.0:
                continue
            d3 = abs(dx * cj + dy * sj)
            pen3 = rad - d3
            if pen3 <= 0.0:
                continue
            d4 = abs(-dx * sj + dy * cj)
            pen4 = rad - d4
            if pen4 <= 0.0:
                continue
            pen = min(min(pen1, pen2), min(pen3, pen4))
            if pen > max_pen:
                max_pen = pen
    return max_pen, L


@nb.njit(cache=True)
def sat_penalty_grad(
    state: np.ndarray,
    L: float,
    w_wall: float,
    w_pair: float,
    side: float = 1.0,
) -> tuple[float, np.ndarray]:
    """Quadratic SAT penetration + wall violation energy and exact subgradient for side `side`."""
    n = len(state) // 3
    x = state[:n]
    y = state[n : 2 * n]
    th = state[2 * n :]
    c = np.cos(th)
    s = np.sin(th)
    half = 0.5 * side
    hx = half * (np.abs(c) + np.abs(s))
    sc_sgn = np.where(c >= 0.0, 1.0, -1.0)
    ss_sgn = np.where(s >= 0.0, 1.0, -1.0)
    dhx = half * (-sc_sgn * s + ss_sgn * c)

    grad = np.zeros(3 * n, dtype=np.float64)
    energy = 0.0

    for i in range(n):
        v = hx[i] - x[i]
        if v > 0.0:
            energy += w_wall * v * v
            g = 2.0 * w_wall * v
            grad[i] -= g
            grad[2 * n + i] += g * dhx[i]
        v = x[i] + hx[i] - L
        if v > 0.0:
            energy += w_wall * v * v
            g = 2.0 * w_wall * v
            grad[i] += g
            grad[2 * n + i] += g * dhx[i]
        v = hx[i] - y[i]
        if v > 0.0:
            energy += w_wall * v * v
            g = 2.0 * w_wall * v
            grad[n + i] -= g
            grad[2 * n + i] += g * dhx[i]
        v = y[i] + hx[i] - L
        if v > 0.0:
            energy += w_wall * v * v
            g = 2.0 * w_wall * v
            grad[n + i] += g
            grad[2 * n + i] += g * dhx[i]

    for i in range(n):
        xi, yi, ci, si, hxi = x[i], y[i], c[i], s[i], hx[i]
        for j in range(i + 1, n):
            dx = x[j] - xi
            if abs(dx) >= hxi + hx[j]:
                continue
            dy = y[j] - yi
            if abs(dy) >= hxi + hx[j]:
                continue
            cj, sj = c[j], s[j]
            cr = ci * cj + si * sj
            sr = ci * sj - si * cj
            rad = half * (1.0 + abs(cr) + abs(sr))

            p1 = dx * ci + dy * si
            pen1 = rad - abs(p1)
            if pen1 <= 0.0:
                continue
            p2 = -dx * si + dy * ci
            pen2 = rad - abs(p2)
            if pen2 <= 0.0:
                continue
            p3 = dx * cj + dy * sj
            pen3 = rad - abs(p3)
            if pen3 <= 0.0:
                continue
            p4 = -dx * sj + dy * cj
            pen4 = rad - abs(p4)
            if pen4 <= 0.0:
                continue

            best_k = 0
            pen = pen1
            if pen2 < pen:
                pen = pen2
                best_k = 1
            if pen3 < pen:
                pen = pen3
                best_k = 2
            if pen4 < pen:
                pen = pen4
                best_k = 3

            energy += w_pair * pen * pen
            g = 2.0 * w_pair * pen

            sgn_cr = 1.0 if cr >= 0.0 else -1.0
            sgn_sr = 1.0 if sr >= 0.0 else -1.0
            drad_dr = half * (-sgn_cr * sr + sgn_sr * cr)

            if best_k == 0:
                sp = 1.0 if p1 >= 0.0 else -1.0
                dpen_dx = -sp * ci
                dpen_dy = -sp * si
                grad[i] -= g * dpen_dx
                grad[j] += g * dpen_dx
                grad[n + i] -= g * dpen_dy
                grad[n + j] += g * dpen_dy
                grad[2 * n + i] += g * (-drad_dr - sp * p2)
                grad[2 * n + j] += g * drad_dr
            elif best_k == 1:
                sp = 1.0 if p2 >= 0.0 else -1.0
                dpen_dx = sp * si
                dpen_dy = -sp * ci
                grad[i] -= g * dpen_dx
                grad[j] += g * dpen_dx
                grad[n + i] -= g * dpen_dy
                grad[n + j] += g * dpen_dy
                grad[2 * n + i] += g * (-drad_dr + sp * p1)
                grad[2 * n + j] += g * drad_dr
            elif best_k == 2:
                sp = 1.0 if p3 >= 0.0 else -1.0
                dpen_dx = -sp * cj
                dpen_dy = -sp * sj
                grad[i] -= g * dpen_dx
                grad[j] += g * dpen_dx
                grad[n + i] -= g * dpen_dy
                grad[n + j] += g * dpen_dy
                grad[2 * n + i] += g * (-drad_dr)
                grad[2 * n + j] += g * (drad_dr - sp * p4)
            else:
                sp = 1.0 if p4 >= 0.0 else -1.0
                dpen_dx = sp * sj
                dpen_dy = -sp * cj
                grad[i] -= g * dpen_dx
                grad[j] += g * dpen_dx
                grad[n + i] -= g * dpen_dy
                grad[n + j] += g * dpen_dy
                grad[2 * n + i] += g * (-drad_dr)
                grad[2 * n + j] += g * (drad_dr + sp * p3)
    return energy, grad


@nb.njit(cache=True)
def sat_pair_clear_against_group(
    cx: float,
    cy: float,
    ca: float,
    sa: float,
    hxa: float,
    gx: np.ndarray,
    gy: np.ndarray,
    gc: np.ndarray,
    gs: np.ndarray,
    ghx: np.ndarray,
    min_gap: float = 0.0,
) -> bool:
    """Return True if candidate square at (cx, cy, ca, sa) has >= min_gap SAT clearance to all squares in g."""
    for j in range(len(gx)):
        dx = cx - gx[j]
        if abs(dx) >= hxa + ghx[j] + min_gap:
            continue
        dy = cy - gy[j]
        if abs(dy) >= hxa + ghx[j] + min_gap:
            continue
        cj, sj = gc[j], gs[j]
        rad = 0.5 * (1.0 + abs(ca * cj + sa * sj) + abs(ca * sj - sa * cj)) + min_gap
        if abs(dx * ca + dy * sa) >= rad:
            continue
        if abs(-dx * sa + dy * ca) >= rad:
            continue
        if abs(dx * cj + dy * sj) >= rad:
            continue
        if abs(-dx * sj + dy * cj) >= rad:
            continue
        return False
    return True


def relax_lbfgs(
    x: np.ndarray,
    y: np.ndarray,
    th: np.ndarray,
    L: float,
    free_mask: np.ndarray | None = None,
    maxiter: int = 300,
    w_wall: float = 1.0,
    w_pair: float = 1.0,
    side: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Minimize SAT penalty inside [0, L]^2 using L-BFGS-B."""
    n = len(x)
    s0 = np.concatenate([x, y, th]).astype(np.float64)
    if free_mask is None:
        res = minimize(
            lambda s: sat_penalty_grad(s, L, w_wall, w_pair, side),
            s0,
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": maxiter, "ftol": 1e-15, "gtol": 1e-12},
        )
        s = res.x
    else:
        mask3 = np.concatenate([free_mask, free_mask, free_mask])
        idx = np.where(mask3)[0]
        if len(idx) == 0:
            pen, _ = sat_max_overlap_and_bbox(x, y, th, side)
            return x.copy(), y.copy(), th.copy(), pen
        s_work = s0.copy()

        def obj(v):
            s_work[idx] = v
            e, g = sat_penalty_grad(s_work, L, w_wall, w_pair, side)
            return e, g[idx]

        res = minimize(
            obj,
            s0[idx],
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": maxiter, "ftol": 1e-15, "gtol": 1e-12},
        )
        s_work[idx] = res.x
        s = s_work

    nx, ny, nth = s[:n], s[n : 2 * n], s[2 * n :]
    pen, _ = sat_max_overlap_and_bbox(nx, ny, nth, side)
    # Also include max wall violation in pen
    hx = 0.5 * side * (np.abs(np.cos(nth)) + np.abs(np.sin(nth)))
    wall_pen = max(
        0.0,
        float(np.max(hx - nx)),
        float(np.max(nx + hx - L)),
        float(np.max(hx - ny)),
        float(np.max(ny + hx - L)),
    )
    return nx, ny, nth, max(pen, wall_pen)


def fixed_angle_lp_tighten(
    x: np.ndarray,
    y: np.ndarray,
    th: np.ndarray,
    rounds: int = 5,
    margin: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray, float, bool]:
    """Solve sequential fixed-angle LP for minimum bounding L."""
    n = len(x)
    c = np.cos(th)
    s = np.sin(th)
    hx = 0.5 * (np.abs(c) + np.abs(s))
    cx = x.copy()
    cy = y.copy()
    pen0, best_L = sat_max_overlap_and_bbox(cx, cy, th)
    best_x, best_y = cx.copy(), cy.copy()
    if pen0 > 1e-11:
        best_L = 1e18

    for _ in range(rounds):
        rows_idx: list[int] = []
        cols_idx: list[int] = []
        vals: list[float] = []
        ub: list[float] = []
        row = 0

        for i in range(n):
            rows_idx.append(row); cols_idx.append(i); vals.append(-1.0); ub.append(-(hx[i] + margin)); row += 1
            rows_idx.extend([row, row]); cols_idx.extend([i, 2 * n]); vals.extend([1.0, -1.0]); ub.append(-(hx[i] + margin)); row += 1
            rows_idx.append(row); cols_idx.append(n + i); vals.append(-1.0); ub.append(-(hx[i] + margin)); row += 1
            rows_idx.extend([row, row]); cols_idx.extend([n + i, 2 * n]); vals.extend([1.0, -1.0]); ub.append(-(hx[i] + margin)); row += 1

        for i in range(n):
            xi, yi, ci, si, hxi = cx[i], cy[i], c[i], s[i], hx[i]
            for j in range(i + 1, n):
                dx = cx[j] - xi
                if abs(dx) > hxi + hx[j] + 0.5:
                    continue
                dy = cy[j] - yi
                if abs(dy) > hxi + hx[j] + 0.5:
                    continue
                cj, sj = c[j], s[j]
                rad = 0.5 + 0.5 * (abs(ci * cj + si * sj) + abs(ci * sj - si * cj))
                axes = (
                    (ci, si, dx * ci + dy * si),
                    (-si, ci, -dx * si + dy * ci),
                    (cj, sj, dx * cj + dy * sj),
                    (-sj, cj, -dx * sj + dy * cj),
                )
                ax, ay, p = max(axes, key=lambda a: abs(a[2]) - rad)
                sgn = 1.0 if p >= 0.0 else -1.0
                rows_idx.extend([row, row, row, row])
                cols_idx.extend([i, n + i, j, n + j])
                vals.extend([sgn * ax, sgn * ay, -sgn * ax, -sgn * ay])
                ub.append(-(rad + margin))
                row += 1

        A = coo_matrix((vals, (rows_idx, cols_idx)), shape=(row, 2 * n + 1)).tocsc()
        obj = np.zeros(2 * n + 1)
        obj[2 * n] = 1.0
        bounds = [(None, None)] * (2 * n) + [(0.0, None)]
        res = linprog(obj, A_ub=A, b_ub=np.array(ub), bounds=bounds, method="highs")
        if not res.success:
            return best_x, best_y, best_L, best_L < 1e17
        nx = res.x[:n]
        ny = res.x[n : 2 * n]
        pen, cur_L = sat_max_overlap_and_bbox(nx, ny, th)
        if pen > 1e-12:
            return best_x, best_y, best_L, best_L < 1e17
        cx, cy = nx, ny
        if cur_L < best_L:
            best_L = cur_L
            best_x, best_y = cx.copy(), cy.copy()
        else:
            break
    return best_x, best_y, best_L, best_L < 1e17


def verify_official(n: int, x: np.ndarray, y: np.ndarray, th: np.ndarray) -> dict:
    """Certify literal + pandas roundtrip via official squares.core metric."""
    hx = 0.5 * (np.abs(np.cos(th)) + np.abs(np.sin(th)))
    sx = x - (x - hx).min() + 1e-12
    sy = y - (y - hx).min() + 1e-12
    rows = [
        {
            "square_id": str(i),
            "x": repr(float(sx[i])),
            "y": repr(float(sy[i])),
            "angle": repr(float(th[i])),
        }
        for i in range(n)
    ]
    _, _, side_lit, reason_lit = sc.evaluate(n, rows)
    if reason_lit != "ok" or side_lit is None:
        return {"ok": False, "reason_lit": reason_lit, "L_lit": None, "L_pandas": None}
    rt = sc.kaggle_roundtrip({n: rows})
    _, _, side_pd, reason_pd = sc.evaluate(n, rt[n])
    if reason_pd != "ok" or side_pd is None:
        return {"ok": False, "reason_lit": reason_lit, "reason_pd": reason_pd, "L_lit": side_lit, "L_pandas": None}
    return {
        "ok": True,
        "L_lit": side_lit,
        "L_pandas": side_pd,
        "L_float": float(max(Decimal(side_lit), Decimal(side_pd))),
        "rows": rows,
    }


def load_incumbent_and_frontier(ns: list[int]) -> dict[int, dict[str, float]]:
    """Return dict n -> {'L_ours': float, 'L_front': float}."""
    import pandas as pd

    merged = pd.read_csv(REPO_ROOT / "poc" / "pell_corrugation" / "data" / "border-scan" / "merged.csv")
    gap_df = pd.read_csv(REPO_ROOT / "poc" / "frontier_nov12" / "docs" / "per_n_gap.csv").set_index("n")
    out = {}
    for n in ns:
        sub = merged[merged["n"] == n]
        x, y, a = sub["x"].to_numpy(), sub["y"].to_numpy(), sub["angle"].to_numpy()
        _, L_ours = sat_max_overlap_and_bbox(x, y, a)
        out[n] = {"L_ours": float(L_ours), "L_front": float(gap_df.loc[n, "L_frontier"])}
    return out
