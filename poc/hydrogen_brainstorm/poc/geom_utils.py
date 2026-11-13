"""Shared geometry, numba SAT energy/gradient, LP/SLP tighten, and official certification helpers."""
from __future__ import annotations

import math
import sys
from decimal import Decimal
from pathlib import Path

import numba as nb
import numpy as np
from scipy.optimize import linprog, minimize

REPO_ROOT = Path(__file__).resolve().parents[3]
BASELINE_SRC = REPO_ROOT / "poc" / "baseline" / "poc" / "src"
if str(BASELINE_SRC) not in sys.path:
    sys.path.insert(0, str(BASELINE_SRC))

import squares.core as sc  # noqa: E402


@nb.njit(cache=True)
def sat_max_overlap_and_bbox(x: np.ndarray, y: np.ndarray, th: np.ndarray) -> tuple[float, float]:
    """Return (max_penetration, bbox_L). Penetration > 0 means overlap."""
    n = len(x)
    c = np.cos(th)
    s = np.sin(th)
    hx = 0.5 * (np.abs(c) + np.abs(s))
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
            # 4 separating axes: u_i, v_i, u_j, v_j
            cj, sj = c[j], s[j]
            # Project onto u_i = (ci, si)
            d1 = abs(dx * ci + dy * si)
            r1 = 0.5 + 0.5 * (abs(ci * cj + si * sj) + abs(-ci * sj + si * cj))
            pen1 = r1 - d1
            if pen1 <= 0.0:
                continue
            # v_i = (-si, ci)
            d2 = abs(-dx * si + dy * ci)
            r2 = 0.5 + 0.5 * (abs(-si * cj + ci * sj) + abs(si * sj + ci * cj))
            pen2 = r2 - d2
            if pen2 <= 0.0:
                continue
            # u_j = (cj, sj)
            d3 = abs(dx * cj + dy * sj)
            pen3 = r1 - d3
            if pen3 <= 0.0:
                continue
            # v_j = (-sj, cj)
            d4 = abs(-dx * sj + dy * cj)
            pen4 = r2 - d4
            if pen4 <= 0.0:
                continue
            pen = min(min(pen1, pen2), min(pen3, pen4))
            if pen > max_pen:
                max_pen = pen
    return max_pen, L


@nb.njit(cache=True)
def sat_penalty_grad(state: np.ndarray, L: float, w_wall: float, w_pair: float) -> tuple[float, np.ndarray]:
    """Quadratic SAT penetration + wall violation energy and exact subgradient."""
    n = len(state) // 3
    x = state[:n]
    y = state[n : 2 * n]
    th = state[2 * n :]
    c = np.cos(th)
    s = np.sin(th)
    ac = np.abs(c)
    sac = np.abs(s)
    hx = 0.5 * (ac + sac)
    # dhx/dth
    sc = np.where(c >= 0.0, 1.0, -1.0)
    ss = np.where(s >= 0.0, 1.0, -1.0)
    dhx = 0.5 * (-sc * s + ss * c)

    grad = np.zeros(3 * n, dtype=np.float64)
    energy = 0.0

    # Wall penalty in [0, L]^2
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

    # Pairwise SAT penalty
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
            # Relative cos/sin: cos(th_j - th_i), sin(th_j - th_i)
            cr = ci * cj + si * sj
            sr = ci * sj - si * cj
            acr = abs(cr)
            asr = abs(sr)
            rad = 0.5 + 0.5 * (acr + asr)

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

            # Find minimum penetration axis
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

            # drad/d(th_j - th_i)
            sgn_cr = 1.0 if cr >= 0.0 else -1.0
            sgn_sr = 1.0 if sr >= 0.0 else -1.0
            drad_dr = 0.5 * (-sgn_cr * sr + sgn_sr * cr)

            if best_k == 0:
                # axis u_i = (ci, si)
                sp = 1.0 if p1 >= 0.0 else -1.0
                dpen_dx = -sp * ci
                dpen_dy = -sp * si
                grad[i] -= g * dpen_dx
                grad[j] += g * dpen_dx
                grad[n + i] -= g * dpen_dy
                grad[n + j] += g * dpen_dy
                # dp1/dth_i = -dx*si + dy*ci = p2
                grad[2 * n + i] += g * (-drad_dr - sp * p2)
                grad[2 * n + j] += g * drad_dr
            elif best_k == 1:
                # axis v_i = (-si, ci)
                sp = 1.0 if p2 >= 0.0 else -1.0
                dpen_dx = sp * si
                dpen_dy = -sp * ci
                grad[i] -= g * dpen_dx
                grad[j] += g * dpen_dx
                grad[n + i] -= g * dpen_dy
                grad[n + j] += g * dpen_dy
                # dp2/dth_i = -dx*ci - dy*si = -p1
                grad[2 * n + i] += g * (-drad_dr - sp * (-p1))
                grad[2 * n + j] += g * drad_dr
            elif best_k == 2:
                # axis u_j = (cj, sj)
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
                # axis v_j = (-sj, cj)
                sp = 1.0 if p4 >= 0.0 else -1.0
                dpen_dx = sp * sj
                dpen_dy = -sp * cj
                grad[i] -= g * dpen_dx
                grad[j] += g * dpen_dx
                grad[n + i] -= g * dpen_dy
                grad[n + j] += g * dpen_dy
                grad[2 * n + i] += g * (-drad_dr)
                grad[2 * n + j] += g * (drad_dr - sp * (-p3))
    return energy, grad


def relax_lbfgs(
    x: np.ndarray,
    y: np.ndarray,
    th: np.ndarray,
    L: float,
    free_mask: np.ndarray | None = None,
    maxiter: int = 300,
    w_wall: float = 1.0,
    w_pair: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Minimize SAT penalty inside [0, L]^2 using L-BFGS-B."""
    n = len(x)
    s0 = np.concatenate([x, y, th]).astype(np.float64)
    if free_mask is None:
        def obj(s):
            return sat_penalty_grad(s, L, w_wall, w_pair)

        res = minimize(obj, s0, method="L-BFGS-B", jac=True, options={"maxiter": maxiter, "ftol": 1e-15, "gtol": 1e-12})
        s = res.x
    else:
        mask3 = np.concatenate([free_mask, free_mask, free_mask])
        idx = np.where(mask3)[0]
        if len(idx) == 0:
            pen, _ = sat_max_overlap_and_bbox(x, y, th)
            return x.copy(), y.copy(), th.copy(), pen
        s_work = s0.copy()
        v0 = s0[idx]

        def obj(v):
            s_work[idx] = v
            e, g = sat_penalty_grad(s_work, L, w_wall, w_pair)
            return e, g[idx]

        res = minimize(obj, v0, method="L-BFGS-B", jac=True, options={"maxiter": maxiter, "ftol": 1e-15, "gtol": 1e-12})
        s_work[idx] = res.x
        s = s_work
    nx, ny, nth = s[:n], s[n : 2 * n], s[2 * n :]
    pen, _ = sat_max_overlap_and_bbox(nx, ny, nth)
    return nx, ny, nth, pen


def fixed_angle_lp_tighten(
    x: np.ndarray,
    y: np.ndarray,
    th: np.ndarray,
    rounds: int = 6,
    margin: float = 2e-9,
) -> tuple[np.ndarray, np.ndarray, float, bool]:
    """Fixed-angle SLP to minimize box side L with exact separation margin."""
    n = len(x)
    cx = x.copy()
    cy = y.copy()
    c = np.cos(th)
    s = np.sin(th)
    hx = 0.5 * (np.abs(c) + np.abs(s))
    best_L = float(max((cx + hx).max() - (cx - hx).min(), (cy + hx).max() - (cy - hx).min()))
    best_x, best_y = cx.copy(), cy.copy()

    for _ in range(rounds):
        # Variables: x_0..n-1, y_0..n-1, L (index 2*n)
        rows_idx = []
        cols_idx = []
        vals = []
        ub = []
        row = 0

        # Wall constraints: x_i >= hx_i + margin, x_i - L <= -hx_i - margin, same y
        for i in range(n):
            # -x_i <= -(hx_i + margin)
            rows_idx.append(row); cols_idx.append(i); vals.append(-1.0); ub.append(-(hx[i] + margin)); row += 1
            # x_i - L <= -hx_i - margin
            rows_idx.extend([row, row]); cols_idx.extend([i, 2 * n]); vals.extend([1.0, -1.0]); ub.append(-(hx[i] + margin)); row += 1
            # -y_i <= -(hx_i + margin)
            rows_idx.append(row); cols_idx.append(n + i); vals.append(-1.0); ub.append(-(hx[i] + margin)); row += 1
            # y_i - L <= -hx_i - margin
            rows_idx.extend([row, row]); cols_idx.extend([n + i, 2 * n]); vals.extend([1.0, -1.0]); ub.append(-(hx[i] + margin)); row += 1

        # Pair constraints for pairs within cutoff 2.2
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
                axes = [
                    (ci, si, dx * ci + dy * si),
                    (-si, ci, -dx * si + dy * ci),
                    (cj, sj, dx * cj + dy * sj),
                    (-sj, cj, -dx * sj + dy * cj),
                ]
                # Pick axis with largest |proj| - rad (most separating)
                best_ax = max(axes, key=lambda a: abs(a[2]) - rad)
                ax, ay, p = best_ax
                sgn = 1.0 if p >= 0.0 else -1.0
                # sgn * (ax*(xj - xi) + ay*(yj - yi)) >= rad + margin
                # => sgn*ax*xi + sgn*ay*yi - sgn*ax*xj - sgn*ay*yj <= -(rad + margin)
                rows_idx.extend([row, row, row, row])
                cols_idx.extend([i, n + i, j, n + j])
                vals.extend([sgn * ax, sgn * ay, -sgn * ax, -sgn * ay])
                ub.append(-(rad + margin))
                row += 1

        from scipy.sparse import coo_matrix

        A = coo_matrix((vals, (rows_idx, cols_idx)), shape=(row, 2 * n + 1)).tocsc()
        obj = np.zeros(2 * n + 1)
        obj[2 * n] = 1.0
        bounds = [(None, None)] * (2 * n) + [(0.0, None)]
        res = linprog(obj, A_ub=A, b_ub=np.array(ub), bounds=bounds, method="highs")
        if not res.success:
            return best_x, best_y, best_L, False
        nx = res.x[:n]
        ny = res.x[n : 2 * n]
        pen, cur_L = sat_max_overlap_and_bbox(nx, ny, th)
        if pen > 1e-12:
            return best_x, best_y, best_L, False
        cx, cy = nx, ny
        if cur_L < best_L:
            best_L = cur_L
            best_x, best_y = cx.copy(), cy.copy()
        else:
            break
    return best_x, best_y, best_L, True


def verify_official(n: int, x: np.ndarray, y: np.ndarray, th: np.ndarray) -> dict:
    """Certify literal + pandas roundtrip via official squares.core metric."""
    # Shift to positive quadrant with small margin
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


def load_incumbent_Ls() -> dict[int, float]:
    """Load exact float L of current merged.csv for n=1..1000."""
    import pandas as pd

    csv_path = REPO_ROOT / "poc" / "pell_corrugation" / "data" / "border-scan" / "merged.csv"
    df = pd.read_csv(csv_path)
    rad = df["angle"].values
    hx = 0.5 * (np.abs(np.cos(rad)) + np.abs(np.sin(rad)))
    tmp = pd.DataFrame({
        "n": df["n"].values,
        "x0": df["x"].values - hx,
        "x1": df["x"].values + hx,
        "y0": df["y"].values - hx,
        "y1": df["y"].values + hx,
    })
    g = tmp.groupby("n").agg(minx=("x0", "min"), maxx=("x1", "max"), miny=("y0", "min"), maxy=("y1", "max"))
    g["L"] = np.maximum(g["maxx"] - g["minx"], g["maxy"] - g["miny"])
    return {int(k): float(v) for k, v in g["L"].items()}
