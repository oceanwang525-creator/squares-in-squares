"""Shared geometry, Numba SAT energy/gradient, joint (x, y, dtheta) HiGHS SLP, and official certification."""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import math
import sys
from decimal import Decimal
from pathlib import Path

import numba as nb
import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix, csc_matrix

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


def relax_lbfgs(
    x: np.ndarray,
    y: np.ndarray,
    th: np.ndarray,
    L: float,
    free_mask: np.ndarray | None = None,
    maxiter: int = 250,
    w_wall: float = 2.0,
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
    rounds: int = 6,
    margin: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray, float, bool]:
    """Solve sequential fixed-angle LP for minimum bounding L (HiGHS threads=1)."""
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
        res = linprog(obj, A_ub=A, b_ub=np.array(ub), bounds=bounds, method="highs", options={"threads": 1})
        if not res.success or res.x is None:
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


def _build_joint_slp_sparse(
    x: np.ndarray,
    y: np.ndarray,
    th: np.ndarray,
    tr_xy: float,
    tr_th: float,
    margin: float = 1e-8,
) -> tuple[csc_matrix, np.ndarray]:
    """Build sparse A_ub, b_ub for joint (dx, dy, dtheta, L) LP."""
    n = len(x)
    c = np.cos(th)
    s = np.sin(th)

    s1 = np.array([-0.5, -0.5, 0.5, 0.5], dtype=np.float64)
    s2 = np.array([-0.5, 0.5, -0.5, 0.5], dtype=np.float64)
    dx_c = c[:, None] * s1[None, :] - s[:, None] * s2[None, :]
    dy_c = s[:, None] * s1[None, :] + c[:, None] * s2[None, :]
    X_c = x[:, None] + dx_c
    Y_c = y[:, None] + dy_c

    rows_list = []
    cols_list = []
    vals_list = []
    b_list = []
    row_counter = 0

    curv_pad = 0.25 * (tr_th * tr_th)
    idx_n = np.arange(n, dtype=np.int32)
    order_x = np.argsort(X_c, axis=1)
    order_y = np.argsort(Y_c, axis=1)
    L_est_x = float(np.max(X_c))
    L_est_y = float(np.max(Y_c))

    for k_slot in range(2):
        c_min_x = order_x[:, k_slot]
        x_val = X_c[idx_n, c_min_x]
        dy_val = dy_c[idx_n, c_min_x]
        mask = x_val < 2.5
        if np.any(mask):
            m_idx = idx_n[mask]
            m_len = len(m_idx)
            r_ids = np.arange(row_counter, row_counter + m_len, dtype=np.int32)
            row_counter += m_len
            rows_list.extend([r_ids, r_ids])
            cols_list.extend([m_idx, 2 * n + m_idx])
            vals_list.extend([np.full(m_len, -1.0), dy_val[mask]])
            b_list.append(np.maximum(x_val[mask] - curv_pad, -0.4 * tr_xy))

        c_max_x = order_x[:, 3 - k_slot]
        x_val = X_c[idx_n, c_max_x]
        dy_val = dy_c[idx_n, c_max_x]
        mask = x_val > L_est_x - 2.5
        if np.any(mask):
            m_idx = idx_n[mask]
            m_len = len(m_idx)
            r_ids = np.arange(row_counter, row_counter + m_len, dtype=np.int32)
            row_counter += m_len
            rows_list.extend([r_ids, r_ids, r_ids])
            cols_list.extend([m_idx, 2 * n + m_idx, np.full(m_len, 3 * n, dtype=np.int32)])
            vals_list.extend([np.ones(m_len), -dy_val[mask], np.full(m_len, -1.0)])
            b_list.append(-x_val[mask] - curv_pad)

        c_min_y = order_y[:, k_slot]
        y_val = Y_c[idx_n, c_min_y]
        dx_val = dx_c[idx_n, c_min_y]
        mask = y_val < 2.5
        if np.any(mask):
            m_idx = idx_n[mask]
            m_len = len(m_idx)
            r_ids = np.arange(row_counter, row_counter + m_len, dtype=np.int32)
            row_counter += m_len
            rows_list.extend([r_ids, r_ids])
            cols_list.extend([n + m_idx, 2 * n + m_idx])
            vals_list.extend([np.full(m_len, -1.0), -dx_val[mask]])
            b_list.append(np.maximum(y_val[mask] - curv_pad, -0.4 * tr_xy))

        c_max_y = order_y[:, 3 - k_slot]
        y_val = Y_c[idx_n, c_max_y]
        dx_val = dx_c[idx_n, c_max_y]
        mask = y_val > L_est_y - 2.5
        if np.any(mask):
            m_idx = idx_n[mask]
            m_len = len(m_idx)
            r_ids = np.arange(row_counter, row_counter + m_len, dtype=np.int32)
            row_counter += m_len
            rows_list.extend([r_ids, r_ids, r_ids])
            cols_list.extend([n + m_idx, 2 * n + m_idx, np.full(m_len, 3 * n, dtype=np.int32)])
            vals_list.extend([np.ones(m_len), dx_val[mask], np.full(m_len, -1.0)])
            b_list.append(-y_val[mask] - curv_pad)

    dx_mat = x[None, :] - x[:, None]
    dy_mat = y[None, :] - y[:, None]
    near_i, near_j = np.where(
        (np.abs(dx_mat) < 1.82) & (np.abs(dy_mat) < 1.82) & (np.arange(n)[:, None] < np.arange(n)[None, :])
    )
    m_pairs = len(near_i)
    if m_pairs > 0:
        dxi = dx_mat[near_i, near_j]
        dyi = dy_mat[near_i, near_j]
        ci, si = c[near_i], s[near_i]
        cj, sj = c[near_j], s[near_j]

        nx_all = np.stack([ci, -si, cj, -sj], axis=1)
        ny_all = np.stack([si, ci, sj, cj], axis=1)
        proj = dxi[:, None] * nx_all + dyi[:, None] * ny_all
        s_dir = np.where(proj >= 0.0, 1.0, -1.0)
        abs_proj = np.abs(proj)

        cth = ci * cj + si * sj
        sth = ci * sj - si * cj
        rad_other = 0.5 * (np.abs(cth) + np.abs(sth))
        gaps = abs_proj - (0.5 + rad_other[:, None])

        best_a = np.argmax(gaps, axis=1)
        i_owns = best_a < 2
        p_idx = np.where(i_owns, near_i, near_j)
        q_idx = np.where(i_owns, near_j, near_i)

        pair_arange = np.arange(m_pairs)
        ux = nx_all[pair_arange, best_a] * s_dir[pair_arange, best_a] * np.where(i_owns, 1.0, -1.0)
        uy = ny_all[pair_arange, best_a] * s_dir[pair_arange, best_a] * np.where(i_owns, 1.0, -1.0)

        dx_q = dx_c[q_idx]
        dy_q = dy_c[q_idx]
        dX = (x[q_idx, None] + dx_q) - x[p_idx, None]
        dY = (y[q_idx, None] + dy_q) - y[p_idx, None]
        f_all = dX * ux[:, None] + dY * uy[:, None] - 0.5

        order_f = np.argsort(f_all, axis=1)
        for k_slot in range(2):
            c_idx = order_f[:, k_slot]
            f_val = f_all[pair_arange, c_idx]
            if k_slot == 1:
                f_min = f_all[pair_arange, order_f[:, 0]]
                active_mask = (f_val - f_min) <= (2.5 * tr_th + 0.025)
            else:
                active_mask = f_val < 0.45

            if not np.any(active_mask):
                continue

            mp = p_idx[active_mask]
            mq = q_idx[active_mask]
            mux = ux[active_mask]
            muy = uy[active_mask]
            mdX = dX[active_mask, c_idx[active_mask]]
            mdY = dY[active_mask, c_idx[active_mask]]
            mdx_q = dx_q[active_mask, c_idx[active_mask]]
            mdy_q = dy_q[active_mask, c_idx[active_mask]]
            mf = f_val[active_mask]

            m_len = len(mp)
            r_ids = np.arange(row_counter, row_counter + m_len, dtype=np.int32)
            row_counter += m_len

            c_th_p = mdX * muy - mdY * mux
            c_th_q = mdy_q * mux - mdx_q * muy

            rows_list.extend([r_ids, r_ids, r_ids, r_ids, r_ids, r_ids])
            cols_list.extend([mp, n + mp, 2 * n + mp, mq, n + mq, 2 * n + mq])
            vals_list.extend([mux, muy, c_th_p, -mux, -muy, c_th_q])
            b_list.append(np.maximum(mf - margin - curv_pad, -0.35 * tr_xy))

    all_rows = np.concatenate(rows_list)
    all_cols = np.concatenate(cols_list)
    all_vals = np.concatenate(vals_list)
    all_b = np.concatenate(b_list)
    A = csc_matrix((all_vals, (all_rows, all_cols)), shape=(row_counter, 3 * n + 1))
    return A, all_b


def joint_xytheta_slp(
    x: np.ndarray,
    y: np.ndarray,
    th: np.ndarray,
    rounds: int = 16,
    tr_th0: float = 0.025,
    tr_xy0: float = 0.12,
    margin: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, bool]:
    """Jointly optimize (x, y, th, L) via Sequential Linear Programming + L-BFGS + final LP polish."""
    n = len(x)
    k = math.isqrt(n - 1)
    cur_x = x.copy()
    cur_y = y.copy()
    cur_th = th.copy()

    hx = 0.5 * (np.abs(np.cos(cur_th)) + np.abs(np.sin(cur_th)))
    cur_x = cur_x - float(np.min(cur_x - hx))
    cur_y = cur_y - float(np.min(cur_y - hx))

    pen0, bbox0 = sat_max_overlap_and_bbox(cur_x, cur_y, cur_th)
    if pen0 > 1e-7:
        cur_x, cur_y, cur_th, pen0 = relax_lbfgs(cur_x, cur_y, cur_th, max(bbox0, k + 0.50), maxiter=220)
        pen0, bbox0 = sat_max_overlap_and_bbox(cur_x, cur_y, cur_th)

    best_x, best_y, best_th = cur_x.copy(), cur_y.copy(), cur_th.copy()
    best_L = bbox0 if pen0 <= 1e-7 else 1e9
    best_ok = pen0 <= 1e-7

    obj = np.zeros(3 * n + 1, dtype=np.float64)
    obj[3 * n] = 1.0

    tr_th = tr_th0
    tr_xy = tr_xy0

    for _ in range(rounds):
        A_ub, b_ub = _build_joint_slp_sparse(cur_x, cur_y, cur_th, tr_xy=tr_xy, tr_th=tr_th, margin=margin)
        bounds = (
            [(-tr_xy, tr_xy)] * (2 * n)
            + [(-tr_th, tr_th)] * n
            + [(float(k) + 0.45, float(max(bbox0, k + 0.65) + 0.35))]
        )
        res = linprog(obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs", options={"threads": 1})
        if not res.success or res.x is None:
            tr_th *= 0.6
            tr_xy *= 0.7
            continue

        dx = res.x[:n]
        dy = res.x[n : 2 * n]
        dth = res.x[2 * n : 3 * n]
        lp_L = float(res.x[3 * n])

        accepted = False
        for alpha in (1.0, 0.5):
            cand_x = cur_x + alpha * dx
            cand_y = cur_y + alpha * dy
            cand_th = cur_th + alpha * dth
            pen_c, bbox_c = sat_max_overlap_and_bbox(cand_x, cand_y, cand_th)
            if pen_c > 1e-8:
                target_L = max(lp_L, min(bbox_c, bbox0) - 0.002)
                cand_x, cand_y, cand_th, pen_c = relax_lbfgs(
                    cand_x, cand_y, cand_th, target_L, maxiter=90, w_wall=2.0, w_pair=1.0
                )
                pen_c, bbox_c = sat_max_overlap_and_bbox(cand_x, cand_y, cand_th)

            if pen_c <= 1e-7 and bbox_c < best_L:
                best_L = bbox_c
                best_x, best_y, best_th = cand_x.copy(), cand_y.copy(), cand_th.copy()
                best_ok = True

            if pen_c <= max(5e-4, pen0 * 0.9) and bbox_c <= bbox0 + 0.01:
                cur_x, cur_y, cur_th = cand_x, cand_y, cand_th
                pen0, bbox0 = pen_c, bbox_c
                accepted = True
                break

        if accepted:
            tr_th = max(2e-5, tr_th * 0.84)
            tr_xy = max(5e-4, tr_xy * 0.88)
        else:
            tr_th = max(1e-5, tr_th * 0.5)
            tr_xy = max(5e-4, tr_xy * 0.6)

    for cx, cy, cth in [(best_x, best_y, best_th), (cur_x, cur_y, cur_th)]:
        pen_f, bbox_f = sat_max_overlap_and_bbox(cx, cy, cth)
        if pen_f > 1e-9:
            cx, cy, cth, pen_f = relax_lbfgs(cx, cy, cth, bbox_f, maxiter=250, w_wall=2.0, w_pair=1.0)
        tx, ty, tL, tok = fixed_angle_lp_tighten(cx, cy, cth, rounds=5, margin=1e-8)
        if tok and tL < best_L:
            best_x, best_y, best_th, best_L, best_ok = tx, ty, cth.copy(), tL, True

    return best_x, best_y, best_th, best_L, best_ok


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
        "L_lit_d": Decimal(side_lit),
        "L_pd_d": Decimal(side_pd),
        "L_float": float(max(Decimal(side_lit), Decimal(side_pd))),
        "rows": rows,
        "x": sx,
        "y": sy,
        "th": th.copy(),
    }


def load_incumbent_exact(n: int, merged_df: pd.DataFrame | None = None) -> tuple[Decimal, np.ndarray, np.ndarray, np.ndarray]:
    """Return exact Decimal incumbent side L and shifted (x, y, th) from merged.csv."""
    if merged_df is None:
        merged_df = pd.read_csv(REPO_ROOT / "poc" / "pell_corrugation" / "data" / "border-scan" / "merged.csv")
    sub = merged_df[merged_df["n"] == n]
    rows = [
        {"square_id": str(i), "x": repr(float(r.x)), "y": repr(float(r.y)), "angle": repr(float(r.angle))}
        for i, r in enumerate(sub.itertuples())
    ]
    _, _, side_lit, reason = sc.evaluate(n, rows)
    assert reason == "ok" and side_lit is not None
    x = sub["x"].to_numpy(dtype=np.float64)
    y = sub["y"].to_numpy(dtype=np.float64)
    th = sub["angle"].to_numpy(dtype=np.float64)
    hx = 0.5 * (np.abs(np.cos(th)) + np.abs(np.sin(th)))
    x = x - float(np.min(x - hx))
    y = y - float(np.min(y - hx))
    return Decimal(side_lit), x, y, th


def write_final_csv_from_wins(wins_dir: Path, out_path: Path) -> None:
    """Write poc/swarm_replace45/final.csv in exact CI format (id,x,y,deg) from certified win CSVs."""
    all_rows = []
    for p in sorted(wins_dir.glob("win_n*.csv")):
        df = pd.read_csv(p)
        n = int(df["n"].iloc[0])
        for i, r in enumerate(df.itertuples()):
            deg = math.degrees(float(r.angle))
            all_rows.append({"id": f"{n:03d}_{i}", "x": repr(float(r.x)), "y": repr(float(r.y)), "deg": repr(deg)})
    if all_rows:
        pd.DataFrame(all_rows).to_csv(out_path, index=False)
