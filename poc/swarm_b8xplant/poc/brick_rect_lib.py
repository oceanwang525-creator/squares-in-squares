#!/usr/bin/env python3
"""Pinwheel sub-brick extraction, rectangular SLP, 5-region pinwheel assembly, and seam melt."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import csc_matrix

REPO = Path(__file__).resolve().parents[3]
HOMO_POC = REPO / "poc" / "swarm_homology" / "poc"
HIGHD_POC = REPO / "poc" / "swarm_highd" / "poc"
NEW159_POC = REPO / "poc" / "swarm_new159" / "poc"
GAD_POC = REPO / "poc" / "rectangle_gadgets" / "poc"
for p in (HOMO_POC, HIGHD_POC, NEW159_POC, GAD_POC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import gadlib  # noqa: E402
import pocket_seeds as ps  # noqa: E402
import rgad  # noqa: E402
import utils as u  # noqa: E402
from cpp_wrap import cpp_eval_sat_pen_bbox, cpp_expand_sym, full_to_free_sym, run_cpp_masked_melt  # noqa: E402
from joint_slp import _build_joint_slp_sparse, joint_xytheta_slp  # noqa: E402


def joint_xytheta_slp_tight(
    x: np.ndarray,
    y: np.ndarray,
    th: np.ndarray,
    stages: list[tuple[int, float, float, float]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, bool]:
    """Multi-stage joint (x, y, th, L) SLP decaying trust region down to ~1e-4 rad so curv_pad -> ~0."""
    if stages is None:
        stages = [
            (10, 0.020, 0.09, 1e-8),
            (10, 0.005, 0.025, 1e-10),
            (8, 0.0012, 0.006, 1e-11),
            (6, 0.00025, 0.0015, 1e-12),
        ]
    cx, cy, cth = x.copy(), y.copy(), th.copy()
    pen0, L0 = cpp_eval_sat_pen_bbox(cx, cy, cth)
    best_x, best_y, best_th, best_L, best_ok = cx.copy(), cy.copy(), cth.copy(), L0, (pen0 <= 1e-9)

    for rounds, tr_th0, tr_xy0, tol in stages:
        xs, ys, ts, Ls, ok = joint_xytheta_slp(
            cx, cy, cth, rounds=rounds, tr_th0=tr_th0, tr_xy0=tr_xy0, margin=1e-8, tol=tol, patience=2
        )
        if ok:
            cx, cy, cth = xs, ys, ts
            if not best_ok or Ls < best_L:
                best_x, best_y, best_th, best_L, best_ok = xs.copy(), ys.copy(), ts.copy(), Ls, True
    return best_x, best_y, best_th, best_L, best_ok


def rect_bbox(x: np.ndarray, y: np.ndarray, th: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    hx = 0.5 * (np.abs(np.cos(th)) + np.abs(np.sin(th)))
    nx = x - float((x - hx).min())
    ny = y - float((y - hx).min())
    W = float((nx + hx).max())
    H = float((ny + hx).max())
    return nx, ny, W, H


def relax_rect_lbfgs(
    x: np.ndarray, y: np.ndarray, th: np.ndarray, Lx: float, Ly: float, maxiter: int = 400
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    X = np.empty(3 * len(x), dtype=np.float64)
    X[0::3] = x
    X[1::3] = y
    X[2::3] = th
    res = minimize(lambda v: rgad.eg(v, Lx, Ly), X, jac=True, method="L-BFGS-B",
                   options={"maxiter": maxiter, "ftol": 1e-30, "gtol": 1e-16, "maxcor": 25})
    rx = res.x
    return rx[0::3].copy(), rx[1::3].copy(), rx[2::3].copy(), float(res.fun)


def _build_rect_slp_sparse(
    x: np.ndarray, y: np.ndarray, th: np.ndarray, w: int, h: int, tr_xy: float, tr_th: float, margin: float = 1e-8
) -> tuple[csc_matrix, np.ndarray]:
    """Build sparse A_ub, b_ub for joint (dx, dy, dtheta, f) LP in a (w+f) x (h+f) box."""
    n = len(x)
    c = np.cos(th)
    s = np.sin(th)
    s1 = np.array([-0.5, -0.5, 0.5, 0.5], dtype=np.float64)
    s2 = np.array([-0.5, 0.5, -0.5, 0.5], dtype=np.float64)
    dx_c = c[:, None] * s1[None, :] - s[:, None] * s2[None, :]
    dy_c = s[:, None] * s1[None, :] + c[:, None] * s2[None, :]
    X_c = x[:, None] + dx_c
    Y_c = y[:, None] + dy_c

    rows_list, cols_list, vals_list, b_list = [], [], [], []
    row_counter = 0
    curv_pad = 0.25 * (tr_th * tr_th)
    idx_n = np.arange(n, dtype=np.int32)
    order_x = np.argsort(X_c, axis=1)
    order_y = np.argsort(Y_c, axis=1)
    W_est = float(np.max(X_c))
    H_est = float(np.max(Y_c))

    for k_slot in range(2):
        c_min_x = order_x[:, k_slot]
        x_val = X_c[idx_n, c_min_x]
        dy_val = dy_c[idx_n, c_min_x]
        mask = x_val < 2.2
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
        mask = x_val > W_est - 2.2
        if np.any(mask):
            m_idx = idx_n[mask]
            m_len = len(m_idx)
            r_ids = np.arange(row_counter, row_counter + m_len, dtype=np.int32)
            row_counter += m_len
            rows_list.extend([r_ids, r_ids, r_ids])
            cols_list.extend([m_idx, 2 * n + m_idx, np.full(m_len, 3 * n, dtype=np.int32)])
            vals_list.extend([np.ones(m_len), -dy_val[mask], np.full(m_len, -1.0)])
            b_list.append(float(w) - x_val[mask] - curv_pad)

        c_min_y = order_y[:, k_slot]
        y_val = Y_c[idx_n, c_min_y]
        dx_val = dx_c[idx_n, c_min_y]
        mask = y_val < 2.2
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
        mask = y_val > H_est - 2.2
        if np.any(mask):
            m_idx = idx_n[mask]
            m_len = len(m_idx)
            r_ids = np.arange(row_counter, row_counter + m_len, dtype=np.int32)
            row_counter += m_len
            rows_list.extend([r_ids, r_ids, r_ids])
            cols_list.extend([n + m_idx, 2 * n + m_idx, np.full(m_len, 3 * n, dtype=np.int32)])
            vals_list.extend([np.ones(m_len), dx_val[mask], np.full(m_len, -1.0)])
            b_list.append(float(h) - y_val[mask] - curv_pad)

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

    A = csc_matrix((np.concatenate(vals_list), (np.concatenate(rows_list), np.concatenate(cols_list))), shape=(row_counter, 3 * n + 1))
    return A, np.concatenate(b_list)


def fixed_angle_lp_tighten_rect(
    x: np.ndarray, y: np.ndarray, th: np.ndarray, w: int, h: int, rounds: int = 5, margin: float = 1e-8
) -> tuple[np.ndarray, np.ndarray, float, bool]:
    """Fixed-angle LP tightening for (w+f) x (h+f) rectangle minimizing f."""
    n = len(x)
    cx, cy, W, H = rect_bbox(x, y, th)
    f0 = max(W - w, H - h)
    pen0, _ = cpp_eval_sat_pen_bbox(cx, cy, th)
    if pen0 > 1e-7:
        return cx, cy, f0, False
    best_x, best_y, best_f = cx.copy(), cy.copy(), f0

    obj = np.zeros(2 * n + 1, dtype=np.float64)
    obj[2 * n] = 1.0
    hx = 0.5 * (np.abs(np.cos(th)) + np.abs(np.sin(th)))
    c, s = np.cos(th), np.sin(th)
    s1 = np.array([-0.5, -0.5, 0.5, 0.5])
    s2 = np.array([-0.5, 0.5, -0.5, 0.5])
    dx_c = c[:, None] * s1[None, :] - s[:, None] * s2[None, :]
    dy_c = s[:, None] * s1[None, :] + c[:, None] * s2[None, :]

    for _ in range(rounds):
        rows_l, cols_l, vals_l, b_l = [], [], [], []
        rc = 0
        # Wall constraints
        idx = np.arange(n, dtype=np.int32)
        # x - hx + dx >= 0 => -dx <= x - hx
        rows_l.append(np.arange(rc, rc + n)); cols_l.append(idx); vals_l.append(np.full(n, -1.0)); b_l.append(cx - hx); rc += n
        # x + hx + dx <= w + f => dx - f <= w - x - hx
        rows_l.extend([np.arange(rc, rc + n), np.arange(rc, rc + n)])
        cols_l.extend([idx, np.full(n, 2 * n)]); vals_l.extend([np.ones(n), np.full(n, -1.0)]); b_l.append(w - cx - hx); rc += n
        # y - hx + dy >= 0
        rows_l.append(np.arange(rc, rc + n)); cols_l.append(n + idx); vals_l.append(np.full(n, -1.0)); b_l.append(cy - hx); rc += n
        # y + hx + dy <= h + f
        rows_l.extend([np.arange(rc, rc + n), np.arange(rc, rc + n)])
        cols_l.extend([n + idx, np.full(n, 2 * n)]); vals_l.extend([np.ones(n), np.full(n, -1.0)]); b_l.append(h - cy - hx); rc += n

        dx_mat = cx[None, :] - cx[:, None]
        dy_mat = cy[None, :] - cy[:, None]
        ni, nj = np.where((np.abs(dx_mat) < 1.82) & (np.abs(dy_mat) < 1.82) & (np.arange(n)[:, None] < np.arange(n)[None, :]))
        if len(ni) > 0:
            dxi, dyi = dx_mat[ni, nj], dy_mat[ni, nj]
            ci, si, cj, sj = c[ni], s[ni], c[nj], s[nj]
            nx_all = np.stack([ci, -si, cj, -sj], axis=1)
            ny_all = np.stack([si, ci, sj, cj], axis=1)
            proj = dxi[:, None] * nx_all + dyi[:, None] * ny_all
            s_dir = np.where(proj >= 0.0, 1.0, -1.0)
            cth, sth = ci * cj + si * sj, ci * sj - si * cj
            rad_other = 0.5 * (np.abs(cth) + np.abs(sth))
            gaps = np.abs(proj) - (0.5 + rad_other[:, None])
            ba = np.argmax(gaps, axis=1)
            i_owns = ba < 2
            p_idx = np.where(i_owns, ni, nj)
            q_idx = np.where(i_owns, nj, ni)
            pa = np.arange(len(ni))
            ux = nx_all[pa, ba] * s_dir[pa, ba] * np.where(i_owns, 1.0, -1.0)
            uy = ny_all[pa, ba] * s_dir[pa, ba] * np.where(i_owns, 1.0, -1.0)
            dX = (cx[q_idx, None] + dx_c[q_idx]) - cx[p_idx, None]
            dY = (cy[q_idx, None] + dy_c[q_idx]) - cy[p_idx, None]
            f_val = np.min(dX * ux[:, None] + dY * uy[:, None] - 0.5, axis=1)
            act = f_val < 0.35
            if np.any(act):
                mp, mq, mux, muy, mf = p_idx[act], q_idx[act], ux[act], uy[act], f_val[act]
                ml = len(mp)
                r_ids = np.arange(rc, rc + ml, dtype=np.int32)
                rc += ml
                rows_l.extend([r_ids, r_ids, r_ids, r_ids])
                cols_l.extend([mp, n + mp, mq, n + mq])
                vals_l.extend([mux, muy, -mux, -muy])
                b_l.append(np.maximum(mf - margin, -0.05))

        A = csc_matrix((np.concatenate(vals_l), (np.concatenate(rows_l), np.concatenate(cols_l))), shape=(rc, 2 * n + 1))
        b = np.concatenate(b_l)
        res = u._highs_linprog(obj, A_ub=A, b_ub=b, bounds=[(-0.08, 0.08)] * (2 * n) + [(0.2, best_f + 0.1)])
        if not res.success or res.x is None:
            break
        cand_x = cx + res.x[:n]
        cand_y = cy + res.x[n : 2 * n]
        cand_x, cand_y, Wc, Hc = rect_bbox(cand_x, cand_y, th)
        fc = max(Wc - w, Hc - h)
        pen_c, _ = cpp_eval_sat_pen_bbox(cand_x, cand_y, th)
        if pen_c <= 1e-10 and fc < best_f - 1e-12:
            best_x, best_y, best_f = cand_x, cand_y, fc
            cx, cy = cand_x, cand_y
        else:
            break
    return best_x, best_y, best_f, True


def joint_xytheta_slp_rect(
    x: np.ndarray,
    y: np.ndarray,
    th: np.ndarray,
    w: int,
    h: int,
    rounds: int = 10,
    tr_th0: float = 0.020,
    tr_xy0: float = 0.09,
    margin: float = 1e-8,
    tol: float = 1e-9,
    patience: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, bool]:
    """Jointly optimize (x, y, th, f) in a (w+f) x (h+f) rectangular box."""
    n = len(x)
    cur_x, cur_y, W0, H0 = rect_bbox(x, y, th)
    cur_th = th.copy()
    f0 = max(W0 - w, H0 - h)
    pen0, _ = cpp_eval_sat_pen_bbox(cur_x, cur_y, cur_th)
    if pen0 > 1e-7:
        cur_x, cur_y, cur_th, _ = relax_rect_lbfgs(cur_x, cur_y, cur_th, w + max(f0, 0.70), h + max(f0, 0.70), maxiter=250)
        cur_x, cur_y, W0, H0 = rect_bbox(cur_x, cur_y, cur_th)
        f0 = max(W0 - w, H0 - h)
        pen0, _ = cpp_eval_sat_pen_bbox(cur_x, cur_y, cur_th)

    best_x, best_y, best_th = cur_x.copy(), cur_y.copy(), cur_th.copy()
    best_f = f0 if pen0 <= 1e-8 else 1e9
    best_ok = pen0 <= 1e-8

    obj = np.zeros(3 * n + 1, dtype=np.float64)
    obj[3 * n] = 1.0
    tr_th, tr_xy = tr_th0, tr_xy0
    f_hi = float(max(f0, 1.0) + 0.25)
    stall = 0

    for _ in range(rounds):
        prev_best_f = best_f
        A_ub, b_ub = _build_rect_slp_sparse(cur_x, cur_y, cur_th, w, h, tr_xy=tr_xy, tr_th=tr_th, margin=margin)
        bounds = [(-tr_xy, tr_xy)] * (2 * n) + [(-tr_th, tr_th)] * n + [(0.20, f_hi)]
        res = u._highs_linprog(obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds)
        if not res.success or res.x is None:
            tr_th *= 0.6
            tr_xy *= 0.7
            continue
        dx, dy, dth = res.x[:n], res.x[n : 2 * n], res.x[2 * n : 3 * n]
        lp_f = float(res.x[3 * n])

        accepted = False
        for alpha in (1.0, 0.5):
            cand_x, cand_y, cand_th = cur_x + alpha * dx, cur_y + alpha * dy, cur_th + alpha * dth
            cand_x, cand_y, Wc, Hc = rect_bbox(cand_x, cand_y, cand_th)
            fc = max(Wc - w, Hc - h)
            pen_c, _ = cpp_eval_sat_pen_bbox(cand_x, cand_y, cand_th)
            if pen_c > 1e-8:
                tf = max(lp_f, min(fc, f0) - 0.002)
                cand_x, cand_y, cand_th, _ = relax_rect_lbfgs(cand_x, cand_y, cand_th, w + tf, h + tf, maxiter=140)
                cand_x, cand_y, Wc, Hc = rect_bbox(cand_x, cand_y, cand_th)
                fc = max(Wc - w, Hc - h)
                pen_c, _ = cpp_eval_sat_pen_bbox(cand_x, cand_y, cand_th)
            if pen_c <= 1e-9 and fc < best_f:
                best_f = fc
                best_x, best_y, best_th = cand_x.copy(), cand_y.copy(), cand_th.copy()
                best_ok = True
            if pen_c <= max(5e-4, pen0 * 0.9) and fc <= f0 + 0.01:
                cur_x, cur_y, cur_th = cand_x, cand_y, cand_th
                pen0, f0 = pen_c, fc
                accepted = True
                break
        if accepted:
            tr_th = max(2e-5, tr_th * 0.84)
            tr_xy = max(5e-4, tr_xy * 0.88)
        else:
            tr_th = max(1e-5, tr_th * 0.5)
            tr_xy = max(5e-4, tr_xy * 0.6)
        if best_ok and (prev_best_f - best_f) < tol:
            stall += 1
            if stall >= patience:
                break
        else:
            stall = 0

    for cx, cy, cth in ((best_x, best_y, best_th), (cur_x, cur_y, cur_th)):
        pen_f, _ = cpp_eval_sat_pen_bbox(cx, cy, cth)
        if pen_f > 1e-9:
            _, _, Wf, Hf = rect_bbox(cx, cy, cth)
            ff = max(Wf - w, Hf - h)
            cx, cy, cth, _ = relax_rect_lbfgs(cx, cy, cth, w + ff, h + ff, maxiter=250)
        tx, ty, tf, tok = fixed_angle_lp_tighten_rect(cx, cy, cth, w, h, rounds=4, margin=1e-8)
        if tok and tf < best_f:
            best_x, best_y, best_th, best_f, best_ok = tx, ty, cth.copy(), tf, True

    return best_x, best_y, best_th, best_f, best_ok


def extract_subbricks(packs: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, float]]) -> dict[tuple[int, int, int], dict]:
    """Extract best (w, h, g) bricks from frontier packings (both clean-cut and forced same-tilt rects)."""
    best: dict[tuple[int, int, int], dict] = {}

    def _add_brick(w: int, h: int, g: int, x: np.ndarray, y: np.ndarray, th: np.ndarray, src_n: int, tag: str):
        if g <= 0 or w <= 0 or h <= 0 or len(x) != w * h + g:
            return
        pen, _ = cpp_eval_sat_pen_bbox(x, y, th)
        if pen > 1e-7:
            return
        nx, ny, W, H = rect_bbox(x, y, th)
        fx, fy = W - w, H - h
        f = max(fx, fy)
        if f >= 1.0 - 1e-7:
            return
        for rot in (0, 1):
            if rot == 0:
                rw, rh, rx, ry, rt, rfx, rfy = w, h, nx.copy(), ny.copy(), th.copy(), fx, fy
            else:
                rw, rh = h, w
                rx, ry, rt = ny.copy(), (W - nx).copy(), (th + 0.5 * math.pi).copy()
                rx, ry, W2, H2 = rect_bbox(rx, ry, rt)
                rfx, rfy = W2 - rw, H2 - rh
            k = (rw, rh, g)
            if k not in best or f < best[k]["f"] - 1e-12:
                best[k] = {
                    "w": rw, "h": rh, "g": g, "f": max(rfx, rfy), "fx": rfx, "fy": rfy,
                    "src_n": src_n, "tag": tag, "x": rx, "y": ry, "th": rt,
                }

    for n, (x, y, th, L) in sorted(packs.items()):
        a = np.column_stack([x, y, th])
        e = 0.5 * (np.abs(np.cos(th)) + np.abs(np.sin(th)))
        K = int(L + 1e-9)
        if n > K * K and L - K < 1.0 - 1e-7:
            _add_brick(K, K, n - K * K, x, y, th, n, f"whole_{n}")

        t = np.degrees(np.mod(th, np.pi / 2))
        for thresh in (0.5, 2.0):
            tilt = np.minimum(t, 90.0 - t) > thresh
            if not np.any(tilt):
                continue
            for cl in gadlib.clusters(a, tilt, link=1.42):
                x0, x1 = (a[cl, 0] - e[cl]).min(), (a[cl, 0] + e[cl]).max()
                y0, y1 = (a[cl, 1] - e[cl]).min(), (a[cl, 1] + e[cl]).max()
                X0, X1, Y0, Y1, idx = gadlib.clean_rect(a, e, x0, x1, y0, y1)
                W, H = X1 - X0, Y1 - Y0
                if W > L - 0.5 and H > L - 0.5:
                    continue
                w, h = int(W + 1e-7), int(H + 1e-7)
                g = len(idx) - w * h
                if g > 0:
                    _add_brick(w, h, g, a[idx, 0] - X0, a[idx, 1] - Y0, a[idx, 2], n, f"clean_{n}")

    # Forced extraction of n=910's two clean rectangular regions (10x21,+6) and (19x9,+4)
    if 910 in packs:
        x, y, th, L = packs[910]
        e = 0.5 * (np.abs(np.cos(th)) + np.abs(np.sin(th)))
        for X0, X1, Y0, Y1, w, h, tag in [
            (0.0, 10.66, 9.0, 30.66, 10, 21, "forced_910_L"),
            (11.0, 30.66, 0.0, 9.66, 19, 9, "forced_910_B"),
        ]:
            in_r = (x - e >= X0 - 1e-3) & (x + e <= X1 + 1e-3) & (y - e >= Y0 - 1e-3) & (y + e <= Y1 + 1e-3)
            idx = np.where(in_r)[0]
            if len(idx) > w * h:
                _add_brick(w, h, len(idx) - w * h, x[idx], y[idx], th[idx], 910, tag)

    return best


def assemble_pinwheel(
    m: int, b1: dict, b2: dict, align_mode: str = "center"
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float] | None:
    """Assemble 5-region pinwheel in [0, m+f]^2 from TL brick b1 and BR brick b2."""
    w1, h1, g1 = b1["w"], b1["h"], b1["g"]
    w2, h2, g2 = b2["w"], b2["h"], b2["g"]
    if not ((w1 + w2 <= m - 1 and h1 + h2 <= m) or (h1 + h2 <= m - 1 and w1 + w2 <= m)):
        return None
    f = max(b1["f"], b2["f"])
    L = m + f

    # Place b1 in TL [0, w1+f] x [m-h1, m+f]
    if align_mode == "flush_outer":
        dx1, dy1 = 0.0, (m + f) - (h1 + b1["fy"])
        dx2, dy2 = (m + f) - (w2 + b2["fx"]), 0.0
    elif align_mode == "flush_inner":
        dx1, dy1 = (w1 + f) - (w1 + b1["fx"]), float(m - h1)
        dx2, dy2 = float(m - w2), (h2 + f) - (h2 + b2["fy"])
    else:  # center inside slack
        dx1 = 0.5 * (f - b1["fx"])
        dy1 = (m - h1) + 0.5 * (f - b1["fy"])
        dx2 = (m - w2) + 0.5 * (f - b2["fx"])
        dy2 = 0.5 * (f - b2["fy"])

    xs = [b1["x"] + dx1, b2["x"] + dx2]
    ys = [b1["y"] + dy1, b2["y"] + dy2]
    ths = [b1["th"], b2["th"]]

    gx, gy = [], []
    for i in range(m):
        for j in range(m):
            if i < w1 and j >= m - h1:
                continue
            if i >= m - w2 and j < h2:
                continue
            if i < m - w2 and j < m - h1:
                gx.append(i + 0.5)
                gy.append(j + 0.5)
            else:
                gx.append(i + f + 0.5)
                gy.append(j + f + 0.5)

    xs.append(np.array(gx, dtype=np.float64))
    ys.append(np.array(gy, dtype=np.float64))
    ths.append(np.zeros(len(gx), dtype=np.float64))
    ax = np.concatenate(xs)
    ay = np.concatenate(ys)
    at = np.concatenate(ths)
    return ax, ay, at, L


def build_pinwheel_seam_seeds(
    n: int,
    x_base: np.ndarray,
    y_base: np.ndarray,
    th_base: np.ndarray,
    L_base: float,
    rng: np.random.Generator,
    L_expand: float = 0.025,
) -> tuple[list[np.ndarray], list[np.ndarray], list[list[float]], list[str]]:
    """Build masked-melt seeds targeting pinwheel seam corners, slack-sharing shifts, and Idea-6 pockets."""
    L0 = L_base + L_expand
    sc = L0 / L_base
    xb, yb, tb = x_base * sc, y_base * sc, th_base.copy()
    sched = [L0, L_base + 0.006, L_base - 3e-4, L_base - 0.0025]

    q_list, m_list, ls_list, labels = [], [], [], []

    # 1. Include 10 Idea-6 seeds (5 void + 5 hot)
    q_h13, m_h13, _, lb_h13 = ps.build_h13_seeds(n, x_base, y_base, th_base, L_base, rng, L_expand=L_expand)
    for idx in list(range(0, 5)) + list(range(10, 15)):
        q_list.append(q_h13[idx])
        m_list.append(m_h13[idx])
        ls_list.append(sched)
        labels.append(lb_h13[idx])

    # 2. Detect tilted cluster rectangles & seam points in the packing
    a = np.column_stack([xb, yb, tb])
    e = 0.5 * (np.abs(np.cos(tb)) + np.abs(np.sin(tb)))
    t_deg = np.degrees(np.mod(tb, np.pi / 2))
    tilt = np.minimum(t_deg, 90.0 - t_deg) > 2.0
    cls = gadlib.clusters(a, tilt, link=1.42) if np.any(tilt) else []

    seam_pts = []
    rects = []
    for cl in cls:
        x0, x1 = float((a[cl, 0] - e[cl]).min()), float((a[cl, 0] + e[cl]).max())
        y0, y1 = float((a[cl, 1] - e[cl]).min()), float((a[cl, 1] + e[cl]).max())
        rects.append((x0, x1, y0, y1))
        # Inner seam corners & edge midpoints of each tilted cluster
        for cx, cy in [
            (x0, y0), (x0, y1), (x1, y0), (x1, y1),
            (0.5 * (x0 + x1), y0), (0.5 * (x0 + x1), y1),
            (x0, 0.5 * (y0 + y1)), (x1, 0.5 * (y0 + y1)),
            (0.5 * (x0 + x1), 0.5 * (y0 + y1)),
        ]:
            if 1.0 < cx < L0 - 1.0 or 1.0 < cy < L0 - 1.0:
                seam_pts.append((cx, cy))

    if not seam_pts:
        seam_pts = [(0.5 * L0, 0.5 * L0), (0.3 * L0, 0.7 * L0), (0.7 * L0, 0.3 * L0)]

    # 3. 12 pinwheel seam-corner & brick-edge melt seeds
    for s in range(12):
        x, y, th = xb.copy(), yb.copy(), tb.copy()
        cx, cy = seam_pts[s % len(seam_pts)]
        rad = float(rng.uniform(4.2, 7.8))
        near = ((x - cx) ** 2 + (y - cy) ** 2) <= rad * rad
        nm = int(near.sum())
        if nm > 0:
            x[near] += rng.uniform(-0.13, 0.13, nm)
            y[near] += rng.uniform(-0.13, 0.13, nm)
            th[near] += rng.uniform(-0.24, 0.24, nm)
        q_list.append(full_to_free_sym(x, y, th, L0))
        m_list.append(np.concatenate([near.astype(np.uint8)] * 3))
        ls_list.append(sched)
        labels.append(f"pw_seam_{s}")

    # 4. 6 asymmetric slack-sharing seeds: melt whole seam + shift central grid away from tilted brick
    for s in range(6):
        x, y, th = xb.copy(), yb.copy(), tb.copy()
        if rects:
            rx0, rx1, ry0, ry1 = rects[s % len(rects)]
            # Target the central band adjacent to this brick
            cx = rx1 if rx0 < 0.3 * L0 else rx0
            cy = ry1 if ry0 < 0.3 * L0 else ry0
            rad = float(rng.uniform(6.0, 9.5))
            near = ((x - cx) ** 2 + (y - cy) ** 2) <= rad * rad
            # Directional nudge away from brick center toward opposite corner
            bx, by = 0.5 * (rx0 + rx1), 0.5 * (ry0 + ry1)
            sx_dir = 1.0 if bx < 0.5 * L0 else -1.0
            sy_dir = 1.0 if by < 0.5 * L0 else -1.0
            nm = int(near.sum())
            if nm > 0:
                in_b = near & (x >= rx0 - 0.1) & (x <= rx1 + 0.1) & (y >= ry0 - 0.1) & (y <= ry1 + 0.1)
                out_b = near & (~in_b)
                if np.any(out_b):
                    x[out_b] += sx_dir * float(rng.uniform(0.015, 0.045))
                    y[out_b] += sy_dir * float(rng.uniform(0.015, 0.045))
                x[near] += rng.uniform(-0.11, 0.11, nm)
                y[near] += rng.uniform(-0.11, 0.11, nm)
                th[near] += rng.uniform(-0.20, 0.20, nm)
        else:
            near = np.ones(n, dtype=bool)
        q_list.append(full_to_free_sym(x, y, th, L0))
        m_list.append(np.concatenate([near.astype(np.uint8)] * 3))
        ls_list.append(sched)
        labels.append(f"pw_slack_{s}")

    return q_list, m_list, ls_list, labels
