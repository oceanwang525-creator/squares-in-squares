"""Joint (x, y, dtheta, L) Sequential Linear Program (HiGHS, threads=1) for free-angle compression."""
from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np
from scipy.sparse import csc_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
import utils as u


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
    L_est = float(np.max(X_c))
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
        mask = x_val > L_est - 2.5
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

    A = csc_matrix((np.concatenate(vals_list), (np.concatenate(rows_list), np.concatenate(cols_list))), shape=(row_counter, 3 * n + 1))
    return A, np.concatenate(b_list)


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
    k = math.isqrt(n)
    cur_x = x.copy()
    cur_y = y.copy()
    cur_th = th.copy()

    pen0, bbox0 = u.sat_max_overlap_and_bbox(cur_x, cur_y, cur_th)
    if pen0 > 1e-7:
        cur_x, cur_y, cur_th, pen0 = u.relax_lbfgs(cur_x, cur_y, cur_th, max(bbox0, k + 0.70), maxiter=220)
        pen0, bbox0 = u.sat_max_overlap_and_bbox(cur_x, cur_y, cur_th)

    best_x, best_y, best_th = cur_x.copy(), cur_y.copy(), cur_th.copy()
    best_L = bbox0 if pen0 <= 1e-7 else 1e9
    best_ok = pen0 <= 1e-7

    obj = np.zeros(3 * n + 1, dtype=np.float64)
    obj[3 * n] = 1.0

    tr_th = tr_th0
    tr_xy = tr_xy0
    L_hi = float(max(bbox0, k + 1.0) + 0.25)

    for _ in range(rounds):
        A_ub, b_ub = _build_joint_slp_sparse(cur_x, cur_y, cur_th, tr_xy=tr_xy, tr_th=tr_th, margin=margin)
        bounds = (
            [(-tr_xy, tr_xy)] * (2 * n)
            + [(-tr_th, tr_th)] * n
            + [(float(k) + 0.50, L_hi)]
        )
        res = u._highs_linprog(obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds)
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
            pen_c, bbox_c = u.sat_max_overlap_and_bbox(cand_x, cand_y, cand_th)
            if pen_c > 1e-8:
                target_L = max(lp_L, min(bbox_c, bbox0) - 0.002)
                cand_x, cand_y, cand_th, pen_c = u.relax_lbfgs(
                    cand_x, cand_y, cand_th, target_L, maxiter=90, w_wall=2.0, w_pair=1.0
                )
                pen_c, bbox_c = u.sat_max_overlap_and_bbox(cand_x, cand_y, cand_th)

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
        pen_f, bbox_f = u.sat_max_overlap_and_bbox(cx, cy, cth)
        if pen_f > 1e-9:
            cx, cy, cth, pen_f = u.relax_lbfgs(cx, cy, cth, bbox_f, maxiter=250, w_wall=2.0, w_pair=1.0)
        tx, ty, tL, tok = u.fixed_angle_lp_tighten(cx, cy, cth, rounds=4, margin=1e-8)
        if tok and tL < best_L:
            best_x, best_y, best_th, best_L, best_ok = tx, ty, cth.copy(), tL, True

    return best_x, best_y, best_th, best_L, best_ok
