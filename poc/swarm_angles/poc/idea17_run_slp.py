"""Idea 17: Un-grid-locked continuous (x, y, theta) optimization with Joint (x, y, dtheta) SLP.

Combines continuous L-BFGS penalty relaxation (utils.relax_lbfgs) with a first-order
Sequential Linear Program (SLP) solved via HiGHS that jointly optimizes (dx_i, dy_i, dtheta_i, L)
using exact smooth corner-to-wall and corner-to-face SAT linearizations.

Tests 4 continuum families across n in {401, 442, 530, 626, 842}:
  1. surg: Direct skewed-band neighbor surgery from merged.csv (n=403->401, 444->442, 532->530, 629->626, 845/843->842)
  2. skew: Parametric w=2 and w=3 skewed diagonal bands (theta in [22..34.5 deg], dv = tan(45 deg - theta))
  3. pin4: Smooth 4-domain chiral pinwheel continuum + central tilted pocket
  4. pocket: Continuous corner-pocket & block-rotation deformations of n=401, 442, 530, 626, 842 incumbents
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import linprog
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

    # 4 corners per square: (sigma1, sigma2) in {(-1,-1), (-1,+1), (+1,-1), (+1,+1)}
    s1 = np.array([-0.5, -0.5, 0.5, 0.5], dtype=np.float64)
    s2 = np.array([-0.5, 0.5, -0.5, 0.5], dtype=np.float64)
    # shape (n, 4)
    dx_c = c[:, None] * s1[None, :] - s[:, None] * s2[None, :]
    dy_c = s[:, None] * s1[None, :] + c[:, None] * s2[None, :]
    X_c = x[:, None] + dx_c
    Y_c = y[:, None] + dy_c

    rows_list = []
    cols_list = []
    vals_list = []
    b_list = []
    row_counter = 0

    # Curvature safety pad for angle step: 0.5 * r * dth^2
    curv_pad = 0.25 * (tr_th * tr_th)

    # 1. Wall constraints: for each square i, take the 2 lowest-X, 2 highest-X, 2 lowest-Y, 2 highest-Y corners
    idx_n = np.arange(n, dtype=np.int32)
    order_x = np.argsort(X_c, axis=1)
    order_y = np.argsort(Y_c, axis=1)

    for k_slot in range(2):
        # Left wall: X_{i,c} + dx_i - dy_{i,c} dth_i >= curv_pad  <=>  -dx_i + dy_{i,c} dth_i <= X_{i,c} - curv_pad
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

        # Right wall: X_{i,c} + dx_i - dy_{i,c} dth_i - L <= -curv_pad
        c_max_x = order_x[:, 3 - k_slot]
        x_val = X_c[idx_n, c_max_x]
        dy_val = dy_c[idx_n, c_max_x]
        L_est = float(np.max(X_c))
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

        # Bottom wall: Y_{i,c} + dy_i + dx_{i,c} dth_i >= curv_pad <=> -dy_i - dx_{i,c} dth_i <= Y_{i,c} - curv_pad
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

        # Top wall: Y_{i,c} + dy_i + dx_{i,c} dth_i - L <= -curv_pad
        c_max_y = order_y[:, 3 - k_slot]
        y_val = Y_c[idx_n, c_max_y]
        dx_val = dx_c[idx_n, c_max_y]
        L_est_y = float(np.max(Y_c))
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

    # 2. Pairwise SAT separation constraints for pairs within 1.82
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

        # 4 candidate normals: a=0 (ci, si), a=1 (-si, ci), a=2 (cj, sj), a=3 (-sj, cj)
        nx_all = np.stack([ci, -si, cj, -sj], axis=1)
        ny_all = np.stack([si, ci, sj, cj], axis=1)

        proj = dxi[:, None] * nx_all + dyi[:, None] * ny_all
        s_dir = np.where(proj >= 0.0, 1.0, -1.0)
        abs_proj = np.abs(proj)

        cth = ci * cj + si * sj
        sth = ci * sj - si * cj
        rad_other = 0.5 * (np.abs(cth) + np.abs(sth))
        gaps = abs_proj - (0.5 + rad_other[:, None])

        # Best separating axis for each pair
        best_a = np.argmax(gaps, axis=1)
        # If best_a in {0, 1}, p = near_i (face owner), q = near_j (corner owner)
        # If best_a in {2, 3}, p = near_j (face owner), q = near_i (corner owner), and flip direction
        i_owns = best_a < 2
        p_idx = np.where(i_owns, near_i, near_j)
        q_idx = np.where(i_owns, near_j, near_i)

        pair_arange = np.arange(m_pairs)
        ux = nx_all[pair_arange, best_a] * s_dir[pair_arange, best_a] * np.where(i_owns, 1.0, -1.0)
        uy = ny_all[pair_arange, best_a] * s_dir[pair_arange, best_a] * np.where(i_owns, 1.0, -1.0)

        # Corners of q: shape (m_pairs, 4)
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

            # Coeffs: ux dx_p + uy dy_p + (dX uy - dY ux) dth_p - ux dx_q - uy dy_q + (dy_q ux - dx_q uy) dth_q <= f - margin
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
    k = int(round(math.sqrt(n - 1)))
    cur_x = x.copy()
    cur_y = y.copy()
    cur_th = th.copy()

    pen0, bbox0 = u.sat_max_overlap_and_bbox(cur_x, cur_y, cur_th)
    if pen0 > 1e-7:
        cur_x, cur_y, cur_th, pen0 = u.relax_lbfgs(cur_x, cur_y, cur_th, max(bbox0, k + 0.50), maxiter=220)
        pen0, bbox0 = u.sat_max_overlap_and_bbox(cur_x, cur_y, cur_th)

    best_x, best_y, best_th = cur_x.copy(), cur_y.copy(), cur_th.copy()
    best_L = bbox0 if pen0 <= 1e-7 else 1e9
    best_ok = pen0 <= 1e-7

    obj = np.zeros(3 * n + 1, dtype=np.float64)
    obj[3 * n] = 1.0

    tr_th = tr_th0
    tr_xy = tr_xy0

    for r in range(rounds):
        A_ub, b_ub = _build_joint_slp_sparse(cur_x, cur_y, cur_th, tr_xy=tr_xy, tr_th=tr_th, margin=margin)
        bounds = (
            [(-tr_xy, tr_xy)] * (2 * n)
            + [(-tr_th, tr_th)] * n
            + [(float(k) + 0.45, float(max(bbox0, k + 0.65) + 0.25))]
        )
        res = linprog(obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
        if not res.success or res.x is None:
            tr_th *= 0.6
            tr_xy *= 0.7
            continue

        dx = res.x[:n]
        dy = res.x[n : 2 * n]
        dth = res.x[2 * n : 3 * n]
        lp_L = float(res.x[3 * n])

        # Line search / trust-region acceptance with short L-BFGS repair
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

            # Accept iterate if penalty is small or improved
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

    # Final exact certification polish via L-BFGS + fixed_angle_lp_tighten on both cur and best
    for cx, cy, cth in [(best_x, best_y, best_th), (cur_x, cur_y, cur_th)]:
        pen_f, bbox_f = u.sat_max_overlap_and_bbox(cx, cy, cth)
        if pen_f > 1e-9:
            cx, cy, cth, pen_f = u.relax_lbfgs(cx, cy, cth, bbox_f, maxiter=250, w_wall=2.0, w_pair=1.0)
        tx, ty, tL, tok = u.fixed_angle_lp_tighten(cx, cy, cth, rounds=4, margin=1e-8)
        if tok and tL < best_L:
            best_x, best_y, best_th, best_L, best_ok = tx, ty, cth.copy(), tL, True

    return best_x, best_y, best_th, best_L, best_ok


def generate_family1_surg_seeds(n: int, df_merged: pd.DataFrame) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    """Family 1: Direct surgery on skewed-band neighbors in merged.csv."""
    k = int(round(math.sqrt(n - 1)))
    L_target = k + 0.520
    seeds = []

    neighbor_map = {
        401: [403, 404],
        442: [444, 443],
        530: [532, 533],
        626: [629, 628],
        842: [845, 843],
    }
    for n_src in neighbor_map.get(n, []):
        sub = df_merged[df_merged.n == n_src]
        if len(sub) != n_src:
            continue
        x0 = sub.x.values.copy()
        y0 = sub.y.values.copy()
        th0 = sub.angle.values.copy()
        # Shift into [0, L]^2 if centered at 0
        c0, s0 = np.cos(th0), np.sin(th0)
        hx = 0.5 * (np.abs(c0) + np.abs(s0))
        x0 = x0 - float(np.min(x0 - hx))
        y0 = y0 - float(np.min(y0 - hx))
        L_src = float(max(np.max(x0 + hx), np.max(y0 + hx)))

        dn = n_src - n
        deg = np.mod(np.degrees(th0), 90.0)
        deg_f = np.minimum(deg, 90.0 - deg)
        tilted_idx = np.where(deg_f > 3.0)[0]
        up_idx = np.where(deg_f <= 3.0)[0]

        if len(tilted_idx) >= dn:
            # Sort tilted squares along diagonal u = x + y
            u_coord = x0[tilted_idx] + y0[tilted_idx]
            order_u = np.argsort(u_coord)
            # Mode 1A: remove dn squares from high-u end, low-u end, or split
            rem_patterns = [("hi", order_u[-dn:]), ("lo", order_u[:dn])]
            if dn >= 2:
                rem_patterns.append((" ends", np.concatenate([order_u[:1], order_u[-(dn - 1) :]])))

            for pat_name, rem_local in rem_patterns:
                rem_global = tilted_idx[rem_local]
                keep = np.setdiff1d(np.arange(n_src), rem_global)
                for dth_deg in (0.0, -2.5, -5.0, -7.5):
                    xk = x0[keep].copy() * (L_target / L_src)
                    yk = y0[keep].copy() * (L_target / L_src)
                    thk = th0[keep].copy()
                    k_deg = np.mod(np.degrees(thk), 90.0)
                    k_tilted = np.minimum(k_deg, 90.0 - k_deg) > 3.0
                    thk[k_tilted] += math.radians(dth_deg)
                    seeds.append((f"surg_{n_src}_{pat_name}_dth{dth_deg:.1f}", xk, yk, thk))

        # Mode 1B: for dn == 2 and w=3 band, remove 3 end-slice tilted squares and insert 1 upright square
        if dn == 2 and len(tilted_idx) >= 3:
            u_coord = x0[tilted_idx] + y0[tilted_idx]
            order_u = np.argsort(u_coord)
            for end_tag, rem_local in [("slice_hi", order_u[-3:]), ("slice_lo", order_u[:3])]:
                rem_global = tilted_idx[rem_local]
                keep = np.setdiff1d(np.arange(n_src), rem_global)
                # Place 1 upright replacement at the centroid of the 3 removed squares
                rx = float(np.mean(x0[rem_global]))
                ry = float(np.mean(y0[rem_global]))
                xk = np.append(x0[keep], rx) * (L_target / L_src)
                yk = np.append(y0[keep], ry) * (L_target / L_src)
                thk = np.append(th0[keep], 0.0)
                for dth_deg in (-2.0, -5.0):
                    th_mod = thk.copy()
                    k_deg = np.mod(np.degrees(th_mod), 90.0)
                    k_tilted = np.minimum(k_deg, 90.0 - k_deg) > 3.0
                    th_mod[k_tilted] += math.radians(dth_deg)
                    seeds.append((f"surg_{n_src}_{end_tag}_swap_dth{dth_deg:.1f}", xk.copy(), yk.copy(), th_mod))

    return seeds


def generate_family2_skew_seeds(n: int) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    """Family 2: Parametric w=2 and w=3 skewed diagonal bands with dv = tan(45 - theta) and continuous staircases."""
    k = int(round(math.sqrt(n - 1)))
    L_init = k + 0.525
    seeds = []

    for w in (2, 3):
        for th_deg in (22.0, 25.2, 28.5, 31.5, 34.0):
            th_rad = math.radians(th_deg)
            delta = math.radians(45.0 - th_deg)
            tan_d = math.tan(delta)
            c_th, s_th = math.cos(th_rad), math.sin(th_rad)
            hx_t = 0.5 * (abs(c_th) + abs(s_th))

            for v_off in (-0.6, 0.0):
                # Build skewed band slices along 45 deg diagonal shifted by v_off
                bx, by, bth = [], [], []
                cx0 = L_init * 0.5 - v_off / math.sqrt(2.0)
                cy0 = L_init * 0.5 + v_off / math.sqrt(2.0)
                for s_idx in range(-k - 4, k + 5):
                    for m_idx in range(w):
                        m_centered = m_idx - 0.5 * (w - 1)
                        e1 = float(s_idx) + 0.04 * m_centered
                        e2 = float(m_centered) + float(s_idx) * tan_d
                        px = cx0 + e1 * c_th - e2 * s_th
                        py = cy0 + e1 * s_th + e2 * c_th
                        if hx_t + 0.02 <= px <= L_init - hx_t - 0.02 and hx_t + 0.02 <= py <= L_init - hx_t - 0.02:
                            bx.append(px)
                            by.append(py)
                            bth.append(th_rad)

                nt = len(bx)
                if nt < k or nt > n - 10:
                    continue
                n_rem = n - nt

                # Generate candidate NW (top-left) and SE (bottom-right) upright staircase sites
                # and slide the innermost square of each row/col continuously toward the band
                nw_sites = []
                se_sites = []
                for r in range(k):
                    for c_col in range(k):
                        # NW candidate flush with top-left walls: (0.5 + c_col, L_init - 0.5 - r)
                        px_nw = 0.5 + c_col
                        py_nw = L_init - 0.5 - r
                        # Signed transverse coordinate relative to band center
                        v_nw = (py_nw - cy0) * math.cos(math.radians(45.0)) - (px_nw - cx0) * math.sin(math.radians(45.0))
                        if v_nw > 0.5 * w * math.cos(delta) + 0.15:
                            nw_sites.append(( -v_nw, px_nw, py_nw ))

                        # SE candidate flush with bottom-right walls: (L_init - 0.5 - c_col, 0.5 + r)
                        px_se = L_init - 0.5 - c_col
                        py_se = 0.5 + r
                        v_se = (py_se - cy0) * math.cos(math.radians(45.0)) - (px_se - cx0) * math.sin(math.radians(45.0))
                        if v_se < -(0.5 * w * math.cos(delta) + 0.15):
                            se_sites.append(( v_se, px_se, py_se ))

                nw_sites.sort(key=lambda t: t[0])
                se_sites.sort(key=lambda t: t[0])
                combined = []
                max_len = max(len(nw_sites), len(se_sites))
                for idx in range(max_len):
                    if idx < len(nw_sites):
                        combined.append((nw_sites[idx][1], nw_sites[idx][2]))
                    if idx < len(se_sites):
                        combined.append((se_sites[idx][1], se_sites[idx][2]))

                if len(combined) < n_rem:
                    continue

                sel_up = combined[:n_rem]
                ux = [p[0] for p in sel_up]
                uy = [p[1] for p in sel_up]
                uth = [0.0] * n_rem

                x_all = np.array(bx + ux, dtype=np.float64)
                y_all = np.array(by + uy, dtype=np.float64)
                th_all = np.array(bth + uth, dtype=np.float64)
                seeds.append((f"skew_w{w}_th{th_deg:.1f}_v{v_off:.1f}", x_all, y_all, th_all))

    return seeds


def generate_family3_pin4_seeds(n: int) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    """Family 3: Smooth 4-domain chiral pinwheel continuum + central tilted pocket."""
    k = int(round(math.sqrt(n - 1)))
    L_init = k + 0.525
    seeds = []

    for m_core in (3, 5):
        n_core = m_core * m_core
        for th_core_deg in (28.0, 36.0, 45.0):
            th_c = math.radians(th_core_deg)
            cc, sc = math.cos(th_c), math.sin(th_c)
            cx0, cy0 = 0.5 * L_init, 0.5 * L_init

            bx, by, bth = [], [], []
            for i in range(m_core):
                for j in range(m_core):
                    u_loc = i - 0.5 * (m_core - 1)
                    v_loc = j - 0.5 * (m_core - 1)
                    bx.append(cx0 + u_loc * cc - v_loc * sc)
                    by.append(cy0 + u_loc * sc + v_loc * cc)
                    bth.append(th_c)

            for th_dom_deg in (0.8, 2.0, 3.8):
                th_d = math.radians(th_dom_deg)
                cd, sd = math.cos(th_d), math.sin(th_d)
                hx_d = 0.5 * (abs(cd) + abs(sd))

                # Build 4 chiral outer quadrants rotated by th_d around their outer corners
                cand_outer = []
                half_k = k // 2 + 2
                for q_id, (ox, oy, ux_q, uy_q, vx_q, vy_q, q_th) in enumerate([
                    (hx_d, hx_d, cd, sd, -sd, cd, th_d),
                    (L_init - hx_d, hx_d, -sd, cd, -cd, -sd, th_d),
                    (L_init - hx_d, L_init - hx_d, -cd, -sd, sd, -cd, th_d),
                    (hx_d, L_init - hx_d, sd, -cd, cd, sd, th_d),
                ]):
                    for i in range(half_k):
                        for j in range(half_k):
                            px = ox + i * ux_q + j * vx_q
                            py = oy + i * uy_q + j * vy_q
                            if hx_d <= px <= L_init - hx_d and hx_d <= py <= L_init - hx_d:
                                dist_center = max(abs(px - cx0), abs(py - cy0))
                                cand_outer.append((-dist_center, px, py, q_th))

                cand_outer.sort(key=lambda t: t[0])
                # Deduplicate close candidates
                kept_outer = []
                for _, px, py, q_th in cand_outer:
                    if all(abs(px - kx) > 0.75 or abs(py - ky) > 0.75 for kx, ky, _ in kept_outer):
                        kept_outer.append((px, py, q_th))
                    if len(kept_outer) >= n - n_core:
                        break

                if len(kept_outer) == n - n_core:
                    x_all = np.array(bx + [p[0] for p in kept_outer], dtype=np.float64)
                    y_all = np.array(by + [p[1] for p in kept_outer], dtype=np.float64)
                    th_all = np.array(bth + [p[2] for p in kept_outer], dtype=np.float64)
                    seeds.append((f"pin4_c{m_core}_tc{th_core_deg:.0f}_td{th_dom_deg:.1f}", x_all, y_all, th_all))

    return seeds


def generate_family4_pocket_seeds(n: int, df_merged: pd.DataFrame) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    """Family 4: Continuous corner-pocket & block-rotation deformations of n incumbent."""
    sub = df_merged[df_merged.n == n]
    if len(sub) != n:
        return []
    x0 = sub.x.values.copy()
    y0 = sub.y.values.copy()
    th0 = sub.angle.values.copy()
    c0, s0 = np.cos(th0), np.sin(th0)
    hx = 0.5 * (np.abs(c0) + np.abs(s0))
    x0 = x0 - float(np.min(x0 - hx))
    y0 = y0 - float(np.min(y0 - hx))
    L0 = float(max(np.max(x0 + hx), np.max(y0 + hx)))

    deg = np.mod(np.degrees(th0), 90.0)
    deg_f = np.minimum(deg, 90.0 - deg)
    tilted_mask = deg_f > 10.0

    seeds = []
    # Mode 4A: Rotate the 45-deg block as a rigid/semi-rigid pocket by dth_block and shear surrounding uprights
    if np.any(tilted_mask):
        cx_t = float(np.mean(x0[tilted_mask]))
        cy_t = float(np.mean(y0[tilted_mask]))
        for dth_deg in (-6.0, -3.5, -1.5, -0.6, 0.6, 1.5, 3.5):
            dth = math.radians(dth_deg)
            cd, sd = math.cos(dth), math.sin(dth)
            xk = x0.copy()
            yk = y0.copy()
            thk = th0.copy()
            dx_t = xk[tilted_mask] - cx_t
            dy_t = yk[tilted_mask] - cy_t
            xk[tilted_mask] = cx_t + dx_t * cd - dy_t * sd
            yk[tilted_mask] = cy_t + dx_t * sd + dy_t * cd
            thk[tilted_mask] += dth
            seeds.append((f"pocket_rot_{dth_deg:+.1f}", xk, yk, thk))

    # Mode 4B: Smooth corner-pocket bridge tilt on upright squares adjacent to the tilted block
    if np.any(tilted_mask):
        tx_min, tx_max = float(np.min(x0[tilted_mask])), float(np.max(x0[tilted_mask]))
        ty_min, ty_max = float(np.min(y0[tilted_mask])), float(np.max(y0[tilted_mask]))
        bridge_mask = (~tilted_mask) & (
            (np.abs(x0 - tx_max) < 3.5) & (np.abs(y0 - ty_max) < 3.5)
            | (np.abs(x0 - tx_min) < 3.5) & (np.abs(y0 - ty_min) < 3.5)
        )
        for b_deg in (-3.0, -1.2, 1.2, 3.0, 5.5):
            xk = x0.copy() * ((L0 - 0.008) / L0)
            yk = y0.copy() * ((L0 - 0.008) / L0)
            thk = th0.copy()
            thk[bridge_mask] += math.radians(b_deg)
            seeds.append((f"pocket_bridge_{b_deg:+.1f}", xk, yk, thk))

    return seeds


def _eval_seed_worker(args: tuple) -> dict:
    """Worker to run continuous L-BFGS + joint (x, y, dtheta) SLP on a single seed."""
    n, fam, name, x0, y0, th0, L_base = args
    t0 = time.time()
    k = int(round(math.sqrt(n - 1)))

    # Initial multi-stage L-BFGS warm relaxation at L_warm -> L_target
    cur_x, cur_y, cur_th = x0.copy(), y0.copy(), th0.copy()
    for L_stage, maxit in [(k + 0.56, 180), (k + 0.53, 220)]:
        cur_x, cur_y, cur_th, pen = u.relax_lbfgs(cur_x, cur_y, cur_th, L_stage, maxiter=maxit, w_wall=1.5, w_pair=1.0)

    # Joint (x, y, dtheta) SLP optimization
    best_x, best_y, best_th, best_L, ok = joint_xytheta_slp(
        cur_x, cur_y, cur_th, rounds=14, tr_th0=0.025, tr_xy0=0.12, margin=1e-8
    )

    # If close to feasible or promising, run a fine continuation pass
    pen_now, bbox_now = u.sat_max_overlap_and_bbox(best_x, best_y, best_th)
    if ok and best_L < L_base + 0.015:
        for step in range(4):
            target_L = min(best_L, L_base) - 0.0015
            rx, ry, rth, rpen = u.relax_lbfgs(best_x, best_y, best_th, target_L, maxiter=200, w_wall=2.0, w_pair=1.0)
            sx, sy, sth, sL, sok = joint_xytheta_slp(rx, ry, rth, rounds=8, tr_th0=0.008, tr_xy0=0.04, margin=1e-8)
            if sok and sL < best_L:
                best_x, best_y, best_th, best_L, ok = sx, sy, sth, sL, True

    pen_final, bbox_final = u.sat_max_overlap_and_bbox(best_x, best_y, best_th)
    deg = np.mod(np.degrees(best_th), 90.0)
    deg_f = np.minimum(deg, 90.0 - deg)
    n_non45 = int(np.sum((deg_f > 0.5) & (np.abs(deg_f - 45.0) > 0.5)))

    res = {
        "n": n,
        "fam": fam,
        "name": name,
        "ok": bool(ok and pen_final <= 1e-7),
        "L": float(best_L if ok else bbox_final),
        "dL": float((best_L if ok else bbox_final) - L_base),
        "pen": float(pen_final),
        "n_non45": n_non45,
        "elapsed": round(time.time() - t0, 2),
    }
    if ok and best_L < L_base - 1e-6:
        ver = u.verify_official(n, best_x, best_y, best_th)
        res["verified"] = bool(ver["ok"])
        res["L_lit"] = float(ver["L_lit"])
        res["L_pandas"] = float(ver["L_pandas"])
        if ver["ok"] and max(ver["L_lit"], ver["L_pandas"]) < L_base - 1e-6:
            res["win_x"] = best_x.tolist()
            res["win_y"] = best_y.tolist()
            res["win_th"] = best_th.tolist()
    return res


def main() -> None:
    t_start = time.time()
    df_merged = pd.read_csv("poc/pell_corrugation/data/border-scan/merged.csv")
    targets = [401, 442, 530, 626, 842]

    tasks = []
    inc_L_map = {}
    for n in targets:
        sub = df_merged[df_merged.n == n]
        _, L_base = u.sat_max_overlap_and_bbox(sub.x.values, sub.y.values, sub.angle.values)
        inc_L_map[n] = float(L_base)

        for name, x0, y0, th0 in generate_family1_surg_seeds(n, df_merged):
            tasks.append((n, "surg", name, x0, y0, th0, L_base))
        for name, x0, y0, th0 in generate_family2_skew_seeds(n):
            tasks.append((n, "skew", name, x0, y0, th0, L_base))
        for name, x0, y0, th0 in generate_family3_pin4_seeds(n):
            tasks.append((n, "pin4", name, x0, y0, th0, L_base))
        for name, x0, y0, th0 in generate_family4_pocket_seeds(n, df_merged):
            tasks.append((n, "pocket", name, x0, y0, th0, L_base))

    print(f"Idea 17: launching {len(tasks)} continuous (x, y, dtheta) SLP tasks across {targets} on 7 workers...")
    results = []
    wins = []

    with ProcessPoolExecutor(max_workers=7) as ex:
        futs = [ex.submit(_eval_seed_worker, t) for t in tasks]
        for idx, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            results.append(r)
            if r.get("verified") and "win_x" in r:
                wins.append(r)
                print(f"*** WIN on n={r['n']} ({r['name']}): L_lit={r['L_lit']:.6f} vs {inc_L_map[r['n']]:.6f} ***")
            if idx % 20 == 0 or idx == len(tasks):
                print(f"  [{idx}/{len(tasks)}] elapsed={time.time()-t_start:.1f}s")

    summary = {"elapsed_s": round(time.time() - t_start, 2), "total_tasks": len(tasks), "wins": len(wins), "per_n": {}}
    for n in targets:
        n_res = [r for r in results if r["n"] == n]
        feas = [r for r in n_res if r["ok"]]
        feas.sort(key=lambda r: r["L"])
        by_fam = {}
        for fam in ("surg", "skew", "pin4", "pocket"):
            f_all = [r for r in n_res if r["fam"] == fam]
            f_ok = [r for r in feas if r["fam"] == fam]
            by_fam[fam] = {
                "total": len(f_all),
                "feasible": len(f_ok),
                "best_L": round(f_ok[0]["L"], 6) if f_ok else None,
                "best_dL": round(f_ok[0]["dL"], 6) if f_ok else None,
                "best_name": f_ok[0]["name"] if f_ok else (f_all[0]["name"] if f_all else None),
                "min_pen": round(min((r["pen"] for r in f_all), default=99.0), 6),
            }
        summary["per_n"][str(n)] = {
            "L_base": round(inc_L_map[n], 6),
            "total": len(n_res),
            "feasible": len(feas),
            "best_L": round(feas[0]["L"], 6) if feas else None,
            "best_dL": round(feas[0]["dL"], 6) if feas else None,
            "best_name": feas[0]["name"] if feas else None,
            "families": by_fam,
            "top5": [
                {k: v for k, v in r.items() if not k.startswith("win_")}
                for r in (feas[:5] if feas else sorted(n_res, key=lambda x: x["pen"])[:5])
            ],
        }
        print(
            f"n={n} (L_base={inc_L_map[n]:.6f}): feas={len(feas)}/{len(n_res)}, "
            f"best={summary['per_n'][str(n)]['best_L']} (dL={summary['per_n'][str(n)]['best_dL']}, {summary['per_n'][str(n)]['best_name']})"
        )

    out_path = Path("poc/swarm_angles/docs/idea17_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved {out_path} in {summary['elapsed_s']}s")


if __name__ == "__main__":
    main()
