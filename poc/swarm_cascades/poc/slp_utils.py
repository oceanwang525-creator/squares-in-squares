"""Shared utilities for swarm_cascades: C++ SAT melt bindings, Joint (x, y, dtheta) SLP,
fixed-angle LP tighten, official certification, and final.csv management.
"""
from __future__ import annotations

import ctypes
import csv
import io
import math
import os
import re
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, csc_matrix

REPO_ROOT = Path(__file__).resolve().parents[3]
BASELINE_SRC = REPO_ROOT / "poc" / "baseline" / "poc" / "src"
if str(BASELINE_SRC) not in sys.path:
    sys.path.insert(0, str(BASELINE_SRC))

import squares.core as sc  # noqa: E402

CPP_PATH = Path(__file__).resolve().parent / "cascades_cpp.cpp"
SO_PATH = Path("/tmp/libcascades.so")
MERGED_CSV = REPO_ROOT / "poc" / "pell_corrugation" / "data" / "border-scan" / "merged.csv"
PER_N_GAP_CSV = REPO_ROOT / "poc" / "frontier_nov12" / "docs" / "per_n_gap.csv"
FINAL_CSV = Path(__file__).resolve().parents[1] / "final.csv"


def load_cpp_lib():
    """Compile (if needed) and load /tmp/libcascades.so."""
    if (not SO_PATH.exists()) or (CPP_PATH.stat().st_mtime > SO_PATH.stat().st_mtime):
        cmd = [
            "g++",
            "-O3",
            "-march=native",
            "-shared",
            "-fPIC",
            "-std=c++17",
            "-pthread",
            str(CPP_PATH),
            "-o",
            str(SO_PATH),
        ]
        subprocess.run(cmd, check=True)
    lib = ctypes.CDLL(str(SO_PATH))

    c_dp = ctypes.POINTER(ctypes.c_double)
    c_u8p = ctypes.POINTER(ctypes.c_uint8)

    lib.eval_sym_sat_energy_grad.argtypes = [
        ctypes.c_int, ctypes.c_int, c_dp,
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, c_dp,
    ]
    lib.eval_sym_sat_energy_grad.restype = ctypes.c_double

    lib.expand_sym_coords.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_double, c_dp, c_dp,
    ]
    lib.expand_sym_coords.restype = None

    lib.eval_sat_pen_bbox.argtypes = [
        ctypes.c_int, c_dp, c_dp, c_dp, ctypes.c_double, c_dp,
    ]
    lib.eval_sat_pen_bbox.restype = ctypes.c_double

    lib.batch_melt_solve.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        c_dp, c_dp, c_u8p, ctypes.c_int,
        ctypes.c_int, c_dp,
        ctypes.c_double, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_uint64,
        c_dp, c_dp, c_dp, c_dp,
    ]
    lib.batch_melt_solve.restype = None
    return lib


_CPP = None


def cpp():
    global _CPP
    if _CPP is None:
        _CPP = load_cpp_lib()
    return _CPP


def sat_max_overlap_and_bbox(
    x: np.ndarray, y: np.ndarray, th: np.ndarray, side: float = 1.0
) -> tuple[float, float]:
    """Fast C++ spatial-hash SAT max penetration and axis-aligned bounding box L."""
    n = len(x)
    xa = np.ascontiguousarray(x, dtype=np.float64)
    ya = np.ascontiguousarray(y, dtype=np.float64)
    tha = np.ascontiguousarray(th, dtype=np.float64)
    out_L = ctypes.c_double(0.0)
    pen = cpp().eval_sat_pen_bbox(
        n,
        xa.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ya.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        tha.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        float(side),
        ctypes.byref(out_L),
    )
    return float(pen), float(out_L.value)


def cpp_lbfgs_relax(
    x: np.ndarray,
    y: np.ndarray,
    th: np.ndarray,
    L: float,
    a_start: float = 1.0,
    n_melt_stages: int = 1,
    full_iters: int = 240,
    seam_mask_full: np.ndarray | None = None,
    seam_iters: int = 0,
    n_ruin_iters: int = 0,
    seed: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Single-trajectory C++ L-BFGS SAT relaxation (sym_mode=1) returning (nx, ny, nth, pen, L_bbox)."""
    n = len(x)
    hc = 0.5 * L
    q_red = np.ascontiguousarray(np.concatenate([x - hc, y - hc, th]), dtype=np.float64)
    q_feas = np.zeros_like(q_red)
    L_sched = np.array([L], dtype=np.float64)
    best_Lf = np.zeros(1, dtype=np.float64)
    out_E = np.zeros(1, dtype=np.float64)
    out_pen = np.zeros(1, dtype=np.float64)
    out_Lb = np.zeros(1, dtype=np.float64)

    has_sm = 1 if seam_mask_full is not None else 0
    if seam_mask_full is not None:
        sm = np.ascontiguousarray(
            np.concatenate([seam_mask_full, seam_mask_full, seam_mask_full]).astype(np.uint8)
        )
        sm_ptr = sm.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    else:
        sm = np.zeros(1, dtype=np.uint8)
        sm_ptr = sm.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))

    cpp().batch_melt_solve(
        1, n, 1,
        q_red.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        q_feas.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        sm_ptr, has_sm,
        1, L_sched.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        float(a_start), int(n_melt_stages),
        int(seam_iters), int(full_iters), int(n_ruin_iters),
        1, int(seed),
        best_Lf.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out_E.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out_pen.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out_Lb.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    use_q = q_feas if best_Lf[0] < 1e17 else q_red
    z_full = np.zeros(3 * n, dtype=np.float64)
    cpp().expand_sym_coords(
        n, 1, float(L),
        use_q.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        z_full.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    nx, ny, nth = z_full[:n].copy(), z_full[n : 2 * n].copy(), z_full[2 * n :].copy()
    pen, Lb = sat_max_overlap_and_bbox(nx, ny, nth)
    hx = 0.5 * (np.abs(np.cos(nth)) + np.abs(np.sin(nth)))
    wall_pen = max(
        0.0,
        float(np.max(hx - nx)),
        float(np.max(nx + hx - L)),
        float(np.max(hx - ny)),
        float(np.max(ny + hx - L)),
    )
    return nx, ny, nth, max(pen, wall_pen), Lb


def cpp_batch_melt(
    coords_list: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    L_sched_list: list[list[float]],
    sym_mode: int = 1,
    a_start: float = 0.94,
    n_melt_stages: int = 4,
    seam_masks: list[np.ndarray] | None = None,
    seam_iters: int = 180,
    full_iters: int = 240,
    n_ruin_iters: int = 8,
    n_threads: int = 4,
    base_seed: int = 42,
) -> list[dict]:
    """Run batch of trajectories in C++ (threads capped <= 4)."""
    batch_size = len(coords_list)
    if batch_size == 0:
        return []
    n = len(coords_list[0][0])
    has_c = 0 if sym_mode == 1 else (n % sym_mode)
    m_orb = n if sym_mode == 1 else ((n - has_c) // sym_mode)
    dim = 3 * m_orb + has_c
    n_L_steps = len(L_sched_list[0])
    threads = max(1, min(4, int(n_threads)))

    q_red_batch = np.zeros((batch_size, dim), dtype=np.float64)
    q_feas_batch = np.zeros((batch_size, dim), dtype=np.float64)
    L_sched_batch = np.array(L_sched_list, dtype=np.float64)
    has_sm = 1 if seam_masks is not None else 0
    seam_batch = np.zeros((batch_size, dim), dtype=np.uint8) if has_sm else np.zeros(1, dtype=np.uint8)

    for b, (x, y, th) in enumerate(coords_list):
        L0 = float(L_sched_list[b][0])
        hc = 0.5 * L0
        if sym_mode == 1:
            q_red_batch[b, :n] = x - hc
            q_red_batch[b, n : 2 * n] = y - hc
            q_red_batch[b, 2 * n :] = th
            if has_sm:
                m_arr = seam_masks[b].astype(np.uint8)
                seam_batch[b, :n] = m_arr
                seam_batch[b, n : 2 * n] = m_arr
                seam_batch[b, 2 * n :] = m_arr
        else:
            # Expect caller to pass orbit-ordered arrays (m_orb orbits of size sym_mode + optional center)
            idx_orb = np.arange(0, sym_mode * m_orb, sym_mode)
            q_red_batch[b, :m_orb] = x[idx_orb] - hc
            q_red_batch[b, m_orb : 2 * m_orb] = y[idx_orb] - hc
            q_red_batch[b, 2 * m_orb : 3 * m_orb] = th[idx_orb]
            if has_c:
                q_red_batch[b, 3 * m_orb] = th[sym_mode * m_orb]
            if has_sm:
                m_orb_arr = seam_masks[b][idx_orb].astype(np.uint8)
                seam_batch[b, :m_orb] = m_orb_arr
                seam_batch[b, m_orb : 2 * m_orb] = m_orb_arr
                seam_batch[b, 2 * m_orb : 3 * m_orb] = m_orb_arr
                if has_c:
                    seam_batch[b, 3 * m_orb] = np.uint8(seam_masks[b][sym_mode * m_orb])

    best_Lf = np.zeros(batch_size, dtype=np.float64)
    out_E = np.zeros(batch_size, dtype=np.float64)
    out_pen = np.zeros(batch_size, dtype=np.float64)
    out_Lb = np.zeros(batch_size, dtype=np.float64)

    cpp().batch_melt_solve(
        batch_size, n, sym_mode,
        q_red_batch.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        q_feas_batch.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        seam_batch.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        has_sm,
        n_L_steps,
        L_sched_batch.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        float(a_start), int(n_melt_stages),
        int(seam_iters), int(full_iters), int(n_ruin_iters),
        threads, int(base_seed),
        best_Lf.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out_E.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out_pen.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out_Lb.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )

    results = []
    z_full = np.zeros(3 * n, dtype=np.float64)
    for b in range(batch_size):
        L_fin = float(L_sched_list[b][-1])
        has_feas = bool(best_Lf[b] < 1e17)
        q_use = np.ascontiguousarray(q_feas_batch[b] if has_feas else q_red_batch[b], dtype=np.float64)
        cpp().expand_sym_coords(
            n, sym_mode, L_fin,
            q_use.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            z_full.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        nx, ny, nth = z_full[:n].copy(), z_full[n : 2 * n].copy(), z_full[2 * n :].copy()
        pen, Lb = sat_max_overlap_and_bbox(nx, ny, nth)
        results.append({
            "x": nx,
            "y": ny,
            "th": nth,
            "pen": float(pen),
            "L_bbox": float(Lb),
            "best_L_feas": float(best_Lf[b]) if has_feas else None,
            "E": float(out_E[b]),
        })
    return results


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
    min_x = float(np.min(X_c))
    min_y = float(np.min(Y_c))

    for k_slot in range(2):
        # Left wall (>= min_x or 0): X_c + dx - dy_c dth >= curv_pad
        c_min_x = order_x[:, k_slot]
        x_val = X_c[idx_n, c_min_x]
        dy_val = dy_c[idx_n, c_min_x]
        mask = x_val < min_x + 2.5
        if np.any(mask):
            m_idx = idx_n[mask]
            m_len = len(m_idx)
            r_ids = np.arange(row_counter, row_counter + m_len, dtype=np.int32)
            row_counter += m_len
            rows_list.extend([r_ids, r_ids])
            cols_list.extend([m_idx, 2 * n + m_idx])
            vals_list.extend([np.full(m_len, -1.0), dy_val[mask]])
            b_list.append(np.maximum(x_val[mask] - curv_pad, -0.4 * tr_xy))

        # Right wall: X_c + dx - dy_c dth - L <= -curv_pad
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

        # Bottom wall: Y_c + dy + dx_c dth >= curv_pad
        c_min_y = order_y[:, k_slot]
        y_val = Y_c[idx_n, c_min_y]
        dx_val = dx_c[idx_n, c_min_y]
        mask = y_val < min_y + 2.5
        if np.any(mask):
            m_idx = idx_n[mask]
            m_len = len(m_idx)
            r_ids = np.arange(row_counter, row_counter + m_len, dtype=np.int32)
            row_counter += m_len
            rows_list.extend([r_ids, r_ids])
            cols_list.extend([n + m_idx, 2 * n + m_idx])
            vals_list.extend([np.full(m_len, -1.0), -dx_val[mask]])
            b_list.append(np.maximum(y_val[mask] - curv_pad, -0.4 * tr_xy))

        # Top wall: Y_c + dy + dx_c dth - L <= -curv_pad
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


def fixed_angle_lp_tighten(
    x: np.ndarray,
    y: np.ndarray,
    th: np.ndarray,
    rounds: int = 5,
    margin: float = 1e-8,
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

        dx_mat = cx[None, :] - cx[:, None]
        dy_mat = cy[None, :] - cy[:, None]
        h_sum = hx[:, None] + hx[None, :] + 0.5
        ni, nj = np.where((np.abs(dx_mat) <= h_sum) & (np.abs(dy_mat) <= h_sum) & (np.arange(n)[:, None] < np.arange(n)[None, :]))

        for idx in range(len(ni)):
            i, j = int(ni[idx]), int(nj[idx])
            dx = float(dx_mat[i, j])
            dy = float(dy_mat[i, j])
            ci, si, cj, sj = c[i], s[i], c[j], s[j]
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
        res = linprog(obj, A_ub=A, b_ub=np.array(ub), bounds=bounds, method="highs", options={"threads": 1, "time_limit": 10.0})
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


def joint_xytheta_slp(
    x: np.ndarray,
    y: np.ndarray,
    th: np.ndarray,
    rounds: int = 16,
    tr_th0: float = 0.025,
    tr_xy0: float = 0.12,
    margin: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, bool]:
    """Jointly optimize (x, y, th, L) via Sequential Linear Programming + C++ L-BFGS + fixed-angle LP."""
    n = len(x)
    k = int(math.floor(math.sqrt(n)))
    hx = 0.5 * (np.abs(np.cos(th)) + np.abs(np.sin(th)))
    cur_x = x - np.min(x - hx)
    cur_y = y - np.min(y - hx)
    cur_th = th.copy()

    pen0, bbox0 = sat_max_overlap_and_bbox(cur_x, cur_y, cur_th)
    if pen0 > 1e-8:
        cur_x, cur_y, cur_th, pen0, bbox0 = cpp_lbfgs_relax(cur_x, cur_y, cur_th, max(bbox0, k + 0.45), full_iters=220)
        hx = 0.5 * (np.abs(np.cos(cur_th)) + np.abs(np.sin(cur_th)))
        cur_x = cur_x - np.min(cur_x - hx)
        cur_y = cur_y - np.min(cur_y - hx)
        pen0, bbox0 = sat_max_overlap_and_bbox(cur_x, cur_y, cur_th)

    best_x, best_y, best_th = cur_x.copy(), cur_y.copy(), cur_th.copy()
    best_L = bbox0 if pen0 <= 1e-9 else 1e9
    best_ok = pen0 <= 1e-9

    obj = np.zeros(3 * n + 1, dtype=np.float64)
    obj[3 * n] = 1.0

    tr_th = tr_th0
    tr_xy = tr_xy0

    for _ in range(rounds):
        A_ub, b_ub = _build_joint_slp_sparse(cur_x, cur_y, cur_th, tr_xy=tr_xy, tr_th=tr_th, margin=margin)
        bounds = (
            [(-tr_xy, tr_xy)] * (2 * n)
            + [(-tr_th, tr_th)] * n
            + [(float(k) + 0.40, float(max(bbox0, k + 0.65) + 0.35))]
        )
        res = linprog(obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs", options={"threads": 1, "time_limit": 10.0})
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
            chx = 0.5 * (np.abs(np.cos(cand_th)) + np.abs(np.sin(cand_th)))
            cand_x = cand_x - np.min(cand_x - chx)
            cand_y = cand_y - np.min(cand_y - chx)
            pen_c, bbox_c = sat_max_overlap_and_bbox(cand_x, cand_y, cand_th)
            if pen_c > 1e-9:
                target_L = max(lp_L, min(bbox_c, bbox0) - 0.0015)
                cand_x, cand_y, cand_th, pen_c, bbox_c = cpp_lbfgs_relax(
                    cand_x, cand_y, cand_th, target_L, full_iters=140
                )
                chx = 0.5 * (np.abs(np.cos(cand_th)) + np.abs(np.sin(cand_th)))
                cand_x = cand_x - np.min(cand_x - chx)
                cand_y = cand_y - np.min(cand_y - chx)
                pen_c, bbox_c = sat_max_overlap_and_bbox(cand_x, cand_y, cand_th)

            if pen_c <= 1e-9 and bbox_c < best_L:
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
        if pen_f > 1e-10:
            cx, cy, cth, pen_f, bbox_f = cpp_lbfgs_relax(cx, cy, cth, bbox_f, full_iters=250)
        tx, ty, tL, tok = fixed_angle_lp_tighten(cx, cy, cth, rounds=4, margin=1e-8)
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
        "L_float": float(max(Decimal(side_lit), Decimal(side_pd))),
        "rows": rows,
    }


def verify_final_csv_format(path: Path = FINAL_CSV) -> None:
    """Validate final.csv matches .github/workflows/final-csv-check.yml."""
    text = path.read_text(encoding="utf-8")
    reader = csv.reader(io.StringIO(text, newline=""), strict=True)
    if next(reader, None) != ["id", "x", "y", "deg"]:
        raise ValueError("expected header id,x,y,deg")
    groups: dict[int, set[int]] = {}
    for line, row in enumerate(reader, 2):
        if len(row) != 4:
            raise ValueError(f"line {line}: expected four columns")
        m = re.fullmatch(r"([0-9]{3}|1000)_(0|[1-9][0-9]*)", row[0])
        if not m:
            raise ValueError(f"line {line}: invalid id {row[0]}")
        n, idx = int(m.groups()[0]), int(m.groups()[1])
        if not 1 <= n <= 1000 or not 0 <= idx < n:
            raise ValueError(f"line {line}: id out of range")
        s = groups.setdefault(n, set())
        if idx in s:
            raise ValueError(f"line {line}: duplicate id")
        s.add(idx)
        for val in row[1:]:
            if not math.isfinite(float(val[1:] if val.startswith("s") else val)):
                raise ValueError(f"line {line}: non-finite number")
    for n, s in groups.items():
        if s != set(range(n)):
            raise ValueError(f"n={n}: incomplete group ({len(s)}/{n})")


def save_win_to_final_csv(n: int, x: np.ndarray, y: np.ndarray, th: np.ndarray) -> dict:
    """Verify strict literal+pandas win against merged.csv and update poc/swarm_cascades/final.csv."""
    merged_puzzles = sc.read_submission(MERGED_CSV)
    inc_L_lit = Decimal(sc.certify(n, merged_puzzles[n]))

    hx = 0.5 * (np.abs(np.cos(th)) + np.abs(np.sin(th)))
    sx = x - (x - hx).min() + 1e-12
    sy = y - (y - hx).min() + 1e-12
    deg = np.degrees(th)
    # Use degrees roundtrip: verify exact rad(deg) in squares.core too
    th_from_deg = np.radians(np.array([float(repr(float(d))) for d in deg]))
    res = verify_official(n, sx, sy, th_from_deg)
    if not res["ok"]:
        return {"win": False, "reason": res}

    L_lit = Decimal(res["L_lit"])
    L_pd = Decimal(res["L_pandas"])
    if not (L_lit < inc_L_lit and L_pd < inc_L_lit):
        return {
            "win": False,
            "reason": "not strictly better",
            "L_lit": str(L_lit),
            "L_pd": str(L_pd),
            "inc_L": str(inc_L_lit),
        }

    # Load existing final.csv entries if present, else empty dict
    existing: dict[int, list[dict[str, str]]] = {}
    if FINAL_CSV.exists():
        with FINAL_CSV.open("r", encoding="utf-8", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                nid, idx = row["id"].split("_")
                existing.setdefault(int(nid), []).append(row)

    new_rows = [
        {
            "id": f"{n:03d}_{i}" if n < 1000 else f"1000_{i}",
            "x": repr(float(sx[i])),
            "y": repr(float(sy[i])),
            "deg": repr(float(deg[i])),
        }
        for i in range(n)
    ]
    existing[n] = new_rows

    FINAL_CSV.parent.mkdir(parents=True, exist_ok=True)
    with FINAL_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "x", "y", "deg"])
        w.writeheader()
        for k_n in sorted(existing):
            for row in existing[k_n]:
                w.writerow(row)

    verify_final_csv_format(FINAL_CSV)
    return {
        "win": True,
        "n": n,
        "L_lit": str(L_lit),
        "L_pd": str(L_pd),
        "inc_L": str(inc_L_lit),
        "dL": float(inc_L_lit - max(L_lit, L_pd)),
    }


def load_merged_and_gap(ns: list[int]) -> tuple[pd.DataFrame, dict[int, dict[str, float]]]:
    """Load merged.csv and incumbent/frontier L for requested n."""
    df = pd.read_csv(MERGED_CSV)
    gap = pd.read_csv(PER_N_GAP_CSV).set_index("n")
    info = {}
    for n in ns:
        sub = df[df["n"] == n]
        x, y, a = sub["x"].to_numpy(copy=True), sub["y"].to_numpy(copy=True), sub["angle"].to_numpy(copy=True)
        _, L_ours = sat_max_overlap_and_bbox(x, y, a)
        info[n] = {"L_ours": float(L_ours), "L_front": float(gap.loc[n, "L_frontier"])}
    return df, info
