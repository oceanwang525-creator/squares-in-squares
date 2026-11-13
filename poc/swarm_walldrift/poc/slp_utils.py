"""Shared utilities for swarm_walldrift: C++ SAT melt + directed drift bindings,
Joint (x, y, dtheta) SLP, fixed-angle LP tighten, and official union.py certification.
"""
from __future__ import annotations

import ctypes
import csv
import importlib.util
import io
import math
import os
import re
import subprocess
import sys
import warnings
from decimal import Decimal
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeWarning, linprog
from scipy.sparse import coo_matrix, csc_matrix

warnings.filterwarnings("ignore", category=OptimizeWarning, message="Unrecognized options detected.*")

REPO_ROOT = Path(__file__).resolve().parents[3]
BASELINE_SRC = REPO_ROOT / "poc" / "baseline" / "poc" / "src"
if str(BASELINE_SRC) not in sys.path:
    sys.path.insert(0, str(BASELINE_SRC))

import squares.core as sc  # noqa: E402

CPP_PATH = Path(__file__).resolve().parent / "walldrift_cpp.cpp"
SO_PATH = Path("/tmp/libwalldrift.so")
MERGED_CSV = REPO_ROOT / "poc" / "pell_corrugation" / "data" / "border-scan" / "merged.csv"
PER_N_GAP_CSV = REPO_ROOT / "poc" / "frontier_nov12" / "docs" / "per_n_gap.csv"
FINAL_CSV = Path(__file__).resolve().parents[1] / "final.csv"

_CPP = None
_UNION = None


def get_union_module():
    """Load poc/auto_union/poc/union.py dynamically."""
    global _UNION
    if _UNION is None:
        spec = importlib.util.spec_from_file_location(
            "auto_union_mod", REPO_ROOT / "poc" / "auto_union" / "poc" / "union.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _UNION = mod
    return _UNION


def load_cpp_lib():
    """Compile (if needed) and load /tmp/libwalldrift.so."""
    if (not SO_PATH.exists()) or (CPP_PATH.stat().st_mtime > SO_PATH.stat().st_mtime):
        cmd = [
            "g++",
            "-O3",
            "-march=native",
            "-shared",
            "-fPIC",
            "-std=c++17",
            "-pthread",
            "-Wall",
            "-Wextra",
            "-Werror",
            str(CPP_PATH),
            "-o",
            str(SO_PATH),
        ]
        subprocess.run(cmd, check=True)
    lib = ctypes.CDLL(str(SO_PATH))

    c_dp = ctypes.POINTER(ctypes.c_double)
    c_ip = ctypes.POINTER(ctypes.c_int)
    c_u8p = ctypes.POINTER(ctypes.c_uint8)

    lib.eval_sym_sat_energy_grad.argtypes = [
        ctypes.c_int, ctypes.c_int, c_dp,
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, c_dp,
    ]
    lib.eval_sym_sat_energy_grad.restype = ctypes.c_double

    lib.eval_sym_sat_drift_energy_grad.argtypes = [
        ctypes.c_int, ctypes.c_int, c_dp,
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
        c_dp, ctypes.c_double, c_dp,
    ]
    lib.eval_sym_sat_drift_energy_grad.restype = ctypes.c_double

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
        c_dp, ctypes.c_int, c_dp,
        ctypes.c_int, c_dp,
        ctypes.c_double, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_uint64,
        c_dp, c_dp, c_dp, c_dp,
    ]
    lib.batch_melt_solve.restype = None

    lib.batch_masked_melt_solve.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        c_dp, c_dp, c_u8p,
        c_dp, ctypes.c_int, c_dp,
        ctypes.c_int, c_dp, c_ip,
        ctypes.c_double, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_uint64,
        c_dp, c_dp, c_dp, c_dp,
    ]
    lib.batch_masked_melt_solve.restype = None
    return lib


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
    full_iters: int = 200,
    seam_mask_full: np.ndarray | None = None,
    seam_iters: int = 0,
    n_ruin_iters: int = 0,
    drift_xy: np.ndarray | None = None,
    drift_strength: float = 0.0,
    seed: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Single-trajectory C++ L-BFGS SAT relaxation returning (nx, ny, nth, pen, L_bbox)."""
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
    else:
        sm = np.zeros(1, dtype=np.uint8)

    has_dr = 1 if (drift_xy is not None and drift_strength != 0.0) else 0
    if has_dr:
        dr = np.ascontiguousarray(drift_xy, dtype=np.float64)
        ds = np.array([float(drift_strength)], dtype=np.float64)
    else:
        dr = np.zeros(1, dtype=np.float64)
        ds = np.zeros(1, dtype=np.float64)

    cpp().batch_melt_solve(
        1, n, 1,
        q_red.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        q_feas.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        sm.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), has_sm,
        dr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), has_dr,
        ds.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
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
    a_start: float = 0.86,
    n_melt_stages: int = 4,
    seam_masks: list[np.ndarray] | None = None,
    seam_iters: int = 180,
    full_iters: int = 220,
    n_ruin_iters: int = 6,
    drift_xy_list: list[np.ndarray] | None = None,
    drift_strengths: list[float] | None = None,
    n_threads: int = 4,
    base_seed: int = 42,
) -> list[dict]:
    """Run batch of full/seam-melt trajectories with optional directed drift in C++."""
    batch_size = len(coords_list)
    if batch_size == 0:
        return []
    n = len(coords_list[0][0])
    dim = 3 * n
    n_L_steps = len(L_sched_list[0])
    threads = max(1, min(4, int(n_threads)))

    q_red_batch = np.zeros((batch_size, dim), dtype=np.float64)
    q_feas_batch = np.zeros((batch_size, dim), dtype=np.float64)
    L_sched_batch = np.array(L_sched_list, dtype=np.float64)
    has_sm = 1 if seam_masks is not None else 0
    seam_batch = np.zeros((batch_size, dim), dtype=np.uint8) if has_sm else np.zeros(1, dtype=np.uint8)

    has_dr = 1 if drift_xy_list is not None else 0
    drift_batch = np.zeros((batch_size, 2 * n), dtype=np.float64) if has_dr else np.zeros(1, dtype=np.float64)
    ds_batch = np.zeros(batch_size, dtype=np.float64) if has_dr else np.zeros(1, dtype=np.float64)

    for b, (x, y, th) in enumerate(coords_list):
        hc = 0.5 * float(L_sched_list[b][0])
        q_red_batch[b, :n] = x - hc
        q_red_batch[b, n : 2 * n] = y - hc
        q_red_batch[b, 2 * n :] = th
        if has_sm:
            m_arr = seam_masks[b].astype(np.uint8)
            seam_batch[b, :n] = m_arr
            seam_batch[b, n : 2 * n] = m_arr
            seam_batch[b, 2 * n :] = m_arr
        if has_dr:
            drift_batch[b] = drift_xy_list[b]
            ds_batch[b] = float(drift_strengths[b]) if drift_strengths is not None else 0.0

    best_Lf = np.zeros(batch_size, dtype=np.float64)
    out_E = np.zeros(batch_size, dtype=np.float64)
    out_pen = np.zeros(batch_size, dtype=np.float64)
    out_Lb = np.zeros(batch_size, dtype=np.float64)

    cpp().batch_melt_solve(
        batch_size, n, 1,
        q_red_batch.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        q_feas_batch.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        seam_batch.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), has_sm,
        drift_batch.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), has_dr,
        ds_batch.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        n_L_steps, L_sched_batch.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        float(a_start), int(n_melt_stages),
        int(seam_iters), int(full_iters), int(n_ruin_iters),
        threads, int(base_seed),
        best_Lf.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out_E.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out_pen.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out_Lb.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )

    return _unpack_batch_results(
        batch_size, n, L_sched_list, q_red_batch, q_feas_batch, best_Lf, out_E, out_pen
    )


def cpp_batch_masked_melt(
    coords_list: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    active_masks: list[np.ndarray],
    L_sched_list: list[list[float]],
    ils_iters_per_L: list[int],
    a_start: float = 0.78,
    n_melt_stages: int = 4,
    masked_iters: int = 180,
    full_iters: int = 240,
    drift_xy_list: list[np.ndarray] | None = None,
    drift_strengths: list[float] | None = None,
    n_threads: int = 4,
    base_seed: int = 77001,
) -> list[dict]:
    """Run batch of masked-pocket cavity-relocation melts with optional directed drift."""
    batch_size = len(coords_list)
    if batch_size == 0:
        return []
    n = len(coords_list[0][0])
    dim = 3 * n
    n_L_steps = len(L_sched_list[0])
    threads = max(1, min(4, int(n_threads)))

    q_red_batch = np.zeros((batch_size, dim), dtype=np.float64)
    q_feas_batch = np.zeros((batch_size, dim), dtype=np.float64)
    mask_batch = np.zeros((batch_size, dim), dtype=np.uint8)
    L_sched_batch = np.array(L_sched_list, dtype=np.float64)
    ils_arr = np.ascontiguousarray(ils_iters_per_L, dtype=np.int32)

    has_dr = 1 if drift_xy_list is not None else 0
    drift_batch = np.zeros((batch_size, 2 * n), dtype=np.float64) if has_dr else np.zeros(1, dtype=np.float64)
    ds_batch = np.zeros(batch_size, dtype=np.float64) if has_dr else np.zeros(1, dtype=np.float64)

    for b, (x, y, th) in enumerate(coords_list):
        hc = 0.5 * float(L_sched_list[b][0])
        q_red_batch[b, :n] = x - hc
        q_red_batch[b, n : 2 * n] = y - hc
        q_red_batch[b, 2 * n :] = th
        m_arr = active_masks[b].astype(np.uint8)
        mask_batch[b, :n] = m_arr
        mask_batch[b, n : 2 * n] = m_arr
        mask_batch[b, 2 * n :] = m_arr
        if has_dr:
            drift_batch[b] = drift_xy_list[b]
            ds_batch[b] = float(drift_strengths[b]) if drift_strengths is not None else 0.0

    best_Lf = np.zeros(batch_size, dtype=np.float64)
    out_E = np.zeros(batch_size, dtype=np.float64)
    out_pen = np.zeros(batch_size, dtype=np.float64)
    out_Lb = np.zeros(batch_size, dtype=np.float64)

    cpp().batch_masked_melt_solve(
        batch_size, n, 1,
        q_red_batch.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        q_feas_batch.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        mask_batch.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        drift_batch.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), has_dr,
        ds_batch.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        n_L_steps, L_sched_batch.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ils_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        float(a_start), int(n_melt_stages), int(masked_iters), int(full_iters),
        threads, int(base_seed),
        best_Lf.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out_E.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out_pen.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out_Lb.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )

    return _unpack_batch_results(
        batch_size, n, L_sched_list, q_red_batch, q_feas_batch, best_Lf, out_E, out_pen
    )


def _unpack_batch_results(
    batch_size: int,
    n: int,
    L_sched_list: list[list[float]],
    q_red_batch: np.ndarray,
    q_feas_batch: np.ndarray,
    best_Lf: np.ndarray,
    out_E: np.ndarray,
    out_pen: np.ndarray,
) -> list[dict]:
    results = []
    z_full = np.zeros(3 * n, dtype=np.float64)
    for b in range(batch_size):
        has_feas = bool(best_Lf[b] < 1e17)
        L_ref = float(best_Lf[b]) if has_feas else float(L_sched_list[b][-1])
        q_use = np.ascontiguousarray(q_feas_batch[b] if has_feas else q_red_batch[b], dtype=np.float64)
        cpp().expand_sym_coords(
            n, 1, L_ref,
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
            "cpp_pen": float(out_pen[b]),
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

    rows_list, cols_list, vals_list, b_list = [], [], [], []
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
        rows_idx, cols_idx, vals, ub = [], [], [], []
        row = 0

        for i in range(n):
            rows_idx.append(row); cols_idx.append(i); vals.append(-1.0); ub.append(-(hx[i] + margin)); row += 1
            rows_idx.extend([row, row]); cols_idx.extend([i, 2 * n]); vals.extend([1.0, -1.0]); ub.append(-(hx[i] + margin)); row += 1
            rows_idx.append(row); cols_idx.append(n + i); vals.append(-1.0); ub.append(-(hx[i] + margin)); row += 1
            rows_idx.extend([row, row]); cols_idx.extend([n + i, 2 * n]); vals.extend([1.0, -1.0]); ub.append(-(hx[i] + margin)); row += 1

        dx_mat = cx[None, :] - cx[:, None]
        dy_mat = cy[None, :] - cy[:, None]
        h_sum = hx[:, None] + hx[None, :] + 0.5
        ni, nj = np.where(
            (np.abs(dx_mat) <= h_sum) & (np.abs(dy_mat) <= h_sum) & (np.arange(n)[:, None] < np.arange(n)[None, :])
        )

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
        res = linprog(
            obj, A_ub=A, b_ub=np.array(ub), bounds=bounds, method="highs",
            options={"threads": 1, "time_limit": 10.0},
        )
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
    rounds: int = 12,
    tr_th0: float = 0.015,
    tr_xy0: float = 0.08,
    margin: float = 1e-8,
    tol: float = 1e-10,
    patience: int = 3,
    lbfgs_iters: int = 120,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, bool]:
    """Jointly optimize (x, y, th, L) via exact signed-Jacobian SLP + C++ L-BFGS + LP."""
    n = len(x)
    k = int(math.floor(math.sqrt(n)))
    hx = 0.5 * (np.abs(np.cos(th)) + np.abs(np.sin(th)))
    cur_x = x - np.min(x - hx)
    cur_y = y - np.min(y - hx)
    cur_th = th.copy()

    pen0, bbox0 = sat_max_overlap_and_bbox(cur_x, cur_y, cur_th)
    if pen0 > 1e-8:
        cur_x, cur_y, cur_th, pen0, bbox0 = cpp_lbfgs_relax(
            cur_x, cur_y, cur_th, max(bbox0, k + 0.45), full_iters=max(120, lbfgs_iters)
        )
        hx = 0.5 * (np.abs(np.cos(cur_th)) + np.abs(np.sin(cur_th)))
        cur_x = cur_x - np.min(cur_x - hx)
        cur_y = cur_y - np.min(cur_y - hx)
        pen0, bbox0 = sat_max_overlap_and_bbox(cur_x, cur_y, cur_th)

    best_x, best_y, best_th = cur_x.copy(), cur_y.copy(), cur_th.copy()
    best_L = bbox0 if pen0 <= 1e-12 else 1e9
    best_ok = pen0 <= 1e-12
    if not best_ok and pen0 <= 1e-6:
        tx, ty, tL, tok = fixed_angle_lp_tighten(cur_x, cur_y, cur_th, rounds=4, margin=1e-7)
        if tok:
            best_x, best_y, best_th, best_L, best_ok = tx, ty, cur_th.copy(), tL, True

    obj = np.zeros(3 * n + 1, dtype=np.float64)
    obj[3 * n] = 1.0

    tr_th = tr_th0
    tr_xy = tr_xy0
    stale = 0

    for _ in range(rounds):
        A_ub, b_ub = _build_joint_slp_sparse(cur_x, cur_y, cur_th, tr_xy=tr_xy, tr_th=tr_th, margin=margin)
        bounds = (
            [(-tr_xy, tr_xy)] * (2 * n)
            + [(-tr_th, tr_th)] * n
            + [(float(k) + 0.40, float(max(bbox0, k + 0.65) + 0.35))]
        )
        res = linprog(
            obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs",
            options={"threads": 1, "time_limit": 10.0},
        )
        if not res.success or res.x is None:
            tr_th *= 0.6
            tr_xy *= 0.7
            stale += 1
            if stale >= patience:
                break
            continue

        dx = res.x[:n]
        dy = res.x[n : 2 * n]
        dth = res.x[2 * n : 3 * n]
        lp_L = float(res.x[3 * n])

        accepted = False
        improved = False
        for alpha in (1.0, 0.5):
            cand_x = cur_x + alpha * dx
            cand_y = cur_y + alpha * dy
            cand_th = cur_th + alpha * dth
            chx = 0.5 * (np.abs(np.cos(cand_th)) + np.abs(np.sin(cand_th)))
            cand_x = cand_x - np.min(cand_x - chx)
            cand_y = cand_y - np.min(cand_y - chx)
            pen_c, bbox_c = sat_max_overlap_and_bbox(cand_x, cand_y, cand_th)
            if pen_c > 1e-9 and lbfgs_iters > 0:
                target_L = max(lp_L, min(bbox_c, bbox0) - 0.0015)
                cand_x, cand_y, cand_th, pen_c, bbox_c = cpp_lbfgs_relax(
                    cand_x, cand_y, cand_th, target_L, full_iters=lbfgs_iters
                )
                chx = 0.5 * (np.abs(np.cos(cand_th)) + np.abs(np.sin(cand_th)))
                cand_x = cand_x - np.min(cand_x - chx)
                cand_y = cand_y - np.min(cand_y - chx)
                pen_c, bbox_c = sat_max_overlap_and_bbox(cand_x, cand_y, cand_th)

            if pen_c <= 1e-12 and bbox_c < best_L:
                if best_L - bbox_c > tol:
                    improved = True
                best_L = bbox_c
                best_x, best_y, best_th = cand_x.copy(), cand_y.copy(), cand_th.copy()
                best_ok = True
            elif pen_c <= 1e-7 and bbox_c < best_L + 0.002:
                tx, ty, tL, tok = fixed_angle_lp_tighten(cand_x, cand_y, cand_th, rounds=3, margin=1e-7)
                if tok and tL < best_L:
                    if best_L - tL > tol:
                        improved = True
                    best_L = tL
                    best_x, best_y, best_th = tx, ty, cand_th.copy()
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

        stale = 0 if improved else (stale + 1)
        if stale >= patience:
            break

    for cx, cy, cth in [(best_x, best_y, best_th), (cur_x, cur_y, cur_th)]:
        pen_f, bbox_f = sat_max_overlap_and_bbox(cx, cy, cth)
        if pen_f > 1e-11:
            cx, cy, cth, pen_f, bbox_f = cpp_lbfgs_relax(cx, cy, cth, bbox_f + 1e-4, full_iters=160)
        tx, ty, tL, tok = fixed_angle_lp_tighten(cx, cy, cth, rounds=5, margin=1e-7)
        if tok and (not best_ok or tL < best_L):
            best_x, best_y, best_th, best_L, best_ok = tx, ty, cth.copy(), tL, True

    return best_x, best_y, best_th, best_L, best_ok


def verify_official(n: int, x: np.ndarray, y: np.ndarray, th: np.ndarray) -> dict:
    """Certify literal + pandas roundtrip via official squares.core + union.py check."""
    hx = 0.5 * (np.abs(np.cos(th)) + np.abs(np.sin(th)))
    sx = x - (x - hx).min() + 1e-12
    sy = y - (y - hx).min() + 1e-12
    rows = []
    for i in range(n):
        deg_f = float(np.degrees(float(th[i])))
        ang_rt = float(np.radians(float(repr(deg_f))))
        rows.append({
            "square_id": str(i),
            "x": repr(float(sx[i])),
            "y": repr(float(sy[i])),
            "angle": repr(ang_rt),
            "deg": repr(deg_f),
        })
    core_rows = [{k: r[k] for k in ("square_id", "x", "y", "angle")} for r in rows]
    _, _, side_lit, reason_lit = sc.evaluate(n, core_rows)
    if reason_lit != "ok" or side_lit is None:
        return {"ok": False, "reason_lit": reason_lit, "L_lit": None, "L_pandas": None}
    rt = sc.kaggle_roundtrip({n: core_rows})
    _, _, side_pd, reason_pd = sc.evaluate(n, rt[n])
    if reason_pd != "ok" or side_pd is None:
        return {"ok": False, "reason_lit": reason_lit, "reason_pd": reason_pd, "L_lit": side_lit, "L_pandas": None}
    # Also verify exact final.csv serialization roundtrip through auto_union load + check
    union_mod = get_union_module()
    txt = union_mod.final_text({n: core_rows})
    lit_g, pd_g = union_mod.parse_text(txt), union_mod.parse_text(
        pd.read_csv(io.StringIO(txt)).to_csv(index=False)
    )
    chk = union_mod.check((n, lit_g[n], pd_g[n]))
    if chk is None:
        return {"ok": False, "reason_lit": "union_check_failed", "L_lit": side_lit, "L_pandas": side_pd}
    return {
        "ok": True,
        "L_lit": chk[0],
        "L_pandas": chk[1],
        "L_float": float(max(Decimal(chk[0]), Decimal(chk[1]))),
        "rows": rows,
    }


def verify_final_csv_with_union(path: Path = FINAL_CSV, baseline_path: Path = MERGED_CSV) -> dict:
    """Validate final.csv via poc/auto_union/poc/union.py load + check (literal & pandas) vs merged.csv."""
    union_mod = get_union_module()
    lit_groups, pd_groups = union_mod.load(path)
    if not lit_groups:
        return {"count": 0, "results": {}}
    base_puzzles = sc.read_submission(baseline_path)
    out = {}
    for n in sorted(lit_groups):
        res = union_mod.check((n, lit_groups[n], pd_groups[n]))
        if res is None:
            raise ValueError(f"auto_union check failed for n={n} in {path}")
        inc_L = Decimal(sc.certify(n, base_puzzles[n]))
        L_lit, L_pd = Decimal(res[0]), Decimal(res[1])
        dL = float(max(L_lit, L_pd) - inc_L)
        if not (L_lit < inc_L and L_pd < inc_L):
            raise ValueError(f"n={n} in {path} does not strictly beat merged.csv (dL={dL:+.3e})")
        out[n] = {"L_lit": res[0], "L_pd": res[1], "inc_L": str(inc_L), "dL": dL}
    return {"count": len(out), "results": out}


def write_final_csv_from_wins(wins_by_n: dict[int, dict], out_path: Path = FINAL_CSV) -> None:
    """Write certified wins/ties into final.csv (`id,x,y,deg`, canonical s-prefixed format)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not wins_by_n:
        out_path.write_text("id,x,y,deg\n", encoding="utf-8")
        return
    union_mod = get_union_module()
    groups = {}
    for n in sorted(wins_by_n):
        rows = wins_by_n[n]["rows"]
        groups[n] = [
            {
                "square_id": str(r["square_id"]),
                "x": r["x"],
                "y": r["y"],
                "angle": r["angle"],
            }
            for r in rows
        ]
    out_path.write_text(union_mod.final_text(groups), encoding="utf-8")


def load_incumbent_and_frontier(ns: list[int]) -> tuple[dict[int, tuple], dict[int, dict[str, float]]]:
    """Load normalized incumbent arrays and exact Decimal L_inc + L_front for requested n."""
    merged = pd.read_csv(MERGED_CSV)
    gap_df = pd.read_csv(PER_N_GAP_CSV).set_index("n")
    base_puzzles = sc.read_submission(MERGED_CSV)
    mats: dict[int, tuple] = {}
    info: dict[int, dict[str, float]] = {}

    for n, sub in merged.groupby("n"):
        sub = sub.sort_values("square_id")
        x = sub["x"].to_numpy(np.float64, copy=True)
        y = sub["y"].to_numpy(np.float64, copy=True)
        th = sub["angle"].to_numpy(np.float64, copy=True)
        hx = 0.5 * (np.abs(np.cos(th)) + np.abs(np.sin(th)))
        x -= float((x - hx).min())
        y -= float((y - hx).min())
        L = float(max((x + hx).max(), (y + hx).max()))
        mats[int(n)] = (x, y, th, L)

    for n in ns:
        inc_dec = Decimal(sc.certify(n, base_puzzles[n]))
        info[n] = {
            "L_ours": float(inc_dec),
            "L_inc_str": str(inc_dec),
            "L_front": float(gap_df.loc[n, "L_frontier"]),
        }
    return mats, info
