"""Idea 18: C++ OpenMP Spatial-Hash Soft-Core Inflation + Basin-Hopping + HiGHS Joint (x, y, dtheta) SLP."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import ctypes
from decimal import Decimal
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import utils as u  # noqa: E402
from idea17_run_slp import joint_xytheta_slp  # noqa: E402

CPP_SRC = SCRIPT_DIR / "idea18_inflate_cpp.cpp"
LIB_PATH = Path("/tmp/libidea18.so")

_LIB: ctypes.CDLL | None = None


def get_cpp_lib(force_rebuild: bool = False) -> ctypes.CDLL:
    """Compile (if missing or stale) and load /tmp/libidea18.so."""
    global _LIB
    if _LIB is not None and not force_rebuild:
        return _LIB

    need_compile = force_rebuild or (not LIB_PATH.exists())
    if not need_compile and CPP_SRC.stat().st_mtime > LIB_PATH.stat().st_mtime:
        need_compile = True

    if need_compile:
        cmd = [
            "g++", "-O3", "-march=native", "-shared", "-fPIC", "-std=c++17",
            "-pthread", "-Wall", "-Wextra", str(CPP_SRC), "-o", str(LIB_PATH),
        ]
        subprocess.run(cmd, check=True)

    lib = ctypes.CDLL(str(LIB_PATH))
    c_dp = ctypes.POINTER(ctypes.c_double)
    c_ip = ctypes.POINTER(ctypes.c_int)

    lib.eval_sat_pen_bbox.argtypes = [ctypes.c_int, c_dp, c_dp, c_dp, ctypes.c_double, c_dp]
    lib.eval_sat_pen_bbox.restype = ctypes.c_double

    lib.eval_sat_energy_grad.argtypes = [
        ctypes.c_int, c_dp, c_dp, c_dp,
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, c_dp,
    ]
    lib.eval_sat_energy_grad.restype = ctypes.c_double

    lib.batch_inflate_solve.argtypes = [
        ctypes.c_int, ctypes.c_int,
        c_dp, c_dp,
        ctypes.c_int, c_dp, c_ip, c_dp,
        ctypes.c_int, ctypes.c_double, ctypes.c_double,
        ctypes.c_int, ctypes.c_uint64,
        c_dp, c_dp, c_dp,
    ]
    lib.batch_inflate_solve.restype = None

    _LIB = lib
    return lib


def cpp_eval_sat_pen_bbox(
    x: np.ndarray, y: np.ndarray, th: np.ndarray, side: float = 1.0
) -> tuple[float, float]:
    """C++ spatial-hash wrapper matching utils.sat_max_overlap_and_bbox."""
    lib = get_cpp_lib()
    xa = np.ascontiguousarray(x, dtype=np.float64)
    ya = np.ascontiguousarray(y, dtype=np.float64)
    ta = np.ascontiguousarray(th, dtype=np.float64)
    out_L = ctypes.c_double(0.0)
    pen = lib.eval_sat_pen_bbox(
        len(xa),
        xa.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ya.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ta.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        float(side),
        ctypes.byref(out_L),
    )
    return float(pen), float(out_L.value)


def cpp_eval_energy_grad(
    x: np.ndarray, y: np.ndarray, th: np.ndarray,
    L: float, w_wall: float = 2.0, w_pair: float = 1.0, side: float = 1.0,
) -> tuple[float, np.ndarray]:
    """C++ spatial-hash wrapper matching utils.sat_penalty_grad."""
    lib = get_cpp_lib()
    n = len(x)
    xa = np.ascontiguousarray(x, dtype=np.float64)
    ya = np.ascontiguousarray(y, dtype=np.float64)
    ta = np.ascontiguousarray(th, dtype=np.float64)
    grad = np.zeros(3 * n, dtype=np.float64)
    e = lib.eval_sat_energy_grad(
        n,
        xa.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ya.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ta.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        float(L), float(w_wall), float(w_pair), float(side),
        grad.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    return float(e), grad


def run_cpp_batch_inflate(
    states_list: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    L_targets: list[float],
    a_sched: list[float],
    stage_iters: list[int],
    stage_wang: list[float],
    n_kicks: int = 2,
    kick_ang: float = 0.22,
    kick_xy: float = 0.09,
    n_threads: int = 7,
    base_seed: int = 18001,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run batch of inflation trajectories in C++ using <= 7 threads."""
    lib = get_cpp_lib()
    batch_size = len(states_list)
    if batch_size == 0:
        raise ValueError("Empty states_list")
    n = len(states_list[0][0])

    buf = np.empty((batch_size, 3 * n), dtype=np.float64)
    for b, (x, y, th) in enumerate(states_list):
        buf[b, :n] = x
        buf[b, n : 2 * n] = y
        buf[b, 2 * n :] = th

    L_arr = np.ascontiguousarray(L_targets, dtype=np.float64)
    a_arr = np.ascontiguousarray(a_sched, dtype=np.float64)
    it_arr = np.ascontiguousarray(stage_iters, dtype=np.int32)
    wa_arr = np.ascontiguousarray(stage_wang, dtype=np.float64)
    out_E = np.empty(batch_size, dtype=np.float64)
    out_pen = np.empty(batch_size, dtype=np.float64)
    out_L = np.empty(batch_size, dtype=np.float64)

    threads = max(1, min(7, int(n_threads)))
    lib.batch_inflate_solve(
        batch_size, n,
        buf.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        L_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        len(a_arr),
        a_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        it_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        wa_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        int(n_kicks), float(kick_ang), float(kick_xy),
        threads, int(base_seed),
        out_E.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out_pen.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out_L.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    return buf, out_E, out_pen, out_L


# -------------------- Seed Family Generators -------------------- #

def _rotated_core_points(n_core: int, th_rad: float, cx: float, cy: float, pitch: float = 0.97) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate `n_core` points around (cx, cy) forming a compact rotated core."""
    if n_core == 1:
        return np.array([cx]), np.array([cy]), np.array([th_rad])
    if n_core == 2:
        pts = [(-0.5, 0.0), (0.5, 0.0)]
    elif n_core == 6:
        pts = [(i - 1.0, j - 0.5) for i in range(3) for j in range(2)]
    else:
        r = int(math.ceil(math.sqrt(n_core))) + 2
        cands = []
        for i in range(-r, r + 1):
            for j in range(-r, r + 1):
                # Circular/octagonal preference keeps corners compact
                d2 = i * i + j * j
                cands.append((d2, max(abs(i), abs(j)), abs(i) + abs(j), float(i), float(j)))
        cands.sort()
        pts = [(t[3], t[4]) for t in cands[:n_core]]
        mx = sum(p[0] for p in pts) / n_core
        my = sum(p[1] for p in pts) / n_core
        pts = [(p[0] - mx, p[1] - my) for p in pts]

    cc, ss = math.cos(th_rad), math.sin(th_rad)
    xs = np.array([cx + pitch * (u_p * cc - v_p * ss) for u_p, v_p in pts], dtype=np.float64)
    ys = np.array([cy + pitch * (u_p * ss + v_p * cc) for u_p, v_p in pts], dtype=np.float64)
    ths = np.full(n_core, th_rad, dtype=np.float64)
    return xs, ys, ths


def _fill_corners_excluding_core(
    n_need: int,
    L: float,
    core_x: np.ndarray,
    core_y: np.ndarray,
    th_corner_deg: float = 0.0,
    pitch: float = 1.00,
    min_core_sep: float = 0.85,
    quad_mask: tuple[bool, bool, bool, bool] = (True, True, True, True),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate `n_need` corner-lattice squares outside core cleanly (always returns exact `n_need`)."""
    k_lim = int(L) + 2
    th_c = math.radians(th_corner_deg)
    cc, ss = math.cos(th_c), math.sin(th_c)
    hx = 0.5 * (abs(cc) + abs(ss))
    # Build selected corner lattices and sort by distance from own corner
    cands = []
    for i in range(k_lim):
        for j in range(k_lim):
            p0x = 0.5 + i * pitch * cc - j * pitch * ss
            p0y = 0.5 + i * pitch * ss + j * pitch * cc
            locs = [
                (p0x, p0y, th_c),
                (L - p0y, p0x, th_c),
                (L - p0x, L - p0y, th_c),
                (p0y, L - p0x, th_c),
            ]
            for q_id, (px, py, qt) in enumerate(locs):
                if not quad_mask[q_id]:
                    continue
                if hx * 0.85 <= px <= L - hx * 0.85 and hx * 0.85 <= py <= L - hx * 0.85:
                    d_corner = i + j + 0.02 * abs(i - j) + 0.001 * q_id
                    cands.append((d_corner, px, py, qt))

    cands.sort(key=lambda t: t[0])
    # Deduplicate points from different corners near the midline
    kept = []
    core_d2_cut = min_core_sep * min_core_sep
    # First pass: avoid core + avoid near-duplicates
    deferred = []
    for _, px, py, qt in cands:
        if any((px - kx) ** 2 + (py - ky) ** 2 < 0.72 * 0.72 for kx, ky, _ in kept):
            continue
        if len(core_x) > 0 and np.min((core_x - px) ** 2 + (core_y - py) ** 2) < core_d2_cut:
            deferred.append((px, py, qt))
            continue
        kept.append((px, py, qt))
        if len(kept) == n_need:
            break

    # Fallback pass if we still need more squares (safe because soft-core a=0.86 separates!)
    if len(kept) < n_need:
        for px, py, qt in deferred:
            if any((px - kx) ** 2 + (py - ky) ** 2 < 0.60 * 0.60 for kx, ky, _ in kept):
                continue
            kept.append((px, py, qt))
            if len(kept) == n_need:
                break

    if len(kept) < n_need:
        rng = np.random.default_rng(1818 + n_need)
        while len(kept) < n_need:
            kept.append((float(rng.uniform(0.6, L - 0.6)), float(rng.uniform(0.6, L - 0.6)), th_c))

    arr = np.array(kept[:n_need], dtype=np.float64)
    return arr[:, 0], arr[:, 1], arr[:, 2]


def generate_pin4_fixed_seeds(n: int, L_tgt: float, seed_rng: np.random.Generator) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    """Family 1: Exact C4 4-quadrant chiral pinwheel + tilted central pocket (always exact count n)."""
    k = int(round(math.sqrt(n - 1)))
    seeds = []
    rem_mod4 = n % 4
    core_candidates = [m for m in (1, 5, 9, 13, 17, 25, 29, 37) if m % 4 == rem_mod4 and m < n]
    if not core_candidates:
        core_candidates = [m for m in (2, 6, 10, 14, 18, 26, 30, 38) if m % 4 == rem_mod4 and m < n]

    for n_core in core_candidates[:4]:
        q_need = (n - n_core) // 4
        for th_core_deg in (18.0, 28.0, 38.0, 45.0):
            for th_dom_deg in (1.5, 4.0, 8.5):
                cx0, cy0 = 0.5 * L_tgt, 0.5 * L_tgt
                cx_arr, cy_arr, ct_arr = _rotated_core_points(n_core, math.radians(th_core_deg), cx0, cy0, pitch=0.96)

                # Build one quadrant template starting at corner (0, 0), tilted by th_dom_deg
                td = math.radians(th_dom_deg)
                cd, sd = math.cos(td), math.sin(td)
                pitch = 0.94
                q_cands = []
                lim = k // 2 + 4
                for i in range(lim):
                    for j in range(lim):
                        px = 0.48 + pitch * (i * cd - j * sd)
                        py = 0.48 + pitch * (i * sd + j * cd)
                        # Keep inside first quadrant territory mostly
                        if 0.35 <= px <= 0.62 * L_tgt and 0.35 <= py <= 0.62 * L_tgt:
                            d_center = (px - cx0) ** 2 + (py - cy0) ** 2
                            if d_center > (0.28 * math.sqrt(n_core)) ** 2:
                                q_cands.append((max(i, j), i + j, -d_center, px, py))
                q_cands.sort()
                # If slightly fewer, relax inner exclusions
                if len(q_cands) < q_need:
                    for i in range(lim):
                        for j in range(lim):
                            px = 0.48 + pitch * (i * cd - j * sd)
                            py = 0.48 + pitch * (i * sd + j * cd)
                            if 0.3 <= px <= 0.68 * L_tgt and 0.3 <= py <= 0.68 * L_tgt:
                                d_center = (px - cx0) ** 2 + (py - cy0) ** 2
                                q_cands.append((max(i, j) + 100, i + j, -d_center, px, py))
                    # deduplicate
                    seen = set()
                    uniq = []
                    for item in q_cands:
                        key = (round(item[3], 4), round(item[4], 4))
                        if key not in seen:
                            seen.add(key)
                            uniq.append(item)
                    q_cands = uniq

                q0 = q_cands[:q_need]
                qx, qy, qth = [], [], []
                for _, _, _, px, py in q0:
                    dx, dy = px - cx0, py - cy0
                    # Rotate 0, 90, 180, 270 around (cx0, cy0)
                    qx.extend([cx0 + dx, cx0 - dy, cx0 - dx, cx0 + dy])
                    qy.extend([cy0 + dy, cy0 + dx, cy0 - dy, cy0 - dx])
                    qth.extend([td, td, td, td])

                x_all = np.concatenate([cx_arr, np.array(qx, dtype=np.float64)])
                y_all = np.concatenate([cy_arr, np.array(qy, dtype=np.float64)])
                th_all = np.concatenate([ct_arr, np.array(qth, dtype=np.float64)])
                if len(x_all) == n:
                    seeds.append((f"pin4_c{n_core}_tc{th_core_deg:.0f}_td{th_dom_deg:.1f}", x_all, y_all, th_all))
                    # Break exact C4 degeneracy on a perturbed sibling
                    if abs(th_dom_deg - 4.0) < 0.1:
                        xj = x_all + seed_rng.uniform(-0.04, 0.04, size=n)
                        yj = y_all + seed_rng.uniform(-0.04, 0.04, size=n)
                        tj = th_all + seed_rng.uniform(-0.04, 0.04, size=n)
                        seeds.append((f"pin4_c{n_core}_tc{th_core_deg:.0f}_td{th_dom_deg:.1f}_brk", xj, yj, tj))
    return seeds


def generate_pinched_frame4_seeds(n: int, L_tgt: float) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    """Family 2: 4-edge pinched tilted wall strips (or 2-wall L-trough) around inner (k-m)^2 core."""
    k = int(round(math.sqrt(n - 1)))
    seeds = []

    for m in (2, 3, 4):
        k_in = k - m
        if k_in < 6:
            continue
        n_in = k_in * k_in
        n_frame = n - n_in

        for core_deg in (0.0, 26.0, 45.0):
            if core_deg == 0.0:
                # Upright centered core
                off = 0.5 * (L_tgt - k_in) + 0.5
                xs_in = [off + i for i in range(k_in) for _ in range(k_in)]
                ys_in = [off + j for _ in range(k_in) for j in range(k_in)]
                th_in = [0.0] * n_in
            else:
                # Tilted core centered (takes k_in^2 or fits if small enough)
                if k_in * math.sqrt(2) > L_tgt + 1.5 and core_deg == 45.0:
                    continue
                xs_in_a, ys_in_a, th_in_a = _rotated_core_points(n_in, math.radians(core_deg), 0.5 * L_tgt, 0.5 * L_tgt, pitch=0.95)
                xs_in, ys_in, th_in = xs_in_a.tolist(), ys_in_a.tolist(), th_in_a.tolist()

            for th_strip_deg in (8.0, 15.0, 22.0, 30.0):
                for chiral_mode in (1, -1):  # 1: all cyclic same sign; -1: alternating
                    th_s = math.radians(th_strip_deg)
                    hx_s = 0.5 * (abs(math.cos(th_s)) + abs(math.sin(th_s)))
                    fx, fy, fth = [], [], []
                    base_cnt = n_frame // 4
                    rem = n_frame % 4
                    # 4 edges in cyclic order: bottom, right, top, left
                    for e in range(4):
                        cnt = base_cnt + (1 if e < rem else 0)
                        ang_e = th_s if (chiral_mode == 1 or e % 2 == 0) else -th_s
                        tiers = 1 if m == 2 else 2
                        per_tier = (cnt + tiers - 1) // tiers
                        placed = 0
                        for t_row in range(tiers):
                            row_n = min(per_tier, cnt - placed)
                            if row_n <= 0:
                                break
                            depth = 0.52 * hx_s + 0.48 + t_row * 0.78
                            span_start = 0.55 + 0.35 * t_row
                            span_end = L_tgt - 0.55 - 0.35 * t_row
                            pos = np.linspace(span_start, span_end, row_n)
                            for p in pos:
                                if e == 0:
                                    fx.append(float(p)); fy.append(float(depth))
                                elif e == 1:
                                    fx.append(float(L_tgt - depth)); fy.append(float(p))
                                elif e == 2:
                                    fx.append(float(L_tgt - p)); fy.append(float(L_tgt - depth))
                                else:
                                    fx.append(float(depth)); fy.append(float(L_tgt - p))
                                fth.append(float(ang_e))
                            placed += row_n

                    x_all = np.array(xs_in + fx, dtype=np.float64)
                    y_all = np.array(ys_in + fy, dtype=np.float64)
                    th_all = np.array(th_in + fth, dtype=np.float64)
                    if len(x_all) == n:
                        cm = "cy" if chiral_mode == 1 else "al"
                        seeds.append((f"pf4_m{m}_c{core_deg:.0f}_s{th_strip_deg:.0f}_{cm}", x_all, y_all, th_all))

        # L-trough variant for m=2, 3: upright core pushed to (0, 0), tilted double-strip on right+top L
        if m in (2, 3):
            xs_in = [0.5 + i for i in range(k_in) for _ in range(k_in)]
            ys_in = [0.5 + j for _ in range(k_in) for j in range(k_in)]
            th_in = [0.0] * n_in
            for th_strip_deg in (15.0, 22.0, 28.0, 35.0):
                th_s = math.radians(th_strip_deg)
                # Right vertical slab: x in [k_in + 0.4, L_tgt - 0.45], y in [0.45, L_tgt - 0.45]
                # Top horizontal slab: x in [0.45, k_in - 0.2], y in [k_in + 0.4, L_tgt - 0.45]
                n_right = (n_frame * (k + 1)) // (2 * k + 1)
                n_top = n_frame - n_right
                fx, fy, fth = [], [], []
                for idx in range(n_right):
                    col = idx % m
                    row_i = idx // m
                    n_rows_col = max(1, (n_right + m - 1 - col) // m)
                    xr = k_in + 0.55 + col * ((L_tgt - k_in - 1.05) / max(1, m - 1))
                    yr = 0.55 + (row_i + 0.3 * (col % 2)) * ((L_tgt - 1.15) / max(1, n_rows_col))
                    fx.append(xr); fy.append(min(L_tgt - 0.45, yr)); fth.append(th_s)
                for idx in range(n_top):
                    row_i = idx % m
                    col_i = idx // m
                    n_cols_row = max(1, (n_top + m - 1 - row_i) // m)
                    yt = k_in + 0.55 + row_i * ((L_tgt - k_in - 1.05) / max(1, m - 1))
                    xt = 0.55 + (col_i + 0.3 * (row_i % 2)) * ((k_in - 0.7) / max(1, n_cols_row))
                    fx.append(min(k_in - 0.2, xt)); fy.append(yt); fth.append(-th_s)

                x_all = np.array(xs_in + fx, dtype=np.float64)
                y_all = np.array(ys_in + fy, dtype=np.float64)
                th_all = np.array(th_in + fth, dtype=np.float64)
                if len(x_all) == n:
                    seeds.append((f"ltrough_m{m}_s{th_strip_deg:.0f}", x_all, y_all, th_all))
    return seeds


def generate_chamfer_core_seeds(n: int, L_tgt: float) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    """Family 3: Rotated/chamfered (a x b) core (center & shifted) + inflated corner wedges."""
    k = int(round(math.sqrt(n - 1)))
    seeds = []
    # Core sizes from ~k*0.25 up to ~(k*0.7)^2
    core_sizes = sorted({max(9, int(round(r * n))) for r in (0.12, 0.22, 0.32, 0.44)})
    for n_core in core_sizes:
        for th_core_deg in (16.0, 24.0, 30.0, 36.0, 41.0, 44.5):
            # Make sure diameter fits inside L_tgt
            if math.sqrt(n_core) * (math.cos(math.radians(th_core_deg)) + math.sin(math.radians(th_core_deg))) > L_tgt - 0.5:
                continue
            for dx_c, dy_c in ((0.0, 0.0), (-0.5, 0.0), (-0.5, -0.5)):
                cx0 = 0.5 * L_tgt + dx_c
                cy0 = 0.5 * L_tgt + dy_c
                cx_arr, cy_arr, ct_arr = _rotated_core_points(n_core, math.radians(th_core_deg), cx0, cy0, pitch=1.00)
                for th_corner_deg in (0.0,):
                    ox, oy, oth = _fill_corners_excluding_core(
                        n - n_core, L_tgt, cx_arr, cy_arr, th_corner_deg=th_corner_deg, pitch=1.00
                    )
                    x_all = np.concatenate([cx_arr, ox])
                    y_all = np.concatenate([cy_arr, oy])
                    th_all = np.concatenate([ct_arr, oth])
                    seeds.append((f"cham_n{n_core}_tc{th_core_deg:.0f}_dx{dx_c:.1f}_tcorn{th_corner_deg:.1f}", x_all, y_all, th_all))
    return seeds


def generate_taper_band_seeds(n: int, L_tgt: float) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    """Family 4: Skewed / rational (3,4,5) / cross-tapered diagonal corridor w=2..6 + triangular lobes."""
    seeds = []
    diag_len = L_tgt * math.sqrt(2.0)
    for w in (2, 3, 4, 5, 6):
        for th_deg in (26.565, 33.0, 36.87, 42.0):
            th_rad = math.radians(th_deg)
            for taper_mode in (0.0, 4.5, -3.5):
                bx, by, bth = [], [], []
                m_u = int(diag_len / 0.92) + 2
                c45 = math.sqrt(0.5)
                for iu in range(-m_u, m_u + 1):
                    u_loc = iu * 0.95
                    for iv in range(w):
                        v_loc = (iv - 0.5 * (w - 1)) * 0.96
                        px = 0.5 * L_tgt + c45 * (u_loc - v_loc)
                        py = 0.5 * L_tgt + c45 * (u_loc + v_loc)
                        if 0.55 <= px <= L_tgt - 0.55 and 0.55 <= py <= L_tgt - 0.55:
                            if taper_mode >= 0.0:
                                ang_loc = th_rad + math.radians(taper_mode) * (2.0 * abs(u_loc) / max(1.0, diag_len) - 0.5)
                            else:
                                # Cross-band n=37 style counter-tilt (side wings opposite signs around center)
                                side_sgn = 0.0 if w % 2 == 1 and iv == w // 2 else (-1.0 if iv < 0.5 * (w - 1) else 1.0)
                                ang_loc = math.radians(45.0) + side_sgn * math.radians(abs(taper_mode))
                            bx.append(px); by.append(py); bth.append(ang_loc)
                if len(bx) >= n - 10:
                    continue
                core_x = np.array(bx, dtype=np.float64)
                core_y = np.array(by, dtype=np.float64)
                core_th = np.array(bth, dtype=np.float64)
                ox, oy, oth = _fill_corners_excluding_core(
                    n - len(core_x), L_tgt, core_x, core_y, th_corner_deg=0.0, pitch=1.00,
                    quad_mask=(False, True, False, True),
                )
                seeds.append((
                    f"taper_w{w}_th{th_deg:.1f}_tm{taper_mode:+.1f}",
                    np.concatenate([core_x, ox]),
                    np.concatenate([core_y, oy]),
                    np.concatenate([core_th, oth]),
                ))
    return seeds


def generate_denovo_kick_seeds(
    n: int, L_tgt: float, df_merged: pd.DataFrame, seed_rng: np.random.Generator, n_random: int = 36
) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    """Family 5: Perturbed incumbents + de novo multi-domain soft-core starts."""
    k = int(round(math.sqrt(n - 1)))
    seeds = []

    sub = df_merged[df_merged["n"] == n]
    if len(sub) == n:
        x0 = sub["x"].to_numpy(np.float64).copy()
        y0 = sub["y"].to_numpy(np.float64).copy()
        th0 = sub["angle"].to_numpy(np.float64).copy()
        c0, s0 = np.cos(th0), np.sin(th0)
        hx = 0.5 * (np.abs(c0) + np.abs(s0))
        x0 -= float((x0 - hx).min())
        y0 -= float((y0 - hx).min())
        L0 = float(max((x0 + hx).max(), (y0 + hx).max()))
        xs = x0 * (L_tgt / L0)
        ys = y0 * (L_tgt / L0)

        deg_f = np.minimum(np.mod(np.degrees(th0), 90.0), 90.0 - np.mod(np.degrees(th0), 90.0))
        tilted = deg_f > 8.0
        cx_t = float(np.mean(xs[tilted])) if np.any(tilted) else 0.5 * L_tgt
        cy_t = float(np.mean(ys[tilted])) if np.any(tilted) else 0.5 * L_tgt

        # Rotate tilted core + kick transition shell
        for r_deg in (-22.0, -14.0, -8.0, 8.0, 14.0, 22.0):
            dth = math.radians(r_deg)
            cd, sd = math.cos(dth), math.sin(dth)
            xk, yk, tk = xs.copy(), ys.copy(), th0.copy()
            if np.any(tilted):
                dx_t = xk[tilted] - cx_t
                dy_t = yk[tilted] - cy_t
                xk[tilted] = cx_t + dx_t * cd - dy_t * sd
                yk[tilted] = cy_t + dx_t * sd + dy_t * cd
                tk[tilted] += dth
            dist_c = np.hypot(xk - cx_t, yk - cy_t)
            halo = (~tilted) & (dist_c < 0.38 * L_tgt)
            tk[halo] += seed_rng.uniform(-0.18, 0.18, size=int(np.sum(halo)))
            seeds.append((f"inc_rot{r_deg:+.0f}_halo", xk, yk, tk))

        # Radial vortex / shear resets of inner 30%, 55%, 100% squares
        for frac in (0.30, 0.55, 1.00):
            for mode in ("vortex", "chiral4", "rand"):
                xk, yk, tk = xs.copy(), ys.copy(), th0.copy()
                r_dist = np.hypot(xk - 0.5 * L_tgt, yk - 0.5 * L_tgt)
                cutoff = float(np.quantile(r_dist, frac)) if frac < 1.0 else 1e9
                mask = r_dist <= cutoff
                if mode == "vortex":
                    tk[mask] = 0.5 * np.arctan2(yk[mask] - 0.5 * L_tgt, xk[mask] - 0.5 * L_tgt)
                elif mode == "chiral4":
                    q = ((xk > 0.5 * L_tgt).astype(int) + 2 * (yk > 0.5 * L_tgt).astype(int))
                    angles = np.array([0.35, -0.35, 0.35, -0.35])
                    tk[mask] = angles[q[mask]] + seed_rng.uniform(-0.08, 0.08, size=int(np.sum(mask)))
                else:
                    tk[mask] = seed_rng.uniform(-math.pi / 4, math.pi / 4, size=int(np.sum(mask)))
                xk[mask] += seed_rng.uniform(-0.12, 0.12, size=int(np.sum(mask)))
                yk[mask] += seed_rng.uniform(-0.12, 0.12, size=int(np.sum(mask)))
                seeds.append((f"inc_{mode}_f{int(frac*100)}", xk, yk, tk))

    # Neighbor donor cavity melts (n+1..n+4 -> n): drop seam/tilted boundary squares and melt seam
    for n_don in range(n + 1, n + 5):
        sub_d = df_merged[df_merged["n"] == n_don]
        if len(sub_d) != n_don:
            continue
        xd = sub_d["x"].to_numpy(np.float64).copy()
        yd = sub_d["y"].to_numpy(np.float64).copy()
        td = sub_d["angle"].to_numpy(np.float64).copy()
        cd, sd = np.cos(td), np.sin(td)
        hxd = 0.5 * (np.abs(cd) + np.abs(sd))
        xd -= float((xd - hxd).min())
        yd -= float((yd - hxd).min())
        Ld = float(max((xd + hxd).max(), (yd + hxd).max()))
        xs_d = xd * (L_tgt / Ld)
        ys_d = yd * (L_tgt / Ld)
        # Drop (n_don - n) squares either near tilted seam midpoint or near center
        deg_d = np.minimum(np.mod(np.degrees(td), 90.0), 90.0 - np.mod(np.degrees(td), 90.0))
        d_diag = np.abs((xs_d - 0.5 * L_tgt) - (ys_d - 0.5 * L_tgt))
        for drop_mode in ("center", "seam", "corner"):
            if drop_mode == "center":
                score_drop = np.hypot(xs_d - 0.5 * L_tgt, ys_d - 0.5 * L_tgt)
            elif drop_mode == "seam":
                score_drop = np.where(deg_d > 5.0, d_diag, 1e6 + d_diag)
            else:
                score_drop = np.minimum(xs_d, L_tgt - xs_d) + np.minimum(ys_d, L_tgt - ys_d)
            keep_idx = np.argsort(score_drop)[n_don - n:]
            xk, yk, tk = xs_d[keep_idx].copy(), ys_d[keep_idx].copy(), td[keep_idx].copy()
            # Melt seam angles slightly
            seam_m = np.minimum(np.mod(np.degrees(tk), 90.0), 90.0 - np.mod(np.degrees(tk), 90.0)) > 5.0
            if np.any(seam_m):
                tk[seam_m] += seed_rng.uniform(-0.22, 0.22, size=int(np.sum(seam_m)))
            seeds.append((f"don_{n_don}_{drop_mode}", xk, yk, tk))

    # Pure de novo uniform & structured multi-angle soft-core starts
    grid_x = np.linspace(0.5, L_tgt - 0.5, k)
    gx, gy = np.meshgrid(grid_x, grid_x)
    base_x = np.concatenate([gx.ravel(), [0.5 * L_tgt + 0.15]])
    base_y = np.concatenate([gy.ravel(), [0.5 * L_tgt - 0.15]])

    for idx in range(n_random):
        kind = idx % 5
        if kind == 4:
            # Pure uniform fluid box start (like calibrated n=17 fluid phase)
            xk = seed_rng.uniform(0.5, L_tgt - 0.5, size=n)
            yk = seed_rng.uniform(0.5, L_tgt - 0.5, size=n)
            tk = seed_rng.uniform(-math.pi / 4, math.pi / 4, size=n)
        else:
            xk = base_x + seed_rng.uniform(-0.22, 0.22, size=n)
            yk = base_y + seed_rng.uniform(-0.22, 0.22, size=n)
            if kind == 0:
                tk = seed_rng.uniform(-math.pi / 4, math.pi / 4, size=n)
            elif kind == 1:
                ang_q = seed_rng.uniform(0.15, 0.65)
                sgn = np.where((xk > 0.5 * L_tgt) ^ (yk > 0.5 * L_tgt), -1.0, 1.0)
                tk = sgn * ang_q + seed_rng.uniform(-0.12, 0.12, size=n)
            elif kind == 2:
                r_center = np.maximum(np.abs(xk - 0.5 * L_tgt), np.abs(yk - 0.5 * L_tgt))
                ang_c = seed_rng.uniform(0.25, 0.72)
                tk = np.where(r_center < 0.30 * L_tgt, ang_c, 0.0) + seed_rng.uniform(-0.08, 0.08, size=n)
            else:
                r_rad = np.hypot(xk - 0.5 * L_tgt, yk - 0.5 * L_tgt)
                tk = 0.5 * np.sin(r_rad * 0.6 + seed_rng.uniform(0, 6.28)) + seed_rng.uniform(-0.1, 0.1, size=n)
        seeds.append((f"denovo_k{kind}_{idx}", xk, yk, tk))
    return seeds


def _polish_candidate_worker(args: tuple) -> dict:
    """Finish a promising C++ basin with HiGHS joint_xytheta_slp + fixed_angle_lp_tighten + verify_official."""
    idx, label, fam, n, x, y, th, cpp_E, cpp_pen, cpp_L = args
    t0 = time.time()

    # Shift coordinates flush to [0, cpp_L]^2 before L-BFGS / SLP
    hx = 0.5 * (np.abs(np.cos(th)) + np.abs(np.sin(th)))
    x = x - float((x - hx).min())
    y = y - float((y - hx).min())

    # If tiny overlap remains from C++, relax slightly then pass to joint_xytheta_slp
    if cpp_pen > 1e-7 and cpp_pen < 0.05:
        L_try = cpp_L + max(0.002, 1.8 * cpp_pen)
        xr, yr, thr, _ = u.relax_lbfgs(x, y, th, L_try, maxiter=240, w_wall=4.0, w_pair=1.0, side=1.0)
        pen_r, L_r = cpp_eval_sat_pen_bbox(xr, yr, thr, side=1.0)
        if pen_r <= cpp_pen + 1e-6:
            x, y, th = xr, yr, thr
            cpp_pen, cpp_L = pen_r, L_r

    pen_pre, L_pre = cpp_eval_sat_pen_bbox(x, y, th, side=1.0)
    ok_slp = False
    L_slp = L_pre
    x_f, y_f, th_f = x.copy(), y.copy(), th.copy()

    if pen_pre < 0.04:
        xs, ys, ts, Ls, ok_s = joint_xytheta_slp(x, y, th, rounds=16, tr_th0=0.025, tr_xy0=0.12, margin=1e-8)
        if ok_s:
            x_f, y_f, th_f, L_slp, ok_slp = xs, ys, ts, Ls, True

    xf2, yf2, L_lp2, ok_lp2 = u.fixed_angle_lp_tighten(x_f, y_f, th_f, rounds=6, margin=1e-9)
    if ok_lp2 and L_lp2 < L_slp:
        x_f, y_f, L_slp, ok_slp = xf2, yf2, L_lp2, True

    pen_fin, L_fin = cpp_eval_sat_pen_bbox(x_f, y_f, th_f, side=1.0)
    ang_deg = np.degrees(th_f) % 90.0
    n45 = int(np.sum(np.abs(ang_deg - 45.0) < 0.2))
    n0 = int(np.sum((ang_deg < 0.2) | (ang_deg > 89.8)))

    ver = {"ok": False}
    if pen_fin < 1e-7:
        ver = u.verify_official(n, x_f, y_f, th_f)

    return {
        "idx": idx,
        "label": label,
        "family": fam,
        "n": n,
        "cpp_E": float(cpp_E),
        "cpp_pen": float(cpp_pen),
        "cpp_L": float(cpp_L),
        "ok_slp": bool(ok_slp),
        "pen_fin": float(pen_fin),
        "L_fin": float(L_fin),
        "n0": n0,
        "n45": n45,
        "non45_non0": int(n - n0 - n45),
        "verify": {k: v for k, v in ver.items() if k != "rows"},
        "rows": ver.get("rows"),
        "dt": round(time.time() - t0, 2),
    }


def write_final_csv_from_wins(wins_by_n: dict[int, dict], out_path: Path) -> None:
    """Format strict certified wins into poc/swarm_angles/final.csv matching CI schema (id,x,y,deg)."""
    records = []
    for n in sorted(wins_by_n.keys()):
        rows = wins_by_n[n]["rows"]
        for r in rows:
            sid = int(r["square_id"])
            id_str = f"{n:03d}_{sid}" if n < 1000 else f"{n}_{sid}"
            deg_val = repr(float(np.degrees(float(r["angle"]))))
            records.append({"id": id_str, "x": r["x"], "y": r["y"], "deg": deg_val})
    df_out = pd.DataFrame(records, columns=["id", "x", "y", "deg"])
    df_out.to_csv(out_path, index=False)


def run_calibration(n_threads: int = 7) -> dict:
    """Calibrate C++ soft-core inflation on n=50 (non-45 target L=7.571429) & n=65 (L=8.535534)."""
    t0 = time.time()
    out = {}
    for n, L_tgt in [(50, 7.574), (65, 8.538)]:
        rng = np.random.default_rng(1850 + n)
        k = int(round(math.sqrt(n - 1)))
        gx, gy = np.meshgrid(np.linspace(0.5, L_tgt - 0.5, k), np.linspace(0.5, L_tgt - 0.5, k))
        bx = np.concatenate([gx.ravel(), [0.5 * L_tgt]])
        by = np.concatenate([gy.ravel(), [0.5 * L_tgt]])
        states = []
        for i in range(56):
            xk = bx + rng.uniform(-0.25, 0.25, size=n)
            yk = by + rng.uniform(-0.25, 0.25, size=n)
            tk = rng.uniform(-math.pi / 4, math.pi / 4, size=n)
            states.append((xk, yk, tk))

        a_sched = [0.84, 0.87, 0.90, 0.93, 0.955, 0.972, 0.985, 0.993, 0.997, 1.000]
        iters = [120, 120, 120, 150, 180, 200, 220, 250, 280, 320]
        wang = [0.0] * len(a_sched)
        buf, Es, pens, Ls = run_cpp_batch_inflate(
            states, [L_tgt] * len(states), a_sched, iters, wang,
            n_kicks=3, kick_ang=0.24, kick_xy=0.09, n_threads=n_threads, base_seed=9000 + n,
        )
        n_feas = int(np.sum(pens < 1e-6))
        best_i = int(np.argmin(Es + 10.0 * pens))
        x_b, y_b, th_b = buf[best_i, :n], buf[best_i, n : 2 * n], buf[best_i, 2 * n :]
        _, _, _, L_slp, ok_slp = joint_xytheta_slp(x_b, y_b, th_b, rounds=14, tr_th0=0.025, tr_xy0=0.12)
        out[n] = {
            "L_tgt": L_tgt,
            "n_feas": n_feas,
            "best_E": float(Es[best_i]),
            "best_pen": float(pens[best_i]),
            "best_L_cpp": float(Ls[best_i]),
            "L_slp": float(L_slp),
            "ok_slp": bool(ok_slp),
        }
    out["elapsed_s"] = round(time.time() - t0, 2)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=int, nargs="+", default=[401, 442, 530, 626])
    parser.add_argument("--threads", type=int, default=7)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-calib", action="store_true")
    args = parser.parse_args()

    n_threads = max(1, min(7, args.threads))
    get_cpp_lib(force_rebuild=True)

    results: dict = {"targets": args.targets, "threads": n_threads, "per_n": {}, "wins": []}
    if not args.skip_calib:
        cal = run_calibration(n_threads=n_threads)
        results["calibration"] = cal
        print(f"[CALIB] {json.dumps(cal)}", flush=True)

    df_merged = pd.read_csv(REPO_ROOT / "poc" / "pell_corrugation" / "data" / "border-scan" / "merged.csv")
    inc_front = u.load_incumbent_and_frontier(args.targets)

    wins_by_n: dict[int, dict] = {}
    t_total = time.time()

    for n in args.targets:
        t_n = time.time()
        L_inc = inc_front[n]["L_ours"]
        L_front = inc_front[n]["L_front"]
        rng = np.random.default_rng(180000 + n)

        # Two target box tiers: slightly below incumbent & near frontier
        L_mid = round(0.5 * (L_inc + L_front), 6)
        L_tight = round(L_front + 0.0015, 6)
        L_near = round(L_inc - 0.0025, 6)

        # Build all 5 continuum families
        fam1 = [("pin4_fixed", *s) for s in generate_pin4_fixed_seeds(n, L_near, rng)]
        fam2 = [("pinched_frame4", *s) for s in generate_pinched_frame4_seeds(n, L_near)]
        fam3 = [("chamfer_core", *s) for s in generate_chamfer_core_seeds(n, L_near)]
        fam4 = [("taper_band", *s) for s in generate_taper_band_seeds(n, L_near)]
        fam5 = [("denovo_kick", *s) for s in generate_denovo_kick_seeds(n, L_near, df_merged, rng, n_random=12 if args.smoke else 44)]

        # Cap per structured family cleanly so Phase 1 finishes in ~3 min total
        cap = 10 if args.smoke else 64
        def _take(lst: list, m: int) -> list:
            if len(lst) <= m:
                return lst
            idx = np.linspace(0, len(lst) - 1, m, dtype=int)
            return [lst[i] for i in idx]

        struct_all = _take(fam1, cap) + _take(fam2, cap) + _take(fam3, cap) + _take(fam4, cap)
        denovo_all = _take(fam5, 16 if args.smoke else 96)

        # Assign target boxes cycling across (L_relax, L_near, L_mid, L_tight)
        L_relax = round(L_inc + 0.0015, 6)
        tiers_s = [L_relax, L_near, L_mid, L_tight]
        tiers_d = [L_near, L_mid, L_tight]
        L_struct = [tiers_s[i % 4] for i in range(len(struct_all))]
        L_denovo = [tiers_d[i % 3] for i in range(len(denovo_all))]

        # Run structured priors (gentle tether early, free angles late)
        a_s1 = [0.90, 0.935, 0.96, 0.978, 0.990, 0.996, 0.9988, 1.000]
        it_s1 = [130, 140, 160, 180, 200, 240, 260, 300]
        wa_s1 = [0.04, 0.015, 0.003, 0.0, 0.0, 0.0, 0.0, 0.0]

        t_cpp = time.time()
        buf1, E1, pen1, L1 = run_cpp_batch_inflate(
            [(s[2], s[3], s[4]) for s in struct_all],
            L_struct, a_s1, it_s1, wa_s1,
            n_kicks=2, kick_ang=0.16, kick_xy=0.07, n_threads=n_threads, base_seed=20000 + n,
        )

        # Run de novo + kicks (0.84 -> 1.00, zero tether)
        a_s2 = [0.84, 0.875, 0.905, 0.935, 0.955, 0.972, 0.985, 0.993, 0.9975, 1.000]
        it_s2 = [130, 130, 140, 160, 180, 200, 220, 250, 280, 320]
        wa_s2 = [0.0] * len(a_s2)
        buf2, E2, pen2, L2 = run_cpp_batch_inflate(
            [(s[2], s[3], s[4]) for s in denovo_all],
            L_denovo, a_s2, it_s2, wa_s2,
            n_kicks=4, kick_ang=0.26, kick_xy=0.09, n_threads=n_threads, base_seed=50000 + n,
        )
        dt_cpp = time.time() - t_cpp

        # Combine records & pick top near-feasible + lowest-E per family for Python SLP polish
        cands = []
        for idx, ((fam, lbl, _, _, _), lt) in enumerate(zip(struct_all, L_struct)):
            cands.append((fam, lbl, lt, buf1[idx, :n], buf1[idx, n:2*n], buf1[idx, 2*n:], E1[idx], pen1[idx], L1[idx]))
        for idx, ((fam, lbl, _, _, _), lt) in enumerate(zip(denovo_all, L_denovo)):
            cands.append((fam, lbl, lt, buf2[idx, :n], buf2[idx, n:2*n], buf2[idx, 2*n:], E2[idx], pen2[idx], L2[idx]))

        fam_stats = {}
        polish_jobs = []
        seen_idx = set()
        for fam_name in ("pin4_fixed", "pinched_frame4", "chamfer_core", "taper_band", "denovo_kick"):
            fc = [(i, c) for i, c in enumerate(cands) if c[0] == fam_name]
            fc.sort(key=lambda t: (0 if t[1][7] < 1e-6 else 1, t[1][8] if t[1][7] < 1e-6 else t[1][6] + 5.0 * t[1][7]))
            fam_stats[fam_name] = {
                "count": len(fc),
                "n_feas_cpp": int(sum(1 for _, c in fc if c[7] < 1e-6)),
                "best_E": float(min(c[6] for _, c in fc)) if fc else None,
                "best_pen": float(min(c[7] for _, c in fc)) if fc else None,
                "best_label": fc[0][1][1] if fc else None,
            }
            top_k = 2 if args.smoke else 6
            for i, c in fc[:top_k]:
                if i not in seen_idx:
                    seen_idx.add(i)
                    polish_jobs.append((i, c[1], c[0], n, c[3], c[4], c[5], c[6], c[7], c[8]))

        polished = []
        with ProcessPoolExecutor(max_workers=n_threads) as ex:
            for res in ex.map(_polish_candidate_worker, polish_jobs):
                polished.append(res)

        best_valid = None
        for r in polished:
            v = r["verify"]
            if v.get("ok"):
                L_ver = float(v["L_float"])
                if best_valid is None or L_ver < best_valid["L_float"]:
                    best_valid = {**r, "L_float": L_ver}
                if L_ver < L_inc - 1e-9:
                    wins_by_n[n] = {"L_float": L_ver, "L_inc": L_inc, "rows": r["rows"], "label": r["label"], "family": r["family"]}
                    results["wins"].append({k: r[k] for k in ("n", "family", "label", "L_fin", "verify", "n0", "n45", "non45_non0")})

        # Strip raw rows from JSON summary
        polished_clean = [{k: v for k, v in r.items() if k != "rows"} for r in polished]
        polished_clean.sort(key=lambda r: (0 if r["verify"].get("ok") else 1, r["L_fin"] + 10.0 * r["pen_fin"]))

        results["per_n"][str(n)] = {
            "L_inc": L_inc,
            "L_front": L_front,
            "n_cpp_runs": len(cands),
            "dt_cpp_s": round(dt_cpp, 2),
            "dt_total_s": round(time.time() - t_n, 2),
            "fam_stats": fam_stats,
            "top_polished": polished_clean[:8],
            "best_valid": {k: v for k, v in best_valid.items() if k != "rows"} if best_valid else None,
        }
        b_str = f"{best_valid['L_float']:.7f}" if best_valid else f"none(min_E={min(c[6] for c in cands):.3e}, min_pen={min(c[7] for c in cands):.4f})"
        print(f"[n={n}] inc={L_inc:.6f} front={L_front:.6f} runs={len(cands)} cpp={dt_cpp:.1f}s best_valid={b_str}", flush=True)

    results["elapsed_s"] = round(time.time() - t_total, 2)
    out_json = SCRIPT_DIR.parent / "docs" / ("idea18_smoke.json" if args.smoke else "idea18_results.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2) + "\n")

    if wins_by_n and not args.smoke:
        final_csv = SCRIPT_DIR.parent / "final.csv"
        write_final_csv_from_wins(wins_by_n, final_csv)
        print(f"[WINS] Wrote {len(wins_by_n)} winning group(s) to {final_csv}", flush=True)
    else:
        print(f"[DONE] wins={len(wins_by_n)} elapsed={results['elapsed_s']}s -> {out_json}", flush=True)


if __name__ == "__main__":
    main()
