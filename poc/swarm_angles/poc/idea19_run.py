"""Idea 19: Symmetry-Reduced (C4 / C2 / Free) C++ Spatial-Hash Cavity-Relocate ILS
with L-Continuation and Multi-Architecture Non-45° Seed Tracks.

Run calibration first on n=50 (`C2` non-45° baseline):
    PYTHONPATH=.:poc/swarm_angles/poc OMP_NUM_THREADS=1 /mnt/e/squares-in-squares/.venv/bin/python \
        poc/swarm_angles/poc/idea19_run.py --calibrate-only --threads 7

Then run deep Phase 1 search on n=401 (or --targets 401 442 530):
    PYTHONPATH=.:poc/swarm_angles/poc OMP_NUM_THREADS=1 /mnt/e/squares-in-squares/.venv/bin/python \
        poc/swarm_angles/poc/idea19_run.py --targets 401 --threads 7
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import multiprocessing as mp
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

import utils as u
from idea18_run import _polish_candidate_worker, cpp_eval_sat_pen_bbox, write_final_csv_from_wins

REPO_ROOT = Path(__file__).resolve().parents[3]
MERGED_CSV = REPO_ROOT / "poc" / "pell_corrugation" / "data" / "border-scan" / "merged.csv"
CPP_SRC_19 = Path(__file__).resolve().parent / "idea19_ils_cpp.cpp"
CPP_LIB_19 = Path("/tmp/libidea19.so")
OUT_JSON_19 = Path(__file__).resolve().parent.parent / "docs" / "idea19_results.json"
FINAL_CSV = Path(__file__).resolve().parent.parent / "final.csv"

_LIB19: ctypes.CDLL | None = None


def get_cpp_lib19(force_rebuild: bool = False) -> ctypes.CDLL:
    """Compile (if needed) and load /tmp/libidea19.so with ctypes signatures."""
    global _LIB19
    if _LIB19 is not None and not force_rebuild:
        return _LIB19
    need_build = (
        force_rebuild
        or (not CPP_LIB_19.exists())
        or (CPP_SRC_19.stat().st_mtime > CPP_LIB_19.stat().st_mtime)
    )
    if need_build:
        cmd = [
            "g++",
            "-O3",
            "-march=native",
            "-shared",
            "-fPIC",
            "-pthread",
            "-Wall",
            "-Wextra",
            "-Werror",
            str(CPP_SRC_19),
            "-o",
            str(CPP_LIB_19),
        ]
        subprocess.run(cmd, check=True)

    lib = ctypes.CDLL(str(CPP_LIB_19))
    c_dp = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    c_ip = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")

    lib.eval_sym_sat_energy_grad.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        c_dp,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        c_dp,
    ]
    lib.eval_sym_sat_energy_grad.restype = ctypes.c_double

    lib.expand_sym_coords.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_double, c_dp, c_dp]
    lib.expand_sym_coords.restype = None

    lib.eval_sat_pen_bbox.argtypes = [ctypes.c_int, c_dp, c_dp, c_dp, ctypes.c_double, c_dp]
    lib.eval_sat_pen_bbox.restype = ctypes.c_double

    lib.batch_ils_solve.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        c_dp,
        c_dp,
        ctypes.c_int,
        c_dp,
        c_ip,
        ctypes.c_double,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_uint64,
        c_dp,
        c_dp,
        c_dp,
        c_dp,
    ]
    lib.batch_ils_solve.restype = None

    _LIB19 = lib
    return lib


def sym_dim(n: int, sym_mode: int) -> tuple[int, int, int]:
    """Return (m_orbits, has_center, dim) for sym_mode in {1, 2, 4}."""
    has_c = 0 if sym_mode == 1 else (n % sym_mode)
    if sym_mode in (2, 4) and has_c > 1:
        raise ValueError(f"sym_mode={sym_mode} requires n % sym_mode in {{0, 1}}, got n={n}")
    m_orb = n if sym_mode == 1 else (n - has_c) // sym_mode
    return m_orb, has_c, 3 * m_orb + has_c


def cpp_eval_sym_energy_grad(
    n: int, sym_mode: int, q_red: np.ndarray, L: float,
    w_wall: float = 2.0, w_pair: float = 1.0, side: float = 1.0,
) -> tuple[float, np.ndarray]:
    """Evaluate symmetry-reduced SAT energy and gradient in C++."""
    lib = get_cpp_lib19()
    q_arr = np.ascontiguousarray(q_red, dtype=np.float64)
    g_out = np.empty_like(q_arr)
    e = lib.eval_sym_sat_energy_grad(n, sym_mode, q_arr, float(L), float(w_wall), float(w_pair), float(side), g_out)
    return float(e), g_out


def cpp_expand_sym(n: int, sym_mode: int, L: float, q_red: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Expand reduced orbit coords q_red around (L/2, L/2) into full (x, y, th) in [0, L]^2."""
    lib = get_cpp_lib19()
    q_arr = np.ascontiguousarray(q_red, dtype=np.float64)
    z_full = np.empty(3 * n, dtype=np.float64)
    lib.expand_sym_coords(n, sym_mode, float(L), q_arr, z_full)
    return z_full[:n].copy(), z_full[n : 2 * n].copy(), z_full[2 * n : 3 * n].copy()


def full_to_free_sym(x: np.ndarray, y: np.ndarray, th: np.ndarray, L: float) -> np.ndarray:
    """Center full [0, L]^2 coordinates around L/2 for sym_mode=1."""
    hc = 0.5 * L
    return np.concatenate([x - hc, y - hc, th]).astype(np.float64)


def run_cpp_batch_ils(
    n: int,
    sym_mode: int,
    q_list: list[np.ndarray],
    L_sched_list: list[list[float]] | list[np.ndarray],
    ils_iters_per_L: list[int],
    a_start: float = 0.90,
    n_melt_stages: int = 3,
    n_threads: int = 7,
    base_seed: int = 19001,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run batch of C++ ILS + L-continuation trajectories across <= 7 CPU threads."""
    lib = get_cpp_lib19()
    b = len(q_list)
    _, _, dim = sym_dim(n, sym_mode)
    n_L = len(ils_iters_per_L)

    q_batch = np.zeros((b, dim), dtype=np.float64)
    L_batch = np.zeros((b, n_L), dtype=np.float64)
    for i in range(b):
        assert len(q_list[i]) == dim, f"seed {i} len {len(q_list[i])} != dim {dim}"
        assert len(L_sched_list[i]) == n_L
        q_batch[i] = q_list[i]
        L_batch[i] = L_sched_list[i]

    q_feas = np.zeros_like(q_batch)
    ils_arr = np.ascontiguousarray(ils_iters_per_L, dtype=np.int32)
    out_Lf = np.empty(b, dtype=np.float64)
    out_E = np.empty(b, dtype=np.float64)
    out_pen = np.empty(b, dtype=np.float64)
    out_Lb = np.empty(b, dtype=np.float64)

    lib.batch_ils_solve(
        b,
        n,
        sym_mode,
        np.ascontiguousarray(q_batch.ravel()),
        np.ascontiguousarray(q_feas.ravel()),
        n_L,
        np.ascontiguousarray(L_batch.ravel()),
        ils_arr,
        float(a_start),
        int(n_melt_stages),
        min(7, max(1, int(n_threads))),
        int(base_seed),
        out_Lf,
        out_E,
        out_pen,
        out_Lb,
    )
    return q_batch, q_feas, out_Lf, out_E, out_pen, out_Lb


# ---------------------------------------------------------------------------
# Seed Construction Helpers (Clean Geometry without Naively Overlapping Grids)
# ---------------------------------------------------------------------------


def _greedy_fill_orbits(
    existing_pts: list[tuple[float, float, float]],
    target_n: int,
    L: float,
    sym_mode: int,
    rng: np.random.Generator,
    candidate_angles_deg: tuple[float, ...] = (0.0, 22.5, 30.0, 35.0, 45.0),
) -> list[tuple[float, float, float]]:
    """Trim excess or greedily place non-overlapping (or low-overlap) orbits up to target_n."""
    m_orb, has_c, _ = sym_dim(target_n, sym_mode)
    hc = 0.5 * L

    # Normalize existing_pts to orbit generators if sym_mode > 1, or full points if sym_mode == 1
    orbs = list(existing_pts)
    if sym_mode == 1:
        if len(orbs) > target_n:
            orbs = orbs[:target_n]
    else:
        if len(orbs) > m_orb:
            orbs = orbs[:m_orb]

    need = (target_n - len(orbs)) if sym_mode == 1 else (m_orb - len(orbs))
    if need <= 0:
        return orbs

    # Build dense spatial grid of occupied centers so we pick actual open pockets
    pts_full: list[tuple[float, float]] = []
    for rx, ry, _ in orbs:
        if sym_mode == 1:
            pts_full.append((rx, ry))
        elif sym_mode == 2:
            pts_full.append((hc + rx, hc + ry))
            pts_full.append((hc - rx, hc - ry))
        elif sym_mode == 4:
            pts_full.extend([
                (hc + rx, hc + ry),
                (hc - ry, hc + rx),
                (hc - rx, hc - ry),
                (hc + ry, hc - rx),
            ])
    if sym_mode in (2, 4) and has_c:
        pts_full.append((hc, hc))

    arr_xy = np.array(pts_full, dtype=np.float64) if pts_full else np.zeros((0, 2), dtype=np.float64)

    # Candidate positions on 0.45-step grid ranked by distance to nearest square center
    xs = np.arange(0.55, L - 0.54, 0.45)
    ys = np.arange(0.55, L - 0.54, 0.45)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    cands = np.column_stack([gx.ravel(), gy.ravel()])
    if sym_mode == 2:
        cands = cands[cands[:, 0] - hc >= -0.1]
    elif sym_mode == 4:
        cands = cands[(cands[:, 0] - hc >= -0.1) & (cands[:, 1] - hc >= -0.1)]

    if len(arr_xy) > 0 and len(cands) > 0:
        d2 = ((cands[:, None, :] - arr_xy[None, :, :]) ** 2).sum(axis=2).min(axis=1)
        order = np.argsort(-d2)
        cands = cands[order]
    else:
        rng.shuffle(cands)

    # Filter out self-overlapping central positions for C2/C4
    chosen_xy: list[tuple[float, float]] = []
    min_sep2 = 0.88 ** 2
    for px, py in cands:
        rx, ry = float(px - hc), float(py - hc)
        if sym_mode == 2 and (rx * rx + ry * ry) < 0.28:
            continue
        if sym_mode == 4 and (rx * rx + ry * ry) < 0.55:
            continue
        # Ensure newly added candidate centers aren't stacked right on top of each other
        ok = True
        for qx, qy in chosen_xy:
            if (px - qx) ** 2 + (py - qy) ** 2 < min_sep2:
                ok = False
                break
        if ok:
            chosen_xy.append((px, py))
            ang = math.radians(float(rng.choice(candidate_angles_deg)))
            orbs.append((rx if sym_mode > 1 else px, ry if sym_mode > 1 else py, ang))
            if len(chosen_xy) >= need:
                break

    # Random fallback if still short
    while len(orbs) < (target_n if sym_mode == 1 else m_orb):
        px = float(rng.uniform(0.55, L - 0.55))
        py = float(rng.uniform(0.55, L - 0.55))
        if sym_mode == 2 and px < hc:
            px = L - px
        elif sym_mode == 4:
            px = hc + abs(px - hc)
            py = hc + abs(py - hc)
        rx, ry = px - hc, py - hc
        ang = math.radians(float(rng.choice(candidate_angles_deg)))
        orbs.append((rx if sym_mode > 1 else px, ry if sym_mode > 1 else py, ang))
    return orbs


def _orbs_to_q(orbs: list[tuple[float, float, float]], n: int, sym_mode: int, th_center: float = 0.0) -> np.ndarray:
    """Pack orbit list into C++ q_red array."""
    m_orb, has_c, dim = sym_dim(n, sym_mode)
    q = np.zeros(dim, dtype=np.float64)
    for j in range(m_orb):
        q[j] = orbs[j][0]
        q[m_orb + j] = orbs[j][1]
        q[2 * m_orb + j] = orbs[j][2]
    if has_c:
        q[3 * m_orb] = th_center
    return q


def _c4_pinwheel_staircase_seed(
    n: int, L: float, m_core: int, th_core_deg: float, c4_split: int, rng: np.random.Generator,
    max_need: int | None = None,
) -> np.ndarray | None:
    """Build C4 seed: central (m_core x m_core) rotated core + non-overlapping C4 upright corner staircases."""
    m_orb, _, _ = sym_dim(n, 4)
    hc = 0.5 * L
    th = math.radians(th_core_deg)
    ct, st = math.cos(th), math.sin(th)
    orbs: list[tuple[float, float, float]] = []

    # Core: m_core must be odd so m_core^2 = 1 (mod 4) cleanly maps into (m_core^2 - 1)/4 C4 orbits + 1 center
    half_m = m_core // 2
    for u in range(-half_m, half_m + 1):
        for v in range(-half_m, half_m + 1):
            if u == 0 and v == 0:
                continue
            # Keep exactly one representative per C4 orbit: (u > 0 and v >= 0)
            if u > 0 and v >= 0:
                rx = 0.99 * (u * ct - v * st)
                ry = 0.99 * (u * st + v * ct)
                orbs.append((rx, ry, th))

    # Corner upright quadrant (bottom-left [0..s] x [0..t] where s + t = floor(L))
    kmax = int(math.floor(L))
    s = c4_split
    t = kmax - s
    core_w = m_core * (ct + st)
    hx_u = 0.5 * m_core + 0.5 * (ct + st)
    hx_x = 0.5 + 0.5 * core_w

    corner_cands: list[tuple[float, float, float]] = []
    for i in range(s):
        for j in range(t):
            px, py = i + 0.5, j + 0.5
            dx, dy = px - hc, py - hc
            # Exact SAT clearance against centered rotated square [0, m_core]^2
            clr = (
                abs(dx) >= hx_x - 0.02
                or abs(dy) >= hx_x - 0.02
                or abs(ct * dx + st * dy) >= hx_u - 0.02
                or abs(-st * dx + ct * dy) >= hx_u - 0.02
            )
            if clr:
                # Rank by corner-most distance first so boundary layers are kept intact
                dist_corner = px + py
                corner_cands.append((dist_corner, dx, dy))

    corner_cands.sort(key=lambda t3: t3[0])
    for _, dx, dy in corner_cands:
        if len(orbs) >= m_orb:
            break
        orbs.append((dx, dy, 0.0))

    if max_need is not None and (m_orb - len(orbs)) > max_need:
        return None
    orbs = _greedy_fill_orbits(orbs, n, L, 4, rng, candidate_angles_deg=(0.0, th_core_deg, 90.0 - th_core_deg))
    return _orbs_to_q(orbs, n, 4, th_center=th)


def _c2_shifted_2block_seed(
    n: int, L: float, a_blk: int, b_blk: int, th_deg: float, cx_off: float, cy_off: float,
    phase_dx: float, phase_dy: float, rng: np.random.Generator, max_need: int | None = None,
) -> np.ndarray | None:
    """Build C2-symmetric n=631/535-style seed: two C2-paired off-diagonal tilted blocks + shifted upright lattices."""
    m_orb, has_c, _ = sym_dim(n, 2)
    hc = 0.5 * L
    th = math.radians(th_deg)
    ct, st = math.cos(th), math.sin(th)
    orbs: list[tuple[float, float, float]] = []

    # Block center relative to L/2 at (+cx_off, +cy_off); C2 mirror puts second block at (-cx_off, -cy_off)
    for ia in range(a_blk):
        for ib in range(b_blk):
            lu = ia - 0.5 * (a_blk - 1)
            lv = ib - 0.5 * (b_blk - 1)
            rx = cx_off + 0.99 * (lu * ct - lv * st)
            ry = cy_off + 0.99 * (lu * st + lv * ct)
            px, py = hc + rx, hc + ry
            if 0.55 <= px <= L - 0.55 and 0.55 <= py <= L - 0.55:
                if (rx * rx + ry * ry) >= 0.30:
                    orbs.append((rx, ry, th))

    # Surround with C2-paired upright squares using a stretched k x k grid covering [0.5, L - 0.5]
    # Fundamental C2 domain keeps rx >= 0 (and ry > 0.51 when rx == 0) with max(|rx|, |ry|) >= 0.51
    up_cands: list[tuple[float, float, float]] = []
    hx_u = 0.5 * a_blk + 0.5 * (ct + st)
    hv_v = 0.5 * b_blk + 0.5 * (ct + st)
    hx_x = 0.5 + 0.5 * (a_blk * ct + b_blk * st)
    hx_y = 0.5 + 0.5 * (a_blk * st + b_blk * ct)

    k_int = int(math.floor(L))
    step_f = (L - 1.0) / max(1, k_int - 1)
    for ix in range(k_int):
        for iy in range(k_int):
            px_base = 0.5 + ix * step_f
            py_base = 0.5 + iy * step_f
            if py_base > hc + 0.25:
                px = max(0.5, min(L - 0.5, px_base - phase_dx))
                py = max(0.5, min(L - 0.5, py_base - phase_dy))
            else:
                px = px_base
                py = py_base
            rx, ry = px - hc, py - hc
            # Keep exactly one C2 representative per pair, avoiding center overlap
            if rx < -0.02 or (abs(rx) <= 0.02 and ry <= 0.51):
                continue
            if abs(rx) < 0.51 and abs(ry) < 0.51:
                continue
            # Clear both +blk and -blk
            clr = True
            for sx, sy in ((cx_off, cy_off), (-cx_off, -cy_off)):
                dx, dy = rx - sx, ry - sy
                c1 = (
                    abs(dx) >= hx_x + 0.02
                    or abs(dy) >= hx_y + 0.02
                    or abs(ct * dx + st * dy) >= hx_u + 0.02
                    or abs(-st * dx + ct * dy) >= hv_v + 0.02
                )
                if not c1:
                    clr = False
                    break
            if clr:
                wall_score = max(abs(rx), abs(ry))
                up_cands.append((wall_score, rx, ry))

    up_cands.sort(key=lambda t3: -t3[0])
    for _, rx, ry in up_cands:
        if len(orbs) >= m_orb:
            break
        orbs.append((rx, ry, 0.0))

    if max_need is not None and (m_orb - len(orbs)) > max_need:
        return None
    orbs = _greedy_fill_orbits(orbs, n, L, 2, rng, candidate_angles_deg=(0.0, th_deg, 35.0, 45.0))
    return _orbs_to_q(orbs, n, 2, th_center=0.0)


def _diag_band_phase_seed(
    n: int, L: float, band_w: int, th_deg: float, phase_shift: float, rng: np.random.Generator,
    max_need: int | None = None,
) -> np.ndarray | None:
    """Build Free (sym_mode=1) n=227/403-style diagonal band + 2 shifted complementary upright halves."""
    th = math.radians(th_deg)
    ct, st = math.cos(th), math.sin(th)
    hc = 0.5 * L
    pts: list[tuple[float, float, float]] = []

    # Diagonal band centered on line (x - hc) + (y - hc) = 0 (running BL -> TR or TL -> BR)
    # Unit tangent along band at angle th: wait for line from (L,0) to (0,L), normal is (cos45, sin45)
    # Better: corridor rotated by th around (hc, hc), spanning [-L*0.72, +L*0.72] x [-(band_w-1)/2, +(band_w-1)/2]
    u_lim = int(math.ceil(L * 0.75))
    hmargin = 0.5 * (ct + st)
    for iu in range(-u_lim, u_lim + 1):
        for iv in range(band_w):
            v = iv - 0.5 * (band_w - 1)
            px = hc + 1.003 * (iu * ct - v * st)
            py = hc + 1.003 * (iu * st + v * ct)
            if hmargin <= px <= L - hmargin and hmargin <= py <= L - hmargin:
                pts.append((px, py, th))

    # Build upright halves in complementary TL (0, L) and BR (L, 0) corners clearing the BL->TR corridor
    band_arr = np.array(pts, dtype=np.float64) if pts else np.zeros((0, 3), dtype=np.float64)

    def clears_band(px: float, py: float) -> bool:
        if len(band_arr) == 0:
            return True
        dx = band_arr[:, 0] - px
        dy = band_arr[:, 1] - py
        close = (np.abs(dx) < 0.5 + hmargin + 0.02) & (np.abs(dy) < 0.5 + hmargin + 0.02)
        if not np.any(close):
            return True
        dxc, dyc = dx[close], dy[close]
        rad = 0.5 * (1.0 + ct + st) + 0.02
        sep_x = np.abs(dxc) >= 0.5 + hmargin + 0.02
        sep_y = np.abs(dyc) >= 0.5 + hmargin + 0.02
        sep_u = np.abs(ct * dxc + st * dyc) >= rad
        sep_v = np.abs(-st * dxc + ct * dyc) >= rad
        return bool(np.all(sep_x | sep_y | sep_u | sep_v))

    k_int = int(math.floor(L))
    step_up = min(1.02, (L - 1.0) / max(1, k_int - 1))
    up_pts: list[tuple[float, float, float]] = []
    # Half 1: Top-Left corner anchored at (0.5, L - 0.5)
    for i in range(k_int):
        for j in range(k_int):
            px, py = 0.5 + i * step_up, L - 0.5 - j * step_up
            if -(px - hc) * st + (py - hc) * ct > 0.5 * band_w:
                if clears_band(px, py):
                    up_pts.append((px + (L - py), px, py))

    # Half 2: Bottom-Right corner anchored at (L - 0.5 - phase_shift, 0.5)
    for i in range(k_int):
        for j in range(k_int):
            px, py = L - 0.5 - i * step_up - phase_shift, 0.5 + j * step_up
            if px < 0.5 or px > L - 0.5 or py < 0.5 or py > L - 0.5:
                continue
            if -(px - hc) * st + (py - hc) * ct < -0.5 * band_w:
                if clears_band(px, py):
                    up_pts.append(((L - px) + py, px, py))

    # Sort upright so outer corners stay, trim if band+up > n
    up_pts.sort(key=lambda t3: t3[0])
    all_pts = list(pts)
    for _, px, py in up_pts:
        if len(all_pts) >= n:
            break
        all_pts.append((px, py, 0.0))

    if max_need is not None and (n - len(all_pts)) > max_need:
        return None
    all_pts = _greedy_fill_orbits(all_pts, n, L, 1, rng, candidate_angles_deg=(0.0, th_deg, 45.0))
    return _orbs_to_q(all_pts, n, 1)


def _incumbent_melt_seed(
    inc_x: np.ndarray, inc_y: np.ndarray, inc_th: np.ndarray, L_inc: float, L_0: float,
    rng: np.random.Generator, melt_mode: str,
) -> np.ndarray:
    """Perturb incumbent into tilted local pocket or multi-drop relocate seed at L_0."""
    scale = L_0 / L_inc
    x = inc_x * scale
    y = inc_y * scale
    th = inc_th.copy()
    n = len(x)

    if melt_mode == "pocket":
        cx = float(rng.uniform(2.5, L_0 - 2.5))
        cy = float(rng.uniform(2.5, L_0 - 2.5))
        rad2 = float(rng.uniform(2.5, 4.8)) ** 2
        tgt_th = math.radians(float(rng.choice([22.5, 26.6, 30.0, 34.5, 36.9])))
        mask = ((x - cx) ** 2 + (y - cy) ** 2) <= rad2
        th[mask] = tgt_th + rng.uniform(-0.08, 0.08, int(mask.sum()))
        x[mask] += rng.uniform(-0.18, 0.18, int(mask.sum()))
        y[mask] += rng.uniform(-0.18, 0.18, int(mask.sum()))
    elif melt_mode == "band_flip":
        # Rotate existing 45-deg squares to ~33-38 deg with shear perturbation
        ang_mod = np.degrees(th) % 90.0
        tilted = (ang_mod > 5.0) & (ang_mod < 85.0)
        new_th = math.radians(float(rng.choice([26.6, 31.0, 34.6, 36.9])))
        th[tilted] = new_th + rng.uniform(-0.06, 0.06, int(tilted.sum()))
        x[tilted] += rng.uniform(-0.12, 0.12, int(tilted.sum()))
        y[tilted] += rng.uniform(-0.12, 0.12, int(tilted.sum()))
    else:  # drop_reloc: randomly teleport 3-6 squares into random voids
        idx_pick = rng.choice(n, size=min(6, n), replace=False)
        x[idx_pick] = rng.uniform(0.6, L_0 - 0.6, len(idx_pick))
        y[idx_pick] = rng.uniform(0.6, L_0 - 0.6, len(idx_pick))
        th[idx_pick] = rng.uniform(0.0, math.pi / 2, len(idx_pick))

    return full_to_free_sym(x, y, th, L_0)


def _seed_1corner_melt(
    n: int, k: int, L_0: float, s_box: int, th_deg: float, mode: str, rng: np.random.Generator,
) -> np.ndarray:
    """Pack clean (k x k) grid shifted BL, insert +1 square, and melt TR corner box or L-trough."""
    xs = np.empty(n, dtype=np.float64)
    ys = np.empty(n, dtype=np.float64)
    ths = np.zeros(n, dtype=np.float64)
    step = min(1.003, (L_0 - 1.05) / max(1, k - 1))
    idx = 0
    pocket_mask = np.zeros(n, dtype=bool)
    for ix in range(k):
        for iy in range(k):
            px = 0.505 + ix * step
            py = 0.505 + iy * step
            xs[idx] = px
            ys[idx] = py
            if mode == "corner_box":
                if ix >= k - s_box and iy >= k - s_box:
                    pocket_mask[idx] = True
            else:  # "L_trough": melt top/right outer L strips of width 2..3 + TR tip
                if ix >= k - 2 or iy >= k - 2 or (ix + iy >= 2 * k - s_box):
                    pocket_mask[idx] = True
            idx += 1

    # Insert the +1 extra square in the outermost TR void
    xs[idx] = L_0 - 0.505
    ys[idx] = L_0 - 0.505
    pocket_mask[idx] = True

    th_rad = math.radians(th_deg)
    n_p = int(pocket_mask.sum())
    # Gently open pocket into the top/right 0.85 trough without colliding back into upright wall
    xs[pocket_mask] = np.clip(xs[pocket_mask] + rng.uniform(0.0, 0.16, n_p), 0.52, L_0 - 0.505)
    ys[pocket_mask] = np.clip(ys[pocket_mask] + rng.uniform(0.0, 0.16, n_p), 0.52, L_0 - 0.505)
    xs[idx] = L_0 - 0.505
    ys[idx] = L_0 - 0.505
    ths[pocket_mask] = th_rad + rng.uniform(-0.14, 0.14, n_p)
    return full_to_free_sym(xs, ys, ths, L_0)


def _seed_donor_non45_drop(
    dx: np.ndarray, dy: np.ndarray, dth: np.ndarray, L_don: float, L_0: float,
    drop_mode: str, melt_r: float, rng: np.random.Generator,
) -> np.ndarray:
    """Drop 2 squares from certified (n+2) non-45° packing and melt local seam at L_0."""
    n_don = len(dx)
    scale = L_0 / L_don
    x = dx * scale
    y = dy * scale
    th = dth.copy()
    L = L_0

    ang_mod = np.degrees(th) % 90.0
    tilted_idx = np.where((ang_mod > 5.0) & (ang_mod < 85.0))[0]
    upright_idx = np.where((ang_mod <= 5.0) | (ang_mod >= 85.0))[0]

    if drop_mode == "tilted_ends" and len(tilted_idx) >= 2:
        score = np.abs(x[tilted_idx] - 0.5 * L) + np.abs(y[tilted_idx] - 0.5 * L)
        drop = tilted_idx[np.argsort(-score)[:2]]
    elif drop_mode == "tilted_mid" and len(tilted_idx) >= 2:
        score = (x[tilted_idx] - 0.5 * L) ** 2 + (y[tilted_idx] - 0.5 * L) ** 2
        drop = tilted_idx[np.argsort(score)[:2]]
    elif drop_mode == "upright_seam" and len(upright_idx) >= 2 and len(tilted_idx) > 0:
        cx_t, cy_t = float(np.mean(x[tilted_idx])), float(np.mean(y[tilted_idx]))
        # Upright squares nearest band midline |x - y| or tilted centroid
        dist_band = np.min(
            (x[upright_idx, None] - x[None, tilted_idx]) ** 2
            + (y[upright_idx, None] - y[None, tilted_idx]) ** 2,
            axis=1,
        )
        # Rank seam squares biased near TR/BL wall stress
        rank_s = dist_band + 0.05 * ((x[upright_idx] - cx_t) ** 2 + (y[upright_idx] - cy_t) ** 2)
        drop = upright_idx[np.argsort(rank_s)[:2]]
    else:  # "extreme_bbox": pick 2 squares pushing outermost walls
        hx = 0.5 * (np.abs(np.cos(th)) + np.abs(np.sin(th)))
        margin = np.minimum(np.minimum(x - hx, L - x - hx), np.minimum(y - hx, L - y - hx))
        drop = np.argsort(margin)[:2]

    drop_cx = float(np.mean(x[drop]))
    drop_cy = float(np.mean(y[drop]))
    keep = np.ones(n_don, dtype=bool)
    keep[drop] = False
    x, y, th = x[keep], y[keep], th[keep]

    # Melt neighborhood within melt_r around dropped pair plus light jitter on tilted band
    dist2 = (x - drop_cx) ** 2 + (y - drop_cy) ** 2
    melt_mask = dist2 <= melt_r * melt_r
    nm = int(melt_mask.sum())
    x[melt_mask] += rng.uniform(-0.18, 0.18, nm)
    y[melt_mask] += rng.uniform(-0.18, 0.18, nm)
    th[melt_mask] += rng.uniform(-0.16, 0.16, nm)

    ang_m2 = np.degrees(th) % 90.0
    t2 = (ang_m2 > 5.0) & (ang_m2 < 85.0)
    nt2 = int(t2.sum())
    if nt2 > 0:
        x[t2] += rng.uniform(-0.08, 0.08, nt2)
        y[t2] += rng.uniform(-0.08, 0.08, nt2)
        th[t2] += rng.uniform(-0.07, 0.07, nt2)

    return full_to_free_sym(x, y, th, L_0)


# ---------------------------------------------------------------------------
# Calibration Gate: n=50 (7^2 + 1, non-45° C2 target L <= 7.575)
# ---------------------------------------------------------------------------


def calibrate_n50(n_threads: int = 7) -> dict:
    """Prove C2 Cavity-Relocate ILS + L-Continuation discovers n=50 <= 7.575 from cold starts."""
    t0 = time.time()
    n = 50
    sym_mode = 2
    m_orb, _, _ = sym_dim(n, sym_mode)
    rng = np.random.default_rng(19050)

    # Known non-45° C2 packing has L=7.571429 (34 upright + 16 @ 36.87 deg).
    # Test 64 cold C2 fluid + upright-hybrid starts (NO prior 36.87 solution injected!).
    n_starts = 64
    L_sched = [7.72, 7.65, 7.60, 7.582, 7.573]
    ils_iters = [8, 12, 16, 20, 20]

    q_list: list[np.ndarray] = []
    L_scheds: list[list[float]] = []
    for i in range(n_starts):
        L0 = L_sched[0]
        hc = 0.5 * L0
        if i < 40:
            # Pure random C2 fluid orbits in right half-plane
            rx = rng.uniform(0.25, hc - 0.52, m_orb)
            ry = rng.uniform(-(hc - 0.52), hc - 0.52, m_orb)
            rt = rng.uniform(0.0, math.pi / 2, m_orb)
        else:
            # Mix outer 7x7 upright C2 orbits + random tilted center cluster
            pts = []
            for ix in range(4, 7):
                for iy in range(7):
                    pts.append((ix + 0.5 - hc, iy + 0.5 - hc, 0.0))
            rng.shuffle(pts)
            pts = pts[:18]
            while len(pts) < m_orb:
                pts.append((
                    float(rng.uniform(0.3, 2.2)),
                    float(rng.uniform(-2.2, 2.2)),
                    math.radians(float(rng.uniform(15.0, 45.0))),
                ))
            rx = np.array([p[0] for p in pts])
            ry = np.array([p[1] for p in pts])
            rt = np.array([p[2] for p in pts])
        q_list.append(np.concatenate([rx, ry, rt]))
        L_scheds.append(L_sched)

    q_fin, q_feas, L_feas, E_out, pen_out, L_bbox = run_cpp_batch_ils(
        n, sym_mode, q_list, L_scheds, ils_iters,
        a_start=0.88, n_melt_stages=3, n_threads=n_threads, base_seed=190500,
    )

    # Polish top 12 candidates with HiGHS joint_xytheta_slp
    scored = []
    for i in range(n_starts):
        L_ref = float(L_feas[i]) if L_feas[i] < 1e9 else float(L_bbox[i])
        q_use = q_feas[i] if L_feas[i] < 1e9 else q_fin[i]
        x, y, th = cpp_expand_sym(n, sym_mode, L_ref if L_feas[i] < 1e9 else L_sched[-1], q_use)
        scored.append((float(pen_out[i]) if L_feas[i] > 1e9 else 0.0, L_ref, i, x, y, th))
    scored.sort(key=lambda t: (round(t[0], 6), t[1]))

    tasks = [
        (t[2], f"cal50_{t[2]}", "c2_cal", n, t[3], t[4], t[5], E_out[t[2]], pen_out[t[2]], t[1])
        for t in scored[:12]
    ]
    with mp.Pool(processes=min(7, len(tasks))) as pool:
        polished = pool.map(_polish_candidate_worker, tasks)

    valids = [r for r in polished if r["verify"].get("ok")]
    valids.sort(key=lambda r: float(r["verify"]["L_float"]))
    non45_wins = [r for r in valids if r["verify"]["L_float"] < 7.60]
    best_r = valids[0] if valids else None
    out = {
        "n": n,
        "n_starts": n_starts,
        "n_cpp_feas": int(np.sum(L_feas < 1e9)),
        "n_valid_polished": len(valids),
        "n_below_7_60": len(non45_wins),
        "best_L": best_r["verify"]["L_float"] if best_r else None,
        "best_non45": best_r["non45_non0"] if best_r else None,
        "best_label": best_r["label"] if best_r else None,
        "dt_s": round(time.time() - t0, 2),
    }
    print("[Calibrate n=50]", json.dumps({k: v for k, v in out.items() if k != "rows"}))
    return out


# ---------------------------------------------------------------------------
# Phase 1 Search per D=1 Target n
# ---------------------------------------------------------------------------


def _load_inc_xyz(n: int, df_merged: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Load incumbent (x, y, th, L) normalized to [0, L]^2 from merged.csv."""
    sub = df_merged[df_merged["n"] == n]
    x = sub["x"].to_numpy(np.float64).copy()
    y = sub["y"].to_numpy(np.float64).copy()
    th = sub["angle"].to_numpy(np.float64).copy()
    hx = 0.5 * (np.abs(np.cos(th)) + np.abs(np.sin(th)))
    x -= float((x - hx).min())
    y -= float((y - hx).min())
    L0 = float(max((x + hx).max(), (y + hx).max()))
    return x, y, th, L0


def solve_target_n(
    n: int, df_merged: pd.DataFrame, inc_front: dict[int, dict[str, float]],
    n_threads: int = 7, quick: bool = False,
) -> dict:
    """Run all Idea 19 tracks (C4, C2, Diagonal Band, Free ILS Melt) for one D=1 target n."""
    t0 = time.time()
    k = int(math.isqrt(n))
    inc_x, inc_y, inc_th, _ = _load_inc_xyz(n, df_merged)
    L_inc = inc_front[n]["L_ours"]
    L_front = inc_front[n]["L_front"]
    print(f"\n=== Idea 19 n={n} (k={k}) | L_inc={L_inc:.9f} | L_front={L_front:.9f} ===")

    rng = np.random.default_rng(190000 + n)
    all_candidates: list[dict] = []

    def eval_and_collect(
        fam: str, labels: list[str], sym_mode: int, q_batch: np.ndarray, q_feas: np.ndarray,
        L_feas: np.ndarray, E_out: np.ndarray, pen_out: np.ndarray, L_bbox: np.ndarray, L_end: float,
    ) -> dict:
        best_feas = float(np.min(L_feas)) if np.any(L_feas < 1e9) else None
        n_feas = int(np.sum(L_feas < 1e9))
        idx_min_p = int(np.argmin(pen_out))
        print(
            f"  [{fam}] N={len(labels)} sym={sym_mode} feas={n_feas} "
            f"best_Lf={best_feas if best_feas else 'None'} "
            f"best_pen={pen_out[idx_min_p]:.6f} ({labels[idx_min_p]})"
        )
        # Keep best feasible and best low-penalty candidates for HiGHS polish
        order = sorted(
            range(len(labels)),
            key=lambda i: (0 if L_feas[i] < 1e9 else 1, float(L_feas[i]) if L_feas[i] < 1e9 else float(pen_out[i]), float(L_bbox[i])),
        )
        keep_n = 6 if quick else 10
        for i in order[:keep_n]:
            use_feas = L_feas[i] < 1e9
            L_ref = float(L_feas[i]) if use_feas else L_end
            q_use = q_feas[i] if use_feas else q_batch[i]
            x, y, th = cpp_expand_sym(n, sym_mode, L_ref, q_use)
            all_candidates.append({
                "label": labels[i],
                "family": fam,
                "n": n,
                "x": x,
                "y": y,
                "th": th,
                "cpp_E": float(E_out[i]),
                "cpp_pen": 0.0 if use_feas else float(pen_out[i]),
                "cpp_L": float(L_feas[i]) if use_feas else float(L_bbox[i]),
            })
        return {
            "count": len(labels),
            "sym_mode": sym_mode,
            "n_feas": n_feas,
            "best_L_feas": best_feas,
            "best_pen": float(pen_out[idx_min_p]),
            "best_label": labels[idx_min_p],
        }

    fam_summary: dict[str, dict] = {}
    L0 = k + 0.82
    L_tgt = L_inc - 0.008
    L_sched = [L0, k + 0.68, k + 0.59, k + 0.55, L_inc - 0.001, max(L_front + 0.0005, L_tgt)]
    ils_iters = [3, 5, 7, 10, 12, 14] if quick else [4, 7, 10, 14, 18, 22]

    # --- Track 19A: C4 (if n % 4 == 1) Pinwheel Core + Unjammed Soft-Core Crystal-Swirl Melts ---
    if n % 4 == 1:
        m_orb, _, _ = sym_dim(n, 4)
        q_c4, lab_c4, ls_c4 = [], [], []
        hc = 0.5 * L0
        splits = [k // 2, k // 2 - 1]
        for m_core in [3, 5, 7, 9]:
            for th_deg in [18.0, 26.6, 32.0, 36.9, 41.0, 44.0]:
                if m_core * (math.cos(math.radians(th_deg)) + math.sin(math.radians(th_deg))) > L0 + 0.1:
                    continue
                for sp in splits:
                    q_seed = _c4_pinwheel_staircase_seed(n, L0, m_core, th_deg, sp, rng, max_need=8)
                    if q_seed is None:
                        continue
                    q_c4.append(q_seed)
                    lab_c4.append(f"c4_pin_m{m_core}_t{int(round(th_deg))}_s{sp}")
                    ls_c4.append(L_sched)
        # Crystal-Swirl Melts: start from clean L0-scaled k x k pinwheel, melt core/annulus with a_start=0.68
        half_k = k // 2
        step = (L0 - 0.04) / k
        base_pin = []
        for ix in range(half_k):
            for iy in range(half_k):
                rx = -hc + 0.52 + ix * step
                ry = -hc + 0.52 + iy * step
                base_pin.append((rx, ry))
        base_pin = base_pin[:m_orb]
        for R_cut in [3.5, 5.2, 6.8, 8.5]:
            for th_deg in [22.0, 26.6, 31.0, 35.0, 38.0, 42.0]:
                for rep in range(1 if quick else 2):
                    orbs_s = []
                    th_rad = math.radians(th_deg)
                    for rx, ry in base_pin:
                        rad = math.hypot(rx, ry)
                        in_zone = (rad <= R_cut) if (rep % 2 == 0) else (abs(rad - R_cut) <= 2.4 or rad <= 1.25)
                        if in_zone:
                            # Mild radial breathing room near (0,0) center orbit
                            push = 1.14 if rad <= 1.5 else 1.0
                            orbs_s.append((
                                rx * push + float(rng.uniform(-0.14, 0.14)),
                                ry * push + float(rng.uniform(-0.14, 0.14)),
                                th_rad + float(rng.uniform(-0.14, 0.14)),
                            ))
                        else:
                            orbs_s.append((rx, ry, 0.0))
                    q_c4.append(_orbs_to_q(orbs_s, n, 4, th_center=th_rad if rep == 0 else math.pi / 4))
                    lab_c4.append(f"c4_swirl_R{R_cut:.1f}_t{int(round(th_deg))}_{rep}")
                    ls_c4.append(L_sched)
        if q_c4:
            qb, qf, lf, eo, po, lb = run_cpp_batch_ils(
                n, 4, q_c4, ls_c4, ils_iters, a_start=0.68, n_melt_stages=5,
                n_threads=n_threads, base_seed=194000 + n,
            )
            fam_summary["c4_sym"] = eval_and_collect("c4_sym", lab_c4, 4, qb, qf, lf, eo, po, lb, L_sched[-1])

    # --- Track 19B: C2 Dual-Pocket Zipper + Unjammed Soft-Core Crystal 2-Pocket Melts ---
    m_orb2, has_c2, _ = sym_dim(n, 2)
    q_c2, lab_c2, ls_c2 = [], [], []
    hc = 0.5 * L0
    for a_b, b_b in [(4, 6), (4, 8), (5, 7), (6, 8)]:
        for th_deg in [24.5, 26.4, 31.0, 34.6, 36.9]:
            for cx_o, cy_o in [(0.22 * L0, 0.14 * L0), (0.28 * L0, 0.24 * L0)]:
                for pdx, pdy in [(0.52, 0.14), (0.31, 0.31)]:
                    if quick and pdx == 0.31:
                        continue
                    q_seed = _c2_shifted_2block_seed(
                        n, L0, a_b, b_b, th_deg, cx_o, cy_o, pdx, pdy, rng, max_need=18
                    )
                    if q_seed is None:
                        continue
                    q_c2.append(q_seed)
                    lab_c2.append(f"c2_2blk_{a_b}x{b_b}_t{int(round(th_deg))}_c{int(cx_o)}_{int(pdx*10)}")
                    ls_c2.append(L_sched)
    # Crystal 2-Pocket Melts: clean C2-paired lattice inside L0, melt two C2 pockets at a_start=0.68
    step = (L0 - 0.04) / k
    c2_base = []
    if k % 2 == 0:
        for ix in range(k // 2, k):
            for iy in range(k):
                c2_base.append(((ix + 0.5) * step - hc, (iy + 0.5) * step - hc))
    else:
        mid = k // 2
        for ix in range(mid + 1, k):
            for iy in range(k):
                c2_base.append(((ix + 0.5) * step - hc, (iy + 0.5) * step - hc))
        for iy in range(mid + 1, k):
            c2_base.append(((mid + 0.5) * step - hc, (iy + 0.5) * step - hc))
    c2_base = c2_base[:m_orb2]
    for px_f, py_f in [(0.16, 0.14), (0.24, 0.22), (0.20, -0.18), (0.12, 0.0)]:
        for R_p in [2.6, 3.8, 5.0]:
            for th_deg in [22.0, 26.4, 31.0, 34.6, 36.9]:
                if quick and R_p == 5.0:
                    continue
                th_r = math.radians(th_deg)
                px_c, py_c = px_f * L0, py_f * L0
                shear_y = 0.28 if px_f > 0.18 else 0.0
                # For odd k (even n=k^2+1), insert the extra orbit inside the melted C2 pocket rather than (0.65,0)
                base_pts = list(c2_base)
                while len(base_pts) < m_orb2:
                    base_pts.append((px_c + float(rng.uniform(-0.3, 0.3)), py_c + float(rng.uniform(-0.3, 0.3))))
                orbs_2 = []
                for rx, ry in base_pts[:m_orb2]:
                    rad = math.hypot(rx, ry)
                    in_p = ((rx - px_c) ** 2 + (ry - py_c) ** 2 <= R_p * R_p) or (has_c2 and rad <= 1.15)
                    if in_p:
                        push = 1.18 if (has_c2 and rad <= 1.2) else 1.0
                        orbs_2.append((
                            rx * push + float(rng.uniform(-0.14, 0.14)),
                            ry * push + float(rng.uniform(-0.14, 0.14)),
                            th_r + float(rng.uniform(-0.12, 0.12)),
                        ))
                    else:
                        orbs_2.append((rx, ry + (shear_y * 0.5 if ry > 0 else -shear_y * 0.5) * 0.4, 0.0))
                q_c2.append(_orbs_to_q(orbs_2, n, 2, th_center=th_r if has_c2 else 0.0))
                lab_c2.append(f"c2_cry_{int(px_f*100)}_{int(py_f*100)}_R{R_p:.1f}_t{int(round(th_deg))}")
                ls_c2.append(L_sched)
    if q_c2:
        qb, qf, lf, eo, po, lb = run_cpp_batch_ils(
            n, 2, q_c2, ls_c2, ils_iters, a_start=0.68, n_melt_stages=5,
            n_threads=n_threads, base_seed=192000 + n,
        )
        fam_summary["c2_sym"] = eval_and_collect("c2_sym", lab_c2, 2, qb, qf, lf, eo, po, lb, L_sched[-1])

    # --- Track 19C: Fixed TL/BR Diagonal Non-45° Corridor (n=227/403 style) ---
    q_db, lab_db, ls_db = [], [], []
    for bw in [1, 2, 3, 4]:
        for th_deg in [18.0, 24.5, 29.0, 34.6, 36.9, 41.0]:
            for psh in [0.0, 0.065, 0.25, 0.50]:
                if quick and bw == 4:
                    continue
                q_seed = _diag_band_phase_seed(n, L0, bw, th_deg, psh, rng, max_need=28)
                if q_seed is None:
                    continue
                q_db.append(q_seed)
                lab_db.append(f"db_w{bw}_t{th_deg:.1f}_p{int(psh*100)}")
                ls_db.append(L_sched)
    if q_db:
        qb, qf, lf, eo, po, lb = run_cpp_batch_ils(
            n, 1, q_db, ls_db, ils_iters, a_start=0.70, n_melt_stages=5,
            n_threads=n_threads, base_seed=193000 + n,
        )
        fam_summary["diag_band_phase"] = eval_and_collect("diag_band_phase", lab_db, 1, qb, qf, lf, eo, po, lb, L_sched[-1])

    # --- Track 19D: Free ILS Incumbent Dual-Diamond Pocket Melts ---
    q_fr, lab_fr, ls_fr = [], [], []
    L_inc_sched = [L_inc + 0.12, L_inc + 0.05, L_inc + 0.015, L_inc - 0.0005, L_inc - 0.004, max(L_front + 0.0005, L_inc - 0.010)]
    n_melts = 24 if quick else 48
    for j in range(n_melts):
        mode = ["pocket", "band_flip", "drop_reloc"][j % 3]
        q_fr.append(_incumbent_melt_seed(inc_x, inc_y, inc_th, L_inc, L_inc_sched[0], rng, mode))
        lab_fr.append(f"inc_{mode}_{j}")
        ls_fr.append(L_inc_sched)
    qb, qf, lf, eo, po, lb = run_cpp_batch_ils(
        n, 1, q_fr, ls_fr, ils_iters, a_start=0.72, n_melt_stages=4,
        n_threads=n_threads, base_seed=191000 + n,
    )
    fam_summary["free_ils_melt"] = eval_and_collect("free_ils_melt", lab_fr, 1, qb, qf, lf, eo, po, lb, L_inc_sched[-1])

    # --- Track 19E: Asymmetric 1-Corner / L-Trough Pocket Melts (Free ILS) ---
    q_cm, lab_cm, ls_cm = [], [], []
    for cmode in ["corner_box", "L_trough"]:
        for s_box in ([8, 11] if quick else [8, 10, 12, 14]):
            for th_deg in ([22.0, 31.0, 36.9] if quick else [20.0, 26.4, 31.0, 34.6, 36.9, 41.0]):
                q_cm.append(_seed_1corner_melt(n, k, L0, s_box, th_deg, cmode, rng))
                lab_cm.append(f"cm_{cmode[:3]}_s{s_box}_t{int(round(th_deg))}")
                ls_cm.append(L_sched)
    qb, qf, lf, eo, po, lb = run_cpp_batch_ils(
        n, 1, q_cm, ls_cm, ils_iters, a_start=0.68, n_melt_stages=5,
        n_threads=n_threads, base_seed=195000 + n,
    )
    fam_summary["free_1corner_melt"] = eval_and_collect("free_1corner_melt", lab_cm, 1, qb, qf, lf, eo, po, lb, L_sched[-1])

    # --- Track 19F: Certified Non-45° Donor (n+2 -> n) Corridor Drops ---
    donor_n = n + 2
    if int((df_merged["n"] == donor_n).sum()) == donor_n:
        dx, dy, dth, L_don = _load_inc_xyz(donor_n, df_merged)
        L_don0 = max(L_don + 0.05, L_inc + 0.07)
        L_don_sched = [
            L_don0,
            L_don - 0.004,
            0.5 * (L_don + L_inc),
            L_inc + 0.002,
            L_inc - 0.001,
            max(L_front + 0.0005, L_inc - 0.008),
        ]
        q_don, lab_don, ls_don = [], [], []
        n_don_seeds = 24 if quick else 48
        for j in range(n_don_seeds):
            dmode = ["tilted_ends", "tilted_mid", "upright_seam", "extreme_bbox"][j % 4]
            mr = [4.5, 6.5, 9.0][(j // 4) % 3]
            q_don.append(_seed_donor_non45_drop(dx, dy, dth, L_don, L_don0, dmode, mr, rng))
            lab_don.append(f"don{donor_n}_{dmode[:6]}_R{mr:.1f}_{j}")
            ls_don.append(L_don_sched)
        qb, qf, lf, eo, po, lb = run_cpp_batch_ils(
            n, 1, q_don, ls_don, ils_iters, a_start=0.72, n_melt_stages=4,
            n_threads=n_threads, base_seed=196000 + n,
        )
        fam_summary["donor_non45_drop"] = eval_and_collect("donor_non45_drop", lab_don, 1, qb, qf, lf, eo, po, lb, L_don_sched[-1])

    # --- Polish Top Candidates via HiGHS SLP + Official Verification ---
    polish_tasks = [
        (i, c["label"], c["family"], n, c["x"], c["y"], c["th"], c["cpp_E"], c["cpp_pen"], c["cpp_L"])
        for i, c in enumerate(all_candidates)
    ]
    print(f"  Polishing {len(polish_tasks)} candidates across <=7 workers...")
    with mp.Pool(processes=min(7, max(1, len(polish_tasks)))) as pool:
        polished = pool.map(_polish_candidate_worker, polish_tasks)

    polished.sort(key=lambda r: (0 if r["verify"].get("ok") else 1, r["pen_fin"], r["L_fin"]))
    best_win = None
    best_valid = None
    top_summary = []
    for r in polished[:12]:
        ver = r["verify"]
        ok = bool(ver.get("ok"))
        row_s = {k2: v2 for k2, v2 in r.items() if k2 != "rows"}
        top_summary.append(row_s)
        if ok:
            Lf = float(ver["L_float"])
            if best_valid is None or Lf < float(best_valid["verify"]["L_float"]):
                best_valid = r
            if Lf < L_inc - 1e-11:
                if best_win is None or Lf < float(best_win["verify"]["L_float"]):
                    best_win = r

    if best_valid is not None:
        v = best_valid["verify"]
        print(
            f"  BEST VALID n={n}: {best_valid['label']} L_lit={v['L_lit'][:16]} "
            f"(inc={L_inc:.9f}, win={best_win is not None}, n0={best_valid['n0']}, n45={best_valid['n45']}, non45={best_valid['non45_non0']})"
        )
    else:
        print(f"  NO VALID POLISHED SOLUTION for n={n} (top L_fin={polished[0]['L_fin']:.6f}, pen={polished[0]['pen_fin']:.3e})")

    return {
        "n": n,
        "L_inc": float(L_inc),
        "L_front": float(L_front),
        "dt_s": round(time.time() - t0, 2),
        "fam_summary": fam_summary,
        "top_polished": top_summary,
        "best_valid": {k2: v2 for k2, v2 in best_valid.items() if k2 != "rows"} if best_valid else None,
        "win": best_win,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=int, nargs="+", default=[401])
    parser.add_argument("--threads", type=int, default=7)
    parser.add_argument("--calibrate-only", action="store_true")
    parser.add_argument("--skip-calib", action="store_true")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    t_all = time.time()
    out: dict = {"targets": args.targets, "threads": args.threads}
    if not args.skip_calib:
        out["calib_n50"] = calibrate_n50(n_threads=args.threads)
        if args.calibrate_only:
            OUT_JSON_19.parent.mkdir(parents=True, exist_ok=True)
            OUT_JSON_19.write_text(json.dumps(out, indent=2))
            return

    df_merged = pd.read_csv(MERGED_CSV)
    inc_front = u.load_incumbent_and_frontier(args.targets)
    wins_by_n: dict[int, dict] = {}
    per_n: dict[str, dict] = {}
    for n in args.targets:
        res = solve_target_n(n, df_merged, inc_front, n_threads=args.threads, quick=args.quick)
        win = res.pop("win")
        per_n[str(n)] = res
        if win is not None:
            wins_by_n[n] = win
            # Save partial win immediately!
            win_csv = Path(__file__).resolve().parent.parent / "data" / f"win_n{n}.csv"
            win_csv.parent.mkdir(parents=True, exist_ok=True)
            write_final_csv_from_wins({n: win}, win_csv)

    out["per_n"] = per_n
    out["wins"] = [
        {"n": n, "L_lit": w["verify"]["L_lit"], "L_pandas": w["verify"]["L_pandas"], "label": w["label"]}
        for n, w in sorted(wins_by_n.items())
    ]
    out["elapsed_s"] = round(time.time() - t_all, 2)

    OUT_JSON_19.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON_19.write_text(json.dumps(out, indent=2))

    if wins_by_n:
        write_final_csv_from_wins(wins_by_n, FINAL_CSV)
        print(f"\nWROTE STRICT CERTIFIED WINS to {FINAL_CSV}: {list(wins_by_n.keys())}")
    else:
        print("\nNO STRICT WINS FOUND (final.csv untouched/absent).")


if __name__ == "__main__":
    main()
