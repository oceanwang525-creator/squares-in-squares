"""Idea 1: Donor-Adapted Non-45 deg Contraction/Expansion + C++ L-BFGS a-Melt + Joint (x, y, dtheta) HiGHS SLP.

Targets (initial wave): n in {581, 850, 974}.
Strictly CPU-only, max 4 workers/threads, BLAS=1, HiGHS threads=1.
"""
from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from concurrent.futures import ProcessPoolExecutor, as_completed
import ctypes
from decimal import Decimal
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

POC_DIR = Path(__file__).resolve().parent
ROOT_DIR = POC_DIR.parent
REPO_ROOT = ROOT_DIR.parent.parent
sys.path.insert(0, str(POC_DIR))

import utils as u  # noqa: E402

LIB_PATH = Path("/tmp/libreplace45_melt.so")


def _load_cpp():
    lib = ctypes.CDLL(str(LIB_PATH))
    lib.batch_melt_solve.argtypes = [
        ctypes.c_int,  # batch_size
        ctypes.c_int,  # n
        ctypes.POINTER(ctypes.c_double),  # z_batch
        ctypes.POINTER(ctypes.c_double),  # z_feas_batch
        ctypes.c_int,  # n_L_steps
        ctypes.POINTER(ctypes.c_double),  # L_sched_batch
        ctypes.c_int,  # n_a_steps
        ctypes.POINTER(ctypes.c_double),  # a_sched
        ctypes.POINTER(ctypes.c_int),  # iters_per_a
        ctypes.c_int,  # ils_jitter_rounds
        ctypes.POINTER(ctypes.c_int),  # free_mask_batch
        ctypes.c_int,  # n_threads
        ctypes.c_uint64,  # base_seed
        ctypes.POINTER(ctypes.c_double),  # out_best_L_feas
        ctypes.POINTER(ctypes.c_double),  # out_E
        ctypes.POINTER(ctypes.c_double),  # out_pen
        ctypes.POINTER(ctypes.c_double),  # out_L_bbox
    ]
    lib.batch_melt_solve.restype = None
    return lib


def run_cpp_melt_batch(
    seeds: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]],
    L_sched: np.ndarray,
    a_sched: np.ndarray,
    iters_per_a: np.ndarray,
    ils_jitter_rounds: int = 8,
    n_threads: int = 4,
    base_seed: int = 12345,
) -> list[dict]:
    """Run C++ batch_melt_solve on a list of (tag, x, y, th, free_mask) seeds."""
    if not seeds:
        return []
    lib = _load_cpp()
    B = len(seeds)
    n = len(seeds[0][1])
    n_L = len(L_sched)
    n_a = len(a_sched)

    z_batch = np.zeros((B, 3 * n), dtype=np.float64)
    z_feas = np.zeros((B, 3 * n), dtype=np.float64)
    L_batch = np.tile(L_sched.astype(np.float64), (B, 1))
    a_arr = np.ascontiguousarray(a_sched, dtype=np.float64)
    it_arr = np.ascontiguousarray(iters_per_a, dtype=np.int32)

    use_mask = any(s[4] is not None for s in seeds)
    if use_mask:
        fmask = np.ones((B, n), dtype=np.int32)
        for b, (_, _, _, _, m) in enumerate(seeds):
            if m is not None:
                fmask[b] = m.astype(np.int32)
        fmask_ptr = fmask.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    else:
        fmask_ptr = ctypes.POINTER(ctypes.c_int)()

    for b, (_, x, y, th, _) in enumerate(seeds):
        hx = 0.5 * (np.abs(np.cos(th)) + np.abs(np.sin(th)))
        sx = x - float(np.min(x - hx))
        sy = y - float(np.min(y - hx))
        z_batch[b, :n] = sx
        z_batch[b, n : 2 * n] = sy
        z_batch[b, 2 * n :] = th
    z_feas[:] = z_batch

    out_L_feas = np.full(B, 1e18, dtype=np.float64)
    out_E = np.zeros(B, dtype=np.float64)
    out_pen = np.zeros(B, dtype=np.float64)
    out_L_bbox = np.zeros(B, dtype=np.float64)

    lib.batch_melt_solve(
        B,
        n,
        z_batch.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        z_feas.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        n_L,
        L_batch.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        n_a,
        a_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        it_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        int(ils_jitter_rounds),
        fmask_ptr,
        int(min(4, max(1, n_threads))),
        int(base_seed),
        out_L_feas.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out_E.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out_pen.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out_L_bbox.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )

    results = []
    for b, (tag, _, _, _, _) in enumerate(seeds):
        has_feas = out_L_feas[b] < 1e17
        arr = z_feas[b] if has_feas else z_batch[b]
        results.append(
            {
                "tag": tag,
                "has_feas": bool(has_feas),
                "L_feas": float(out_L_feas[b]),
                "E": float(out_E[b]),
                "pen": float(out_pen[b]),
                "L_bbox": float(out_L_bbox[b]),
                "x": arr[:n].copy(),
                "y": arr[n : 2 * n].copy(),
                "th": arr[2 * n :].copy(),
            }
        )
    return results


def _load_donor(n_src: int, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    sub = df[df["n"] == n_src]
    x = sub["x"].to_numpy(dtype=np.float64)
    y = sub["y"].to_numpy(dtype=np.float64)
    th = sub["angle"].to_numpy(dtype=np.float64)
    hx = 0.5 * (np.abs(np.cos(th)) + np.abs(np.sin(th)))
    x = x - float(np.min(x - hx))
    y = y - float(np.min(y - hx))
    L = float(max(np.max(x + hx), np.max(y + hx)))
    return x, y, th, L


def _find_void_sites(
    x: np.ndarray, y: np.ndarray, th: np.ndarray, L: float, top_k: int = 16
) -> list[tuple[float, float, float]]:
    """Find best low-overlap candidate sites (cx, cy, ang) inside [0.5, L-0.5]^2."""
    gx = np.linspace(0.55, L - 0.55, 56)
    gy = np.linspace(0.55, L - 0.55, 56)
    XX, YY = np.meshgrid(gx, gy, indexing="ij")
    pts_x = XX.ravel()
    pts_y = YY.ravel()
    # Quick min center-distance filter
    dx = pts_x[:, None] - x[None, :]
    dy = pts_y[:, None] - y[None, :]
    d2 = dx * dx + dy * dy
    min_d2 = np.min(d2, axis=1)
    top_idx = np.argsort(-min_d2)[: min(80, len(min_d2))]

    ang_cands = [0.0, math.radians(24.5), math.radians(40.97), math.radians(65.5)]
    scored = []
    for idx in top_idx:
        px, py = float(pts_x[idx]), float(pts_y[idx])
        best_e = 1e18
        best_ang = 0.0
        # Evaluate SAT penalty against nearby squares
        near = np.where((np.abs(x - px) < 1.8) & (np.abs(y - py) < 1.8))[0]
        for ang in ang_cands:
            ci, si = math.cos(ang), math.sin(ang)
            hxi = 0.5 * (abs(ci) + abs(si))
            e = 0.0
            for v in (hxi - px, px + hxi - L, hxi - py, py + hxi - L):
                if v > 0.0:
                    e += 2.0 * v * v
            for j in near:
                ddx, ddy = x[j] - px, y[j] - py
                cj, sj = math.cos(th[j]), math.sin(th[j])
                rad = 0.5 * (1.0 + abs(ci * cj + si * sj) + abs(ci * sj - si * cj))
                p1 = rad - abs(ddx * ci + ddy * si)
                if p1 <= 0.0:
                    continue
                p2 = rad - abs(-ddx * si + ddy * ci)
                if p2 <= 0.0:
                    continue
                p3 = rad - abs(ddx * cj + ddy * sj)
                if p3 <= 0.0:
                    continue
                p4 = rad - abs(-ddx * sj + ddy * cj)
                if p4 <= 0.0:
                    continue
                pen = min(p1, p2, p3, p4)
                e += pen * pen
            if e < best_e:
                best_e = e
                best_ang = ang
        scored.append((best_e, px, py, best_ang))
    scored.sort(key=lambda t: t[0])
    # Deduplicate spatially so we get diverse sites
    chosen = []
    for e, px, py, ang in scored:
        if all((px - cx) ** 2 + (py - cy) ** 2 > 1.2 for cx, cy, _ in chosen):
            chosen.append((px, py, ang))
            if len(chosen) >= top_k:
                break
    return chosen


def _contract_donor_k(
    x: np.ndarray, y: np.ndarray, th: np.ndarray, L_src: float, dn_remove: int, cut_x: float, cut_y: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Contract a k_src packing by removing 1 x-strip + 1 y-strip (dn_remove squares) along (cut_x, cut_y)."""
    deg90 = np.degrees(th) % 90.0
    is_up = (deg90 <= 1.5) | (deg90 >= 88.5)
    # Distance to the L-cut cross: prefer upright squares closest to cut_x or cut_y
    dist_x = np.abs(x - cut_x)
    dist_y = np.abs(y - cut_y)
    score = np.minimum(dist_x, dist_y) + np.where(is_up, 0.0, 2.5)
    rem = np.argsort(score)[:dn_remove]
    keep = np.setdiff1d(np.arange(len(x)), rem)
    kx = x[keep].copy()
    ky = y[keep].copy()
    kth = th[keep].copy()
    # Shift coordinates past the cuts by -1.0
    kx[kx > cut_x] -= 1.0
    ky[ky > cut_y] -= 1.0
    return kx, ky, kth


def _contract_donor_2k(
    x: np.ndarray, y: np.ndarray, th: np.ndarray, dn_remove: int, cx1: float, cy1: float, cx2: float, cy2: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Contract a (k+2) packing to k by removing two x-strips and two y-strips."""
    deg90 = np.degrees(th) % 90.0
    is_up = (deg90 <= 1.5) | (deg90 >= 88.5)
    score = np.minimum(
        np.minimum(np.abs(x - cx1), np.abs(x - cx2)),
        np.minimum(np.abs(y - cy1), np.abs(y - cy2)),
    ) + np.where(is_up, 0.0, 3.0)
    rem = np.argsort(score)[:dn_remove]
    keep = np.setdiff1d(np.arange(len(x)), rem)
    kx = x[keep].copy()
    ky = y[keep].copy()
    kth = th[keep].copy()
    kx = kx - (kx > cx1).astype(float) - (kx > cx2).astype(float)
    ky = ky - (ky > cy1).astype(float) - (ky > cy2).astype(float)
    return kx, ky, kth


def build_seeds_581(df: pd.DataFrame) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]]:
    seeds = []
    # 1. Contract 630 (25, 5, 40.97 deg, L=25.591378) -> 581 (24, 5)
    x630, y630, th630, L630 = _load_donor(630, df)
    for tag, cx, cy in [
        ("c630_mid", 11.4, 16.9),
        ("c630_br", 25.09, 0.5),
        ("c630_tl", 0.5, 25.09),
        ("c630_tr", 25.09, 25.09),
        ("c630_bl", 0.5, 0.5),
        ("c630_m2", 11.4, 25.09),
        ("c630_m3", 25.09, 16.9),
        ("c630_m4", 15.5, 11.5),
    ]:
        kx, ky, kth = _contract_donor_k(x630, y630, th630, L630, 49, cx, cy)
        seeds.append((tag, kx, ky, kth, None))

    # 1b. Exact component-preserving 630 -> 581: keep comp0 (33) shifted (0,-1) + comp1 (56) unshifted + 492 upright!
    deg90 = np.degrees(th630) % 90.0
    t_mask = (deg90 > 1.0) & (deg90 < 89.0)
    comp0 = t_mask & (y630 > 16.85)  # 33 tilted squares
    comp1 = t_mask & (y630 <= 16.85)  # 56 tilted squares
    L_tgt = 24.591378
    for shift_mode, dy0, dx1 in [("c0y_m1", -1.0, 0.0), ("c1x_m1", 0.0, -1.0), ("both_m05", -0.5, -0.5)]:
        tx = np.concatenate([x630[comp0], x630[comp1] + dx1])
        ty = np.concatenate([y630[comp0] + dy0, y630[comp1]])
        tth = np.concatenate([th630[comp0], th630[comp1]])
        # Generate candidate upright positions on BL (0.5 + i) and TR (L_tgt - 0.5 - i)
        # Split by anti-diagonal x + y
        up_x, up_y = [], []
        for ix in range(24):
            for iy in range(24):
                # Test BL vs TR based on tilted squares' x,y
                ux_bl, uy_bl = 0.5 + ix, 0.5 + iy
                ux_tr, uy_tr = L_tgt - 0.5 - (23 - ix), L_tgt - 0.5 - (23 - iy)
                # Pick whichever is clearer of tilted squares, or test clearance
                for ux, uy in ((ux_bl, uy_bl), (ux_tr, uy_tr)):
                    # Check min SAT clearance against tilted squares
                    ddx = np.abs(tx - ux)
                    ddy = np.abs(ty - uy)
                    # rough SAT proxy
                    if np.all(np.maximum(ddx, ddy) > 1.10):
                        up_x.append(ux)
                        up_y.append(uy)
                        break
        # Pad or trim to 492 upright squares (581 - 89 = 492)
        need_up = 581 - len(tx)
        if len(up_x) < need_up:
            sites = _find_void_sites(
                np.concatenate([tx, np.array(up_x)]),
                np.concatenate([ty, np.array(up_y)]),
                np.concatenate([tth, np.zeros(len(up_x))]),
                L_tgt,
                top_k=need_up - len(up_x) + 10,
            )
            for px, py, _ in sites:
                if len(up_x) >= need_up:
                    break
                up_x.append(px)
                up_y.append(py)
            # Fallback if deduplicated sites were fewer than need_up
            rng_pad = np.random.default_rng(42)
            while len(up_x) < need_up:
                up_x.append(float(rng_pad.uniform(1.0, L_tgt - 1.0)))
                up_y.append(float(rng_pad.uniform(1.0, L_tgt - 1.0)))
        else:
            up_x, up_y = up_x[:need_up], up_y[:need_up]
        ax = np.concatenate([tx, np.array(up_x[:need_up])])
        ay = np.concatenate([ty, np.array(up_y[:need_up])])
        ath = np.concatenate([tth, np.zeros(need_up)])
        fmask = np.ones(581, dtype=np.int32)
        fmask[: len(tx)] = 0  # freeze tilted first or keep free? let's also add free copy!
        seeds.append((f"630tile_{shift_mode}_frz", ax.copy(), ay.copy(), ath.copy(), fmask))
        seeds.append((f"630tile_{shift_mode}_free", ax.copy(), ay.copy(), ath.copy(), None))

    # 2. Contract 681 (26, 5, 40.97 deg) -> 581 (24, 5)
    x681, y681, th681, _ = _load_donor(681, df)
    for tag, cx1, cy1, cx2, cy2 in [
        ("c681_ends", 0.5, 0.5, 26.0, 26.0),
        ("c681_mids", 5.5, 19.0, 15.0, 8.2),
    ]:
        kx, ky, kth = _contract_donor_2k(x681, y681, th681, 100, cx1, cy1, cx2, cy2)
        seeds.append((tag, kx, ky, kth, None))

    # 3. Expand 580 (24, 4, 39.65 deg, L=24.583782) -> 581 (24, 5) via top void insertions
    x580, y580, th580, L580 = _load_donor(580, df)
    sites580 = _find_void_sites(x580, y580, th580, L580, top_k=12)
    for k_i, (px, py, ang) in enumerate(sites580):
        seeds.append(
            (
                f"e580_v{k_i}",
                np.append(x580, px),
                np.append(y580, py),
                np.append(th580, ang),
                None,
            )
        )

    # 4. Contract 582 (24, 6, 24.51 deg, L=24.633983) -> 581 by deleting 1 tilted tip square
    x582, y582, th582, _ = _load_donor(582, df)
    deg582 = np.degrees(th582) % 90.0
    t582 = np.where((deg582 > 1.0) & (deg582 < 89.0))[0]
    for tag, idx in [
        ("d582_tminx", t582[np.argmin(x582[t582])]),
        ("d582_tmaxx", t582[np.argmax(x582[t582])]),
        ("d582_tminy", t582[np.argmin(y582[t582])]),
        ("d582_tmaxy", t582[np.argmax(y582[t582])]),
    ]:
        keep = np.setdiff1d(np.arange(582), [idx])
        seeds.append((tag, x582[keep].copy(), y582[keep].copy(), th582[keep].copy(), None))
    return seeds


def build_seeds_850(df: pd.DataFrame) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]]:
    seeds = []
    # 1. Expand 849 (29, 8, 24.10 deg, L=29.637029) -> 850 (29, 9) via top voids
    x849, y849, th849, L849 = _load_donor(849, df)
    sites849 = _find_void_sites(x849, y849, th849, L849, top_k=14)
    for k_i, (px, py, ang) in enumerate(sites849):
        seeds.append((f"e849_v{k_i}", np.append(x849, px), np.append(y849, py), np.append(th849, ang), None))

    # 2. Contract 909 (30, 9, L=30.652397) -> 850 (29, 9)
    x909, y909, th909, L909 = _load_donor(909, df)
    for tag, cx, cy in [
        ("c909_tr", 30.15, 30.15),
        ("c909_bl", 0.5, 0.5),
        ("c909_mid", 15.5, 15.5),
        ("c909_tl", 0.5, 30.15),
        ("c909_br", 30.15, 0.5),
        ("c909_m2", 18.5, 10.5),
    ]:
        kx, ky, kth = _contract_donor_k(x909, y909, th909, L909, 59, cx, cy)
        seeds.append((tag, kx, ky, kth, None))

    # 3. Contract 970 (31, 9, L=31.638982) -> 850 (29, 9)
    x970, y970, th970, _ = _load_donor(970, df)
    for tag, cx1, cy1, cx2, cy2 in [
        ("c970_ends", 0.5, 0.5, 31.1, 31.1),
        ("c970_mid", 14.5, 14.5, 22.5, 22.5),
    ]:
        kx, ky, kth = _contract_donor_2k(x970, y970, th970, 120, cx1, cy1, cx2, cy2)
        seeds.append((tag, kx, ky, kth, None))

    # 4. Delete 1 from 851 (29, 10, L=29.684144) -> 850
    x851, y851, th851, _ = _load_donor(851, df)
    deg851 = np.degrees(th851) % 90.0
    t851 = np.where((deg851 > 1.0) & (deg851 < 89.0))[0]
    u851 = np.where((deg851 <= 1.0) | (deg851 >= 89.0))[0]
    for tag, idx in [
        ("d851_tminx", t851[np.argmin(x851[t851])]),
        ("d851_tmaxx", t851[np.argmax(x851[t851])]),
        ("d851_tminy", t851[np.argmin(y851[t851])]),
        ("d851_tmaxy", t851[np.argmax(y851[t851])]),
        ("d851_uminx", u851[np.argmin(x851[u851])]),
        ("d851_umaxx", u851[np.argmax(x851[u851])]),
    ]:
        keep = np.setdiff1d(np.arange(851), [idx])
        seeds.append((tag, x851[keep].copy(), y851[keep].copy(), th851[keep].copy(), None))
    return seeds


def build_seeds_974(df: pd.DataFrame) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]]:
    seeds = []
    # 1. Expand 973 (31, 12, 24.45 deg, L=31.696609) -> 974 (31, 13)
    x973, y973, th973, L973 = _load_donor(973, df)
    sites973 = _find_void_sites(x973, y973, th973, L973, top_k=14)
    for k_i, (px, py, ang) in enumerate(sites973):
        seeds.append((f"e973_v{k_i}", np.append(x973, px), np.append(y973, py), np.append(th973, ang), None))

    # 2. Expand 972 (31, 11, 24.75 deg, L=31.683990) -> 974 (+2 squares)
    x972, y972, th972, L972 = _load_donor(972, df)
    sites972 = _find_void_sites(x972, y972, th972, L972, top_k=10)
    for k_i in range(min(6, len(sites972) // 2)):
        p1, p2 = sites972[2 * k_i], sites972[2 * k_i + 1]
        seeds.append(
            (
                f"e972_pair{k_i}",
                np.append(x972, [p1[0], p2[0]]),
                np.append(y972, [p1[1], p2[1]]),
                np.append(th972, [p1[2], p2[2]]),
                None,
            )
        )

    # 3. Delete 1 from 975 (31, 14, 19.09 deg, L=31.721540) -> 974
    x975, y975, th975, _ = _load_donor(975, df)
    deg975 = np.degrees(th975) % 90.0
    t975 = np.where((deg975 > 1.0) & (deg975 < 89.0))[0]
    u975 = np.where((deg975 <= 1.0) | (deg975 >= 89.0))[0]
    for tag, idx in [
        ("d975_tminx", t975[np.argmin(x975[t975])]),
        ("d975_tmaxx", t975[np.argmax(x975[t975])]),
        ("d975_tminy", t975[np.argmin(y975[t975])]),
        ("d975_tmaxy", t975[np.argmax(y975[t975])]),
        ("d975_umin", u975[np.argmin(x975[u975] + y975[u975])]),
        ("d975_umax", u975[np.argmax(x975[u975] + y975[u975])]),
    ]:
        keep = np.setdiff1d(np.arange(975), [idx])
        seeds.append((tag, x975[keep].copy(), y975[keep].copy(), th975[keep].copy(), None))
    return seeds


def _slp_worker(args: tuple) -> dict:
    tag, x, y, th, rounds = args
    t0 = time.perf_counter()
    sx, sy, sth, sL, sok = u.joint_xytheta_slp(x, y, th, rounds=rounds, tr_th0=0.03, tr_xy0=0.12)
    pen, bbox = u.sat_max_overlap_and_bbox(sx, sy, sth)
    return {
        "tag": tag,
        "ok": bool(sok and pen <= 1e-11),
        "L_bbox": float(bbox),
        "pen": float(pen),
        "x": sx,
        "y": sy,
        "th": sth,
        "dt": time.perf_counter() - t0,
    }


def solve_target(n: int, seeds: list, L_sched: np.ndarray, inc_L: Decimal, L_front: float) -> dict:
    t0 = time.perf_counter()
    a_sched = np.array([0.92, 0.95, 0.975, 0.99, 0.997, 1.0], dtype=np.float64)
    iters = np.array([180, 180, 180, 200, 220, 320], dtype=np.int32)

    print(f"\n=== Target n={n} (inc={inc_L}, front={L_front:.6f}), {len(seeds)} seeds ===")
    melt_res = run_cpp_melt_batch(seeds, L_sched, a_sched, iters, ils_jitter_rounds=8, n_threads=4, base_seed=1000 + n)

    feas_cands = [r for r in melt_res if r["has_feas"]]
    near_cands = sorted(melt_res, key=lambda r: (r["pen"], r["L_bbox"]))[:8]
    print(f"  C++ melt done in {time.perf_counter()-t0:.1f}s: {len(feas_cands)}/{len(melt_res)} feas")
    for r in sorted(feas_cands, key=lambda r: r["L_feas"])[:5]:
        print(f"    feas {r['tag']:18s}: L_feas={r['L_feas']:.9f} (d_inc={r['L_feas']-float(inc_L):+.6f})")
    if not feas_cands:
        for r in near_cands[:4]:
            print(f"    near {r['tag']:18s}: L_bbox={r['L_bbox']:.6f}, pen={r['pen']:.3e}, E={r['E']:.3e}")

    # Select top candidates (up to 12) for joint (x, y, dtheta) HiGHS SLP
    slp_pool = sorted(feas_cands, key=lambda r: r["L_feas"])[:8]
    seen_tags = {r["tag"] for r in slp_pool}
    for r in near_cands:
        if r["tag"] not in seen_tags and len(slp_pool) < 12:
            slp_pool.append(r)
            seen_tags.add(r["tag"])

    tasks = [(r["tag"], r["x"], r["y"], r["th"], 16) for r in slp_pool]
    slp_res = []
    with ProcessPoolExecutor(max_workers=4) as ex:
        futs = [ex.submit(_slp_worker, t) for t in tasks]
        for fut in as_completed(futs):
            slp_res.append(fut.result())

    best_ver = None
    best_cand = None
    for sr in sorted(slp_res, key=lambda r: (0 if r["ok"] else 1, r["L_bbox"])):
        if sr["ok"]:
            ver = u.verify_official(n, sr["x"], sr["y"], sr["th"])
            if ver["ok"]:
                if best_ver is None or ver["L_lit_d"] < best_ver["L_lit_d"]:
                    best_ver = ver
                    best_cand = sr
                print(
                    f"    SLP {sr['tag']:18s}: L_lit={ver['L_lit']} L_pd={ver['L_pandas']} "
                    f"(vs inc={ver['L_lit_d'] - inc_L:+.9f}) [{sr['dt']:.1f}s]"
                )
        else:
            print(f"    SLP-infeas {sr['tag']:18s}: L={sr['L_bbox']:.6f} pen={sr['pen']:.3e} [{sr['dt']:.1f}s]")

    win = False
    if best_ver is not None and best_ver["L_lit_d"] < inc_L and best_ver["L_pd_d"] < inc_L:
        win = True
        data_dir = ROOT_DIR / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        win_path = data_dir / f"win_n{n}.csv"
        pd.DataFrame(
            [
                {"n": n, "square_id": r["square_id"], "x": r["x"], "y": r["y"], "angle": r["angle"]}
                for r in best_ver["rows"]
            ]
        ).to_csv(win_path, index=False)
        u.write_final_csv_from_wins(data_dir, ROOT_DIR / "final.csv")
        print(f"  >>> CERTIFIED WIN n={n}: L_lit={best_ver['L_lit']} < inc={inc_L}! Saved {win_path}")

    return {
        "n": n,
        "inc_L": str(inc_L),
        "L_front": L_front,
        "n_seeds": len(seeds),
        "n_melt_feas": len(feas_cands),
        "best_tag": best_cand["tag"] if best_cand else None,
        "best_L_lit": best_ver["L_lit"] if best_ver else None,
        "best_L_pd": best_ver["L_pandas"] if best_ver else None,
        "win": win,
        "elapsed_s": round(time.perf_counter() - t0, 2),
    }


def main():
    df = pd.read_csv(REPO_ROOT / "poc" / "pell_corrugation" / "data" / "border-scan" / "merged.csv")
    gap_df = pd.read_csv(REPO_ROOT / "poc" / "frontier_nov12" / "docs" / "per_n_gap.csv").set_index("n")

    configs = [
        (581, build_seeds_581(df), np.array([24.64, 24.615, 24.602, 24.594])),
        (850, build_seeds_850(df), np.array([29.69, 29.668, 29.654, 29.644])),
        (974, build_seeds_974(df), np.array([31.73, 31.712, 31.703, 31.697])),
    ]

    summary = []
    for n, seeds, L_sched in configs:
        inc_L, _, _, _ = u.load_incumbent_exact(n, df)
        Lf = float(gap_df.loc[n, "L_frontier"])
        res = solve_target(n, seeds, L_sched, inc_L, Lf)
        summary.append(res)

    docs_dir = ROOT_DIR / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    out_json = docs_dir / "idea1_results.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print("\nSummary written to", out_json)


if __name__ == "__main__":
    main()
