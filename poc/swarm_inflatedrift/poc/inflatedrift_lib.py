"""Coupled Inflation (Over-Inflation & Soft-a C++ Melt) + WallDrift + 4-Corner PSLP + LPSlack + Mechwalk."""
from __future__ import annotations

import ctypes
import math
import os
import subprocess
import sys
import time
from decimal import Decimal
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

POC_DIR = Path(__file__).resolve().parent
REPO_ROOT = POC_DIR.parents[2]
LPSLACK_POC = REPO_ROOT / "poc" / "swarm_lpslack" / "poc"
MECHWALK_POC = REPO_ROOT / "poc" / "swarm_mechwalk" / "poc"
HIGHD_POC = REPO_ROOT / "poc" / "swarm_highd" / "poc"

for _p in (POC_DIR, LPSLACK_POC, MECHWALK_POC, HIGHD_POC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import arm_model as am  # noqa: E402
import lpslack  # noqa: E402
import mechwalk_lib as ml  # noqa: E402
import pslp_4c as pslp  # noqa: E402

CPP_PATH = POC_DIR / "inflatedrift_cpp.cpp"
SO_PATH = Path("/tmp/libinflatedrift.so")
_CPP = None


def load_cpp_lib():
    """Compile (if needed) and load /tmp/libinflatedrift.so."""
    if (not SO_PATH.exists()) or (CPP_PATH.stat().st_mtime > SO_PATH.stat().st_mtime):
        cmd = [
            "g++", "-O3", "-march=native", "-shared", "-fPIC", "-std=c++17",
            "-pthread", "-Wall", "-Wextra", "-Werror",
            str(CPP_PATH), "-o", str(SO_PATH),
        ]
        subprocess.run(cmd, check=True)
    lib = ctypes.CDLL(str(SO_PATH))
    c_dp = ctypes.POINTER(ctypes.c_double)
    c_ip = ctypes.POINTER(ctypes.c_int)
    c_u8p = ctypes.POINTER(ctypes.c_uint8)

    lib.expand_sym_coords.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_double, c_dp, c_dp]
    lib.expand_sym_coords.restype = None

    lib.eval_sat_pen_bbox.argtypes = [ctypes.c_int, c_dp, c_dp, c_dp, ctypes.c_double, c_dp]
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


def sat_pen_bbox(x: np.ndarray, y: np.ndarray, th: np.ndarray, side: float = 1.0) -> tuple[float, float]:
    xa = np.ascontiguousarray(x, dtype=np.float64)
    ya = np.ascontiguousarray(y, dtype=np.float64)
    tha = np.ascontiguousarray(th, dtype=np.float64)
    out_L = ctypes.c_double(0.0)
    pen = cpp().eval_sat_pen_bbox(
        len(xa),
        xa.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ya.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        tha.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        float(side),
        ctypes.byref(out_L),
    )
    return float(pen), float(out_L.value)


def cpp_batch_melt(
    coords_list: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    L_sched_list: list[list[float]],
    a_start: float,
    n_melt_stages: int = 4,
    seam_masks: list[np.ndarray] | None = None,
    seam_iters: int = 160,
    full_iters: int = 200,
    n_ruin_iters: int = 4,
    drift_xy_list: list[np.ndarray] | None = None,
    drift_strengths: list[float] | None = None,
    n_threads: int = 1,
    base_seed: int = 1,
) -> list[dict]:
    B = len(coords_list)
    if B == 0:
        return []
    n = len(coords_list[0][0])
    dim = 3 * n
    n_L = len(L_sched_list[0])
    q_batch = np.zeros((B, dim), dtype=np.float64)
    q_feas = np.zeros((B, dim), dtype=np.float64)
    L_sched_arr = np.asarray(L_sched_list, dtype=np.float64)

    for b, (x, y, th) in enumerate(coords_list):
        hc = 0.5 * L_sched_arr[b, 0]
        q_batch[b, :n] = x - hc
        q_batch[b, n : 2 * n] = y - hc
        q_batch[b, 2 * n :] = th

    has_sm = 1 if seam_masks is not None else 0
    if has_sm:
        sm_arr = np.zeros((B, dim), dtype=np.uint8)
        for b, m in enumerate(seam_masks):
            mb = np.asarray(m, dtype=np.uint8)
            sm_arr[b, :n] = mb
            sm_arr[b, n : 2 * n] = mb
            sm_arr[b, 2 * n :] = mb
    else:
        sm_arr = np.zeros(1, dtype=np.uint8)

    has_dr = 1 if (drift_xy_list is not None and drift_strengths is not None) else 0
    if has_dr:
        dr_arr = np.ascontiguousarray(np.vstack(drift_xy_list), dtype=np.float64)
        ds_arr = np.ascontiguousarray(drift_strengths, dtype=np.float64)
    else:
        dr_arr = np.zeros(1, dtype=np.float64)
        ds_arr = np.zeros(1, dtype=np.float64)

    out_best_L = np.zeros(B, dtype=np.float64)
    out_E = np.zeros(B, dtype=np.float64)
    out_pen = np.zeros(B, dtype=np.float64)
    out_Lb = np.zeros(B, dtype=np.float64)

    cpp().batch_melt_solve(
        B, n, 1,
        q_batch.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        q_feas.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        sm_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), has_sm,
        dr_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), has_dr,
        ds_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        n_L, L_sched_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        float(a_start), int(n_melt_stages),
        int(seam_iters), int(full_iters), int(n_ruin_iters),
        int(n_threads), int(base_seed),
        out_best_L.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out_E.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out_pen.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out_Lb.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )

    results: list[dict] = []
    L_fin = float(L_sched_arr[0, -1])
    z_full = np.zeros(dim, dtype=np.float64)
    for b in range(B):
        use_q = q_feas[b] if out_best_L[b] < 1e17 else q_batch[b]
        cpp().expand_sym_coords(
            n, 1, float(L_sched_arr[b, -1]),
            use_q.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            z_full.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        nx, ny, nth = z_full[:n].copy(), z_full[n : 2 * n].copy(), z_full[2 * n :].copy()
        nx, ny, nth, Lb = ml.normalize(nx, ny, nth)
        pen, _ = sat_pen_bbox(nx, ny, nth)
        results.append({
            "x": nx, "y": ny, "th": nth,
            "pen": float(pen), "L_bbox": float(Lb),
            "best_L_feas": float(out_best_L[b]), "E": float(out_E[b]),
        })
    return results


def cpp_batch_masked_melt(
    coords_list: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    masks_list: list[np.ndarray],
    L_sched_list: list[list[float]],
    ils_iters_per_L: list[int],
    a_start: float = 0.82,
    n_melt_stages: int = 4,
    masked_iters: int = 160,
    full_iters: int = 200,
    drift_xy_list: list[np.ndarray] | None = None,
    drift_strengths: list[float] | None = None,
    n_threads: int = 1,
    base_seed: int = 1,
) -> list[dict]:
    B = len(coords_list)
    if B == 0:
        return []
    n = len(coords_list[0][0])
    dim = 3 * n
    n_L = len(L_sched_list[0])
    q_batch = np.zeros((B, dim), dtype=np.float64)
    q_feas = np.zeros((B, dim), dtype=np.float64)
    mask_arr = np.zeros((B, dim), dtype=np.uint8)
    L_sched_arr = np.asarray(L_sched_list, dtype=np.float64)
    ils_arr = np.asarray(ils_iters_per_L, dtype=np.int32)

    for b, ((x, y, th), m) in enumerate(zip(coords_list, masks_list)):
        hc = 0.5 * L_sched_arr[b, 0]
        q_batch[b, :n] = x - hc
        q_batch[b, n : 2 * n] = y - hc
        q_batch[b, 2 * n :] = th
        mb = np.asarray(m, dtype=np.uint8)
        if mb.size == dim:
            mask_arr[b] = mb
        else:
            mask_arr[b, :n] = mb
            mask_arr[b, n : 2 * n] = mb
            mask_arr[b, 2 * n :] = mb

    has_dr = 1 if (drift_xy_list is not None and drift_strengths is not None) else 0
    if has_dr:
        dr_arr = np.ascontiguousarray(np.vstack(drift_xy_list), dtype=np.float64)
        ds_arr = np.ascontiguousarray(drift_strengths, dtype=np.float64)
    else:
        dr_arr = np.zeros(1, dtype=np.float64)
        ds_arr = np.zeros(1, dtype=np.float64)

    out_best_L = np.zeros(B, dtype=np.float64)
    out_E = np.zeros(B, dtype=np.float64)
    out_pen = np.zeros(B, dtype=np.float64)
    out_Lb = np.zeros(B, dtype=np.float64)

    cpp().batch_masked_melt_solve(
        B, n, 1,
        q_batch.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        q_feas.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        mask_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        dr_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), has_dr,
        ds_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        n_L, L_sched_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ils_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        float(a_start), int(n_melt_stages),
        int(masked_iters), int(full_iters),
        int(n_threads), int(base_seed),
        out_best_L.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out_E.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out_pen.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out_Lb.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )

    results: list[dict] = []
    z_full = np.zeros(dim, dtype=np.float64)
    for b in range(B):
        use_q = q_feas[b] if out_best_L[b] < 1e17 else q_batch[b]
        cpp().expand_sym_coords(
            n, 1, float(L_sched_arr[b, -1]),
            use_q.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            z_full.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        nx, ny, nth = z_full[:n].copy(), z_full[n : 2 * n].copy(), z_full[2 * n :].copy()
        nx, ny, nth, Lb = ml.normalize(nx, ny, nth)
        pen, _ = sat_pen_bbox(nx, ny, nth)
        results.append({
            "x": nx, "y": ny, "th": nth,
            "pen": float(pen), "L_bbox": float(Lb),
            "best_L_feas": float(out_best_L[b]), "E": float(out_E[b]),
        })
    return results


def clearance_points(
    x: np.ndarray, y: np.ndarray, th: np.ndarray, L: float, n_pts: int = 8, step: float = 0.16
) -> list[tuple[float, float, float]]:
    g = np.arange(0.5, L - 0.5 + 1e-9, step)
    gx, gy = np.meshgrid(g, g, indexing="ij")
    px, py = gx.ravel(), gy.ravel()
    clr = np.minimum(np.minimum(px, L - px), np.minimum(py, L - py)) - 0.5
    c, s = np.cos(th), np.sin(th)
    for a in range(0, len(px), 4096):
        dx = px[a : a + 4096, None] - x[None, :]
        dy = py[a : a + 4096, None] - y[None, :]
        d = np.maximum(
            np.abs(c[None, :] * dx + s[None, :] * dy),
            np.abs(-s[None, :] * dx + c[None, :] * dy),
        ) - 0.5
        clr[a : a + 4096] = np.minimum(clr[a : a + 4096], d.min(axis=1))
    order = np.argsort(-clr)
    out: list[tuple[float, float, float]] = []
    for i in order:
        if len(out) >= n_pts:
            break
        if all((px[i] - q[0]) ** 2 + (py[i] - q[1]) ** 2 > 0.85 ** 2 for q in out):
            out.append((float(px[i]), float(py[i]), float(clr[i])))
    return out


def analyze_walls(x: np.ndarray, y: np.ndarray, th: np.ndarray, L: float) -> dict:
    hx = 0.5 * (np.abs(np.cos(th)) + np.abs(np.sin(th)))
    margins = {"L": x - hx, "R": L - (x + hx), "B": y - hx, "T": L - (y + hx)}
    inward = {"L": (1.0, 0.0), "R": (-1.0, 0.0), "B": (0.0, 1.0), "T": (0.0, -1.0)}
    opp = {"L": "R", "R": "L", "B": "T", "T": "B"}
    c_tight = {w: int(np.sum(m < 1e-4)) for w, m in margins.items()}
    weak_w = min(margins.keys(), key=lambda w: c_tight[w])
    heavy_w = opp[weak_w]
    weak_idx = np.where(margins[weak_w] < 5e-4)[0]
    if len(weak_idx) == 0:
        weak_idx = np.argsort(margins[weak_w])[:4]
    heavy_idx = np.where(margins[heavy_w] < 5e-4)[0]
    if len(heavy_idx) == 0:
        heavy_idx = np.argsort(margins[heavy_w])[:8]
    return {
        "counts": c_tight,
        "weak_wall": weak_w,
        "heavy_wall": heavy_w,
        "weak_idx": weak_idx,
        "heavy_idx": heavy_idx,
        "inward": inward[weak_w],
        "opp_inward": inward[heavy_w],
        "margins": margins,
    }


def _cap_mask(
    x: np.ndarray, y: np.ndarray, cx: float, cy: float, rad: float,
    must_include: np.ndarray | None = None, min_sq: int = 35, max_sq: int = 95,
) -> np.ndarray:
    d2 = (x - cx) ** 2 + (y - cy) ** 2
    mask = d2 <= rad * rad
    if must_include is not None and len(must_include) > 0:
        mask[must_include] = True
    if int(mask.sum()) > max_sq:
        order = np.argsort(d2)
        mask = np.zeros(len(x), dtype=bool)
        if must_include is not None and len(must_include) > 0:
            mask[must_include] = True
        for i in order:
            if int(mask.sum()) >= max_sq:
                break
            mask[i] = True
    elif int(mask.sum()) < min_sq:
        mask[np.argsort(d2)[: min(len(x), min_sq)]] = True
    return mask


def build_overinflate_drift_tasks(
    n: int, x_b: np.ndarray, y_b: np.ndarray, th_b: np.ndarray, L_b: float, rng: np.random.Generator
) -> tuple[list[dict], list[dict]]:
    """Build full-box and masked-pocket over-inflation (a_start > 1.0) + directional wall-drift tasks."""
    wb = analyze_walls(x_b, y_b, th_b, L_b)
    voids = clearance_points(x_b, y_b, th_b, L_b, n_pts=6, step=0.18)
    deg = np.degrees(th_b) % 90.0
    tilted_idx = np.where((deg > 2.0) & (deg < 88.0))[0]
    nx_in, ny_in = wb["inward"]
    ox_in, oy_in = wb["opp_inward"]

    full_tasks: list[dict] = []
    a_vals = (1.0008, 1.0018, 1.0032)
    for idx_s in range(12):
        a_start = a_vals[idx_s % len(a_vals)]
        L0 = L_b + [0.0015, 0.0040, 0.0080][idx_s % 3]
        L_fin = L_b + [0.0002, -0.0003][(idx_s // 3) % 2]
        sched = [L0, 0.5 * (L0 + L_fin), L_fin]
        sc = L0 / L_b
        x, y, th = x_b * sc, y_b * sc, th_b.copy()

        fx = np.zeros(n, dtype=np.float64)
        fy = np.zeros(n, dtype=np.float64)
        mode = idx_s % 3
        if mode == 0:
            cx_m, cy_m = 0.5 * L0, 0.5 * L0
            dx, dy = cx_m - x, cy_m - y
            norm = np.maximum(1.0, np.hypot(dx, dy))
            fx, fy = dx / norm, dy / norm
            if len(tilted_idx) > 0:
                th[tilted_idx] += rng.uniform(-0.020, 0.020, len(tilted_idx))
            x += rng.uniform(-0.010, 0.010, n)
            y += rng.uniform(-0.010, 0.010, n)
            strength = 0.05
        elif mode == 1:
            fx[wb["weak_idx"]] = nx_in
            fy[wb["weak_idx"]] = ny_in
            fx[wb["heavy_idx"]] = 0.5 * ox_in
            fy[wb["heavy_idx"]] = 0.5 * oy_in
            if len(tilted_idx) > 0:
                pi = int(tilted_idx[(idx_s * 5) % len(tilted_idx)])
                near = ((x - x[pi]) ** 2 + (y - y[pi]) ** 2) <= 5.5 ** 2
                x[near] += rng.uniform(-0.03, 0.03, int(near.sum()))
                y[near] += rng.uniform(-0.03, 0.03, int(near.sum()))
                th[near] += rng.uniform(-0.05, 0.05, int(near.sum()))
            strength = 0.07
        else:
            vx, vy, _ = voids[(idx_s // 3) % len(voids)] if voids else (0.5 * L0, 0.5 * L0, 0.0)
            near = ((x - vx) ** 2 + (y - vy) ** 2) <= 6.0 ** 2
            dx, dy = vx - x[near], vy - y[near]
            norm = np.maximum(0.5, np.hypot(dx, dy))
            fx[near], fy[near] = dx / norm, dy / norm
            x[near] += rng.uniform(-0.04, 0.04, int(near.sum()))
            y[near] += rng.uniform(-0.04, 0.04, int(near.sum()))
            th[near] += rng.uniform(-0.06, 0.06, int(near.sum()))
            strength = 0.06

        full_tasks.append({
            "x": x, "y": y, "th": th,
            "a_start": a_start,
            "sched": sched,
            "drift": np.concatenate([fx, fy]),
            "strength": strength,
            "label": f"overinf_m{mode}_a{a_start:.4f}_{idx_s}",
        })

    # Masked-pocket over-inflation (keeps exterior locked while over-inflated pocket buckles)
    mpk_tasks: list[dict] = []
    for idx_s in range(8):
        a_start = (1.0015, 1.0030)[idx_s % 2]
        L0 = L_b + 0.004
        sched = [L0, L_b + 0.0005, L_b - 0.0002]
        sc = L0 / L_b
        x, y, th = x_b * sc, y_b * sc, th_b.copy()
        fx = np.zeros(n, dtype=np.float64)
        fy = np.zeros(n, dtype=np.float64)
        if idx_s % 2 == 0 and len(tilted_idx) > 0:
            pi = int(tilted_idx[(idx_s * 7) % len(tilted_idx)])
            cx, cy = float(x[pi]), float(y[pi])
            near = _cap_mask(x, y, cx, cy, 6.0)
        else:
            wi = int(wb["weak_idx"][idx_s % len(wb["weak_idx"])])
            near = _cap_mask(x, y, float(x[wi]), float(y[wi]), 6.0, must_include=wb["weak_idx"])
            fx[wb["weak_idx"]], fy[wb["weak_idx"]] = nx_in, ny_in
        cnt = int(near.sum())
        x[near] += rng.uniform(-0.04, 0.04, cnt)
        y[near] += rng.uniform(-0.04, 0.04, cnt)
        th[near] += rng.uniform(-0.08, 0.08, cnt)
        mpk_tasks.append({
            "x": x, "y": y, "th": th, "mask": near,
            "a_start": a_start, "sched": sched,
            "drift": np.concatenate([fx, fy]), "strength": 0.07,
            "label": f"overinf_mpk_a{a_start:.4f}_{idx_s}",
        })
    return full_tasks, mpk_tasks


def build_soft_walldrift_tasks(
    n: int, x_b: np.ndarray, y_b: np.ndarray, th_b: np.ndarray, L_b: float, rng: np.random.Generator
) -> tuple[list[dict], list[dict]]:
    """Build masked-pocket + full-box soft-a melt + wall-drift + strip-release tasks."""
    wb = analyze_walls(x_b, y_b, th_b, L_b)
    voids = clearance_points(x_b, y_b, th_b, L_b, n_pts=8, step=0.16)
    deg = np.degrees(th_b) % 90.0
    tilted_idx = np.where((deg > 2.0) & (deg < 88.0))[0]
    tilt = math.radians(float(np.median(deg[(deg > 1.5) & (deg < 88.5)]))) if len(tilted_idx) else math.pi / 4.0
    nx_in, ny_in = wb["inward"]
    ox_in, oy_in = wb["opp_inward"]

    masked_tasks: list[dict] = []
    L0_mpk = L_b + 0.024
    sc_m = L0_mpk / L_b
    xm0, ym0, tm0 = x_b * sc_m, y_b * sc_m, th_b.copy()
    sched_m = [L0_mpk, L_b + 0.004, L_b - 0.0005]

    for s in range(16):
        x, y, th = xm0.copy(), ym0.copy(), tm0.copy()
        fx = np.zeros(n, dtype=np.float64)
        fy = np.zeros(n, dtype=np.float64)
        mode = s % 4
        if mode == 0:
            # Sparse-wall -> internal void teleport + melt
            wi = int(wb["weak_idx"][s % len(wb["weak_idx"])])
            wx, wy = float(x[wi]), float(y[wi])
            vx, vy, _ = voids[s % len(voids)] if voids else (0.5 * L0_mpk, 0.5 * L0_mpk, 0.0)
            cx, cy = 0.45 * wx + 0.55 * vx, 0.45 * wy + 0.55 * vy
            near = _cap_mask(x, y, cx, cy, float(rng.uniform(5.0, 7.2)), must_include=wb["weak_idx"])
            for j in wb["weak_idx"]:
                dx, dy = vx - float(x[j]), vy - float(y[j])
                nm = max(1e-6, math.hypot(dx, dy))
                fx[j] = 0.7 * (dx / nm) + 0.3 * nx_in
                fy[j] = 0.7 * (dy / nm) + 0.3 * ny_in
            if s % 8 < 4:
                x[wi] = vx + float(rng.uniform(-0.08, 0.08))
                y[wi] = vy + float(rng.uniform(-0.08, 0.08))
                th[wi] = [0.0, math.pi / 4.0, tilt][(s // 4) % 3] + float(rng.uniform(-0.08, 0.08))
                near |= ((x - wx) ** 2 + (y - wy) ** 2) <= 3.5 ** 2
            strength = 0.09
            lbl = f"wd_mpk_void_{s}"
        elif mode == 1:
            # Weak-wall inward drift pocket
            wi = int(wb["weak_idx"][s % len(wb["weak_idx"])])
            near = _cap_mask(x, y, float(x[wi]), float(y[wi]), float(rng.uniform(5.0, 7.2)), must_include=wb["weak_idx"])
            fx[near], fy[near] = 0.5 * nx_in, 0.5 * ny_in
            fx[wb["weak_idx"]], fy[wb["weak_idx"]] = nx_in, ny_in
            strength = 0.10
            lbl = f"wd_mpk_wall_{s}"
        elif mode == 2:
            # Tilted-arm / upright seam unlock pocket
            if len(tilted_idx) > 0:
                pi = int(tilted_idx[(s * 11) % len(tilted_idx)])
                cx, cy = float(x[pi]), float(y[pi])
            else:
                cx, cy = 0.5 * L0_mpk, 0.5 * L0_mpk
            near = _cap_mask(x, y, cx, cy, float(rng.uniform(5.2, 7.5)))
            if voids:
                vx, vy, _ = voids[0]
                dx, dy = vx - x[near], vy - y[near]
                nm = np.maximum(0.5, np.hypot(dx, dy))
                fx[near], fy[near] = 0.5 * dx / nm, 0.5 * dy / nm
            strength = 0.06
            lbl = f"wd_mpk_seam_{s}"
        else:
            # Heavy-wall push toward weak wall
            hi = int(wb["heavy_idx"][(s * 3) % len(wb["heavy_idx"])])
            near = _cap_mask(x, y, float(x[hi]), float(y[hi]), float(rng.uniform(5.0, 7.4)))
            fx[near], fy[near] = 0.7 * ox_in, 0.7 * oy_in
            strength = 0.08
            lbl = f"wd_mpk_opp_{s}"

        cnt = int(near.sum())
        x[near] += rng.uniform(-0.12, 0.12, cnt)
        y[near] += rng.uniform(-0.12, 0.12, cnt)
        th[near] += rng.uniform(-0.22, 0.22, cnt)
        masked_tasks.append({
            "x": x, "y": y, "th": th, "mask": near,
            "drift": np.concatenate([fx, fy]), "strength": strength,
            "sched": sched_m, "label": lbl,
        })

    # Full-box soft-a melt tasks (0.90 -> 1.00) with wall-drift & global strip release
    full_tasks: list[dict] = []
    L0_f = L_b + 0.020
    sc_f = L0_f / L_b
    xf0, yf0, tf0 = x_b * sc_f, y_b * sc_f, th_b.copy()
    sched_f = [L0_f, L_b + 0.002, L_b - 0.0005]
    for s in range(12):
        x, y, th = xf0.copy(), yf0.copy(), tf0.copy()
        fx = np.zeros(n, dtype=np.float64)
        fy = np.zeros(n, dtype=np.float64)
        if s % 3 == 0:
            # Asymmetric wall-slack shift
            drift_step = float(rng.uniform(0.03, 0.07))
            if abs(nx_in) > 0.5:
                loaded = (x > L0_f * 0.30) if nx_in > 0 else (x < L0_f * 0.70)
                x[loaded] -= nx_in * drift_step
                fx[wb["weak_idx"]] = nx_in
                fx[loaded] = -0.35 * nx_in
            else:
                loaded = (y > L0_f * 0.30) if ny_in > 0 else (y < L0_f * 0.70)
                y[loaded] -= ny_in * drift_step
                fy[wb["weak_idx"]] = ny_in
                fy[loaded] = -0.35 * ny_in
            near = ((x - 0.5 * L0_f) ** 2 + (y - 0.5 * L0_f) ** 2) <= 7.0 ** 2
            x[near] += rng.uniform(-0.10, 0.10, int(near.sum()))
            y[near] += rng.uniform(-0.10, 0.10, int(near.sum()))
            th[near] += rng.uniform(-0.14, 0.14, int(near.sum()))
            strength = 0.08
            lbl = f"soft_drift_{s}"
        elif s % 3 == 1:
            # Tilted-arm + boundary junction melt
            if len(tilted_idx) > 0:
                pi = int(tilted_idx[(s * 7) % len(tilted_idx)])
                near = ((x - x[pi]) ** 2 + (y - y[pi]) ** 2) <= 6.5 ** 2
            else:
                near = np.ones(n, dtype=bool)
            x[near] += rng.uniform(-0.10, 0.10, int(near.sum()))
            y[near] += rng.uniform(-0.10, 0.10, int(near.sum()))
            th[near] += rng.uniform(-0.16, 0.16, int(near.sum()))
            strength = 0.0
            lbl = f"soft_tilted_{s}"
        else:
            # Void collapse pull
            vx, vy, _ = voids[(s // 3) % len(voids)] if voids else (0.5 * L0_f, 0.5 * L0_f, 0.0)
            near = ((x - vx) ** 2 + (y - vy) ** 2) <= 6.5 ** 2
            dx, dy = vx - x[near], vy - y[near]
            nm = np.maximum(0.5, np.hypot(dx, dy))
            fx[near], fy[near] = dx / nm, dy / nm
            x[near] += 0.05 * dx + rng.uniform(-0.09, 0.09, int(near.sum()))
            y[near] += 0.05 * dy + rng.uniform(-0.09, 0.09, int(near.sum()))
            th[near] += rng.uniform(-0.14, 0.14, int(near.sum()))
            strength = 0.07
            lbl = f"soft_void_{s}"
        full_tasks.append({
            "x": x, "y": y, "th": th,
            "drift": np.concatenate([fx, fy]), "strength": strength,
            "sched": sched_f, "label": lbl,
        })

    return masked_tasks, full_tasks


def coupled_pslp_lpslack_polish(
    n: int,
    x: np.ndarray,
    y: np.ndarray,
    th: np.ndarray,
    old_L: Decimal | None = None,
    use_reduced: bool = True,
    pslp_iters: int = 160,
    run_lpslack: bool = True,
    lpslack_time_limit: float = 6.0,
    lpslack_gap_gate: float = 5e-5,
) -> dict:
    """Run 4-corner reduced pslp -> full pslp -> (if near-incumbent) lpslack_polish on one candidate."""
    a0 = np.column_stack([x, y, th])
    a0, L0 = am.normalize_arr(a0)
    K = int(math.floor(L0))
    v0 = am.full_viol(a0, L0)

    a_cur, L_cur, v_cur = a0, L0, v0
    trace = [f"init:L={L0:.9f}(v={v0:.1e})"]

    # 1. 4-corner reduced arm-row free-angle kink-safe LP squeeze
    if use_reduced:
        m = am.build_model(a_cur, L_cur)
        if m["reduced"]:
            a_red, L_red, _, _ = am.run_squeeze_reduced(
                a_cur, L_cur, K, model=m, max_it=pslp_iters, trust0=0.015, trust_min=2e-8
            )
            v_red = am.full_viol(a_red, L_red)
            if v_red < 1e-7 and (L_red < L_cur or v_cur > 1e-8):
                a_cur, _ = am.normalize_arr(a_red)
                _, L_cur = am.normalize_arr(a_cur)
                v_cur = am.full_viol(a_cur, L_cur)
                trace.append(f"pslp4c(kept={m['block_counts']['kept']}/{n}):L={L_cur:.9f}(v={v_cur:.1e})")

    # 2. Full-row pslp squeeze (skip if reduced already converged feasibly far above old_L)
    near_inc = (old_L is None) or (L_cur <= float(old_L) + 5e-4) or (v_cur > 1e-11)
    if near_inc:
        a_full, L_full, v_full, _ = am.run_squeeze_full(
            a_cur, L_cur, K, max_it=pslp_iters, trust0=0.012, trust_min=1e-8
        )
        a_full, L_full = am.normalize_arr(a_full)
        v_full = am.full_viol(a_full, L_full)
        if v_full <= 1e-11 and (v_cur > 1e-11 or L_full < L_cur):
            a_cur, L_cur, v_cur = a_full, L_full, v_full
            trace.append(f"pslp_full:L={L_cur:.9f}(v={v_cur:.1e})")

    # 3. 2nd-order exact contact-slack LP (only if competitive with incumbent)
    if run_lpslack and v_cur <= 1e-9 and (old_L is None or L_cur <= float(old_L) + lpslack_gap_gate):
        rows_in = lpslack.rows_from_arrays(a_cur[:, :2], a_cur[:, 2])
        ref_L_str = str(old_L) if old_L is not None else f"{L_cur + 1.0:.12f}"
        # Run lpslack optimize_group (without failing if old_L wasn't beaten; we want best_centers/angles too)
        # Call lpslack internals cleanly via fixed_lp_loop + soft_angle_slp
        c_cur = a_cur[:, :2].copy()
        ang_cur = a_cur[:, 2].copy()
        L_lp = lpslack.exact_side(c_cur, ang_cur)
        for margin in (1e-10, 1e-9):
            fx = lpslack.fixed_lp_loop(ang_cur, c_cur, rounds=10, margin=margin, time_limit=lpslack_time_limit)
            if fx is not None:
                s_fx, c_fx, _, _ = fx
                if lpslack.min_sat_gap(c_fx, ang_cur) > 1e-12 and s_fx < L_lp - 1e-12:
                    c_cur, L_lp = c_fx.copy(), s_fx
        # 1 pass of lpslack dual_non90 + non90
        for mode in ("dual_non90", "non90"):
            for tr_th0 in (3e-4, 5e-5):
                fx_i = lpslack.solve_fixed_lp(ang_cur, c_cur, margin=1e-10, time_limit=lpslack_time_limit)
                pairs_i, marg_i = (fx_i[2], fx_i[3]) if fx_i is not None else (None, None)
                fmask = lpslack.select_free_mask(ang_cur, mode, pairs_i, marg_i, top_k=150)
                if not np.any(fmask):
                    continue
                tr_th = tr_th0
                tr_xy = min(0.02, max(2e-3, 10.0 * tr_th))
                for _ in range(5):
                    slp_out = lpslack.solve_soft_angle_slp(
                        ang_cur, c_cur, fmask, tr_th=tr_th, tr_xy=tr_xy,
                        margin=1e-8 + 2.0 * tr_th * tr_xy, time_limit=lpslack_time_limit,
                    )
                    if slp_out is None:
                        break
                    _, pc_arr, pa_arr, max_dth = slp_out
                    fx_out = lpslack.fixed_lp_loop(pa_arr, pc_arr, rounds=8, margin=1e-10, time_limit=lpslack_time_limit)
                    if fx_out is not None:
                        s_fx, c_fx, _, _ = fx_out
                        if lpslack.min_sat_gap(c_fx, pa_arr) > 1e-12 and s_fx < L_lp - 1e-12:
                            c_cur, ang_cur, L_lp = c_fx.copy(), pa_arr.copy(), s_fx
                            continue
                    tr_th *= 0.5
                    tr_xy = min(0.02, max(1e-3, 10.0 * tr_th))
                    if tr_th < 1e-6 or max_dth < 1e-9:
                        break
        a_lp = np.column_stack([c_cur, ang_cur])
        a_lp, L_lp_n = am.normalize_arr(a_lp)
        v_lp = am.full_viol(a_lp, L_lp_n)
        if v_lp <= 1e-11 and L_lp_n < L_cur:
            a_cur, L_cur, v_cur = a_lp, L_lp_n, v_lp
            trace.append(f"lpslack:L={L_cur:.9f}")

    return {
        "x": a_cur[:, 0].copy(),
        "y": a_cur[:, 1].copy(),
        "th": a_cur[:, 2].copy(),
        "L": float(L_cur),
        "viol": float(v_cur),
        "trace": " -> ".join(trace),
    }


def mechwalk_pslp_round(
    n: int,
    x: np.ndarray,
    y: np.ndarray,
    th: np.ndarray,
    L_cur: float,
    rng_seed: int = 1,
    n_threads: int = 1,
    top_k_polish: int = 4,
) -> list[dict]:
    """Run one mechwalk soft-mode walk + C++ masked melt, returning top basins ready for coupled_pslp_lpslack_polish."""
    all_modes: list[tuple[float, float, np.ndarray, np.ndarray, str]] = []
    for tol in (2e-5, 2e-4, 1e-3):
        jac, _ = ml.contact_jacobian(x, y, th, contact_tol=tol)
        sv, raw_modes = ml.soft_modes(jac, k=12)
        sel = ml.select_diverse_modes(raw_modes, sv, x, y, max_keep=5)
        for _, sval, mvec, neff in sel:
            if any(abs(float(np.dot(mvec, prev[2]))) > 0.90 for prev in all_modes):
                continue
            disp, _ = ml.mode_participation(mvec, n)
            all_modes.append((sval, neff, mvec, disp, f"t{tol:.0e}"))
    all_modes = all_modes[:8]

    coords_list, masks_list, ls_list, labels = [], [], [], []
    for m_idx, (_, _, mvec, disp, t_lbl) in enumerate(all_modes):
        max_disp = float(np.max(disp))
        if max_disp < 1e-8:
            continue
        top_idx = np.argsort(disp)[-max(6, min(14, n // 30)):]
        core = disp >= max(0.25 * max_disp, float(np.quantile(disp, 0.85)))
        core[top_idx] = True
        d2 = (x[:, None] - x[None, core]) ** 2 + (y[:, None] - y[None, core]) ** 2
        active_bool = np.min(d2, axis=1) <= 2.4 ** 2
        if np.sum(active_bool) > 0.50 * n:
            active_bool = disp >= float(np.quantile(disp, 0.65))
        for step in (0.008, 0.025, 0.075):
            amp = min(step / max_disp, 2.5)
            for sgn, s_lbl in ((1.0, "p"), (-1.0, "m")):
                wx, wy, wth, _ = ml.normalize(
                    x + sgn * amp * mvec[:n],
                    y + sgn * amp * mvec[n : 2 * n],
                    th + sgn * amp * mvec[2 * n :],
                )
                coords_list.append((wx, wy, wth))
                masks_list.append(active_bool.astype(np.uint8))
                ls_list.append([L_cur + 0.010, L_cur + 0.0015, L_cur - 0.0004])
                labels.append(f"mw_m{m_idx}_{t_lbl}_{s_lbl}_{step:.3g}")

    if not coords_list:
        return []
    res = cpp_batch_masked_melt(
        coords_list, masks_list, ls_list, ils_iters_per_L=[3, 5, 7],
        a_start=0.88, n_melt_stages=3, masked_iters=140, full_iters=180,
        n_threads=n_threads, base_seed=rng_seed,
    )
    for r, lbl in zip(res, labels):
        r["label"] = lbl
    res.sort(key=lambda r: (0 if r["pen"] <= 1e-7 else 1, r["L_bbox"] + 2.0 * r["pen"]))
    return res[:top_k_polish]
