"""Variant 3 for n=634: True mitered SW corner seam (pure diagonal clip, 45-deg seam spine, or shrunk sw_u + mitered fringe)."""
from __future__ import annotations

import os
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_var] = "1"

import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import pocket_core as pc  # noqa: E402
import geom_utils as gu  # noqa: E402

N_TARGET = 634


def _overlaps_any(gx: float, gy: float, ga: float, ax: np.ndarray, ay: np.ndarray, ac: np.ndarray, as_: np.ndarray, margin: float = 1e-7) -> bool:
    if len(ax) == 0:
        return False
    return float(pc._can_place_sat(gx, gy, ga, ax, ay, ac, as_, 1e6, margin)) < 0.0


def build_mitered_l(
    L_probe: float,
    P: int,
    w: int,
    th: float,
    s1: int,
    s2: int,
    sw_u: int,
    miter_mode: str,
    b_miter: float,
    y_phase: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[float, float, float, float]]]:
    K = int(math.floor(L_probe))
    Kx = Ky = K - P
    W = L_probe - Kx

    # Main NE upright grid
    ux, uy, uth = pc.make_upright_grid(L_probe - Kx, L_probe, L_probe - Ky, L_probe)

    # Optional smaller SW upright block [0, sw_u]^2
    if sw_u > 0:
        swx, swy, swth = pc.make_upright_grid(0.0, float(sw_u), 0.0, float(sw_u))
    else:
        swx = swy = swth = np.empty(0, dtype=np.float64)

    # Optional 45-degree diamond spine along SW corner diagonal (x=y) starting past sw_u
    dxs, dys, dths = [], [], []
    if miter_mode == "diag45":
        s2_inv = 1.0 / math.sqrt(2.0)
        start_d = float(sw_u) + s2_inv if sw_u == 0 else float(sw_u) + 0.72
        k_d = 0
        while True:
            px = start_d + k_d * s2_inv
            py = px
            if px + s2_inv > W - 1e-6:
                break
            dxs.append(px)
            dys.append(py)
            dths.append(math.pi / 4.0)
            k_d += 1
    dx_arr = np.array(dxs, dtype=np.float64)
    dy_arr = np.array(dys, dtype=np.float64)
    dt_arr = np.array(dths, dtype=np.float64)

    # Extend BOTH full sheared arms down to y_min=0 (West) and x_min=0 (South)
    wx, wy, wth = pc.make_sheared_strip(W - 1e-5, 0.0, L_probe, w, th, s1, y_phase=y_phase)
    sx0, sy0, sth0 = pc.make_sheared_strip(W - 1e-5, 0.0, L_probe, w, th, s2, y_phase=y_phase)
    sx, sy, sth = sy0, sx0, -sth0

    # Filter West & South squares against SW upright block, 45-deg spine, and miter plane
    keep_w = np.ones(len(wx), dtype=bool)
    keep_s = np.ones(len(sx), dtype=bool)
    if sw_u > 0:
        hx_th = 0.5 * (abs(math.cos(th)) + abs(math.sin(th)))
        keep_w &= ~((wx - hx_th < sw_u - 1e-7) & (wy - hx_th < sw_u - 1e-7))
        keep_s &= ~((sx - hx_th < sw_u - 1e-7) & (sy - hx_th < sw_u - 1e-7))

    # Miter split inside [0, W]^2 (and overlap zone): West keeps y >= x + b_miter, South keeps y < x + b_miter
    in_sw_w = (wx < W + 0.6) & (wy < W + 0.6)
    in_sw_s = (sx < W + 0.6) & (sy < W + 0.6)
    keep_w &= (~in_sw_w) | (wy >= wx + b_miter)
    keep_s &= (~in_sw_s) | (sy < sx + b_miter)

    wx, wy, wth = wx[keep_w], wy[keep_w], wth[keep_w]
    sx, sy, sth = sx[keep_s], sy[keep_s], sth[keep_s]

    # Build fixed obstacle array (NE + sw_u + diag45)
    base_x = np.concatenate([ux, swx, dx_arr])
    base_y = np.concatenate([uy, swy, dy_arr])
    base_t = np.concatenate([uth, swth, dt_arr])
    base_c = np.cos(base_t)
    base_s = np.sin(base_t)

    # Greedily add West & South non-overlapping squares, preferring squares deeper inside interior first
    w_order = np.argsort(-(wy - wx))
    s_order = np.argsort(-(sx - sy))

    px = list(base_x)
    py = list(base_y)
    pt = list(base_t)
    ax, ay, ac, as_ = base_x.copy(), base_y.copy(), base_c.copy(), base_s.copy()

    # Alternate adding farthest-from-seam first so miter locks cleanly
    wi, si_idx = 0, 0
    while wi < len(w_order) or si_idx < len(s_order):
        if wi < len(w_order):
            idx = int(w_order[wi])
            wi += 1
            if float(pc._can_place_sat(float(wx[idx]), float(wy[idx]), float(wth[idx]), ax, ay, ac, as_, L_probe, -1e-9)) >= 0.0:
                px.append(float(wx[idx])); py.append(float(wy[idx])); pt.append(float(wth[idx]))
                ax = np.append(ax, wx[idx]); ay = np.append(ay, wy[idx])
                ac = np.append(ac, math.cos(wth[idx])); as_ = np.append(as_, math.sin(wth[idx]))
        if si_idx < len(s_order):
            idx = int(s_order[si_idx])
            si_idx += 1
            if float(pc._can_place_sat(float(sx[idx]), float(sy[idx]), float(sth[idx]), ax, ay, ac, as_, L_probe, -1e-9)) >= 0.0:
                px.append(float(sx[idx])); py.append(float(sy[idx])); pt.append(float(sth[idx]))
                ax = np.append(ax, sx[idx]); ay = np.append(ay, sy[idx])
                ac = np.append(ac, math.cos(sth[idx])); as_ = np.append(as_, math.sin(sth[idx]))

    pockets = [
        (0.0, W + 0.5, 0.0, W + 0.5),
        (0.0, W, max(0.0, W - 2.5), L_probe),
        (max(0.0, W - 2.5), L_probe, 0.0, W),
    ]
    return ax, ay, np.array(pt, dtype=np.float64), pockets


def _eval_worker(task: tuple) -> dict:
    L_probe, P, w, s1, s2, sw_u, miter_mode, b_miter, y_phase = task
    K = int(math.floor(L_probe))
    W = L_probe - (K - P)
    th = pc.solve_shear_angle(w, W - 1e-5)
    if th is None or th > 0.65:
        return {"cap": 0, "feasible_cap": 0}

    x, y, th_arr, pockets = build_mitered_l(L_probe, P, w, th, s1, s2, sw_u, miter_mode, b_miter, y_phase)
    cands = [0.0, math.pi / 4.0, -math.pi / 4.0, s1 * th, -s2 * th, 0.5 * s1 * th, -0.5 * s2 * th]
    need_extra = max(0, N_TARGET - len(x))
    x_f, y_f, th_f, soft_holes = pc.fill_end_and_void_pockets(
        x, y, th_arr, L_probe, pockets, cands, grid_step=0.125, extra_soft_candidates=max(4, need_extra + 2)
    )
    rigid_cap = len(x_f)
    pen0, L0 = gu.sat_max_overlap_and_bbox(x_f, y_f, th_f)
    if (pen0 > 1e-9 or L0 > L_probe + 1e-8) and pen0 <= 0.01:
        x_f, y_f, th_f, _ = pc.relax_tight(x_f, y_f, th_f, L_probe, maxiter=250)
        pen0, L0 = gu.sat_max_overlap_and_bbox(x_f, y_f, th_f)
    best_cap = rigid_cap if (pen0 <= 1e-9 and L0 <= L_probe + 1e-5) else 0
    best_state = (x_f, y_f, th_f) if best_cap > 0 else None

    if best_cap < N_TARGET and len(soft_holes) >= (N_TARGET - rigid_cap) and (N_TARGET - rigid_cap) <= 6:
        k_add = N_TARGET - rigid_cap
        xs_t = np.concatenate([x_f, [h[1] for h in soft_holes[:k_add]]])
        ys_t = np.concatenate([y_f, [h[2] for h in soft_holes[:k_add]]])
        ts_t = np.concatenate([th_f, [h[3] for h in soft_holes[:k_add]]])
        rx, ry, rth, _ = pc.relax_tight(xs_t, ys_t, ts_t, L_probe, maxiter=450)
        rpen, rL = gu.sat_max_overlap_and_bbox(rx, ry, rth)
        if rpen <= 1e-9 and rL <= L_probe + 1e-5:
            best_cap = len(rx)
            best_state = (rx, ry, rth)

    return {
        "L_probe": L_probe,
        "P": P,
        "w": w,
        "th_deg": round(math.degrees(th), 3),
        "signs": (s1, s2),
        "sw_u": sw_u,
        "miter_mode": miter_mode,
        "b_miter": b_miter,
        "y_phase": y_phase,
        "rigid_cap": rigid_cap,
        "feasible_cap": best_cap,
        "state": best_state,
    }


def main() -> None:
    t0 = time.time()
    inc = gu.load_incumbent_Ls()
    L_ours = inc[N_TARGET]
    L_probe = L_ours - 0.002

    _ = gu.sat_max_overlap_and_bbox(np.array([0.5]), np.array([0.5]), np.array([0.0]))
    _ = pc._can_place_sat(0.5, 0.5, 0.0, np.array([2.5]), np.array([2.5]), np.array([1.0]), np.array([0.0]), 5.0, 1e-7)

    tasks = []
    for P in (5, 6, 7, 8, 9):
        W = P + (L_probe - int(math.floor(L_probe)))
        w = int(math.floor(W)) + 1
        if pc.solve_shear_angle(w, W - 1e-5) is None:
            continue
        for s1, s2 in ((1, 1), (-1, -1), (1, -1), (-1, 1)):
            for sw_u in (0, 3, 4, 5):
                for miter_mode in ("diag_clip", "diag45"):
                    for b_miter in (-0.4, 0.0, 0.4):
                        if miter_mode == "diag45" and b_miter != 0.0:
                            continue
                        for y_phase in (0.0, 0.5):
                            tasks.append((L_probe, P, w, s1, s2, sw_u, miter_mode, b_miter, y_phase))

    n_workers = os.cpu_count() or 4
    print(f"Running Variant 3 (mitered SW corner) on n={N_TARGET} ({len(tasks)} tasks, {n_workers} workers)...", flush=True)
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        results = list(ex.map(_eval_worker, tasks, chunksize=8))

    results.sort(key=lambda r: (r["feasible_cap"], r.get("rigid_cap", 0)), reverse=True)
    best = results[0]
    best_cap = best["feasible_cap"]
    deficit = N_TARGET - best_cap

    best_L = None
    certified_win = None
    if best_cap >= N_TARGET and best["state"] is not None:
        bx, by, bth = best["state"]
        bx, by, bth = bx[:N_TARGET], by[:N_TARGET], bth[:N_TARGET]
        lx, ly, lth, lL, lok = pc.squeeze_L_by_relax_and_slp(bx, by, bth, L_probe)
        best_L = lL
        if lok and lL < L_ours - 1e-9:
            cert = gu.verify_official(N_TARGET, lx, ly, lth)
            if cert["ok"] and cert["L_float"] < L_ours:
                certified_win = {k: v for k, v in cert.items() if k != "rows"}

    summary = {
        "variant": "v3_miter_corner",
        "n": N_TARGET,
        "L_ours": L_ours,
        "L_probe": L_probe,
        "tasks": len(tasks),
        "best_cap": best_cap,
        "deficit": deficit,
        "best_cfg": {k: v for k, v in best.items() if k != "state"},
        "top5": [{k: v for k, v in r.items() if k != "state"} for r in results[:5]],
        "best_L": best_L,
        "certified_win": certified_win,
        "seconds": round(time.time() - t0, 2),
    }
    out_path = THIS_DIR.parent / "docs" / "v3_miter_corner.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(
        f"Variant 3 done in {summary['seconds']}s: best_cap={best_cap}, deficit={deficit}, "
        f"cfg={summary['best_cfg']}, best_L={best_L}, win={certified_win is not None}",
        flush=True,
    )


if __name__ == "__main__":
    main()
