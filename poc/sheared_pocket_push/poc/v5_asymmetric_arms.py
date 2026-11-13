"""Variant 5 for n=634: Asymmetric split between arms (2-arm Kx!=Ky + 4-arm pinwheel fractional strip split)."""
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


def _eval_2arm_asym(task: tuple) -> dict:
    _, L_probe, P1, P2, u1, u2, s1, s2, yp1, yp2 = task
    K = int(math.floor(L_probe))
    Kx = K - P1
    Ky = K - P2
    W_W = L_probe - Kx
    W_S = L_probe - Ky
    if W_W < 1.5 or W_S < 1.5:
        return {"cap": 0, "feasible_cap": 0}

    w1 = int(math.floor(W_W - u1)) + 1
    w2 = int(math.floor(W_S - u2)) + 1
    th1 = pc.solve_shear_angle(w1, W_W - u1 - 1e-5)
    th2 = pc.solve_shear_angle(w2, W_S - u2 - 1e-5)
    if th1 is None or th2 is None:
        return {"cap": 0, "feasible_cap": 0}

    sw_x = int(math.floor(W_W))
    sw_y = int(math.floor(W_S))
    ux, uy, uth = pc.make_upright_grid(L_probe - Kx, L_probe, L_probe - Ky, L_probe)
    swx, swy, swth = pc.make_upright_grid(0.0, float(sw_x), 0.0, float(sw_y))

    wx, wy, wth = pc.make_sheared_strip(W_W - 1e-5, float(sw_y) + 1e-5, L_probe, w1, th1, s1, y_phase=yp1, u_cols_right=u1)
    sx0, sy0, sth0 = pc.make_sheared_strip(W_S - 1e-5, float(sw_x) + 1e-5, L_probe, w2, th2, s2, y_phase=yp2, u_cols_right=u2)
    sx, sy, sth = sy0, sx0, -sth0

    x = np.concatenate([ux, swx, wx, sx])
    y = np.concatenate([uy, swy, wy, sy])
    th_arr = np.concatenate([uth, swth, wth, sth])

    pockets = [(0.0, W_W, float(sw_y), L_probe), (float(sw_x), L_probe, 0.0, W_S)]
    cands = [0.0, s1 * th1, -s2 * th2, math.pi / 4.0, -math.pi / 4.0]
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
        "mode": "2arm_asym",
        "L_probe": L_probe,
        "P1": P1,
        "P2": P2,
        "u1_u2": (u1, u2),
        "signs": (s1, s2),
        "yp": (yp1, yp2),
        "rigid_cap": rigid_cap,
        "feasible_cap": best_cap,
        "state": best_state,
    }


def _build_pinwheel_arm_strip(W: float, H: float, w: int, th: float, sign: int, y_phase: float, u_cols: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return pc.make_sheared_strip(W - 1e-5, 0.0, H, w, th, sign, y_phase=y_phase, u_cols_right=u_cols)


def _eval_4arm_pinwheel(task: tuple) -> dict:
    _, L_probe, Kx, Ky, W_W, W_S, u_cfg, signs, yps = task
    W_E = L_probe - Kx - W_W
    W_N = L_probe - Ky - W_S
    widths = (W_W, W_N, W_E, W_S)
    lengths = (W_S + Ky, W_W + Kx, W_N + Ky, W_E + Kx)
    ws = []
    ths = []
    for W_arm, u_arm in zip(widths, u_cfg):
        if W_arm < 1.0:
            return {"cap": 0, "feasible_cap": 0}
        if W_arm - u_arm < 1.2:
            w_a = 1
            th_a = 0.0
        else:
            w_a = int(math.floor(W_arm - u_arm)) + 1
            th_a = pc.solve_shear_angle(w_a, W_arm - u_arm - 1e-5)
            if th_a is None:
                # Try w_a - 1 upright-ish fallback with angle 0 if nearly integer
                if W_arm - u_arm >= w_a - 1:
                    w_a = w_a - 1
                    th_a = 0.0
                else:
                    return {"cap": 0, "feasible_cap": 0}
        ws.append(w_a)
        ths.append(th_a)

    # Central upright core [W_W, W_W + Kx] x [W_S, W_S + Ky]
    cx, cy, cth = pc.make_upright_grid(W_W, W_W + Kx, W_S, W_S + Ky)

    # West: [0, W_W] x [0, W_S + Ky]
    wx, wy, wt = _build_pinwheel_arm_strip(W_W, lengths[0], ws[0], ths[0], signs[0], yps[0], u_cfg[0])
    # North: [0, W_W + Kx] x [W_S + Ky, L_probe] -> rotate/map strip of width W_N, length W_W+Kx
    nx0, ny0, nt0 = _build_pinwheel_arm_strip(W_N, lengths[1], ws[1], ths[1], signs[1], yps[1], u_cfg[1])
    nx = ny0
    ny = L_probe - nx0
    nt = -nt0
    # East: [W_W + Kx, L_probe] x [W_S, L_probe]
    ex0, ey0, et0 = _build_pinwheel_arm_strip(W_E, lengths[2], ws[2], ths[2], signs[2], yps[2], u_cfg[2])
    ex = L_probe - ex0
    ey = L_probe - ey0
    et = et0
    # South: [W_W, L_probe] x [0, W_S]
    sx0, sy0, st0 = _build_pinwheel_arm_strip(W_S, lengths[3], ws[3], ths[3], signs[3], yps[3], u_cfg[3])
    sx = L_probe - sy0
    sy = sx0
    st = -st0

    x = np.concatenate([cx, wx, nx, ex, sx])
    y = np.concatenate([cy, wy, ny, ey, sy])
    th_arr = np.concatenate([cth, wt, nt, et, st])

    pockets = [
        (0.0, W_W, 0.0, W_S + Ky),
        (0.0, W_W + Kx, W_S + Ky, L_probe),
        (W_W + Kx, L_probe, W_S, L_probe),
        (W_W, L_probe, 0.0, W_S),
    ]
    cands = [0.0, ths[0], -ths[0], ths[1], -ths[1], math.pi / 4.0, -math.pi / 4.0]
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
        "mode": "4arm_pinwheel",
        "L_probe": L_probe,
        "Kx_Ky": (Kx, Ky),
        "W_W_S": (round(W_W, 4), round(W_S, 4)),
        "ws": ws,
        "u_cfg": u_cfg,
        "rigid_cap": rigid_cap,
        "feasible_cap": best_cap,
        "state": best_state,
    }


def _worker(task: tuple) -> dict:
    if task[0] == "2arm":
        return _eval_2arm_asym(task)
    return _eval_4arm_pinwheel(task)


def main() -> None:
    t0 = time.time()
    inc = gu.load_incumbent_Ls()
    L_ours = inc[N_TARGET]
    L_probe = L_ours - 0.002

    _ = gu.sat_max_overlap_and_bbox(np.array([0.5]), np.array([0.5]), np.array([0.0]))
    _ = pc._can_place_sat(0.5, 0.5, 0.0, np.array([2.5]), np.array([2.5]), np.array([1.0]), np.array([0.0]), 5.0, -1e-9)

    tasks = []
    # 1) Asymmetric 2-arm (P1 != P2 as well as P1 == P2)
    for P1 in range(4, 10):
        for P2 in range(4, 10):
            for u1 in (0, 1):
                for u2 in (0, 1):
                    for s1, s2 in ((1, 1), (-1, -1), (1, -1)):
                        for yp1, yp2 in ((0.0, 0.0), (0.5, 0.5), (0.0, 0.5)):
                            tasks.append(("2arm", L_probe, P1, P2, u1, u2, s1, s2, yp1, yp2))

    # 2) 4-arm pinwheel asymmetric fractional strip splits
    # Test Kx=Ky in (15, 16, 17, 18, 19) with asymmetric fractional splits where both widths admit clean w*cos + sin
    for Kx in (15, 16, 17, 18, 19):
        Ky = Kx
        tot = L_probe - Kx
        splits = [round(f * tot, 5) for f in (0.25, 0.30, 0.35, 0.40, 0.45, 0.50)]
        for W_W in splits:
            W_E = tot - W_W
            for W_S in splits:
                W_N = tot - W_S
                # Check if widths have valid sheared solution with u_cols=0
                valid = True
                for W_arm in (W_W, W_N, W_E, W_S):
                    w_a = int(math.floor(W_arm)) + 1
                    if pc.solve_shear_angle(w_a, W_arm - 1e-5) is None:
                        valid = False
                        break
                if not valid:
                    continue
                for signs in ((1, 1, 1, 1), (1, -1, 1, -1)):
                    for yps in ((0.0, 0.0, 0.0, 0.0), (0.5, 0.5, 0.5, 0.5)):
                        tasks.append(("4arm", L_probe, Kx, Ky, W_W, W_S, (0, 0, 0, 0), signs, yps))

    n_workers = os.cpu_count() or 4
    print(f"Running Variant 5 (asymmetric 2-arm + 4-arm pinwheel) on n={N_TARGET} ({len(tasks)} tasks, {n_workers} workers)...", flush=True)
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        results = list(ex.map(_worker, tasks, chunksize=8))

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
        "variant": "v5_asymmetric_arms",
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
    out_path = THIS_DIR.parent / "docs" / "v5_asymmetric_arms.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(
        f"Variant 5 done in {summary['seconds']}s: best_cap={best_cap}, deficit={deficit}, "
        f"cfg={summary['best_cfg']}, best_L={best_L}, win={certified_win is not None}",
        flush=True,
    )


if __name__ == "__main__":
    main()
