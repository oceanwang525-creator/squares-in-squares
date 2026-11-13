"""Variant 1 for n=634: Add continuous pocket angle (sweep theta around shear angle + joint angle/position L-BFGS relaxation).

Keeps symmetric arm layout (Kx=Ky=K-P, P1=P2=P, sw_upright, y_phase=0), no miter, no end-taper yet.
"""
from __future__ import annotations

import os
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_var] = "1"

import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import pocket_core as pc  # noqa: E402
import geom_utils as gu  # noqa: E402

N_TARGET = 634


def build_symmetric_cont_angle(
    L: float, P: int, w: int, th_rad: float, sign1: int, sign2: int, dy_scale: float = 1.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[float, float, float, float]]]:
    """Symmetric L-pocket (Kx=Ky=floor(L)-P) at arbitrary continuous angle th_rad."""
    K = int(math.floor(L))
    Kx = Ky = K - P
    W = L - Kx
    sw_p = int(math.floor(W))
    # Ensure width fits w squares at th_rad (if wider, trim? No, require valid strip width or slightly compressed if relaxing)
    ux, uy, uth = pc.make_upright_grid(L - Kx, L, L - Ky, L)
    swx, swy, swth = pc.make_upright_grid(0.0, float(sw_p), 0.0, float(sw_p))

    # Strip at continuous th_rad: allow slight pre-overlap up to 0.08 if dy_scale < 1 or W slightly < span
    c, s = math.cos(th_rad), math.sin(th_rad)
    hx = 0.5 * (abs(c) + abs(s))
    span_x = w * c + abs(s)
    # If span_x > W slightly, squeeze x spacing linearly to fit [0, W] so L-BFGS can unroll angle
    x_scale = min(1.0, (W - 2 * hx - 1e-4) / max((w - 1) * c, 1e-9)) if span_x > W - 1e-5 else 1.0
    dy_row = (1.0 / max(c, 1e-9)) * dy_scale

    def make_arm(y_min: float, y_max: float, sgn: int):
        xs, ys, ts = [], [], []
        r_min = -int(math.ceil(w * abs(s) / dy_row)) - 2
        r_max = int(math.ceil((y_max - y_min + w * abs(s)) / dy_row)) + 2
        x_pad = max(0.0, 0.5 * (W - (2 * hx + (w - 1) * c * x_scale)))
        for r in range(r_min, r_max + 1):
            y0 = y_min + hx + r * dy_row
            for k in range(w):
                xk = x_pad + hx + k * c * x_scale
                yk = y0 + sgn * k * s
                if xk - hx >= -1e-9 and xk + hx <= W + 1e-9 and yk - hx >= y_min - 1e-9 and yk + hx <= y_max + 1e-9:
                    xs.append(xk)
                    ys.append(yk)
                    ts.append(sgn * th_rad)
        return np.array(xs), np.array(ys), np.array(ts)

    wx, wy, wth = make_arm(float(sw_p) + 1e-5, L, sign1)
    sx, sy, sth = make_arm(float(sw_p) + 1e-5, L, sign2)
    # Transpose south arm
    sx, sy, sth = sy, sx, -sth

    x = np.concatenate([ux, swx, wx, sx])
    y = np.concatenate([uy, swy, wy, sy])
    th = np.concatenate([uth, swth, wth, sth])

    pockets = [
        (0.0, W, float(sw_p), L),
        (float(sw_p), L, 0.0, W),
    ]
    return x, y, th, pockets


def _eval_worker(task: tuple[float, int, int, float, int, int]) -> dict:
    L_probe, P, w, dth_deg, s1, s2 = task
    W = P + (L_probe - int(math.floor(L_probe)))
    th0 = pc.solve_shear_angle(w, W - 1e-5)
    if th0 is None:
        return {"cap": 0, "feasible_cap": 0}
    th = th0 + math.radians(dth_deg)
    if th <= 0.02 or th >= math.pi / 4.0 + 0.02:
        return {"cap": 0, "feasible_cap": 0}

    x, y, th_arr, pockets = build_symmetric_cont_angle(L_probe, P, w, th, s1, s2, dy_scale=1.0)
    cands = [0.0, s1 * th, -s2 * th, math.pi / 4.0, -math.pi / 4.0, 0.5 * s1 * th]
    # Rigid end fill first + collect soft holes up to missing count
    pen0, L0 = gu.sat_max_overlap_and_bbox(x, y, th_arr)
    if pen0 > 0.08 or L0 > L_probe + 0.05:
        return {"cap": 0, "feasible_cap": 0}
    if pen0 > 1e-10 or L0 > L_probe + 1e-8:
        x, y, th_arr, _ = pc.relax_tight(x, y, th_arr, L_probe, maxiter=250)
        pen0, L0 = gu.sat_max_overlap_and_bbox(x, y, th_arr)
        if pen0 > 1e-9 or L0 > L_probe + 1e-5:
            return {"cap": 0, "feasible_cap": 0}

    need_extra = max(0, N_TARGET - len(x))
    x_f, y_f, th_f, soft_holes = pc.fill_end_and_void_pockets(
        x, y, th_arr, L_probe, pockets, cands, grid_step=0.125, extra_soft_candidates=max(4, need_extra + 2)
    )
    rigid_cap = len(x_f)

    pen_r, L_r = gu.sat_max_overlap_and_bbox(x_f, y_f, th_f)
    best_relaxed_cap = rigid_cap if (pen_r <= 1e-9 and L_r <= L_probe + 1e-5) else 0
    best_state = (x_f, y_f, th_f) if best_relaxed_cap > 0 else None

    # If still short of 634, try soft-inserting up to 6 missing squares and relaxing angles+positions
    if rigid_cap < N_TARGET and len(soft_holes) >= (N_TARGET - rigid_cap) and (N_TARGET - rigid_cap) <= 6:
        k_add = N_TARGET - rigid_cap
        add_x = np.array([h[1] for h in soft_holes[:k_add]])
        add_y = np.array([h[2] for h in soft_holes[:k_add]])
        add_t = np.array([h[3] for h in soft_holes[:k_add]])
        xs_t = np.concatenate([x_f, add_x])
        ys_t = np.concatenate([y_f, add_y])
        ts_t = np.concatenate([th_f, add_t])
        rx, ry, rth, _ = pc.relax_tight(xs_t, ys_t, ts_t, L_probe, maxiter=450)
        rpen, rL = gu.sat_max_overlap_and_bbox(rx, ry, rth)
        if rpen <= 1e-9 and rL <= L_probe + 1e-5:
            best_relaxed_cap = len(rx)
            best_state = (rx, ry, rth)

    return {
        "L_probe": L_probe,
        "P": P,
        "w": w,
        "dth_deg": dth_deg,
        "th_deg": round(math.degrees(th), 3),
        "signs": (s1, s2),
        "rigid_cap": rigid_cap,
        "feasible_cap": best_relaxed_cap,
        "state": best_state,
    }


def main() -> None:
    t0 = time.time()
    inc = gu.load_incumbent_Ls()
    L_ours = inc[N_TARGET]
    L_probe = L_ours - 0.002  # 25.705107

    # Warm Numba JIT in main process
    _ = gu.sat_max_overlap_and_bbox(np.array([0.5]), np.array([0.5]), np.array([0.0]))
    _ = pc._can_place_sat(0.5, 0.5, 0.0, np.array([2.5]), np.array([2.5]), np.array([1.0]), np.array([0.0]), 5.0, 1e-7)

    tasks = []
    for P in range(4, 14):
        W = P + (L_probe - int(math.floor(L_probe)))
        for w in (P + 1, P + 2):
            if pc.solve_shear_angle(w, W - 1e-5) is None:
                continue
            for dth_deg in (-3.0, -2.0, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 2.0, 3.0):
                for s1, s2 in ((1, 1), (-1, -1), (1, -1)):
                    tasks.append((L_probe, P, w, dth_deg, s1, s2))

    n_workers = os.cpu_count() or 4
    print(f"Running Variant 1 (continuous pocket angle) on n={N_TARGET} ({len(tasks)} tasks, {n_workers} workers)...", flush=True)
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        results = list(ex.map(_eval_worker, tasks, chunksize=4))

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
        "variant": "v1_continuous_angle",
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
    out_path = THIS_DIR.parent / "docs" / "v1_continuous_angle.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(
        f"Variant 1 done in {summary['seconds']}s: best_cap={best_cap}, deficit={deficit}, "
        f"cfg={summary['best_cfg']}, best_L={best_L}, win={certified_win is not None}",
        flush=True,
    )


if __name__ == "__main__":
    main()
