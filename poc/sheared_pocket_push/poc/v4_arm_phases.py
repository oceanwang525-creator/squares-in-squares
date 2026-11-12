"""Variant 4 for n=634: Independent row counts (w1, u1 vs w2, u2), phases, x-offsets, and partial corner tapers per arm."""
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


def _build_arm_variant(
    W: float,
    y_min: float,
    y_max: float,
    w: int,
    u_cols: int,
    dth_deg: float,
    sign: int,
    y_phase: float,
    x_pad_mode: str,
    corner_taper: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    W_sh = W - u_cols
    th0 = pc.solve_shear_angle(w, W_sh - 1e-5)
    if th0 is None:
        return None
    th = th0 + math.radians(dth_deg)
    if th <= 0.02 or th >= 0.72:
        return None
    ax, ay, at = pc.make_sheared_strip(
        W - 1e-5,
        y_min,
        y_max,
        w,
        th,
        sign,
        x_offset=0.0,
        y_phase=y_phase,
        dy_scale=1.0,
        x_pad_mode=x_pad_mode,
        u_cols_right=u_cols,
    )
    if len(ax) == 0:
        return None

    # Optional partial flush upright wedge fill (only keeps upright squares if they beat sheared overlaps)
    if corner_taper > 0 and W_sh >= 1.5:
        cands_xy = []
        # Bottom empty wedge (opposite slanted start) and top empty wedge
        for j in range(corner_taper):
            yb = y_min + 0.5 + j
            yt = y_max - 0.5 - j
            for i in range(int(math.floor(W_sh))):
                xl = 0.5 + i
                xr = W_sh - 0.5 - i
                cands_xy.append((xl, yb))
                cands_xy.append((xr, yb))
                cands_xy.append((xl, yt))
                cands_xy.append((xr, yt))
        ac = np.cos(at)
        as_ = np.sin(at)
        for gx, gy in cands_xy:
            if gx < 0.5 - 1e-7 or gx > W_sh - 0.5 + 1e-7 or gy < y_min + 0.5 - 1e-7 or gy > y_max - 0.5 + 1e-7:
                continue
            if float(pc._can_place_sat(gx, gy, 0.0, ax, ay, ac, as_, 1e6, -1e-9)) >= 0.0:
                ax = np.append(ax, gx)
                ay = np.append(ay, gy)
                at = np.append(at, 0.0)
                ac = np.append(ac, 1.0)
                as_ = np.append(as_, 0.0)
    return ax, ay, at


def _eval_worker(task: tuple) -> dict:
    L_probe, P, arm1_cfg, arm2_cfg, sw_extend = task
    K = int(math.floor(L_probe))
    Kx = Ky = K - P
    W = L_probe - Kx
    sw_p = int(math.floor(W))

    ux, uy, uth = pc.make_upright_grid(L_probe - Kx, L_probe, L_probe - Ky, L_probe)
    swx, swy, swth = pc.make_upright_grid(0.0, float(sw_p), 0.0, float(sw_p))

    w1, u1, dth1, s1, yp1, xp1, ct1 = arm1_cfg
    w2, u2, dth2, s2, yp2, xp2, ct2 = arm2_cfg

    y_start = float(sw_p) + 1e-5
    r1 = _build_arm_variant(W, y_start, L_probe, w1, u1, dth1, s1, yp1, xp1, ct1)
    r2 = _build_arm_variant(W, y_start, L_probe, w2, u2, dth2, s2, yp2, xp2, ct2)
    if r1 is None or r2 is None:
        return {"cap": 0, "feasible_cap": 0}

    wx, wy, wth = r1
    sx0, sy0, sth0 = r2
    sx, sy, sth = sy0, sx0, -sth0

    x = np.concatenate([ux, swx, wx, sx])
    y = np.concatenate([uy, swy, wy, sy])
    th_arr = np.concatenate([uth, swth, wth, sth])

    # Check base overlaps first (relax if tiny overlap from dth)
    pen0, L0 = gu.sat_max_overlap_and_bbox(x, y, th_arr)
    if pen0 > 0.08 or L0 > L_probe + 0.05:
        return {"cap": 0, "feasible_cap": 0}
    if pen0 > 1e-10 or L0 > L_probe + 1e-8:
        x, y, th_arr, _ = pc.relax_tight(x, y, th_arr, L_probe, maxiter=200)
        pen0, L0 = gu.sat_max_overlap_and_bbox(x, y, th_arr)
        if pen0 > 1e-9 or L0 > L_probe + 1e-5:
            return {"cap": 0, "feasible_cap": 0}

    pockets = [(0.0, W, float(sw_p), L_probe), (float(sw_p), L_probe, 0.0, W)]
    if sw_extend:
        pockets.append((0.0, W + 0.5, 0.0, W + 0.5))
    cands = [0.0, float(np.median(wth)), -float(np.median(wth)), math.pi / 4.0, -math.pi / 4.0]
    need_extra = max(0, N_TARGET - len(x))
    x_f, y_f, th_f, soft_holes = pc.fill_end_and_void_pockets(
        x, y, th_arr, L_probe, pockets, cands, grid_step=0.125, extra_soft_candidates=max(4, need_extra + 2)
    )
    rigid_cap = len(x_f)
    pen_r, L_r = gu.sat_max_overlap_and_bbox(x_f, y_f, th_f)
    if (pen_r > 1e-9 or L_r > L_probe + 1e-8) and pen_r <= 0.01:
        x_f, y_f, th_f, _ = pc.relax_tight(x_f, y_f, th_f, L_probe, maxiter=250)
        pen_r, L_r = gu.sat_max_overlap_and_bbox(x_f, y_f, th_f)
    best_cap = rigid_cap if (pen_r <= 1e-9 and L_r <= L_probe + 1e-5) else 0
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
        "arm1": arm1_cfg,
        "arm2": arm2_cfg,
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
    for P in (5, 6, 7, 8):
        W = P + (L_probe - int(math.floor(L_probe)))
        # Build single-arm candidates (w, u_cols, dth_deg, sign, y_phase, x_pad_mode, corner_taper)
        arms = []
        for u_cols in (0, 1):
            w = int(math.floor(W - u_cols)) + 1
            if pc.solve_shear_angle(w, (W - u_cols) - 1e-5) is None:
                continue
            for sign in (1, -1):
                for yp in (0.0, 0.25, 0.5, 0.75):
                    for dth_deg, xp in ((0.0, "center"), (-0.5, "left"), (-0.5, "right")):
                        arms.append((w, u_cols, dth_deg, sign, yp, xp, 2))
        # Pair distinct arm configs (limit to high-diversity cross pairs)
        for i, a1 in enumerate(arms):
            # Include all same-(w,u) phase mismatches + mixed-(w,u) combos
            for a2 in arms:
                if a1[2] != 0.0 and a2[2] != 0.0 and a1[2] != a2[2]:
                    continue
                if a1[4] != a2[4] and (a1[5] != "center" or a2[5] != "center"):
                    continue
                tasks.append((L_probe, P, a1, a2, False))

    n_workers = os.cpu_count() or 4
    print(f"Running Variant 4 (independent arm counts/phases/offsets) on n={N_TARGET} ({len(tasks)} tasks, {n_workers} workers)...", flush=True)
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        results = list(ex.map(_eval_worker, tasks, chunksize=16))

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
        "variant": "v4_arm_phases",
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
    out_path = THIS_DIR.parent / "docs" / "v4_arm_phases.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(
        f"Variant 4 done in {summary['seconds']}s: best_cap={best_cap}, deficit={deficit}, "
        f"cfg={summary['best_cfg']}, best_L={best_L}, win={certified_win is not None}",
        flush=True,
    )


if __name__ == "__main__":
    main()
