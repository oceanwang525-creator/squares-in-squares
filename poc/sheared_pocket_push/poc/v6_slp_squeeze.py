"""Variant 6 for n=634: Expanded-L feasible seeding + sequential insert + joint (dx,dy,dth,L) SLP contact squeeze."""
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


def _build_asym_seed(L_build: float, P1: int, P2: int, u1: int, u2: int, s1: int, s2: int, yp1: float, yp2: float):
    K = int(math.floor(L_build))
    Kx = K - P1
    Ky = K - P2
    W_W = L_build - Kx
    W_S = L_build - Ky
    if W_W < 1.5 or W_S < 1.5:
        return None
    w1 = int(math.floor(W_W - u1)) + 1
    w2 = int(math.floor(W_S - u2)) + 1
    th1 = pc.solve_shear_angle(w1, W_W - u1 - 1e-5)
    th2 = pc.solve_shear_angle(w2, W_S - u2 - 1e-5)
    if th1 is None or th2 is None:
        return None

    sw_x = int(math.floor(W_W))
    sw_y = int(math.floor(W_S))
    ux, uy, uth = pc.make_upright_grid(L_build - Kx, L_build, L_build - Ky, L_build)
    swx, swy, swth = pc.make_upright_grid(0.0, float(sw_x), 0.0, float(sw_y))
    wx, wy, wth = pc.make_sheared_strip(W_W - 1e-5, float(sw_y) + 1e-5, L_build, w1, th1, s1, y_phase=yp1, u_cols_right=u1)
    sx0, sy0, sth0 = pc.make_sheared_strip(W_S - 1e-5, float(sw_x) + 1e-5, L_build, w2, th2, s2, y_phase=yp2, u_cols_right=u2)
    sx, sy, sth = sy0, sx0, -sth0

    x = np.concatenate([ux, swx, wx, sx])
    y = np.concatenate([uy, swy, wy, sy])
    th_arr = np.concatenate([uth, swth, wth, sth])
    pockets = [(0.0, W_W, float(sw_y), L_build), (float(sw_x), L_build, 0.0, W_S), (0.0, W_W + 0.5, 0.0, W_S + 0.5)]
    cands = [0.0, s1 * th1, -s2 * th2, math.pi / 4.0, -math.pi / 4.0, 0.5 * s1 * th1, -0.5 * s2 * th2]
    return x, y, th_arr, pockets, cands


def _sequential_insert_and_squeeze(task: tuple) -> dict:
    P1, P2, u1, u2, s1, s2, yp1, yp2, L_expand, strategy = task
    seed = _build_asym_seed(L_expand, P1, P2, u1, u2, s1, s2, yp1, yp2)
    if seed is None:
        return {"cap": 0, "best_L": 1e9, "ok": False}
    x, y, th_arr, pockets, cands = seed

    # Rigid fill at L_expand first
    x, y, th_arr, holes = pc.fill_end_and_void_pockets(
        x, y, th_arr, L_expand, pockets, cands, grid_step=0.1, extra_soft_candidates=10
    )

    # Sequentially insert remaining squares up to N_TARGET one-by-one with relaxation
    rng = np.random.default_rng(1000 + P1 * 37 + P2 * 13 + (0 if s1 > 0 else 7))
    cur_L = L_expand
    while len(x) < N_TARGET:
        # Refresh holes if depleted
        _, _, _, holes = pc.fill_end_and_void_pockets(
            x, y, th_arr, cur_L, pockets, cands, grid_step=0.1, extra_soft_candidates=6
        )
        if not holes:
            break
        # Pick best hole (or randomized top-2 for diversity)
        h_idx = 0 if strategy == "greedy" or len(holes) == 1 else int(rng.integers(0, min(2, len(holes))))
        _, gx, gy, ga = holes[h_idx]
        x = np.append(x, gx)
        y = np.append(y, gy)
        th_arr = np.append(th_arr, ga)
        x, y, th_arr, pen = pc.relax_tight(x, y, th_arr, cur_L, maxiter=350)
        pen, cur_L_meas = gu.sat_max_overlap_and_bbox(x, y, th_arr)
        if pen > 1e-8 or cur_L_meas > cur_L + 0.08:
            # Allow expanding cur_L slightly to keep feasible
            cur_L = max(cur_L + 0.04, cur_L_meas + 0.01)
            x, y, th_arr, _ = pc.relax_tight(x, y, th_arr, cur_L, maxiter=400)
            pen, cur_L_meas = gu.sat_max_overlap_and_bbox(x, y, th_arr)
            if pen > 1e-8:
                break

    if len(x) < N_TARGET:
        return {"cap": len(x), "best_L": 1e9, "ok": False, "cfg": task}

    x, y, th_arr = x[:N_TARGET], y[:N_TARGET], th_arr[:N_TARGET]
    # Now squeeze L aggressively + SLP angle/center polish
    lx, ly, lth, lL, lok = pc.squeeze_L_by_relax_and_slp(x, y, th_arr, cur_L, step=0.015, max_bisect=10)
    # Second fine SLP polish pass from tighter trust region
    lx2, ly2, lth2, lL2, lok2 = pc.slp_angle_center_polish(lx, ly, lth, rounds=16, trust_xy=0.05, trust_th0=0.025)
    if lok2 and lL2 < lL:
        lx, ly, lth, lL, lok = lx2, ly2, lth2, lL2, lok2

    pen_f, meas_L = gu.sat_max_overlap_and_bbox(lx, ly, lth)
    return {
        "cap": len(lx),
        "best_L": float(meas_L),
        "ok": bool(lok and pen_f <= 0.0),
        "pen": float(pen_f),
        "cfg": task,
        "state": (lx, ly, lth) if (pen_f <= 0.0) else None,
    }


def main() -> None:
    t0 = time.time()
    inc = gu.load_incumbent_Ls()
    L_ours = inc[N_TARGET]

    _ = gu.sat_max_overlap_and_bbox(np.array([0.5]), np.array([0.5]), np.array([0.0]))
    _ = pc._can_place_sat(0.5, 0.5, 0.0, np.array([2.5]), np.array([2.5]), np.array([1.0]), np.array([0.0]), 5.0, -1e-9)

    # Top sheared basins from v1-v5
    basins = [
        (6, 6, 0, 0, 1, 1, 0.0, 0.0),
        (6, 6, 0, 0, -1, -1, 0.0, 0.0),
        (6, 6, 0, 0, 1, -1, 0.0, 0.5),
        (9, 4, 0, 0, -1, -1, 0.0, 0.5),
        (5, 9, 0, 0, -1, -1, 0.0, 0.0),
        (8, 8, 0, 0, 1, 1, 0.0, 0.0),
        (7, 7, 0, 0, 1, 1, 0.0, 0.5),
        (5, 5, 0, 0, 1, 1, 0.0, 0.0),
    ]
    tasks = []
    for P1, P2, u1, u2, s1, s2, yp1, yp2 in basins:
        for L_exp in (L_ours + 0.04, L_ours + 0.14):
            for strat in ("greedy", "div"):
                tasks.append((P1, P2, u1, u2, s1, s2, yp1, yp2, L_exp, strat))

    n_workers = os.cpu_count() or 4
    print(f"Running Variant 6 (expand + sequential insert + SLP squeeze) on n={N_TARGET} ({len(tasks)} tasks, {n_workers} workers)...", flush=True)
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        results = list(ex.map(_sequential_insert_and_squeeze, tasks, chunksize=1))

    results.sort(key=lambda r: (-r["cap"], r["best_L"] if r["ok"] else 1e9))
    best = results[0]
    best_cap = best["cap"]
    best_L = best["best_L"] if best["ok"] else None

    certified_win = None
    if best["ok"] and best["best_L"] < L_ours - 1e-9 and best["state"] is not None:
        lx, ly, lth = best["state"]
        cert = gu.verify_official(N_TARGET, lx, ly, lth)
        if cert["ok"] and cert["L_float"] < L_ours:
            certified_win = {k: v for k, v in cert.items() if k != "rows"}

    summary = {
        "variant": "v6_slp_squeeze",
        "n": N_TARGET,
        "L_ours": L_ours,
        "tasks": len(tasks),
        "best_cap": best_cap,
        "deficit": N_TARGET - best_cap,
        "best_L": best_L,
        "best_cfg": {k: v for k, v in best.items() if k != "state"},
        "top5": [{k: v for k, v in r.items() if k != "state"} for r in results[:5]],
        "certified_win": certified_win,
        "seconds": round(time.time() - t0, 2),
    }
    out_path = THIS_DIR.parent / "docs" / "v6_slp_squeeze.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(
        f"Variant 6 done in {summary['seconds']}s: best_cap={best_cap}, best_L={best_L} vs L_ours={L_ours:.6f}, "
        f"win={certified_win is not None}",
        flush=True,
    )


if __name__ == "__main__":
    main()
