"""Variant 2 for n=634: Add POC3 end-taper / upright_bottom_taper & fan unfolding caps at both pocket ends.

Keeps symmetric arm decomposition (Kx=Ky=K-P, P1=P2=P, sw_upright or bottom-tapered SW boundary),
adds upright staircase end-caps flush against walls + 1-2 fan-angle transition rows + soft-insert relax.
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


def _squares_overlap(gx: float, gy: float, ga: float, ox: float, oy: float, oa: float, margin: float = 1e-7) -> bool:
    dx = gx - ox
    dy = gy - oy
    if abs(dx) >= 1.45 or abs(dy) >= 1.45:
        return False
    ca, sa = math.cos(ga), math.sin(ga)
    cj, sj = math.cos(oa), math.sin(oa)
    rad = 0.5 + 0.5 * (abs(ca * cj + sa * sj) + abs(ca * sj - sa * cj))
    return max(abs(dx * ca + dy * sa), abs(-dx * sa + dy * ca), abs(dx * cj + dy * sj), abs(-dx * sj + dy * cj)) < rad + margin


def build_tapered_arm(
    W: float,
    y_min: float,
    y_max: float,
    w: int,
    th: float,
    sign: int,
    taper_rows_bot: int,
    taper_rows_top: int,
    u_cols: int = 0,
    y_phase: float = 0.0,
    mode: str = "splice",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build sheared strip [0, W] x [y_min, y_max] with flush upright + fan-angled end-taper caps."""
    W_sh = W - u_cols
    if W_sh < 1.5:
        return np.empty(0), np.empty(0), np.empty(0)
    c, s = math.cos(th), math.sin(th)
    hx = 0.5 * (c + s)
    dy_row = 1.0 / c
    x_pad = 0.5 * max(0.0, W_sh - (w * c + s))

    y_lo_sh = y_min + (taper_rows_bot if mode == "reserve" else 0)
    y_hi_sh = y_max - (taper_rows_top if mode == "reserve" else 0)

    uxs, uys, uths = [], [], []
    # Optional upright columns along right border [W_sh, W]
    if u_cols > 0:
        ny = int(math.floor(y_max - y_min + 1e-9))
        for ci in range(u_cols):
            for rj in range(ny):
                uxs.append(W_sh + 0.5 + ci)
                uys.append(y_min + 0.5 + rj)
                uths.append(0.0)

    # Sheared core
    sxs, sys_, sths = [], [], []
    r_min = -int(math.ceil(w * s / dy_row)) - 2
    r_max = int(math.ceil((y_hi_sh - y_lo_sh + w * s) / dy_row)) + 2
    for r in range(r_min, r_max + 1):
        y0 = y_lo_sh + hx + (r + y_phase) * dy_row
        for k in range(w):
            xk = x_pad + hx + k * c
            yk = y0 + sign * k * s
            if xk - hx >= -1e-9 and xk + hx <= W_sh + 1e-9 and yk - hx >= y_lo_sh - 1e-9 and yk + hx <= y_hi_sh + 1e-9:
                sxs.append(xk)
                sys_.append(yk)
                sths.append(sign * th)

    P_sh = int(math.floor(W_sh + 1e-9))
    if mode == "splice" and (taper_rows_bot > 0 or taper_rows_top > 0):
        # Place flush upright end rows across [0, W_sh] (trying full row P_sh split at best gap)
        # and ONLY remove conflicting sheared squares if cap >= conflict count!
        cap_pts: list[tuple[float, float, float]] = []
        for j in range(taper_rows_bot):
            yc = y_min + 0.5 + j
            if yc > y_max - 0.5:
                continue
            for i in range(P_sh):
                # Split left-flush and right-flush at midpoint so gap aligns cleanly
                xc = (0.5 + i) if i < (P_sh + 1) // 2 else (W_sh - 0.5 - (P_sh - 1 - i))
                cap_pts.append((xc, yc, 0.0))
        for j in range(taper_rows_top):
            yc = y_max - 0.5 - j
            if yc < y_min + 0.5:
                continue
            if any(abs(yc - oy) < 0.99 for _, oy, _ in cap_pts):
                continue
            for i in range(P_sh):
                xc = (0.5 + i) if i < (P_sh + 1) // 2 else (W_sh - 0.5 - (P_sh - 1 - i))
                cap_pts.append((xc, yc, 0.0))

        keep = [True] * len(sxs)
        for sx, sy, st in cap_pts:
            for idx_s in range(len(sxs)):
                if keep[idx_s] and _squares_overlap(sx, sy, st, sxs[idx_s], sys_[idx_s], sths[idx_s]):
                    keep[idx_s] = False
        sxs = [sxs[i] for i in range(len(sxs)) if keep[i]]
        sys_ = [sys_[i] for i in range(len(sys_)) if keep[i]]
        sths = [sths[i] for i in range(len(sths)) if keep[i]]
        uxs.extend(p[0] for p in cap_pts)
        uys.extend(p[1] for p in cap_pts)
        uths.extend(p[2] for p in cap_pts)

    xs = uxs + sxs
    ys = uys + sys_
    ths = uths + sths
    arr_x = np.array(xs, dtype=np.float64)
    arr_y = np.array(ys, dtype=np.float64)
    arr_t = np.array(ths, dtype=np.float64)
    arr_c = np.cos(arr_t)
    arr_s = np.sin(arr_t)

    # Extra wall-flush + fan-angled greedy end fill inside [0, W_sh]
    x_upright_cands = []
    for i in range(P_sh + 1):
        if 0.5 + i <= W_sh - 0.5 + 1e-7:
            x_upright_cands.append(0.5 + i)
        if W_sh - 0.5 - i >= 0.5 - 1e-7:
            x_upright_cands.append(W_sh - 0.5 - i)
    x_upright_cands = sorted(set(round(v, 7) for v in x_upright_cands))

    cand_pts = []
    for j in range(max(4, taper_rows_bot + 3)):
        yc = y_min + 0.5 + j
        if yc <= y_max - 0.5:
            for xc in x_upright_cands:
                cand_pts.append((xc, yc))
    for j in range(max(4, taper_rows_top + 3)):
        yc = y_max - 0.5 - j
        if yc >= y_min + 0.5:
            for xc in x_upright_cands:
                cand_pts.append((xc, yc))

    angs = [0.0, 0.35 * sign * th, 0.65 * sign * th, sign * th]
    for gx, gy in cand_pts:
        if len(arr_x) > 0 and float(np.min((arr_x - gx) ** 2 + (arr_y - gy) ** 2)) < 0.72:
            continue
        for ang in angs:
            ca, sa = math.cos(ang), math.sin(ang)
            hxa = 0.5 * (abs(ca) + abs(sa))
            if gx - hxa < -1e-7 or gx + hxa > W_sh + 1e-7 or gy - hxa < y_min - 1e-7 or gy + hxa > y_max + 1e-7:
                continue
            ok = True
            for idx in range(len(arr_x)):
                dx = gx - arr_x[idx]
                dy = gy - arr_y[idx]
                if abs(dx) >= 1.45 or abs(dy) >= 1.45:
                    continue
                cj, sj = arr_c[idx], arr_s[idx]
                rad = 0.5 + 0.5 * (abs(ca * cj + sa * sj) + abs(ca * sj - sa * cj))
                if max(abs(dx * ca + dy * sa), abs(-dx * sa + dy * ca), abs(dx * cj + dy * sj), abs(-dx * sj + dy * cj)) < rad + 1e-7:
                    ok = False
                    break
            if ok:
                xs.append(gx)
                ys.append(gy)
                ths.append(ang)
                arr_x = np.append(arr_x, gx)
                arr_y = np.append(arr_y, gy)
                arr_t = np.append(arr_t, ang)
                arr_c = np.append(arr_c, ca)
                arr_s = np.append(arr_s, sa)
                break

    return arr_x, arr_y, arr_t


def _eval_worker(task: tuple) -> dict:
    L_probe, P, w, u_cols, s1, s2, tb, tt, y_phase, mode = task
    K = int(math.floor(L_probe))
    Kx = Ky = K - P
    W = L_probe - Kx
    W_sh = W - u_cols
    th = pc.solve_shear_angle(w, W_sh - 1e-5)
    if th is None or th > 0.65:
        return {"cap": 0, "feasible_cap": 0}

    sw_p = int(math.floor(W))
    ux, uy, uth = pc.make_upright_grid(L_probe - Kx, L_probe, L_probe - Ky, L_probe)
    swx, swy, swth = pc.make_upright_grid(0.0, float(sw_p), 0.0, float(sw_p))

    wx, wy, wth = build_tapered_arm(W, float(sw_p) + 1e-5, L_probe, w, th, s1, tb, tt, u_cols=u_cols, y_phase=y_phase, mode=mode)
    sx0, sy0, sth0 = build_tapered_arm(W, float(sw_p) + 1e-5, L_probe, w, th, s2, tb, tt, u_cols=u_cols, y_phase=y_phase, mode=mode)
    sx, sy, sth = sy0, sx0, -sth0

    x = np.concatenate([ux, swx, wx, sx])
    y = np.concatenate([uy, swy, wy, sy])
    th_arr = np.concatenate([uth, swth, wth, sth])

    pockets = [(0.0, W, float(sw_p), L_probe), (float(sw_p), L_probe, 0.0, W)]
    cands = [0.0, 0.35 * s1 * th, 0.65 * s1 * th, s1 * th, -s2 * th, math.pi / 4.0, -math.pi / 4.0]
    need_extra = max(0, N_TARGET - len(x))
    x_f, y_f, th_f, soft_holes = pc.fill_end_and_void_pockets(
        x, y, th_arr, L_probe, pockets, cands, grid_step=0.125, extra_soft_candidates=max(4, need_extra + 2)
    )
    rigid_cap = len(x_f)
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
        "u_cols": u_cols,
        "th_deg": round(math.degrees(th), 3),
        "signs": (s1, s2),
        "taper_bot": tb,
        "taper_top": tt,
        "y_phase": y_phase,
        "mode": mode,
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
    for P in range(5, 11):
        W = P + (L_probe - int(math.floor(L_probe)))
        for u_cols in (0, 1):
            W_sh = W - u_cols
            w_sh = int(math.floor(W_sh)) + 1
            if pc.solve_shear_angle(w_sh, W_sh - 1e-5) is None:
                continue
            for s1, s2 in ((1, 1), (-1, -1), (1, -1)):
                for tb, tt in ((0, 1), (1, 0), (1, 1), (2, 0), (0, 2), (2, 1), (1, 2), (2, 2)):
                    for y_phase in (0.0, 0.25, 0.5, 0.75):
                        for mode in ("splice", "reserve"):
                            tasks.append((L_probe, P, w_sh, u_cols, s1, s2, tb, tt, y_phase, mode))

    n_workers = os.cpu_count() or 4
    print(f"Running Variant 2 (end-taper + fan unfolding) on n={N_TARGET} ({len(tasks)} tasks, {n_workers} workers)...", flush=True)
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
        "variant": "v2_end_taper",
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
    out_path = THIS_DIR.parent / "docs" / "v2_end_taper.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(
        f"Variant 2 done in {summary['seconds']}s: best_cap={best_cap}, deficit={deficit}, "
        f"cfg={summary['best_cfg']}, best_L={best_L}, win={certified_win is not None}",
        flush=True,
    )


if __name__ == "__main__":
    main()
