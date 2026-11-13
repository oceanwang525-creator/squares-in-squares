#!/usr/bin/env python3
"""Idea 4: Frozen-Exterior Masked-Pocket C++ a-Melt + Void/Weak-Wall Re-Packing + Fine SLP."""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import utils as u
from cpp_wrap import cpp_eval_sat_pen_bbox, cpp_expand_sym, full_to_free_sym, run_cpp_masked_melt
from joint_slp import joint_xytheta_slp
import idea1_highd_pilot as i1
import idea2_highd_sweep as i2

HERE = Path(__file__).resolve().parent
OUT_JSON = HERE.parent / "docs" / "idea4_results.json"
FINAL_CSV = HERE.parent / "final.csv"
DATA_DIR = HERE.parent / "data"


def _mask_from_bool(active: np.ndarray) -> np.ndarray:
    """Convert n-length boolean array into 3*n uint8 mask for sym_mode=1."""
    m = active.astype(np.uint8)
    return np.concatenate([m, m, m])


def build_masked_pocket_seeds(
    n: int,
    x_inc: np.ndarray,
    y_inc: np.ndarray,
    th_inc: np.ndarray,
    L_inc: float,
    L0: float,
    mats: dict,
    rng: np.random.Generator,
) -> tuple[list[np.ndarray], list[np.ndarray], list[str]]:
    """Build masked pocket seeds: void-bridge, boundary-pin, upright/tilted seam, and (n+1)->n del."""
    sc = L0 / L_inc
    xb, yb, tb = x_inc * sc, y_inc * sc, th_inc.copy()
    hx = 0.5 * (np.abs(np.cos(tb)) + np.abs(np.sin(tb)))
    margin = np.minimum.reduce([xb - hx, L0 - xb - hx, yb - hx, L0 - yb - hx])
    tight_idx = np.argsort(margin)[:18]
    voids = i1.clearance_points(xb, yb, tb, L0, n_pts=8, step=0.14)
    tilt = i1.donor_tilt(tb)

    deg = np.degrees(tb) % 90.0
    deg_m = np.minimum(deg, 90.0 - deg)
    up_mask = deg_m < 1.0
    tilted_mask = ~up_mask
    tilted_idx = np.where(tilted_mask)[0]
    seam_idx = tilted_idx
    if np.any(up_mask) and np.any(tilted_mask):
        d_ut = np.hypot(xb[tilted_mask, None] - xb[None, up_mask], yb[tilted_mask, None] - yb[None, up_mask])
        close_t = tilted_idx[np.where(d_ut.min(axis=1) < 1.35)[0]]
        if len(close_t) > 0:
            seam_idx = close_t

    q_list, m_list, labels = [], [], []

    # 1. Void-bridge pockets (12 seeds)
    for s in range(12):
        x, y, th = xb.copy(), yb.copy(), tb.copy()
        vx, vy, _ = voids[s % len(voids)] if voids else (float(L0 * 0.5), float(L0 * 0.5), 0.1)
        rad = float(rng.uniform(4.2, 6.2))
        near = ((x - vx) ** 2 + (y - vy) ** 2) <= rad * rad
        if s % 2 == 0 and len(tight_idx) > 0:
            pi = int(tight_idx[(s // 2) % len(tight_idx)])
            px_old, py_old = float(x[pi]), float(y[pi])
            x[pi] = vx + float(rng.uniform(-0.06, 0.06))
            y[pi] = vy + float(rng.uniform(-0.06, 0.06))
            th[pi] = [0.0, np.pi / 4.0, tilt][(s // 2) % 3] + float(rng.uniform(-0.08, 0.08))
            near |= ((x - px_old) ** 2 + (y - py_old) ** 2) <= 3.2 ** 2
            near[pi] = True
        nm = int(near.sum())
        x[near] += rng.uniform(-0.10, 0.10, nm)
        y[near] += rng.uniform(-0.10, 0.10, nm)
        th[near] += rng.uniform(-0.22, 0.22, nm)
        q_list.append(full_to_free_sym(x, y, th, L0))
        m_list.append(_mask_from_bool(near))
        labels.append(f"mvoid_{s}")

    # 2. Boundary-pin pockets (12 seeds)
    for s in range(12):
        x, y, th = xb.copy(), yb.copy(), tb.copy()
        pi = int(tight_idx[s % len(tight_idx)])
        cx, cy = float(x[pi]), float(y[pi])
        rad = float(rng.uniform(4.5, 6.8))
        near = ((x - cx) ** 2 + (y - cy) ** 2) <= rad * rad
        nm = int(near.sum())
        x[near] += rng.uniform(-0.11, 0.11, nm)
        y[near] += rng.uniform(-0.11, 0.11, nm)
        th[near] += rng.uniform(-0.25, 0.25, nm)
        q_list.append(full_to_free_sym(x, y, th, L0))
        m_list.append(_mask_from_bool(near))
        labels.append(f"mpin_{s}")

    # 3. Seam pockets along upright/tilted corridor interface (12 seeds)
    for s in range(12):
        x, y, th = xb.copy(), yb.copy(), tb.copy()
        if len(seam_idx) > 0:
            pi = int(seam_idx[(s * 13) % len(seam_idx)])
            cx, cy = float(x[pi]), float(y[pi])
        else:
            cx, cy = float(L0 * 0.5), float(L0 * 0.5)
        rad = float(rng.uniform(4.8, 6.8))
        near = ((x - cx) ** 2 + (y - cy) ** 2) <= rad * rad
        nm = int(near.sum())
        x[near] += rng.uniform(-0.11, 0.11, nm)
        y[near] += rng.uniform(-0.11, 0.11, nm)
        th[near] += rng.uniform(-0.25, 0.25, nm)
        q_list.append(full_to_free_sym(x, y, th, L0))
        m_list.append(_mask_from_bool(near))
        labels.append(f"mseam_{s}")

    # 4. Masked (n+1)->n 1-square deletion pockets (6 seeds)
    if (n + 1) in mats:
        x1, y1, th1, L1 = mats[n + 1]
        if L1 <= L_inc + 0.05:
            sc1 = L0 / L1
            xb1, yb1, tb1 = x1 * sc1, y1 * sc1, th1.copy()
            hx1 = 0.5 * (np.abs(np.cos(tb1)) + np.abs(np.sin(tb1)))
            wm1 = np.minimum.reduce([xb1 - hx1, L0 - xb1 - hx1, yb1 - hx1, L0 - yb1 - hx1])
            cand1 = np.argsort(wm1)[:12]
            for s in range(6):
                drop = int(cand1[s % len(cand1)])
                cx, cy = float(xb1[drop]), float(yb1[drop])
                keep = np.ones(n + 1, dtype=bool)
                keep[drop] = False
                x, y, th = xb1[keep].copy(), yb1[keep].copy(), tb1[keep].copy()
                near = ((x - cx) ** 2 + (y - cy) ** 2) <= 5.2 ** 2
                nm = int(near.sum())
                x[near] += rng.uniform(-0.11, 0.11, nm)
                y[near] += rng.uniform(-0.11, 0.11, nm)
                th[near] += rng.uniform(-0.22, 0.22, nm)
                q_list.append(full_to_free_sym(x, y, th, L0))
                m_list.append(_mask_from_bool(near))
                labels.append(f"mdel_{s}")

    return q_list, m_list, labels


def _polish_worker(args: tuple) -> dict:
    idx, label, n, L_inc, x, y, th, cpp_E, cpp_pen, cpp_L = args
    t0 = time.time()
    hx = 0.5 * (np.abs(np.cos(th)) + np.abs(np.sin(th)))
    x = x - float((x - hx).min())
    y = y - float((y - hx).min())

    if cpp_pen > 1e-7 and cpp_pen < 0.025:
        L_try = cpp_L + max(0.002, 1.8 * cpp_pen)
        xr, yr, thr, _ = u.relax_lbfgs(x, y, th, L_try, maxiter=160, w_wall=4.0, w_pair=1.0, side=1.0)
        pen_r, L_r = cpp_eval_sat_pen_bbox(xr, yr, thr, side=1.0)
        if pen_r <= cpp_pen + 1e-6:
            x, y, th, cpp_pen, cpp_L = xr, yr, thr, pen_r, L_r

    pen_pre, L_pre = cpp_eval_sat_pen_bbox(x, y, th, side=1.0)
    ok_slp, L_slp = False, L_pre
    x_f, y_f, th_f = x.copy(), y.copy(), th.copy()

    if pen_pre < 0.015:
        xs, ys, ts, Ls, ok_s = joint_xytheta_slp(
            x, y, th, rounds=8, tr_th0=0.018, tr_xy0=0.08, margin=1e-8, tol=1e-8, patience=2
        )
        if ok_s:
            x_f, y_f, th_f, L_slp, ok_slp = xs, ys, ts, Ls, True
            if L_slp < L_inc + 4e-4:
                xs2, ys2, ts2, Ls2, ok_s2 = joint_xytheta_slp(
                    x_f, y_f, th_f, rounds=12, tr_th0=0.005, tr_xy0=0.02, margin=1e-8, tol=1e-11, patience=2
                )
                if ok_s2 and Ls2 < L_slp:
                    x_f, y_f, th_f, L_slp = xs2, ys2, ts2, Ls2

    ver: dict = {"ok": False}
    best_ver_L = 1e9
    for m_lp in (1e-7, 2e-8, 5e-9, 1e-6):
        xf2, yf2, L_lp2, ok_lp2 = u.fixed_angle_lp_tighten(x_f, y_f, th_f, rounds=5, margin=m_lp)
        if ok_lp2 and L_lp2 <= L_slp + 5e-6:
            if L_lp2 < L_inc - 1e-11 or not ver.get("ok"):
                cand_ver = u.verify_official(n, xf2, yf2, th_f)
                if cand_ver.get("ok"):
                    cL = float(cand_ver["L_float"])
                    if cL < best_ver_L:
                        best_ver_L = cL
                        x_f, y_f, L_slp, ok_slp, ver = xf2, yf2, L_lp2, True, cand_ver
                elif not ver.get("ok"):
                    ver = cand_ver

    pen_fin, L_fin = cpp_eval_sat_pen_bbox(x_f, y_f, th_f, side=1.0)
    return {
        "idx": idx, "label": label, "n": n,
        "cpp_pen": float(cpp_pen), "cpp_L": float(cpp_L),
        "ok_slp": bool(ok_slp), "pen_fin": float(pen_fin), "L_fin": float(L_fin),
        "verify": {k: v for k, v in ver.items() if k != "rows"},
        "rows": ver.get("rows"),
        "dt": round(time.time() - t0, 2),
    }


def solve_target(n: int, mats: dict, inc_front: dict, current_wins: dict, n_threads: int) -> dict:
    assert n in i1.OWNED_TARGETS
    t0 = time.time()
    L_inc = inc_front[n]["L_ours"]
    L_front = inc_front[n]["L_front"]
    # Start from best known configuration for n (including prior win!)
    if n in current_wins:
        sub = pd.read_csv(DATA_DIR / f"win_n{n}.csv")
        sub["sid"] = sub["id"].astype(str).str.split("_").str[1].astype(int)
        sub = sub.sort_values("sid")
        x_inc = sub["x"].to_numpy(np.float64)
        y_inc = sub["y"].to_numpy(np.float64)
        th_inc = np.radians(sub["deg"].to_numpy(np.float64))
        L_base = float(current_wins[n]["verify"]["L_float"])
    else:
        x_inc, y_inc, th_inc, L_base = mats[n]

    rng = np.random.default_rng(40400 + n)
    L0 = L_base + 0.004
    L_sched = [L0, L_base - 1.5e-4, L_base - 0.0012]
    ils_iters = [4, 7, 9]

    q_list, m_list, labels = build_masked_pocket_seeds(n, x_inc, y_inc, th_inc, L_base, L0, mats, rng)
    ls_list = [L_sched] * len(q_list)

    qb, qf, lf, eo, po, lb = run_cpp_masked_melt(
        n, 1, q_list, m_list, ls_list, ils_iters,
        a_start=0.80, n_melt_stages=4, masked_iters=160, full_iters=220,
        n_threads=n_threads, base_seed=66000 + n,
    )
    n_feas = int(np.sum(lf < 1e9))
    best_feas = float(np.min(lf)) if n_feas else None
    print(f"\n=== Idea 4 Masked n={n} | L_base={L_base:.9f} | N={len(labels)} feas={n_feas} best_Lf={best_feas} min_pen={float(np.min(po)):.2e} ===", flush=True)

    score = lambda i: (0 if lf[i] < 1e9 else 1, float(lf[i]) if lf[i] < 1e9 else float(po[i]))
    order = sorted(range(len(labels)), key=score)
    keep, used = [], set()
    for pref, qta in (("mvoid_", 2), ("mpin_", 2), ("mseam_", 2), ("mdel_", 1)):
        c = 0
        for i in order:
            if i not in used and labels[i].startswith(pref):
                keep.append(i); used.add(i); c += 1
                if c >= qta or len(keep) >= 6:
                    break
    for i in order:
        if len(keep) >= 6:
            break
        if i not in used:
            keep.append(i); used.add(i)

    tasks = []
    for i in keep:
        use_f = lf[i] < 1e9
        L_ref = float(lf[i]) if use_f else L_sched[-1]
        x, y, th = cpp_expand_sym(n, 1, L_ref, qf[i] if use_f else qb[i])
        tasks.append((i, labels[i], n, L_base, x, y, th, float(eo[i]), 0.0 if use_f else float(po[i]),
                      float(lf[i]) if use_f else float(lb[i])))

    with mp.Pool(processes=min(4, max(1, n_threads), len(tasks))) as pool:
        polished = pool.map(_polish_worker, tasks)
    polished.sort(key=lambda r: (0 if r["verify"].get("ok") else 1, float(r["verify"].get("L_float", 1e9)) if r["verify"].get("ok") else r["pen_fin"]))

    best_valid, best_win = None, None
    for r in polished:
        if r["verify"].get("ok"):
            Lf = float(r["verify"]["L_float"])
            if best_valid is None or Lf < float(best_valid["verify"]["L_float"]):
                best_valid = r
            if Lf < L_base - 1e-11 and (best_win is None or Lf < float(best_win["verify"]["L_float"])):
                best_win = r

    if best_valid is not None:
        v = best_valid["verify"]
        print(f"  BEST VALID n={n}: {best_valid['label']} L_lit={v['L_lit'][:16]} (dL_base={float(v['L_float']) - L_base:+.3e}, win={best_win is not None}) [{time.time()-t0:.1f}s]", flush=True)
    else:
        print(f"  NO VALID POLISHED n={n} [{time.time()-t0:.1f}s]", flush=True)

    return {
        "n": n, "L_inc": L_inc, "L_base": L_base, "L_front": L_front, "dt_s": round(time.time() - t0, 2),
        "n_starts": len(labels), "n_feas": n_feas, "best_cpp_Lf": best_feas,
        "top_polished": [{k: v for k, v in r.items() if k != "rows"} for r in polished],
        "win": best_win,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", type=int, nargs="+", default=[639, 698, 918, 545])
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    n_threads = min(4, max(1, args.threads))

    for t in args.targets:
        if t not in i1.OWNED_TARGETS:
            raise ValueError(f"Refusing non-owned target n={t}")

    t_all = time.time()
    df = pd.read_csv(i1.MERGED_CSV)
    mats = {int(n): i1._norm(sub) for n, sub in df.groupby("n")}
    inc_front = u.load_incumbent_and_frontier(args.targets)
    wins = i2.load_verified_disk_wins(DATA_DIR)

    results = []
    for n in args.targets:
        res = solve_target(n, mats, inc_front, wins, n_threads=n_threads)
        w = res.pop("win")
        results.append(res)
        if w is not None:
            wins[n] = w
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            u.write_final_csv_from_wins({n: w}, DATA_DIR / f"win_n{n}.csv")
            print(f"  [SAVED WIN] {DATA_DIR / f'win_n{n}.csv'} L_pd={w['verify']['L_pandas'][:16]}", flush=True)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({"results": results, "dt_total_s": round(time.time() - t_all, 2)}, indent=2))
    print(f"\nIdea 4 done in {time.time()-t_all:.1f}s | wins={sorted(wins.keys())}", flush=True)

    if wins:
        u.write_final_csv_from_wins(wins, FINAL_CSV)
        print(f"Updated {FINAL_CSV} with wins={sorted(wins.keys())}", flush=True)


if __name__ == "__main__":
    main()
