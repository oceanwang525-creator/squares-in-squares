#!/usr/bin/env python3
"""Idea 6: Masked-Pocket Cavity-Relocation (L0=L_inc+0.032, a=0.78) + Frozen-Core Border-Seam Peels."""
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
OUT_JSON = HERE.parent / "docs" / "idea6_results.json"
FINAL_CSV = HERE.parent / "final.csv"
DATA_DIR = HERE.parent / "data"


def _mask3(active: np.ndarray) -> np.ndarray:
    m = active.astype(np.uint8)
    return np.concatenate([m, m, m])


def build_structural_seeds(
    n: int,
    x_base: np.ndarray,
    y_base: np.ndarray,
    th_base: np.ndarray,
    L_base: float,
    L0_pocket: float,
    mats: dict,
    rng: np.random.Generator,
) -> tuple[list[np.ndarray], list[np.ndarray], list[list[float]], list[str]]:
    """Build Mechanism A (L0=+0.032 masked pocket cavity-relocate) & Mechanism B (frozen-core surplus peel)."""
    q_list, m_list, ls_list, labels = [], [], [], []

    sc = L0_pocket / L_base
    xb, yb, tb = x_base * sc, y_base * sc, th_base.copy()
    hx = 0.5 * (np.abs(np.cos(tb)) + np.abs(np.sin(tb)))
    margin = np.minimum.reduce([xb - hx, L0_pocket - xb - hx, yb - hx, L0_pocket - yb - hx])
    tight_idx = np.argsort(margin)[:18]
    voids = i1.clearance_points(xb, yb, tb, L0_pocket, n_pts=8, step=0.14)
    tilt = i1.donor_tilt(tb)

    deg = np.degrees(tb) % 90.0
    deg_m = np.minimum(deg, 90.0 - deg)
    up_mask = deg_m < 1.0
    tilted_idx = np.where(~up_mask)[0]
    seam_idx = tilted_idx
    if np.any(up_mask) and len(tilted_idx) > 0:
        d_ut = np.hypot(xb[tilted_idx, None] - xb[None, up_mask], yb[tilted_idx, None] - yb[None, up_mask])
        close_t = tilted_idx[np.where(d_ut.min(axis=1) < 1.35)[0]]
        if len(close_t) > 0:
            seam_idx = close_t

    mL = xb - hx - float((xb - hx).min())
    mR = float((xb + hx).max()) - (xb + hx)
    mB = yb - hx - float((yb - hx).min())
    mT = float((yb + hx).max()) - (yb + hx)
    cL, cR = int(np.sum(mL < 5e-4)), int(np.sum(mR < 5e-4))
    cB, cT = int(np.sum(mB < 5e-4)), int(np.sum(mT < 5e-4))
    dx_sign = 1.0 if cR <= cL else -1.0
    dy_sign = 1.0 if cT <= cB else -1.0

    sched_p = [L0_pocket, L_base + 0.008, L_base - 2e-4, L_base - 0.002]

    # Mechanism A1: Void-centered masked pockets with L0=+0.032 + cavity relocate + wall-pin swap (12 seeds)
    for s in range(12):
        x, y, th = xb.copy(), yb.copy(), tb.copy()
        vx, vy, _ = voids[s % len(voids)] if voids else (0.5 * L0_pocket, 0.5 * L0_pocket, 0.1)
        rad = float(rng.uniform(5.2, 7.2))
        near = ((x - vx) ** 2 + (y - vy) ** 2) <= rad * rad
        if len(tight_idx) > 0 and s % 2 == 0:
            pi = int(tight_idx[(s // 2) % len(tight_idx)])
            px_old, py_old = float(x[pi]), float(y[pi])
            x[pi] = vx + float(rng.uniform(-0.08, 0.08))
            y[pi] = vy + float(rng.uniform(-0.08, 0.08))
            th[pi] = [0.0, np.pi / 4.0, tilt][(s // 2) % 3] + float(rng.uniform(-0.10, 0.10))
            near |= ((x - px_old) ** 2 + (y - py_old) ** 2) <= 3.5 ** 2
            near[pi] = True
        nm = int(near.sum())
        x[near] += rng.uniform(-0.14, 0.14, nm)
        y[near] += rng.uniform(-0.14, 0.14, nm)
        th[near] += rng.uniform(-0.28, 0.28, nm)
        q_list.append(full_to_free_sym(x, y, th, L0_pocket))
        m_list.append(_mask3(near))
        ls_list.append(sched_p)
        labels.append(f"mpk_void_{s}")

    # Mechanism A2: Weak-wall / boundary-pin pocket with asymmetric slack drift (12 seeds)
    for s in range(12):
        x, y, th = xb.copy(), yb.copy(), tb.copy()
        pi = int(tight_idx[s % len(tight_idx)])
        cx, cy = float(x[pi]), float(y[pi])
        rad = float(rng.uniform(5.2, 7.4))
        near = ((x - cx) ** 2 + (y - cy) ** 2) <= rad * rad
        nm = int(near.sum())
        drift = float(rng.uniform(0.03, 0.07))
        x[near] += dx_sign * drift * 0.5 + rng.uniform(-0.13, 0.13, nm)
        y[near] += dy_sign * drift * 0.5 + rng.uniform(-0.13, 0.13, nm)
        th[near] += rng.uniform(-0.30, 0.30, nm)
        q_list.append(full_to_free_sym(x, y, th, L0_pocket))
        m_list.append(_mask3(near))
        ls_list.append(sched_p)
        labels.append(f"mpk_wall_{s}")

    # Mechanism A3: Corridor seam pockets (8 seeds)
    for s in range(8):
        x, y, th = xb.copy(), yb.copy(), tb.copy()
        if len(seam_idx) > 0:
            pi = int(seam_idx[(s * 17) % len(seam_idx)])
            cx, cy = float(x[pi]), float(y[pi])
        else:
            cx, cy = 0.5 * L0_pocket, 0.5 * L0_pocket
        rad = float(rng.uniform(5.2, 7.2))
        near = ((x - cx) ** 2 + (y - cy) ** 2) <= rad * rad
        nm = int(near.sum())
        x[near] += rng.uniform(-0.13, 0.13, nm)
        y[near] += rng.uniform(-0.13, 0.13, nm)
        th[near] += rng.uniform(-0.28, 0.28, nm)
        q_list.append(full_to_free_sym(x, y, th, L0_pocket))
        m_list.append(_mask3(near))
        ls_list.append(sched_p)
        labels.append(f"mpk_seam_{s}")

    # Mechanism B: Frozen-core border-seam surplus peels (top 2 donors, 4 seeds each)
    peels = i1.collect_surplus_peels(n, L_base, mats)[:2]
    for p in peels:
        L_box = p["L"]
        L0_s = max(L0_pocket, min(L_base + 0.12, L_box + min(0.09, p["max_oh_rem"]) + 0.02))
        sched_s = [L0_s, L_base + 0.010, L_base - 2e-4, L_base - 0.002]
        x0, y0, th0, oh0 = p["x"], p["y"], p["th"], p["oh"]
        surplus = p["surplus"]
        hx0 = 0.5 * (np.abs(np.cos(th0)) + np.abs(np.sin(th0)))
        rank = np.argsort(-oh0)
        pool_idx = rank[: max(surplus + 5, min(len(rank), 12))]
        sc_s = L0_s / L_box
        for s in range(4):
            if surplus == 0:
                keep = np.ones(len(x0), dtype=bool)
            elif s == 0:
                keep = np.ones(len(x0), dtype=bool)
                keep[rank[:surplus]] = False
            else:
                keep = np.ones(len(x0), dtype=bool)
                keep[rng.choice(pool_idx, size=surplus, replace=False)] = False
            x = x0[keep] * sc_s
            y = y0[keep] * sc_s
            th = th0[keep].copy()
            hxs = hx0[keep]
            ohs = oh0[keep]
            x = np.clip(x, hxs + 0.004, L0_s - hxs - 0.004)
            y = np.clip(y, hxs + 0.004, L0_s - hxs - 0.004)
            # Freeze deep interior where oh == 0 and away from walls / seam; melt overhang border strip
            deg_k = np.degrees(th) % 90.0
            is_tilt_k = np.minimum(deg_k, 90.0 - deg_k) > 1.5
            near_edge = (ohs > 1e-4) | (x < 3.0) | (x > L0_s - 3.0) | (y < 3.0) | (y > L0_s - 3.0) | is_tilt_k
            if np.sum(near_edge) > int(0.65 * n):
                near_edge = (ohs > 1e-4) | (x < 2.5) | (x > L0_s - 2.5) | (y < 2.5) | (y > L0_s - 2.5)
            nm = int(near_edge.sum())
            if nm > 0:
                x[near_edge] += rng.uniform(-0.09, 0.09, nm)
                y[near_edge] += rng.uniform(-0.09, 0.09, nm)
                th[near_edge] += rng.uniform(-0.18, 0.18, nm)
            q_list.append(full_to_free_sym(x, y, th, L0_s))
            m_list.append(_mask3(near_edge))
            ls_list.append(sched_s)
            labels.append(f"fpeel_{p['src']}_s{s}")

    return q_list, m_list, ls_list, labels


def _polish_worker(args: tuple) -> dict:
    idx, label, n, L_base, x, y, th, cpp_pen, cpp_L = args
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
            x, y, th, rounds=10, tr_th0=0.020, tr_xy0=0.09, margin=1e-8, tol=1e-8, patience=2
        )
        if ok_s:
            x_f, y_f, th_f, L_slp, ok_slp = xs, ys, ts, Ls, True
            if L_slp < L_base + 5e-4:
                xs2, ys2, ts2, Ls2, ok_s2 = joint_xytheta_slp(
                    x_f, y_f, th_f, rounds=12, tr_th0=0.006, tr_xy0=0.025, margin=1e-8, tol=1e-11, patience=2
                )
                if ok_s2 and Ls2 < L_slp:
                    x_f, y_f, th_f, L_slp = xs2, ys2, ts2, Ls2

    ver: dict = {"ok": False}
    best_ver_L = 1e9
    for m_lp in (1e-7, 2e-8, 5e-9, 1e-6):
        xf2, yf2, L_lp2, ok_lp2 = u.fixed_angle_lp_tighten(x_f, y_f, th_f, rounds=5, margin=m_lp)
        if ok_lp2 and L_lp2 <= L_slp + 5e-6:
            if L_lp2 < L_base - 1e-11 or not ver.get("ok"):
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


def solve_target(n: int, mats: dict, inc_front: dict, wins: dict, n_threads: int) -> dict:
    assert n in i1.OWNED_TARGETS
    t0 = time.time()
    L_inc = inc_front[n]["L_ours"]
    L_front = inc_front[n]["L_front"]
    if n in wins:
        sub = pd.read_csv(DATA_DIR / f"win_n{n}.csv")
        sub["sid"] = sub["id"].astype(str).str.split("_").str[1].astype(int)
        sub = sub.sort_values("sid")
        x_b = sub["x"].to_numpy(np.float64)
        y_b = sub["y"].to_numpy(np.float64)
        th_b = np.radians(sub["deg"].to_numpy(np.float64))
        L_b = float(wins[n]["verify"]["L_float"])
    else:
        x_b, y_b, th_b, L_b = mats[n]

    rng = np.random.default_rng(60600 + n)
    L0_pocket = L_b + 0.032
    q_list, m_list, ls_list, labels = build_structural_seeds(n, x_b, y_b, th_b, L_b, L0_pocket, mats, rng)
    ils_iters = [5, 8, 12, 15]

    qb, qf, lf, eo, po, lb = run_cpp_masked_melt(
        n, 1, q_list, m_list, ls_list, ils_iters,
        a_start=0.78, n_melt_stages=4, masked_iters=180, full_iters=240,
        n_threads=n_threads, base_seed=77000 + n,
    )
    n_feas = int(np.sum(lf < 1e9))
    best_feas = float(np.min(lf)) if n_feas else None
    print(f"\n=== Idea 6 n={n} | L_b={L_b:.9f} | N={len(labels)} feas={n_feas} best_Lf={best_feas} min_pen={float(np.min(po)):.2e} ===", flush=True)

    score = lambda i: (0 if lf[i] < 1e9 else 1, float(lf[i]) if lf[i] < 1e9 else float(po[i]))
    order = sorted(range(len(labels)), key=score)
    keep, used = [], set()
    for pref, qta in (("mpk_void_", 2), ("mpk_wall_", 2), ("mpk_seam_", 1), ("fpeel_", 2)):
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
        L_ref = float(lf[i]) if use_f else ls_list[i][-1]
        x, y, th = cpp_expand_sym(n, 1, L_ref, qf[i] if use_f else qb[i])
        tasks.append((i, labels[i], n, L_b, x, y, th, 0.0 if use_f else float(po[i]),
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
            if Lf < L_b - 1e-11 and (best_win is None or Lf < float(best_win["verify"]["L_float"])):
                best_win = r

    if best_valid is not None:
        v = best_valid["verify"]
        dL = float(v["L_float"]) - L_b
        kind = "REAL WIN" if dL <= -1e-9 else ("TIE" if dL < 0.0 else "no_win")
        print(f"  BEST VALID n={n}: {best_valid['label']} L_lit={v['L_lit'][:16]} (dL={dL:+.3e}, {kind}) [{time.time()-t0:.1f}s]", flush=True)
    else:
        print(f"  NO VALID POLISHED n={n} [{time.time()-t0:.1f}s]", flush=True)

    return {
        "n": n, "L_inc": L_inc, "L_base": L_b, "L_front": L_front, "dt_s": round(time.time() - t0, 2),
        "n_starts": len(labels), "n_feas": n_feas, "best_cpp_Lf": best_feas,
        "top_polished": [{k: v for k, v in r.items() if k != "rows"} for r in polished],
        "win": best_win,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", type=int, nargs="+", default=[698, 860, 545, 703])
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
    print(f"\nIdea 6 done in {time.time()-t_all:.1f}s | wins={sorted(wins.keys())}", flush=True)

    if wins:
        u.write_final_csv_from_wins(wins, FINAL_CSV)
        print(f"Updated {FINAL_CSV} with wins={sorted(wins.keys())}", flush=True)


if __name__ == "__main__":
    main()
