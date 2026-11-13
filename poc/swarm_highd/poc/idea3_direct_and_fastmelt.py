#!/usr/bin/env python3
"""Idea 3: 14-Target Direct Joint (x, y, dtheta) SLP Audit + Ultra-Fast Dual-Regime (a=0.78 & a=0.92) Melts."""
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
from cpp_wrap import cpp_eval_sat_pen_bbox, cpp_expand_sym, full_to_free_sym, run_cpp_batch_ils
from joint_slp import joint_xytheta_slp
import idea1_highd_pilot as i1
import idea2_highd_sweep as i2

HERE = Path(__file__).resolve().parent
OUT_JSON_DIRECT = HERE.parent / "docs" / "idea3_direct_results.json"
OUT_JSON_MELT = HERE.parent / "docs" / "idea3_melt_results.json"
FINAL_CSV = HERE.parent / "final.csv"
DATA_DIR = HERE.parent / "data"


def _direct_worker(args: tuple) -> dict:
    n, x_inc, y_inc, th_inc, L_inc, L_front = args
    t0 = time.time()
    rng = np.random.default_rng(90000 + n)
    best_x, best_y, best_th = x_inc.copy(), y_inc.copy(), th_inc.copy()
    best_ver: dict = {"ok": False}
    best_L = 1e9
    best_tag = "raw"

    starts = [("fine", 12, 0.004, 0.015, 0.0, 0.0), ("med", 14, 0.025, 0.10, 0.0, 0.0)]
    for j in range(4):
        starts.append((f"jit{j}", 12, 0.012, 0.04, 1e-4, np.radians(0.02)))

    best_lp_L = 1e9
    for tag, rds, tth, txy, p_jit, a_jit in starts:
        if tag == "med" and best_lp_L < L_inc + 1e-6:
            x0, y0, th0 = best_x.copy(), best_y.copy(), best_th.copy()
        elif p_jit > 0.0:
            x0 = x_inc + rng.uniform(-p_jit, p_jit, n)
            y0 = y_inc + rng.uniform(-p_jit, p_jit, n)
            th0 = th_inc + rng.uniform(-a_jit, a_jit, n)
        else:
            x0, y0, th0 = x_inc.copy(), y_inc.copy(), th_inc.copy()

        xs, ys, ts, Ls, ok_s = joint_xytheta_slp(
            x0, y0, th0, rounds=rds, tr_th0=tth, tr_xy0=txy, margin=1e-8, tol=1e-11, patience=2
        )
        cx, cy, cth = (xs, ys, ts) if ok_s else (x0, y0, th0)
        for m_lp in (1e-7, 2e-8, 5e-9):
            xf, yf, L_lp, ok_lp = u.fixed_angle_lp_tighten(cx, cy, cth, rounds=5, margin=m_lp)
            if not ok_lp:
                continue
            if L_lp < L_inc - 1e-11:
                ver = u.verify_official(n, xf, yf, cth)
                if ver.get("ok"):
                    Lf = float(ver["L_float"])
                    if Lf < best_L:
                        best_L = Lf
                        best_x, best_y, best_th = xf.copy(), yf.copy(), cth.copy()
                        best_ver = ver
                        best_tag = f"{tag}_m{m_lp:.0e}"
            elif L_lp < best_lp_L:
                best_lp_L = L_lp
                if not best_ver.get("ok"):
                    best_x, best_y, best_th = xf.copy(), yf.copy(), cth.copy()
                    best_tag = f"{tag}_m{m_lp:.0e}"

    if not best_ver.get("ok"):
        best_ver = u.verify_official(n, best_x, best_y, best_th)
        if best_ver.get("ok"):
            best_L = float(best_ver["L_float"])

    win = best_ver.get("ok", False) and best_L < L_inc - 1e-11
    return {
        "n": n,
        "L_inc": L_inc,
        "L_front": L_front,
        "best_tag": best_tag,
        "L_best": best_L if best_ver.get("ok") else None,
        "dL": (best_L - L_inc) if best_ver.get("ok") else None,
        "win": bool(win),
        "verify": {k: v for k, v in best_ver.items() if k != "rows"},
        "rows": best_ver.get("rows") if win else None,
        "dt": round(time.time() - t0, 2),
    }


def _fast_polish_worker(args: tuple) -> dict:
    """Fast polish (<3s/candidate): 8-round early-stop SLP; upgrade only if L_slp < L_inc + 5e-4."""
    idx, label, fam, n, L_inc, x, y, th, cpp_E, cpp_pen, cpp_L = args
    t0 = time.time()

    hx = 0.5 * (np.abs(np.cos(th)) + np.abs(np.sin(th)))
    x = x - float((x - hx).min())
    y = y - float((y - hx).min())

    if cpp_pen > 1e-7 and cpp_pen < 0.03:
        L_try = cpp_L + max(0.002, 1.8 * cpp_pen)
        xr, yr, thr, _ = u.relax_lbfgs(x, y, th, L_try, maxiter=180, w_wall=4.0, w_pair=1.0, side=1.0)
        pen_r, L_r = cpp_eval_sat_pen_bbox(xr, yr, thr, side=1.0)
        if pen_r <= cpp_pen + 1e-6:
            x, y, th, cpp_pen, cpp_L = xr, yr, thr, pen_r, L_r

    pen_pre, L_pre = cpp_eval_sat_pen_bbox(x, y, th, side=1.0)
    ok_slp, L_slp = False, L_pre
    x_f, y_f, th_f = x.copy(), y.copy(), th.copy()

    if pen_pre < 0.02:
        xs, ys, ts, Ls, ok_s = joint_xytheta_slp(
            x, y, th, rounds=8, tr_th0=0.020, tr_xy0=0.10, margin=1e-8, tol=5e-8, patience=2
        )
        if ok_s:
            x_f, y_f, th_f, L_slp, ok_slp = xs, ys, ts, Ls, True
            if L_slp < L_inc + 5e-4:
                xs2, ys2, ts2, Ls2, ok_s2 = joint_xytheta_slp(
                    x_f, y_f, th_f, rounds=14, tr_th0=0.006, tr_xy0=0.025, margin=1e-8, tol=1e-10, patience=3
                )
                if ok_s2 and Ls2 < L_slp:
                    x_f, y_f, th_f, L_slp = xs2, ys2, ts2, Ls2

    ver: dict = {"ok": False}
    best_ver_L = 1e9
    for m_lp in (1e-9, 1e-8, 5e-8, 2e-7, 1e-6):
        xf2, yf2, L_lp2, ok_lp2 = u.fixed_angle_lp_tighten(x_f, y_f, th_f, rounds=5, margin=m_lp)
        if ok_lp2 and L_lp2 <= L_slp + 5e-6:
            cand_ver = u.verify_official(n, xf2, yf2, th_f)
            if cand_ver.get("ok"):
                cL = float(cand_ver["L_float"])
                if cL < best_ver_L:
                    best_ver_L = cL
                    x_f, y_f, L_slp, ok_slp, ver = xf2, yf2, L_lp2, True, cand_ver
                if m_lp <= 5e-8:
                    break
            elif not ver.get("ok"):
                ver = cand_ver

    pen_fin, L_fin = cpp_eval_sat_pen_bbox(x_f, y_f, th_f, side=1.0)
    if not ver.get("ok") and pen_fin < 1e-6:
        ver = u.verify_official(n, x_f, y_f, th_f)
    if not ver.get("ok") and pen_fin < 5e-6:
        for eps in (2e-9, 1e-8, 5e-8, 2e-7, 1e-6, 5e-6):
            sc = 1.0 + eps
            cand_ver = u.verify_official(n, x_f * sc, y_f * sc, th_f)
            if cand_ver.get("ok"):
                x_f, y_f = x_f * sc, y_f * sc
                pen_fin, L_fin = cpp_eval_sat_pen_bbox(x_f, y_f, th_f, side=1.0)
                ver = cand_ver
                break

    return {
        "idx": idx, "label": label, "family": fam, "n": n,
        "cpp_E": float(cpp_E), "cpp_pen": float(cpp_pen), "cpp_L": float(cpp_L),
        "ok_slp": bool(ok_slp), "pen_fin": float(pen_fin), "L_fin": float(L_fin),
        "verify": {k: v for k, v in ver.items() if k != "rows"},
        "rows": ver.get("rows"),
        "dt": round(time.time() - t0, 2),
    }


def solve_fast_melt(n: int, mats: dict, inc_front: dict, n_threads: int) -> dict:
    assert n in i1.OWNED_TARGETS
    t0 = time.time()
    L_inc = inc_front[n]["L_ours"]
    L_front = inc_front[n]["L_front"]
    rng = np.random.default_rng(30400 + n)

    peels = i1.collect_surplus_peels(n, L_inc, mats)[:2]
    x_inc, y_inc, th_inc, _ = mats[n]
    print(f"\n=== Idea 3 Melt n={n} | L_inc={L_inc:.9f} L_front={L_front:.9f} | peels={len(peels)} ===", flush=True)

    # Batch A: Deep global melt a_start=0.78
    L0_78 = L_inc + 0.04
    sched_78 = [L0_78, L_inc + 0.012, L_inc - 2e-4, L_inc - 0.002, max(L_front + 2e-4, L_inc - 0.006)]
    iters_78 = [5, 8, 11, 13, 15]
    q_78, lbl_78, ls_78 = [], [], []
    for p in peels:
        L0_p = max(L0_78, min(L_inc + 0.14, p["L"] + min(0.11, p["max_oh_rem"]) + 0.02))
        ls_p = [L0_p, L_inc + 0.015, L_inc - 2e-4, L_inc - 0.002, sched_78[-1]]
        for i, q in enumerate(i1._surplus_drop_seeds(p, n, L0_p, 3, rng)):
            q_78.append(q); lbl_78.append(f"{p['src']}_s{i}"); ls_78.append(ls_p)
    for i, q in enumerate(i1._incumbent_bottleneck_seeds(x_inc, y_inc, th_inc, L_inc, L0_78, 24, rng)):
        q_78.append(q); lbl_78.append(f"inc78_{i}"); ls_78.append(sched_78)
    qb1, qf1, lf1, eo1, po1, lb1 = run_cpp_batch_ils(
        n, 1, q_78, ls_78, iters_78, a_start=0.78, n_melt_stages=4, n_threads=n_threads, base_seed=44000 + n
    )

    # Batch B: Gentle melt a_start=0.92 (relaxes local seams while keeping global L-corridor)
    L0_92 = L_inc + 0.015
    sched_92 = [L0_92, L_inc + 0.002, L_inc - 1.5e-4, L_inc - 0.0015]
    iters_92 = [6, 9, 12, 15]
    q_92, lbl_92, ls_92 = [], [], []
    for i, q in enumerate(i2._local_pocket_seeds(x_inc, y_inc, th_inc, L_inc, L0_92, 24, rng)):
        q_92.append(q); lbl_92.append(f"inc92_{i}"); ls_92.append(sched_92)
    qb2, qf2, lf2, eo2, po2, lb2 = run_cpp_batch_ils(
        n, 1, q_92, ls_92, iters_92, a_start=0.92, n_melt_stages=3, n_threads=n_threads, base_seed=55000 + n
    )

    print(
        f"  [78] feas={int(np.sum(lf1 < 1e9))} min_pen={float(np.min(po1)):.2e} | [92] feas={int(np.sum(lf2 < 1e9))} min_pen={float(np.min(po2)):.2e}",
        flush=True,
    )

    lbl_all = lbl_78 + lbl_92
    qb_all = np.concatenate([qb1, qb2], axis=0)
    qf_all = np.concatenate([qf1, qf2], axis=0)
    lf_all = np.concatenate([lf1, lf2], axis=0)
    eo_all = np.concatenate([eo1, eo2], axis=0)
    po_all = np.concatenate([po1, po2], axis=0)
    lb_all = np.concatenate([lb1, lb2], axis=0)
    ls_all = ls_78 + ls_92

    score = lambda i: (0 if lf_all[i] < 1e9 else 1, float(lf_all[i]) if lf_all[i] < 1e9 else float(po_all[i]))
    order = sorted(range(len(lbl_all)), key=score)
    keep, used = [], set()
    for pref, qta in (("inc92_", 3), ("inc78_", 2), ("s", 1), ("x", 1)):
        c = 0
        for i in order:
            if i not in used and lbl_all[i].startswith(pref):
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
        use_f = lf_all[i] < 1e9
        L_ref = float(lf_all[i]) if use_f else ls_all[i][-1]
        x, y, th = cpp_expand_sym(n, 1, L_ref, qf_all[i] if use_f else qb_all[i])
        fam = "inc92" if lbl_all[i].startswith("inc92_") else ("inc78" if lbl_all[i].startswith("inc78_") else "surplus")
        tasks.append((i, lbl_all[i], fam, n, L_inc, x, y, th, float(eo_all[i]), 0.0 if use_f else float(po_all[i]),
                      float(lf_all[i]) if use_f else float(lb_all[i])))

    with mp.Pool(processes=min(4, max(1, n_threads), len(tasks))) as pool:
        polished = pool.map(_fast_polish_worker, tasks)
    polished.sort(key=lambda r: (0 if r["verify"].get("ok") else 1, float(r["verify"].get("L_float", 1e9)) if r["verify"].get("ok") else r["pen_fin"]))

    best_valid, best_win = None, None
    for r in polished:
        if r["verify"].get("ok"):
            Lf = float(r["verify"]["L_float"])
            if best_valid is None or Lf < float(best_valid["verify"]["L_float"]):
                best_valid = r
            if Lf < L_inc - 1e-11 and (best_win is None or Lf < float(best_win["verify"]["L_float"])):
                best_win = r

    if best_valid is not None:
        v = best_valid["verify"]
        print(f"  BEST VALID n={n}: {best_valid['label']} L_lit={v['L_lit'][:16]} (inc={L_inc:.9f}, dL={float(v['L_float']) - L_inc:+.3e}, win={best_win is not None}) [{time.time()-t0:.1f}s]", flush=True)
    else:
        print(f"  NO VALID POLISHED for n={n} [{time.time()-t0:.1f}s]", flush=True)

    return {
        "n": n, "L_inc": L_inc, "L_front": L_front, "dt_s": round(time.time() - t0, 2),
        "n_starts": len(lbl_all), "n_feas": int(np.sum(lf_all < 1e9)),
        "top_polished": [{k: v for k, v in r.items() if k != "rows"} for r in polished],
        "win": best_win,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["direct", "melt", "both"], default="direct")
    ap.add_argument("--targets", type=int, nargs="+", default=sorted(i1.OWNED_TARGETS))
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    n_threads = min(4, max(1, args.threads))

    for t in args.targets:
        if t not in i1.OWNED_TARGETS:
            raise ValueError(f"Refusing non-owned target n={t}")

    df = pd.read_csv(i1.MERGED_CSV)
    mats = {int(n): i1._norm(sub) for n, sub in df.groupby("n")}
    inc_front = u.load_incumbent_and_frontier(args.targets)
    wins = i2.load_verified_disk_wins(DATA_DIR)

    if args.mode in ("direct", "both"):
        t0 = time.time()
        tasks = [(n, mats[n][0], mats[n][1], mats[n][2], inc_front[n]["L_ours"], inc_front[n]["L_front"]) for n in args.targets]
        d_res = []
        with mp.Pool(processes=min(4, n_threads, len(tasks))) as pool:
            for r in pool.imap_unordered(_direct_worker, tasks):
                rows = r.pop("rows", None)
                d_res.append(r)
                print(f"[Direct n={r['n']}] tag={r['best_tag']} dL={r['dL']} win={r['win']} ({r['dt']}s)", flush=True)
                if r["win"] and rows is not None:
                    nw = r["n"]
                    w_obj = {"n": nw, "label": r["best_tag"], "rows": rows, "verify": r["verify"]}
                    prev = wins.get(nw)
                    if prev is None or float(r["verify"]["L_float"]) < float(prev["verify"]["L_float"]):
                        wins[nw] = w_obj
                        DATA_DIR.mkdir(parents=True, exist_ok=True)
                        u.write_final_csv_from_wins({nw: w_obj}, DATA_DIR / f"win_n{nw}.csv")
        d_res.sort(key=lambda r: r["n"])
        OUT_JSON_DIRECT.parent.mkdir(parents=True, exist_ok=True)
        OUT_JSON_DIRECT.write_text(json.dumps({"results": d_res, "dt_s": round(time.time() - t0, 2)}, indent=2))
        print(f"Direct audit done in {time.time()-t0:.1f}s | wins={sorted(wins.keys())}", flush=True)

    if args.mode in ("melt", "both"):
        t0 = time.time()
        m_res = []
        for n in args.targets:
            res = solve_fast_melt(n, mats, inc_front, n_threads=n_threads)
            w = res.pop("win")
            m_res.append(res)
            if w is not None:
                prev = wins.get(n)
                if prev is None or float(w["verify"]["L_float"]) < float(prev["verify"]["L_float"]):
                    wins[n] = w
                    DATA_DIR.mkdir(parents=True, exist_ok=True)
                    u.write_final_csv_from_wins({n: w}, DATA_DIR / f"win_n{n}.csv")
        OUT_JSON_MELT.parent.mkdir(parents=True, exist_ok=True)
        OUT_JSON_MELT.write_text(json.dumps({"results": m_res, "dt_s": round(time.time() - t0, 2)}, indent=2))
        print(f"Melt sweep done in {time.time()-t0:.1f}s | wins={sorted(wins.keys())}", flush=True)

    if wins:
        u.write_final_csv_from_wins(wins, FINAL_CSV)
        print(f"Updated {FINAL_CSV} with wins={sorted(wins.keys())}", flush=True)


if __name__ == "__main__":
    main()
