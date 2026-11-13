#!/usr/bin/env python3
"""Idea 2: Fast Early-Stop Joint SLP + Dual-Regime (a=0.78 Global & a=1.00 Local) Pocket Melts.

Combines:
  A. Deep global melt (a_start=0.78, L0=L_inc+0.04, 7b385ca schedule) across 6 incumbent bottleneck
     modes + top surplus peels + (n-1)+1 void insertions.
  B. Local-only pocket melt (a_start=1.00, L_sched=[L_inc+0.003, L_inc-1.5e-4, L_inc-1e-3, L_inc-3.5e-3])
     preserving the outer upright+tilted lattice while unlocking local seams/voids and (n+1)->n deletions.
  C. Fast early-stopping joint_xytheta_slp (rounds=10, upgraded to 16 when within +0.002 of L_inc)
     + multi-margin fixed_angle_lp_tighten + official interval certification.
"""
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

HERE = Path(__file__).resolve().parent
OUT_JSON = HERE.parent / "docs" / "idea2_results.json"
FINAL_CSV = HERE.parent / "final.csv"
DATA_DIR = HERE.parent / "data"


def _local_pocket_seeds(
    x_inc: np.ndarray,
    y_inc: np.ndarray,
    th_inc: np.ndarray,
    L_inc: float,
    L0: float,
    n_seeds: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Local-only perturbations scaled cleanly to L0 without shattering the global crystal."""
    n = len(x_inc)
    sc = L0 / L_inc
    xb, yb, tb = x_inc * sc, y_inc * sc, th_inc.copy()
    hx = 0.5 * (np.abs(np.cos(tb)) + np.abs(np.sin(tb)))
    margin = np.minimum.reduce([xb - hx, L0 - xb - hx, yb - hx, L0 - yb - hx])
    tight_idx = np.argsort(margin)[:18]
    voids = i1.clearance_points(xb, yb, tb, L0, n_pts=6, step=0.14)
    tilt = i1.donor_tilt(tb)

    deg = np.degrees(tb) % 90.0
    deg_m = np.minimum(deg, 90.0 - deg)
    up_mask = deg_m < 1.0
    tilted_idx = np.where(~up_mask)[0]

    mL = xb - hx - float((xb - hx).min())
    mR = float((xb + hx).max()) - (xb + hx)
    mB = yb - hx - float((yb - hx).min())
    mT = float((yb + hx).max()) - (yb + hx)
    cL, cR = int(np.sum(mL < 5e-4)), int(np.sum(mR < 5e-4))
    cB, cT = int(np.sum(mB < 5e-4)), int(np.sum(mT < 5e-4))
    dx_sign = 1.0 if cR <= cL else -1.0
    dy_sign = 1.0 if cT <= cB else -1.0

    seeds = []
    for s in range(n_seeds):
        x, y, th = xb.copy(), yb.copy(), tb.copy()
        mode = s % 4
        if mode == 0 and len(voids) > 0:
            pi = int(tight_idx[s % len(tight_idx)])
            vx, vy, _ = voids[(s // 4) % len(voids)]
            x[pi] = vx + float(rng.uniform(-0.05, 0.05))
            y[pi] = vy + float(rng.uniform(-0.05, 0.05))
            th[pi] = [0.0, np.pi / 4.0, tilt][(s // 4) % 3] + float(rng.uniform(-0.06, 0.06))
            near = ((x - vx) ** 2 + (y - vy) ** 2) <= 3.8 ** 2
            nm = int(near.sum())
            x[near] += rng.uniform(-0.08, 0.08, nm)
            y[near] += rng.uniform(-0.08, 0.08, nm)
            th[near] += rng.uniform(-0.07, 0.07, nm)
        elif mode == 1:
            pi = int(tight_idx[(s // 4) % len(tight_idx)])
            cx, cy = float(x[pi]), float(y[pi])
            near = ((x - cx) ** 2 + (y - cy) ** 2) <= float(rng.uniform(3.8, 5.8)) ** 2
            nm = int(near.sum())
            x[near] += rng.uniform(-0.09, 0.09, nm)
            y[near] += rng.uniform(-0.09, 0.09, nm)
            th[near] += rng.uniform(-0.10, 0.10, nm)
        elif mode == 2 and len(tilted_idx) > 0:
            pi = int(tilted_idx[(s * 7) % len(tilted_idx)])
            cx, cy = float(x[pi]), float(y[pi])
            near = ((x - cx) ** 2 + (y - cy) ** 2) <= float(rng.uniform(4.2, 6.2)) ** 2
            nm = int(near.sum())
            x[near] += rng.uniform(-0.08, 0.08, nm)
            y[near] += rng.uniform(-0.08, 0.08, nm)
            th[near] += rng.uniform(-0.11, 0.11, nm)
        else:
            drift = float(rng.uniform(0.015, 0.05))
            half_x = (x < L0 * 0.6) if dx_sign > 0 else (x > L0 * 0.4)
            half_y = (y < L0 * 0.6) if dy_sign > 0 else (y > L0 * 0.4)
            x[half_x] += dx_sign * drift
            y[half_y] += dy_sign * 0.6 * drift
            cx, cy = float(x[tight_idx[s % 8]]), float(y[tight_idx[s % 8]])
            near = ((x - cx) ** 2 + (y - cy) ** 2) <= 5.0 ** 2
            nm = int(near.sum())
            x[near] += rng.uniform(-0.08, 0.08, nm)
            y[near] += rng.uniform(-0.08, 0.08, nm)
            th[near] += rng.uniform(-0.09, 0.09, nm)
        seeds.append(full_to_free_sym(x, y, th, L0))
    return seeds


def _polish_worker(args: tuple) -> dict:
    """Polish one C++ basin with fast early-stop joint SLP + upgraded precision pass if promising."""
    idx, label, fam, n, L_inc, x, y, th, cpp_E, cpp_pen, cpp_L = args
    t0 = time.time()

    hx = 0.5 * (np.abs(np.cos(th)) + np.abs(np.sin(th)))
    x = x - float((x - hx).min())
    y = y - float((y - hx).min())

    if cpp_pen > 1e-7 and cpp_pen < 0.04:
        L_try = cpp_L + max(0.002, 1.8 * cpp_pen)
        xr, yr, thr, _ = u.relax_lbfgs(x, y, th, L_try, maxiter=220, w_wall=4.0, w_pair=1.0, side=1.0)
        pen_r, L_r = cpp_eval_sat_pen_bbox(xr, yr, thr, side=1.0)
        if pen_r <= cpp_pen + 1e-6:
            x, y, th, cpp_pen, cpp_L = xr, yr, thr, pen_r, L_r

    pen_pre, L_pre = cpp_eval_sat_pen_bbox(x, y, th, side=1.0)
    ok_slp, L_slp = False, L_pre
    x_f, y_f, th_f = x.copy(), y.copy(), th.copy()

    if pen_pre < 0.03:
        xs, ys, ts, Ls, ok_s = joint_xytheta_slp(
            x, y, th, rounds=10, tr_th0=0.025, tr_xy0=0.12, margin=1e-8, tol=1e-8, patience=2
        )
        if ok_s:
            x_f, y_f, th_f, L_slp, ok_slp = xs, ys, ts, Ls, True
            if L_slp < L_inc + 0.002:
                xs2, ys2, ts2, Ls2, ok_s2 = joint_xytheta_slp(
                    x_f, y_f, th_f, rounds=14, tr_th0=0.012, tr_xy0=0.05, margin=1e-8, tol=1e-10, patience=3
                )
                if ok_s2 and Ls2 < L_slp:
                    x_f, y_f, th_f, L_slp = xs2, ys2, ts2, Ls2

    ver: dict = {"ok": False}
    best_ver_L = 1e9
    for m_lp in (1e-9, 1e-8, 5e-8, 2e-7, 1e-6):
        xf2, yf2, L_lp2, ok_lp2 = u.fixed_angle_lp_tighten(x_f, y_f, th_f, rounds=6, margin=m_lp)
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

    ang_deg = np.degrees(th_f) % 90.0
    n45 = int(np.sum(np.abs(ang_deg - 45.0) < 0.2))
    n0 = int(np.sum((ang_deg < 0.2) | (ang_deg > 89.8)))
    return {
        "idx": idx,
        "label": label,
        "family": fam,
        "n": n,
        "cpp_E": float(cpp_E),
        "cpp_pen": float(cpp_pen),
        "cpp_L": float(cpp_L),
        "ok_slp": bool(ok_slp),
        "pen_fin": float(pen_fin),
        "L_fin": float(L_fin),
        "n0": n0,
        "n45": n45,
        "non45_non0": int(n - n0 - n45),
        "verify": {k: v for k, v in ver.items() if k != "rows"},
        "rows": ver.get("rows"),
        "dt": round(time.time() - t0, 2),
    }


def _select_diverse(labels: list[str], lf: np.ndarray, po: np.ndarray, k_keep: int) -> list[int]:
    score = lambda i: (0 if lf[i] < 1e9 else 1, float(lf[i]) if lf[i] < 1e9 else float(po[i]))
    order = sorted(range(len(labels)), key=score)
    chosen: list[int] = []
    used = set()
    for prefix, quota in (("inc78_", 4), ("incloc_", 4), ("s", 2), ("x", 1), ("del", 1), ("m", 1)):
        cnt = 0
        for i in order:
            if i not in used and labels[i].startswith(prefix):
                chosen.append(i)
                used.add(i)
                cnt += 1
                if cnt >= quota or len(chosen) >= k_keep:
                    break
    for i in order:
        if len(chosen) >= k_keep:
            break
        if i not in used:
            chosen.append(i)
            used.add(i)
    return chosen


def solve_target(n: int, mats: dict, inc_front: dict, n_threads: int, quick: bool, max_donors: int) -> dict:
    assert n in i1.OWNED_TARGETS, f"Refusing non-owned target n={n}"
    t0 = time.time()
    L_inc = inc_front[n]["L_ours"]
    L_front = inc_front[n]["L_front"]
    rng = np.random.default_rng(20400 + n)

    peels = i1.collect_surplus_peels(n, L_inc, mats)[:max_donors]
    print(f"\n=== Idea 2 n={n} | L_inc={L_inc:.9f} L_front={L_front:.9f} (gap={L_inc-L_front:+.6f}) | peels={len(peels)} ===", flush=True)
    for p in peels:
        print(f"   peel {p['src']}: surp=+{p['surplus']} L={p['L']:.6f} oh_rem={p['max_oh_rem']:.4f}", flush=True)

    x_inc, y_inc, th_inc, _ = mats[n]

    # Batch 1: Deep global melt (a_start=0.78, 7b385ca recipe)
    L0_78 = L_inc + 0.04
    sched_78 = [L0_78, L_inc + 0.012, L_inc - 2e-4, L_inc - 0.002, max(L_front + 2e-4, L_inc - 0.006)]
    iters_78 = [5, 8, 11, 13, 15] if quick else [7, 10, 14, 17, 20]

    q_78, lbl_78, ls_78 = [], [], []
    n_peel_seeds = 3 if quick else 5
    for p in peels:
        L0_p = max(L0_78, min(L_inc + 0.14, p["L"] + min(0.11, p["max_oh_rem"]) + 0.02))
        ls_p = [L0_p, L_inc + 0.015, L_inc - 2e-4, L_inc - 0.002, sched_78[-1]]
        for i, q in enumerate(i1._surplus_drop_seeds(p, n, L0_p, n_peel_seeds, rng)):
            q_78.append(q)
            lbl_78.append(f"{p['src']}_s{i}")
            ls_78.append(ls_p)

    n_inc78 = 28 if quick else 40
    for i, q in enumerate(i1._incumbent_bottleneck_seeds(x_inc, y_inc, th_inc, L_inc, L0_78, n_inc78, rng)):
        q_78.append(q)
        lbl_78.append(f"inc78_{i}")
        ls_78.append(sched_78)

    if (n - 1) in mats and mats[n - 1][3] < L_inc:
        x1, y1, th1, L1 = mats[n - 1]
        sc1 = (L0_78 - 1.0) / max(1e-9, L1 - 1.0)
        xb1, yb1 = 0.5 + (x1 - 0.5) * sc1, 0.5 + (y1 - 0.5) * sc1
        pts = i1.clearance_points(xb1, yb1, th1, L0_78, n_pts=4, step=0.14)
        tilt1 = i1.donor_tilt(th1)
        for s in range(4):
            px, py, _ = pts[s % len(pts)]
            xs = np.append(xb1, px + float(rng.normal(0, 0.05)))
            ys = np.append(yb1, py + float(rng.normal(0, 0.05)))
            ts = np.append(th1, [0.0, np.pi / 4.0, tilt1][s % 3] + float(rng.normal(0, 0.04)))
            q_78.append(full_to_free_sym(xs, ys, ts, L0_78))
            lbl_78.append(f"m{n-1}_{s}")
            ls_78.append(sched_78)

    qb1, qf1, lf1, eo1, po1, lb1 = run_cpp_batch_ils(
        n, 1, q_78, ls_78, iters_78, a_start=0.78, n_melt_stages=4, n_threads=n_threads, base_seed=22000 + n
    )
    print(
        f"  [Batch78] N={len(lbl_78)} feas={int(np.sum(lf1 < 1e9))} best_Lf={float(np.min(lf1)) if np.any(lf1 < 1e9) else None} min_pen={float(np.min(po1)):.2e}",
        flush=True,
    )

    # Batch 2: Local-only pocket melt (a_start=1.00, no global shrink) + (n+1)->n deletion
    L0_loc = L_inc + 0.003
    sched_loc = [L0_loc, L_inc - 1.5e-4, L_inc - 0.001, max(L_front + 2e-4, L_inc - 0.0035)]
    iters_loc = [6, 10, 13, 16] if quick else [8, 12, 16, 20]
    q_loc, lbl_loc, ls_loc = [], [], []
    n_incloc = 24 if quick else 32
    for i, q in enumerate(_local_pocket_seeds(x_inc, y_inc, th_inc, L_inc, L0_loc, n_incloc, rng)):
        q_loc.append(q)
        lbl_loc.append(f"incloc_{i}")
        ls_loc.append(sched_loc)

    for lbl_d, q_d in i1._del_n1_n2_seeds(mats, n, L_inc, L0_loc, 6 if quick else 10, rng):
        if lbl_d.startswith(f"del{n+1}_"):
            q_loc.append(q_d)
            lbl_loc.append(lbl_d)
            ls_loc.append(sched_loc)

    qb2, qf2, lf2, eo2, po2, lb2 = run_cpp_batch_ils(
        n, 1, q_loc, ls_loc, iters_loc, a_start=1.0, n_melt_stages=0, n_threads=n_threads, base_seed=33000 + n
    )
    print(
        f"  [BatchLoc] N={len(lbl_loc)} feas={int(np.sum(lf2 < 1e9))} best_Lf={float(np.min(lf2)) if np.any(lf2 < 1e9) else None} min_pen={float(np.min(po2)):.2e}",
        flush=True,
    )

    # Combine candidates and polish top diverse subset
    lbl_all = lbl_78 + lbl_loc
    qb_all = np.concatenate([qb1, qb2], axis=0)
    qf_all = np.concatenate([qf1, qf2], axis=0)
    lf_all = np.concatenate([lf1, lf2], axis=0)
    eo_all = np.concatenate([eo1, eo2], axis=0)
    po_all = np.concatenate([po1, po2], axis=0)
    lb_all = np.concatenate([lb1, lb2], axis=0)
    ls_all = ls_78 + ls_loc

    keep = _select_diverse(lbl_all, lf_all, po_all, 12 if quick else 16)
    tasks = []
    for i in keep:
        use_f = lf_all[i] < 1e9
        L_ref = float(lf_all[i]) if use_f else ls_all[i][-1]
        x, y, th = cpp_expand_sym(n, 1, L_ref, qf_all[i] if use_f else qb_all[i])
        fam = (
            "inc78" if lbl_all[i].startswith("inc78_")
            else "incloc" if lbl_all[i].startswith("incloc_")
            else "del" if lbl_all[i].startswith("del")
            else "m1" if lbl_all[i].startswith("m")
            else "surplus_peel"
        )
        tasks.append((i, lbl_all[i], fam, n, L_inc, x, y, th, float(eo_all[i]), 0.0 if use_f else float(po_all[i]),
                      float(lf_all[i]) if use_f else float(lb_all[i])))

    with mp.Pool(processes=min(4, max(1, n_threads), len(tasks))) as pool:
        polished = pool.map(_polish_worker, tasks)
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
        print(f"  BEST VALID n={n}: {best_valid['label']} L_lit={v['L_lit'][:16]} (inc={L_inc:.9f}, dL={float(v['L_float']) - L_inc:+.3e}, win={best_win is not None})", flush=True)
    else:
        print(f"  NO VALID POLISHED for n={n} (min_pen_fin={min(r['pen_fin'] for r in polished):.2e})", flush=True)

    return {
        "n": n,
        "L_inc": L_inc,
        "L_front": L_front,
        "dt_s": round(time.time() - t0, 2),
        "peels": [{"src": p["src"], "surplus": p["surplus"], "L": p["L"], "max_oh_rem": p["max_oh_rem"]} for p in peels],
        "n_starts": len(lbl_all),
        "n_feas": int(np.sum(lf_all < 1e9)),
        "best_cpp_Lf": float(np.min(lf_all)) if np.any(lf_all < 1e9) else None,
        "top_polished": [{k: v for k, v in r.items() if k != "rows"} for r in polished[:10]],
        "win": best_win,
    }


def load_verified_disk_wins(data_dir: Path) -> dict[int, dict]:
    paths = sorted(data_dir.glob("win_n*.csv")) if data_dir.exists() else []
    if not paths:
        return {}
    ns = [int(p.stem.replace("win_n", "")) for p in paths]
    inc_map = u.load_incumbent_and_frontier(ns)
    out: dict[int, dict] = {}
    for p in paths:
        nw = int(p.stem.replace("win_n", ""))
        sub = pd.read_csv(p)
        sub["sid"] = sub["id"].astype(str).str.split("_").str[1].astype(int)
        sub = sub.sort_values("sid")
        x = sub["x"].to_numpy(np.float64)
        y = sub["y"].to_numpy(np.float64)
        th = np.radians(sub["deg"].to_numpy(np.float64))
        ver = u.verify_official(nw, x, y, th)
        if ver.get("ok") and float(ver["L_float"]) < inc_map[nw]["L_ours"] - 1e-11:
            out[nw] = {
                "n": nw,
                "label": p.stem,
                "rows": ver["rows"],
                "verify": {k: v for k, v in ver.items() if k != "rows"},
            }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", type=int, nargs="+", default=[698, 860, 339, 921])
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--max-donors", type=int, default=3)
    args = ap.parse_args()
    n_threads = min(4, max(1, args.threads))

    for t in args.targets:
        if t not in i1.OWNED_TARGETS:
            raise ValueError(f"Target n={t} is outside OWNED_TARGETS={sorted(i1.OWNED_TARGETS)}")

    t_all = time.time()
    df = pd.read_csv(i1.MERGED_CSV)
    mats = {int(n): i1._norm(sub) for n, sub in df.groupby("n")}
    inc_front = u.load_incumbent_and_frontier(args.targets)

    results = []
    wins = load_verified_disk_wins(DATA_DIR)
    for n in args.targets:
        res = solve_target(n, mats, inc_front, n_threads=n_threads, quick=args.quick, max_donors=args.max_donors)
        w = res.pop("win")
        results.append(res)
        if w is not None:
            prev_w = wins.get(n)
            if prev_w is None or float(w["verify"]["L_float"]) < float(prev_w["verify"]["L_float"]):
                wins[n] = w
                DATA_DIR.mkdir(parents=True, exist_ok=True)
                u.write_final_csv_from_wins({n: w}, DATA_DIR / f"win_n{n}.csv")
                print(f"  [SAVED WIN] {DATA_DIR / f'win_n{n}.csv'} L_pd={w['verify']['L_pandas'][:16]}", flush=True)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({"results": results, "dt_total_s": round(time.time() - t_all, 2)}, indent=2))
    print(f"\nSaved {OUT_JSON} in {time.time() - t_all:.1f}s | wins={sorted(wins.keys())}", flush=True)

    if wins:
        u.write_final_csv_from_wins(wins, FINAL_CSV)
        print(f"FINAL_CSV written to {FINAL_CSV} for wins={sorted(wins.keys())}", flush=True)


if __name__ == "__main__":
    main()
