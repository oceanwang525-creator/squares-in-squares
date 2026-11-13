#!/usr/bin/env python3
"""Idea 21: Fine Interior Cross-Cuts (a=0.94->1.00) + Topological Pocket Ruin-and-Recreate.

Combines:
1. Fast vectorized 0.25-step interior cross-cut search across all donors (c in 1, 2, 3),
   selecting exact/low-surplus cuts with minimal tilted-pocket disruption and slack-opened seams.
2. Gentle-breathing C++ ILS (a_start=0.94->1.00) on cross-cut seeds so perimeter tilted
   pockets remain crystal-locked while the thin cross-seam relaxes.
3. Topological incumbent ruin-and-recreate (R=5.5..8.0 pocket, +/-0.35 position shake,
   angle scramble, and weak-wall drift) at a_start=0.88->1.00 to escape incumbent basins.
4. Two-stage HiGHS joint_xytheta_slp (threads=1) + multi-margin LP tighten + exact verify.
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pandas as pd

import utils as u
from idea17_run_slp import joint_xytheta_slp
from idea18_run import cpp_eval_sat_pen_bbox, write_final_csv_from_wins
from idea19_run import MERGED_CSV, cpp_expand_sym, full_to_free_sym, run_cpp_batch_ils
from idea20_partner_shift import (
    DATA_DIR,
    FINAL_CSV,
    _norm,
    clearance_points,
    donor_tilt,
    load_verified_disk_wins,
)

HERE = Path(__file__).resolve().parent
OUT_JSON_21 = HERE.parent / "docs" / "idea21_results.json"


def find_fine_cross_cuts(
    n: int, L_inc: float, mats: dict[int, tuple], step: float = 0.25, max_cuts: int = 8,
) -> list[dict]:
    """Vectorized 0.25-step interior cross-cut search over c in (1, 2, 3) donors."""
    k = math.isqrt(n)
    pre_cands: list[tuple] = []

    for n2 in range(n + 1, min(1000, n + 5 * k + 35) + 1):
        if n2 not in mats:
            continue
        x, y, th, L = mats[n2]
        deg = np.degrees(th) % 90.0
        tilt = (deg > 2.0) & (deg < 88.0)
        n_tilt = int(tilt.sum())
        for c in (1, 2, 3):
            L_box = L - c
            if not (L_inc - 0.10 <= L_box <= L_inc - 0.0005):
                continue
            x_cuts = np.arange(1.25, L - 1.25 - c + 1e-9, step)
            y_cuts = np.arange(1.25, L - 1.25 - c + 1e-9, step)
            if len(x_cuts) == 0 or len(y_cuts) == 0:
                continue
            KX = ((x[None, :] < x_cuts[:, None]) | (x[None, :] > x_cuts[:, None] + c)).astype(np.float32)
            KY = ((y[None, :] < y_cuts[:, None]) | (y[None, :] > y_cuts[:, None] + c)).astype(np.float32)
            counts = np.rint(KX @ KY.T).astype(np.int32)
            valid = np.argwhere((counts >= n) & (counts <= n + 4))
            if len(valid) == 0:
                continue
            if n_tilt > 0:
                t_kept = np.rint(KX[:, tilt] @ KY[:, tilt].T).astype(np.int32)
                t_cut = n_tilt - t_kept[valid[:, 0], valid[:, 1]]
            else:
                t_cut = np.zeros(len(valid), dtype=np.int32)
            surp = counts[valid[:, 0], valid[:, 1]] - n
            # Rank locally before exact SAT check
            order = np.lexsort((t_cut, surp))[:6]
            for idx in order:
                ix, iy = int(valid[idx, 0]), int(valid[idx, 1])
                pre_cands.append((int(surp[idx]), int(t_cut[idx]), L_box, n2, c, float(x_cuts[ix]), float(y_cuts[iy])))

    pre_cands.sort(key=lambda r: (r[0], r[1], r[2]))
    out: list[dict] = []
    seen = set()
    for surp, tc, L_box, n2, c, xc, yc in pre_cands[:36]:
        x, y, th, _ = mats[n2]
        keep = ((x < xc) | (x > xc + c)) & ((y < yc) | (y > yc + c))
        xs = np.where(x[keep] > xc + c, x[keep] - c, x[keep])
        ys = np.where(y[keep] > yc + c, y[keep] - c, y[keep])
        ts = th[keep].copy()
        if surp > 0:
            # Drop surplus squares closest to cross-seam intersection
            d_seam = np.minimum(np.abs(xs - xc), np.abs(ys - yc))
            drop = np.argsort(d_seam)[:surp]
            mask = np.ones(len(xs), dtype=bool)
            mask[drop] = False
            xs, ys, ts = xs[mask], ys[mask], ts[mask]
        pen, _ = u.sat_max_overlap_and_bbox(xs, ys, ts)
        key = (n2, c, round(pen, 3), round(L_box, 5))
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "src": f"xc{n2}_c{c}_{int(round(xc*10))}_{int(round(yc*10))}",
            "n2": n2, "c": c, "xc": xc, "yc": yc, "L": L_box, "surplus": surp,
            "tilt_cut": tc, "pen0": float(pen), "x": xs, "y": ys, "th": ts,
        })
    out.sort(key=lambda r: (r["pen0"], r["L"]))
    return out[:max_cuts]


def _cross_cut_seeds(cut: dict, L_inc: float, n_seeds: int, rng: np.random.Generator) -> tuple[list[np.ndarray], list[float]]:
    """Open quadrant slack to L0 and lightly relax cross-seam before gentle C++ breathing."""
    xs0, ys0, ts0 = cut["x"], cut["y"], cut["th"]
    xc, yc, L_box = cut["xc"], cut["yc"], cut["L"]
    L0 = min(L_inc + 0.012, max(L_inc - 0.0005, L_box + min(0.06, 0.45 * cut["pen0"])))
    slack = max(0.0, L0 - L_box)
    L_sched = [L0, L_inc + 0.002, L_inc - 1e-4, L_inc - 0.002, max(L_box, L_inc - 0.006)]

    seeds = []
    for s in range(n_seeds):
        xs = np.where(xs0 > xc, xs0 + slack, xs0)
        ys = np.where(ys0 > yc, ys0 + slack, ys0)
        ts = ts0.copy()
        seam = (np.abs(xs - xc) < 1.6) | (np.abs(ys - yc) < 1.6)
        if s > 0:
            ns = int(seam.sum())
            xs[seam] += rng.uniform(-0.10, 0.10, ns)
            ys[seam] += rng.uniform(-0.10, 0.10, ns)
            ts[seam] += rng.uniform(-0.10, 0.10, ns)
        xs, ys, ts, _ = u.relax_lbfgs(xs, ys, ts, L0, maxiter=140, w_wall=4.0, w_pair=1.0)
        seeds.append(full_to_free_sym(xs, ys, ts, L0))
    return seeds, L_sched


def _ruin_recreate_seeds(
    x_inc: np.ndarray, y_inc: np.ndarray, th_inc: np.ndarray, L_inc: float, L0: float,
    n_seeds: int, rng: np.random.Generator,
) -> list[np.ndarray]:
    """Topological pocket ruin-and-recreate (+/-0.35 shake + angle scramble) to break incumbent basin."""
    n = len(x_inc)
    sc = L0 / L_inc
    xb, yb, tb = x_inc * sc, y_inc * sc, th_inc.copy()
    hx = 0.5 * (np.abs(np.cos(tb)) + np.abs(np.sin(tb)))
    margin = np.minimum.reduce([xb - hx, L0 - xb - hx, yb - hx, L0 - yb - hx])
    tight_idx = np.argsort(margin)[:16]
    voids = clearance_points(xb, yb, tb, L0, n_pts=8, step=0.14)
    tilt = donor_tilt(tb)

    deg = np.degrees(tb) % 90.0
    tilted_idx = np.where((deg > 2.0) & (deg < 88.0))[0]

    seeds = []
    for s in range(n_seeds):
        x, y, th = xb.copy(), yb.copy(), tb.copy()
        mode = s % 3
        if mode == 0 and len(voids) > 0:
            vx, vy, _ = voids[(s // 3) % len(voids)]
            rad = float(rng.uniform(5.2, 7.8))
            near = ((x - vx) ** 2 + (y - vy) ** 2) <= rad * rad
        elif mode == 1:
            pi = int(tight_idx[(s // 3) % len(tight_idx)])
            rad = float(rng.uniform(5.5, 8.0))
            near = ((x - x[pi]) ** 2 + (y - y[pi]) ** 2) <= rad * rad
        else:
            pi = int(tilted_idx[(s * 7) % len(tilted_idx)]) if len(tilted_idx) else 0
            rad = float(rng.uniform(5.5, 8.2))
            near = ((x - x[pi]) ** 2 + (y - y[pi]) ** 2) <= rad * rad

        nm = int(near.sum())
        x[near] = np.clip(x[near] + rng.uniform(-0.32, 0.32, nm), 0.6, L0 - 0.6)
        y[near] = np.clip(y[near] + rng.uniform(-0.32, 0.32, nm), 0.6, L0 - 0.6)
        # Scramble 50% of pocket square angles across {0, tilt, pi/4, random} to jump discrete basins
        flip = rng.random(nm) < 0.50
        new_th = th[near].copy()
        choices = rng.choice([0.0, tilt, math.pi / 4], size=int(flip.sum()))
        new_th[flip] = choices + rng.uniform(-0.22, 0.22, int(flip.sum()))
        new_th[~flip] += rng.uniform(-0.18, 0.18, int((~flip).sum()))
        th[near] = new_th
        seeds.append(full_to_free_sym(x, y, th, L0))
    return seeds


def _fast_polish_repair_worker(args: tuple) -> dict:
    """Two-stage HiGHS joint_xytheta_slp + multi-margin LP tighten + exact verification."""
    idx, label, fam, n, x, y, th, cpp_E, cpp_pen, cpp_L, L_inc = args
    t0 = time.time()

    hx = 0.5 * (np.abs(np.cos(th)) + np.abs(np.sin(th)))
    x = x - float((x - hx).min())
    y = y - float((y - hx).min())

    if 1e-7 < cpp_pen < 0.05:
        L_try = cpp_L + max(0.002, 1.8 * cpp_pen)
        xr, yr, thr, _ = u.relax_lbfgs(x, y, th, L_try, maxiter=220, w_wall=4.0, w_pair=1.0, side=1.0)
        pen_r, L_r = cpp_eval_sat_pen_bbox(xr, yr, thr, side=1.0)
        if pen_r <= cpp_pen + 1e-6:
            x, y, th, cpp_pen, cpp_L = xr, yr, thr, pen_r, L_r

    pen_pre, L_pre = cpp_eval_sat_pen_bbox(x, y, th, side=1.0)
    ok_slp, L_slp = False, L_pre
    x_f, y_f, th_f = x.copy(), y.copy(), th.copy()

    if pen_pre < 0.04:
        xs, ys, ts, Ls, ok_s = joint_xytheta_slp(x, y, th, rounds=10, tr_th0=0.03, tr_xy0=0.14, margin=1e-8)
        if ok_s:
            x_f, y_f, th_f, L_slp, ok_slp = xs, ys, ts, Ls, True
            if Ls < L_inc + 0.004:
                xs2, ys2, ts2, Ls2, ok_s2 = joint_xytheta_slp(xs, ys, ts, rounds=10, tr_th0=0.01, tr_xy0=0.04, margin=1e-8)
                if ok_s2 and Ls2 < L_slp:
                    x_f, y_f, th_f, L_slp = xs2, ys2, ts2, Ls2

    ver: dict = {"ok": False}
    for m_lp in (1e-9, 5e-8, 2e-7, 1e-6):
        xf2, yf2, L_lp2, ok_lp2 = u.fixed_angle_lp_tighten(x_f, y_f, th_f, rounds=6, margin=m_lp)
        if ok_lp2 and (m_lp == 1e-9 and L_lp2 < L_slp or m_lp > 1e-9 and L_lp2 < L_slp + 5e-6):
            cand_ver = u.verify_official(n, xf2, yf2, th_f)
            if cand_ver.get("ok"):
                x_f, y_f, L_slp, ok_slp, ver = xf2, yf2, L_lp2, True, cand_ver
                break
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
        "idx": idx, "label": label, "family": fam, "n": n,
        "cpp_E": float(cpp_E), "cpp_pen": float(cpp_pen), "cpp_L": float(cpp_L),
        "ok_slp": bool(ok_slp), "pen_fin": float(pen_fin), "L_fin": float(L_fin),
        "n0": n0, "n45": n45, "non45_non0": int(n - n0 - n45),
        "verify": {k: v for k, v in ver.items() if k != "rows"},
        "rows": ver.get("rows"),
        "dt": round(time.time() - t0, 2),
    }


def solve_target(n: int, mats: dict, inc_front: dict, n_threads: int, quick: bool) -> dict:
    t0 = time.time()
    L_inc = inc_front[n]["L_ours"]
    L_front = inc_front[n]["L_front"]
    rng = np.random.default_rng(21000 + n)
    cuts = find_fine_cross_cuts(n, L_inc, mats, max_cuts=6 if quick else 10)
    print(f"\n=== Idea 21 n={n} | L_inc={L_inc:.9f} L_front={L_front:.9f} | fine_cuts={len(cuts)} ===", flush=True)
    for c in cuts:
        print(f"   cut {c['src']}: surp=+{c['surplus']} tcut={c['tilt_cut']} L={c['L']:.6f} pen0={c['pen0']:.4f}", flush=True)

    # Track A: Gentle breathing a_start=0.94 on fine interior cross-cuts
    q_a, lbl_a, ls_a = [], [], []
    for c in cuts:
        seeds_c, lsc = _cross_cut_seeds(c, L_inc, 3 if quick else 5, rng)
        for i, q in enumerate(seeds_c):
            q_a.append(q)
            lbl_a.append(f"{c['src']}_s{i}")
            ls_a.append(lsc)

    # Track B: Topological incumbent ruin-and-recreate at a_start=0.88
    x_inc, y_inc, th_inc, _ = mats[n]
    L0_r = L_inc + 0.025
    ls_r = [L0_r, L_inc + 0.005, L_inc - 1e-4, L_inc - 0.002, max(L_front + 2e-4, L_inc - 0.006)]
    q_b = _ruin_recreate_seeds(x_inc, y_inc, th_inc, L_inc, L0_r, 18 if quick else 30, rng)
    lbl_b = [f"ruin_{i}" for i in range(len(q_b))]
    ls_b = [ls_r] * len(q_b)

    ils_iters = [5, 7, 10, 12, 14] if quick else [6, 9, 13, 16, 18]
    all_cands = []
    if q_a:
        qb, qf, lf, eo, po, lb = run_cpp_batch_ils(
            n, 1, q_a, ls_a, ils_iters, a_start=0.94, n_melt_stages=4,
            n_threads=n_threads, base_seed=215000 + n,
        )
        print(f"  [TrackA cross] N={len(q_a)} feas={int(np.sum(lf < 1e9))} min_pen={float(np.min(po)):.3e}", flush=True)
        for i in range(len(q_a)):
            all_cands.append(("cross_cut", lbl_a[i], qf[i] if lf[i] < 1e9 else qb[i], float(lf[i]), float(eo[i]), float(po[i]), float(lb[i]), ls_a[i][-1]))

    qb2, qf2, lf2, eo2, po2, lb2 = run_cpp_batch_ils(
        n, 1, q_b, ls_b, ils_iters, a_start=0.88, n_melt_stages=4,
        n_threads=n_threads, base_seed=216000 + n,
    )
    print(f"  [TrackB ruin]  N={len(q_b)} feas={int(np.sum(lf2 < 1e9))} min_pen={float(np.min(po2)):.3e}", flush=True)
    for i in range(len(q_b)):
        all_cands.append(("ruin_rec", lbl_b[i], qf2[i] if lf2[i] < 1e9 else qb2[i], float(lf2[i]), float(eo2[i]), float(po2[i]), float(lb2[i]), ls_b[i][-1]))

    # Stratified pick of top candidates from both Track A and Track B (keep <= 8 in quick)
    score = lambda r: (0 if r[3] < 1e9 else 1, r[3] if r[3] < 1e9 else r[5])
    top_a = sorted([r for r in all_cands if r[0] == "cross_cut"], key=score)[: (3 if quick else 6)]
    top_b = sorted([r for r in all_cands if r[0] == "ruin_rec"], key=score)[: (5 if quick else 8)]
    chosen = top_a + top_b

    tasks = []
    for k_idx, (fam, lbl, qv, lfi, eoi, poi, lbi, lref_end) in enumerate(chosen):
        use_f = lfi < 1e9
        x, y, th = cpp_expand_sym(n, 1, lfi if use_f else lref_end, qv)
        tasks.append((k_idx, lbl, fam, n, x, y, th, eoi, 0.0 if use_f else poi, lfi if use_f else lbi, L_inc))

    with mp.Pool(processes=min(4, max(1, n_threads), len(tasks))) as pool:
        polished = pool.map(_fast_polish_repair_worker, tasks)
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
        print(f"  NO VALID POLISHED for n={n}", flush=True)
    return {
        "n": n, "L_inc": L_inc, "L_front": L_front, "dt_s": round(time.time() - t0, 2),
        "cuts": [{"src": c["src"], "surplus": c["surplus"], "tilt_cut": c["tilt_cut"], "L": c["L"], "pen0": c["pen0"]} for c in cuts],
        "top_polished": [{k: v for k, v in r.items() if k != "rows"} for r in polished],
        "win": best_win,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", type=int, nargs="+", default=[748, 790, 795, 852])
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    n_threads = min(4, max(1, args.threads))

    t_all = time.time()
    df = pd.read_csv(MERGED_CSV)
    mats = {int(n): _norm(sub) for n, sub in df.groupby("n")}
    inc_front = u.load_incumbent_and_frontier(args.targets)

    per_n: dict = {}
    if OUT_JSON_21.exists():
        try:
            per_n = json.loads(OUT_JSON_21.read_text()).get("per_n", {})
        except Exception:
            per_n = {}

    all_wins = load_verified_disk_wins(DATA_DIR)
    for n in args.targets:
        res = solve_target(n, mats, inc_front, n_threads, args.quick)
        w = res.pop("win")
        per_n[str(n)] = res
        if w is not None:
            prev_w = all_wins.get(n)
            if prev_w is None or float(w["verify"]["L_float"]) < float(prev_w["verify"]["L_float"]):
                all_wins[n] = w
                DATA_DIR.mkdir(parents=True, exist_ok=True)
                write_final_csv_from_wins({n: w}, DATA_DIR / f"win_n{n}.csv")

    out = {
        "targets": sorted(int(k) for k in per_n),
        "per_n": per_n,
        "elapsed_s": round(time.time() - t_all, 2),
        "wins": [{"n": n, "L_lit": w["verify"]["L_lit"], "L_pandas": w["verify"]["L_pandas"], "label": w["label"]}
                 for n, w in sorted(all_wins.items())],
    }
    OUT_JSON_21.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON_21.write_text(json.dumps(out, indent=2, default=float))
    if all_wins:
        write_final_csv_from_wins(all_wins, FINAL_CSV)
        print(f"\nWROTE STRICT CERTIFIED WINS to {FINAL_CSV}: {sorted(all_wins)}")


if __name__ == "__main__":
    main()
