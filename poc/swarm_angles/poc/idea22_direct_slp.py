#!/usr/bin/env python3
"""Idea 22: Direct Multi-Margin joint_xytheta_slp + 4 Micro-Jitter Harvest on merged.csv.

Runs signed-Jacobian joint_xytheta_slp directly on merged.csv[n] (and existing wins)
with 1 unperturbed + 4 micro-jitter starts (1e-4 pos, 0.02 deg angle) and multi-margin
final LP clearance [5e-9, 2e-8, 1e-7, 5e-7] verified against squares.core + pandas.
"""
from __future__ import annotations

import argparse
from decimal import Decimal
import json
import math
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pandas as pd

import utils as u
from idea17_run_slp import _build_joint_slp_sparse
from idea18_run import write_final_csv_from_wins
from idea19_run import MERGED_CSV
from idea20_partner_shift import DATA_DIR, FINAL_CSV, _norm, load_verified_disk_wins

HERE = Path(__file__).resolve().parent
OUT_JSON_22 = HERE.parent / "docs" / "idea22_results.json"
TARGETS_12 = [228, 340, 456, 491, 694, 748, 790, 795, 848, 852, 965, 982]


def direct_slp_multimargin(
    n: int,
    x: np.ndarray,
    y: np.ndarray,
    th: np.ndarray,
    rounds: int = 16,
    tr_th0: float = 0.025,
    tr_xy0: float = 0.12,
    margins: tuple[float, ...] = (5e-9, 2e-8, 1e-7, 5e-7),
) -> tuple[dict, float]:
    """Run joint (dx, dy, dth, L) SLP and certify tightest passing margin in `margins`."""
    k = math.isqrt(n)
    hx = 0.5 * (np.abs(np.cos(th)) + np.abs(np.sin(th)))
    cur_x = x - np.min(x - hx)
    cur_y = y - np.min(y - hx)
    cur_th = th.copy()

    pen0, bbox0 = u.sat_max_overlap_and_bbox(cur_x, cur_y, cur_th)
    if pen0 > 1e-7:
        cur_x, cur_y, cur_th, pen0 = u.relax_lbfgs(cur_x, cur_y, cur_th, max(bbox0, k + 0.45), maxiter=200)
        pen0, bbox0 = u.sat_max_overlap_and_bbox(cur_x, cur_y, cur_th)

    best_x, best_y, best_th = cur_x.copy(), cur_y.copy(), cur_th.copy()
    best_L = bbox0 if pen0 <= 1e-12 else 1e9
    if best_L > 1e8 and pen0 <= 1e-6:
        tx, ty, tL, tok = u.fixed_angle_lp_tighten(cur_x, cur_y, cur_th, rounds=4, margin=1e-7)
        if tok:
            best_x, best_y, best_th, best_L = tx, ty, cur_th.copy(), tL

    obj = np.zeros(3 * n + 1, dtype=np.float64)
    obj[3 * n] = 1.0
    tr_th, tr_xy = tr_th0, tr_xy0

    for _ in range(rounds):
        A_ub, b_ub = _build_joint_slp_sparse(cur_x, cur_y, cur_th, tr_xy=tr_xy, tr_th=tr_th, margin=1e-8)
        bounds = (
            [(-tr_xy, tr_xy)] * (2 * n)
            + [(-tr_th, tr_th)] * n
            + [(float(k) + 0.40, float(max(bbox0, k + 0.65) + 0.30))]
        )
        res = u._highs_linprog(obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds)
        if not res.success or res.x is None:
            tr_th *= 0.6
            tr_xy *= 0.7
            continue

        dx, dy, dth, lp_L = res.x[:n], res.x[n : 2 * n], res.x[2 * n : 3 * n], float(res.x[3 * n])
        accepted = False
        for alpha in (1.0, 0.5):
            cand_x = cur_x + alpha * dx
            cand_y = cur_y + alpha * dy
            cand_th = cur_th + alpha * dth
            chx = 0.5 * (np.abs(np.cos(cand_th)) + np.abs(np.sin(cand_th)))
            cand_x -= np.min(cand_x - chx)
            cand_y -= np.min(cand_y - chx)
            pen_c, bbox_c = u.sat_max_overlap_and_bbox(cand_x, cand_y, cand_th)
            if pen_c > 1e-8:
                target_L = max(lp_L, min(bbox_c, bbox0) - 0.0015)
                cand_x, cand_y, cand_th, _ = u.relax_lbfgs(
                    cand_x, cand_y, cand_th, target_L, maxiter=90, w_wall=2.0, w_pair=1.0
                )
                chx = 0.5 * (np.abs(np.cos(cand_th)) + np.abs(np.sin(cand_th)))
                cand_x -= np.min(cand_x - chx)
                cand_y -= np.min(cand_y - chx)
                pen_c, bbox_c = u.sat_max_overlap_and_bbox(cand_x, cand_y, cand_th)

            if pen_c <= 1e-12 and bbox_c < best_L:
                best_x, best_y, best_th, best_L = cand_x.copy(), cand_y.copy(), cand_th.copy(), bbox_c
            elif pen_c <= 1e-7 and bbox_c < best_L + 0.002:
                tx, ty, tL, tok = u.fixed_angle_lp_tighten(cand_x, cand_y, cand_th, rounds=3, margin=1e-7)
                if tok and tL < best_L:
                    best_x, best_y, best_th, best_L = tx, ty, cand_th.copy(), tL

            if pen_c <= max(5e-4, pen0 * 0.9) and bbox_c <= bbox0 + 0.01:
                cur_x, cur_y, cur_th = cand_x, cand_y, cand_th
                pen0, bbox0 = pen_c, bbox_c
                accepted = True
                break

        if accepted:
            tr_th = max(2e-5, tr_th * 0.84)
            tr_xy = max(5e-4, tr_xy * 0.88)
        else:
            tr_th = max(1e-5, tr_th * 0.5)
            tr_xy = max(5e-4, tr_xy * 0.6)

    # Multi-margin final LP clearance across both best and final iterate
    best_ver: dict = {"ok": False}
    best_L_cert = 1e18
    for cx, cy, cth in [(best_x, best_y, best_th), (cur_x, cur_y, cur_th)]:
        p, b = u.sat_max_overlap_and_bbox(cx, cy, cth)
        if p > 1e-8:
            cx, cy, cth, _ = u.relax_lbfgs(cx, cy, cth, b, maxiter=160, w_wall=2.0, w_pair=1.0)
        for m in margins:
            tx, ty, tL, tok = u.fixed_angle_lp_tighten(cx, cy, cth, rounds=5, margin=m)
            if tok and tL < best_L_cert + 1e-4:
                ver = u.verify_official(n, tx, ty, cth)
                if ver.get("ok"):
                    L_v = float(ver["L_float"])
                    if L_v < best_L_cert:
                        best_L_cert = L_v
                        best_ver = ver
                    break
    return best_ver, best_L_cert


def _worker_task(args: tuple) -> dict:
    n, label, x, y, th = args
    t0 = time.time()
    ver, L_cert = direct_slp_multimargin(n, x, y, th)
    return {
        "n": n, "label": label, "ok": bool(ver.get("ok")),
        "L_cert": float(L_cert) if ver.get("ok") else None,
        "verify": {k: v for k, v in ver.items() if k != "rows"},
        "rows": ver.get("rows"),
        "dt": round(time.time() - t0, 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", type=int, nargs="+", default=TARGETS_12)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    n_threads = min(4, max(1, args.threads))

    t0 = time.time()
    df = pd.read_csv(MERGED_CSV)
    groups = {int(n): sub.sort_values("square_id") for n, sub in df.groupby("n")}
    all_wins = load_verified_disk_wins(DATA_DIR)

    # Exact baseline incumbent L_lit from merged.csv (without float coordinate shift)
    inc_exact: dict[int, Decimal] = {}
    tasks = []
    pos_eps = 1e-4
    ang_eps = math.radians(0.02)

    for n in args.targets:
        sub = groups[n]
        raw_rows = [
            {"square_id": str(int(r.square_id)), "x": repr(float(r.x)), "y": repr(float(r.y)), "angle": repr(float(r.angle))}
            for r in sub.itertuples()
        ]
        _, _, L_inc_lit, reason = u.sc.evaluate(n, raw_rows)
        assert reason == "ok" and L_inc_lit is not None, f"Incumbent n={n} failed: {reason}"
        inc_exact[n] = Decimal(L_inc_lit)
        x0 = sub["x"].to_numpy(np.float64)
        y0 = sub["y"].to_numpy(np.float64)
        th0 = sub["angle"].to_numpy(np.float64)
        tasks.append((n, "merged_exact", x0.copy(), y0.copy(), th0.copy()))
        rng = np.random.default_rng(22000 + n)
        for j in range(4):
            xj = x0 + rng.uniform(-pos_eps, pos_eps, n)
            yj = y0 + rng.uniform(-pos_eps, pos_eps, n)
            tj = th0 + rng.uniform(-ang_eps, ang_eps, n)
            tasks.append((n, f"jitter_{j}", xj, yj, tj))
        if n in all_wins:
            w_sub = pd.read_csv(DATA_DIR / f"win_n{n}.csv")
            w_sub["sid"] = w_sub["id"].astype(str).str.split("_").str[1].astype(int)
            w_sub = w_sub.sort_values("sid")
            tasks.append((n, "disk_win", w_sub["x"].to_numpy(np.float64), w_sub["y"].to_numpy(np.float64),
                          np.radians(w_sub["deg"].to_numpy(np.float64))))

    with mp.Pool(processes=n_threads) as pool:
        results = pool.map(_worker_task, tasks)

    per_n: dict = {}
    for n in args.targets:
        sub_res = [r for r in results if r["n"] == n]
        ok_res = [r for r in sub_res if r["ok"]]
        ok_res.sort(key=lambda r: max(Decimal(r["verify"]["L_lit"]), Decimal(r["verify"]["L_pandas"])))
        best = ok_res[0] if ok_res else None
        inc_dec = inc_exact[n]
        win_flag = False
        if best is not None:
            best_dec = max(Decimal(best["verify"]["L_lit"]), Decimal(best["verify"]["L_pandas"]))
            dL = float(best_dec - inc_dec)
            if best_dec < inc_dec:
                win_flag = True
                prev = all_wins.get(n)
                prev_dec = max(Decimal(prev["verify"]["L_lit"]), Decimal(prev["verify"]["L_pandas"])) if prev else inc_dec
                if best_dec < prev_dec:
                    all_wins[n] = best
                    DATA_DIR.mkdir(parents=True, exist_ok=True)
                    write_final_csv_from_wins({n: best}, DATA_DIR / f"win_n{n}.csv")
            print(f"n={n:3d}: {best['label']:12s} L={best['verify']['L_lit'][:18]} vs inc={str(inc_dec)[:18]} dL={dL:+.6e} win={win_flag}", flush=True)
        else:
            print(f"n={n:3d}: NO VALID CERTIFIED", flush=True)
        per_n[str(n)] = {
            "inc_exact": str(inc_dec),
            "win": win_flag,
            "best": {k: v for k, v in best.items() if k != "rows"} if best else None,
            "runs": [{k: v for k, v in r.items() if k not in ("rows", "verify")} for r in sub_res],
        }

    out = {
        "targets": args.targets,
        "elapsed_s": round(time.time() - t0, 2),
        "per_n": per_n,
        "wins": [{"n": n, "L_lit": w["verify"]["L_lit"], "L_pandas": w["verify"]["L_pandas"], "label": w["label"]}
                 for n, w in sorted(all_wins.items())],
    }
    OUT_JSON_22.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON_22.write_text(json.dumps(out, indent=2, default=float))
    if all_wins:
        write_final_csv_from_wins(all_wins, FINAL_CSV)
        print(f"\nElapsed {out['elapsed_s']}s | ALL WINS in {FINAL_CSV}: {sorted(all_wins)}")


if __name__ == "__main__":
    main()
