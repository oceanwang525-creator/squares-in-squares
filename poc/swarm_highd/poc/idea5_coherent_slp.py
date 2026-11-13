#!/usr/bin/env python3
"""Idea 5: Iterative n=862 Continuation + Coherent No-Melt Pocket/Tilted Jitter Joint SLP."""
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
from joint_slp import joint_xytheta_slp
import idea1_highd_pilot as i1
import idea2_highd_sweep as i2

HERE = Path(__file__).resolve().parent
OUT_JSON = HERE.parent / "docs" / "idea5_results.json"
FINAL_CSV = HERE.parent / "final.csv"
DATA_DIR = HERE.parent / "data"


def _coherent_seeds(
    n: int,
    x_base: np.ndarray,
    y_base: np.ndarray,
    th_base: np.ndarray,
    L_base: float,
    n_jit_seeds: int,
    rng: np.random.Generator,
) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray, int, float, float]]:
    """Generate non-destructive coherent perturbations for joint_xytheta_slp."""
    deg = np.degrees(th_base) % 90.0
    deg_m = np.minimum(deg, 90.0 - deg)
    tilted = deg_m > 1.5
    hx = 0.5 * (np.abs(np.cos(th_base)) + np.abs(np.sin(th_base)))

    mL = x_base - hx - float((x_base - hx).min())
    mR = float((x_base + hx).max()) - (x_base + hx)
    mB = y_base - hx - float((y_base - hx).min())
    mT = float((y_base + hx).max()) - (y_base + hx)
    cL, cR = int(np.sum(mL < 5e-4)), int(np.sum(mR < 5e-4))
    cB, cT = int(np.sum(mB < 5e-4)), int(np.sum(mT < 5e-4))
    dx_sign = 1.0 if cR <= cL else -1.0
    dy_sign = 1.0 if cT <= cB else -1.0

    voids = i1.clearance_points(x_base, y_base, th_base, L_base, n_pts=4, step=0.18)

    # Approximate pairwise center distance to find loosest squares
    dx_m = x_base[:, None] - x_base[None, :]
    dy_m = y_base[:, None] - y_base[None, :]
    d2 = dx_m * dx_m + dy_m * dy_m + np.eye(n) * 1e6
    nn_dist = np.sqrt(d2.min(axis=1))
    loose_idx = np.argsort(-nn_dist)[:16]

    seeds: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, int, float, float]] = []
    if n == 862:
        seeds.append(("cont_fine", x_base.copy(), y_base.copy(), th_base.copy(), 16, 0.006, 0.02))
        seeds.append(("cont_med", x_base.copy(), y_base.copy(), th_base.copy(), 16, 0.016, 0.06))

    # 1. Clean micro-jitter seeds (different RNG draws from Idea 3)
    for k in range(n_jit_seeds):
        pj = 8e-5 if k % 2 == 0 else 2.5e-4
        aj = np.radians(0.02 if k % 2 == 0 else 0.05)
        x = x_base + rng.uniform(-pj, pj, n)
        y = y_base + rng.uniform(-pj, pj, n)
        th = th_base + rng.uniform(-aj, aj, n)
        seeds.append((f"mjit_{k}", x, y, th, 12, 0.014, 0.05))

    # 2. Tilted-corridor-only coherent jitters (upright core untouched)
    for tag, pj, aj_deg in (("tilt_a", 3e-4, 0.12), ("tilt_b", 8e-4, 0.25)):
        x, y, th = x_base.copy(), y_base.copy(), th_base.copy()
        nt = int(tilted.sum())
        if nt > 0:
            x[tilted] += rng.uniform(-pj, pj, nt)
            y[tilted] += rng.uniform(-pj, pj, nt)
            th[tilted] += rng.uniform(-np.radians(aj_deg), np.radians(aj_deg), nt)
        seeds.append((tag, x, y, th, 12, 0.015, 0.05))

    # 3. Void-pocket coherent nudge (R=4.5 around top void)
    if voids:
        vx, vy, _ = voids[0]
        x, y, th = x_base.copy(), y_base.copy(), th_base.copy()
        near = ((x - vx) ** 2 + (y - vy) ** 2) <= 4.5 ** 2
        nm = int(near.sum())
        x[near] += 1.5e-3 * (vx - x[near]) + rng.uniform(-5e-4, 5e-4, nm)
        y[near] += 1.5e-3 * (vy - y[near]) + rng.uniform(-5e-4, 5e-4, nm)
        th[near] += rng.uniform(-np.radians(0.20), np.radians(0.20), nm)
        seeds.append(("void_nudge", x, y, th, 12, 0.015, 0.05))

    # 4. Weak-wall drift + loose-square rattle nudge
    x, y, th = x_base.copy(), y_base.copy(), th_base.copy()
    half_x = (x < L_base * 0.6) if dx_sign > 0 else (x > L_base * 0.4)
    half_y = (y < L_base * 0.6) if dy_sign > 0 else (y > L_base * 0.4)
    x[half_x] += dx_sign * 4e-4
    y[half_y] += dy_sign * 3e-4
    x[loose_idx] += rng.uniform(-2e-3, 2e-3, len(loose_idx))
    y[loose_idx] += rng.uniform(-2e-3, 2e-3, len(loose_idx))
    th[loose_idx] += rng.uniform(-np.radians(0.35), np.radians(0.35), len(loose_idx))
    seeds.append(("wall_loose", x, y, th, 12, 0.016, 0.06))

    return seeds


def _worker(args: tuple) -> dict:
    n, x_base, y_base, th_base, L_inc, L_base, L_front, n_jit = args
    t0 = time.time()
    rng = np.random.default_rng(50500 + n)
    seeds = _coherent_seeds(n, x_base, y_base, th_base, L_base, n_jit, rng)

    best_L = 1e9
    best_ver: dict = {"ok": False}
    best_tag = "none"
    best_rows = None

    for tag, x0, y0, th0, rds, tth, txy in seeds:
        # Pre-relax at L_base + 5e-4 if perturbed into tiny overlap
        pen0, _ = u.sat_max_overlap_and_bbox(x0, y0, th0)
        if pen0 > 1e-8:
            x0, y0, th0, _ = u.relax_lbfgs(x0, y0, th0, L_base + 5e-4, maxiter=90, w_wall=3.0, w_pair=1.0)
        xs, ys, ts, Ls, ok_s = joint_xytheta_slp(
            x0, y0, th0, rounds=rds, tr_th0=tth, tr_xy0=txy, margin=1e-8, tol=1e-11, patience=2
        )
        cx, cy, cth = (xs, ys, ts) if ok_s else (x0, y0, th0)
        for m_lp in (1e-7, 2e-8, 5e-9):
            xf, yf, L_lp, ok_lp = u.fixed_angle_lp_tighten(cx, cy, cth, rounds=5, margin=m_lp)
            if not ok_lp:
                continue
            if L_lp < L_base - 1e-11:
                ver = u.verify_official(n, xf, yf, cth)
                if ver.get("ok"):
                    Lf = float(ver["L_float"])
                    if Lf < best_L:
                        best_L = Lf
                        best_ver = ver
                        best_tag = f"{tag}_m{m_lp:.0e}"
                        best_rows = ver["rows"]

    win = best_ver.get("ok", False) and best_L < L_base - 1e-11
    return {
        "n": n,
        "L_inc": L_inc,
        "L_base": L_base,
        "L_front": L_front,
        "best_tag": best_tag,
        "L_best": best_L if win else None,
        "dL_base": (best_L - L_base) if win else None,
        "dL_inc": (best_L - L_inc) if win else None,
        "win": bool(win),
        "verify": {k: v for k, v in best_ver.items() if k != "rows"},
        "rows": best_rows,
        "dt": round(time.time() - t0, 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", type=int, nargs="+", default=[862, 339, 545, 639, 698, 703, 805])
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--njit", type=int, default=3)
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

    tasks = []
    for n in args.targets:
        if n in wins:
            sub = pd.read_csv(DATA_DIR / f"win_n{n}.csv")
            sub["sid"] = sub["id"].astype(str).str.split("_").str[1].astype(int)
            sub = sub.sort_values("sid")
            xb = sub["x"].to_numpy(np.float64)
            yb = sub["y"].to_numpy(np.float64)
            tb = np.radians(sub["deg"].to_numpy(np.float64))
            Lb = float(wins[n]["verify"]["L_float"])
        else:
            xb, yb, tb, Lb = mats[n]
        tasks.append((n, xb, yb, tb, inc_front[n]["L_ours"], Lb, inc_front[n]["L_front"], args.njit))

    res_list = []
    with mp.Pool(processes=min(4, n_threads, len(tasks))) as pool:
        for r in pool.imap_unordered(_worker, tasks):
            rows = r.pop("rows", None)
            res_list.append(r)
            print(f"[Idea5 n={r['n']}] tag={r['best_tag']} dL_base={r['dL_base']} dL_inc={r['dL_inc']} win={r['win']} ({r['dt']}s)", flush=True)
            if r["win"] and rows is not None:
                nw = r["n"]
                w_obj = {"n": nw, "label": r["best_tag"], "rows": rows, "verify": r["verify"]}
                wins[nw] = w_obj
                DATA_DIR.mkdir(parents=True, exist_ok=True)
                u.write_final_csv_from_wins({nw: w_obj}, DATA_DIR / f"win_n{nw}.csv")

    res_list.sort(key=lambda r: r["n"])
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({"results": res_list, "dt_s": round(time.time() - t_all, 2)}, indent=2))
    print(f"Idea 5 done in {time.time()-t_all:.1f}s | wins={sorted(wins.keys())}", flush=True)

    if wins:
        u.write_final_csv_from_wins(wins, FINAL_CSV)
        print(f"Updated {FINAL_CSV} with wins={sorted(wins.keys())}", flush=True)


if __name__ == "__main__":
    main()
