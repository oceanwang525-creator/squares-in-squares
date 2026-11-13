"""Idea 15A: Fine 16-step L-continuation L-BFGS + fixed-angle LP tighten on n=401, 530, 842."""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import json
import math
import sys
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd

THIS_DIR = str(Path(__file__).resolve().parent)
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from poc.swarm_angles.poc import utils as u
from poc.swarm_angles.poc.idea14_lbfgs_compress import fill_best_holes
from poc.swarm_angles.poc.idea12_highs_search import make_block
from poc.swarm_angles.poc.idea13_chevron_highs import solve_grid9_tl


def run_fine_seed(args):
    tag, n, L_base, x0, y0, th0, sched = args
    x = np.asarray(x0, dtype=np.float64).copy()
    y = np.asarray(y0, dtype=np.float64).copy()
    th = np.asarray(th0, dtype=np.float64).copy()

    best_L = 1e18
    best_cfg = None
    last_pen = 1e9

    for Lw in sched:
        hx = 0.5 * (np.abs(np.cos(th)) + np.abs(np.sin(th)))
        minx, maxx = float(np.min(x - hx)), float(np.max(x + hx))
        miny, maxy = float(np.min(y - hx)), float(np.max(y + hx))
        span_x = max(1e-6, maxx - minx)
        span_y = max(1e-6, maxy - miny)
        if span_x > Lw or span_y > Lw:
            scale = Lw / max(span_x, span_y)
            x = (x - minx) * scale
            y = (y - miny) * scale
        else:
            x = x - minx + 0.5 * (Lw - span_x)
            y = y - miny + 0.5 * (Lw - span_y)

        x, y, th, pen = u.relax_lbfgs(x, y, th, Lw, maxiter=900, w_wall=2.5, w_pair=1.0)
        last_pen = pen
        if pen < 5e-4:
            lx, ly, lL, ok = u.fixed_angle_lp_tighten(x, y, th, rounds=6, margin=1e-9)
            if ok and lL < best_L:
                best_L = lL
                best_cfg = (lx.copy(), ly.copy(), th.copy())
                x, y = lx.copy(), ly.copy()

    return (tag, n, best_L, last_pen, best_cfg)


def build_fine_seeds(n: int, k: int, L_base: float, df_merged: pd.DataFrame, nm_list: list) -> list:
    seeds = []
    # Fine 16-step continuation from L_base+0.08 down to L_base-0.014
    sched = [
        L_base + 0.08, L_base + 0.04, L_base + 0.02, L_base + 0.01,
        L_base + 0.005, L_base + 0.002, L_base + 0.0005,
        L_base - 0.0005, L_base - 0.0015, L_base - 0.003,
        L_base - 0.005, L_base - 0.008, L_base - 0.012,
    ]
    warm_sched = [k + 1.2, k + 0.85, k + 0.68, k + 0.59] + sched

    # 1) Chiral A x B pinwheels around D=1 cores
    for B in [4, 5, 6, 7]:
        A = k - B
        m = A - B
        n_core = m * m + 1
        sub_c = df_merged[df_merged["n"] == n_core]
        if len(sub_c) != n_core:
            continue
        cx_raw = sub_c["x"].to_numpy()
        cy_raw = sub_c["y"].to_numpy()
        cth_raw = sub_c["angle"].to_numpy()
        chx = 0.5 * (np.abs(np.cos(cth_raw)) + np.abs(np.sin(cth_raw)))
        cx_raw = cx_raw - (cx_raw - chx).min()
        cy_raw = cy_raw - (cy_raw - chx).min()
        Lc = max((cx_raw + chx).max(), (cy_raw + chx).max())
        Lw0 = k + 0.85
        for deg in [0.0, 1.5, 3.0]:
            th = math.radians(deg)
            xs, ys, ths = [], [], []
            arms = [
                (A, B, th, 0.5 * A + 0.1, 0.5 * B + 0.1),
                (B, A, th, Lw0 - 0.5 * B - 0.1, 0.5 * A + 0.1),
                (A, B, th, Lw0 - 0.5 * A - 0.1, Lw0 - 0.5 * B - 0.1),
                (B, A, th, 0.5 * B + 0.1, Lw0 - 0.5 * A - 0.1),
            ]
            for uu, vv, aa, bcx, bcy in arms:
                ca, sa = math.cos(aa), math.sin(aa)
                for i in range(uu):
                    for j in range(vv):
                        u_ = i - 0.5 * (uu - 1)
                        v_ = j - 0.5 * (vv - 1)
                        xs.append(bcx + u_ * ca - v_ * sa)
                        ys.append(bcy + u_ * sa + v_ * ca)
                        ths.append(aa)
            xs.extend((cx_raw - 0.5 * Lc + 0.5 * Lw0).tolist())
            ys.extend((cy_raw - 0.5 * Lc + 0.5 * Lw0).tolist())
            ths.extend(cth_raw.tolist())
            seeds.append((f"pin_A{A}_B{B}_d{deg}", n, L_base, xs, ys, ths, warm_sched))

    # 2) Global small-tilt lattice + 1 inserted square
    for deg in [1.0, 2.5, 4.5]:
        th = math.radians(deg)
        ca, sa = math.cos(th), math.sin(th)
        Lw0 = k + 0.90
        xs, ys, ths = [], [], []
        for i in range(k):
            for j in range(k):
                u_ = (i - 0.5 * (k - 1)) * 1.02
                v_ = (j - 0.5 * (k - 1)) * 1.02
                xs.append(0.5 * Lw0 + u_ * ca - v_ * sa)
                ys.append(0.5 * Lw0 + u_ * sa + v_ * ca)
                ths.append(th)
        for pos_tag, px, py, pdeg in [("corner", 0.6, 0.6, 45.0), ("edge", 0.5 * Lw0, 0.6, 36.87)]:
            seeds.append((f"glob_d{deg}_{pos_tag}", n, L_base, xs + [px], ys + [py], ths + [math.radians(pdeg)], warm_sched))

    # 3) Near-misses from 14A with hole insertion (on 401/442)
    if n == 401:
        b12 = make_block(20.534534, 12, 12, math.pi / 4, 0.5 * 20.534534, 0.5 * 20.534534, 0)
        r12 = solve_grid9_tl(("diam12", "diam", 20.534534, b12, 0.0, False, 3.0))
        if r12[2] < n:
            til = np.asarray(b12)
            up = r12[5]
            x0 = np.concatenate([til[:, 0], up[:, 0]])
            y0 = np.concatenate([til[:, 1], up[:, 1]])
            th0 = np.concatenate([til[:, 2], np.zeros(len(up))])
            for add_deg in [0.0, 22.5, 36.87]:
                hx, hy, hth = fill_best_holes(x0, y0, th0, 20.534534, n - r12[2], math.radians(add_deg))
                seeds.append((f"diam12_ins_{add_deg}", n, L_base, hx, hy, hth, warm_sched[2:]))

    for idx, nm in enumerate(nm_list[:3]):
        tot = nm["tot"]
        if tot >= n:
            continue
        til = np.asarray(nm["tilted"], dtype=np.float64)
        up = np.asarray(nm["up"], dtype=np.float64)
        x0 = np.concatenate([til[:, 0], up[:, 0]]) if len(til) > 0 else up[:, 0]
        y0 = np.concatenate([til[:, 1], up[:, 1]]) if len(til) > 0 else up[:, 1]
        th0 = np.concatenate([til[:, 2], np.zeros(len(up))]) if len(til) > 0 else np.zeros(len(up))
        for add_deg in [0.0, 36.87]:
            hx, hy, hth = fill_best_holes(x0, y0, th0, L_base - 0.001, n - tot, math.radians(add_deg))
            seeds.append((f"nm{idx}_def{n - tot}_{add_deg}", n, L_base, hx, hy, hth, warm_sched[2:]))

    # 4) Multi-angle kicks on merged.csv incumbent
    sub = df_merged[df_merged["n"] == n]
    if len(sub) == n:
        mx = sub["x"].to_numpy()
        my = sub["y"].to_numpy()
        mth = sub["angle"].to_numpy()
        mhx = 0.5 * (np.abs(np.cos(mth)) + np.abs(np.sin(mth)))
        mx = mx - (mx - mhx).min()
        my = my - (my - mhx).min()
        rng = np.random.default_rng(1000 + n)
        for sig in [0.08, 0.16, 0.25]:
            for sd in [1, 2]:
                nth = mth + rng.normal(0.0, sig, size=n)
                seeds.append((f"kick_s{sig}_{sd}", n, L_base, mx * 1.004, my * 1.004, nth, warm_sched[2:]))

    return seeds


def main():
    t0 = time.time()
    df_merged = pd.read_csv("poc/pell_corrugation/data/border-scan/merged.csv")
    nm_path = Path("/home/lucas/.gemini/antigravity-cli/brain/95f98074-e284-43bd-8857-8c64d023144c/scratch/idea14a_nearmisses.json")
    nm_all = json.loads(nm_path.read_text()) if nm_path.exists() else {}

    targets = [
        (401, 20, 20.535533905933),
        (530, 23, 23.535533905933),
        (842, 29, 29.520815280171),
    ]
    summary = []
    with ProcessPoolExecutor(max_workers=7, mp_context=mp.get_context("spawn")) as ex:
        for n, k, L_base in targets:
            tn = time.time()
            seeds = build_fine_seeds(n, k, L_base, df_merged, nm_all.get(str(n), []))
            print(f"\nn={n} (L_base={L_base:.6f}): running {len(seeds)} fine 16-step seeds...", flush=True)
            res = list(ex.map(run_fine_seed, seeds, chunksize=1))
            feas = sorted([r for r in res if r[4] is not None], key=lambda r: r[2])
            print(f"  feasible={len(feas)}/{len(seeds)}", flush=True)
            for tag, _, bL, lpen, _ in feas[:5]:
                print(f"    feas {tag:22s}: L={bL:.9f} (dL={bL - L_base:+.9f})", flush=True)
            if not feas:
                mp_r = min(res, key=lambda r: r[3])
                print(f"    no feas; best last_pen={mp_r[3]:.6e} ({mp_r[0]})", flush=True)

            win_L = L_base
            for tag, _, bL, _, cfg in feas[:3]:
                if bL < win_L - 1e-11:
                    ver = u.verify_official(n, cfg[0], cfg[1], cfg[2])
                    if ver["ok"] and ver["L_float"] < win_L - 1e-11:
                        win_L = ver["L_float"]
                        df_out = pd.DataFrame(ver["rows"])
                        df_out["n"] = n
                        df_out["id"] = [f"{n}_{sid}" for sid in df_out["square_id"]]
                        Path("poc/swarm_angles/data").mkdir(parents=True, exist_ok=True)
                        df_out[["id", "n", "square_id", "x", "y", "angle"]].to_csv(
                            f"poc/swarm_angles/data/cand15a_n{n}.csv", index=False
                        )
                        print(f"  *** CERTIFIED WIN n={n}: L={win_L:.12f} (< {L_base:.12f}) via {tag}! ***", flush=True)

            summary.append({
                "n": n, "L_base": L_base, "seeds": len(seeds), "feasible": len(feas),
                "best_L": feas[0][2] if feas else None, "best_tag": feas[0][0] if feas else None,
                "win_L": win_L, "seconds": round(time.time() - tn, 2),
            })

    with open("poc/swarm_angles/docs/idea15a_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nIdea 15A finished in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
