"""Idea 14B: Deterministic multi-start L-BFGS + fixed-angle LP warm-box compression (NO SA)
on n=401 and n=442 across:
  1) Top 14A near-misses + diam12 (400/401) with hole-inserted missing squares
  2) C4 chiral A x B @ theta 4-arm pinwheels around D=1 central cores (17, 26, 37, 50, 101)
  3) Global small-tilt k x k lattice + 1 inserted edge/corner/center square
  4) Macro-perturbed merged.csv n=401 / n=442 with non-45deg block tilts & quadrant chiral twists.
"""
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
from poc.swarm_angles.poc.idea12_highs_search import make_block


def fill_best_holes(x: np.ndarray, y: np.ndarray, th: np.ndarray, L: float, need: int, add_th: float):
    """Greedily place `need` extra squares at max-clearance points in [0.5, L-0.5]^2."""
    xs, ys, ths = list(x), list(y), list(th)
    grid = np.linspace(0.6, L - 0.6, 45)
    gx, gy = np.meshgrid(grid, grid, indexing="ij")
    cand = np.stack([gx.ravel(), gy.ravel()], axis=1)
    for _ in range(need):
        ax = np.asarray(xs)
        ay = np.asarray(ys)
        d2 = (cand[:, 0:1] - ax[None, :]) ** 2 + (cand[:, 1:2] - ay[None, :]) ** 2
        md2 = np.min(d2, axis=1)
        best = int(np.argmax(md2))
        xs.append(float(cand[best, 0]))
        ys.append(float(cand[best, 1]))
        ths.append(float(add_th))
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64), np.asarray(ths, dtype=np.float64)


def run_seed(args):
    tag, n, L_base, x0, y0, th0, L_warm_list = args
    x = np.asarray(x0, dtype=np.float64).copy()
    y = np.asarray(y0, dtype=np.float64).copy()
    th = np.asarray(th0, dtype=np.float64).copy()
    assert len(x) == n

    best_L = 1e18
    best_cfg = None
    last_pen = 1e9

    # Warm-to-tight continuation
    for Lw in L_warm_list:
        hx = 0.5 * (np.abs(np.cos(th)) + np.abs(np.sin(th)))
        cur_L = max(Lw, float(np.max(x + hx) - np.min(x - hx)), float(np.max(y + hx) - np.min(y - hx)))
        # Center inside [0, Lw] if coming from larger scale
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

        x, y, th, pen = u.relax_lbfgs(x, y, th, Lw, maxiter=550, w_wall=2.0, w_pair=1.0)
        last_pen = pen
        if pen < 5e-4:
            lx, ly, lL, ok = u.fixed_angle_lp_tighten(x, y, th, rounds=6, margin=1e-9)
            if ok and lL < best_L:
                best_L = lL
                best_cfg = (lx.copy(), ly.copy(), th.copy())
                x, y = lx.copy(), ly.copy()

    # Final squeeze attempt just below L_base if feasible or near-feasible
    if best_cfg is not None and best_L < L_base + 0.05:
        for dL in [0.002, 0.005, 0.009]:
            L_tgt = min(best_L, L_base) - dL
            tx, ty, tth, pen = u.relax_lbfgs(x, y, th, L_tgt, maxiter=650, w_wall=3.0, w_pair=1.0)
            if pen < 5e-4:
                lx, ly, lL, ok = u.fixed_angle_lp_tighten(tx, ty, tth, rounds=6, margin=1e-9)
                if ok and lL < best_L:
                    best_L = lL
                    best_cfg = (lx.copy(), ly.copy(), tth.copy())
                    x, y, th = lx.copy(), ly.copy(), tth.copy()

    return (tag, n, best_L, last_pen, best_cfg)


def build_seeds(n: int, k: int, L_base: float, df_merged: pd.DataFrame, near_misses: list) -> list:
    seeds = []
    warm_sched = [k + 0.72, k + 0.62, k + 0.56, k + 0.538, L_base - 0.001, L_base - 0.004, L_base - 0.008]

    # 1) Hole-inserted 14A near-misses + diam12 (on n=401)
    if n == 401:
        b12 = make_block(20.534534, 12, 12, math.pi / 4, 0.5 * 20.534534, 0.5 * 20.534534, 0)
        from poc.swarm_angles.poc.idea13_chevron_highs import solve_grid9_tl
        r12 = solve_grid9_tl(("diam12", "diam", 20.534534, b12, 0.0, False, 3.0))
        if r12[2] < n:
            til = np.asarray(b12)
            up = r12[5]
            x0 = np.concatenate([til[:, 0], up[:, 0]])
            y0 = np.concatenate([til[:, 1], up[:, 1]])
            th0 = np.concatenate([til[:, 2], np.zeros(len(up))])
            for add_deg in [0.0, 22.5, 36.87, 45.0]:
                hx, hy, hth = fill_best_holes(x0, y0, th0, 20.534534, n - r12[2], math.radians(add_deg))
                seeds.append((f"diam12_ins_{add_deg}", n, L_base, hx, hy, hth, warm_sched))

    for idx, nm in enumerate(near_misses[:5]):
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
            seeds.append((f"nm{idx}_def{n - tot}_{add_deg}", n, L_base, hx, hy, hth, warm_sched))

    # 2) C4 chiral A x B @ theta 4-arm pinwheels around central core (D=1 subpackings from merged.csv)
    for B in [4, 5, 6, 7, 8]:
        A = k - B
        m = A - B  # central square size m^2 + 1
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
        for deg in [0.0, 1.5, 3.0, 6.0]:
            th = math.radians(deg)
            xs, ys, ths = [], [], []
            # 4 rotated A x B blocks in C4 pinwheel
            # SW: [0, A]x[0, B], SE: [Lw0-B, Lw0]x[0, A], NE: [Lw0-A, Lw0]x[Lw0-B, Lw0], NW: [0, B]x[Lw0-A, Lw0]
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
            # Central core scaled cleanly into middle
            xs.extend((cx_raw - 0.5 * Lc + 0.5 * Lw0).tolist())
            ys.extend((cy_raw - 0.5 * Lc + 0.5 * Lw0).tolist())
            ths.extend(cth_raw.tolist())
            assert len(xs) == n
            seeds.append((f"pin_A{A}_B{B}_d{deg}", n, L_base, xs, ys, ths, warm_sched))

    # 3) Global small-tilt k x k lattice + 1 inserted square
    for deg in [0.8, 2.0, 4.0, 7.0]:
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
        for pos_tag, px, py, pdeg in [
            ("corner", 0.6, 0.6, 45.0),
            ("edge", 0.5 * Lw0, 0.6, 36.87),
            ("center", 0.5 * Lw0, 0.5 * Lw0, 45.0),
        ]:
            seeds.append((
                f"glob_d{deg}_{pos_tag}", n, L_base,
                xs + [px], ys + [py], ths + [math.radians(pdeg)],
                [k + 1.5, k + 0.95, k + 0.72] + warm_sched[1:],
            ))

    # 4) Macro-perturbed merged.csv incumbent
    sub = df_merged[df_merged["n"] == n]
    if len(sub) == n:
        mx = sub["x"].to_numpy()
        my = sub["y"].to_numpy()
        mth = sub["angle"].to_numpy()
        mhx = 0.5 * (np.abs(np.cos(mth)) + np.abs(np.sin(mth)))
        mx = mx - (mx - mhx).min()
        my = my - (my - mhx).min()
        til_mask = (np.abs(mth % (math.pi / 2)) > 0.05) & (np.abs((mth % (math.pi / 2)) - math.pi / 2) > 0.05)
        # Rotate tilted squares away from 45deg to 28..41deg or apply chiral quadrant twist
        for tgt_deg in [28.0, 34.0, 37.0, 41.0]:
            nth = mth.copy()
            nth[til_mask] = math.radians(tgt_deg)
            seeds.append((f"pert_til{tgt_deg}", n, L_base, mx * 1.003, my * 1.003, nth, warm_sched[2:]))
        for qdeg in [1.0, 2.5, 4.0]:
            nth = mth.copy()
            mid = 0.5 * L_base
            q1 = (~til_mask) & (mx < mid) & (my < mid)
            q2 = (~til_mask) & (mx >= mid) & (my < mid)
            q3 = (~til_mask) & (mx >= mid) & (my >= mid)
            q4 = (~til_mask) & (mx < mid) & (my >= mid)
            r = math.radians(qdeg)
            nth[q1] += r
            nth[q2] -= r
            nth[q3] += r
            nth[q4] -= r
            seeds.append((f"pert_quad{qdeg}", n, L_base, mx * 1.005, my * 1.005, nth, warm_sched[1:]))

    return seeds


def main():
    t0 = time.time()
    df_merged = pd.read_csv("poc/pell_corrugation/data/border-scan/merged.csv")
    nm_path = Path("/home/lucas/.gemini/antigravity-cli/brain/95f98074-e284-43bd-8857-8c64d023144c/scratch/idea14a_nearmisses.json")
    nm_all = json.loads(nm_path.read_text()) if nm_path.exists() else {}

    targets = [
        (401, 20, 20.535533905933),
        (442, 21, 21.535533905933),
    ]
    summary = []
    with ProcessPoolExecutor(max_workers=7, mp_context=mp.get_context("spawn")) as ex:
        for n, k, L_base in targets:
            tn = time.time()
            seeds = build_seeds(n, k, L_base, df_merged, nm_all.get(str(n), []))
            print(f"\nn={n} (L_base={L_base:.6f}): running {len(seeds)} pure L-BFGS+LP seeds...", flush=True)
            res = list(ex.map(run_seed, seeds, chunksize=1))
            feas = [r for r in res if r[4] is not None]
            feas.sort(key=lambda r: r[2])
            print(f"  feasible={len(feas)}/{len(seeds)}", flush=True)
            for tag, _, bL, lpen, _ in feas[:5]:
                print(f"    feas {tag:22s}: L={bL:.9f} (dL={bL - L_base:+.9f})", flush=True)
            if not feas:
                min_pen = min(res, key=lambda r: r[3])
                print(f"    no feas; best last_pen={min_pen[3]:.6e} ({min_pen[0]})", flush=True)

            win_L = L_base
            for tag, _, bL, _, cfg in feas[:3]:
                if bL < win_L - 1e-11:
                    ver = u.verify_official(n, cfg[0], cfg[1], cfg[2])
                    if ver["ok"] and ver["L_float"] < win_L - 1e-11:
                        win_L = ver["L_float"]
                        df_out = pd.DataFrame(ver["rows"])
                        df_out["n"] = n
                        df_out["id"] = [f"{n}_{sid}" for sid in df_out["square_id"]]
                        df_out[["id", "n", "square_id", "x", "y", "angle"]].to_csv(
                            f"poc/swarm_angles/data/cand14b_n{n}.csv", index=False
                        )
                        print(f"  *** CERTIFIED WIN n={n}: L={win_L:.12f} (< {L_base:.12f}) via {tag}! ***", flush=True)

            summary.append({
                "n": n, "L_base": L_base, "seeds": len(seeds), "feasible": len(feas),
                "best_L": feas[0][2] if feas else None, "best_tag": feas[0][0] if feas else None,
                "win_L": win_L, "seconds": round(time.time() - tn, 2),
            })

    with open("poc/swarm_angles/docs/idea14b_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nIdea 14B finished in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
