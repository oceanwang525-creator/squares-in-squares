"""Idea 3: (A) <3 min direct multi-margin joint_xytheta_slp harvest across all 12 targets (#90)
and (B) Frozen-interior masked wall/void/seam surgery on surplus_peel & interior cross-cuts."""
from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import ctypes
import json
import math
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
SWARM_DIR = THIS_DIR.parent
REPO_ROOT = SWARM_DIR.parents[1]
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import utils as u  # noqa: E402
import idea2_peel_and_melt as i2  # noqa: E402

TARGETS_ALL = [147, 262, 330, 366, 448, 534, 581, 585, 847, 850, 853, 974]
DATA_DIR = SWARM_DIR / "data"
DOCS_DIR = SWARM_DIR / "docs"
MERGED_CSV = REPO_ROOT / "poc" / "pell_corrugation" / "data" / "border-scan" / "merged.csv"


def _harvest_one_n(n: int) -> dict:
    """Direct multi-margin joint_xytheta_slp + 4 micro-jitter starts on merged.csv[n]."""
    t0 = time.time()
    df = pd.read_csv(MERGED_CSV)
    inc_L_d, x0, y0, th0 = u.load_incumbent_exact(n, df)
    rng = np.random.default_rng(90000 + n)
    ang_jitter = math.radians(0.02)

    best_win = None
    best_cand_L = float(inc_L_d)

    for s in range(5):
        if s == 0:
            xs, ys, ts = x0.copy(), y0.copy(), th0.copy()
        else:
            xs = x0 + rng.normal(0.0, 1e-4, n)
            ys = y0 + rng.normal(0.0, 1e-4, n)
            ts = th0 + rng.normal(0.0, ang_jitter, n)

        bx, by, bth, bL, bok = u.joint_xytheta_slp(xs, ys, ts, rounds=10 if n >= 700 else 14, tr_th0=0.015, tr_xy0=0.06, margin=1e-7)
        if not bok:
            continue

        for m_lp in (1e-7, 2e-8, 5e-9):
            fx, fy, fL, fok = u.fixed_angle_lp_tighten(bx, by, bth, rounds=6, margin=m_lp)
            if not fok:
                continue
            ver = u.verify_official(n, fx, fy, bth)
            if not ver.get("ok"):
                for eps in (2e-9, 1e-8, 5e-8, 2e-7, 1e-6):
                    sc = 1.0 + eps
                    cand = u.verify_official(n, fx * sc, fy * sc, bth)
                    if cand.get("ok"):
                        ver = cand
                        break
            if ver.get("ok"):
                L_lit_d = Decimal(ver["L_lit"])
                L_pd_d = Decimal(ver["L_pandas"])
                if L_lit_d < inc_L_d and L_pd_d < inc_L_d:
                    cand_d = max(L_lit_d, L_pd_d)
                    if best_win is None or cand_d < max(Decimal(best_win["L_lit"]), Decimal(best_win["L_pandas"])):
                        best_win = ver
                        best_cand_L = float(cand_d)
                else:
                    best_cand_L = min(best_cand_L, ver["L_float"])

    dt = round(time.time() - t0, 2)
    if best_win is not None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        win_df = pd.DataFrame([{"n": n, "x": r["x"], "y": r["y"], "angle": r["angle"]} for r in best_win["rows"]])
        win_path = DATA_DIR / f"win_n{n}.csv"
        win_df.to_csv(win_path, index=False)
        u.write_final_csv_from_wins(DATA_DIR, SWARM_DIR / "final.csv")
        return {
            "n": n,
            "inc_L": str(inc_L_d),
            "win": True,
            "L_lit": best_win["L_lit"],
            "L_pandas": best_win["L_pandas"],
            "dL": float(max(Decimal(best_win["L_lit"]), Decimal(best_win["L_pandas"])) - inc_L_d),
            "dt": dt,
        }
    return {"n": n, "inc_L": str(inc_L_d), "win": False, "best_L": best_cand_L, "dt": dt}


def run_harvest(targets: list[int], n_threads: int) -> None:
    t0 = time.time()
    print(f"=== Idea 3A Direct Multi-Margin SLP Harvest on {len(targets)} targets (threads={min(4, n_threads)}) ===", flush=True)
    with ProcessPoolExecutor(max_workers=min(4, n_threads)) as ex:
        results = list(ex.map(_harvest_one_n, targets))
    wins = [r for r in results if r["win"]]
    for r in results:
        if r["win"]:
            print(f"  [WIN n={r['n']}] L_lit={r['L_lit'][:18]} vs inc={r['inc_L'][:18]} (dL={r['dL']:.3e}) dt={r['dt']}s", flush=True)
        else:
            print(f"  [no win n={r['n']}] best={r['best_L']:.9f} vs inc={r['inc_L'][:12]} dt={r['dt']}s", flush=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    out_p = DOCS_DIR / "idea3a_harvest.json"
    out_p.write_text(json.dumps({"dt": round(time.time() - t0, 2), "wins": len(wins), "results": results}, indent=2))
    print(f"Harvest complete in {time.time() - t0:.1f}s | wins={len(wins)} -> {out_p}", flush=True)


def _make_interior_dk_seeds(
    n: int, L_inc: float, mats: dict[int, tuple], rng: np.random.Generator, max_seeds: int = 8
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, str]]:
    """cascades-style interior dk cross-cut seeds returning (x, y, th, free_mask, L_box, label)."""
    k = math.isqrt(n)
    out = []
    donors = []
    for n2, (x2, y2, th2, L2) in mats.items():
        k2 = math.isqrt(n2)
        dk = k2 - k
        if not (1 <= dk <= 3):
            continue
        L_box = L2 - dk
        if not (L_inc - 0.05 <= L_box <= L_inc - 0.0005):
            continue
        f2 = L2 - k2
        if f2 >= (L_inc - k) - 0.002:
            continue
        donors.append((f2, dk, n2, x2, y2, th2, L2, L_box))
    donors.sort()

    for _, dk, n2, x, y, th, L2, L_box in donors[:3]:
        degs = np.degrees(th) % 90.0
        up = np.minimum(degs, 90.0 - degs) < 0.5
        for cx in np.linspace(3.0, L2 - dk - 3.0, 4):
            for cy in np.linspace(3.0, L2 - dk - 3.0, 4):
                rem_up = up & (((x >= cx) & (x < cx + dk)) | ((y >= cy) & (y < cy + dk)))
                best_keep = None
                best_diff = 999999
                best_wt = float(dk)
                for wt in np.linspace(max(0.5, dk - 0.35), dk + 0.65, 25):
                    rem_t = (~up) & (((x >= cx) & (x < cx + wt)) | ((y >= cy) & (y < cy + wt)))
                    keep = ~(rem_up | rem_t)
                    diff = abs(int(keep.sum()) - n)
                    if diff < best_diff:
                        best_diff = diff
                        best_keep = keep
                        best_wt = float(wt)
                        if diff == 0:
                            break
                if best_keep is None or best_diff > 5:
                    continue
                keep = best_keep.copy()
                cnt = int(keep.sum())
                if cnt > n:
                    dist = np.minimum(np.abs(x - (cx + 0.5 * best_wt)), np.abs(y - (cy + 0.5 * best_wt)))
                    cand = np.where(keep & (~up))[0]
                    if len(cand) < cnt - n:
                        cand = np.where(keep)[0]
                    keep[cand[np.argsort(dist[cand])[: cnt - n]]] = False
                elif cnt < n:
                    dist = np.minimum(np.abs(x - (cx + 0.5 * best_wt)), np.abs(y - (cy + 0.5 * best_wt)))
                    cand = np.where(~keep)[0]
                    keep[cand[np.argsort(-dist[cand])[: n - cnt]]] = True
                if int(keep.sum()) != n:
                    continue
                nx = np.where(x[keep] >= cx + 0.5 * dk, x[keep] - dk, x[keep])
                ny = np.where(y[keep] >= cy + 0.5 * dk, y[keep] - dk, y[keep])
                nth = th[keep].copy()
                fmask = ((np.abs(nx - cx) <= 3.8) | (np.abs(ny - cy) <= 3.8)).astype(np.int32)
                out.append((nx, ny, nth, fmask, L_box, f"xcut{n2}_dk{dk}_{int(cx)}_{int(cy)}"))
                if len(out) >= max_seeds:
                    return out
    return out


def solve_masked_peel_n(
    n: int, df_merged: pd.DataFrame, mats: dict[int, tuple], lib: ctypes.CDLL, n_threads: int, quick: bool
) -> dict:
    """Run Idea 3B: frozen-interior wall/void/seam surgery on surplus_peel & interior dk cuts."""
    t0 = time.time()
    inc_L_d, x_inc, y_inc, th_inc = u.load_incumbent_exact(n, df_merged)
    L_inc = float(inc_L_d)
    rng = np.random.default_rng(45300 + n)

    peels = i2.collect_surplus_peels(n, L_inc, mats)[: (5 if quick else 8)]
    xcuts = _make_interior_dk_seeds(n, L_inc, mats, rng, max_seeds=6 if quick else 10)

    seed_triples: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    fmask_list: list[np.ndarray] = []
    ls_list: list[np.ndarray] = []
    labels: list[str] = []

    for p in peels:
        x0, y0, th0, oh0, L_box, surp = p["x"], p["y"], p["th"], p["oh"], p["L"], p["surplus"]
        hx0 = 0.5 * (np.abs(np.cos(th0)) + np.abs(np.sin(th0)))
        rank = np.argsort(-oh0)
        keep_base = np.ones(len(x0), dtype=bool)
        if surp > 0:
            keep_base[rank[:surp]] = False
        xb, yb, tb, ohb, hxb = x0[keep_base], y0[keep_base], th0[keep_base], oh0[keep_base], hx0[keep_base]

        # Variant M1: Frozen interior crystal, melt ONLY squares near overhanging squares (R = 3.0, 4.5, 6.0)
        hot_idx = np.where(ohb > 1e-4)[0]
        for rad in (3.2, 4.8, 6.5):
            xm = np.clip(xb, hxb + 1e-3, L_box - hxb - 1e-3)
            ym = np.clip(yb, hxb + 1e-3, L_box - hxb - 1e-3)
            tm = tb.copy()
            near = np.zeros(n, dtype=bool)
            for hi in hot_idx:
                near |= ((xm - xm[hi]) ** 2 + (ym - ym[hi]) ** 2) <= rad * rad
            # Also include wall strips where overhanging squares live
            if len(hot_idx) == 0:
                near |= (xm < 2.0) | (ym < 2.0) | (xm > L_box - 2.0) | (ym > L_box - 2.0)
            nm = int(near.sum())
            if nm > 0 and rad > 3.5:
                xm[near] = np.clip(xm[near] + rng.uniform(-0.08, 0.08, nm), hxb[near] + 1e-3, L_box - hxb[near] - 1e-3)
                ym[near] = np.clip(ym[near] + rng.uniform(-0.08, 0.08, nm), hxb[near] + 1e-3, L_box - hxb[near] - 1e-3)
                tm[near] += rng.uniform(-0.07, 0.07, nm)
            seed_triples.append((xm, ym, tm))
            fmask_list.append(near.astype(np.int32))
            L_try0 = min(L_inc + 0.015, L_box + min(0.08, p["max_oh_rem"] * 0.5 + 0.01))
            ls_list.append(np.array([L_try0, L_inc - 2e-4, max(L_box + 5e-4, L_inc - 0.005)], dtype=np.float64))
            labels.append(f"m1_{p['src']}_r{int(rad)}")

        # Variant M2: Drop the remaining overhanging squares (<= 6), insert into max-clearance interior voids, melt ONLY void pockets
        n_bad = len(hot_idx)
        if 1 <= n_bad <= 6:
            good = np.ones(n, dtype=bool)
            good[hot_idx] = False
            xg, yg, tg = xb[good].copy(), yb[good].copy(), tb[good].copy()
            pts = i2.clearance_points(xg, yg, tg, L_box, n_pts=max(6, n_bad + 2), step=0.14)
            t_ang = i2.donor_tilt(tg)
            for s_rep in range(3):
                xs, ys, ts = xg.copy(), yg.copy(), tg.copy()
                near = np.zeros(len(xs), dtype=bool)
                new_flags = []
                for j in range(n_bad):
                    px, py, _ = pts[(j + s_rep) % len(pts)]
                    xs = np.append(xs, px + float(rng.uniform(-0.04, 0.04)))
                    ys = np.append(ys, py + float(rng.uniform(-0.04, 0.04)))
                    ang_pick = [0.0, t_ang, tg[hot_idx[j]]][(j + s_rep) % 3]
                    ts = np.append(ts, ang_pick + float(rng.uniform(-0.04, 0.04)))
                    new_flags.append((px, py))
                near_full = np.zeros(n, dtype=bool)
                near_full[len(xg) :] = True
                for px, py in new_flags:
                    near_full |= ((xs - px) ** 2 + (ys - py) ** 2) <= (3.5 + 1.0 * s_rep) ** 2
                nm = int(near_full.sum())
                if s_rep > 0:
                    xs[near_full] += rng.uniform(-0.09, 0.09, nm)
                    ys[near_full] += rng.uniform(-0.09, 0.09, nm)
                    ts[near_full] += rng.uniform(-0.08, 0.08, nm)
                seed_triples.append((xs, ys, ts))
                fmask_list.append(near_full.astype(np.int32))
                ls_list.append(np.array([L_inc - 2e-4, max(L_box + 5e-4, L_inc - 0.004), L_box + 1e-4], dtype=np.float64))
                labels.append(f"m2_{p['src']}_v{s_rep}")

    # Variant M3: Interior dk cross-cuts with seam free_mask
    for nx, ny, nth, fmask, L_box, lbl in xcuts:
        seed_triples.append((nx, ny, nth))
        fmask_list.append(fmask)
        ls_list.append(np.array([L_box + 0.03, L_inc - 2e-4, max(L_box + 5e-4, L_inc - 0.004)], dtype=np.float64))
        labels.append(lbl)

    if not seed_triples:
        return {"n": n, "inc_L": str(inc_L_d), "dt": 0.0, "batch": 0, "win": None}

    batch = len(seed_triples)
    z_batch = np.zeros((batch, 3 * n), dtype=np.float64)
    fmask_batch = np.zeros((batch, n), dtype=np.int32)
    for b, ((xs, ys, ts), fm) in enumerate(zip(seed_triples, fmask_list)):
        z_batch[b, :n] = xs
        z_batch[b, n : 2 * n] = ys
        z_batch[b, 2 * n :] = ts
        fmask_batch[b, :] = fm
    z_feas = np.zeros_like(z_batch)
    L_sched_batch = np.ascontiguousarray(np.vstack(ls_list), dtype=np.float64)

    # Phase 1: Masked local soft-a melt holding interior crystal frozen
    a_sched1 = np.array([0.82, 0.90, 0.96, 1.0], dtype=np.float64)
    iters1 = np.array([100, 110, 130, 180], dtype=np.int32)
    out_best_L_feas = np.full(batch, 1e18, dtype=np.float64)
    out_E = np.zeros(batch, dtype=np.float64)
    out_pen = np.zeros(batch, dtype=np.float64)
    out_L_bbox = np.zeros(batch, dtype=np.float64)

    print(f"\n=== Idea 3B n={n} | inc={L_inc:.9f} | batch={batch} (peels={len(peels)}, xcuts={len(xcuts)}) ===", flush=True)
    lib.batch_melt_solve(
        batch,
        n,
        z_batch.ravel(),
        z_feas.ravel(),
        L_sched_batch.shape[1],
        L_sched_batch.ravel(),
        len(a_sched1),
        a_sched1,
        iters1,
        4,
        fmask_batch.ctypes.data_as(ctypes.c_void_p),
        min(4, n_threads),
        453000 + n,
        out_best_L_feas,
        out_E,
        out_pen,
        out_L_bbox,
    )

    # Phase 2: Short unmasked global settle on the promising basins (holding near L_inc - 2e-4)
    n_feas1 = int(np.sum(out_best_L_feas < 1e9))
    print(f"  [Phase1 masked] feas={n_feas1}/{batch} min_pen={float(np.min(out_pen)):.3e} ({labels[int(np.argmin(out_pen))]})", flush=True)

    # Polish top k_keep candidates via joint_xytheta_slp (now using C++ relax + time_limit=10s!)
    order = sorted(range(batch), key=lambda i: (0 if out_best_L_feas[i] < 1e9 else 1, float(out_best_L_feas[i]) if out_best_L_feas[i] < 1e9 else float(out_pen[i]), float(out_L_bbox[i])))
    k_keep = min(batch, 8 if quick else 12)
    tasks = []
    for i in order[:k_keep]:
        use_f = out_best_L_feas[i] < 1e9
        src_z = z_feas[i] if use_f else z_batch[i]
        tasks.append((
            i,
            labels[i],
            "masked_peel",
            n,
            src_z[:n].copy(),
            src_z[n : 2 * n].copy(),
            src_z[2 * n :].copy(),
            float(out_E[i]),
            0.0 if use_f else float(out_pen[i]),
            float(out_best_L_feas[i]) if use_f else float(out_L_bbox[i]),
        ))

    if min(4, n_threads) > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=min(4, n_threads)) as ex:
            polished = list(ex.map(i2._polish_and_repair_worker, tasks))
    else:
        polished = [i2._polish_and_repair_worker(t) for t in tasks]

    best_win = None
    best_near = None
    for r in polished:
        ver = r["verify"]
        if ver.get("ok"):
            L_lit_d = Decimal(ver["L_lit"])
            L_pd_d = Decimal(ver["L_pandas"])
            print(
                f"   [POLISH OK] {r['label']}: L_lit={ver['L_lit'][:16]} non45={r['non45_non0']} n45={r['n45']} dt={r['dt']}s",
                flush=True,
            )
            if L_lit_d < inc_L_d and L_pd_d < inc_L_d:
                cand_d = max(L_lit_d, L_pd_d)
                if best_win is None or cand_d < max(Decimal(best_win["verify"]["L_lit"]), Decimal(best_win["verify"]["L_pandas"])):
                    best_win = r
        else:
            if best_near is None or (r["pen_fin"], r["L_fin"]) < (best_near["pen_fin"], best_near["L_fin"]):
                best_near = r

    if best_win is not None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        win_df = pd.DataFrame([{"n": n, "x": r["x"], "y": r["y"], "angle": r["angle"]} for r in best_win["rows"]])
        win_path = DATA_DIR / f"win_n{n}.csv"
        win_df.to_csv(win_path, index=False)
        u.write_final_csv_from_wins(DATA_DIR, SWARM_DIR / "final.csv")
        print(f"  >>> WIN CERTIFIED n={n}: {best_win['verify']['L_lit']} < inc {inc_L_d} -> {win_path}", flush=True)
    elif best_near is not None:
        print(f"  [NO WIN n={n}] best_near={best_near['label']} L={best_near['L_fin']:.6f} pen={best_near['pen_fin']:.3e}", flush=True)

    return {
        "n": n,
        "inc_L": str(inc_L_d),
        "dt": round(time.time() - t0, 2),
        "batch": batch,
        "phase1_feas": n_feas1,
        "win": None if best_win is None else {"label": best_win["label"], "L_lit": best_win["verify"]["L_lit"], "L_pandas": best_win["verify"]["L_pandas"], "non45_non0": best_win["non45_non0"]},
        "best_near": None if best_near is None else {"label": best_near["label"], "L_fin": best_near["L_fin"], "pen_fin": best_near["pen_fin"]},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["harvest", "masked_peel"], required=True)
    parser.add_argument("--targets", type=int, nargs="+", default=TARGETS_ALL)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--tag", type=str, default="run1")
    args = parser.parse_args()

    if args.mode == "harvest":
        run_harvest(args.targets, args.threads)
        return

    lib = i2.ensure_cpp_lib()
    df_merged, mats = i2.load_merged_mats()
    results = [solve_masked_peel_n(n, df_merged, mats, lib, min(4, args.threads), args.quick) for n in args.targets]
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = DOCS_DIR / f"idea3b_{args.tag}.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\nSaved -> {out_json}", flush=True)


if __name__ == "__main__":
    main()
