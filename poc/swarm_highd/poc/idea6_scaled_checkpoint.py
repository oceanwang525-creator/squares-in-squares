#!/usr/bin/env python3
"""Checkpointed long-running scaling of proven Idea 6 across owned & unclaimed high-gap targets."""
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
from cpp_wrap import cpp_expand_sym, full_to_free_sym, run_cpp_masked_melt
import idea1_highd_pilot as i1
import idea2_highd_sweep as i2
import idea6_structural_masked as i6

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "data"
FINAL_CSV = HERE.parent / "final.csv"
CKPT_DIR = HERE.parent / "docs" / "idea6_ckpt"
SUMMARY_JSON = HERE.parent / "docs" / "idea6_scaled_summary.json"

# Strict exclusion set: Mac/hydrogen + all other swarm workers' pools and won n outside ours.
EXCLUDED_TARGETS: set[int] = {
    # hydrogen / Mac
    634, 861, 911, 973,
    # pockets
    587, 598, 746, 747, 792, 793, 794, 802, 803, 804, 907, 908, 909,
    # angles
    401, 442, 485, 530, 577, 626, 677, 730, 785, 842, 901, 962,
    228, 340, 456, 491, 694, 748, 790, 795, 848, 852, 965, 982,
    # inflation
    333, 367, 687, 791, 796, 851, 910, 969, 971, 972,
    # bicrystal
    328, 365, 379, 492, 686, 741, 854, 913, 963, 975,
    # replace45
    147, 262, 330, 366, 448, 534, 581, 585, 847, 850, 853, 974,
    # cascades
    263, 499, 582, 593, 640, 646, 695, 738, 749, 786, 863, 902, 904, 912,
    # walldrift (#113 + #122)
    696, 680, 936, 967, 859, 692, 920,
    754, 810, 916, 997, 979, 992, 977, 799, 637, 993, 915, 925, 647, 811, 978, 806,
    # donor (#121)
    591, 736, 731, 495, 843, 678, 627, 362, 325, 290, 964,
}

REMAINING_OWNED = [805, 639, 339, 237, 269, 698]


FRONTIER_CSV = u.REPO_ROOT / "poc" / "frontier_nov12" / "docs" / "per_n_gap.csv"


def rank_unclaimed_targets(mats: dict, min_n: int = 200) -> list[int]:
    """Return unclaimed non0>0 n>=min_n targets ordered by frontier gap_score descending."""
    df_g = pd.read_csv(FRONTIER_CSV).set_index("n")
    rows: list[tuple[int, float]] = []
    for n, (_, _, th, _) in mats.items():
        if n < min_n or n in EXCLUDED_TARGETS or n in i1.OWNED_TARGETS or n not in df_g.index:
            continue
        ang = np.abs(np.degrees(th) % 90.0)
        if int(np.sum((ang > 0.5) & (ang < 89.5))) > 0:
            rows.append((n, float(df_g.loc[n, "gap_score"])))
    rows.sort(key=lambda r: -r[1])
    return [r[0] for r in rows]


def build_scaled_seeds(
    n: int,
    x_b: np.ndarray,
    y_b: np.ndarray,
    th_b: np.ndarray,
    L_b: float,
    L0_pocket: float,
    mats: dict,
    rng: np.random.Generator,
) -> tuple[list[np.ndarray], list[np.ndarray], list[list[float]], list[str]]:
    """Reuse Idea 6 seeds and add 4 extra non-teleport void seeds matching n=698 mpk_void_11."""
    q_list, m_list, ls_list, labels = i6.build_structural_seeds(n, x_b, y_b, th_b, L_b, L0_pocket, mats, rng)
    sc = L0_pocket / L_b
    xb, yb, tb = x_b * sc, y_b * sc, th_b.copy()
    voids = i1.clearance_points(xb, yb, tb, L0_pocket, n_pts=4, step=0.14)
    sched_p = [L0_pocket, L_b + 0.008, L_b - 2e-4, L_b - 0.002]
    for s in range(4):
        x, y, th = xb.copy(), yb.copy(), tb.copy()
        vx, vy, _ = voids[s % len(voids)] if voids else (0.5 * L0_pocket, 0.5 * L0_pocket, 0.1)
        rad = float(rng.uniform(4.8, 7.6))
        near = ((x - vx) ** 2 + (y - vy) ** 2) <= rad * rad
        nm = int(near.sum())
        if nm > 0:
            x[near] += rng.uniform(-0.11, 0.11, nm)
            y[near] += rng.uniform(-0.11, 0.11, nm)
            th[near] += rng.uniform(-0.24, 0.24, nm)
        q_list.append(full_to_free_sym(x, y, th, L0_pocket))
        m_list.append(i6._mask3(near))
        ls_list.append(sched_p)
        labels.append(f"mpk_voidx_{s}")
    return q_list, m_list, ls_list, labels


def solve_one(n: int, mats: dict, inc_front: dict, wins: dict, n_threads: int) -> tuple[dict, dict | None]:
    """Run scaled Idea 6 on a single allowed target n and return (summary_record, win_record_or_None)."""
    if n in EXCLUDED_TARGETS:
        raise ValueError(f"Refusing excluded target n={n}")
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
    q_list, m_list, ls_list, labels = build_scaled_seeds(n, x_b, y_b, th_b, L_b, L0_pocket, mats, rng)
    ils_iters = [5, 8, 12, 15]

    qb, qf, lf, _, po, lb = run_cpp_masked_melt(
        n, 1, q_list, m_list, ls_list, ils_iters,
        a_start=0.78, n_melt_stages=4, masked_iters=180, full_iters=240,
        n_threads=n_threads, base_seed=77000 + n,
    )
    n_feas = int(np.sum(lf < 1e9))
    best_feas = float(np.min(lf)) if n_feas else None
    print(f"\n=== Idea 6S n={n} | L_b={L_b:.9f} | N={len(labels)} feas={n_feas} best_Lf={best_feas} min_pen={float(np.min(po)):.2e} ===", flush=True)

    order = sorted(range(len(labels)), key=lambda i: (0 if lf[i] < 1e9 else 1, float(lf[i]) if lf[i] < 1e9 else float(po[i])))
    keep, used = [], set()
    for pref, qta in (("mpk_void", 3), ("mpk_wall_", 2), ("mpk_seam_", 1), ("fpeel_", 2)):
        c = 0
        for i in order:
            if i not in used and labels[i].startswith(pref):
                keep.append(i); used.add(i); c += 1
                if c >= qta or len(keep) >= 8:
                    break
    for i in order:
        if len(keep) >= 8:
            break
        if i not in used:
            keep.append(i); used.add(i)

    tasks = []
    for i in keep:
        use_f = lf[i] < 1e9
        L_ref = float(lf[i]) if use_f else ls_list[i][-1]
        x, y, th = cpp_expand_sym(n, 1, L_ref, qf[i] if use_f else qb[i])
        tasks.append((i, labels[i], n, L_b, x, y, th, 0.0 if use_f else float(po[i]), float(lf[i]) if use_f else float(lb[i])))

    with mp.Pool(processes=min(4, max(1, n_threads), len(tasks))) as pool:
        polished = pool.map(i6._polish_worker, tasks)
    polished.sort(key=lambda r: (0 if r["verify"].get("ok") else 1, float(r["verify"].get("L_float", 1e9)) if r["verify"].get("ok") else r["pen_fin"]))

    best_valid, best_win = None, None
    for r in polished:
        if r["verify"].get("ok"):
            Lf = float(r["verify"]["L_float"])
            if best_valid is None or Lf < float(best_valid["verify"]["L_float"]):
                best_valid = r
            if Lf < L_b - 1e-11 and (best_win is None or Lf < float(best_win["verify"]["L_float"])):
                best_win = r

    status = "no_valid"
    best_dL = None
    if best_valid is not None:
        v = best_valid["verify"]
        best_dL = float(v["L_float"]) - L_b
        status = "REAL_WIN" if best_dL <= -1e-9 else ("TIE" if best_dL < 0.0 else "no_win")
        print(f"  BEST VALID n={n}: {best_valid['label']} L_lit={v['L_lit'][:18]} (dL={best_dL:+.3e}, {status}) [{time.time()-t0:.1f}s]", flush=True)
    else:
        print(f"  NO VALID POLISHED n={n} [{time.time()-t0:.1f}s]", flush=True)

    rec = {
        "n": n, "L_inc": L_inc, "L_base": L_b, "L_front": L_front, "dt_s": round(time.time() - t0, 2),
        "n_starts": len(labels), "n_feas": n_feas, "best_cpp_Lf": best_feas,
        "status": status, "best_dL": best_dL,
        "top_polished": [{k: v for k, v in r.items() if k != "rows"} for r in polished],
    }
    return rec, best_win


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", type=int, nargs="*", default=None)
    ap.add_argument("--include-unclaimed", action="store_true", default=True)
    ap.add_argument("--max-targets", type=int, default=60)
    ap.add_argument("--max-misses", type=int, default=12)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--rerun", action="store_true")
    args = ap.parse_args()
    n_threads = min(4, max(1, args.threads))

    df = pd.read_csv(i1.MERGED_CSV)
    mats = {int(n): i1._norm(sub) for n, sub in df.groupby("n")}

    targets: list[int] = []
    if args.targets:
        targets.extend(args.targets)
    else:
        targets.extend(REMAINING_OWNED)
    if args.include_unclaimed:
        seen = set(targets)
        for n in rank_unclaimed_targets(mats):
            if n not in seen:
                targets.append(n); seen.add(n)
    targets = targets[: args.max_targets]

    for t in targets:
        if t in EXCLUDED_TARGETS:
            raise ValueError(f"Refusing excluded target n={t}")

    inc_front = u.load_incumbent_and_frontier(targets)
    wins = i2.load_verified_disk_wins(DATA_DIR)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    t_all = time.time()
    records: list[dict] = []
    consec_misses = 0
    real_wins: list[int] = []
    ties: list[int] = []

    print(f"Starting Idea 6 Scaled sweep: {len(targets)} targets, max_misses={args.max_misses}, threads={n_threads}", flush=True)
    print(f"First 15 targets: {targets[:15]}", flush=True)

    for idx, n in enumerate(targets, 1):
        ckpt = CKPT_DIR / f"n_{n}.json"
        if ckpt.exists() and not args.rerun:
            rec = json.loads(ckpt.read_text())
            records.append(rec)
            st = rec.get("status", "no_win")
            if st == "REAL_WIN":
                consec_misses = 0; real_wins.append(n)
            else:
                consec_misses += 1
                if st == "TIE":
                    ties.append(n)
            print(f"[{idx}/{len(targets)}] n={n} CACHED ({st}, consec_misses={consec_misses})", flush=True)
            if consec_misses >= args.max_misses:
                print(f"Stopping early on cached streak: {consec_misses} >= {args.max_misses}", flush=True)
                break
            continue

        rec, win = solve_one(n, mats, inc_front, wins, n_threads=n_threads)
        ckpt.write_text(json.dumps(rec, indent=2))
        records.append(rec)

        st = rec["status"]
        if win is not None:
            wins[n] = win
            u.write_final_csv_from_wins({n: win}, DATA_DIR / f"win_n{n}.csv")
            u.write_final_csv_from_wins(wins, FINAL_CSV)
            v = win["verify"]
            if st == "REAL_WIN":
                consec_misses = 0
                real_wins.append(n)
                print(f">>> [REAL WIN] n={n} L_lit={v['L_lit']} L_pd={v['L_pandas']} vs L_base={rec['L_base']:.12f} (dL={rec['best_dL']:+.6e}) -> saved win_n{n}.csv & final.csv", flush=True)
            else:
                consec_misses += 1
                ties.append(n)
                print(f">>> [TIE] n={n} dL={rec['best_dL']:+.3e} -> updated final.csv (consec_misses={consec_misses}/{args.max_misses})", flush=True)
        else:
            consec_misses += 1
            print(f"--- [{idx}/{len(targets)}] n={n} {st} (consec_misses={consec_misses}/{args.max_misses})", flush=True)

        SUMMARY_JSON.write_text(json.dumps({
            "completed": len(records),
            "real_wins": real_wins,
            "ties": ties,
            "consec_misses": consec_misses,
            "dt_total_s": round(time.time() - t_all, 2),
            "results": records,
        }, indent=2))

        if consec_misses >= args.max_misses:
            print(f"\nStopping cleanly: {consec_misses} consecutive non-REAL-WIN targets reached max_misses={args.max_misses}.", flush=True)
            break

    print(f"\nIdea 6 Scaled sweep finished in {time.time()-t_all:.1f}s | real_wins={real_wins} | ties={ties}", flush=True)


if __name__ == "__main__":
    main()
