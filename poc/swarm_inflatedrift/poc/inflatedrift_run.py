#!/usr/bin/env python3
"""Hypothesis H124 Runner: Coupled Over-Inflation / WallDrift C++ Melt -> 4-Corner PSLP -> LPSlack -> Mechwalk."""
from __future__ import annotations

import os

for _v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_v] = "1"

import argparse
import json
import math
import multiprocessing as mp
import sys
import time
from decimal import Decimal
from pathlib import Path

import numpy as np

POC_DIR = Path(__file__).resolve().parent
SWARM_DIR = POC_DIR.parent
DOCS_DIR = SWARM_DIR / "docs"
DATA_DIR = SWARM_DIR / "data"

if str(POC_DIR) not in sys.path:
    sys.path.insert(0, str(POC_DIR))

import inflatedrift_lib as il  # noqa: E402
import portfolio as pf  # noqa: E402


def _make_win_entry(n: int, label: str, ver: dict, inc: dict) -> dict:
    L_lit = Decimal(ver["L_lit"])
    L_pd = Decimal(ver["L_pandas"])
    L_worst = max(L_lit, L_pd)
    return {
        "n": int(n),
        "label": label,
        "L_lit": ver["L_lit"],
        "L_pandas": ver["L_pandas"],
        "L_inc": str(inc["L_dec"]),
        "inc_source": inc["source"],
        "dL": float(L_worst - inc["L_dec"]),
        "rows": ver["rows"],
    }


def _try_certify(
    n: int,
    label: str,
    x: np.ndarray,
    y: np.ndarray,
    th: np.ndarray,
    incumbent: dict,
    best_dec: Decimal,
    best_win: dict | None,
) -> tuple[dict, Decimal, dict | None, bool]:
    ver = pf.certify_state(n, x, y, th)
    if not ver.get("ok"):
        return ver, best_dec, best_win, False
    L_lit = Decimal(ver["L_lit"])
    L_pd = Decimal(ver["L_pandas"])
    L_worst = max(L_lit, L_pd)
    improved = L_worst < best_dec and L_lit < incumbent["L_lit"] and L_pd < incumbent["L_pandas"]
    if improved:
        best_dec = L_worst
        best_win = _make_win_entry(n, label, ver, incumbent)
        pf.save_win_locked(best_win)
    return ver, best_dec, best_win, improved


def _select_diverse(cands: list[dict], top_k: int, pen_cap: float = 0.025) -> list[dict]:
    valid = [c for c in cands if c["pen"] <= pen_cap]
    if not valid:
        valid = sorted(cands, key=lambda c: c["pen"])[:top_k]
    valid.sort(key=lambda c: (0 if c["pen"] <= 1e-8 else 1, c["L_bbox"] + 2.5 * c["pen"]))
    selected: list[dict] = []
    for c in valid:
        if len(selected) >= top_k:
            break
        # Deduplicate nearly identical L_bbox + angle fingerprint
        if any(
            abs(c["L_bbox"] - s["L_bbox"]) < 1e-6 and float(np.mean(np.abs(c["th"] - s["th"]))) < 1e-4
            for s in selected
        ):
            continue
        selected.append(c)
    return selected


def run_one(
    n: int,
    incumbent: dict,
    cpp_threads: int = 1,
    run_mechwalk: bool = True,
    pslp_iters: int = 240,
) -> dict:
    t0 = time.time()
    L_inc_dec: Decimal = incumbent["L_dec"]
    best_dec = L_inc_dec
    best_win: dict | None = None

    x_b = incumbent["x"].copy()
    y_b = incumbent["y"].copy()
    th_b = incumbent["th"].copy()
    L_b = float(incumbent["L"])
    wb = il.analyze_walls(x_b, y_b, th_b, L_b)

    print(
        f"[H124] Start n={n} (m={math.isqrt(n)}, r=+{n - math.isqrt(n)**2}) "
        f"inc={float(L_inc_dec):.9f} ({incumbent['source']}) walls={wb['counts']}",
        flush=True,
    )

    stage_summary: dict[str, dict] = {}

    # Stage 0: Direct 4-corner pslp -> full pslp -> lpslack on incumbent
    s0 = il.coupled_pslp_lpslack_polish(
        n, x_b, y_b, th_b, old_L=L_inc_dec, use_reduced=True, pslp_iters=pslp_iters, run_lpslack=True
    )
    ver0, best_dec, best_win, imp0 = _try_certify(
        n, "stage0_pslp4c_lpslack", s0["x"], s0["y"], s0["th"], incumbent, best_dec, best_win
    )
    dL0 = float(max(Decimal(ver0["L_lit"]), Decimal(ver0["L_pandas"])) - L_inc_dec) if ver0.get("ok") else 1e9
    stage_summary["stage0"] = {"L": s0["L"], "viol": s0["viol"], "dL": dL0, "imp": imp0, "trace": s0["trace"]}
    if imp0:
        x_b, y_b, th_b = ver0["x"].copy(), ver0["y"].copy(), ver0["th"].copy()
        _, _, _, L_b = pf.normalize(x_b, y_b, th_b)
    print(f"  [n={n} Stage0] L={s0['L']:.9f} dL={dL0:+.3e} imp={imp0} | {s0['trace']}", flush=True)

    # Track 1: Over-Inflation (a_start > 1.0) + Directional Wall-Drift -> 4-Corner PSLP -> LPSlack
    rng = np.random.default_rng(124000 + n)
    t1_full_tasks, t1_mpk_tasks = il.build_overinflate_drift_tasks(n, x_b, y_b, th_b, L_b, rng)
    t1_raw: list[dict] = []
    by_a: dict[float, list[dict]] = {}
    for t in t1_full_tasks:
        by_a.setdefault(t["a_start"], []).append(t)
    for a_val, grp in sorted(by_a.items()):
        res = il.cpp_batch_melt(
            [(g["x"], g["y"], g["th"]) for g in grp],
            [g["sched"] for g in grp],
            a_start=a_val,
            n_melt_stages=4,
            seam_masks=None,
            seam_iters=0,
            full_iters=240,
            n_ruin_iters=3,
            drift_xy_list=[g["drift"] for g in grp],
            drift_strengths=[g["strength"] for g in grp],
            n_threads=cpp_threads,
            base_seed=124100 + n + int(round(a_val * 1000)),
        )
        for r, g in zip(res, grp):
            r["label"] = g["label"]
            t1_raw.append(r)
    by_a_m: dict[float, list[dict]] = {}
    for t in t1_mpk_tasks:
        by_a_m.setdefault(t["a_start"], []).append(t)
    for a_val, grp in sorted(by_a_m.items()):
        res = il.cpp_batch_masked_melt(
            [(g["x"], g["y"], g["th"]) for g in grp],
            [g["mask"] for g in grp],
            [g["sched"] for g in grp],
            ils_iters_per_L=[3, 5, 7],
            a_start=a_val,
            n_melt_stages=4,
            masked_iters=140,
            full_iters=200,
            drift_xy_list=[g["drift"] for g in grp],
            drift_strengths=[g["strength"] for g in grp],
            n_threads=cpp_threads,
            base_seed=124150 + n + int(round(a_val * 1000)),
        )
        for r, g in zip(res, grp):
            r["label"] = g["label"]
            t1_raw.append(r)

    t1_sel = _select_diverse(t1_raw, top_k=3)
    t1_best_dL = 1e9
    t1_imp = False
    for cand in t1_sel:
        pol = il.coupled_pslp_lpslack_polish(
            n, cand["x"], cand["y"], cand["th"], old_L=L_inc_dec,
            use_reduced=True, pslp_iters=pslp_iters, run_lpslack=True,
        )
        ver, best_dec, best_win, imp = _try_certify(
            n, cand["label"], pol["x"], pol["y"], pol["th"], incumbent, best_dec, best_win
        )
        if ver.get("ok"):
            dL = float(max(Decimal(ver["L_lit"]), Decimal(ver["L_pandas"])) - L_inc_dec)
            t1_best_dL = min(t1_best_dL, dL)
            if imp:
                t1_imp = True
                x_b, y_b, th_b = ver["x"].copy(), ver["y"].copy(), ver["th"].copy()
                _, _, _, L_b = pf.normalize(x_b, y_b, th_b)
    stage_summary["track1_overinflate"] = {"n_raw": len(t1_raw), "n_pol": len(t1_sel), "best_dL": t1_best_dL, "imp": t1_imp}
    print(f"  [n={n} Track1 OverInflate+Drift] pol={len(t1_sel)} best_dL={t1_best_dL:+.3e} imp={t1_imp}", flush=True)

    # Track 2: Masked-Pocket + Full-Box Soft-a Melt + Wall-Drift -> 4-Corner PSLP -> LPSlack
    mpk_tasks, full_tasks = il.build_soft_walldrift_tasks(n, x_b, y_b, th_b, L_b, rng)
    t2_mpk = il.cpp_batch_masked_melt(
        [(g["x"], g["y"], g["th"]) for g in mpk_tasks],
        [g["mask"] for g in mpk_tasks],
        [g["sched"] for g in mpk_tasks],
        ils_iters_per_L=[4, 7, 10],
        a_start=0.82,
        n_melt_stages=4,
        masked_iters=150,
        full_iters=200,
        drift_xy_list=[g["drift"] for g in mpk_tasks],
        drift_strengths=[g["strength"] for g in mpk_tasks],
        n_threads=cpp_threads,
        base_seed=124200 + n,
    )
    for r, g in zip(t2_mpk, mpk_tasks):
        r["label"] = g["label"]

    t2_full = il.cpp_batch_melt(
        [(g["x"], g["y"], g["th"]) for g in full_tasks],
        [g["sched"] for g in full_tasks],
        a_start=0.88,
        n_melt_stages=4,
        seam_masks=None,
        seam_iters=0,
        full_iters=200,
        n_ruin_iters=5,
        drift_xy_list=[g["drift"] for g in full_tasks],
        drift_strengths=[g["strength"] for g in full_tasks],
        n_threads=cpp_threads,
        base_seed=124300 + n,
    )
    for r, g in zip(t2_full, full_tasks):
        r["label"] = g["label"]

    t2_sel = _select_diverse(t2_mpk + t2_full, top_k=3)
    t2_best_dL = 1e9
    t2_imp = False
    for cand in t2_sel:
        pol = il.coupled_pslp_lpslack_polish(
            n, cand["x"], cand["y"], cand["th"], old_L=L_inc_dec,
            use_reduced=True, pslp_iters=pslp_iters, run_lpslack=True,
        )
        ver, best_dec, best_win, imp = _try_certify(
            n, cand["label"], pol["x"], pol["y"], pol["th"], incumbent, best_dec, best_win
        )
        if ver.get("ok"):
            dL = float(max(Decimal(ver["L_lit"]), Decimal(ver["L_pandas"])) - L_inc_dec)
            t2_best_dL = min(t2_best_dL, dL)
            if imp:
                t2_imp = True
                x_b, y_b, th_b = ver["x"].copy(), ver["y"].copy(), ver["th"].copy()
                _, _, _, L_b = pf.normalize(x_b, y_b, th_b)
    stage_summary["track2_walldrift"] = {"n_raw": len(t2_mpk) + len(t2_full), "n_pol": len(t2_sel), "best_dL": t2_best_dL, "imp": t2_imp}
    print(f"  [n={n} Track2 WallDrift+SoftMelt] pol={len(t2_sel)} best_dL={t2_best_dL:+.3e} imp={t2_imp}", flush=True)

    # Track 3: Mechwalk soft-mode walk -> 4-corner pslp -> lpslack
    t3_best_dL = 1e9
    t3_imp = False
    t3_pol = 0
    if run_mechwalk:
        mw_cands = il.mechwalk_pslp_round(
            n, x_b, y_b, th_b, L_b, rng_seed=124400 + n, n_threads=cpp_threads, top_k_polish=4
        )
        t3_pol = len(mw_cands)
        for cand in mw_cands:
            pol = il.coupled_pslp_lpslack_polish(
                n, cand["x"], cand["y"], cand["th"], old_L=L_inc_dec,
                use_reduced=True, pslp_iters=pslp_iters, run_lpslack=True,
            )
            ver, best_dec, best_win, imp = _try_certify(
                n, cand["label"], pol["x"], pol["y"], pol["th"], incumbent, best_dec, best_win
            )
            if ver.get("ok"):
                dL = float(max(Decimal(ver["L_lit"]), Decimal(ver["L_pandas"])) - L_inc_dec)
                t3_best_dL = min(t3_best_dL, dL)
                if imp:
                    t3_imp = True
        stage_summary["track3_mechwalk"] = {"n_pol": t3_pol, "best_dL": t3_best_dL, "imp": t3_imp}
        print(f"  [n={n} Track3 Mechwalk+PSLP] pol={t3_pol} best_dL={t3_best_dL:+.3e} imp={t3_imp}", flush=True)

    elapsed = round(time.time() - t0, 2)
    win_flag = best_win is not None
    final_dL = float(best_dec - L_inc_dec)
    print(
        f"[H124] Done n={n} ({elapsed}s) | win={win_flag} best_L={float(best_dec):.12f} dL={final_dL:+.3e}",
        flush=True,
    )
    return {
        "n": int(n),
        "m": math.isqrt(n),
        "r": n - math.isqrt(n) ** 2,
        "inc_L": str(L_inc_dec),
        "inc_source": incumbent["source"],
        "best_L": str(best_dec),
        "dL": final_dL,
        "win": win_flag,
        "win_label": best_win["label"] if best_win else None,
        "elapsed_s": elapsed,
        "stages": stage_summary,
    }


def _worker_star(args: tuple) -> dict:
    return run_one(*args)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", type=int, nargs="+", default=[974, 975, 976, 916, 915, 914, 743, 459])
    ap.add_argument("--workers", type=int, default=2, help="Max 2 worker processes per CPU fairness.")
    ap.add_argument("--pslp-iters", type=int, default=160)
    ap.add_argument("--skip-mechwalk", action="store_true")
    ap.add_argument("--tag", type=str, default="gate")
    args = ap.parse_args()

    workers = max(1, min(2, args.workers))
    cpp_threads = 1 if workers == 2 else 2

    t0 = time.time()
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    portfolio = pf.load_live_portfolio(args.targets)
    missing = [n for n in args.targets if n not in portfolio]
    if missing:
        raise RuntimeError(f"Missing targets in portfolio: {missing}")

    tasks = [
        (n, portfolio[n], cpp_threads, not args.skip_mechwalk, args.pslp_iters)
        for n in args.targets
    ]

    results: list[dict] = []
    ckpt_path = DOCS_DIR / f"{args.tag}_results.json"

    if workers == 1 or len(tasks) == 1:
        for task in tasks:
            res = run_one(*task)
            results.append(res)
            ckpt_path.write_text(json.dumps({"tag": args.tag, "results": results}, indent=2), encoding="utf-8")
    else:
        with mp.Pool(processes=workers) as pool:
            for res in pool.imap_unordered(_worker_star, tasks):
                results.append(res)
                results_sorted = sorted(results, key=lambda r: r["n"])
                ckpt_path.write_text(json.dumps({"tag": args.tag, "results": results_sorted}, indent=2), encoding="utf-8")
        results.sort(key=lambda r: r["n"])

    n_wins = pf.rebuild_final_csv()
    wins = [r for r in results if r["win"]]
    print(
        f"\n[H124 Summary tag={args.tag}] {len(wins)}/{len(results)} wins in {time.time() - t0:.1f}s "
        f"(total win files exported: {n_wins})",
        flush=True,
    )
    for r in wins:
        print(f"  WIN n={r['n']}: dL={r['dL']:+.3e} ({r['win_label']})", flush=True)


if __name__ == "__main__":
    main()
