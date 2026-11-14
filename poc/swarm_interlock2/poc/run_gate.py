#!/usr/bin/env python3
"""30-Minute Gate for H101 (interlock2): 2-D Overhang kx+ky<=2, |I|=0 Pinwheel Composer."""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import csv
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

POC_DIR = Path(__file__).resolve().parent
if str(POC_DIR) not in sys.path:
    sys.path.insert(0, str(POC_DIR))

import interlock2_lib as il  # noqa: E402
import interlock2_eval as ie  # noqa: E402


def load_merged_rows() -> dict[int, list[dict[str, str]]]:
    merged_rows: dict[int, list[dict[str, str]]] = {}
    merged_csv = il.REPO / "poc" / "pell_corrugation" / "data" / "border-scan" / "merged.csv"
    with merged_csv.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            merged_rows.setdefault(int(r["n"]), []).append(
                {"square_id": r["square_id"], "x": r["x"], "y": r["y"], "angle": r["angle"]}
            )
    for n in merged_rows:
        merged_rows[n].sort(key=lambda z: int(z["square_id"]))
    return merged_rows


_G: dict = {}


def _init_worker(merged_rows, best_rows):
    _G["merged"] = merged_rows
    _G["best"] = best_rows


def _eval_zg_worker(item):
    cand, L_inc_str = item
    from decimal import Decimal
    return ie.eval_zero_gap_candidate(
        cand, _G["merged"], _G["best"], Decimal(L_inc_str), n_threads=1
    )


def main():
    t0 = time.time()
    print("=== H101 2-D Overhang Pinwheel Gate ===", flush=True)
    best_rows, inc_L = il.load_portfolio_best()
    merged_rows = il.load_merged_rows()
    scan5_path = il.DATA_DIR / "scanned_5rows.json"
    by_whg = il.load_all_pockets(best_rows, extra_scan_path=scan5_path)
    print(f"Loaded {len(by_whg)} unique (w,h,g) keys ({sum(len(v) for v in by_whg.values())} items)", flush=True)

    # Regime 1: zero_gap (D = g1 + g2 - (kx + ky))
    zg_cands = il.find_overhang_candidates(by_whg, inc_L, regime="zero_gap", dL_margin=0.004)
    print(f"\n[Regime 1: zero_gap] Found {len(zg_cands)} targets with dL_pred <= +0.004:", flush=True)
    for c in zg_cands[:15]:
        print(
            f"  n={c['n']:4d} (K={c['K']},D={c['D']}) kx={c['kx']} ky={c['ky']} "
            f"dL_pred={c['dL_pred']:+.6f} | {c['p1']['uid']} + {c['p2']['uid']}",
            flush=True,
        )

    # Regime 2: seam_touch (D = g1 + g2)
    st_cands = il.find_overhang_candidates(by_whg, inc_L, regime="seam_touch", dL_margin=0.002)
    print(f"\n[Regime 2: seam_touch] Found {len(st_cands)} targets with dL_pred <= +0.002:", flush=True)
    for c in st_cands[:15]:
        print(
            f"  n={c['n']:4d} (K={c['K']},D={c['D']}) kx={c['kx']} ky={c['ky']} "
            f"dL_pred={c['dL_pred']:+.6f} | {c['p1']['uid']} + {c['p2']['uid']}",
            flush=True,
        )

    # Select top 8 Regime 1 candidates (prefer non-trivial dL_pred > 1e-8 so we don't waste slots on exact self-rebuilds)
    zg_top = [c for c in zg_cands if abs(c["dL_pred"]) > 1e-8][:8]
    if len(zg_top) < 8:
        zg_top = zg_cands[:8]

    # Select top 6 Regime 2 candidates: 3 with kx+ky==1 and 3 with kx+ky==2
    st_k1 = [c for c in st_cands if c["kx"] + c["ky"] == 1][:3]
    st_k2 = [c for c in st_cands if c["kx"] + c["ky"] == 2][:3]
    st_top = st_k1 + st_k2

    print("\nCertifying exact incumbents for gate targets...", flush=True)
    gate_ns = sorted({c["n"] for c in zg_top + st_top})
    exact_inc = {n: il.exact_incumbent_L(n, best_rows) for n in gate_ns}
    for n in gate_ns:
        print(f"  exact inc n={n}: {exact_inc[n]}", flush=True)

    results = []
    if zg_top:
        print(f"\n=== Evaluating {len(zg_top)} Regime 1 (zero_gap) gate targets (max_workers=2) ===", flush=True)
        tasks = [(c, str(exact_inc[c["n"]])) for c in zg_top]
        with mp.get_context("fork").Pool(processes=2, initializer=_init_worker, initargs=(merged_rows, best_rows)) as pool:
            zg_res = pool.map(_eval_zg_worker, tasks)
        results.extend(zg_res)
        for r in zg_res:
            print(f"  [ZG DONE] n={r['n']} status={r['status']} dL={r.get('dL')} dt={r['dt']}s", flush=True)

    print(f"\n=== Evaluating {len(st_top)} Regime 2 (seam_touch) gate targets (n_threads=2) ===", flush=True)
    for c in st_top:
        print(f"  Running seam_touch n={c['n']} (kx={c['kx']}, ky={c['ky']}, dL_pred={c['dL_pred']:+.6f})...", flush=True)
        r = ie.eval_seam_touch_candidate(c, merged_rows, best_rows, exact_inc[c["n"]], n_threads=2)
        results.append(r)
        print(f"  [ST DONE] n={r['n']} status={r['status']} dL={r.get('dL')} dt={r['dt']}s", flush=True)

    il.DOCS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = il.DOCS_DIR / "gate_results.json"
    out_path.write_text(json.dumps({"elapsed_s": round(time.time() - t0, 1), "results": results}, indent=2))
    wins = [r for r in results if r.get("status") == "WIN"]
    print(f"\n=== GATE SUMMARY: {len(wins)}/{len(results)} WINS in {time.time()-t0:.1f}s ===", flush=True)
    for w in wins:
        print(f"  WIN n={w['n']} ({w['regime']}): dL={w['dL']:+.6e} L={w['L_lit']}", flush=True)


if __name__ == "__main__":
    main()
