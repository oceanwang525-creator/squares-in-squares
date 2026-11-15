#!/usr/bin/env python3
"""Round 2 (H118 watchcasc): homocasc2 (peel + xcross + C++ melt + mechwalk) + pslp.squeeze from public_watch_nov14 + row_ratchet_980 donors."""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import os
import sys
import time
from decimal import Decimal
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "poc" / "swarm_homocasc2" / "poc"))
sys.path.insert(0, str(REPO / "poc" / "row_ratchet_980" / "poc"))
sys.path.insert(0, str(REPO / "poc" / "baseline" / "poc" / "src"))

import homocasc2_lib as hc  # noqa: E402
import cascade_engine2 as ce2  # noqa: E402
import pslp  # noqa: E402
from squares.core import certify  # noqa: E402

hc.save_win_and_rebuild_final = lambda *a, **kw: None
hc.rebuild_final_csv = lambda *a, **kw: None

WINS_DIR = REPO / "poc" / "swarm_liftpolish3" / "data" / "wins_r2"
REPORT_PATH = REPO / "poc" / "swarm_liftpolish3" / "docs" / "r2_watchcasc_results.json"

# Priority pairs from origin/main 1f337f9 (public_watch_nov14 + row_ratchet_980)
# Format: (n_dst, n_src, layers)
DEFAULT_PAIRS = [
    # #586 r=+8/+7 targets
    (908, 969, 1),
    (736, 791, 1),
    (849, 969, 2),
    (683, 791, 2),
    # #586 r=+16/+15 targets
    (916, 977, 1),
    (799, 856, 1),
    (857, 977, 2),
    (744, 856, 2),
    # row_ratchet_980 r=+19 family
    (919, 980, 1),
    (860, 980, 2),
    # Top remaining dm=-1 pairs from 86 new rows
    (790, 847, 1),
    (854, 913, 1),
    (858, 917, 1),
    (859, 918, 1),
    (930, 991, 1),
    (926, 987, 1),
    (922, 983, 1),
    (793, 850, 1),
    (412, 453, 1),
    (935, 996, 1),
    (875, 934, 1),
    (816, 873, 1),
    (817, 874, 1),
    (650, 701, 1),
]


def select_diverse_seeds(seeds: list, max_keep: int = 12) -> list:
    """Keep top seeds per family (BL/TL/BR/TR/xcross) sorted by initial overlap penalty."""
    scored = []
    for s in seeds:
        name, x, y, th, _, _ = s
        pen, _ = hc.cpp_eval_sat_pen_bbox(x, y, th)
        fam = "xcross" if "xcross" in name else next(
            (c for c in ("_BL_", "_TL_", "_BR_", "_TR_") if c in name), "other"
        )
        scored.append((pen, fam, s))
    scored.sort(key=lambda t: t[0])
    chosen = []
    counts: dict[str, int] = {}
    used = set()
    for idx, (pen, fam, s) in enumerate(scored):
        if counts.get(fam, 0) < 3:
            chosen.append(s)
            counts[fam] = counts.get(fam, 0) + 1
            used.add(idx)
            if len(chosen) >= max_keep:
                break
    for idx, (_, _, s) in enumerate(scored):
        if len(chosen) >= max_keep:
            break
        if idx not in used:
            chosen.append(s)
    return chosen


def save_win(n_target: int, cert: dict, label: str) -> None:
    WINS_DIR.mkdir(parents=True, exist_ok=True)
    out_p = WINS_DIR / f"win_n{n_target}.json"
    if out_p.exists():
        try:
            old = json.loads(out_p.read_text(encoding="utf-8"))
            if Decimal(old["L_lit"]) <= Decimal(cert["L_lit"]):
                return
        except Exception:
            pass
    payload = {
        "n": n_target,
        "label": label,
        "L_lit": cert["L_lit"],
        "L_pan": cert["L_pan"],
        "L_inc": cert["L_inc"],
        "dL": cert["dL"],
        "rows": cert["rows"],
    }
    out_p.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_one_pair(task: tuple[int, int, int, np.ndarray, np.ndarray, np.ndarray, float, Decimal]) -> dict:
    n_dst, n_src, layers, xd, yd, td, Ld, L_inc_dec = task
    t0 = time.time()
    hc.save_win_and_rebuild_final = lambda *a, **kw: None
    hc.rebuild_final_csv = lambda *a, **kw: None
    L_inc = float(L_inc_dec)
    rng = np.random.default_rng(20261114 + n_dst * 1009 + n_src)

    seeds = ce2.generate_peel_seeds(
        n_dst, n_src, xd, yd, td, Ld, L_inc, rng,
        layers=layers, max_prot=20, max_rem_tilt=10,
    )
    if layers == 1:
        seeds += ce2.generate_interior_cross_seeds(
            n_dst, n_src, xd, yd, td, Ld, L_inc, max_seeds=6,
        )

    if not seeds:
        return {
            "n_dst": n_dst, "n_src": n_src, "layers": layers,
            "status": "NO_SEEDS", "L_inc": str(L_inc_dec),
            "elapsed_s": round(time.time() - t0, 1),
        }

    seeds = select_diverse_seeds(seeds, max_keep=12)
    res = ce2.solve_target(
        n_dst, seeds, L_inc_dec, threads=1, rng=rng,
        n_seeds_per_spec=3, masked_iters=160, full_iters=200,
        top_polish=4, run_mechwalk=False,
    )

    best_cert = None
    best_label = res.get("best_label")
    best_seen_L = res.get("best_seen_L")
    if res.get("win") and res.get("L_win") is not None:
        bx, by, bt, _ = res["best_pack"]
        c0, _ = hc.polish_candidate(n_dst, bx, by, bt, L_inc_dec, use_lpslack=False)
        if c0 and c0["is_real_win"]:
            best_cert = c0
            save_win(n_dst, best_cert, best_label or "homocasc2")

    # Apply mechwalk + pslp.squeeze only on promising basins (< L_inc + 0.008)
    bp = res.get("best_pack")
    pslp_dL = None
    if bp is not None and best_seen_L is not None and best_seen_L < L_inc + 0.008:
        bx, by, bt, _ = bp
        mw_cert, mw_pack = hc.mechwalk_compress(n_dst, bx, by, bt, L_inc_dec, rng, rounds=3, threads=1)
        if mw_pack is not None and mw_pack[3] < best_seen_L:
            bp = mw_pack
            best_seen_L = mw_pack[3]
        if mw_cert is not None and mw_cert["is_real_win"]:
            if best_cert is None or Decimal(mw_cert["L_lit"]) < Decimal(best_cert["L_lit"]):
                best_cert = mw_cert
                best_label = f"{best_label or 'basin'}_mw"
                save_win(n_dst, best_cert, best_label)

        bx, by, bt, _ = bp
        xs, ys, ts, L_norm = hc.norm_pack(bx, by, bt)
        K = int(math.floor(L_norm))
        f0 = L_norm - K
        try:
            f_sq, Xn, Yn, Tn, vn, it = pslp.squeeze(xs, ys, ts, K, K, f0, max_it=300)
            if vn <= 1e-11 and (K + f_sq) < L_inc - 1e-9:
                c_sq, _ = hc.polish_candidate(n_dst, Xn, Yn, Tn, L_inc_dec, use_lpslack=True)
                if c_sq is not None:
                    pslp_dL = c_sq["dL"]
                    if c_sq["is_real_win"] and (best_cert is None or Decimal(c_sq["L_lit"]) < Decimal(best_cert["L_lit"])):
                        best_cert = c_sq
                        best_label = f"{best_label or 'basin'}_pslp"
                        save_win(n_dst, best_cert, best_label)
        except Exception as exc:
            print(f"[{n_dst}<-{n_src}] pslp error: {exc}", flush=True)

    return {
        "n_dst": n_dst,
        "n_src": n_src,
        "layers": layers,
        "status": "WIN" if best_cert else "NO_WIN",
        "L_inc": str(L_inc_dec),
        "L_win": best_cert["L_lit"] if best_cert else None,
        "dL": best_cert["dL"] if best_cert else res.get("dL"),
        "pslp_dL": pslp_dL,
        "best_label": best_label,
        "n_seeds": res["n_seeds"],
        "n_feas": res["n_feas"],
        "elapsed_s": round(time.time() - t0, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    pairs = DEFAULT_PAIRS[: args.limit] if args.limit > 0 else DEFAULT_PAIRS
    all_ns = {n for nd, ns, _ in pairs for n in (nd, ns)}
    t_load = time.time()
    by: dict[int, list[dict[str, str]]] = {}
    merged_csv = REPO / "poc" / "pell_corrugation" / "data" / "border-scan" / "merged.csv"
    with merged_csv.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            n = int(r["n"])
            if n in all_ns:
                by.setdefault(n, []).append({
                    "square_id": r["square_id"],
                    "x": r["x"],
                    "y": r["y"],
                    "angle": r["angle"],
                })
    targets_dec = {}
    donors_arr = {}
    for n in sorted(all_ns):
        rows = sorted(by[n], key=lambda r: int(r["square_id"]))
        targets_dec[n] = Decimal(certify(n, rows))
        x = np.array([float(r["x"]) for r in rows], dtype=np.float64)
        y = np.array([float(r["y"]) for r in rows], dtype=np.float64)
        th = np.array([float(r["angle"]) for r in rows], dtype=np.float64)
        xn, yn, tn, Ln = hc.norm_pack(x, y, th)
        donors_arr[n] = (xn, yn, tn, Ln)
    print(f"Loaded {len(all_ns)} groups from merged.csv in {time.time()-t_load:.1f}s", flush=True)
    tasks = [
        (nd, ns, layers, *donors_arr[ns], targets_dec[nd])
        for nd, ns, layers in pairs
    ]
    results = []
    t0 = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(2, args.workers)) as ex:
        futs = {ex.submit(run_one_pair, t): (t[0], t[1]) for t in tasks}
        for fut in concurrent.futures.as_completed(futs):
            r = fut.result()
            results.append(r)
            print(
                f"DONE {r['n_dst']}<-{r['n_src']} (L{r['layers']}): {r['status']} "
                f"feas={r.get('n_feas')} dL={r.get('dL')} ({r['elapsed_s']}s)",
                flush=True,
            )
            REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
            REPORT_PATH.write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")

    wins = [r for r in results if r["status"] == "WIN"]
    print(f"Finished {len(results)} pairs in {time.time()-t0:.1f}s: {len(wins)} WIN(s)", flush=True)


if __name__ == "__main__":
    main()
