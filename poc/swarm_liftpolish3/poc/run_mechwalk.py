#!/usr/bin/env python3
"""Run 4-round SVD mechwalk_compress + lpslack polish on target n values."""
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
import csv
import json
import multiprocessing as mp
import sys
import time
from decimal import Decimal
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "poc" / "swarm_homocasc2" / "poc"))
sys.path.insert(0, str(REPO / "poc" / "auto_union" / "poc"))
sys.path.insert(0, str(REPO / "poc" / "baseline" / "poc" / "src"))

import homocasc2_lib as hc  # noqa: E402
import union as auto_union  # noqa: E402
from squares.core import certify  # noqa: E402


def load_incumbent_map(csv_path: Path, need: set[int]) -> dict[int, tuple[list[dict[str, str]], Decimal]]:
    by: dict[int, list[dict[str, str]]] = {}
    with csv_path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            n = int(r["n"])
            if n in need:
                by.setdefault(n, []).append({
                    "square_id": r["square_id"],
                    "x": r["x"],
                    "y": r["y"],
                    "angle": r["angle"],
                })
    out: dict[int, tuple[list[dict[str, str]], Decimal]] = {}
    for n in sorted(need):
        rows = sorted(by[n], key=lambda r: int(r["square_id"]))
        L = Decimal(certify(n, rows))
        out[n] = (rows, L)
    return out


def _worker(args):
    n, rows, old_L_str, rounds, seed, out_dir = args
    t0 = time.time()
    old_L = Decimal(old_L_str)
    x, y, th, _ = hc._rows_to_arrays(rows)
    rng = np.random.default_rng(seed)
    cert, pack = hc.mechwalk_compress(n, x, y, th, old_L, rng, rounds=rounds, threads=1)
    elapsed = round(time.time() - t0, 2)
    if cert is not None and cert.get("is_real_win"):
        win_path = Path(out_dir) / f"win_n{n}.json"
        win_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "n": n,
            "label": "mechwalk_r4",
            "old_L": old_L_str,
            "L_lit": cert["L_lit"],
            "L_pd": cert["L_pd"],
            "L_kr": cert.get("L_kr", cert["L_pd"]),
            "L_float": cert["L_float"],
            "dL": cert["dL"],
            "rows": cert["rows"],
        }
        win_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return {
            "n": n,
            "status": "WIN",
            "old_L": old_L_str,
            "new_L": max(cert["L_lit"], cert["L_pd"]),
            "dL": cert["dL"],
            "elapsed_s": elapsed,
        }
    return {
        "n": n,
        "status": "NO_WIN",
        "old_L": old_L_str,
        "best_float_L": pack[3] if pack else None,
        "elapsed_s": elapsed,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(REPO / "poc/swarm_liftpolish3/data/combined_incumbent.csv"))
    ap.add_argument("--n-list", required=True)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--seed-lpslack-json", default="")
    ap.add_argument("--wins-dir", default=str(REPO / "poc/swarm_liftpolish3/data/wins"))
    ap.add_argument("--out", default=str(REPO / "poc/swarm_liftpolish3/data/mechwalk_results.json"))
    args = ap.parse_args()

    need = sorted({int(x) for x in args.n_list.split(",") if x.strip()}, reverse=True)
    inc = load_incumbent_map(Path(args.csv), set(need))

    # Optionally start from improved lpslack geometry if available
    start_rows: dict[int, list[dict[str, str]]] = {n: inc[n][0] for n in need}
    if args.seed_lpslack_json and Path(args.seed_lpslack_json).exists():
        lp_data = json.loads(Path(args.seed_lpslack_json).read_text(encoding="utf-8"))
        for rec in lp_data:
            n = int(rec["n"])
            if n in start_rows and "rows" in rec and rec.get("status") in ("WIN", "TINY_WIN"):
                # Verify against our incumbent too
                c = hc.certify_rows_against_frontier(n, rec["rows"], inc[n][1])
                if c is not None and Decimal(max(c["L_lit"], c["L_pd"])) < inc[n][1]:
                    start_rows[n] = rec["rows"]
                    inc[n] = (rec["rows"], Decimal(max(c["L_lit"], c["L_pd"])))
                    print(f"Seeded n={n} from lpslack win L={inc[n][1]}", flush=True)

    tasks = [(n, start_rows[n], str(inc[n][1]), args.rounds, 202611 + n, args.wins_dir) for n in need]
    results = []
    with mp.get_context("fork").Pool(min(args.jobs, len(tasks))) as pool:
        for res in pool.imap_unordered(_worker, tasks, chunksize=1):
            results.append(res)
            print(f"[mechwalk] n={res['n']} status={res['status']} dL={res.get('dL', 0)} t={res['elapsed_s']}s", flush=True)

    results.sort(key=lambda r: r["n"])
    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
