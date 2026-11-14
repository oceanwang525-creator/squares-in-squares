#!/usr/bin/env python3
"""H131 adjacent-row pass with fresh streams and incumbent-preserving export."""
from __future__ import annotations

import argparse
from decimal import Decimal
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("BLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_gate as gate  # noqa: E402

ADJACENT_TARGETS = [633, 634, 635, 696, 697, 860, 862, 863]
ADJACENT_CKPT = gate.DOCS_DIR / "adjacent_ckpt"


def upstream_geometry(n: int, puzzles: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    rows = sorted(puzzles[n], key=lambda row: int(row["square_id"]))
    x = np.array([float(row["x"]) for row in rows], dtype=np.float64)
    y = np.array([float(row["y"]) for row in rows], dtype=np.float64)
    th = np.array([float(row["angle"]) for row in rows], dtype=np.float64)
    hx = 0.5 * (np.abs(np.cos(th)) + np.abs(np.sin(th)))
    x -= float((x - hx).min())
    y -= float((y - hx).min())
    L = float(max((x + hx).max(), (y + hx).max()))
    return x, y, th, L


def baseline_pairs(n: int, puzzles: dict) -> tuple[Decimal, Decimal, Decimal, Decimal, str]:
    upstream_lit = Decimal(gate.certify(n, puzzles[n]))
    transported = gate.kaggle_roundtrip({n: puzzles[n]})
    upstream_pd = Decimal(gate.certify(n, transported[n]))

    h128_path = gate.H128_DATA / f"win_n{n}.csv"
    if not h128_path.exists():
        return upstream_lit, upstream_pd, upstream_lit, upstream_pd, "upstream-only"

    union_mod = gate._load_union_mod()
    h128_lit, h128_pd = union_mod.load(h128_path)
    result = union_mod.check((n, h128_lit[n], h128_pd[n]))
    if result is None:
        raise ValueError(f"H128 source group does not certify: n={n}")
    h_lit, h_pd = map(Decimal, result)
    return upstream_lit, upstream_pd, h_lit, h_pd, "H128-child"


def load_current_wins() -> dict[int, dict]:
    union_mod = gate._load_union_mod()
    literal, transported = union_mod.load(gate.FINAL_CSV)
    if set(literal) != set(transported):
        raise ValueError("current final does not have matching transport groups")
    wins = {}
    for n in sorted(literal):
        result = union_mod.check((n, literal[n], transported[n]))
        if result is None:
            raise ValueError(f"current final group does not certify: n={n}")
        wins[n] = {"rows": literal[n]}
    return wins


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=int, nargs="+", default=ADJACENT_TARGETS)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--masked-iters", type=int, default=240)
    parser.add_argument("--full-iters", type=int, default=240)
    parser.add_argument("--a-start", type=float, default=0.82)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--out-json", type=Path, default=gate.DOCS_DIR / "adjacent_results.json")
    args = parser.parse_args()
    threads = min(4, max(1, args.threads))

    started = time.time()
    puzzles = gate.read_submission(gate.MERGED_CSV)
    wins = load_current_wins()
    protected = sorted(wins)
    ADJACENT_CKPT.mkdir(parents=True, exist_ok=True)
    results = []

    for n in args.targets:
        checkpoint = ADJACENT_CKPT / f"n_{n}.json"
        if checkpoint.exists() and not args.rerun:
            rec = json.loads(checkpoint.read_text(encoding="utf-8"))
            if rec.get("runner") != "H131-adjacent-v1":
                checkpoint.unlink()
            else:
                results.append(rec)
                status = rec.get("status", "MISS")
                if status == "REAL_WIN":
                    win_path = gate.DATA_DIR / f"adjacent_n{n}.csv"
                    union_mod = gate._load_union_mod()
                    literal, _ = union_mod.load(win_path)
                    wins[n] = {"rows": literal[n]}
                print(
                    f"[CACHED] n={n} status={status} "
                    f"dUp={rec.get('best_dL_upstream')} dCmp={rec.get('best_dL_H128')}",
                    flush=True,
                )
                continue

        h128_path = gate.H128_DATA / f"win_n{n}.csv"
        if h128_path.exists():
            x, y, th, seed_L = gate.load_seed_geometry(n)
        else:
            x, y, th, seed_L = upstream_geometry(n, puzzles)

        upstream_lit, upstream_pd, comp_lit, comp_pd, comparison_source = baseline_pairs(
            n, puzzles
        )
        rec = gate.solve_target(
            n,
            x,
            y,
            th,
            seed_L,
            upstream_lit,
            upstream_pd,
            comp_lit,
            comp_pd,
            n_threads=threads,
            masked_iters=args.masked_iters,
            full_iters=args.full_iters,
            a_start=args.a_start,
            top_k=args.top_k,
            rng_offset=232100,
            base_seed_offset=232700,
            l_expand=0.046,
            generation="H131-adjacent-v1",
        )
        win = rec.pop("win")
        rec["runner"] = "H131-adjacent-v1"
        rec["comparison_source"] = comparison_source
        checkpoint.write_text(json.dumps(rec, indent=2), encoding="utf-8")
        results.append(rec)

        if win is not None:
            wins[n] = win
            gate.write_final_csv({n: win}, gate.DATA_DIR / f"adjacent_n{n}.csv")
        gate.write_final_csv(wins, gate.FINAL_CSV)
        args.out_json.write_text(
            json.dumps(
                {
                    "results": results,
                    "protected": protected,
                    "dt_total_s": round(time.time() - started, 2),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    gate.write_final_csv(wins, gate.FINAL_CSV)
    args.out_json.write_text(
        json.dumps(
            {
                "results": results,
                "protected": protected,
                "dt_total_s": round(time.time() - started, 2),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    real = [row["n"] for row in results if row["status"] == "REAL_WIN"]
    print(
        f"\nH131 adjacent run done in {time.time()-started:.1f}s | "
        f"REAL_WINS={real} | protected={protected}",
        flush=True,
    )


if __name__ == "__main__":
    main()
