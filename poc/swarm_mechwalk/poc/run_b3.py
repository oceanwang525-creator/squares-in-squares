#!/usr/bin/env python3
"""Batch 3 (H39) runner: iterated mechanism_walk + masked melt on newly landed structural basins."""
from __future__ import annotations

import argparse
import json
import sys
from decimal import Decimal
from pathlib import Path

import numpy as np

POC_DIR = Path(__file__).resolve().parent
if str(POC_DIR) not in sys.path:
    sys.path.insert(0, str(POC_DIR))

import mechwalk_lib as ml  # noqa: E402
import run_gate  # noqa: E402
import union as auto_union  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=int, nargs="+", required=True)
    parser.add_argument("--extra-json", type=str, nargs="*", default=[])
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--beam", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--tag", type=str, default="b3")
    parser.add_argument("--out-prefix", type=str, default="b3_")
    args = parser.parse_args()

    ml.DATA_DIR.mkdir(parents=True, exist_ok=True)
    ml.DOCS_DIR.mkdir(parents=True, exist_ok=True)

    portfolio = ml.load_origin_main_portfolio(args.targets)

    for jpath in args.extra_json:
        p = Path(jpath)
        if not p.exists():
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        rows = data.get("rows") or []
        n = int(data.get("n") or len(rows))
        if n not in portfolio or len(rows) != n:
            continue
        norm_rows = [
            {
                "square_id": int(r["square_id"]),
                "x": str(r["x"]),
                "y": str(r["y"]),
                "angle": str(r["angle"]),
            }
            for r in sorted(rows, key=lambda r: int(r["square_id"]))
        ]
        L_dec = Decimal(auto_union.certify(n, norm_rows))
        if L_dec < portfolio[n]["L_dec"]:
            x = np.array([float(r["x"]) for r in norm_rows], dtype=np.float64)
            y = np.array([float(r["y"]) for r in norm_rows], dtype=np.float64)
            th = np.array([float(r["angle"]) for r in norm_rows], dtype=np.float64)
            x, y, th, L_f = ml.normalize(x, y, th)
            portfolio[n] = {
                "x": x,
                "y": y,
                "th": th,
                "L": L_f,
                "L_dec": L_dec,
                "source": str(p),
            }

    results: list[dict] = []
    wins = 0
    for n in args.targets:
        summary, win_entry = run_gate.solve_target(
            n,
            portfolio[n],
            n_rounds=args.rounds,
            beam_width=args.beam,
            top_k_polish=args.top_k,
            n_threads=args.threads,
        )
        results.append(summary)
        if win_entry is not None:
            wins += 1
            (ml.DATA_DIR / f"{args.out_prefix}win_n{n}.json").write_text(
                json.dumps(win_entry, indent=2), encoding="utf-8"
            )
        (ml.DOCS_DIR / f"{args.tag}_results.json").write_text(
            json.dumps(results, indent=2), encoding="utf-8"
        )

    print(f"\n=== H39 {args.tag.upper()} COMPLETE: {wins}/{len(args.targets)} REAL WINs ===", flush=True)


if __name__ == "__main__":
    main()
