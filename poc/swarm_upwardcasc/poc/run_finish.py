#!/usr/bin/env python3
"""Targeted follow-up against origin/main (254bb0f): re-polish 332 & 582 + updated-donor cascades."""
from __future__ import annotations

import json
import os
import sys
from decimal import Decimal
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

HERE = Path(__file__).resolve().parent
DOCS_DIR = HERE.parent / "docs"
DATA_DIR = HERE.parent / "data"
sys.path.insert(0, str(HERE))

import run_scale as rs  # noqa: E402
import upward_lib as ul  # noqa: E402


PAIRS = [
    (530, 485),
    (577, 530),
    (585, 538),
    (853, 796),
    (965, 904),
    (969, 908),
    (978, 917),
]


def main() -> None:
    need_ns = sorted({n for pair in PAIRS for n in pair} | {332, 582})
    port = ul.load_portfolio(need_ns, include_prs=False)
    targets = port["targets"]
    inc_L = {n: targets[n]["L_dec"] for n in need_ns}
    rng = np.random.default_rng(2026111499)

    # 1. Re-polish existing 332 and 582 wins
    for n_w in [332, 582]:
        w_path = DATA_DIR / f"win_n{n_w}.json"
        if not w_path.exists():
            continue
        w = json.loads(w_path.read_text(encoding="utf-8"))
        Lw_old = Decimal(w["L_lit"])
        t_w = rs._win_to_target(w)
        print(f"\n=== Re-polish n={n_w} (cur={Lw_old}) ===", flush=True)
        mc_cert, (xm, ym, tm, _) = ul.mechwalk_compress(
            n_w, t_w["x"], t_w["y"], t_w["th"], Lw_old, rng, rounds=4, threads=2
        )
        lp_cert, _ = ul.run_lpslack_polish(n_w, xm, ym, tm, inc_L[n_w], outer_iters=12)
        best_c = None
        for c in (mc_cert, lp_cert):
            if c is not None and Decimal(c["L_lit"]) < Lw_old:
                if best_c is None or Decimal(c["L_lit"]) < Decimal(best_c["L_lit"]):
                    best_c = c
        if best_c is not None:
            saved = ul.save_win_and_rebuild_final(n_w, best_c, "repolish_mech_lpslack_v2")
            print(f"  IMPROVED n={n_w}: {best_c['L_lit']} (dL={best_c['dL']:+.3e}, saved={saved})", flush=True)
        else:
            print(f"  n={n_w} unchanged at {Lw_old}", flush=True)

    # 2. Run updated-donor pairs against 254bb0f
    results = []
    for n_tgt, n_don in PAIRS:
        print(f"\n=== Targeted {n_tgt} <- {n_don} ===", flush=True)
        # If donor is 530 and we won 530, reload win_n530
        w_don = DATA_DIR / f"win_n{n_don}.json"
        if w_don.exists():
            wd = json.loads(w_don.read_text(encoding="utf-8"))
            if Decimal(wd["L_lit"]) < targets[n_don]["L_dec"]:
                targets[n_don] = rs._win_to_target(wd)
        res = rs.solve_fast_upward(
            n_tgt, n_don, targets[n_don], inc_L[n_tgt], rng, threads=2, mech_rounds=3, lpslack_iters=8
        )
        results.append(res)
        print(f"  RESULT {n_tgt}<-{n_don}: {res}", flush=True)

    (DOCS_DIR / "finish_results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print("\nFinish sweep complete.", flush=True)


if __name__ == "__main__":
    main()
