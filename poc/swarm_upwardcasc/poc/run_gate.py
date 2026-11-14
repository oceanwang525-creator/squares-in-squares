#!/usr/bin/env python3
"""30-minute self-judging gate for Hypothesis H98 (swarm_upwardcasc): {853<-796, 591<-544, 582<-535}."""
from __future__ import annotations

import json
import os
import sys
import time
from decimal import Decimal
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import upward_engine as ue  # noqa: E402
import upward_lib as ul  # noqa: E402


GATE_PAIRS = [
    (853, 796),
    (591, 544),
    (582, 535),
]


def run_pair(n_target: int, n_donor: int, loaded: dict, seed_int: int) -> dict:
    rng = np.random.default_rng(seed_int)
    tgt = loaded["targets"][n_target]
    don = loaded["donors"][n_donor]
    L_inc_dec = tgt["L_dec"]
    L_inc = float(L_inc_dec)
    pr_best = loaded["pr_best"][n_target]

    seeds = ue.generate_lborder_seeds(n_target, n_donor, don["x"], don["y"], don["th"], don["L"])
    seeds += ue.generate_xcross_expand_seeds(n_target, n_donor, don["x"], don["y"], don["th"], don["L"])

    print(
        f"\n=== GATE [{n_target}<-{n_donor}] L_inc={L_inc:.9f} ({tgt['source']}) | "
        f"L_don={don['L']:.9f} ({don['source']}) | pr_best={float(pr_best):.9f} | seeds={len(seeds)} ===",
        flush=True,
    )
    res = ue.solve_upward_target(
        n_target,
        seeds,
        L_inc_dec,
        threads=2,
        rng=rng,
        top_polish=8,
        run_mechwalk=True,
    )
    res["donor"] = n_donor
    res["pr_best"] = float(pr_best)
    if res["best_seen_L"] is not None:
        res["dL_vs_pr"] = float(Decimal(repr(res["best_seen_L"])) - pr_best)
    else:
        res["dL_vs_pr"] = None
    return res


def main() -> None:
    ul.DOCS_DIR.mkdir(parents=True, exist_ok=True)
    ul.DATA_DIR.mkdir(parents=True, exist_ok=True)
    ul.rebuild_final_csv()

    need_ns = {n for pair in GATE_PAIRS for n in pair}
    loaded = ul.load_portfolio(need_ns, include_prs=True)

    t0 = time.time()
    results = []
    for idx, (nt, nd) in enumerate(GATE_PAIRS):
        res = run_pair(nt, nd, loaded, seed_int=20269801 + idx)
        results.append(res)
        (ul.DOCS_DIR / "gate_results.json").write_text(json.dumps(results, indent=2))

    print(f"\n=== H98 GATE COMPLETE in {time.time() - t0:.1f}s ===", flush=True)
    for r in results:
        print(
            f"n={r['n']}<-{r['donor']}: win={r['win']} best_L={r.get('best_seen_L')} "
            f"dL_main={r.get('dL')} dL_pr={r.get('dL_vs_pr')} label={r.get('best_label')} ({r['elapsed_s']}s)",
            flush=True,
        )


if __name__ == "__main__":
    main()
