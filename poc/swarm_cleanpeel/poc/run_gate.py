#!/usr/bin/env python3
"""15-minute falsification gate for Hypothesis H16 (swarm_cleanpeel): 582<-631, 692<-745, 577<-626."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import cleanpeel_lib as cp  # noqa: E402
import peel_engine as pe  # noqa: E402

GATE_PAIRS = [
    (582, 631),
    (692, 745),
    (577, 626),
]


def main():
    cp.DOCS_DIR.mkdir(parents=True, exist_ok=True)
    cp.DATA_DIR.mkdir(parents=True, exist_ok=True)
    cp.rebuild_final_csv()

    need_ns = {n for pair in GATE_PAIRS for n in pair}
    packs, best_L, best_src = cp.load_frontier(target_ns=need_ns)
    for n in sorted(need_ns):
        print(f"Frontier n={n}: L={best_L[n]:.12f} ({best_src[n]})", flush=True)

    rng = np.random.default_rng(20261116)
    results = []
    t_start = time.time()

    for n_target, n_donor in GATE_PAIRS:
        xd, yd, td, Ld = packs[n_donor]
        L_inc_dec = best_L[n_target]
        L_inc = float(L_inc_dec)
        print(f"\n=== GATE {n_target} <- {n_donor} (L_inc={L_inc:.9f}, Lt={Ld-1:.9f}, gap={L_inc-(Ld-1):+.6f}) ===", flush=True)
        seeds = pe.generate_cleanpeel_seeds(
            n_target, n_donor, xd, yd, td, Ld, L_inc, rng, max_prot=3, allow_tilted_border=False
        )
        print(f"Generated {len(seeds)} seeds: {[s[0] for s in seeds]}", flush=True)
        res = pe.solve_cleanpeel_target(
            n_target,
            seeds,
            L_inc_dec,
            threads=4,
            rng=rng,
            n_seeds_per_spec=6,
            masked_iters=200,
            full_iters=220,
            top_polish=6,
        )
        res["donor"] = n_donor
        results.append(res)
        (cp.DOCS_DIR / "gate_results.json").write_text(json.dumps(results, indent=2))

    print(f"\n=== GATE COMPLETE in {time.time() - t_start:.1f}s ===", flush=True)
    for r in results:
        print(
            f"n={r['n']}<-{r['donor']}: win={r['win']} feas={r.get('n_feas')}/{r.get('n_jobs')} "
            f"best_seen_L={r.get('best_seen_L')} dL={r.get('dL')}",
            flush=True,
        )


if __name__ == "__main__":
    main()
