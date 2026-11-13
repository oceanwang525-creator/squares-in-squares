#!/usr/bin/env python3
"""15-minute falsification gate for Hypothesis H30 (swarm_homocasc): 850<-909, 635<-686, 852<-911, 793<-909."""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from decimal import Decimal
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import cascade_engine as ce  # noqa: E402
import homocasc_lib as hc  # noqa: E402


def _run_one_pair(args: tuple) -> dict:
    n_target, n_donor, xd, yd, td, Ld, L_inc_dec, layers, use_xcross, seed_int = args
    rng = np.random.default_rng(seed_int)
    L_inc = float(L_inc_dec)
    seeds = ce.generate_peel_seeds(
        n_target, n_donor, xd, yd, td, Ld, L_inc, rng, layers=layers, max_prot=18, max_rem_tilt=2
    )
    if use_xcross and layers == 1:
        seeds += ce.generate_interior_cross_seeds(n_target, n_donor, xd, yd, td, Ld, L_inc)
    # Keep top 24 diverse seeds max so gate completes well within 15 min
    if len(seeds) > 24:
        seeds = seeds[:24]
    print(f"[{n_target}<-{n_donor}] L_inc={L_inc:.9f}, Lt={Ld-layers:.9f}, gap={L_inc-(Ld-layers):+.6f}, seeds={len(seeds)}", flush=True)
    res = ce.solve_target(
        n_target,
        seeds,
        L_inc_dec,
        threads=4,
        rng=rng,
        n_seeds_per_spec=4,
        masked_iters=200,
        full_iters=230,
        top_polish=8,
        run_mechwalk=True,
    )
    res["donor"] = n_donor
    res["layers"] = layers
    return res


def _strip_pack(res: dict) -> dict:
    return {k: v for k, v in res.items() if k != "best_pack"}


def main():
    hc.DOCS_DIR.mkdir(parents=True, exist_ok=True)
    hc.DATA_DIR.mkdir(parents=True, exist_ok=True)
    hc.rebuild_final_csv()

    need_ns = {850, 793, 909, 635, 686, 852, 911}
    loaded = hc.load_portfolio(need_ns, include_brickslp=True)
    targets = loaded["targets"]
    donors = loaded["donors"]

    for n in sorted(need_ns):
        t_src = targets[n]["source"]
        d_src = donors[n]["source"]
        print(f"Frontier n={n:4d}: target_L={targets[n]['L']:.12f} ({t_src}) | donor_L={donors[n]['L']:.12f} ({d_src})", flush=True)

    t_start = time.time()
    wave1_specs = [
        (850, 909, donors[909]["x"], donors[909]["y"], donors[909]["th"], donors[909]["L"], targets[850]["L_dec"], 1, True, 20261131),
        (635, 686, donors[686]["x"], donors[686]["y"], donors[686]["th"], donors[686]["L"], targets[635]["L_dec"], 1, True, 20261132),
        (852, 911, donors[911]["x"], donors[911]["y"], donors[911]["th"], donors[911]["L"], targets[852]["L_dec"], 1, True, 20261133),
    ]

    results = []
    res_by_n = {}
    with ProcessPoolExecutor(max_workers=3) as pool:
        for res in pool.map(_run_one_pair, wave1_specs):
            res_by_n[res["n"]] = res
            results.append(_strip_pack(res))
            (hc.DOCS_DIR / "gate_results.json").write_text(json.dumps(results, indent=2))

    # Cascade 793 <- 850 (if 850 produced a feasible basin) + direct 2-layer 793 <- 909
    rng793 = np.random.default_rng(20261134)
    L_inc_793 = targets[793]["L_dec"]
    seeds_793 = ce.generate_peel_seeds(
        793, 909, donors[909]["x"], donors[909]["y"], donors[909]["th"], donors[909]["L"], float(L_inc_793), rng793,
        layers=2, max_prot=18, max_rem_tilt=2,
    )
    if res_by_n[850].get("best_pack") is not None:
        x850, y850, t850, L850 = res_by_n[850]["best_pack"]
        seeds_793 += ce.generate_peel_seeds(
            793, 850, x850, y850, t850, L850, float(L_inc_793), rng793,
            layers=1, max_prot=18, max_rem_tilt=2,
        )
        seeds_793 += ce.generate_interior_cross_seeds(793, 850, x850, y850, t850, L850, float(L_inc_793))
    if len(seeds_793) > 28:
        seeds_793 = seeds_793[:28]

    print(f"[793<-909/850] L_inc={float(L_inc_793):.9f}, seeds={len(seeds_793)}", flush=True)
    res793 = ce.solve_target(
        793, seeds_793, L_inc_793, threads=4, rng=rng793,
        n_seeds_per_spec=4, masked_iters=200, full_iters=230, top_polish=8, run_mechwalk=True,
    )
    res793["donor"] = "909+850"
    res793["layers"] = "2/1"
    results.append(_strip_pack(res793))
    (hc.DOCS_DIR / "gate_results.json").write_text(json.dumps(results, indent=2))

    print(f"\n=== H30 GATE COMPLETE in {time.time() - t_start:.1f}s ===", flush=True)
    for r in results:
        print(
            f"n={r['n']}<-{r['donor']}: win={r['win']} feas={r.get('n_feas')}/{r.get('n_jobs')} "
            f"best_seen_L={r.get('best_seen_L')} dL={r.get('dL')}",
            flush=True,
        )


if __name__ == "__main__":
    main()
