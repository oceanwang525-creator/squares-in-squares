#!/usr/bin/env python3
"""30-minute self-judging gate for Hypothesis H78 (swarm_homocasc2): {909, 912, 850, 911, 852}."""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import cascade_engine2 as ce  # noqa: E402
import homocasc2_lib as hc  # noqa: E402


def _prioritize_seeds(seeds: list, max_keep: int = 24) -> list:
    if len(seeds) <= max_keep:
        return seeds

    def score(s):
        name, x, y, th, _, _ = s
        pen, _ = hc.cpp_eval_sat_pen_bbox(x, y, th)
        tier = (
            0 if "clean" in name else
            1 if "relocRep" in name else
            2 if "xcross" in name else
            3 if "seamSurg" in name else
            4 if "reloc" in name else
            5
        )
        return (tier, pen)

    return sorted(seeds, key=score)[:max_keep]


def _run_one_pair(args: tuple) -> dict:
    n_target, n_donor, xd, yd, td, Ld, L_inc_dec, layers, use_xcross, seed_int = args
    rng = np.random.default_rng(seed_int)
    L_inc = float(L_inc_dec)
    seeds = ce.generate_peel_seeds(
        n_target, n_donor, xd, yd, td, Ld, L_inc, rng, layers=layers, max_prot=18, max_rem_tilt=8
    )
    if use_xcross and layers == 1:
        seeds += ce.generate_interior_cross_seeds(n_target, n_donor, xd, yd, td, Ld, L_inc, max_seeds=6)
    seeds = _prioritize_seeds(seeds, max_keep=24)
    print(
        f"[{n_target}<-{n_donor}] L_inc={L_inc:.9f}, Lt={Ld-layers:.9f}, gap={L_inc-(Ld-layers):+.6f}, seeds={len(seeds)}",
        flush=True,
    )
    res = ce.solve_target(
        n_target,
        seeds,
        L_inc_dec,
        threads=4,
        rng=rng,
        n_seeds_per_spec=4,
        masked_iters=180,
        full_iters=220,
        top_polish=8,
        run_mechwalk=True,
    )
    res["donor"] = n_donor
    res["layers"] = layers
    return res


def _strip_pack(res: dict) -> dict:
    return {k: v for k, v in res.items() if k != "best_pack"}


def main() -> None:
    hc.DOCS_DIR.mkdir(parents=True, exist_ok=True)
    hc.DATA_DIR.mkdir(parents=True, exist_ok=True)
    hc.rebuild_final_csv()

    need_ns = {850, 852, 909, 911, 912, 970, 972, 973}
    loaded = hc.load_portfolio(need_ns, include_pr2=True)
    targets = loaded["targets"]
    donors = loaded["donors"]

    for n in sorted(need_ns):
        t_src = targets[n]["source"]
        d_src = donors[n]["source"]
        print(
            f"Frontier n={n:4d}: target_L={targets[n]['L']:.12f} ({t_src}) | donor_L={donors[n]['L']:.12f} ({d_src})",
            flush=True,
        )

    t_start = time.time()
    wave1_specs = [
        (909, 970, donors[970]["x"], donors[970]["y"], donors[970]["th"], donors[970]["L"], targets[909]["L_dec"], 1, True, 20262101),
        (912, 973, donors[973]["x"], donors[973]["y"], donors[973]["th"], donors[973]["L"], targets[912]["L_dec"], 1, True, 20262102),
        (852, 911, donors[911]["x"], donors[911]["y"], donors[911]["th"], donors[911]["L"], targets[852]["L_dec"], 1, True, 20262103),
        (911, 972, donors[972]["x"], donors[972]["y"], donors[972]["th"], donors[972]["L"], targets[911]["L_dec"], 1, True, 20262104),
    ]

    results = []
    res_by_n = {}
    with ProcessPoolExecutor(max_workers=4) as pool:
        for res in pool.map(_run_one_pair, wave1_specs):
            res_by_n[res["n"]] = res
            results.append(_strip_pack(res))
            (hc.DOCS_DIR / "gate_results.json").write_text(json.dumps(results, indent=2))

    # Wave 2: 850 <- 909 (using updated 909 if 909 improved) + 850 <- 970 (2-layer peel)
    rng850 = np.random.default_rng(20262105)
    L_inc_850 = targets[850]["L_dec"]
    seeds_850 = ce.generate_peel_seeds(
        850, 970, donors[970]["x"], donors[970]["y"], donors[970]["th"], donors[970]["L"], float(L_inc_850), rng850,
        layers=2, max_prot=18, max_rem_tilt=8,
    )
    if res_by_n[909].get("best_pack") is not None and res_by_n[909]["best_pack"][3] < donors[909]["L"]:
        x909, y909, t909, L909 = res_by_n[909]["best_pack"]
        src909 = "new_909"
    else:
        x909, y909, t909, L909 = donors[909]["x"], donors[909]["y"], donors[909]["th"], donors[909]["L"]
        src909 = "pr2_909"
    seeds_850 += ce.generate_peel_seeds(
        850, 909, x909, y909, t909, L909, float(L_inc_850), rng850, layers=1, max_prot=18, max_rem_tilt=8
    )
    seeds_850 += ce.generate_interior_cross_seeds(850, 909, x909, y909, t909, L909, float(L_inc_850), max_seeds=6)
    seeds_850 = _prioritize_seeds(seeds_850, max_keep=26)

    print(f"[850<-970+{src909}] L_inc={float(L_inc_850):.9f}, seeds={len(seeds_850)}", flush=True)
    res850 = ce.solve_target(
        850, seeds_850, L_inc_850, threads=4, rng=rng850,
        n_seeds_per_spec=4, masked_iters=180, full_iters=220, top_polish=8, run_mechwalk=True,
    )
    res850["donor"] = f"970+{src909}"
    res850["layers"] = "2/1"
    results.append(_strip_pack(res850))
    (hc.DOCS_DIR / "gate_results.json").write_text(json.dumps(results, indent=2))

    print(f"\n=== H78 GATE COMPLETE in {time.time() - t_start:.1f}s ===", flush=True)
    for r in results:
        print(
            f"n={r['n']}<-{r['donor']}: win={r['win']} feas={r.get('n_feas')}/{r.get('n_jobs')} "
            f"best_seen_L={r.get('best_seen_L')} dL={r.get('dL')}",
            flush=True,
        )


if __name__ == "__main__":
    main()
