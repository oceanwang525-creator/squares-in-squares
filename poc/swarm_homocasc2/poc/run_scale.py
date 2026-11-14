#!/usr/bin/env python3
"""H78 Scale-Up Runner across constant-r homology cascade targets (r=+9,+10,+11,+12)."""
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
            1 if "ext" in name else
            2 if "relocRep" in name else
            3 if "xcross" in name else
            4 if "seamSurg" in name else
            5 if "reloc" in name else
            6
        )
        return (tier, pen)

    return sorted(seeds, key=score)[:max_keep]


def _run_job(spec: tuple) -> dict:
    mode, n_target, donor_pairs, L_inc_dec, seed_int = spec
    rng = np.random.default_rng(seed_int)
    L_inc = float(L_inc_dec)
    seeds: list = []

    for n_donor, xd, yd, td, Ld, layers, mkind in donor_pairs:
        if mkind == "upward":
            for tag, ex, ey, et, eL in ce.extend_one_layer_all_corners(xd, yd, td, Ld):
                if len(ex) == n_target:
                    seeds.append((f"{tag}_{n_target}_from{n_donor}", ex, ey, et, min(eL, L_inc - 1e-4), np.array([[0.5 * eL, 0.5 * eL]])))
        else:
            seeds += ce.generate_peel_seeds(
                n_target, n_donor, xd, yd, td, Ld, L_inc, rng, layers=layers, max_prot=18, max_rem_tilt=8
            )
            if layers == 1:
                seeds += ce.generate_interior_cross_seeds(n_target, n_donor, xd, yd, td, Ld, L_inc, max_seeds=6)

    seeds = _prioritize_seeds(seeds, max_keep=24)
    donor_desc = "+".join(f"{p[0]}(L{p[5]})" for p in donor_pairs)
    print(f"[SCALE {n_target}<-{donor_desc} ({mode})] L_inc={L_inc:.9f}, seeds={len(seeds)}", flush=True)
    if not seeds:
        return {"n": n_target, "donor": donor_desc, "mode": mode, "win": False, "n_seeds": 0, "n_jobs": 0, "n_feas": 0}

    res = ce.solve_target(
        n_target,
        seeds,
        L_inc_dec,
        threads=4,
        rng=rng,
        n_seeds_per_spec=1 if mode == "upward" else 4,
        masked_iters=80 if mode == "upward" else 180,
        full_iters=120 if mode == "upward" else 220,
        top_polish=10,
        run_mechwalk=True,
    )
    res["donor"] = donor_desc
    res["mode"] = mode
    return res


def _strip(res: dict) -> dict:
    return {k: v for k, v in res.items() if k != "best_pack"}


def main() -> None:
    hc.DOCS_DIR.mkdir(parents=True, exist_ok=True)
    hc.DATA_DIR.mkdir(parents=True, exist_ok=True)
    hc.rebuild_final_csv()

    all_ns = {
        685, 688, 738, 740, 741, 793, 794, 795, 796, 797,
        850, 851, 852, 853, 854, 855, 909, 910, 911, 912, 913, 914,
        970, 971, 972, 973, 975,
    }
    loaded = hc.load_portfolio(all_ns, include_pr2=True)
    targets = loaded["targets"]
    donors = loaded["donors"]

    for n in sorted(all_ns):
        print(f"n={n:4d}: target_L={targets[n]['L']:.12f} ({targets[n]['source']}) | donor_L={donors[n]['L']:.12f} ({donors[n]['source']})", flush=True)

    t0 = time.time()

    def d_tup(n_d: int, layers: int, mkind: str) -> tuple:
        d = donors[n_d]
        return (n_d, d["x"], d["y"], d["th"], d["L"], layers, mkind)

    # Wave 1: independent & gate-cascaded high-leverage pairs
    wave1 = [
        ("upward", 853, [d_tup(796, 1, "upward")], targets[853]["L_dec"], 20263101),
        ("peel", 795, [d_tup(852, 1, "peel"), d_tup(911, 2, "peel")], targets[795]["L_dec"], 20263102),
        ("peel", 852, [d_tup(911, 1, "peel")], targets[852]["L_dec"], 20263103),
        ("peel", 738, [d_tup(793, 1, "peel")], targets[738]["L_dec"], 20263104),
        ("peel", 688, [d_tup(741, 1, "peel")], targets[688]["L_dec"], 20263105),
        ("peel", 793, [d_tup(850, 1, "peel"), d_tup(909, 2, "peel")], targets[793]["L_dec"], 20263106),
        ("peel", 685, [d_tup(738, 1, "peel"), d_tup(793, 2, "peel")], targets[685]["L_dec"], 20263107),
        ("peel", 912, [d_tup(973, 1, "peel")], targets[912]["L_dec"], 20263108),
        ("peel", 909, [d_tup(970, 1, "peel")], targets[909]["L_dec"], 20263109),
        ("peel", 741, [d_tup(796, 1, "peel")], targets[741]["L_dec"], 20263110),
        ("peel", 794, [d_tup(851, 1, "peel")], targets[794]["L_dec"], 20263111),
        ("peel", 910, [d_tup(971, 1, "peel")], targets[910]["L_dec"], 20263112),
        ("peel", 854, [d_tup(913, 1, "peel")], targets[854]["L_dec"], 20263113),
        ("peel", 914, [d_tup(975, 1, "peel")], targets[914]["L_dec"], 20263114),
        ("peel", 797, [d_tup(854, 1, "peel")], targets[797]["L_dec"], 20263115),
        ("peel", 855, [d_tup(914, 1, "peel")], targets[855]["L_dec"], 20263116),
    ]

    results = []
    with ProcessPoolExecutor(max_workers=2) as pool:
        for res in pool.map(_run_job, wave1):
            results.append(_strip(res))
            (hc.DOCS_DIR / "scale_results.json").write_text(json.dumps(results, indent=2))

    # Reload portfolio + local wins for Wave 2 cascades
    loaded = hc.load_portfolio(all_ns, include_pr2=True)
    targets = loaded["targets"]
    donors = loaded["donors"]
    wave2 = []
    if "homocasc2" in donors[912]["source"] and donors[912]["L"] < targets[853]["L"] + 1.0:
        wave2.append(("peel", 853, [d_tup(912, 1, "peel")], targets[853]["L_dec"], 20263201))
    if "homocasc2" in donors[912]["source"] and donors[912]["L"] + 1.0 < donors[973]["L"]:
        wave2.append(("upward", 973, [d_tup(912, 1, "upward")], targets[973]["L_dec"], 20263202))
    if "homocasc2" in donors[793]["source"]:
        wave2.append(("peel", 738, [d_tup(793, 1, "peel")], targets[738]["L_dec"], 20263203))
    if "homocasc2" in donors[738]["source"]:
        wave2.append(("peel", 685, [d_tup(738, 1, "peel")], targets[685]["L_dec"], 20263204))
    if "homocasc2" in donors[795]["source"] and donors[795]["L"] - 1.0 < targets[740]["L"]:
        wave2.append(("peel", 740, [d_tup(795, 1, "peel")], targets[740]["L_dec"], 20263205))
    if "homocasc2" in donors[853]["source"] and donors[853]["L"] + 1.0 < targets[912]["L"]:
        wave2.append(("upward", 912, [d_tup(853, 1, "upward")], targets[912]["L_dec"], 20263206))

    if wave2:
        with ProcessPoolExecutor(max_workers=2) as pool:
            for res in pool.map(_run_job, wave2):
                results.append(_strip(res))
                (hc.DOCS_DIR / "scale_results.json").write_text(json.dumps(results, indent=2))

    print(f"\n=== H78 SCALE COMPLETE in {time.time() - t0:.1f}s ===", flush=True)
    for r in results:
        print(
            f"n={r['n']}<-{r['donor']} ({r['mode']}): win={r['win']} feas={r.get('n_feas')}/{r.get('n_jobs')} "
            f"best_seen_L={r.get('best_seen_L')} dL={r.get('dL')}",
            flush=True,
        )


if __name__ == "__main__":
    main()
