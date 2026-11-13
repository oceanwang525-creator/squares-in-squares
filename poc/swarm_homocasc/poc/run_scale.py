#!/usr/bin/env python3
"""H30 Scale-Up Runner across cascaded and brickslp-enabled homology pairs."""
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


def _load_best_known(ns: set[int]) -> tuple[dict[int, dict], dict[int, dict]]:
    loaded = hc.load_portfolio(ns, include_brickslp=True)
    targets = loaded["targets"]
    donors = loaded["donors"]
    for wp in sorted(hc.DATA_DIR.glob("win_n*.json")):
        d = json.loads(wp.read_text())
        n = int(d["n"])
        if n not in ns:
            continue
        L_w = max(Decimal(d["L_lit"]), Decimal(d.get("L_pd", d["L_lit"])))
        rows = sorted(d["rows"], key=lambda r: int(r["square_id"]))
        x = np.array([float(r["x"]) for r in rows], dtype=np.float64)
        y = np.array([float(r["y"]) for r in rows], dtype=np.float64)
        th = np.array([float(r["angle"]) for r in rows], dtype=np.float64)
        x, y, th, L_f = hc.norm_pack(x, y, th)
        if n not in donors or L_w < donors[n]["L_dec"]:
            donors[n] = {"x": x, "y": y, "th": th, "L": L_f, "L_dec": L_w, "source": f"homocasc:{wp.name}"}
    return targets, donors


def _prioritize_seeds(seeds: list, max_keep: int = 32) -> list:
    """Keep clean, reloc, xcross, and low-penetration clamp seeds without truncation bias."""
    if len(seeds) <= max_keep:
        return seeds

    def score(s):
        name, x, y, th, _, _ = s
        pen, _ = hc.cpp_eval_sat_pen_bbox(x, y, th)
        tier = (
            0 if "clean" in name else
            1 if "xcross" in name else
            2 if "reloc" in name else
            3
        )
        return (tier, pen)

    ranked = sorted(seeds, key=score)
    return ranked[:max_keep]


def _run_job(spec: tuple) -> dict:
    mode, n_target, n_donor, xd, yd, td, Ld, L_inc_dec, layers, seed_int = spec
    rng = np.random.default_rng(seed_int)
    L_inc = float(L_inc_dec)

    if mode == "upward":
        ex, ey, et, eL = ce.extend_one_layer(xd, yd, td, Ld)
        assert len(ex) == n_target
        seeds = [(f"ext_{n_target}_from{n_donor}", ex, ey, et, min(eL, L_inc - 1e-4), np.array([[0.5 * eL, 0.5 * eL]]))]
        res = ce.solve_target(
            n_target, seeds, L_inc_dec, threads=4, rng=rng,
            n_seeds_per_spec=6, masked_iters=180, full_iters=220, top_polish=6, run_mechwalk=True,
        )
    else:
        seeds = ce.generate_peel_seeds(
            n_target, n_donor, xd, yd, td, Ld, L_inc, rng, layers=layers, max_prot=18, max_rem_tilt=6
        )
        if layers == 1:
            seeds += ce.generate_interior_cross_seeds(n_target, n_donor, xd, yd, td, Ld, L_inc)
        seeds = _prioritize_seeds(seeds, max_keep=30)
        res = ce.solve_target(
            n_target, seeds, L_inc_dec, threads=4, rng=rng,
            n_seeds_per_spec=4, masked_iters=200, full_iters=230, top_polish=8, run_mechwalk=True,
        )
    res["donor"] = n_donor
    res["mode"] = mode
    res["layers"] = layers
    return res


def main():
    hc.DOCS_DIR.mkdir(parents=True, exist_ok=True)
    hc.DATA_DIR.mkdir(parents=True, exist_ok=True)

    all_ns = {
        586, 635, 686, 740, 793, 794, 795, 850, 851, 852, 855,
        909, 910, 911, 914, 970, 971, 972, 975,
    }
    targets, donors = _load_best_known(all_ns)
    t_start = time.time()

    # Wave 1: highest-leverage cascade & brickslp pairs (including 909<-850 upward, 852<-911 deep, 793<-850 deep!)
    wave1 = [
        ("upward", 909, 850, donors[850]["x"], donors[850]["y"], donors[850]["th"], donors[850]["L"], targets[909]["L_dec"], 1, 20261200),
        ("peel", 852, 911, donors[911]["x"], donors[911]["y"], donors[911]["th"], donors[911]["L"], targets[852]["L_dec"], 1, 20261201),
        ("peel", 793, 850, donors[850]["x"], donors[850]["y"], donors[850]["th"], donors[850]["L"], targets[793]["L_dec"], 1, 20261202),
        ("peel", 851, 910, donors[910]["x"], donors[910]["y"], donors[910]["th"], donors[910]["L"], targets[851]["L_dec"], 1, 20261203),
        ("peel", 855, 914, donors[914]["x"], donors[914]["y"], donors[914]["th"], donors[914]["L"], targets[855]["L_dec"], 1, 20261204),
        ("peel", 910, 971, donors[971]["x"], donors[971]["y"], donors[971]["th"], donors[971]["L"], targets[910]["L_dec"], 1, 20261205),
        ("peel", 795, 911, donors[911]["x"], donors[911]["y"], donors[911]["th"], donors[911]["L"], targets[795]["L_dec"], 2, 20261206),
        ("peel", 586, 635, donors[635]["x"], donors[635]["y"], donors[635]["th"], donors[635]["L"], targets[586]["L_dec"], 1, 20261207),
        ("upward", 972, 911, donors[911]["x"], donors[911]["y"], donors[911]["th"], donors[911]["L"], targets[972]["L_dec"], 1, 20261208),
        ("upward", 975, 914, donors[914]["x"], donors[914]["y"], donors[914]["th"], donors[914]["L"], targets[975]["L_dec"], 1, 20261209),
    ]

    results = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        for res in pool.map(_run_job, wave1):
            results.append({k: v for k, v in res.items() if k != "best_pack"})
            (hc.DOCS_DIR / "scale_results.json").write_text(json.dumps(results, indent=2))

    # Reload donors (pick up any new wave1 wins in 909, 852, 851, 795, etc.) and run Wave 2 cascade
    targets, donors = _load_best_known(all_ns)
    wave2 = [
        ("upward", 970, 909, donors[909]["x"], donors[909]["y"], donors[909]["th"], donors[909]["L"], targets[970]["L_dec"], 1, 20261220),
        ("peel", 740, 795, donors[795]["x"], donors[795]["y"], donors[795]["th"], donors[795]["L"], targets[740]["L_dec"], 1, 20261221),
        ("peel", 794, 851, donors[851]["x"], donors[851]["y"], donors[851]["th"], donors[851]["L"], targets[794]["L_dec"], 1, 20261222),
        ("peel", 794, 910, donors[910]["x"], donors[910]["y"], donors[910]["th"], donors[910]["L"], targets[794]["L_dec"], 2, 20261223),
    ]
    if "homocasc" in donors[852]["source"]:
        wave2.append(("peel", 795, 852, donors[852]["x"], donors[852]["y"], donors[852]["th"], donors[852]["L"], targets[795]["L_dec"], 1, 20261224))

    with ProcessPoolExecutor(max_workers=4) as pool:
        for res in pool.map(_run_job, wave2):
            results.append({k: v for k, v in res.items() if k != "best_pack"})
            (hc.DOCS_DIR / "scale_results.json").write_text(json.dumps(results, indent=2))

    print(f"\n=== H30 SCALE COMPLETE in {time.time() - t_start:.1f}s ===", flush=True)
    for r in results:
        print(
            f"n={r['n']}<-{r['donor']} ({r['mode']}): win={r['win']} feas={r.get('n_feas')}/{r.get('n_jobs')} "
            f"best_seen_L={r.get('best_seen_L')} dL={r.get('dL')}",
            flush=True,
        )


if __name__ == "__main__":
    main()
