#!/usr/bin/env python3
"""H30 Wave 3: Cascade interlock win_n851 -> {794, 739, 910} and win_n850 -> {738, 685, 634}."""
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

INTERLOCK_851 = Path(
    "/home/lucas/.gemini/antigravity-cli/worktrees/a4fb0b93-b63b-426d-aa12-67454e99bc08"
    "/swarm-interlock-16f43b0b/poc/swarm_interlock/data/win_n851.json"
)


def _load_win_json(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    d = json.loads(path.read_text())
    rows = sorted(d["rows"], key=lambda r: int(r["square_id"]))
    x = np.array([float(r["x"]) for r in rows], dtype=np.float64)
    y = np.array([float(r["y"]) for r in rows], dtype=np.float64)
    th = np.array([float(r["angle"]) for r in rows], dtype=np.float64)
    return hc.norm_pack(x, y, th)


def _prioritize_seeds(seeds: list, max_keep: int = 30) -> list:
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

    return sorted(seeds, key=score)[:max_keep]


def _run_wave3_job(spec: tuple) -> dict:
    # Stage Wave 3 outputs separately so pinned PR branch final.csv is not mutated mid-verification
    hc.DATA_DIR = hc.HOMOCASC_DIR / "data_wave3"
    hc.FINAL_CSV = hc.HOMOCASC_DIR / "docs" / "wave3_final.csv"
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
    need = {634, 685, 738, 739, 794, 850, 851, 910}
    targets = hc.load_portfolio(need)

    x850, y850, t850, L850 = _load_win_json(hc.DATA_DIR / "win_n850.json")
    x851, y851, t851, L851 = _load_win_json(INTERLOCK_851)

    t_start = time.time()
    jobs = [
        ("peel", 794, 851, x851, y851, t851, L851, targets[794]["L_dec"], 1, 20261301),
        ("peel", 739, 851, x851, y851, t851, L851, targets[739]["L_dec"], 2, 20261302),
        ("upward", 910, 851, x851, y851, t851, L851, targets[910]["L_dec"], 1, 20261303),
        ("peel", 738, 850, x850, y850, t850, L850, targets[738]["L_dec"], 2, 20261304),
        ("peel", 685, 850, x850, y850, t850, L850, targets[685]["L_dec"], 3, 20261305),
        ("peel", 634, 850, x850, y850, t850, L850, targets[634]["L_dec"], 4, 20261306),
    ]

    results = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        for res in pool.map(_run_wave3_job, jobs):
            results.append({k: v for k, v in res.items() if k != "best_pack"})
            (hc.DOCS_DIR / "wave3_results.json").write_text(json.dumps(results, indent=2))

    print(f"\n=== H30 WAVE 3 COMPLETE in {time.time() - t_start:.1f}s ===", flush=True)
    for r in results:
        print(
            f"n={r['n']}<-{r['donor']} ({r['mode']}): win={r['win']} feas={r.get('n_feas')}/{r.get('n_jobs')} "
            f"best_seen_L={r.get('best_seen_L')} dL={r.get('dL')}",
            flush=True,
        )


if __name__ == "__main__":
    main()
