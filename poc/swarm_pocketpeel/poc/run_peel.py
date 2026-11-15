#!/usr/bin/env python3
"""Step 1 & 2: Extract donor pockets + peel 1 row/col + compress_brick_slp + brick_mechwalk."""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import concurrent.futures
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import pocketpeel_lib as pp  # noqa: E402
import cleanpeel2_lib as cp2  # noqa: E402


def collect_donor_bricks(include_pareto: bool = True) -> dict[str, dict]:
    """Extract donor bricks from merged.csv (852, 912, 974, 975) and Pareto pocket JSONs."""
    fr = pp.load_merged_frontier(certify=False)
    donors: dict[str, dict] = {}

    explicit_specs = [
        (852, "BR", 20, 11, 6, "n852_20x11_g6"),
        (852, "TL", 9, 17, 5, "n852_9x17_g5"),
        (912, "TL", 9, 21, 6, "n912_9x21_g6"),
        (912, "BR", 21, 9, 6, "n912_21x9_g6"),
        (974, "TL", 9, 21, 6, "n974_9x21_g6"),
        (974, "BR", 22, 10, 7, "n974_22x10_g7"),
        (975, "TL", 9, 22, 7, "n975_9x22_g7"),
        (975, "BR", 22, 9, 7, "n975_22x9_g7"),
    ]
    for n, corner, w, h, g, name in explicit_specs:
        if n not in fr:
            continue
        b = cp2.extract_corner_brick(fr[n]["x"], fr[n]["y"], fr[n]["th"], fr[n]["L"], corner, w, h, g, name, margin=0.78)
        if b is not None:
            donors[name] = b
            pp.save_brick(b, name)

    if include_pareto:
        json_paths = [
            pp.REPO_ROOT / "poc" / "swarm_liftpolish3" / "data" / "pockets_partb_sane.json",
            pp.REPO_ROOT / "poc" / "pocket_lift_scan3" / "data" / "pockets_pareto_e3_sane.json",
        ]
        best_per_key: dict[tuple[int, int, int], tuple[float, dict]] = {}
        for jp in json_paths:
            if not jp.exists():
                continue
            for pk in json.loads(jp.read_text(encoding="utf-8")):
                src = int(pk["src"])
                w, h, g = int(pk["w"]), int(pk["h"]), int(pk["g"])
                f = float(pk["f"])
                if src >= 1001 or src not in fr:
                    continue
                if not (4 <= g <= 10 and min(w, h) <= 13 and max(w, h) <= 24 and f <= 0.76):
                    continue
                b = pp.extract_pocket_bbox(fr[src], pk, f"p_n{src}_{w}x{h}_g{g}")
                if b is None:
                    continue
                # Keep canonical orientation (w <= h) + original
                key = (min(w, h), max(w, h), g)
                if key not in best_per_key or b["f"] < best_per_key[key][0] - 1e-9:
                    best_per_key[key] = (b["f"], b)
        for (_, _, _), (_, b) in sorted(best_per_key.items()):
            donors[b["tag"]] = b
            pp.save_brick(b, b["tag"])

    return donors


def _peel_worker(args: tuple[str, dict, int, int, int]) -> dict:
    name, b, n_trials, seed, walk_rounds = args
    t0 = time.time()
    cb = pp.compress_brick_slp(b, n_trials=n_trials, seed=seed)
    wb = pp.brick_mechwalk(cb, name, max_rounds=walk_rounds, save=True)
    dt = round(time.time() - t0, 1)
    print(
        f"  [PEEL] {name:34s} ({wb['w']:2d}x{wb['h']:2d}+{wb['g']}) "
        f"raw={b['f']:.6f} -> slp={cb['f']:.6f} -> walk={wb['f']:.9f} ({dt}s)",
        flush=True,
    )
    return {
        "name": name,
        "w": wb["w"],
        "h": wb["h"],
        "g": wb["g"],
        "raw_f": float(b["f"]),
        "slp_f": float(cb["f"]),
        "f": float(wb["f"]),
        "fx": float(wb["fx"]),
        "fy": float(wb["fy"]),
        "dt": dt,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-jobs", type=int, default=40)
    ap.add_argument("--n-trials", type=int, default=10)
    ap.add_argument("--walk-rounds", type=int, default=2)
    args = ap.parse_args()

    t0 = time.time()
    pp.ensure_dirs()
    donors = collect_donor_bricks(include_pareto=True)
    print(f"Collected {len(donors)} donor bricks in {time.time()-t0:.1f}s", flush=True)

    # Generate 1-row / 1-col peel candidates and rank by raw_f and priority
    priority_shapes = {
        (11, 19, 6), (19, 11, 6), (10, 20, 6), (20, 10, 6),
        (17, 8, 5), (8, 17, 5), (16, 9, 5), (9, 16, 5),
        (9, 20, 6), (20, 9, 6), (8, 21, 6), (21, 8, 6),
        (21, 10, 7), (10, 21, 7), (22, 9, 7), (9, 22, 7),
        (9, 21, 7), (21, 9, 7), (8, 22, 7), (22, 8, 7),
        (12, 19, 7), (19, 12, 7), (11, 20, 7), (20, 11, 7),
        (11, 19, 7), (19, 11, 7), (10, 20, 7), (20, 10, 7),
        (19, 9, 7), (9, 19, 7), (19, 10, 7), (10, 19, 7),
        (18, 9, 5), (9, 18, 5), (19, 9, 5), (9, 19, 5), (18, 10, 5), (10, 18, 5),
        (11, 20, 8), (20, 11, 8), (12, 19, 8), (19, 12, 8), (19, 11, 8), (11, 19, 8),
        (10, 20, 8), (20, 10, 8), (19, 10, 8), (10, 19, 8),
    }

    cands: list[tuple[float, str, dict]] = []
    seen_peels: dict[tuple[int, int, int], list[tuple[float, str, dict]]] = {}
    for dname, db in sorted(donors.items()):
        for side in ("top", "bottom", "left", "right"):
            pb = pp.peel_outer_unit_layer(db, side, tag=f"{dname}_peel_{side}")
            if pb is None:
                continue
            # Filter out torn cuts where raw_f > 0.96 unless it's a core priority shape
            shp = (pb["w"], pb["h"], pb["g"])
            is_prio = shp in priority_shapes
            if pb["f"] > 0.99 and not is_prio:
                continue
            if pb["f"] > 1.05:
                continue
            canon_shp = (min(pb["w"], pb["h"]), max(pb["w"], pb["h"]), pb["g"])
            score = pb["f"] - (0.08 if is_prio else 0.0) - (0.05 if dname.startswith(("n852_", "n912_", "n974_", "n975_")) else 0.0)
            seen_peels.setdefault(canon_shp, []).append((score, pb["tag"], pb))

    # Keep top 2 candidates per canonical (min, max, g) shape
    selected: list[tuple[float, str, dict]] = []
    for canon_shp, lst in sorted(seen_peels.items()):
        lst.sort(key=lambda t: t[0])
        selected.extend(lst[:2])
    selected.sort(key=lambda t: t[0])
    selected = selected[: args.max_jobs]

    print(f"Selected {len(selected)} peel jobs (from {sum(len(v) for v in seen_peels.values())} raw peels):", flush=True)
    for sc, nm, pb in selected:
        print(f"  {nm:36s} ({pb['w']:2d}x{pb['h']:2d}+{pb['g']}) raw_fx={pb['fx']:.6f} raw_fy={pb['fy']:.6f} raw_f={pb['f']:.6f}", flush=True)

    if args.dry_run:
        return

    jobs = []
    for idx, (_, nm, pb) in enumerate(selected):
        out_p = pp.BRICKS_DIR / f"brick_{nm}.json"
        if out_p.exists():
            continue
        jobs.append((nm, pb, args.n_trials, 113000 + idx, args.walk_rounds))

    print(f"Running {len(jobs)} compression+mechwalk jobs with max_workers=2...", flush=True)
    results = []
    if jobs:
        with concurrent.futures.ProcessPoolExecutor(max_workers=2) as ex:
            for r in ex.map(_peel_worker, jobs):
                results.append(r)

    summary = sorted(results, key=lambda r: (r["g"], r["w"], r["h"], r["f"]))
    (pp.DOCS_DIR / "peel_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Done in {time.time()-t0:.1f}s ({len(results)} new bricks).", flush=True)


if __name__ == "__main__":
    main()
