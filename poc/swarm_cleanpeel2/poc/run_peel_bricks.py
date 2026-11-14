#!/usr/bin/env python3
"""Peel outer unit rows/columns from H74 donor bricks, compress via SLP, and run standalone brick_mechwalk."""
from __future__ import annotations

import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import cleanpeel2_lib as cp2  # noqa: E402

H60_B5_20X11_PATH = Path(
    "/home/lucas/.gemini/antigravity-cli/worktrees/a4fb0b93-b63b-426d-aa12-67454e99bc08/swarm-b8xplant-7f6c04d2/poc/swarm_b8xplant/data/brick_H60_B5_20x11_n970s6.json"
)
MW972_PATH = Path(
    "/home/lucas/.gemini/antigravity-cli/worktrees/a4fb0b93-b63b-426d-aa12-67454e99bc08/swarm-mechwalk-8abd0e01/poc/swarm_mechwalk/data/win_n972.json"
)


def _process_one_brick(args: tuple[str, dict, int, int, int]) -> dict:
    name, b, n_trials, seed, walk_rounds = args
    t0 = time.time()
    print(f"[START] {name}: {b['w']}x{b['h']} g=+{b['g']} raw f={b['f']:.9f} (fx={b['fx']:.6f}, fy={b['fy']:.6f})", flush=True)
    cb = cp2.compress_brick_slp(b, n_trials=n_trials, seed=seed)
    print(f"  [SLP {name}] f={cb['f']:.9f} (fx={cb['fx']:.6f}, fy={cb['fy']:.6f}) in {time.time()-t0:.1f}s", flush=True)
    wb = cp2.brick_mechwalk(cb, name, max_rounds=walk_rounds, save=True)
    print(f"[DONE] {name}: {wb['w']}x{wb['h']} g=+{wb['g']} final f={wb['f']:.9f} (fx={wb['fx']:.6f}, fy={wb['fy']:.6f}) in {time.time()-t0:.1f}s", flush=True)
    return {
        "name": name, "w": wb["w"], "h": wb["h"], "g": wb["g"],
        "raw_f": b["f"], "slp_f": cb["f"], "f": wb["f"], "fx": wb["fx"], "fy": wb["fy"],
        "dt": round(time.time() - t0, 1),
    }


def main() -> None:
    t0 = time.time()
    cp2.DATA_DIR.mkdir(parents=True, exist_ok=True)
    cp2.DOCS_DIR.mkdir(parents=True, exist_ok=True)

    # Load source bricks
    b5_20x11 = cp2.load_brick_json(H60_B5_20X11_PATH, tag="H60_B5_20x11_n970s6")
    cp2.save_brick_json(b5_20x11, "H60_B5_20x11_n970s6")

    x972, y972, t972, L972 = cp2.load_json_pack(MW972_PATH)
    b6_10x21 = cp2.extract_corner_brick(x972, y972, t972, L972, "BR", 10, 21, 6, "B6_10x21_mw972", margin=0.75)
    assert b6_10x21 is not None, "Failed to extract B6_10x21_mw972"
    b5_20x10_mw972 = cp2.extract_corner_brick(x972, y972, t972, L972, "TL", 20, 10, 5, "B5_20x10_mw972", margin=0.75)
    assert b5_20x10_mw972 is not None, "Failed to extract B5_20x10_mw972"

    frontier = cp2.fast_load_true_frontier([796, 850], certify=False)
    x796, y796, t796, L796 = frontier[796]["x"], frontier[796]["y"], frontier[796]["th"], frontier[796]["L"]
    b6_9x18_tl = cp2.extract_corner_brick(x796, y796, t796, L796, "TL", 9, 18, 6, "B6_9x18_n796_TL", margin=0.75)
    b6_18x9_br = cp2.extract_corner_brick(x796, y796, t796, L796, "BR", 18, 9, 6, "B6_18x9_n796_BR", margin=0.75)
    assert b6_9x18_tl is not None and b6_18x9_br is not None
    cp2.save_brick_json(b6_9x18_tl, "B6_9x18_n796_TL")

    x850, y850, t850, L850 = frontier[850]["x"], frontier[850]["y"], frontier[850]["th"], frontier[850]["L"]
    b5_20x11_850 = cp2.extract_corner_brick(x850, y850, t850, L850, "BR", 20, 11, 5, "B5_20x11_n850", margin=0.75)
    b5_20x10_850 = cp2.extract_corner_brick(x850, y850, t850, L850, "BR", 20, 10, 5, "B5_20x10_n850", margin=0.75)

    jobs: list[tuple[str, dict, int, int, int]] = []

    # Part (a): Peel H60_B5_20x11_n970s6
    for side in ("top", "bottom"):
        p = cp2.peel_outer_unit_layer(b5_20x11, side, f"B5_20x10_peel_{side}")
        if p is not None:
            jobs.append((p["tag"], p, 16, 7401 + len(jobs), 3))
    p_19x11 = cp2.peel_outer_unit_layer(b5_20x11, "right", "B5_19x11_peel_right")
    if p_19x11 is not None:
        jobs.append((p_19x11["tag"], p_19x11, 16, 7410, 3))
        p_19x10_a = cp2.peel_outer_unit_layer(p_19x11, "top", "B5_19x10_peel_rt")
        if p_19x10_a is not None:
            jobs.append((p_19x10_a["tag"], p_19x10_a, 16, 7411, 3))

    # Part (b): Walk B6_10x21_mw972 + peel 10x20 + 10x19; also walk B5_20x10_mw972 + peel 19x10
    jobs.append(("B6_10x21_mw972_walk", b6_10x21, 12, 7420, 3))
    jobs.append(("B5_20x10_mw972_walk", b5_20x10_mw972, 12, 7421, 3))
    b5_19x10_mw = cp2.peel_outer_unit_layer(b5_20x10_mw972, "right", "B5_19x10_mw972_peelR")
    if b5_19x10_mw is not None:
        jobs.append((b5_19x10_mw["tag"], b5_19x10_mw, 16, 7422, 3))

    for side in ("top", "bottom"):
        p10x20 = cp2.peel_outer_unit_layer(b6_10x21, side, f"B6_10x20_mw972_{side}")
        if p10x20 is not None:
            jobs.append((p10x20["tag"], p10x20, 16, 7430 + len(jobs), 3))
            p10x19 = cp2.peel_outer_unit_layer(p10x20, "bottom" if side == "top" else "top", f"B6_10x19_mw972_{side}opp")
            if p10x19 is not None:
                jobs.append((p10x19["tag"], p10x19, 16, 7440 + len(jobs), 3))

    # Part (c): Peel B6_9x18_n796 -> 9x17, and B6_18x9_n796 -> 17x9
    for side in ("top", "bottom"):
        p9x17 = cp2.peel_outer_unit_layer(b6_9x18_tl, side, f"B6_9x17_n796_{side}")
        if p9x17 is not None:
            jobs.append((p9x17["tag"], p9x17, 16, 7450 + len(jobs), 3))
    p17x9 = cp2.peel_outer_unit_layer(b6_18x9_br, "right", "B6_17x9_n796_right")
    if p17x9 is not None:
        jobs.append((p17x9["tag"], p17x9, 16, 7460, 3))

    if b5_20x10_850 is not None:
        b5_19x10_850 = cp2.peel_outer_unit_layer(b5_20x10_850, "right", "B5_19x10_n850_peelR")
        if b5_19x10_850 is not None:
            jobs.append((b5_19x10_850["tag"], b5_19x10_850, 16, 7470, 3))

    print(f"Dispatching {len(jobs)} brick peel+SLP+mechwalk jobs...", flush=True)
    with mp.Pool(processes=min(8, len(jobs))) as pool:
        results = pool.map(_process_one_brick, jobs)

    summary = sorted(results, key=lambda r: (r["g"], r["w"], r["h"], r["f"]))
    (cp2.DOCS_DIR / "peel_bricks_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n=== ALL BRICKS COMPLETE in {time.time()-t0:.1f}s ===", flush=True)
    for r in summary:
        print(f"  {r['name']:28s} {r['w']:2d}x{r['h']:2d} g=+{r['g']} raw={r['raw_f']:.6f} -> slp={r['slp_f']:.6f} -> walk={r['f']:.9f}", flush=True)


if __name__ == "__main__":
    main()
