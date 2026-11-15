#!/usr/bin/env python3
"""Step 1: Extract donor bricks from n=492, 852, 911, 975, synthesize vstrip (g+1, g-1) & peel derivatives."""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vstrip492_lib as vl  # noqa: E402


def _orient_tall(b: dict) -> dict:
    if b["w"] > b["h"]:
        bt = vl.bl.transform_brick(b, "T")
        bt["tag"] = b["tag"]
        return bt
    return b


def _worker_compress_walk(job: tuple[str, dict, tuple[float, ...], int, int, int]) -> tuple[str, int, int, int, float] | None:
    name, b, f_sched, n_trials, walk_rounds, seed = job
    out_p = vl.DATA_DIR / f"brick_{name}.json"
    if out_p.exists():
        loaded = vl.load_brick_json(out_p, name)
        return name, loaded["w"], loaded["h"], loaded["g"], loaded["f"]
    res = vl.compress_and_walk_brick(
        b, name, f_targets=f_sched, n_trials=n_trials, walk_rounds=walk_rounds, seed=seed, save=True
    )
    if res is None:
        return None
    return name, res["w"], res["h"], res["g"], res["f"]


def _run_jobs(jobs: list[tuple], label: str) -> dict[str, dict]:
    print(f"[{label}] Running {len(jobs)} jobs (max_workers=2)...", flush=True)
    out: dict[str, dict] = {}
    with ProcessPoolExecutor(max_workers=2) as pool:
        futs = {pool.submit(_worker_compress_walk, j): j[0] for j in jobs}
        for fut in as_completed(futs):
            nm = futs[fut]
            try:
                res = fut.result()
            except Exception as exc:
                print(f"  [ERR] {nm}: {exc}", flush=True)
                continue
            if res is None:
                print(f"  [FAIL] {nm}", flush=True)
            else:
                r_nm, w, h, g, f = res
                print(f"  [OK] {r_nm:32s} {w:2d}x{h:2d} g=+{g} f={f:.6f}", flush=True)
                out[r_nm] = vl.load_brick_json(vl.DATA_DIR / f"brick_{r_nm}.json", r_nm)
    return out


def main() -> None:
    t0 = time.time()
    vl.DATA_DIR.mkdir(parents=True, exist_ok=True)
    vl.DOCS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Extract and compress/walk raw donor sub-bricks from 492, 852, 911, 975
    raw_ext = vl.extract_donor_subbricks([492, 852, 911, 975])
    keep_tags = [
        "B3_5x13_n492_BR",
        "B3_13x9_n492_TL",
        "B4_15x9_n492_TL",
        "B5_16x9_n492_TL",
        "B4_17x9_n852_BR",
        "B5_18x9_n852_BR",
        "B6_10x20_n852_TL",
        "B3_10x15_n911_BR",
        "B4_10x18_n911_BR",
        "B5_10x19_n911_BR",
        "B6_10x21_n911_BR",
        "B4_9x18_n975_TL",
        "B5_9x19_n975_TL",
        "B6_9x21_n975_TL",
        "B7_9x22_n975_TL",
    ]
    donor_jobs = []
    for idx, tag in enumerate(keep_tags):
        if tag in raw_ext:
            b = _orient_tall(raw_ext[tag])
            donor_jobs.append((f"{tag}_mw", b, (0.72, 0.695, 0.675), 6, 2, 1000 + idx))

    walked_donors = _run_jobs(donor_jobs, "1/3 Donor Compress+Walk")

    # Add top external bricks as parents for vstrip/peel derivatives
    parents = dict(walked_donors)
    ext_refs = {
        "EXT_B7_10x22_vstrip": vl.REPO_ROOT / "poc/swarm_gstep/data/brick_B7_10x22_vstrip.json",
        "EXT_B6_11x21_k5": vl.REPO_ROOT / "poc/swarm_gstep/data/brick_B6_11x21_k5.json",
        "EXT_B6_10x21_mw972": vl.REPO_ROOT / "poc/swarm_cleanpeel2/data/brick_B6_10x21_mw972_walk.json",
        "EXT_B5_20x10_peel_top": vl.REPO_ROOT / "poc/swarm_cleanpeel2/data/brick_B5_20x10_peel_top.json",
    }
    for nm, p in ext_refs.items():
        if p.exists():
            parents[nm] = _orient_tall(vl.load_brick_json(p, nm))

    # 2. Generate vstrip (+1), delvs (-1), hcol (+1), and 1-row/1-col peel derivatives
    synth_jobs = []
    seed_i = 2000

    # Target parent list for vstrip / delvs / peel
    for p_name, pb in sorted(parents.items()):
        # vstrip (g -> g+1, h -> h+1)
        if pb["g"] <= 7 and pb["h"] <= 23:
            for c_idx, (_, cand) in enumerate(vl.gl.splice_vertical_strip(pb)[:2]):
                nm = f"{p_name}_vs{c_idx}_B{cand['g']}_{cand['w']}x{cand['h']}"
                synth_jobs.append((nm, cand, (0.74, 0.71, 0.688, 0.672), 6, 2, seed_i))
                seed_i += 1
        # remove_vertical_strip (g -> g-1, h -> h-1)
        if pb["g"] >= 4 and pb["h"] >= 15:
            for c_idx, (_, cand) in enumerate(vl.remove_vertical_strip(pb)[:1]):
                nm = f"{p_name}_dv{c_idx}_B{cand['g']}_{cand['w']}x{cand['h']}"
                synth_jobs.append((nm, cand, (0.74, 0.71, 0.688, 0.672), 6, 2, seed_i))
                seed_i += 1
        # 1-row/1-col peels (preserving g)
        if not p_name.startswith("EXT_"):
            for side in ("left", "right", "top", "bottom"):
                if side in ("left", "right") and pb["w"] <= 4:
                    continue
                if side in ("top", "bottom") and pb["h"] <= 10:
                    continue
                pc = vl.gl.peel_outer_unit_layer(pb, side)
                if pc is not None:
                    nm = f"{p_name}_p{side[0]}_B{pc['g']}_{pc['w']}x{pc['h']}"
                    synth_jobs.append((nm, pc, (0.74, 0.71, 0.69, 0.675), 5, 2, seed_i))
                    seed_i += 1

    # Also test hcol on compact 492/911 bricks
    for p_name in ("B3_5x13_n492_BR_mw", "B5_16x9_n492_TL_mw", "B4_10x18_n911_BR_mw"):
        if p_name in parents:
            for c_idx, (_, cand) in enumerate(vl.gl.splice_horizontal_column(parents[p_name])[:1]):
                nm = f"{p_name}_hc{c_idx}_B{cand['g']}_{cand['w']}x{cand['h']}"
                synth_jobs.append((nm, cand, (0.75, 0.715, 0.69), 8, 2, seed_i))
                seed_i += 1

    # Limit duplicate shapes if too many; prioritize donor-derived and thinner parents
    walked_r2 = _run_jobs(synth_jobs, "2/3 vstrip/delvs/peel Derivatives")

    # 3. Round-2 peels on best newly synthesized vstrip bricks (e.g. 9x17 g=6 -> 8x17/9x16, 10x22 g=7 -> 9x22/10x21)
    r3_jobs = []
    # Pick best brick per (w, h, g) from walked_r2 where _vs appears
    best_vs_by_shape: dict[tuple[int, int, int], tuple[float, str, dict]] = {}
    for nm, b in walked_r2.items():
        if "_vs" in nm and b["f"] < 0.725:
            k = (b["w"], b["h"], b["g"])
            if k not in best_vs_by_shape or b["f"] < best_vs_by_shape[k][0]:
                best_vs_by_shape[k] = (b["f"], nm, b)
    for _, nm, b in sorted(best_vs_by_shape.values()):
        for side in ("left", "top"):
            pc = vl.gl.peel_outer_unit_layer(b, side)
            if pc is not None:
                p_nm = f"{nm}_p{side[0]}_B{pc['g']}_{pc['w']}x{pc['h']}"
                r3_jobs.append((p_nm, pc, (0.74, 0.715, 0.695), 8, 2, seed_i))
                seed_i += 1
        # Also test 2nd vstrip on thin g<=6 (e.g. 9x17 g=6 -> 9x18 g=7) and pad on 9x17 g=6 -> 9x18/9x19
        if b["g"] == 6 and b["w"] == 9 and b["f"] < 0.71:
            for c_idx, (_, cand) in enumerate(vl.gl.splice_vertical_strip(b)[:1]):
                v2_nm = f"{nm}_vs2_B{cand['g']}_{cand['w']}x{cand['h']}"
                r3_jobs.append((v2_nm, cand, (0.74, 0.715, 0.695), 8, 2, seed_i))
                seed_i += 1
            for side in ("top", "bottom"):
                pd_b = vl.gl.pad_outer_unit_layer(b, side)
                pd_nm = f"{nm}_pad{side[0]}_B{pd_b['g']}_{pd_b['w']}x{pd_b['h']}"
                r3_jobs.append((pd_nm, pd_b, (0.73, 0.705, 0.69), 6, 1, seed_i))
                seed_i += 1

    if r3_jobs:
        _run_jobs(r3_jobs, "3/3 Round-2 Peel/vstrip/Pad")

    # Save summary of all bricks in data/
    summary = []
    for p in sorted(vl.DATA_DIR.glob("brick_*.json")):
        b = vl.load_brick_json(p)
        summary.append({
            "name": b["tag"], "w": b["w"], "h": b["h"], "g": b["g"],
            "fx": round(b["fx"], 6), "fy": round(b["fy"], 6), "f": round(b["f"], 6),
        })
    summary.sort(key=lambda d: (d["g"], d["f"], d["w"] + d["h"]))
    (vl.DOCS_DIR / "synth_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"DONE synth_bricks ({len(summary)} total bricks) in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
