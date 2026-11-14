#!/usr/bin/env python3
"""Fast parallel (max_workers=2) synthesis of B6_synth, B7_synth, B8_synth and peeled/padded variants."""
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
import gstep_lib as gl  # noqa: E402


def _compress_job(job: tuple[str, dict, tuple[float, ...], int]) -> dict | None:
    name, cand, f_sched, seed = job
    out_p = gl.DATA_DIR / f"brick_{name}.json"
    if out_p.exists():
        b = gl.load_brick_json(out_p, name)
        return {"name": name, "w": b["w"], "h": b["h"], "g": b["g"], "fx": b["fx"], "fy": b["fy"], "f": b["f"], "cached": True}
    comp = gl.relax_and_compress_brick(cand, f_targets=f_sched, n_trials=6, seed=seed)
    if comp is None:
        return None
    comp["tag"] = name
    gl.save_brick_json(comp, name)
    return {"name": name, "w": comp["w"], "h": comp["h"], "g": comp["g"], "fx": comp["fx"], "fy": comp["fy"], "f": comp["f"], "cached": False}


def run_batch(jobs: list[tuple[str, dict, tuple[float, ...], int]]) -> dict[str, dict]:
    res_map: dict[str, dict] = {}
    with ProcessPoolExecutor(max_workers=2) as pool:
        futs = {pool.submit(_compress_job, j): j[0] for j in jobs}
        for fut in as_completed(futs):
            nm = futs[fut]
            r = fut.result()
            if r is not None:
                res_map[nm] = r
                print(f"  [BRICK] {nm:28s}: {r['w']}x{r['h']} g=+{r['g']} f={r['f']:.6f} (cached={r['cached']})", flush=True)
            else:
                print(f"  [BRICK] {nm:28s}: FAILED", flush=True)
    return res_map


def main() -> None:
    t0 = time.time()
    bricks = gl.load_source_bricks()
    b6_10x21 = bricks["B6_10x21_mw972_walk"]
    b6_10x20 = bricks["B6_10x20_mw972_top"]
    b6_9x18 = bricks["B6_9x18_n796_TL"]
    b5_11x20 = bricks["B5_11x20"]
    b7_10x22 = gl.load_brick_json(gl.DATA_DIR / "brick_B7_10x22_vstrip.json", "B7_10x22_vstrip")

    # Stage 1: Primary g-step splices + B6 9x19/9x20 pads/peels
    jobs1: list[tuple[str, dict, tuple[float, ...], int]] = []
    # Middle k cuts on B6_10x21 -> B7_10x22
    for cand_tag, cand in gl.splice_vertical_strip(b6_10x21):
        if cand_tag.endswith(("_k8", "_k9", "_k10")):
            jobs1.append((f"B7_10x22_{cand_tag.split('_')[-1]}", cand, (0.74, 0.70, 0.68, 0.665), 11))

    # B5_11x20 -> B6_11x21
    for cand_tag, cand in gl.splice_vertical_strip(b5_11x20)[:2]:
        jobs1.append((f"B6_11x21_{cand_tag.split('_')[-1]}", cand, (0.74, 0.70, 0.675, 0.655), 21))

    # B7_10x22 -> B8_10x23
    for cand_tag, cand in gl.splice_vertical_strip(b7_10x22)[:2]:
        jobs1.append((f"B8_10x23_{cand_tag.split('_')[-1]}", cand, (0.76, 0.72, 0.695, 0.68), 31))

    # B6_9x18 -> pad bottom/top -> B6_9x19
    for side in ("bottom", "top"):
        jobs1.append((f"B6_9x19_pad_{side}", gl.pad_outer_unit_layer(b6_9x18, side), (0.72, 0.695, 0.68, 0.665), 41))

    # B6_10x21 -> peel left/right -> B6_9x21
    for side in ("left", "right"):
        pb = gl.peel_outer_unit_layer(b6_10x21, side)
        if pb is not None:
            jobs1.append((f"B6_9x21_peel_{side}", pb, (0.74, 0.70, 0.68, 0.66), 51))

    # B6_10x20 -> peel left/right -> B6_9x20
    for side in ("left", "right"):
        pb = gl.peel_outer_unit_layer(b6_10x20, side)
        if pb is not None:
            jobs1.append((f"B6_9x20_peel_{side}", pb, (0.74, 0.71, 0.69, 0.675), 61))

    # B7_10x22 -> peel bottom/left/right -> B7_10x21 / B7_9x22
    for side in ("bottom", "left", "right"):
        pb = gl.peel_outer_unit_layer(b7_10x22, side)
        if pb is not None:
            jobs1.append((f"B7_{pb['w']}x{pb['h']}_peel_{side}", pb, (0.75, 0.72, 0.70, 0.685), 71))

    print(f"Stage 1: {len(jobs1)} brick compression jobs...", flush=True)
    res1 = run_batch(jobs1)

    # Stage 2: Second-step peels/pads from best Stage 1 bricks
    jobs2: list[tuple[str, dict, tuple[float, ...], int]] = []
    # Best B6_9x19 -> pad -> B6_9x20
    b9x19_cands = [nm for nm in res1 if nm.startswith("B6_9x19_pad_")]
    if b9x19_cands:
        best_9x19 = min(b9x19_cands, key=lambda nm: res1[nm]["f"])
        b_19 = gl.load_brick_json(gl.DATA_DIR / f"brick_{best_9x19}.json", best_9x19)
        for side in ("bottom", "top"):
            jobs2.append((f"B6_9x20_from9x18_{side}", gl.pad_outer_unit_layer(b_19, side), (0.71, 0.69, 0.67, 0.655), 81))

    # Best B6_9x21 -> peel top/bottom -> B6_9x20
    b9x21_cands = [nm for nm in res1 if nm.startswith("B6_9x21_peel_")]
    if b9x21_cands:
        best_9x21 = min(b9x21_cands, key=lambda nm: res1[nm]["f"])
        b_21 = gl.load_brick_json(gl.DATA_DIR / f"brick_{best_9x21}.json", best_9x21)
        for side in ("top", "bottom"):
            pb = gl.peel_outer_unit_layer(b_21, side)
            if pb is not None:
                jobs2.append((f"B6_9x20_from9x21_{side}", pb, (0.73, 0.70, 0.68, 0.665), 91))

    # Best B6_11x21 -> peel top/bottom/left/right -> B6_11x20 / B6_10x21
    b11x21_cands = [nm for nm in res1 if nm.startswith("B6_11x21_")]
    if b11x21_cands:
        best_11x21 = min(b11x21_cands, key=lambda nm: res1[nm]["f"])
        b_11x21 = gl.load_brick_json(gl.DATA_DIR / f"brick_{best_11x21}.json", best_11x21)
        for side in ("top", "bottom", "left", "right"):
            pb = gl.peel_outer_unit_layer(b_11x21, side)
            if pb is not None:
                jobs2.append((f"B6_{pb['w']}x{pb['h']}_from11x21_{side}", pb, (0.73, 0.70, 0.68, 0.665), 101))

    # Best B8_10x23 -> peel top/bottom -> B8_10x22
    b8_cands = [nm for nm in res1 if nm.startswith("B8_10x23_")]
    if b8_cands:
        best_8 = min(b8_cands, key=lambda nm: res1[nm]["f"])
        b_8 = gl.load_brick_json(gl.DATA_DIR / f"brick_{best_8}.json", best_8)
        for side in ("top", "bottom"):
            pb = gl.peel_outer_unit_layer(b_8, side)
            if pb is not None:
                jobs2.append((f"B8_10x22_from23_{side}", pb, (0.75, 0.72, 0.70, 0.685), 111))

    if jobs2:
        print(f"Stage 2: {len(jobs2)} brick compression jobs...", flush=True)
        res2 = run_batch(jobs2)
        res1.update(res2)

    gl.DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (gl.DOCS_DIR / "fast_synth_summary.json").write_text(json.dumps(res1, indent=2), encoding="utf-8")
    print(f"ALL SYNTH DONE in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
