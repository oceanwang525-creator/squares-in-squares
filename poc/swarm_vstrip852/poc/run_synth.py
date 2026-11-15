#!/usr/bin/env python3
"""H114 (vstrip852) Stage 1 & Stage 2 periodic-strip brick synthesis + pslp shake_squeeze (max_workers=2)."""
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
import vstrip_lib as vl  # noqa: E402
import gstep_lib as gl  # noqa: E402


def _job(spec: tuple[str, dict, tuple[float, ...], int, int, int]) -> dict | None:
    name, cand, f_sched, n_trials, walk_rounds, seed = spec
    out_p = vl.DATA_DIR / f"brick_{name}.json"
    if out_p.exists():
        b = vl.load_brick(out_p, name)
        return {"name": name, "w": b["w"], "h": b["h"], "g": b["g"], "f": b["f"], "cached": True}
    comp = vl.compress_and_walk_brick(
        cand, name, f_targets=f_sched, n_trials=n_trials, walk_rounds=walk_rounds, seed=seed
    )
    if comp is None:
        return None
    return {"name": name, "w": comp["w"], "h": comp["h"], "g": comp["g"], "f": comp["f"], "cached": False}


def run_batch(jobs: list[tuple[str, dict, tuple[float, ...], int, int, int]]) -> dict[str, dict]:
    res: dict[str, dict] = {}
    with ProcessPoolExecutor(max_workers=2) as pool:
        futs = {pool.submit(_job, j): j[0] for j in jobs}
        for fut in as_completed(futs):
            nm = futs[fut]
            r = fut.result()
            if r is not None:
                res[nm] = r
                print(
                    f"  [BRICK] {nm:32s}: {r['w']}x{r['h']} g=+{r['g']} f={r['f']:.6f} (cached={r['cached']})",
                    flush=True,
                )
            else:
                print(f"  [BRICK] {nm:32s}: FAILED", flush=True)
    return res


def _pick_cuts(cands: list[tuple[str, dict]], max_n: int = 4) -> list[tuple[str, dict]]:
    if len(cands) <= max_n:
        return cands
    step = (len(cands) - 1) / float(max_n - 1)
    idxs = sorted({int(round(i * step)) for i in range(max_n)})
    return [cands[i] for i in idxs]


def main() -> None:
    t0 = time.time()
    base = vl.extract_champion_bricks()
    for nm, b in base.items():
        vl.save_brick(b, f"base_{nm}")

    # Re-squeeze any cached bricks with pslp shake_squeeze if not already done
    for p in sorted(vl.DATA_DIR.glob("brick_*.json")):
        nm = p.stem.replace("brick_", "")
        if nm.startswith("base_"):
            continue
        b = vl.load_brick(p, nm)
        b2 = vl.pslp_squeeze_brick(b, shake_rounds=5, seed=777)
        if b2["f"] < b["f"] - 1e-8:
            print(f"  [RESQUEEZE] {nm:30s}: {b['f']:.6f} -> {b2['f']:.6f}", flush=True)
            vl.save_brick(b2, nm)

    # Stage 1: 1-step vstrip and 1-step peel -> vstrip
    jobs1: list[tuple[str, dict, tuple[float, ...], int, int, int]] = []

    b_11x20 = base["n852_11x20"]
    for i, (_, cand) in enumerate(_pick_cuts(gl.splice_vertical_strip(b_11x20), 4)):
        jobs1.append((f"n852_11x21_vs_{i}", cand, (0.76, 0.72, 0.70, 0.688), 5, 6, 100 + i))

    b_9x17 = base["n852_9x17"]
    for i, (_, cand) in enumerate(_pick_cuts(gl.splice_vertical_strip(b_9x17), 4)):
        jobs1.append((f"n852_9x18_vs_{i}", cand, (0.76, 0.72, 0.70, 0.688), 5, 6, 140 + i))

    b_11x21 = base["gstep_11x21_k5"]
    for i, (_, cand) in enumerate(_pick_cuts(gl.splice_vertical_strip(b_11x21), 4)):
        jobs1.append((f"gs_11x22_vs_{i}", cand, (0.74, 0.70, 0.685, 0.672), 5, 6, 180 + i))

    b_9x21 = base["n912_9x21"]
    for i, (_, cand) in enumerate(_pick_cuts(gl.splice_vertical_strip(b_9x21), 3)):
        jobs1.append((f"n912_9x22_vs_{i}", cand, (0.76, 0.73, 0.71, 0.698), 5, 6, 220 + i))

    b_10x22 = base["n974_10x22"]
    for i, (_, cand) in enumerate(_pick_cuts(gl.splice_vertical_strip(b_10x22), 3)):
        jobs1.append((f"n974_10x23_vs_{i}", cand, (0.77, 0.74, 0.72, 0.705), 5, 6, 240 + i))

    b_gs10x22 = base["gstep_10x22_vs"]
    for i, (_, cand) in enumerate(_pick_cuts(gl.splice_vertical_strip(b_gs10x22), 3)):
        jobs1.append((f"gs_10x23_vs_{i}", cand, (0.77, 0.74, 0.715, 0.698), 5, 6, 260 + i))

    # Peel n852_11x20 -> 10x20(+6), n852_9x17 -> 8x17(+5)
    for side in ("left", "right"):
        pb = gl.peel_outer_unit_layer(b_11x20, side)
        if pb is not None:
            jobs1.append((f"n852_10x20_peel_{side}", pb, (0.76, 0.73, 0.705, 0.690), 5, 5, 280 + len(jobs1)))
        pb2 = gl.peel_outer_unit_layer(b_9x17, side)
        if pb2 is not None:
            jobs1.append((f"n852_8x17_peel_{side}", pb2, (0.76, 0.73, 0.705, 0.690), 5, 5, 290 + len(jobs1)))

    print(f"Stage 1: {len(jobs1)} brick jobs...", flush=True)
    res1 = run_batch(jobs1)

    # Stage 2: 2nd vstrip + peels on best Stage 1 bricks
    jobs2: list[tuple[str, dict, tuple[float, ...], int, int, int]] = []

    def best_brick(prefix: str) -> dict | None:
        cands = [nm for nm in res1 if nm.startswith(prefix)]
        if not cands:
            return None
        best_nm = min(cands, key=lambda nm: res1[nm]["f"])
        return vl.load_brick(vl.DATA_DIR / f"brick_{best_nm}.json", best_nm)

    # Best n852 (11, 21, +7) -> (11, 22, +8) & peels (10, 21, +7), (11, 20, +7)
    b7_11x21 = best_brick("n852_11x21_vs_")
    if b7_11x21 is not None:
        for i, (_, cand) in enumerate(_pick_cuts(gl.splice_vertical_strip(b7_11x21), 3)):
            jobs2.append((f"n852_11x22_g8_vs_{i}", cand, (0.78, 0.75, 0.725, 0.705), 5, 6, 300 + i))
        for side in ("left", "right", "top", "bottom"):
            pb = gl.peel_outer_unit_layer(b7_11x21, side)
            if pb is not None:
                jobs2.append((f"n852_11x21_peel_{side}", pb, (0.76, 0.73, 0.715, 0.702), 5, 5, 340 + len(jobs2)))

    # Best gs (11, 22, +7) -> (11, 23, +8) & peels (10, 22, +7), (11, 21, +7)
    b7_11x22 = best_brick("gs_11x22_vs_")
    if b7_11x22 is not None:
        for i, (_, cand) in enumerate(_pick_cuts(gl.splice_vertical_strip(b7_11x22), 2)):
            jobs2.append((f"gs_11x23_g8_vs_{i}", cand, (0.78, 0.75, 0.72, 0.70), 5, 6, 350 + i))
        for side in ("top", "bottom", "left", "right"):
            pb = gl.peel_outer_unit_layer(b7_11x22, side)
            if pb is not None:
                jobs2.append((f"gs_11x22_peel_{side}", pb, (0.76, 0.73, 0.705, 0.688), 5, 5, 360 + len(jobs2)))

    # Best n852 (9, 18, +6) -> (9, 19, +7) & peels (8, 18, +6), (9, 17, +6)
    b6_9x18 = best_brick("n852_9x18_vs_")
    if b6_9x18 is not None:
        for i, (_, cand) in enumerate(_pick_cuts(gl.splice_vertical_strip(b6_9x18), 3)):
            jobs2.append((f"n852_9x19_g7_vs_{i}", cand, (0.78, 0.75, 0.725, 0.708), 5, 6, 400 + i))
        for side in ("left", "right", "top", "bottom"):
            pb = gl.peel_outer_unit_layer(b6_9x18, side)
            if pb is not None:
                jobs2.append((f"n852_9x18_peel_{side}", pb, (0.76, 0.73, 0.715, 0.702), 5, 5, 440 + len(jobs2)))

    # Best n912 (9, 22, +7) -> (9, 23, +8) & peels (8, 22, +7), (9, 21, +7)
    b7_9x22 = best_brick("n912_9x22_vs_")
    if b7_9x22 is not None:
        for i, (_, cand) in enumerate(_pick_cuts(gl.splice_vertical_strip(b7_9x22), 2)):
            jobs2.append((f"n912_9x23_g8_vs_{i}", cand, (0.79, 0.76, 0.735, 0.715), 5, 6, 460 + i))
        for side in ("left", "right", "top", "bottom"):
            pb = gl.peel_outer_unit_layer(b7_9x22, side)
            if pb is not None:
                jobs2.append((f"n912_9x22_peel_{side}", pb, (0.78, 0.75, 0.725, 0.712), 5, 5, 470 + len(jobs2)))

    # vstrip on peeled (10, 20, +6) -> (10, 21, +7) and (8, 17, +5) -> (8, 18, +6)
    b6_10x20 = best_brick("n852_10x20_peel_")
    if b6_10x20 is not None:
        for i, (_, cand) in enumerate(_pick_cuts(gl.splice_vertical_strip(b6_10x20), 2)):
            jobs2.append((f"n852_10x21_g7_vs_{i}", cand, (0.77, 0.74, 0.715, 0.700), 5, 6, 480 + i))
    b5_8x17 = best_brick("n852_8x17_peel_")
    if b5_8x17 is not None:
        for i, (_, cand) in enumerate(_pick_cuts(gl.splice_vertical_strip(b5_8x17), 2)):
            jobs2.append((f"n852_8x18_g6_vs_{i}", cand, (0.77, 0.74, 0.715, 0.700), 5, 6, 490 + i))

    if jobs2:
        print(f"\nStage 2: {len(jobs2)} brick jobs...", flush=True)
        res2 = run_batch(jobs2)
        res1.update(res2)

    vl.DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (vl.DOCS_DIR / "synth_summary.json").write_text(json.dumps(res1, indent=2), encoding="utf-8")
    print(f"\nALL SYNTH DONE in {time.time()-t0:.1f}s ({len(res1)} bricks)", flush=True)


if __name__ == "__main__":
    main()
