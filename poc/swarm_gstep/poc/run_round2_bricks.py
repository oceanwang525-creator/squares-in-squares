#!/usr/bin/env python3
"""Round-2 targeted synthesis (vstrip on B6_10x20, B6_9x21) + brick_mechwalk polish."""
from __future__ import annotations

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


def _compress_cand(name: str, cand: dict, f_sched: tuple[float, ...], seed: int) -> tuple[str, int, int, int, float] | None:
    out_p = gl.DATA_DIR / f"brick_{name}.json"
    if out_p.exists():
        b = gl.load_brick_json(out_p, name)
        return name, b["w"], b["h"], b["g"], b["f"]
    comp = gl.relax_and_compress_brick(cand, f_targets=f_sched, n_trials=8, seed=seed)
    if comp is None:
        return None
    comp["tag"] = name
    gl.save_brick_json(comp, name)
    return name, comp["w"], comp["h"], comp["g"], comp["f"]


def _mechwalk_job(src_path: str, out_name: str, rounds: int) -> tuple[str, float, float]:
    out_p = gl.DATA_DIR / f"brick_{out_name}.json"
    b = gl.load_brick_json(Path(src_path), tag=out_name)
    f0 = b["f"]
    if out_p.exists():
        bn = gl.load_brick_json(out_p, tag=out_name)
        return out_name, f0, bn["f"]
    bp = gl.brick_mechwalk(b, out_name, max_rounds=rounds, save=True)
    return out_name, f0, bp["f"]


def main() -> None:
    t0 = time.time()
    src = gl.load_source_bricks()

    # Step 1: vstrip surgery on B6_10x20_mw972_top -> B7_10x21, and B6_9x21_peel_left -> B7_9x22
    jobs = []
    b6_10x20 = src["B6_10x20_mw972_top"]
    for tag, cand in gl.splice_vertical_strip(b6_10x20)[:3]:
        nm = f"B7_10x21_vs_{tag.split('_')[-1]}"
        jobs.append((nm, cand, (0.695, 0.708, 0.722), 1000 + len(jobs)))

    p_9x21 = gl.DATA_DIR / "brick_B6_9x21_peel_left.json"
    if p_9x21.exists():
        b6_9x21 = gl.load_brick_json(p_9x21, tag="B6_9x21")
        for tag, cand in gl.splice_vertical_strip(b6_9x21)[:2]:
            nm = f"B7_9x22_vs_{tag.split('_')[-1]}"
            jobs.append((nm, cand, (0.700, 0.715, 0.728), 2000 + len(jobs)))

    print(f"Step 1: {len(jobs)} vstrip synthesis jobs...", flush=True)
    with ProcessPoolExecutor(max_workers=2) as pool:
        futs = [pool.submit(_compress_cand, nm, c, fs, sd) for nm, c, fs, sd in jobs]
        for fut in as_completed(futs):
            res = fut.result()
            if res is not None:
                nm, w, h, g, f = res
                print(f"  [SYNTH] {nm:24s}: {w}x{h} g=+{g} f={f:.6f}", flush=True)

    # Step 2: Pick best per target shape and run 2-round brick_mechwalk
    mw_specs: list[tuple[Path, str, int]] = []
    # 9x20 (g=+6): prefer from9x21_bottom or bottom
    for cand_nm in ("brick_B6_9x20_from9x21_bottom.json", "brick_B6_9x20_from9x18_bottom.json"):
        cp = gl.DATA_DIR / cand_nm
        if cp.exists():
            mw_specs.append((cp, cp.stem.replace("brick_", "") + "_mw2", 2))
            break

    # Best B7_10x21
    b7_10x21_list = sorted(
        (gl.load_brick_json(p), p) for p in gl.DATA_DIR.glob("brick_B7_10x21*.json") if "_mw" not in p.stem
    ) if False else []
    cands_10x21 = []
    for p in gl.DATA_DIR.glob("brick_B7_10x21*.json"):
        if "_mw" not in p.stem:
            b = gl.load_brick_json(p)
            cands_10x21.append((b["f"], p))
    if cands_10x21:
        cands_10x21.sort()
        mw_specs.append((cands_10x21[0][1], cands_10x21[0][1].stem.replace("brick_", "") + "_mw2", 2))

    # Best B7_9x22, B7_9x21, B6_10x20, B6_9x21
    for pat in ("brick_B7_9x22*.json", "brick_B7_9x21*.json", "brick_B6_10x20_bottom.json", "brick_B6_9x21_peel_left.json"):
        cands = []
        for p in gl.DATA_DIR.glob(pat):
            if "_mw" not in p.stem:
                b = gl.load_brick_json(p)
                cands.append((b["f"], p))
        if cands:
            cands.sort()
            mw_specs.append((cands[0][1], cands[0][1].stem.replace("brick_", "") + "_mw2", 2))

    print(f"Step 2: {len(mw_specs)} brick_mechwalk jobs (max_workers=2)...", flush=True)
    with ProcessPoolExecutor(max_workers=2) as pool:
        futs = [pool.submit(_mechwalk_job, str(p), out_nm, rnds) for p, out_nm, rnds in mw_specs]
        for fut in as_completed(futs):
            nm, f0, f1 = fut.result()
            print(f"  [MW2] {nm:30s}: {f0:.6f} -> {f1:.6f} (df={f1-f0:+.6f})", flush=True)

    print(f"Round-2 bricks complete in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
