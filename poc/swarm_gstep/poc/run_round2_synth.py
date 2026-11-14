#!/usr/bin/env python3
"""Round-2 targeted brick synthesis (B6_9x20, B7_10x20, B7_10x19, B7_9x21, B8_11x20, B8_12x19)."""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gstep_lib as gl  # noqa: E402

REPO_ROOT = gl.REPO_ROOT


def _load_json_brick(path: Path, tag: str) -> dict:
    d = json.loads(path.read_text(encoding="utf-8"))
    return gl.make_brick(
        int(d["w"]), int(d["h"]), int(d["g"]),
        np.array(d["x"], dtype=np.float64),
        np.array(d["y"], dtype=np.float64),
        np.array(d["th"], dtype=np.float64),
        tag=tag,
    )


def _job(spec: tuple[str, dict, tuple[float, ...], int]) -> tuple[str, int, int, int, float] | None:
    name, cand, sched, seed = spec
    out = gl.DATA_DIR / f"brick_{name}.json"
    if out.exists():
        b = gl.load_brick_json(out, tag=name)
        return name, b["w"], b["h"], b["g"], b["f"]
    comp = gl.relax_and_compress_brick(cand, f_targets=sched, n_trials=8, seed=seed)
    if comp is None:
        return None
    comp["tag"] = name
    gl.save_brick_json(comp, name)
    return name, comp["w"], comp["h"], comp["g"], comp["f"]


def main() -> None:
    t0 = time.time()
    src = gl.load_source_bricks()
    b7x_dir = REPO_ROOT / "poc" / "swarm_b7xplant" / "data"

    jobs = []
    # 1. B6_9x20 from B6_10x20_mw972_top (0.68649 -> zero overlap n=912!)
    b6_10x20 = src["B6_10x20_mw972_top"]
    for side in ("left", "right"):
        p = gl.peel_outer_unit_layer(b6_10x20, side, f"B6_9x20_10x20_{side}")
        if p is not None:
            jobs.append((f"B6_9x20_10x20_{side}", p, (0.688, 0.700, 0.715), 3100 + len(jobs)))

    # 2. B7_10x20 and B7_9x21 from B7_10x21_vs_k5 (0.71864) and B7_9x22_vs_k3 (0.71583)
    p_10x21 = gl.DATA_DIR / "brick_B7_10x21_vs_k5.json"
    if p_10x21.exists():
        b7_10x21 = gl.load_brick_json(p_10x21, "B7_10x21_vs_k5")
        for side in ("top", "bottom"):
            p = gl.peel_outer_unit_layer(b7_10x21, side, f"B7_10x20_vs_{side}")
            if p is not None:
                jobs.append((f"B7_10x20_vs_{side}", p, (0.700, 0.718, 0.735), 3200 + len(jobs)))

    p_9x22 = gl.DATA_DIR / "brick_B7_9x22_vs_k3.json"
    if p_9x22.exists():
        b7_9x22 = gl.load_brick_json(p_9x22, "B7_9x22_vs_k3")
        for side in ("top", "bottom"):
            p = gl.peel_outer_unit_layer(b7_9x22, side, f"B7_9x21_vs_{side}")
            if p is not None:
                jobs.append((f"B7_9x21_vs_{side}", p, (0.710, 0.722, 0.736), 3300 + len(jobs)))
        for side in ("left", "right"):
            p = gl.peel_outer_unit_layer(b7_9x22, side, f"B7_8x22_vs_{side}")
            if p is not None:
                jobs.append((f"B7_8x22_vs_{side}", p, (0.710, 0.724, 0.738), 3350 + len(jobs)))

    # 2b. B6_23x8, B6_22x8 from B6_24x8_b8_slp (0.67774) and B7_8x23 from 8x22
    p_24x8 = b7x_dir / "brick_B6_24x8_b8_slp.json"
    if p_24x8.exists():
        b6_24x8 = _load_json_brick(p_24x8, "B6_24x8")
        for side in ("left", "right"):
            p = gl.peel_outer_unit_layer(b6_24x8, side, f"B6_23x8_{side}")
            if p is not None:
                jobs.append((f"B6_23x8_{side}", p, (0.675, 0.690, 0.708), 3380 + len(jobs)))
        b6_8x24 = gl.bl.transform_brick(b6_24x8, "T")
        b6_8x24["tag"] = "B6_8x24"
        for tag, cand in gl.splice_vertical_strip(b6_8x24)[:2]:
            nm = f"B7_8x25_vs_{tag.split('_')[-1]}"
            jobs.append((nm, cand, (0.680, 0.698, 0.718), 3390 + len(jobs)))

    # 3. B7_10x19, B7_11x18, B8_11x20 from B7_11x19_n975_cut (0.72154)
    p_11x19 = b7x_dir / "brick_B7_11x19_n975_cut.json"
    if p_11x19.exists():
        b7_11x19 = _load_json_brick(p_11x19, "B7_11x19")
        for side in ("left", "right"):
            p = gl.peel_outer_unit_layer(b7_11x19, side, f"B7_10x19_{side}")
            if p is not None:
                jobs.append((f"B7_10x19_{side}", p, (0.710, 0.724, 0.740), 3400 + len(jobs)))
        for side in ("top", "bottom"):
            p = gl.peel_outer_unit_layer(b7_11x19, side, f"B7_11x18_{side}")
            if p is not None:
                jobs.append((f"B7_11x18_{side}", p, (0.712, 0.726, 0.742), 3500 + len(jobs)))
        for tag, cand in gl.splice_vertical_strip(b7_11x19)[:2]:
            nm = f"B8_11x20_vs_{tag.split('_')[-1]}"
            jobs.append((nm, cand, (0.718, 0.732, 0.748), 3600 + len(jobs)))

    # 4. B8_11x20 and B8_12x19 from B8_12x20_n914_cut (0.73121)
    p_12x20 = b7x_dir / "brick_B8_12x20_n914_cut.json"
    if p_12x20.exists():
        b8_12x20 = _load_json_brick(p_12x20, "B8_12x20")
        for side in ("left", "right"):
            p = gl.peel_outer_unit_layer(b8_12x20, side, f"B8_11x20_peel_{side}")
            if p is not None:
                jobs.append((f"B8_11x20_peel_{side}", p, (0.720, 0.734, 0.750), 3700 + len(jobs)))
        for side in ("top", "bottom"):
            p = gl.peel_outer_unit_layer(b8_12x20, side, f"B8_12x19_peel_{side}")
            if p is not None:
                jobs.append((f"B8_12x19_peel_{side}", p, (0.722, 0.736, 0.752), 3800 + len(jobs)))

    # 4b. B8_9x22 from B7_9x21_bottom (0.72580 -> w=9, g=+8!)
    p_b7_9x21 = gl.DATA_DIR / "brick_B7_9x21_bottom.json"
    if p_b7_9x21.exists():
        b7_9x21 = gl.load_brick_json(p_b7_9x21, "B7_9x21")
        for tag, cand in gl.splice_vertical_strip(b7_9x21)[:2]:
            nm = f"B8_9x22_vs_{tag.split('_')[-1]}"
            jobs.append((nm, cand, (0.718, 0.730, 0.745), 3900 + len(jobs)))

    print(f"Round-2 Synth: {len(jobs)} jobs (max_workers=2)...", flush=True)
    with ProcessPoolExecutor(max_workers=2) as pool:
        futs = [pool.submit(_job, j) for j in jobs]
        for fut in as_completed(futs):
            r = fut.result()
            if r is not None:
                nm, w, h, g, f = r
                print(f"  [R2SYNTH] {nm:24s}: {w}x{h} g=+{g} f={f:.6f}", flush=True)
    print(f"Done Round-2 Synth in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
