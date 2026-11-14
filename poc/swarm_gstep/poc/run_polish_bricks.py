#!/usr/bin/env python3
"""Fast Stage-3 brick synthesis (B6_10x20, B6_10x19, B7_9x21) + 1-round mechwalk polish."""
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


def _compress_and_save(name: str, b: dict) -> tuple[str, int, int, int, float]:
    out = gl.DATA_DIR / f"brick_{name}.json"
    if out.exists():
        bc = gl.load_brick_json(out, tag=name)
        return name, bc["w"], bc["h"], bc["g"], bc["f"]
    bc = gl.relax_and_compress_brick(
        b, f_targets=(b["f"] - 0.02, b["f"], b["f"] + 0.02),
        n_trials=6, seed=abs(hash(name)) % 100000,
    )
    if bc is None:
        return name, b["w"], b["h"], b["g"], b["f"]
    bc["tag"] = name
    gl.save_brick_json(bc, name)
    return name, bc["w"], bc["h"], bc["g"], bc["f"]


def _mechwalk_one(src_name: str, out_name: str, max_rounds: int = 1) -> tuple[str, float, float]:
    out = gl.DATA_DIR / f"brick_{out_name}.json"
    if out.exists():
        b_old = gl.load_brick_json(gl.DATA_DIR / f"brick_{src_name}.json")
        b_new = gl.load_brick_json(out)
        return out_name, b_old["f"], b_new["f"]
    b = gl.load_brick_json(gl.DATA_DIR / f"brick_{src_name}.json", tag=src_name)
    f0 = b["f"]
    bp = gl.brick_mechwalk(b, out_name, max_rounds=max_rounds, save=True)
    return out_name, f0, bp["f"]


def main() -> None:
    t0 = time.time()
    src = gl.load_source_bricks()
    b6 = src["B6_10x21_mw972_walk"]

    # Fast extra peels: B6_10x20 (g=+6), B6_10x19 (g=+6), B7_9x21 (g=+7)
    extra_jobs = {}
    for side in ("top", "bottom"):
        p = gl.peel_outer_unit_layer(b6, side, f"B6_10x20_{side}")
        if p is not None:
            extra_jobs[f"B6_10x20_{side}"] = p

    b7_9x22_p = gl.DATA_DIR / "brick_B7_9x22_peel_left.json"
    if b7_9x22_p.exists():
        b7_9x22 = gl.load_brick_json(b7_9x22_p)
        for side in ("top", "bottom"):
            p = gl.peel_outer_unit_layer(b7_9x22, side, f"B7_9x21_{side}")
            if p is not None:
                extra_jobs[f"B7_9x21_{side}"] = p

    print(f"Extra peel compress jobs: {len(extra_jobs)}", flush=True)
    with ProcessPoolExecutor(max_workers=2) as ex:
        futs = [ex.submit(_compress_and_save, nm, b) for nm, b in extra_jobs.items()]
        for fut in as_completed(futs):
            nm, w, h, g, f = fut.result()
            print(f"  [EXTRA] {nm:24s}: {w}x{h} g=+{g} f={f:.6f}", flush=True)

    # Also 10x19 from best B6_10x20
    cands_10x20 = sorted(gl.DATA_DIR.glob("brick_B6_10x20_*.json"))
    if cands_10x20:
        b6_10x20 = min((gl.load_brick_json(p) for p in cands_10x20), key=lambda x: x["f"])
        j2 = {}
        for side in ("top", "bottom"):
            p = gl.peel_outer_unit_layer(b6_10x20, side, f"B6_10x19_{side}")
            if p is not None:
                j2[f"B6_10x19_{side}"] = p
        with ProcessPoolExecutor(max_workers=2) as ex:
            futs = [ex.submit(_compress_and_save, nm, b) for nm, b in j2.items()]
            for fut in as_completed(futs):
                nm, w, h, g, f = fut.result()
                print(f"  [EXTRA2] {nm:23s}: {w}x{h} g=+{g} f={f:.6f}", flush=True)

    if "--no-mw" in sys.argv:
        print(f"Done extra peels in {time.time()-t0:.1f}s", flush=True)
        return

    # 1-round mechwalk on thinnest new bricks per useful shape
    by_shape: dict[tuple[int, int, int], tuple[str, float]] = {}
    for p in sorted(gl.DATA_DIR.glob("brick_*.json")):
        if "_mw" in p.stem:
            continue
        b = gl.load_brick_json(p)
        key = (b["w"], b["h"], b["g"])
        nm = p.stem.replace("brick_", "")
        if key not in by_shape or b["f"] < by_shape[key][1]:
            by_shape[key] = (nm, b["f"])

    only_shapes = {(9, 21, 6), (9, 20, 6), (11, 20, 6), (10, 20, 6), (10, 21, 7), (9, 22, 7)}
    mw_targets = []
    for key in sorted(by_shape):
        if key not in only_shapes:
            continue
        nm, f0 = by_shape[key]
        if f0 < 0.725 and nm != "B7_10x22_vstrip":
            mw_targets.append((nm, f"{nm}_mw1"))

    print(f"Running 1-round mechwalk on {len(mw_targets)} top shapes...", flush=True)
    with ProcessPoolExecutor(max_workers=2) as ex:
        futs = [ex.submit(_mechwalk_one, s, o, 1) for s, o in mw_targets]
        for fut in as_completed(futs):
            out_nm, f0, f1 = fut.result()
            print(f"  [MW1] {out_nm:28s}: {f0:.6f} -> {f1:.6f} (df={f1-f0:+.6f})", flush=True)

    print(f"Done polish in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
