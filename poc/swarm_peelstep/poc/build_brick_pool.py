#!/usr/bin/env python3
"""Extract sub-bricks from win_n974, win_n912 (stepdown & gstep), and n=852,
then generate 1-row/1-col peeled & SLP-compressed variants in poc/swarm_peelstep/data/."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import peelstep_lib as pl  # noqa: E402


def load_win_json(p: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    d = json.loads(p.read_text(encoding="utf-8"))
    rs = sorted(d["rows"], key=lambda r: int(r["square_id"]))
    ang = np.array([float(r["angle"]) for r in rs], dtype=np.float64)
    if float(np.max(np.abs(ang))) > 6.3:
        ang = np.radians(ang)
    return pl.bl.normalize(
        np.array([float(r["x"]) for r in rs], dtype=np.float64),
        np.array([float(r["y"]) for r in rs], dtype=np.float64),
        ang,
    )


def main() -> None:
    t0 = time.time()
    pl.DATA_DIR.mkdir(parents=True, exist_ok=True)
    pl.DOCS_DIR.mkdir(parents=True, exist_ok=True)

    sources = [
        (
            "n974cp2",
            pl.WORKTREES_ROOT / "swarm-cleanpeel2-8a77ac94" / "poc" / "swarm_cleanpeel2" / "data" / "win_n974.json",
            [("TL", 9, 21, 6), ("BR", 22, 10, 7)],
        ),
        (
            "n912sd",
            pl.WORKTREES_ROOT / "swarm-stepdown-9d2222ea" / "poc" / "swarm_stepdown" / "data" / "win_n912.json",
            [("TL", 9, 21, 6), ("BR", 21, 9, 6)],
        ),
        (
            "n912gs",
            pl.WORKTREES_ROOT / "swarm-gstep-6d3db1b1" / "poc" / "swarm_gstep" / "data" / "win_n912.json",
            [("TL", 9, 21, 6), ("BR", 21, 9, 6)],
        ),
    ]

    extracted: list[dict] = []
    for tag, jp, specs in sources:
        if not jp.exists():
            print(f"WARN: missing {jp}", flush=True)
            continue
        x, y, th, L = load_win_json(jp)
        for corner, w, h, g in specs:
            b = pl.cp2.extract_corner_brick(x, y, th, L, corner, w, h, g, f"{tag}_{corner}_{w}x{h}_g{g}", margin=0.82)
            if b is not None:
                pl.save_brick_json(b, b["tag"])
                extracted.append(b)
                print(f"[EXTRACTED] {b['tag']}: {w}x{h}+{g} f={b['f']:.6f}", flush=True)

    # Extract n=852 from origin/main frontier
    fr852 = pl.cp2.fast_load_true_frontier([852], certify=False)[852]
    for corner, w, h, g in [("TL", 18, 10, 5), ("BR", 11, 20, 6)]:
        b = pl.cp2.extract_corner_brick(
            fr852["x"], fr852["y"], fr852["th"], fr852["L"],
            corner, w, h, g, f"n852_{corner}_{w}x{h}_g{g}", margin=0.82
        )
        if b is not None:
            pl.save_brick_json(b, b["tag"])
            extracted.append(b)
            print(f"[EXTRACTED] {b['tag']}: {w}x{h}+{g} f={b['f']:.6f}", flush=True)

    # Generate 1-row / 1-column peeled variants (w-1, h and w, h-1)
    # Deduplicate source bricks for peeling (keep n912gs over n912sd for identical shapes, plus n974 and n852)
    peel_bases = [b for b in extracted if not b["tag"].startswith("n912sd_")]
    peeled_best: dict[tuple[str, int, int, int], dict] = {}

    seed = 10900
    for b in peel_bases:
        for side in ("left", "right", "top", "bottom"):
            seed += 1
            pb = pl.cp2.peel_outer_unit_layer(b, side, tag=f"{b['tag']}_peel_{side}")
            if pb is None:
                continue
            cb = pl.cp2.compress_brick_slp(pb, n_trials=10, seed=seed)
            pen, _ = pl.cpp_eval_sat_pen_bbox(cb["x"], cb["y"], cb["th"])
            if pen > 1e-7:
                continue
            print(
                f"  [PEEL] {b['tag']} -> {side} ({cb['w']}x{cb['h']}+{cb['g']}): raw_f={pb['f']:.6f} -> slp_f={cb['f']:.6f}",
                flush=True,
            )
            key = (b["tag"], cb["w"], cb["h"], cb["g"])
            if key not in peeled_best or cb["f"] < peeled_best[key]["f"]:
                peeled_best[key] = cb

    # Save best peeled bricks and run 1 round of brick_mechwalk on promising ones (f < 0.725)
    summary = []
    for key in sorted(peeled_best):
        cb = peeled_best[key]
        name = f"{cb['tag']}"
        if cb["f"] < 0.725:
            wb = pl.cp2.brick_mechwalk(cb, name=name + "_walk", max_rounds=1, save=False)
            if wb["f"] < cb["f"] - 1e-7:
                print(f"  [WALK] {name}: {cb['f']:.6f} -> {wb['f']:.6f}", flush=True)
                cb = wb
                name = wb["tag"]
        pl.save_brick_json(cb, name)
        summary.append({"tag": name, "w": cb["w"], "h": cb["h"], "g": cb["g"], "f": round(cb["f"], 6)})

    (pl.DOCS_DIR / "peeled_bricks_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pooled = pl.load_all_pooled_bricks(top_per_shape=2)
    print(f"\nBuilt pool in {time.time()-t0:.1f}s: {len(extracted)} extracted, {len(summary)} peeled, {len(pooled)} total pooled oriented bricks", flush=True)


if __name__ == "__main__":
    main()
