#!/usr/bin/env python3
"""Checkpointed Detached Scale-Up Sweep for H35 (Interlocking Corner-Shared Sub-Brick Pinwheels)."""
from __future__ import annotations

import json
import math
import os
import sys
import time
from decimal import Decimal
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import interlock_lib as il  # noqa: E402
import interlock_eval as ie  # noqa: E402

SCALE_TARGETS = [
    794, 739, 852, 633, 584, 537, 738, 685, 634, 585, 741, 688, 637,
]
CKPT_PATH = il.DOCS_DIR / "scale_ckpt.json"


def load_ckpt() -> dict:
    if CKPT_PATH.exists():
        return json.loads(CKPT_PATH.read_text())
    return {"targets": {}}


def save_ckpt(ckpt: dict) -> None:
    il.DOCS_DIR.mkdir(parents=True, exist_ok=True)
    il.frontier_io.auto_union.atomic(CKPT_PATH, json.dumps(ckpt, indent=2).encode())


def discover_extra_targets(bricks: dict, packs: dict, best_L: dict, existing: set[int]) -> list[tuple[float, int]]:
    """Find additional n in 400..1000 where an interlocking pair has estimated f_max < f_inc - 0.002."""
    extras = []
    for n in range(400, 1001):
        if n in existing or n not in best_L:
            continue
        m = int(math.isqrt(n))
        while m * m > n:
            m -= 1
        r = n - m * m
        if r <= 1:
            continue
        f_inc = float(best_L[n]) - m
        # Quick check if any candidate exists with f_max < f_inc - 0.002
        cands = ie.find_candidates_for_n(n, bricks, float(best_L[n]), max_pairs=1)
        if cands:
            df = max(cands[0]["b1"]["f"], cands[0]["b2"]["f"]) - f_inc
            if df < -0.002:
                extras.append((df, n))
    extras.sort()
    return extras


def main():
    t0 = time.time()
    print("=== H35 Scale-Up Sweep ===", flush=True)
    packs, best_L = il.fast_load_frontier(target_ns=SCALE_TARGETS)
    bricks = il.extract_all_bricks(packs, compress_key_bricks=True)
    print(f"Loaded frontier & extracted {len(bricks)} bricks in {time.time()-t0:.1f}s", flush=True)

    extras = discover_extra_targets(bricks, packs, best_L, set(SCALE_TARGETS) | {851, 795, 684})
    extra_ns = [n for _, n in extras[:15]]
    print(f"Discovered {len(extras)} extra targets (gap < -0.002); adding top {len(extra_ns)}: {extra_ns}", flush=True)

    all_targets = SCALE_TARGETS + extra_ns
    # Re-load exact Decimal L_inc for all targets
    packs, best_L = il.fast_load_frontier(target_ns=all_targets)

    ckpt = load_ckpt()
    consecutive_misses = 0

    for idx, n in enumerate(all_targets):
        key = str(n)
        if key in ckpt["targets"] and ckpt["targets"][key].get("status") in ("WIN", "OK", "NO_CANDS"):
            print(f"[Skip] n={n} already in ckpt: {ckpt['targets'][key]}", flush=True)
            continue

        print(f"\n[Scale {idx+1}/{len(all_targets)}] n={n}, L_inc={best_L[n]}", flush=True)
        cands = ie.find_candidates_for_n(n, bricks, float(best_L[n]), max_pairs=6)
        if not cands:
            print(f"  No interlocking candidates for n={n}", flush=True)
            ckpt["targets"][key] = {"n": n, "status": "NO_CANDS"}
            save_ckpt(ckpt)
            continue

        res = ie.eval_target(n, cands, best_L[n], n_threads=3)
        ckpt["targets"][key] = res
        save_ckpt(ckpt)
        print(f"  [Result n={n}] -> {res}", flush=True)

        if res.get("status") == "WIN":
            consecutive_misses = 0
        else:
            consecutive_misses += 1
            if consecutive_misses >= 12:
                print(f"[Stop] {consecutive_misses} consecutive non-wins; stopping per protocol.", flush=True)
                break

    print(f"\n=== H35 Scale-Up Complete in {time.time()-t0:.1f}s ===", flush=True)


if __name__ == "__main__":
    main()
