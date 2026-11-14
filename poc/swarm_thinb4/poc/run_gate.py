#!/usr/bin/env python3
"""H52 Step 1: Extract & compress compact g=+4/+3/+5/+6 sub-bricks and run 15-min falsification gate on {684, 849, 737}."""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import thinb4_lib as tl  # noqa: E402


def build_gate_candidates(n: int, bank: dict[str, dict], L_inc: float) -> list[dict]:
    """Build exact H52 pair assemblies for n, prioritizing zero-overlap pairs."""
    m = int(math.isqrt(n))
    r = n - m * m
    f_inc = L_inc - m
    cands: list[dict] = []
    seen = set()

    bricks = list(bank.values())
    pair_list = []
    for b1 in bricks:
        if b1["g"] >= r or b1["w"] >= m or b1["h"] >= m:
            continue
        for b2 in bricks:
            if b1["g"] + b2["g"] != r or b2["w"] >= m or b2["h"] >= m:
                continue
            kx = b1["w"] + b2["w"] - m
            ky = b1["h"] + b2["h"] - m
            if kx > 2 or ky > 2 or (kx > 0 and ky > 0):
                continue
            f_max = max(b1["f"], b2["f"])
            f_tight = max(b1["fx"], b2["fy"])
            if min(f_max, f_tight) > f_inc + 0.015:
                continue
            # Canonical dedup up to swap + transpose
            k1 = (b1["tag"].replace("_T", ""), b1["w"], b1["h"], b1["g"])
            k2 = (b2["tag"].replace("_T", ""), b2["w"], b2["h"], b2["g"])
            k1t = (b1["tag"].replace("_T", ""), b1["h"], b1["w"], b1["g"])
            k2t = (b2["tag"].replace("_T", ""), b2["h"], b2["w"], b2["g"])
            sig = tuple(sorted((k1, k2)))
            sigt = tuple(sorted((k1t, k2t)))
            if sig in seen or sigt in seen:
                continue
            seen.add(sig)
            seen.add(sigt)
            zw_pen = 0.0 if (kx <= 0 and ky <= 0) else (0.010 + 0.005 * max(0, kx) + 0.005 * max(0, ky))
            score = (0 if (kx <= 0 and ky <= 0) else 1, min(f_max, f_tight) + zw_pen, f_max)
            pair_list.append((score, b1, b2))

    pair_list.sort(key=lambda z: z[0])
    for _, b1, b2 in pair_list[:6]:
        asms = tl.assemble_pair_best(m, r, b1, b2, shift_fracs=(0.0, 0.5, 0.8))
        cands.extend(asms)

    cands.sort(key=lambda c: (0 if c["pen0"] < 1e-5 else 1, c["L"] + 0.05 * c["pen0"]))
    return cands[:6]


def main() -> None:
    t0 = time.time()
    gate_ns = [684, 849, 737]
    src_ns = [632, 684, 736, 737, 793, 848, 849, 908, 909, 969, 972]
    print("=== Hypothesis H52 Step 1: Extraction, Standalone Compression & 15-Min Gate ===", flush=True)
    packs, best_L = tl.fast_load_frontier(target_ns=sorted(set(gate_ns + src_ns)))
    print(f"Loaded frontier in {time.time() - t0:.1f}s.", flush=True)

    bank = tl.build_h52_bank(packs, compress=True)
    print(f"Built compressed H52 bank with {len(bank)} oriented bricks in {time.time() - t0:.1f}s.", flush=True)

    results = {}
    for n in gate_ns:
        m = int(math.isqrt(n))
        r = n - m * m
        print(f"\n[Gate] Target n={n} (m={m}, r=+{r}) L_inc={best_L[n]}", flush=True)
        cands = build_gate_candidates(n, bank, float(best_L[n]))
        if not cands:
            print(f"  No candidates for n={n}", flush=True)
            results[str(n)] = {"n": n, "status": "NO_CANDS"}
            continue
        res = tl.eval_target(n, cands, best_L[n], n_threads=4, n_seeds=20, do_mechwalk_on_win=False)
        results[str(n)] = res
        print(f"  [Gate Result n={n}] -> {res}", flush=True)

    tl.DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (tl.DOCS_DIR / "gate_report.json").write_text(json.dumps(results, indent=2))
    print(f"\n=== GATE COMPLETE in {time.time() - t0:.1f}s ===", flush=True)


if __name__ == "__main__":
    main()
