#!/usr/bin/env python3
"""15-Minute Falsification Gate for H35 (Interlocking Corner-Shared Pinwheel + Masked Seam Melt)."""
from __future__ import annotations

import json
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

import interlock_lib as il  # noqa: E402
import interlock_eval as ie  # noqa: E402


def main():
    t0 = time.time()
    gate_ns = [851, 795, 684]
    print("=== H35 15-Minute Falsification Gate (n = 851, 795, 684) ===", flush=True)
    packs, best_L = il.fast_load_frontier(target_ns=gate_ns)
    bricks = il.extract_all_bricks(packs, compress_key_bricks=True)
    print(f"Loaded frontier & extracted {len(bricks)} bricks in {time.time()-t0:.1f}s", flush=True)

    report = {}
    for n in gate_ns:
        print(f"\n[Gate] Target n={n}, L_inc={best_L[n]}", flush=True)
        cands = ie.find_candidates_for_n(n, bricks, float(best_L[n]), max_pairs=5)
        if not cands:
            print(f"  No interlocking candidates found for n={n}", flush=True)
            report[n] = {"n": n, "status": "NO_CANDS"}
            continue
        res = ie.eval_target(n, cands, best_L[n], n_threads=3)
        report[n] = res
        print(f"[Gate Summary n={n}] -> {res}", flush=True)

    il.DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (il.DOCS_DIR / "gate_report.json").write_text(json.dumps(report, indent=2))
    print(f"\n=== Gate Complete in {time.time()-t0:.1f}s ===", flush=True)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
