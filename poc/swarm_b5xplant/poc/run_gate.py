#!/usr/bin/env python3
"""15-Minute Falsification Gate for H40 (B5_10x20 / B4_18x9 Harvest & Broadcast on n = 793, 910, 911, 971, 972)."""
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

import b5_lib as bl  # noqa: E402


def main():
    t0 = time.time()
    gate_ns = [971, 972, 793, 910, 911]
    print(f"=== H40 Falsification Gate (n = {gate_ns}) ===", flush=True)
    packs, best_L = bl.fast_load_frontier(target_ns=gate_ns)
    bank = bl.load_b5_bank(packs, compress_b5_b4=True)
    print(f"Loaded frontier & {len(bank)} bricks in {time.time()-t0:.1f}s", flush=True)

    report = {}
    for n in gate_ns:
        print(f"\n[Gate] Target n={n}, L_inc={best_L[n]}", flush=True)
        cands = bl.find_b5_candidates(n, bank, float(best_L[n]), max_cands=4)
        if not cands:
            print(f"  No candidates for n={n}", flush=True)
            report[n] = {"n": n, "status": "NO_CANDS"}
            continue
        res = bl.eval_target(n, cands, best_L[n], n_threads=4)
        report[n] = res
        print(f"[Gate Summary n={n}] -> {res}", flush=True)

    bl.DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (bl.DOCS_DIR / "gate_report.json").write_text(json.dumps(report, indent=2))
    print(f"\n=== Gate Complete in {time.time()-t0:.1f}s ===", flush=True)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
