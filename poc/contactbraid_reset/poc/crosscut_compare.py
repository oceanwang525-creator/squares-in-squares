#!/usr/bin/env python3
"""Bounded strip-deletion comparison on the same braid-reset targets."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
ANGLE_POC = ROOT / "poc" / "swarm_angles" / "poc"
if str(ANGLE_POC) not in sys.path:
    sys.path.insert(0, str(ANGLE_POC))

import idea21_seam_cross as i21  # noqa: E402
from utils import load_incumbent_and_frontier  # noqa: E402

DOCS = Path(__file__).resolve().parents[1] / "docs"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=int, nargs="+", default=[592, 928])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    frame = pd.read_csv(i21.MERGED_CSV)
    mats = {int(n): i21._norm(sub) for n, sub in frame.groupby("n")}
    front = load_incumbent_and_frontier(args.targets)
    results = {}
    for n in args.targets:
        result = i21.solve_target(n, mats, front, args.threads, args.quick)
        results[str(n)] = result
        DOCS.mkdir(parents=True, exist_ok=True)
        (DOCS / f"crosscut_n{n}.json").write_text(json.dumps(result, indent=2) + "\n")
    (DOCS / "crosscut_summary.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
