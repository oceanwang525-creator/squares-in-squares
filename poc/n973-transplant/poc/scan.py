#!/usr/bin/env python3
"""Run a focused n=973 dual-pocket plus motif transplant."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


POC_DIR = Path(__file__).resolve().parents[1]
ROOT = POC_DIR.parents[1]
SEARCH = ROOT / "poc" / "corner-pocket-chirality" / "poc" / "core_motif_fill.py"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline",
        type=Path,
        default=ROOT
        / "poc"
        / "pell_corrugation"
        / "data"
        / "border-scan"
        / "merged.csv",
    )
    parser.add_argument("--side", type=float, default=31.6666382096706)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--grid-step", type=float, default=0.25)
    parser.add_argument("--placements-per-motif", type=int, default=2)
    parser.add_argument("--motif-limit", type=int, default=200)
    parser.add_argument("--max-box-area", type=float, default=4.0)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--candidate-out", required=True, type=Path)
    args = parser.parse_args()

    command = [
        sys.executable,
        str(SEARCH),
        "--baseline",
        str(args.baseline),
        "--core-source",
        "971",
        "--core-mode",
        "tilted",
        "--side",
        str(args.side),
        "--target",
        "973",
        "--sources",
        "969,970,971,972,973",
        "--max-box-area",
        str(args.max_box_area),
        "--grid-step",
        str(args.grid_step),
        "--placements-per-motif",
        str(args.placements_per_motif),
        "--motif-limit",
        str(args.motif_limit),
        "--jobs",
        str(args.jobs),
        "--reach",
        "8",
        "--milp-limit",
        "3",
        "--summary",
        str(args.summary),
        "--candidate-out",
        str(args.candidate_out),
    ]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
