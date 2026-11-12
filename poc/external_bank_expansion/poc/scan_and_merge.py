#!/usr/bin/env python3
"""Scan a bank tree for strict per-n witnesses and merge the winners.

Reuses the harvesters from ``poc/external_witness_refresh`` (rational corners,
center-basis witnesses, algebraic-number-field scalars, ``.cert`` files, npz
banks, generic CSV/JSON layouts) and its certifying merge step, so the exact
same gate applies: literal decimal strings and actual pandas 3.1.0 transport
must both improve.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
SCAN = REPO / "poc/external_witness_refresh/poc/scan_new.py"
MERGE = REPO / "poc/external_witness_refresh/poc/merge_witnesses.py"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repos", required=True, help="root of the cloned banks")
    parser.add_argument("--incumbent", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--wins", required=True)
    parser.add_argument("--tol", type=float, default=1e-13,
                        help="float prefilter margin below the incumbent L")
    args = parser.parse_args()

    subprocess.run([sys.executable, str(SCAN),
                    "--repo-root", args.repos,
                    "--incumbent-report", args.report,
                    "--candidates", args.candidates,
                    "--since-regex", ".",
                    "--known-best",
                    "--tol", str(args.tol)], check=True)
    subprocess.run([sys.executable, str(MERGE),
                    "--incumbent", args.incumbent,
                    "--candidates", args.candidates,
                    "--output", args.output,
                    "--wins", args.wins], check=True)


if __name__ == "__main__":
    main()
