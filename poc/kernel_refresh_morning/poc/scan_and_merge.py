#!/usr/bin/env python3
"""Scan downloaded kernel CSVs and merge every strict per-n winner.

Chains the two shared tools:
- ``poc/post_resweep_refresh/poc/scan_refresh.py`` (float-bbox prefilter)
- ``poc/external_witness_refresh/poc/merge_witnesses.py`` (exact literal and
  pandas certification, minimal-dilation repair, row replacement)

The only work here is converting the scan's list rows into the merge's dict
rows, so one command reproduces the whole loop from a directory of CSVs.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCAN = REPO / "poc/post_resweep_refresh/poc/scan_refresh.py"
MERGE = REPO / "poc/external_witness_refresh/poc/merge_witnesses.py"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--incumbent", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--wins", required=True)
    args = parser.parse_args()

    subprocess.run([sys.executable, str(SCAN),
                    "--incumbent-report", args.report,
                    "--inputs", args.inputs,
                    "--candidates", args.candidates], check=True)

    raw = json.load(open(args.candidates, encoding="utf-8"))
    converted = {
        n: [dict(float_side=entry[0], source=entry[1], rows=entry[2])
            for entry in entries]
        for n, entries in raw.items()
    }
    with open(args.candidates, "w", encoding="utf-8") as handle:
        json.dump(converted, handle, indent=1)

    subprocess.run([sys.executable, str(MERGE),
                    "--incumbent", args.incumbent,
                    "--candidates", args.candidates,
                    "--output", args.output,
                    "--wins", args.wins], check=True)


if __name__ == "__main__":
    main()
