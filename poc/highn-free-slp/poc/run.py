#!/usr/bin/env python3
"""Run collaborator kink-free SLP on Codex's n=600..1000 range."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


POC_DIR = Path(__file__).resolve().parents[1]
ROOT = POC_DIR.parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", required=True, type=Path)
    parser.add_argument("--out-rows", required=True, type=Path)
    parser.add_argument("--out-report", required=True, type=Path)
    parser.add_argument("--jobs", default="32")
    parser.add_argument("--iterations", default="30")
    parser.add_argument("--trust", default="0.002")
    args = parser.parse_args()

    search = ROOT / "poc" / "claude-midn" / "poc" / "free_slp.py"
    if not search.exists():
        raise SystemExit(
            "missing collaborator search script: "
            f"{search}"
        )

    command = [
        sys.executable,
        str(search),
        str(args.submission),
        "600-1000",
        args.jobs,
        str(args.out_rows),
        str(args.out_report),
        args.iterations,
        args.trust,
    ]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
