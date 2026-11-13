#!/usr/bin/env python3
"""Render one packing CSV and report its largest uncovered components."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import probe_layout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--side", type=float, required=True)
    parser.add_argument("--scale", type=int, default=24)
    parser.add_argument("--minimum-gap-area", type=float, default=0.02)
    parser.add_argument("--png-out", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()

    with args.candidate.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        squares = np.asarray(
            [
                (float(row["x"]), float(row["y"]), float(row["angle"]))
                for row in reader
            ],
            dtype=np.float64,
        )
    image = probe_layout.rasterize(squares, args.side, args.scale)
    probe_layout.write_png(args.png_out, image)
    report = {
        "candidate": str(args.candidate.resolve()),
        "count": len(squares),
        "side": args.side,
        "tilted_count": int(np.count_nonzero(np.abs(squares[:, 2]) > 1e-8)),
        "gaps": probe_layout.gap_components(
            squares,
            args.side,
            args.scale,
            args.minimum_gap_area,
        ),
        "png": str(args.png_out.resolve()),
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"count={len(squares)} tilted={report['tilted_count']} "
        f"gaps={len(report['gaps'])}"
    )


if __name__ == "__main__":
    main()
