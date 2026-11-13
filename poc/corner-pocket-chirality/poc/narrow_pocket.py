#!/usr/bin/env python3
"""Narrow one extracted tilted pocket inside a fixed rectangle."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import chirality_scan as cs
import scan


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "poc" / "mondrian-narrow" / "poc"))
import narrowscan  # noqa: E402


def parse_pair(value: str) -> tuple[int, int]:
    parts = tuple(int(part) for part in value.split(","))
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("expected SOURCE,COMPONENT")
    return parts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--sources", required=True)
    parser.add_argument("--pocket", type=parse_pair, required=True)
    parser.add_argument("--reduce-width", type=float, default=0.0)
    parser.add_argument("--reduce-height", type=float, default=0.0)
    parser.add_argument("--stages", type=int, default=20)
    parser.add_argument("--fixed-angles", action="store_true")
    parser.add_argument("--trust", type=float, default=0.5)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    sources = tuple(int(value) for value in args.sources.split(","))
    packings = scan.read_groups(args.baseline)
    pockets = cs.extract_pockets(packings, sources, 80, 0.1, 1.5)
    pocket = next(
        item
        for item in pockets
        if (item.source_n, item.component_id) == args.pocket
    )
    squares = pocket.squares
    angles = squares[:, 2].copy()
    centers = squares[:, :2].copy()
    radius = narrowscan.extents(angles)
    width = float((centers[:, 0] + radius).max())
    height = float((centers[:, 1] + radius).max())
    target_width = width - args.reduce_width
    target_height = height - args.reduce_height
    piece = narrowscan.Piece(
        crop=SimpleNamespace(capacity=len(squares)),
        width=width,
        height=height,
        centers=centers,
        angles=angles,
    )
    report: dict[str, object] = {
        "source_n": pocket.source_n,
        "component_id": pocket.component_id,
        "count": len(squares),
        "original_width": width,
        "original_height": height,
        "target_width": target_width,
        "target_height": target_height,
        "stages": args.stages,
    }
    try:
        if args.fixed_angles:
            centers = narrowscan.fixed_wall_lp(
                centers,
                angles,
                (0.0, target_width, 0.0, target_height),
                args.trust,
                repair=True,
            )
            solved = (
                None
                if centers is None
                else (centers, angles, target_width, target_height)
            )
        else:
            solved = narrowscan.narrow_piece(
                piece,
                target_width,
                target_height,
                args.stages,
            )
    except Exception as error:  # noqa: BLE001
        solved = None
        report["error"] = f"{type(error).__name__}: {error}"
    if solved is None:
        report["status"] = "infeasible"
    else:
        new_centers, new_angles, actual_width, actual_height = solved
        valid = narrowscan.wall_state_ok(
            new_centers,
            new_angles,
            (0.0, target_width, 0.0, target_height),
            tol=1e-9,
        )
        report.update(
            {
                "status": "ok" if valid else "invalid",
                "actual_width": actual_width,
                "actual_height": actual_height,
                "squares": [
                    [float(x), float(y), float(angle)]
                    for (x, y), angle in zip(new_centers, new_angles)
                ],
            }
        )
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"pocket={pocket.source_n}:{pocket.component_id} count={len(squares)} "
        f"orig={width:.9f}x{height:.9f} "
        f"target={target_width:.9f}x{target_height:.9f} "
        f"status={report['status']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
