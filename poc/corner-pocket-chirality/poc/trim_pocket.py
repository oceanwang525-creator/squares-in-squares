#!/usr/bin/env python3
"""Remove one boundary layer from an extracted pocket and write a summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import chirality_scan as cs


EDGES = {
    "xmax": (0, False),
    "xmin": (0, True),
    "ymax": (1, False),
    "ymin": (1, True),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--drop", type=int, required=True)
    parser.add_argument("--edge", choices=tuple(EDGES), required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    report = json.loads(args.source.read_text(encoding="utf-8"))
    squares = np.asarray(report["squares"], dtype=np.float64).reshape(-1, 3)
    if not 0 <= args.drop < len(squares):
        raise ValueError("--drop must leave at least one square")
    axis, ascending = EDGES[args.edge]
    order = np.argsort(squares[:, axis], kind="stable")
    keep = order[args.drop:] if ascending else order[: len(squares) - args.drop]
    kept = squares[np.sort(keep)]
    normalized, _ = cs.normalize_bundle(
        kept, np.empty((0, 3), dtype=np.float64)
    )
    xlo, xhi, ylo, yhi = cs.polygon_bounds(normalized)
    result = {
        "source_n": int(report["source_n"]),
        "component_id": int(report["component_id"]),
        "count": len(normalized),
        "source_summary": str(args.source.resolve()),
        "drop": args.drop,
        "edge": args.edge,
        "actual_width": float(xhi.max() - xlo.min()),
        "actual_height": float(yhi.max() - ylo.min()),
        "status": "ok",
        "squares": [
            [float(x), float(y), float(angle)]
            for (x, y), angle in zip(normalized[:, :2], normalized[:, 2])
        ],
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"source={result['source_n']}:{result['component_id']} "
        f"count={result['count']} edge={args.edge} drop={args.drop} "
        f"bounds={result['actual_width']:.9f}x{result['actual_height']:.9f}"
    )


if __name__ == "__main__":
    main()
