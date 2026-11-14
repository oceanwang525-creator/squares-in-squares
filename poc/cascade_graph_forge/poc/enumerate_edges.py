#!/usr/bin/env python3
"""Rank donor-to-child L-border peel edges over the current portfolio."""
from __future__ import annotations

import csv
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

REPO = Path(__file__).resolve().parents[3]
HOMO_POC = REPO / "poc" / "swarm_homocasc2" / "poc"
if str(HOMO_POC) not in sys.path:
    sys.path.insert(0, str(HOMO_POC))

import homocasc2_lib as hc  # noqa: E402

MERGED_CSV = (
    REPO / "poc" / "pell_corrugation" / "data" / "border-scan" / "merged.csv"
)
OUT = Path(__file__).resolve().parents[1] / "data" / "edge_ranking.json"


def load_groups() -> dict[int, list[dict[str, str]]]:
    groups: dict[int, list[dict[str, str]]] = {}
    with MERGED_CSV.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            n = int(row["n"])
            groups.setdefault(n, []).append(
                {
                    "square_id": row["square_id"],
                    "x": hc._strip_s(row["x"]),
                    "y": hc._strip_s(row["y"]),
                    "angle": hc._strip_s(row["angle"]),
                }
            )
    return groups


def side_length(rows: list[dict[str, str]]) -> float:
    x, y, theta, _ = hc._rows_to_arrays(rows)
    extent = np.abs(np.cos(theta)) + np.abs(np.sin(theta))
    return float(max(np.ptp(x) + extent.max(), np.ptp(y) + extent.max()))


def main() -> None:
    groups = load_groups()
    sides = {n: side_length(rows) for n, rows in groups.items()}
    edges = []
    for donor_n, donor_l in sides.items():
        ceil_l = int(math.ceil(donor_l - 1e-12))
        border_squares = 2 * ceil_l - 1
        for layers in (1, 2, 3):
            child_n = donor_n - layers * border_squares
            if child_n not in sides:
                continue
            projected_l = donor_l - layers
            gain = (sides[child_n] - projected_l) / math.sqrt(child_n)
            edges.append(
                {
                    "child": child_n,
                    "donor": donor_n,
                    "layers": layers,
                    "border_squares": border_squares,
                    "child_L": sides[child_n],
                    "donor_L": donor_l,
                    "projected_L": projected_l,
                    "projected_gain": gain,
                }
            )
    edges.sort(key=lambda item: item["projected_gain"], reverse=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(edges, indent=2) + "\n", encoding="utf-8")
    for edge in edges[:40]:
        print(
            f"{edge['donor']:4d}->{edge['child']:4d} "
            f"L{edge['layers']} gain={edge['projected_gain']:+.9e} "
            f"child={edge['child_L']:.9f} projected={edge['projected_L']:.9f}"
        )


if __name__ == "__main__":
    main()
