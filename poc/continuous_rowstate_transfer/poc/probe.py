#!/usr/bin/env python3
"""Capacity probe for continuous vertical row-state transfer."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
INCUMBENT = ROOT / "poc/pell_corrugation/data/border-scan/merged.csv"


def read_groups(path: Path) -> dict[int, list[dict[str, str]]]:
    groups: dict[int, list[dict[str, str]]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            groups.setdefault(int(row["n"]), []).append(row)
    return groups


def side(row: dict[str, str]) -> float:
    angle = float(row["angle"])
    return 0.5 * (abs(math.cos(angle)) + abs(math.sin(angle)))


def box_side(rows: list[dict[str, str]]) -> float:
    x0 = min(float(r["x"]) - side(r) for r in rows)
    x1 = max(float(r["x"]) + side(r) for r in rows)
    y0 = min(float(r["y"]) - side(r) for r in rows)
    y1 = max(float(r["y"]) + side(r) for r in rows)
    return max(x1 - x0, y1 - y0)


def lattice_count(
    width: float,
    height: float,
    angle: float,
    phase_x: float,
    phase_y: float,
) -> tuple[int, list[tuple[float, float]]]:
    """Count a rotated unit lattice wholly inside a ribbon.

    The basis vectors are the unit square axes. Integer translations therefore
    tile the plane exactly. Only square bounding boxes are clipped.
    """
    c, s = math.cos(angle), math.sin(angle)
    half = 0.5 * (abs(c) + abs(s))
    if width + 1e-12 < 2 * half or height + 1e-12 < 2 * half:
        return 0, []

    # Corners of the allowed center rectangle, transformed to lattice indices.
    xs = (half + phase_x, width - half + phase_x)
    ys = (half + phase_y, height - half + phase_y)
    i_corners = []
    j_corners = []
    for x in xs:
        for y in ys:
            i_corners.append(c * x + s * y)
            j_corners.append(-s * x + c * y)
    i0 = math.floor(min(i_corners)) - 1
    i1 = math.ceil(max(i_corners)) + 1
    j0 = math.floor(min(j_corners)) - 1
    j1 = math.ceil(max(j_corners)) + 1

    ii, jj = np.meshgrid(
        np.arange(i0, i1 + 1, dtype=np.float64),
        np.arange(j0, j1 + 1, dtype=np.float64),
        indexing="ij",
    )
    x = c * ii - s * jj
    y = s * ii + c * jj
    inside = (
        (x >= half - phase_x - 1e-11)
        & (x <= width - half - phase_x + 1e-11)
        & (y >= half - phase_y - 1e-11)
        & (y <= height - half - phase_y + 1e-11)
    )
    selected = np.argwhere(inside)
    coords = [
        (float(x[i, j]), float(y[i, j]))
        for i, j in selected
    ]
    return len(coords), coords


def capacity_catalog(width: float, height: float) -> dict:
    best = {
        "count": 0,
        "angle_deg": 0.0,
        "phase": [0.0, 0.0],
        "coords": [],
    }
    angles = [0.0]
    angles.extend(math.radians(float(a)) for a in range(-45, 46, 3) if a)
    phases = ((0.0, 0.0), (0.25, 0.0), (0.5, 0.0), (0.75, 0.0),
              (0.0, 0.5), (0.25, 0.5), (0.5, 0.5), (0.75, 0.5))
    for angle in angles:
        half = 0.5 * (abs(math.cos(angle)) + abs(math.sin(angle)))
        if 2 * half > width + 1e-12 or 2 * half > height + 1e-12:
            continue
        for px, py in phases:
            count, coords = lattice_count(width, height, angle, px, py)
            if count > best["count"]:
                best = {
                    "count": count,
                    "angle_deg": math.degrees(angle),
                    "phase": [px, py],
                    "coords": coords,
                }
    return best


def best_partition(l_target: float, cache: dict[float, dict]) -> dict:
    """Best two side ribbons plus an exact upright central bulk."""
    widths = [round(0.25 * i, 2) for i in range(1, int(l_target / 0.25) + 1)]
    best = {
        "count": 0,
        "left": 0.0,
        "center": 0.0,
        "right": 0.0,
        "center_count": 0,
    }
    for left in widths:
        for right in widths:
            center = l_target - left - right
            if center < 0.25:
                continue
            center_capacity = math.floor(center + 1e-12) * math.floor(l_target + 1e-12)
            count = cache[left]["count"] + center_capacity + cache[right]["count"]
            if count > best["count"]:
                best = {
                    "count": count,
                    "left": left,
                    "center": center,
                    "right": right,
                    "center_count": center_capacity,
                }
    return best


def probe(n: int, rows: list[dict[str, str]]) -> dict:
    incumbent_l = box_side(rows)
    max_width = int(math.ceil(incumbent_l))
    cache_widths = {
        round(0.25 * i, 2)
        for i in range(1, 4 * max_width + 1)
    }
    cache = {
        width: capacity_catalog(width, incumbent_l)
        for width in sorted(cache_widths)
    }

    candidates = []
    for delta in (1e-9, 1e-6, 1e-3, 0.01):
        target_l = incumbent_l - delta
        result = best_partition(target_l, cache)
        candidates.append({
            "delta": delta,
            "target_L": target_l,
            "best": result,
            "contains_n": result["count"] >= n,
        })
    return {
        "n": n,
        "K": math.isqrt(n),
        "incumbent_L": incumbent_l,
        "candidates": candidates,
        "best_count": max(c["best"]["count"] for c in candidates),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", default="861,747,804")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "docs/probe.json",
    )
    args = parser.parse_args()

    groups = read_groups(INCUMBENT)
    results = [probe(n, groups[n]) for n in map(int, args.targets.split(","))]
    report = {
        "method": "three independent vertical strips; best continuous lattice state per width",
        "targets": results,
        "wins": [],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for row in results:
        print(
            f"n={row['n']} K={row['K']} L={row['incumbent_L']:.15g} "
            f"best_count={row['best_count']}"
        )


if __name__ == "__main__":
    main()
