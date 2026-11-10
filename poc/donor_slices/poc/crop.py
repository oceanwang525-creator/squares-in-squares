"""Axis-aligned donor crops on an integer cut grid.

A crop is the set of donor squares that lie fully inside an integer rectangle.
The emitted crop uses the tight fractional bounding box of those squares, so
mixed-angle donors are measured from their actual vertices.  The eight
dihedral orientations are exposed; capacity and container side depend only on
the two bounding-box dimension orders, so the search keeps the first
orientation for each order (identity or a quarter turn).
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from decimal import Decimal, getcontext
from pathlib import Path

import numpy as np

getcontext().prec = 60
EPS = 1e-12


@dataclass(frozen=True)
class Crop:
    donor_n: int
    capacity: int
    width: float
    height: float
    min_x: float
    min_y: float
    cut: tuple[int, int, int, int]
    indices: tuple[int, ...]


def load_portfolio(path: str | Path) -> dict[int, list[tuple[str, str, str, str]]]:
    packings: dict[int, list[tuple[str, str, str, str]]] = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            n = int(row["n"])
            packings.setdefault(n, []).append(
                (row["square_id"], row["x"], row["y"], row["angle"])
            )
    return packings


def boxes(rows: list[tuple[str, str, str, str]]) -> np.ndarray:
    out = np.empty((len(rows), 4), dtype=float)
    for i, (_, x_text, y_text, angle_text) in enumerate(rows):
        x = float(x_text)
        y = float(y_text)
        angle = float(angle_text)
        extent = 0.5 * (abs(math.cos(angle)) + abs(math.sin(angle)))
        out[i] = (x - extent, x + extent, y - extent, y + extent)
    return out


def _contained(box: np.ndarray, x0: int, x1: int, y0: int, y1: int) -> np.ndarray:
    return np.flatnonzero(
        (box[:, 0] >= x0 - EPS)
        & (box[:, 1] <= x1 + EPS)
        & (box[:, 2] >= y0 - EPS)
        & (box[:, 3] <= y1 + EPS)
    )


def iter_crops(
    rows: list[tuple[str, str, str, str]],
    donor_n: int,
    min_width: int,
    max_width: int,
    min_height: int,
    max_height: int,
):
    """Yield distinct tight crops whose integer cut rectangle has the given size."""
    box = boxes(rows)
    if len(box) == 0:
        return
    min_x = math.floor(float(box[:, 0].min()))
    max_x = math.ceil(float(box[:, 1].max()))
    min_y = math.floor(float(box[:, 2].min()))
    max_y = math.ceil(float(box[:, 3].max()))
    seen: set[tuple[int, float, float, float, float]] = set()

    for width in range(min_width, max_width + 1):
        x0_min = min_x
        x0_max = max_x - width
        if x0_max < x0_min:
            continue
        xs = np.arange(x0_min, x0_max + 1)
        for height in range(min_height, max_height + 1):
            y0_min = min_y
            y0_max = max_y - height
            if y0_max < y0_min:
                continue
            ys = np.arange(y0_min, y0_max + 1)

            diff = np.zeros((len(xs) + 1, len(ys) + 1), dtype=np.int32)
            x_lo = np.ceil(box[:, 1] - EPS).astype(int) - width
            x_hi = np.floor(box[:, 0] + EPS).astype(int)
            y_lo = np.ceil(box[:, 3] - EPS).astype(int) - height
            y_hi = np.floor(box[:, 2] + EPS).astype(int)
            for i in range(len(box)):
                a = max(int(x_lo[i]), x0_min)
                c = min(int(x_hi[i]), x0_max)
                d = max(int(y_lo[i]), y0_min)
                e = min(int(y_hi[i]), y0_max)
                if a <= c and d <= e:
                    diff[a - x0_min, d - y0_min] += 1
                    diff[c + 1 - x0_min, d - y0_min] -= 1
                    diff[a - x0_min, e + 1 - y0_min] -= 1
                    diff[c + 1 - x0_min, e + 1 - y0_min] += 1

            counts = diff.cumsum(0).cumsum(1)[: len(xs), : len(ys)]
            ii, jj = np.where(counts > 0)
            for xi, yj in zip(ii, jj):
                x0 = int(xs[xi])
                y0 = int(ys[yj])
                index = _contained(box, x0, x0 + width, y0, y0 + height)
                if len(index) == 0:
                    continue
                selected = box[index]
                tight_width = float(selected[:, 1].max() - selected[:, 0].min())
                tight_height = float(selected[:, 3].max() - selected[:, 2].min())
                min_x_tight = float(selected[:, 0].min())
                min_y_tight = float(selected[:, 2].min())
                key = (
                    len(index),
                    round(tight_width, 12),
                    round(tight_height, 12),
                    round(min_x_tight, 12),
                    round(min_y_tight, 12),
                )
                if key in seen:
                    continue
                seen.add(key)
                yield Crop(
                    donor_n=donor_n,
                    capacity=len(index),
                    width=tight_width,
                    height=tight_height,
                    min_x=min_x_tight,
                    min_y=min_y_tight,
                    cut=(x0, x0 + width, y0, y0 + height),
                    indices=tuple(int(i) for i in index),
                )


ORIENTATIONS = (
    "identity",
    "rot90",
    "rot180",
    "rot270",
    "reflect_x",
    "reflect_y",
    "transpose",
    "anti_transpose",
)


def oriented_dims(crop: Crop) -> list[tuple[str, float, float]]:
    """Return all eight orientations with their bounding-box dimensions."""
    w, h = crop.width, crop.height
    return [
        ("identity", w, h),
        ("rot90", h, w),
        ("rot180", w, h),
        ("rot270", h, w),
        ("reflect_x", w, h),
        ("reflect_y", w, h),
        ("transpose", h, w),
        ("anti_transpose", h, w),
    ]


def canonical_orientation(crop: Crop, name: str) -> str:
    """Map any orientation to identity or rot90, which cover both dim orders."""
    if name in ("rot90", "rot270", "transpose", "anti_transpose"):
        return "rot90"
    return "identity"


def transform_rows(
    rows: list[tuple[str, str, str, str]],
    crop: Crop,
    orientation: str,
) -> list[tuple[str, Decimal, Decimal, Decimal]]:
    """Translate a crop to the origin and apply identity or a quarter turn."""
    origin_x = Decimal(str(crop.min_x))
    origin_y = Decimal(str(crop.min_y))
    width = Decimal(str(crop.width))
    height = Decimal(str(crop.height))
    out: list[tuple[str, Decimal, Decimal, Decimal]] = []
    for i in crop.indices:
        square_id, x_text, y_text, angle_text = rows[i]
        x = Decimal(x_text) - origin_x
        y = Decimal(y_text) - origin_y
        angle = Decimal(angle_text)
        if orientation == "rot90":
            x, y = y, width - x
        elif orientation == "rot180":
            x, y = width - x, height - y
        elif orientation == "rot270":
            x, y = height - y, x
        out.append((square_id, x, y, angle))
    return out
