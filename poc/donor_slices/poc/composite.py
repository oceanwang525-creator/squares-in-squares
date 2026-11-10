"""Composite of a donor crop in a corner and an upright grid in the L-shape.

The crop keeps its exact donor coordinates, translated to the origin.  The
remaining L-shape is filled with the maximal upright unit grid for the chosen
container side.  Capacity and side are computed from the actual square
vertices, not from a nominal container.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from crop import (
    Crop,
    canonical_orientation,
    iter_crops,
    oriented_dims,
    transform_rows,
)

GAP = 1e-7
EPS_L = 1e-5


@dataclass(frozen=True)
class Placement:
    donor_n: int
    crop: Crop
    orientation: str
    width: float
    height: float
    columns_right: int
    rows_right: int
    columns_top: int
    rows_top: int
    grid_capacity: int
    total_capacity: int
    side: float


def grid_capacity(side: float, width: float, height: float) -> int:
    if width >= side - 1e-12 or height >= side - 1e-12:
        return 0
    columns = math.floor(side - width + 1e-12)
    rows = math.floor(side + 1e-12)
    top_columns = math.floor(width + 1e-12)
    top_rows = math.floor(side - height + 1e-12)
    return max(0, columns) * rows + top_columns * max(0, top_rows)


def _side(width: float, height: float, columns: int, rows: int,
          top_columns: int, top_rows: int) -> float:
    return max(
        width,
        height,
        width + GAP + columns,
        height + GAP + top_rows,
        rows,
        top_columns,
    )


def min_side_for_capacity(
    capacity: int,
    width: float,
    height: float,
    target: int,
    exact: bool = False,
) -> tuple[float, int, int, int, int] | None:
    """Smallest exact side whose crop plus maximal L-grid has capacity >= target.

    With exact=True only total capacity equal to target is accepted.
    """
    if capacity == target:
        return max(width, height), 0, 0, 0, 0
    if capacity > target:
        return None if exact else (max(width, height), 0, 0, 0, 0)

    lower = max(width, height, math.sqrt(target))
    upper = max(lower, math.ceil(math.sqrt(target)) + 2.0)
    candidates: set[float] = set()
    for value in range(math.floor(lower) - 1, math.ceil(upper) + 2):
        candidates.add(float(value))
    for base in (width, height):
        for step in range(math.floor(lower - base) - 1, math.ceil(upper - base) + 2):
            candidates.add(base + step)

    best: tuple[float, int, int, int, int] | None = None
    for side in sorted(candidates):
        if side < lower - 1e-12:
            continue
        columns = math.floor(side - width + 1e-12)
        rows = math.floor(side + 1e-12)
        top_columns = math.floor(width + 1e-12)
        top_rows = math.floor(side - height + 1e-12)
        if columns < 0 or top_rows < 0:
            continue
        total = capacity + columns * rows + top_columns * top_rows
        if exact:
            if total != target:
                continue
        elif total < target:
            continue
        exact_side = _side(width, height, columns, rows, top_columns, top_rows)
        if best is None or exact_side < best[0]:
            best = (exact_side, columns, rows, top_columns, top_rows)
    return best


def search_target(
    packings: dict[int, list[tuple[str, str, str, str]]],
    target: int,
    incumbent: float,
    min_width: int,
    max_width: int,
    min_height: int,
    max_height: int,
) -> tuple[Placement | None, Placement | None]:
    """Return the best below-incumbent near miss and the best capacity match."""

    def scan(exact: bool) -> tuple[Placement | None, Placement | None]:
        below_side = incumbent - EPS_L
        best_below: Placement | None = None
        best_match: Placement | None = None

        for donor_n in range(target + 1, 1001):
            rows = packings.get(donor_n)
            if not rows:
                continue
            for crop in iter_crops(
                rows, donor_n, min_width, max_width, min_height, max_height
            ):
                seen: set[tuple[float, float]] = set()
                for name, width, height in oriented_dims(crop):
                    key = (round(width, 10), round(height, 10))
                    if key in seen:
                        continue
                    seen.add(key)
                    orientation = canonical_orientation(crop, name)

                    if width < below_side and height < below_side:
                        columns = math.floor(below_side - width + 1e-12)
                        rows_right = math.floor(below_side + 1e-12)
                        top_columns = math.floor(width + 1e-12)
                        top_rows = math.floor(below_side - height + 1e-12)
                        capacity = (
                            crop.capacity
                            + columns * rows_right
                            + top_columns * top_rows
                        )
                        side = _side(
                            width, height, columns, rows_right,
                            top_columns, top_rows,
                        )
                        if side < incumbent:
                            candidate = Placement(
                                donor_n=donor_n,
                                crop=crop,
                                orientation=orientation,
                                width=width,
                                height=height,
                                columns_right=columns,
                                rows_right=rows_right,
                                columns_top=top_columns,
                                rows_top=top_rows,
                                grid_capacity=(
                                    columns * rows_right
                                    + top_columns * top_rows
                                ),
                                total_capacity=capacity,
                                side=side,
                            )
                            if (
                                best_below is None
                                or candidate.total_capacity
                                > best_below.total_capacity
                                or (
                                    candidate.total_capacity
                                    == best_below.total_capacity
                                    and candidate.side < best_below.side
                                )
                            ):
                                best_below = candidate

                    if best_match is not None:
                        upper = crop.capacity + grid_capacity(
                            best_match.side, width, height
                        )
                        if upper < target:
                            continue

                    result = min_side_for_capacity(
                        crop.capacity, width, height, target, exact=exact
                    )
                    if result is None:
                        continue
                    side, columns, rows_right, top_columns, top_rows = result
                    candidate = Placement(
                        donor_n=donor_n,
                        crop=crop,
                        orientation=orientation,
                        width=width,
                        height=height,
                        columns_right=columns,
                        rows_right=rows_right,
                        columns_top=top_columns,
                        rows_top=top_rows,
                        grid_capacity=(
                            columns * rows_right + top_columns * top_rows
                        ),
                        total_capacity=(
                            crop.capacity + columns * rows_right
                            + top_columns * top_rows
                        ),
                        side=side,
                    )
                    if best_match is None or candidate.side < best_match.side:
                        best_match = candidate

        return best_below, best_match

    below, match = scan(exact=True)
    if match is not None:
        return below, match
    return scan(exact=False)

def _grid_rows(placement: Placement) -> list[tuple[float, float]]:
    width = placement.width
    height = placement.height
    rows: list[tuple[float, float]] = []
    for column in range(placement.columns_right):
        for row in range(placement.rows_right):
            rows.append((width + GAP + column + 0.5, row + 0.5))
    for top in range(placement.rows_top):
        for column in range(placement.columns_top):
            rows.append((column + 0.5, height + GAP + top + 0.5))
    return rows


def build_rows(
    packings: dict[int, list[tuple[str, str, str, str]]],
    placement: Placement,
    target: int,
) -> list[tuple[str, str, str, str]]:
    """Emit exactly target rows for a placement, trimming surplus grid squares."""
    source = packings[placement.donor_n]
    crop_rows = transform_rows(source, placement.crop, placement.orientation)
    grid_rows = _grid_rows(placement)
    if len(crop_rows) >= target:
        crop_rows = crop_rows[:target]
        grid_rows = []
    else:
        needed = target - len(crop_rows)
        if needed > len(grid_rows):
            raise ValueError("placement capacity is below target")
        grid_rows = grid_rows[:needed]

    out: list[tuple[str, str, str, str]] = []
    for index, (_, x, y, angle) in enumerate(crop_rows):
        out.append((str(index), str(x), str(y), str(angle)))
    next_id = len(out)
    for index, (x, y) in enumerate(grid_rows):
        out.append((str(next_id + index), format(x, ".17g"), format(y, ".17g"), "0"))
    return out


def rows_side(rows: list[tuple[str, str, str, str]]) -> float:
    min_x = min_y = math.inf
    max_x = max_y = -math.inf
    for _, x_text, y_text, angle_text in rows:
        x = float(x_text)
        y = float(y_text)
        angle = float(angle_text)
        extent = 0.5 * (abs(math.cos(angle)) + abs(math.sin(angle)))
        min_x = min(min_x, x - extent)
        max_x = max(max_x, x + extent)
        min_y = min(min_y, y - extent)
        max_y = max(max_y, y + extent)
    return max(max_x - min_x, max_y - min_y)
