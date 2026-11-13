#!/usr/bin/env python3
"""Search two opposite-corner Mondrian compositions at selected high n."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import tempfile
from dataclasses import dataclass
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "poc" / "baseline" / "poc" / "src"))
from squares import core  # noqa: E402


COLS = ("n", "square_id", "x", "y", "angle")
EPS = 1e-10


@dataclass(frozen=True)
class Crop:
    donor_n: int
    corner: str
    capacity: int
    width: float
    height: float
    cut: tuple[int, int, int, int]
    indices: tuple[int, ...]


def read_groups(path: Path) -> dict[int, list[dict[str, str]]]:
    groups: dict[int, list[dict[str, str]]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            n = int(row["n"])
            groups.setdefault(n, []).append({key: row[key] for key in COLS})
    for n, rows in groups.items():
        rows.sort(key=lambda row: int(row["square_id"]))
        if len(rows) != n:
            raise ValueError(f"n={n}: found {len(rows)} rows")
    return groups


def square_boxes(rows: list[dict[str, str]]) -> np.ndarray:
    out = np.empty((len(rows), 4), dtype=float)
    for index, row in enumerate(rows):
        x = float(row["x"])
        y = float(row["y"])
        angle = float(row["angle"])
        extent = 0.5 * (abs(math.cos(angle)) + abs(math.sin(angle)))
        out[index] = (x - extent, x + extent, y - extent, y + extent)
    return out


def contained(
    boxes: np.ndarray, x0: float, x1: float, y0: float, y1: float
) -> np.ndarray:
    return np.flatnonzero(
        (boxes[:, 0] >= x0 - EPS)
        & (boxes[:, 1] <= x1 + EPS)
        & (boxes[:, 2] >= y0 - EPS)
        & (boxes[:, 3] <= y1 + EPS)
    )


def corner_crops(
    rows: list[dict[str, str]],
    donor_n: int,
    min_side: int,
    max_side: int,
) -> list[Crop]:
    boxes = square_boxes(rows)
    if len(boxes) == 0:
        return []
    donor_x0 = math.floor(float(boxes[:, 0].min()))
    donor_x1 = math.ceil(float(boxes[:, 1].max()))
    donor_y0 = math.floor(float(boxes[:, 2].min()))
    donor_y1 = math.ceil(float(boxes[:, 3].max()))
    out: list[Crop] = []
    seen: set[tuple[str, int, float, float]] = set()

    for corner in ("bl", "br", "tl", "tr"):
        for width in range(min_side, max_side + 1):
            for height in range(min_side, max_side + 1):
                if corner in ("bl", "tl"):
                    x0, x1 = donor_x0, donor_x0 + width
                else:
                    x0, x1 = donor_x1 - width, donor_x1
                if corner in ("bl", "br"):
                    y0, y1 = donor_y0, donor_y0 + height
                else:
                    y0, y1 = donor_y1 - height, donor_y1
                indices = contained(boxes, x0, x1, y0, y1)
                if len(indices) == 0:
                    continue
                selected = boxes[indices]
                tight_width = float(selected[:, 1].max() - selected[:, 0].min())
                tight_height = float(selected[:, 3].max() - selected[:, 2].min())
                key = (
                    corner,
                    len(indices),
                    round(tight_width, 10),
                    round(tight_height, 10),
                )
                if key in seen:
                    continue
                seen.add(key)
                out.append(
                    Crop(
                        donor_n=donor_n,
                        corner=corner,
                        capacity=len(indices),
                        width=tight_width,
                        height=tight_height,
                        cut=(x0, x1, y0, y1),
                        indices=tuple(int(i) for i in indices),
                    )
                )
    return out


def touched_cells(
    side: float, width: float, height: float, from_top_right: bool
) -> int:
    limit = math.floor(side + EPS)
    if from_top_right:
        columns = limit - math.floor(side - width + EPS)
        rows = limit - math.floor(side - height + EPS)
    else:
        columns = math.ceil(width - EPS)
        rows = math.ceil(height - EPS)
    return max(0, columns) * max(0, rows)


def crop_gain(crop: Crop, side: float, from_top_right: bool) -> int:
    return crop.capacity - touched_cells(
        side, crop.width, crop.height, from_top_right
    )


def crop_rows(
    rows: list[dict[str, str]], crop: Crop, side: float, top_right: bool
) -> list[tuple[str, str, str]]:
    selected = [rows[index] for index in crop.indices]
    bbox_min_x = min(
        float(row["x"])
        - 0.5
        * (abs(math.cos(float(row["angle"]))) + abs(math.sin(float(row["angle"]))))
        for row in selected
    )
    bbox_min_y = min(
        float(row["y"])
        - 0.5
        * (abs(math.cos(float(row["angle"]))) + abs(math.sin(float(row["angle"]))))
        for row in selected
    )
    offset_x = side - crop.width if top_right else 0.0
    offset_y = side - crop.height if top_right else 0.0
    out: list[tuple[str, str, str]] = []
    for row in selected:
        x = float(row["x"]) - bbox_min_x + offset_x
        y = float(row["y"]) - bbox_min_y + offset_y
        out.append((format(x, ".17g"), format(y, ".17g"), row["angle"]))
    return out


def grid_rows(
    side: float,
    crop_a: Crop,
    crop_b: Crop,
) -> list[tuple[str, str]]:
    limit = math.floor(side + EPS)
    a_boxes = (0.0, crop_a.width, 0.0, crop_a.height)
    b_boxes = (
        side - crop_b.width,
        side,
        side - crop_b.height,
        side,
    )
    out: list[tuple[str, str]] = []
    for i in range(limit):
        for j in range(limit):
            x0 = float(i)
            x1 = float(i + 1)
            y0 = float(j)
            y1 = float(j + 1)
            blocked = False
            for bx0, bx1, by0, by1 in (a_boxes, b_boxes):
                if x1 > bx0 + EPS and x0 < bx1 - EPS and y1 > by0 + EPS and y0 < by1 - EPS:
                    blocked = True
                    break
            if not blocked:
                out.append((format(i + 0.5, ".17g"), format(j + 0.5, ".17g")))
    return out


def choose_crops(
    packings: dict[int, list[dict[str, str]]],
    side: float,
    target: int,
    min_side: int,
    max_side: int,
) -> tuple[tuple[Crop, Crop] | None, dict[str, object]]:
    limit = math.floor(side + EPS)
    need = target - limit * limit
    bottom_crops: list[Crop] = []
    top_crops: list[Crop] = []
    for donor_n, rows in packings.items():
        for crop in corner_crops(rows, donor_n, min_side, max_side):
            if crop.corner in ("bl", "br"):
                bottom_crops.append(crop)
            if crop.corner in ("tl", "tr"):
                top_crops.append(crop)

    bottom_gain = np.array(
        [crop_gain(crop, side, False) for crop in bottom_crops], dtype=np.int16
    )
    top_gain = np.array(
        [crop_gain(crop, side, True) for crop in top_crops], dtype=np.int16
    )
    best_total = -10**9
    best_pair: tuple[Crop, Crop] | None = None
    best_record: tuple[Crop, Crop] | None = None

    # The best gains are sparse: huge cuts lose grid cells. Keeping the top
    # slice makes the opposite-corner pair scan exact for the observed range.
    bottom_order = np.argsort(bottom_gain)[::-1][:3000]
    top_order = np.argsort(top_gain)[::-1][:3000]
    for ia in bottom_order:
        crop_a = bottom_crops[int(ia)]
        for ib in top_order:
            crop_b = top_crops[int(ib)]
            if crop_a.width + crop_b.width > side + EPS:
                continue
            if crop_a.height + crop_b.height > side + EPS:
                continue
            gain = int(bottom_gain[ia]) + int(top_gain[ib])
            total = limit * limit + gain
            if total > best_total:
                best_total = total
                best_record = (crop_a, crop_b)
            if total >= target and best_pair is None:
                best_pair = (crop_a, crop_b)

    chosen = best_pair or best_record
    report = {
        "target": target,
        "side": format(side, ".17g"),
        "grid_side": limit,
        "grid_capacity": limit * limit,
        "needed_gain": need,
        "crop_counts": [len(bottom_crops), len(top_crops)],
        "max_gain": best_total - limit * limit,
        "capacity": best_total,
        "reaches_target": best_pair is not None,
    }
    if chosen is not None:
        report["crops"] = [
            {
                "donor_n": crop.donor_n,
                "corner": crop.corner,
                "capacity": crop.capacity,
                "width": format(crop.width, ".17g"),
                "height": format(crop.height, ".17g"),
                "cut": list(crop.cut),
                "gain": crop_gain(crop, side, crop.corner in ("tl", "tr")),
            }
            for crop in chosen
        ]
    return best_pair, report


def build_target(
    packings: dict[int, list[dict[str, str]]],
    pair: tuple[Crop, Crop],
    side: float,
    target: int,
) -> list[dict[str, str]]:
    crop_a, crop_b = pair
    first = crop_rows(packings[crop_a.donor_n], crop_a, side, False)
    second = crop_rows(packings[crop_b.donor_n], crop_b, side, True)
    rows = first + second
    if len(rows) > target:
        raise ValueError("corner crops alone exceed target")
    grid = grid_rows(side, crop_a, crop_b)
    needed = target - len(rows)
    if needed > len(grid):
        raise ValueError("composition capacity is below target")
    rows.extend((x, y, "0") for x, y in grid[:needed])
    return [
        {"square_id": str(index), "x": x, "y": y, "angle": angle}
        for index, (x, y, angle) in enumerate(rows)
    ]


def certify_both(rows: list[dict[str, str]]) -> tuple[str, str]:
    literal = Decimal(core.certify(len(rows), rows))
    roundtrip = core.kaggle_roundtrip({len(rows): rows})[len(rows)]
    pandas_side = Decimal(core.certify(len(rows), roundtrip))
    return str(literal), str(pandas_side)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--submission",
        type=Path,
        default=ROOT
        / "poc"
        / "pell_corrugation"
        / "data"
        / "border-scan"
        / "merged.csv",
    )
    parser.add_argument("--targets", default="686,687,908,910,911,969,970,971,972,973")
    parser.add_argument("--min-crop", type=int, default=4)
    parser.add_argument("--max-crop", type=int, default=16)
    parser.add_argument("--margin", default="1e-9")
    parser.add_argument("--summary", type=Path, default=Path("/tmp/mondrian-summary.json"))
    parser.add_argument("--rows-out", type=Path, default=Path("/tmp/mondrian-rows.csv"))
    args = parser.parse_args()

    packings = read_groups(args.submission)
    incumbent = {
        n: Decimal(core.certify(n, packings[n])) for n in map(int, args.targets.split(","))
    }
    accepted: dict[int, list[dict[str, str]]] = {}
    report: dict[str, object] = {
        "submission": str(args.submission),
        "margin": args.margin,
        "min_crop": args.min_crop,
        "max_crop": args.max_crop,
        "targets": {},
    }
    for target in map(int, args.targets.split(",")):
        side = incumbent[target] - Decimal(args.margin)
        pair, target_report = choose_crops(
            packings,
            float(side),
            target,
            args.min_crop,
            args.max_crop,
        )
        target_report["incumbent_L"] = str(incumbent[target])
        if pair is not None:
            rows = build_target(packings, pair, float(side), target)
            literal, pandas_side = certify_both(rows)
            target_report["certified_literal_L"] = literal
            target_report["certified_pandas_L"] = pandas_side
            target_report["strict_win"] = (
                Decimal(literal) < incumbent[target]
                and Decimal(pandas_side) < incumbent[target]
            )
            if target_report["strict_win"]:
                accepted[target] = rows
        else:
            target_report["strict_win"] = False
        report["targets"][str(target)] = target_report
        print(json.dumps({str(target): target_report}, indent=1), flush=True)

    report["accepted_n"] = sorted(accepted)
    report["accepted"] = len(accepted)
    args.summary.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    with args.rows_out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=COLS, lineterminator="\n"
        )
        writer.writeheader()
        for n in sorted(accepted):
            writer.writerows(
                {"n": n, **{key: row[key] for key in COLS if key != "n"}}
                for row in accepted[n]
            )
    print(f"accepted {len(accepted)}/{len(report['targets'])}", flush=True)


if __name__ == "__main__":
    main()
