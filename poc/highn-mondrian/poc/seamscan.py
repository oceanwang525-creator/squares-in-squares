#!/usr/bin/env python3
"""Exact upright refill for many Mondrian crop pairs."""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import sys
from decimal import Decimal
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fillfast  # noqa: E402
import scan  # noqa: E402


def top_crops(
    packings: dict[int, list[dict[str, str]]],
    side: float,
    min_crop: int,
    max_crop: int,
    keep: int,
) -> tuple[list[tuple[int, scan.Crop]], list[tuple[int, scan.Crop]]]:
    bottom_heap: list[tuple[int, int, scan.Crop]] = []
    top_heap: list[tuple[int, int, scan.Crop]] = []
    serial = 0
    for donor_n, rows in packings.items():
        for crop in scan.corner_crops(rows, donor_n, min_crop, max_crop):
            from_top = crop.corner in ("tl", "tr")
            gain = scan.crop_gain(crop, side, from_top)
            heap = top_heap if from_top else bottom_heap
            item = (gain, serial, crop)
            serial += 1
            if len(heap) < keep:
                heapq.heappush(heap, item)
            elif gain > heap[0][0]:
                heapq.heapreplace(heap, item)
    bottom = [(gain, crop) for gain, _, crop in sorted(bottom_heap, reverse=True)]
    top = [(gain, crop) for gain, _, crop in sorted(top_heap, reverse=True)]
    return bottom, top


def best_pairs(
    bottom: list[tuple[int, scan.Crop]],
    top: list[tuple[int, scan.Crop]],
    side: float,
    grid_capacity: int,
    keep: int,
) -> list[tuple[int, int, int, scan.Crop, scan.Crop]]:
    heap: list[tuple[int, int, int, scan.Crop, scan.Crop]] = []
    serial = 0
    for gain_a, crop_a in bottom:
        for gain_b, crop_b in top:
            if crop_a.width + crop_b.width > side + scan.EPS:
                continue
            if crop_a.height + crop_b.height > side + scan.EPS:
                continue
            bound = grid_capacity + gain_a + gain_b
            item = (bound, serial, gain_a + gain_b, crop_a, crop_b)
            serial += 1
            if len(heap) < keep:
                heapq.heappush(heap, item)
            elif bound > heap[0][0]:
                heapq.heapreplace(heap, item)
    return sorted(heap, reverse=True, key=lambda item: (item[0], -item[1]))


def local_piece(
    rows: list[dict[str, str]], crop: scan.Crop
) -> tuple[float, float, list[tuple[float, float, float]]]:
    selected = [rows[index] for index in crop.indices]
    values = []
    for row in selected:
        x = float(row["x"])
        y = float(row["y"])
        angle = float(row["angle"])
        extent = 0.5 * (abs(math.cos(angle)) + abs(math.sin(angle)))
        values.append((x, y, angle, extent))
    x0 = min(x - extent for x, _, _, extent in values)
    y0 = min(y - extent for _, y, _, extent in values)
    x1 = max(x + extent for x, _, _, extent in values)
    y1 = max(y + extent for _, y, _, extent in values)
    width, height = x1 - x0, y1 - y0
    piece = [
        (x - x0, y - y0, angle)
        for x, y, angle, _extent in values
    ]
    return width, height, piece


def transformed_piece(
    width: float,
    height: float,
    piece: list[tuple[float, float, float]],
    mirror: bool,
    rotations: int,
) -> tuple[float, float, list[tuple[float, float, float]]]:
    values = [
        (width - x if mirror else x, y, -angle if mirror else angle)
        for x, y, angle in piece
    ]
    for _ in range(rotations % 4):
        values = [(height - y, x, angle) for x, y, angle in values]
        width, height = height, width
    return width, height, values


def place_piece(
    piece: list[tuple[float, float, float]],
    width: float,
    height: float,
    phase_x: float,
    phase_y: float,
    top_right: bool,
    side: float,
) -> list[tuple[float, float, float]]:
    x0 = side - width - phase_x if top_right else phase_x
    y0 = side - height - phase_y if top_right else phase_y
    return [(x + x0, y + y0, angle) for x, y, angle in piece]


def touched_grid_cells(
    side: float, x0: float, x1: float, y0: float, y1: float
) -> int:
    limit = math.floor(side + scan.EPS)
    first_x = max(0, math.floor(x0 + scan.EPS))
    last_x = min(limit - 1, math.ceil(x1 - scan.EPS) - 1)
    first_y = max(0, math.floor(y0 + scan.EPS))
    last_y = min(limit - 1, math.ceil(y1 - scan.EPS) - 1)
    return max(0, last_x - first_x + 1) * max(0, last_y - first_y + 1)


def tilted_rows(
    packings: dict[int, list[dict[str, str]]],
    crop_a: scan.Crop,
    crop_b: scan.Crop,
    side: float,
    transforms: bool,
    variants_per_pair: int,
) -> list[tuple[int, list[tuple[float, float, float]]]]:
    if not transforms:
        return [
            (
                math.floor(side + scan.EPS) ** 2
                + scan.crop_gain(crop_a, side, False)
                + scan.crop_gain(crop_b, side, True),
                [
                    (float(x), float(y), float(angle))
                    for x, y, angle in scan.crop_rows(
                        packings[crop_a.donor_n], crop_a, side, top_right=False
                    )
                    + scan.crop_rows(
                        packings[crop_b.donor_n], crop_b, side, top_right=True
                    )
                ],
            )
        ]

    wa, ha, piece_a = local_piece(packings[crop_a.donor_n], crop_a)
    wb, hb, piece_b = local_piece(packings[crop_b.donor_n], crop_b)
    variants_a = [
        transformed_piece(wa, ha, piece_a, mirror, rotations)
        for mirror in (False, True)
        for rotations in range(4)
    ]
    variants_b = [
        transformed_piece(wb, hb, piece_b, mirror, rotations)
        for mirror in (False, True)
        for rotations in range(4)
    ]
    variants: dict[
        tuple[tuple[int, int, int], ...],
        tuple[int, list[tuple[float, float, float]]],
    ] = {}
    phases = (0.0, 0.5)
    for width_a, height_a, shape_a in variants_a:
        for width_b, height_b, shape_b in variants_b:
            for phase_ax in phases:
                for phase_ay in phases:
                    for phase_bx in phases:
                        for phase_by in phases:
                            if width_a + phase_ax + width_b + phase_bx > side + scan.EPS:
                                continue
                            if height_a + phase_ay + height_b + phase_by > side + scan.EPS:
                                continue
                            first = place_piece(
                                shape_a,
                                width_a,
                                height_a,
                                phase_ax,
                                phase_ay,
                                False,
                                side,
                            )
                            second = place_piece(
                                shape_b,
                                width_b,
                                height_b,
                                phase_bx,
                                phase_by,
                                True,
                                side,
                            )
                            combined = first + second
                            touched = touched_grid_cells(
                                side,
                                phase_ax,
                                phase_ax + width_a,
                                phase_ay,
                                phase_ay + height_a,
                            )
                            touched += touched_grid_cells(
                                side,
                                side - phase_bx - width_b,
                                side - phase_bx,
                                side - phase_by - height_b,
                                side - phase_by,
                            )
                            bound = len(combined) + math.floor(side + scan.EPS) ** 2 - touched
                            key = tuple(
                                sorted(
                                    (
                                        round(x, 9),
                                        round(y, 9),
                                        round(angle, 9),
                                    )
                                    for x, y, angle in combined
                                )
                            )
                            old = variants.get(key)
                            if old is None or bound > old[0]:
                                variants[key] = (bound, combined)
    ordered = sorted(variants.values(), reverse=True, key=lambda item: item[0])
    return ordered[:variants_per_pair]


def exact_capacity(side: float, tilted: list[tuple[float, float, float]], limit: int) -> int:
    filled = fillfast.fill(side, np.asarray(tilted, dtype=float), mode="exact", time_limit=limit)
    return len(filled)


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
    parser.add_argument("--targets", default="971")
    parser.add_argument("--min-crop", type=int, default=4)
    parser.add_argument("--max-crop", type=int, default=16)
    parser.add_argument("--keep-crops", type=int, default=1500)
    parser.add_argument("--top-pairs", type=int, default=2500)
    parser.add_argument("--milp-limit", type=int, default=10)
    parser.add_argument("--margin", default="1e-9")
    parser.add_argument("--transforms", action="store_true")
    parser.add_argument("--variants-per-pair", type=int, default=64)
    parser.add_argument("--summary", type=Path, default=Path("/tmp/seamscan-summary.json"))
    parser.add_argument("--rows-out", type=Path, default=Path("/tmp/seamscan-rows.csv"))
    args = parser.parse_args()

    packings = scan.read_groups(args.submission)
    incumbent = {
        n: Decimal(scan.core.certify(n, packings[n]))
        for n in map(int, args.targets.split(","))
    }
    accepted: dict[int, list[dict[str, str]]] = {}
    report: dict[str, object] = {
        "submission": str(args.submission),
        "margin": args.margin,
        "min_crop": args.min_crop,
        "max_crop": args.max_crop,
        "keep_crops": args.keep_crops,
        "top_pairs": args.top_pairs,
        "transforms": args.transforms,
        "variants_per_pair": args.variants_per_pair,
        "targets": {},
    }

    for target in map(int, args.targets.split(",")):
        side = incumbent[target] - Decimal(args.margin)
        side_f = float(side)
        limit = math.floor(side_f + scan.EPS)
        bottom, top = top_crops(
            packings,
            side_f,
            args.min_crop,
            args.max_crop,
            args.keep_crops,
        )
        pairs = best_pairs(
            bottom,
            top,
            side_f,
            limit * limit,
            args.top_pairs,
        )
        best: list[dict[str, object]] = []
        for bound, serial, gain, crop_a, crop_b in pairs:
            variants = tilted_rows(
                packings,
                crop_a,
                crop_b,
                side_f,
                args.transforms,
                args.variants_per_pair,
            )
            for variant_bound, tilted in variants:
                capacity = exact_capacity(side_f, tilted, args.milp_limit)
                record = {
                    "bound": variant_bound,
                    "gain": gain,
                    "capacity": capacity,
                    "short": target - capacity,
                    "serial": serial,
                    "crop_a": {
                        "donor_n": crop_a.donor_n,
                        "corner": crop_a.corner,
                        "capacity": crop_a.capacity,
                        "width": crop_a.width,
                        "height": crop_a.height,
                        "cut": list(crop_a.cut),
                    },
                    "crop_b": {
                        "donor_n": crop_b.donor_n,
                        "corner": crop_b.corner,
                        "capacity": crop_b.capacity,
                        "width": crop_b.width,
                        "height": crop_b.height,
                        "cut": list(crop_b.cut),
                    },
                }
                best.append(record)
                if len(best) > 20:
                    best.sort(key=lambda item: int(item["capacity"]), reverse=True)
                    del best[20:]
                if capacity >= target:
                    filled = fillfast.fill(
                        side_f,
                        np.asarray(tilted, dtype=float),
                        mode="exact",
                        time_limit=args.milp_limit,
                    )
                    rows = [
                        {
                            "square_id": str(index),
                            "x": format(float(x), ".17g"),
                            "y": format(float(y), ".17g"),
                            "angle": format(float(angle), ".17g"),
                        }
                        for index, (x, y, angle) in enumerate(filled[:target])
                    ]
                    literal, pandas_side = scan.certify_both(rows)
                    record["certified_literal_L"] = literal
                    record["certified_pandas_L"] = pandas_side
                    record["strict_win"] = (
                        Decimal(literal) < incumbent[target]
                        and Decimal(pandas_side) < incumbent[target]
                    )
                    if record["strict_win"]:
                        accepted[target] = rows
                        report["targets"][str(target)] = {
                            "side": str(side),
                            "best": best,
                            "winner": record,
                        }
                        break
            if target in accepted:
                break
        else:
            report["targets"][str(target)] = {
                "side": str(side),
                "best": best,
                "winner": None,
            }
        print(
            json.dumps(
                {
                    str(target): {
                        "best_capacity": best[0]["capacity"] if best else None,
                        "best_short": best[0]["short"] if best else None,
                        "win": target in accepted,
                    }
                }
            ),
            flush=True,
        )

    report["accepted_n"] = sorted(accepted)
    report["accepted"] = len(accepted)
    args.summary.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    with args.rows_out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=scan.COLS, lineterminator="\n")
        writer.writeheader()
        for n in sorted(accepted):
            writer.writerows(
                {"n": n, **{key: row[key] for key in scan.COLS if key != "n"}}
                for row in accepted[n]
            )
    print(f"accepted {len(accepted)}/{len(report['targets'])}", flush=True)


if __name__ == "__main__":
    main()
