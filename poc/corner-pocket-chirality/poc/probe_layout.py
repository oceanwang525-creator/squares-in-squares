#!/usr/bin/env python3
"""Materialize one replayed pocket cell for residual-gap inspection."""

from __future__ import annotations

import argparse
import csv
import json
import math
import struct
import zlib
from decimal import Decimal
from pathlib import Path

import numpy as np
from scipy import ndimage

import chirality_scan as cs
import fillfast
import scan


def read_baseline(path: Path) -> dict[int, list[dict[str, str]]]:
    return scan.read_groups(path)


def reconstruct(
    baseline: Path,
    legacy_summary: Path,
    target: int,
    cell: int,
    margin: str,
    first_phase: tuple[float, float] | None,
    second_phase: tuple[float, float] | None,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    packings = read_baseline(baseline)
    legacy = json.loads(legacy_summary.read_text(encoding="utf-8"))
    sources = tuple(int(value) for value in legacy["sources"])
    pockets = cs.extract_pockets(packings, sources, 80, 0.1, 1.5)
    incumbent = Decimal(scan.core.certify(target, packings[target]))
    side = float(incumbent - Decimal(margin))
    phases = tuple((x, y) for x in (0.0, 0.5) for y in (0.0, 0.5))
    options = [cs.placement_options(pocket, side, phases) for pocket in pockets]
    record = legacy["targets"][str(target)]["top"][cell]
    first_option = int(record["first_option"])
    second_option = int(record["second_option"])
    if first_phase is None or second_phase is None:
        first = options[int(record["first_index"])][first_option]
        second = options[int(record["second_index"])][second_option]
    else:
        first_pocket = pockets[int(record["first_index"])]
        second_pocket = pockets[int(record["second_index"])]
        first_symmetry = cs.SYMMETRIES[first_option // 8]
        second_symmetry = cs.SYMMETRIES[second_option // 8]
        first_squares, first_supports = cs.transform_bundle(
            first_pocket.squares, first_pocket.supports, first_symmetry
        )
        second_squares, second_supports = cs.transform_bundle(
            second_pocket.squares, second_pocket.supports, second_symmetry
        )
        first_offset = cs.placement_offset(first_squares, side, "LL", first_phase)
        second_offset = cs.placement_offset(second_squares, side, "UR", second_phase)
        first = {
            "squares": cs.shift(first_squares, first_offset),
            "supports": cs.shift(first_supports, first_offset),
        }
        second = {
            "squares": cs.shift(second_squares, second_offset),
            "supports": cs.shift(second_supports, second_offset),
        }
    return (
        side,
        first["squares"],
        second["squares"],
        np.vstack([first["supports"], second["supports"]])
        .reshape(-1, 3)
        .astype(np.float64),
        record,
    )


def square_polygon(square: np.ndarray) -> np.ndarray:
    x, y, angle = square
    cosine, sine = math.cos(angle), math.sin(angle)
    local = np.array(
        [
            (-0.5, -0.5),
            (0.5, -0.5),
            (0.5, 0.5),
            (-0.5, 0.5),
        ],
        dtype=np.float64,
    )
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    return local @ rotation.T + np.array([x, y])


def rasterize(squares: np.ndarray, side: float, scale: int) -> np.ndarray:
    size = int(math.ceil(side * scale))
    yy, xx = np.mgrid[0:size, 0:size]
    px = (xx + 0.5) / scale
    py = (yy + 0.5) / scale
    image = np.full((size, size, 4), 255, dtype=np.uint8)
    image[:, :, 3] = 255
    for square in squares:
        polygon = square_polygon(square)
        lo = np.maximum(np.floor((polygon.min(axis=0) - 0.5) * scale), 0)
        hi = np.minimum(np.ceil((polygon.max(axis=0) + 0.5) * scale), size)
        x0, y0 = int(lo[0]), int(lo[1])
        x1, y1 = int(hi[0]), int(hi[1])
        inside = np.ones((y1 - y0, x1 - x0), dtype=bool)
        for start, end in zip(polygon, np.roll(polygon, -1, axis=0)):
            edge = end - start
            local_x = px[y0:y1, x0:x1] - start[0]
            local_y = py[y0:y1, x0:x1] - start[1]
            cross = edge[0] * local_y - edge[1] * local_x
            inside &= cross >= -1e-9
        color = (
            np.array([224, 141, 62, 255], dtype=np.uint8)
            if abs(square[2]) > 1e-8
            else np.array([103, 153, 206, 255], dtype=np.uint8)
        )
        image[y0:y1, x0:x1][inside] = color
    return image


def write_png(path: Path, image: np.ndarray) -> None:
    height, width, _ = image.shape

    def chunk(kind: bytes, payload: bytes) -> bytes:
        body = kind + payload
        return struct.pack(">I", len(payload)) + body + struct.pack(
            ">I", zlib.crc32(body) & 0xFFFFFFFF
        )

    rows = b"".join(b"\x00" + row.tobytes() for row in image)
    payload = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(rows, 9))
        + chunk(b"IEND", b"")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def gap_components(
    squares: np.ndarray, side: float, scale: int, minimum_area: float
) -> list[dict[str, object]]:
    image = rasterize(squares, side, scale)
    occupied = np.any(image[:, :, :3] != 255, axis=2)
    empty = ~occupied
    labels, count = ndimage.label(empty)
    objects = ndimage.find_objects(labels)
    result = []
    for label, bounds in enumerate(objects, start=1):
        if bounds is None:
            continue
        area = float(np.count_nonzero(labels == label)) / (scale * scale)
        if area < minimum_area:
            continue
        result.append(
            {
                "area": area,
                "bbox": [
                    bounds[1].start / scale,
                    bounds[1].stop / scale,
                    bounds[0].start / scale,
                    bounds[0].stop / scale,
                ],
            }
        )
    result.sort(key=lambda item: float(item["area"]), reverse=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--legacy-summary", type=Path, required=True)
    parser.add_argument("--target", type=int, required=True)
    parser.add_argument("--cell", type=int, default=0)
    parser.add_argument("--margin", default="1e-9")
    parser.add_argument("--first-phase-x", type=float)
    parser.add_argument("--first-phase-y", type=float)
    parser.add_argument("--second-phase-x", type=float)
    parser.add_argument("--second-phase-y", type=float)
    parser.add_argument("--reach", type=int, default=8)
    parser.add_argument("--milp-limit", type=float, default=10.0)
    parser.add_argument("--scale", type=int, default=24)
    parser.add_argument("--minimum-gap-area", type=float, default=0.02)
    parser.add_argument("--csv-out", type=Path, required=True)
    parser.add_argument("--png-out", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()

    side, first, second, supports, record = reconstruct(
        args.baseline,
        args.legacy_summary,
        args.target,
        args.cell,
        args.margin,
        (
            None
            if args.first_phase_x is None or args.first_phase_y is None
            else (args.first_phase_x, args.first_phase_y)
        ),
        (
            None
            if args.second_phase_x is None or args.second_phase_y is None
            else (args.second_phase_x, args.second_phase_y)
        ),
    )
    if not cs.separated(first, second):
        raise ValueError("replayed pockets overlap")
    fixed = np.vstack([first, second])
    filled = cs.exact_fill(
        side,
        fixed,
        supports,
        args.reach,
        args.milp_limit,
    )
    image = rasterize(filled, side, args.scale)
    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["n", "square_id", "x", "y", "angle"])
        for square_id, (x, y, angle) in enumerate(filled):
            writer.writerow(
                [
                    args.target,
                    square_id,
                    repr(float(x)),
                    repr(float(y)),
                    repr(float(angle)),
                ]
            )
    write_png(args.png_out, image)
    occupied_area = float(len(filled))
    report = {
        "target": args.target,
        "side": side,
        "cell": args.cell,
        "record": record,
        "count": int(len(filled)),
        "short": args.target - int(len(filled)),
        "capacity_area": side * side,
        "used_area": occupied_area,
        "unused_area": side * side - occupied_area,
        "fillfast": dict(fillfast.LAST),
        "gaps": gap_components(
            filled, side, args.scale, args.minimum_gap_area
        ),
        "csv": str(args.csv_out.resolve()),
        "png": str(args.png_out.resolve()),
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"count={report['count']} short={report['short']} "
        f"unused_area={report['unused_area']:.6f} "
        f"gaps={len(report['gaps'])}",
        flush=True,
    )
    for gap in report["gaps"][:12]:
        print(
            f"  area={gap['area']:.4f} bbox={gap['bbox']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
