#!/usr/bin/env python3
"""Replay dual-pocket cells with pocket/support normalization kept rigid."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from decimal import Decimal
from itertools import product
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
MONDRIAN = ROOT / "poc" / "highn-mondrian" / "poc"
sys.path.insert(0, str(MONDRIAN))
import fillfast  # noqa: E402
import scan  # noqa: E402


SYMMETRIES = tuple(
    (swap, sx, sy)
    for swap in (False, True)
    for sx in (1, -1)
    for sy in (1, -1)
)


@dataclass(frozen=True)
class Pocket:
    source_n: int
    component_id: int
    squares: np.ndarray
    supports: np.ndarray


def polygon_bounds(squares: np.ndarray) -> tuple[np.ndarray, ...]:
    x, y, angle = squares.T
    half_extent = 0.5 * (np.abs(np.cos(angle)) + np.abs(np.sin(angle)))
    return (
        x - half_extent,
        x + half_extent,
        y - half_extent,
        y + half_extent,
    )


def box_gap(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    xlo, xhi, ylo, yhi = polygon_bounds(first)
    xlo2, xhi2, ylo2, yhi2 = polygon_bounds(second)
    gx = np.maximum(xlo - xhi2, xlo2 - xhi)
    gy = np.maximum(ylo - yhi2, ylo2 - yhi)
    return np.maximum(gx, gy)


def tilted(squares: np.ndarray, tolerance_deg: float = 2.0) -> np.ndarray:
    angle = np.degrees(squares[:, 2]) % 90.0
    return np.minimum(angle, 90.0 - angle) > tolerance_deg


def components(squares: np.ndarray, gap_tolerance: float) -> list[list[int]]:
    count = len(squares)
    adjacency = [set() for _ in range(count)]
    for start in range(0, count, 256):
        rows = squares[start : start + 256]
        xlo, xhi, ylo, yhi = polygon_bounds(rows)
        all_xlo, all_xhi, all_ylo, all_yhi = polygon_bounds(squares)
        gx = np.maximum(
            xlo[:, None] - all_xhi[None, :],
            all_xlo[None, :] - xhi[:, None],
        )
        gy = np.maximum(
            ylo[:, None] - all_yhi[None, :],
            all_ylo[None, :] - yhi[:, None],
        )
        near = np.maximum(gx, gy) <= gap_tolerance
        for offset, row in enumerate(near):
            index = start + offset
            row[index] = False
            for other in np.flatnonzero(row):
                adjacency[index].add(int(other))
                adjacency[int(other)].add(index)

    seen: set[int] = set()
    result = []
    for start in range(count):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        component = []
        while stack:
            current = stack.pop()
            component.append(current)
            for other in adjacency[current]:
                if other not in seen:
                    seen.add(other)
                    stack.append(other)
        result.append(component)
    return result


def exit_supports(
    all_squares: np.ndarray,
    pocket: np.ndarray,
    margin: float,
) -> np.ndarray:
    axis = ~tilted(all_squares, tolerance_deg=1.0)
    xlo, xhi, ylo, yhi = polygon_bounds(pocket)
    bounds = (
        xlo.min() - margin,
        xhi.max() + margin,
        ylo.min() - margin,
        yhi.max() + margin,
    )
    selected = []
    for index in np.flatnonzero(axis):
        square = all_squares[index : index + 1]
        if box_gap(square, pocket).min() > 1e-6:
            continue
        center = all_squares[index, :2]
        if (
            bounds[0] <= center[0] <= bounds[1]
            and bounds[2] <= center[1] <= bounds[3]
        ):
            selected.append(index)
    return all_squares[selected].copy().reshape(-1, 3)


def normalize_bundle(
    squares: np.ndarray,
    supports: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    normalized_squares = np.asarray(squares, dtype=np.float64).copy()
    normalized_supports = np.asarray(supports, dtype=np.float64).copy()
    xlo, _, ylo, _ = polygon_bounds(normalized_squares)
    low = np.array([float(xlo.min()), float(ylo.min())])
    normalized_squares[:, :2] -= low
    if len(normalized_supports):
        normalized_supports[:, :2] -= low
    return normalized_squares, normalized_supports


def extract_pockets(
    packings: dict[int, list[dict[str, str]]],
    sources: tuple[int, ...],
    min_count: int,
    gap_tolerance: float,
    support_margin: float,
) -> list[Pocket]:
    result = []
    for source_n in sources:
        rows = packings[source_n]
        all_squares = np.asarray(
            [(float(row["x"]), float(row["y"]), float(row["angle"])) for row in rows],
            dtype=np.float64,
        )
        selected = all_squares[tilted(all_squares)]
        found = []
        for indices in components(selected, gap_tolerance):
            if len(indices) >= min_count:
                found.append(selected[indices])
        found.sort(key=lambda value: (-len(value), float(value[:, 1].max())))
        for component_id, squares in enumerate(found):
            supports = exit_supports(all_squares, squares, support_margin)
            normalized_squares, normalized_supports = normalize_bundle(
                squares, supports
            )
            result.append(
                Pocket(
                    source_n=source_n,
                    component_id=component_id,
                    squares=normalized_squares,
                    supports=normalized_supports,
                )
            )
    return result


def transform_raw(
    squares: np.ndarray,
    symmetry: tuple[bool, int, int],
) -> np.ndarray:
    result = np.asarray(squares, dtype=np.float64).copy()
    swap, sx, sy = symmetry
    x, y, angle = result.T
    if swap:
        x, y = y, x
        angle = math.pi / 2.0 - angle
    x *= sx
    y *= sy
    angle = np.arctan2(np.sin(angle) * sy, np.cos(angle) * sx)
    return np.column_stack([x, y, angle])


def transform_bundle(
    squares: np.ndarray,
    supports: np.ndarray,
    symmetry: tuple[bool, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    oriented_squares = transform_raw(squares, symmetry)
    oriented_supports = transform_raw(supports, symmetry)
    return normalize_bundle(oriented_squares, oriented_supports)


def placement_offset(
    squares: np.ndarray,
    side: float,
    anchor: str,
    phase: tuple[float, float],
) -> np.ndarray:
    xlo, xhi, ylo, yhi = polygon_bounds(squares)
    px, py = phase
    if anchor == "LL":
        return np.array([px - xlo.min(), py - ylo.min()])
    if anchor == "UR":
        return np.array(
            [
                side - px - xhi.max(),
                side - py - yhi.max(),
            ]
        )
    raise ValueError(anchor)


def shift(squares: np.ndarray, offset: np.ndarray) -> np.ndarray:
    result = np.asarray(squares, dtype=np.float64).copy()
    result[:, :2] += offset
    return result


def placement_options(
    pocket: Pocket,
    side: float,
    phases: tuple[tuple[float, float], ...],
) -> list[dict[str, object]]:
    options = []
    seen: set[bytes] = set()
    for symmetry_id, symmetry in enumerate(SYMMETRIES):
        oriented, oriented_supports = transform_bundle(
            pocket.squares, pocket.supports, symmetry
        )
        for phase_id, phase in enumerate(phases):
            for anchor in ("LL", "UR"):
                offset = placement_offset(oriented, side, anchor, phase)
                placed = shift(oriented, offset)
                xlo, xhi, ylo, yhi = polygon_bounds(placed)
                if (
                    xlo.min() < -1e-7
                    or ylo.min() < -1e-7
                    or xhi.max() > side + 1e-7
                    or yhi.max() > side + 1e-7
                ):
                    continue
                ordered = placed[
                    np.lexsort((placed[:, 2], placed[:, 1], placed[:, 0]))
                ]
                pose_key = np.round(ordered, 10).tobytes()
                if pose_key in seen:
                    continue
                seen.add(pose_key)
                options.append(
                    {
                        "symmetry": symmetry_id,
                        "phase": phase_id,
                        "phase_xy": phase,
                        "anchor": anchor,
                        "squares": placed,
                        "supports": shift(oriented_supports, offset),
                    }
                )
    return options


def separated(first: np.ndarray, second: np.ndarray, tolerance: float = 2e-8) -> bool:
    if not len(first) or not len(second):
        return True
    xlo, xhi, ylo, yhi = polygon_bounds(first)
    xlo2, xhi2, ylo2, yhi2 = polygon_bounds(second)
    candidate = (
        (xhi[:, None] >= xlo2[None, :] - tolerance)
        & (xhi2[None, :] >= xlo[:, None] - tolerance)
        & (yhi[:, None] >= ylo2[None, :] - tolerance)
        & (yhi2[None, :] >= ylo[:, None] - tolerance)
    )
    rows, cols = np.nonzero(candidate)
    if not len(rows):
        return True
    dx = second[cols, 0] - first[rows, 0]
    dy = second[cols, 1] - first[rows, 1]
    maximum = np.full(len(rows), -np.inf)
    for owner_is_first in (True, False):
        for quarter_turn in (0.0, math.pi / 2.0):
            owner = first if owner_is_first else second
            owner_index = rows if owner_is_first else cols
            angle = owner[owner_index, 2] + quarter_turn
            ax, ay = np.cos(angle), np.sin(angle)
            radius_owner = 0.5 * (np.abs(np.cos(angle)) + np.abs(np.sin(angle)))
            if owner_is_first:
                other_angle = second[cols, 2]
                other_dx, other_dy = dx, dy
            else:
                other_angle = first[rows, 2]
                other_dx, other_dy = -dx, -dy
            relative = angle - other_angle
            radius_other = 0.5 * (
                np.abs(np.cos(relative)) + np.abs(np.sin(relative))
            )
            separation = np.abs(other_dx * ax + other_dy * ay) - (
                radius_owner + radius_other
            )
            maximum = np.maximum(maximum, separation)
    return bool((maximum > tolerance).all())


def exact_fill(
    side: float,
    fixed: np.ndarray,
    extra: np.ndarray,
    reach: int,
    time_limit: float,
    lattice_offset: tuple[float, float] | None = None,
) -> np.ndarray:
    candidates = fillfast.candidates(side, fixed, reach=reach)
    filtered = []
    for square in extra:
        square = square.reshape(1, 3)
        xlo, xhi, ylo, yhi = polygon_bounds(square)
        if (
            xlo[0] < 0.5 - 1e-9
            or ylo[0] < 0.5 - 1e-9
            or xhi[0] > side - 0.5 + 1e-9
            or yhi[0] > side - 0.5 + 1e-9
        ):
            continue
        if separated(square, fixed):
            filtered.append((float(square[0, 0]), float(square[0, 1]), 0.0))
    if lattice_offset is not None:
        low = 0.5
        high = side - 0.5
        offset_x, offset_y = lattice_offset
        base_x = low + offset_x % 1.0
        base_y = low + offset_y % 1.0
        x_values = np.arange(
            base_x - math.floor(base_x - low),
            high + 1e-10,
            1.0,
        )
        y_values = np.arange(
            base_y - math.floor(base_y - low),
            high + 1e-10,
            1.0,
        )
        x_values = x_values[(x_values >= low) & (x_values <= high)]
        y_values = y_values[(y_values >= low) & (y_values <= high)]
        for x in x_values:
            for y in y_values:
                square = np.array([[x, y, 0.0]], dtype=np.float64)
                if separated(square, fixed):
                    filtered.append((float(x), float(y), 0.0))
    all_candidates = np.asarray(candidates + filtered, dtype=np.float64).reshape(-1, 3)
    all_candidates = np.unique(np.round(all_candidates, 12), axis=0)
    x, y, adjacency = fillfast.conflict_graph(all_candidates)
    taken, alive = fillfast.corner_reduce(x, y, adjacency)
    chosen = list(taken)
    for component in fillfast.components(alive, adjacency):
        if len(component) == 1:
            chosen.append(component[0])
        else:
            chosen += fillfast._milp(component, adjacency, time_limit)
    return np.r_[fixed, all_candidates[chosen]]


def add_supports(fixed: np.ndarray, supports: np.ndarray) -> tuple[np.ndarray, int]:
    accepted = []
    current = fixed
    for square in supports:
        square = square.reshape(1, 3)
        if not separated(square, current, tolerance=1e-9):
            continue
        if np.min(np.max(np.abs(current - square), axis=1)) <= 1e-9:
            continue
        xlo, xhi, ylo, yhi = polygon_bounds(square)
        side = max(float(xhi[0]), float(yhi[0]))
        if xlo[0] < -1e-7 or ylo[0] < -1e-7:
            continue
        accepted.append(square[0])
        current = np.vstack([current, square])
    if accepted:
        current = np.vstack([fixed, np.asarray(accepted)])
    return current, len(accepted)


def evaluate(
    first: np.ndarray,
    second: np.ndarray,
    support_a: np.ndarray,
    support_b: np.ndarray,
    side: float,
    support_mode: str,
    reach: int,
    time_limit: float,
) -> dict[str, object]:
    if not separated(first, second):
        return {"count": -1, "valid": False, "reason": "pocket_overlap"}
    tilted_squares = np.vstack([first, second])
    supports = np.vstack([support_a, support_b]).reshape(-1, 3)
    used_supports = 0
    if support_mode == "fixed":
        fixed, used_supports = add_supports(tilted_squares, supports)
        filled = exact_fill(
            side,
            fixed,
            np.empty((0, 3), dtype=np.float64),
            reach,
            time_limit,
        )
    elif support_mode == "candidate":
        fixed = tilted_squares
        filled = exact_fill(side, fixed, supports, reach, time_limit)
    elif support_mode == "none":
        fixed = tilted_squares
        filled = exact_fill(
            side,
            fixed,
            np.empty((0, 3), dtype=np.float64),
            reach,
            time_limit,
        )
    else:
        raise ValueError(support_mode)
    return {
        "count": int(len(filled)),
        "valid": True,
        "tilted_count": int(len(tilted_squares)),
        "used_support_count": used_supports,
        "upright_count": int(len(filled) - len(fixed)),
        "squares": filled,
    }


def record_key(record: dict[str, object]) -> tuple[int, ...]:
    return (
        int(record["first_index"]),
        int(record["second_index"]),
        int(record["first_option"]),
        int(record["second_option"]),
    )


def write_candidate(path: Path, n: int, squares: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["n", "square_id", "x", "y", "angle"])
        for square_id, (x, y, angle) in enumerate(squares[:n]):
            writer.writerow(
                [n, square_id, repr(float(x)), repr(float(y)), repr(float(angle))]
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline",
        type=Path,
        default=ROOT
        / "poc"
        / "pell_corrugation"
        / "data"
        / "border-scan"
        / "merged.csv",
    )
    parser.add_argument("--legacy-summary", type=Path, required=True)
    parser.add_argument("--target", type=int, required=True)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--margin", default="1e-9")
    parser.add_argument("--min-pocket-count", type=int, default=80)
    parser.add_argument("--gap-tolerance", type=float, default=0.1)
    parser.add_argument("--support-margin", type=float, default=1.5)
    parser.add_argument("--phase-step", type=float, choices=(0.25, 0.5), default=0.5)
    parser.add_argument("--reach", type=int, default=8)
    parser.add_argument("--milp-limit", type=float, default=5.0)
    parser.add_argument(
        "--support-modes",
        default="none,candidate,fixed",
        help="comma-separated subset of none,candidate,fixed",
    )
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--candidate-out", type=Path)
    args = parser.parse_args()

    packings = scan.read_groups(args.baseline)
    legacy = json.loads(args.legacy_summary.read_text(encoding="utf-8"))
    sources = tuple(int(value) for value in legacy["sources"])
    pockets = extract_pockets(
        packings,
        sources,
        args.min_pocket_count,
        args.gap_tolerance,
        args.support_margin,
    )
    actual = [
        (
            pocket.source_n,
            pocket.component_id,
            len(pocket.squares),
            len(pocket.supports),
        )
        for pocket in pockets
    ]
    expected = [
        (
            int(item["source_n"]),
            int(item["component_id"]),
            int(item["tilted_count"]),
            int(item["axis_support_count"]),
        )
        for item in legacy["pockets"]
    ]
    if actual != expected:
        raise ValueError(f"legacy pocket roster changed: expected {expected}, got {actual}")

    incumbent = Decimal(scan.core.certify(args.target, packings[args.target]))
    side = incumbent - Decimal(args.margin)
    side_float = float(side)
    phases = tuple(
        (x, y)
        for x in (0.0, args.phase_step)
        for y in (0.0, args.phase_step)
    )
    option_sets = [
        placement_options(pocket, side_float, phases) for pocket in pockets
    ]
    target_data = legacy["targets"][str(args.target)]
    records = target_data["top"][: args.top]
    modes = tuple(value.strip() for value in args.support_modes.split(","))

    seen = set()
    replay = []
    for record in records:
        key = record_key(record)
        if key in seen:
            continue
        seen.add(key)
        first_index = int(record["first_index"])
        second_index = int(record["second_index"])
        first_option_index = int(record["first_option"])
        second_option_index = int(record["second_option"])
        if first_option_index >= len(option_sets[first_index]):
            continue
        if second_option_index >= len(option_sets[second_index]):
            continue
        first_option = option_sets[first_index][first_option_index]
        second_option = option_sets[second_index][second_option_index]
        if first_option["anchor"] != "LL" or second_option["anchor"] != "UR":
            continue
        for mode in modes:
            outcome = evaluate(
                first_option["squares"],
                second_option["squares"],
                first_option["supports"],
                second_option["supports"],
                side_float,
                mode,
                args.reach,
                args.milp_limit,
            )
            replay.append(
                {
                    "mode": mode,
                    "reach": args.reach,
                    "count": outcome["count"],
                    "valid": outcome["valid"],
                    "tilted_count": outcome.get("tilted_count"),
                    "used_support_count": outcome.get("used_support_count"),
                    "upright_count": outcome.get("upright_count"),
                    "first_index": first_index,
                    "second_index": second_index,
                    "first_source": pockets[first_index].source_n,
                    "first_component": pockets[first_index].component_id,
                    "second_source": pockets[second_index].source_n,
                    "second_component": pockets[second_index].component_id,
                    "first_option": first_option_index,
                    "second_option": second_option_index,
                    "first_symmetry": first_option["symmetry"],
                    "second_symmetry": second_option["symmetry"],
                    "first_phase_xy": first_option["phase_xy"],
                    "second_phase_xy": second_option["phase_xy"],
                    "squares": outcome.get("squares"),
                }
            )

    replay.sort(
        key=lambda item: (
            int(item["count"]),
            int(item["used_support_count"] or 0),
        ),
        reverse=True,
    )
    hits = [item for item in replay if int(item["count"]) >= args.target]
    if hits and args.candidate_out:
        best = hits[0]
        write_candidate(args.candidate_out, args.target, best["squares"])
        best["candidate_path"] = str(args.candidate_out.resolve())
        best["candidate_L"] = scan.core.certify(
            args.target,
            [
                {
                    "n": str(args.target),
                    "square_id": str(index),
                    "x": repr(float(square[0])),
                    "y": repr(float(square[1])),
                    "angle": repr(float(square[2])),
                }
                for index, square in enumerate(best["squares"][: args.target])
            ],
        )

    clean = [
        {key: value for key, value in item.items() if key != "squares"}
        for item in replay
    ]
    report = {
        "baseline": str(args.baseline),
        "legacy_summary": str(args.legacy_summary),
        "target": args.target,
        "incumbent_L": str(incumbent),
        "side": str(side),
        "margin": args.margin,
        "sources": sources,
        "pockets": [
            {
                "source_n": pocket.source_n,
                "component_id": pocket.component_id,
                "tilted_count": len(pocket.squares),
                "support_count": len(pocket.supports),
            }
            for pocket in pockets
        ],
        "replayed_cells": len(seen),
        "support_modes": modes,
        "reach": args.reach,
        "best_count": int(replay[0]["count"]) if replay else None,
        "hits": len(hits),
        "top": clean[:40],
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"target={args.target} side={side} replayed={len(seen)} "
        f"best={report['best_count']} hits={len(hits)}",
        flush=True,
    )
    for item in clean[:10]:
        print(
            f"  {item['mode']:9s} count={item['count']} "
            f"supports={item['used_support_count']} "
            f"cell={item['first_source']}:{item['first_component']}x"
            f"{item['second_source']}:{item['second_component']} "
            f"opts={item['first_option']},{item['second_option']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
