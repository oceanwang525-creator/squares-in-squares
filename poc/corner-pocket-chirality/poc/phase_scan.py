#!/usr/bin/env python3
"""Scan continuous corner phases for one pocket/chirality cell."""

from __future__ import annotations

import argparse
import csv
import json
from decimal import Decimal
from pathlib import Path

import numpy as np

import chirality_scan as cs
import scan


def option_symmetry(option_index: int) -> int:
    return option_index // 8


def option_phase(option_index: int) -> tuple[float, float]:
    phase_id = (option_index % 8) // 2
    return ((0.0, 0.0), (0.0, 0.5), (0.5, 0.0), (0.5, 0.5))[phase_id]


def placed_bundle(
    pocket: cs.Pocket,
    symmetry_id: int,
    side: float,
    anchor: str,
    phase: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    oriented, oriented_supports = cs.transform_bundle(
        pocket.squares,
        pocket.supports,
        cs.SYMMETRIES[symmetry_id],
    )
    offset = cs.placement_offset(oriented, side, anchor, phase)
    return cs.shift(oriented, offset), cs.shift(oriented_supports, offset)


def write_candidate(path: Path, target: int, squares: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["n", "square_id", "x", "y", "angle"])
        for square_id, (x, y, angle) in enumerate(squares[:target]):
            writer.writerow(
                [target, square_id, repr(float(x)), repr(float(y)), repr(float(angle))]
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--legacy-summary", type=Path, required=True)
    parser.add_argument("--target", type=int, required=True)
    parser.add_argument("--cell", type=int, default=0)
    parser.add_argument("--margin", default="1e-9")
    parser.add_argument("--phase-min", type=float, default=0.0)
    parser.add_argument("--phase-max", type=float, default=0.95)
    parser.add_argument("--phase-step", type=float, default=0.1)
    parser.add_argument("--phase-x-min", type=float)
    parser.add_argument("--phase-x-max", type=float)
    parser.add_argument("--phase-x-step", type=float)
    parser.add_argument("--phase-y-min", type=float)
    parser.add_argument("--phase-y-max", type=float)
    parser.add_argument("--phase-y-step", type=float)
    parser.add_argument(
        "--scan-pocket",
        choices=("first", "second"),
        default="first",
    )
    parser.add_argument("--first-phase-x", type=float)
    parser.add_argument("--first-phase-y", type=float)
    parser.add_argument("--second-phase-x", type=float)
    parser.add_argument("--second-phase-y", type=float)
    parser.add_argument("--reach", type=int, default=8)
    parser.add_argument("--milp-limit", type=float, default=10.0)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--candidate-out", type=Path)
    args = parser.parse_args()

    packings = scan.read_groups(args.baseline)
    legacy = json.loads(args.legacy_summary.read_text(encoding="utf-8"))
    sources = tuple(int(value) for value in legacy["sources"])
    pockets = cs.extract_pockets(packings, sources, 80, 0.1, 1.5)
    incumbent = Decimal(scan.core.certify(args.target, packings[args.target]))
    side = float(incumbent - Decimal(args.margin))
    record = legacy["targets"][str(args.target)]["top"][args.cell]
    first_index = int(record["first_index"])
    second_index = int(record["second_index"])
    first_option = int(record["first_option"])
    second_option = int(record["second_option"])
    first_symmetry = option_symmetry(first_option)
    second_symmetry = option_symmetry(second_option)
    measured_first_phase = option_phase(first_option)
    measured_second_phase = option_phase(second_option)
    fixed_first_phase = (
        measured_first_phase[0]
        if args.first_phase_x is None
        else args.first_phase_x,
        measured_first_phase[1]
        if args.first_phase_y is None
        else args.first_phase_y,
    )
    fixed_second_phase = (
        measured_second_phase[0]
        if args.second_phase_x is None
        else args.second_phase_x,
        measured_second_phase[1]
        if args.second_phase_y is None
        else args.second_phase_y,
    )
    phase_x_min = (
        args.phase_min if args.phase_x_min is None else args.phase_x_min
    )
    phase_x_max = (
        args.phase_max if args.phase_x_max is None else args.phase_x_max
    )
    phase_x_step = (
        args.phase_step if args.phase_x_step is None else args.phase_x_step
    )
    phase_y_min = (
        args.phase_min if args.phase_y_min is None else args.phase_y_min
    )
    phase_y_max = (
        args.phase_max if args.phase_y_max is None else args.phase_y_max
    )
    phase_y_step = (
        args.phase_step if args.phase_y_step is None else args.phase_y_step
    )
    phase_x = np.arange(
        phase_x_min,
        phase_x_max + 0.5 * phase_x_step,
        phase_x_step,
    )
    phase_y = np.arange(
        phase_y_min,
        phase_y_max + 0.5 * phase_y_step,
        phase_y_step,
    )
    phase_pairs = [
        (round(float(x), 10), round(float(y), 10))
        for x in phase_x
        for y in phase_y
    ]
    results = []
    for phase_index, phase in enumerate(phase_pairs):
        first_phase = phase if args.scan_pocket == "first" else fixed_first_phase
        second_phase = phase if args.scan_pocket == "second" else fixed_second_phase
        first, _ = placed_bundle(
            pockets[first_index], first_symmetry, side, "LL", first_phase
        )
        second, _ = placed_bundle(
            pockets[second_index], second_symmetry, side, "UR", second_phase
        )
        if not cs.separated(first, second):
            results.append(
                {
                    "phase_xy": phase,
                    "first_phase_xy": first_phase,
                    "second_phase_xy": second_phase,
                    "count": -1,
                    "valid": False,
                    "reason": "pocket_overlap",
                }
            )
            continue
        fixed = np.vstack([first, second])
        filled = cs.exact_fill(
            side,
            fixed,
            np.empty((0, 3), dtype=np.float64),
            args.reach,
            args.milp_limit,
        )
        results.append(
            {
                "phase_xy": phase,
                "first_phase_xy": first_phase,
                "second_phase_xy": second_phase,
                "count": int(len(filled)),
                "valid": True,
                "squares": filled,
            }
        )
        if phase_index % 20 == 19:
            best = max(
                int(item["count"]) for item in results if int(item["count"]) >= 0
            )
            print(
                f"phase={phase} tested={phase_index + 1}/{len(phase_pairs)} "
                f"best={best}",
                flush=True,
            )

    results.sort(key=lambda item: int(item["count"]), reverse=True)
    hits = [item for item in results if int(item["count"]) >= args.target]
    if hits and args.candidate_out:
        write_candidate(args.candidate_out, args.target, hits[0]["squares"])
        hits[0]["candidate_path"] = str(args.candidate_out.resolve())
        hits[0]["candidate_L"] = scan.core.certify(
            args.target,
            [
                {
                    "n": str(args.target),
                    "square_id": str(index),
                    "x": repr(float(square[0])),
                    "y": repr(float(square[1])),
                    "angle": repr(float(square[2])),
                }
                for index, square in enumerate(hits[0]["squares"][: args.target])
            ],
        )

    clean = [
        {key: value for key, value in item.items() if key != "squares"}
        for item in results
    ]
    report = {
        "baseline": str(args.baseline),
        "legacy_summary": str(args.legacy_summary),
        "target": args.target,
        "cell": args.cell,
        "incumbent_L": str(incumbent),
        "side": side,
        "margin": args.margin,
        "reach": args.reach,
        "first_index": first_index,
        "second_index": second_index,
        "first_symmetry": first_symmetry,
        "second_symmetry": second_symmetry,
        "scan_pocket": args.scan_pocket,
        "measured_first_phase_xy": measured_first_phase,
        "measured_second_phase_xy": measured_second_phase,
        "fixed_first_phase_xy": fixed_first_phase,
        "fixed_second_phase_xy": fixed_second_phase,
        "phase_grid": {
            "x_min": phase_x_min,
            "x_max": phase_x_max,
            "x_step": phase_x_step,
            "y_min": phase_y_min,
            "y_max": phase_y_max,
            "y_step": phase_y_step,
            "count": len(phase_pairs),
        },
        "best_count": int(results[0]["count"]) if results else None,
        "hits": len(hits),
        "top": clean[:80],
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"target={args.target} side={side:.12f} "
        f"best={report['best_count']} hits={len(hits)}",
        flush=True,
    )
    for item in clean[:10]:
        print(
            f"  phase={item['phase_xy']} count={item['count']} "
            f"valid={item['valid']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
