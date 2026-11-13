#!/usr/bin/env python3
"""Scan one selected pair from the extracted pocket catalog."""

from __future__ import annotations

import argparse
import csv
import json
from decimal import Decimal
from pathlib import Path

import numpy as np

import chirality_scan as cs
import scan


def parse_pair(value: str) -> tuple[int, int]:
    parts = tuple(int(part) for part in value.split(","))
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("expected INDEX,COMPONENT")
    return parts


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
    parser.add_argument("--sources", required=True)
    parser.add_argument("--first-pocket", type=parse_pair, required=True)
    parser.add_argument("--second-pocket", type=parse_pair, required=True)
    parser.add_argument("--first-symmetry", type=int, required=True)
    parser.add_argument("--second-symmetry", type=int, required=True)
    parser.add_argument("--all-symmetries", action="store_true")
    parser.add_argument("--target", type=int, required=True)
    parser.add_argument("--margin", default="1e-9")
    parser.add_argument("--phase-step", type=float, default=0.5)
    parser.add_argument("--first-phase-x", type=float)
    parser.add_argument("--first-phase-y", type=float)
    parser.add_argument("--second-phase-x", type=float)
    parser.add_argument("--second-phase-y", type=float)
    parser.add_argument("--reach", type=int, default=8)
    parser.add_argument("--milp-limit", type=float, default=10.0)
    parser.add_argument("--min-pocket-count", type=int, default=80)
    parser.add_argument("--gap-tolerance", type=float, default=0.1)
    parser.add_argument("--support-margin", type=float, default=1.5)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--candidate-out", type=Path)
    args = parser.parse_args()

    sources = tuple(int(value) for value in args.sources.split(","))
    packings = scan.read_groups(args.baseline)
    pockets = cs.extract_pockets(
        packings,
        sources,
        args.min_pocket_count,
        args.gap_tolerance,
        args.support_margin,
    )

    def locate(selector: tuple[int, int]) -> tuple[int, cs.Pocket]:
        source_n, component_id = selector
        for index, pocket in enumerate(pockets):
            if (
                pocket.source_n == source_n
                and pocket.component_id == component_id
            ):
                return index, pocket
        raise ValueError(f"pocket {selector} not found")

    first_index, first_pocket = locate(args.first_pocket)
    second_index, second_pocket = locate(args.second_pocket)
    incumbent = Decimal(scan.core.certify(args.target, packings[args.target]))
    side = float(incumbent - Decimal(args.margin))
    fixed_phase_values = (
        args.first_phase_x,
        args.first_phase_y,
        args.second_phase_x,
        args.second_phase_y,
    )
    if any(value is not None for value in fixed_phase_values):
        if any(value is None for value in fixed_phase_values):
            raise ValueError("supply all four fixed phase coordinates")
        phases = (
            (
                float(args.first_phase_x),
                float(args.first_phase_y),
                float(args.second_phase_x),
                float(args.second_phase_y),
            ),
        )
    else:
        phase_values = np.arange(
            0.0,
            1.0 - 0.5 * args.phase_step,
            args.phase_step,
        )
        phases = tuple(
            (round(float(x), 10), round(float(y), 10))
            for x in phase_values
            for y in phase_values
        )
    results = []
    symmetry_pairs = (
        tuple(
            (first_symmetry, second_symmetry)
            for first_symmetry in range(len(cs.SYMMETRIES))
            for second_symmetry in range(len(cs.SYMMETRIES))
        )
        if args.all_symmetries
        else ((args.first_symmetry, args.second_symmetry),)
    )
    for first_symmetry, second_symmetry in symmetry_pairs:
        first_squares, first_supports = cs.transform_bundle(
            first_pocket.squares,
            first_pocket.supports,
            cs.SYMMETRIES[first_symmetry],
        )
        second_squares, second_supports = cs.transform_bundle(
            second_pocket.squares,
            second_pocket.supports,
            cs.SYMMETRIES[second_symmetry],
        )
        for phase_record in phases:
            if len(phase_record) == 4:
                first_phase = phase_record[:2]
                second_phase = phase_record[2:]
                phase = (first_phase, second_phase)
            else:
                first_phase = phase_record
                second_phase = phase_record
                phase = phase_record
            first_offset = cs.placement_offset(
                first_squares, side, "LL", first_phase
            )
            second_offset = cs.placement_offset(
                second_squares, side, "UR", second_phase
            )
            first = cs.shift(first_squares, first_offset)
            second = cs.shift(second_squares, second_offset)
            first_axis = cs.shift(first_supports, first_offset)
            second_axis = cs.shift(second_supports, second_offset)
            if not cs.separated(first, second):
                results.append(
                    {
                        "first_symmetry": first_symmetry,
                        "second_symmetry": second_symmetry,
                        "phase_xy": phase,
                        "count": -1,
                        "valid": False,
                        "reason": "pocket_overlap",
                    }
                )
                continue
            fixed = np.vstack([first, second])
            supports = np.vstack([first_axis, second_axis]).reshape(-1, 3)
            filled = cs.exact_fill(
                side,
                fixed,
                supports,
                args.reach,
                args.milp_limit,
            )
            results.append(
                {
                    "first_symmetry": first_symmetry,
                    "second_symmetry": second_symmetry,
                    "phase_xy": phase,
                    "count": int(len(filled)),
                    "valid": True,
                    "tilted_count": int(len(fixed)),
                    "squares": filled,
                }
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
        "sources": sources,
        "target": args.target,
        "incumbent_L": str(incumbent),
        "side": side,
        "margin": args.margin,
        "reach": args.reach,
        "pockets": [
            {
                "index": index,
                "source_n": pocket.source_n,
                "component_id": pocket.component_id,
                "tilted_count": len(pocket.squares),
                "support_count": len(pocket.supports),
            }
            for index, pocket in enumerate(pockets)
        ],
        "first_index": first_index,
        "second_index": second_index,
        "first_symmetry": args.first_symmetry,
        "second_symmetry": args.second_symmetry,
        "all_symmetries": args.all_symmetries,
        "phase_step": args.phase_step,
        "best_count": int(results[0]["count"]) if results else None,
        "hits": len(hits),
        "top": clean[:40],
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"target={args.target} side={side:.12f} "
        f"pockets={first_pocket.source_n}:{first_pocket.component_id}x"
        f"{second_pocket.source_n}:{second_pocket.component_id} "
        f"sym={args.first_symmetry},{args.second_symmetry} "
        f"best={report['best_count']} hits={len(hits)}",
        flush=True,
    )
    for item in clean[:8]:
        print(
            f"  sym={item['first_symmetry']},{item['second_symmetry']} "
            f"phase={item['phase_xy']} count={item['count']} valid={item['valid']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
