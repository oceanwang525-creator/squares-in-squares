#!/usr/bin/env python3
"""Compose two narrowed pockets and exactly refill the upright remainder."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from decimal import Decimal
from pathlib import Path

for _var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_var] = "1"

import numpy as np

import chirality_scan as cs
import scan


def load_pocket(path: Path) -> cs.Pocket:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "ok":
        raise ValueError(f"{path} is not a successful narrowing result")
    squares = np.asarray(report["squares"], dtype=np.float64).reshape(-1, 3)
    normalized, supports = cs.normalize_bundle(
        squares, np.empty((0, 3), dtype=np.float64)
    )
    return cs.Pocket(
        source_n=int(report["source_n"]),
        component_id=int(report["component_id"]),
        squares=normalized,
        supports=supports,
    )


def rotate_pocket(pocket: cs.Pocket, degrees: float) -> cs.Pocket:
    if abs(degrees) <= 1e-15:
        return pocket
    squares = pocket.squares.copy()
    supports = pocket.supports.copy()
    xlo, xhi, ylo, yhi = cs.polygon_bounds(squares)
    pivot = np.array([0.5 * (xlo.min() + xhi.max()), 0.5 * (ylo.min() + yhi.max())])
    angle = math.radians(degrees)
    rotation = np.array(
        [
            [math.cos(angle), -math.sin(angle)],
            [math.sin(angle), math.cos(angle)],
        ]
    )
    squares[:, :2] = (squares[:, :2] - pivot) @ rotation.T + pivot
    squares[:, 2] += angle
    if len(supports):
        supports[:, :2] = (supports[:, :2] - pivot) @ rotation.T + pivot
        supports[:, 2] += angle
    normalized, supports = cs.normalize_bundle(
        squares, supports
    )
    return cs.Pocket(
        source_n=pocket.source_n,
        component_id=pocket.component_id,
        squares=normalized,
        supports=supports,
    )


def attach_original_supports(
    pocket: cs.Pocket,
    packings: dict[int, list[dict[str, str]]],
) -> cs.Pocket:
    candidates = cs.extract_pockets(
        packings,
        (pocket.source_n,),
        80,
        0.1,
        1.5,
    )
    original = next(
        item
        for item in candidates
        if item.component_id == pocket.component_id
    )
    return cs.Pocket(
        source_n=pocket.source_n,
        component_id=pocket.component_id,
        squares=pocket.squares,
        supports=original.supports,
    )


def phase_grid(
    low_x: float,
    high_x: float,
    step_x: float,
    low_y: float,
    high_y: float,
    step_y: float,
) -> tuple[tuple[float, float], ...]:
    if step_x <= 0.0 or step_y <= 0.0:
        raise ValueError("phase steps must be positive")
    x_values = np.arange(low_x, high_x + 0.5 * step_x, step_x)
    y_values = np.arange(low_y, high_y + 0.5 * step_y, step_y)
    return tuple(
        (round(float(x), 10), round(float(y), 10))
        for x in x_values
        for y in y_values
    )


def phase_cases(
    mode: str,
    grid: tuple[tuple[float, float], ...],
    first_fixed: tuple[float, float],
    second_fixed: tuple[float, float],
) -> tuple[tuple[tuple[float, float], tuple[float, float]], ...]:
    if mode == "fixed":
        return ((first_fixed, second_fixed),)
    if mode == "same":
        return tuple((phase, phase) for phase in grid)
    if mode == "first":
        return tuple((phase, second_fixed) for phase in grid)
    if mode == "second":
        return tuple((first_fixed, phase) for phase in grid)
    if mode == "cartesian":
        return tuple((first, second) for first in grid for second in grid)
    raise ValueError(f"unknown phase mode {mode}")


def write_candidate(path: Path, target: int, squares: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["n", "square_id", "x", "y", "angle"])
        for square_id, (x, y, angle) in enumerate(squares[:target]):
            writer.writerow(
                [target, square_id, repr(float(x)), repr(float(y)), repr(float(angle))]
            )


def evaluate_case(
    task: tuple[
        int,
        int,
        tuple[float, float],
        tuple[float, float],
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        float,
        int,
        float,
        tuple[float, float] | None,
        bool,
    ],
) -> dict[str, object]:
    (
        first_symmetry,
        second_symmetry,
        first_phase,
        second_phase,
        first_oriented,
        second_oriented,
        first_supports,
        second_supports,
        side,
        reach,
        milp_limit,
        lattice_offset,
        candidate_supports,
    ) = task
    first_offset = cs.placement_offset(first_oriented, side, "LL", first_phase)
    second_offset = cs.placement_offset(second_oriented, side, "UR", second_phase)
    first = cs.shift(first_oriented, first_offset)
    second = cs.shift(second_oriented, second_offset)
    base = {
        "first_symmetry": first_symmetry,
        "second_symmetry": second_symmetry,
        "first_phase_xy": first_phase,
        "second_phase_xy": second_phase,
    }
    if not cs.separated(first, second):
        return {
            **base,
            "count": -1,
            "valid": False,
            "reason": "pocket_overlap",
        }
    fixed = np.vstack([first, second])
    extra = (
        np.vstack([first_supports, second_supports]).reshape(-1, 3)
        if candidate_supports
        else np.empty((0, 3), dtype=np.float64)
    )
    filled = cs.exact_fill(
        side,
        fixed,
        extra,
        reach,
        milp_limit,
        lattice_offset,
    )
    return {
        **base,
        "count": int(len(filled)),
        "valid": True,
        "tilted_count": int(len(fixed)),
        "squares": filled,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--first-summary", type=Path, required=True)
    parser.add_argument("--second-summary", type=Path, required=True)
    parser.add_argument("--first-rotation-deg", type=float, default=0.0)
    parser.add_argument("--second-rotation-deg", type=float, default=0.0)
    parser.add_argument("--candidate-supports", action="store_true")
    parser.add_argument("--target", type=int, required=True)
    parser.add_argument("--first-symmetry", type=int)
    parser.add_argument("--second-symmetry", type=int)
    parser.add_argument("--all-symmetries", action="store_true")
    parser.add_argument(
        "--phase-mode",
        choices=("fixed", "same", "first", "second", "cartesian"),
        default="first",
    )
    parser.add_argument("--phase-min", type=float, default=0.0)
    parser.add_argument("--phase-max", type=float, default=0.95)
    parser.add_argument("--phase-step", type=float, default=0.1)
    parser.add_argument("--phase-x-min", type=float)
    parser.add_argument("--phase-x-max", type=float)
    parser.add_argument("--phase-x-step", type=float)
    parser.add_argument("--phase-y-min", type=float)
    parser.add_argument("--phase-y-max", type=float)
    parser.add_argument("--phase-y-step", type=float)
    parser.add_argument("--first-phase-x", type=float, default=0.0)
    parser.add_argument("--first-phase-y", type=float, default=0.0)
    parser.add_argument("--second-phase-x", type=float, default=0.0)
    parser.add_argument("--second-phase-y", type=float, default=0.0)
    parser.add_argument("--margin", default="1e-9")
    parser.add_argument("--reach", type=int, default=8)
    parser.add_argument("--milp-limit", type=float, default=10.0)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--upright-offset")
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--candidate-out", type=Path)
    parser.add_argument("--best-out", type=Path)
    args = parser.parse_args()
    if args.jobs < 1:
        raise ValueError("--jobs must be positive")

    if args.all_symmetries:
        if args.first_symmetry is not None or args.second_symmetry is not None:
            raise ValueError("--all-symmetries excludes fixed symmetry arguments")
        symmetry_pairs = tuple(
            (first, second)
            for first in range(len(cs.SYMMETRIES))
            for second in range(len(cs.SYMMETRIES))
        )
    else:
        if args.first_symmetry is None or args.second_symmetry is None:
            raise ValueError("supply both fixed symmetries or --all-symmetries")
        symmetry_pairs = ((args.first_symmetry, args.second_symmetry),)

    packings = scan.read_groups(args.baseline)
    incumbent = Decimal(scan.core.certify(args.target, packings[args.target]))
    side = float(incumbent - Decimal(args.margin))
    lattice_offset = (
        None
        if args.upright_offset is None
        else tuple(float(value) for value in args.upright_offset.split(","))
    )
    if lattice_offset is not None and len(lattice_offset) != 2:
        raise ValueError("--upright-offset expects X,Y")
    first_pocket = load_pocket(args.first_summary)
    second_pocket = load_pocket(args.second_summary)
    if args.candidate_supports:
        first_pocket = attach_original_supports(first_pocket, packings)
        second_pocket = attach_original_supports(second_pocket, packings)
    first_pocket = rotate_pocket(first_pocket, args.first_rotation_deg)
    second_pocket = rotate_pocket(second_pocket, args.second_rotation_deg)
    phase_x_min = args.phase_min if args.phase_x_min is None else args.phase_x_min
    phase_x_max = args.phase_max if args.phase_x_max is None else args.phase_x_max
    phase_x_step = args.phase_step if args.phase_x_step is None else args.phase_x_step
    phase_y_min = args.phase_min if args.phase_y_min is None else args.phase_y_min
    phase_y_max = args.phase_max if args.phase_y_max is None else args.phase_y_max
    phase_y_step = args.phase_step if args.phase_y_step is None else args.phase_y_step
    grid = phase_grid(
        phase_x_min,
        phase_x_max,
        phase_x_step,
        phase_y_min,
        phase_y_max,
        phase_y_step,
    )
    cases = phase_cases(
        args.phase_mode,
        grid,
        (args.first_phase_x, args.first_phase_y),
        (args.second_phase_x, args.second_phase_y),
    )

    tasks = []
    for first_symmetry, second_symmetry in symmetry_pairs:
        first_oriented, _ = cs.transform_bundle(
            first_pocket.squares,
            first_pocket.supports,
            cs.SYMMETRIES[first_symmetry],
        )
        second_oriented, _ = cs.transform_bundle(
            second_pocket.squares,
            second_pocket.supports,
            cs.SYMMETRIES[second_symmetry],
        )
        for first_phase, second_phase in cases:
            tasks.append(
                (
                    first_symmetry,
                    second_symmetry,
                    first_phase,
                    second_phase,
                    first_oriented,
                    second_oriented,
                    first_pocket.supports,
                    second_pocket.supports,
                    side,
                    args.reach,
                    args.milp_limit,
                    lattice_offset,
                    args.candidate_supports,
                )
            )
    results = []
    tested = 0
    if args.jobs == 1:
        iterator = map(evaluate_case, tasks)
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=args.jobs)
        iterator = executor.map(evaluate_case, tasks, chunksize=1)
    try:
        for result in iterator:
            results.append(result)
            tested += 1
            if tested % 16 == 0 or tested == len(tasks):
                best = max(int(item["count"]) for item in results)
                print(f"tested={tested}/{len(tasks)} best={best}", flush=True)
    finally:
        if executor is not None:
            executor.shutdown()

    results.sort(key=lambda item: int(item["count"]), reverse=True)
    if args.best_out and results and int(results[0]["count"]) >= 0:
        write_candidate(args.best_out, args.target, results[0]["squares"])
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
        "target": args.target,
        "incumbent_L": str(incumbent),
        "side": side,
        "margin": args.margin,
        "first_summary": str(args.first_summary),
        "second_summary": str(args.second_summary),
        "first_rotation_deg": args.first_rotation_deg,
        "second_rotation_deg": args.second_rotation_deg,
        "candidate_supports": args.candidate_supports,
        "first_pocket": {
            "source_n": first_pocket.source_n,
            "component_id": first_pocket.component_id,
            "count": len(first_pocket.squares),
        },
        "second_pocket": {
            "source_n": second_pocket.source_n,
            "component_id": second_pocket.component_id,
            "count": len(second_pocket.squares),
        },
        "symmetry_pairs": [list(pair) for pair in symmetry_pairs],
        "phase_mode": args.phase_mode,
        "phase_grid": {
            "x_min": phase_x_min,
            "x_max": phase_x_max,
            "x_step": phase_x_step,
            "y_min": phase_y_min,
            "y_max": phase_y_max,
            "y_step": phase_y_step,
            "count": len(cases),
        },
        "tested": tested,
        "reach": args.reach,
        "upright_offset": lattice_offset,
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
        f"target={args.target} side={side:.12f} tested={tested} "
        f"best={report['best_count']} hits={len(hits)}",
        flush=True,
    )
    for item in clean[:10]:
        print(
            f"  sym={item['first_symmetry']},{item['second_symmetry']} "
            f"phase={item['first_phase_xy']},{item['second_phase_xy']} "
            f"count={item['count']} valid={item['valid']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
