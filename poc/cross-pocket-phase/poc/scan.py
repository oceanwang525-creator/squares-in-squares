#!/usr/bin/env python3
"""Scan independent cross-pocket phases and exactly refill upright squares."""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import sys
import time
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

for variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[variable] = "1"

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "poc" / "corner-pocket-chirality" / "poc"))
sys.path.insert(0, str(ROOT / "poc" / "highn-mondrian" / "poc"))

import chirality_scan as cs  # noqa: E402
import scan as mondrian  # noqa: E402


@dataclass(frozen=True)
class Task:
    task_id: int
    first_symmetry: int
    second_symmetry: int
    first_phase: tuple[float, float]
    second_phase: tuple[float, float]
    first: np.ndarray
    second: np.ndarray
    first_supports: np.ndarray
    second_supports: np.ndarray
    side: float
    reach: int
    milp_limit: float
    upright_offset: tuple[float, float] | None
    include_supports: bool


def locate_pocket(
    pockets: list[cs.Pocket],
    source_n: int,
    component_id: int,
) -> cs.Pocket:
    for pocket in pockets:
        if (
            pocket.source_n == source_n
            and pocket.component_id == component_id
        ):
            return pocket
    raise ValueError(f"pocket {source_n}:{component_id} was not extracted")


def evaluate(task: Task) -> dict[str, object]:
    first_offset = cs.placement_offset(
        task.first,
        task.side,
        "LL",
        task.first_phase,
    )
    second_offset = cs.placement_offset(
        task.second,
        task.side,
        "UR",
        task.second_phase,
    )
    first = cs.shift(task.first, first_offset)
    second = cs.shift(task.second, second_offset)
    base = {
        "task_id": task.task_id,
        "first_symmetry": task.first_symmetry,
        "second_symmetry": task.second_symmetry,
        "first_phase_xy": task.first_phase,
        "second_phase_xy": task.second_phase,
    }
    if not cs.separated(first, second):
        return {
            **base,
            "count": -1,
            "valid": False,
            "reason": "pocket_overlap",
        }

    fixed = np.vstack([first, second])
    supports = (
        np.vstack(
            [
                cs.shift(task.first_supports, first_offset),
                cs.shift(task.second_supports, second_offset),
            ]
        ).reshape(-1, 3)
        if task.include_supports
        else np.empty((0, 3), dtype=np.float64)
    )
    filled = cs.exact_fill(
        task.side,
        fixed,
        supports,
        task.reach,
        task.milp_limit,
        task.upright_offset,
    )
    return {
        **base,
        "count": int(len(filled)),
        "valid": True,
        "tilted_count": int(len(fixed)),
        "upright_count": int(len(filled) - len(fixed)),
        "upright_offset": task.upright_offset,
        "squares": filled,
    }


def write_candidate(path: Path, target: int, squares: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["n", "square_id", "x", "y", "angle"])
        for square_id, (x, y, angle) in enumerate(squares[:target]):
            writer.writerow(
                [
                    target,
                    square_id,
                    repr(float(x)),
                    repr(float(y)),
                    repr(float(angle)),
                ]
            )


def phase_values(step: float) -> tuple[float, ...]:
    if not 0.0 < step <= 1.0:
        raise ValueError("--phase-step must be in (0, 1]")
    count = int(round(1.0 / step))
    if not math.isclose(count * step, 1.0, abs_tol=1e-12):
        raise ValueError("--phase-step must divide 1 exactly")
    return tuple(float(index * step) for index in range(count))


def parse_upright_offsets(
    value: str,
) -> tuple[tuple[float, float] | None, ...]:
    result: list[tuple[float, float] | None] = []
    for item in value.split(";"):
        item = item.strip()
        if item == "corner":
            result.append(None)
            continue
        parts = tuple(float(part) for part in item.split(","))
        if len(parts) != 2:
            raise ValueError("--upright-offsets entries must be corner or X,Y")
        result.append((parts[0], parts[1]))
    if not result:
        raise ValueError("--upright-offsets cannot be empty")
    return tuple(result)


def phase_cases(args: argparse.Namespace) -> tuple[
    tuple[
        int | None,
        int | None,
        tuple[float, float],
        tuple[float, float],
    ],
    ...,
]:
    phase_grid = tuple(
        (x, y)
        for x in phase_values(args.phase_step)
        for y in phase_values(args.phase_step)
    )
    if args.refine_summary is None:
        return tuple(
            (None, None, first_phase, second_phase)
            for first_phase in phase_grid
            for second_phase in phase_grid
        )

    if args.top_records < 1:
        raise ValueError("--top-records must be positive")
    report = json.loads(args.refine_summary.read_text(encoding="utf-8"))
    records = sorted(
        (
            record
            for record in report["records"]
            if int(record["count"]) >= 0
        ),
        key=lambda record: int(record["count"]),
        reverse=True,
    )
    cases = []
    seen = set()
    for record in records:
        key = (
            int(record["first_symmetry"]),
            int(record["second_symmetry"]),
            tuple(float(value) for value in record["first_phase_xy"]),
            tuple(float(value) for value in record["second_phase_xy"]),
        )
        if key in seen:
            continue
        seen.add(key)
        cases.append((key[0], key[1], key[2], key[3]))
        if len(cases) == args.top_records:
            break
    return tuple(cases)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline",
        type=Path,
        default=(
            ROOT
            / "poc"
            / "pell_corrugation"
            / "data"
            / "border-scan"
            / "merged.csv"
        ),
    )
    parser.add_argument("--target", type=int, required=True)
    parser.add_argument("--first-source", type=int, required=True)
    parser.add_argument("--first-component", type=int, required=True)
    parser.add_argument("--second-source", type=int, required=True)
    parser.add_argument("--second-component", type=int, required=True)
    parser.add_argument("--phase-step", type=float, default=0.5)
    parser.add_argument("--margin", default="1e-9")
    parser.add_argument("--min-pocket-count", type=int, default=80)
    parser.add_argument("--gap-tolerance", type=float, default=0.1)
    parser.add_argument("--support-margin", type=float, default=1.5)
    parser.add_argument("--reach", type=int, default=8)
    parser.add_argument("--milp-limit", type=float, default=5.0)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--support-candidates", action="store_true")
    parser.add_argument(
        "--upright-offsets",
        default="corner",
        help="semicolon-separated corner or X,Y upright lattice offsets",
    )
    parser.add_argument(
        "--refine-summary",
        type=Path,
        help="re-evaluate only geometry/phase cases from an earlier summary",
    )
    parser.add_argument("--top-records", type=int, default=64)
    parser.add_argument("--keep-records", type=int, default=128)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--best-out", type=Path, required=True)
    parser.add_argument("--candidate-out", type=Path)
    args = parser.parse_args()
    if args.jobs < 1:
        raise ValueError("--jobs must be positive")
    if args.keep_records < 1:
        raise ValueError("--keep-records must be positive")

    started = time.time()
    packings = mondrian.read_groups(args.baseline)
    incumbent = Decimal(mondrian.core.certify(args.target, packings[args.target]))
    side = float(incumbent - Decimal(args.margin))
    sources = tuple(dict.fromkeys((args.first_source, args.second_source)))
    pockets = cs.extract_pockets(
        packings,
        sources,
        args.min_pocket_count,
        args.gap_tolerance,
        args.support_margin,
    )
    first_pocket = locate_pocket(
        pockets,
        args.first_source,
        args.first_component,
    )
    second_pocket = locate_pocket(
        pockets,
        args.second_source,
        args.second_component,
    )

    cases = phase_cases(args)
    upright_offsets = parse_upright_offsets(args.upright_offsets)
    task_geometry = {}
    for first_symmetry, first_symmetry_value in enumerate(cs.SYMMETRIES):
        first, first_supports = cs.transform_bundle(
            first_pocket.squares,
            first_pocket.supports,
            first_symmetry_value,
        )
        for second_symmetry, second_symmetry_value in enumerate(cs.SYMMETRIES):
            second, second_supports = cs.transform_bundle(
                second_pocket.squares,
                second_pocket.supports,
                second_symmetry_value,
            )
            task_geometry[(first_symmetry, second_symmetry)] = (
                first,
                second,
                first_supports,
                second_supports,
            )

    tasks = []
    for (
        selected_first_symmetry,
        selected_second_symmetry,
        first_phase,
        second_phase,
    ) in cases:
        symmetry_pairs = (
            tuple(task_geometry)
            if selected_first_symmetry is None
            else (
                (
                    int(selected_first_symmetry),
                    int(selected_second_symmetry),
                ),
            )
        )
        for first_symmetry, second_symmetry in symmetry_pairs:
            first, second, first_supports, second_supports = task_geometry[
                (first_symmetry, second_symmetry)
            ]
            for upright_offset in upright_offsets:
                tasks.append(
                    Task(
                        task_id=len(tasks),
                        first_symmetry=first_symmetry,
                        second_symmetry=second_symmetry,
                        first_phase=first_phase,
                        second_phase=second_phase,
                        first=first,
                        second=second,
                        first_supports=first_supports,
                        second_supports=second_supports,
                        side=side,
                        reach=args.reach,
                        milp_limit=args.milp_limit,
                        upright_offset=upright_offset,
                        include_supports=args.support_candidates,
                    )
                )

    print(
        f"target={args.target} side={side:.12f} "
        f"pockets={args.first_source}:{args.first_component}"
        f"x{args.second_source}:{args.second_component} "
        f"phase_cases={len(cases)} upright_offsets={len(upright_offsets)} "
        f"tasks={len(tasks)}",
        flush=True,
    )
    records: list[dict[str, object]] = []
    best: dict[str, object] | None = None
    with mp.get_context("fork").Pool(args.jobs) as pool:
        for completed, result in enumerate(
            pool.imap_unordered(evaluate, tasks, chunksize=1),
            start=1,
        ):
            records.append(
                {key: value for key, value in result.items() if key != "squares"}
            )
            if int(result["count"]) >= 0 and (
                best is None or int(result["count"]) > int(best["count"])
            ):
                best = result
                print(
                    f"best={result['count']}/{args.target} "
                    f"task={result['task_id']}",
                    flush=True,
                )
            if completed % 64 == 0 or completed == len(tasks):
                print(
                    f"progress={completed}/{len(tasks)} "
                    f"best={None if best is None else best['count']}",
                    flush=True,
                )

    if best is None:
        raise RuntimeError("no feasible task was evaluated")
    squares = np.asarray(best["squares"], dtype=np.float64)
    write_candidate(args.best_out, args.target, squares)

    hits = [record for record in records if int(record["count"]) >= args.target]
    if hits and args.candidate_out:
        write_candidate(
            args.candidate_out,
            args.target,
            np.asarray(best["squares"], dtype=np.float64),
        )

    sorted_records = sorted(
        records,
        key=lambda record: int(record["count"]),
        reverse=True,
    )
    summary = {
        "baseline": str(args.baseline),
        "target": args.target,
        "incumbent_L": str(incumbent),
        "side": side,
        "margin": args.margin,
        "first_pocket": {
            "source_n": args.first_source,
            "component_id": args.first_component,
            "tilted_count": len(first_pocket.squares),
            "support_count": len(first_pocket.supports),
        },
        "second_pocket": {
            "source_n": args.second_source,
            "component_id": args.second_component,
            "tilted_count": len(second_pocket.squares),
            "support_count": len(second_pocket.supports),
        },
        "reach": args.reach,
        "support_candidates": args.support_candidates,
        "phase_step": args.phase_step,
        "phase_cases": len(cases),
        "refine_summary": (
            None if args.refine_summary is None else str(args.refine_summary)
        ),
        "upright_offsets": [
            None if offset is None else list(offset)
            for offset in upright_offsets
        ],
        "task_count": len(tasks),
        "elapsed_seconds": time.time() - started,
        "best": {key: value for key, value in best.items() if key != "squares"},
        "hits": len(hits),
        "stored_records": min(len(records), args.keep_records),
        "records_truncated": len(records) > args.keep_records,
        "records": sorted_records[: args.keep_records],
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(
            summary,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"best={best['count']}/{args.target} hits={len(hits)} "
        f"elapsed={time.time() - started:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
