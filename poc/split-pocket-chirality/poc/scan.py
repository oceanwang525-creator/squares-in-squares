#!/usr/bin/env python3
"""Search four-corner placements of split n=908 tilted pockets."""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

for variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[variable] = "1"

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
CORNER_POC = ROOT / "poc" / "corner-pocket-chirality" / "poc"
sys.path.insert(0, str(CORNER_POC))
sys.path.insert(0, str(ROOT / "poc" / "highn-mondrian" / "poc"))

import chirality_scan as cs  # noqa: E402
import scan  # noqa: E402


ANCHORS = ("LL", "LR", "UL", "UR")


@dataclass(frozen=True)
class Option:
    pocket: int
    anchor: str
    symmetry: int
    phase: int
    squares: np.ndarray


@dataclass(frozen=True)
class Task:
    task_id: int
    side: float
    options: tuple[Option, Option, Option, Option]


def split_component(squares: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centred = squares[:, :2] - squares[:, :2].mean(axis=0)
    _, _, transpose = np.linalg.svd(centred, full_matrices=False)
    projection = centred @ transpose[0]
    order = np.argsort(projection, kind="stable")
    cut = len(order) // 2
    first = squares[order[:cut]].copy()
    second = squares[order[cut:]].copy()
    first, _ = cs.normalize_bundle(first, np.empty((0, 3), dtype=np.float64))
    second, _ = cs.normalize_bundle(second, np.empty((0, 3), dtype=np.float64))
    return first, second


def offset_for_anchor(
    squares: np.ndarray,
    side: float,
    anchor: str,
    phase: tuple[float, float],
) -> np.ndarray:
    xlo, xhi, ylo, yhi = cs.polygon_bounds(squares)
    px, py = phase
    if anchor == "LL":
        return np.array([px - xlo.min(), py - ylo.min()])
    if anchor == "LR":
        return np.array([side - px - xhi.max(), py - ylo.min()])
    if anchor == "UL":
        return np.array([px - xlo.min(), side - py - yhi.max()])
    if anchor == "UR":
        return np.array([side - px - xhi.max(), side - py - yhi.max()])
    raise ValueError(anchor)


def placement_options(
    pockets: tuple[np.ndarray, ...],
    side: float,
    phases: np.ndarray,
) -> dict[str, list[Option]]:
    result = {anchor: [] for anchor in ANCHORS}
    for pocket_index, pocket in enumerate(pockets):
        for symmetry_id, symmetry in enumerate(cs.SYMMETRIES):
            oriented, _ = cs.transform_bundle(
                pocket,
                np.empty((0, 3), dtype=np.float64),
                symmetry,
            )
            for phase_id, phase in enumerate(phases):
                for anchor in ANCHORS:
                    offset = offset_for_anchor(
                        oriented,
                        side,
                        anchor,
                        (float(phase[0]), float(phase[1])),
                    )
                    squares = cs.shift(oriented, offset)
                    xlo, xhi, ylo, yhi = cs.polygon_bounds(squares)
                    if (
                        xlo.min() < -1e-8
                        or ylo.min() < -1e-8
                        or xhi.max() > side + 1e-8
                        or yhi.max() > side + 1e-8
                    ):
                        continue
                    ordered = squares[
                        np.lexsort(
                            (squares[:, 2], squares[:, 1], squares[:, 0])
                        )
                    ]
                    key = np.round(ordered, 10).tobytes()
                    if any(
                        np.round(existing.squares, 10).tobytes() == key
                        for existing in result[anchor]
                    ):
                        continue
                    result[anchor].append(
                        Option(
                            pocket=pocket_index,
                            anchor=anchor,
                            symmetry=symmetry_id,
                            phase=phase_id,
                            squares=squares,
                        )
                    )
    return result


def pairwise_separated(options: tuple[Option, ...]) -> bool:
    for first in range(4):
        for second in range(first + 1, 4):
            if not cs.separated(
                options[first].squares,
                options[second].squares,
            ):
                return False
    return True


def evaluate(task: Task) -> dict[str, object]:
    fixed = np.vstack([option.squares for option in task.options])
    filled = cs.exact_fill(
        task.side,
        fixed,
        np.empty((0, 3), dtype=np.float64),
        reach=8,
        time_limit=2.0,
    )
    return {
        "task_id": task.task_id,
        "side": task.side,
        "count": int(len(filled)),
        "options": [
            {
                "anchor": option.anchor,
                "pocket": option.pocket,
                "symmetry": option.symmetry,
                "phase": option.phase,
            }
            for option in task.options
        ],
        "squares": filled.tolist(),
    }


def write_candidate(path: Path, target: int, squares: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        handle.write("n,square_id,x,y,angle\n")
        for index, (x, y, angle) in enumerate(squares[:target]):
            handle.write(
                f"{target},{index},{repr(float(x))},"
                f"{repr(float(y))},{repr(float(angle))}\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--source", type=int, default=908)
    parser.add_argument("--target", type=int, default=971)
    parser.add_argument("--side", required=True, type=float)
    parser.add_argument("--phase-step", type=float, default=0.5)
    parser.add_argument("--tasks", type=int, default=300)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261113)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--candidate-out", required=True, type=Path)
    args = parser.parse_args()

    started = time.time()
    packings = scan.read_groups(args.baseline)
    pockets = cs.extract_pockets(
        packings,
        (args.source,),
        min_count=80,
        gap_tolerance=0.1,
        support_margin=1.5,
    )
    components = [split_component(pocket.squares) for pocket in pockets]
    halves = tuple(component for pair in components for component in pair)
    phases = np.asarray(
        [
            (x, y)
            for x in np.arange(0.0, 1.0, args.phase_step)
            for y in np.arange(0.0, 1.0, args.phase_step)
        ],
        dtype=np.float64,
    )
    options = placement_options(halves, args.side, phases)
    print(
        "halves="
        + ",".join(str(len(half)) for half in halves)
        + " options="
        + ",".join(f"{anchor}:{len(options[anchor])}" for anchor in ANCHORS),
        flush=True,
    )

    rng = np.random.default_rng(args.seed)
    tasks = []
    seen = set()
    attempts = 0
    while len(tasks) < args.tasks and attempts < 1000000:
        attempts += 1
        selected = tuple(
            options[anchor][int(rng.integers(len(options[anchor])))]
            for anchor in ANCHORS
        )
        key = tuple(
            (option.pocket, option.symmetry, option.phase)
            for option in selected
        )
        if key in seen:
            continue
        seen.add(key)
        if not pairwise_separated(selected):
            continue
        tasks.append(Task(len(tasks), args.side, selected))
    print(
        f"attempts={attempts} separated_tasks={len(tasks)}",
        flush=True,
    )

    records = []
    best = None
    with mp.get_context("fork").Pool(args.jobs) as pool:
        for completed, record in enumerate(
            pool.imap_unordered(evaluate, tasks),
            start=1,
        ):
            records.append(
                {key: value for key, value in record.items() if key != "squares"}
            )
            if best is None or int(record["count"]) > int(best["count"]):
                best = record
                print(
                    f"best={record['count']}/{args.target} "
                    f"task={record['task_id']}",
                    flush=True,
                )
            if completed % 20 == 0:
                print(
                    f"progress={completed}/{len(tasks)} "
                    f"best={None if best is None else best['count']}",
                    flush=True,
                )

    report = {
        "source": args.source,
        "target": args.target,
        "side": args.side,
        "phase_step": args.phase_step,
        "halves": [len(half) for half in halves],
        "task_count": len(tasks),
        "attempts": attempts,
        "elapsed_seconds": time.time() - started,
        "best": None
        if best is None
        else {key: value for key, value in best.items() if key != "squares"},
        "records": sorted(
            records,
            key=lambda item: int(item["count"]),
            reverse=True,
        ),
    }
    if best is not None:
        squares = np.asarray(best["squares"], dtype=np.float64)
        write_candidate(args.candidate_out, args.target, squares)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"best={None if best is None else best['count']}/{args.target} "
        f"elapsed={time.time() - started:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
