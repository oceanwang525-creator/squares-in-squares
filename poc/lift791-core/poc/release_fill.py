#!/usr/bin/env python3
"""Release core squares and exactly refill the upright region."""

from __future__ import annotations

import argparse
import json
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


POC_DIR = Path(__file__).resolve().parents[1]
ROOT = POC_DIR.parents[1]
sys.path.insert(0, str(ROOT / "poc" / "corner-pocket-chirality" / "poc"))
sys.path.insert(0, str(ROOT / "poc" / "highn-mondrian" / "poc"))
import chirality_scan as cs  # noqa: E402
import motif_gap_scan as mgs  # noqa: E402
import scan  # noqa: E402


SIDE = 0.0
TARGET = 0
REACH = 8
MILP_LIMIT = 1.0


@dataclass(frozen=True)
class Task:
    removed: tuple[int, ...]
    fixed: np.ndarray


def init_worker(
    side: float,
    target: int,
    reach: int,
    milp_limit: float,
) -> None:
    global SIDE, TARGET, REACH, MILP_LIMIT
    SIDE = side
    TARGET = target
    REACH = reach
    MILP_LIMIT = milp_limit


def evaluate(task: Task) -> dict[str, object]:
    started = time.time()
    filled = cs.exact_fill(
        SIDE,
        task.fixed,
        np.empty((0, 3), dtype=np.float64),
        REACH,
        MILP_LIMIT,
    )
    return {
        "removed": list(task.removed),
        "count": int(len(filled)),
        "elapsed_seconds": time.time() - started,
        "squares": filled.tolist(),
    }


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
    parser.add_argument("--source-n", type=int, default=791)
    parser.add_argument("--side", type=float, required=True)
    parser.add_argument("--target", type=int, required=True)
    parser.add_argument("--remove-count", type=int, default=1)
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--reach", type=int, default=8)
    parser.add_argument("--milp-limit", type=float, default=1.0)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--candidate-out", type=Path, required=True)
    args = parser.parse_args()

    if args.remove_count not in (1, 2):
        raise SystemExit("--remove-count must be 1 or 2")
    started = time.time()
    packings = scan.read_groups(args.baseline)
    core = np.asarray(
        [
            (float(row["x"]), float(row["y"]), float(row["angle"]))
            for row in packings[args.source_n]
        ],
        dtype=np.float64,
    )
    removed_sets = [
        (first,)
        for first in range(len(core))
    ]
    if args.remove_count == 2:
        removed_sets = [
            (first, second)
            for first in range(len(core))
            for second in range(first + 1, len(core))
        ]
    if args.max_tasks is not None:
        removed_sets = removed_sets[: args.max_tasks]
    tasks = [
        Task(
            removed=removed,
            fixed=np.delete(core, removed, axis=0),
        )
        for removed in removed_sets
    ]
    print(
        f"core={len(core)} remove={args.remove_count} "
        f"tasks={len(tasks)}",
        flush=True,
    )

    best = None
    records = []
    context = mp.get_context("fork")
    with context.Pool(
        args.jobs,
        initializer=init_worker,
        initargs=(
            args.side,
            args.target,
            args.reach,
            args.milp_limit,
        ),
    ) as pool:
        for completed, record in enumerate(
            pool.imap_unordered(evaluate, tasks),
            start=1,
        ):
            records.append(
                {
                    key: record[key]
                    for key in ("removed", "count", "elapsed_seconds")
                }
            )
            if best is None or int(record["count"]) > int(best["count"]):
                best = record
                print(
                    f"best={record['count']}/{args.target} "
                    f"removed={record['removed']}",
                    flush=True,
                )
            if completed % 50 == 0:
                print(
                    f"progress={completed}/{len(tasks)} best={best['count']}",
                    flush=True,
                )
            if int(best["count"]) >= args.target:
                pool.terminate()
                break

    report = {
        "baseline": str(args.baseline.resolve()),
        "source_n": args.source_n,
        "source_count": len(core),
        "side": args.side,
        "target": args.target,
        "remove_count": args.remove_count,
        "submitted_tasks": len(tasks),
        "jobs": args.jobs,
        "reach": args.reach,
        "milp_limit": args.milp_limit,
        "elapsed_seconds": time.time() - started,
        "best": None if best is None else {
            key: value for key, value in best.items() if key != "squares"
        },
        "records": sorted(
            records,
            key=lambda item: int(item["count"]),
            reverse=True,
        ),
    }
    if best is not None:
        squares = np.asarray(best["squares"], dtype=np.float64)
        mgs.write_candidate(args.candidate_out, args.target, squares)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"tasks={len(records)} best={None if best is None else best['count']} "
        f"target={args.target} elapsed={time.time() - started:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
