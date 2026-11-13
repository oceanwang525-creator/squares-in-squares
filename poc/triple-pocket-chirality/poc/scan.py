#!/usr/bin/env python3
"""Search three-corner placements with one full and two split pockets."""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import multiprocessing as mp
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
SPLIT_POC = ROOT / "poc" / "split-pocket-chirality" / "poc"
sys.path.insert(0, str(SPLIT_POC))
sys.path.insert(0, str(ROOT / "poc" / "corner-pocket-chirality" / "poc"))
sys.path.insert(0, str(ROOT / "poc" / "highn-mondrian" / "poc"))

import chirality_scan as cs  # noqa: E402
import scan  # noqa: E402

_split_spec = importlib.util.spec_from_file_location(
    "split_scan",
    SPLIT_POC / "scan.py",
)
if _split_spec is None or _split_spec.loader is None:
    raise ImportError("cannot load split-pocket scan helpers")
split_scan = importlib.util.module_from_spec(_split_spec)
sys.modules["split_scan"] = split_scan
_split_spec.loader.exec_module(split_scan)


ANCHORS = split_scan.ANCHORS


@dataclass(frozen=True)
class Task:
    task_id: int
    side: float
    options: tuple[object, object, object]
    anchors: tuple[str, str, str]


def separated(options: tuple[object, ...]) -> bool:
    for first, second in itertools.combinations(range(len(options)), 2):
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
        "count": len(filled),
        "anchors": task.anchors,
        "options": [
            {
                "pocket": option.pocket,
                "symmetry": option.symmetry,
                "phase": option.phase,
            }
            for option in task.options
        ],
        "squares": filled.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--source", type=int, required=True)
    parser.add_argument("--target", type=int, default=971)
    parser.add_argument("--side", type=float, required=True)
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
    if len(pockets) < 2:
        raise SystemExit("source must contain at least two pockets")
    full = pockets[0].squares.copy()
    first_half, second_half = split_scan.split_component(pockets[1].squares)
    roster = (full, first_half, second_half)
    phases = np.asarray(
        [
            (x, y)
            for x in np.arange(0.0, 1.0, args.phase_step)
            for y in np.arange(0.0, 1.0, args.phase_step)
        ],
        dtype=np.float64,
    )
    options_by_anchor = split_scan.placement_options(roster, args.side, phases)
    print(
        f"roster={','.join(str(len(squares)) for squares in roster)} "
        + " ".join(
            f"{anchor}:{len(options_by_anchor[anchor])}" for anchor in ANCHORS
        ),
        flush=True,
    )

    rng = np.random.default_rng(args.seed)
    tasks = []
    seen = set()
    attempts = 0
    while len(tasks) < args.tasks and attempts < 1000000:
        attempts += 1
        anchors = tuple(
            ANCHORS[index]
            for index in rng.choice(len(ANCHORS), size=3, replace=False)
        )
        selected = tuple(
            options_by_anchor[anchor][
                int(rng.integers(len(options_by_anchor[anchor])))
            ]
            for anchor in anchors
        )
        key = (
            anchors,
            tuple(
                (option.pocket, option.symmetry, option.phase)
                for option in selected
            ),
        )
        if key in seen:
            continue
        seen.add(key)
        if not separated(selected):
            continue
        tasks.append(
            Task(
                task_id=len(tasks),
                side=args.side,
                options=selected,
                anchors=anchors,
            )
        )
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
                    f"progress={completed}/{len(tasks)} best={best['count']}",
                    flush=True,
                )

    report = {
        "source": args.source,
        "target": args.target,
        "side": args.side,
        "phase_step": args.phase_step,
        "roster": [len(squares) for squares in roster],
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
        split_scan.write_candidate(
            args.candidate_out,
            args.target,
            squares,
        )
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
