#!/usr/bin/env python3
"""Place compact tilted motifs around an exact core, then refill uprights."""

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
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "poc" / "highn-mondrian" / "poc"))
import chirality_scan as cs  # noqa: E402
import motif_gap_scan as mgs  # noqa: E402
import scan  # noqa: E402


CORE = np.empty((0, 3), dtype=np.float64)
FULL = np.empty((0, 3), dtype=np.float64)
FULL_TILTED = np.empty(0, dtype=bool)
SIDE = 0.0
REACH = 8
MILP_LIMIT = 3.0
REMOVE_CONFLICTS = False


@dataclass(frozen=True)
class Task:
    motif: int
    placement: np.ndarray
    clearance: float
    translation: tuple[float, float]


def read_core(
    packings: dict[int, list[dict[str, str]]],
    source: int,
    mode: str,
) -> np.ndarray:
    squares = np.asarray(
        [
            (float(row["x"]), float(row["y"]), float(row["angle"]))
            for row in packings[source]
        ],
        dtype=np.float64,
    )
    if mode == "tilted":
        return squares[cs.tilted(squares, tolerance_deg=2.0)]
    return squares


def placement_tasks(
    core: np.ndarray,
    motifs: list[np.ndarray],
    side: float,
    step: float,
    per_motif: int,
    minimum_separation: float,
) -> list[Task]:
    fixed_tree = cKDTree(core[:, :2])
    tasks = []
    for motif_index, motif in enumerate(motifs):
        xlo, xhi, ylo, yhi = cs.polygon_bounds(motif)
        width = float(xhi.max() - xlo.min())
        height = float(yhi.max() - ylo.min())
        tx = np.arange(0.0, side - width + 1e-12, step)
        ty = np.arange(0.0, side - height + 1e-12, step)
        occupied = mgs.forbidden_translation(core, motif, tx, ty, step)
        free_y, free_x = np.nonzero(~occupied)
        if not len(free_x):
            continue
        px = tx[free_x] + 0.5 * width
        py = ty[free_y] + 0.5 * height
        clearance, _ = fixed_tree.query(np.column_stack([px, py]))
        order = np.argsort(clearance)[::-1]
        selected = []
        for candidate in order:
            point = np.array([px[candidate], py[candidate]])
            if any(
                np.linalg.norm(point - existing[0]) < minimum_separation
                for existing in selected
            ):
                continue
            selected.append(
                (
                    point,
                    int(candidate),
                    float(clearance[candidate]),
                )
            )
            if len(selected) >= per_motif:
                break
        for _, candidate, clearance_value in selected:
            placed = motif.copy()
            placed[:, 0] += tx[free_x[candidate]]
            placed[:, 1] += ty[free_y[candidate]]
            if not cs.separated(placed, core):
                continue
            tasks.append(
                Task(
                    motif=motif_index,
                    placement=placed,
                    clearance=clearance_value,
                    translation=(
                        float(tx[free_x[candidate]]),
                        float(ty[free_y[candidate]]),
                    ),
                )
            )
    return tasks


def init_worker(
    core: np.ndarray,
    full: np.ndarray,
    full_tilted: np.ndarray,
    side: float,
    reach: int,
    limit: float,
    remove_conflicts: bool,
) -> None:
    global CORE, FULL, FULL_TILTED, SIDE, REACH, MILP_LIMIT, REMOVE_CONFLICTS
    CORE = core
    FULL = full
    FULL_TILTED = full_tilted
    SIDE = side
    REACH = reach
    MILP_LIMIT = limit
    REMOVE_CONFLICTS = remove_conflicts


def evaluate(task: Task) -> dict[str, object]:
    removed = 0
    if REMOVE_CONFLICTS:
        keep = np.ones(len(FULL), dtype=bool)
        for square in task.placement:
            for index, existing in enumerate(FULL):
                if keep[index] and not cs.separated(
                    square.reshape(1, 3),
                    existing.reshape(1, 3),
                ):
                    if FULL_TILTED[index]:
                        return {
                            "motif": task.motif,
                            "translation": task.translation,
                            "clearance": task.clearance,
                            "motif_squares": task.placement.tolist(),
                            "count": -1,
                            "removed": 0,
                            "squares": [],
                        }
                    keep[index] = False
                    removed += 1
        fixed = np.vstack([FULL[keep], task.placement])
    else:
        fixed = np.vstack([CORE, task.placement])
    filled = cs.exact_fill(
        SIDE,
        fixed,
        np.empty((0, 3), dtype=np.float64),
        REACH,
        MILP_LIMIT,
    )
    return {
        "motif": task.motif,
        "translation": task.translation,
        "clearance": task.clearance,
        "motif_squares": task.placement.tolist(),
        "removed": removed,
        "count": int(len(filled)),
        "squares": filled.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--core-source", type=int, default=908)
    parser.add_argument("--core-mode", choices=("full", "tilted"), default="tilted")
    parser.add_argument("--seed-csv", type=Path)
    parser.add_argument("--remove-conflicts", action="store_true")
    parser.add_argument("--side", required=True, type=float)
    parser.add_argument("--target", required=True, type=int)
    parser.add_argument("--sources", default="908,969,970,971,972")
    parser.add_argument("--max-box-area", type=float, default=3.0)
    parser.add_argument("--grid-step", type=float, default=0.25)
    parser.add_argument("--placements-per-motif", type=int, default=2)
    parser.add_argument("--minimum-separation", type=float, default=2.5)
    parser.add_argument("--motif-limit", type=int)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--reach", type=int, default=8)
    parser.add_argument("--milp-limit", type=float, default=3.0)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--candidate-out", required=True, type=Path)
    args = parser.parse_args()

    started = time.time()
    packings = scan.read_groups(args.baseline)
    core = read_core(packings, args.core_source, args.core_mode)
    full = (
        mgs.read_squares(args.seed_csv)
        if args.seed_csv is not None
        else core
    )
    full_tilted = cs.tilted(full, tolerance_deg=2.0)
    if args.remove_conflicts and args.seed_csv is None:
        raise SystemExit("--remove-conflicts requires --seed-csv")
    sources = tuple(int(value) for value in args.sources.split(","))
    motifs = mgs.pair_motifs(packings, sources, args.max_box_area)
    if args.motif_limit is not None:
        motifs = motifs[: args.motif_limit]
    tasks = placement_tasks(
        core,
        motifs,
        args.side,
        args.grid_step,
        args.placements_per_motif,
        args.minimum_separation,
    )
    print(
        f"core={len(core)} motifs={len(motifs)} tasks={len(tasks)}",
        flush=True,
    )
    records = []
    best = None
    context = mp.get_context("fork")
    with context.Pool(
        args.jobs,
        initializer=init_worker,
        initargs=(
            core,
            full,
            full_tilted,
            args.side,
            args.reach,
            args.milp_limit,
            args.remove_conflicts,
        ),
    ) as pool:
        for completed, record in enumerate(
            pool.imap_unordered(evaluate, tasks),
            start=1,
        ):
            records.append(
                {
                    key: record[key]
                    for key in (
                        "motif",
                        "translation",
                        "clearance",
                        "motif_squares",
                        "removed",
                        "count",
                    )
                }
            )
            if int(record["count"]) >= 0 and (
                best is None or int(record["count"]) > int(best["count"])
            ):
                best = record
                print(
                    f"best={record['count']}/{args.target} "
                    f"motif={record['motif']}",
                    flush=True,
                )
            if completed % 25 == 0:
                print(
                    f"progress={completed}/{len(tasks)} "
                    f"best={best['count'] if best is not None else None}",
                    flush=True,
                )
    report: dict[str, object] = {
        "baseline": str(args.baseline.resolve()),
        "core_source": args.core_source,
        "core_mode": args.core_mode,
        "seed_csv": None if args.seed_csv is None else str(args.seed_csv.resolve()),
        "remove_conflicts": args.remove_conflicts,
        "core_count": len(core),
        "side": args.side,
        "target": args.target,
        "sources": sources,
        "max_box_area": args.max_box_area,
        "grid_step": args.grid_step,
        "placements_per_motif": args.placements_per_motif,
        "minimum_separation": args.minimum_separation,
        "motif_limit": args.motif_limit,
        "jobs": args.jobs,
        "reach": args.reach,
        "milp_limit": args.milp_limit,
        "motif_count": len(motifs),
        "task_count": len(tasks),
        "elapsed_seconds": time.time() - started,
        "best": None,
        "records": sorted(records, key=lambda item: int(item["count"]), reverse=True),
    }
    if best is not None:
        squares = np.asarray(best["squares"], dtype=np.float64)
        report["best"] = {
            key: value
            for key, value in best.items()
            if key != "squares"
        }
        mgs.write_candidate(args.candidate_out, args.target, squares)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"tasks={len(tasks)} best={None if best is None else best['count']} "
        f"target={args.target} elapsed={time.time() - started:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
