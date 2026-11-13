#!/usr/bin/env python3
"""Exact local set-packing search around a certified high-n seed layout."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from collections import Counter
from dataclasses import dataclass
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
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree

import chirality_scan as cs


@dataclass(frozen=True)
class Graph:
    candidates: np.ndarray
    edges: np.ndarray
    indptr: np.ndarray
    indices: np.ndarray
    seed_indices: np.ndarray


def read_squares(path: Path) -> np.ndarray:
    with path.open(newline="", encoding="utf-8") as handle:
        return np.asarray(
            [
                (float(row["x"]), float(row["y"]), float(row["angle"]))
                for row in csv.DictReader(handle)
            ],
            dtype=np.float64,
        )


def angle_key(angle: float) -> float:
    folded = angle % (0.5 * math.pi)
    return round(float(folded), 10)


def candidate_grid(
    side: float,
    seed: np.ndarray,
    step: float,
    angle_step_deg: float,
    common_angles: int,
) -> np.ndarray:
    angles = set(
        np.radians(np.arange(0.0, 90.0, angle_step_deg)).round(12).tolist()
    )
    counts = Counter(angle_key(float(angle)) for angle in seed[:, 2])
    angles.update(angle for angle, _ in counts.most_common(common_angles))
    generated = []
    for angle in sorted(angles):
        extent = 0.5 * (abs(math.cos(angle)) + abs(math.sin(angle)))
        values = np.arange(extent, side - extent + 1e-12, step)
        grid_x, grid_y = np.meshgrid(values, values, indexing="ij")
        generated.append(
            np.column_stack(
                [
                    grid_x.ravel(),
                    grid_y.ravel(),
                    np.full(grid_x.size, angle, dtype=np.float64),
                ]
            )
        )
    generated.append(np.asarray(seed, dtype=np.float64).reshape(-1, 3))
    return np.unique(np.round(np.vstack(generated), 10), axis=0)


def conflict_edges(candidates: np.ndarray) -> np.ndarray:
    tree = cKDTree(candidates[:, :2])
    pairs = tree.query_pairs(1.45, output_type="ndarray")
    if not len(pairs):
        return pairs.reshape(0, 2)
    first, second = pairs.T
    dx = candidates[second, 0] - candidates[first, 0]
    dy = candidates[second, 1] - candidates[first, 1]
    first_angle = candidates[first, 2]
    second_angle = candidates[second, 2]
    delta = second_angle - first_angle
    radius = 0.5 * (
        1.0 + np.abs(np.cos(delta)) + np.abs(np.sin(delta))
    )
    projection = np.abs(
        np.cos(first_angle) * dx + np.sin(first_angle) * dy
    )
    projection = np.maximum(
        projection,
        np.abs(-np.sin(first_angle) * dx + np.cos(first_angle) * dy),
    )
    projection = np.maximum(
        projection,
        np.abs(np.cos(second_angle) * dx + np.sin(second_angle) * dy),
    )
    projection = np.maximum(
        projection,
        np.abs(-np.sin(second_angle) * dx + np.cos(second_angle) * dy),
    )
    return pairs[projection < radius - 1e-9].astype(np.uint32)


def compressed_adjacency(n: int, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not len(edges):
        return np.zeros(n + 1, dtype=np.int64), np.zeros(0, dtype=np.uint32)
    source = np.concatenate([edges[:, 0], edges[:, 1]])
    target = np.concatenate([edges[:, 1], edges[:, 0]])
    order = np.argsort(source, kind="stable")
    source = source[order]
    target = target[order]
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.add.at(indptr, source + 1, 1)
    indptr = np.cumsum(indptr)
    return indptr, target.astype(np.uint32)


def build_graph(
    side: float,
    seed: np.ndarray,
    step: float,
    angle_step_deg: float,
    common_angles: int,
) -> Graph:
    candidates = candidate_grid(
        side, seed, step, angle_step_deg, common_angles
    )
    keys = {
        tuple(row): index
        for index, row in enumerate(np.round(candidates, 10))
    }
    seed_indices = np.asarray(
        [keys[tuple(row)] for row in np.round(seed, 10)], dtype=np.int64
    )
    edges = conflict_edges(candidates)
    indptr, indices = compressed_adjacency(len(candidates), edges)
    return Graph(candidates, edges, indptr, indices, seed_indices)


def solve_window(
    graph: Graph,
    selected: np.ndarray,
    window: np.ndarray,
    time_limit: float,
    presolve: bool,
) -> np.ndarray | None:
    selected_in_window = np.flatnonzero(selected[window])
    fixed = np.zeros(len(selected), dtype=bool)
    fixed[selected] = True
    fixed[window] = False

    keep = np.ones(len(window), dtype=bool)
    for local, candidate in enumerate(window):
        for offset in range(
            graph.indptr[candidate], graph.indptr[candidate + 1]
        ):
            other = graph.indices[offset]
            if fixed[other]:
                keep[local] = False
                break
    reduced = window[keep]
    if not len(reduced):
        return None
    local_index = np.full(len(selected), -1, dtype=np.int64)
    local_index[reduced] = np.arange(len(reduced), dtype=np.int64)
    local_edges = []
    for local, candidate in enumerate(reduced):
        neighbours = graph.indices[
            graph.indptr[candidate] : graph.indptr[candidate + 1]
        ]
        other = local_index[neighbours]
        valid = other > local
        if valid.any():
            local_edges.append(
                np.column_stack(
                    [
                        np.full(np.count_nonzero(valid), local),
                        other[valid],
                    ]
                )
            )
    reduced_edges = (
        np.vstack(local_edges).astype(np.int64)
        if local_edges
        else np.empty((0, 2), dtype=np.int64)
    )
    if len(reduced_edges):
        rows = np.repeat(np.arange(len(reduced_edges)), 2)
        columns = np.column_stack(
            [
                reduced_edges[:, 0],
                reduced_edges[:, 1],
            ]
        )
        matrix = coo_matrix(
            (
                np.ones(2 * len(reduced_edges)),
                (rows, columns.ravel()),
            ),
            shape=(len(reduced_edges), len(reduced)),
        ).tocsr()
        constraints: object = LinearConstraint(matrix, -np.inf, 1.0)
    else:
        constraints = ()
    result = milp(
        -np.ones(len(reduced)),
        constraints=constraints,
        integrality=np.ones(len(reduced)),
        bounds=Bounds(0, 1),
        options={"time_limit": time_limit, "presolve": presolve},
    )
    if result.x is None:
        return None
    return reduced[result.x > 0.5]


def write_candidate(path: Path, target: int, squares: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["n", "square_id", "x", "y", "angle"])
        for index, (x, y, angle) in enumerate(squares[:target]):
            writer.writerow(
                [
                    target,
                    index,
                    repr(float(x)),
                    repr(float(y)),
                    repr(float(angle)),
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-csv", required=True, type=Path)
    parser.add_argument("--side", required=True, type=float)
    parser.add_argument("--target", required=True, type=int)
    parser.add_argument("--grid-step", type=float, default=0.5)
    parser.add_argument("--angle-step", type=float, default=5.0)
    parser.add_argument("--common-angles", type=int, default=10)
    parser.add_argument("--window-size", type=float, default=5.0)
    parser.add_argument("--window-count", type=int, default=300)
    parser.add_argument("--time-limit", type=float, default=5.0)
    parser.add_argument(
        "--presolve",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--rng-seed", type=int, default=20261112)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--best-out", type=Path)
    parser.add_argument("--candidate-out", type=Path)
    args = parser.parse_args()

    seed_squares = read_squares(args.seed_csv)
    if len(seed_squares) >= args.target:
        raise ValueError("seed already reaches target")
    started = time.time()
    graph = build_graph(
        args.side,
        seed_squares,
        args.grid_step,
        args.angle_step,
        args.common_angles,
    )
    selected = np.zeros(len(graph.candidates), dtype=bool)
    selected[graph.seed_indices] = True
    best_count = int(selected.sum())
    best_selected = selected.copy()
    rng = np.random.default_rng(args.rng_seed)
    records = []
    print(
        f"candidates={len(graph.candidates)} edges={len(graph.edges)} "
        f"seed={best_count}",
        flush=True,
    )

    for iteration in range(args.window_count):
        if iteration < 4:
            centers = (
                (0.0, 0.0),
                (args.side, 0.0),
                (0.0, args.side),
                (args.side, args.side),
            )
            x0, y0 = centers[iteration]
            x0 = min(max(0.0, x0 - 0.5 * args.window_size), args.side - args.window_size)
            y0 = min(max(0.0, y0 - 0.5 * args.window_size), args.side - args.window_size)
        else:
            x0 = rng.uniform(0.0, args.side - args.window_size)
            y0 = rng.uniform(0.0, args.side - args.window_size)
        window = np.flatnonzero(
            (graph.candidates[:, 0] >= x0)
            & (graph.candidates[:, 0] <= x0 + args.window_size)
            & (graph.candidates[:, 1] >= y0)
            & (graph.candidates[:, 1] <= y0 + args.window_size)
        )
        old_count = int(selected[window].sum())
        chosen = solve_window(
            graph,
            selected,
            window,
            args.time_limit,
            args.presolve,
        )
        if chosen is None:
            if iteration % 10 == 0:
                print(
                    f"progress={iteration + 1}/{args.window_count} "
                    f"best={best_count} no-solution",
                    flush=True,
                )
            continue
        new_count = len(chosen)
        accept = new_count > old_count or (
            new_count == old_count and rng.random() < 0.08
        )
        if not accept:
            continue
        selected[window] = False
        selected[chosen] = True
        count = int(selected.sum())
        if count >= best_count:
            best_count = count
            best_selected = selected.copy()
        record = {
            "iteration": iteration,
            "window": [x0, y0, x0 + args.window_size, y0 + args.window_size],
            "old": old_count,
            "new": new_count,
            "total": count,
            "best": best_count,
        }
        records.append(record)
        if new_count > old_count or count >= best_count:
            print(
                f"iteration={iteration} window=({x0:.3f},{y0:.3f}) "
                f"old={old_count} new={new_count} total={count} "
                f"best={best_count}",
                flush=True,
            )
        if best_count >= args.target:
            break
        if iteration % 10 == 0:
            print(
                f"progress={iteration + 1}/{args.window_count} "
                f"best={best_count}",
                flush=True,
            )

    best_squares = graph.candidates[best_selected]
    if args.best_out:
        write_candidate(args.best_out, len(best_squares), best_squares)
    report = {
        "seed": str(args.seed_csv.resolve()),
        "seed_count": len(seed_squares),
        "side": args.side,
        "target": args.target,
        "candidates": len(graph.candidates),
        "edges": len(graph.edges),
        "grid_step": args.grid_step,
        "angle_step_deg": args.angle_step,
        "common_angles": args.common_angles,
        "window_size": args.window_size,
        "window_count": args.window_count,
        "presolve": args.presolve,
        "best_count": best_count,
        "elapsed_seconds": time.time() - started,
        "records": records,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if best_count >= args.target and args.candidate_out:
        write_candidate(args.candidate_out, args.target, best_squares)
    if len(best_squares) > args.target:
        best_squares = best_squares[: args.target]
    print(
        f"seed={len(seed_squares)} best={best_count} target={args.target} "
        f"elapsed={time.time() - started:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
