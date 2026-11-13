#!/usr/bin/env python3
"""Fine-coordinate exact repacking in windows around measured residual gaps."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

import chirality_scan as cs
import fillfast


def read_squares(path: Path) -> np.ndarray:
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append((float(row["x"]), float(row["y"]), float(row["angle"])))
    return np.asarray(rows, dtype=np.float64)


def bounds(square: np.ndarray) -> tuple[float, float, float, float]:
    xlo, xhi, ylo, yhi = cs.polygon_bounds(square.reshape(1, 3))
    return float(xlo[0]), float(xhi[0]), float(ylo[0]), float(yhi[0])


def overlaps_window(
    square: np.ndarray,
    window: tuple[float, float, float, float],
) -> bool:
    xlo, xhi, ylo, yhi = bounds(square)
    return (
        xhi > window[0]
        and xlo < window[1]
        and yhi > window[2]
        and ylo < window[3]
    )


def filter_candidates(
    points: np.ndarray,
    side: float,
    fixed: np.ndarray,
) -> np.ndarray:
    candidates = np.column_stack(
        [points, np.zeros(len(points), dtype=np.float64)]
    )
    keep = np.ones(len(candidates), dtype=bool)
    for start in range(0, len(candidates), 512):
        batch = candidates[start : start + 512]
        xlo, xhi, ylo, yhi = cs.polygon_bounds(batch)
        keep[start : start + len(batch)] = (
            (xlo >= -1e-9)
            & (ylo >= -1e-9)
            & (xhi <= side + 1e-9)
            & (yhi <= side + 1e-9)
        )
        for fixed_start in range(0, len(fixed), 128):
            obstacle = fixed[fixed_start : fixed_start + 128]
            if not len(obstacle):
                continue
            oxlo, oxhi, oylo, oyhi = cs.polygon_bounds(obstacle)
            near = (
                (xlo[:, None] <= oxhi[None, :] + 2e-8)
                & (oxlo[None, :] <= xhi[:, None] + 2e-8)
                & (ylo[:, None] <= oyhi[None, :] + 2e-8)
                & (oylo[None, :] <= yhi[:, None] + 2e-8)
            )
            rows = np.flatnonzero(keep[start : start + len(batch)])
            for row in rows:
                cols = np.flatnonzero(near[row])
                if len(cols) and not cs.separated(
                    batch[row : row + 1],
                    obstacle[cols],
                ):
                    keep[start + row] = False
    return candidates[keep]


def candidate_grid(
    window: tuple[float, float, float, float],
    side: float,
    step: float,
    shift: float,
    fixed: np.ndarray,
) -> np.ndarray:
    x0 = max(0.5, window[0])
    x1 = min(side - 0.5, window[1])
    y0 = max(0.5, window[2])
    y1 = min(side - 0.5, window[3])
    if x1 < x0 or y1 < y0:
        return np.empty((0, 3), dtype=np.float64)
    x_start = math.ceil((x0 - shift) / step) * step + shift
    y_start = math.ceil((y0 - shift) / step) * step + shift
    xs = np.arange(x_start, x1 + 0.5 * step, step)
    ys = np.arange(y_start, y1 + 0.5 * step, step)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    points = np.column_stack([gx.ravel(), gy.ravel()]).astype(np.float64)
    return filter_candidates(points, side, fixed)


def solve_candidates(
    candidates: np.ndarray,
    time_limit: float,
) -> np.ndarray:
    if not len(candidates):
        return candidates
    values = [tuple(square) for square in candidates]
    x, y, adjacency = fillfast.conflict_graph(values)
    taken, alive = fillfast.corner_reduce(x, y, adjacency)
    chosen = list(taken)
    for component in fillfast.components(alive, adjacency):
        if len(component) == 1:
            chosen.append(component[0])
        else:
            chosen += fillfast._milp(component, adjacency, time_limit)
    return candidates[chosen]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--probe-json", type=Path, required=True)
    parser.add_argument("--window-count", type=int, default=8)
    parser.add_argument("--window-start", type=int, default=0)
    parser.add_argument("--expand", type=float, default=0.75)
    parser.add_argument("--step", type=float, default=0.1)
    parser.add_argument("--grid-shift", type=float, default=0.0)
    parser.add_argument("--local-radius", type=float, default=0.2)
    parser.add_argument("--local-step", type=float, default=0.025)
    parser.add_argument("--time-limit", type=float, default=30.0)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    squares = read_squares(args.csv)
    probe = json.loads(args.probe_json.read_text(encoding="utf-8"))
    side = float(probe["side"])
    upright = np.abs(squares[:, 2]) <= 1e-8
    results = []
    current = squares
    indexed_gaps = list(enumerate(probe["gaps"]))
    for index, gap in indexed_gaps[
        args.window_start : args.window_start + args.window_count
    ]:
        x0, x1, y0, y1 = (float(value) for value in gap["bbox"])
        window = (
            max(0.0, x0 - args.expand),
            min(side, x1 + args.expand),
            max(0.0, y0 - args.expand),
            min(side, y1 + args.expand),
        )
        moved = np.array(
            [
                overlaps_window(square, window)
                for square in current[np.abs(current[:, 2]) <= 1e-8]
            ],
            dtype=bool,
        )
        upright_current = current[np.abs(current[:, 2]) <= 1e-8]
        old_window = upright_current[moved]
        fixed = np.vstack(
            [
                current[current[:, 2] != 0],
                upright_current[~moved],
            ]
        )
        grid_candidates = candidate_grid(
            window, side, args.step, args.grid_shift, fixed
        )
        if args.local_radius > 0 and len(old_window):
            offsets = np.arange(
                -args.local_radius,
                args.local_radius + 0.5 * args.local_step,
                args.local_step,
            )
            dx, dy = np.meshgrid(offsets, offsets, indexing="ij")
            displacement = np.column_stack([dx.ravel(), dy.ravel()])
            local_points = (
                old_window[None, :, :2] + displacement[:, None, :]
            ).reshape(-1, 2)
            local_points = local_points[
                (local_points[:, 0] >= window[0])
                & (local_points[:, 0] <= window[1])
                & (local_points[:, 1] >= window[2])
                & (local_points[:, 1] <= window[3])
            ]
            local_candidates = filter_candidates(local_points, side, fixed)
        else:
            local_candidates = np.empty((0, 3), dtype=np.float64)
        candidates = np.vstack(
            [old_window, grid_candidates, local_candidates]
        ).reshape(-1, 3)
        candidates = np.unique(np.round(candidates, 10), axis=0)
        if len(candidates) and len(old_window):
            distances = np.max(
                np.abs(
                    old_window[:, None, :2]
                    - candidates[None, :, :2]
                ),
                axis=2,
            )
            rediscovered = int((distances.min(axis=1) <= 1e-8).sum())
        else:
            rediscovered = 0
        control = solve_candidates(old_window, args.time_limit)
        selected = solve_candidates(candidates, args.time_limit)
        before = int(len(old_window))
        after = int(len(selected))
        accepted = after > before
        if accepted:
            current = np.vstack([fixed, selected])
        results.append(
            {
                "index": index,
                "window": window,
                "source_gap_area": float(gap["area"]),
                "old_upright": before,
                "candidates": int(len(candidates)),
                "rediscovered_old": rediscovered,
                "old_control": int(len(control)),
                "new_upright": after,
                "gain": after - before,
                "accepted": accepted,
            }
        )
        print(
            f"window={index} old={before} candidates={len(candidates)} "
            f"rediscovered={rediscovered} control={len(control)} "
            f"new={after} gain={after - before} accepted={accepted}",
            flush=True,
        )
    result = {
        "input": str(args.csv),
        "probe": str(args.probe_json),
        "side": side,
        "initial_count": int(len(squares)),
        "final_count": int(len(current)),
        "expand": args.expand,
        "window_start": args.window_start,
        "window_count": args.window_count,
        "step": args.step,
        "grid_shift": args.grid_shift,
        "local_radius": args.local_radius,
        "local_step": args.local_step,
        "results": results,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"initial={len(squares)} final={len(current)} "
        f"gain={len(current) - len(squares)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
