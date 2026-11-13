#!/usr/bin/env python3
"""Fine-grid exact set packing around the largest voids of a high-n seed."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

for variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[variable] = "1"

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[3]
CORNER_POC = ROOT / "poc" / "corner-pocket-chirality" / "poc"
sys.path.insert(0, str(CORNER_POC))

import chirality_scan as cs  # noqa: E402
import probe_layout  # noqa: E402
import setpack_tail as st  # noqa: E402


def read_squares(path: Path) -> np.ndarray:
    with path.open(newline="", encoding="utf-8") as handle:
        return np.asarray(
            [
                (float(row["x"]), float(row["y"]), float(row["angle"]))
                for row in csv.DictReader(handle)
            ],
            dtype=np.float64,
        )


def local_candidates(
    bounds: tuple[float, float, float, float],
    side: float,
    centre_step: float,
    angle_step_deg: float,
) -> np.ndarray:
    x0, x1, y0, y1 = bounds
    angles = np.radians(
        np.arange(0.0, 90.0, angle_step_deg, dtype=np.float64)
    )
    generated = []
    for angle in angles:
        extent = 0.5 * (abs(math.cos(angle)) + abs(math.sin(angle)))
        low_x = max(extent, x0)
        high_x = min(side - extent, x1)
        low_y = max(extent, y0)
        high_y = min(side - extent, y1)
        if low_x > high_x or low_y > high_y:
            continue
        xs = np.arange(low_x, high_x + 1e-12, centre_step)
        ys = np.arange(low_y, high_y + 1e-12, centre_step)
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        generated.append(
            np.column_stack(
                [
                    gx.ravel(),
                    gy.ravel(),
                    np.full(gx.size, angle, dtype=np.float64),
                ]
            )
        )
    if not generated:
        return np.empty((0, 3), dtype=np.float64)
    return np.unique(np.round(np.vstack(generated), 10), axis=0)


def without_fixed_conflicts(
    candidates: np.ndarray,
    fixed: np.ndarray,
    tree: cKDTree,
) -> np.ndarray:
    if not len(candidates):
        return candidates
    fixed_index = tree.query_ball_point(
        0.5 * (candidates[:, :2].min(axis=0) + candidates[:, :2].max(axis=0)),
        r=0.5 * float(np.ptp(candidates[:, 0])) + 0.5 * float(np.ptp(candidates[:, 1])) + 2.0,
    )
    nearby = fixed[np.asarray(fixed_index, dtype=np.int64)]
    fixed_u = np.column_stack([np.cos(nearby[:, 2]), np.sin(nearby[:, 2])])
    fixed_v = np.column_stack([-fixed_u[:, 1], fixed_u[:, 0]])
    keep = np.zeros(len(candidates), dtype=bool)
    for angle in np.unique(candidates[:, 2]):
        angle_mask = np.isclose(candidates[:, 2], angle, rtol=0.0, atol=1e-14)
        points = candidates[angle_mask, :2]
        test_u = np.array([math.cos(angle), math.sin(angle)])
        test_v = np.array([-test_u[1], test_u[0]])
        dx = nearby[None, :, 0] - points[:, None, 0]
        dy = nearby[None, :, 1] - points[:, None, 1]
        clear = np.zeros(len(points), dtype=bool)
        for axis in (test_u, test_v):
            projection = np.abs(dx * axis[..., 0] + dy * axis[..., 1])
            radius = 0.5 * (
                np.abs(axis @ test_u)
                + np.abs(axis @ test_v)
                + np.abs(fixed_u @ axis)
                + np.abs(fixed_v @ axis)
            )
            clear |= (projection - radius[None, :] >= -1e-9).all(axis=1)
        for axis in (fixed_u, fixed_v):
            projection = np.abs(
                dx * axis[None, :, 0] + dy * axis[None, :, 1]
            )
            radius = 0.5 * (
                np.abs(test_u @ axis.T)
                + np.abs(test_v @ axis.T)
                + np.abs(np.einsum("ij,ij->i", fixed_u, axis))
                + np.abs(np.einsum("ij,ij->i", fixed_v, axis))
            )
            clear |= (projection - radius[None, :] >= -1e-9).all(axis=1)
        keep[angle_mask] = clear
    return candidates[keep]


def solve_local(
    local: np.ndarray,
    time_limit: float,
) -> np.ndarray | None:
    if not len(local):
        return None
    edges = st.conflict_edges(local)
    if len(edges):
        rows = np.repeat(np.arange(len(edges)), 2)
        columns = np.column_stack([edges[:, 0], edges[:, 1]]).astype(np.int64)
        matrix = coo_matrix(
            (np.ones(2 * len(edges)), (rows, columns.ravel())),
            shape=(len(edges), len(local)),
        ).tocsr()
        constraints: object = LinearConstraint(matrix, -np.inf, 1.0)
    else:
        constraints = ()
    result = milp(
        -np.ones(len(local)),
        constraints=constraints,
        integrality=np.ones(len(local)),
        bounds=Bounds(0, 1),
        options={"time_limit": time_limit, "presolve": True},
    )
    if result.x is None:
        return None
    return local[result.x > 0.5]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-csv", required=True, type=Path)
    parser.add_argument("--side", required=True, type=float)
    parser.add_argument("--target", required=True, type=int)
    parser.add_argument("--centre-step", type=float, default=0.1)
    parser.add_argument("--angle-step", type=float, default=0.5)
    parser.add_argument("--window-margin", type=float, default=0.6)
    parser.add_argument("--minimum-gap-area", type=float, default=0.3)
    parser.add_argument("--window-limit", type=int, default=80)
    parser.add_argument("--raster-scale", type=int, default=32)
    parser.add_argument("--time-limit", type=float, default=10.0)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--candidate-out", required=True, type=Path)
    args = parser.parse_args()

    started = time.time()
    current = read_squares(args.seed_csv)
    initial_count = len(current)
    gaps = probe_layout.gap_components(
        current,
        args.side,
        args.raster_scale,
        args.minimum_gap_area,
    )
    interior = [
        gap
        for gap in gaps
        if not (
            gap["bbox"][0] == 0.0
            and gap["bbox"][2] == 0.0
            and gap["bbox"][1] >= args.side - 1.0
        )
    ]
    windows = interior[: args.window_limit]
    records = []
    for iteration, gap in enumerate(windows):
        x0, x1, y0, y1 = (float(value) for value in gap["bbox"])
        bounds = (
            max(0.0, x0 - args.window_margin),
            min(args.side, x1 + args.window_margin),
            max(0.0, y0 - args.window_margin),
            min(args.side, y1 + args.window_margin),
        )
        in_window = (
            (current[:, 0] >= bounds[0])
            & (current[:, 0] <= bounds[1])
            & (current[:, 1] >= bounds[2])
            & (current[:, 1] <= bounds[3])
        )
        fixed = current[~in_window]
        old_local = current[in_window]
        candidates = local_candidates(
            bounds,
            args.side,
            args.centre_step,
            args.angle_step,
        )
        tree = cKDTree(fixed[:, :2])
        free = without_fixed_conflicts(candidates, fixed, tree)
        local = np.unique(
            np.round(np.vstack([old_local, free]), 10),
            axis=0,
        )
        chosen = solve_local(local, args.time_limit)
        new_count = 0 if chosen is None else len(chosen)
        old_count = len(old_local)
        accepted = new_count > old_count
        if accepted:
            current = np.vstack([fixed, chosen])
        total = len(current)
        records.append(
            {
                "iteration": iteration,
                "gap_area": gap["area"],
                "bounds": bounds,
                "candidates": len(candidates),
                "free": len(free),
                "local": len(local),
                "old": old_count,
                "new": new_count,
                "total": total,
                "accepted": accepted,
            }
        )
        print(
            f"iteration={iteration} gap={gap['area']:.3f} "
            f"candidates={len(candidates)} free={len(free)} "
            f"local={len(local)} old={old_count} new={new_count} total={total}",
            flush=True,
        )
        if total >= args.target:
            break

    st.write_candidate(args.candidate_out, args.target, current)
    report = {
        "seed_csv": str(args.seed_csv.resolve()),
        "side": args.side,
        "target": args.target,
        "initial_count": initial_count,
        "best_count": len(current),
        "centre_step": args.centre_step,
        "angle_step_deg": args.angle_step,
        "window_margin": args.window_margin,
        "minimum_gap_area": args.minimum_gap_area,
        "raster_scale": args.raster_scale,
        "elapsed_seconds": time.time() - started,
        "records": records,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"best={len(current)}/{args.target}", flush=True)


if __name__ == "__main__":
    main()
