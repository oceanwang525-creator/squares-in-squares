#!/usr/bin/env python3
"""Narrow Mondrian crops in their own rectangles, then refill exactly."""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import os
import sys
from dataclasses import dataclass
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
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[3]
MONDRIAN = ROOT / "poc" / "highn-mondrian" / "poc"
CLAUDE = ROOT / "poc" / "claude-midn" / "poc"
sys.path.insert(0, str(MONDRIAN))
sys.path.insert(0, str(CLAUDE))
import fillfast  # noqa: E402
import scan  # noqa: E402
import seamscan  # noqa: E402
import tight_lp  # noqa: E402


MARGIN = 5e-13
PAIR_RADIUS = 1.65
W = 0.5 * np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]], float)


@dataclass(frozen=True)
class Piece:
    crop: scan.Crop
    width: float
    height: float
    centers: np.ndarray
    angles: np.ndarray


def perp(v: np.ndarray) -> np.ndarray:
    return np.stack([-v[..., 1], v[..., 0]], -1)


def corners_rel(theta: np.ndarray) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.stack(
        [
            c[:, None] * W[None, :, 0] - s[:, None] * W[None, :, 1],
            s[:, None] * W[None, :, 0] + c[:, None] * W[None, :, 1],
        ],
        -1,
    )


def extents(theta: np.ndarray) -> np.ndarray:
    return 0.5 * (np.abs(np.cos(theta)) + np.abs(np.sin(theta)))


def normalized_piece(
    packings: dict[int, list[dict[str, str]]], crop: scan.Crop
) -> Piece:
    rows = [packings[crop.donor_n][index] for index in crop.indices]
    centers = np.array(
        [[float(row["x"]), float(row["y"])] for row in rows], dtype=float
    )
    angles = np.array([float(row["angle"]) for row in rows], dtype=float)
    radius = extents(angles)
    low = np.array(
        [(centers[:, 0] - radius).min(), (centers[:, 1] - radius).min()]
    )
    centers -= low
    return Piece(crop, crop.width, crop.height, centers, angles)


def wall_state_ok(
    centers: np.ndarray,
    angles: np.ndarray,
    walls: tuple[float, float, float, float],
    tol: float = 1e-12,
) -> bool:
    radius = extents(angles)
    x0, x1, y0, y1 = walls
    if (centers[:, 0] - radius).min() < x0 - tol:
        return False
    if (centers[:, 0] + radius).max() > x1 + tol:
        return False
    if (centers[:, 1] - radius).min() < y0 - tol:
        return False
    if (centers[:, 1] + radius).max() > y1 + tol:
        return False
    return tight_lp.pair_axes(centers, angles)[3].min() >= -1e-11


def fixed_wall_lp(
    centers: np.ndarray,
    angles: np.ndarray,
    walls: tuple[float, float, float, float],
    trust: float,
    repair: bool = False,
) -> np.ndarray | None:
    """Exact fixed-angle centre LP inside fixed rectangular walls."""
    n = len(centers)
    i, j, axes, gap = tight_lp.pair_axes(centers, angles)
    rows = np.arange(len(i))
    rhs = np.maximum(gap - MARGIN, 0.0)
    if repair:
        rhs = np.where(gap < 0.0, gap - MARGIN, rhs)
    row_parts = [
        np.repeat(rows, 4),
    ]
    col_parts = [
        np.column_stack((2 * i, 2 * i + 1, 2 * j, 2 * j + 1)).ravel(),
    ]
    val_parts = [
        np.column_stack(
            (axes[:, 0], axes[:, 1], -axes[:, 0], -axes[:, 1])
        ).ravel(),
    ]
    rhs_parts = [rhs]

    radius = extents(angles)
    x0, x1, y0, y1 = walls
    row = len(i)
    index = np.arange(n)
    for axis, low, high in ((0, x0, x1), (1, y0, y1)):
        lo_rows = row + index
        row_parts.append(lo_rows)
        col_parts.append(2 * index + axis)
        val_parts.append(-np.ones(n))
        rhs_parts.append(centers[:, axis] - radius - low)
        row += n

        hi_rows = row + index
        row_parts.append(hi_rows)
        col_parts.append(2 * index + axis)
        val_parts.append(np.ones(n))
        rhs_parts.append(high - radius - centers[:, axis])
        row += n

    matrix = coo_matrix(
        (
            np.concatenate(val_parts),
            (np.concatenate(row_parts), np.concatenate(col_parts)),
        ),
        shape=(row, 2 * n),
    ).tocsr()
    result = linprog(
        np.zeros(2 * n),
        A_ub=matrix,
        b_ub=np.concatenate(rhs_parts),
        bounds=[(-trust, trust)] * (2 * n),
        method="highs-ds",
        options={"time_limit": 30},
    )
    if not result.success:
        return None
    candidate = centers + result.x.reshape(n, 2)
    if not wall_state_ok(candidate, angles, walls):
        return None
    return candidate


def wall_slp_step(
    centers: np.ndarray,
    angles: np.ndarray,
    walls: tuple[float, float, float, float],
    trust_move: float,
    trust_angle: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """One free-angle feasibility step with fixed rectangular walls."""
    n = len(centers)
    u = np.c_[np.cos(angles), np.sin(angles)]
    v = perp(u)
    pairs = np.array(
        sorted(cKDTree(centers).query_pairs(PAIR_RADIUS)), dtype=int
    )
    if len(pairs):
        i, j = pairs[:, 0], pairs[:, 1]
        delta = centers[j] - centers[i]
        candidates = (
            (u[i], i, j),
            (v[i], i, j),
            (u[j], j, i),
            (v[j], j, i),
        )
        best = np.full(len(pairs), -np.inf)
        face = np.zeros((len(pairs), 2))
        owner = np.zeros(len(pairs), dtype=int)
        other = np.zeros(len(pairs), dtype=int)
        for axis, own, oth in candidates:
            projection = np.einsum("ij,ij->i", axis, delta)
            radius = 0.5 * (
                np.abs(np.einsum("ij,ij->i", axis, u[i]))
                + np.abs(np.einsum("ij,ij->i", axis, v[i]))
                + np.abs(np.einsum("ij,ij->i", axis, u[j]))
                + np.abs(np.einsum("ij,ij->i", axis, v[j]))
            )
            gap = np.abs(projection) - radius
            better = gap > best
            sign = np.sign(np.einsum("ij,ij->i", axis, centers[oth] - centers[own]))
            face[better] = (axis * sign[:, None])[better]
            best[better] = gap[better]
            owner[better] = own[better]
            other[better] = oth[better]
    else:
        i = np.zeros(0, dtype=int)
        j = np.zeros(0, dtype=int)
        best = np.zeros(0)
        face = np.zeros((0, 2))
        owner = np.zeros(0, dtype=int)
        other = np.zeros(0, dtype=int)

    rel = corners_rel(angles)
    row_parts: list[np.ndarray] = []
    col_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    rhs_parts: list[np.ndarray] = []
    row = 0

    if len(pairs):
        for corner in range(4):
            rk = rel[other, corner]
            vector = centers[other] - centers[owner] + rk
            gap = np.einsum("ij,ij->i", face, vector) - 0.5
            keep = gap < 3.0 * trust_move + 3.0 * trust_angle
            idx = np.flatnonzero(keep)
            if len(idx) == 0:
                continue
            count = len(idx)
            rr = row + np.arange(count)
            axis = face[idx]
            d_gap_owner = np.einsum("ij,ij->i", perp(axis), vector[idx])
            d_gap_other = np.einsum(
                "ij,ij->i", axis, perp(rk[idx])
            )
            own = owner[idx]
            oth = other[idx]
            row_parts.append(np.repeat(rr, 6))
            col_parts.append(
                np.column_stack(
                    (
                        2 * oth,
                        2 * oth + 1,
                        2 * own,
                        2 * own + 1,
                        2 * n + own,
                        2 * n + oth,
                    )
                ).ravel()
            )
            val_parts.append(
                np.column_stack(
                    (
                        -axis[:, 0],
                        -axis[:, 1],
                        axis[:, 0],
                        axis[:, 1],
                        -d_gap_owner,
                        -d_gap_other,
                    )
                ).ravel()
            )
            margin = np.minimum(MARGIN, np.maximum(gap[idx], 0.0))
            rhs_parts.append(np.maximum(gap[idx] - margin, 0.0))
            row += count

    x0, x1, y0, y1 = walls
    for axis, low, high in ((0, x0, x1), (1, y0, y1)):
        for corner in range(4):
            position = centers[:, axis] + rel[:, corner, axis]
            d_rot = perp(rel[:, corner])[:, axis]
            low_slack = position - low
            idx = np.flatnonzero(
                low_slack < trust_move + 0.8 * trust_angle + 1e-9
            )
            if len(idx):
                count = len(idx)
                rr = row + np.arange(count)
                row_parts.append(np.repeat(rr, 3))
                col_parts.append(
                    np.column_stack(
                        (
                            2 * idx + axis,
                            2 * n + idx,
                            np.full(count, 3 * n),
                        )
                    ).ravel()
                )
                val_parts.append(
                    np.column_stack(
                        (
                            -np.ones(count),
                            -d_rot[idx],
                            np.ones(count),
                        )
                    ).ravel()
                )
                rhs_parts.append(np.maximum(low_slack[idx], 0.0))
                row += count

            high_slack = high - position
            idx = np.flatnonzero(
                high_slack < trust_move + 0.8 * trust_angle + 1e-9
            )
            if len(idx):
                count = len(idx)
                rr = row + np.arange(count)
                row_parts.append(np.repeat(rr, 3))
                col_parts.append(
                    np.column_stack(
                        (
                            2 * idx + axis,
                            2 * n + idx,
                            np.full(count, 3 * n),
                        )
                    ).ravel()
                )
                val_parts.append(
                    np.column_stack(
                        (
                            np.ones(count),
                            d_rot[idx],
                            np.ones(count),
                        )
                    ).ravel()
                )
                rhs_parts.append(np.maximum(high_slack[idx], 0.0))
                row += count

    if row == 0:
        return None
    matrix = coo_matrix(
        (
            np.concatenate(val_parts),
            (np.concatenate(row_parts), np.concatenate(col_parts)),
        ),
        shape=(row, 3 * n + 1),
    ).tocsr()
    objective = np.zeros(3 * n + 1)
    objective[-1] = 1.0
    bounds = (
        [(-trust_move, trust_move)] * (2 * n)
        + [(-trust_angle, trust_angle)] * n
        + [(None, 0.0)]
    )
    result = linprog(
        objective,
        A_ub=matrix,
        b_ub=np.concatenate(rhs_parts),
        bounds=bounds,
        method="highs-ds",
        options={"time_limit": 60},
    )
    if not result.success:
        return None
    proposal = result.x
    return (
        centers + proposal[: 2 * n].reshape(n, 2),
        angles + proposal[2 * n : 3 * n],
    )


def solve_walls(
    centers: np.ndarray,
    angles: np.ndarray,
    walls: tuple[float, float, float, float],
    rounds: int = 20,
) -> tuple[np.ndarray, np.ndarray] | None:
    centers = centers.copy()
    angles = angles.copy()
    radius = extents(angles)
    x0, x1, y0, y1 = walls
    centers[:, 0] = np.clip(
        centers[:, 0], x0 + radius, x1 - radius
    )
    centers[:, 1] = np.clip(
        centers[:, 1], y0 + radius, y1 - radius
    )
    repaired = fixed_wall_lp(centers, angles, walls, 0.05, repair=True)
    if repaired is not None:
        centers = repaired

    trust_move = 0.04
    trust_angle = 0.02
    for _ in range(rounds):
        step = wall_slp_step(
            centers, angles, walls, trust_move, trust_angle
        )
        if step is None:
            trust_move *= 0.5
            trust_angle *= 0.5
            if trust_move < 1e-8:
                break
            continue
        proposed_centers, proposed_angles = step
        accepted = False
        for alpha in (1.0, 0.5, 0.25, 0.125):
            q = centers + alpha * (proposed_centers - centers)
            theta = angles + alpha * (proposed_angles - angles)
            q = fixed_wall_lp(q, theta, walls, 0.05, repair=True)
            if q is not None:
                centers, angles = q, theta
                accepted = True
                break
        if accepted:
            trust_move = min(0.04, trust_move * 1.5)
            trust_angle = min(0.02, trust_angle * 1.5)
        else:
            trust_move *= 0.5
            trust_angle *= 0.5
            if trust_move < 1e-8:
                break

    if not wall_state_ok(centers, angles, walls, tol=1e-10):
        return None
    return centers, angles


def narrow_piece(
    piece: Piece,
    target_width: float,
    target_height: float,
    stages: int,
) -> tuple[np.ndarray, np.ndarray, float, float] | None:
    if target_width <= 0.0 or target_height <= 0.0:
        return None
    if target_width * target_height + 1e-10 < piece.crop.capacity:
        return None
    if (
        target_width >= piece.width - 1e-12
        and target_height >= piece.height - 1e-12
    ):
        return (
            piece.centers.copy(),
            piece.angles.copy(),
            piece.width,
            piece.height,
        )
    centers, angles = piece.centers.copy(), piece.angles.copy()
    width, height = piece.width, piece.height
    # A tiny initial wall margin makes the unchanged saturated geometry feasible.
    walls = (0.0, width + 1e-9, 0.0, height + 1e-9)
    if not wall_state_ok(centers, angles, walls):
        return None
    for stage in range(1, stages + 1):
        fraction = float(stage) / stages
        next_width = width + fraction * (target_width - width)
        next_height = height + fraction * (target_height - height)
        next_walls = (0.0, next_width, 0.0, next_height)
        solved = solve_walls(centers, angles, next_walls)
        if solved is None:
            return None
        centers, angles = solved
    radius = extents(angles)
    actual_width = float((centers[:, 0] + radius).max())
    actual_height = float((centers[:, 1] + radius).max())
    if actual_width > target_width + 1e-10:
        return None
    if actual_height > target_height + 1e-10:
        return None
    return centers, angles, actual_width, actual_height


def fit_splits(
    deficit: float,
    width_a: float,
    width_b: float,
    side: float,
) -> tuple[float, ...]:
    """Include fractions plus every split that changes a grid-cell count."""
    if deficit <= scan.EPS:
        return (0.0,)
    values = {
        0.0,
        deficit / 3.0,
        deficit / 2.0,
        2.0 * deficit / 3.0,
        deficit,
    }
    lo = max(0.0, deficit - width_b)
    hi = min(deficit, width_a)
    margin = 1e-10
    for boundary in range(
        math.floor(width_a - hi) - 1,
        math.ceil(width_a - lo) + 2,
    ):
        # Bottom-piece extent crosses an integer from the left.
        value = width_a - boundary
        if lo - margin <= value <= hi + margin:
            values.add(min(hi, max(lo, value)))
    top_low = side - width_b
    top_high = top_low + deficit
    for boundary in range(
        math.floor(top_low) - 1, math.ceil(top_high) + 2
    ):
        # Top-piece extent crosses an integer-relative container boundary.
        value = top_high - boundary
        if lo - margin <= value <= hi + margin:
            values.add(min(hi, max(lo, value)))
    return tuple(sorted(values))


def shrink_phase(
    bottom: scan.Crop,
    top: scan.Crop,
    side: float,
    target_aw: float,
    target_ah: float,
    target_bw: float,
    target_bh: float,
) -> tuple[int, float]:
    """Projected exact upright-cell capacity for fixed transformed crops."""
    touched = scan.touched_cells(
        side, target_aw, target_ah, from_top_right=False
    ) + scan.touched_cells(
        side, target_bw, target_bh, from_top_right=True
    )
    projected = (
        bottom.capacity
        + top.capacity
        + math.floor(side + scan.EPS) ** 2
        - touched
    )
    shrink = (
        bottom.width
        + top.width
        - side
        + bottom.height
        + top.height
        - side
    )
    return projected, shrink


def pair_candidates(
    bottom: list[tuple[int, scan.Crop]],
    top: list[tuple[int, scan.Crop]],
    side: float,
    need: int,
    keep: int,
    max_shrink: float,
) -> list[tuple[float, int, int, scan.Crop, scan.Crop, float, float]]:
    heap: list[
        tuple[
            float,
            int,
            int,
            scan.Crop,
            scan.Crop,
            float,
            float,
        ]
    ] = []
    serial = 0
    for gain_a, crop_a in bottom:
        for gain_b, crop_b in top:
            gain = gain_a + gain_b
            deficit_x = max(0.0, crop_a.width + crop_b.width - side)
            deficit_y = max(0.0, crop_a.height + crop_b.height - side)
            if deficit_x + deficit_y <= scan.EPS:
                continue
            if deficit_x > max_shrink or deficit_y > max_shrink:
                continue
            best: tuple[int, float] | None = None
            for shrink_ax in fit_splits(
                deficit_x, crop_a.width, crop_b.width, side
            ):
                for shrink_ay in fit_splits(
                    deficit_y,
                    crop_a.height,
                    crop_b.height,
                    side,
                ):
                    projected, shrink = shrink_phase(
                        crop_a,
                        crop_b,
                        side,
                        crop_a.width - shrink_ax,
                        crop_a.height - shrink_ay,
                        crop_b.width - (deficit_x - shrink_ax),
                        crop_b.height - (deficit_y - shrink_ay),
                    )
                    target_aw = crop_a.width - shrink_ax
                    target_ah = crop_a.height - shrink_ay
                    target_bw = crop_b.width - (deficit_x - shrink_ax)
                    target_bh = crop_b.height - (deficit_y - shrink_ay)
                    if (
                        target_aw * target_ah + 1e-10
                        < crop_a.capacity
                        or target_bw * target_bh + 1e-10
                        < crop_b.capacity
                    ):
                        continue
                    score = (projected, -shrink)
                    if best is None or score > best:
                        best = score
            if best is None:
                continue
            projected, negative_shrink = best
            if projected < need:
                continue
            priority = projected + 1e-3 * negative_shrink
            item = (
                priority,
                serial,
                gain,
                crop_a,
                crop_b,
                deficit_x,
                deficit_y,
            )
            serial += 1
            if len(heap) < keep:
                heapq.heappush(heap, item)
            elif priority > heap[0][0]:
                heapq.heapreplace(heap, item)
    return sorted(heap, key=lambda item: item[0], reverse=True)


def compose_variants(
    piece_a: Piece,
    piece_b: Piece,
    side: float,
    deficit_x: float,
    deficit_y: float,
    stages: int,
    max_solves: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[dict[str, object]]]:
    combined: list[tuple[np.ndarray, np.ndarray]] = []
    records: list[dict[str, object]] = []
    seen: set[tuple[int, int, int, int]] = set()
    solves = 0
    for shrink_ax in fit_splits(
        deficit_x, piece_a.width, piece_b.width, side
    ):
        for shrink_ay in fit_splits(
            deficit_y, piece_a.height, piece_b.height, side
        ):
            target_aw = piece_a.width - shrink_ax
            target_ah = piece_a.height - shrink_ay
            target_bw = piece_b.width - (deficit_x - shrink_ax)
            target_bh = piece_b.height - (deficit_y - shrink_ay)
            key = tuple(
                int(round(value * 1e9))
                for value in (target_aw, target_ah, target_bw, target_bh)
            )
            if key in seen:
                continue
            seen.add(key)
            solved_a = narrow_piece(
                piece_a, target_aw, target_ah, stages
            )
            solves += 1
            if solved_a is None:
                records.append(
                    {
                        "target": [target_aw, target_ah, target_bw, target_bh],
                        "a": "infeasible",
                        "b": None,
                    }
                )
            else:
                solved_b = narrow_piece(
                    piece_b, target_bw, target_bh, stages
                )
                solves += 1
                if solved_b is None:
                    records.append(
                        {
                            "target": [
                                target_aw,
                                target_ah,
                                target_bw,
                                target_bh,
                            ],
                            "a": "ok",
                            "b": "infeasible",
                        }
                    )
                else:
                    centers_a, angles_a, actual_aw, actual_ah = solved_a
                    centers_b, angles_b, actual_bw, actual_bh = solved_b
                    placed_a = centers_a
                    placed_b = centers_b + np.array(
                        [side - actual_bw, side - actual_bh]
                    )
                    combined.append(
                        (
                            np.vstack((placed_a, placed_b)),
                            np.concatenate((angles_a, angles_b)),
                        )
                    )
                    records.append(
                        {
                            "target": [
                                target_aw,
                                target_ah,
                                target_bw,
                                target_bh,
                            ],
                            "a": "ok",
                            "b": "ok",
                            "actual": [
                                actual_aw,
                                actual_ah,
                                actual_bw,
                                actual_bh,
                            ],
                        }
                    )
            if solves >= max_solves:
                return combined, records
    return combined, records


def exact_fill(
    side: float, centers: np.ndarray, angles: np.ndarray, limit: int
) -> list[tuple[float, float, float]]:
    tilted = np.column_stack((centers, angles))
    filled = fillfast.fill(
        side, tilted, mode="exact", time_limit=limit
    )
    return list(filled)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--submission",
        type=Path,
        default=ROOT
        / "poc"
        / "pell_corrugation"
        / "data"
        / "border-scan"
        / "merged.csv",
    )
    parser.add_argument("--targets", default="908,971")
    parser.add_argument(
        "--donors",
        default="791,908,971,972",
        help='comma-separated donor n values, or "all"',
    )
    parser.add_argument("--min-crop", type=int, default=4)
    parser.add_argument("--max-crop", type=int, default=20)
    parser.add_argument("--keep-crops", type=int, default=800)
    parser.add_argument("--top-pairs", type=int, default=30)
    parser.add_argument("--max-shrink", type=float, default=2.5)
    parser.add_argument("--stages", type=int, default=8)
    parser.add_argument("--max-solves", type=int, default=20)
    parser.add_argument("--milp-limit", type=int, default=10)
    parser.add_argument("--margin", default="1e-9")
    parser.add_argument(
        "--summary", type=Path, default=Path("/tmp/mondrian-narrow.json")
    )
    parser.add_argument(
        "--rows-out", type=Path, default=Path("/tmp/mondrian-narrow.csv")
    )
    args = parser.parse_args()

    packings = scan.read_groups(args.submission)
    if args.donors == "all":
        donor_packings = packings
    else:
        donors = {int(value) for value in args.donors.split(",") if value}
        donor_packings = {
            n: rows for n, rows in packings.items() if n in donors
        }
        if not donor_packings:
            raise SystemExit("no requested donor packings were found")
    report: dict[str, object] = {
        "submission": str(args.submission),
        "margin": args.margin,
        "targets": {},
    }
    accepted: dict[int, list[dict[str, str]]] = {}

    for target in map(int, args.targets.split(",")):
        incumbent = Decimal(scan.core.certify(target, packings[target]))
        side_dec = incumbent - Decimal(args.margin)
        side = float(side_dec)
        limit = math.floor(side + scan.EPS)
        need = target - limit * limit
        bottom, top = seamscan.top_crops(
            donor_packings,
            side,
            args.min_crop,
            args.max_crop,
            args.keep_crops,
        )
        candidates = pair_candidates(
            bottom,
            top,
            side,
            need,
            args.top_pairs,
            args.max_shrink,
        )
        target_record: dict[str, object] = {
            "incumbent": str(incumbent),
            "side": str(side_dec),
            "limit": limit,
            "need": need,
            "bottom_crops": len(bottom),
            "top_crops": len(top),
            "candidate_pairs": len(candidates),
            "attempts": [],
            "winner": None,
        }
        print(
            f"n={target} K={limit} need={need} "
            f"bottom={len(bottom)} top={len(top)} "
            f"pairs={len(candidates)}",
            flush=True,
        )

        for priority, serial, gain, crop_a, crop_b, dx, dy in candidates:
            piece_a = normalized_piece(packings, crop_a)
            piece_b = normalized_piece(packings, crop_b)
            variants, solves = compose_variants(
                piece_a,
                piece_b,
                side,
                dx,
                dy,
                args.stages,
                args.max_solves,
            )
            attempt: dict[str, object] = {
                "serial": serial,
                "priority": priority,
                "gain": gain,
                "deficit": [dx, dy],
                "crop_a": {
                    "donor_n": crop_a.donor_n,
                    "corner": crop_a.corner,
                    "capacity": crop_a.capacity,
                    "width": crop_a.width,
                    "height": crop_a.height,
                },
                "crop_b": {
                    "donor_n": crop_b.donor_n,
                    "corner": crop_b.corner,
                    "capacity": crop_b.capacity,
                    "width": crop_b.width,
                    "height": crop_b.height,
                },
                "narrowing": solves,
                "variants": len(variants),
                "best_capacity": -1,
            }
            print(
                f" pair={serial} gain={gain} deficit={dx:.4f},{dy:.4f} "
                f"narrowed={len(variants)}",
                flush=True,
            )
            for centers, angles in variants:
                filled = exact_fill(
                    side, centers, angles, args.milp_limit
                )
                capacity = len(filled)
                if capacity > int(attempt["best_capacity"]):
                    attempt["best_capacity"] = capacity
                    attempt["best_short"] = target - capacity
                if capacity < target:
                    continue
                rows = [
                    {
                        "square_id": str(index),
                        "x": format(float(x), ".17g"),
                        "y": format(float(y), ".17g"),
                        "angle": format(float(angle), ".17g"),
                    }
                    for index, (x, y, angle) in enumerate(filled[:target])
                ]
                literal, pandas_side = scan.certify_both(rows)
                attempt["winner_capacity"] = capacity
                attempt["certified_literal_L"] = literal
                attempt["certified_pandas_L"] = pandas_side
                if Decimal(literal) < incumbent and Decimal(pandas_side) < incumbent:
                    attempt["strict_win"] = True
                    accepted[target] = rows
                    target_record["winner"] = attempt
                    break
            target_record["attempts"].append(attempt)
            if target in accepted:
                break
        report["targets"][str(target)] = target_record
        print(
            f"n={target} accepted={target in accepted} "
            f"best_short={min((a['best_short'] for a in target_record['attempts'] if 'best_short' in a), default=None)}",
            flush=True,
        )

    report["accepted_n"] = sorted(accepted)
    args.summary.write_text(
        json.dumps(report, indent=1) + "\n", encoding="utf-8"
    )
    with args.rows_out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=scan.COLS, lineterminator="\n"
        )
        writer.writeheader()
        for n in sorted(accepted):
            writer.writerows(
                {
                    "n": n,
                    **{
                        key: row[key]
                        for key in scan.COLS
                        if key != "n"
                    },
                }
                for row in accepted[n]
            )
    print(f"accepted {len(accepted)}/{len(report['targets'])}", flush=True)


if __name__ == "__main__":
    main()
