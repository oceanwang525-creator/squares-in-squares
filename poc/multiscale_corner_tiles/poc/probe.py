#!/usr/bin/env python3
"""Bounded exact probe for two/three-layer corner boundary tiles."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from decimal import Decimal
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree

POC = Path(__file__).resolve().parents[1]
ROOT = POC.parents[1]
INCUMBENT = ROOT / "poc/pell_corrugation/data/border-scan/merged.csv"
sys.path.insert(0, str(ROOT / "poc/baseline/poc/src"))

from squares import core  # noqa: E402

EPS = 2e-9
GAP = 1e-8


def extent(angle: float) -> float:
    return 0.5 * (abs(math.cos(angle)) + abs(math.sin(angle)))


def read_groups(path: Path) -> dict[int, list[dict[str, str]]]:
    groups: dict[int, list[dict[str, str]]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            groups.setdefault(int(row["n"]), []).append(row)
    return groups


def side_of(rows: list[dict[str, str]]) -> Decimal:
    return Decimal(core.certify(int(rows[0]["n"]), rows))


def float_side(rows: list[dict[str, str]]) -> float:
    xmin = ymin = math.inf
    xmax = ymax = -math.inf
    for row in rows:
        h = extent(float(row["angle"]))
        x, y = float(row["x"]), float(row["y"])
        xmin, ymin = min(xmin, x - h), min(ymin, y - h)
        xmax, ymax = max(xmax, x + h), max(ymax, y + h)
    return max(xmax - xmin, ymax - ymin)


def upright_candidates(
    side: float, phase: tuple[float, float]
) -> list[tuple[float, float, float]]:
    count = math.floor(side + 1e-10)
    out: list[tuple[float, float, float]] = []
    x0, y0 = phase
    for i in range(count):
        x = x0 + i + 0.5
        if x + 0.5 > side + 1e-9:
            break
        for j in range(count):
            y = y0 + j + 0.5
            if y + 0.5 <= side + 1e-9:
                out.append((x, y, 0.0))
    return out


def wall_transform(
    wall: str, normal: float, tangent: float, angle: float, side: float
) -> tuple[float, float, float]:
    if wall == "left":
        return normal, tangent, angle
    if wall == "right":
        return side - normal, tangent, math.pi / 2.0 - angle
    if wall == "bottom":
        return tangent, normal, math.pi / 2.0 - angle
    return tangent, side - normal, angle


def layer_band(
    side: float,
    wall: str,
    angle: float,
    depth0: float,
    depth1: float,
    phase_normal: float,
    phase_tangent: float,
) -> list[tuple[float, float, float]]:
    """Points of one rotated square lattice inside a wall-depth band."""
    c = math.cos(angle)
    s = math.sin(angle)
    # The lattice basis is the square's own axes.  Phases are fractions of a
    # lattice step, so different layers interlock instead of sharing one seam.
    origin_n = depth0 + phase_normal
    origin_t = -2.0 + phase_tangent
    reach = int(math.ceil(side + 5.0))
    out: list[tuple[float, float, float]] = []
    h = extent(angle)
    for i in range(-reach, reach + 1):
        for j in range(-reach, reach + 1):
            n = origin_n + i * c - j * s
            t = origin_t + i * s + j * c
            if n < depth0 - 1e-10 or n > depth1 + 1e-10:
                continue
            if n < h - 1e-9 or t < h - 1e-9:
                continue
            if n > side - h + 1e-9 or t > side - h + 1e-9:
                continue
            out.append(wall_transform(wall, n, t, angle, side))
    return out


def candidate_pool(
    side: float,
    wall: str,
    angles: tuple[float, ...],
    cuts: tuple[float, ...],
    grid_phase: tuple[float, float],
) -> tuple[np.ndarray, list[dict[str, float | str]]]:
    candidates = upright_candidates(side, grid_phase)
    layers: list[dict[str, float | str]] = []
    total_depth = cuts[-1]
    for layer, angle_deg in enumerate(angles):
        depth0 = 0.5 if layer == 0 else cuts[layer - 1]
        depth1 = cuts[layer]
        for pn, pt in ((0.0, 0.0), (0.25, 0.0), (0.5, 0.25), (0.75, 0.5)):
            pts = layer_band(
                side,
                wall,
                math.radians(angle_deg),
                depth0,
                depth1,
                pn,
                pt,
            )
            candidates.extend(pts)
            if pts:
                layers.append(
                    dict(
                        wall=wall,
                        angle_deg=angle_deg,
                        depth0=depth0,
                        depth1=depth1,
                        phase_normal=pn,
                        phase_tangent=pt,
                        points=len(pts),
                    )
                )
    arr = np.unique(np.round(np.asarray(candidates, dtype=np.float64), 10), axis=0)
    return np.asarray(arr), layers


def overlap_conflicts(candidates: np.ndarray) -> np.ndarray:
    if len(candidates) < 2:
        return np.empty((0, 2), dtype=np.int64)
    tree = cKDTree(candidates[:, :2])
    pairs = tree.query_pairs(math.sqrt(2.0) + 2e-7, output_type="ndarray")
    if len(pairs) == 0:
        return np.empty((0, 2), dtype=np.int64)
    i, j = pairs[:, 0], pairs[:, 1]
    dx = candidates[i, 0] - candidates[j, 0]
    dy = candidates[i, 1] - candidates[j, 1]
    ai, aj = candidates[i, 2], candidates[j, 2]
    ci, si = np.cos(ai), np.sin(ai)
    p1 = np.abs(dx * ci + dy * si)
    p2 = np.abs(-dx * si + dy * ci)
    radius = 0.5 * (
        1.0 + np.abs(np.cos(ai - aj)) + np.abs(np.sin(ai - aj))
    )
    conflict = (p1 < radius - EPS) & (p2 < radius - EPS)
    return pairs[conflict]


def solve_mis(
    candidates: np.ndarray, conflicts: np.ndarray, seconds: float
) -> tuple[np.ndarray, str]:
    if not len(conflicts):
        return np.arange(len(candidates), dtype=np.int64), "no conflicts"
    rows = np.repeat(np.arange(len(conflicts)), 2)
    cols = conflicts.ravel()
    matrix = coo_matrix(
        (np.ones(len(rows)), (rows, cols)),
        shape=(len(conflicts), len(candidates)),
    ).tocsr()
    result = milp(
        -np.ones(len(candidates)),
        constraints=LinearConstraint(matrix, -np.inf, 1.0),
        integrality=np.ones(len(candidates)),
        bounds=Bounds(0.0, 1.0),
        options=dict(time_limit=max(0.2, seconds), disp=False),
    )
    if result.x is None:
        return np.empty(0, dtype=np.int64), result.message
    return np.flatnonzero(result.x > 0.5), result.message


def rows_from_points(points: np.ndarray, n: int) -> list[dict[str, str]]:
    points = points[np.lexsort((points[:, 0], points[:, 1], points[:, 2]))][:n]
    return [
        dict(
            square_id=str(i),
            x=format(float(row[0]), ".15g"),
            y=format(float(row[1]), ".15g"),
            angle=format(float(row[2]), ".15g"),
        )
        for i, row in enumerate(points)
    ]


def dual_certify(
    n: int, rows: list[dict[str, str]], incumbent: Decimal
) -> dict[str, object]:
    literal = core.certify(n, rows)
    transported = core.kaggle_roundtrip({n: rows})[n]
    pandas = core.certify(n, transported)
    return dict(
        literal=literal,
        pandas=pandas,
        strict_win=Decimal(literal) < incumbent and Decimal(pandas) < incumbent,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", default="328,365,367")
    parser.add_argument("--delta", type=float, default=0.0001)
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument(
        "--output", type=Path, default=POC / "docs/results.json"
    )
    args = parser.parse_args()
    groups = read_groups(INCUMBENT)
    schedules = [
        ((44.0, 46.0), (1.6, 3.2)),
        ((40.0, 50.0), (1.6, 3.2)),
        ((36.87, 53.13), (1.6, 3.2)),
        ((40.0, 45.0, 50.0), (1.25, 2.5, 3.75)),
        ((30.0, 45.0, 60.0), (1.25, 2.5, 3.75)),
    ]
    report: dict[str, object] = {
        "targets": [],
        "delta": args.delta,
        "seconds_per_milp": args.seconds,
        "started_unix": time.time(),
    }
    wins: dict[int, list[dict[str, str]]] = {}
    for n in [int(x) for x in args.targets.split(",") if x]:
        rows = groups[n]
        incumbent = side_of(rows)
        target_side = float(incumbent) - args.delta
        target_report: dict[str, object] = {
            "n": n,
            "incumbent_L": str(incumbent),
            "probe_side": target_side,
            "schedule_results": [],
            "best_count": 0,
        }
        frac = target_side - math.floor(target_side)
        grid_phases = (
            (0.0, 0.0),
            (frac / 2.0, frac / 2.0),
            (frac, frac),
        )
        for schedule_index, (angles, cuts) in enumerate(schedules):
            for wall in ("left", "right", "bottom", "top"):
                for grid_phase in grid_phases:
                    candidates, layers = candidate_pool(
                        target_side, wall, angles, cuts, grid_phase
                    )
                    conflicts = overlap_conflicts(candidates)
                    selected, status = solve_mis(
                        candidates, conflicts, args.seconds
                    )
                    count = len(selected)
                    tile_candidates = candidates[np.abs(candidates[:, 2]) > 1e-14]
                    tile_selected, _ = solve_mis(
                        tile_candidates,
                        overlap_conflicts(tile_candidates),
                        args.seconds,
                    )
                    upright = candidates[np.abs(candidates[:, 2]) <= 1e-14]
                    depth = cuts[-1]
                    if wall == "left":
                        strip_count = int(np.sum(upright[:, 0] <= depth + 1e-10))
                    elif wall == "right":
                        strip_count = int(
                            np.sum(upright[:, 0] >= target_side - depth - 1e-10)
                        )
                    elif wall == "bottom":
                        strip_count = int(np.sum(upright[:, 1] <= depth + 1e-10))
                    else:
                        strip_count = int(
                            np.sum(upright[:, 1] >= target_side - depth - 1e-10)
                        )
                    if len(tile_selected) > target_report.get(
                        "best_boundary_tile_count", 0
                    ):
                        target_report["best_boundary_tile_count"] = len(
                            tile_selected
                        )
                        target_report["best_upright_strip_count"] = strip_count
                    if count > target_report["best_count"]:
                        target_report["best_count"] = count
                        target_report["best_schedule"] = schedule_index
                        target_report["best_wall"] = wall
                        target_report["best_grid_phase"] = grid_phase
                        target_report["best_candidates"] = len(candidates)
                        target_report["best_conflicts"] = len(conflicts)
                        target_report["best_layers"] = layers
                        target_report["milp_status"] = status
                    target_report["schedule_results"].append(
                        dict(
                            schedule=schedule_index,
                            angles=angles,
                            cuts=cuts,
                            wall=wall,
                            grid_phase=grid_phase,
                            candidates=len(candidates),
                            conflicts=len(conflicts),
                            count=count,
                            boundary_tile_count=len(tile_selected),
                            upright_strip_count=strip_count,
                        )
                    )
                    if count >= n:
                        chosen = candidates[selected]
                        if len(chosen) >= n:
                            candidate_rows = rows_from_points(chosen, n)
                            cert = dual_certify(n, candidate_rows, incumbent)
                            target_report["certification"] = cert
                            if cert["strict_win"]:
                                wins[n] = candidate_rows
                    print(
                        f"n={n} wall={wall} sched={schedule_index} "
                        f"phase={grid_phase[0]:.4g} cand={len(candidates)} "
                        f"count={count} best={target_report['best_count']}",
                        flush=True,
                    )
        report["targets"].append(target_report)
    report["wins"] = {str(n): rows for n, rows in wins.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if wins:
        with (POC / "final.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["id", "x", "y", "deg"])
            for n in sorted(wins):
                for i, row in enumerate(wins[n]):
                    writer.writerow(
                        [
                            f"{n}_{i}",
                            row["x"],
                            row["y"],
                            str(float(row["angle"]) * 180.0 / math.pi),
                        ]
                    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
