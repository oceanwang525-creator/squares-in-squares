#!/usr/bin/env python3
"""Raster probe for completing an n-1 donor inside target n's box."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "poc/baseline/poc/src"))
from squares import core  # noqa: E402


def packing(rows: list[dict]) -> np.ndarray:
    return np.asarray(
        [[float(r["x"]), float(r["y"]), float(r["angle"])] for r in rows],
        dtype=np.float64,
    )


def half_extent(theta: np.ndarray) -> np.ndarray:
    return 0.5 * (np.abs(np.cos(theta)) + np.abs(np.sin(theta)))


def valid_insertions(
    obstacles: np.ndarray,
    candidates: np.ndarray,
    side: float,
) -> np.ndarray:
    """Return candidate indices that fit the target square and every obstacle."""
    hc = half_extent(candidates[:, 2])
    inside = np.flatnonzero(
        (candidates[:, 0] >= hc)
        & (candidates[:, 1] >= hc)
        & (candidates[:, 0] + hc <= side)
        & (candidates[:, 1] + hc <= side)
    )
    if not len(inside):
        return inside

    tree = cKDTree(obstacles[:, :2])
    neigh = tree.query_ball_point(candidates[inside, :2], 1.5)
    ci, oi = [], []
    for local_idx, hits in enumerate(neigh):
        if hits:
            idx = inside[local_idx]
            ci.extend([idx] * len(hits))
            oi.extend(hits)
    if not ci:
        return inside

    ci = np.asarray(ci, dtype=np.int64)
    oi = np.asarray(oi, dtype=np.int64)
    c = candidates[ci]
    o = obstacles[oi]
    dx = o[:, 0] - c[:, 0]
    dy = o[:, 1] - c[:, 1]
    cc, sc = np.cos(c[:, 2]), np.sin(c[:, 2])
    co, so = np.cos(o[:, 2]), np.sin(o[:, 2])
    separated = np.zeros(len(ci), dtype=bool)
    for ax in ((cc, -sc, co, -so), (sc, cc, so, co)):
        projection = dx * ax[0] + dy * ax[1] + dx * ax[2] + dy * ax[3]
        cth = cc * co + sc * so
        sth = cc * so - sc * co
        radius = 0.5 * (1.0 + np.abs(cth) + np.abs(sth))
        separated |= np.abs(projection) >= radius - 1e-12
    blocked = np.unique(ci[~separated])
    return np.setdiff1d(inside, blocked, assume_unique=True)


def candidate_grid(side: float, step: float, angles: tuple[float, ...]) -> np.ndarray:
    coords = np.arange(step * 0.5, side, step)
    xx, yy = np.meshgrid(coords, coords, indexing="ij")
    xy = np.column_stack((xx.ravel(), yy.ravel()))
    return np.vstack([
        np.column_stack((xy, np.full(len(xy), angle)))
        for angle in angles
    ])


def probe(
    incumbent: dict[int, list[dict]],
    n: int,
    step: float,
    shifts: tuple[float, ...] = (0.0, 0.5, 1.0),
) -> dict:
    target = packing(incumbent[n])
    donor = packing(incumbent[n - 1])
    target_side = float(core.certify(n, incumbent[n]))
    donor_side = float(core.certify(n - 1, incumbent[n - 1]))
    slack = target_side - donor_side
    if slack < 0:
        return {"n": n, "status": "donor-larger", "slack": slack}

    angles = tuple(x * np.pi / 180.0 for x in (0.0, 15.0, 30.0, 45.0, 60.0, 75.0))
    candidates_all = candidate_grid(target_side, step, angles)
    checked = 0
    for fx in shifts:
        for fy in shifts:
            shifted = donor.copy()
            shifted[:, 0] += fx * slack
            shifted[:, 1] += fy * slack
            valid = valid_insertions(shifted, candidates_all, target_side)
            checked += len(candidates_all)
            for idx in valid[:8]:
                c = candidates_all[idx]
                rows = [
                    dict(
                        square_id=str(i),
                        x=format(float(r["x"]), ".50g"),
                        y=format(float(r["y"]), ".50g"),
                        angle=format(float(r["angle"]), ".50g"),
                    )
                    for i, r in enumerate(incumbent[n - 1])
                ]
                if fx or fy:
                    rows = [
                        dict(
                            row,
                            x=format(float(row["x"]) + fx * slack, ".50g"),
                            y=format(float(row["y"]) + fy * slack, ".50g"),
                        )
                        for row in rows
                    ]
                rows.append(dict(
                    square_id=str(len(rows)),
                    x=format(float(c[0]), ".50g"),
                    y=format(float(c[1]), ".50g"),
                    angle=format(float(c[2]), ".50g"),
                ))
                try:
                    L = core.certify(n, rows)
                except Exception:
                    continue
                if L < target_side:
                    return {
                        "n": n,
                        "status": "candidate",
                        "donor": n - 1,
                        "shift_fraction": [fx, fy],
                        "x": rows[-1]["x"],
                        "y": rows[-1]["y"],
                        "angle": rows[-1]["angle"],
                        "candidate_L": L,
                        "target_L": format(target_side, ".50g"),
                        "checked": checked,
                        "rows": rows,
                    }
    return {
        "n": n,
        "status": "no-insertion",
        "checked": checked,
        "target_L": format(target_side, ".50g"),
        "donor_L": format(donor_side, ".50g"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, nargs="+", default=[803, 791, 908])
    parser.add_argument("--step", type=float, default=0.1)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "poc/rigid_cluster_crossover/docs/insertion_probe.json",
    )
    args = parser.parse_args()
    incumbent = core.read_submission(ROOT / "poc/pell_corrugation/data/border-scan/merged.csv")
    results = [probe(incumbent, n, args.step) for n in args.n]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(
        [{k: v for k, v in r.items() if k != "rows"} for r in results],
        indent=2,
    ))


if __name__ == "__main__":
    main()
