"""Allow the strip splice's closest seams to choose a side per square."""
from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
from itertools import product
import json
import math
from pathlib import Path
from time import perf_counter
import warnings

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from splice import (
    DATA, EPS, INCUMBENT, STRICT, Cut, Trial, certify_baseline, certify_trial,
    crossing_collisions, extents, length, load_source, trim_indices, write_submission,
)


def placement_choices(geometry, trial, shoulder):
    original_lo, original_hi = extents(geometry)
    box_lo = original_lo.min(0)
    widths = np.array([trial.x_width, trial.y_width])
    box_hi = original_hi.max(0) - widths
    starts = np.array([trial.x.start, trial.y.start])
    half = starts + widths / 2
    poses, origins, sides = [], [], []
    for i, row in enumerate(geometry):
        choices = [
            (False, True) if start-shoulder <= coordinate <= start+width+shoulder
            else (bool(coordinate > middle),)
            for coordinate, start, width, middle in zip(row[:2], starts, widths, half)
        ]
        for flags in product(*choices):
            delta = widths * flags
            if np.any(original_lo[i] - delta < box_lo - EPS):
                continue
            if np.any(original_hi[i] - delta > box_hi + EPS):
                continue
            poses.append([*(row[:2] - delta), row[2]])
            origins.append(i)
            sides.append(flags)
    return np.array(poses), np.array(origins, dtype=int), np.array(sides, dtype=bool)


def assign(geometry, trial, shoulder, seconds):
    poses, origins, shifts = placement_choices(geometry, trial, shoulder)
    if not len(poses):
        return None, None, None, {"capacity": 0, "upper_bound": 0, "status": 0}
    edges = crossing_collisions(poses, *shifts.T)
    by_origin = {}
    for i, origin in enumerate(origins):
        by_origin.setdefault(origin, []).append(i)
    constraints = [list(edge) for edge in edges]
    constraints.extend(indices for indices in by_origin.values() if len(indices) > 1)
    if not constraints:
        keep = np.arange(len(poses))
        return poses, origins, shifts, {"capacity": len(keep), "upper_bound": len(keep),
                                         "status": 0, "keep": keep}

    cols = np.concatenate(constraints)
    active = np.unique(cols)
    rows = np.repeat(np.arange(len(constraints)), [len(group) for group in constraints])
    matrix = coo_matrix((np.ones(len(rows)), (rows, np.searchsorted(active, cols))),
                        shape=(len(constraints), len(active))).tocsc()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Unrecognized options detected")
        result = milp(-np.ones(len(active)), integrality=np.ones(len(active)),
                      bounds=Bounds(0, 1), constraints=LinearConstraint(matrix, 0, 1),
                      options={"time_limit": seconds, "threads": 1})
    if result.x is None:
        return poses, origins, shifts, {"capacity": None, "upper_bound": None,
                                        "status": result.status}
    fixed = len(poses) - len(active)
    keep = np.ones(len(poses), dtype=bool)
    keep[active] = result.x > 0.5
    selected = np.flatnonzero(keep)
    if len(set(origins[selected])) != len(selected) or np.any(keep[edges[:, 0]] & keep[edges[:, 1]]):
        raise RuntimeError("Assigned placements collide or reuse a donor square")
    bound = fixed - result.mip_dual_bound
    return poses, origins, shifts, {
        "capacity": len(selected), "upper_bound": math.floor(bound + 1e-6),
        "status": result.status, "keep": selected,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan", type=Path, default=DATA / "scan.json")
    parser.add_argument("--cases", type=int, default=20)
    parser.add_argument("--shoulder", type=float, default=0.6)
    parser.add_argument("--seconds-per-case", type=float, default=2.0)
    parser.add_argument("--output", type=Path, default=DATA)
    args = parser.parse_args()
    started = perf_counter()
    scan = json.loads(args.scan.read_text(encoding="utf-8"))
    if hashlib.sha256(INCUMBENT.read_bytes()).hexdigest() != scan["incumbent_sha256"]:
        raise ValueError("Scan and incumbent refer to different donor bytes")
    source = load_source(INCUMBENT)
    wins, details, baselines = {}, [], {}
    for row in scan["best_capacity_by_target"][:args.cases]:
        n, donor = row["target"], row["donor"]
        x, y = (Cut(start, 0, 0) for start in row["start"])
        trial = Trial(donor, n, row["nominal"], *row["width"], x, y)
        geometry = np.array(source[donor], dtype=float)
        poses, origins, flags, result = assign(geometry, trial, args.shoulder, args.seconds_per_case)
        selected = result.pop("keep", None)
        detail = dict(row, assignment=result)
        details.append(detail)
        if selected is None or len(selected) < n:
            continue
        for target in range(n, min(len(selected), donor-1) + 1):
            chosen = trim_indices(selected, poses, target)
            if length(poses[chosen]) > length(np.array(source[target], dtype=float)) - STRICT:
                continue
            if target not in baselines:
                baselines[target] = certify_baseline(target, source[target])
            right = np.zeros(donor, dtype=bool)
            upper = np.zeros(donor, dtype=bool)
            indices = origins[chosen]
            right[indices], upper[indices] = flags[chosen].T
            certified = certify_trial(target, source[donor], trial, indices, right, upper, baselines[target])
            if certified is None:
                continue
            rows, exact, transport, slack = certified
            if target not in wins or Decimal(exact) < wins[target][1]:
                wins[target] = (rows, Decimal(exact))
                detail.setdefault("wins", {})[target] = dict(
                    literal=exact, pandas=transport, dilation=slack,
                    literal_baseline=str(baselines[target][0]),
                    pandas_baseline=str(baselines[target][1]),
                )
    args.output.mkdir(parents=True, exist_ok=True)
    result = dict(incumbent_sha256=scan["incumbent_sha256"], tested=len(details), wins=sorted(wins),
                  shoulder=args.shoulder, seconds_per_case=args.seconds_per_case,
                  results=details, seconds=round(perf_counter()-started, 3))
    (args.output / "assignment.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if wins:
        write_submission(args.output / "assignment-candidates.csv", {n: row[0] for n, row in wins.items()})
    print(f"{len(details)} seam assignment cases, {len(wins)} wins, {result['seconds']} seconds", flush=True)


if __name__ == "__main__":
    main()
