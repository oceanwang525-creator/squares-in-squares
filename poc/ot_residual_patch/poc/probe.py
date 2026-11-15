#!/usr/bin/env python3
"""Patch only high-stress neighborhoods of H7's near-feasible transport seeds."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[3]
H7_PATH = ROOT / "poc/optimal_transport_basin_transfer/poc/probe.py"
BASELINE = ROOT / "poc/baseline/poc/src"
if str(BASELINE) not in sys.path:
    sys.path.insert(0, str(BASELINE))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


h7 = load_module("ot_residual_h7", H7_PATH)
overlap = h7.overlap

SEEDS = {
    328: {"donor": 326, "epsilon": 0.025, "delta": 1e-5, "alpha": 0.2},
    365: {"donor": 364, "epsilon": 0.025, "delta": 1e-6, "alpha": 0.2},
    367: {"donor": 366, "epsilon": 0.025, "delta": 1e-8, "alpha": 0.2},
}


def contact_state(q: np.ndarray, target_side: float):
    pairs = overlap.pairs(q)
    depth = overlap.pair_depth(q, pairs)
    wall, _ = overlap.wall_violation(q, target_side)
    mask = depth > 1e-12
    return pairs[mask], depth[mask], wall


def components(pairs: np.ndarray, n: int) -> list[list[int]]:
    adjacency = [set() for _ in range(n)]
    for i, j in pairs:
        adjacency[int(i)].add(int(j))
        adjacency[int(j)].add(int(i))
    seen: set[int] = set()
    result = []
    for root in range(n):
        if root in seen or not adjacency[root]:
            continue
        stack = [root]
        seen.add(root)
        component = []
        while stack:
            node = stack.pop()
            component.append(node)
            for other in adjacency[node]:
                if other not in seen:
                    seen.add(other)
                    stack.append(other)
        result.append(component)
    result.sort(key=len, reverse=True)
    return result


def patch_centers(
    q: np.ndarray,
    pairs: np.ndarray,
    depth: np.ndarray,
    target_side: float,
    radius: float,
) -> list[np.ndarray]:
    """Return separated patches around the worst-stress contact pairs."""
    if not len(pairs):
        return []
    order = np.argsort(depth)[::-1]
    centers = []
    for edge in order:
        mid = 0.5 * (q[pairs[edge, 0], :2] + q[pairs[edge, 1], :2])
        if any(np.linalg.norm(mid - old) < radius * 1.25 for old in centers):
            continue
        centers.append(mid)
        if len(centers) >= 8:
            break
    return centers


def patch_once(
    q: np.ndarray,
    center: np.ndarray,
    radius: float,
    target_side: float,
    iterations: int,
) -> tuple[np.ndarray, float]:
    active = np.flatnonzero(np.linalg.norm(q[:, :2] - center, axis=1) <= radius)
    if len(active) < 2:
        return q, float(overlap.penalty(q, target_side, grad=False))
    columns = np.concatenate((3 * active, 3 * active + 1, 3 * active + 2))
    columns.sort()
    base = q.ravel()

    def objective(changes: np.ndarray):
        candidate = base.copy()
        candidate[columns] += changes
        trial = candidate.reshape(-1, 3)
        _, grad = overlap.penalty(trial, target_side, grad=True)
        return float(overlap.penalty(trial, target_side, grad=False)), grad.ravel()[columns]

    result = minimize(
        objective,
        np.zeros(len(columns)),
        method="L-BFGS-B",
        jac=True,
        options={
            "maxiter": iterations,
            "maxls": 30,
            "ftol": 1e-15,
            "gtol": 1e-12,
        },
    )
    out = base.copy()
    out[columns] += result.x
    return out.reshape(-1, 3), float(result.fun)


def max_state(q: np.ndarray, target_side: float):
    pairs, depth, wall = contact_state(q, target_side)
    return (
        float(depth.max()) if len(depth) else 0.0,
        float(wall.max()),
        int(len(depth)),
    )


def probe_target(n: int, groups, seconds: float, iterations: int) -> dict:
    seed = SEEDS[n]
    current = groups[n]
    incumbent = h7.side(current)
    target_side = incumbent - seed["delta"]
    xy, angles, _, _ = h7.transport_barycenter(
        current, groups[seed["donor"]], seed["epsilon"]
    )
    q0 = h7.place_centered(
        current, xy, angles, target_side, seed["alpha"]
    )
    q, _, initial_global = h7.repair(q0, target_side, iterations=5000)

    pairs, depth, _ = contact_state(q, target_side)
    comps = components(pairs, n)
    initial_max, initial_wall, initial_pairs = max_state(q, target_side)
    patches = []
    history = []

    for radius in (1.25, 2.0, 3.2):
        centers = patch_centers(q, pairs, depth, target_side, radius)
        for center in centers:
            previous_max, previous_wall, previous_pairs = max_state(q, target_side)
            trial, penalty = patch_once(
                q, center, radius, target_side, iterations
            )
            trial_max, trial_wall, trial_pairs = max_state(trial, target_side)
            improvement = max(previous_max, previous_wall) - max(trial_max, trial_wall)
            history.append(
                {
                    "radius": radius,
                    "center": center.tolist(),
                    "before_max": previous_max,
                    "after_max": trial_max,
                    "before_wall": previous_wall,
                    "after_wall": trial_wall,
                    "before_pairs": previous_pairs,
                    "after_pairs": trial_pairs,
                    "penalty": penalty,
                    "improvement": improvement,
                }
            )
            if improvement > 1e-13:
                q = trial
                pairs, depth, _ = contact_state(q, target_side)
        patches.append(
            {
                "radius": radius,
                "centers": len(centers),
                "max_after": list(max_state(q, target_side)),
            }
        )

    final_max, final_wall, final_pairs = max_state(q, target_side)
    certified = None
    if max(final_max, final_wall) <= 1e-10:
        centers, dilation = h7.polish.minimal_feasible_dilation(
            q[:, :2], q[:, 2], tau=1e-10
        )
        q2 = np.column_stack([centers, q[:, 2]])
        side2 = h7.polish.exact_bbox_side(q2[:, :2], q2[:, 2])
        ov2, wall2, _ = overlap.max_violation(q2, side2)
        if max(ov2, wall2) <= 1e-8 and side2 < incumbent:
            centers_lp, side_lp = h7.polish.fixed_angle_lp_polish(
                q2[:, :2], q2[:, 2], rounds=10, time_limit=30.0
            )
            q_lp = np.column_stack([centers_lp, q2[:, 2]])
            ov_lp, wall_lp, _ = overlap.max_violation(q_lp, side_lp)
            if max(ov_lp, wall_lp) <= 1e-8:
                certified = h7.certify_candidate(q_lp, n, incumbent)

    return {
        "n": n,
        "seed": seed,
        "incumbent_side": incumbent,
        "target_side": target_side,
        "initial_global_violation": initial_global,
        "initial": {
            "max_overlap": initial_max,
            "max_wall": initial_wall,
            "pairs": initial_pairs,
            "components": [len(x) for x in comps[:8]],
        },
        "patches": patches,
        "history": history,
        "final": {
            "max_overlap": final_max,
            "max_wall": final_wall,
            "pairs": final_pairs,
        },
        "certified": certified,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", default="328,365,367")
    parser.add_argument("--seconds", type=float, default=840.0)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "poc/ot_residual_patch/docs/results.json",
    )
    args = parser.parse_args()
    groups = h7.load_groups()
    results = []
    for target in [int(x) for x in args.targets.split(",") if x]:
        result = probe_target(
            target, groups, min(args.seconds, 840.0), args.iterations
        )
        results.append(result)
        print(json.dumps(result, indent=2), flush=True)
        if args.seconds <= 0:
            break

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"results": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
