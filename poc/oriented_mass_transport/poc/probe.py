#!/usr/bin/env python3
"""Joint centre/folded-orientation Sinkhorn transport followed by repair."""
from __future__ import annotations

import argparse
import importlib.util
import io
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[3]
H7_PATH = ROOT / "poc/optimal_transport_basin_transfer/poc/probe.py"
FINAL = ROOT / "poc/oriented_mass_transport/final.csv"
DOCS = ROOT / "poc/oriented_mass_transport/docs"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


h7 = load_module("oriented_base_transport", H7_PATH)
overlap = h7.overlap
union = h7.union
polish = h7.polish


def folded_embedding(theta: np.ndarray) -> np.ndarray:
    return np.column_stack((np.cos(4.0 * theta), np.sin(4.0 * theta)))


def joint_cost(
    source: np.ndarray,
    donor: np.ndarray,
    lambda_angle: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = h7.normalize_centers(source)
    xd = h7.normalize_centers(donor)
    spatial_sq = cdist(xs, xd, metric="sqeuclidean")
    osrc = folded_embedding(source[:, 2])
    odon = folded_embedding(donor[:, 2])
    orientation_sq = np.maximum(
        0.0,
        2.0 - 2.0 * (osrc @ odon.T),
    )
    return np.sqrt(spatial_sq + lambda_angle**2 * orientation_sq), xs, xd


def oriented_transport(
    source: np.ndarray,
    donor: np.ndarray,
    epsilon: float,
    lambda_angle: float,
):
    cost, xs, xd = joint_cost(source, donor, lambda_angle)
    coupling = h7.sinkhorn(cost, epsilon=epsilon)
    weights = coupling / np.maximum(coupling.sum(axis=1, keepdims=True), 1e-300)

    centers = weights @ xd
    donor_z = folded_embedding(donor[:, 2])
    mean_z = weights @ donor_z
    angles = 0.25 * np.arctan2(mean_z[:, 1], mean_z[:, 0])

    spatial_sq = cdist(xs, xd, metric="sqeuclidean")
    center_mass = float((coupling * np.sqrt(spatial_sq)).sum())
    return centers, angles, coupling, float((coupling * cost).sum()), center_mass


def smooth_angle_patches(
    current_angles: np.ndarray,
    mapped_angles: np.ndarray,
    radius: float = 0.09,
) -> np.ndarray:
    """Average mapped angles only among squares in one folded class."""

    folded = np.mod(current_angles, math.pi / 2)
    out = mapped_angles.copy()
    for center in folded:
        delta = h7.wrap_angle(folded - center)
        mask = np.abs(delta) <= radius
        if mask.sum() < 2:
            continue
        z = folded_embedding(mapped_angles[mask]).mean(axis=0)
        out[mask] = 0.25 * np.arctan2(z[1], z[0])
    return out


def probe_target(
    n: int,
    groups: dict[int, np.ndarray],
    seconds: float,
    repair_iterations: int,
    top_starts: int,
) -> dict:
    started = time.monotonic()
    current = groups[n]
    incumbent = h7.side(current)
    donors = [m for m in (n - 3, n - 2, n - 1, n + 1, n + 2, n + 3) if m in groups]
    epsilons = (0.025, 0.04, 0.07)
    lambdas = (0.0, 0.12, 0.25, 0.45, 0.8)
    deltas = (1e-6, 1e-5, 5e-5, 1e-4, 2e-4)
    alphas = (0.2, 0.4, 0.65, 0.9)

    starts = []
    maps = 0
    for donor_n in donors:
        donor = groups[donor_n]
        for epsilon in epsilons:
            for lambda_angle in lambdas:
                mapped_xy, mapped_angles, coupling, cost, center_mass = oriented_transport(
                    current, donor, epsilon, lambda_angle
                )
                maps += 1
                if lambda_angle > 0:
                    mapped_angles = smooth_angle_patches(current[:, 2], mapped_angles)
                for delta in deltas:
                    target_side = incumbent - delta
                    for alpha in alphas:
                        q0 = h7.place_centered(
                            current, mapped_xy, mapped_angles, target_side, alpha
                        )
                        initial = float(overlap.penalty(q0, target_side, grad=False))
                        starts.append(
                            dict(
                                initial_penalty=initial,
                                donor=donor_n,
                                epsilon=epsilon,
                                lambda_angle=lambda_angle,
                                delta=delta,
                                alpha=alpha,
                                target_side=target_side,
                                mass=float(coupling.sum()),
                                transport_cost=cost,
                                center_mass=center_mass,
                                q0=q0,
                            )
                        )
                if time.monotonic() - started >= seconds:
                    break
            if time.monotonic() - started >= seconds:
                break
        if time.monotonic() - started >= seconds:
            break

    starts.sort(key=lambda item: item["initial_penalty"])
    attempts = []
    best = None
    for item in starts[:top_starts]:
        q, penalty, violation = h7.repair(
            item["q0"],
            item["target_side"],
            iterations=repair_iterations,
        )
        row = {k: v for k, v in item.items() if k != "q0"}
        row.update(final_penalty=penalty, max_violation=violation, q=q)
        attempts.append(row)
        if best is None or (penalty, violation) < (
            best["final_penalty"],
            best["max_violation"],
        ):
            best = row

    attempts.sort(key=lambda row: (row["final_penalty"], row["max_violation"]))
    realized = []
    for row in attempts[:12]:
        q = row["q"]
        centers, dilation = polish.minimal_feasible_dilation(
            q[:, :2], q[:, 2], tau=1e-10
        )
        qd = np.column_stack((centers, q[:, 2]))
        dside = polish.exact_bbox_side(qd[:, :2], qd[:, 2])
        ov, wall, _ = overlap.max_violation(qd, dside)
        row["dilation"] = float(dilation)
        row["dilated_side"] = float(dside)
        row["dilated_violation"] = float(max(ov, wall))
        if max(ov, wall) <= 1e-8 and dside < incumbent:
            realized.append((dside, qd))

    realized.sort(key=lambda pair: pair[0])
    certified = None
    for cside, q in realized:
        certified = h7.certify_candidate(q, n, incumbent)
        if certified is not None:
            certified["realized_side"] = cside
            break

    return dict(
        n=n,
        incumbent_side=incumbent,
        maps=maps,
        screened_starts=len(starts),
        repaired_starts=len(attempts),
        best=None
        if best is None
        else {k: v for k, v in best.items() if k != "q"},
        attempts=[
            {k: v for k, v in row.items() if k != "q"}
            for row in attempts[:12]
        ],
        certified=certified,
        elapsed_seconds=time.monotonic() - started,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", default="803,328,908")
    parser.add_argument("--seconds-per-target", type=float, default=120.0)
    parser.add_argument("--repair-iterations", type=int, default=3000)
    parser.add_argument("--top-starts", type=int, default=18)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--out", default=str(DOCS / "results.json"))
    args = parser.parse_args()

    targets = [int(item) for item in args.targets.split(",") if item]
    groups = h7.load_groups()
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results = list(
                pool.map(
                    probe_target,
                    targets,
                    [groups] * len(targets),
                    [args.seconds_per_target] * len(targets),
                    [args.repair_iterations] * len(targets),
                    [args.top_starts] * len(targets),
                )
            )
    else:
        results = [
            probe_target(
                n,
                groups,
                args.seconds_per_target,
                args.repair_iterations,
                args.top_starts,
            )
            for n in targets
        ]

    wins = {
        str(result["n"]): result["certified"]
        for result in results
        if result["certified"] is not None
    }
    existing = union.parse_text(FINAL.read_text()) if FINAL.exists() else {}
    for result in results:
        item = result["certified"]
        if item is None:
            continue
        parsed = union.parse_text(item["final_text"])
        existing[result["n"]] = parsed[result["n"]]
    if not existing:
        FINAL.write_text("id,x,y,deg\n", encoding="utf-8")
    else:
        FINAL.write_text(union.final_text(existing), encoding="utf-8")

    payload = dict(
        method="joint centre/folded-orientation Sinkhorn plus patch smoothing",
        targets=targets,
        results=[
            {k: v for k, v in result.items() if k != "certified"}
            for result in results
        ],
        wins=wins,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
