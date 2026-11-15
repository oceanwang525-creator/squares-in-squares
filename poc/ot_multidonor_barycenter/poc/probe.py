#!/usr/bin/env python3
"""Screen multi-donor Sinkhorn barycenters for unit-square packings."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import io
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "poc/pell_corrugation/data/border-scan/merged.csv"
OUTPUT_DIR = ROOT / "poc/ot_multidonor_barycenter"
FINAL = OUTPUT_DIR / "final.csv"
DOCS = OUTPUT_DIR / "docs"
BASELINE = ROOT / "poc/baseline/poc/src"
if str(BASELINE) not in sys.path:
    sys.path.insert(0, str(BASELINE))

from squares import core  # noqa: E402


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


overlap = _load_module("multi_ot_overlap", ROOT / "poc/k2p10_fill2/poc/a_overlap.py")
union = _load_module("multi_ot_union", ROOT / "poc/auto_union/poc/union.py")
polish = _load_module(
    "multi_ot_polish", ROOT / "poc/swarm_inflation/poc/polish_and_certify.py"
)


def extents(theta: np.ndarray) -> np.ndarray:
    return 0.5 * (np.abs(np.cos(theta)) + np.abs(np.sin(theta)))


def side(q: np.ndarray) -> float:
    e = extents(q[:, 2])
    return float(
        max(
            (q[:, 0] + e).max() - (q[:, 0] - e).min(),
            (q[:, 1] + e).max() - (q[:, 1] - e).min(),
        )
    )


def load_groups() -> dict[int, np.ndarray]:
    data: dict[int, list[tuple[float, float, float]]] = {}
    with BASE.open(newline="") as handle:
        for row in csv.DictReader(handle):
            data.setdefault(int(row["n"]), []).append(
                (float(row["x"]), float(row["y"]), float(row["angle"]))
            )
    return {n: np.asarray(rows, dtype=np.float64) for n, rows in data.items()}


def normalize_centers(q: np.ndarray) -> np.ndarray:
    e = extents(q[:, 2])
    lo = (q[:, :2] - e[:, None]).min(axis=0)
    hi = (q[:, :2] + e[:, None]).max(axis=0)
    span = hi - lo
    return (q[:, :2] - lo) / span


def wrap_angle(delta: np.ndarray | float) -> np.ndarray | float:
    period = math.pi / 2
    return (delta + period / 2) % period - period / 2


def sinkhorn(cost: np.ndarray, epsilon: float, rounds: int = 70) -> np.ndarray:
    scaled = cost / max(epsilon, 1e-12)
    scaled -= scaled.min()
    kernel = np.exp(-scaled)
    n, m = kernel.shape
    row_mass = np.full(n, 1.0 / n)
    col_mass = np.full(m, 1.0 / m)
    u = np.ones(n)
    v = np.ones(m)
    for _ in range(rounds):
        u = row_mass / np.maximum(kernel @ v, 1e-300)
        v = col_mass / np.maximum(kernel.T @ u, 1e-300)
    return u[:, None] * kernel * v[None, :]


def transport_map(
    source: np.ndarray, donor: np.ndarray, epsilon: float
) -> tuple[np.ndarray, np.ndarray, float]:
    xs = normalize_centers(source)
    xd = normalize_centers(donor)
    spatial = cdist(xs, xd, metric="euclidean")
    angle_delta = wrap_angle(source[:, None, 2] - donor[None, :, 2])
    angle_delta = np.asarray(angle_delta, dtype=np.float64)
    cost = spatial + 0.08 * np.abs(angle_delta)
    coupling = sinkhorn(cost, epsilon)
    weights = coupling / np.maximum(coupling.sum(axis=1, keepdims=True), 1e-300)
    centers = weights @ xd
    angles = donor[:, 2][None, :] + np.asarray(
        wrap_angle(donor[:, 2][None, :] - source[:, 2][:, None]),
        dtype=np.float64,
    )
    cos_mean = (weights * np.cos(2 * angles)).sum(axis=1)
    sin_mean = (weights * np.sin(2 * angles)).sum(axis=1)
    mapped_angles = 0.5 * np.arctan2(sin_mean, cos_mean)
    value = float((coupling * cost).sum())
    return centers, mapped_angles, value


def place_centered(
    current: np.ndarray,
    mapped_xy: np.ndarray,
    mapped_angles: np.ndarray,
    target_side: float,
    alpha: float,
) -> np.ndarray:
    current_xy = normalize_centers(current)
    desired_xy = (1.0 - alpha) * current_xy + alpha * mapped_xy
    desired_xy *= target_side
    target_angle = current[:, 2] + alpha * np.asarray(
        wrap_angle(mapped_angles - current[:, 2]), dtype=np.float64
    )
    q = np.column_stack([desired_xy, target_angle])
    e = extents(q[:, 2])
    lo = (q[:, :2] - e[:, None]).min(axis=0)
    hi = (q[:, :2] + e[:, None]).max(axis=0)
    q[:, :2] += 0.5 * (target_side - (lo + hi))
    return q


def repair(
    q0: np.ndarray, target_side: float, iterations: int
) -> tuple[np.ndarray, float, float]:
    def objective(flat: np.ndarray):
        q = flat.reshape(-1, 3)
        return overlap.penalty(q, target_side, grad=True)

    result = minimize(
        objective,
        q0.ravel(),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": iterations, "maxls": 30, "ftol": 1e-14, "gtol": 1e-10},
    )
    q = result.x.reshape(-1, 3)
    max_overlap, max_wall, _ = overlap.max_violation(q, target_side)
    return q, float(result.fun), max(max_overlap, max_wall)


def to_groups(q: np.ndarray, n: int) -> dict[int, list[dict[str, str]]]:
    return {
        n: [
            {
                "square_id": str(i),
                "x": format(float(x), ".17g"),
                "y": format(float(y), ".17g"),
                "angle": format(float(a), ".17g"),
            }
            for i, (x, y, a) in enumerate(q)
        ]
    }


def certify_candidate(q: np.ndarray, n: int, incumbent_side: float):
    literal_text = union.final_text(to_groups(q, n))
    literal = union.parse_text(literal_text)
    pandas_text = pd.read_csv(io.StringIO(literal_text)).to_csv(index=False)
    transported = union.parse_text(pandas_text)
    if len(literal[n]) != n or len(transported[n]) != n:
        raise ValueError("incomplete group")
    literal_side = core.certify(n, literal[n])
    pandas_side = core.certify(n, transported[n])
    if float(literal_side) >= incumbent_side or float(pandas_side) >= incumbent_side:
        return None
    return {
        "literal_side": literal_side,
        "pandas_side": pandas_side,
        "final_text": literal_text,
    }


def schedules(offsets: list[int]) -> list[tuple[tuple[int, float], ...]]:
    result: list[tuple[tuple[int, float], ...]] = []
    for left in offsets:
        for right in offsets:
            if left < right:
                result.extend(
                    [
                        ((left, 1.0), (right, 1.0)),
                        ((left, 2.0), (right, 1.0)),
                        ((left, 1.0), (right, 2.0)),
                    ]
                )
    triples = [
        (-3, -1, 1),
        (-2, -1, 1),
        (-1, 1, 2),
        (-1, 1, 3),
        (-2, 1, 2),
        (-3, -1, 2),
        (-2, 1, 3),
    ]
    for a, b, c in triples:
        if all(x in offsets for x in (a, b, c)):
            result.extend(
                [
                    ((a, 1.0), (b, 1.0), (c, 1.0)),
                    ((a, 2.0), (b, 1.0), (c, 1.0)),
                    ((a, 1.0), (b, 1.0), (c, 2.0)),
                ]
            )
    return result


def probe_target(
    n: int,
    groups: dict[int, np.ndarray],
    seconds: float,
    repair_iterations: int,
    top_starts: int,
) -> dict:
    started = time.monotonic()
    current = groups[n]
    incumbent_side = side(current)
    offsets = [k for k in (-4, -3, -2, -1, 1, 2, 3, 4) if n + k in groups]
    maps: dict[tuple[int, float], tuple[np.ndarray, np.ndarray, float]] = {}
    for offset in offsets:
        for epsilon in (0.025, 0.06):
            if time.monotonic() - started >= seconds:
                break
            maps[(offset, epsilon)] = transport_map(
                current, groups[n + offset], epsilon
            )
        if time.monotonic() - started >= seconds:
            break

    screened = []
    valid_schedules = schedules(offsets)
    for epsilon in (0.025, 0.06):
        for schedule in valid_schedules:
            if not all((offset, epsilon) in maps for offset, _ in schedule):
                continue
            total = sum(weight for _, weight in schedule)
            mixed_xy = np.zeros((n, 2), dtype=np.float64)
            cos_sum = np.zeros(n, dtype=np.float64)
            sin_sum = np.zeros(n, dtype=np.float64)
            transport_value = 0.0
            for offset, weight in schedule:
                mapped_xy, mapped_angles, value = maps[(offset, epsilon)]
                w = weight / total
                mixed_xy += w * mapped_xy
                cos_sum += w * np.cos(2 * mapped_angles)
                sin_sum += w * np.sin(2 * mapped_angles)
                transport_value += w * value
            mixed_angles = 0.5 * np.arctan2(sin_sum, cos_sum)
            for delta in (1e-6, 1e-5, 1e-4, 2e-4):
                target_side = incumbent_side - delta
                for alpha in (0.2, 0.4, 0.65):
                    q0 = place_centered(
                        current, mixed_xy, mixed_angles, target_side, alpha
                    )
                    initial_penalty = float(
                        overlap.penalty(q0, target_side, grad=False)
                    )
                    screened.append(
                        (
                            initial_penalty,
                            epsilon,
                            schedule,
                            delta,
                            alpha,
                            target_side,
                            q0,
                            transport_value,
                        )
                    )

    screened.sort(key=lambda row: row[0])
    repaired = []
    for row in screened[:top_starts]:
        if time.monotonic() - started >= seconds * 2:
            break
        initial_penalty, epsilon, schedule, delta, alpha, target_side, q0, value = row
        q, penalty, violation = repair(q0, target_side, repair_iterations)
        repaired.append(
            {
                "epsilon": epsilon,
                "schedule": [list(item) for item in schedule],
                "delta": delta,
                "alpha": alpha,
                "initial_penalty": initial_penalty,
                "transport_value": value,
                "final_penalty": penalty,
                "max_violation": violation,
                "_q": q,
            }
        )
    repaired.sort(key=lambda row: (row["final_penalty"], row["max_violation"]))

    realized = []
    for entry in repaired[:16]:
        q = entry["_q"]
        centers, dilation = polish.minimal_feasible_dilation(
            q[:, :2], q[:, 2], tau=1e-10
        )
        q_dilated = np.column_stack([centers, q[:, 2]])
        dilation_side = polish.exact_bbox_side(q_dilated[:, :2], q_dilated[:, 2])
        max_overlap, max_wall, _ = overlap.max_violation(q_dilated, dilation_side)
        entry["dilation"] = dilation
        entry["dilated_side"] = dilation_side
        if max(max_overlap, max_wall) <= 1e-8 and dilation_side < incumbent_side:
            realized.append((dilation_side, q_dilated))
            centers_lp, lp_side = polish.fixed_angle_lp_polish(
                q_dilated[:, :2],
                q_dilated[:, 2],
                rounds=10,
                time_limit=30.0,
            )
            q_lp = np.column_stack([centers_lp, q_dilated[:, 2]])
            lp_ov, lp_wall, _ = overlap.max_violation(q_lp, lp_side)
            if max(lp_ov, lp_wall) <= 1e-8:
                realized.append((lp_side, q_lp))
                entry["fixed_angle_lp_side"] = lp_side
        # If global dilation is over budget but residual is tiny, retain it as
        # a diagnostic for the separate residual-patch hypothesis.
        if entry["max_violation"] <= 1e-4:
            centers_small, dilation_small = polish.minimal_feasible_dilation(
                q[:, :2], q[:, 2], tau=1e-10
            )
            q_small = np.column_stack([centers_small, q[:, 2]])
            entry["small_dilation"] = dilation_small
            entry["small_dilation_side"] = polish.exact_bbox_side(
                q_small[:, :2], q_small[:, 2]
            )

    realized.sort(key=lambda item: item[0])
    certified = None
    for candidate_side, candidate in realized:
        if candidate_side >= incumbent_side:
            continue
        certified = certify_candidate(candidate, n, incumbent_side)
        if certified:
            break

    return {
        "n": n,
        "incumbent_side": incumbent_side,
        "donor_offsets": offsets,
        "maps": len(maps),
        "weight_schedules": len(valid_schedules),
        "screened_starts": len(screened),
        "repaired_starts": len(repaired),
        "attempts": [
            {key: value for key, value in entry.items() if key != "_q"}
            for entry in repaired[:16]
        ],
        "certified": certified,
        "elapsed_seconds": time.monotonic() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", default="803,367,328")
    parser.add_argument("--seconds-per-target", type=float, default=120.0)
    parser.add_argument("--repair-iterations", type=int, default=5000)
    parser.add_argument("--top-starts", type=int, default=24)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--out", default=str(DOCS / "results.json"))
    args = parser.parse_args()
    targets = [int(item) for item in args.targets.split(",") if item]
    groups = load_groups()
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
        result["n"]: result["certified"]
        for result in results
        if result["certified"] is not None
    }
    existing = union.parse_text(FINAL.read_text(encoding="utf-8"))
    for n, item in wins.items():
        existing[n] = union.parse_text(item["final_text"])[n]
    FINAL.write_text(union.final_text(existing), encoding="utf-8")

    payload = {
        "method": "multi-donor Sinkhorn mass barycenter plus fixed-side repair",
        "targets": targets,
        "results": [
            {key: value for key, value in result.items() if key != "certified"}
            for result in results
        ],
        "wins": {
            str(n): {
                "literal_side": item["literal_side"],
                "pandas_side": item["pandas_side"],
            }
            for n, item in wins.items()
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
