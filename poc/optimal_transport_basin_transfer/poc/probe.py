#!/usr/bin/env python3
"""Screen nonlocal center/angle transport fields for square-packings."""
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
FINAL = ROOT / "poc/optimal_transport_basin_transfer/final.csv"
DOCS = ROOT / "poc/optimal_transport_basin_transfer/docs"
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


overlap = _load_module("transport_overlap", ROOT / "poc/k2p10_fill2/poc/a_overlap.py")
union = _load_module("transport_union", ROOT / "poc/auto_union/poc/union.py")
polish = _load_module(
    "transport_polish", ROOT / "poc/swarm_inflation/poc/polish_and_certify.py"
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
    data = {}
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


def wrap_angle(delta: np.ndarray) -> np.ndarray:
    period = math.pi / 2
    return (delta + period / 2) % period - period / 2


def sinkhorn(cost: np.ndarray, epsilon: float, rounds: int = 80) -> np.ndarray:
    """Small stable balanced Sinkhorn solver for equal/unequal masses."""
    scaled = cost / max(epsilon, 1e-12)
    scaled -= scaled.min()
    kernel = np.exp(-scaled)
    n, m = kernel.shape
    a = np.full(n, 1.0 / n)
    b = np.full(m, 1.0 / m)
    u = np.ones(n)
    v = np.ones(m)
    for _ in range(rounds):
        u = a / np.maximum(kernel @ v, 1e-300)
        v = b / np.maximum(kernel.T @ u, 1e-300)
    return u[:, None] * kernel * v[None, :]


def transport_barycenter(
    source: np.ndarray,
    donor: np.ndarray,
    epsilon: float,
    angle_weight: float = 0.08,
):
    xs = normalize_centers(source)
    xd = normalize_centers(donor)
    spatial = cdist(xs, xd, metric="euclidean")
    angle_delta = wrap_angle(source[:, None, 2] - donor[None, :, 2])
    cost = spatial + angle_weight * np.abs(angle_delta)
    coupling = sinkhorn(cost, epsilon=epsilon)
    weights = coupling / np.maximum(coupling.sum(axis=1, keepdims=True), 1e-300)
    centers = weights @ xd
    cos_mean = weights @ np.cos(2 * donor[:, 2])
    sin_mean = weights @ np.sin(2 * donor[:, 2])
    angles = 0.5 * np.arctan2(sin_mean, cos_mean)
    return centers, angles, coupling, cost


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
    target_angle = current[:, 2] + alpha * wrap_angle(mapped_angles - current[:, 2])
    q = np.column_stack([desired_xy, target_angle])
    e = extents(q[:, 2])
    lo = (q[:, :2] - e[:, None]).min(axis=0)
    hi = (q[:, :2] + e[:, None]).max(axis=0)
    q[:, :2] += 0.5 * (target_side - (lo + hi))
    return q


def repair(q0: np.ndarray, target_side: float, iterations: int) -> tuple[np.ndarray, float, float]:
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
    groups = to_groups(q, n)
    literal_text = union.final_text(groups)
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


def probe_target(
    n: int,
    groups: dict[int, np.ndarray],
    seconds: float,
    repair_iterations: int = 900,
    top_starts: int = 18,
    donor_offsets: tuple[int, ...] = (-2, -1, 1, 2),
    epsilons: tuple[float, ...] = (0.025, 0.06),
    deltas: tuple[float, ...] = (1e-8, 1e-7, 1e-6, 1e-5, 1e-4),
    alphas: tuple[float, ...] = (0.2, 0.4, 0.65, 0.9),
    angle_weight: float = 0.08,
    include_same_d: bool = False,
) -> dict:
    started = time.monotonic()
    current = groups[n]
    incumbent_side = side(current)
    donors = [n + offset for offset in donor_offsets if n + offset in groups]
    if include_same_d:
        defect = n - math.isqrt(n) ** 2
        same_d = [
            k * k + defect
            for k in range(max(1, math.isqrt(n) - 3), math.isqrt(n) + 4)
            if 1 <= k * k + defect <= 1000 and k * k + defect != n
        ]
        donors.extend(same_d)
    donors = sorted(set(donors))
    # Begin at a barely perceptible reduction so a genuinely more flexible
    # transport topology can be separated from a mere penalty-margin failure.
    screened = []
    transports = 0
    for donor_n in donors:
        donor = groups[donor_n]
        for epsilon in epsilons:
            mapped_xy, mapped_angles, coupling, cost = transport_barycenter(
                current, donor, epsilon, angle_weight
            )
            transports += 1
            transport_mass = float(coupling.sum())
            transport_cost = float((coupling * cost).sum())
            for delta in deltas:
                target_side = incumbent_side - delta
                for alpha in alphas:
                    q0 = place_centered(
                        current, mapped_xy, mapped_angles, target_side, alpha
                    )
                    initial_penalty = float(overlap.penalty(q0, target_side, grad=False))
                    screened.append(
                        (
                            initial_penalty,
                            donor_n,
                            epsilon,
                            delta,
                            alpha,
                            target_side,
                            q0,
                            transport_mass,
                            transport_cost,
                        )
                    )
            if time.monotonic() - started >= seconds:
                break
        if time.monotonic() - started >= seconds:
            break

    screened.sort(key=lambda row: row[0])
    best = None
    attempts = []
    for row in screened[:top_starts]:
        if time.monotonic() - started >= seconds * 2:
            break
        initial_penalty, donor_n, epsilon, delta, alpha, target_side, q0, mass, cost = row
        q, penalty, violation = repair(q0, target_side, iterations=repair_iterations)
        attempts.append(
            {
                "donor": donor_n,
                "epsilon": epsilon,
                "delta": delta,
                "alpha": alpha,
                "initial_penalty": initial_penalty,
                "final_penalty": penalty,
                "max_violation": violation,
                "_q": q,
            }
        )
        if best is None or (penalty, violation) < (best["penalty"], best["violation"]):
            best = {
                "donor": donor_n,
                "epsilon": epsilon,
                "delta": delta,
                "alpha": alpha,
                "target_side": target_side,
                "penalty": penalty,
                "violation": violation,
                "q": q,
            }

    attempts.sort(key=lambda row: (row["final_penalty"], row["max_violation"]))
    certified = None
    realized = []
    for entry in attempts[:12]:
        q = entry["_q"]
        # Resolve the last quadrature-level overlaps by the minimum isotropic
        # expansion, then let fixed-angle LP search for a smaller exact box.
        centers, dilation = polish.minimal_feasible_dilation(
            q[:, :2], q[:, 2], tau=1e-10
        )
        q_dilated = np.column_stack([centers, q[:, 2]])
        dilation_side = polish.exact_bbox_side(q_dilated[:, :2], q_dilated[:, 2])
        max_overlap, max_wall, _ = overlap.max_violation(q_dilated, dilation_side)
        entry["dilation"] = dilation
        entry["dilated_side"] = dilation_side
        entry["dilated_violation"] = max(max_overlap, max_wall)
        if max(max_overlap, max_wall) <= 1e-8 and dilation_side < incumbent_side:
            centers_lp, lp_side = polish.fixed_angle_lp_polish(
                q_dilated[:, :2],
                q_dilated[:, 2],
                rounds=10,
                time_limit=30.0,
            )
            q_lp = np.column_stack([centers_lp, q_dilated[:, 2]])
            lp_ov, lp_wall, _ = overlap.max_violation(q_lp, lp_side)
            entry["fixed_angle_lp_side"] = lp_side
            entry["fixed_angle_lp_violation"] = max(lp_ov, lp_wall)
            if max(lp_ov, lp_wall) <= 1e-8:
                realized.append((lp_side, q_lp))
            realized.append((dilation_side, q_dilated))
    realized.sort(key=lambda pair: pair[0])
    for candidate_side, q_candidate in realized:
        if candidate_side >= incumbent_side:
            continue
        certified = certify_candidate(q_candidate, n, incumbent_side)
        if certified:
            best["realized_side"] = candidate_side
            break
    return {
        "n": n,
        "incumbent_side": incumbent_side,
        "transport_maps": transports,
        "screened_starts": len(screened),
        "repaired_starts": len(attempts),
        "best": None if best is None else {
            key: value for key, value in best.items() if key not in ("q", "_q")
        },
        "attempts": [
            {key: value for key, value in row.items() if key != "_q"}
            for row in attempts[:12]
        ],
        "certified": certified,
        "elapsed_seconds": time.monotonic() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", default="803,791,908")
    parser.add_argument("--seconds-per-target", type=float, default=240.0)
    parser.add_argument("--repair-iterations", type=int, default=900)
    parser.add_argument("--top-starts", type=int, default=18)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--donor-offsets", default="-2,-1,1,2")
    parser.add_argument("--epsilons", default="0.025,0.06")
    parser.add_argument("--deltas", default="1e-8,1e-7,1e-6,1e-5,1e-4")
    parser.add_argument("--alphas", default="0.2,0.4,0.65,0.9")
    parser.add_argument("--angle-weight", type=float, default=0.08)
    parser.add_argument("--include-same-d", action="store_true")
    parser.add_argument("--out", default=str(DOCS / "results.json"))
    args = parser.parse_args()
    targets = [int(item) for item in args.targets.split(",") if item]
    donor_offsets = tuple(
        int(item) for item in args.donor_offsets.split(",") if item
    )
    epsilons = tuple(float(item) for item in args.epsilons.split(",") if item)
    deltas = tuple(float(item) for item in args.deltas.split(",") if item)
    alphas = tuple(float(item) for item in args.alphas.split(",") if item)
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
                    [donor_offsets] * len(targets),
                    [epsilons] * len(targets),
                    [deltas] * len(targets),
                    [alphas] * len(targets),
                    [args.angle_weight] * len(targets),
                    [args.include_same_d] * len(targets),
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
                donor_offsets,
                epsilons,
                deltas,
                alphas,
                args.angle_weight,
                args.include_same_d,
            )
            for n in targets
        ]

    wins = {}
    for result in results:
        if result["certified"]:
            wins[result["n"]] = result["certified"]
    existing = union.parse_text(FINAL.read_text(encoding="utf-8")) if FINAL.exists() else {}
    for n, item in wins.items():
        existing[n] = union.parse_text(item["final_text"])[n]
    FINAL.write_text(union.final_text(existing) if existing else "id,x,y,deg\n", encoding="utf-8")

    DOCS.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": "Sinkhorn barycentric center/angle transfer plus fixed-side repair",
        "targets": targets,
        "results": [
            {key: value for key, value in result.items() if key != "certified"}
            for result in results
        ],
        "wins": {
            str(n): {
                "literal_side": value["literal_side"],
                "pandas_side": value["pandas_side"],
            }
            for n, value in wins.items()
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
