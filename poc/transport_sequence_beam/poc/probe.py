#!/usr/bin/env python3
"""Beam search over short Sinkhorn-transport/repair sequences."""
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
OUTPUT_DIR = ROOT / "poc/transport_sequence_beam"
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


overlap = _load_module("beam_overlap", ROOT / "poc/k2p10_fill2/poc/a_overlap.py")
union = _load_module("beam_union", ROOT / "poc/auto_union/poc/union.py")
polish = _load_module(
    "beam_polish", ROOT / "poc/swarm_inflation/poc/polish_and_certify.py"
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
    return (q[:, :2] - lo) / (hi - lo)


def wrap_angle(delta: np.ndarray) -> np.ndarray:
    period = math.pi / 2
    return (delta + period / 2) % period - period / 2


def sinkhorn(cost: np.ndarray, epsilon: float, rounds: int = 65) -> np.ndarray:
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
    angle_delta = np.asarray(
        wrap_angle(source[:, None, 2] - donor[None, :, 2]), dtype=np.float64
    )
    cost = spatial + 0.08 * np.abs(angle_delta)
    coupling = sinkhorn(cost, epsilon)
    weights = coupling / np.maximum(coupling.sum(axis=1, keepdims=True), 1e-300)
    centers = weights @ xd
    cos_mean = weights @ np.cos(2 * donor[:, 2])
    sin_mean = weights @ np.sin(2 * donor[:, 2])
    angles = 0.5 * np.arctan2(sin_mean, cos_mean)
    return centers, angles, float((coupling * cost).sum())


def place_centered(
    source: np.ndarray,
    mapped_xy: np.ndarray,
    mapped_angles: np.ndarray,
    target_side: float,
    alpha: float,
) -> np.ndarray:
    current_xy = normalize_centers(source)
    desired_xy = (1.0 - alpha) * current_xy + alpha * mapped_xy
    desired_xy *= target_side
    target_angle = source[:, 2] + alpha * wrap_angle(
        mapped_angles - source[:, 2]
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
        return overlap.penalty(flat.reshape(-1, 3), target_side, grad=True)

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


def diverse_top(rows: list[dict], count: int) -> list[dict]:
    """Keep the best residual while preventing one donor path filling the beam."""
    rows.sort(key=lambda row: (row["violation"], row["penalty"], row["path"]))
    result = []
    seen = set()
    for row in rows:
        first = row["path"][0] if row["path"] else ()
        if first not in seen:
            result.append(row)
            seen.add(first)
        if len(result) >= count:
            return result
    keys = {tuple(row["path"]) for row in result}
    for row in rows:
        if tuple(row["path"]) not in keys:
            result.append(row)
            keys.add(tuple(row["path"]))
        if len(result) >= count:
            break
    return result


def realize(q: np.ndarray, n: int, incumbent_side: float):
    centers, dilation = polish.minimal_feasible_dilation(
        q[:, :2], q[:, 2], tau=1e-10
    )
    q_dilated = np.column_stack([centers, q[:, 2]])
    dilated_side = polish.exact_bbox_side(q_dilated[:, :2], q_dilated[:, 2])
    ov, wall, _ = overlap.max_violation(q_dilated, dilated_side)
    entry = {
        "dilation": float(dilation),
        "dilated_side": float(dilated_side),
        "dilated_violation": float(max(ov, wall)),
    }
    options = []
    if entry["dilated_violation"] <= 1e-8 and dilated_side < incumbent_side:
        options.append((dilated_side, q_dilated))
        centers_lp, lp_side = polish.fixed_angle_lp_polish(
            q_dilated[:, :2],
            q_dilated[:, 2],
            rounds=8,
            time_limit=20.0,
        )
        q_lp = np.column_stack([centers_lp, q_dilated[:, 2]])
        lp_ov, lp_wall, _ = overlap.max_violation(q_lp, lp_side)
        entry["fixed_angle_lp_side"] = float(lp_side)
        entry["fixed_angle_lp_violation"] = float(max(lp_ov, lp_wall))
        if entry["fixed_angle_lp_violation"] <= 1e-8:
            options.append((lp_side, q_lp))
    return entry, options


def probe_target(
    n: int,
    groups: dict[int, np.ndarray],
    seconds: float,
    beam_width: int,
    stage1_starts: int,
    stage2_starts: int,
    stage1_repair: int,
    stage2_repair: int,
    incumbent_side: float,
) -> dict:
    started = time.monotonic()
    current = groups[n]
    offsets = (-4, -3, -2, -1, 1, 2, 3, 4)
    donors = [n + offset for offset in offsets if n + offset in groups]
    epsilons = (0.0125, 0.025, 0.05)
    alphas1 = (0.05, 0.1, 0.2, 0.35)
    alphas2 = (0.04, 0.08, 0.15, 0.3)
    deltas = (1e-4, 2e-4, 5e-4)
    stage_sides = (incumbent_side, incumbent_side + 0.005)

    maps1 = {
        (donor_n, epsilon): transport_map(current, groups[donor_n], epsilon)
        for donor_n in donors
        for epsilon in epsilons
    }
    screened1 = []
    for donor_n in donors:
        for epsilon in epsilons:
            mapped_xy, mapped_angles, value = maps1[(donor_n, epsilon)]
            for stage_side in stage_sides:
                for alpha in alphas1:
                    q0 = place_centered(
                        current, mapped_xy, mapped_angles, stage_side, alpha
                    )
                    penalty = float(overlap.penalty(q0, stage_side, grad=False))
                    screened1.append(
                        {
                            "penalty": penalty,
                            "violation": penalty,
                            "path": [("transport", donor_n, epsilon, alpha)],
                            "q0": q0,
                            "side": stage_side,
                            "transport_value": value,
                        }
                    )
    screened1.sort(key=lambda row: (row["violation"], row["penalty"]))
    stage1 = []
    for row in screened1[:stage1_starts]:
        q, penalty, violation = repair(
            row["q0"], row["side"], iterations=stage1_repair
        )
        rec = dict(row)
        rec.update(
            {"q0": q, "penalty": penalty, "violation": violation}
        )
        stage1.append(rec)
    beam = diverse_top(stage1, beam_width)

    screened2 = []
    for parent_idx, parent in enumerate(beam):
        for donor_n in donors:
            for epsilon in epsilons:
                mapped_xy, mapped_angles, value = transport_map(
                    parent["q0"], groups[donor_n], epsilon
                )
                for delta in deltas:
                    target_side = incumbent_side - delta
                    for alpha in alphas2:
                        q0 = place_centered(
                            parent["q0"],
                            mapped_xy,
                            mapped_angles,
                            target_side,
                            alpha,
                        )
                        penalty = float(
                            overlap.penalty(q0, target_side, grad=False)
                        )
                        screened2.append(
                            {
                                "penalty": penalty,
                                "violation": penalty,
                                "path": parent["path"]
                                + [
                                    (
                                        "transport",
                                        donor_n,
                                        epsilon,
                                        alpha,
                                        delta,
                                    )
                                ],
                                "q0": q0,
                                "side": target_side,
                                "parent": parent_idx,
                                "transport_value": value,
                            }
                        )
                    if time.monotonic() - started > seconds * 1.5:
                        break
                if time.monotonic() - started > seconds * 1.5:
                    break
            if time.monotonic() - started > seconds * 1.5:
                break

    screened2.sort(key=lambda row: (row["violation"], row["penalty"]))
    attempts = []
    for row in screened2[:stage2_starts]:
        q, penalty, violation = repair(
            row["q0"], row["side"], iterations=stage2_repair
        )
        rec = dict(row)
        rec.update(
            {"q0": q, "penalty": penalty, "violation": violation}
        )
        attempts.append(rec)
    attempts.sort(key=lambda row: (row["violation"], row["penalty"]))

    certified = None
    realizations = []
    for entry in attempts[:16]:
        realized, options = realize(entry["q0"], n, incumbent_side)
        entry.update(realized)
        for candidate_side, q in options:
            candidate = certify_candidate(q, n, incumbent_side)
            realizations.append(
                {
                    "side": float(candidate_side),
                    "certified": candidate,
                    "path": entry["path"],
                }
            )
            if candidate and (
                certified is None
                or float(candidate["literal_side"])
                < float(certified["literal_side"])
            ):
                certified = candidate

    def public_row(row: dict) -> dict:
        return {
            key: value
            for key, value in row.items()
            if key not in ("q0",)
        }

    return {
        "n": n,
        "incumbent_side": incumbent_side,
        "stage1_maps": len(maps1),
        "stage1_screened": len(screened1),
        "stage1_repaired": len(stage1),
        "beam_width": len(beam),
        "stage2_screened": len(screened2),
        "stage2_repaired": len(attempts),
        "beam": [public_row(row) for row in beam],
        "attempts": [public_row(row) for row in attempts[:16]],
        "realizations": [
            {
                "side": row["side"],
                "path": row["path"],
                "certified": None
                if row["certified"] is None
                else {
                    "literal_side": row["certified"]["literal_side"],
                    "pandas_side": row["certified"]["pandas_side"],
                },
            }
            for row in realizations
        ],
        "certified": certified,
        "elapsed_seconds": time.monotonic() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", default="803,367,328")
    parser.add_argument("--seconds-per-target", type=float, default=180.0)
    parser.add_argument("--beam-width", type=int, default=8)
    parser.add_argument("--stage1-starts", type=int, default=8)
    parser.add_argument("--stage2-starts", type=int, default=20)
    parser.add_argument("--stage1-repair", type=int, default=500)
    parser.add_argument("--stage2-repair", type=int, default=2500)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--out", default=str(DOCS / "results.json"))
    args = parser.parse_args()
    targets = [int(item) for item in args.targets.split(",") if item]
    groups = load_groups()
    existing = union.parse_text(FINAL.read_text(encoding="utf-8"))
    incumbent_sides = {}
    for n in targets:
        best = side(groups[n])
        if n in existing:
            rows = sorted(existing[n], key=lambda row: int(row["square_id"]))
            q = np.asarray(
                [(float(r["x"]), float(r["y"]), float(r["angle"])) for r in rows]
            )
            best = min(best, side(q))
        incumbent_sides[n] = best

    worker = (
        targets,
        [groups] * len(targets),
        [args.seconds_per_target] * len(targets),
        [args.beam_width] * len(targets),
        [args.stage1_starts] * len(targets),
        [args.stage2_starts] * len(targets),
        [args.stage1_repair] * len(targets),
        [args.stage2_repair] * len(targets),
        [incumbent_sides[n] for n in targets],
    )
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(probe_target, *worker))
    else:
        results = [
            probe_target(
                n,
                groups,
                args.seconds_per_target,
                args.beam_width,
                args.stage1_starts,
                args.stage2_starts,
                args.stage1_repair,
                args.stage2_repair,
                incumbent_sides[n],
            )
            for n in targets
        ]

    wins = {
        result["n"]: result["certified"]
        for result in results
        if result["certified"] is not None
    }
    for n, item in wins.items():
        existing[n] = union.parse_text(item["final_text"])[n]
    FINAL.write_text(union.final_text(existing), encoding="utf-8")
    payload = {
        "method": "two-stage Sinkhorn transport/repair beam",
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
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
