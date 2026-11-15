#!/usr/bin/env python3
"""Transport-seeded, contact-local masked repair for unit-square packings."""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[3]
H16_POC = ROOT / "poc/ot_multidonor_barycenter"
H16_SCRIPT = H16_POC / "poc/probe.py"
OUT = Path(__file__).resolve().parents[1]
DOCS = OUT / "docs"
FINAL = OUT / "final.csv"
BASELINE = ROOT / "poc/baseline/poc/src"
for path in (BASELINE,):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


h16 = _load("transport_masked_h16", H16_SCRIPT)
overlap = h16.overlap
polish = h16.polish
union = h16.union
core = h16.core


def canonical_group(n: int, groups: dict[int, np.ndarray]) -> np.ndarray:
    q = groups[n].copy()
    e = h16.extents(q[:, 2])
    lo = (q[:, :2] - e[:, None]).min(axis=0)
    q[:, :2] -= lo
    return q


def parse_schedule(items) -> tuple[tuple[int, float], ...]:
    return tuple((int(offset), float(weight)) for offset, weight in items)


def best_recorded_seed(n: int, groups):
    entries = []
    for path in sorted(glob.glob(str(H16_POC / "docs" / "results*.json"))):
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for target in payload.get("results", []):
            if int(target.get("n", -1)) != n:
                continue
            for attempt in target.get("attempts", []):
                if "schedule" not in attempt:
                    continue
                entries.append(
                    (
                        float(attempt.get("max_violation", float("inf"))),
                        float(attempt.get("initial_penalty", float("inf"))),
                        attempt,
                    )
                )
    if not entries:
        return None
    entries.sort(key=lambda item: (item[0], item[1]))
    return entries[0][2]


def mix_transport(current: np.ndarray, groups, attempt) -> np.ndarray:
    epsilon = float(attempt["epsilon"])
    schedule = parse_schedule(attempt["schedule"])
    total = sum(weight for _, weight in schedule)
    mixed_xy = np.zeros((len(current), 2), dtype=np.float64)
    cos_sum = np.zeros(len(current), dtype=np.float64)
    sin_sum = np.zeros(len(current), dtype=np.float64)
    for offset, weight in schedule:
        mapped_xy, mapped_angles, _ = h16.transport_map(
            current, groups[len(current) + offset], epsilon
        )
        fraction = weight / total
        mixed_xy += fraction * mapped_xy
        cos_sum += fraction * np.cos(2.0 * mapped_angles)
        sin_sum += fraction * np.sin(2.0 * mapped_angles)
    angles = 0.5 * np.arctan2(sin_sum, cos_sum)
    n = len(current)
    incumbent = h16.side(current)
    target_side = incumbent - float(attempt["delta"])
    return h16.place_centered(
        current, mixed_xy, angles, target_side, float(attempt["alpha"])
    )


def conflict_centers(q: np.ndarray, target_side: float, count: int = 10) -> np.ndarray:
    pair_idx, depths = overlap.overlap_pairs(q)
    centers = []
    if len(pair_idx):
        order = np.argsort(depths)[::-1][:count]
        for k in order:
            pair = pair_idx[k]
            centers.append(0.5 * (q[pair[0], :2] + q[pair[1], :2]))
    wall_violation, _ = overlap.wall_violation(q, target_side)
    stress = wall_violation.max(axis=1)
    for index in np.argsort(stress)[::-1][: min(4, len(q))]:
        if stress[index] > 0.0:
            centers.append(q[index, :2])
    return np.asarray(centers, dtype=np.float64)


def masked_repair(
    q0: np.ndarray,
    target_side: float,
    rounds: int,
    max_iterations: int,
):
    q = q0.copy()
    best = q.copy()
    best_penalty = float(overlap.penalty(q, target_side, grad=False))
    history = []
    for round_index in range(rounds):
        centers = conflict_centers(q, target_side)
        if not len(centers):
            break
        radius = (2.0, 3.0, 4.5, 6.0)[round_index % 4]
        distances = np.min(
            np.linalg.norm(q[:, None, :2] - centers[None, :, :], axis=2), axis=1
        )
        active = np.where(distances <= radius)[0]
        if len(active) < 4:
            active = np.argsort(distances)[:8]
        active = np.unique(active)
        base = q.copy()

        def objective(flat):
            trial = base.copy()
            trial[active] = flat.reshape(-1, 3)
            value, gradient = overlap.penalty(trial, target_side, grad=True)
            return float(value), gradient[active].ravel()

        result = minimize(
            objective,
            base[active].ravel(),
            method="L-BFGS-B",
            jac=True,
            options={
                "maxiter": max_iterations,
                "maxls": 25,
                "ftol": 1e-14,
                "gtol": 1e-10,
            },
        )
        trial = base.copy()
        trial[active] = result.x.reshape(-1, 3)
        penalty = float(overlap.penalty(trial, target_side, grad=False))
        if penalty < best_penalty:
            q = trial
            best = trial.copy()
            best_penalty = penalty
        max_overlap, max_wall, pairs = overlap.max_violation(best, target_side)
        history.append(
            {
                "round": round_index,
                "radius": radius,
                "active": int(len(active)),
                "penalty": best_penalty,
                "max_overlap": float(max_overlap),
                "max_wall": float(max_wall),
                "pairs": int(pairs),
            }
        )
    return best, best_penalty, history


def certify(q: np.ndarray, n: int, incumbent_side: float):
    centers, dilation = polish.minimal_feasible_dilation(
        q[:, :2], q[:, 2], tau=1e-10
    )
    q_safe = np.column_stack([centers, q[:, 2]])
    safe_side = polish.exact_bbox_side(q_safe[:, :2], q_safe[:, 2])
    if safe_side >= incumbent_side:
        return None, {
            "dilation": dilation,
            "safe_side": safe_side,
            "reason": "safe dilation over incumbent",
        }
    centers_lp, lp_side = polish.fixed_angle_lp_polish(
        q_safe[:, :2], q_safe[:, 2], rounds=10, time_limit=30.0
    )
    q_lp = np.column_stack([centers_lp, q_safe[:, 2]])
    result = h16.certify_candidate(q_lp, n, incumbent_side)
    if result is None:
        return None, {
            "dilation": dilation,
            "safe_side": safe_side,
            "lp_side": lp_side,
            "reason": "official certification did not improve",
        }
    return result, {
        "dilation": dilation,
        "safe_side": safe_side,
        "lp_side": lp_side,
    }


def group_from_file(path: Path, n: int) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        puzzles = union.parse_text(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if n not in puzzles:
        return None
    rows = sorted(puzzles[n], key=lambda row: int(row["square_id"]))
    return np.asarray(
        [(float(row["x"]), float(row["y"]), float(row["angle"])) for row in rows]
    )


def run_target(n: int, groups, repair_iterations: int):
    started = time.monotonic()
    current = canonical_group(n, groups)
    incumbent_side = h16.side(current)
    existing_candidates = [
        group_from_file(h16.FINAL, n),
        group_from_file(
            ROOT / "poc/optimal_transport_basin_transfer/final.csv", n
        ),
        group_from_file(FINAL, n),
    ]
    existing_candidates = [item for item in existing_candidates if item is not None]
    existing = (
        min(existing_candidates, key=h16.side) if existing_candidates else None
    )
    if existing is not None:
        incumbent_side = min(incumbent_side, h16.side(existing))
    seed = existing.copy() if existing is not None else None
    attempt = best_recorded_seed(n, groups)
    seed_source = "existing"
    if seed is None:
        seed = current.copy()
        seed_source = "canonical"
    target_side = incumbent_side - 1e-6
    if attempt is not None and seed is None:
        try:
            transported = mix_transport(current, groups, attempt)
            target_side = incumbent_side - float(attempt["delta"])
            repaired, _, violation = h16.repair(
                transported, target_side, repair_iterations
            )
            if violation < 0.05:
                seed = repaired
                seed_source = "transport"
        except Exception as error:
            attempt = {"error": repr(error)}

    initial_penalty = float(overlap.penalty(seed, target_side, grad=False))
    local, local_penalty, history = masked_repair(
        seed, target_side, rounds=12, max_iterations=900
    )
    max_overlap, max_wall, pairs = overlap.max_violation(local, target_side)
    certified, detail = certify(local, n, incumbent_side)
    return {
        "n": n,
        "seed_source": seed_source,
        "recorded_attempt": attempt,
        "incumbent_side": incumbent_side,
        "initial_penalty": initial_penalty,
        "local_penalty": local_penalty,
        "max_overlap": float(max_overlap),
        "max_wall": float(max_wall),
        "overlap_pairs": int(pairs),
        "history": history,
        "certification": detail,
        "certified": certified,
        "elapsed_seconds": time.monotonic() - started,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", default="803,328,367,969,861,791,908")
    parser.add_argument("--repair-iterations", type=int, default=5000)
    parser.add_argument("--out", default=str(DOCS / "results.json"))
    args = parser.parse_args()
    targets = [int(item) for item in args.targets.split(",") if item]
    groups = h16.load_groups()
    results = [run_target(n, groups, args.repair_iterations) for n in targets]

    puzzles = {}
    for result in results:
        item = result.pop("certified", None)
        if item is not None:
            parsed = union.parse_text(item["final_text"])
            puzzles[int(result["n"])] = parsed[int(result["n"])]
    if puzzles:
        existing = {}
        if FINAL.exists():
            try:
                existing = union.parse_text(FINAL.read_text(encoding="utf-8"))
            except Exception:
                existing = {}
        existing.update(puzzles)
        FINAL.write_text(union.final_text(existing), encoding="utf-8")

    payload = {
        "method": "transport-seeded contact-local masked repair",
        "results": results,
        "wins": {
            str(result["n"]): result.get("certification")
            for result in results
            if result.get("certification", {}).get("reason") is None
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
