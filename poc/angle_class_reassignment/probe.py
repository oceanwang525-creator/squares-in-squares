#!/usr/bin/env python3
"""Reassign folded-angle classes after proven centre-only transport."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[2]
H16_PATH = ROOT / "poc/ot_multidonor_barycenter/poc/probe.py"
OUTPUT_DIR = ROOT / "poc/angle_class_reassignment"
FINAL = OUTPUT_DIR / "final.csv"
DOCS = OUTPUT_DIR / "docs"
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


h16 = load_module("angle_class_h16", H16_PATH)
union = h16.union
polish = h16.polish
overlap = h16.overlap


def circular_class_mean(values: np.ndarray, weights: np.ndarray) -> float:
    if values.size == 0:
        raise ValueError("empty angle class")
    anchor = float(values[0])
    unwrapped = anchor + h16.wrap_angle(values - anchor)
    return float(np.average(unwrapped, weights=weights))


def cluster_folded_angles(
    values: np.ndarray, weights: np.ndarray, count: int
) -> tuple[np.ndarray, np.ndarray]:
    """Small deterministic 1-D k-means on the [-pi/4, pi/4] angle chart."""
    folded = np.asarray(h16.wrap_angle(values), dtype=np.float64)
    count = max(2, min(count, len(folded)))
    centers = np.quantile(
        folded,
        np.linspace(0.5 / count, 1.0 - 0.5 / count, count),
    )
    labels = np.zeros(len(folded), dtype=np.int32)
    for _ in range(24):
        new_labels = np.argmin(
            np.abs(folded[:, None] - centers[None, :]), axis=1
        ).astype(np.int32)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for index in range(count):
            mask = labels == index
            if mask.any():
                centers[index] = np.average(folded[mask], weights=weights[mask])
    # Convert chart centers back to a representative in [-pi/4, pi/4].
    representatives = np.asarray(
        [
            circular_class_mean(values[labels == index], weights[labels == index])
            if np.any(labels == index)
            else centers[index]
            for index in range(count)
        ]
    )
    return labels, representatives


def smooth_labels(
    xy: np.ndarray,
    labels: np.ndarray,
    neighbours: int = 8,
    rounds: int = 1,
) -> np.ndarray:
    distance = cdist(xy, xy)
    order = np.argsort(distance, axis=1)[:, 1 : neighbours + 1]
    work = labels.copy()
    classes = int(labels.max()) + 1
    for _ in range(rounds):
        updated = work.copy()
        for index in range(len(work)):
            counts = np.bincount(work[order[index]], minlength=classes)
            updated[index] = int(np.argmax(counts))
        work = updated
    return work


def reassigned_angles(
    mixed_xy: np.ndarray,
    mixed_angles: np.ndarray,
    current: np.ndarray,
    class_count: int,
    mode: str,
    smoothing: float,
) -> tuple[np.ndarray, dict]:
    weights = np.full(len(mixed_angles), 1.0 / len(mixed_angles))
    labels, representatives = cluster_folded_angles(
        mixed_angles, weights, class_count
    )
    mixed_xy = np.asarray(mixed_xy, dtype=np.float64)
    lo = mixed_xy.min(axis=0)
    span = np.maximum(mixed_xy.max(axis=0) - lo, 1e-12)
    mapped_xy = (mixed_xy - lo) / span

    if mode == "pointwise":
        labels = smooth_labels(mapped_xy, labels, rounds=1)
    elif mode == "region":
        class_xy = np.asarray(
            [
                np.average(mapped_xy[labels == index], axis=0)
                if np.any(labels == index)
                else mapped_xy.mean(axis=0)
                for index in range(class_count)
            ]
        )
        current_xy = h16.normalize_centers(current)
        labels = np.argmin(
            cdist(current_xy, class_xy), axis=1
        ).astype(np.int32)
    else:
        raise ValueError(f"unknown class mode: {mode}")

    result = representatives[labels].copy()
    blend = float(np.clip(smoothing, 0.0, 1.0))
    result = mixed_angles + blend * h16.wrap_angle(result - mixed_angles)
    hist = np.bincount(labels, minlength=class_count).tolist()
    return result, {
        "mode": mode,
        "classes": int(class_count),
        "smoothing": blend,
        "class_counts": hist,
        "representatives": representatives.tolist(),
    }


def incumbent_groups(targets: list[int], groups: dict[int, np.ndarray]):
    result: dict[int, np.ndarray] = {n: groups[n] for n in targets}
    for path in sorted((ROOT / "poc").glob("*/final.csv")):
        try:
            literal, transported = union.load(path)
        except Exception:
            continue
        for n in targets:
            if n not in literal or n not in transported:
                continue
            if len(literal[n]) != n or len(transported[n]) != n:
                continue
            rows = sorted(literal[n], key=lambda row: int(row["square_id"]))
            candidate = np.asarray(
                [
                    (float(row["x"]), float(row["y"]), float(row["angle"]))
                    for row in rows
                ],
                dtype=np.float64,
            )
            if h16.side(candidate) < h16.side(result[n]):
                result[n] = candidate
    return result


def probe_target(
    n: int,
    groups: dict[int, np.ndarray],
    incumbent: np.ndarray,
    seconds: float,
    repair_iterations: int,
    top_starts: int,
) -> dict:
    started = time.monotonic()
    incumbent_side = h16.side(incumbent)
    offsets = [-4, -3, -2, -1, 1, 2, 3, 4]
    epsilons = [0.012, 0.025, 0.05]
    deltas = [1e-6, 1e-5, 1e-4, 2e-4]
    alphas = [0.2, 0.4, 0.65]
    ratios = [(1.0, 1.0), (2.0, 1.0), (1.0, 2.0)]
    schedules = h16.schedules(offsets, ratios, include_triples=False)
    maps = {}
    for offset in offsets:
        for epsilon in epsilons:
            maps[(offset, epsilon)] = h16.transport_map(
                incumbent, groups[n + offset], epsilon
            )

    screened = []
    for epsilon in epsilons:
        for schedule in schedules:
            total = sum(weight for _, weight in schedule)
            mixed_xy = np.zeros((n, 2), dtype=np.float64)
            cos_sum = np.zeros(n, dtype=np.float64)
            sin_sum = np.zeros(n, dtype=np.float64)
            for offset, weight in schedule:
                mapped_xy, mapped_angles, _ = maps[(offset, epsilon)]
                part = weight / total
                mixed_xy += part * mapped_xy
                cos_sum += part * np.cos(2 * mapped_angles)
                sin_sum += part * np.sin(2 * mapped_angles)
            mixed_angles = 0.5 * np.arctan2(sin_sum, cos_sum)
            for classes in (2, 3, 4):
                for mode in ("pointwise", "region"):
                    for smoothing in (0.45, 1.0):
                        rebuilt, meta = reassigned_angles(
                            mixed_xy,
                            mixed_angles,
                            incumbent,
                            classes,
                            mode,
                            smoothing,
                        )
                        for delta in deltas:
                            target_side = incumbent_side - delta
                            for alpha in alphas:
                                q0 = h16.place_centered(
                                    incumbent,
                                    mixed_xy,
                                    rebuilt,
                                    target_side,
                                    alpha,
                                )
                                penalty = float(
                                    overlap.penalty(q0, target_side, grad=False)
                                )
                                screened.append(
                                    (
                                        penalty,
                                        dict(
                                            epsilon=epsilon,
                                            donor_schedule=schedule,
                                            delta=delta,
                                            alpha=alpha,
                                            target_side=target_side,
                                            angle_classes=meta,
                                        ),
                                        q0,
                                    )
                                )
                        if time.monotonic() - started >= seconds:
                            break
                    if time.monotonic() - started >= seconds:
                        break
                if time.monotonic() - started >= seconds:
                    break
            if time.monotonic() - started >= seconds:
                break
        if time.monotonic() - started >= seconds:
            break

    screened.sort(key=lambda row: row[0])
    repaired = []
    for penalty0, meta, q0 in screened[:top_starts]:
        q, final_penalty, violation = h16.repair(
            q0, meta["target_side"], repair_iterations
        )
        centers, dilation = polish.minimal_feasible_dilation(
            q[:, :2], q[:, 2], tau=1e-10
        )
        q_dilated = np.column_stack([centers, q[:, 2]])
        dilated_side = polish.exact_bbox_side(
            q_dilated[:, :2], q_dilated[:, 2]
        )
        repaired.append(
            {
                **meta,
                "initial_penalty": penalty0,
                "final_penalty": final_penalty,
                "max_violation": violation,
                "dilation": dilation,
                "dilated_side": dilated_side,
                "_q": q_dilated,
            }
        )
    repaired.sort(key=lambda row: (row["final_penalty"], row["dilated_side"]))

    realized = []
    for entry in repaired:
        q = entry["_q"]
        if entry["dilated_side"] < incumbent_side:
            centers_lp, lp_side = polish.fixed_angle_lp_polish(
                q[:, :2],
                q[:, 2],
                rounds=10,
                time_limit=30.0,
            )
            q_lp = np.column_stack([centers_lp, q[:, 2]])
            lp_ov, lp_wall, _ = overlap.max_violation(q_lp, lp_side)
            entry["lp_side"] = lp_side
            entry["lp_violation"] = max(lp_ov, lp_wall)
            if lp_side < incumbent_side and max(lp_ov, lp_wall) <= 1e-8:
                realized.append((lp_side, q_lp))

    realized.sort(key=lambda item: item[0])
    certified = None
    for _, candidate in realized:
        certified = h16.certify_candidate(candidate, n, incumbent_side)
        if certified:
            break

    return {
        "n": n,
        "incumbent_side": incumbent_side,
        "transport_maps": len(maps),
        "screened_starts": len(screened),
        "repaired_starts": len(repaired),
        "best": [
            {
                key: value
                for key, value in entry.items()
                if key != "_q"
            }
            for entry in repaired[:12]
        ],
        "realized": len(realized),
        "certified": certified,
        "elapsed_seconds": time.monotonic() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", default="803,328,791")
    parser.add_argument("--seconds-per-target", type=float, default=180.0)
    parser.add_argument("--repair-iterations", type=int, default=12000)
    parser.add_argument("--top-starts", type=int, default=48)
    parser.add_argument("--out", default=str(DOCS / "results.json"))
    args = parser.parse_args()
    targets = [int(item) for item in args.targets.split(",") if item]
    groups = h16.load_groups()
    incumbents = incumbent_groups(targets, groups)
    results = [
        probe_target(
            n,
            groups,
            incumbents[n],
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
        "method": "angle-class reassignment after centre transport",
        "targets": targets,
        "results": [
            {
                key: value
                for key, value in result.items()
                if key != "certified"
            }
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
    Path(args.out).write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
