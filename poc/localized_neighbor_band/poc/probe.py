#!/usr/bin/env python3
"""Probe position-localized barycenters on unresolved neighboring rows."""
from __future__ import annotations

import argparse
import heapq
import importlib.util
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
H16_PATH = ROOT / "poc/ot_multidonor_barycenter/poc/probe.py"
OUT = ROOT / "poc/localized_neighbor_band"
FINAL = OUT / "final.csv"
DOCS = OUT / "docs"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


h16 = load_module("localized_neighbors_h16", H16_PATH)
union = h16.union
overlap = h16.overlap
polish = h16.polish


def load_best_groups() -> dict[int, np.ndarray]:
    """Overlay all complete certified same-n final groups on canonical rows."""
    groups = h16.load_groups()
    for path in sorted((ROOT / "poc").glob("*/final.csv")):
        try:
            improved = union.parse_text(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for n, rows in improved.items():
            if len(rows) != n:
                continue
            ordered = sorted(rows, key=lambda row: int(row["square_id"]))
            q = np.asarray(
                [(float(r["x"]), float(r["y"]), float(r["angle"])) for r in ordered],
                dtype=np.float64,
            )
            if n not in groups or h16.side(q) < h16.side(groups[n]):
                groups[n] = q
    return groups


def local_weights(
    xy: np.ndarray, base: float, strength: float, pattern: str
) -> np.ndarray:
    if pattern == "x":
        field = xy[:, 0] - 0.5
    elif pattern == "y":
        field = xy[:, 1] - 0.5
    elif pattern == "d+":
        field = 0.5 * ((xy[:, 0] - 0.5) + (xy[:, 1] - 0.5))
    elif pattern == "d-":
        field = 0.5 * ((xy[:, 0] - 0.5) - (xy[:, 1] - 0.5))
    elif pattern == "quadrant":
        field = (xy[:, 0] - 0.5) * (xy[:, 1] - 0.5)
    else:
        raise ValueError(pattern)
    field /= max(float(np.max(np.abs(field))), 1e-12)
    logit = np.log(max(base, 1e-6) / max(1.0 - base, 1e-6))
    return np.clip(1.0 / (1.0 + np.exp(-(logit + strength * field))), 0.02, 0.98)


def local_mixture(
    source_xy: np.ndarray,
    maps: tuple[tuple[np.ndarray, np.ndarray, float], ...],
    base: float,
    strength: float,
    pattern: str,
):
    wa = local_weights(source_xy, base, strength, pattern)
    wb = 1.0 - wa
    xy_a, angle_a, value_a = maps[0]
    xy_b, angle_b, value_b = maps[1]
    xy = wa[:, None] * xy_a + wb[:, None] * xy_b
    angle = 0.5 * np.arctan2(
        wa * np.sin(2 * angle_a) + wb * np.sin(2 * angle_b),
        wa * np.cos(2 * angle_a) + wb * np.cos(2 * angle_b),
    )
    return xy, angle, float((wa * value_a + wb * value_b).mean())


def probe_target(
    n: int,
    groups: dict[int, np.ndarray],
    seconds: float,
    repair_iterations: int,
    top_starts: int,
    offsets: tuple[int, ...],
    epsilons: tuple[float, ...],
    alphas: tuple[float, ...],
    deltas: tuple[float, ...],
    bases: tuple[float, ...],
    strengths: tuple[float, ...],
    patterns: tuple[str, ...],
):
    started = time.monotonic()
    current = groups[n]
    source_xy = h16.normalize_centers(current)
    incumbent = h16.side(current)
    maps: dict[tuple[int, float], tuple[np.ndarray, np.ndarray, float]] = {}
    for offset in offsets:
        donor = n + offset
        if donor not in groups:
            continue
        for epsilon in epsilons:
            if time.monotonic() - started >= seconds:
                break
            maps[(offset, epsilon)] = h16.transport_map(
                current, groups[donor], epsilon
            )
        if time.monotonic() - started >= seconds:
            break

    best = []  # min-heap of the worst retained item: (-penalty, serial, item)
    serial = 0
    screened = 0
    for left_i, left in enumerate(offsets):
        for right in offsets[left_i + 1 :]:
            for epsilon in epsilons:
                if (left, epsilon) not in maps or (right, epsilon) not in maps:
                    continue
                pair = (maps[(left, epsilon)], maps[(right, epsilon)])
                for base in bases:
                    for strength in strengths:
                        for pattern in patterns:
                            xy, angles, transport_value = local_mixture(
                                source_xy, pair, base, strength, pattern
                            )
                            for delta in deltas:
                                target_side = incumbent - delta
                                for alpha in alphas:
                                    q0 = h16.place_centered(
                                        current, xy, angles, target_side, alpha
                                    )
                                    penalty = float(
                                        overlap.penalty(q0, target_side, grad=False)
                                    )
                                    item = {
                                        "n": n,
                                        "incumbent_side": incumbent,
                                        "penalty": penalty,
                                        "schedule": [left, right, base, strength, pattern],
                                        "epsilon": epsilon,
                                        "delta": delta,
                                        "alpha": alpha,
                                        "target_side": target_side,
                                        "transport_value": transport_value,
                                        "q0": q0,
                                    }
                                    screened += 1
                                    if len(best) < top_starts:
                                        heapq.heappush(best, (-penalty, serial, item))
                                    elif penalty < -best[0][0]:
                                        heapq.heapreplace(best, (-penalty, serial, item))
                                    serial += 1

    repaired = []
    for _, _, row in sorted(best, key=lambda x: -x[0]):
        if time.monotonic() - started >= 2.2 * seconds:
            break
        q, penalty, violation = h16.repair(
            row["q0"], row["target_side"], repair_iterations
        )
        repaired.append(dict(row, final_penalty=penalty, violation=violation, q=q))
    repaired.sort(key=lambda row: (row["final_penalty"], row["violation"]))

    realized = []
    for row in repaired[:16]:
        q = row["q"]
        centers, dilation = polish.minimal_feasible_dilation(
            q[:, :2], q[:, 2], tau=1e-10
        )
        qd = np.column_stack([centers, q[:, 2]])
        safe_side = polish.exact_bbox_side(qd[:, :2], qd[:, 2])
        max_ov, max_wall, _ = overlap.max_violation(qd, safe_side)
        row["dilation"] = dilation
        row["safe_side"] = safe_side
        row["safe_violation"] = max(max_ov, max_wall)
        if max(max_ov, max_wall) <= 1e-8 and safe_side < incumbent:
            centers_lp, lp_side = polish.fixed_angle_lp_polish(
                qd[:, :2], qd[:, 2], rounds=10, time_limit=30.0
            )
            qlp = np.column_stack([centers_lp, qd[:, 2]])
            lp_ov, lp_wall, _ = overlap.max_violation(qlp, lp_side)
            if max(lp_ov, lp_wall) <= 1e-8:
                row["lp_side"] = lp_side
                realized.append((lp_side, qlp))

    realized.sort(key=lambda item: item[0])
    certified = None
    for candidate_side, candidate in realized:
        if candidate_side >= incumbent:
            continue
        certified = h16.certify_candidate(candidate, n, incumbent)
        if certified is not None:
            break
    return {
        "n": n,
        "incumbent_side": incumbent,
        "maps": len(maps),
        "screened_starts": screened,
        "repaired_starts": len(repaired),
        "attempts": [
            {k: v for k, v in row.items() if k not in ("q", "q0")}
            for row in repaired[:16]
        ],
        "certified": certified,
        "elapsed_seconds": time.monotonic() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", default="790,792,907")
    parser.add_argument("--seconds-per-target", type=float, default=180)
    parser.add_argument("--repair-iterations", type=int, default=10000)
    parser.add_argument("--top-starts", type=int, default=32)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--donor-offsets", default="1,2,3,4")
    parser.add_argument("--epsilons", default="0.0125,0.018,0.025")
    parser.add_argument("--alphas", default="0.04,0.08,0.15,0.25")
    parser.add_argument("--deltas", default="1e-5,1e-4,3e-4,6e-4,1e-3")
    parser.add_argument("--bases", default="0.25,0.4,0.5,0.65,0.75")
    parser.add_argument("--strengths", default="0.75,1.5,2.5")
    parser.add_argument("--patterns", default="x,y,d+,d-,quadrant")
    parser.add_argument("--out", default=str(DOCS / "results.json"))
    args = parser.parse_args()

    targets = [int(x) for x in args.targets.split(",") if x]
    groups = load_best_groups()
    offsets = tuple(int(x) for x in args.donor_offsets.split(",") if x)
    epsilons = tuple(float(x) for x in args.epsilons.split(",") if x)
    alphas = tuple(float(x) for x in args.alphas.split(",") if x)
    deltas = tuple(float(x) for x in args.deltas.split(",") if x)
    bases = tuple(float(x) for x in args.bases.split(",") if x)
    strengths = tuple(float(x) for x in args.strengths.split(",") if x)
    patterns = tuple(x for x in args.patterns.split(",") if x)

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
                    [offsets] * len(targets),
                    [epsilons] * len(targets),
                    [alphas] * len(targets),
                    [deltas] * len(targets),
                    [bases] * len(targets),
                    [strengths] * len(targets),
                    [patterns] * len(targets),
                )
            )
    else:
        results = [
            probe_target(
                n, groups, args.seconds_per_target, args.repair_iterations,
                args.top_starts, offsets, epsilons, alphas, deltas, bases,
                strengths, patterns,
            )
            for n in targets
        ]

    existing = union.parse_text(FINAL.read_text())
    wins = {}
    for result in results:
        item = result["certified"]
        if item is None:
            continue
        n = result["n"]
        existing[n] = union.parse_text(item["final_text"])[n]
        wins[str(n)] = {
            "literal_side": item["literal_side"],
            "pandas_side": item["pandas_side"],
        }
    FINAL.write_text(union.final_text(existing), encoding="utf-8")
    payload = {
        "method": "position-localized two-donor transport on neighbor band",
        "targets": targets,
        "results": [
            {k: v for k, v in r.items() if k != "certified"} for r in results
        ],
        "wins": wins,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
