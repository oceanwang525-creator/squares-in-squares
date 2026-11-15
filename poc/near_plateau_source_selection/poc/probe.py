#!/usr/bin/env python3
"""Select transport donors by geometric plateau and contact compatibility."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = ROOT / "poc/near_plateau_source_selection"
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


source = load_module(
    "near_plateau_transport_source",
    ROOT / "poc/ot_multidonor_barycenter/poc/probe.py",
)
union = load_module("near_plateau_union", ROOT / "poc/auto_union/poc/union.py")
core = source.core


def angle_class(theta: np.ndarray) -> np.ndarray:
    folded = np.mod(theta, math.pi / 2)
    return np.minimum((folded / (math.pi / 8)).astype(np.int64), 3)


def geometry_signature(q: np.ndarray, common_side: float) -> np.ndarray:
    """Stable coarse contact/wall signature used only for donor ranking."""
    classes = angle_class(q[:, 2])
    orient = np.bincount(classes, minlength=4).astype(float) / len(q)
    e = source.extents(q[:, 2])
    tol = 0.04
    walls = np.asarray(
        [
            np.mean(q[:, 0] - e <= tol),
            np.mean(q[:, 0] + e >= common_side - tol),
            np.mean(q[:, 1] - e <= tol),
            np.mean(q[:, 1] + e >= common_side - tol),
        ],
        dtype=float,
    )
    distances = cdist(q[:, :2], q[:, :2])
    np.fill_diagonal(distances, np.inf)
    unit = np.mean(np.min(distances, axis=1) <= 1.02)
    diag = np.mean(np.min(distances, axis=1) <= math.sqrt(2) + 0.02)
    return np.concatenate([orient, walls, [unit, diag]])


def rank_donors(
    target_n: int,
    groups: dict[int, np.ndarray],
    limit: int,
    max_side_gap: float,
) -> tuple[list[int], dict[int, float]]:
    target = groups[target_n]
    target_side = source.side(target)
    target_signature = geometry_signature(target, target_side)
    ranked: list[tuple[float, int]] = []
    diagnostics: dict[int, float] = {}
    for donor_n, donor in groups.items():
        if donor_n == target_n:
            continue
        donor_side = source.side(donor)
        side_gap = abs(donor_side - target_side)
        if side_gap > max_side_gap:
            continue
        signature = geometry_signature(donor, max(target_side, donor_side))
        side_term = side_gap / max_side_gap
        orient_term = float(np.abs(signature[:4] - target_signature[:4]).sum())
        wall_term = float(np.abs(signature[4:8] - target_signature[4:8]).sum())
        contact_term = float(np.abs(signature[8:] - target_signature[8:]).sum())
        index_term = min(abs(donor_n - target_n), 16) / 16.0
        score = (
            0.38 * side_term
            + 0.22 * orient_term
            + 0.22 * wall_term
            + 0.12 * contact_term
            + 0.06 * index_term
        )
        diagnostics[donor_n] = score
        ranked.append((score, donor_n))
    ranked.sort()
    return [n for _, n in ranked[:limit]], diagnostics


def valid_existing_groups() -> dict[int, list[dict[str, str]]]:
    if not FINAL.exists():
        return {}
    return union.parse_text(FINAL.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", default="803,328,791")
    parser.add_argument("--seconds-per-target", type=float, default=120.0)
    parser.add_argument("--repair-iterations", type=int, default=6000)
    parser.add_argument("--top-starts", type=int, default=24)
    parser.add_argument("--donors", type=int, default=8)
    parser.add_argument("--max-side-gap", type=float, default=0.02)
    parser.add_argument("--out", default=str(DOCS / "results.json"))
    args = parser.parse_args()
    targets = [int(value) for value in args.targets.split(",") if value]
    groups = source.load_groups()
    existing = valid_existing_groups()

    results = []
    for n in targets:
        donor_ns, ranking = rank_donors(
            n, groups, args.donors, args.max_side_gap
        )
        incumbent = source.side(groups[n])
        # Use any strict group already present in this POC as a floor.
        if n in existing:
            rows = sorted(existing[n], key=lambda row: int(row["square_id"]))
            q = np.asarray(
                [
                    (float(row["x"]), float(row["y"]), float(row["angle"]))
                    for row in rows
                ],
                dtype=np.float64,
            )
            incumbent = min(incumbent, source.side(q))
        result = source.probe_target(
            n=n,
            groups=groups,
            seconds=args.seconds_per_target,
            repair_iterations=args.repair_iterations,
            top_starts=args.top_starts,
            donor_offsets=[donor_n - n for donor_n in donor_ns],
            epsilons=[0.0185, 0.025, 0.04],
            alphas=[0.05, 0.1, 0.2, 0.35],
            deltas=[1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 2e-4],
            weight_ratios=[(1.0, 0.85), (1.0, 1.0), (1.0, 1.5), (1.0, 2.0)],
            include_triples=False,
            incumbent_sides={n: incumbent},
        )
        result["selected_donors"] = donor_ns
        result["donor_ranking"] = [
            [donor_n, ranking[donor_n]] for donor_n in donor_ns
        ]
        results.append(result)

    wins = {}
    for result in results:
        if result["certified"] is not None:
            n = result["n"]
            wins[n] = result["certified"]
            existing[n] = union.parse_text(result["certified"]["final_text"])[n]
    FINAL.write_text(
        union.final_text(existing) if existing else "id,x,y,deg\n",
        encoding="utf-8",
    )

    payload = {
        "method": "near-side/contact-profile donor selection + H16 transport repair",
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
