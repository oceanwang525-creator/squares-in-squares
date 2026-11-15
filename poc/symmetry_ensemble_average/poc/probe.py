#!/usr/bin/env python3
"""Contact-weighted dihedral consensus probe for square packings."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import io
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment, minimize

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "poc/pell_corrugation/data/border-scan/merged.csv"
HERE = ROOT / "poc/symmetry_ensemble_average"
FINAL = HERE / "final.csv"
DOCS = HERE / "docs"
sys.path.insert(0, str(ROOT / "poc/baseline/poc/src"))

from squares import core  # noqa: E402


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


overlap = _load("ensemble_overlap", ROOT / "poc/k2p10_fill2/poc/a_overlap.py")
union = _load("ensemble_union", ROOT / "poc/auto_union/poc/union.py")


def extents(angles: np.ndarray) -> np.ndarray:
    return 0.5 * (np.abs(np.cos(angles)) + np.abs(np.sin(angles)))


def bbox_side(q: np.ndarray) -> float:
    e = extents(q[:, 2])
    return float(
        max(
            (q[:, 0] + e).max() - (q[:, 0] - e).min(),
            (q[:, 1] + e).max() - (q[:, 1] - e).min(),
        )
    )


def normalize(q: np.ndarray) -> tuple[np.ndarray, float]:
    """Return centers in [0,1]^2 and the exact occupied side."""
    e = extents(q[:, 2])
    lo = (q[:, :2] - e[:, None]).min(axis=0)
    hi = (q[:, :2] + e[:, None]).max(axis=0)
    side = float(max(hi - lo))
    xy = (q[:, :2] - lo) / side
    return np.column_stack((xy, q[:, 2])), side


def transform(q: np.ndarray, transform_id: int) -> np.ndarray:
    norm, side = normalize(q)
    xy = norm[:, :2].copy()
    angle = norm[:, 2].copy()
    if transform_id & 1:
        xy[:, 0] = 1.0 - xy[:, 0]
        angle = -angle
    if transform_id & 2:
        xy[:, 1] = 1.0 - xy[:, 1]
        angle = math.pi - angle
    if transform_id & 4:
        xy = np.column_stack((1.0 - xy[:, 1], xy[:, 0]))
        angle += math.pi / 2
    return np.column_stack((xy * side, angle))


def angle_delta(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    period = math.pi / 2
    return (a - b + period / 2) % period - period / 2


def align(variant: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, float, int]:
    """Choose the best dihedral copy and return its Hungarian alignment."""
    ref_norm, _ = normalize(reference)
    best = None
    for transform_id in range(8):
        moved = transform(variant, transform_id)
        moved_norm, _ = normalize(moved)
        spatial = np.linalg.norm(
            moved_norm[:, None, :2] - ref_norm[None, :, :2], axis=2
        )
        angle_cost = np.abs(
            angle_delta(rearrange_angle(moved_norm[:, 2])[:, None],
                        ref_norm[None, :, 2])
        )
        cost = spatial + 0.08 * angle_cost
        row, col = linear_sum_assignment(cost)
        score = float(cost[row, col].mean())
        aligned = np.empty_like(ref_norm)
        aligned[col] = moved_norm[row]
        if best is None or score < best[1]:
            best = (aligned, score, transform_id)
    assert best is not None
    return best


def rearrange_angle(a: np.ndarray) -> np.ndarray:
    return a


def load_base() -> dict[int, np.ndarray]:
    groups: dict[int, list[list[float]]] = {}
    with BASE.open(newline="") as handle:
        for row in csv.DictReader(handle):
            groups.setdefault(int(row["n"]), []).append(
                [float(row["x"]), float(row["y"]), float(row["angle"])]
            )
    return {n: np.asarray(rows, dtype=float) for n, rows in groups.items()}


def load_same_n_alternatives(n: int) -> list[tuple[Path, np.ndarray]]:
    alternatives = []
    prefix = f"{n:03d}_"
    for path in sorted((ROOT / "poc").glob("*/final.csv")):
        text = path.read_text(encoding="utf-8")
        rows = []
        for line in text.splitlines()[1:]:
            fields = line.split(",")
            if len(fields) != 4 or not fields[0].startswith(prefix):
                continue
            try:
                rows.append(
                    [
                        float(fields[1].removeprefix("s")),
                        float(fields[2].removeprefix("s")),
                        math.radians(float(fields[3].removeprefix("s"))),
                    ]
                )
            except ValueError:
                rows = []
                break
        if len(rows) == n:
            alternatives.append((path, np.asarray(rows, dtype=float)))
    return alternatives


def contact_weights(q: np.ndarray) -> np.ndarray:
    pairs = overlap.pairs(q, margin=1e-6)
    degree = np.zeros(len(q), dtype=float)
    if len(pairs):
        np.add.at(degree, pairs[:, 0], 1.0)
        np.add.at(degree, pairs[:, 1], 1.0)
    return 1.0 + 0.25 * degree


def place(reference: np.ndarray, consensus: np.ndarray, weight: np.ndarray,
          target_side: float, alpha: float) -> np.ndarray:
    ref_norm, _ = normalize(reference)
    con_norm, _ = normalize(consensus)
    xy = (1.0 - alpha) * ref_norm[:, :2] + alpha * con_norm[:, :2]
    phi = 4.0 * ((1.0 - alpha) * ref_norm[:, 2] + alpha * con_norm[:, 2])
    angle = np.arctan2(np.sin(phi), np.cos(phi)) / 4.0
    q = np.column_stack((xy * target_side, angle))
    e = extents(angle)
    lo = (q[:, :2] - e[:, None]).min(axis=0)
    hi = (q[:, :2] + e[:, None]).max(axis=0)
    q[:, :2] += 0.5 * (target_side - lo - hi)
    return q


def repair(q0: np.ndarray, target_side: float, iterations: int) -> tuple[np.ndarray, float, float, int]:
    def objective(flat: np.ndarray):
        q = flat.reshape(-1, 3)
        return overlap.penalty(q, target_side, grad=True)

    result = minimize(
        objective,
        q0.ravel(),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": iterations, "maxls": 25, "ftol": 1e-14, "gtol": 1e-10},
    )
    q = result.x.reshape(-1, 3)
    od, wv, count = overlap.max_violation(q, target_side)
    return q, float(result.fun), max(od, wv), count


def dilated_side(q: np.ndarray) -> tuple[np.ndarray, float]:
    """Find the smallest center dilation making all SAT pairs disjoint."""
    q = q.copy()
    e = extents(q[:, 2])
    q[:, :2] -= (q[:, :2] - e[:, None]).min(axis=0)
    side = bbox_side(q)
    center = q[:, :2].mean(axis=0)
    lo, hi = 1.0, 1.25
    expanded = None
    for _ in range(45):
        mid = 0.5 * (lo + hi)
        candidate = q.copy()
        candidate[:, :2] = center + mid * (q[:, :2] - center)
        candidate_side = box_side(candidate)
        overlap_depth, _, _ = overlap.max_violation(candidate, candidate_side)
        # Keep a full unit-margin clearance for exact decimal transport and
        # rounded official SAT checks.
        if overlap_depth <= -1e-8:
            hi = mid
            expanded = candidate
        else:
            lo = mid
    if expanded is None:
        candidate = q.copy()
        candidate[:, :2] = center + hi * (q[:, :2] - center)
        expanded = candidate
    e = extents(expanded[:, 2])
    expanded[:, :2] -= (expanded[:, :2] - e[:, None]).min(axis=0)
    return expanded, hi


def box_side(q: np.ndarray) -> float:
    return bbox_side(q)


def to_text(q: np.ndarray, n: int) -> str:
    groups = {
        n: [
            {
                "square_id": str(i),
                "x": format(float(x), ".17g"),
                "y": format(float(y), ".17g"),
                "angle": format(float(angle), ".17g"),
            }
            for i, (x, y, angle) in enumerate(q)
        ]
    }
    return union.final_text(groups)


def certify(q: np.ndarray, n: int, incumbent: float):
    text = to_text(q, n)
    literal = union.parse_text(text)
    transported = union.parse_text(pd.read_csv(io.StringIO(text)).to_csv(index=False))
    if len(literal.get(n, [])) != n or len(transported.get(n, [])) != n:
        raise ValueError("incomplete candidate")
    try:
        lit = core.certify(n, literal[n])
        pd_side = core.certify(n, transported[n])
    except Exception as exc:
        return None, str(exc)
    if float(lit) >= incumbent or float(pd_side) >= incumbent:
        return None, None
    return {"literal_side": lit, "pandas_side": pd_side, "final_text": text}, None


def probe(
    n: int,
    base_groups: dict[int, np.ndarray],
    iterations: int,
    top_alternatives: int,
) -> dict:
    started = time.monotonic()
    reference = base_groups[n]
    incumbent = bbox_side(reference)
    alternatives = load_same_n_alternatives(n)
    ref_norm, _ = normalize(reference)
    weights = contact_weights(reference)
    aligned = []
    source_rows = []
    for path, variant in alternatives:
        moved, alignment_score, transform_id = align(variant, reference)
        if alignment_score < 1e-10:
            continue
        aligned.append(moved)
        source_rows.append(
            {
                "path": str(path.relative_to(ROOT)),
                "transform": transform_id,
                "alignment_score": alignment_score,
            }
        )
    ranked = sorted(
        zip(source_rows, aligned), key=lambda item: item[0]["alignment_score"]
    )[:top_alternatives]
    source_rows = [item[0] for item in ranked]
    aligned = [item[1] for item in ranked]
    if not aligned:
        consensus = ref_norm.copy()
    else:
        stack = np.stack([ref_norm] + aligned)
        weighted = stack.copy()
        weighted[:, :, :2] *= weights[None, :, None]
        consensus_xy = weighted[:, :, :2].sum(axis=0) / weights.sum()
        z = weights[None, :] * np.exp(4j * stack[:, :, 2])
        phase = z.sum(axis=0)
        consensus = np.column_stack(
            (consensus_xy, np.angle(phase) / 4.0)
        )

    attempts = []
    best = None
    for delta in (1e-8, 1e-7, 1e-6, 1e-5, 1e-4):
        target = incumbent - delta
        for alpha in (0.2, 0.4, 0.65, 0.9):
            q0 = place(reference, consensus, weights, target, alpha)
            q, energy, violation, pairs = repair(q0, target, iterations)
            measured = bbox_side(q)
            dilated_q, dilation = dilated_side(q) if violation > -1e-8 else (q, 1.0)
            feasible_side = bbox_side(dilated_q)
            row = {
                "delta": delta,
                "alpha": alpha,
                "energy": energy,
                "max_violation": violation,
                "overlap_pairs": pairs,
                "repair_side": measured,
                "dilation": dilation,
                "feasible_side": feasible_side,
            }
            attempts.append(row)
            if feasible_side < incumbent and (best is None or feasible_side < best[0]):
                best = (feasible_side, dilated_q, row)
    certified, certification_error = (
        certify(best[1], n, incumbent) if best else (None, None)
    )
    return {
        "n": n,
        "incumbent_side": incumbent,
        "alternatives": source_rows,
        "attempts": sorted(attempts, key=lambda row: row["feasible_side"])[:12],
        "best": None if best is None else best[2],
        "certified": certified,
        "certification_error": certification_error,
        "elapsed_seconds": time.monotonic() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", default="803,328,908")
    parser.add_argument("--repair-iterations", type=int, default=2400)
    parser.add_argument("--top-alternatives", type=int, default=3)
    parser.add_argument("--out", default=str(DOCS / "results.json"))
    args = parser.parse_args()
    targets = [int(item) for item in args.targets.split(",") if item]
    groups = load_base()
    results = [
        probe(n, groups, args.repair_iterations, args.top_alternatives)
        for n in targets
    ]
    wins = {
        str(row["n"]): row["certified"]
        for row in results if row["certified"] is not None
    }
    if wins:
        groups_out = {}
        for n, item in wins.items():
            groups_out[int(n)] = union.parse_text(item["final_text"])[int(n)]
        FINAL.write_text(union.final_text(groups_out), encoding="utf-8")
    else:
        FINAL.write_text("id,x,y,deg\n", encoding="utf-8")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": "dihedral Hungarian alignment plus contact-weighted consensus",
        "targets": targets,
        "results": [
            {key: value for key, value in row.items() if key != "certified"}
            for row in results
        ],
        "wins": {
            n: {"literal_side": item["literal_side"], "pandas_side": item["pandas_side"]}
            for n, item in wins.items()
        },
    }
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
