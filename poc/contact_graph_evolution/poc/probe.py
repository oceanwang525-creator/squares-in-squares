#!/usr/bin/env python3
"""Evolve local contact graphs and re-realize their coordinates."""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from decimal import Decimal
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[3]
OVERLAP_DIR = ROOT / "poc" / "k2p10_fill2" / "poc"
if str(OVERLAP_DIR) not in sys.path:
    sys.path.insert(0, str(OVERLAP_DIR))
import a_overlap as ao  # noqa: E402

sys.path.insert(0, str(ROOT / "poc" / "baseline" / "poc" / "src"))
from squares.core import certify, kaggle_roundtrip  # noqa: E402

INCUMBENT = ROOT / "poc/pell_corrugation/data/border-scan/merged.csv"
HEADER = "id,x,y,deg\n"


def load_group(n: int) -> np.ndarray:
    frame = pd.read_csv(INCUMBENT, usecols=["n", "x", "y", "angle"])
    group = frame[frame["n"] == n]
    if len(group) != n:
        raise ValueError(f"expected {n} rows for n={n}, got {len(group)}")
    return group[["x", "y", "angle"]].to_numpy(dtype=np.float64)


def graph_at(q: np.ndarray, tol: float = 2e-5) -> dict:
    pairs = ao.pairs(q, margin=0.04)
    depths = ao.pair_depth(q, pairs) if len(pairs) else np.zeros(0)
    active = pairs[np.abs(depths) <= tol] if len(pairs) else np.empty((0, 2), int)
    adjacency = [set() for _ in range(len(q))]
    for i, j in active:
        adjacency[int(i)].add(int(j))
        adjacency[int(j)].add(int(i))
    return {
        "active": active,
        "adjacency": adjacency,
        "pair_depths": depths,
    }


def _fallback_rows(q: np.ndarray) -> list[dict[str, str]]:
    return [
        {
            "square_id": str(i),
            "x": format(float(row[0]), ".50g"),
            "y": format(float(row[1]), ".50g"),
            "angle": format(float(row[2]), ".50g"),
        }
        for i, row in enumerate(q)
    ]


def _combined_final_text(wins: dict[int, list[dict[str, str]]]) -> str:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(["id", "x", "y", "deg"])
    for n, rows in sorted(wins.items()):
        for row in sorted(rows, key=lambda item: int(item["square_id"])):
            writer.writerow(
                [
                    f"{n:03}_{row['square_id']}",
                    "s" + row["x"],
                    "s" + row["y"],
                    "s" + format(float(row["angle"]) * 180.0 / math.pi, ".50g"),
                ]
            )
    return output.getvalue()


def certify_candidate(n: int, q: np.ndarray) -> dict | None:
    rows = _fallback_rows(q)
    try:
        literal = certify(n, rows)
        transported = kaggle_roundtrip({n: rows})
        pandas_side = certify(n, transported[n])
        return {
            "rows": rows,
            "L_literal": literal,
            "L_pandas": pandas_side,
        }
    except Exception:
        return None


def rewrite_seed(q: np.ndarray, graph: dict, rng: np.random.Generator) -> tuple[np.ndarray, dict] | None:
    adjacency = graph["adjacency"]
    candidates = [
        i
        for i, neighbours in enumerate(adjacency)
        if 3 <= len(neighbours) <= 8
    ]
    if not candidates:
        return None
    rng.shuffle(candidates)
    tree = cKDTree(q[:, :2])
    for pivot in candidates[: min(80, len(candidates))]:
        old = list(adjacency[pivot])
        removed = int(old[int(rng.integers(len(old)))])
        nearby = tree.query_ball_point(q[pivot, :2], 1.35)
        addable = [
            int(j)
            for j in nearby
            if int(j) != pivot and int(j) not in adjacency[pivot]
        ]
        if not addable:
            continue
        added = int(addable[int(rng.integers(len(addable)))])
        changed = q.copy()
        delta = changed[added, :2] - changed[pivot, :2]
        norm = float(np.linalg.norm(delta))
        if norm < 1e-10:
            continue
        direction = delta / norm
        target = changed[added, :2] - direction * 1.0
        changed[pivot, :2] += 0.55 * (target - changed[pivot, :2])
        # Preserve the square's angle. Angle mutation would turn this into the
        # already-saturated free-angle local polish family.
        return changed, {
            "pivot": pivot,
            "removed_contact": removed,
            "added_contact": added,
            "old_degree": len(adjacency[pivot]),
            "added_distance": norm,
        }
    return None


def objective(q: np.ndarray, target_side: float) -> float:
    value, _ = ao.penalty(
        q.reshape(-1, 3),
        target_side,
        w_wall=40.0,
        grad=True,
        margin=2e-8,
    )
    return float(value)


def relax_rewrite(
    base: np.ndarray,
    changed: np.ndarray,
    target_side: float,
    maxiter: int,
) -> tuple[np.ndarray, dict]:
    h = 0.5 * (np.abs(np.cos(base[:, 2])) + np.abs(np.sin(base[:, 2])))
    center = (base[:, :2].min(axis=0) + base[:, :2].max(axis=0)) / 2.0
    scale = target_side / max(float(ao.bbox_side(base)), 1e-12)
    seed = base.copy()
    seed[:, :2] = center + (seed[:, :2] - center) * scale
    delta = changed[:, :2] - base[:, :2]
    seed[:, :2] += 0.65 * delta
    seed[:, 0] -= float(np.min(seed[:, 0] - h))
    seed[:, 1] -= float(np.min(seed[:, 1] - h))
    max_side = max(
        float(np.max(seed[:, 0] + h)),
        float(np.max(seed[:, 1] + h)),
    )
    if max_side > target_side:
        seed[:, :2] *= target_side / max_side

    result = minimize(
        lambda flat: objective(flat, target_side),
        seed.ravel(),
        method="L-BFGS-B",
        options={"maxiter": int(maxiter), "ftol": 1e-14, "gtol": 1e-9},
    )
    relaxed = result.x.reshape(-1, 3)
    overlap, wall, count = ao.max_violation(relaxed, target_side)
    return relaxed, {
        "optimizer_success": bool(result.success),
        "iterations": int(getattr(result, "nit", 0)),
        "objective": float(result.fun),
        "max_overlap": float(overlap),
        "max_wall": float(wall),
        "overlap_pairs": int(count),
        "side": float(ao.bbox_side(relaxed)),
    }


def run_target(n: int, rewrites: int, generations: int, maxiter: int, seed: int) -> dict:
    started = time.time()
    rng = np.random.default_rng(seed * 10007 + n)
    base = load_group(n)
    incumbent_side = float(ao.bbox_side(base))
    target_side = incumbent_side - 1e-4
    graph = graph_at(base)
    degrees = np.asarray([len(row) for row in graph["adjacency"]], dtype=int)
    attempts: list[dict] = []
    elites: list[tuple[float, np.ndarray, dict]] = []

    for generation in range(generations):
        for rewrite_index in range(rewrites):
            generated = rewrite_seed(base, graph, rng)
            if generated is None:
                continue
            changed, labels = generated
            relaxed, diagnostics = relax_rewrite(base, changed, target_side, maxiter)
            penalty = diagnostics["max_overlap"] + diagnostics["max_wall"]
            row = {
                "generation": generation,
                "rewrite": rewrite_index,
                **labels,
                **diagnostics,
                "penetration": float(penalty),
            }
            attempts.append(row)
            elites.append((penalty, relaxed, row))
        elites.sort(key=lambda item: item[0])
        elites = elites[:3]

    best = min(attempts, key=lambda row: row["penetration"], default=None)
    certified = None
    if best is not None and best["penetration"] <= 1e-10:
        # The best relaxed candidate is regenerated deterministically only if
        # its exact coordinates happen to certify; no proxy is promoted.
        for penalty, relaxed, row in elites:
            if penalty > 1e-10:
                continue
            verdict = certify_candidate(n, relaxed)
            if verdict is not None:
                literal = Decimal(verdict["L_literal"])
                transported = Decimal(verdict["L_pandas"])
                if literal < Decimal(str(incumbent_side)) and transported < Decimal(
                    str(incumbent_side)
                ):
                    certified = {**row, **verdict}
                    break

    return {
        "n": n,
        "incumbent_L": incumbent_side,
        "target_L": target_side,
        "vertices": n,
        "contact_edges": int(len(graph["active"])),
        "degree_min": int(degrees.min()),
        "degree_median": float(np.median(degrees)),
        "degree_max": int(degrees.max()),
        "low_degree_vertices": int(np.sum(degrees <= 2)),
        "pivot_vertices": int(np.sum((degrees >= 3) & (degrees <= 8))),
        "attempts": len(attempts),
        "best": best,
        "certified": certified,
        "records": attempts,
        "seconds": round(time.time() - started, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=int, nargs="+", default=[803, 791, 908])
    parser.add_argument("--rewrites", type=int, default=20)
    parser.add_argument("--generations", type=int, default=2)
    parser.add_argument("--maxiter", type=int, default=90)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "docs" / "results.json")
    parser.add_argument("--final", type=Path, default=Path(__file__).resolve().parents[1] / "final.csv")
    args = parser.parse_args()

    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [
            pool.submit(
                run_target,
                int(n),
                args.rewrites,
                args.generations,
                args.maxiter,
                args.seed,
            )
            for n in args.targets
        ]
        results = [future.result() for future in futures]

    payload = {"targets": args.targets, "results": results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    wins = {
        row["n"]: row["certified"]["rows"]
        for row in results
        if row.get("certified") is not None
    }
    args.final.write_text(
        _combined_final_text(wins) if wins else HEADER,
        encoding="utf-8",
    )

    for result in results:
        best = result["best"]
        print(
            "n=%d edges=%d attempts=%d best_pen=%s certified=%s seconds=%.1f"
            % (
                result["n"],
                result["contact_edges"],
                result["attempts"],
                None if best is None else best["penetration"],
                result["certified"] is not None,
                result["seconds"],
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
