#!/usr/bin/env python3
"""Discrete face-cycle ownership flips followed by local SAT re-jamming."""
from __future__ import annotations

import argparse
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
os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
INCUMBENT = ROOT / "poc/pell_corrugation/data/border-scan/merged.csv"
BASELINE_SRC = ROOT / "poc/baseline/poc/src"
GEOM_POC = ROOT / "poc/hydrogen_brainstorm/poc"
UNION_POC = ROOT / "poc/auto_union/poc"
for path in (BASELINE_SRC, GEOM_POC, UNION_POC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import geom_utils as gu  # noqa: E402
import union as auto_union  # noqa: E402
from squares import core as sc  # noqa: E402


def half_extent(theta: np.ndarray) -> np.ndarray:
    return 0.5 * (np.abs(np.cos(theta)) + np.abs(np.sin(theta)))


def load_group(n: int):
    frame = pd.read_csv(INCUMBENT, usecols=["n", "square_id", "x", "y", "angle"])
    group = frame[frame["n"] == n].sort_values("square_id")
    if len(group) != n:
        raise ValueError(f"n={n}: found {len(group)} rows")
    x = group["x"].to_numpy(np.float64, copy=True)
    y = group["y"].to_numpy(np.float64, copy=True)
    theta = group["angle"].to_numpy(np.float64, copy=True)
    h = half_extent(theta)
    x -= float(np.min(x - h))
    y -= float(np.min(y - h))
    rows = [
        {
            "square_id": str(int(r.square_id)),
            "x": str(r.x),
            "y": str(r.y),
            "angle": str(r.angle),
        }
        for r in group.itertuples(index=False)
    ]
    _, _, side, reason = sc.evaluate(n, rows)
    if reason != "ok" or side is None:
        raise ValueError(f"n={n}: incumbent failed certification: {reason}")
    return x, y, theta, Decimal(side), rows


def contact_graph(x: np.ndarray, y: np.ndarray, theta: np.ndarray, tol: float):
    n = len(x)
    c, s = np.cos(theta), np.sin(theta)
    ax = np.stack((c, -s, c, -s), axis=1)
    ay = np.stack((s, c, s, c), axis=1)
    adjacency = [set() for _ in range(n)]
    edges: dict[tuple[int, int], float] = {}
    for i in range(n):
        dx = x[i + 1 :] - x[i]
        dy = y[i + 1 :] - y[i]
        h_i = half_extent(theta[i])
        h_j = half_extent(theta[i + 1 :])
        near = (np.abs(dx) <= h_i + h_j + 1e-6) & (
            np.abs(dy) <= h_i + h_j + 1e-6
        )
        js = np.flatnonzero(near) + i + 1
        for j in js:
            ddx, ddy = x[j] - x[i], y[j] - y[i]
            proj = ddx * ax[i] + ddy * ay[i]
            cth = c[i] * c[j] + s[i] * s[j]
            sth = c[i] * s[j] - s[i] * c[j]
            radius = 0.5 * (1.0 + abs(float(cth)) + abs(float(sth)))
            gap = float(np.max(np.abs(proj) - radius))
            if gap <= tol:
                adjacency[i].add(int(j))
                adjacency[j].add(int(i))
                edges[(i, int(j))] = gap
    return adjacency, edges


def canonical_cycle(path: list[int]) -> tuple[int, ...]:
    rotations = []
    seq = path
    for direction in (seq, list(reversed(seq))):
        for k in range(len(seq)):
            rotations.append(tuple(direction[k:] + direction[:k]))
    return min(rotations)


def chordless_cycles(
    adjacency: list[set[int]], max_length: int = 8, cap: int = 20000
) -> list[tuple[int, ...]]:
    found: set[tuple[int, ...]] = set()
    n = len(adjacency)
    # The q_jam contact graph is nearly triangulated, so its elementary faces
    # are triangles even when the filtered graph has no chordless 4-cycles.
    for i in range(n):
        for j in adjacency[i]:
            if j <= i:
                continue
            for k in adjacency[j]:
                if k > j and k in adjacency[i]:
                    found.add((i, j, k))
    for start in range(n):
        if len(found) >= cap:
            break
        # Requiring start to be the smallest id avoids cycling each component
        # once per member while retaining every simple cycle.
        def dfs(u: int, path: list[int], used: set[int]):
            if len(found) >= cap or len(path) > max_length:
                return
            for v in sorted(adjacency[u]):
                if v < start:
                    continue
                if v == start:
                    if len(path) >= 3:
                        found.add(canonical_cycle(path))
                    continue
                if v in used:
                    continue
                # A face is induced: a chord means this cycle is not a planar
                # ownership boundary.
                if any(v in adjacency[w] for w in path[:-1]):
                    continue
                used.add(v)
                path.append(v)
                dfs(v, path, used)
                path.pop()
                used.remove(v)

        dfs(start, [start], {start})
    return sorted(found)


def polygon_area(points: np.ndarray) -> float:
    q = np.roll(points, -1, axis=0)
    return float(abs(np.sum(points[:, 0] * q[:, 1] - q[:, 0] * points[:, 1])) / 2)


def face_rank(cycle: tuple[int, ...], x, y, edges) -> dict:
    pts = np.column_stack((x[list(cycle)], y[list(cycle)]))
    area = polygon_area(pts)
    gaps = []
    for a, b in zip(cycle, cycle[1:] + cycle[:1]):
        gaps.append(edges.get((min(a, b), max(a, b)), 0.0))
    # Large centre-polygon area and tight cycle contacts indicate a load-bearing
    # face whose boundary may own hidden slack.
    score = area * (1.0 + 0.15 * len(cycle)) / (
        1e-6 + float(np.mean(np.abs(gaps)))
    )
    return {
        "cycle": list(cycle),
        "length": len(cycle),
        "area": area,
        "mean_contact_gap": float(np.mean(gaps)),
        "rank": float(score),
    }


def transform(ids, x, y, theta, kind: str):
    nx, ny, nt = x.copy(), y.copy(), theta.copy()
    pts = np.column_stack((x[ids], y[ids]))
    center = pts.mean(axis=0)
    rel = pts - center
    angles = theta[ids].copy()
    if kind.startswith("rot"):
        degrees = float(kind[3:])
        a = math.radians(degrees)
        mat = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        pts2 = rel @ mat.T + center
        angles += a
    elif kind.startswith("ref"):
        degrees = float(kind[3:])
        a = math.radians(degrees)
        ux, uy = math.cos(a), math.sin(a)
        mat = np.array(
            [[ux * ux - uy * uy, 2 * ux * uy],
             [2 * ux * uy, uy * uy - ux * ux]]
        )
        pts2 = rel @ mat.T + center
        # Reflection of a direction vector across the axis.
        angles = 2 * a - angles
    elif kind.startswith("scale"):
        factor = float(kind[5:])
        pts2 = rel * factor + center
    else:
        raise ValueError(kind)
    nx[ids], ny[ids] = pts2[:, 0], pts2[:, 1]
    nt[ids] = angles
    return nx, ny, nt


def ring_mask(
    cycle: tuple[int, ...], adjacency: list[set[int]], hops: int
) -> np.ndarray:
    chosen = set(cycle)
    frontier = set(cycle)
    for _ in range(hops):
        nxt = set()
        for i in frontier:
            nxt.update(adjacency[i])
        frontier = nxt - chosen
        chosen.update(frontier)
        if not frontier:
            break
    mask = np.zeros(len(adjacency), dtype=bool)
    mask[list(chosen)] = True
    return mask


def export_candidate(n: int, x, y, theta):
    h = half_extent(theta)
    x = x - float(np.min(x - h)) + 1e-12
    y = y - float(np.min(y - h)) + 1e-12
    rows = [
        {
            "square_id": str(i),
            "x": format(float(x[i]), ".60g"),
            "y": format(float(y[i]), ".60g"),
            "angle": format(float(theta[i]), ".60g"),
        }
        for i in range(n)
    ]
    _, _, lit, lit_reason = sc.evaluate(n, rows)
    if lit_reason != "ok" or lit is None:
        return None
    transported = sc.kaggle_roundtrip({n: rows})[n]
    _, _, pd_side, pd_reason = sc.evaluate(n, transported)
    if pd_reason != "ok" or pd_side is None:
        return None
    return {
        "rows": rows,
        "L_lit": lit,
        "L_pandas": pd_side,
    }


def probe_target(task):
    n, seconds = task
    started = time.monotonic()
    x, y, theta, incumbent, _ = load_group(n)
    adjacency, edges = contact_graph(x, y, theta, tol=2e-5)
    cycles = chordless_cycles(adjacency, max_length=8, cap=20000)
    ranked = sorted(
        (face_rank(c, x, y, edges) for c in cycles if len(c) <= 7),
        key=lambda r: r["rank"],
        reverse=True,
    )
    # Rank by geometric area, then retain a deterministic stratified sample so
    # a useful hinge is not missed merely because its centre polygon is small.
    top = ranked[:12]
    remaining = ranked[12:]
    if remaining:
        rng = np.random.default_rng(n)
        take = min(8, len(remaining))
        for idx in rng.choice(len(remaining), size=take, replace=False):
            top.append(remaining[int(idx)])
    ranked = top
    transforms = [
        "rot45",
        "rot90",
        "rot135",
        "rot-90",
        "rot180",
        "ref0",
        "ref90",
        "ref22.5",
        "ref45",
        "ref67.5",
        "ref135",
        "scale0.82",
        "scale1.10",
    ]
    incumbent_float = float(incumbent)
    stages = [
        incumbent_float + 0.008,
        incumbent_float + 0.001,
        incumbent_float - 2e-5,
    ]
    attempts = []
    winners: dict[int, dict] = {}
    for face_rank_i, face in enumerate(ranked):
        cycle = tuple(face["cycle"])
        ids = np.asarray(cycle, dtype=np.int64)
        for hops in (0, 1, 2):
            mask = ring_mask(cycle, adjacency, hops)
            for kind in transforms:
                if time.monotonic() - started > seconds:
                    break
                sx, sy, st = transform(ids, x, y, theta, kind)
                best_pen = float("inf")
                best_side = float("inf")
                stage_attempts = []
                for stage_i, side in enumerate(stages):
                    sx, sy, st, pen = gu.relax_lbfgs(
                        sx, sy, st, side, free_mask=mask, maxiter=55
                    )
                    _, measured = gu.sat_max_overlap_and_bbox(sx, sy, st)
                    best_pen = min(best_pen, pen)
                    best_side = min(best_side, measured)
                    stage_attempts.append(
                        {
                            "target_side": side,
                            "penetration": pen,
                            "measured_side": measured,
                        }
                    )
                    if pen <= 1e-11 and measured <= side + 2e-8:
                        candidate = export_candidate(n, sx, sy, st)
                        if candidate is not None:
                            meaningful_lit = (
                                Decimal(candidate["L_lit"])
                                <= incumbent - Decimal("1e-9")
                            )
                            meaningful_pd = (
                                Decimal(candidate["L_pandas"])
                                <= incumbent - Decimal("1e-9")
                            )
                            if meaningful_lit and meaningful_pd:
                                winners[n] = candidate
                    if stage_i == 0 and pen > 0.2:
                        # The flipped topology is too broken for the released
                        # ring to repair; do not spend the smaller-side stages.
                        break
                attempts.append(
                    {
                        "face": face_rank_i,
                        "cycle": list(cycle),
                        "transform": kind,
                        "release_hops": hops,
                        "released_squares": int(mask.sum()),
                        "best_penetration": best_pen,
                        "best_measured_side": best_side,
                        "stage_attempts": stage_attempts,
                    }
                )
        if time.monotonic() - started > seconds:
            break

    near = min(attempts, key=lambda r: r["best_penetration"], default=None)
    return {
        "n": n,
        "incumbent": str(incumbent),
        "contact_edges": len(edges),
        "chordless_cycles": len(cycles),
        "top_faces": ranked,
        "attempts": attempts,
        "best_attempt": near,
        "winner": winners.get(n),
        "elapsed_seconds": time.monotonic() - started,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", default="969,908,791")
    parser.add_argument("--seconds-per-target", type=int, default=240)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    targets = [int(v) for v in args.targets.split(",") if v]
    tasks = [(n, args.seconds_per_target) for n in targets]
    with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as pool:
        results = list(pool.map(probe_target, tasks))

    winners = {
        int(r["n"]): r["winner"]
        for r in results
        if r["winner"] is not None
    }
    final_path = ROOT / "poc/face_cycle_flip/final.csv"
    if winners:
        groups = {n: w["rows"] for n, w in winners.items()}
        final_path.write_text(auto_union.final_text(groups), encoding="utf-8")
    report = {
        "targets": targets,
        "seconds_per_target": args.seconds_per_target,
        "winners": {
            str(n): {
                "L_lit": w["L_lit"],
                "L_pandas": w["L_pandas"],
            }
            for n, w in winners.items()
        },
        "results": results,
    }
    out = ROOT / "poc/face_cycle_flip/docs/results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for r in results:
        best = r["best_attempt"]
        print(
            json.dumps(
                {
                    "n": r["n"],
                    "faces": r["chordless_cycles"],
                    "attempts": len(r["attempts"]),
                    "winner": r["winner"] is not None,
                    "best_pen": None if best is None else best["best_penetration"],
                    "best_side": None if best is None else best["best_measured_side"],
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
