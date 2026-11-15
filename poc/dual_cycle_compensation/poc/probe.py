#!/usr/bin/env python3
"""Paired discrete contact-face flips followed by joint SAT re-jamming."""
from __future__ import annotations

import argparse
import itertools
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
for path in (
    ROOT / "poc/baseline/poc/src",
    ROOT / "poc/hydrogen_brainstorm/poc",
    ROOT / "poc/auto_union/poc",
):
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
    adjacency = [set() for _ in range(n)]
    edges: dict[tuple[int, int], float] = {}
    for i in range(n):
        dx = x[i + 1 :] - x[i]
        dy = y[i + 1 :] - y[i]
        hi = half_extent(theta[i])
        hj = half_extent(theta[i + 1 :])
        near = (np.abs(dx) <= hi + hj + 1e-6) & (
            np.abs(dy) <= hi + hj + 1e-6
        )
        for offset in np.flatnonzero(near):
            j = i + int(offset) + 1
            ddx, ddy = x[j] - x[i], y[j] - y[i]
            cth = float(c[i] * c[j] + s[i] * s[j])
            sth = float(c[i] * s[j] - s[i] * c[j])
            radius = 0.5 * (1.0 + abs(cth) + abs(sth))
            axes = (
                ddx * c[i] + ddy * s[i],
                -ddx * s[i] + ddy * c[i],
                ddx * c[j] + ddy * s[j],
                -ddx * s[j] + ddy * c[j],
            )
            gap = max(abs(v) - radius for v in axes)
            if gap <= tol:
                adjacency[i].add(j)
                adjacency[j].add(i)
                edges[(i, j)] = gap
    return adjacency, edges


def canonical_cycle(path: list[int]) -> tuple[int, ...]:
    rotations = []
    for direction in (path, list(reversed(path))):
        for k in range(len(path)):
            rotations.append(tuple(direction[k:] + direction[:k]))
    return min(rotations)


def chordless_cycles(
    adjacency: list[set[int]], max_length: int = 8, cap: int = 12000
) -> list[tuple[int, ...]]:
    found: set[tuple[int, ...]] = set()
    n = len(adjacency)
    # The incumbent graphs are nearly triangulated; enumerate triangles first.
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
                if v in used or any(v in adjacency[w] for w in path[:-1]):
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
    gaps = [
        edges.get((min(a, b), max(a, b)), 0.0)
        for a, b in zip(cycle, cycle[1:] + cycle[:1])
    ]
    area = polygon_area(pts)
    score = area * (1.0 + 0.15 * len(cycle)) / (
        1e-6 + float(np.mean(np.abs(gaps)))
    )
    return {
        "cycle": list(cycle),
        "length": len(cycle),
        "area": area,
        "center": [float(pts[:, 0].mean()), float(pts[:, 1].mean())],
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
        a = math.radians(float(kind[3:]))
        mat = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        pts2 = rel @ mat.T + center
        angles += a
    elif kind.startswith("ref"):
        a = math.radians(float(kind[3:]))
        ux, uy = math.cos(a), math.sin(a)
        mat = np.array(
            [[ux * ux - uy * uy, 2 * ux * uy],
             [2 * ux * uy, uy * uy - ux * ux]]
        )
        pts2 = rel @ mat.T + center
        angles = 2 * a - angles
    else:
        raise ValueError(kind)
    nx[ids], ny[ids] = pts2[:, 0], pts2[:, 1]
    nt[ids] = angles
    return nx, ny, nt


def ring_mask(cycles, adjacency: list[set[int]], hops: int) -> np.ndarray:
    chosen = set(itertools.chain.from_iterable(cycles))
    frontier = set(chosen)
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


def pair_distance(face_a: dict, face_b: dict) -> float:
    ca = np.asarray(face_a["center"])
    cb = np.asarray(face_b["center"])
    return float(np.linalg.norm(ca - cb))


def graph_distance(
    cycle_a: tuple[int, ...], cycle_b: tuple[int, ...], adjacency: list[set[int]]
) -> int:
    frontier = set(cycle_a)
    target = set(cycle_b)
    seen = set(frontier)
    for distance in range(4):
        if frontier & target:
            return distance
        nxt = set()
        for i in frontier:
            nxt.update(adjacency[i])
        frontier = nxt - seen
        seen.update(frontier)
        if not frontier:
            break
    return 99


def select_faces(ranked: list[dict], adjacency: list[set[int]], n: int):
    """Choose high-area faces plus a deterministic spatial sample."""
    chosen = ranked[:24]
    rest = ranked[24:]
    if rest:
        rng = np.random.default_rng(81000 + n)
        for idx in rng.choice(len(rest), size=min(12, len(rest)), replace=False):
            chosen.append(rest[int(idx)])
    unique = {}
    for f in chosen:
        unique[tuple(f["cycle"])] = f
    faces = sorted(unique.values(), key=lambda q: q["rank"], reverse=True)

    pairs = []
    for i, face_a in enumerate(faces):
        for j in range(i + 1, len(faces)):
            face_b = faces[j]
            a = tuple(face_a["cycle"])
            b = tuple(face_b["cycle"])
            if set(a) & set(b):
                continue
            d = pair_distance(face_a, face_b)
            gd = graph_distance(a, b, adjacency)
            if d <= 4.8 or gd <= 1:
                # Nearby paired faces are most likely to share a load path.
                pairs.append((gd, d, i, j))
    pairs.sort()
    return faces, pairs[:40]


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
    _, _, lit, reason = sc.evaluate(n, rows)
    if reason != "ok" or lit is None:
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


PAIR_TRANSFORMS = [
    ("rot90", "rot-90"),
    ("rot-90", "rot90"),
    ("rot90", "rot90"),
    ("rot-90", "rot-90"),
    ("rot135", "rot-90"),
    ("rot-90", "rot135"),
    ("ref0", "ref0"),
    ("ref45", "ref45"),
    ("ref0", "ref90"),
    ("ref90", "ref0"),
    ("rot45", "rot-90"),
    ("rot-90", "rot45"),
]


def probe_target(task):
    n, seconds = task
    started = time.monotonic()
    x, y, theta, incumbent, _ = load_group(n)
    adjacency, edges = contact_graph(x, y, theta, tol=2e-5)
    cycles = chordless_cycles(adjacency, max_length=8, cap=12000)
    ranked = sorted(
        (face_rank(c, x, y, edges) for c in cycles if len(c) <= 7),
        key=lambda r: r["rank"],
        reverse=True,
    )
    faces, pair_indices = select_faces(ranked, adjacency, n)
    L = float(incumbent)
    attempts = []
    screened = []

    for pair_rank, (gd, cd, ia, ib) in enumerate(pair_indices):
        if time.monotonic() - started > seconds * 0.70:
            break
        cycle_a = tuple(faces[ia]["cycle"])
        cycle_b = tuple(faces[ib]["cycle"])
        id_a = np.asarray(cycle_a, dtype=np.int64)
        id_b = np.asarray(cycle_b, dtype=np.int64)
        for kind_a, kind_b in PAIR_TRANSFORMS:
            if time.monotonic() - started > seconds * 0.70:
                break
            sx, sy, st = transform(id_a, x, y, theta, kind_a)
            sx, sy, st = transform(id_b, sx, sy, st, kind_b)
            mask = ring_mask((cycle_a, cycle_b), adjacency, hops=1)
            _, _, _, raw_pen = gu.relax_lbfgs(
                sx, sy, st, L + 0.008, free_mask=mask, maxiter=22
            )
            screened.append(
                {
                    "pair": pair_rank,
                    "graph_distance": gd,
                    "center_distance": cd,
                    "face_a": list(cycle_a),
                    "face_b": list(cycle_b),
                    "transform_a": kind_a,
                    "transform_b": kind_b,
                    "screen_penetration": raw_pen,
                    "state": (sx, sy, st),
                }
            )

    screened.sort(key=lambda q: q["screen_penetration"])
    # Re-finalize the best diverse pair/transform combinations only.
    keep = []
    seen_pair = {}
    for item in screened:
        count = seen_pair.get(item["pair"], 0)
        if count >= 3:
            continue
        seen_pair[item["pair"]] = count + 1
        keep.append(item)
        if len(keep) >= 64:
            break
    winners = {}
    for item in keep:
        if time.monotonic() - started > seconds:
            break
        cycle_a = tuple(item["face_a"])
        cycle_b = tuple(item["face_b"])
        mask = ring_mask((cycle_a, cycle_b), adjacency, hops=1)
        sx, sy, st = (arr.copy() for arr in item["state"])
        stage_results = []
        for target_side, iterations in ((L + 0.001, 55), (L - 2e-5, 65)):
            sx, sy, st, pen = gu.relax_lbfgs(
                sx, sy, st, target_side, free_mask=mask, maxiter=iterations
            )
            _, measured = gu.sat_max_overlap_and_bbox(sx, sy, st)
            stage_results.append(
                {
                    "target_side": target_side,
                    "penetration": pen,
                    "measured_side": measured,
                }
            )
            if pen <= 1e-11 and measured <= target_side + 2e-8:
                candidate = export_candidate(n, sx, sy, st)
                if candidate is not None:
                    if (
                        Decimal(candidate["L_lit"]) <= incumbent - Decimal("1e-9")
                        and Decimal(candidate["L_pandas"])
                        <= incumbent - Decimal("1e-9")
                    ):
                        winners[n] = candidate
                        break
        attempts.append(
            {
                "pair": item["pair"],
                "graph_distance": item["graph_distance"],
                "center_distance": item["center_distance"],
                "face_a": item["face_a"],
                "face_b": item["face_b"],
                "transform_a": item["transform_a"],
                "transform_b": item["transform_b"],
                "screen_penetration": item["screen_penetration"],
                "stages": stage_results,
                "released_squares": int(mask.sum()),
            }
        )

    best = min(attempts, key=lambda q: q["stages"][-1]["penetration"], default=None)
    return {
        "n": n,
        "incumbent": str(incumbent),
        "contact_edges": len(edges),
        "chordless_cycles": len(cycles),
        "selected_faces": len(faces),
        "selected_pairs": len(pair_indices),
        "screened_combinations": len(screened),
        "finalized_combinations": len(attempts),
        "best_attempt": best,
        "winner": winners.get(n),
        "elapsed_seconds": time.monotonic() - started,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", default="969,908")
    parser.add_argument("--seconds-per-target", type=int, default=300)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    targets = [int(v) for v in args.targets.split(",") if v]
    tasks = [(n, args.seconds_per_target) for n in targets]
    with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as pool:
        results = list(pool.map(probe_target, tasks))

    winners = {int(r["n"]): r["winner"] for r in results if r["winner"]}
    final_path = ROOT / "poc/dual_cycle_compensation/final.csv"
    if winners:
        final_path.write_text(
            auto_union.final_text({n: w["rows"] for n, w in winners.items()}),
            encoding="utf-8",
        )
    else:
        final_path.write_text("id,x,y,deg\n", encoding="utf-8")

    report = {
        "targets": targets,
        "seconds_per_target": args.seconds_per_target,
        "pair_transforms": PAIR_TRANSFORMS,
        "winners": {
            str(n): {"L_lit": w["L_lit"], "L_pandas": w["L_pandas"]}
            for n, w in winners.items()
        },
        "results": results,
    }
    out = ROOT / "poc/dual_cycle_compensation/docs/results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for result in results:
        best = result["best_attempt"]
        print(
            json.dumps(
                {
                    "n": result["n"],
                    "faces": result["chordless_cycles"],
                    "pairs": result["selected_pairs"],
                    "screened": result["screened_combinations"],
                    "finalized": result["finalized_combinations"],
                    "winner": result["winner"] is not None,
                    "best_penetration": None
                    if best is None
                    else best["stages"][-1]["penetration"],
                    "best_side": None
                    if best is None
                    else best["stages"][-1]["measured_side"],
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
