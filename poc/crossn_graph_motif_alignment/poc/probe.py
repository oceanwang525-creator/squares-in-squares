#!/usr/bin/env python3
"""Cross-n labeled contact-motif alignment followed by exact certification."""
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


def load_group(n: int):
    import pandas as pd

    frame = pd.read_csv(INCUMBENT, usecols=["n", "square_id", "x", "y", "angle"])
    group = frame[frame["n"] == n].sort_values("square_id")
    if len(group) != n:
        raise ValueError(f"n={n}: found {len(group)} rows")
    x = group["x"].to_numpy(np.float64, copy=True)
    y = group["y"].to_numpy(np.float64, copy=True)
    theta = group["angle"].to_numpy(np.float64, copy=True)
    h = 0.5 * (np.abs(np.cos(theta)) + np.abs(np.sin(theta)))
    x -= float(np.min(x - h))
    y -= float(np.min(y - h))
    x += 1e-11
    y += 1e-11
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
        raise ValueError(f"n={n}: canonical row failed certification: {reason}")
    return x, y, theta, Decimal(side), rows


def contact_graph(x: np.ndarray, y: np.ndarray, theta: np.ndarray, tol: float):
    n = len(x)
    c, s = np.cos(theta), np.sin(theta)
    half = 0.5 * (np.abs(c) + np.abs(s))
    adjacency = [set() for _ in range(n)]
    edge_gap: dict[tuple[int, int], float] = {}
    for i in range(n):
        dx = x[i + 1 :] - x[i]
        dy = y[i + 1 :] - y[i]
        reach = half[i] + half[i + 1 :] + tol + 1e-9
        near = (np.abs(dx) <= reach) & (np.abs(dy) <= reach)
        for off in np.flatnonzero(near):
            j = i + 1 + int(off)
            ddx, ddy = x[j] - x[i], y[j] - y[i]
            cj, sj = c[j], s[j]
            cth = c[i] * cj + s[i] * sj
            sth = c[i] * sj - s[i] * cj
            radius = 0.5 * (1.0 + abs(cth) + abs(sth))
            axes = (
                ddx * c[i] + ddy * s[i],
                -ddx * s[i] + ddy * c[i],
                ddx * cj + ddy * sj,
                -ddx * sj + ddy * cj,
            )
            gap = float(max(abs(value) - radius for value in axes))
            if gap <= tol:
                adjacency[i].add(j)
                adjacency[j].add(i)
                edge_gap[(i, j)] = gap
    return adjacency, edge_gap


def orientation_class(angle: float) -> int:
    # Fold away the square's fourfold symmetry and retain 2.5-degree classes.
    folded = angle % (0.5 * math.pi)
    return int(round(folded / math.radians(2.5)))


def wall_mask(i, x, y, theta, side: Decimal, tol: float) -> int:
    h = 0.5 * (abs(math.cos(theta[i])) + abs(math.sin(theta[i])))
    mask = 0
    if x[i] - h <= tol:
        mask |= 1
    if side - (x[i] + h) <= tol:
        mask |= 2
    if y[i] - h <= tol:
        mask |= 4
    if side - (y[i] + h) <= tol:
        mask |= 8
    return mask


def node_label(i, adjacency, x, y, theta, side, tol):
    return (len(adjacency[i]), orientation_class(theta[i]), wall_mask(i, x, y, theta, side, tol))


def motif_table(x, y, theta, side, adjacency, min_degree=3):
    side_f = float(side)
    labels = [node_label(i, adjacency, x, y, theta, side_f, 2e-4) for i in range(len(x))]
    table: dict[tuple, list[tuple[int, tuple[int, ...]]]] = {}
    for center, neighbors in enumerate(adjacency):
        if len(neighbors) < min_degree:
            continue
        ordered = tuple(sorted(neighbors))
        signature = (labels[center], tuple(sorted(labels[j] for j in ordered)))
        table.setdefault(signature, []).append((center, ordered))
    return table, labels


def polar_sort(ids, center, x, y):
    dx = x[list(ids)] - x[center]
    dy = y[list(ids)] - y[center]
    angles = np.mod(np.arctan2(dy, dx), 2 * math.pi)
    return tuple(int(ids[k]) for k in np.argsort(angles))


def dihedral(ids, center, x, y, theta, kind: int):
    px = x[list(ids)] - x[center]
    py = y[list(ids)] - y[center]
    if kind < 4:
        angle = kind * math.pi / 2
        c, s = math.cos(angle), math.sin(angle)
        tx = c * px - s * py
        ty = s * px + c * py
        nt = theta[list(ids)] + angle
    else:
        # Reflect first, then rotate by 0, 90, 180, or 270 degrees.
        px, py = px, -py
        angle = (kind - 4) * math.pi / 2
        c, s = math.cos(angle), math.sin(angle)
        tx = c * px - s * py
        ty = s * px + c * py
        nt = -theta[list(ids)] + angle
    return tx, ty, nt


def one_ring(motif, adjacency):
    selected = set(int(i) for i in motif)
    for i in tuple(selected):
        selected.update(adjacency[i])
    return np.asarray(sorted(selected), dtype=np.int64)


def candidate_rows(n, x, y, theta):
    h = 0.5 * (np.abs(np.cos(theta)) + np.abs(np.sin(theta)))
    x = x - float(np.min(x - h)) + 1e-12
    y = y - float(np.min(y - h)) + 1e-12
    return [
        {
            "square_id": str(i),
            "x": format(float(x[i]), ".60g"),
            "y": format(float(y[i]), ".60g"),
            "angle": format(float(theta[i]), ".60g"),
        }
        for i in range(n)
    ]


def certify_candidate(n, x, y, theta, incumbent):
    rows = candidate_rows(n, x, y, theta)
    _, _, literal, reason = sc.evaluate(n, rows)
    if reason != "ok" or literal is None:
        return None
    transported = sc.kaggle_roundtrip({n: rows})[n]
    _, _, pandas_side, pandas_reason = sc.evaluate(n, transported)
    if pandas_reason != "ok" or pandas_side is None:
        return None
    margin = Decimal("1e-9")
    if Decimal(literal) > incumbent - margin or Decimal(pandas_side) > incumbent - margin:
        return None
    return {"rows": rows, "L_lit": literal, "L_pandas": pandas_side}


def donor_numbers(n):
    k = math.isqrt(n)
    d = n - k * k
    result = [v for v in (n - 2, n - 1, n + 1, n + 2) if 1 <= v <= 1000]
    for kk in (k - 1, k + 1):
        nn = kk * kk + d
        if 1 <= nn <= 1000 and nn != n:
            result.append(nn)
    return sorted(set(result))


def probe_target(task):
    n, seconds = task
    started = time.monotonic()
    x, y, theta, incumbent, _ = load_group(n)
    adjacency, _ = contact_graph(x, y, theta, tol=2e-5)
    target_table, target_labels = motif_table(x, y, theta, incumbent, adjacency)
    donors = []
    all_matches = []
    for donor_n in donor_numbers(n):
        dx, dy, dtheta, dside, _ = load_group(donor_n)
        donor_adj, _ = contact_graph(dx, dy, dtheta, tol=2e-5)
        donor_table, _ = motif_table(dx, dy, dtheta, dside, donor_adj)
        common = sorted(
            set(target_table).intersection(donor_table),
            key=lambda sig: (-len(sig[1]), sig),
        )
        meaningful = [sig for sig in common if sig[0][0] >= 3]
        donors.append(
            {
                "n": donor_n,
                "common_star_signatures": len(common),
                "meaningful_stars": len(meaningful),
            }
        )
        for sig in meaningful[:24]:
            all_matches.append((donor_n, sig, dx, dy, dtheta, donor_adj, donor_table[sig]))

    # Prefer rare signatures shared across several donors, then high degree.
    frequency = {}
    for item in all_matches:
        frequency[item[1]] = frequency.get(item[1], 0) + 1
    all_matches.sort(
        key=lambda item: (
            -frequency[item[1]],
            -item[1][0][0],
            item[0],
            item[1],
        )
    )

    attempts = []
    winner = None
    seen_motifs = set()
    for donor_n, sig, dx, dy, dtheta, donor_adj, donor_instances in all_matches:
        target_instances = target_table[sig]
        for target_center, target_neighbors in target_instances[:4]:
            motif_key = (donor_n, sig, target_center)
            if motif_key in seen_motifs:
                continue
            seen_motifs.add(motif_key)
            target_ids = tuple(sorted((target_center,) + target_neighbors))
            free = one_ring(target_ids, adjacency)
            target_order = polar_sort(target_neighbors, target_center, x, y)
            for donor_center, donor_neighbors in donor_instances[:3]:
                donor_ids = (donor_center,) + donor_neighbors
                donor_order = polar_sort(donor_neighbors, donor_center, dx, dy)
                for kind in range(8):
                    if time.monotonic() - started > seconds:
                        break
                    tx, ty, nt = dihedral(donor_ids, donor_center, dx, dy, dtheta, kind)
                    # Map center + dihedral-transformed neighbors to the target motif.
                    new_x, new_y, new_t = x.copy(), y.copy(), theta.copy()
                    new_x[target_center] = x[target_center]
                    new_y[target_center] = y[target_center]
                    new_t[target_center] = theta[target_center]
                    by_id = {sid: p for p, sid in enumerate(donor_ids)}
                    rel = np.column_stack((tx, ty))
                    rel -= rel[by_id[donor_center]]
                    donor_angles = np.asarray(nt)
                    center_angle = float(donor_angles[by_id[donor_center]])
                    donor_angles_abs = np.arctan2(rel[:, 1], rel[:, 0])
                    donor_sorted = sorted(
                        donor_neighbors,
                        key=lambda sid: donor_angles_abs[by_id[sid]] % (2 * math.pi),
                    )
                    for idx, donor_neighbor in enumerate(donor_sorted):
                        local = rel[by_id[donor_neighbor]]
                        local_angle = float(donor_angles[by_id[donor_neighbor]]) - center_angle
                        target_neighbor = target_order[idx % len(target_order)]
                        new_x[target_neighbor] = x[target_center] + local[0]
                        new_y[target_neighbor] = y[target_center] + local[1]
                        new_t[target_neighbor] = theta[target_center] + local_angle
                    stages = (
                        float(incumbent) + 0.006,
                        float(incumbent) + 0.0008,
                        float(incumbent) - 2e-5,
                    )
                    stage_result = []
                    best_pen = float("inf")
                    for stage in stages:
                        new_x, new_y, new_t, pen = gu.relax_lbfgs(
                            new_x,
                            new_y,
                            new_t,
                            stage,
                            free_mask=free,
                            maxiter=90,
                        )
                        best_pen = min(best_pen, pen)
                        stage_result.append(
                            {
                                "side": stage,
                                "penetration": pen,
                                "measured_side": gu.sat_max_overlap_and_bbox(
                                    new_x, new_y, new_t
                                )[1],
                            }
                        )
                        if pen <= 1e-11:
                            certified = certify_candidate(
                                n, new_x, new_y, new_t, incumbent
                            )
                            if certified is not None:
                                winner = certified
                                break
                    attempts.append(
                        {
                            "donor_n": donor_n,
                            "signature_degree": sig[0][0],
                            "target_center": int(target_center),
                            "release_squares": int(len(free)),
                            "transform": kind,
                            "best_penetration": best_pen,
                            "stages": stage_result,
                        }
                    )
                    if winner is not None or time.monotonic() - started > seconds:
                        break
                if winner is not None or time.monotonic() - started > seconds:
                    break
            if winner is not None or time.monotonic() - started > seconds:
                break
        if winner is not None or time.monotonic() - started > seconds:
            break

    best = min(attempts, key=lambda item: item["best_penetration"], default=None)
    return {
        "n": n,
        "incumbent": str(incumbent),
        "target_star_signatures": len(target_table),
        "donors": donors,
        "matches_considered": len(seen_motifs),
        "attempts": attempts,
        "best_attempt": best,
        "winner": winner,
        "elapsed_seconds": time.monotonic() - started,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", default="803,791,908")
    parser.add_argument("--seconds-per-target", type=int, default=240)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    targets = [int(value) for value in args.targets.split(",") if value]
    with ProcessPoolExecutor(max_workers=min(args.workers, len(targets))) as pool:
        results = list(pool.map(probe_target, [(n, args.seconds_per_target) for n in targets]))
    winners = {
        int(result["n"]): result["winner"]
        for result in results
        if result["winner"] is not None
    }
    out = ROOT / "poc/crossn_graph_motif_alignment/docs/results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "targets": targets,
                "seconds_per_target": args.seconds_per_target,
                "winners": {
                    str(n): {"L_lit": value["L_lit"], "L_pandas": value["L_pandas"]}
                    for n, value in winners.items()
                },
                "results": results,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    final_path = ROOT / "poc/crossn_graph_motif_alignment/final.csv"
    final_path.write_text(
        auto_union.final_text(
            {n: value["rows"] for n, value in winners.items()}
        ),
        encoding="utf-8",
    )
    for result in results:
        best = result["best_attempt"]
        print(
            json.dumps(
                {
                    "n": result["n"],
                    "target_signatures": result["target_star_signatures"],
                    "common_donor_max": max(
                        (d["meaningful_stars"] for d in result["donors"]), default=0
                    ),
                    "matches": result["matches_considered"],
                    "attempts": len(result["attempts"]),
                    "best_penetration": None if best is None else best["best_penetration"],
                    "winner": result["winner"] is not None,
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
