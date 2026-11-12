"""Probe forced rotations of an incumbent's connected angle groups."""

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "poc/baseline/poc/src"))
from squares._official_metric import evaluate_single_puzzle

FIELDS = ["id", "n", "square_id", "x", "y", "angle"]
PAD = 8e-13


def half_extent(angles):
    return (np.abs(np.cos(angles)) + np.abs(np.sin(angles))) / 2


def fit(p):
    h = half_extent(p[:, 2])[:, None]
    q = p.copy()
    q[:, :2] -= (p[:, :2] - h).min(axis=0)
    length = (q[:, :2] + h).max()
    q[:, 2] = (q[:, 2] + math.pi / 4) % (math.pi / 2) - math.pi / 4
    return q, float(length)


def separation(p, radius):
    i, j = cKDTree(p[:, :2]).query_pairs(radius, output_type="ndarray").T
    dxy = p[j, :2] - p[i, :2]
    theta = p[:, 2]
    directions = np.column_stack((theta[i], theta[i] + math.pi / 2,
                                  theta[j], theta[j] + math.pi / 2))
    nx, ny = np.cos(directions), np.sin(directions)
    projections = dxy[:, 0, None] * nx + dxy[:, 1, None] * ny
    best = np.abs(projections).argmax(axis=1)
    k = np.arange(len(i))
    signs = np.where(projections[k, best] < 0, -1, 1)
    owner = np.where(best < 2, i, j)
    other = np.where(best < 2, j, i)
    ax, ay = signs * nx[k, best], signs * ny[k, best]
    derivative = -ay * dxy[:, 0] + ax * dxy[:, 1]
    return i, j, owner, other, ax, ay, np.abs(projections[k, best]), derivative


def groups_for(p):
    """Group parallel and mirrored squares that meet in the contact graph."""
    theta = p[:, 2]
    magnitude = np.abs(theta)
    ids = np.full(len(p), -1, dtype=int)
    i, j, owner, other, _, _, projection, _ = separation(p, 1.42)
    delta = theta[other] - theta[owner]
    close = projection < (1 + np.abs(np.cos(delta)) + np.abs(np.sin(delta))) / 2 + 0.025
    coherent = close & (np.abs(magnitude[i] - magnitude[j]) < 2e-5)
    adjacency = defaultdict(list)
    for a, b in zip(i[coherent], j[coherent]):
        adjacency[a].append(b)
        adjacency[b].append(a)
    count = 0
    for seed in np.flatnonzero(magnitude > 1e-6):
        if ids[seed] >= 0:
            continue
        todo = [seed]
        ids[seed] = count
        while todo:
            for neighbor in adjacency[todo.pop()]:
                if ids[neighbor] < 0:
                    ids[neighbor] = count
                    todo.append(neighbor)
        count += 1
    return ids, np.sign(theta), count


def step(p, clusters, xy, rotation, forced=None):
    n = len(p)
    p, length = fit(p)
    ids, signs, count = clusters
    angle_col = np.where(ids >= 0, 2 * n + ids, 2 * n + count)
    length_col = 2 * n + count + 1
    i, j, owner, other, ax, ay, projection, axis_derivative = separation(
        p, math.sqrt(2) + 2 * math.sqrt(2) * xy + 1e-8
    )
    a = p[:, 2]
    delta = a[other] - a[owner]
    dc, ds = np.cos(delta), np.sin(delta)
    rr, cc, vv, bb = [], [], [], []

    def add(cols, vals, limits, keep):
        k = np.arange(keep.sum()) + len(bb)
        for col, value in zip(cols, vals):
            cc.extend(np.broadcast_to(col, keep.shape)[keep])
            vv.extend(np.broadcast_to(value, keep.shape)[keep])
            rr.extend(k)
        bb.extend(limits[keep])

    # Every support is the maximum of four signed trigonometric branches.
    for u, v in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        support = (1 + u * dc + v * ds) / 2
        slope = (-u * ds + v * dc) / 2
        slack = projection - support - PAD
        add((i, j, n+i, n+j, angle_col[owner], angle_col[other]),
            (ax, -ax, ay, -ay,
             (-axis_derivative-slope)*signs[owner], slope*signs[other]),
            slack, slack < 3 * (xy + rotation) + 0.005)

        extent = (u*np.cos(a) + v*np.sin(a)) / 2
        extent_slope = (-u*np.sin(a) + v*np.cos(a)) * signs / 2
        for dim in (0, 1):
            center_col = np.arange(n) + dim * n
            lower = p[:, dim] - extent
            upper = length - p[:, dim] - extent
            add((center_col, angle_col), (-1, extent_slope),
                lower, lower < xy + 2*rotation + 0.005)
            add((center_col, angle_col, length_col), (1, extent_slope, -1),
                upper, upper < xy + 2*rotation + 0.005)

    matrix = coo_matrix((vv, (rr, cc)), shape=(len(bb), length_col + 1)).tocsr()
    bounds = [(-xy, xy)] * (2*n) + [(-rotation, rotation)] * count
    if forced is not None:
        group, delta = forced
        bounds[2*n + group] = (delta, delta)
    bounds.extend(((0, 0), (-0.3, 0.3)))
    objective = np.zeros(length_col + 1)
    objective[-1] = 1
    result = linprog(objective, A_ub=matrix, b_ub=bb, bounds=bounds,
                     method="highs", options={"time_limit": 3,
                                              "primal_feasibility_tolerance": 1e-9})
    if not result.success:
        return None
    q = p + np.column_stack((result.x[:n], result.x[n:2*n], result.x[angle_col]*signs))
    return fit(q)[0]


def make_feasible(p):
    i, j, owner, other, _, _, projection, _ = separation(p, 1.415)
    delta = p[other, 2] - p[owner, 2]
    ratio = (1 + np.abs(np.cos(delta)) + np.abs(np.sin(delta))) / (2 * projection)
    q = p.copy()
    q[:, :2] *= max(1.0, float(np.max(ratio, initial=1))) + PAD
    return fit(q)


def probe(job):
    n, p = job
    p, start = fit(p)
    clusters = groups_for(p)
    ids, _, count = clusters
    if not count:
        return n, start, start, 0, None
    counts = np.bincount(ids[ids >= 0], minlength=count)
    options = [None]
    for g in np.argsort(-counts)[:3]:
        if counts[g] >= 3:
            options.extend((g, delta) for delta in (-0.018, -0.006, 0.006, 0.018))
    best_side, best = start, None
    for forced in options:
        current = step(p, clusters, 0.065, 0.02, forced)
        if current is None:
            continue
        previous = None
        for radius in (0.018, 0.007, 0.002, 0.0005, 0.00008):
            candidate, length = make_feasible(current)
            if length < best_side - 2e-11:
                best_side, best = length, candidate
            if previous is not None and abs(previous-length) < 5e-11:
                break
            previous = length
            q = step(current, clusters, 2.5*radius, radius)
            if q is None:
                break
            current = q
    return n, start, best_side, len(options), best


def certify(n, rows):
    import io
    import pandas as pd

    original = evaluate_single_puzzle(n, rows)
    body = pd.DataFrame(rows).to_csv(index=False)
    roundtrip = pd.read_csv(io.StringIO(body)).to_csv(index=False)
    converted = evaluate_single_puzzle(n, list(csv.DictReader(io.StringIO(roundtrip))))
    return original, converted


def main():
    from decimal import Decimal

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("incumbent", type=Path)
    parser.add_argument("candidate", type=Path, help="Output CSV with winning groups only")
    parser.add_argument("report", type=Path)
    parser.add_argument("targets", nargs="+", type=int)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    targets = set(args.targets)
    groups = defaultdict(list)
    with args.incumbent.open(newline="", encoding="utf-8-sig") as source:
        for row in csv.DictReader(source):
            if int(row["n"]) in targets:
                groups[int(row["n"])].append(row)
    assert set(groups) == targets
    jobs = [(n, np.array([[float(r[k]) for k in ("x", "y", "angle")] for r in rows]))
            for n, rows in groups.items()]
    report, wins = [], {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for n, length, best, starts, q in pool.map(probe, jobs):
            result = dict(n=n, initial=length, best=best, starts=starts, accepted=False)
            if q is not None:
                rows = [dict(r, x=repr(float(x)), y=repr(float(y)), angle=repr(float(a)))
                        for r, (x, y, a) in zip(groups[n], q)]
                incumbent = certify(n, groups[n])
                candidate = certify(n, rows)
                result.update(incumbent=incumbent, candidate=candidate)
                if all(new[3] == old[3] == "ok" and Decimal(new[2]) < Decimal(old[2])
                       for old, new in zip(incumbent, candidate)):
                    wins[n] = rows
                    result["accepted"] = True
            report.append(result)
            print(n, starts, "delta", best-length, "accepted", result["accepted"], flush=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if wins:
        with args.candidate.open("w", newline="", encoding="utf-8") as target:
            writer = csv.DictWriter(target, fieldnames=FIELDS)
            writer.writeheader()
            for n in sorted(wins):
                writer.writerows(wins[n])


if __name__ == "__main__":
    main()
