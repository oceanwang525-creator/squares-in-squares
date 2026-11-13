"""Grow interlocked square packings by inserting a layer along each axis."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import itertools
import json
import math
import os
import sys
from decimal import Decimal
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
from numba import njit
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

HEADER = ("id", "n", "square_id", "x", "y", "angle")
REACH = math.sqrt(2.0)
MARGIN = 2e-11


def extents(angles):
    return (np.abs(np.cos(angles)) + np.abs(np.sin(angles))) / 2


def normalize(points):
    points = points.copy()
    points[:, :2] -= (points[:, :2] - extents(points[:, 2])[:, None]).min(axis=0)
    return points


def side(points):
    h = extents(points[:, 2])[:, None]
    return float(((points[:, :2] + h).max(axis=0) - (points[:, :2] - h).min(axis=0)).max())


@njit
def pair_gap(a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    if dx * dx + dy * dy > 2.0:
        return 0.5
    ca, sa = np.cos(a[2]), np.sin(a[2])
    cb, sb = np.cos(b[2]), np.sin(b[2])
    distance = max(abs(dx * ca + dy * sa), abs(dy * ca - dx * sa),
                   abs(dx * cb + dy * sb), abs(dy * cb - dx * sb))
    dt = b[2] - a[2]
    return distance - 0.5 * (1.0 + abs(np.cos(dt)) + abs(np.sin(dt)))


@njit
def insert_layer(points, cut, axis, allowance, clearance):
    n = len(points)
    result = np.empty((2 * n, 3))
    result[:n] = points
    pitch = 1.0 / np.cos(points[:, 2])
    shift = 1.0
    for i in range(n):
        if cut - 1.3 < points[i, axis] <= cut:
            shift = max(shift, pitch[i])
    shift += clearance
    for i in range(n):
        if points[i, axis] > cut:
            result[i, axis] += shift
    used = n
    for i in np.argsort(-points[:, axis]):
        if not cut - 2.0 <= points[i, axis] <= cut:
            continue
        copy = points[i].copy()
        copy[axis] += pitch[i]
        for j in range(used):
            if pair_gap(copy, result[j]) < -allowance:
                break
        else:
            result[used] = copy
            used += 1
    return result[:used].copy()


def sat(points, left, right):
    displacement = points[right, :2] - points[left, :2]
    a, b = points[left, 2], points[right, 2]
    angles = np.column_stack((a, a + math.pi / 2, b, b + math.pi / 2))
    directions = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
    projects = np.einsum("ij,ikj->ik", displacement, directions)
    winner = np.abs(projects).argmax(axis=1)
    rows = np.arange(len(left))
    signed = directions[rows, winner] * np.where(projects[rows, winner] >= 0, 1., -1.)[:, None]
    angle_difference = b - a
    required = (1. + np.abs(np.cos(angle_difference)) + np.abs(np.sin(angle_difference))) / 2
    gap = np.abs(projects[rows, winner]) - required
    return gap, signed, winner, angle_difference, displacement


def minimum_gap(points):
    i, j = np.triu_indices(len(points), k=1)
    return float(sat(points, i, j)[0].min())


def touching_pairs(points, cutoff):
    left, right = np.triu_indices(len(points), k=1)
    near = np.square(points[left, :2] - points[right, :2]).sum(axis=1) <= cutoff**2
    return left[near], right[near]


def derivatives(points, left, right):
    gap, axes, winner, delta, distance = sat(points, left, right)
    radius_derivative = (-np.sign(np.cos(delta))*np.sin(delta)
                         + np.sign(np.sin(delta))*np.cos(delta)) / 2
    rotation_derivative = axes[:, 0]*distance[:, 1] - axes[:, 1]*distance[:, 0]
    angle_left = radius_derivative + (winner < 2)*rotation_derivative
    angle_right = -radius_derivative + (winner >= 2)*rotation_derivative
    return gap, axes, angle_left, angle_right


def settle(points, length, iterations, radius=0.37):
    """Minimize true SAT penetration and wall overflow at a fixed box side."""
    p = normalize(points)
    h = extents(p[:, 2])[:, None]
    p[:, :2] += (length - (p[:, :2] + h).max(axis=0)) / 2
    n = len(p)
    left, right = touching_pairs(p, REACH + 2*math.sqrt(2.)*radius + 1e-6)
    def energy(flat):
        q = flat.reshape(-1, 3)
        gaps, axes, gl, gr = derivatives(q, left, right)
        overlaps = np.minimum(gaps, 0.)
        weight = 2*overlaps
        a = q[:, 2]
        c, s = np.cos(a), np.sin(a)
        h = (np.abs(c) + np.abs(s)) / 2
        dh = (-np.sign(c)*s + np.sign(s)*c) / 2
        lo = np.maximum(h[:, None] - q[:, :2], 0.)
        hi = np.maximum(q[:, :2] + h[:, None] - length, 0.)
        g = np.column_stack((2*(hi-lo), 2*(lo+hi).sum(axis=1)*dh))
        for k in (0, 1):
            term = weight*axes[:, k]
            g[:, k] += np.bincount(right, term, minlength=n) - np.bincount(left, term, minlength=n)
        g[:, 2] += (np.bincount(left, weight*gl, minlength=n)
                   + np.bincount(right, weight*gr, minlength=n))
        return float(np.square(overlaps).sum() + np.square(lo).sum() + np.square(hi).sum()), g.ravel()
    span = np.broadcast_to((radius, radius, .14), p.shape).ravel()
    x = p.ravel()
    result = minimize(energy, x, method="L-BFGS-B", jac=True,
                      bounds=list(zip(x-span, x+span)),
                      options={"maxiter": iterations, "ftol": 1e-15, "gtol": 3e-10})
    return result.x.reshape(-1, 3)


def lp_step(points, trust, angle_trust):
    """Linearize active separating axes and the enclosing square."""
    n = len(points)
    size = 3*n+2
    delta_side, violation = size-2, size-1
    length = side(points)
    left, right = touching_pairs(points, REACH + 2*math.sqrt(2.)*trust + 1e-6)
    gap, a, theta_left, theta_right = derivatives(points, left, right)
    pair_cols = np.column_stack((3*left, 3*left+1, 3*right, 3*right+1,
                                 3*left+2, 3*right+2, np.full(len(left), violation)))
    pair_data = np.column_stack((a, -a, -theta_left, -theta_right, -np.ones(len(left))))
    parts = [(pair_cols, pair_data, gap-MARGIN)]
    c, s = np.cos(points[:, 2]), np.sin(points[:, 2])
    h = (np.abs(c) + np.abs(s)) / 2
    dh = (-np.sign(c)*s + np.sign(s)*c) / 2
    idx = np.arange(n)
    for axis in (0, 1):
        for sign in (-1, 1):
            rhs = points[:, axis] - h if sign < 0 else length - points[:, axis] - h
            cols = np.column_stack((3*idx+axis, 3*idx+2, np.full(n, delta_side), np.full(n, violation)))
            data = np.column_stack((np.full(n, sign), dh, np.full(n, -(sign > 0)), -np.ones(n)))
            parts.append((cols, data, rhs))
    all_cols, all_vals, all_rows, all_rhs = [], [], [], []
    offset = 0
    for cols, data, rhs in parts:
        all_rows.extend(np.repeat(np.arange(len(rhs))+offset, cols.shape[1]))
        all_cols.extend(cols.ravel())
        all_vals.extend(data.ravel())
        all_rhs.extend(rhs)
        offset += len(rhs)
    matrix = coo_matrix((all_vals, (all_rows, all_cols)), shape=(offset, size)).tocsr()
    cost = np.zeros(size)
    cost[delta_side], cost[violation] = 1., 40.
    bounds = [(-trust, trust), (-trust, trust), (-angle_trust, angle_trust)]*n
    result = linprog(cost, A_ub=matrix, b_ub=all_rhs,
                     bounds=bounds+[(-2*trust, 2*trust), (0, None)], method="highs",
                     options={"primal_feasibility_tolerance": 1e-9, "dual_feasibility_tolerance": 1e-9})
    return result.x[:3*n].reshape(-1, 3) if result.success else None


def compact(points, rounds):
    p = normalize(points)
    def merit(q):
        return side(q) + 40*max(0., -minimum_gap(q))
    value = merit(p)
    for t in range(rounds):
        trust = max(.003, .09*.86**t)
        d = lp_step(p, trust, min(.025, trust/3))
        if d is None:
            break
        for rate in (1., .5, .25, .125):
            q = normalize(p+rate*d)
            new_value = merit(q)
            if new_value < value-1e-11:
                p, value = q, new_value
                break
    fixed = lp_step(p, .02, 0.)
    if fixed is not None:
        q = normalize(p+fixed)
        if merit(q) < value:
            p = q
    return p


def layout_rows(points, n):
    return [dict(zip(HEADER, (n*(n-1)//2+j, n, j, *map(repr, map(float, row)))))
            for j, row in enumerate(points)]


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADER, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def roundtrip(rows, pandas):
    body = io.StringIO()
    writer = csv.DictWriter(body, fieldnames=HEADER)
    writer.writeheader()
    writer.writerows(rows)
    body.seek(0)
    return list(csv.DictReader(io.StringIO(pandas.read_csv(body).to_csv(index=False))))


def certify(points, n, old, old_pandas):
    import pandas as pd
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"baseline/poc/src"))
    from squares._official_metric import evaluate_single_puzzle
    gap = minimum_gap(points)
    if gap < -1e-7:
        return None
    result = points.copy()
    result[:, :2] *= 1.0 + max(MARGIN, -gap*4 + MARGIN)
    rows = layout_rows(result, n)
    reference = evaluate_single_puzzle(n, old)
    reference_pandas = evaluate_single_puzzle(n, old_pandas)
    literal = evaluate_single_puzzle(n, rows)
    transported = evaluate_single_puzzle(n, roundtrip(rows, pd))
    if any(r[3] != "ok" for r in (literal, transported, reference, reference_pandas)):
        return None
    if Decimal(literal[2]) >= Decimal(reference[2]) or Decimal(transported[2]) >= Decimal(reference_pandas[2]):
        return None
    return rows, {"old_L": reference[2], "new_L": literal[2], "old_pandas_L": reference_pandas[2],
                  "new_pandas_L": transported[2], "score_gain": reference[1]-literal[1]}


def points_from(rows):
    return np.asarray([[float(r[k]) for k in ("x", "y", "angle")] for r in rows])


def seed_cost(points, target):
    q = normalize(points)
    q[:, :2] *= min(1., target/side(q))
    i, j = touching_pairs(q, REACH)
    gaps = sat(q, i, j)[0]
    return float(np.square(np.minimum(gaps, 0.)).sum())


def seeds(points, target, n, cuts, limit, known):
    p = normalize(points)
    p[:, 2] = (p[:, 2] + math.pi/4) % (math.pi/2) - math.pi/4
    axis_cuts = np.linspace(1.1, side(p)-1.1, cuts)
    layouts, tested, maximum_count = [], set(), 0
    combinations = [(x, y) for x in axis_cuts for y in axis_cuts]
    combinations += known
    for allowance, clearance in ((.15, .01), (.18, .025), (.12, .005)):
        first_layers = {y: insert_layer(p, y, 1, allowance, clearance) for y in sorted({y for _, y in combinations})}
        for x, y in combinations:
            q = insert_layer(first_layers[y], x, 0, allowance, clearance)
            maximum_count = max(maximum_count, len(q))
            if len(q) < n or len(q) > n+2:
                continue
            full_count = len(q)
            if full_count > n:
                # Only newly inserted squares can be deleted in this sweep.
                for _ in range(full_count-n):
                    i, j = touching_pairs(q, REACH)
                    depth = np.maximum(-sat(q, i, j)[0], 0.)
                    weights = np.bincount(np.r_[i, j], np.r_[depth, depth], minlength=len(q))
                    q = np.delete(q, len(p)+int(np.argmax(weights[len(p):])), axis=0)
            key = np.round(q, 10).tobytes()
            if key not in tested:
                tested.add(key)
                layouts.append((seed_cost(q, target), x, y, allowance, clearance, full_count, q))
    layouts.sort(key=lambda item: item[0])
    return layouts[:limit], {"cut_variants": 3*len(combinations), "max_count": maximum_count,
                             "unique_target_seeds": len(layouts)}


def search(args):
    experiments = [(int(a), int(b)) for a, b in (pair.split(":") for pair in args.family)]
    sizes = set(itertools.chain.from_iterable(experiments))
    with args.incumbent.open(encoding="utf-8", newline="") as source:
        groups = {}
        for n, rows in itertools.groupby(csv.DictReader(source), key=lambda row: int(row["n"])):
            if n > max(sizes):
                break
            if n in sizes:
                groups[n] = list(rows)
    report = {"incumbent_sha256": hashlib.sha256(args.incumbent.read_bytes()).hexdigest(),
              "cuts": args.cuts, "shortlist": args.shortlist, "relax_iters": args.relax_iters,
              "lp_rounds": args.lp_rounds, "families": []}
    wins = {}
    incumbent_pandas = None
    args.out.mkdir(parents=True, exist_ok=True)
    for donor, n in experiments:
        length = side(points_from(groups[n]))
        known = {(109, 132): [(3.48, 8.61)], (132, 157): [(9.71, 3.33)]}.get((donor, n), [])
        shortlist, counts = seeds(points_from(groups[donor]), length, n, args.cuts, args.shortlist, known)
        log = {"donor_n": donor, "target_n": n, "incumbent_side_float": length, **counts, "trials": []}
        print(f"{donor}->{n}: {counts}, testing {len(shortlist)}", flush=True)
        for k, (cost, x, y, allowance, clearance, full_count, seed) in enumerate(shortlist, 1):
            relaxed = settle(seed, length-.00005, args.relax_iters)
            final = compact(relaxed, args.lp_rounds)
            measured, gap = side(final), minimum_gap(final)
            item = {"x": float(x), "y": float(y), "copy_overlap_limit": allowance,
                    "shift_clearance": clearance, "seed_count": full_count, "seed_penalty": cost,
                    "side_float": measured, "minimum_gap_float": gap}
            log["trials"].append(item)
            if measured < length-1e-9 and gap >= -1e-7:
                if incumbent_pandas is None:
                    import pandas as pd
                    frame = pd.read_csv(args.incumbent)
                    text = frame[frame.n.isin(sizes)].to_csv(index=False)
                    incumbent_pandas = {k: list(rs) for k, rs in itertools.groupby(
                        csv.DictReader(io.StringIO(text)), key=lambda r: int(r["n"]))}
                checked = certify(final, n, groups[n], incumbent_pandas[n])
                if checked:
                    rows, details = checked
                    item["verified"] = details
                    if n not in wins or Decimal(details["new_L"]) < Decimal(wins[n][1]["new_L"]):
                        wins[n] = (rows, details)
            print(f"  {k}/{len(shortlist)} L={measured:.9f} min_gap={gap:+.3e}"
                  f"{' verified win' if 'verified' in item else ''}", flush=True)
        report["families"].append(log)
        report["wins"] = {str(n): details for n, (_, details) in wins.items()}
        (args.out/"scan.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
        if wins:
            write_csv(args.out/"improved_rows.csv", (r for n in sorted(wins) for r in wins[n][0]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--incumbent", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--family", nargs="+", default=["157:184", "183:212", "211:242"])
    parser.add_argument("--cuts", type=int, default=23)
    parser.add_argument("--shortlist", type=int, default=10)
    parser.add_argument("--relax-iters", type=int, default=480)
    parser.add_argument("--lp-rounds", type=int, default=18)
    search(parser.parse_args())
