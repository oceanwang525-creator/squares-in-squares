"""Contract a larger packing across two internal seams, preserving rigid pieces."""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from decimal import Decimal, localcontext
import hashlib
import json
import math
from pathlib import Path
import sys
from time import perf_counter
import warnings

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[3]
DATA = Path(__file__).resolve().parents[1] / "data"
INCUMBENT = ROOT / "poc/pell_corrugation/data/border-scan/merged.csv"
FRONTIER = ROOT / "poc/frontier_nov12/docs/per_n_gap.csv"
sys.path.insert(0, str(ROOT / "poc/baseline/poc/src"))
from squares.core import certify, kaggle_roundtrip, write_submission

EPS = 2e-12
STRICT = 2e-8
TOP_GAPS = 24
SEAM_OFFSETS = (0.25, 0.5, 0.75)


@dataclass(frozen=True)
class Cut:
    start: float
    survivors: int
    count: int


@dataclass(frozen=True)
class Trial:
    donor: int
    target: int
    nominal: float
    x_width: float
    y_width: float
    x: Cut
    y: Cut

    @property
    def kept(self):
        return self.x.survivors & self.y.survivors

    @property
    def capacity(self):
        return self.kept.bit_count()


def load_source(path):
    groups = [[] for _ in range(1001)]
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        if next(reader) != ["id", "n", "square_id", "x", "y", "angle"]:
            raise ValueError("Unexpected submission header")
        for i, row in enumerate(reader):
            index, n, sid = map(int, row[:3])
            if index != i or not 1 <= n <= 1000 or sid != len(groups[n]):
                raise ValueError(f"Unexpected puzzle or square ID at row {i}")
            groups[n].append(tuple(row[3:]))
    if any(len(group) != n for n, group in enumerate(groups)):
        raise ValueError("Incomplete donor file")
    return groups


def certify_baseline(n, source):
    rows = [dict(square_id=str(i), x=x, y=y, angle=angle)
            for i, (x, y, angle) in enumerate(source)]
    return Decimal(certify(n, rows)), Decimal(certify(n, kaggle_roundtrip({n: rows})[n]))


def extents(geometry):
    h = (abs(np.cos(geometry[:, 2])) + abs(np.sin(geometry[:, 2])))[:, None] / 2
    return geometry[:, :2] - h, geometry[:, :2] + h


def length(geometry):
    lo, hi = extents(geometry)
    return float(np.max(hi.max(0) - lo.min(0)))


def axis_cuts(lo, hi, width, min_count):
    n = len(lo)
    min_side = max(1, n // 20)
    all_bits = (1 << n) - 1
    if width <= EPS:
        return [Cut(float(hi.max()) + 1, all_bits, n)]
    left_order = np.argsort(hi)
    right_ends = lo - width
    right_order = np.argsort(right_ends)
    events = np.unique(np.concatenate((hi, right_ends)))
    left = 0
    right = all_bits
    li = ri = 0
    cuts = {}
    for start in events:
        while li < n and hi[left_order[li]] <= start + EPS:
            left |= 1 << int(left_order[li])
            li += 1
        while ri < n and right_ends[right_order[ri]] < start - EPS:
            right ^= 1 << int(right_order[ri])
            ri += 1
        if li < min_side or n - ri < min_side:
            continue
        keep = left | right
        count = keep.bit_count()
        if count >= min_count:
            cuts.setdefault(keep, Cut(float(start), keep, count))
    return sorted(cuts.values(), key=lambda cut: -cut.count)


def best_pair(donor, target, width_x, width_y, nominal, lo, hi, near):
    minimum = target - near
    cuts_x = axis_cuts(lo[:, 0], hi[:, 0], width_x, minimum)
    cuts_y = axis_cuts(lo[:, 1], hi[:, 1], width_y, minimum)
    if len(cuts_x) * len(cuts_y) >= 16_384:
        # The event indices at which one square is deleted on an axis form an
        # interval. Its joint deletions form a rectangle in the cut-pair table.
        cuts = [sorted(group, key=lambda cut: cut.start) for group in (cuts_x, cuts_y)]
        first, last = [], []
        for axis, width in enumerate((width_x, width_y)):
            starts = np.array([cut.start for cut in cuts[axis]])
            first.append(np.searchsorted(starts - EPS, lo[:, axis] - width, side="right"))
            last.append(np.searchsorted(starts + EPS, hi[:, axis], side="left"))
        active = (first[0] < last[0]) & (first[1] < last[1])
        x0, y0 = (limits[active] for limits in first)
        x1, y1 = (limits[active] for limits in last)
        difference = np.zeros((len(cuts_x) + 1, len(cuts_y) + 1), dtype=np.int16)
        np.add.at(difference, (x0, y0), 1)
        np.add.at(difference, (x0, y1), -1)
        np.add.at(difference, (x1, y0), -1)
        np.add.at(difference, (x1, y1), 1)
        joint = difference.cumsum(0, dtype=np.int16).cumsum(1, dtype=np.int16)[:-1, :-1]
        counts_x, counts_y = (np.array([cut.count for cut in group], dtype=np.int16)
                              for group in cuts)
        capacity = counts_x[:, None] + counts_y[None, :] - donor + joint
        i, j = np.unravel_index(capacity.argmax(), capacity.shape)
        trial = Trial(donor, target, nominal, width_x, width_y, cuts[0][i], cuts[1][j])
        if trial.capacity != capacity[i, j]:
            raise RuntimeError("Cut event counts disagree with survivor masks")
        return trial if trial.capacity >= minimum else None

    best = None
    best_count = minimum - 1
    for x in cuts_x:
        if x.count <= best_count:
            break
        for y in cuts_y:
            if y.count <= best_count:
                break
            count = (x.survivors & y.survivors).bit_count()
            if count > best_count:
                best = Trial(donor, target, nominal, width_x, width_y, x, y)
                best_count = count
    return best


def center_shifts(trial, geometry, off_x, off_y):
    right = geometry[:, 0] > trial.x.start + off_x * trial.x_width
    upper = geometry[:, 1] > trial.y.start + off_y * trial.y_width
    moved = geometry.copy()
    moved[right, 0] -= trial.x_width
    moved[upper, 1] -= trial.y_width
    return moved, right, upper


def crossing_collisions(moved, right, upper):
    pairs = cKDTree(moved[:, :2]).query_pairs(math.sqrt(2.0) + EPS, output_type="ndarray")
    i, j = pairs.T
    across = (right[i] != right[j]) | (upper[i] != upper[j])
    i, j = i[across], j[across]
    dx, dy = (moved[i, :2] - moved[j, :2]).T
    angle = moved[:, 2]
    c, s = np.cos(angle), np.sin(angle)
    diff = angle[i] - angle[j]
    width = (1 + abs(np.cos(diff)) + abs(np.sin(diff))) / 2
    projection = np.maximum.reduce(
        (abs(dx*c[i] + dy*s[i]), abs(dy*c[i] - dx*s[i]),
         abs(dx*c[j] + dy*s[j]), abs(dy*c[j] - dx*s[j]))
    )
    overlap = projection < width - EPS
    return np.column_stack((i[overlap], j[overlap]))


def keep_after_conflicts(size, edges):
    """Find the maximum independent set in a bipartite seam, or solve a small MILP."""
    active = np.unique(edges)
    adjacency = {int(v): [] for v in active}
    for i, j in edges.tolist():
        adjacency[i].append(j)
        adjacency[j].append(i)
    color = {}
    bipartite = True
    for node in adjacency:
        if node in color:
            continue
        color[node] = 0
        stack = [node]
        while stack:
            current = stack.pop()
            for neighbor in adjacency[current]:
                if neighbor not in color:
                    color[neighbor] = 1 - color[current]
                    stack.append(neighbor)
                elif color[neighbor] == color[current]:
                    bipartite = False
    keep = np.ones(size, dtype=bool)
    if not len(edges):
        return keep, False
    if not bipartite:
        row = np.repeat(np.arange(len(edges)), 2)
        col = np.searchsorted(active, edges.ravel())
        matrix = coo_matrix((np.ones(len(row)), (row, col)), shape=(len(edges), len(active))).tocsc()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Unrecognized options detected")
            result = milp(
                -np.ones(len(active)), integrality=np.ones(len(active)),
                bounds=Bounds(0, 1), constraints=LinearConstraint(matrix, 0, 1),
                options={"time_limit": 0.25, "threads": 1},
            )
        if result.x is None:
            return None, True
        keep[active] = result.x > 0.5
        if np.any(keep[edges[:, 0]] & keep[edges[:, 1]]):
            raise RuntimeError("Conflict solver returned adjacent survivors")
        return keep, True

    left = sorted(v for v, value in color.items() if value == 0)
    right = sorted(v for v, value in color.items() if value == 1)
    right_index = {v: i for i, v in enumerate(right)}
    row = [i for i, v in enumerate(left) for _ in adjacency[v]]
    col = [right_index[w] for v in left for w in adjacency[v]]
    graph = csr_matrix((np.ones(len(row)), (row, col)), shape=(len(left), len(right)))
    matches = maximum_bipartite_matching(graph, perm_type="column")
    owners = {right[j]: left[i] for i, j in enumerate(matches) if j >= 0}
    matching = {left[i]: right[j] for i, j in enumerate(matches) if j >= 0}
    reachable_left = set(v for v in left if v not in matching)
    reachable_right = set()
    stack = list(reachable_left)
    while stack:
        vertex = stack.pop()
        for neighbor in adjacency[vertex]:
            if neighbor == matching.get(vertex) or neighbor in reachable_right:
                continue
            reachable_right.add(neighbor)
            owner = owners.get(neighbor)
            if owner is not None and owner not in reachable_left:
                reachable_left.add(owner)
                stack.append(owner)
    cover = (set(left) - reachable_left) | reachable_right
    keep[list(cover)] = False
    if np.any(keep[edges[:, 0]] & keep[edges[:, 1]]):
        raise RuntimeError("Matching cover leaves an unresolved conflict")
    return keep, False


def integer_trials(geometries, side, minimum_donor, maximum_cut, near):
    for donor in range(minimum_donor, 1001):
        lo, hi = extents(geometries[donor])
        original = float((hi.max(0) - lo.min(0)).max())
        for width in range(1, maximum_cut + 1):
            nominal = original - width
            if nominal <= 0:
                continue
            upper = min(donor-1, int(nominal*nominal + EPS))
            targets = np.flatnonzero(side[:upper+1] > nominal + STRICT)
            if len(targets) == 0:
                continue
            target = int(targets[0])
            yield best_pair(donor, target, width, width, nominal, lo, hi, near)


def frontier_trials(geometries, side, minimum_donor, near):
    with FRONTIER.open(newline="", encoding="utf-8") as handle:
        targets = sorted(csv.DictReader(handle), key=lambda r: -float(r["gap_score"]))
    targets = [int(r["n"]) for r in targets if float(r["gap_score"]) > STRICT][:TOP_GAPS]
    bounds = [None] + [extents(geometry) for geometry in geometries[1:]]
    spans = np.array([[0.0, 0.0]] + [hi.max(0) - lo.min(0) for lo, hi in bounds[1:]])
    for target in targets:
        nominal = side[target] - STRICT
        widths = spans - nominal
        area_difference = np.prod(spans, axis=1) - nominal**2
        counts = np.arange(len(spans))
        possible = (
            (counts >= max(target + 1, minimum_donor))
            & (widths.min(1) >= 0.55) & (widths.max(1) <= 2.35)
            & (abs(counts - target - area_difference) <= 12)
        )
        for donor in np.flatnonzero(possible):
            lo, hi = bounds[donor]
            yield best_pair(
                int(donor), target, *map(float, widths[donor]), float(nominal), lo, hi, near,
            )


def trim_indices(indices, moved, count):
    """Drop extreme survivors greedily when the seam has spare capacity."""
    indices = indices.copy()
    lo, hi = extents(moved)
    while len(indices) > count:
        extremes = np.unique(np.concatenate((indices[np.argmin(lo[indices], 0)],
                                             indices[np.argmax(hi[indices], 0)])))
        def removal_side(square):
            keep = indices[indices != square]
            return (hi[keep].max(0) - lo[keep].min(0)).max()
        drop = min(extremes, key=removal_side)
        indices = indices[indices != drop]
    return indices


def certify_trial(n, source, trial, indices, right, upper, baseline):
    with localcontext() as ctx:
        ctx.prec = 60
        cut_x = Decimal(str(trial.x_width))
        cut_y = Decimal(str(trial.y_width))
        coordinates = [
            (Decimal(source[i][0]) - (cut_x if right[i] else 0),
             Decimal(source[i][1]) - (cut_y if upper[i] else 0), source[i][2])
            for i in indices
        ]
        x0, y0, _ = coordinates[0]
        for slack in ("0", "1e-14", "1e-13", "5e-13", "1e-12", "5e-12"):
            factor = Decimal(1) + Decimal(slack)
            rows = [
                dict(square_id=str(i), x=str((x-x0)*factor), y=str((y-y0)*factor), angle=a)
                for i, (x, y, a) in enumerate(coordinates)
            ]
            try:
                measured = certify(n, rows)
                if Decimal(measured) >= baseline[0]:
                    return None
                measured_pandas = certify(n, kaggle_roundtrip({n: rows})[n])
                if Decimal(measured_pandas) >= baseline[1]:
                    continue
                return rows, measured, measured_pandas, slack
            except ValueError:
                continue
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-donor", type=int, default=100)
    parser.add_argument("--max-cut", type=int, default=4)
    parser.add_argument("--near", type=int, default=5)
    parser.add_argument("--graph-cases", type=int, default=48)
    parser.add_argument("--output", type=Path, default=DATA)
    args = parser.parse_args()
    if not 1 <= args.min_donor <= 1000 or min(args.max_cut, args.near, args.graph_cases) < 0:
        parser.error("Invalid donor range or negative search limit")
    started = perf_counter()
    source = load_source(INCUMBENT)
    geometries = [np.array(rows, dtype=float) for rows in source]
    side = np.array([0.0] + [length(geometry) for geometry in geometries[1:]])
    print(f"Loaded {len(geometries)-1} donor puzzles", flush=True)
    trials = []
    stats = {}
    streams = (
        ("integer", integer_trials(geometries, side, args.min_donor, args.max_cut, args.near)),
        ("frontier", frontier_trials(geometries, side, args.min_donor, args.near)),
    )
    for name, stream in streams:
        count, close = 0, []
        for trial in stream:
            count += 1
            if trial is not None:
                close.append(trial)
            if count % 500 == 0:
                print(name, count, "cut-width pairs", len(close), "within capacity threshold", flush=True)
        trials.extend(close)
        stats[name] = {
            "cut_width_pairs": count, "within_near_count": len(close),
            "capacity_reaches_target": sum(t.capacity >= t.target for t in close),
            "seconds_so_far": round(perf_counter() - started, 3),
        }
        print(name, stats[name], flush=True)

    def rank(trial):
        gain = (side[trial.target] - trial.nominal) / math.sqrt(trial.target)
        return trial.target - trial.capacity, -gain

    trials.sort(key=rank)
    graph_count = milp_count = proposals = 0
    near_records = {}
    baselines = {}
    baseline_rejections = {}
    wins = {}
    for index, trial in enumerate(trials):
        geometry = geometries[trial.donor]
        moved, right, upper = center_shifts(trial, geometry, 0.5, 0.5)
        keep = np.array([(trial.kept >> i) & 1 for i in range(trial.donor)], dtype=bool)
        options = [(int(keep.sum()), moved, right, upper, keep, 0.5, 0.5)]
        if index < args.graph_cases:
            donor_lo, donor_hi = extents(geometry)
            box_lo = donor_lo.min(0)
            box_hi = donor_hi.max(0) - [trial.x_width, trial.y_width]
            for ox in SEAM_OFFSETS:
                for oy in SEAM_OFFSETS:
                    moved, right, upper = center_shifts(trial, geometry, ox, oy)
                    edges = crossing_collisions(moved, right, upper)
                    moved_lo, moved_hi = extents(moved)
                    inside = np.all((moved_lo >= box_lo - EPS) & (moved_hi <= box_hi + EPS), axis=1)
                    edges = edges[inside[edges].all(1)]
                    keep, used_milp = keep_after_conflicts(trial.donor, edges)
                    graph_count += 1
                    milp_count += used_milp
                    if keep is not None:
                        keep &= inside
                        options.append((int(keep.sum()), moved, right, upper, keep, ox, oy))
        options.sort(key=lambda option: -option[0])
        count, *_ = options[0]
        note = dict(
            donor=trial.donor, target=trial.target, incumbent_side=side[trial.target],
            nominal=trial.nominal, safe_capacity=trial.capacity, best_capacity=count,
            width=[trial.x_width, trial.y_width],
            start=[trial.x.start, trial.y.start],
        )
        old = near_records.get(trial.target)
        if old is None or count > old["best_capacity"]:
            near_records[trial.target] = note
        for count, moved, right, upper, keep, ox, oy in options:
            if count < trial.target:
                break
            indices = np.flatnonzero(keep)
            upper_n = min(count, trial.donor-1)
            for n in range(trial.target, upper_n+1):
                chosen = trim_indices(indices, moved, n)
                estimated = length(moved[chosen])
                best_side = float(wins[n][1]["literal"]) if n in wins else side[n]
                if estimated > best_side - STRICT or n in baseline_rejections:
                    continue
                if n not in baselines:
                    try:
                        baselines[n] = certify_baseline(n, source[n])
                    except ValueError as error:
                        baseline_rejections[n] = str(error)
                        continue
                baseline = baselines[n]
                proposals += 1
                candidate = certify_trial(n, source[trial.donor], trial, chosen, right, upper, baseline)
                if candidate is None:
                    continue
                rows, exact, transported, slack = candidate
                if float(exact) < best_side:
                    details = dict(note, n=n, literal=exact, pandas=transported, dilation=slack,
                                   offsets=[ox, oy], literal_baseline=str(baseline[0]),
                                   pandas_baseline=str(baseline[1]))
                    wins[n] = rows, details
                    print("strict dual win", details, flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "incumbent_sha256": hashlib.sha256(INCUMBENT.read_bytes()).hexdigest(),
        "settings": dict(vars(args), output=str(args.output.name)),
        "scans": stats, "graph_configurations": graph_count, "non_bipartite_configurations": milp_count,
        "certification_attempts": proposals,
        "baseline_rejections": baseline_rejections,
        "best_capacity_by_target": sorted(
            near_records.values(), key=lambda r: (r["target"] - r["best_capacity"], r["target"])
        )[:40],
        "wins": [details for _, details in wins.values()],
        "seconds": round(perf_counter()-started, 3),
    }
    (args.output / "scan.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if wins:
        write_submission(args.output / "candidates.csv", {n: rows for n, (rows, _) in wins.items()})
    print(f"{graph_count} graph variants, {proposals} certification attempts, {len(wins)} wins, "
          f"{report['seconds']} seconds", flush=True)


if __name__ == "__main__":
    main()
