"""Replant an occupied upright/tilted interface patch from a neighboring puzzle."""

import argparse
import csv
import hashlib
import io
import json
import math
import os
import sys
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
from scipy.optimize import linear_sum_assignment, minimize
from scipy.spatial import cKDTree

POC = Path(__file__).resolve().parents[1]
ROOT = POC.parents[1]
INCUMBENT = ROOT / 'poc/pell_corrugation/data/border-scan/merged.csv'
FIELDS = ('id', 'n', 'square_id', 'x', 'y', 'angle')
DEFAULT_TARGETS = (908, 910, 911, 969, 971, 972, 973)


def fold(angle):
    return (angle + np.pi / 4) % (np.pi / 2) - np.pi / 4


def extent(angle):
    return (np.abs(np.cos(angle)) + np.abs(np.sin(angle))) / 2


def normalize(packing):
    q = packing.copy()
    h = extent(q[:, 2])[:, None]
    q[:, :2] -= (q[:, :2] - h).min(axis=0)
    return q


def side(packing):
    h = extent(packing[:, 2])[:, None]
    return float(np.max((packing[:, :2] + h).max(axis=0) - (packing[:, :2] - h).min(axis=0)))


def read_selected(path, ns):
    groups = {n: [] for n in ns}
    with path.open(newline='', encoding='utf-8-sig') as file:
        for row in csv.DictReader(file):
            n = int(row['n'])
            if n in groups:
                groups[n].append(row)
    for n, rows in groups.items():
        if len(rows) != n or {int(row['square_id']) for row in rows} != set(range(n)):
            raise ValueError(f'Incomplete group n={n}')
    return groups


def from_rows(rows):
    q = np.array([[float(row[k]) for k in FIELDS[3:]] for row in rows])
    q[:, 2] = fold(q[:, 2])
    return normalize(q)


def frame(packing, sx, sy, swap=False):
    q = packing.copy()
    q[:, :2] *= (sx, sy)
    q[:, 2] *= sx * sy
    if swap:
        q[:, :2] = q[:, 1::-1]
        q[:, 2] *= -1
    return normalize(q)


def gap_values(q, i, j):
    dx, dy = (q[i, :2] - q[j, :2]).T
    a, b = q[i, 2], q[j, 2]
    ax = np.stack((np.cos(a), -np.sin(a), np.cos(b), -np.sin(b)), axis=1)
    ay = np.stack((np.sin(a), np.cos(a), np.sin(b), np.cos(b)), axis=1)
    projection = dx[:, None] * ax + dy[:, None] * ay
    selected = np.abs(projection).argmax(axis=1)
    indices = np.arange(len(i))
    signed = projection[indices, selected]
    ax, ay = ax[indices, selected], ay[indices, selected]
    da = a - b
    sine, cosine = np.sin(da), np.cos(da)
    radius = (1 + np.abs(sine) + np.abs(cosine)) / 2
    radius_derivative = (np.sign(sine) * cosine - np.sign(cosine) * sine) / 2
    return np.abs(signed) - radius, np.sign(signed), ax, ay, dx, dy, selected, radius_derivative


def pair_indices(q, max_move):
    radius = math.sqrt(2) * (1 + 2 * max_move) + 1e-9
    return cKDTree(q[:, :2]).query_pairs(radius, output_type='ndarray').T


@dataclass
class Patch:
    packing: np.ndarray
    indices: np.ndarray
    anchor: np.ndarray
    grid_size: int
    corner: tuple
    contact: tuple
    tree: cKDTree

    @property
    def relative(self):
        q = self.packing[self.indices].copy()
        q[:, :2] -= self.anchor
        return q


def corner_patches(packing, k, per_corner, min_grid=4):
    """Use touching tilt/upright pairs where the tilt enters an occupied corner grid."""
    found = []
    for sx, sy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        q = frame(packing, sx, sy)
        tree = cKDTree(q[:, :2])
        upright = np.abs(q[:, 2]) < .014
        grid_xy = np.rint(q[:, :2] - .5).astype(int)
        residual = np.max(np.abs(q[:, :2] - (grid_xy + .5)), axis=1)
        lattice = upright & (residual < .055) & (grid_xy >= 0).all(axis=1)
        grid_side = 0
        limit = math.floor(side(q))
        for width in range(min_grid, limit + 1):
            inside = (grid_xy < width).all(axis=1)
            if np.count_nonzero(lattice & inside) >= .93 * width**2:
                grid_side = width
        if grid_side == 0:
            continue
        grid = lattice & (grid_xy < grid_side).all(axis=1)
        i, j = pair_indices(q, 0)
        i, j = np.where(grid[i], i, j), np.where(grid[i], j, i)
        tilt = np.abs(q[:, 2]) > .09
        h = extent(q[:, 2])
        edge = np.max(grid_xy, axis=1) >= grid_side - 3
        intersects = (q[:, :2] - h[:, None] < grid_side - .012).all(axis=1)
        touching = gap_values(q, i, j)[0] <= .035
        contacts = np.flatnonzero(grid[i] & edge[i] & tilt[j] & intersects[j] & touching)
        if not len(contacts):
            continue
        # Favor the inner corner, then cover distinct points on its two edges.
        tip = np.full(2, grid_side - .5)
        order = contacts[np.argsort(np.linalg.norm(q[i[contacts], :2] - tip, axis=1))]
        anchors = []
        for index in order:
            a, b = i[index], j[index]
            anchor = q[a, :2]
            if any(np.linalg.norm(anchor - other) < 1.6 for other in anchors):
                continue
            middle = (anchor + q[b, :2]) / 2
            nearest = np.argsort(np.linalg.norm(q[:, :2] - middle, axis=1))[:k]
            if a not in nearest or b not in nearest:
                continue
            found.append(Patch(q, nearest, anchor, grid_side, (sx, sy), (int(a), int(b)), tree))
            anchors.append(anchor)
            if len(anchors) == per_corner:
                break
    return found


def transplant(recipient, donor, swap):
    original = recipient.relative
    patch = donor.relative
    if swap:
        patch[:, :2] = patch[:, 1::-1]
        patch[:, 2] *= -1
    delta = original[:, None, :2] - patch[None, :, :2]
    angle = fold(original[:, None, 2] - patch[None, :, 2])
    ii, jj = linear_sum_assignment(np.sum(delta**2, axis=2) + .2 * angle**2)
    difference = float(np.max(np.linalg.norm(delta[ii, jj], axis=1)))
    angle_change = float(np.max(np.abs(angle[ii, jj])))
    if difference < .12 and angle_change < .08:
        return None
    patch[:, :2] += recipient.anchor
    q = recipient.packing.copy()
    q[recipient.indices[ii]] = patch[jj]
    near = recipient.tree.query_ball_point(q[recipient.indices, :2], math.sqrt(2) + 1e-9)
    replaced = set(recipient.indices)
    pairs = [(a, b) for a, neighbors in zip(recipient.indices, near) for b in neighbors if b not in replaced]
    i, j = np.array(pairs, dtype=int).reshape(-1, 2).T
    gap = gap_values(q, i, j)[0]
    penetration = np.maximum(-gap, 0)
    if penetration.max(initial=0) > .48 or side(q) > side(recipient.packing) + .22:
        return None
    rank = float(penetration @ penetration) + .0005 * difference
    return rank, q, difference, angle_change


def optimize(seed, active, length, move, angle_move, iterations, target_gap):
    """Solve exact SAT overlap penalties with analytic gradients and bounded motion."""
    n = len(seed)
    i, j = pair_indices(seed, move)
    i, j = i[active[i] | active[j]], j[active[i] | active[j]]
    ids = np.flatnonzero(active)
    center = seed[ids]
    change = np.full((len(ids), 3), move)
    change[:, 2] = angle_move
    pull = 2e-10

    def energy(flat):
        q = seed.copy()
        q[ids] = flat.reshape(-1, 3)
        gap, direction, ax, ay, dx, dy, selected, dr = gap_values(q, i, j)
        penetration = np.maximum(target_gap - gap, 0)
        scale = 2 * penetration
        sx, sy = scale * direction * ax, scale * direction * ay
        dtheta = direction * (-dx * ay + dy * ax)
        gradient = np.empty((n, 3))
        gradient[:, 0] = np.bincount(j, sx, n) - np.bincount(i, sx, n)
        gradient[:, 1] = np.bincount(j, sy, n) - np.bincount(i, sy, n)
        gradient[:, 2] = (np.bincount(i, scale * (dr - dtheta * (selected < 2)), n)
                          + np.bincount(j, scale * (-dr - dtheta * (selected >= 2)), n))
        c, s = np.cos(q[:, 2]), np.sin(q[:, 2])
        h = (np.abs(c) + np.abs(s)) / 2
        dh = (np.sign(s) * c - np.sign(c) * s) / 2
        lower = np.maximum(h[:, None] - q[:, :2], 0)
        upper = np.maximum(q[:, :2] + h[:, None] - length, 0)
        gradient[:, :2] += 2 * (upper - lower)
        gradient[:, 2] += 2 * dh * (upper + lower).sum(axis=1)
        delta = q[ids] - center
        loss = penetration @ penetration + np.sum(lower**2 + upper**2) + pull * np.sum(delta**2)
        return float(loss), (gradient[ids] + 2 * pull * delta).ravel()

    bounds = list(zip((center - change).ravel(), (center + change).ravel()))
    opt = minimize(energy, center.ravel(), method='L-BFGS-B', jac=True, bounds=bounds,
                   options={'maxiter': iterations, 'gtol': 2e-10, 'ftol': 2e-15, 'maxls': 25})
    q = seed.copy()
    q[ids] = opt.x.reshape(-1, 3)
    return q, float(opt.fun), int(opt.nit)


def residual(q):
    i, j = pair_indices(q, 0)
    return max(0., -float(gap_values(q, i, j)[0].min(initial=0)))


def harden(q):
    # Center dilation improves every separating projection by at least the dilation.
    increase = (residual(q) + 2e-12) * 1.05
    q = q.copy()
    q[:, :2] *= 1 + increase
    return normalize(q)


def finish(q, polish):
    hardened = harden(q)
    if not polish:
        return hardened, None
    fixed = polish(len(q), q[:, :2], q[:, 2], rounds=3, time_limit=4.0)
    if fixed is None:
        return hardened, None
    projected = q.copy()
    projected[:, :2] = fixed[1]
    projected = harden(projected)
    lp_side = side(projected)
    return (projected if lp_side < side(hardened) else hardened), lp_side


def cert_pair(n, rows):
    import pandas as pd
    sys.path.insert(0, str(ROOT / 'poc/baseline/poc/src'))
    from squares._official_metric import evaluate_single_puzzle

    literal = evaluate_single_puzzle(n, rows)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(rows)
    readback = pd.read_csv(io.StringIO(buffer.getvalue())).to_csv(index=False)
    transported = evaluate_single_puzzle(n, list(csv.DictReader(io.StringIO(readback))))
    return literal, transported


def winner(n, q, incumbent, out):
    rows = []
    for i, values in enumerate(q):
        r = dict(id=n*(n-1)//2 + i, n=n, square_id=i)
        r.update(zip(FIELDS[3:], (repr(float(v)) for v in values)))
        rows.append(r)
    control = cert_pair(n, incumbent)
    result = cert_pair(n, rows)
    if any(ref[3] != 'ok' or new[3] != 'ok' or Decimal(new[2]) >= Decimal(ref[2])
           for ref, new in zip(control, result)):
        return None
    with (out / f'n{n}.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return {'literal': result[0][2], 'pandas': result[1][2],
            'baseline_literal': control[0][2], 'baseline_pandas': control[1][2]}


def searches(n, packing, patches, args, polish):
    original = side(packing)
    trials = []
    for size in args.patch_sizes:
        recipient_patches = patches[n, size]
        offered = []
        for index, recipient in enumerate(recipient_patches):
            for other in args.donors[n]:
                for donor in patches[other, size]:
                    for swap in (False, True):
                        seed = transplant(recipient, donor, swap)
                        if seed is not None:
                            offered.append((seed[0], index, other, swap, recipient, donor, seed))
        offered.sort(key=lambda item: item[0])
        used = set()
        for rank, index, other, swap, recipient, donor, seed in offered:
            key = (index, other)
            if key in used:
                continue
            used.add(key)
            _, q, difference, angle_change = seed
            active = np.any(np.linalg.norm(q[:, None, :2] - q[recipient.indices][None, :, :2], axis=2) < 1.65,
                            axis=1)
            repaired, local_loss, local_iters = optimize(q, active, original, .27, .20, 350, 1e-8)
            local_residual = residual(repaired)
            # The exact same wall shrink is used for the incumbent control.
            target = original - args.shrink
            repaired[:, :2] += (target - original) * .5
            whole, loss, iterations = optimize(repaired, np.ones(n, bool), target, .16, .11, args.iterations, 4e-9)
            q, lp_side = finish(whole, polish)
            trials.append((q, {'n': n, 'donor': other, 'patch_size': size, 'recipient_corner': recipient.corner,
                           'donor_corner': donor.corner, 'recipient_grid_size': recipient.grid_size,
                           'donor_grid_size': donor.grid_size, 'recipient_contact': recipient.contact,
                           'donor_contact': donor.contact, 'swap': swap, 'seed_rank': rank,
                           'seed_move': difference, 'seed_angle_change': angle_change,
                           'halo_size': int(active.sum()), 'local_loss': local_loss, 'local_iterations': local_iters,
                           'local_residual': local_residual, 'loss': loss, 'iterations': iterations,
                           'raw_residual': residual(whole), 'lp_side': lp_side, 'candidate_side': side(q),
                           'baseline_side': original, 'target_side': target}))
            if len(used) == args.trials:
                break
    return trials


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, default=INCUMBENT)
    parser.add_argument('--out', type=Path, default=POC / 'data')
    parser.add_argument('--targets', type=int, nargs='+', default=DEFAULT_TARGETS)
    parser.add_argument('--patch-sizes', type=int, nargs='+', default=(9, 16))
    parser.add_argument('--trials', type=int, default=2, help='number of attempted donors per patch size and target')
    parser.add_argument('--iterations', type=int, default=420)
    parser.add_argument('--shrink', type=float, default=.001)
    parser.add_argument('--polish', action='store_true', help='reuse the repository fixed-angle LP after the repair')
    args = parser.parse_args()
    polish = None
    if args.polish:
        sys.path.insert(0, str(ROOT / 'poc/alln_fixed_lp/poc'))
        from alln_fixed_lp import refine as polish
    args.donors = {n: sorted({n + step for step in (-63,-62,-61,-2,-1,1,2,61,62,63)
                            if 1 <= n + step <= 1000}) for n in args.targets}
    ns = set(args.targets) | {d for donors in args.donors.values() for d in donors}
    args.out.mkdir(parents=True, exist_ok=True)
    originals = read_selected(args.baseline, ns)
    packings = {n: from_rows(rows) for n, rows in originals.items()}
    patches = {(n, size): corner_patches(q, size, per_corner=3)
               for n, q in packings.items() for size in args.patch_sizes}
    report = {'baseline_sha256': hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
              'configuration': {'targets': args.targets, 'patch_sizes': args.patch_sizes, 'trials': args.trials,
                                'iterations': args.iterations, 'shrink': args.shrink, 'polish': args.polish},
              'groups': []}
    report_name = 'results_lp.json' if args.polish else 'results.json'
    for n in args.targets:
        p = packings[n]
        target = side(p) - args.shrink
        seed = p.copy()
        seed[:, :2] -= args.shrink * .5
        control, loss, iterations = optimize(seed, np.ones(n, bool), target, .16, .11, args.iterations, 4e-9)
        trials = searches(n, p, patches, args, polish)
        final_control, control_lp_side = finish(control, polish)
        group = {'n': n, 'baseline_side': side(p), 'control_side': side(final_control), 'control_loss': loss,
                 'control_lp_side': control_lp_side,
                 'control_residual': residual(control), 'control_iterations': iterations,
                 'recipient_patch_counts': {size: len(patches[n, size]) for size in args.patch_sizes},
                 'trials': [r for _, r in trials], 'win': None}
        for q, row in sorted(trials, key=lambda item: item[1]['candidate_side']):
            if row['candidate_side'] >= side(p) - 1e-9:
                break
            win = winner(n, q, originals[n], args.out)
            if win:
                group['win'] = {**win, 'donor': row['donor'], 'patch_size': row['patch_size']}
                break
        report['groups'].append(group)
        (args.out / report_name).write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        best = min((r['candidate_side'] for _, r in trials), default=None)
        print(f'n={n} patches={group["recipient_patch_counts"]} trials={len(trials)} '
              f'baseline={side(p):.12f} control={group["control_side"]:.12f} best={best} win={group["win"]}', flush=True)


if __name__ == '__main__':
    main()
