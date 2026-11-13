"""Rotate/translate the tilted cores of midrange layouts and refill upright pockets."""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from numba import njit


@njit
def clear_upright(x, y, core):
    for tx, ty, c, s, sep in core:
        dx, dy = x - tx, y - ty
        if abs(dx) < sep-1e-10 and abs(dy) < sep-1e-10:
            if abs(dx*c + dy*s) < sep-1e-10 and abs(dy*c - dx*s) < sep-1e-10:
                return False
    return True


@njit
def scan(core, pocket, shifts, limit):
    best = np.empty((0, 3))
    for k in range(len(shifts)):
        x, y, theta = shifts[k]
        ca, sa = math.cos(theta), math.sin(theta)
        q = np.empty((len(core), 5))
        feasible = True
        for i in range(len(core)):
            u, v, a = core[i]
            q[i, 0] = u*ca - v*sa + x
            q[i, 1] = u*sa + v*ca + y
            q[i, 2] = math.cos(a + theta)
            q[i, 3] = math.sin(a + theta)
            h = (abs(q[i, 2]) + abs(q[i, 3]))/2
            q[i, 4] = h + .5
            if min(q[i, 0], q[i, 1]) < h or max(q[i, 0], q[i, 1]) > limit-h:
                feasible = False
                break
        if not feasible:
            continue
        ok = np.zeros(len(pocket), dtype=np.bool_)
        for i in range(len(pocket)):
            ok[i] = clear_upright(pocket[i, 0], pocket[i, 1], q)
        if ok.sum()+len(core) <= len(best):
            continue
        # The four corner lattices can only conflict near their shared seams.
        # Retry different orders of the corner pockets, picking outward sites first.
        chosen = np.empty((len(pocket), 2))
        for start in range(4):
            nc = 0
            for j in range(len(pocket)):
                i = (j + start*(len(pocket)//4)) % len(pocket)
                if not ok[i]:
                    continue
                px, py = pocket[i]
                keep = True
                for h in range(nc):
                    if abs(px-chosen[h, 0]) < 1-1e-10 and abs(py-chosen[h, 1]) < 1-1e-10:
                        keep = False
                        break
                if keep:
                    chosen[nc] = px, py
                    nc += 1
            if nc+len(core) <= len(best):
                continue
            best = np.zeros((nc+len(core), 3))
            for i in range(len(core)):
                best[i, 0] = q[i, 0]
                best[i, 1] = q[i, 1]
                best[i, 2] = core[i, 2] + theta
            best[len(core):, :2] = chosen[:nc]
    return best


def read(path):
    groups = {}
    with path.open(newline='', encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            n = int(r['n'])
            if 130 <= n <= 210:
                groups.setdefault(n, []).append([float(r[k]) for k in ('x', 'y', 'angle')])
            if n > 210:
                break
    return {n: np.array(p) for n, p in groups.items()}


def make_pockets(length):
    m = math.floor(length)
    ij = sorted(((i, j) for i in range(m) for j in range(m)), key=lambda p: (sum(p), p))
    a = .5 + np.array(ij)
    return np.vstack([np.where(flip, length-a, a) for flip in [(0, 0), (0, 1), (1, 1), (1, 0)]])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('baseline', type=Path)
    ap.add_argument('output', type=Path)
    args = ap.parse_args()
    layouts = read(args.baseline)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    wins, results = {}, {}
    for n in [147, 148, 199, 146, 149, 150, 153, 154, 156, 171, 172, 173, 175, 179, 198, 200]:
        p = layouts[n]
        h = (abs(np.cos(p[:, 2])) + abs(np.sin(p[:, 2])))[:, None]/2
        bound = float(max((p[:, :2]+h).max(0)-(p[:, :2]-h).min(0)))
        goal = bound - 1e-6
        pocket = make_pockets(goal)
        best, trials = None, 0
        for donor in sorted(set([n, 147, 148, 173, 200, *range(n-2, n+3)])):
            q = layouts[donor]
            h = (abs(np.cos(q[:, 2])) + abs(np.sin(q[:, 2])))[:, None]/2
            center = ((q[:, :2]+h).max(0)+(q[:, :2]-h).min(0))/2
            core = q[abs(np.sin(q[:, 2]*2))>1e-5].copy()
            if not len(core) or len(core)>n:
                continue
            core[:, :2] -= center
            offset = np.linspace(-.20, .20, 15)
            angles = np.linspace(-.035, .035, 21)
            shifts = np.array([(goal/2+x, goal/2+y, a) for a in angles for x in offset for y in offset])
            result = scan(core, pocket, shifts, goal)
            trials += len(shifts)
            if best is None or len(result)>best[0]:
                best = (len(result), donor, result)
        count, donor, q = best
        results[n] = dict(best_count=count, source=donor, side_limit=goal, poses=trials)
        print(n, results[n], flush=True)
        if count >= n:
            q = q[:n].copy()
            q[:, :2] *= 1+3e-10
            wins[n] = q
        (output/'scan.json').write_text(json.dumps(results, indent=2) + '\n')
    if wins:
        with (output/'pocket_candidates.csv').open('w', newline='') as f:
            w = csv.writer(f, lineterminator='\n')
            w.writerow(('id', 'n', 'square_id', 'x', 'y', 'angle'))
            for n, geometry in sorted(wins.items()):
                for sid, row in enumerate(geometry):
                    w.writerow((n*(n-1)//2 + sid, n, sid, *row))


if __name__ == '__main__':
    main()
