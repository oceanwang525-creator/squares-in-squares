"""Find the best square enclosure orientation for each archived midrange packing."""
import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull


HEADER = ('id', 'n', 'square_id', 'x', 'y', 'angle')
HALF_PI = math.pi / 2
EPS = 1e-10


def read(path, lower, upper):
    with path.open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        if not {'n', 'x', 'y', 'angle'}.issubset(reader.fieldnames or ()):
            return
        for size, rows in itertools.groupby(reader, key=lambda row: row['n']):
            n = int(size)
            if not lower <= n <= upper:
                continue
            rows = list(rows)
            if len(rows) != n:
                continue
            p = np.array([[float(r[k]) for k in HEADER[-3:]] for r in rows])
            if np.isfinite(p).all():
                yield n, rows, p


def half_extent(p):
    return (abs(np.cos(p[:, 2])) + abs(np.sin(p[:, 2]))) / 2


def side(p):
    h = half_extent(p)[:, None]
    return float(max((p[:, :2] + h).max(axis=0) - (p[:, :2] - h).min(axis=0)))


def best_rotation(p):
    a = p[:, 2]
    c, s = np.cos(a) / 2, np.sin(a) / 2
    u, v = np.column_stack((c, s)), np.column_stack((-s, c))
    corners = np.concatenate([p[:, :2] + i*u + j*v for i in (-1, 1) for j in (-1, 1)])
    hull = corners[ConvexHull(corners).vertices]
    edges = np.roll(hull, -1, axis=0) - hull
    angles = np.mod(-np.arctan2(edges[:, 1], edges[:, 0]), HALF_PI)
    bounds = np.unique(np.r_[0, np.round(angles, 14), HALF_PI])
    mid = (bounds[:-1] + bounds[1:]) / 2

    def projections(theta):
        c, s = np.cos(theta), np.sin(theta)
        x, y = hull.T
        return c[:, None]*x - s[:, None]*y, s[:, None]*x + c[:, None]*y

    x, y = projections(mid)
    dx = hull[x.argmax(1)] - hull[x.argmin(1)]
    dy = hull[y.argmax(1)] - hull[y.argmin(1)]
    # Width and height are each positive sinusoids, hence concave between
    # hull events. A minimum is an event or an equal-width-and-height crossing.
    cross = np.mod(np.arctan2(dx[:, 0] - dy[:, 1], dx[:, 1] + dy[:, 0]), math.pi)
    active = (cross > bounds[:-1]) & (cross < bounds[1:])
    trials = np.r_[bounds, cross[active]]
    x, y = projections(trials)
    lengths = np.maximum(np.ptp(x, axis=1), np.ptp(y, axis=1))
    i = int(lengths.argmin())
    theta = float((trials[i] + HALF_PI/2) % HALF_PI - HALF_PI/2)
    return float(lengths[i]), theta


def turn(p, theta):
    c, s = math.cos(theta), math.sin(theta)
    q = p.copy()
    q[:, :2] = q[:, :2] @ np.array([[c, s], [-s, c]])
    q[:, 2] = (q[:, 2] + theta + HALF_PI/2) % HALF_PI - HALF_PI/2
    h = half_extent(q)[:, None]
    q[:, :2] -= (q[:, :2] - h).min(axis=0)
    # Preserve contact clearance across float serialization of the rotation.
    q[:, :2] *= 1 + 3e-13
    return q


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('baseline', type=Path)
    ap.add_argument('output', type=Path)
    ap.add_argument('sources', nargs='*', type=Path)
    args = ap.parse_args()
    base = dict((n, side(p)) for n, _, p in read(args.baseline, 100, 200))
    if sorted(base) != list(range(100, 201)):
        raise ValueError('Incomplete incumbent')
    args.output.mkdir(exist_ok=True, parents=True)
    paths = list(dict.fromkeys(path.resolve() for src in [args.baseline, *args.sources]
                              for path in (sorted(src.rglob('*.csv')) if src.is_dir() else [src])))
    seen, wins, failed, usable, nearest = set(), {}, [], 0, {}
    for path in paths:
        try:
            groups = 0
            for n, raw, p in read(path, 100, 200):
                groups += 1
                key = hashlib.blake2b(p.tobytes(), digest_size=16).digest()
                if key in seen:
                    continue
                seen.add(key)
                best, theta = best_rotation(p)
                prior = wins[n][0] if n in wins else base[n]
                delta = best - base[n]
                if delta < nearest.get(n, (math.inf,))[0]:
                    nearest[n] = (delta, theta, path.name)
                if best >= prior - EPS:
                    continue
                if abs(theta) < 1e-12:
                    values = [tuple(r[k] for k in HEADER[-3:]) for r in raw]
                    length = side(p)
                else:
                    q = turn(p, theta)
                    length = side(q)
                    values = q.tolist()
                if length < prior - EPS:
                    wins[n] = (length, str(path), theta, values)
                    print('candidate', n, base[n], length, theta, path.name, flush=True)
            usable += groups > 0
        except (UnicodeError, ValueError, KeyError, OSError, csv.Error) as exc:
            failed.append([path.name, type(exc).__name__])
    report = dict(baseline_sha256=hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
                  paths=len(paths), usable_csv=usable, distinct_layouts=len(seen),
                  unreadable=failed, nearest=nearest,
                  candidates={n: {'side': x[0], 'incumbent': base[n], 'source': Path(x[1]).name,
                                  'rotation_rad': x[2]} for n, x in sorted(wins.items())})
    (args.output/'scan.json').write_text(json.dumps(report, indent=2) + '\n')
    if wins:
        with (args.output/'hull_candidates.csv').open('w', newline='') as f:
            w = csv.writer(f, lineterminator='\n')
            w.writerow(HEADER)
            for n, (_, _, _, geometry) in sorted(wins.items()):
                for sid, row in enumerate(geometry):
                    w.writerow((n*(n-1)//2 + sid, n, sid, *row))
    print({k: report[k] for k in ('paths', 'usable_csv', 'distinct_layouts', 'candidates')}, flush=True)


if __name__ == '__main__':
    main()
