"""Uniform shrink: if every neighbouring pair has a positive SAT gap, scale all centres toward
the bbox centre by s = 1 - delta (angles unchanged), delta = 0.5 * min(gap / |d|).
Certified literal + pandas like the merger.
usage: shrink_pass.py full.csv lit.json pandas.json out.json [survey]
"""
import csv, json, math, sys
from decimal import Decimal, getcontext, ROUND_HALF_EVEN
from multiprocessing import Pool
import numpy as np
import numba as nb
import merge_public as mp

getcontext().prec = 60


@nb.njit(cache=True)
def min_ratio(X, Y, T):
    n = X.shape[0]
    best = 1e9; bestgap = 1e9
    for i in range(n):
        for j in range(i + 1, n):
            dx = X[j] - X[i]; dy = Y[j] - Y[i]
            dd = math.sqrt(dx * dx + dy * dy)
            if dd >= 1.5:
                continue
            gap = -1e9
            for a in range(4):
                if a < 2:
                    th = T[i] + a * (math.pi / 2); to = T[j]
                else:
                    th = T[j] + (a - 2) * (math.pi / 2); to = T[i]
                ux = math.cos(th); uy = math.sin(th)
                d = abs(dx * ux + dy * uy)
                ro = 0.5 * (abs(math.cos(to - th)) + abs(math.sin(to - th)))
                g = d - 0.5 - ro
                if g > gap:
                    gap = g
            if gap < bestgap:
                bestgap = gap
            r = gap / dd
            if r < best:
                best = r
    return best, bestgap


def load(path):
    by = {}
    with open(path, newline='') as f:
        for r in csv.DictReader(f):
            by.setdefault(int(r['n']), []).append({k: r[k] for k in ('n', 'square_id', 'x', 'y', 'angle')})
    return by


def analyse(rows):
    X = np.array([float(r['x']) for r in rows]); Y = np.array([float(r['y']) for r in rows])
    T = np.array([float(r['angle']) for r in rows])
    return min_ratio(X, Y, T)


def shrink(rows, delta):
    e = [0.5 * (abs(math.cos(float(r['angle']))) + abs(math.sin(float(r['angle'])))) for r in rows]
    xs = [float(r['x']) for r in rows]; ys = [float(r['y']) for r in rows]
    cx = Decimal(repr((min(x - a for x, a in zip(xs, e)) + max(x + a for x, a in zip(xs, e))) / 2))
    cy = Decimal(repr((min(y - a for y, a in zip(ys, e)) + max(y + a for y, a in zip(ys, e))) / 2))
    s = Decimal(1) - Decimal(repr(delta))
    q = Decimal('1e-25')
    out = []
    for r in rows:
        x = cx + s * (Decimal(r['x'].strip()) - cx); y = cy + s * (Decimal(r['y'].strip()) - cy)
        out.append({'n': r['n'], 'square_id': r['square_id'], 'x': format(x.quantize(q, ROUND_HALF_EVEN), 'f'),
                    'y': format(y.quantize(q, ROUND_HALF_EVEN), 'f'), 'angle': r['angle']})
    return out


def main():
    full, litp, pdp, outp = sys.argv[1:5]
    survey = len(sys.argv) > 5
    lit = {r['n']: Decimal(r['L']) for r in json.load(open(litp))['results']}
    pdl = {r['n']: Decimal(r['L']) for r in json.load(open(pdp))['results']}
    by = load(full)
    jobs = []; hist = {}
    for n in range(2, 1001):
        ratio, gap = analyse(by[n])
        b = 'touching(<1e-13)' if gap < 1e-13 else ('<1e-11' if gap < 1e-11 else ('<1e-9' if gap < 1e-9 else '>=1e-9'))
        hist[b] = hist.get(b, 0) + 1
        if gap >= 1e-12 and ratio > 0:
            delta = float(__import__("os").environ.get("SHRINK_FACTOR", "0.5")) * ratio
            jobs.append((n, shrink(by[n], delta), delta))
    print('min-gap histogram:', hist, '| shrink candidates:', len(jobs), flush=True)
    if survey:
        return
    wins = {}
    with Pool(8) as pool:
        for idx, (n, res, why) in enumerate(pool.imap(mp.certify, [(n, rows) for n, rows, _ in jobs])):
            if res is None:
                print('reject', n, why[:70], flush=True); continue
            a, b = Decimal(res[0]), Decimal(res[1])
            if a < lit[n] and b < pdl[n]:
                wins[n] = {'L': str(max(a, b)), 'rows': jobs[idx][1], 'src': f'shrink-{jobs[idx][2]:.2e}'}
                print(f'win n={n} dL={max(lit[n], pdl[n]) - max(a, b):.3e} delta={jobs[idx][2]:.2e}', flush=True)
    json.dump(wins, open(outp, 'w'))
    print('wins', len(wins), 'score gain', sum(float(max(lit[n], pdl[n]) - Decimal(w['L'])) / math.sqrt(n) for n, w in wins.items()))


if __name__ == '__main__':
    main()
