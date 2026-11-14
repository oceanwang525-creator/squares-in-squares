"""Build a single-pocket staircase packing for n = K^2 + g from a library gadget.

Construction (the m = 1 case of the composition rule sum w <= K, sum h <= K,
sum (w+h+1) <= 2K+1):

  box side S = K + max(fx, fy);  pocket = the gadget's (w+fx) x (h+fy) rectangle
  flush with the top-right corner, i.e. translated so that its bounding box is
  [K-w, K-w+W] x [K-h, K-h+H];  the rest of the box holds the upright unit grid
  restricted to the L-shape  {i < K-w} u {j < K-h}, which is exactly K^2 - w*h
  squares.  Total = K^2 - w*h + (w*h + g) = K^2 + g = n.

All arithmetic on the copied squares is done at 50 digits on the *original
decimal strings* of the donor row, so the donor's internal contacts survive
unchanged; a 1e-12 inset keeps the pocket off the grid.

usage: build_pocket.py --n 442 --src 401 --w 20 --h 17 [--out dir]
"""
import argparse
import csv
import collections
import json
import math
import os
import sys
from decimal import Decimal

import mpmath as mp
import numpy as np

sys.path.insert(0, 'poc/baseline/poc/src')
sys.path.insert(0, 'poc/faultline_dp2/poc')
from squares.core import certify, kaggle_roundtrip
import gadlib

EPS = mp.mpf('1e-12')


def load_rows(path):
    by = collections.OrderedDict()
    for r in csv.DictReader(open(path)):
        by.setdefault(int(r['n']), []).append((r['square_id'], r['x'], r['y'], r['angle']))
    return by


def find_gadget(rows, w, h, gg):
    """repeat gadlib's float extraction for one n and return the idx of the gadget."""
    a = np.array([[float(x), float(y), float(t)] for _, x, y, t in rows])
    e = 0.5 * (np.abs(np.cos(a[:, 2])) + np.abs(np.sin(a[:, 2])))
    a[:, 0] -= (a[:, 0] - e).min()
    a[:, 1] -= (a[:, 1] - e).min()
    L = max((a[:, 0] + e).max(), (a[:, 1] + e).max())
    cands = []
    t = np.degrees(np.mod(a[:, 2], np.pi / 2))
    tilt = np.minimum(t, 90 - t) > 0.5
    cls = gadlib.clusters(a, tilt)
    seeds = list(cls)
    for i in range(len(cls)):
        for j in range(i + 1, len(cls)):
            seeds.append(np.concatenate([cls[i], cls[j]]))
    seen = set()
    for cl in seeds:
        x0 = (a[cl, 0] - e[cl]).min(); x1 = (a[cl, 0] + e[cl]).max()
        y0 = (a[cl, 1] - e[cl]).min(); y1 = (a[cl, 1] + e[cl]).max()
        X0, X1, Y0, Y1, idx = gadlib.clean_rect(a, e, x0, x1, y0, y1)
        W, H = X1 - X0, Y1 - Y0
        wi, hi = int(W + 1e-7), int(H + 1e-7)
        g = len(idx) - wi * hi
        key = (round(W, 6), round(H, 6), len(idx))
        if key in seen:
            continue
        seen.add(key)
        if {wi, hi} == {w, h} or (wi == w and hi == h) or (wi == h and hi == w):
            if g != gg:
                continue
            cands.append((max(W - wi, H - hi), idx, wi, hi, W, H, X0, X1, Y0, Y1))
    if not cands:
        raise SystemExit(f'no gadget ({w},{h},{gg}) found in n=401-style extraction')
    cands.sort(key=lambda c: c[0])
    return cands[0], a, e


def build(n, src, rows_all, plan):
    K = math.isqrt(n)
    j = n - K * K
    assert j == plan['g'], (j, plan)
    w, h, g = plan['w'], plan['h'], plan['g']
    rows = rows_all[src]
    (f, idx, wi, hi, W, H, X0, X1, Y0, Y1), a, e = find_gadget(rows, w, h, g)
    assert len(idx) == w * h + g, (len(idx), w * h + g)
    xs = [rows[i][1] for i in idx]
    ys = [rows[i][2] for i in idx]
    ts = [rows[i][3] for i in idx]
    with mp.workdps(50):
        ex = [mp.mpf(0.5) * (abs(mp.cos(mp.mpf(t))) + abs(mp.sin(mp.mpf(t)))) for t in ts]
        xmin = min(mp.mpf(x) - e1 for x, e1 in zip(xs, ex))
        ymin = min(mp.mpf(y) - e1 for y, e1 in zip(ys, ex))
        dx = mp.mpf(K - w) + EPS - xmin
        dy = mp.mpf(K - h) + EPS - ymin
        keep = 0.5 * (abs(mp.cos(mp.mpf(rows[idx[0]][3]))) + abs(mp.sin(mp.mpf(rows[idx[0]][3]))))
        out = []
        for x, y, t in zip(xs, ys, ts):
            out.append(dict(square_id=None, x=mp.nstr(mp.mpf(x) + dx, 40),
                            y=mp.nstr(mp.mpf(y) + dy, 40), angle=t))
    if any('e' in r['x'] or 'e' in r['y'] for r in out):
        raise SystemExit('exponent notation in output')
    n_pocket = len(out)
    grid = []
    for i in range(K):
        for jj in range(K):
            if i < K - w or jj < K - h:
                grid.append(dict(square_id=None, x=f'{i}.5', y=f'{jj}.5', angle='0'))
    allrows = out + grid
    assert len(allrows) == n, (len(allrows), n)
    for i, r in enumerate(allrows):
        r['square_id'] = str(i)
    assert len(allrows) == K * K - w * h + n_pocket
    return allrows, dict(K=K, j=j, w=w, h=h, g=g, src=src, f=f, pocket=n_pocket, grid=len(grid))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, nargs='+', required=True)
    ap.add_argument('--src', type=int, default=401)
    ap.add_argument('--w', type=int, default=20)
    ap.add_argument('--h', type=int, default=17)
    ap.add_argument('--g', type=int, default=1)
    ap.add_argument('--csv', default='poc/pell_corrugation/data/border-scan/merged.csv')
    ap.add_argument('--report', default='poc/faultline_dp2/data/report.json')
    ap.add_argument('--outdir', default='poc/faultline_dp2/data/built')
    ap.add_argument('--final')
    a = ap.parse_args()
    rows_all = load_rows(a.csv)
    L = {r['n']: r['L'] for r in json.load(open(a.report))['results']}
    done = {}
    for n in a.n:
        rows, info = build(n, a.src, rows_all, dict(w=a.w, h=a.h, g=a.g))
        Llit = certify(n, rows)
        Lpan = certify(n, kaggle_roundtrip({n: rows})[n])
        inc = Decimal(L[n])
        dlit = Decimal(Llit) - inc
        dpan = Decimal(Lpan) - inc
        gain = min(dlit, dpan) / Decimal(n).sqrt()
        print(f'n={n} K={info["K"]} j={info["j"]} gadget ({a.w},{a.h},+{a.g}) from n={a.src} '
              f'f={info["f"]:.9f}')
        print(f'   incumbent {inc}  ->  literal {Llit}  (dL {dlit:.3e})')
        print(f'                          pandas  {Lpan}  (dL {dpan:.3e})   gain {gain:.4e}')
        print(f'   pocket {info["pocket"]} sq + grid {info["grid"]} sq = {n}  '
              f'{"STRICT WIN" if dlit < 0 and dpan < 0 else "no win"}')
        if dlit < 0 and dpan < 0:
            done[n] = rows
            os.makedirs(a.outdir, exist_ok=True)
            json.dump(rows, open(os.path.join(a.outdir, f'n{n}.json'), 'w'))
    if a.final and done:
        with open(a.final, 'w', newline='') as fh:
            wtr = csv.writer(fh, lineterminator='\n')
            wtr.writerow(['id', 'x', 'y', 'deg'])
            with mp.workdps(95):
                for n, rows in sorted(done.items()):
                    for r in sorted(rows, key=lambda r: int(r['square_id'])):
                        deg = mp.nstr(mp.mpf(r['angle']) * 180 / mp.pi, 75)
                        wtr.writerow([f'{n:03}_{r["square_id"]}', 's' + r['x'], 's' + r['y'],
                                      's' + deg])
        print(f'wrote {a.final} ({sum(len(v) for v in done.values())} rows, '
              f'groups {sorted(done)})')


if __name__ == '__main__':
    main()
