"""Build the m=1 staircase composition from a pocket found here, and certify it.

Same geometry as `poc/faultline_dp2/poc/build_pocket.py` (m = 1 case of
sum w <= K, sum h <= K, sum(w+h+1) <= 2K+1):

    box side  S = K + f,   f = max(W_pocket - w, H_pocket - h)
    pocket flush with the top-right corner: bbox translated to
        [K-w, K-w+W] x [K-h, K-h+H]        (with a 1e-12 inset)
    upright unit grid on the L-shape {i < K-w} u {j < K-h}  ->  K^2 - w*h squares
    total = K^2 - w*h + (w*h + g) = K^2 + g = n

Certification uses the unchanged official metric in BOTH transports:
  literal decimal strings, and a real pandas read_csv -> to_csv round trip.

usage:
  build.py --n 10 --w 3 --h 2 --g 1 --pocket data/shape_v2_3_2_1.json --final final.csv
"""
import argparse
import collections
import csv
import json
import math
import os
import sys
from decimal import Decimal

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'poc', 'baseline', 'poc', 'src'))
sys.path.insert(0, HERE)

import mpmath as mp                                   # noqa: E402
from squares.core import certify, kaggle_roundtrip    # noqa: E402

REPORT = os.path.join(ROOT, 'poc', 'faultline_dp2', 'data', 'report.json')
INSET = mp.mpf('1e-12')


def compose(pocket, w, h, n, ndps=45, dilate=1e-11):
    """pocket: (N,3) float array (x, y, angle) whose bbox is W x H, W>w, H>h.

    `dilate` is a uniform relative dilation of the centres about the pocket's
    bbox centre: a numerically-optimal pocket has contacts at ~1e-16 and the
    official metric still rejects any overlap, so the pocket is inflated by a
    hair (costs ~1e-11 of side, far below any real win)."""
    K = math.isqrt(n)
    assert n - K * K <= w * h + 1e9
    q = np.asarray(pocket, float).copy()
    e = 0.5 * (np.abs(np.cos(q[:, 2])) + np.abs(np.sin(q[:, 2])))
    cx = 0.5 * ((q[:, 0] - e).min() + (q[:, 0] + e).max())
    cy = 0.5 * ((q[:, 1] - e).min() + (q[:, 1] + e).max())
    q[:, 0] = cx + (q[:, 0] - cx) * (1 + dilate)
    q[:, 1] = cy + (q[:, 1] - cy) * (1 + dilate)
    e = e * (1 + dilate)
    W = float((q[:, 0] + e).max() - (q[:, 0] - e).min())
    H = float((q[:, 1] + e).max() - (q[:, 1] - e).min())
    f = max(W - w, H - h)
    S = K + f
    with mp.workdps(ndps):
        rows = []
        x0 = mp.mpf(repr(float((q[:, 0] - e).min())))
        y0 = mp.mpf(repr(float((q[:, 1] - e).min())))
        dx = mp.mpf(K - w) + INSET - x0
        dy = mp.mpf(K - h) + INSET - y0
        for (x, y, t) in q:
            rows.append(dict(square_id=None,
                             x=mp.nstr(mp.mpf(repr(float(x))) + dx, ndps),
                             y=mp.nstr(mp.mpf(repr(float(y))) + dy, ndps),
                             angle=mp.nstr(mp.mpf(repr(float(t))), ndps)))
        grid = []
        for i in range(K):
            for j in range(K):
                if i < K - w or j < K - h:
                    grid.append(dict(square_id=None, x=f'{i}.5', y=f'{j}.5', angle='0'))
    rows = rows + grid
    assert len(rows) == n, (len(rows), n)
    for i, r in enumerate(rows):
        r['square_id'] = str(i)
    return rows, dict(K=K, f=f, S=S, W=W, H=H, pocket=len(q), grid=len(grid))


def cert(n, rows):
    Llit = certify(n, rows)
    Lpan = certify(n, kaggle_roundtrip({n: rows})[n])
    return Llit, Lpan


def write_final(path, groups):
    with open(path, 'w', newline='') as fh:
        wtr = csv.writer(fh, lineterminator='\n')
        wtr.writerow(['id', 'x', 'y', 'deg'])
        with mp.workdps(95):
            for n, rows in sorted(groups.items()):
                for r in sorted(rows, key=lambda r: int(r['square_id'])):
                    deg = mp.nstr(mp.mpf(r['angle']) * 180 / mp.pi, 75)
                    wtr.writerow([f'{n:03}_{r["square_id"]}', 's' + r['x'], 's' + r['y'],
                                  's' + deg])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pocket', required=True, help='json with key "q" = (N,3) list')
    ap.add_argument('--n', type=int, nargs='+', required=True)
    ap.add_argument('--w', type=int, required=True)
    ap.add_argument('--h', type=int, required=True)
    ap.add_argument('--final', default=None)
    a = ap.parse_args()
    d = json.load(open(a.pocket))
    q = np.array(d['q'], float)
    inc = {r['n']: r['L'] for r in json.load(open(REPORT))['results']}
    done = {}
    for n in a.n:
        rows, info = compose(q, a.w, a.h, n)
        Llit, Lpan = cert(n, rows)
        L0 = Decimal(inc[n])
        dlit = Decimal(Llit) - L0
        dpan = Decimal(Lpan) - L0
        print('n=%d K=%d  pocket (%d,%d,+%d) %d sq + grid %d sq   f=%.15f  side=%.15f'
              % (n, info['K'], a.w, a.h, n - info['K'] ** 2, info['pocket'], info['grid'],
                 info['f'], info['S']))
        print('   incumbent %s' % L0)
        print('   literal   %s   dL %+.3e' % (Llit, float(dlit)))
        print('   pandas    %s   dL %+.3e' % (Lpan, float(dpan)))
        print('   %s' % ('STRICT WIN' if (dlit < 0 and dpan < 0) else 'no win (not strictly better)'))
        if dlit < 0 and dpan < 0:
            done[n] = rows
    if a.final:
        if done:
            write_final(a.final, done)
            print('wrote %s: groups %s' % (a.final, sorted(done)))
        else:
            with open(a.final, 'w', newline='') as fh:
                csv.writer(fh, lineterminator='\n').writerow(['id', 'x', 'y', 'deg'])
            print('wrote %s: header only (no strict win)' % a.final)


if __name__ == '__main__':
    main()
