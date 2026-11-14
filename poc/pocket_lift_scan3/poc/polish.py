"""Stage C²: two-edge exact polish of the surviving pockets.

`scan_pockets.py` stage C walks one edge at a time over the row's exact boundary
values (lx/hx/ly/hy of its squares).  That is a coordinate descent, so it can
never reach a pocket that needs *two* edges moved simultaneously (the
intermediate state is worse or infeasible).  Here every pair of edges is moved
over its four nearest boundary values in each direction, and every set visited
is classified -- cheap, because it runs on the ~5e2 Pareto-surviving pockets
instead of on all 1e3 rows.

usage: polish.py pockets_pareto.json out.json [--procs 30] [--k 4]
"""
import argparse
import collections
import json
import math
import sys

import numpy as np

sys.path.insert(0, 'poc/pocket_lift_scan3/poc')
from scan_pockets import Row, load_rows


def neighbours(vals, cur, k):
    i = int(np.searchsorted(vals, cur))
    lo = [vals[j] for j in range(max(0, i - k), min(len(vals), i + 1))]
    hi = [vals[j] for j in range(min(len(vals), i + 1), min(len(vals), i + 1 + k))]
    return lo + hi


def polish(row, box, fmax, nmax, store, k=4):
    X0, X1, Y0, Y1 = box
    lists = (row.LX, row.HX, row.LY, row.HY)
    cur = (X0, X1, Y0, Y1)
    W = []
    for e in range(4):                              # single-edge moves
        vals = neighbours(lists[e], cur[e], 2 * k)
        for v in vals:
            w = list(cur)
            w[e] = v
            W.append(w)
    for a in range(4):                              # two-edge moves
        for b in range(a + 1, 4):
            va = neighbours(lists[a], cur[a], k)
            vb = neighbours(lists[b], cur[b], k)
            for u in va:
                for v in vb:
                    w = list(cur)
                    w[a], w[b] = u, v
                    W.append(w)
    W = np.array(W, float)
    W = W[(W[:, 0] < W[:, 1] - 1e-9) & (W[:, 2] < W[:, 3] - 1e-9)]
    if W.shape[0] == 0:
        return 0
    boxes = row.refine_many(W)
    n = 0
    for b in boxes:
        cls = row.classify(b, fmax, nmax)
        if cls is None:
            continue
        key = (cls[0], cls[1], cls[2])
        store(key, cls, b)
        n += 1
    return n


_G = {}


def _init(csv, opts):
    _G['rows'] = load_rows(csv)
    _G['o'] = opts
    _G['row'] = {}


def _work(rec):
    src = rec['src']
    if src not in _G['row']:
        _G['row'][src] = Row(src, _G['rows'][src])
    row = _G['row'][src]
    o = _G['o']
    box = (rec['x0'], rec['x1'], rec['y0'], rec['y1'])
    out = {}

    def store(key, cls, b):
        w, h, g, f, cnt = cls
        cur = out.get(key)
        if cur is None or f < cur['f'] - 1e-12:
            out[key] = dict(src=src, w=w, h=h, g=g, f=f, cnt=cnt,
                            x0=b[0], x1=b[1], y0=b[2], y1=b[3])

    polish(row, box, o['fmax'], o['nmax'], store, o['k'])
    return list(out.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pockets')
    ap.add_argument('out')
    ap.add_argument('--csv', default='poc/pell_corrugation/data/border-scan/merged.csv')
    ap.add_argument('--procs', type=int, default=30)
    ap.add_argument('--k', type=int, default=4)
    ap.add_argument('--fmax', type=float, default=1.0)
    ap.add_argument('--nmax', type=int, default=31)
    a = ap.parse_args()
    recs = json.load(open(a.pockets))
    print(f'{len(recs)} pockets from {a.pockets}', flush=True)
    opts = dict(k=a.k, fmax=a.fmax, nmax=a.nmax)
    import multiprocessing as mp
    with mp.Pool(a.procs, initializer=_init, initargs=(a.csv, opts)) as pool:
        res = pool.map(_work, recs, chunksize=2)
    out = {}
    for gs in res:
        for g in gs:
            k = (g['w'], g['h'], g['g'])
            if k not in out or g['f'] < out[k]['f'] - 1e-12:
                out[k] = g
    lib = list(out.values())
    json.dump(lib, open(a.out, 'w'), indent=0)
    print(f'{len(lib)} distinct (w,h,g) types -> {a.out}')


if __name__ == '__main__':
    main()
