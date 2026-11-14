"""Exhaustive proper-sub-pocket scan (hypothesis C1).

Idea
----
`gadlib.py` (faultline_dp2) only cuts *minimal clean rectangles around tilted
clusters*, so the library contains whole pockets but almost none of their
sub-rectangles -- and in particular nothing cut out of the upright region of a
row.  But any axis-aligned sub-rectangle of a row that (a) no square straddles
and (b) has a tight bounding box is itself a legal composition gadget

    (w, h, +g, f):  w*h + g squares in a (w+f) x (h+f) box,

and the staircase composition only ever asks the pocket to fit inside its own
box, so it can be re-anchored at the origin and used directly.

This script scans **every tracked row** for such sub-rectangles:

  stage A  enumerate axis-aligned windows on a coarse offset grid (STEP, in
           units of squares), screen them with an exact 4-sided containment
           count (2-D prefix table per y-window, O(1)/window), keep windows
           that hold at least w*h+1 squares;
  stage B  re-measure each surviving window *exactly*: shrink it to the tight
           bounding box of the squares it contains, read off
           w = floor(W), h = floor(H), g = count - w*h, f = max(W-w, H-h);
  stage C  exact local search: the tight edge of a true pocket is always the
           bounding-box edge of some square, so coordinate descent over the
           exact boundary values (all lx/hx/ly/hy of the row) recovers the
           resolution the grid lost.  Every set visited is recorded, so this
           is a systematic sweep of the neighbourhood of every grid pocket.

Only (w,h,g) types with g>=1, f<=FMAX, w,h<=31 are kept, and globally only the
smallest f per (w,h,g) type is stored (the DP needs nothing else).

usage:
  scan_pockets.py merged.csv out.json [--step 0.25] [--fmax 0.9] [--procs 30]
                  [--nmax 31] [--rows 1,2,3] [--cap 400]
"""
import argparse
import csv
import collections
import json
import math
import os
import sys
import time

import numpy as np


# ---------------------------------------------------------------- row loading
def load_rows(path):
    by = collections.defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            by[int(r['n'])].append((float(r['x']), float(r['y']), float(r['angle'])))
    return {n: np.array(v) for n, v in by.items()}


def prep(a):
    """translate the row so that its bounding box starts at the origin."""
    e = 0.5 * (np.abs(np.cos(a[:, 2])) + np.abs(np.sin(a[:, 2])))
    a = a.copy()
    a[:, 0] -= (a[:, 0] - e).min()
    a[:, 1] -= (a[:, 1] - e).min()
    return a, e


# ------------------------------------------------------------------- stage A
def window_pairs(ng, step, maxw=31.75):
    """grid (p,q) pairs with 1 <= step*(q-p) <= maxw."""
    lo = int(round(1.0 / step))
    hi = int(round(maxw / step))
    out = []
    for p in range(ng):
        for q in range(p + lo, min(p + hi, ng - 1) + 1):
            out.append((p, q))
    return np.array(out, dtype=np.int64)


def stage_a(lx, hx, ly, hy, step, maxw=31.75, off=(0.0, 0.0)):
    """grid windows whose contained-square count is >= w*h+1 (window w,h)."""
    L = max(hx.max(), hy.max())
    ng = int(math.ceil((L + 1.0) / step)) + 2
    ox, oy = off
    lo = lambda v, o: np.clip(np.floor((v - o) / step + 1e-9).astype(np.int64), 0, ng - 1)
    hi = lambda v, o: np.clip(np.ceil((v - o) / step - 1e-9).astype(np.int64), 0, ng - 1)
    lo_, hi_ = lo, hi
    lo = lambda v: lo_(v, ox)
    hi = lambda v: hi_(v, ox)
    pl, qh = lo(lx), hi(hx)          # lx >= ox+step*p  <=>  p <= pl
    pc, qd = lo_(ly, oy), hi_(hy, oy)
    P = window_pairs(ng, step, maxw)
    Wv = step * (P[:, 1] - P[:, 0])
    wi = np.floor(Wv + 1e-9).astype(np.int64)
    Q = P
    flat = pl * ng + qh
    cands = []
    for c, d in Q:
        m = (pc >= c) & (qd <= d)
        cnt = int(m.sum())
        if cnt < 2:
            continue
        S = np.cumsum(np.bincount(flat[m], minlength=ng * ng).reshape(ng, ng), axis=1)
        S = np.cumsum(S[::-1], axis=0)[::-1]
        cc = S[P[:, 0], P[:, 1]]
        hh = int(math.floor(step * (d - c) + 1e-9))
        ok = np.nonzero(cc - wi * hh >= 1)[0]
        if ok.size:
            cands.append(np.column_stack([P[ok, 0] * step + ox, P[ok, 1] * step + ox,
                                          np.full(ok.size, c * step + oy),
                                          np.full(ok.size, d * step + oy)]))
    if not cands:
        return np.zeros((0, 4))
    return np.vstack(cands)


# --------------------------------------------------------------- stage B / C
class Row:
    """pre-computed bounding boxes + exact boundary lists of one row."""

    def __init__(self, n, a):
        a, e = prep(a)
        self.n = n
        self.lx, self.hx = a[:, 0] - e, a[:, 0] + e
        self.ly, self.hy = a[:, 1] - e, a[:, 1] + e
        self.LX = np.unique(np.round(self.lx, 9))
        self.HX = np.unique(np.round(self.hx, 9))
        self.LY = np.unique(np.round(self.ly, 9))
        self.HY = np.unique(np.round(self.hy, 9))
        self.L = float(max(self.hx.max(), self.hy.max()))

    def mask(self, X0, X1, Y0, Y1):
        return ((self.lx >= X0 - 1e-12) & (self.hx <= X1 + 1e-12) &
                (self.ly >= Y0 - 1e-12) & (self.hy <= Y1 + 1e-12))

    def tight(self, X0, X1, Y0, Y1):
        """tight bbox of the squares inside the window; None if < 2 squares."""
        m = self.mask(X0, X1, Y0, Y1)
        c = int(m.sum())
        if c < 2:
            return None
        return (float(self.lx[m].min()), float(self.hx[m].max()),
                float(self.ly[m].min()), float(self.hy[m].max()), c)

    def refine_many(self, win):
        """vectorised tight bbox for a batch of windows (B,4)."""
        X0, X1, Y0, Y1 = (win[:, i][:, None] for i in range(4))
        M = ((self.lx[None] >= X0 - 1e-12) & (self.hx[None] <= X1 + 1e-12) &
             (self.ly[None] >= Y0 - 1e-12) & (self.hy[None] <= Y1 + 1e-12))
        cnt = M.sum(1)
        inf = np.inf
        mnx = np.where(M, self.lx[None], inf).min(1)
        mxx = np.where(M, self.hx[None], -inf).max(1)
        mny = np.where(M, self.ly[None], inf).min(1)
        mxy = np.where(M, self.hy[None], -inf).max(1)
        ok = cnt >= 2
        return np.column_stack([mnx, mxx, mny, mxy, cnt])[ok]

    def classify(self, box, fmax, nmax):
        """tight box -> (w,h,g,f,cnt) or None."""
        X0, X1, Y0, Y1, cnt = box
        W, H = X1 - X0, Y1 - Y0
        w = int(math.floor(W + 1e-6))
        h = int(math.floor(H + 1e-6))
        g = int(round(cnt)) - w * h
        f = max(W - w, H - h)
        if g < 1 or f > fmax or w < 1 or h < 1 or w > nmax or h > nmax:
            return None
        return (w, h, g, f, int(round(cnt)))


def edge_values(vals, cur, span):
    """exact boundary candidates for one edge within +-span of `cur`."""
    i = np.searchsorted(vals, cur - span)
    j = np.searchsorted(vals, cur + span, side='right')
    return vals[i:j]


def stage_c(row, box, fmax, nmax, store, span=0.75, iters=4):
    """coordinate descent on the four exact edges; records every set visited."""
    X0, X1, Y0, Y1, _ = box
    best = None
    seen = {}
    for _ in range(iters):
        moved = False
        for edge in range(4):
            cur = (X0, X1, Y0, Y1)
            vals = edge_values([row.LX, row.HX, row.LY, row.HY][edge], cur[edge], span)
            if vals.size == 0:
                continue
            win = np.repeat(np.array([cur], float), vals.size, axis=0)
            win[:, edge] = vals
            win = win[(win[:, 0] < win[:, 1] - 1e-9) & (win[:, 2] < win[:, 3] - 1e-9)]
            if win.shape[0] == 0:
                continue
            boxes = row.refine_many(win)
            for b in boxes:
                cls = row.classify(b, fmax, nmax)
                if cls is None:
                    continue
                key = (cls[0], cls[1], cls[2])
                if key not in seen or cls[3] < seen[key][3]:
                    seen[key] = cls
                store(key, cls, b)
                sc = (cls[3], -cls[4])
                if best is None or sc < best[0]:
                    best = (sc, b)
        if best is not None:
            nb = (best[1][0], best[1][1], best[1][2], best[1][3])
            if nb != (X0, X1, Y0, Y1):
                X0, X1, Y0, Y1 = nb
                moved = True
        if not moved:
            break
    return seen


# --------------------------------------------------------------- row scanner
def scan_row(n, a, step, fmax, nmax, cap=400, off=(0.0, 0.0), verbose=False):
    row = Row(n, a)
    t0 = time.time()
    win = stage_a(row.lx, row.hx, row.ly, row.hy, step, maxw=float(nmax) + 0.75, off=off)
    if win.shape[0] == 0:
        return [], row
    cands = []
    B = 4000
    for s in range(0, win.shape[0], B):
        boxes = row.refine_many(win[s:s + B])
        for b in boxes:
            cands.append(b)
    # dedupe stage-B pockets by tight bbox
    uniq = {}
    for b in cands:
        k = (round(b[0], 6), round(b[1], 6), round(b[2], 6), round(b[3], 6))
        if k not in uniq or b[4] > uniq[k][4]:
            uniq[k] = b
    seeds = list(uniq.values())
    # keep the most promising seeds if there are too many (big-|g| first)
    if len(seeds) > cap:
        def score(b):
            W, H = b[1] - b[0], b[3] - b[2]
            w, h = math.floor(W + 1e-6), math.floor(H + 1e-6)
            f = max(W - w, H - h)
            return (f, -(b[4] - w * h))
        seeds.sort(key=score)
        seeds = seeds[:cap]
    out = {}

    def store(key, cls, box=None):
        w, h, g, f, cnt = cls
        rec = out.get(key)
        if rec is None or f < rec['f'] - 1e-12:
            out[key] = dict(w=w, h=h, g=g, f=f, cnt=cnt,
                            x0=(box[0] if box is not None else 0.0),
                            x1=(box[1] if box is not None else 0.0),
                            y0=(box[2] if box is not None else 0.0),
                            y1=(box[3] if box is not None else 0.0))

    for b in seeds:
        cls = row.classify(b, fmax, nmax)
        if cls is not None:
            store((cls[0], cls[1], cls[2]), cls, b)
        stage_c(row, b, fmax, nmax, store)
    if verbose:
        print(f'  n={n} win={win.shape[0]} pockets={len(uniq)} seeds={len(seeds)} '
              f'types={len(out)} {time.time() - t0:.2f}s', flush=True)
    # store geometry: the representative tight box of each type
    res = []
    for (w, h, g), v in out.items():
        res.append(dict(src=n, w=w, h=h, g=g, f=v['f'], cnt=v['cnt'],
                        x0=v['x0'], x1=v['x1'], y0=v['y0'], y1=v['y1']))
    return res, row


_G = {}


def _init(csv_path, opts):
    _G['rows'] = load_rows(csv_path)
    _G['opts'] = opts


def _work(n):
    o = _G['opts']
    return n, scan_row(n, _G['rows'][n], o['step'], o['fmax'], o['nmax'], o['cap'],
                        o['off'])[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('csv')
    ap.add_argument('out')
    ap.add_argument('--step', type=float, default=0.25)
    ap.add_argument('--fmax', type=float, default=0.9)
    ap.add_argument('--nmax', type=int, default=31)
    ap.add_argument('--cap', type=int, default=400)
    ap.add_argument('--procs', type=int, default=30)
    ap.add_argument('--rows')
    ap.add_argument('--offx', type=float, default=0.0)
    ap.add_argument('--offy', type=float, default=0.0)
    a = ap.parse_args()
    rows = load_rows(a.csv)
    ns = sorted(rows) if not a.rows else [int(v) for v in a.rows.split(',')]
    print(f'{len(ns)} rows, step={a.step} fmax={a.fmax} nmax={a.nmax} cap={a.cap} '
          f'off=({a.offx},{a.offy})', flush=True)
    t0 = time.time()
    opts = dict(step=a.step, fmax=a.fmax, nmax=a.nmax, cap=a.cap, off=(a.offx, a.offy))
    if a.procs > 1:
        import multiprocessing as mp
        with mp.Pool(a.procs, initializer=_init, initargs=(a.csv, opts)) as pool:
            res = pool.map(_work, ns, chunksize=4)
    else:
        _init(a.csv, opts)
        res = [_work(n) for n in ns]
    lib = [g for _, gs in res for g in gs]
    json.dump(lib, open(a.out, 'w'))
    best = {}
    for g in lib:
        k = (g['w'], g['h'], g['g'])
        if k not in best or g['f'] < best[k]['f']:
            best[k] = g
    print(f'{time.time() - t0:.1f}s  {len(lib)} gadgets -> {len(best)} distinct (w,h,g)', flush=True)
    print(f'wrote {a.out}')


if __name__ == '__main__':
    main()
