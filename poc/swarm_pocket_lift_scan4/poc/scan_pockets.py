"""Exhaustive proper-sub-pocket scan -- swarm_pocket_lift_scan4.

Same 3 stages as `poc/pocket_lift_scan/scan_pockets.py`
    A  offset-lattice window enumeration + exact containment count screen
    B  exact tight bbox of the contained squares -> (w,h,g,f)
    C  coordinate descent over the row's exact boundary values
plus three changes that make `--step 0.0625` both affordable and stronger:

1. **Per-mask screen cache (lossless).**  The screen `count >= w*h + 1` for a
   y-window depends only on (contained-square set, integer window height hh).
   At step 0.0625 there are ~66k-130k y-windows with >= 2 squares but only
   ~2k-12k distinct (set, hh) pairs, and only ~1k-6k distinct sets, so the
   O(ng^2) 2-D cumulative count runs per *set*.  Measured on n=686: stage A
   74 s -> ~9 s per row.  (The earlier attempt to key on the set alone lost
   boxes -- two windows with the same set but different `hh` screen against a
   different `w*h` -- so the key is (set, hh).)

2. **Stage B over every window, no `--cap` truncation.**  `--cap` now only
   limits the *stage-C local search* seeds.  Every stage-B tight box is
   classified, so the type library is a superset of what the old
   cap-truncated pipeline could see (measured on n=686 at step 0.0625: 2972
   types instead of 1526, 624 of the old types with a smaller f, and the
   decisive `(9,16,+5)` recovered at its exact 0.6967165439563).

3. **Grid extended one cell below the origin.**  With `--offx/--offy > 0` the old
   grid started *at* the offset, so the row's own first squares (lx or ly < off)
   could never be inside a window; the lattice is now shifted down by one cell
   (same phase mod step) and `ng` grown accordingly, so whole-row pockets are
   reachable at any offset.

usage:
  scan_pockets.py merged.csv out.json [--step 0.25] [--fmax 0.9] [--procs 30]
                  [--nmax 31] [--rows 1,2,3] [--cap 1200] [--offx 0] [--offy 0]
"""
import argparse
import collections
import csv as _csv
import json
import math
import time

import numpy as np


# ---------------------------------------------------------------- row loading
def load_rows(path):
    by = collections.defaultdict(list)
    with open(path) as f:
        for r in _csv.DictReader(f):
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


def stage_a(lx, hx, ly, hy, step, maxw=31.75, off=(0.0, 0.0), stats=None):
    """grid windows whose contained-square count could be >= w*h+1.

    Returns a (B,4) array of `(X0, X1, Y0, Y1)` grid windows.
    """
    L = max(hx.max(), hy.max())
    ox, oy = off[0] - step, off[1] - step      # same lattice, one cell lower
    ng = int(math.ceil((L + step - oy) / step)) + 2
    cl = lambda v, o: np.clip(np.floor((v - o) / step + 1e-9).astype(np.int64), 0, ng - 1)
    ch = lambda v, o: np.clip(np.ceil((v - o) / step - 1e-9).astype(np.int64), 0, ng - 1)
    pl, qh = cl(lx, ox), ch(hx, ox)
    pc, qd = cl(ly, oy), ch(hy, oy)
    P = window_pairs(ng, step, maxw)
    wi = np.floor(step * (P[:, 1] - P[:, 0]) + 1e-9).astype(np.int64)
    flat = pl * ng + qh
    # pass 1: group the y-windows by contained-square set
    groups = {}
    order = []
    for k in range(P.shape[0]):
        c, d = int(P[k, 0]), int(P[k, 1])
        m = (pc >= c) & (qd <= d)
        if m.sum() < 2:
            continue
        key = np.packbits(m).tobytes()
        g = groups.get(key)
        if g is None:
            groups[key] = g = (np.nonzero(m)[0], [])
            order.append(key)
        g[1].append(k)
    npass = sum(len(g[1]) for g in groups.values())
    # pass 2: one 2-D cumulative count per set, then screen every (k, hh)
    rows = []
    for key in order:
        idx, ks = groups[key]
        Sc = np.bincount(flat[idx], minlength=ng * ng).reshape(ng, ng)
        Sc = np.cumsum(Sc, axis=1)
        Sc = np.cumsum(Sc[::-1], axis=0)[::-1]
        cc = Sc[P[:, 0], P[:, 1]]
        cache = {}
        for k in ks:
            c, d = int(P[k, 0]), int(P[k, 1])
            hh = int(math.floor(step * (d - c) + 1e-9))
            ok = cache.get(hh)
            if ok is None:
                ok = cache[hh] = np.nonzero(cc - wi * hh >= 1)[0]
            if ok.size:
                rows.append((k, ok))
    if stats is not None:
        stats['ypass'] = npass
        stats['ymasks'] = len(groups)
    cands = []
    for k, ok in rows:
        c, d = P[k, 0], P[k, 1]
        y0, y1 = c * step + oy, d * step + oy
        cands.append(np.column_stack([P[ok, 0] * step + ox, P[ok, 1] * step + ox,
                                      np.full(ok.size, y0), np.full(ok.size, y1)]))
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
def seed_score(b):
    """stage-C priority: small f first, then large surplus, then stable geometry."""
    W, H = b[1] - b[0], b[3] - b[2]
    w, h = math.floor(W + 1e-6), math.floor(H + 1e-6)
    f = max(W - w, H - h)
    return (f, -(b[4] - w * h), b[0], b[2])


def scan_row(n, a, step, fmax, nmax, cap=400, off=(0.0, 0.0), verbose=False):
    row = Row(n, a)
    t0 = time.time()
    stats = {}
    win = stage_a(row.lx, row.hx, row.ly, row.hy, step, maxw=float(nmax) + 0.75,
                  off=off, stats=stats)
    tA = time.time() - t0
    if win.shape[0] == 0:
        stats.update(tA=tA, tB=0.0, tC=0.0, win=0, pockets=0, seeds=0)
        return [], row, stats
    # ---- stage B on EVERY window
    cands = []
    B = 4000
    for s in range(0, win.shape[0], B):
        cands.extend(row.refine_many(win[s:s + B]))
    uniq = {}
    for b in cands:
        k = (round(b[0], 6), round(b[1], 6), round(b[2], 6), round(b[3], 6))
        if k not in uniq or b[4] > uniq[k][4]:
            uniq[k] = b
    tB = time.time() - t0 - tA
    seeds = sorted(uniq.values(), key=seed_score)
    nseeds = len(seeds)
    if cap and len(seeds) > cap:
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

    nB = 0
    for b in uniq.values():          # every stage-B box, no truncation
        cls = row.classify(b, fmax, nmax)
        if cls is not None:
            store((cls[0], cls[1], cls[2]), cls, b)
            nB += 1
    for b in seeds:                  # stage C on the best `cap` seeds
        stage_c(row, b, fmax, nmax, store)
    tC = time.time() - t0 - tA - tB
    res = []
    for (w, h, g), v in out.items():
        res.append(dict(src=n, w=w, h=h, g=g, f=v['f'], cnt=v['cnt'],
                        x0=v['x0'], x1=v['x1'], y0=v['y0'], y1=v['y1']))
    if verbose:
        print(f'  n={n} win={win.shape[0]} masks={stats.get("ymasks")} '
              f'pockets={len(uniq)} seeds={nseeds} used={len(seeds)} '
              f'types={len(out)}  A {tA:.2f}s B {tB:.2f}s C {tC:.2f}s', flush=True)
    stats.update(tA=tA, tB=tB, tC=tC, win=int(win.shape[0]), pockets=len(uniq),
                 seeds=nseeds, used=len(seeds), types=len(out))
    return res, row, stats


_G = {}


def _init(csv_path, opts):
    _G['rows'] = load_rows(csv_path)
    _G['opts'] = opts


def _work(n):
    o = _G['opts']
    res, _, st = scan_row(n, _G['rows'][n], o['step'], o['fmax'], o['nmax'], o['cap'],
                          o['off'], verbose=o['verbose'])
    return n, res, st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('csv')
    ap.add_argument('out')
    ap.add_argument('--step', type=float, default=0.25)
    ap.add_argument('--fmax', type=float, default=0.9)
    ap.add_argument('--nmax', type=int, default=31)
    ap.add_argument('--cap', type=int, default=1200, help='stage-C seed cap (0 = all)')
    ap.add_argument('--procs', type=int, default=30)
    ap.add_argument('--rows')
    ap.add_argument('--offx', type=float, default=0.0)
    ap.add_argument('--offy', type=float, default=0.0)
    ap.add_argument('--verbose', action='store_true')
    a = ap.parse_args()
    rows = load_rows(a.csv)
    ns = sorted(rows) if not a.rows else [int(v) for v in a.rows.split(',')]
    print(f'{len(ns)} rows, step={a.step} fmax={a.fmax} nmax={a.nmax} cap={a.cap} '
          f'off=({a.offx},{a.offy})', flush=True)
    t0 = time.time()
    opts = dict(step=a.step, fmax=a.fmax, nmax=a.nmax, cap=a.cap,
                off=(a.offx, a.offy), verbose=a.verbose)
    if a.procs > 1:
        import multiprocessing as mp
        with mp.Pool(a.procs, initializer=_init, initargs=(a.csv, opts)) as pool:
            res = pool.map(_work, ns, chunksize=2)
    else:
        _init(a.csv, opts)
        res = [_work(n) for n in ns]
    lib = [g for _, gs, _ in res for g in gs]
    json.dump(lib, open(a.out, 'w'))
    best = {}
    for g in lib:
        k = (g['w'], g['h'], g['g'])
        if k not in best or g['f'] < best[k]['f']:
            best[k] = g
    tot = time.time() - t0
    sts = [st for _, _, st in res]
    mean = lambda f: sum(s.get(f, 0) for s in sts) / max(1, len(sts))
    print(f'{tot:.1f}s  {len(lib)} gadgets -> {len(best)} distinct (w,h,g)  '
          f'mean stage A {mean("tA"):.2f}s B {mean("tB"):.2f}s C {mean("tC"):.2f}s '
          f'seeds {mean("seeds"):.0f} used {mean("used"):.0f}', flush=True)
    print(f'sums: windows {sum(s.get("win", 0) for s in sts)} '
          f'ypass {sum(s.get("ypass", 0) for s in sts)} '
          f'masks {sum(s.get("ymasks", 0) for s in sts)} '
          f'seeds {sum(s.get("seeds", 0) for s in sts)} '
          f'used {sum(s.get("used", 0) for s in sts)}', flush=True)
    print(f'wrote {a.out}')


if __name__ == '__main__':
    main()
