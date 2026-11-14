"""Build + certify a staircase composition  (K x K grid  +  pockets on the seam).

Construction (the general m >= 1 case of the rule
`sum w <= K, sum h <= K, sum(w+h+1) <= 2K+1, n = K^2 + sum g`):

  box side  S = K + max_i f_i ;  pocket i sits at integer offset (X_i, Y_i),
  X non-decreasing, Y non-increasing, with the "one extra column or row between
  consecutive pockets" staggering of poc/gadget_squeeze/poc/compose.py;
  the K x K grid fills the two L-shaped corners A (anchored at (0,0)) and
  B (anchored at (S,S)) wherever a unit cell does not meet a pocket box.

Every pocket is copied from the **exact decimal strings** of its donor row at 50
digits (translation only, so the donor's internal contacts survive unchanged)
and inset by 1e-12 off the grid, so no dilation is needed.

usage:
  build_cert.py --pred pred.json --pockets pockets.json --report report.json \
                [--n 442 485] [--final final.csv] [--outdir data/built]
"""
import argparse
import collections
import csv
import json
import math
import os
import sys

import mpmath as mp
import numpy as np

sys.path.insert(0, 'poc/baseline/poc/src')
from squares.core import certify, kaggle_roundtrip

EPS = mp.mpf('1e-12')
SPREAD = mp.mpf('1e-11')   # gaps inside the fractional corner lattice (grid B)


# ------------------------------------------------------------------ row access
def load_rows(path):
    by = collections.OrderedDict()
    with open(path) as f:
        for r in csv.DictReader(f):
            by.setdefault(int(r['n']), []).append((r['square_id'], r['x'], r['y'], r['angle']))
    for n in by:
        by[n].sort(key=lambda r: int(r[0]))
    return by


def prep(rows):
    a = np.array([[float(x), float(y), float(t)] for _, x, y, t in rows])
    e = 0.5 * (np.abs(np.cos(a[:, 2])) + np.abs(np.sin(a[:, 2])))
    a[:, 0] -= (a[:, 0] - e).min()
    a[:, 1] -= (a[:, 1] - e).min()
    return a, e


def pocket_setup(rows, pk, transpose=False):
    """exact 50-digit pocket: local coordinates (bbox min at 0), exact W, H, count.

    The bbox is recomputed at 50 digits from the donor's decimal strings, so the
    pocket's own contacts are exact (only a translation is applied).
    """
    a, e = prep(rows)
    lx, hx, ly, hy = a[:, 0] - e, a[:, 0] + e, a[:, 1] - e, a[:, 1] + e
    # The scan recorded the tight bbox in float; a square sitting exactly on that
    # bbox edge can land 1e-16 on the wrong side once the angles are re-read at
    # 50 digits, which silently drops it from the pocket (measured: n=974's
    # (22,10,+7) came out 125 of 138 squares).  Select with a 1e-9 tolerance and
    # let the exact 50-digit bbox below be authoritative.
    m = ((lx >= pk['x0'] - 1e-9) & (hx <= pk['x1'] + 1e-9) &
         (ly >= pk['y0'] - 1e-9) & (hy <= pk['y1'] + 1e-9))
    idx = np.nonzero(m)[0]
    if len(idx) != pk['cnt']:
        print(f"    note: pocket n={pk['src']} ({pk['w']},{pk['h']},+{pk['g']}): "
              f"{len(idx)} squares selected vs recorded cnt {pk['cnt']} (tolerance)")
    with mp.workdps(50):
        ext = [mp.mpf(0.5) * (abs(mp.cos(mp.mpf(rows[i][3]))) + abs(mp.sin(mp.mpf(rows[i][3]))))
               for i in idx]
        xs = [mp.mpf(rows[i][1]) for i in idx]
        ys = [mp.mpf(rows[i][2]) for i in idx]
        X0 = min(x - t for x, t in zip(xs, ext))
        Y0 = min(y - t for y, t in zip(ys, ext))
        X1 = max(x + t for x, t in zip(xs, ext))
        Y1 = max(y + t for y, t in zip(ys, ext))
        sq = [dict(x=x - X0, y=y - Y0, angle=rows[i][3]) for i, x, y in zip(idx, xs, ys)]
        W, H = X1 - X0, Y1 - Y0
        if transpose:
            # exact 90-degree rotation: (x, y) -> (H - y, x).  A unit square is
            # invariant under a quarter turn, so the angle can stay unchanged.
            sq = [dict(x=H - s['y'], y=s['x'], angle=s['angle']) for s in sq]
            W, H = H, W
    return sq, W, H


# ------------------------------------------------------------------- building
def layout(K, parts):
    """parts: list of (w, h, fx, fy). Returns X, Y offsets or None."""
    m = len(parts)
    sw = sum(p[0] for p in parts)
    sh = sum(p[1] for p in parts)
    if sw > K or sh > K or sw + sh + m - 1 > 2 * K:
        return None
    nex = min(m - 1, K - sw)
    ney = (m - 1) - nex
    if ney > K - sh:
        return None
    ex = [1] * nex + [0] * ney
    X = [0] * m
    Y = [0] * m
    for i in range(1, m):
        X[i] = X[i - 1] + parts[i - 1][0] + ex[i - 1]
    for i in range(m - 2, -1, -1):
        Y[i] = Y[i + 1] + parts[i + 1][1] + (1 - ex[i])
    if X[-1] + parts[-1][0] > K or Y[0] + parts[0][1] > K:
        return None
    return X, Y


def layouts_all(K, parts):
    """every feasible (X, Y) staggering of `parts` along the anti-diagonal seam.

    The extra columns (ex) may go to any of the m-1 gaps, not just the first
    min(m-1, K-sum w) of them, so the grid that is left over can be larger for
    one choice than another (this is what rescues n=492).
    """
    m = len(parts)
    sw = sum(p[0] for p in parts)
    sh = sum(p[1] for p in parts)
    if sw > K or sh > K or sw + sh + m - 1 > 2 * K:
        return []
    out = []
    for nex in range(m):
        ney = (m - 1) - nex
        if nex > K - sw or ney > K - sh:
            continue
        ex = [1] * nex + [0] * ney
        X = [0] * m
        Y = [0] * m
        for i in range(1, m):
            X[i] = X[i - 1] + parts[i - 1][0] + ex[i - 1]
        for i in range(m - 2, -1, -1):
            Y[i] = Y[i + 1] + parts[i + 1][1] + (1 - ex[i])
        if X[-1] + parts[-1][0] > K or Y[0] + parts[0][1] > K:
            continue
        out.append((X, Y))
    return out


def grid_cells(K, S, parts, X, Y):
    """the free integer-lattice cells (grid A) + corner-anchored cells (grid B)."""
    m = len(parts)
    boxes = [(mp.mpf(X[i]), mp.mpf(Y[i]), mp.mpf(X[i]) + parts[i][0] + parts[i][2],
              mp.mpf(Y[i]) + parts[i][1] + parts[i][3]) for i in range(m)]
    grid = []
    for i in range(K):
        for j in range(K):
            if all(i + 1 <= X[t] or j + 1 <= Y[t] for t in range(m)):
                grid.append((mp.mpf(i) + mp.mpf('0.5'), mp.mpf(j) + mp.mpf('0.5')))
    if m > 1:
        for a in range(K):
            for b in range(K):
                # spread by a*SPREAD so neighbouring cells keep a positive gap
                x0 = S - 1 - a - a * SPREAD
                y0 = S - 1 - b - b * SPREAD
                if not any(x0 >= X[t] - mp.mpf('1e-12') and y0 >= Y[t] - mp.mpf('1e-12')
                           for t in range(m)):
                    continue
                if any(x0 < bx1 - EPS and x0 + 1 > bx0 + EPS and
                       y0 < by1 - EPS and y0 + 1 > by0 + EPS
                       for bx0, by0, bx1, by1 in boxes):
                    continue
                grid.append((x0 + mp.mpf('0.5'), y0 + mp.mpf('0.5')))
    return grid


def build(K, pockets, rows_all):
    """pockets: list of dict(src, w, h, g, f, boxh, squares). Returns rows list."""
    m = len(pockets)
    with mp.workdps(50):
        fmax = max(max(p['boxh'][0] - p['w'], p['boxh'][1] - p['h']) for p in pockets)
        S = mp.mpf(K) + fmax
    best = None
    if m > 1:
        # search every pocket order x seam staggering for the largest grid
        import itertools
        for order in itertools.permutations(range(m)):
            pt = [(pockets[i]['w'], pockets[i]['h'],
                   float(pockets[i]['boxh'][0]) - pockets[i]['w'],
                   float(pockets[i]['boxh'][1]) - pockets[i]['h']) for i in order]
            for X, Y in layouts_all(K, pt):
                cells = grid_cells(K, S, pt, X, Y)
                tot = sum(len(pockets[i]['squares']) for i in order) + len(cells)
                if best is None or tot > best[0]:
                    best = (tot, order, X, Y, cells)
        if best is None:
            return None, 'layout'
        _, order, X, Y, cells = best
        pockets = [pockets[i] for i in order]
        pockets_inv = order
    parts = [(p['w'], p['h'], float(p['boxh'][0]) - p['w'],
              float(p['boxh'][1]) - p['h']) for p in pockets]
    if m == 1:
        # proven single-pocket scheme (poc/faultline_dp2/poc/build_pocket.py):
        # pocket flush with the top-right corner, grid = L-shape {i<K-w} u {j<K-h}
        w, h = parts[0][0], parts[0][1]
        X, Y = [K - w], [K - h]
    out = []
    with mp.workdps(50):
        for p, x0, y0 in zip(pockets, X, Y):
            dx, dy = mp.mpf(x0) + EPS, mp.mpf(y0) + EPS
            for s in p['squares']:
                out.append(dict(x=s['x'] + dx, y=s['y'] + dy, angle=s['angle']))
    boxes = [(mp.mpf(X[i]), mp.mpf(Y[i]), mp.mpf(X[i]) + parts[i][0] + parts[i][2],
              mp.mpf(Y[i]) + parts[i][1] + parts[i][3]) for i in range(len(parts))]
    grid = grid_cells(K, S, parts, X, Y)
    for cx, cy in grid:
        out.append(dict(x=cx, y=cy, angle='0'))
    want = K * K + sum(p['g'] for p in pockets)
    return out, dict(S=S, fmax=fmax, total=len(out), want=want,
                     pocket=sum(len(p['squares']) for p in pockets), grid=len(grid),
                     boxes=boxes, X=X, Y=Y)


def to_rows(P, n):
    rows = []
    for i, s in enumerate(P):
        a = s['angle'] if isinstance(s['angle'], str) else mp.nstr(s['angle'], 45)
        rows.append(dict(square_id=str(i),
                         x=mp.nstr(s['x'], 45) if not isinstance(s['x'], str) else s['x'],
                         y=mp.nstr(s['y'], 45) if not isinstance(s['y'], str) else s['y'],
                         angle=a))
    if len(rows) != n:
        return None
    return rows


def write_final(path, groups):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['id', 'x', 'y', 'deg'])
        with mp.workdps(95):
            for n in sorted(groups):
                for r in sorted(groups[n], key=lambda r: int(r['square_id'])):
                    deg = mp.nstr(mp.mpf(str(r['angle'])) * 180 / mp.pi, 75)
                    w.writerow([f'{n:03}_{r["square_id"]}', 's' + str(r['x']),
                                's' + str(r['y']), 's' + deg])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', required=True)
    ap.add_argument('--pockets', required=True)
    ap.add_argument('--report', required=True)
    ap.add_argument('--csv', default='poc/swarm_pocket_lift_scan4/data/upstream-merged.csv')
    ap.add_argument('--n', type=int, nargs='+')
    ap.add_argument('--final')
    ap.add_argument('--outdir', default='poc/swarm_pocket_lift_scan4/data/built')
    a = ap.parse_args()
    rep = {r['n']: r['L'] for r in json.load(open(a.report))['results']}
    pred = json.load(open(a.pred))
    lib = json.load(open(a.pockets))
    by_src = collections.defaultdict(list)
    for g in lib:
        by_src[(g['src'], g['w'], g['h'], g['g'])].append(g)
    rows_all = load_rows(a.csv)
    L_inc = {n: mp.mpf(L) for n, L in rep.items()}
    done = {}
    for n in (a.n or sorted(int(k) for k in pred)):
        rec = pred.get(str(n))
        if rec is None:
            print(f'n={n}: no prediction, skip')
            continue
        K, j = rec['K'], rec['j']
        pockets = []
        ok = True
        for c in rec['comp']:
            cands = by_src.get((c['src'], c['w'], c['h'], c['g']), [])
            if not cands:                       # stored in the other orientation
                cands = by_src.get((c['src'], c['h'], c['w'], c['g']), [])
            if not cands:
                ok = False
                break
            pk = min(cands, key=lambda g: g['f'])
            rows = rows_all[pk['src']]
            tr = (c['w'], c['h']) != (pk['w'], pk['h'])
            sq, W, H = pocket_setup(rows, pk, tr)
            pockets.append(dict(src=pk['src'], w=c['w'], h=c['h'], g=c['g'], f=c['f'],
                                box=pk, boxh=(W, H), squares=sq, tr=tr))
        if not ok and len(rec['comp']) == 1:
            c = rec['comp'][0]
            cands = by_src.get((c['src'], c['h'], c['w'], c['g']), [])
            if cands:      # single pocket: the seam sums are symmetric under w<->h
                pk = min(cands, key=lambda g: g['f'])
                rows = rows_all[pk['src']]
                sq, W, H = pocket_setup(rows, pk, True)
                pockets = [dict(src=pk['src'], w=c['h'], h=c['w'], g=c['g'], f=c['f'],
                                box=pk, boxh=(W, H), squares=sq, tr=True)]
                print(f'  n={n}: using pocket ({c["h"]},{c["w"]},+{c["g"]}) from n={c["src"]} '
                      f'(untransposed)')
                ok = True
        if not ok:
            print(f'  n={n}: cannot realise the DP composition, skip')
            continue
        P, info = build(K, pockets, rows_all)
        if P is None:
            print(f'n={n}: layout failed')
            continue
        if info['total'] < n:
            print(f'n={n}: built only {info["total"]} squares, want {info["want"]} '
                  f'(K={K}), skip')
            continue
        if info['total'] > n:
            # a surplus is harmless: any subset of a valid packing is valid.
            # Drop the grid cells farthest from the origin first (keeps L low).
            npk = info['pocket']
            grid = sorted(range(npk, len(P)), key=lambda i: -(float(P[i]['x']) + float(P[i]['y'])))
            drop = set(grid[:info['total'] - n])
            P = [p for i, p in enumerate(P) if i not in drop]
            print(f'n={n}: dropped {info["total"] - n} surplus grid squares')
        rows = to_rows(P, n)
        if rows is None:
            print(f'n={n}: assembly failed')
            continue
        try:
            Llit = certify(n, rows)
            Lpan = certify(n, kaggle_roundtrip({n: rows})[n])
        except Exception as ex:                  # invalid packing
            print(f'n={n}: certify raised {ex}')
            continue
        inc = L_inc[n]
        dlit, dpan = mp.mpf(Llit) - inc, mp.mpf(Lpan) - inc
        gain = min(dlit, dpan) / mp.sqrt(n)
        comp = ' + '.join(f"({p['w']},{p['h']},+{p['g']})<n={p['src']}>" for p in pockets)
        print(f'n={n} K={K} j={j} f={rec["f"]:.9f} side={info["S"]:.9f}  {comp}')
        print(f'   incumbent {inc}  ->  literal {Llit}  (dL {mp.nstr(dlit, 8)})')
        print(f'                          pandas  {Lpan}  (dL {mp.nstr(dpan, 8)})')
        win = dlit < mp.mpf('-1e-9') and dpan < mp.mpf('-1e-9')
        print(f'   pocket {info["pocket"]} + grid {info["grid"]} = {info["total"]} squares '
              f'{"STRICT WIN" if win else "NO WIN"}   gain {mp.nstr(gain, 5)}')
        if win:
            done[n] = rows
            if a.outdir:
                os.makedirs(a.outdir, exist_ok=True)
                json.dump(rows, open(os.path.join(a.outdir, f'n{n}.json'), 'w'))
    if a.final is not None:
        write_final(a.final, done)
        print(f'wrote {a.final}: {len(done)} groups, {sum(len(v) for v in done.values())} rows '
              f'{sorted(done)}')
        if done:
            tot = sum((L_inc[n] - mp.mpf(certify(n, done[n]))) / mp.sqrt(n) for n in done)
            print(f'total score gain {mp.nstr(tot, 8)} (literal)')


if __name__ == '__main__':
    main()
