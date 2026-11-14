"""Staircase composition DP over a sub-gadget library (corrected composition rule).

Gadget i: pocket (w_i + f) x (h_i + f) holding w_i*h_i + g_i unit squares.
m gadgets fit along the BL/TR anti-diagonal seam of a (K + f) box iff

    sum w <= K,   sum h <= K,   sum (w + h + 1) <= 2K + 1

and then the packing has n = K^2 + sum g squares at side K + f.

This is a numpy re-implementation of `poc/rectangle_gadgets/poc/gaddp.py`
(which needs numba, not installed in this environment).  Same semantics:
unbounded knapsack over distinct (w, h, g) items with the smallest f, both
orientations; f levels are swept in increasing order and each n keeps the
first (smallest) f that reaches it.

usage:
  gaddp.py --lib gadlib.json --report report.json [--extra extra.json] [--out out.json]
"""
import argparse
import json
import math

import numpy as np

KMAX = 31
NEG = -10 ** 6


def lib_items(lib):
    """distinct (w, h, g) -> smallest f, both orientations kept."""
    items = {}
    for i, g in enumerate(lib):
        f = max(g['fx'], g['fy'])
        for (w, h) in ((g['w'], g['h']), (g['h'], g['w'])):
            k = (w, h, g['g'])
            if k not in items or f < items[k][0]:
                items[k] = (f, i)
    return items


def knap(W, H, G, K):
    """unbounded knapsack: best[x, y, s] = max gain with exact sums; -1 unreachable."""
    S = 2 * K + 1
    best = np.full((K + 1, K + 1, S + 1), NEG, np.int32)
    best[0, 0, 0] = 0
    for w, h, gg in zip(W, H, G):
        d = w + h + 1
        if w > K or h > K or d > S:
            continue
        for x in range(w, K + 1):
            src = best[x - w, :K + 1 - h, :S + 1 - d]
            dst = best[x, h:, d:]
            cand = np.where(src >= 0, src + gg, -1)
            np.maximum(dst, cand, out=dst)
    best[best < 0] = -1
    return best


def reachable_gains(W, H, G, K, S):
    best = knap(W, H, G, K)
    return int(best.max())


def run(items, Kmax=KMAX, verbose=False):
    """pred[n] = (K, f, gain) using the smallest f that reaches n at that K."""
    keys = sorted(items, key=lambda k: items[k][0])
    fs = sorted(set(round(items[k][0], 12) for k in keys))
    pred = {}
    for K in range(2, Kmax + 1):
        todo = [j for j in range(1, 2 * K + 1) if K * K + j <= 1000 and K * K + j not in pred]
        if not todo:
            continue
        for f in fs:
            sel = [k for k in keys if items[k][0] <= f + 1e-12 and k[0] <= K and k[1] <= K]
            if not sel:
                continue
            W = np.array([k[0] for k in sel]); H = np.array([k[1] for k in sel])
            G = np.array([k[2] for k in sel])
            gmax = reachable_gains(W, H, G, K, 2 * K + 1)
            if gmax <= 0:
                continue
            hit = False
            for j in todo:
                if j <= gmax:
                    pred[K * K + j] = dict(K=K, f=f, gain=j)
                    hit = True
            todo = [j for j in todo if K * K + j not in pred]
            if not todo:
                break
            if verbose:
                print(f'  K={K} f={f:.6f} gmax={gmax} hit={hit} left={len(todo)}')
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lib', required=True)
    ap.add_argument('--report', required=True)
    ap.add_argument('--extra', default=None, help='json list of {w,h,g,f,src}')
    ap.add_argument('--out', default=None)
    ap.add_argument('--kmax', type=int, default=KMAX)
    a = ap.parse_args()

    lib = json.load(open(a.lib))
    L = {r['n']: float(r['L']) for r in json.load(open(a.report))['results']}
    items = lib_items(lib)
    n_lib = len(items)
    if a.extra:
        for e in json.load(open(a.extra)):
            for (w, h) in ((e['w'], e['h']), (e['h'], e['w'])):
                k = (w, h, e['g'])
                if k not in items or e['f'] < items[k][0] - 1e-12:
                    items[k] = (e['f'], e.get('src', 'extra'))
        print(f'added {len(json.load(open(a.extra)))} extra gadgets '
              f'-> {len(items)} items (was {n_lib})')

    pred = run(items, Kmax=a.kmax)
    rows = []
    for n, p in sorted(pred.items()):
        d = L[n] - (p['K'] + p['f'])
        if d > 1e-7:
            rows.append((d / math.sqrt(n), n, L[n], p['K'] + p['f'], p))
    rows.sort(key=lambda r: -r[0])
    print(f'library {len(lib)} gadgets; {len(items)} items; predicted cells {len(pred)}; '
          f'DP strictly better at {len(rows)} n; score gain {sum(r[0] for r in rows):.6f}')
    for sg, n, ours, Lp, p in rows[:40]:
        print(f'  n={n} K={p["K"]} j={n - p["K"] ** 2}: ours {ours:.6f} -> {Lp:.6f} '
              f'(gain {sg:.3e})')
    if a.out:
        json.dump({str(r[1]): dict(L=r[3], K=r[4]['K'], f=r[4]['f'], se=1) for r in rows},
                  open(a.out, 'w'))


if __name__ == '__main__':
    main()
