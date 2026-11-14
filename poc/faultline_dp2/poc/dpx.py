"""Fast staircase DP + gadget threshold tool.

Same semantics as `gaddp.py` (unbounded knapsack over (w, h, g) items, both
orientations, f levels swept in increasing order, each n keeps the smallest f
that reaches it), but with Pareto pruning of the item set so a full K <= 31
sweep costs ~1 s instead of ~6 min.

  dpx.py base  --lib gadlib.json --report report.json [--out base.json]
  dpx.py thr   --lib ... --report ... --shape 4,7,3 --lo 0.2 --hi 0.9 [--n 12]
  dpx.py extra --lib ... --report ... --extra extra.json --out win.json
"""
import argparse
import json
import math

import numpy as np

KMAX = 31
NEG = -10 ** 6


def lib_items(lib):
    items = {}
    for i, g in enumerate(lib):
        f = max(g['fx'], g['fy'])
        for (w, h) in ((g['w'], g['h']), (g['h'], g['w'])):
            k = (w, h, g['g'])
            if k not in items or f < items[k][0]:
                items[k] = (f, g['src'])
    return [(w, h, g, v[0], v[1]) for (w, h, g), v in items.items()]


def pareto(items):
    """drop items dominated by another: w<=w', h<=h', g>=g', f<=f' (one of them strict)."""
    it = sorted(items, key=lambda r: (r[3], -r[2], r[0] + r[1]))
    keep = []
    for a in it:
        dom = False
        for b in keep:
            if b[0] <= a[0] and b[1] <= a[1] and b[2] >= a[2]:
                dom = True
                break
        if not dom:
            keep.append(a)
    return keep


def knap(items, K, S=None, best=None):
    """in-place unbounded knapsack; returns `best` (x, y, s) grid."""
    if S is None:
        S = 2 * K + 1
    if best is None:
        best = np.full((K + 1, K + 1, S + 1), NEG, np.int32)
        best[0, 0, 0] = 0
    for w, h, gg, f, _ in items:
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


def dp_sides(items, Kmax=KMAX, upto=None):
    """n -> (K, f) using the smallest f that reaches n.

    One pass per K: items are inserted in increasing f order, so `best.max()`
    right after each f level is exactly the knapsack optimum at that level.
    """
    size = max(r[0] + r[1] + 1 for r in items)
    by_f = {}
    for r in items:
        by_f.setdefault(round(r[3], 12), []).append(r)
    fs = sorted(by_f)
    if upto is not None:
        fs = [f for f in fs if f >= upto - 1e-12]
    pred = {}
    for K in range(2, Kmax + 1):
        todo = [j for j in range(1, 2 * K + 1) if K * K + j <= 1000 and K * K + j not in pred]
        if not todo:
            continue
        best = np.full((K + 1, K + 1, 2 * K + 2), NEG, np.int32)
        best[0, 0, 0] = 0
        for f in fs:
            sel = [r for r in by_f[f] if r[0] <= K and r[1] <= K]
            if sel:
                knap(sel, K, best=best)
            gmax = int(best.max())
            if gmax <= 0:
                continue
            for j in todo:
                if j <= gmax:
                    pred[K * K + j] = (K, f)
            todo = [j for j in todo if K * K + j not in pred]
            if not todo:
                break
    return pred


def improvements(pred, L, tol=1e-7):
    rows = []
    for n, (K, f) in pred.items():
        d = L[n] - (K + f)
        if d > tol:
            rows.append((d / math.sqrt(n), n, L[n], K + f, K, f))
    rows.sort(key=lambda r: -r[0])
    return rows


def load_items(a):
    lib = json.load(open(a.lib))
    items = lib_items(lib)
    n0 = len(items)
    items = pareto(items)
    extra = []
    if getattr(a, 'extra', None):
        for e in json.load(open(a.extra)):
            for (w, h) in ((e['w'], e['h']), (e['h'], e['w'])):
                extra.append((w, h, e['g'], e['f'], e.get('src', 'extra')))
    items = pareto(items + extra)
    print(f'items: {n0} raw -> {len(items)} after pareto (+{len(extra)} extra)')
    return items, lib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['base', 'thr', 'extra'])
    ap.add_argument('--lib', required=True)
    ap.add_argument('--report', required=True)
    ap.add_argument('--extra')
    ap.add_argument('--shape')
    ap.add_argument('--lo', type=float, default=0.2)
    ap.add_argument('--hi', type=float, default=1.2)
    ap.add_argument('--n', type=int, default=10)
    ap.add_argument('--out')
    ap.add_argument('--kmax', type=int, default=KMAX)
    a = ap.parse_args()
    L = {r['n']: float(r['L']) for r in json.load(open(a.report))['results']}
    items, lib = load_items(a)

    if a.mode == 'base':
        pred = dp_sides(items, a.kmax)
        rows = improvements(pred, L)
        print(f'predicted cells {len(pred)}; DP strictly better at {len(rows)} n; '
              f'score gain {sum(r[0] for r in rows):.3e}')
        for sg, n, ours, Lp, K, f in rows[:20]:
            print(f'  n={n} K={K} j={n - K * K}: ours {ours:.9f} -> {Lp:.9f} (gain {sg:.3e})')
        if a.out:
            json.dump({str(n): dict(K=v[0], f=v[1]) for n, v in pred.items()}, open(a.out, 'w'))
        return

    if a.mode == 'thr':
        w, h, g = (int(v) for v in a.shape.split(','))
        base = dp_sides(items, a.kmax)
        base_sides = {n: v[0] + v[1] for n, v in base.items()}

        def gain_at(f):
            it = pareto(items + [(w, h, g, f, 'probe'), (h, w, g, f, 'probe')])
            pred = dp_sides(it, a.kmax)
            tot = 0.0
            hits = []
            for n, (K, ff) in pred.items():
                old = base_sides.get(n, 1e9)
                d = old - (K + ff)
                if d > 1e-7:
                    tot += d / math.sqrt(n)
                    hits.append(n)
            return tot, sorted(hits)

        lo, hi = a.lo, a.hi
        g_lo = gain_at(lo)
        g_hi = gain_at(hi)
        print(f'({w},{h},{g}) f={lo:.4f}: gain {g_lo[0]:.4e} hits {g_lo[1][:a.n]}')
        print(f'({w},{h},{g}) f={hi:.4f}: gain {g_hi[0]:.4e} hits {g_hi[1][:a.n]}')
        for _ in range(a.n):
            mid = (lo + hi) / 2
            gm = gain_at(mid)
            if gm[0] > 0:
                lo, best = mid, gm
            else:
                hi = mid
        print(f'({w},{h},{g}) pays for f <= {lo:.6f}: gain {best[0]:.4e} '
              f'at {len(best[1])} n: {best[1][:24]}')
        if a.out:
            json.dump(dict(shape=[w, h, g], f_max=lo, gain=best[0], hits=best[1]),
                      open(a.out, 'w'))
        return

    pred = dp_sides(items, a.kmax)
    base_items = pareto(lib_items(lib))
    base = dp_sides(base_items, a.kmax)
    rows = improvements(pred, L)
    new = [r for r in rows if r[1] not in base]
    print(f'with extra: DP better at {len(rows)} n; {len(new)} of them are new vs the base library')
    for sg, n, ours, Lp, K, f in rows[:40]:
        tag = 'NEW' if n not in base else '    '
        print(f'  {tag} n={n} K={K} j={n - K * K}: ours {ours:.9f} -> {Lp:.9f} (gain {sg:.3e})')
    if a.out:
        json.dump({str(r[1]): dict(L=r[3], K=r[4], f=r[5], gain=r[0], new=r[1] not in base)
                   for r in rows}, open(a.out, 'w'))


if __name__ == '__main__':
    main()
