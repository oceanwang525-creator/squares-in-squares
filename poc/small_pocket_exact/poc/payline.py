"""Exact pay line of a hypothetical gadget shape against the current incumbent.

Reuses the repo DP of `poc/faultline_dp2/poc/dpx.py` (same semantics as
`gaddp.py`: unbounded knapsack over (w,h,g) items, both orientations,
f levels swept in increasing order) and the official 50-digit side table
`poc/faultline_dp2/data/report.json` (score 22.15325720501231).

  payshape.py --shapes 3,2,1 4,7,3 ... [--n 16]
"""
import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DP2 = os.path.join(HERE, '..', '..', 'faultline_dp2')
sys.path.insert(0, os.path.join(DP2, 'poc'))

import dpx                                        # noqa: E402

LIB = os.path.join(DP2, 'data', 'gadlib.json')
REPORT = os.path.join(DP2, 'data', 'report.json')


def load():
    items = dpx.pareto(dpx.lib_items(json.load(open(LIB))))
    L = {r['n']: float(r['L']) for r in json.load(open(REPORT))['results']}
    base = dpx.dp_sides(items, 31)
    base_sides = {n: v[0] + v[1] for n, v in base.items()}
    return items, L, base_sides


def gain_at(items, L, base_sides, w, h, g, f):
    it = dpx.pareto(items + [(w, h, g, f, 'probe'), (h, w, g, f, 'probe')])
    pred = dpx.dp_sides(it, 31)
    tot = 0.0
    hits = []
    for n, (K, ff) in pred.items():
        old = base_sides.get(n, 1e9)
        d = old - (K + ff)
        if d > 1e-7:
            tot += d / math.sqrt(n)
            hits.append((d / math.sqrt(n), n, K, ff, d))
    hits.sort(reverse=True)
    return tot, hits


def threshold(items, L, base_sides, w, h, g, lo=0.3, hi=1.6, steps=9, nbest=6):
    """largest f with positive gain (gain is monotone decreasing in f in practice)."""
    g_hi = gain_at(items, L, base_sides, w, h, g, hi)[0]
    g_lo = gain_at(items, L, base_sides, w, h, g, lo)[0]
    if g_hi > 0:
        return hi, None
    if g_lo <= 0:
        return None, None
    best = None
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        tot, hits = gain_at(items, L, base_sides, w, h, g, mid)
        if tot > 0:
            lo, best = mid, (tot, hits)
        else:
            hi = mid
    return lo, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shapes', nargs='+', required=True)
    ap.add_argument('--n', type=int, default=6)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    items, L, base_sides = load()
    print('library %d items after pareto; %d base cells' % (len(items), len(base_sides)),
          flush=True)
    out = {}
    for s in a.shapes:
        w, h, g = (int(v) for v in s.split(','))
        N = w * h + g
        fmax, best = threshold(items, L, base_sides, w, h, g)
        rec = dict(shape=[w, h, g], N=N, f_max=fmax,
                   gain=(best[0] if best else 0.0),
                   hits=([(round(x[0], 9), x[1], x[2], round(x[3], 6)) for x in best[1][:a.n]]
                         if best else []))
        out[s] = rec
        if fmax is None:
            print('(%d,%d,+%d) N=%d: NO gain at any f in the scanned range' % (w, h, g, N),
                  flush=True)
        else:
            print('(%d,%d,+%d) N=%d: pays only if f <= %.6f  (gain %.3e, density %.3f) '
                  'best n: %s' % (w, h, g, N, fmax, rec['gain'],
                                  N / ((w + fmax) * (h + fmax)),
                                  ', '.join('%d(g%.1e)' % (x[1], x[0]) for x in best[1][:a.n])),
                  flush=True)
    if a.out:
        json.dump(out, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
