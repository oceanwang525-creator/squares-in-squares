"""New gadget supply: CLEAN SUB-RECTANGLE CUTS of the pockets we already have.

gadlib.py only reports the minimal clean rectangle around each tilted cluster (and around
cluster unions), so the library contains the *whole* pocket but almost none of its
lower-left / lower-right / upper-left / upper-right sub-boxes.  A sub-box of a (w+f) x (h+f)
pocket that no square straddles is itself a legal gadget: (w'+f') x (h'+f') holding
w'h' + g' squares, and the staircase composition only ever asks a pocket to sit inside its
own box, so any clean sub-box can be re-anchored at the origin and used directly.

This turns 889 pockets into tens of thousands of candidate gadgets at near-zero cost, and in
particular produces thin shapes ((4,h,+g) and friends) that our packings never contain as
clusters but the value map says would pay.

usage: cutlib.py gadlib.json out.json [--max-area ...]
"""
import argparse, json, math, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gadlib import clean_rect  # noqa: E402  (identical clean-cut growth rule)


def extents(T):
    return 0.5 * (np.abs(np.cos(T)) + np.abs(np.sin(T)))


def positions(lo, hi, nmax=8):
    """Integer cut positions strictly inside (lo, hi), coarsened when the box is large."""
    a, b = int(math.ceil(lo + 1e-9)), int(math.floor(hi - 1e-9))
    vals = list(range(a, b + 1))
    if len(vals) > nmax:
        idx = np.linspace(0, len(vals) - 1, nmax).round().astype(int)
        vals = sorted({vals[i] for i in idx})
    return vals


def cuts_of(g, nmax):
    sq = np.asarray(g['sq'], float)
    if len(sq) == 0:
        return []
    e = extents(sq[:, 2])
    W = (sq[:, 0] + e).max(); H = (sq[:, 1] + e).max()
    xs = positions(0, W, nmax); ys = positions(0, H, nmax)
    out = []
    seen = set()
    xint = [(0.0, float(x)) for x in xs] + [(float(x), W) for x in xs]
    yint = [(0.0, float(y)) for y in ys] + [(float(y), H) for y in ys]
    for x0, x1 in xint:
        for y0, y1 in yint:
            if x1 - x0 < 2 or y1 - y0 < 2:
                continue
            X0, X1, Y0, Y1, idx = clean_rect(sq, e, x0, x1, y0, y1)
            Wc, Hc = X1 - X0, Y1 - Y0
            w, h = int(Wc + 1e-7), int(Hc + 1e-7)
            cnt = len(idx)
            gg = cnt - w * h
            if gg <= 0 or w < 2 or h < 2:
                continue
            key = (w, h, gg, round(Wc - w, 9), round(Hc - h, 9), round(X0, 6), round(Y0, 6))
            if key in seen:
                continue
            seen.add(key)
            sub = sq[idx].copy(); sub[:, 0] -= X0; sub[:, 1] -= Y0
            out.append(dict(src=g.get('src'), w=w, h=h, g=int(gg), fx=float(Wc - w), fy=float(Hc - h),
                            f=float(max(Wc - w, Hc - h)), cnt=int(cnt), whole=False, sq=sub.tolist(),
                            cut=f'{g["w"]}x{g["h"]}'))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('lib')
    ap.add_argument('out')
    ap.add_argument('--nmax', type=int, default=8)
    ap.add_argument('--max-sq', type=int, default=700, help='skip pockets with more squares')
    a = ap.parse_args()
    lib = json.load(open(a.lib))
    sel = [g for g in lib if len(g['sq']) <= a.max_sq]
    print(f'{len(lib)} pockets, {len(sel)} with <= {a.max_sq} squares', flush=True)
    cuts = []
    for i, g in enumerate(sel):
        cuts += cuts_of(g, a.nmax)
        if (i + 1) % 100 == 0:
            print(f'  {i+1}/{len(sel)} pockets -> {len(cuts)} cuts', flush=True)
    # keep the best f per (w, h, g)
    best = {}
    for c in cuts:
        k = (c['w'], c['h'], c['g'])
        if k not in best or c['f'] < best[k]['f'] - 1e-12:
            best[k] = c
    print(f'{len(cuts)} cut gadgets; {len(best)} distinct (w,h,g)', flush=True)
    cur = {}
    for g in lib:
        k = (g['w'], g['h'], g['g'])
        f = max(g['fx'], g['fy'])
        for kk in (k, (g['h'], g['w'], g['g'])):
            if kk not in cur or f < cur[kk]:
                cur[kk] = f
    new = [c for k, c in best.items() if k not in cur]
    better = [c for k, c in best.items() if k in cur and c['f'] < cur[k] - 1e-9]
    print(f'  of which {len(new)} shapes are new and {len(better)} improve an existing shape')
    for c in sorted(better, key=lambda c: cur[(c['w'], c['h'], c['g'])] - c['f'])[:20]:
        print(f'   improves ({c["w"]},{c["h"]},+{c["g"]}): {cur[(c["w"],c["h"],c["g"])]:.6f} -> {c["f"]:.6f}')
    for c in sorted(new, key=lambda c: c['f'])[:20]:
        print(f'   new ({c["w"]},{c["h"]},+{c["g"]}) f={c["f"]:.6f} from n={c["src"]} cut of {c["cut"]}')
    json.dump(lib + list(best.values()), open(a.out, 'w'))
    print('wrote', a.out, len(lib) + len(best))


if __name__ == '__main__':
    main()
