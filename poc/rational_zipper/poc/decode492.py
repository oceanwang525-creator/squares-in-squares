"""Decode the shipped n=492 build (poc/pocket_lift_scan2/data/built/n492.json).

The build is  pocket(5,13,+3)<-n585  +  pocket(16,9,+5)<-n686  +  275 integer
cells.  Both pockets are used TRANSPOSED (pred2.json).  This script

  1. reads the composition from pred2.json,
  2. re-cuts each pocket's window out of its donor row in merged.csv,
  3. finds the rigid translation that embeds that window in the build and
     verifies the whole window lands on build squares,
  4. reports the leftover (integer-lattice) squares.

usage: decode492.py [--built ...] [--pred ...]
"""
import argparse
import csv
import json
import math
from collections import Counter

import numpy as np


def read_csv_n(path, n):
    out = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if int(r['n']) == n:
                out.append((float(r['x']), float(r['y']), float(r['angle'])))
    return out


def embed(window, built, tol=5e-7):
    """translation t (with the best coverage) of `window` into `built`."""
    B = np.array(built)
    best = (0, None)
    for w in window[:40]:
        for b in built:
            if abs(b[2] - w[2]) > 1e-7:
                continue
            t = (b[0] - w[0], b[1] - w[1])
            ok = 0
            for e in window:
                m = (np.abs(B[:, 0] - e[0] - t[0]) < tol) & \
                    (np.abs(B[:, 1] - e[1] - t[1]) < tol) & \
                    (np.abs(B[:, 2] - e[2]) < 1e-7)
                if m.any():
                    ok += 1
            if ok > best[0]:
                best = (ok, t)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--built', default='poc/pocket_lift_scan2/data/built/n492.json')
    ap.add_argument('--pred', default='poc/pocket_lift_scan2/data/pred2.json')
    ap.add_argument('--lib', default='poc/pocket_lift_scan2/data/pockets_pareto2.json')
    ap.add_argument('--csv', default='poc/pell_corrugation/data/border-scan/merged.csv')
    a = ap.parse_args()

    built = [(float(r['x']), float(r['y']), float(r['angle']))
             for r in json.load(open(a.built))]
    print(f'built n=492: {len(built)} squares')
    cell = json.load(open(a.pred))['492']
    K, j, f = cell['K'], cell['j'], cell['f']
    print(f'pred2: K={K} j={j} f={f:.15f}  side={K + f:.15f}')
    lib = json.load(open(a.lib))
    used = 0
    for it in cell['comp']:
        ent = [p for p in lib if p['src'] == it['src'] and p['w'] == it['h'] and False]
        cand = [p for p in lib if p['src'] == it['src'] and p['g'] == it['g']
                and abs(p['f'] - it['f']) < 1e-9]
        p = cand[0]
        donor = read_csv_n(a.csv, it['src'])
        win = [d for d in donor if p['x0'] - 1e-9 <= d[0] <= p['x1'] + 1e-9
               and p['y0'] - 1e-9 <= d[1] <= p['y1'] + 1e-9]
        if it['transposed']:
            win = [(-d[1], d[0], d[2]) for d in win]
        ok, t = embed(win, built)
        W = p['x1'] - p['x0']
        H = p['y1'] - p['y0']
        fx, fy = W - p['w'], H - p['h']
        print(f"  pocket (w,h,g)=({it['w']},{it['h']},{it['g']}) <- n={it['src']}"
              f"{' (T)' if it['transposed'] else ''}")
        print(f"    donor window {p['x1']-p['x0']:.9f} x {p['y1']-p['y0']:.9f} "
              f"holds {len(win)} squares; declared f={p['f']:.12f} "
              f"(fx={fx:.9f}, fy={fy:.9f})")
        print(f"    rigid embed into the build: {ok}/{len(win)} squares, "
              f"shift ({t[0]:.9f}, {t[1]:.9f})" if t else '    no embed')
        used += ok
    rest = [b for b in built]
    print(f'  pocket squares verified: {used}; grid cells = {492 - used}')
    ang = Counter(round(math.degrees(b[2]) % 90, 3) for b in built)
    print('  build angles (deg mod 90), top 6:', sorted(ang.items(), key=lambda x: -x[1])[:6])


if __name__ == '__main__':
    main()
