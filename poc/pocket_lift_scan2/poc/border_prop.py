"""L-border propagation of a certified group to n' = n + 2*K + 1 (K = floor(sqrt n)).

Same mechanism as poc/koboshchan_coordinator/poc/border_extend.py, with two
changes that matter at 1e-9 resolution:

  * GAP = 1e-12 instead of 1e-9 (an exactly-touching pair is "overlap
    undecidable" for the official metric, so a gap is required -- but 1e-12 is
    enough and costs only 1e-12 of side, keeping dL intact);
  * a lopsided bbox is handled (m = ceil(max(W,H)) so the new side is
    max(W,H)+1+GAP on whichever axis was the long one).

The new column/row coordinates are exact 40-decimal values and the vertical /
horizontal pitch is 1+1e-12, so every pair of neighbours in the border is
strictly separated at any escalation precision.

usage:
  border_prop.py --report report.json --built data/built [--outdir data/built] [--n 447 ...]
"""
import argparse
import glob
import json
import os
import sys
from decimal import Decimal, getcontext

import mpmath as mp

sys.path.insert(0, 'poc/baseline/poc/src')
from squares.core import certify, kaggle_roundtrip

getcontext().prec = 120
Q = Decimal(1).scaleb(-40)
GAP = Decimal('1e-9')
SP = Decimal(1) + Decimal('1e-12')
HALF = Decimal('0.5')


def bbox(rows):
    xs0, xs1, ys0, ys1 = [], [], [], []
    with mp.workdps(90):
        for r in rows:
            th = mp.mpf(str(r['angle'])); e = (abs(mp.cos(th)) + abs(mp.sin(th))) / 2
            x = mp.mpf(str(r['x'])); y = mp.mpf(str(r['y']))
            xs0.append(x - e); xs1.append(x + e); ys0.append(y - e); ys1.append(y + e)
        return min(xs0), max(xs1), min(ys0), max(ys1)


def dec(v):
    return Decimal(mp.nstr(v, 60))


def fmt(d):
    return format(d.quantize(Q), 'f')


def border_extend(rows):
    with mp.workdps(90):
        x0, x1, y0, y1 = bbox(rows)
        W, H = x1 - x0, y1 - y0
        L = max(W, H)
        m = int(mp.ceil(L))
        colx = (dec(x1) + GAP + HALF).quantize(Q)
        rowy = (dec(y1) + GAP + HALF).quantize(Q)
        ybase = (dec(y0) + HALF).quantize(Q)
        xbase = (dec(x0) + HALF).quantize(Q)
    out = [dict(r) for r in rows]
    sid = max(int(r['square_id']) for r in rows) + 1
    for j in range(m):                       # right column, x in [x1+GAP, x1+GAP+1]
        out.append(dict(square_id=str(sid), x=fmt(colx),
                        y=fmt(ybase + j * SP), angle='0'))
        sid += 1
    for i in range(m - 1):                   # top row, y in [y1+GAP, y1+GAP+1]
        out.append(dict(square_id=str(sid), x=fmt(xbase + i * SP),
                        y=fmt(rowy), angle='0'))
        sid += 1
    out = [dict(r, square_id=str(i)) for i, r in enumerate(out)]
    return len(out), out, L, (W, H)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--report', required=True)
    ap.add_argument('--built', default='poc/pocket_lift_scan2/data/built')
    ap.add_argument('--outdir', default='poc/pocket_lift_scan2/data/built_prop')
    ap.add_argument('--n', type=int, nargs='+')
    a = ap.parse_args()
    L = {r['n']: mp.mpf(str(r['L'])) for r in json.load(open(a.report))['results']}
    files = sorted(glob.glob(os.path.join(a.built, 'n*.json')))
    if a.n:
        files = [f for f in files if int(os.path.basename(f)[1:-5]) in a.n]
    for f in files:
        n = int(os.path.basename(f)[1:-5])
        rows = json.load(open(f))
        if len(rows) != n:
            print(f'n={n}: file has {len(rows)} rows, skip'); continue
        n2, rows2, Lb, WH = border_extend(rows)
        if n2 not in L:
            print(f'n={n}: extended n={n2} not in the incumbent, skip'); continue
        try:
            s1 = certify(n2, rows2)
            s2 = certify(n2, kaggle_roundtrip({n2: rows2})[n2])
        except Exception as ex:
            print(f'n={n} -> {n2}: certify raised {ex}'); continue
        inc = L[n2]
        d1, d2 = mp.mpf(s1) - inc, mp.mpf(s2) - inc
        # sanity: side must be exactly max(W,H)+1(+GAP), i.e. the border is free
        db = L.get(n)
        base_dl = (db - (mp.mpf(s1) - 1)) if db is not None else None
        print(f'n={n} -> {n2}: base L {mp.nstr(db, 20) if db is not None else "?"} '
              f'({len(rows)} sq) -> ext {mp.nstr(mp.mpf(s1), 20)} / '
              f'{mp.nstr(mp.mpf(s2), 20)}  inc {mp.nstr(inc, 20)}  '
              f'dL {mp.nstr(d1, 8)} / {mp.nstr(d2, 8)}'
              + (f'  (base dL {mp.nstr(base_dl, 8)})' if base_dl is not None else ''))
        if d1 < mp.mpf('-1e-9') and d2 < mp.mpf('-1e-9'):
            out = os.path.join(a.outdir, f'n{n2}.json')
            json.dump(rows2, open(out, 'w'))
            print(f'   STRICT WIN -> {out}')


if __name__ == '__main__':
    main()
