"""Exact rational-slope bookkeeping for E3.

Two separate objects are tabulated:

1. **Single-layer seam zipper.**  A seam of slope q:p filled with one layer of
   unit squares whose edges are parallel to the seam needs a perpendicular
   separation >= 1 (a unit square's width in any direction is >= 1).  With a
   separation vector (u,v) the perpendicular component is (u*p + v*q)/h,
   h = sqrt(p^2+q^2), and the axis cost is max(u,v).  Minimising max(u,v)
   subject to (u*p+v*q)/h >= 1 gives

        c_zip(p,q) = h/(p+q)   >= 1/sqrt(2),  equality iff p == q.

   c_zip is rational exactly for Pythagorean (p,q,h); the exact table below is
   computed in Fraction/Decimal arithmetic.  No rational slope can beat the
   45-degree zipper, and only p=q attains 1/sqrt(2).

2. **Observed rational coefficient law of the tracked large-n rows.**
   The portfolio's own big rows carry coefficients that are exactly
   p/(p+q) for a slope q:p that is visibly present in the row (e.g. n=585 and
   n=853 sit at 12/17 and contain 96 / 105 squares at atan(5/12) = 22.62 deg).
   That is a *pocket-composition* constant, not a seam-band constant, and it is
   the true content of the "rational slope" hypothesis.

usage:  slope_table.py [--out data/slopes.json]
"""
import argparse
import json
import math
from fractions import Fraction

import mpmath as mp

TARGETS_FALLBACK = [791, 908, 969, 803, 328, 492, 861, 365, 367, 975, 747, 804,
                    974, 848, 456, 852, 366, 228, 795, 982]


def pythagorean(hmax=60):
    out = []
    for p in range(1, hmax):
        for q in range(1, p):
            h = math.isqrt(p * p + q * q)
            if h * h != p * p + q * q or math.gcd(p, q) != 1:
                continue
            out.append((p, q, h))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--report', default='poc/rational_zipper/data/report.json')
    ap.add_argument('--gap', default='poc/koboshchan_coordinator/data/gap_vs_frontier.csv')
    ap.add_argument('--out', default='poc/rational_zipper/data/slopes.json')
    ap.add_argument('--qmax', type=int, default=40)
    a = ap.parse_args()

    mp.mp.dps = 40
    rep = {r['n']: mp.mpf(r['L']) for r in json.load(open(a.report))['results']}

    # ---------------- 1. exact single-layer zipper cost --------------------
    print('=== exact single-layer seam-zipper cost  c_zip = h/(p+q) ===')
    print('best rational slopes (smallest cost), q <= %d:' % a.qmax)
    best = []
    for q in range(1, a.qmax + 1):
        for p in range(q, a.qmax + 1):
            if math.gcd(p, q) != 1:
                continue
            h2 = p * p + q * q
            h = math.isqrt(h2)
            exact = (h * h == h2)
            val = mp.mpf(h2) ** mp.mpf('0.5') / (p + q) if not exact else Fraction(h, p + q)
            best.append((float(val), p, q, h if exact else None,
                         str(Fraction(h, p + q)) if exact else ''))
    best.sort()
    for v, p, q, h, s in best[:12]:
        kind = f'Pythagorean h={h}  exact {s}' if h else 'irrational'
        print(f'  slope q:p = {q}:{p}   cost {v:.12f}   {kind}')
    print(f'  1/sqrt(2) = {float(mp.mpf(1)/mp.sqrt(2)):.12f}  (45deg, (1,1), irrational)')
    py = sorted(((Fraction(h, p + q), p, q, h) for p, q, h in pythagorean(60)),
                key=lambda t: t[0])
    print('  best *Pythagorean* (rational-side) slopes:')
    for fr, p, q, h in py[:6]:
        print(f'    ({p},{q},{h})  slope q:p = {q}:{p}  cost {fr} = {float(fr):.12f}')

    # ---------------- 2. family constants p/(p+q) --------------------------
    fams = sorted({Fraction(p, p + q) for q in range(1, a.qmax + 1)
                   for p in range(q, a.qmax + 1) if math.gcd(p, q) == 1})
    print(f'\n=== observed coefficient law  c = p/(p+q)  (slope q:p) ===')
    print('  smallest constants:', ', '.join(str(f) for f in fams[:8]))

    # ---------------- 3. per target ---------------------------------------
    rows = []
    with open(a.gap) as f:
        f.readline()
        for line in f:
            pp = line.split(',')
            rows.append((float(pp[8]), int(pp[0])))
    rows.sort(reverse=True)
    ns = [n for _, n in rows[:24]]

    floor = mp.mpf(1) / mp.sqrt(2)
    out = {}
    print(f"\n{'n':>4} {'K':>3} {'D':>3} {'L_inc':>15} {'c_inc':>10} {'c_front':>9} "
          f"{'zip_floor':>15} {'zip-L_inc':>11} {'next fam <c_inc':>15} {'prize':>9}")
    for n in ns:
        K = math.isqrt(n)
        L = rep[n]
        c = L - K
        fr = [r for r in rows if r[1] == n][0][0]
        with open(a.gap) as f:
            f.readline()
            frontier = None
            for line in f:
                pp = line.split(',')
                if int(pp[0]) == n:
                    frontier = mp.mpf(pp[4])
        cf = frontier - K
        zf = K + floor
        # smallest family constant strictly below the incumbent coefficient
        cand = [f for f in fams if f < c]
        fam = min(cand) if cand else None
        out[n] = dict(n=n, K=K, D=n - K * K, L_inc=str(L), c_inc=str(c),
                      L_front=str(frontier), c_front=str(cf), zipper_floor=str(zf),
                      zipper_minus_inc=str(zf - L),
                      family_below=(str(fam) if fam else None),
                      family_side=(str(K + fam) if fam else None),
                      family_prize=(str(c - fam) if fam else None))
        print(f"{n:4d} {K:3d} {n-K*K:3d} {mp.nstr(L,15):>15} {mp.nstr(c,8):>10} "
              f"{mp.nstr(cf,7):>9} {mp.nstr(zf,15):>15} {mp.nstr(zf-L,5):>11} "
              f"{(str(fam)+' (='+mp.nstr(mp.mpf(fam.numerator)/fam.denominator,7)+')') if fam else '-':>15} "
              f"{mp.nstr(c-fam,5) if fam else '-':>9}")
    json.dump(out, open(a.out, 'w'), indent=1)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
