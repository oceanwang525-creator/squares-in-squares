"""Per-target verdict for the rational-slope zipper hypothesis (E3).

For every top-gap target n (K = isqrt(n), D = n - K^2) it reports

  c_inc   = L_inc - K                (fresh-main coefficient, data/report.json)
  c_front = L_front - K              (gap_vs_fortier frontier coefficient)
  floor   = K + 1/sqrt(2)            (the best a single-layer *any*-slope seam
                                      zipper can do; rational slopes are all
                                      strictly worse -- see slope_table.py)
  verdict = 'provably worse'   when c_inc <= 1/sqrt(2)  (no rational-slope
                               zipper can even tie the incumbent)
            'capacity short'   when c_inc > 1/sqrt(2): the floor side is
                               reachable in principle, but the measured capacity
                               of the portfolio's own 22 zipper rows is only
                               D_zip(K) ~ floor(0.433*K) extra squares, so the
                               construction holds K^2+D_zip(K) < n squares.

usage: targets.py [--out data/targets.json] [--md docs/targets.md]
"""
import argparse
import json
import math

import mpmath as mp

# the portfolio's own rows that sit *exactly* at K + 1/sqrt(2)
ZIP_ROWS = [(5, 1), (10, 1), (27, 2), (38, 2), (52, 3), (67, 3), (104, 4), (125, 4),
            (149, 5), (174, 5), (201, 5), (231, 6), (262, 6), (296, 7), (331, 7),
            (369, 8), (408, 8), (449, 8), (493, 9), (586, 10), (635, 10), (913, 13)]


def dzip(K):
    """measured single-seam-layer capacity: D_zip(K) = floor(0.433*K)."""
    return int(0.433 * K)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--report', default='poc/rational_zipper/data/report.json')
    ap.add_argument('--gap', default='poc/koboshchan_coordinator/data/gap_vs_frontier.csv')
    ap.add_argument('--out', default='poc/rational_zipper/data/targets.json')
    ap.add_argument('--md', default='')
    ap.add_argument('--topk', type=int, default=24)
    a = ap.parse_args()

    rep = {r['n']: float(r['L']) for r in json.load(open(a.report))['results']}
    gaps = {}
    with open(a.gap) as f:
        f.readline()
        for line in f:
            p = line.split(',')
            gaps[int(p[0])] = (float(p[4]), float(p[8]))
    order = sorted(gaps, key=lambda n: -gaps[n][1])[:a.topk]

    floor = 1.0 / math.sqrt(2.0)
    out = {}
    lines = []
    hdr = (f"| n | K | D | L_inc | c_inc | c_front | c_inc-1/sqrt2 | K+1/sqrt2 | "
           f"D_zip(K) | squares at the floor | verdict |")
    lines.append(hdr)
    lines.append('|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|')
    for n in order:
        K = math.isqrt(n)
        D = n - K * K
        L = rep[n]
        c = L - K
        cf = gaps[n][0] - K
        zf = K + floor
        dz = dzip(K)
        hold = K * K + dz
        if c <= floor:
            verdict = f"provably worse (+{(floor-c)*1000:.1f}e-3 above the floor)"
        else:
            verdict = f"capacity short ({D-dz} squares)"
        out[n] = dict(n=n, K=K, D=D, L_inc=L, c_inc=c, c_front=cf,
                      zipper_floor=zf, floor_minus_inc=zf - L, d_zip=dz,
                      squares_at_floor=hold, short=D - dz, verdict=verdict,
                      gap_score=gaps[n][1])
        lines.append(f"| {n} | {K} | {D} | {L:.9f} | {c:.6f} | {cf:.6f} | "
                     f"{c-floor:+.6f} | {zf:.6f} | {dz} | {hold} | {verdict} |")
        print(f"n={n:4d} K={K:2d} D={D:2d} c_inc={c:.6f} c_front={cf:.6f} "
              f"floor={zf:.6f} ({zf-L:+.6f}) {verdict}")
    json.dump(out, open(a.out, 'w'), indent=1)
    print('wrote', a.out)
    if a.md:
        open(a.md, 'w').write('\n'.join(lines) + '\n')
        print('wrote', a.md)


if __name__ == '__main__':
    main()
