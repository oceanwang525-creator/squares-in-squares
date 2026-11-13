#!/usr/bin/env python3
"""Fault-slab peel scan (interior generalisation of the crop / border peel).

A horizontal *fault line* of a packing is a height y that no square crosses.
Removing every square between two fault lines y1 < y2 and sliding the upper
part down by y2 - y1 gives a valid packing; the same holds for vertical lines.
For every donor n' this enumerates all (y-slab, x-slab) pairs, including empty
slabs on one axis (but not on both), and reports the box side and square count
of the peeled packing.  A
result with count >= m and side < L(m) is a strict win for target m (delete the
surplus squares).

Slabs are limited to MAXW = 2.6 units (up to two rows/columns plus a seam), which
keeps the pair enumeration tractable on grid-like donors.

usage: fault_peel.py ext.bin lo hi > report.csv
"""
import sys
import numpy as np

MAXW = 2.6


def faults(lo, hi, tol=1e-9):
    """Sorted fault coordinates: boundaries that no (lo, hi) interval crosses."""
    order = np.argsort(lo)
    lo, hi = lo[order], hi[order]
    out = [lo[0]]
    reach = hi[0]
    for a, b in zip(lo[1:], hi[1:]):
        if a >= reach - tol:
            out.append(reach)
            if a > reach + tol:
                out.append(a)
        reach = max(reach, b)
    out.append(reach)
    return np.array(out)


def main():
    a = np.fromfile(sys.argv[1]).reshape(-1, 4)
    lo, hi = int(sys.argv[2]), int(sys.argv[3])
    P, o = {}, 0
    for n in range(1, 1001):
        P[n] = a[o:o + n]; o += n
    L = {n: max(P[n][:, 1].max(), P[n][:, 3].max()) for n in P}
    best = {}
    nfault = 0
    for d in range(lo + 1, 1001):
        b = P[d]
        fx, fy = faults(b[:, 0], b[:, 1]), faults(b[:, 2], b[:, 3])
        if len(fx) > 2 or len(fy) > 2:
            nfault += 1
        xs = [(0.0, 0.0)] + [(fx[i], fx[j]) for i in range(len(fx)) for j in range(i + 1, len(fx)) if fx[j] - fx[i] <= MAXW]
        ys = [(0.0, 0.0)] + [(fy[i], fy[j]) for i in range(len(fy)) for j in range(i + 1, len(fy)) if fy[j] - fy[i] <= MAXW]
        W, H = b[:, 1].max(), b[:, 3].max()
        for x1, x2 in xs:
            inx = (b[:, 0] >= x1 - 1e-9) & (b[:, 1] <= x2 + 1e-9) if x2 > x1 else np.zeros(d, bool)
            for y1, y2 in ys:
                if x2 <= x1 and y2 <= y1:
                    continue                   # no slab removed: that is the donor itself
                iny = (b[:, 2] >= y1 - 1e-9) & (b[:, 3] <= y2 + 1e-9) if y2 > y1 else np.zeros(d, bool)
                cnt = d - int((inx | iny).sum())
                side = max(W - (x2 - x1), H - (y2 - y1))
                if cnt < lo:
                    continue
                key = min(cnt, hi)
                if side < best.get(key, (1e9,))[0]:
                    best[key] = (side, d, x1, x2, y1, y2, cnt)
    # deletion closure: a packing of cnt squares serves every m <= cnt
    run = (1e9,)
    rows = []
    for m in range(hi, lo - 1, -1):
        if m in best and best[m][0] < run[0]:
            run = best[m]
        rows.append((m, L[m], run))
    wins = 0
    print("m,ours,peel_side,delta,donor,x1,x2,y1,y2,count")
    for m, ours, r in reversed(rows):
        if len(r) == 1:
            continue
        delta = r[0] - ours
        wins += delta < -1e-9
        print(f"{m},{ours:.12f},{r[0]:.12f},{delta:.3e},{r[1]},{r[2]:.6f},{r[3]:.6f},{r[4]:.6f},{r[5]:.6f},{r[6]}")
    print(f"# donors with an interior fault line: {nfault}; strict wins (delta < -1e-9): {wins}", file=sys.stderr)


if __name__ == "__main__":
    main()
