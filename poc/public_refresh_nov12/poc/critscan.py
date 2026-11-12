"""Tilted m x p rectangle at critical positions (faces touching corner-lattice corners) +
greedy corner-lattice fill. Reports the best capacity per L.
usage: critscan.py n L1 L2 ... [--theta deg]
"""
import math, sys, time
import numpy as np
from multiprocessing import Pool
from bandgen import build

R2 = math.sqrt(2)


def crit_positions(L, m, p, span=3.0):
    uc = R2 * L / 2  # diagonal centre
    us = set()
    for q in range(0, int(2 * L * R2) + 4):
        u0 = q / R2 + 0.5                      # SW face on SW-lattice corner line
        if abs(u0 + (p - 1) / 2 - uc) < span: us.add(round(u0, 12))
        u0 = (2 * L - q) / R2 - p + 0.5        # NE face on NE-lattice corner line
        if abs(u0 + (p - 1) / 2 - uc) < span: us.add(round(u0, 12))
    vs = set()
    for q in range(-int(L) - 4, int(L) + 4):
        vhi = (L - q) / R2                    # SE face
        if abs(vhi - m / 2) < span: vs.add(round(vhi - m / 2, 12))
        vlo = (q - L) / R2                    # NW face
        if abs(vlo + m / 2) < span: vs.add(round(vlo + m / 2, 12))
    return sorted(us), sorted(vs)


def job(args):
    L, m, p, th = args
    us, vs = crit_positions(L, m, p)
    best = (0, None)
    for u0 in us:
        for vc in vs:
            P = build(L, m, vc, u0, th, -1e-12, 0, p)
            if len(P) > best[0]:
                best = (len(P), (m, p, u0, vc))
    return L, best


if __name__ == '__main__':
    n = int(sys.argv[1]); Ls = [float(x) for x in sys.argv[2:]]
    th = math.pi / 4
    k = math.isqrt(n)
    jobs = [(L, m, p, th) for L in Ls for m in range(2, 2 * k) for p in range(m, 2 * k)
            if (m + p) / R2 <= L + 0.01 and m * p >= 9]
    t = time.time()
    res = {}
    with Pool(8) as pool:
        for L, b in pool.imap_unordered(job, jobs, chunksize=4):
            if b[0] > res.get(L, (0,))[0]:
                res[L] = b
    for L in Ls:
        c, cfg = res[L]
        print(f'L={L:.6f} best capacity={c} need={n} {"<-- FITS" if c >= n else ""} cfg={cfg}', flush=True)
    print('time', round(time.time() - t, 1))
