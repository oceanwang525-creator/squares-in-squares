"""Rigid scan: best seed count per looseness for one pocket size."""
import math, sys, time
import numpy as np
import numba as nb
from seedgen import count_seed


@nb.njit(cache=True)
def scan(w, h, f0s, ths, Rs, As, phases, offs, blf):
    best = np.zeros(len(f0s), np.int64); par = np.zeros((len(f0s), 8))
    tot = 0
    for fi in range(len(f0s)):
        W = w + f0s[fi]; H = h + f0s[fi]
        for ti in range(len(ths)):
            for R in Rs:
                for a in As:
                    for ph in phases:
                        for al in range(5):
                            for ox in offs:
                                for oy in offs:
                                    n, nt = count_seed(W, H, ths[ti], W / 2 + ox, H / 2 + oy, a, R, ph, al, blf)
                                    tot += 1
                                    if n > best[fi]:
                                        best[fi] = n
                                        par[fi, 0] = ths[ti]; par[fi, 1] = R; par[fi, 2] = a; par[fi, 3] = ph
                                        par[fi, 4] = al; par[fi, 5] = ox; par[fi, 6] = oy; par[fi, 7] = nt
    return best, par, tot


if __name__ == '__main__':
    w, h = int(sys.argv[1]), int(sys.argv[2]); t0, t1, dt_ = float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
    f0s = np.array([0.62, 0.66, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95])
    ths = np.radians(np.arange(t0, t1 + 1e-9, dt_))
    Rs = np.array([int(v) for v in sys.argv[6].split(',')]); As = np.array([float(v) for v in sys.argv[7].split(',')])
    t = time.time()
    blf = float(sys.argv[8]) if len(sys.argv) > 8 else 0.0
    scan(w, h, f0s[:1], ths[:1], Rs[:1], As[:1], np.array([0.0]), np.array([0.0]), blf)
    t = time.time()
    best, par, tot = scan(w, h, f0s, ths, Rs, As, np.array([0.0, 0.25, 0.5, 0.75]), np.array([-0.5, 0.0, 0.5]), blf)
    d = time.time() - t
    print(f'{tot} seeds in {d:.1f}s = {d/tot*1e3:.3f} ms/seed; grid {w*h}')
    for fi, f0 in enumerate(f0s):
        p = par[fi]
        print(f'  f0={f0:.2f}: best N={best[fi]} (gain {best[fi]-w*h:+d}) theta={math.degrees(p[0]):.1f} R={int(p[1])} a={p[2]} phase={p[3]} align={int(p[4])} off=({p[5]},{p[6]}) tilted={int(p[7])}')
