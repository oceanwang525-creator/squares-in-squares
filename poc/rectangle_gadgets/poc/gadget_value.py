"""Value map for hypothetical rectangle gadgets (a, b, g): the largest f at which adding the gadget to
the square-gadget library improves at least one n, versus the hard lower bound f(b, b, g)."""
import json, math, sys
import numpy as np
import numba as nb
from gadget_dp import table_from_report, square_library


@nb.njit(cache=True)
def knap(ca, cb, g, B):
    best = np.zeros((B + 1, B + 1), dtype=np.int64)
    for x in range(B + 1):
        for y in range(B + 1):
            v = 0
            for k in range(ca.shape[0]):
                if ca[k] <= x and cb[k] <= y:
                    w = best[x - ca[k], y - cb[k]] + g[k]
                    if w > v:
                        v = w
            best[x, y] = v
    return best


def items(lib, f):
    ca, cb, gg = [], [], []
    for a, b, g, fi, _ in lib:
        if fi <= f + 1e-12:
            ca += [a + 1, b + 1]; cb += [b + 1, a + 1]; gg += [g, g]
    return np.array(ca, np.int64), np.array(cb, np.int64), np.array(gg, np.int64)


def improves(lib, extra, f, cK):
    a, b, g = extra
    ca, cb, gg = items(lib, f)
    ca = np.concatenate([ca, [a + 1, b + 1]]); cb = np.concatenate([cb, [b + 1, a + 1]]); gg = np.concatenate([gg, [g, g]])
    best = knap(ca, cb, gg, 32)
    gain = 0.0; hits = []
    for K in range(2, 32):
        for j in range(1, best[K + 1, K + 1] + 1):
            n = K * K + j
            if n <= 1000 and cK.get(n, 9) > f + 1e-9:
                gain += (cK[n] - f) / math.sqrt(n); hits.append(n)
    return gain, hits


if __name__ == '__main__':
    L = table_from_report(sys.argv[1])
    lib = square_library(L)
    cK = {n: L[n] - math.isqrt(n) for n in L}
    fsq = {(a, g): f for a, b, g, f, _ in lib}
    out = []
    for g in (1, 2, 3):
        for b in range(2, 21):
            if (b, g) not in fsq:
                continue
            lo = fsq[(b, g)]
            for a in range(1, b):
                if a * b + g >= (a + 1) * (b + 1):
                    continue
                gain, hits = improves(lib, (a, b, g), lo, cK)
                if gain <= 0:
                    continue
                # largest f in [lo, 1) that still improves something
                fl, fh = lo, 1.0
                for _ in range(25):
                    fm = (fl + fh) / 2
                    if improves(lib, (a, b, g), fm, cK)[0] > 0:
                        fl = fm
                    else:
                        fh = fm
                out.append((a, b, g, lo, fl, gain, hits[:8]))
    out.sort(key=lambda r: -(r[4] - r[3]))
    print('gadget (a,b,g) | lower bound f(b,b,g) | useful if f <= | score gain at lower bound | example n')
    for a, b, g, lo, fs, gain, hits in out[:40]:
        print(f'({a},{b},{g}) | {lo:.4f} | {fs:.4f} | {gain:.4f} | {hits}')
    json.dump(out, open('gadget_value.json', 'w'))
