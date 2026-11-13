"""Staircase gadget composition DP.
Gadget (a, b, g, f): an (a+f) x (b+f) box holding a*b + g squares. m gadgets along the anti-diagonal
staircase of a K + f box need sum(a_i + 1) <= K + 1 and sum(b_i + 1) <= K + 1 and give n = K^2 + sum g_i.
"""
import json, math, sys
import numpy as np


def table_from_report(path):
    L = {r['n']: float(r['L']) for r in json.load(open(path))['results']}
    return L


def square_library(L):
    lib = []
    for n0, l in L.items():
        k0 = math.isqrt(n0); j0 = n0 - k0 * k0
        if j0 >= 1 and l - k0 < 1:
            lib.append((k0, k0, j0, l - k0, f'sq{n0}'))
    return lib


def max_gain(lib, f, B=32):
    """best[x][y] = max total gain with sum(a+1) <= x, sum(b+1) <= y using gadgets with f_i <= f."""
    items = []
    for a, b, g, fi, _ in lib:
        if fi <= f + 1e-12:
            items.append((a + 1, b + 1, g))
            if a != b:
                items.append((b + 1, a + 1, g))
    best = np.zeros((B + 1, B + 1), dtype=np.int64)
    for x in range(B + 1):
        for y in range(B + 1):
            v = 0
            for ca, cb, g in items:
                if ca <= x and cb <= y:
                    w = best[x - ca, y - cb] + g
                    if w > v:
                        v = w
            best[x, y] = v
    return best


def predict(lib, Kmax=31):
    fs = sorted(set(round(x[3], 12) for x in lib))
    pred = {}
    prev = None
    for f in fs:
        best = max_gain(lib, f)
        for K in range(2, Kmax + 1):
            gmax = best[K + 1, K + 1]
            for j in range(1, gmax + 1):
                n = K * K + j
                if n <= 1000 and (n not in pred):
                    pred[n] = (K + f, f)
    return pred


if __name__ == '__main__':
    L = table_from_report(sys.argv[1])
    lib = square_library(L)
    pred = predict(lib)
    worse = better = same = 0; gain = 0.0
    rows = []
    for n, (Lp, f) in sorted(pred.items()):
        d = L[n] - Lp
        if d > 1e-9:
            better += 1; gain += d / math.sqrt(n); rows.append((n, L[n], Lp))
        elif d < -1e-9:
            worse += 1
        else:
            same += 1
    print(f'squares-only library: {len(lib)} gadgets; predicted n: {len(pred)}; same {same}, table better {worse}, DP better {better} (gain {gain:.6f})')
    for r in rows[:20]:
        print('  DP better:', r)
