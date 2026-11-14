"""Compose pockets along the anti-diagonal seam of a (K+f) box (numba-free copy of
poc/pocket_squeeze/poc/compose.py, same layout rule).

Pocket i: lower-left corner at integer (X_i, Y_i), squares inside [0, w_i + f] x [0, h_i + f].
Layout: X_{i+1} >= X_i + w_i + ex_i, Y_i >= Y_{i+1} + h_{i+1} + ey_i, ex_i + ey_i = 1
(one extra column OR one extra row between consecutive pockets), all inside the K x K index
range. Grid A is anchored at (0,0), grid B at (L,L). Total = K^2 + sum of pocket gains.
"""
import json, math
import numpy as np


def transpose(sq):
    a = np.asarray(sq, float)
    return np.column_stack([a[:, 1], a[:, 0], math.pi / 2 - a[:, 2]])


def layout(K, parts):
    m = len(parts)
    sw = sum(p[0] for p in parts); sh = sum(p[1] for p in parts)
    if sw > K or sh > K or sw + sh + m - 1 > 2 * K:
        return None
    nex = min(m - 1, K - sw)
    ney = (m - 1) - nex
    if ney > K - sh:
        return None
    ex = [1] * nex + [0] * ney
    X = [0] * m; Y = [0] * m
    for i in range(1, m):
        X[i] = X[i - 1] + parts[i - 1][0] + ex[i - 1]
    for i in range(m - 2, -1, -1):
        Y[i] = Y[i + 1] + parts[i + 1][1] + (1 - ex[i])
    assert X[-1] + parts[-1][0] <= K and Y[0] + parts[0][1] <= K
    return X, Y


def build(K, f, pockets):
    """pockets: list of dict(w, h, g, sq) in seam order. Returns (N x 3 array, expected count)."""
    L = K + f
    lay = layout(K, [(p['w'], p['h']) for p in pockets])
    if lay is None:
        return None, None
    X, Y = lay
    out = []
    for p, x0, y0 in zip(pockets, X, Y):
        a = np.asarray(p['sq'], float).copy()
        a[:, 0] += x0; a[:, 1] += y0
        out.append(a)
    cells = []
    for a in range(K):                        # grid A: left of / below every pocket corner
        for b in range(K):
            if all((a + 1 <= x0) or (b + 1 <= y0) for x0, y0 in zip(X, Y)):
                cells.append((a + 0.5, b + 0.5, 0.0))
    for a in range(K):                        # grid B: anchored at the top-right corner
        for b in range(K):
            x0 = L - a - 1; y0 = L - b - 1
            if not any((x0 >= px - 1e-12) and (y0 >= py - 1e-12) for px, py in zip(X, Y)):
                continue
            ok = True
            for p, px, py in zip(pockets, X, Y):
                if (x0 < px + p['w'] + f - 1e-12 and x0 + 1 > px + 1e-12
                        and y0 < py + p['h'] + f - 1e-12 and y0 + 1 > py + 1e-12):
                    ok = False; break
            if ok:
                cells.append((x0 + 0.5, y0 + 0.5, 0.0))
    out.append(np.array(cells, float).reshape(-1, 3))
    P = np.vstack(out)
    return P, K * K + sum(p['g'] for p in pockets)


def realise(lib, K, f, parts, n):
    """Build the n-square packing for a predicted cell; returns N x 3 array or None."""
    pockets = []
    for i, tr in parts:
        g = lib[i]
        sq = transpose(g['sq']) if tr else np.asarray(g['sq'], float)
        w, h = (g['h'], g['w']) if tr else (g['w'], g['h'])
        pockets.append(dict(w=w, h=h, g=g['g'], sq=sq))
    P, cnt = build(K, f, pockets)
    if P is None or len(P) != cnt or cnt < n:
        return None
    return P[:n]


def load_lib(path):
    lib = json.load(open(path))
    for g in lib:
        if 'f' not in g:
            g['f'] = max(g['fx'], g['fy'])
    return lib
