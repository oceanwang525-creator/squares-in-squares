"""Motif transplant: cut tilted clusters (+ nearby off-lattice squares) out of good packings and
re-insert them into a fresh K + c container whose remainder is filled greedily by four
corner-anchored upright lattices (exact SAT, touching allowed)."""
import csv, math, sys
import numpy as np
import numba as nb
from bandgen import try_add, sat_overlap, inside


@nb.njit(cache=True)
def fill(L, pre, margin):
    """Insert pre-placed squares (assumed mutually valid) then greedy corner lattices."""
    cap = int((L + 2) * (L + 2)) + 50
    P = np.empty((cap, 3)); G = int(L) + 3
    grid = np.zeros((G, G, 16), np.int64)
    m = 0
    for q in range(pre.shape[0]):
        x, y, t = pre[q, 0], pre[q, 1], pre[q, 2]
        if not inside(x, y, t, L, margin):
            return P[:0]
        gx = int(x); gy = int(y)
        P[m, 0] = x; P[m, 1] = y; P[m, 2] = t
        c = grid[gx, gy, 0]; grid[gx, gy, 1 + c] = m; grid[gx, gy, 0] = c + 1
        m += 1
    K = int(L) + 1
    for ssum in range(2 * K):
        for i in range(ssum + 1):
            j = ssum - i
            if i >= K or j >= K:
                continue
            m = try_add(P, m, L - 0.5 - i, 0.5 + j, 0.0, L, margin, grid, 1.0, G)
            m = try_add(P, m, 0.5 + i, L - 0.5 - j, 0.0, L, margin, grid, 1.0, G)
            m = try_add(P, m, 0.5 + i, 0.5 + j, 0.0, L, margin, grid, 1.0, G)
            m = try_add(P, m, L - 0.5 - i, L - 0.5 - j, 0.0, L, margin, grid, 1.0, G)
    return P[:m]


def load(path, ns):
    by = {}
    for r in csv.DictReader(open(path)):
        n = int(r['n'])
        if n in ns:
            by.setdefault(n, []).append((float(r['x']), float(r['y']), float(r['angle'])))
    out = {}
    for n, rows in by.items():
        a = np.array(rows); e = 0.5 * (np.abs(np.cos(a[:, 2])) + np.abs(np.sin(a[:, 2])))
        a[:, 0] -= (a[:, 0] - e).min(); a[:, 1] -= (a[:, 1] - e).min()
        L = max((a[:, 0] + e).max(), (a[:, 1] + e).max())
        out[n] = (a, L)
    return out


def off_lattice(a, L, tol=1e-6):
    """True for squares that are tilted or not aligned to any of the four corner lattices."""
    t = np.mod(a[:, 2], np.pi / 2)
    tilted = np.minimum(t, np.pi / 2 - t) > tol
    fx = np.mod(a[:, 0] - 0.5, 1.0); fxr = np.mod(L - 0.5 - a[:, 0], 1.0)
    fy = np.mod(a[:, 1] - 0.5, 1.0); fyr = np.mod(L - 0.5 - a[:, 1], 1.0)
    ok = lambda f: (f < tol) | (f > 1 - tol)
    aligned = (ok(fx) | ok(fxr)) & (ok(fy) | ok(fyr))
    return tilted | ~aligned


def clusters(a, mask, link=1.6):
    idx = np.nonzero(mask)[0]; lab = -np.ones(len(a), int); c = 0
    for s in idx:
        if lab[s] >= 0: continue
        stack = [s]; lab[s] = c
        while stack:
            i = stack.pop()
            d = np.hypot(a[idx, 0] - a[i, 0], a[idx, 1] - a[i, 1])
            for j in idx[(d < link) & (lab[idx] < 0)]:
                lab[j] = c; stack.append(j)
        c += 1
    return [np.nonzero(lab == q)[0] for q in range(c)]
