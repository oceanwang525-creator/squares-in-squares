"""Constructive seeds: tilted band along the diagonal + greedy corner-anchored grid fill.
All placements are exact non-overlap (SAT with margin), so a seed with count >= n at side L
is already a valid packing (up to the margin)."""
import numpy as np
import numba as nb

R2 = np.sqrt(2.0)


@nb.njit(cache=True)
def sat_overlap(x1, y1, t1, x2, y2, t2, margin):
    dx = x2 - x1; dy = y2 - y1
    if dx * dx + dy * dy >= 2.0 + 4 * margin:
        return False
    for a in range(4):
        if a < 2:
            th = t1 + a * (np.pi / 2); to = t2
        else:
            th = t2 + (a - 2) * (np.pi / 2); to = t1
        ux = np.cos(th); uy = np.sin(th)
        d = abs(dx * ux + dy * uy)
        ro = 0.5 * (abs(np.cos(to - th)) + abs(np.sin(to - th)))
        if d >= 0.5 + ro + margin:
            return False
    return True


@nb.njit(cache=True)
def inside(x, y, t, L, margin):
    e = 0.5 * (abs(np.cos(t)) + abs(np.sin(t)))
    return x - e >= margin and y - e >= margin and x + e <= L - margin and y + e <= L - margin


@nb.njit(cache=True)
def try_add(P, m, x, y, t, L, margin, grid, cell, G):
    if not inside(x, y, t, L, margin):
        return m
    gx = int(x / cell); gy = int(y / cell)
    for ix in range(max(0, gx - 2), min(G, gx + 3)):
        for iy in range(max(0, gy - 2), min(G, gy + 3)):
            for k in range(grid[ix, iy, 0]):
                q = grid[ix, iy, 1 + k]
                if sat_overlap(x, y, t, P[q, 0], P[q, 1], P[q, 2], margin):
                    return m
    P[m, 0] = x; P[m, 1] = y; P[m, 2] = t
    c = grid[ix if False else gx, gy, 0]
    grid[gx, gy, 1 + c] = m; grid[gx, gy, 0] = c + 1
    return m + 1


@nb.njit(cache=True)
def build(L, w, v0, u0, theta, margin, order, ell=10**6):
    """Band of w rows at angle theta (squares aligned with band direction) through the
    diagonal; then grid lattices anchored at the 4 corners, filled greedily.
    order: 0 = band first then grids, grids filled outward from corners."""
    cap = int((L + 2) * (L + 2)) + 50
    P = np.empty((cap, 3))
    cell = 1.0
    G = int(L / cell) + 3
    grid = np.zeros((G, G, 12), np.int64)
    m = 0
    # band direction: along the main diagonal; squares rotated by theta
    dux = np.cos(np.pi / 4); duy = np.sin(np.pi / 4)
    dvx = np.cos(np.pi / 4 - np.pi / 2); dvy = np.sin(np.pi / 4 - np.pi / 2)
    umax = R2 * L + 2
    for s in range(w):
        v = v0 + s - (w - 1) / 2.0
        i = -2
        placed = 0
        if ell < 10**6:
            i = 0
        while True:
            u = u0 + i
            if u > umax or placed >= ell:
                break
            x = u * dux + v * dvx; y = u * duy + v * dvy
            m2 = try_add(P, m, x, y, theta, L, margin, grid, cell, G)
            if m2 > m:
                placed += 1
            m = m2
            i += 1
    # corner lattices, fill by increasing index sum (outward from corners)
    K = int(L) + 1
    for ssum in range(2 * K):
        for i in range(ssum + 1):
            j = ssum - i
            if i >= K or j >= K:
                continue
            m = try_add(P, m, L - 0.5 - i, 0.5 + j, 0.0, L, margin, grid, cell, G)   # SE
            m = try_add(P, m, 0.5 + i, L - 0.5 - j, 0.0, L, margin, grid, cell, G)   # NW
            m = try_add(P, m, 0.5 + i, 0.5 + j, 0.0, L, margin, grid, cell, G)       # SW
            m = try_add(P, m, L - 0.5 - i, L - 0.5 - j, 0.0, L, margin, grid, cell, G)  # NE
    return P[:m]


def capacity_scan(L, ws, v0s, u0s, thetas, margin=-1e-12, ells=(10**6,)):
    best = (0, None)
    for w in ws:
        for ell in ells:
            for v0 in v0s:
                for u0 in u0s:
                    for th in thetas:
                        P = build(L, w, v0, u0, th, margin, 0, ell)
                        if len(P) > best[0]:
                            best = (len(P), (w, ell, v0, u0, th))
    return best
