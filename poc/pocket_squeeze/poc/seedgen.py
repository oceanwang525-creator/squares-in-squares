"""Rigid pocket seeds: a block of tilted rows (each row filled maximally inside a rotated rectangle
clipped to the pocket) plus upright fill (four corner lattices, then a fine bottom-left-style void fill).

make_seed(W, H, th, cx, cy, a, R, phase, align, blf) -> (P[m, 3], number of tilted squares)
  th     row direction angle in (0, pi/2)
  cx, cy block centre in pocket coordinates
  a      half-length of the block along the rows; R = number of rows
  phase  shift of the row stack across the rows, in [0, 1)
  align  0 left, 1 right, 2 centred, 3 alternate L/R, 4 alternate R/L
"""
import math
import numpy as np
import numba as nb

CAPC = 24


@nb.njit(cache=True)
def _overlap(x1, y1, t1, x2, y2, t2, margin):
    dx = x2 - x1; dy = y2 - y1
    if dx * dx + dy * dy >= 2.0 + 4 * margin:
        return False
    for a in range(4):
        if a < 2:
            th = t1 + a * (math.pi / 2); to = t2
        else:
            th = t2 + (a - 2) * (math.pi / 2); to = t1
        d = abs(dx * math.cos(th) + dy * math.sin(th))
        ro = 0.5 * (abs(math.cos(to - th)) + abs(math.sin(to - th)))
        if d >= 0.5 + ro + margin:
            return False
    return True


@nb.njit(cache=True)
def _add(P, m, x, y, t, W, H, grid, Gx, Gy, margin):
    e = 0.5 * (abs(math.cos(t)) + abs(math.sin(t)))
    if x - e < -1e-9 or y - e < -1e-9 or x + e > W + 1e-9 or y + e > H + 1e-9:
        return m
    gx = int(x); gy = int(y)
    if gx >= Gx or gy >= Gy:
        return m
    for ix in range(max(0, gx - 2), min(Gx, gx + 3)):
        for iy in range(max(0, gy - 2), min(Gy, gy + 3)):
            for k in range(grid[ix, iy, 0]):
                q = grid[ix, iy, 1 + k]
                if _overlap(x, y, t, P[q, 0], P[q, 1], P[q, 2], margin):
                    return m
    c = grid[gx, gy, 0]
    if c >= CAPC - 1 or m >= P.shape[0]:
        return m
    P[m, 0] = x; P[m, 1] = y; P[m, 2] = t
    grid[gx, gy, 1 + c] = m; grid[gx, gy, 0] = c + 1
    return m + 1


@nb.njit(cache=True)
def make_seed(W, H, th, cx, cy, a, R, phase, align, blf):
    cap = int((W + 1) * (H + 1)) + 32
    P = np.empty((cap, 3))
    Gx = int(W) + 2; Gy = int(H) + 2
    grid = np.zeros((Gx, Gy, CAPC), np.int64)
    margin = -1e-9
    c = math.cos(th); s = math.sin(th)
    uc = cx * c + cy * s; vc = -cx * s + cy * c
    v0 = vc - R / 2.0 + phase
    m = 0
    for i in range(R):
        vi = v0 + i
        ulo = max((vi + 1) * s / c, -vi * c / s, uc - a)
        uhi = min((W + vi * s) / c - 1.0, (H - (vi + 1) * c) / s - 1.0, uc + a - 1.0)
        if uhi < ulo - 1e-12:
            continue
        n = int(math.floor(uhi - ulo + 1e-9)) + 1
        if align == 0:
            start = ulo
        elif align == 1:
            start = uhi - (n - 1)
        elif align == 2:
            start = ulo + (uhi - ulo - (n - 1)) / 2
        elif align == 3:
            start = ulo if i % 2 == 0 else uhi - (n - 1)
        else:
            start = uhi - (n - 1) if i % 2 == 0 else ulo
        for k in range(n):
            uu = start + k + 0.5; vv = vi + 0.5
            m = _add(P, m, uu * c - vv * s, uu * s + vv * c, th, W, H, grid, Gx, Gy, margin)
    nt = m
    KX = int(W) + 1; KY = int(H) + 1
    for ssum in range(KX + KY):
        for i in range(ssum + 1):
            j = ssum - i
            if i >= KX or j >= KY:
                continue
            m = _add(P, m, 0.5 + i, 0.5 + j, 0.0, W, H, grid, Gx, Gy, margin)
            m = _add(P, m, W - 0.5 - i, H - 0.5 - j, 0.0, W, H, grid, Gx, Gy, margin)
            m = _add(P, m, W - 0.5 - i, 0.5 + j, 0.0, W, H, grid, Gx, Gy, margin)
            m = _add(P, m, 0.5 + i, H - 0.5 - j, 0.0, W, H, grid, Gx, Gy, margin)
    if blf > 0:
        nx = int((W - 1.0) / blf) + 1; ny = int((H - 1.0) / blf) + 1
        for ssum in range(nx + ny):
            for i in range(ssum + 1):
                j = ssum - i
                if i >= nx or j >= ny:
                    continue
                x = 0.5 + i * blf; y = 0.5 + j * blf
                m = _add(P, m, x, y, 0.0, W, H, grid, Gx, Gy, margin)
                m = _add(P, m, W - x, H - y, 0.0, W, H, grid, Gx, Gy, margin)
                m = _add(P, m, W - x, y, 0.0, W, H, grid, Gx, Gy, margin)
                m = _add(P, m, x, H - y, 0.0, W, H, grid, Gx, Gy, margin)
    return P[:m].copy(), nt


@nb.njit(cache=True)
def count_seed(W, H, th, cx, cy, a, R, phase, align, blf):
    P, nt = make_seed(W, H, th, cx, cy, a, R, phase, align, blf)
    return P.shape[0], nt
