"""Tilted r x s block anywhere in the box + greedy fill by four corner-anchored grids."""
import math, sys, time
import numpy as np
import numba as nb
from multiprocessing import Pool
from bandgen import try_add


@nb.njit(cache=True)
def build_block(L, cx, cy, th, r, s, margin):
    cap = int((L + 2) * (L + 2)) + 50
    P = np.empty((cap, 3)); G = int(L) + 3
    grid = np.zeros((G, G, 12), np.int64)
    m = 0
    c, sn = np.cos(th), np.sin(th)
    for i in range(s):
        for j in range(r):
            a = i - (s - 1) / 2.0; b = j - (r - 1) / 2.0
            m2 = try_add(P, m, cx + a * c - b * sn, cy + a * sn + b * c, th, L, margin, grid, 1.0, G)
            if m2 == m:
                return P[:0]          # block must be complete and inside
            m = m2
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


def job(args):
    r, s, th, c, npos = args
    W = r * math.cos(th) + s * math.sin(th); H = r * math.sin(th) + s * math.cos(th)
    k = int(math.ceil(max(W, H))) + 3
    L = k + c
    best = (-99, None)
    for fx in np.linspace(0, 1, npos, endpoint=False):
        for fy in np.linspace(0, 1, npos, endpoint=False):
            cx = math.floor(L / 2) + fx; cy = math.floor(L / 2) + fy
            P = build_block(L, cx, cy, th, r, s, -1e-12)
            if len(P) == 0:
                continue
            g = len(P) - k * k
            if g > best[0]:
                best = (g, (r, s, round(math.degrees(th), 3), round(fx, 3), round(fy, 3), k))
    return best


if __name__ == '__main__':
    c = float(sys.argv[1]); npos = int(sys.argv[2])
    thetas = np.radians(np.arange(float(sys.argv[3]), float(sys.argv[4]) + 1e-9, float(sys.argv[5])))
    rmax = int(sys.argv[6])
    jobs = [(r, s, th, c, npos) for r in range(2, rmax + 1) for s in range(r, rmax + 1) for th in thetas]
    t = time.time(); found = []
    with Pool(8) as pool:
        for g, cfg in pool.imap_unordered(job, jobs, chunksize=8):
            if g >= 1:
                found.append((g, cfg))
    found.sort(key=lambda x: (-x[0], x[1][0] * x[1][1]))
    print(f'c={c}: {len(found)} block configs with gain>=1 ({len(jobs)} jobs, {time.time()-t:.0f}s)')
    import json; json.dump([[g,[float(x) for x in cfg]] for g,cfg in found], open(f"blocks_c{c}.json","w"))
    for g, cfg in found[:25]:
        print('  gain', g, 'r,s,theta,fx,fy,k =', cfg)
