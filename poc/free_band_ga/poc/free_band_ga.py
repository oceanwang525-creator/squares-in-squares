#!/usr/bin/env python3
"""Population search over a free-angle staircase band.

Genome: (theta, T1, T2, w, gap, phase).  BL upright cells are selected by
n.(i+.5,j+.5) <= T1 with n=(cos,sin); TR reflected cells by the mirrored
condition >= T2.  The band is a square lattice at angle theta occupying w rows
between the two digital staircases.  The count is analytic (no overlap), so a
large population can be evaluated quickly.
"""
import math
import random

import numpy as np

L = 30.6609125615235
K = int(math.ceil(L))
U = np.array([[u, v] for u in (0, 1) for v in (0, 1)], dtype=float)


def pack(theta, T1, T2, w, gap, ph):
    nx, ny = math.cos(theta), math.sin(theta)
    tx, ty = -ny, nx
    # BL
    i = np.arange(K + 1)[:, None]
    j = np.arange(K + 1)[None, :]
    cx = i + 0.5; cy = j + 0.5
    val = nx * cx + ny * cy
    mask = (val <= T1) & (i + 1 <= L + 1e-9) & (j + 1 <= L + 1e-9)
    nBL = int(mask.sum())
    if nBL == 0:
        return 0, None
    C1 = float((nx * (i + 1) + ny * (j + 1))[mask].max())
    # TR: select reflected indices i,j with n.(i+.5,j+.5) <= T2
    valr = nx * (i + 0.5) + ny * (j + 0.5)
    maskr = (valr <= T2) & (L - i - 1 >= -1e-9) & (L - j - 1 >= -1e-9)
    nTR = int(maskr.sum())
    if nTR == 0:
        return 0, None
    C2 = float((nx * (L - i - 1) + ny * (L - j - 1))[maskr].min())
    if C2 <= C1 + 1e-9:
        return 0, None
    e = 0.5 * (abs(nx) + abs(ny))
    bands = []
    for b in range(w):
        C = C1 + gap + 0.5 + b
        if C + 0.5 > C2 - gap + 1e-9:
            break
        # x = C*nx + a*tx, y = C*ny + a*ty ; |x-Cx|<=e, |y-Cy|<=e
        lo = -1e9; hi = 1e9
        if abs(tx) > 1e-12:
            a1 = (e - C * nx) / tx; a2 = (L - e - C * nx) / tx
            lo = max(lo, min(a1, a2)); hi = min(hi, max(a1, a2))
        else:
            if C * nx - e < -1e-9 or C * nx + e > L + 1e-9:
                continue
        if abs(ty) > 1e-12:
            a1 = (e - C * ny) / ty; a2 = (L - e - C * ny) / ty
            lo = max(lo, min(a1, a2)); hi = min(hi, max(a1, a2))
        else:
            if C * ny - e < -1e-9 or C * ny + e > L + 1e-9:
                continue
        if hi < lo:
            continue
        first = math.ceil(lo - ph - 1e-9)
        last = math.floor(hi - ph + 1e-9)
        cnt = max(0, last - first + 1)
        for k in range(first, last + 1):
            a = ph + k
            x = C * nx + a * tx; y = C * ny + a * ty
            bands.append((x, y, theta))
    return nBL + nTR + len(bands), (nBL, nTR, bands)


def mutate(g, rng, scale=1.0):
    theta, T1, T2, w, gap, ph = g
    theta += rng.gauss(0, 0.01 * scale)
    T1 += rng.gauss(0, 1.5 * scale)
    T2 += rng.gauss(0, 1.5 * scale)
    if rng.random() < 0.25:
        w += rng.choice([-1, 1])
    gap += rng.gauss(0, 0.02 * scale)
    ph += rng.gauss(0, 0.05 * scale)
    theta = min(max(theta, 0.30), 0.65)
    T1 = min(max(T1, 5.0), 36.0)
    T2 = min(max(T2, 0.0), 15.0)
    if T1 + T2 > 37.0:
        T1 = 37.0 - T2
    w = int(min(max(w, 2), 9))
    gap = min(max(gap, 0.0), 0.35)
    ph = ph % 1.0
    return (theta, T1, T2, w, gap, ph)


if __name__ == "__main__":
    rng = random.Random(12345)
    # seed with rational 2:1-like configurations and random draws
    pop = []
    for _ in range(64):
        th = rng.uniform(0.42, 0.51)
        t1 = rng.uniform(8, 33)
        t2 = rng.uniform(0, min(12, 36 - t1))
        pop.append((th, t1, t2,
                    rng.randint(2, 7), rng.uniform(0, 0.2), rng.random()))
    best = (0, None, None)
    for gen in range(300):
        scored = []
        for g in pop:
            n, info = pack(*g)
            scored.append((n, g, info))
            if n > best[0]:
                best = (n, g, info)
        scored.sort(key=lambda t: -t[0])
        elite = [s[1] for s in scored[:12]]
        if gen % 20 == 0:
            print(f"gen {gen} best {scored[0][0]} overall {best[0]} "
                  f"g={scored[0][1]}", flush=True)
        pop = elite[:]
        while len(pop) < 64:
            base = rng.choice(elite)
            pop.append(mutate(base, rng, 1.0 if rng.random() < 0.7 else 0.2))
    n, g, info = best
    print("BEST", n, g, "parts", None if info is None else
          (info[0], info[1], len(info[2])))
    if info is not None and n >= 895:
        nBL, nTR, bands = info
        # rebuild BL/TR explicit for certification
        theta, T1, T2, w, gap, ph = g
        nx, ny = math.cos(theta), math.sin(theta)
        rows = []
        for i in range(K + 1):
            for j in range(K + 1):
                if nx * (i + 0.5) + ny * (j + 0.5) <= T1 \
                        and i + 1 <= L + 1e-9 and j + 1 <= L + 1e-9:
                    rows.append((i + 0.5, j + 0.5, 0.0))
        for i in range(K + 1):
            for j in range(K + 1):
                cx = L - i - 0.5; cy = L - j - 0.5
                if nx * cx + ny * cy >= T2 and cx - 0.5 >= -1e-9 and cy - 0.5 >= -1e-9:
                    rows.append((cx, cy, 0.0))
        rows.extend(bands)
        np.save("/tmp/opencode/free_band_best.npy", np.array(rows))
        print("saved", len(rows), "rows")
