"""Rectangle gadgets: pack n = a*b + g unit squares in an (a+f) x (b+f) box, minimising f.
Global search = compress-and-relax from random starts (penalty: SAT penetration^2 + wall^2).
"""
import math, sys, time, json, os
import numpy as np
import numba as nb
from scipy.optimize import minimize

H = 0.5


@nb.njit(cache=True)
def eg(X, Lx, Ly):
    n = X.shape[0] // 3
    G = np.zeros_like(X); E = 0.0
    for i in range(n):
        x = X[3 * i]; y = X[3 * i + 1]; t = X[3 * i + 2]
        c, s = np.cos(t), np.sin(t)
        for sx in (-H, H):
            for sy in (-H, H):
                px = x + sx * c - sy * s; py = y + sx * s + sy * c
                dpx = -sx * s - sy * c; dpy = sx * c - sy * s
                if px < 0:
                    E += px * px; G[3 * i] += 2 * px; G[3 * i + 2] += 2 * px * dpx
                elif px > Lx:
                    d = px - Lx; E += d * d; G[3 * i] += 2 * d; G[3 * i + 2] += 2 * d * dpx
                if py < 0:
                    E += py * py; G[3 * i + 1] += 2 * py; G[3 * i + 2] += 2 * py * dpy
                elif py > Ly:
                    d = py - Ly; E += d * d; G[3 * i + 1] += 2 * d; G[3 * i + 2] += 2 * d * dpy
    for i in range(n):
        for j in range(i + 1, n):
            xi = X[3 * i]; yi = X[3 * i + 1]; ti = X[3 * i + 2]
            xj = X[3 * j]; yj = X[3 * j + 1]; tj = X[3 * j + 2]
            dx = xj - xi; dy = yj - yi
            if dx * dx + dy * dy >= 2.0:
                continue
            best = 1e9; bax = 0; bs = 0.0
            for a in range(4):
                if a < 2:
                    th = ti + a * (np.pi / 2); to = tj
                else:
                    th = tj + (a - 2) * (np.pi / 2); to = ti
                ux = np.cos(th); uy = np.sin(th)
                d = dx * ux + dy * uy
                ro = H * (abs(np.cos(to - th)) + abs(np.sin(to - th)))
                pen = H + ro - abs(d)
                if pen < best:
                    best = pen; bax = a; bs = 1.0 if d >= 0 else -1.0
                if best <= 0:
                    break
            if best <= 0:
                continue
            E += best * best
            a = bax
            if a < 2:
                th = ti + a * (np.pi / 2); to = tj
            else:
                th = tj + (a - 2) * (np.pi / 2); to = ti
            ux = np.cos(th); uy = np.sin(th)
            phi = to - th; cp = np.cos(phi); sp = np.sin(phi)
            dro = H * (np.sign(cp) * (-sp) + np.sign(sp) * cp)
            g = 2 * best
            gdx = -bs * ux; gdy = -bs * uy
            G[3 * j] += g * gdx; G[3 * j + 1] += g * gdy
            G[3 * i] -= g * gdx; G[3 * i + 1] -= g * gdy
            dth = -dro - bs * (dx * (-uy) + dy * ux)
            if a < 2:
                G[3 * i + 2] += g * dth; G[3 * j + 2] += g * dro
            else:
                G[3 * j + 2] += g * dth; G[3 * i + 2] += g * dro
    return E, G


def relax(X, Lx, Ly, it=3000):
    r = minimize(lambda v: eg(v, Lx, Ly), X, jac=True, method='L-BFGS-B',
                 options=dict(maxiter=it, ftol=1e-32, gtol=1e-18, maxcor=30))
    return r.x, r.fun


def compress(a, b, g, rng, f0=1.4, fmin=0.3, tries_per_step=4):
    """Random start in a loose box, then shrink f while a relaxation stays feasible."""
    n = a * b + g
    Lx, Ly = a + f0, b + f0
    X = np.empty(3 * n)
    X[0::3] = rng.uniform(0.5, Lx - 0.5, n); X[1::3] = rng.uniform(0.5, Ly - 0.5, n)
    pt = rng.uniform(0.05, 1.0)
    X[2::3] = np.where(rng.random(n) < pt, rng.uniform(0, np.pi / 2, n), 0.0)
    X, E = relax(X, Lx, Ly)
    if E > 1e-24:
        return None, None
    f = f0; best = None
    step = 0.05
    while f - step >= fmin and step > 2e-6:
        ft = f - step
        sx = (a + ft) / (a + f); sy = (b + ft) / (b + f)
        ok = False
        for _ in range(tries_per_step):
            Y = X.copy(); Y[0::3] *= sx; Y[1::3] *= sy
            if _ > 0:
                sig = 0.02 * _
                Y[0::3] += rng.normal(0, sig, n); Y[1::3] += rng.normal(0, sig, n); Y[2::3] += rng.normal(0, sig, n)
            Y, E = relax(Y, a + ft, b + ft)
            if E < 1e-26:
                ok = True; break
        if ok:
            X, f = Y, ft; best = (f, X.copy())
        else:
            step /= 2
    return best if best else (None, None)


def search(args):
    a, b, g, secs, seed = args
    rng = np.random.default_rng(seed)
    t_end = time.time() + secs; best_f = 9; best_X = None; runs = 0
    while time.time() < t_end:
        f, X = compress(a, b, g, rng)
        runs += 1
        if f is not None and f < best_f:
            best_f, best_X = f, X
    return a, b, g, best_f, (best_X.tolist() if best_X is not None else None), runs
