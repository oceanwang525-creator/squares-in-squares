"""Penalty relaxation of unit squares in a box [0,L]^2 (numba).
Energy = sum over pairs of SAT penetration^2 + sum of wall violations^2.
Variables per square: x, y, theta.
"""
import numpy as np
import numba as nb
from scipy.optimize import minimize

H = 0.5


@nb.njit(cache=True)
def corners(x, y, t):
    c, s = np.cos(t), np.sin(t)
    out = np.empty((4, 2))
    k = 0
    for sx in (-H, H):
        for sy in (-H, H):
            out[k, 0] = x + sx * c - sy * s
            out[k, 1] = y + sx * s + sy * c
            k += 1
    return out


@nb.njit(cache=True)
def build_pairs(X, cut):
    n = X.shape[0] // 3
    xs = X[0::3]; ys = X[1::3]
    cell = cut
    mx = xs.min(); my = ys.min()
    cx = ((xs - mx) / cell).astype(np.int64); cy = ((ys - my) / cell).astype(np.int64)
    W = cx.max() + 1; Hh = cy.max() + 1
    head = -np.ones(W * Hh, np.int64); nxt = -np.ones(n, np.int64)
    for i in range(n):
        c = cx[i] * Hh + cy[i]
        nxt[i] = head[c]; head[c] = i
    pa = np.empty(n * 24, np.int64); pb = np.empty(n * 24, np.int64); m = 0
    c2 = cut * cut
    for i in range(n):
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                gx = cx[i] + dx; gy = cy[i] + dy
                if gx < 0 or gy < 0 or gx >= W or gy >= Hh:
                    continue
                j = head[gx * Hh + gy]
                while j >= 0:
                    if j > i:
                        ddx = xs[i] - xs[j]; ddy = ys[i] - ys[j]
                        if ddx * ddx + ddy * ddy < c2:
                            if m >= pa.shape[0]:
                                pa2 = np.empty(pa.shape[0] * 2, np.int64); pb2 = np.empty(pa.shape[0] * 2, np.int64)
                                pa2[:m] = pa[:m]; pb2[:m] = pb[:m]; pa = pa2; pb = pb2
                            pa[m] = i; pb[m] = j; m += 1
                    j = nxt[j]
    return pa[:m], pb[:m]


@nb.njit(cache=True)
def energy_grad(X, L, pa, pb, wpair):
    n = X.shape[0] // 3
    G = np.zeros_like(X)
    E = 0.0
    # walls: each corner within [0, L]
    for i in range(n):
        x = X[3 * i]; y = X[3 * i + 1]; t = X[3 * i + 2]
        c, s = np.cos(t), np.sin(t)
        for sx in (-H, H):
            for sy in (-H, H):
                px = x + sx * c - sy * s
                py = y + sx * s + sy * c
                dpx_dt = -sx * s - sy * c
                dpy_dt = sx * c - sy * s
                if px < 0:
                    E += px * px; G[3 * i] += 2 * px; G[3 * i + 2] += 2 * px * dpx_dt
                elif px > L:
                    d = px - L; E += d * d; G[3 * i] += 2 * d; G[3 * i + 2] += 2 * d * dpx_dt
                if py < 0:
                    E += py * py; G[3 * i + 1] += 2 * py; G[3 * i + 2] += 2 * py * dpy_dt
                elif py > L:
                    d = py - L; E += d * d; G[3 * i + 1] += 2 * d; G[3 * i + 2] += 2 * d * dpy_dt
    # pairs: SAT penetration depth
    for k in range(pa.shape[0]):
        i = pa[k]; j = pb[k]
        xi = X[3 * i]; yi = X[3 * i + 1]; ti = X[3 * i + 2]
        xj = X[3 * j]; yj = X[3 * j + 1]; tj = X[3 * j + 2]
        dx = xj - xi; dy = yj - yi
        if dx * dx + dy * dy >= 2.0:
            continue
        best = 1e9; bax = 0; bsgn = 0.0
        # axes: (cos ti, sin ti), (-sin ti, cos ti), same for tj
        for a in range(4):
            if a < 2:
                th = ti + a * (np.pi / 2)
                to = tj
            else:
                th = tj + (a - 2) * (np.pi / 2)
                to = ti
            ux = np.cos(th); uy = np.sin(th)
            d = dx * ux + dy * uy
            ro = H * (abs(np.cos(to - th)) + abs(np.sin(to - th)))
            pen = H + ro - abs(d)
            if pen < best:
                best = pen; bax = a; bsgn = 1.0 if d >= 0 else -1.0
            if best <= 0:
                break
        if best <= 0:
            continue
        w = wpair
        E += w * best * best
        # gradient of pen = H + ro(phi) - |d|, phi = to - th, d = (dx,dy).u(th)
        a = bax
        if a < 2:
            th = ti + a * (np.pi / 2); to = tj
        else:
            th = tj + (a - 2) * (np.pi / 2); to = ti
        ux = np.cos(th); uy = np.sin(th)
        phi = to - th
        cp = np.cos(phi); sp = np.sin(phi)
        dro_dphi = H * (np.sign(cp) * (-sp) + np.sign(sp) * cp)
        # d(|d|)/d(dx,dy) = sgn*u ; d(|d|)/dth = sgn*(dx*(-uy)+dy*ux)
        g = 2 * w * best
        gdx = -bsgn * ux; gdy = -bsgn * uy
        # dx = xj - xi
        G[3 * j] += g * gdx; G[3 * j + 1] += g * gdy
        G[3 * i] -= g * gdx; G[3 * i + 1] -= g * gdy
        dpen_dth = -dro_dphi - bsgn * (dx * (-uy) + dy * ux)
        dpen_dto = dro_dphi
        if a < 2:
            G[3 * i + 2] += g * dpen_dth; G[3 * j + 2] += g * dpen_dto
        else:
            G[3 * j + 2] += g * dpen_dth; G[3 * i + 2] += g * dpen_dto
    return E, G


def relax(X, L, iters=3000, tol=1e-30, wpair=1.0):
    X = X.copy()
    for rnd in range(6):
        pa, pb = build_pairs(X, 1.6)
        r = minimize(lambda v: energy_grad(v, L, pa, pb, wpair), X, jac=True, method='L-BFGS-B',
                     options=dict(maxiter=iters, ftol=tol, gtol=1e-16, maxcor=30))
        moved = np.abs(r.x - X).max()
        X = r.x
        if moved < 0.05:
            break
    pa, pb = build_pairs(X, 1.6)
    E, _ = energy_grad(X, L, pa, pb, wpair)
    return X, E


def side(X):
    t = X[2::3]
    e = H * (np.abs(np.cos(t)) + np.abs(np.sin(t)))
    x = X[0::3]; y = X[1::3]
    return max((x + e).max() - (x - e).min(), (y + e).max() - (y - e).min())
