"""Search for low-f rectangle gadgets: pack m = a*b + g unit squares in an (a+f) x (b+f) box.

Numba is not installed here, so the SAT-pair energy of
`poc/rectangle_gadgets/poc/rgad.py` is re-implemented with vectorised numpy
(identical model: squared corner penetration + squared wall violation, best of
the four separating axes).  Descent is scipy L-BFGS-B, one process per restart.

usage:
  gadsearch.py --shapes 4,7,3 4,13,3 --f 0.78 0.70 --secs 60 --jobs 32 --out res.json
  gadsearch.py --selftest
"""
import argparse
import json
import math
import time
from multiprocessing import Pool

import numpy as np
from scipy.optimize import minimize

H = 0.5
SC = np.array([[-H, -H], [H, -H], [-H, H], [H, H]])       # corner offsets (sx, sy)


def energy(X, Lx, Ly):
    n = X.shape[0] // 3
    x = X[0::3]; y = X[1::3]; t = X[2::3]
    c = np.cos(t); s = np.sin(t)
    px = x[:, None] + SC[:, 0] * c[:, None] - SC[:, 1] * s[:, None]
    py = y[:, None] + SC[:, 0] * s[:, None] + SC[:, 1] * c[:, None]
    dpx = -SC[:, 0] * s[:, None] - SC[:, 1] * c[:, None]
    dpy = SC[:, 0] * c[:, None] - SC[:, 1] * s[:, None]

    G = np.zeros(3 * n)
    E = 0.0
    vx = np.where(px < 0, px, np.where(px > Lx, px - Lx, 0.0))
    vy = np.where(py < 0, py, np.where(py > Ly, py - Ly, 0.0))
    E += float((vx * vx).sum() + (vy * vy).sum())
    gx = 2 * vx; gy = 2 * vy
    G[0::3] += gx.sum(1); G[1::3] += gy.sum(1)
    G[2::3] += (gx * dpx).sum(1) + (gy * dpy).sum(1)

    I, J = PAIRS[n]
    dx = x[J] - x[I]; dy = y[J] - y[I]
    ti = t[I]; tj = t[J]
    ci = np.cos(ti); si = np.sin(ti); cj = np.cos(tj); sj = np.sin(tj)
    ux = np.stack([ci, -si, cj, -sj])          # 4 axes x P
    uy = np.stack([si, ci, sj, cj])
    d = dx * ux + dy * uy
    phi = tj - ti
    ro = H * (np.abs(np.cos(phi)) + np.abs(np.sin(phi)))
    pen = (H + ro) - np.abs(d)
    k = pen.argmin(0)
    ar = np.arange(k.size)
    best = pen[k, ar]
    act = best > 0
    if not act.any():
        return E, G
    best = best[act]
    kk = k[act]; aa = ar[act]
    E += float((best * best).sum())
    dk = d[kk, aa]
    uxk = ux[kk, aa]; uyk = uy[kk, aa]
    bs = np.where(dk >= 0, 1.0, -1.0)
    g = 2 * best
    ii = I[aa]; jj = J[aa]
    cp = np.cos(phi[aa]); sp = np.sin(phi[aa])
    dro = H * (-np.sign(cp) * sp + np.sign(sp) * cp)     # d ro / d tj  (d ro / d ti = -dro)
    # d(d)/d theta of the axis-carrying square: axis0 -> d1, axis1 -> -d0, axis2 -> d3, axis3 -> -d2
    dproj = np.stack([d[1], -d[0], d[3], -d[2]])
    dproj = np.take_along_axis(dproj, k[None, :], 0)[0][aa]
    carry_i = kk < 2                                   # axis belongs to square i
    # x, y, theta gradients accumulated with one bincount
    carry = np.where(carry_i, ii, jj)
    gdx = -bs * uxk; gdy = -bs * uyk
    idx = np.concatenate([jj * 3, ii * 3, jj * 3 + 1, ii * 3 + 1,
                          carry * 3 + 2, jj * 3 + 2, ii * 3 + 2])
    wts = np.concatenate([g * gdx, -g * gdx, g * gdy, -g * gdy,
                          -g * bs * dproj, g * dro, -g * dro])
    G += np.bincount(idx, weights=wts, minlength=3 * n)
    return E, G


PAIRS = {}


def pairs(n):
    if n not in PAIRS:
        I, J = np.triu_indices(n, 1)
        PAIRS[n] = (I.astype(np.int64), J.astype(np.int64))
    return PAIRS[n]


def relax(X, Lx, Ly, it=400):
    pairs(X.shape[0] // 3)
    r = minimize(lambda v: energy(v, Lx, Ly), X, jac=True, method='L-BFGS-B',
                 options=dict(maxiter=it, ftol=1e-32, gtol=1e-20, maxcor=30))
    return r.x, float(r.fun)


def start_grid(a, b, g, rng, Lx, Ly, common_tilt=0.0):
    """a x b upright lattice block in the bottom-left corner plus g squares in the L strip."""
    n = a * b + g
    X = np.empty(3 * n)
    k = 0
    for j in range(b):
        for i in range(a):
            X[3 * k: 3 * k + 3] = (i + 0.5, j + 0.5, common_tilt)
            k += 1
    for _ in range(g):
        # L-shaped strip: right strip x in [a, Lx], top strip y in [b, Ly]
        if rng.random() < 0.5:
            X[3 * k: 3 * k + 3] = (rng.uniform(a, Lx), rng.uniform(0.3, Ly), rng.uniform(-1.2, 1.2))
        else:
            X[3 * k: 3 * k + 3] = (rng.uniform(0.3, Lx), rng.uniform(b, Ly), rng.uniform(-1.2, 1.2))
        k += 1
    return X


def start_rand(a, b, g, rng, Lx, Ly, ptilt=0.6):
    n = a * b + g
    X = np.empty(3 * n)
    X[0::3] = rng.uniform(H, Lx - H, n)
    X[1::3] = rng.uniform(H, Ly - H, n)
    X[2::3] = np.where(rng.random(n) < ptilt, rng.uniform(-np.pi / 2, np.pi / 2, n), 0.0)
    return X


def start_tilted_lattice(a, b, g, rng, Lx, Ly):
    """tilted lattice whose bounding box is (a, b) plus loose margin; random slope."""
    n = a * b + g
    th = rng.uniform(0.15, 1.35)
    cols = max(1, int(round(math.sqrt(n * a / max(b, 1)))))
    cols = min(max(cols, 1), n)
    X = np.empty(3 * n)
    X[0::3] = rng.uniform(0, 0.3)
    X[1::3] = rng.uniform(0, 0.3)
    X[2::3] = th
    idx = 0
    for q in range(n):
        i, j = q % cols, q // cols
        X[3 * q] = H + i * abs(math.cos(th)) + j * abs(math.sin(th)) + rng.normal(0, 0.02)
        X[3 * q + 1] = H + i * abs(math.sin(th)) + j * abs(math.cos(th)) + rng.normal(0, 0.02)
    return X


def one(args):
    a, b, g, f_target, seed, secs = args
    rng = np.random.default_rng(seed)
    t_end = time.time() + secs
    best = (9.0, None, None)
    tries = 0
    while time.time() < t_end:
        tries += 1
        kind = rng.integers(0, 3)
        f = f_target + rng.uniform(0.0, 0.5)
        Lx, Ly = a + f, b + f
        if kind == 0:
            X = start_grid(a, b, g, rng, Lx, Ly, 0.0 if rng.random() < 0.5 else rng.uniform(0, 0.4))
        elif kind == 1:
            X = start_rand(a, b, g, rng, Lx, Ly)
        else:
            X = start_tilted_lattice(a, b, g, rng, Lx, Ly)
        X, E = relax(X, Lx, Ly, 400)
        fcur = f
        # shrink toward the target in steps, warm starting from the relaxed state
        while fcur > f_target + 1e-12 and E < 1e-12:
            fnew = max(f_target, fcur - max(0.04, 0.25 * (fcur - f_target)))
            sx = (a + fnew) / (a + fcur); sy = (b + fnew) / (b + fcur)
            Y = X.copy(); Y[0::3] *= sx; Y[1::3] *= sy
            Y, E2 = relax(Y, a + fnew, b + fnew, 250)
            if E2 < 1e-17:
                X, E, fcur = Y, E2, fnew
            else:
                break
        if E < best[0]:
            best = (E, X.copy() if E < 1e-8 else None, fcur)
        if E < 1e-20:
            break
    return a, b, g, best[0], (best[1].tolist() if best[1] is not None else None), best[2], tries


def sat_violation(X, Lx, Ly):
    """exact-ish max violation: max(overlap depth, wall overshoot)."""
    n = X.shape[0] // 3
    x = X[0::3]; y = X[1::3]; t = X[2::3]
    c = np.cos(t); s = np.sin(t)
    px = x[:, None] + SC[:, 0] * c[:, None] - SC[:, 1] * s[:, None]
    py = y[:, None] + SC[:, 0] * s[:, None] + SC[:, 1] * c[:, None]
    v = max(max(0.0, -px.min()), max(0.0, px.max() - Lx),
            max(0.0, -py.min()), max(0.0, py.max() - Ly))
    I, J = pairs(n)
    dx = x[J] - x[I]; dy = y[J] - y[I]
    ti = t[I]; tj = t[J]
    ux = np.stack([np.cos(ti), -np.sin(ti), np.cos(tj), -np.sin(tj)])
    uy = np.stack([np.sin(ti), np.cos(ti), np.sin(tj), np.cos(tj)])
    d = np.abs(dx * ux + dy * uy)
    phi = tj - ti
    ro = H * (np.abs(np.cos(phi)) + np.abs(np.sin(phi)))
    pen = (H + ro) - d
    return max(v, float(pen.min(0).max(initial=-1e9)))


def selftest():
    """known values: n=5 -> f=0.70711, n=10 -> f=0.70711, n=26 (5,5,1) -> f=0.62132"""
    for (a, b, g, fk) in [(2, 2, 1, 0.707107), (3, 3, 1, 0.707107), (5, 5, 1, 0.621320)]:
        r = one((a, b, g, fk, 1, 40))
        print(f'(a,b,g)=({a},{b},{g}) target f={fk:.6f}: E={r[3]:.3e} f_found={r[5]:.6f} '
              f'tries={r[6]}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shapes', nargs='*', default=[])
    ap.add_argument('--f', nargs='*', type=float, default=[])
    ap.add_argument('--secs', type=float, default=60)
    ap.add_argument('--jobs', type=int, default=32)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=None)
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        selftest(); return
    specs = []
    for i, s in enumerate(a.shapes):
        w, h, gg = (int(v) for v in s.split(','))
        specs.append((w, h, gg, a.f[i]))
    tasks = []
    per = max(1, a.jobs // max(1, len(specs)))
    for (w, h, gg, f) in specs:
        for r in range(per):
            tasks.append((w, h, gg, f, a.seed + 1000 * w + 37 * h + 7 * gg + r, a.secs))
    res = {}
    t0 = time.time()
    with Pool(min(a.jobs, len(tasks))) as pool:
        for w, h, gg, E, X, ffin, tries in pool.imap_unordered(one, tasks):
            k = f'{w},{h},{gg}'
            cur = res.get(k)
            if cur is None or E < cur['E']:
                res[k] = dict(E=E, f=ffin, X=X, tries=tries)
            else:
                cur['tries'] += tries
    for k in sorted(res, key=lambda k: tuple(map(int, k.split(',')))):
        r = res[k]
        print(f'({k}) n={int(k.split(",")[0]) * int(k.split(",")[1]) + int(k.split(",")[2])} '
              f'best E={r["E"]:.3e} f={r["f"]:.6f} tries={r["tries"]}', flush=True)
    if a.out:
        json.dump(res, open(a.out, 'w'))
    print(f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
