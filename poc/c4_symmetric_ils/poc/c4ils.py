"""C4-symmetric compress-and-relax with iterated-local-search kicks.
Free variables: one representative (x, y, theta) per orbit of four (coordinates relative to the box
centre) and, if n = 1 mod 4, the angle of a centre square. Copies k = 1..3 are 90-degree rotations.
"""
import json, math, os, sys, time
import numpy as np
from scipy.optimize import minimize
from rgad import eg

PI2 = math.pi / 2


def expand(p, m, centre, L):
    """p = [x0,y0,t0, x1,y1,t1, ..., (tc)] -> full X in box coordinates."""
    c = L / 2
    X = np.empty(3 * (4 * m + centre))
    for k in range(4):
        ck, sk = round(math.cos(k * PI2)), round(math.sin(k * PI2))
        u = p[0:3 * m:3]; v = p[1:3 * m:3]
        X[3 * k * m + 0:3 * (k + 1) * m:3] = c + ck * u - sk * v
        X[3 * k * m + 1:3 * (k + 1) * m:3] = c + sk * u + ck * v
        X[3 * k * m + 2:3 * (k + 1) * m:3] = p[2:3 * m:3] + k * PI2
    if centre:
        X[-3] = c; X[-2] = c; X[-1] = p[-1]
    return X


def energy(p, m, centre, L):
    X = expand(p, m, centre, L)
    E, G = eg(X, L, L)
    g = np.zeros_like(p)
    for k in range(4):
        ck, sk = round(math.cos(k * PI2)), round(math.sin(k * PI2))
        gx = G[3 * k * m + 0:3 * (k + 1) * m:3]; gy = G[3 * k * m + 1:3 * (k + 1) * m:3]
        g[0:3 * m:3] += ck * gx + sk * gy          # R^T applied to (gx, gy)
        g[1:3 * m:3] += -sk * gx + ck * gy
        g[2:3 * m:3] += G[3 * k * m + 2:3 * (k + 1) * m:3]
    if centre:
        g[-1] = G[-1]
    return E, g


def relax(p, m, centre, L, it=2000):
    r = minimize(lambda q: energy(q, m, centre, L), p, jac=True, method='L-BFGS-B',
                 options=dict(maxiter=it, ftol=1e-32, gtol=1e-18, maxcor=40))
    return r.x, r.fun


def random_start(m, centre, L, rng):
    p = np.empty(3 * m + centre)
    h = L / 2
    # representatives in one quadrant-ish wedge: u in (0, h), v in (0, h) (rotations cover the rest)
    p[0:3 * m:3] = rng.uniform(0.0, h - 0.5, m)
    p[1:3 * m:3] = rng.uniform(0.0, h - 0.5, m)
    pt = rng.uniform(0.1, 1.0)
    p[2:3 * m:3] = np.where(rng.random(m) < pt, rng.uniform(0, PI2, m), 0.0)
    if centre:
        p[-1] = rng.uniform(0, PI2)
    return p


def kick(p, m, centre, L, rng):
    q = p.copy()
    r = rng.random()
    if r < 0.4:      # jitter everything a little
        s = rng.choice([0.01, 0.03, 0.08])
        q[0:3 * m:3] += rng.normal(0, s, m); q[1:3 * m:3] += rng.normal(0, s, m); q[2:3 * m:3] += rng.normal(0, s, m)
    elif r < 0.8:    # re-seat a few orbits at random
        idx = rng.choice(m, size=min(m, rng.integers(1, 4)), replace=False)
        q[3 * idx] = rng.uniform(0, L / 2 - 0.5, len(idx)); q[3 * idx + 1] = rng.uniform(0, L / 2 - 0.5, len(idx))
        q[3 * idx + 2] = rng.uniform(0, PI2, len(idx))
    else:            # rotate angles of a few orbits
        idx = rng.choice(m, size=min(m, rng.integers(1, 6)), replace=False)
        q[3 * idx + 2] += rng.normal(0, 0.3, len(idx))
    if centre and rng.random() < 0.3:
        q[-1] += rng.normal(0, 0.2)
    return q


def compress(n, L0, Lmin, rng, kicks=30, t_end=None):
    centre = n % 4
    assert centre in (0, 1), 'C4 needs n = 0 or 1 mod 4'
    m = (n - centre) // 4
    L = L0
    p = random_start(m, centre, L, rng)
    p, E = relax(p, m, centre, L)
    # ILS until feasible at the loose start
    for _ in range(kicks):
        if E < 1e-26: break
        q, Eq = relax(kick(p, m, centre, L, rng), m, centre, L)
        if Eq < E: p, E = q, Eq
    if E >= 1e-26:
        return None, None
    best = (L, p.copy()); step = 0.05
    while step > 1e-5 and L - step >= Lmin and (t_end is None or time.time() < t_end):
        Lt = L - step
        q = p.copy(); q[0:3 * m:3] *= Lt / L; q[1:3 * m:3] *= Lt / L
        q, Eq = relax(q, m, centre, Lt)
        tries = 0
        while Eq >= 1e-26 and tries < kicks:
            r_, Er = relax(kick(q, m, centre, Lt, rng), m, centre, Lt)
            if Er < Eq: q, Eq = r_, Er
            tries += 1
        if Eq < 1e-26:
            p, L = q, Lt; best = (L, p.copy())
        else:
            step /= 2
    return best


def worker(args):
    n, L0, Lmin, secs, seed = args
    rng = np.random.default_rng(seed)
    t_end = time.time() + secs; bestL = 1e9; bestp = None; runs = 0
    while time.time() < t_end:
        L, p = compress(n, L0, Lmin, rng, t_end=t_end)
        runs += 1
        if L is not None and L < bestL:
            bestL, bestp = L, p
    return n, bestL, (bestp.tolist() if bestp is not None else None), runs


if __name__ == '__main__':
    from multiprocessing import Pool
    n, L0, Lmin, secs, W = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5])
    os.makedirs('state/c4', exist_ok=True)
    best = (1e9, None)
    with Pool(W) as pool:
        for n_, L, p, runs in pool.imap_unordered(worker, [(n, L0, Lmin, secs, 7919 * n + s) for s in range(W)]):
            print(time.strftime('%H:%M:%S'), f'n={n_} worker best L={L:.6f} runs={runs}', flush=True)
            if L < best[0]: best = (L, p)
    if best[1] is not None:
        json.dump(dict(n=n, L=best[0], p=best[1]), open(f'state/c4/{n}.json', 'w'))
    print('BEST', n, best[0])


def ils_fixed(n, L, secs, seed, restart_after=400, T=0.0):
    """ILS at a fixed side L: minimise overlap energy; returns best (E, p)."""
    rng = np.random.default_rng(seed)
    centre = n % 4; m = (n - centre) // 4
    t_end = time.time() + secs
    bestE, bestp = 1e9, None; evals = 0
    while time.time() < t_end:
        p, E = relax(random_start(m, centre, L, rng), m, centre, L); evals += 1
        stall = 0
        while time.time() < t_end and stall < restart_after:
            q, Eq = relax(kick(p, m, centre, L, rng), m, centre, L, it=600); evals += 1
            if Eq < E - 1e-15 or (T > 0 and rng.random() < math.exp(-(Eq - E) / T)):
                if Eq < E - 1e-12: stall = 0
                else: stall += 1
                p, E = q, Eq
            else:
                stall += 1
            if E < bestE:
                bestE, bestp = E, p.copy()
            if E < 1e-26:
                return bestE, bestp, evals
    return bestE, bestp, evals


def ils_worker(args):
    n, L, secs, seed = args
    E, p, evals = ils_fixed(n, L, secs, seed)
    return n, L, E, (p.tolist() if p is not None else None), evals
