"""Absorb-one-extra-square search: band seed with capacity n-d at side Lt, insert d squares
in the largest voids, relax with free angles + jiggle. Success = zero-penalty packing of n at Lt.
usage: insert_search.py n Lt seconds workers
"""
import json, os, sys, time
import numpy as np
from multiprocessing import Pool
from bandgen import build
from relax import relax, side

R2 = np.sqrt(2)
os.makedirs('state/insert', exist_ok=True)


def voids(P, L, rng, k):
    pts = rng.uniform(0.5, L - 0.5, (4000, 2))
    d = np.min(np.hypot(pts[:, None, 0] - P[None, :, 0], pts[:, None, 1] - P[None, :, 1]), axis=1)
    idx = np.argsort(-d)[:k * 20]
    return pts[idx[rng.integers(0, len(idx), k)]]


def worker(args):
    n, Lt, secs, seed = args
    rng = np.random.default_rng(seed)
    t_end = time.time() + secs
    tries = 0; best_E = 1e9
    while time.time() < t_end:
        m = int(rng.integers(3, int(Lt * 0.75))); p = int(rng.integers(m, int(Lt * 1.2)))
        th = np.pi / 4 + rng.normal(0, 0.03) * (rng.random() < 0.5)
        du = rng.uniform(-1, 1); dv = rng.uniform(-1, 1)
        P = build(Lt, m, dv, R2 * Lt / 2 - (p - 1) / 2 + du, th, -1e-12, 0, p)
        d = n - len(P)
        if d < 0:
            P = P[:n]; d = 0
        if d > 2 or m * p < 9:
            continue
        X = np.empty(3 * n)
        X[0:3 * len(P):3] = P[:, 0]; X[1:3 * len(P):3] = P[:, 1]; X[2:3 * len(P):3] = P[:, 2]
        V = voids(P, Lt, rng, d)
        for i in range(d):
            k = len(P) + i
            X[3 * k], X[3 * k + 1], X[3 * k + 2] = V[i, 0], V[i, 1], rng.uniform(0, np.pi / 2)
        sig = rng.choice([0.0, 0.02, 0.05])
        X[0::3] += rng.normal(0, sig, n); X[1::3] += rng.normal(0, sig, n); X[2::3] += rng.normal(0, sig / 2, n)
        X, E = relax(X, Lt, iters=6000)
        tries += 1; best_E = min(best_E, E)
        if E < 1e-26:
            s = side(X)
            json.dump(dict(n=n, L=s, X=X.tolist(), seed=[m, p, th, du, dv]), open(f'state/insert/{n}_{seed}_{tries}.json', 'w'))
            return n, seed, tries, best_E, s
    return n, seed, tries, best_E, None


if __name__ == '__main__':
    n, Lt, secs, W = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
    with Pool(W) as pool:
        for r in pool.imap_unordered(worker, [(n, Lt, secs, s) for s in range(W)]):
            print(time.strftime('%H:%M:%S'), 'n=%d seed=%d tries=%d bestE=%.3e found=%s' % r, flush=True)
