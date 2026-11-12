"""Shrink-and-relax search on incumbent packings; writes improved rows (float repr strings).
usage: opt_n.py current.csv seconds_per_n n1 n2 ...
Results: state/opt/{n}.json when a feasible smaller box is found (certified later).
"""
import csv, json, os, sys, time
import numpy as np
from multiprocessing import Pool
from relax import relax, side, build_pairs, energy_grad

OUT = 'state/opt'
os.makedirs(OUT, exist_ok=True)


def load(path, ns):
    by = {}
    with open(path, newline='') as f:
        for r in csv.DictReader(f):
            n = int(r['n'])
            if n in ns:
                by.setdefault(n, []).append(r)
    res = {}
    for n, rows in by.items():
        rows.sort(key=lambda r: int(r['square_id']))
        X = np.empty(3 * n)
        X[0::3] = [float(r['x']) for r in rows]; X[1::3] = [float(r['y']) for r in rows]
        X[2::3] = [float(r['angle']) for r in rows]
        res[n] = X
    return res


def to_box(X):
    t = X[2::3]; e = 0.5 * (np.abs(np.cos(t)) + np.abs(np.sin(t)))
    X = X.copy()
    X[0::3] -= (X[0::3] - e).min(); X[1::3] -= (X[1::3] - e).min()
    return X


def feasible(X, L):
    pa, pb = build_pairs(X, 1.6)
    return energy_grad(X, L, pa, pb, 1.0)[0]


def work(args):
    n, X, secs, seed = args
    rng = np.random.default_rng(seed)
    X = to_box(X); L0 = side(X); Lc = L0; cur = X.copy()
    t_end = time.time() + secs
    deltas = [3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5]
    tries = 0; acc = 0
    while time.time() < t_end:
        d = deltas[rng.integers(len(deltas))]
        Lt = Lc - d
        sig = [0.0, 0.003, 0.01, 0.03, 0.06][rng.integers(5)]
        Y = cur.copy()
        c = Lc / 2
        Y[0::3] = c + (Y[0::3] - c) * (Lt / Lc); Y[1::3] = c + (Y[1::3] - c) * (Lt / Lc)
        Y[0::3] -= (Lc - Lt) / 2; Y[1::3] -= (Lc - Lt) / 2
        if sig > 0:
            k = rng.integers(1, 4)
            if k == 1:   # global jiggle
                Y[0::3] += rng.normal(0, sig, n); Y[1::3] += rng.normal(0, sig, n); Y[2::3] += rng.normal(0, sig / 3, n)
            else:        # local region jiggle
                cx, cy = rng.uniform(0, Lt, 2); r = rng.uniform(2, 6)
                m = (Y[0::3] - cx) ** 2 + (Y[1::3] - cy) ** 2 < r * r
                Y[0::3][m] += rng.normal(0, sig * 3, m.sum()); Y[1::3][m] += rng.normal(0, sig * 3, m.sum())
                Y[2::3][m] += rng.normal(0, sig, m.sum())
        Y, E = relax(Y, Lt, iters=4000)
        tries += 1
        if E < 1e-26:
            s = side(Y)
            if s < Lc - 1e-12:
                cur, Lc = to_box(Y), s; acc += 1
                json.dump(dict(n=n, L0=L0, L=Lc, X=cur.tolist()), open(f'{OUT}/{n}.json', 'w'))
    return n, L0, Lc, tries, acc


if __name__ == '__main__':
    path, secs = sys.argv[1], float(sys.argv[2])
    ns = [int(a) for a in sys.argv[3:]]
    Xs = load(path, set(ns))
    with Pool(int(os.environ.get('WORKERS', 6))) as pool:
        for n, L0, Lc, tries, acc in pool.imap_unordered(work, [(n, Xs[n], secs, n) for n in ns]):
            print(time.strftime('%H:%M:%S'), f'n={n} L0={L0:.9f} best={Lc:.9f} dL={L0-Lc:.3e} tries={tries} acc={acc}', flush=True)
