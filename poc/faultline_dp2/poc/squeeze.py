"""Donor-squeeze probe for a target gadget (a, b, g).

Take the library gadget closest to the target (more squares in a slightly larger
pocket), delete squares until a*b+g remain, then shrink the pocket with the
penalty relaxation of `gadsearch.py`.  Cheap warm start compared with random
restarts, because the donor is already an interlocked dense pocket.

usage: squeeze.py --shape 4,7,3 --donor 5,7,3 --secs 60
"""
import argparse
import json
import time

import numpy as np

import gadsearch as G


def load_gadgets(path):
    best = {}
    for g in json.load(open(path)):
        f = max(g['fx'], g['fy'])
        for (w, h) in ((g['w'], g['h']), (g['h'], g['w'])):
            k = (w, h, g['g'])
            if k not in best or f < best[k][0]:
                best[k] = (f, g)
    return best


def vec(sq):
    a = np.array(sq, float)
    X = np.zeros(3 * len(a))
    X[0::3] = a[:, 0]; X[1::3] = a[:, 1]; X[2::3] = a[:, 2]
    return X


def bbox(X):
    n = X.shape[0] // 3
    e = 0.5 * (np.abs(np.cos(X[2::3])) + np.abs(np.sin(X[2::3])))
    return (X[0::3] - e).min(), (X[0::3] + e).max(), (X[1::3] - e).min(), (X[1::3] + e).max()


def squeeze(X, a, b, t, f_hi, verbose=True):
    """shrink toward the smallest f that stays feasible."""
    f = f_hi
    x0, x1, y0, y1 = bbox(X)
    X[0::3] -= x0 - 1e-9
    X[1::3] -= y0 - 1e-9
    X, E = G.relax(X.copy(), a + f, b + f, 600)
    if E > 1e-14:
        return 9.0, None
    best = (f, X.copy())
    step = 0.04
    while step > 1e-4:
        ft = f - step
        if ft < 0.3:
            break
        sx = (a + ft) / (a + f); sy = (b + ft) / (b + f)
        Y = X.copy(); Y[0::3] *= sx; Y[1::3] *= sy
        Y, E = G.relax(Y, a + ft, b + ft, 600)
        if E < 1e-17:
            X, f = Y, ft
            best = (f, X.copy())
        else:
            step /= 2
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shape', required=True)
    ap.add_argument('--donor', required=True)
    ap.add_argument('--lib', default='poc/faultline_dp2/data/gadlib.json')
    ap.add_argument('--secs', type=float, default=60)
    a = ap.parse_args()
    tg = tuple(int(v) for v in a.shape.split(','))
    dg = tuple(int(v) for v in a.donor.split(','))
    best = load_gadgets(a.lib)
    if dg not in best:
        raise SystemExit(f'donor {dg} not in library')
    f_don, gad = best[dg]
    X = vec(gad['sq'])
    t = tg[0] * tg[1] + tg[2]
    n = X.shape[0] // 3
    print(f'target ({tg}) needs {t} squares; donor ({dg}) has {n} at f={f_don:.6f} '
          f'(n={gad["src"]})')
    e = 0.5 * (np.abs(np.cos(X[2::3])) + np.abs(np.sin(X[2::3])))
    order = np.argsort(X[0::3])
    rng = np.random.default_rng(0)
    t_end = time.time() + a.secs
    bestf = 9.0
    tries = 0
    modes = ['left', 'right', 'rand']
    while time.time() < t_end:
        mode = modes[tries % len(modes)]
        tries += 1
        if mode == 'left':
            keep = order[:t]
        elif mode == 'right':
            keep = order[::-1][:t]
        else:
            # keep a jittered preference plus random deletions
            w = X[0::3] + rng.normal(0, 0.35, n)
            keep = np.argsort(w)[:t]
        Y = np.zeros(3 * t)
        Y[0::3] = X[3 * keep]; Y[1::3] = X[3 * keep + 1]; Y[2::3] = X[3 * keep + 2]
        x0, x1, y0, y1 = bbox(Y)
        f_hi = max(x1 - x0 - tg[0], y1 - y0 - tg[1], 0.35)
        f, Z = squeeze(Y, tg[0], tg[1], t, f_hi)
        if f < bestf:
            bestf = f
            print(f'  {mode}: f={f:.6f} (tries {tries})', flush=True)
            json.dump(dict(f=f, X=Z.tolist(), shape=list(tg), donor=list(dg)),
                      open(f'poc/faultline_dp2/data/squeeze_{tg[0]}_{tg[1]}_{tg[2]}.json', 'w'))
    print(f'({tg}) best f={bestf:.6f} after {tries} tries')


if __name__ == '__main__':
    main()
