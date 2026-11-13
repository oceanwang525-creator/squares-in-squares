"""Pocket factory for one (w, h): rigid scan -> squeeze the seeds with enough gain -> report best f per gain.
usage: factory.py w h gmin max_seeds out.json [th0 th1 dth]
"""
import json, math, os, sys, time
import numpy as np
import numba as nb
from multiprocessing import Pool
from seedgen import count_seed, make_seed
from pslp import squeeze


@nb.njit(cache=True)
def collect(w, h, f0s, ths, Rs, As, phases, offs, gmin, cap):
    out = np.empty((cap, 10)); m = 0; tot = 0
    for fi in range(len(f0s)):
        W = w + f0s[fi]; H = h + f0s[fi]
        for ti in range(len(ths)):
            for R in Rs:
                for a in As:
                    for ph in phases:
                        for al in range(5):
                            for ox in offs:
                                for oy in offs:
                                    n, nt = count_seed(W, H, ths[ti], W / 2 + ox, H / 2 + oy, a, R, ph, al, 0.0)
                                    tot += 1
                                    if n - w * h >= gmin and nt >= 6 and m < cap:
                                        out[m, 0] = f0s[fi]; out[m, 1] = ths[ti]; out[m, 2] = R; out[m, 3] = a; out[m, 4] = ph
                                        out[m, 5] = al; out[m, 6] = ox; out[m, 7] = oy; out[m, 8] = n; out[m, 9] = nt
                                        m += 1
    return out[:m], tot


def work(args):
    w, h, p = args
    f0, th, R, a, ph, al, ox, oy = p[:8]
    W = w + f0; H = h + f0
    P, nt = make_seed(W, H, th, W / 2 + ox, H / 2 + oy, a, int(R), ph, int(al), 0.125)
    try:
        f, X, Y, T, v, it = squeeze(P[:, 0].copy(), P[:, 1].copy(), P[:, 2].copy(), w, h, f0)
    except Exception:
        return None
    if v > 1e-12:
        return None
    return len(P) - w * h, f, [float(q) for q in p[:8]], np.column_stack([X, Y, T]).tolist()


if __name__ == '__main__':
    w, h, gmin, max_seeds, out_p = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    th0, th1, dth = (float(sys.argv[6]), float(sys.argv[7]), float(sys.argv[8])) if len(sys.argv) > 8 else (15.0, 75.0, 1.0)
    f0s = np.array([0.80, 0.85, 0.90, 0.95])
    ths = np.radians(np.arange(th0, th1 + 1e-9, dth))
    Rs = np.arange(3, max(w, h) + 2)
    As = np.arange(1.5, max(w, h) / 2 + 1.01, 0.5)
    t = time.time()
    collect(w, h, f0s[:1], ths[:1], Rs[:1], As[:1], np.array([0.0]), np.array([0.0]), 99, 4)
    S, tot = collect(w, h, f0s, ths, Rs, As, np.array([0.0, 0.25, 0.5, 0.75]), np.array([-0.5, 0.0, 0.5]), gmin, 2000000)
    print(f'({w},{h}): {tot} rigid seeds scanned in {time.time()-t:.0f}s; {len(S)} with gain >= {gmin}', flush=True)
    # dedupe by (count, tilted count, f0, theta, R) then sample
    rng = np.random.default_rng(w * 100 + h)
    key = {}
    for i in rng.permutation(len(S)):
        k = (S[i, 0], round(S[i, 1], 4), S[i, 2], S[i, 8], S[i, 9], S[i, 5])
        if k not in key: key[k] = i
    idx = list(key.values())
    if len(idx) > max_seeds:
        # prefer higher gain and tighter f0
        idx.sort(key=lambda i: (-(S[i, 8] - w * h) + 8 * S[i, 0], rng.random()))
        idx = idx[:max_seeds]
    print(f'  {len(key)} distinct signatures; squeezing {len(idx)}', flush=True)
    best = {}; allres = []
    t = time.time()
    with Pool(7) as pool:
        for r in pool.imap_unordered(work, [(w, h, S[i]) for i in idx], chunksize=4):
            if r is None: continue
            g, f, p, sq = r
            allres.append((g, f))
            if g not in best or f < best[g]['f']:
                best[g] = dict(w=w, h=h, g=g, f=f, par=p, sq=sq, src=f'factory:{w}x{h}')
    json.dump(list(best.values()), open(out_p, 'w'))
    print(f'  squeezed {len(allres)} in {time.time()-t:.0f}s')
    for g in sorted(best):
        fs = sorted(f for gg, f in allres if gg == g)
        print(f'  gain {g:+d}: {len(fs)} seeds; best f={fs[0]:.6f}; 5th={fs[min(4,len(fs)-1)]:.4f}; median={fs[len(fs)//2]:.4f}; best params th={math.degrees(best[g]["par"][1]):.1f} R={int(best[g]["par"][2])} a={best[g]["par"][3]} f0={best[g]["par"][0]}')
