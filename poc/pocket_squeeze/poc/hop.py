"""Basin hopping on pocket types: shake (global or local) + SLP re-squeeze, keep strict improvements.
usage: hop.py gadlib.json slack0.json out_dir rounds max_N [min_N]
"""
import json, os, sys, time
import numpy as np
from multiprocessing import Pool
from pslp import squeeze, viol, box_of


def hop(args):
    key, w, h, g, f, sq, rounds, seed, out_dir = args
    rng = np.random.default_rng(seed)
    a = np.array(sq); n = len(a)
    best = (float(f), a[:, 0].copy(), a[:, 1].copy(), a[:, 2].copy())
    f_start = best[0]; t0 = time.time(); hits = 0
    for r in range(rounds):
        X, Y, T = best[1].copy(), best[2].copy(), best[3].copy()
        mode = rng.random()
        if mode < 0.5:                                   # global shake
            sig = rng.choice([0.002, 0.005, 0.01, 0.02]); sc = rng.choice([1.0, 1.005, 1.01, 1.02])
            cx = (w + best[0]) / 2; cy = (h + best[0]) / 2
            X = cx + (X - cx) * sc + rng.normal(0, sig, n); Y = cy + (Y - cy) * sc + rng.normal(0, sig, n)
            T = T + rng.normal(0, sig, n)
        else:                                            # local shake around a random square
            c = rng.integers(n); rad = rng.choice([1.5, 2.5, 4.0]); sig = rng.choice([0.01, 0.03, 0.06, 0.1])
            m = np.hypot(X - X[c], Y - Y[c]) < rad
            X[m] += rng.normal(0, sig, m.sum()); Y[m] += rng.normal(0, sig, m.sum()); T[m] += rng.normal(0, sig, m.sum())
            sc = rng.choice([1.0, 1.003, 1.01])
            cx = (w + best[0]) / 2; cy = (h + best[0]) / 2
            X = cx + (X - cx) * sc; Y = cy + (Y - cy) * sc
        x0, x1, y0, y1 = box_of(X, Y, T)
        X -= min(x0, 0.0); Y -= min(y0, 0.0)
        x0, x1, y0, y1 = box_of(X, Y, T)
        f0 = max(x1 - w, y1 - h, best[0])
        try:
            fn, Xn, Yn, Tn, vn, it = squeeze(X, Y, T, w, h, f0)
        except Exception:
            continue
        if vn <= 1e-12 and fn < best[0] - 1e-10:
            best = (fn, Xn, Yn, Tn); hits += 1
    res = dict(key=key, w=w, h=h, g=g, f0=f_start, f=best[0], hits=hits, secs=time.time() - t0,
               sq=np.column_stack(best[1:]).tolist())
    json.dump(res, open(f'{out_dir}/{key}.json', 'w'))
    return key, n, f_start, best[0], hits, time.time() - t0


if __name__ == '__main__':
    lib = json.load(open(sys.argv[1])); sl = json.load(open(sys.argv[2])); out_dir = sys.argv[3]
    rounds = int(sys.argv[4]); maxN = int(sys.argv[5]); minN = int(sys.argv[6]) if len(sys.argv) > 6 else 0
    os.makedirs(out_dir, exist_ok=True)
    best = {}
    for i, o in sl.items():
        k = (min(o['w'], o['h']), max(o['w'], o['h']), o['g'])
        if o['v'] <= 1e-12 and (k not in best or o['f'] < best[k]['f']):
            best[k] = o
    jobs = []
    for k, o in sorted(best.items(), key=lambda kv: len(kv[1]['sq'])):
        N = len(o['sq'])
        if N > maxN or N < minN: continue
        key = f'{o["w"]}x{o["h"]}g{o["g"]}'
        if os.path.exists(f'{out_dir}/{key}.json'): continue
        jobs.append((key, o['w'], o['h'], o['g'], o['f'], o['sq'], rounds, hash(k) % 10**6, out_dir))
    print(len(jobs), 'pocket types queued', flush=True)
    with Pool(7) as pool:
        for key, n, f0, f, hits, secs in pool.imap_unordered(hop, jobs):
            print(time.strftime('%H:%M:%S'), f'{key} N={n}: f {f0:.8f} -> {f:.8f} ({f0-f:+.2e}) improvements={hits} {secs:.0f}s', flush=True)
