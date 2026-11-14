"""Drive the search for one shape: `a*b+g` unit squares in `(a+f) x (b+f)`.

Pipeline
  1. CONTINUATION (`packb.Run`): R restarts, each with its own box excess f_r,
     batched normalised-GD on the repo SAT penalty, per-restart shrink /
     restore / kick.  Emits the deepest feasible states (its archive).
  2. POLISH POOL: every deep state is run through the repo sequential-LP squeeze
     (`pslp_np.squeeze`) in a multiprocessing pool, then re-validated with the
     exact all-pairs SAT check.
  3. BASIN HOP: `pslp_np.shake_squeeze` (perturb + re-squeeze, strict
     improvements only) on the best few, in parallel over processes.
  4. Fixed-f probe at the pay line: min penalty energy reached by a fresh
     restart ensemble in exactly the target box (a jam diagnostic).

usage:
  search.py --shape 4,7,3 --fpay 0.777734 --R 1024 --sweeps 30 --iters 80 \
            --npolish 64 --hop 16 --tag s4_7_3
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import packb                        # noqa: E402
import polish                       # noqa: E402
import pslp_np                      # noqa: E402
import satb                         # noqa: E402

A_ = B_ = None


def _pol(args):
    f, q = args
    d = polish.polish(q, A_, B_, max_it=400)
    return d['f'], d['violation'], d['q']


def _hop(args):
    f, q, rounds, seed = args
    rng = np.random.default_rng(seed)
    best = (f, q)
    for _ in range(rounds):
        X = best[1][:, 0].copy(); Y = best[1][:, 1].copy(); T = best[1][:, 2].copy()
        out = pslp_np.shake_squeeze(X, Y, T, A_, B_, best[0], 6, rng)
        qq = np.c_[out[1], out[2], out[3]]
        fp, dep, wall, qt = polish.validate(qq, A_, B_)
        if max(dep, wall) <= 1e-12 and fp < best[0] - 1e-12:
            best = (fp, qt)
    return best[0], best[1]


def main():
    global A_, B_
    ap = argparse.ArgumentParser()
    ap.add_argument('--shape', required=True)
    ap.add_argument('--fpay', type=float, required=True)
    ap.add_argument('--R', type=int, default=1024)
    ap.add_argument('--sweeps', type=int, default=30)
    ap.add_argument('--iters', type=int, default=80)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--tag', default=None)
    ap.add_argument('--npolish', type=int, default=64)
    ap.add_argument('--hop', type=int, default=16, help='total hop rounds, 0=off')
    ap.add_argument('--hopk', type=int, default=4, help='best k states to hop')
    ap.add_argument('--procs', type=int, default=8)
    ap.add_argument('--fstart', type=float, default=None)
    ap.add_argument('--tlimit', type=float, default=None)
    ap.add_argument('--probeR', type=int, default=1024)
    ap.add_argument('--probeiters', type=int, default=800)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    w, h, g = (int(v) for v in a.shape.split(','))
    A_, B_ = w, h
    N = w * h + g
    tag = a.tag or a.shape.replace(',', '_')
    print('=== shape (%d,%d,+%d)  N=%d  pay line f<=%.6f ===' % (w, h, g, N, a.fpay),
          flush=True)
    t0 = time.time()

    # ---- 1. continuation
    rng = np.random.default_rng(a.seed + 991)
    if a.fstart is not None:
        base = a.fstart
    else:
        # start where the seed family is (nearly) exact: the a x b lattice with
        # f = 1 (grid + one-square strip) is feasible for every (a,b,g), so the
        # continuation must begin at/above it or the seeds start jammed.
        base = max(a.fpay + 0.12, 1.0 if N >= 25 else 0.82)
    f0 = base + rng.uniform(0.0, 0.30, a.R)
    run = packb.Run(w, h, g, f0, R=a.R, seed_=a.seed, verbose=True)
    for sw in range(a.sweeps):
        run.sweep(a.iters, sw)
        if a.tlimit is not None and time.time() - t0 > a.tlimit * 0.55:
            print('  (time limit, stop at sweep %d)' % sw, flush=True)
            break
    cands = run.candidates(max(a.npolish, 8))
    print('  continuation: deepest f = %.9f  (%d feasible restarts, %d candidates, %.0fs)'
          % (cands[0][0], int(np.isfinite(run.best_f).sum()), len(cands), time.time() - t0),
          flush=True)

    # ---- 2. polish pool
    best = None
    with Pool(min(a.procs, max(1, len(cands)))) as pool:
        res = pool.map(_pol, cands)
    res = [r for r in res if r[1] <= 1e-12]
    res.sort(key=lambda r: r[0])
    print('  polish: %d/%d valid, best f = %s  (%.0fs)'
          % (len(res), len(cands), ('%.9f' % res[0][0]) if res else 'none', time.time() - t0),
          flush=True)
    if res:
        best = (res[0][0], res[0][2])

    # ---- 3. basin hop
    if a.hop > 0 and res:
        seeds = [(f, q, max(1, a.hop // min(a.hopk, len(res))), a.seed + 17 * j)
                 for j, (f, q, _) in enumerate([(r[0], r[2], 0) for r in res[:a.hopk]])]
        with Pool(min(a.procs, len(seeds))) as pool:
            hr = pool.map(_hop, seeds)
        hr.sort(key=lambda r: r[0])
        print('  hop: best f = %.9f  (%.0fs)' % (hr[0][0], time.time() - t0), flush=True)
        if hr[0][0] < best[0]:
            best = (hr[0][0], hr[0][1])

    if best is not None:
        fp, dep, wall, qt = polish.validate(best[1], w, h)
        print('  BEST validated f = %.9f  violation %.1e  (pay line %.6f, %+.3e)  %.0fs'
              % (fp, max(dep, wall), a.fpay, fp - a.fpay, time.time() - t0), flush=True)
        best = (fp, qt)
    else:
        print('  no validated packing found', flush=True)

    # ---- 4. jam probe at the pay line
    pr = packb.Run(w, h, g, np.full(a.probeR, a.fpay), R=a.probeR, seed_=a.seed + 7,
                   verbose=False)
    pr.relax(a.probeiters, margin=1.5)
    emin = float(np.min(pr.E))
    print('  fixed-f probe at f=%.6f: min penalty energy %.3e  (0 = feasible)'
          % (a.fpay, emin), flush=True)

    res_d = dict(shape=[w, h, g], N=N, fpay=a.fpay,
                 f_best=(best[0] if best else None),
                 f_cont=float(cands[0][0]), probe_emin=emin,
                 pol_best=[float(r[0]) for r in res[:8]],
                 R=a.R, sweeps=a.sweeps, iters=a.iters, npolish=a.npolish,
                 hop=a.hop, seed=a.seed, secs=time.time() - t0)
    if best:
        res_d['q'] = best[1].tolist()
        res_d['violation'] = float(max(dep, wall))
    path = a.out or os.path.join(HERE, '..', 'data', 'shape_%s.json' % tag)
    json.dump(res_d, open(path, 'w'))
    print('  -> %s  (%.0fs)' % (os.path.normpath(path), time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
