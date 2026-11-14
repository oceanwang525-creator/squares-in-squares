"""Multi-start sequential-LP squeezer ("hammer"): the strongest instrument here.

  1. generate R *feasible* packings of `a*b+g` squares in a loose box a+f0, b+f0
     (batched normalised-GD, `packb`), so every start is inside the feasible set;
  2. shrink each start's box to `f_hit` (isotropic about the bbox centre) and run
     the repo SLP `pslp_np.squeeze` on it -- the SLP explores the infeasible path
     (elastic slack) and returns a locally maximal contraction;
  3. validate with the exact all-pairs SAT check.

This is the same "compress-and-relax + exact polish" strategy as
`poc/gadget_squeeze/poc/squeeze.py`, except that the starts come from a batched
penalty descent (thousands of restarts) instead of hand-built seeds.

usage:
  hammer.py --shape 4,4,1 --f0 0.95 --R 2048 --iters 600 [--hit 0.70]
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

A_ = B_ = HIT_ = None
MAXIT = 400


def _work(q):
    a, b = A_, B_
    q = np.asarray(q, float).copy()
    e = 0.5 * (np.abs(np.cos(q[:, 2])) + np.abs(np.sin(q[:, 2])))
    if HIT_ is not None:
        # isotropic shrink of the current bbox to the target box, then centre it
        w = float((q[:, 0] + e).max() - (q[:, 0] - e).min())
        h = float((q[:, 1] + e).max() - (q[:, 1] - e).min())
        sx = (a + HIT_) / max(w, a + HIT_)
        sy = (b + HIT_) / max(h, b + HIT_)
        if sx < 1 or sy < 1:
            s = max(sx, sy)
            cx = 0.5 * ((q[:, 0] - e).min() + (q[:, 0] + e).max())
            cy = 0.5 * ((q[:, 1] - e).min() + (q[:, 1] + e).max())
            q[:, 0] = cx + (q[:, 0] - cx) * s
            q[:, 1] = cy + (q[:, 1] - cy) * s
    d = polish.polish(q, a, b, max_it=MAXIT)
    return d['f'], d['violation'], d['q']


def main():
    global A_, B_, HIT_, MAXIT
    ap = argparse.ArgumentParser()
    ap.add_argument('--shape', required=True)
    ap.add_argument('--f0', type=float, default=None, help='loose box for the starts')
    ap.add_argument('--hit', type=float, default=None, help='pre-shrink target box')
    ap.add_argument('--R', type=int, default=2048)
    ap.add_argument('--iters', type=int, default=600)
    ap.add_argument('--maxit', type=int, default=400)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--procs', type=int, default=16)
    ap.add_argument('--tag', default=None)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    w, h, g = (int(v) for v in a.shape.split(','))
    A_, B_, HIT_, MAXIT = w, h, g, a.maxit
    N = w * h + g
    f0 = a.f0 if a.f0 is not None else 1.0 + (0.6 if N > 30 else 0.0)
    tag = a.tag or a.shape.replace(',', '_')
    print('=== hammer (%d,%d,+%d) N=%d  starts f=%.3f x %d, hit=%s ==='
          % (w, h, g, N, f0, a.R, a.hit), flush=True)
    t0 = time.time()
    r = packb.Run(w, h, g, np.full(a.R, f0), R=a.R, seed_=a.seed, verbose=False)
    r.relax(a.iters, margin=1.5)
    E = r.E
    ok = E <= 1e-11
    print('  starts: %d/%d feasible at f=%.3f (minE %.2e)  %.0fs'
          % (int(ok.sum()), a.R, f0, E.min(), time.time() - t0), flush=True)
    starts = r.Q[ok] if ok.any() else r.Q[np.argsort(E)[:max(1, a.R // 8)]]
    with Pool(a.procs) as pool:
        res = pool.map(_work, list(starts), chunksize=8)
    res = [x for x in res if x[1] <= 1e-12]
    res.sort(key=lambda x: x[0])
    print('  %d/%d valid squeezes; best f = %s   %.0fs'
          % (len(res), len(starts), ('%.12f' % res[0][0]) if res else 'none',
             time.time() - t0), flush=True)
    if res:
        print('  top: %s' % ' '.join('%.9f' % x[0] for x in res[:10]), flush=True)
        json.dump(dict(shape=[w, h, g], N=N, f0=f0, hit=a.hit, R=a.R, iters=a.iters,
                       f_best=res[0][0], q=res[0][2].tolist(),
                       top=[x[0] for x in res[:20]], violation=res[0][1]),
                  open(a.out or os.path.join(HERE, '..', 'data', 'hammer_%s.json' % tag), 'w'))


if __name__ == '__main__':
    main()
