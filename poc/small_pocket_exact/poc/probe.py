"""Fixed-box feasibility probe: how jammed is `(a+f) x (b+f)` at a given f?

All R restarts sit in exactly the same box; the batched optimiser minimises the
SAT penalty.  min E == 0 means a feasible packing was found (the box is
reachable); a positive floor means every restart jammed -- the *size* of that
floor is the jam depth (calibrated against known records, see README).

  probe.py --shape 3,2,1 --f 0.72 0.71 0.7072 0.70625 0.70 --R 4096 --iters 1200
  probe.py --shape 3,2,1 --bisect 0.70 0.72
"""
import argparse
import sys
import time

import numpy as np

import os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import packb                        # noqa: E402


def probe(a, b, g, f, R, iters, seed, verbose=False):
    r = packb.Run(a, b, g, np.full(R, float(f)), R=R, seed_=seed, verbose=verbose)
    r.relax(iters, margin=1.5)
    return float(np.min(r.E)), r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shape', required=True)
    ap.add_argument('--f', nargs='*', type=float, default=[])
    ap.add_argument('--bisect', nargs=2, type=float, default=None)
    ap.add_argument('--steps', type=int, default=12)
    ap.add_argument('--R', type=int, default=4096)
    ap.add_argument('--iters', type=int, default=1200)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--tol', type=float, default=1e-9)
    a = ap.parse_args()
    w, h, g = (int(v) for v in a.shape.split(','))
    for f in a.f:
        t0 = time.time()
        E, _ = probe(w, h, g, f, a.R, a.iters, a.seed)
        print('(%d,%d,+%d) f=%.9f  minE %.3e  %s  (%.0fs)'
              % (w, h, g, f, E, 'FEASIBLE' if E <= a.tol else 'jam', time.time() - t0),
              flush=True)
    if a.bisect:
        lo, hi = a.bisect
        Elo, _ = probe(w, h, g, lo, a.R, a.iters, a.seed)
        Ehi, _ = probe(w, h, g, hi, a.R, a.iters, a.seed)
        print('bracket: f=%.6f -> %.3e | f=%.6f -> %.3e' % (lo, Elo, hi, Ehi), flush=True)
        best = lo if Elo <= a.tol else None
        for _ in range(a.steps):
            mid = 0.5 * (lo + hi)
            Em, _ = probe(w, h, g, mid, a.R, a.iters, a.seed)
            print('  f=%.9f -> %.3e %s' % (mid, Em, 'FEAS' if Em <= a.tol else 'jam'),
                  flush=True)
            if Em <= a.tol:
                lo, best = mid, mid
            else:
                hi = mid
        print('deepest feasible f from probe: %s (bracket %.6f .. %.6f)'
              % (('%.9f' % best) if best is not None else 'none', lo, hi), flush=True)


if __name__ == '__main__':
    main()
