"""Mass multi-start SLP squeeze: take every feasible state the batched
continuation finds at a loose box, hard-shrink it to `--hit` and let the repo
SLP try to make it feasible.  Used as the last-resort test of the boundary
question for a small shape (does anything beat f = 1/sqrt(2) for (3,2,1)?).

  masssqueeze.py --shape 3,2,1 --f0 0.78 --hit 0.7060 --R 8192
"""
import argparse, json, os, sys, time
from multiprocessing import Pool
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import packb, polish

A_ = B_ = HIT_ = None

def _w(q):
    q = np.asarray(q, float).copy()
    e = 0.5 * (np.abs(np.cos(q[:, 2])) + np.abs(np.sin(q[:, 2])))
    w = float((q[:, 0] + e).max() - (q[:, 0] - e).min())
    h = float((q[:, 1] + e).max() - (q[:, 1] - e).min())
    s = min((A_ + HIT_) / w, (B_ + HIT_) / h)
    if s < 1.0:
        cx = 0.5 * ((q[:, 0] - e).min() + (q[:, 0] + e).max())
        cy = 0.5 * ((q[:, 1] - e).min() + (q[:, 1] + e).max())
        q[:, 0] = cx + (q[:, 0] - cx) * s
        q[:, 1] = cy + (q[:, 1] - cy) * s
    d = polish.polish(q, A_, B_, max_it=300)
    return d['f'], d['violation'], d['q']

def main():
    global A_, B_, HIT_
    ap = argparse.ArgumentParser()
    ap.add_argument('--shape', required=True)
    ap.add_argument('--f0', type=float, default=0.78)
    ap.add_argument('--hit', type=float, required=True)
    ap.add_argument('--R', type=int, default=8192)
    ap.add_argument('--iters', type=int, default=400)
    ap.add_argument('--sweeps', type=int, default=5)
    ap.add_argument('--procs', type=int, default=24)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--tag', default=None)
    a = ap.parse_args()
    w, h, g = (int(v) for v in a.shape.split(','))
    A_, B_, HIT_ = w, h, a.hit
    t0 = time.time()
    r = packb.Run(w, h, g, np.full(a.R, a.f0), R=a.R, seed_=a.seed, verbose=False)
    r.relax(a.iters, margin=1.5)
    pool_states = [q for q in r.Q[r.E <= 1e-11]]
    print('states feasible at f=%.4f: %d (%.0fs)' % (a.f0, len(pool_states), time.time() - t0),
          flush=True)
    if len(pool_states) < a.procs and len(pool_states) == 0:
        return
    with Pool(a.procs) as p:
        res = p.map(_w, pool_states, chunksize=8)
    res = [x for x in res if x[1] <= 1e-12]
    res.sort(key=lambda x: x[0])
    print('%d valid; BEST f = %s  (hit box f=%.6f)  top: %s'
          % (len(res), ('%.12f' % res[0][0]) if res else 'none', a.hit,
             ' '.join('%.9f' % x[0] for x in res[:8])), flush=True)
    if res:
        json.dump(dict(shape=[w, h, g], f0=a.f0, hit=a.hit, f_best=res[0][0],
                       top=[x[0] for x in res[:20]], q=res[0][2].tolist()),
                  open(os.path.join(HERE, '..', 'data', 'mass_%s.json'
                                    % (a.tag or a.shape.replace(',', '_'))), 'w'))

if __name__ == '__main__':
    main()
