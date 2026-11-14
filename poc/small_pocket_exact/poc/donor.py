"""Donor squeeze: take a library pocket of the *same* shape and try to push f
below the pay line.

Only one pay-line shape has a same-shape donor in the current library
(`(13,5,+3) @ 0.705887` from n=585, for shape `(5,13,+3)`), so this is the only
shape where the "squeeze the incumbent's own pocket" route can be tested.

  donor.py --shape 5,13,3 --chains 32 --rounds 60
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, HERE)

import polish                       # noqa: E402
import pslp_np                      # noqa: E402

LIB = os.path.join(ROOT, 'poc', 'faultline_dp2', 'data', 'gadlib.json')
A_ = B_ = None


def get_pocket(w, h, g):
    lib = json.load(open(LIB))
    best = None
    for e in lib:
        if {e['w'], e['h']} == {w, h} and e['g'] == g:
            f = max(e['fx'], e['fy'])
            if best is None or f < best[0]:
                best = (f, np.array(e['sq'], float), e['src'], e['w'], e['h'])
    return best


def _chain(args):
    f, q, rounds, seed = args
    rng = np.random.default_rng(seed)
    best = (f, q.copy())
    for r in range(rounds):
        X = best[1][:, 0].copy(); Y = best[1][:, 1].copy(); T = best[1][:, 2].copy()
        out = pslp_np.shake_squeeze(X, Y, T, A_, B_, best[0], 4, rng)
        qq = np.c_[out[1], out[2], out[3]]
        fp, dep, wall, qt = polish.validate(qq, A_, B_)
        if max(dep, wall) <= 1e-12 and fp < best[0] - 1e-12:
            best = (fp, qt)
    return best[0], best[1]


def main():
    global A_, B_
    ap = argparse.ArgumentParser()
    ap.add_argument('--shape', required=True)
    ap.add_argument('--chains', type=int, default=32)
    ap.add_argument('--rounds', type=int, default=60)
    ap.add_argument('--procs', type=int, default=16)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    w, h, g = (int(v) for v in a.shape.split(','))
    A_, B_ = w, h
    got = get_pocket(w, h, g)
    if got is None:
        print('no donor pocket of shape (%d,%d,+%d) in the library' % (w, h, g))
        return
    f0, q, src, dw, dh = got
    fp, dep, wall, qt = polish.validate(q, w, h)
    print('donor (%d,%d,+%d) from n=%s: stored f=%.9f -> realised f=%.9f (viol %.1e)'
          % (dw, dh, g, src, f0, fp, max(dep, wall)), flush=True)
    d = polish.polish(qt, w, h, max_it=600)
    print('  plain SLP squeeze -> f=%.9f' % d['f'], flush=True)
    best = (d['f'], d['q'])
    t0 = time.time()
    tasks = [(best[0], best[1], a.rounds, 1000 + j) for j in range(a.chains)]
    with Pool(a.procs) as pool:
        res = pool.map(_chain, tasks)
    res.sort(key=lambda r: r[0])
    print('  %d shake chains x %d rounds: best f=%.9f  (%.0fs)'
          % (a.chains, a.rounds, res[0][0], time.time() - t0), flush=True)
    if res[0][0] < best[0]:
        best = (res[0][0], res[0][1])
    print('  BEST donor squeeze f = %.9f' % best[0], flush=True)
    json.dump(dict(shape=[w, h, g], src=src, f_donor=fp, f_best=best[0],
                   q=best[1].tolist()),
              open(a.out or os.path.join(HERE, '..', 'data', 'donor_%d_%d_%d.json'
                                         % (w, h, g)), 'w'))


if __name__ == '__main__':
    main()
