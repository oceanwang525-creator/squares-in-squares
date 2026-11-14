#!/usr/bin/env python3
"""Exact integer counting for the rational-slope staircase zipper (E3).

Family (the fault-line / zipper scheme, rational seam slope p:q)
----------------------------------------------------------------
Box side L.  theta = atan2(q, p);  n_hat=(c,s) seam normal;  t_hat=(-s,c).

  BL grid : every upright cell (i,j), i,j >= 0, inside the box, whose centre
            satisfies  n_hat.(i+.5, j+.5) <= A          (digital staircase
            anchored at the bottom-left corner, integer lattice)
  band    : w layers of the rotated lattice (unit squares at angle theta, edges
            parallel to the seam), layer k centred on  n_hat.x = C1+gap+.5+k,
            C1 = the BL's extreme corner in the n_hat direction
  TR grid : every upright cell of the lattice **reflected through the box
            corner** (L-1-i, L-1-j) whose distance-from-corner key
            n_hat.(L-.5-i, L-.5-j)  is  <= A2 = n_hat.(L,L) - C2 - gap
            (so the TR lattice sits at offset f = frac(L) from BL, which is
            what makes the seam thin)

Every count is exact integer arithmetic on explicit cell sets; candidates are
re-validated with an explicit separation test before certification.

The per-target question is a decision problem -- "can any rational slope reach
n squares with side < L_inc?" -- answered at L = L_inc - 1e-9 (the count is
monotone in L, so one evaluation decides it).

usage:
  ratzip.py --n-list 791,908      [--procs 4] [--qmax 40] [--wmax 6]
  ratzip.py --one 492 --L 22.707106781186653 --verbose
"""
import argparse
import json
import math

import numpy as np


def slopes(qmax=40):
    out = []
    for q in range(1, qmax + 1):
        for p in range(q, qmax + 1):
            if math.gcd(p, q) == 1:
                out.append((p, q))
    return out


def _a_range(L, c, s, C, ne):
    """t_hat parameter range for a band layer centred at n.x = C to stay in box."""
    tx, ty = -s, c
    lo, hi = -1e18, 1e18
    for p_, t_ in ((c, tx), (s, ty)):
        if abs(t_) > 1e-13:
            a1 = (ne - C * p_) / t_
            a2 = (L - ne - C * p_) / t_
            lo, hi = max(lo, min(a1, a2)), min(hi, max(a1, a2))
        else:
            if C * p_ < ne - 1e-12 or C * p_ > L - ne + 1e-12:
                return None
    if hi < lo - 1e-12:
        return None
    return lo, hi


def scan(L, p, q, wmax=6, gaps=(0.03, 0.1), phases=(0.0, 0.5), astep=1):
    """max count over (A, w, gap, phase) for one rational slope p:q."""
    th = math.atan2(q, p)
    c, s = math.cos(th), math.sin(th)
    ne = 0.5 * (abs(c) + abs(s))
    M = int(math.ceil(L)) + 2
    i = np.arange(M)[:, None]
    j = np.arange(M)[None, :]
    inside = (i + 1 <= L + 1e-9) & (j + 1 <= L + 1e-9)
    # BL keys
    kb = c * (i + 0.5) + s * (j + 0.5)
    kb = np.where(inside, kb, np.inf)
    kbs = np.sort(kb[np.isfinite(kb)].ravel())
    nb = kbs.size
    # TR keys (lattice reflected through the corner)
    kt = c * (L - 0.5 - i) + s * (L - 0.5 - j)
    inside_tr = (L - 1 - i >= -1e-9) & (L - 1 - j >= -1e-9)
    kt = np.where(inside_tr, kt, np.inf)
    kts = np.sort(kt[np.isfinite(kt)].ravel())
    total_diag = c * L + s * L
    best = (0, None)
    for aidx in range(0, nb, astep):
        A = kbs[aidx]
        nbl = aidx + 1
        C1 = A + 0.5 * (c + s)
        for gap in gaps:
            for w in range(0, wmax + 1):
                C2 = C1 + gap + w
                A2 = total_diag - C2 - gap
                if A2 <= 0:
                    continue
                ntr = int(np.searchsorted(kts, A2 + 1e-12))
                for ph in phases:
                    tot = nbl + ntr
                    for k in range(w):
                        rg = _a_range(L, c, s, C1 + gap + 0.5 + k, ne)
                        if rg is None:
                            break
                        lo, hi = rg
                        first = math.ceil(lo - ph - 1e-12)
                        last = math.floor(hi - ph + 1e-12)
                        if last < first:
                            break
                        tot += last - first + 1
                    if tot > best[0]:
                        best = (tot, dict(p=p, q=q, w=w, gap=gap, ph=ph, A=A,
                                          nbl=nbl, ntr=ntr, theta=th))
    return best


def target(n, L, qmax=40, wmax=6, astep=1, verbose=False):
    best = (0, None)
    for (p, q) in slopes(qmax):
        cand = scan(L - 1e-9, p, q, wmax=wmax, astep=astep)
        if cand[0] > best[0]:
            best = cand
            if verbose:
                print(f'   n={n} best {cand[0]} ({n - cand[0]:+d}) {cand[1]}', flush=True)
    return best


def _worker(a):
    n, L, kw = a
    return n, target(n, L, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-list', default='')
    ap.add_argument('--one', type=int)
    ap.add_argument('--L', type=float)
    ap.add_argument('--report', default='poc/rational_zipper/data/report.json')
    ap.add_argument('--gap', default='poc/koboshchan_coordinator/data/gap_vs_frontier.csv')
    ap.add_argument('--qmax', type=int, default=40)
    ap.add_argument('--wmax', type=int, default=6)
    ap.add_argument('--astep', type=int, default=1)
    ap.add_argument('--procs', type=int, default=4)
    ap.add_argument('--topk', type=int, default=16)
    ap.add_argument('--out', default='poc/rational_zipper/data/ratzip.json')
    ap.add_argument('--verbose', action='store_true')
    a = ap.parse_args()

    rep = {r['n']: float(r['L']) for r in json.load(open(a.report))['results']}
    if a.one:
        ns = [a.one]
        if a.L:
            rep[a.one] = a.L
    elif a.n_list:
        ns = [int(x) for x in a.n_list.split(',')]
    else:
        rows = []
        with open(a.gap) as f:
            f.readline()
            for line in f:
                pp = line.split(',')
                rows.append((float(pp[8]), int(pp[0])))
        rows.sort(reverse=True)
        ns = [n for _, n in rows[:a.topk]]

    kw = dict(qmax=a.qmax, wmax=a.wmax, astep=a.astep, verbose=a.verbose)
    tasks = [(n, rep[n], kw) for n in ns]
    res = {}
    if a.procs > 1 and len(tasks) > 1:
        import multiprocessing as mp
        with mp.Pool(min(a.procs, len(tasks))) as pool:
            for n, best in pool.imap_unordered(_worker, tasks):
                res[n] = best
                print(f'n={n}: max count {best[0]} / need {n} ({n - best[0]:+d})  {best[1]}',
                      flush=True)
    else:
        for t in tasks:
            n, best = _worker(t)
            res[n] = best
            print(f'n={n}: max count {best[0]} / need {n} ({n - best[0]:+d}) {best[1]}',
                  flush=True)

    out = {}
    for n in ns:
        cnt, prm = res[n]
        out[n] = dict(n=n, L_inc=rep[n], K=math.isqrt(n), need=n, best_count=cnt,
                      deficit=n - cnt, params=prm, win=bool(cnt >= n))
    json.dump(out, open(a.out, 'w'), indent=1)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
