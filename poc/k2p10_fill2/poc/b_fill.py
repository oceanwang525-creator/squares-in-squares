"""Approach 3: n-2 (633) staircase-band core + 2 extra squares in pockets, minimise total overlap with all squares free.
usage: b_fill.py cfg_index Lc trials seed [secs]   (cfg_index into the list of 633-count 45deg configs)"""
import sys, time, math, numpy as np
from scipy.optimize import minimize
sys.path.insert(0, "/Users/lucaszhang/squares-in-squares/poc/k2p10_fill2/poc")
sys.path.insert(0, "/Users/lucaszhang/squares-in-squares/poc/k2p10_band_search/poc")
sys.path.insert(0, "/Users/lucaszhang/squares-in-squares/poc/k2p10_out_of_phase/poc")
import b_overlap as bo
from build import build

DOCS = "/Users/lucaszhang/squares-in-squares/poc/k2p10_fill2/docs/"


CFGS = [  # (t, w, a1, a2, b1, b2, ell): 633-square 45deg staircase-band cores
    (1, 8, 18, 18, 6, 6, 25),   # L45 25.65685 (needs Lc >= ~25.69 for 25 columns)
    (1, 13, 15, 14, 9, 10, 20),  # L45 25.69239
    (1, 25, 6, 6, 18, 18, 8),   # L45 25.67767
    (1, 10, 18, 17, 6, 6, 25),
]


def configs():
    return CFGS


def core_of(cfg, K=25):
    t, w, a1, a2, b1, b2, l = cfg
    q, L45 = build(t, w, a1, a2, b1, b2, l, K)
    return q.copy(), L45


def find_pockets(P, Lc, rng, ngrid=0.2):
    """Score free spots for an extra unit square: min over (x,y,angle in 0,45,rand) of penetration energy with P."""
    cand = []
    xs = np.arange(0.5, Lc - 0.5 + 1e-9, ngrid)
    tree = None
    from scipy.spatial import cKDTree
    tree = cKDTree(P[:, :2])
    for ang in (0.0, math.pi / 4):
        e = 0.5 * (abs(math.cos(ang)) + abs(math.sin(ang)))
        gx = xs[(xs >= e) & (xs <= Lc - e)]
        for x in gx:
            for y in gx:
                idx = tree.query_ball_point([x, y], bo.R2)
                if not idx: cand.append((0.0, x, y, ang)); continue
                Q = np.vstack([P[idx], [x, y, ang]])
                I = np.full(len(idx), len(idx)); J = np.arange(len(idx))
                g = bo.pair_gap(Q, I, J)
                d = np.clip(-g, 0, None)
                cand.append((float((d ** 2).sum()), x, y, ang))
    cand.sort()
    return cand


def pick(cand, rng, k=2, topn=60, mind=1.5):
    top = cand[:topn]
    chosen = []
    order = rng.permutation(len(top))
    for i in order:
        c = top[i]
        if all(math.hypot(c[1] - d[1], c[2] - d[2]) > mind for d in chosen):
            chosen.append(c)
        if len(chosen) == k: break
    return chosen


def relax(P, Lc, maxiter=3000):
    n = len(P)
    def f(v):
        E, g, _ = bo.energy(v.reshape(n, 3), Lc)
        return E, g.ravel()
    r = minimize(f, P.ravel(), jac=True, method="L-BFGS-B", options=dict(maxiter=maxiter, maxcor=30, ftol=1e-30, gtol=1e-13))
    return r.x.reshape(n, 3), r.fun


def run(cfg_i, Lc, trials, seed, secs=1e9, cfgs=None):
    cfgs = cfgs or configs()
    cfg = cfgs[cfg_i]
    core, L45 = core_of(cfg)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    best = (1e9, None, None)
    for tr in range(trials):
        if time.time() - t0 > secs: break
        # spread the slack by uniform dilation of the core centres, optionally jittered
        P0 = core.copy()
        P0[:, :2] *= Lc / L45
        P0[:, :2] += rng.normal(0, 0.003 * (tr % 3), P0[:, :2].shape)
        cand = find_pockets(P0, Lc, rng)
        ch = pick(cand, rng)
        ext = np.array([[c[1], c[2], c[3]] for c in ch])
        P = np.vstack([P0, ext])
        # penalty hardening via progressive box: start with enlarged box then shrink to Lc
        for Lb in (Lc + 0.15, Lc + 0.05, Lc):
            P, E = relax(P, Lb, 5000 if Lb == Lc else 1500)
        _, _, info = bo.energy(P, Lc)
        print(f"cfg{cfg_i} {cfg} Lc={Lc} tr{tr} pockets={[(round(c[1],2),round(c[2],2),round(math.degrees(c[3]))) for c in ch]} "
              f"E={E:.3e} maxd={info['maxd']:.4f} npairs={info['npairs']} t={time.time()-t0:.0f}s", flush=True)
        if E < best[0]: best = (E, P.copy(), info)
    return best


if __name__ == "__main__":
    ci, Lc, trials, seed = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    secs = float(sys.argv[5]) if len(sys.argv) > 5 else 1e9
    E, P, info = run(ci, Lc, trials, seed, secs)
    print("BEST", ci, Lc, E, info)
    np.save(DOCS + f"b_best_cfg{ci}_L{Lc}_s{seed}.npy", P)
