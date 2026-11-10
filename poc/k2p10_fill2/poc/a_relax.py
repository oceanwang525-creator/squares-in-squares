"""Build the 633 band config, insert 2 squares, relax with L-BFGS penalty. python a_relax.py L ntrials seed"""
import sys, math, time, numpy as np
from scipy.optimize import minimize
sys.path.insert(0, "/Users/lucaszhang/squares-in-squares/poc/k2p10_out_of_phase/poc")
sys.path.insert(0, "/Users/lucaszhang/squares-in-squares/poc/k2p10_fill2/poc")
from build import build
from a_overlap import penalty, pairs, max_violation, pair_depth

def base633():
    q, L45 = build(1, 8, 18, 18, 6, 6, 25, 25)
    return q, L45

def relax(q0, L, iters=400, rounds=3, fix=None):
    q = q0.copy(); N = len(q)
    for r in range(rounds):
        P = pairs(q, margin=0.6)
        def f(v):
            qq = v.reshape(N, 3)
            E, G = penalty(qq, L, P=P)
            return E, G.ravel()
        res = minimize(f, q.ravel(), jac=True, method="L-BFGS-B", options=dict(maxiter=iters, ftol=1e-18, gtol=1e-12))
        q = res.x.reshape(N, 3)
    return q, res.fun

def free_candidates(q, L, step=0.1):
    """Score grid points (upright & 45deg) by clearance = -max depth against neighbours; return best list."""
    from scipy.spatial import cKDTree
    out = []
    for ang in (0.0, math.pi / 4):
        xs = np.arange(0.5, L - 0.5 + 1e-9, step)
        X, Y = np.meshgrid(xs, xs); X = X.ravel(); Y = Y.ravel()
        t = cKDTree(q[:, :2])
        nb = t.query_ball_point(np.c_[X, Y], 1.45)
        best = np.full(len(X), -9.0)
        for k, lst in enumerate(nb):
            if not lst: best[k] = 1.0; continue
            qq = np.vstack([[[X[k], Y[k], ang]], q[lst]])
            P = np.c_[np.zeros(len(lst), int), np.arange(1, len(lst) + 1)]
            best[k] = -pair_depth(qq, P).max()  # clearance (positive = free)
        for k in np.argsort(-best)[:2000]: out.append((best[k], X[k], Y[k], ang))
    out.sort(reverse=True)
    return out

if __name__ == "__main__":
    L = float(sys.argv[1]); ntr = int(sys.argv[2]); seed = int(sys.argv[3])
    q45, L45 = base633(); print("base", len(q45), "L45", L45)
    q = q45.copy(); q[:, :2] *= L / L45 # dilation opens gaps
    print("base@L overlap", max_violation(q, L))
    cand = free_candidates(q, L)
    print("best clearances", [round(c[0], 4) for c in cand[:5]])
    rng = np.random.default_rng(seed); best = (9, None)
    t0 = time.time()
    for tr in range(ntr):
        # pick 2 of the top candidates (distinct places) at random
        top = cand[:400]; i = rng.integers(len(top)); j = rng.integers(len(top))
        a, b = top[i], top[j]
        qa = np.vstack([q, [a[1], a[2], a[3]], [b[1], b[2], b[3]]])
        qr, E = relax(qa, L)
        mv = max_violation(qr, L)
        if E < best[0]: best = (E, qr)
        print(tr, "E", E, "maxdepth,wall,npairs", mv, round(time.time() - t0, 1), flush=True)
        if time.time() - t0 > 240: break
    np.save(f"/Users/lucaszhang/squares-in-squares/poc/k2p10_fill2/docs/a_best_relax_L{L}.npy", best[1])
