"""Relax the incumbent w=1 chain packing at L0-delta with chain + near-flank squares free."""
import sys, math, numpy as np, pandas as pd, json
sys.path.insert(0, '/Users/lucaszhang/squares-in-squares/poc/k2p10_fill2/poc')
import a_overlap as ov
from scipy.optimize import minimize
INC = '/Users/lucaszhang/squares-in-squares/poc/pell_corrugation/data/border-scan/merged.csv'
def load(n):
    g = pd.read_csv(INC); g = g[g.n == n]
    q = g[['x', 'y', 'angle']].values.copy()
    h = .5 * (abs(np.cos(q[:, 2])) + abs(np.sin(q[:, 2])))
    lo = min((q[:, 0] - h).min(), (q[:, 1] - h).min())
    q[:, :2] -= lo - 0  # walls at 0 ... shift so min edge = 0
    L0 = max((q[:, 0] + h).max(), (q[:, 1] + h).max())
    return q, L0
def setup(q, L0, delta, D):
    v = (q[:, 0] - q[:, 1]) / math.sqrt(2)
    free = np.abs(v) < D
    q2 = q.copy()
    up = ~free
    # lower-right block (x>y): shift x by -delta ; upper-left: shift y by -delta
    m1 = up & (v > 0); m2 = up & (v < 0)
    q2[m1, 0] -= delta; q2[m2, 1] -= delta
    return q2, free
def relax(q0, free, L, rng, jit, it=3000, ww=10.0):
    idx = np.where(free)[0]; nf = len(idx)
    def f(z):
        q = q0.copy(); q[idx] = z.reshape(-1, 3)
        E, G = ov.penalty(q, L, w_wall=ww)
        return E, G[idx].ravel()
    z0 = q0[idx] + rng.normal(0, 1, (nf, 3)) * jit * np.array([1, 1, 3])
    r = minimize(f, z0.ravel(), jac=True, method='L-BFGS-B', options=dict(maxiter=it, ftol=1e-20, gtol=1e-13, maxcor=30))
    q = q0.copy(); q[idx] = r.x.reshape(-1, 3)
    return q, r.fun
if __name__ == '__main__':
    n = int(sys.argv[1]); D = float(sys.argv[2]); starts = int(sys.argv[3]); deltas = list(map(float, sys.argv[4].split(',')))
    q, L0 = load(n); print('L0', L0, 'n', len(q))
    rng = np.random.default_rng(5)
    P = ov.pairs(q); print('incumbent overlap', ov.overlap_pairs(q, 1e-9)[1].max() if len(ov.overlap_pairs(q,1e-9)[1]) else 0)
    for dl in deltas:
        q2, free = setup(q, L0, dl, D)
        best = 1e9
        for s in range(starts):
            qq, E = relax(q2, free, L0 - dl, rng, rng.choice([0.003, 0.01, 0.03]))
            best = min(best, E)
            if E < 1e-14: break
        print(json.dumps(dict(delta=dl, free=int(free.sum()), bestE=best)), flush=True)
