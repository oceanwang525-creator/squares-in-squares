"""Periodic-strip model of a 45deg band between two upright staircases, in the band frame
(u along band, v transverse; band squares angle 0, staircase squares angle pi/4).
Outer staircase layer on each side is a rigid lattice (free phase in u); the next staircase
layer (k diamonds/side) and w rows of m band squares are fully free (x,y,angle).
Cell length P=k*sqrt2. Finds the smallest outer-to-outer distance H with zero overlap."""
import sys, math, numpy as np
sys.path.insert(0, '/Users/lucaszhang/squares-in-squares/poc/k2p10_fill2/poc')
import a_overlap as ov
from scipy.optimize import minimize
R2 = math.sqrt(2); PI4 = math.pi / 4

class Cell:
    def __init__(s, k, m, w):
        s.k, s.m, s.w = k, m, w; s.P = k * R2
        s.n = 2 * k + w * m; s.dphi = None
    def outer(s, phb, pht, H):
        i = np.arange(-1, s.k + 1)  # extra copies beyond cell handled by replication
        i = np.arange(s.k)
        b = np.c_[(i + phb) * R2, np.zeros(s.k), np.full(s.k, PI4)]
        t = np.c_[(i + pht) * R2, np.full(s.k, H), np.full(s.k, PI4)]
        return b, t
    def full(s, z, H):
        q = z[:3 * s.n].reshape(-1, 3)
        if s.dphi is not None: z[-1] = z[-2] + s.dphi
        b, t = s.outer(z[-2], z[-1], H)
        base = np.vstack([q, b, t])
        return np.vstack([base + np.array([d * s.P, 0, 0]) for d in (-1, 0, 1)]), len(base)
    def E(s, z, H):
        Q, nb = s.full(z, H)
        E, G = ov.penalty(Q, 1e9, w_wall=0.0)
        G = G.reshape(3, nb, 3).sum(0)
        g = np.zeros_like(z)
        g[:3 * s.n] = G[:s.n].ravel()
        v = z[:3 * s.n].reshape(-1, 3)[:, 1]
        lo = np.maximum(0, 0.6 - v); hi = np.maximum(0, v - (H - 0.6))
        E += float(np.sum(lo ** 2 + hi ** 2))
        g[1:3 * s.n:3] += 2 * (hi - lo)
        k = s.k
        g[-2] = G[s.n:s.n + k, 0].sum(); g[-1] = G[s.n + k:, 0].sum()
        if s.dphi is not None: g[-2] += g[-1]; g[-1] = 0
        g[-2] *= R2; g[-1] *= R2  # phase is in units of sqrt2
        return E, g
    def init(s, H, rng, jit):
        k, m, w = s.k, s.m, s.w
        phb, pht = rng.uniform(0, 1, 2)
        sq = []
        i = np.arange(k)
        # inner staircase layers: half-period offset, v at sqrt2/2 from outer
        sq.append(np.c_[(i + phb + .5) * R2, np.full(k, R2 / 2), np.full(k, PI4)])
        sq.append(np.c_[(i + pht + .5) * R2, np.full(k, H - R2 / 2), np.full(k, PI4)])
        v0 = R2 / 2 + R2 / 2 + 0.5 + (H - (w + 2 * R2)) / 2
        for r in range(w):
            u0 = rng.uniform(0, s.P)
            j = np.arange(m)
            sq.append(np.c_[u0 + (j + .5) * s.P / m, np.full(m, v0 + r), np.zeros(m)])
        q = np.vstack(sq)
        q[:, :2] += rng.normal(0, jit, (len(q), 2)); q[:, 2] += rng.normal(0, jit, len(q))
        return np.r_[q.ravel(), phb, pht]
    def minimize(s, z, H, it=400):
        r = minimize(s.E, z, args=(H,), jac=True, method='L-BFGS-B', options=dict(maxiter=it, ftol=1e-18, gtol=1e-12))
        return r.x, r.fun
    def waste(s, H):
        return (H + R2 / 2) * s.P - (4 * s.k + s.w * s.m)
