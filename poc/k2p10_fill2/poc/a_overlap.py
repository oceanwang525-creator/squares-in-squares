"""Vectorised unit-square overlap/penalty module (numpy SAT + cKDTree neighbours).
q: (N,3) array of (x, y, angle) for unit squares centred at (x,y). Container [0,L]^2.
"""
import math, numpy as np
from scipy.spatial import cKDTree

def pairs(q, margin=0.0):
    """Candidate pairs (i<j) whose centres are within sqrt2+margin."""
    t = cKDTree(q[:, :2])
    p = t.query_pairs(math.sqrt(2) + margin + 1e-9, output_type="ndarray")
    return p

def _r(psi):
    c, s = np.cos(psi), np.sin(psi)
    return 0.5 * (np.abs(c) + np.abs(s))

def pair_depth(q, P, grad=False):
    """Penetration depth per pair (<=0 means separated; value is max gap-ish: min over axes of d).
    With grad=True also returns (axis index k, dd/dpi (m,2), dd/dpj (m,2), dd/dti, dd/dtj) for the minimising axis."""
    i, j = P[:, 0], P[:, 1]
    D = q[j, :2] - q[i, :2]
    ti, tj = q[i, 2], q[j, 2]
    al = np.stack([ti, ti + math.pi / 2, tj, tj + math.pi / 2], 1)  # m,4
    ux, uy = np.cos(al), np.sin(al)
    dot = D[:, 0:1] * ux + D[:, 1:2] * uy
    d = _r(al - ti[:, None]) + _r(al - tj[:, None]) - np.abs(dot)
    k = np.argmin(d, 1)
    dep = d[np.arange(len(P)), k]
    if not grad:
        return dep
    m = np.arange(len(P))
    a = al[m, k]; s = np.sign(dot[m, k]); s[s == 0] = 1.0
    ux_, uy_ = np.cos(a), np.sin(a)
    gpj = np.stack([-s * ux_, -s * uy_], 1)
    gpi = -gpj
    dDu = -s * (-D[:, 0] * uy_ + D[:, 1] * ux_)  # d(-|D.u|)/dalpha
    fromi = k < 2
    # r of the other square as function of psi = a - theta_other
    psi_o = np.where(fromi, a - tj, a - ti)
    c, sn = np.cos(psi_o), np.sin(psi_o)
    dr = 0.5 * (-np.sign(c) * sn + np.sign(sn) * c)
    dti = np.where(fromi, dr + dDu, -dr)
    dtj = np.where(fromi, -dr, dr + dDu)
    return dep, gpi, gpj, dti, dtj

def overlap_pairs(q, tol=0.0):
    """Return (pairs, depths) of overlapping pairs with depth > tol."""
    P = pairs(q)
    if len(P) == 0: return P, np.zeros(0)
    dep = pair_depth(q, P)
    m = dep > tol
    return P[m], dep[m]

def wall_violation(q, L):
    h = 0.5 * (np.abs(np.cos(q[:, 2])) + np.abs(np.sin(q[:, 2])))
    lo = np.maximum(0, h[:, None] - q[:, :2]); hi = np.maximum(0, q[:, :2] + h[:, None] - L)
    return lo + hi, h

def penalty(q, L, P=None, w_wall=1.0, grad=True, margin=0.0):
    """E = sum_pairs max(depth+margin,0)^2 + w_wall*sum wall_viol^2; analytic gradient wrt q (N,3)."""
    N = len(q)
    if P is None: P = pairs(q)
    E = 0.0; G = np.zeros((N, 3))
    if len(P):
        out = pair_depth(q, P, grad=grad)
        dep = out[0] if grad else out
        m = dep + margin > 0
        if grad: _, gpi, gpj, dti, dtj = out
        e = dep[m] + margin
        E += float(np.sum(e * e))
        if grad:
            Pm = P[m]; f = 2 * e
            np.add.at(G[:, 0], Pm[:, 0], f * gpi[m, 0]); np.add.at(G[:, 1], Pm[:, 0], f * gpi[m, 1])
            np.add.at(G[:, 0], Pm[:, 1], f * gpj[m, 0]); np.add.at(G[:, 1], Pm[:, 1], f * gpj[m, 1])
            np.add.at(G[:, 2], Pm[:, 0], f * dti[m]); np.add.at(G[:, 2], Pm[:, 1], f * dtj[m])
    h = 0.5 * (np.abs(np.cos(q[:, 2])) + np.abs(np.sin(q[:, 2])))
    lo = np.maximum(0, h[:, None] - q[:, :2]); hi = np.maximum(0, q[:, :2] + h[:, None] - L)
    E += w_wall * float(np.sum(lo * lo + hi * hi))
    if grad:
        G[:, :2] += w_wall * 2 * (hi - lo)
        dh = 0.5 * (-np.sign(np.cos(q[:, 2])) * np.sin(q[:, 2]) + np.sign(np.sin(q[:, 2])) * np.cos(q[:, 2]))
        G[:, 2] += w_wall * 2 * dh * (np.sum(lo, 1) + np.sum(hi, 1))
        return E, G
    return E

def max_violation(q, L):
    P, dep = overlap_pairs(q)
    wv, _ = wall_violation(q, L)
    return (dep.max() if len(dep) else 0.0), float(wv.max()), len(P)

def bbox_side(q):
    h = 0.5 * (np.abs(np.cos(q[:, 2])) + np.abs(np.sin(q[:, 2])))
    return max((q[:, 0] + h).max() - (q[:, 0] - h).min(), (q[:, 1] + h).max() - (q[:, 1] - h).min())

# ---- exact scalar reference (independent: corner projection SAT) ----
def corners(x, y, a):
    c, s = math.cos(a), math.sin(a)
    return [(x + c * dx - s * dy, y + s * dx + c * dy) for dx, dy in ((-.5, -.5), (.5, -.5), (.5, .5), (-.5, .5))]

def ref_depth(a, b):
    ca, cb = corners(*a), corners(*b)
    best = 1e9
    for t in (a[2], a[2] + math.pi / 2, b[2], b[2] + math.pi / 2):
        u = (math.cos(t), math.sin(t))
        pa = [p[0] * u[0] + p[1] * u[1] for p in ca]; pb = [p[0] * u[0] + p[1] * u[1] for p in cb]
        best = min(best, min(max(pa) - min(pb), max(pb) - min(pa)))
    return best

if __name__ == "__main__":
    import sys, time
    rng = np.random.default_rng(1)
    # random pair test: depth vs reference
    q = np.c_[rng.uniform(0, 6, (300, 2)), rng.uniform(0, math.pi, 300)]
    P = pairs(q); dep = pair_depth(q, P)
    allp = np.array([(i, j) for i in range(300) for j in range(i + 1, 300)])
    # far pairs are non-overlapping by construction; check only candidates
    ref = np.array([ref_depth(q[i], q[j]) for i, j in P])
    print("pairs", len(P), "max |depth-ref|", np.abs(dep - ref).max())
    ov_ref = {(i, j) for i, j in allp if ref_depth(q[i], q[j]) > 1e-12}
    ov = {tuple(p) for p in P[dep > 1e-12]}
    print("overlap sets equal (vs brute force all pairs):", ov == ov_ref, len(ov))
    # gradient check
    E, G = penalty(q, 6.0)
    Gn = np.zeros_like(G); e = 1e-6
    for n in range(0, 300, 7):
        for c in range(3):
            q2 = q.copy(); q2[n, c] += e; q3 = q.copy(); q3[n, c] -= e
            Gn[n, c] = (penalty(q2, 6.0, grad=False) - penalty(q3, 6.0, grad=False)) / (2 * e)
    sel = np.arange(0, 300, 7)
    print("grad max err", np.abs(G[sel] - Gn[sel]).max(), "scale", np.abs(G[sel]).max())
    # incumbent
    import pandas as pd
    df = pd.read_csv("/Users/lucaszhang/squares-in-squares/poc/pell_corrugation/data/border-scan/merged.csv")
    d = df[df.n == 635]; qq = d[["x", "y", "angle"]].to_numpy(float).copy()
    hh = 0.5 * (np.abs(np.cos(qq[:, 2])) + np.abs(np.sin(qq[:, 2])))
    qq[:, 0] -= (qq[:, 0] - hh).min(); qq[:, 1] -= (qq[:, 1] - hh).min()
    t0 = time.time(); od, wv, npair = max_violation(qq, bbox_side(qq))
    print("incumbent 635: N", len(qq), "bbox side", bbox_side(qq), "max overlap depth", od, "overlap pairs>0", npair, "t", time.time() - t0)
