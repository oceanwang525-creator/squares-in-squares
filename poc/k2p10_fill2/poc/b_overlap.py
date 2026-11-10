"""Vectorised SAT penalty for unit squares (x, y, angle) in the box [0, Lbox]^2.
E = sum over pairs of depth^2 (depth = -max SAT gap over the 4 axes, if >0) + sum of wall violations^2,
with analytic gradient wrt x, y, angle.  Neighbour pairs via cKDTree (centre distance < sqrt2).
"""
import numpy as np
from scipy.spatial import cKDTree

R2 = 2 ** 0.5 + 1e-9
HP = np.pi / 2


def pairs(P, r=R2):
    pr = cKDTree(P[:, :2]).query_pairs(r, output_type="ndarray")
    return pr[:, 0], pr[:, 1]


def pair_gap(P, I, J, grad=False):
    """Max SAT gap over the 4 axes per pair (negative = overlap; value is minus penetration depth).
    If grad: also returns (dgdxi, dgdyi, dgdti, dgdxj, dgdyj, dgdtj), each shape (M,)."""
    X, Y, T = P[:, 0], P[:, 1], P[:, 2]
    M = len(I)
    G = np.empty((4, M))
    if grad:
        D = np.empty((4, 6, M))
    c4 = 0
    for flip in (0, 1):
        own, oth = (I, J) if flip == 0 else (J, I)
        dx = X[oth] - X[own]; dy = Y[oth] - Y[own]
        for k in (0, 1):
            phi = T[own] + k * HP
            c, s = np.cos(phi), np.sin(phi)
            proj = c * dx + s * dy
            sg = np.where(proj >= 0, 1.0, -1.0)
            dl = T[oth] - phi
            cd, sd = np.cos(dl), np.sin(dl)
            G[c4] = sg * proj - 0.5 - 0.5 * (np.abs(cd) + np.abs(sd))
            if grad:
                hp = -sd * np.where(cd >= 0, 1.0, -1.0) + cd * np.where(sd >= 0, 1.0, -1.0)
                gphi = sg * (-s * dx + c * dy) + 0.5 * hp
                gto = -0.5 * hp
                gxo, gyo = sg * c, sg * s
                # order: (xi, yi, ti, xj, yj, tj)
                if flip == 0:
                    D[c4] = np.array([-gxo, -gyo, gphi, gxo, gyo, gto])
                else:  # own = j, oth = i
                    D[c4] = np.array([gxo, gyo, gto, -gxo, -gyo, gphi])
            c4 += 1
    am = np.argmax(G, axis=0)
    ar = np.arange(M)
    g = G[am, ar]
    if grad:
        return g, D[am, :, ar].T
    return g


def energy(P, Lbox, pr=None, grad=True, pen_power=2):
    """Returns E, grad (n,3) (or None), info dict."""
    n = len(P)
    X, Y, T = P[:, 0], P[:, 1], P[:, 2]
    if pr is None:
        pr = pairs(P)
    I, J = pr
    gr = np.zeros((n, 3)) if grad else None
    E = 0.0
    maxd = 0.0; sumd = 0.0; npairs = 0
    if len(I):
        if grad:
            g, D = pair_gap(P, I, J, True)
        else:
            g = pair_gap(P, I, J)
        m = g < 0
        if m.any():
            d = -g[m]
            E += float((d ** 2).sum())
            maxd = float(d.max()); sumd = float(d.sum()); npairs = int(m.sum())
            if grad:
                w = (-2.0 * d)[None, :]  # dE/dg = -2 d ; d=-g
                Dm = D[:, m] * w
                Im, Jm = I[m], J[m]
                for q in range(3):
                    np.add.at(gr[:, q], Im, Dm[q])
                    np.add.at(gr[:, q], Jm, Dm[3 + q])
    # walls
    c, s = np.abs(np.cos(T)), np.abs(np.sin(T))
    e = 0.5 * (c + s)
    de = 0.5 * (-np.sin(T) * np.where(np.cos(T) >= 0, 1.0, -1.0) + np.cos(T) * np.where(np.sin(T) >= 0, 1.0, -1.0))
    wmax = 0.0
    for q, V in ((0, X), (1, Y)):
        lo = e - V            # >0 => violates low wall
        hi = V + e - Lbox     # >0 => violates high wall
        ml = lo > 0; mh = hi > 0
        if ml.any():
            E += float((lo[ml] ** 2).sum()); wmax = max(wmax, float(lo[ml].max()))
            if grad:
                gr[ml, q] += -2 * lo[ml]; gr[ml, 2] += 2 * lo[ml] * de[ml]
        if mh.any():
            E += float((hi[mh] ** 2).sum()); wmax = max(wmax, float(hi[mh].max()))
            if grad:
                gr[mh, q] += 2 * hi[mh]; gr[mh, 2] += 2 * hi[mh] * de[mh]
    return E, gr, dict(maxd=maxd, sumd=sumd, npairs=npairs, wmax=wmax)
