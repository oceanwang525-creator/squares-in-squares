"""Vectorised unit-square SAT overlap module (numpy).

Self-contained re-implementation of the penalty model used elsewhere in this
repo (`poc/k2p10_fill2/poc/a_overlap.py`), reproduced here so that this POC has
no cross-folder import.  `q` is an (N,3) array of (x, y, angle) *centres*.

Separating-axis test for two equal unit squares: the four candidate axes are
the two edge directions of each square, and the projection radius of a unit
square on an axis at angle `a` is r(a) = (|cos a| + |sin a|) / 2.  A pair is
overlapping iff every candidate axis shows a positive penetration, so the
penetration depth is the minimum over the four axes.
"""
import math

import numpy as np
from scipy.spatial import cKDTree


def pairs(q, margin=0.0):
    """Candidate pairs (i<j) whose centres are within sqrt(2) + margin."""
    t = cKDTree(q[:, :2])
    return t.query_pairs(math.sqrt(2.0) + margin + 1e-9, output_type="ndarray")


_ALL = {}


def all_pairs(n):
    """All (i<j) index pairs, cached.  For n <= ~200 this beats a KD-tree."""
    if n not in _ALL:
        _ALL[n] = np.stack(np.triu_indices(n, 1), 1)
    return _ALL[n]


def penalty_fast(q, L, w_wall=3.0, grad=True, margin=0.0):
    """Same penalty as `penalty` over *all* pairs, in the minimal op count.

    Uses  depth_ij = 1/2 + h(ti-tj) - max(|D.ui|, |D.ui_perp|, |D.uj|,
    |D.uj_perp|)  with  h(psi) = (|cos psi| + |sin psi|)/2,  which is the
    smallest of the four separating-axis penetrations.  Intended for
    N <= ~200, where `np.triu_indices` caching and 4 vectorised dot products
    beat a KD-tree rebuild.
    """
    n = len(q)
    P = all_pairs(n)
    i, j = P[:, 0], P[:, 1]
    x, y, t = q[:, 0], q[:, 1], q[:, 2]
    Dx = x[j] - x[i]
    Dy = y[j] - y[i]
    # depth + margin <= 1.2071 - 0.7071*|D|, so pairs with |D| > 1.72 can never
    # contribute (for margin <= 1e-3).
    act = Dx * Dx + Dy * Dy < 3.0
    if not act.all():
        i, j, Dx, Dy = i[act], j[act], Dx[act], Dy[act]
    ci, si = np.cos(t), np.sin(t)
    cj, sj = ci[j], si[j]
    ci_, si_ = ci[i], si[i]
    a1 = Dx * ci_ + Dy * si_
    b1 = -Dx * si_ + Dy * ci_
    a2 = Dx * cj + Dy * sj
    b2 = -Dx * sj + Dy * cj
    dots = np.stack([a1, b1, a2, b2], 0)
    ad = np.abs(dots)
    k = ad.argmax(0)
    m = ad[k, np.arange(len(i))]
    dpsi = t[i] - t[j]
    cpsi, spsi = np.cos(dpsi), np.sin(dpsi)
    hh = 0.5 * (np.abs(cpsi) + np.abs(spsi))
    dep = 0.5 + hh - m
    e = dep + margin
    sel = e > 0.0
    E = float(np.sum(e[sel] * e[sel]))
    G = np.zeros((n, 3))
    if grad and sel.any():
        kk = k[sel]
        idx = np.arange(len(i))[sel]
        ux_all = np.stack([ci_, -si_, cj, -sj], 0)[kk, idx]
        uy_all = np.stack([si_, ci_, sj, cj], 0)[kk, idx]
        dotw = dots[kk, idx]
        s = np.sign(dotw)
        s[s == 0] = 1.0
        f = 2.0 * e[sel]
        # d(-|D.u|)/d(alpha) for the winning axis
        dDu = -s * (-Dx[sel] * uy_all + Dy[sel] * ux_all)
        fromi = kk < 2
        cw, sw = np.cos(dpsi[sel]), np.sin(dpsi[sel])
        dh = 0.5 * (-np.sign(cw) * sw + np.sign(sw) * cw)
        dti = np.where(fromi, dh + dDu, dh)
        dtj = np.where(fromi, -dh, -dh + dDu)
        ii, jj = i[sel], j[sel]
        bc = np.bincount
        gx = -s * ux_all
        gy = -s * uy_all
        G[:, 0] = bc(ii, f * -gx, n) + bc(jj, f * gx, n)
        G[:, 1] = bc(ii, f * -gy, n) + bc(jj, f * gy, n)
        G[:, 2] = bc(ii, f * dti, n) + bc(jj, f * dtj, n)
    h = 0.5 * (np.abs(ci) + np.abs(si))
    lo = np.maximum(0.0, h[:, None] - q[:, :2])
    hi = np.maximum(0.0, q[:, :2] + h[:, None] - L)
    E += w_wall * float(np.sum(lo * lo + hi * hi))
    if grad:
        G[:, :2] += w_wall * 2.0 * (hi - lo)
        dh2 = 0.5 * (-np.sign(ci) * si + np.sign(si) * ci)
        G[:, 2] += w_wall * 2.0 * dh2 * (np.sum(lo, 1) + np.sum(hi, 1))
        return E, G
    return E


def _r(psi):
    c, s = np.cos(psi), np.sin(psi)
    return 0.5 * (np.abs(c) + np.abs(s))


def pair_depth(q, P, grad=False):
    """Penetration depth per pair (<=0 means separated)."""
    i, j = P[:, 0], P[:, 1]
    D = q[j, :2] - q[i, :2]
    ti, tj = q[i, 2], q[j, 2]
    al = np.stack([ti, ti + math.pi / 2, tj, tj + math.pi / 2], 1)
    ux, uy = np.cos(al), np.sin(al)
    dot = D[:, 0:1] * ux + D[:, 1:2] * uy
    d = _r(al - ti[:, None]) + _r(al - tj[:, None]) - np.abs(dot)
    k = np.argmin(d, 1)
    dep = d[np.arange(len(P)), k]
    if not grad:
        return dep
    m = np.arange(len(P))
    a = al[m, k]
    s = np.sign(dot[m, k])
    s[s == 0] = 1.0
    ux_, uy_ = np.cos(a), np.sin(a)
    gpj = np.stack([-s * ux_, -s * uy_], 1)
    gpi = -gpj
    dDu = -s * (-D[:, 0] * uy_ + D[:, 1] * ux_)
    fromi = k < 2
    psi_o = np.where(fromi, a - tj, a - ti)
    c, sn = np.cos(psi_o), np.sin(psi_o)
    dr = 0.5 * (-np.sign(c) * sn + np.sign(sn) * c)
    dti = np.where(fromi, dr + dDu, -dr)
    dtj = np.where(fromi, -dr, dr + dDu)
    return dep, gpi, gpj, dti, dtj


def radii(q):
    return 0.5 * (np.abs(np.cos(q[:, 2])) + np.abs(np.sin(q[:, 2])))


def penalty(q, L, w_wall=3.0, grad=True, P=None, margin=0.0):
    """E = sum_pairs max(depth+margin,0)^2 + w_wall * sum wall_violation^2.

    A positive `margin` demands a strict clearance between every pair, which is
    what makes a float solution survive exact decimal certification.
    """
    N = len(q)
    if P is None:
        if N <= 200:
            return penalty_fast(q, L, w_wall=w_wall, grad=grad, margin=margin)
        P = pairs(q, margin)
    E = 0.0
    G = np.zeros((N, 3))
    if len(P):
        out = pair_depth(q, P, grad=grad)
        dep = out[0] if grad else out
        m = dep + margin > 0.0
        if grad:
            _, gpi, gpj, dti, dtj = out
        e = dep[m] + margin
        E += float(np.sum(e * e))
        if grad:
            Pm = P[m]
            f = 2.0 * e
            bc = np.bincount
            G[:, 0] = (bc(Pm[:, 0], f * gpi[m, 0], N)
                       + bc(Pm[:, 1], f * gpj[m, 0], N))
            G[:, 1] = (bc(Pm[:, 0], f * gpi[m, 1], N)
                       + bc(Pm[:, 1], f * gpj[m, 1], N))
            G[:, 2] = (bc(Pm[:, 0], f * dti[m], N)
                       + bc(Pm[:, 1], f * dtj[m], N))
    h = radii(q)
    lo = np.maximum(0.0, h[:, None] - q[:, :2])
    hi = np.maximum(0.0, q[:, :2] + h[:, None] - L)
    E += w_wall * float(np.sum(lo * lo + hi * hi))
    if grad:
        G[:, :2] += w_wall * 2.0 * (hi - lo)
        dh = 0.5 * (-np.sign(np.cos(q[:, 2])) * np.sin(q[:, 2])
                    + np.sign(np.sin(q[:, 2])) * np.cos(q[:, 2]))
        G[:, 2] += w_wall * 2.0 * dh * (np.sum(lo, 1) + np.sum(hi, 1))
        return E, G
    return E


def violation(q, L):
    """(max pair penetration, max wall violation) in float arithmetic."""
    P = pairs(q)
    if len(P):
        dep = pair_depth(q, P)
        od = float(dep.max())
    else:
        od = 0.0
    h = radii(q)
    lo = np.maximum(0.0, h[:, None] - q[:, :2])
    hi = np.maximum(0.0, q[:, :2] + h[:, None] - L)
    return od, float(max(lo.max(initial=0.0), hi.max(initial=0.0)))


def bbox(q):
    """(width, height) of the axis-aligned bounding box, and the offsets."""
    h = radii(q)
    x0 = float((q[:, 0] - h).min())
    x1 = float((q[:, 0] + h).max())
    y0 = float((q[:, 1] - h).min())
    y1 = float((q[:, 1] + h).max())
    return x1 - x0, y1 - y0, x0, y0


def side(q):
    w, h, _, _ = bbox(q)
    return max(w, h)


def normalise(q):
    """Translate so the bbox lower-left corner is the origin."""
    w, h, x0, y0 = bbox(q)
    q = q.copy()
    q[:, 0] -= x0
    q[:, 1] -= y0
    return q
