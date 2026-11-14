"""Batched SAT overlap penalty for packing unit squares in a rectangle.

Model == `poc/k2p10_fill2/poc/a_overlap.py` (the repo's exact SAT penalty):
for every pair the penetration depth is
    depth = min_k ( rho(alpha_k - t_i) + rho(alpha_k - t_j) - |D . u_k| )
over the four separating axes alpha_k in {t_i, t_i+pi/2, t_j, t_j+pi/2},
rho(psi) = 0.5(|cos psi| + |sin psi|);  E = sum max(depth,0)^2 + sum wall^2.

Key simplification used here (verified against a_overlap in `selftest`):
rho(alpha - t_i) + rho(alpha - t_j) is the SAME (0.5 + rho(t_j - t_i)) for all
four axes, because rho(psi + pi/2) = rho(psi).  Hence

    depth = 0.5 + rho(phi) - max(|D.u_i|, |D.u_i^+|, |D.u_j|, |D.u_j^+|)

i.e. only the two squares' own edge frames matter.  Everything below is
vectorised over R independent restarts *and* the pair list, with an analytic
gradient (same branch selection as a_overlap: the arg-max axis).

Coordinates: Q[r, k] = (x, y, theta) = centre and angle (radians) of square k;
box is [0, W[r]] x [0, H[r]].
"""
import math

import numpy as np

HP = math.pi / 2


def pair_list(N):
    I, J = np.triu_indices(N, 1)
    return I.astype(np.int64), J.astype(np.int64)


def energy_grad(Q, W, H, I, J, want_grad=True):
    """Q (R,N,3); W,H (R,).  Returns E (R,) and G (R,N,3) (None if not wanted)."""
    R, N, _ = Q.shape
    x = Q[:, :, 0]
    y = Q[:, :, 1]
    t = Q[:, :, 2]
    c = np.cos(t)
    s = np.sin(t)
    cI = c[:, I]; sI = s[:, I]; cJ = c[:, J]; sJ = s[:, J]
    dx = x[:, J] - x[:, I]
    dy = y[:, J] - y[:, I]
    phi = t[:, J] - t[:, I]
    cp = np.cos(phi); sp = np.sin(phi)
    ro = 0.5 * (np.abs(cp) + np.abs(sp))

    p0 = np.abs(dx * cI + dy * sI)
    p1 = np.abs(-dx * sI + dy * cI)
    p2 = np.abs(dx * cJ + dy * sJ)
    p3 = np.abs(-dx * sJ + dy * cJ)
    best = p0
    k = np.zeros(p0.shape, np.uint8)
    m = p1 > best; best = np.where(m, p1, best); k[m] = 1
    m = p2 > best; best = np.where(m, p2, best); k[m] = 2
    m = p3 > best; best = np.where(m, p3, best); k[m] = 3
    del p0, p1, p2, p3, m

    pen = 0.5 + ro - best
    del best, ro
    e = np.maximum(pen, 0.0)
    del pen
    E = (e * e).sum(1)

    # ---- wall ----
    hh = 0.5 * (np.abs(c) + np.abs(s))
    lox = np.maximum(0.0, hh - x); hix = np.maximum(0.0, x + hh - W[:, None])
    loy = np.maximum(0.0, hh - y); hiy = np.maximum(0.0, y + hh - H[:, None])
    E += (lox * lox + hix * hix + loy * loy + hiy * hiy).sum(1)
    if not want_grad:
        return E, None

    G = np.zeros((R, N, 3))
    G[:, :, 0] += 2.0 * (hix - lox)
    G[:, :, 1] += 2.0 * (hiy - loy)
    dh = 0.5 * (-np.sign(c) * s + np.sign(s) * c)
    G[:, :, 2] += 2.0 * dh * (lox + hix + loy + hiy)
    del lox, hix, loy, hiy, hh, dh

    # ---- pair gradient on the active pairs only ----
    act = e > 0.0
    if act.any():
        rr, cc = np.nonzero(act)
        ee = e[rr, cc]
        ii = I[cc]; jj = J[cc]
        ka = k[rr, cc]
        dxa = dx[rr, cc]; dya = dy[rr, cc]
        tia = t[rr, ii]; tja = t[rr, jj]
        fromi = ka < 2
        ka = ka.astype(np.float64)
        alpha = np.where(fromi, tia + ka * HP, tja + (ka - 2.0) * HP)
        ux = np.cos(alpha); uy = np.sin(alpha)
        proj = dxa * ux + dya * uy
        sg = np.sign(proj); sg[sg == 0] = 1.0
        psi_o = np.where(fromi, alpha - tja, alpha - tia)
        cpo = np.cos(psi_o); spo = np.sin(psi_o)
        dr = 0.5 * (-np.sign(cpo) * spo + np.sign(spo) * cpo)
        dDu = -sg * (-dxa * uy + dya * ux)
        dti = np.where(fromi, dr + dDu, -dr)
        dtj = np.where(fromi, -dr, dr + dDu)
        f = 2.0 * ee
        gx = f * sg * ux
        gy = f * sg * uy
        base = rr.astype(np.int64) * N
        wi = base + ii
        wj = base + jj
        idx = np.concatenate([wi, wj])
        L = R * N
        G[:, :, 0] += np.bincount(idx, weights=np.concatenate([gx, -gx]), minlength=L).reshape(R, N)
        G[:, :, 1] += np.bincount(idx, weights=np.concatenate([gy, -gy]), minlength=L).reshape(R, N)
        G[:, :, 2] += np.bincount(idx, weights=np.concatenate([f * dti, f * dtj]),
                                  minlength=L).reshape(R, N)
    return E, G


# --------------------------------------------------------------------------
# flat / screened variant: same maths on a 1-D pair list that carries its own
# restart index.  Lets the caller drop pairs that are far apart (neighbour
# lists refreshed once per sweep) -- identical energy, ~5-10x cheaper.
# --------------------------------------------------------------------------
def neighbours(Q, margin=1.0):
    """flat (r, i, j) arrays of pairs whose centres are within sqrt(2)+margin."""
    R, N, _ = Q.shape
    I, J = pair_list(N)
    x = Q[:, :, 0]; y = Q[:, :, 1]
    dx = x[:, J] - x[:, I]
    dy = y[:, J] - y[:, I]
    d2 = dx * dx + dy * dy
    keep = d2 <= (math.sqrt(2.0) + margin) ** 2
    rr, cc = np.nonzero(keep)
    return rr.astype(np.int64), I[cc], J[cc]


def energy_grad_flat(Q, W, H, R_, I_, J_, want_grad=True):
    """Q (R,N,3), W,H (R,), flat pair lists -> E (R,), G (R,N,3)."""
    R, N, _ = Q.shape
    x = Q[:, :, 0]; y = Q[:, :, 1]; t = Q[:, :, 2]
    G = np.zeros((R, N, 3))
    # ---- wall (exact, all squares) ----
    c = np.cos(t); s = np.sin(t)
    hh = 0.5 * (np.abs(c) + np.abs(s))
    lox = np.maximum(0.0, hh - x); hix = np.maximum(0.0, x + hh - W[:, None])
    loy = np.maximum(0.0, hh - y); hiy = np.maximum(0.0, y + hh - H[:, None])
    E = (lox * lox + hix * hix + loy * loy + hiy * hiy).sum(1)
    if not want_grad:
        if I_.size == 0:
            return E, None
        xi = x[R_, I_]; yi = y[R_, I_]; ti = t[R_, I_]
        xj = x[R_, J_]; yj = y[R_, J_]; tj = t[R_, J_]
        dx = xj - xi; dy = yj - yi
        ci = np.cos(ti); si = np.sin(ti); cj = np.cos(tj); sj = np.sin(tj)
        phi = tj - ti
        ro = 0.5 * (np.abs(np.cos(phi)) + np.abs(np.sin(phi)))
        p0 = np.abs(dx * ci + dy * si)
        p1 = np.abs(-dx * si + dy * ci)
        p2 = np.maximum(p0, np.abs(dx * cj + dy * sj))
        p2 = np.maximum(p2, np.abs(-dx * sj + dy * cj))
        p2 = np.maximum(p2, p1)
        pen = 0.5 + ro - p2
        e = np.maximum(pen, 0.0)
        E += np.bincount(R_, weights=e * e, minlength=R)
        return E, None
    G[:, :, 0] += 2.0 * (hix - lox)
    G[:, :, 1] += 2.0 * (hiy - loy)
    dh = 0.5 * (-np.sign(c) * s + np.sign(s) * c)
    G[:, :, 2] += 2.0 * dh * (lox + hix + loy + hiy)
    if I_.size == 0:
        return E, G

    xi = x[R_, I_]; yi = y[R_, I_]; ti = t[R_, I_]
    xj = x[R_, J_]; yj = y[R_, J_]; tj = t[R_, J_]
    dx = xj - xi; dy = yj - yi
    ci = np.cos(ti); si = np.sin(ti); cj = np.cos(tj); sj = np.sin(tj)
    phi = tj - ti
    cp = np.cos(phi); sp = np.sin(phi)
    ro = 0.5 * (np.abs(cp) + np.abs(sp))
    p0 = np.abs(dx * ci + dy * si)
    p1 = np.abs(-dx * si + dy * ci)
    p2 = np.abs(dx * cj + dy * sj)
    p3 = np.abs(-dx * sj + dy * cj)
    best = p0
    k = np.zeros(p0.shape, np.uint8)
    m = p1 > best; best = np.where(m, p1, best); k[m] = 1
    m = p2 > best; best = np.where(m, p2, best); k[m] = 2
    m = p3 > best; best = np.where(m, p3, best); k[m] = 3
    del p0, p1, p2, p3, m
    pen = 0.5 + ro - best
    del best, ro
    e = np.maximum(pen, 0.0)
    del pen
    E += np.bincount(R_, weights=e * e, minlength=R)
    act = e > 0.0
    if act.any():
        rr = R_[act]; ii = I_[act]; jj = J_[act]
        ee = e[act]
        ka = k[act]
        dxa = dx[act]; dya = dy[act]
        tia = ti[act]; tja = tj[act]
        fromi = ka < 2
        kaf = ka.astype(np.float64)
        alpha = np.where(fromi, tia + kaf * HP, tja + (kaf - 2.0) * HP)
        ux = np.cos(alpha); uy = np.sin(alpha)
        proj = dxa * ux + dya * uy
        sg = np.sign(proj); sg[sg == 0] = 1.0
        psi_o = np.where(fromi, alpha - tja, alpha - tia)
        cpo = np.cos(psi_o); spo = np.sin(psi_o)
        dr = 0.5 * (-np.sign(cpo) * spo + np.sign(spo) * cpo)
        dDu = -sg * (-dxa * uy + dya * ux)
        dti = np.where(fromi, dr + dDu, -dr)
        dtj = np.where(fromi, -dr, dr + dDu)
        f = 2.0 * ee
        gx = f * sg * ux
        gy = f * sg * uy
        L = R * N
        wi = rr * N + ii
        wj = rr * N + jj
        idx = np.concatenate([wi, wj])
        G[:, :, 0] += np.bincount(idx, weights=np.concatenate([gx, -gx]),
                                  minlength=L).reshape(R, N)
        G[:, :, 1] += np.bincount(idx, weights=np.concatenate([gy, -gy]),
                                  minlength=L).reshape(R, N)
        G[:, :, 2] += np.bincount(idx, weights=np.concatenate([f * dti, f * dtj]),
                                  minlength=L).reshape(R, N)
    return E, G


# --------------------------------------------------------------------------
# exact (non-batched) feasibility report, independent of the penalty branch
# selection: uses the repo module verbatim when available.
# --------------------------------------------------------------------------
def best_axis_depth(q, P):
    """q (N,3) single configuration -> penetration depth per pair (a_overlap maths)."""
    i, j = P[:, 0], P[:, 1]
    D = q[j, :2] - q[i, :2]
    ti, tj = q[i, 2], q[j, 2]
    al = np.stack([ti, ti + HP, tj, tj + HP], 1)
    ux, uy = np.cos(al), np.sin(al)
    dot = D[:, 0:1] * ux + D[:, 1:2] * uy
    r = 0.5 * (np.abs(np.cos(al - ti[:, None])) + np.abs(np.sin(al - ti[:, None]))
               + np.abs(np.cos(al - tj[:, None])) + np.abs(np.sin(al - tj[:, None])))
    return (r - np.abs(dot)).min(1)


def report(q, W, H):
    """(max pair penetration, max wall overshoot) for one configuration."""
    N = len(q)
    I, J = pair_list(N)
    dep = best_axis_depth(q, np.stack([I, J], 1)) if N > 1 else np.zeros(0)
    hh = 0.5 * (np.abs(np.cos(q[:, 2])) + np.abs(np.sin(q[:, 2])))
    wall = max(0.0, float((hh - q[:, 0]).max()), float((q[:, 0] + hh - W).max()),
               0.0, float((hh - q[:, 1]).max()), float((q[:, 1] + hh - H).max()))
    return (float(dep.max()) if dep.size else 0.0), wall


def bbox(q):
    hh = 0.5 * (np.abs(np.cos(q[:, 2])) + np.abs(np.sin(q[:, 2])))
    x0 = float((q[:, 0] - hh).min()); x1 = float((q[:, 0] + hh).max())
    y0 = float((q[:, 1] - hh).min()); y1 = float((q[:, 1] + hh).max())
    return x1 - x0, y1 - y0


def shape_f(q, a, b):
    """the realised f of a configuration for shape (a,b): max(W-a, H-b) after tightening."""
    w, h = bbox(q)
    return max(w - a, h - b), w, h


def tighten(q):
    """translate so the bbox starts at 0; returns new q (copy) and (W,H)."""
    hh = 0.5 * (np.abs(np.cos(q[:, 2])) + np.abs(np.sin(q[:, 2])))
    q = q.copy()
    q[:, 0] -= float((q[:, 0] - hh).min())
    q[:, 1] -= float((q[:, 1] - hh).min())
    w, h = bbox(q)
    return q, w, h


def selftest():
    import sys
    sys.path.insert(0, '../../k2p10_fill2/poc')
    sys.path.insert(0, '../k2p10_fill2/poc')
    import a_overlap as ao
    rng = np.random.default_rng(7)
    N = 24
    q = np.c_[rng.uniform(0, 5, (N, 2)), rng.uniform(0, 3, N)]
    L = 5.0
    Q = q[None]
    E, G = energy_grad(Q, np.array([L]), np.array([L]), *pair_list(N))
    E0, G0 = ao.penalty(q, L)
    print('energy  mine %.12e ref %.12e  diff %.2e' % (E[0], E0, abs(E[0] - E0)))
    print('grad    max|mine-ref| %.3e   scale %.3e' % (np.abs(G[0] - G0).max(), np.abs(G0).max()))
    # finite differences
    e = 1e-6
    err = 0.0
    for n in range(0, N, 3):
        for cc in range(3):
            qp = q.copy(); qp[n, cc] += e
            qm = q.copy(); qm[n, cc] -= e
            num = (ao.penalty(qp, L, grad=False) - ao.penalty(qm, L, grad=False)) / (2 * e)
            err = max(err, abs(num - G[0, n, cc]))
    print('grad    max|mine-fd|  %.3e' % err)
    # wall gradient check on a tight box
    W = np.array([4.0]); Hh = np.array([3.0])
    E1, G1 = energy_grad(Q, W, Hh, *pair_list(N))
    err = 0.0
    for n in range(0, N, 3):
        for cc in range(3):
            Qp = Q.copy(); Qp[0, n, cc] += e
            Qm = Q.copy(); Qm[0, n, cc] -= e
            num = (energy_grad(Qp, W, Hh, *pair_list(N), want_grad=False)[0][0]
                   - energy_grad(Qm, W, Hh, *pair_list(N), want_grad=False)[0][0]) / (2 * e)
            err = max(err, abs(num - G1[0, n, cc]))
    print('grad    max|fd(tight box)| %.3e' % err)


if __name__ == '__main__':
    selftest()
