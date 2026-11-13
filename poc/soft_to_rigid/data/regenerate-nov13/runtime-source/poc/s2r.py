"""Soft-to-rigid ("hardening") packing of unit squares, CPU numpy/scipy.

Stage 1: superellipses |x/a|^p+|y/a|^p<=1 (a=.5, disk at p=2 -> square at p=inf) with a
differentiable boundary-sample-in-gauge overlap penalty, p raised and box side shrunk.
Stage 2: exact rigid squares (SAT-depth penalty with analytic gradient) at p=inf,
side compressed by continuation, then dilated and certified by the official metric.
"""
import math, sys, time, json
import numpy as np
from scipy.optimize import minimize
sys.path.insert(0, '/mnt/e/squares-poc-regenerate-nov13/poc/baseline/poc/src')
from squares.core import evaluate

K = 40
PHI = np.linspace(0, 2*np.pi, K, endpoint=False) + 0.01

def shape_pts(p):
    c, s = np.cos(PHI), np.sin(PHI)
    r = (np.abs(c)**p + np.abs(s)**p)**(-1.0/p)
    return 0.5*np.stack([c*r, s*r], 1)          # (K,2) local boundary points

def soft_fg(z, n, L, p, S):
    q = z.reshape(n, 3); xy = q[:, :2]; th = q[:, 2]
    co, si = np.cos(th), np.sin(th)
    g = np.zeros_like(q); f = 0.0
    # world boundary points (n,K,2)
    ox = co[:, None]*S[:, 0] - si[:, None]*S[:, 1]
    oy = si[:, None]*S[:, 0] + co[:, None]*S[:, 1]
    # container
    W = np.stack([xy[:, 0, None]+ox, xy[:, 1, None]+oy], -1)
    lo = np.maximum(0, -W); hi = np.maximum(0, W-L)
    f += (lo**2).sum() + (hi**2).sum()
    gw = -2*lo + 2*hi                               # dF/dW (n,K,2)
    gxy = gw.sum(1)
    gth = (gw[..., 0]*(-oy) + gw[..., 1]*ox).sum(1)
    g[:, :2] += gxy; g[:, 2] += gth
    # pairs
    d = xy[:, None, :]-xy[None, :, :]
    I, J = np.where(np.triu(np.hypot(d[..., 0], d[..., 1]) < 1.43, 1))
    if len(I):
        for A, B in ((I, J), (J, I)):               # points of A inside shape B
            dd = W[A]-xy[B][:, None, :]              # (P,K,2)
            cB, sB = co[B][:, None], si[B][:, None]
            wx = cB*dd[..., 0]+sB*dd[..., 1]; wy = -sB*dd[..., 0]+cB*dd[..., 1]
            ax, ay = np.abs(wx)*2, np.abs(wy)*2
            m = np.maximum(np.maximum(ax, ay), 1e-9)
            tx, ty = (ax/m)**p, (ay/m)**p
            sm = tx+ty
            gg = m*sm**(1.0/p)
            pen = np.maximum(0, 1-gg)
            act = pen > 0
            f += (pen**2).sum()
            c0 = np.where(act, -2*pen, 0.0)            # dF/dgg
            base = sm**(1.0/p-1.0)
            dgx = base*(ax/m)**(p-1)*np.sign(wx)*2
            dgy = base*(ay/m)**(p-1)*np.sign(wy)*2
            gwx, gwy = c0*dgx, c0*dgy                  # dF/dw
            # w = R^T d: dF/dd = R gw ; dw/dthetaB = (wy,-wx)
            gdx = cB*gwx - sB*gwy; gdy = sB*gwx + cB*gwy
            gthB = (gwx*wy - gwy*wx).sum(1)
            gA = np.stack([gdx.sum(1), gdy.sum(1)], 1)
            gthA = (gdx*(-oy[A]) + gdy*ox[A]).sum(1)
            np.add.at(g[:, 0], A, gA[:, 0]); np.add.at(g[:, 1], A, gA[:, 1])
            np.add.at(g[:, 2], A, gthA)
            np.add.at(g[:, 0], B, -gA[:, 0]); np.add.at(g[:, 1], B, -gA[:, 1])
            np.add.at(g[:, 2], B, gthB)
    return f, g.ravel()

def rigid_fg(z, n, L):
    q = z.reshape(n, 3); xy = q[:, :2]; th = q[:, 2]
    co, si = np.cos(th), np.sin(th)
    g = np.zeros_like(q); f = 0.0
    ext = (np.abs(co)+np.abs(si))/2
    dex = (-np.sign(co)*si + np.sign(si)*co)/2          # d ext/d th
    for k in range(2):
        e = xy[:, k]-L/2
        v = np.maximum(0, np.abs(e)+ext-L/2)
        f += (v**2).sum()
        g[:, k] += 2*v*np.sign(e); g[:, 2] += 2*v*dex
    d = xy[:, None, :]-xy[None, :, :]
    I, J = np.where(np.triu(np.hypot(d[..., 0], d[..., 1]) < 1.4143, 1))
    if len(I):
        dl = xy[J]-xy[I]; dth = th[J]-th[I]
        cD, sD = np.cos(dth), np.sin(dth)
        R = 0.5*(np.abs(cD)+np.abs(sD))
        dR = 0.5*(-np.sign(cD)*sD + np.sign(sD)*cD)      # dR/d dth
        u = np.stack([co[I], si[I]], 1); v = np.stack([-si[I], co[I]], 1)
        u2 = np.stack([co[J], si[J]], 1); v2 = np.stack([-si[J], co[J]], 1)
        pr = np.stack([(u*dl).sum(1), (v*dl).sum(1), (u2*dl).sum(1), (v2*dl).sum(1)], 1)
        dep = 0.5+R[:, None]-np.abs(pr)
        k = np.argmin(dep, 1); P = np.arange(len(I))
        depth = dep[P, k]; act = depth > 0
        f += (depth[act]**2).sum()
        c = np.where(act, 2*depth, 0.0)
        ax = np.where((k <= 1)[:, None], np.where((k == 0)[:, None], u, v),
                      np.where((k == 2)[:, None], u2, v2))
        sg = np.sign(pr[P, k])
        # depth = .5+R(dth)-|ax.dl|
        gd = -(c*sg)[:, None]*ax                        # dF/d dl
        # d(ax.dl)/d theta of the axis owner
        perp = np.stack([-ax[:, 1], ax[:, 0]], 1)       # d ax/d theta
        dproj = (perp*dl).sum(1)
        gax = -(c*sg)*dproj
        gdt = c*dR
        np.add.at(g[:, 0], J, gd[:, 0]); np.add.at(g[:, 1], J, gd[:, 1])
        np.add.at(g[:, 0], I, -gd[:, 0]); np.add.at(g[:, 1], I, -gd[:, 1])
        own = np.where(k <= 1, I, J)
        np.add.at(g[:, 2], own, gax)
        np.add.at(g[:, 2], J, gdt); np.add.at(g[:, 2], I, -gdt)
    return f, g.ravel()

def harden(z, n, L0, L1, p0=2.0, p1=48.0, stages=30, it=60, rng=None, log=None):
    z = z.copy(); q = z.reshape(n, 3)
    Lc = L0
    for s in range(stages):
        t = s/(stages-1)
        p = p0*(p1/p0)**t; Ln = L0+(L1-L0)*t
        q[:, :2] *= Ln/Lc; Lc = Ln
        S = shape_pts(p)
        r = minimize(soft_fg, z, args=(n, Lc, p, S), jac=True, method='L-BFGS-B',
                     options=dict(maxiter=it, maxls=20))
        z = r.x; q = z.reshape(n, 3)
        if log: log(f'  stage {s} p={p:.1f} L={Lc:.4f} pen={r.fun:.2e}')
    return z

def rigid_min(z, n, L, it=400):
    r = minimize(rigid_fg, z, args=(n, L), jac=True, method='L-BFGS-B',
                 options=dict(maxiter=it, maxls=30, ftol=1e-16, gtol=1e-12))
    return r.x, r.fun

def compress(z, n, L, tol=1e-16, step0=0.02, step_min=2e-5, tmax=60):
    """Raise L until exact-feasible, then shrink by continuation. Returns (z, L) feasible or None."""
    t0 = time.time()
    Lc = L; zc = z.copy(); ok = None
    while Lc < math.ceil(math.sqrt(n)) + 0.5 and time.time()-t0 < tmax:
        zz = zc.copy(); zz.reshape(n, 3)[:, :2] *= 1.0
        zz, f = rigid_min(zz, n, Lc)
        if f < tol: ok = (zz, Lc); break
        zc = zz; Lc += 0.01
    if ok is None: return None
    zc, Lc = ok; step = step0
    while step > step_min and time.time()-t0 < tmax:
        Ln = Lc-step
        zz = zc.copy(); zz.reshape(n, 3)[:, :2] *= Ln/Lc
        zz, f = rigid_min(zz, n, Ln)
        if f < tol: zc, Lc = zz, Ln
        else: step /= 2
    return zc, Lc

def certify_z(z, n, L, dil=1.0+1e-8):
    q = z.reshape(n, 3).copy()
    q[:, :2] -= L/2; q[:, :2] *= dil; q[:, :2] += L/2*dil
    rows = [dict(square_id=str(i), x=format(a, '.17g'), y=format(b, '.17g'), angle=format(t, '.17g'))
            for i, (a, b, t) in enumerate(q)]
    _, _, side, reason = evaluate(n, rows)
    return (float(side) if side else None), reason, rows
