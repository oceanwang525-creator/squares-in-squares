"""Sequential-LP squeeze of unit squares in a rectangle (w + f) x (h + f), minimising f.

Generalized from poc/row_ratchet_980/poc/pslp2.py to support per-square corner rigid-block
constraints (move_x, move_y, frozen_x, frozen_y, frozen_th) for all four box corners:
  - BL_static: frozen_x, frozen_y, frozen_th
  - TR_rigid:  move_x, move_y, frozen_th
  - TL_rigid:  frozen_x, move_y, frozen_th
  - BR_rigid:  move_x, frozen_y, frozen_th
"""
from __future__ import annotations

import math
import numpy as np
import numba as nb
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

HP = math.pi / 2


@nb.njit(cache=True)
def _rho(phi):
    return 0.5 * (abs(math.cos(phi)) + abs(math.sin(phi)))


@nb.njit(cache=True)
def feas(X, Y, T, W, H):
    """(min pair gap, min wall slack); >= 0 means feasible."""
    n = X.shape[0]
    mp = 1e9
    mw = 1e9
    for i in range(n):
        e = _rho(T[i])
        mw = min(mw, X[i] - e, Y[i] - e, W - X[i] - e, H - Y[i] - e)
        for j in range(i + 1, n):
            dx = X[j] - X[i]
            dy = Y[j] - Y[i]
            if dx * dx + dy * dy >= 2.0000001:
                continue
            best = -1e9
            for a in range(4):
                if a < 2:
                    al = T[i] + a * HP
                    phi = T[j] - al
                else:
                    al = T[j] + (a - 2) * HP
                    phi = T[i] - al
                g = abs(dx * math.cos(al) + dy * math.sin(al)) - 0.5 - _rho(phi)
                if g > best:
                    best = g
            if best < mp:
                mp = best
    return mp, mw


@nb.njit(cache=True)
def build_rows(X, Y, T, w, h, f, trust, tau, fx_free, fy_free, capf=1):
    """A_ub z <= b_ub rows. z = [dx(n), dy(n), dth(n), df, t]."""
    n = X.shape[0]
    cap_r = (16 * n + 64) * capf
    R = np.empty(cap_r * 8, np.int64)
    C = np.empty(cap_r * 8, np.int64)
    V = np.empty(cap_r * 8)
    B = np.empty(cap_r)
    r = 0
    k = 0
    IDF = 3 * n
    IT = 3 * n + 1
    reach = 1.41422 + 3.0 * trust
    R2 = reach * reach
    W = w + f
    Hh = h + f
    for i in range(n):
        for j in range(i + 1, n):
            dx = X[j] - X[i]
            dy = Y[j] - Y[i]
            if dx * dx + dy * dy >= R2:
                continue
            best = -1e9
            ba = 0
            for a in range(4):
                if a < 2:
                    al = T[i] + a * HP
                    phi = T[j] - al
                else:
                    al = T[j] + (a - 2) * HP
                    phi = T[i] - al
                g = abs(dx * math.cos(al) + dy * math.sin(al)) - 0.5 - _rho(phi)
                if g > best:
                    best = g
                    ba = a
            if best > 4.0 * trust:
                continue
            if ba < 2:
                own = i
                oth = j
                al = T[i] + ba * HP
            else:
                own = j
                oth = i
                al = T[j] + (ba - 2) * HP
            phi = T[oth] - al
            ux = math.cos(al)
            uy = math.sin(al)
            proj = dx * ux + dy * uy
            s = 1.0 if proj >= 0 else -1.0
            perp = -dx * uy + dy * ux
            c = math.cos(phi)
            sn = math.sin(phi)
            s1 = 1.0 if c >= 0 else -1.0
            s2 = 1.0 if sn >= 0 else -1.0
            for b1 in range(2):
                if b1 == 1 and abs(c) >= tau:
                    continue
                for b2 in range(2):
                    if b2 == 1 and abs(sn) >= tau:
                        continue
                    q1 = s1 if b1 == 0 else -s1
                    q2 = s2 if b2 == 0 else -s2
                    rb = 0.5 * (q1 * c + q2 * sn)
                    drb = 0.5 * (-q1 * sn + q2 * c)
                    g = s * proj - 0.5 - rb
                    if r >= cap_r - 8:
                        return R, C, V, B, -1, k
                    R[k] = r; C[k] = i; V[k] = s * ux; k += 1
                    R[k] = r; C[k] = j; V[k] = -s * ux; k += 1
                    R[k] = r; C[k] = n + i; V[k] = s * uy; k += 1
                    R[k] = r; C[k] = n + j; V[k] = -s * uy; k += 1
                    R[k] = r; C[k] = 2 * n + own; V[k] = -(s * perp + drb); k += 1
                    R[k] = r; C[k] = 2 * n + oth; V[k] = drb; k += 1
                    R[k] = r; C[k] = IT; V[k] = -1.0; k += 1
                    B[r] = g
                    r += 1
    lim = 3.0 * trust + 1e-9
    for i in range(n):
        c = math.cos(T[i])
        sn = math.sin(T[i])
        s1 = 1.0 if c >= 0 else -1.0
        s2 = 1.0 if sn >= 0 else -1.0
        e = 0.5 * (abs(c) + abs(sn))
        nl = X[i] - e < lim
        nr = W - X[i] - e < lim
        nb_ = Y[i] - e < lim
        nt = Hh - Y[i] - e < lim
        if not (nl or nr or nb_ or nt):
            continue
        for b1 in range(2):
            if b1 == 1 and abs(c) >= tau:
                continue
            for b2 in range(2):
                if b2 == 1 and abs(sn) >= tau:
                    continue
                q1 = s1 if b1 == 0 else -s1
                q2 = s2 if b2 == 0 else -s2
                rb = 0.5 * (q1 * c + q2 * sn)
                drb = 0.5 * (-q1 * sn + q2 * c)
                if r >= cap_r - 8:
                    return R, C, V, B, -1, k
                if nl:
                    R[k] = r; C[k] = i; V[k] = -1.0; k += 1
                    R[k] = r; C[k] = 2 * n + i; V[k] = drb; k += 1
                    R[k] = r; C[k] = IT; V[k] = -1.0; k += 1
                    B[r] = X[i] - rb
                    r += 1
                if nr:
                    R[k] = r; C[k] = i; V[k] = 1.0; k += 1
                    R[k] = r; C[k] = 2 * n + i; V[k] = drb; k += 1
                    if fx_free:
                        R[k] = r; C[k] = IDF; V[k] = -1.0; k += 1
                    R[k] = r; C[k] = IT; V[k] = -1.0; k += 1
                    B[r] = W - X[i] - rb
                    r += 1
                if nb_:
                    R[k] = r; C[k] = n + i; V[k] = -1.0; k += 1
                    R[k] = r; C[k] = 2 * n + i; V[k] = drb; k += 1
                    R[k] = r; C[k] = IT; V[k] = -1.0; k += 1
                    B[r] = Y[i] - rb
                    r += 1
                if nt:
                    R[k] = r; C[k] = n + i; V[k] = 1.0; k += 1
                    R[k] = r; C[k] = 2 * n + i; V[k] = drb; k += 1
                    if fy_free:
                        R[k] = r; C[k] = IDF; V[k] = -1.0; k += 1
                    R[k] = r; C[k] = IT; V[k] = -1.0; k += 1
                    B[r] = Hh - Y[i] - rb
                    r += 1
    return R, C, V, B, r, k


def lp_step(
    X,
    Y,
    T,
    w,
    h,
    f,
    trust,
    M=1e3,
    df_lo=None,
    df_hi=None,
    frozen=None,
    center=0.0,
    moving=None,
    move_x=None,
    move_y=None,
    frozen_x=None,
    frozen_y=None,
    frozen_th=None,
):
    """One LP in trust-scaled variables z = trust * u."""
    n = len(X)
    tau = min(0.5, 4.0 * trust + 1e-7)
    capf = 1
    while True:
        R, C, V, B, r, k = build_rows(X, Y, T, float(w), float(h), float(f), trust, tau, True, True, capf)
        if r >= 0:
            break
        capf *= 4
        if capf > 256:
            return None
    nv = 3 * n + 2

    if moving is not None:
        move_x = moving if move_x is None else (move_x | moving)
        move_y = moving if move_y is None else (move_y | moving)
        frozen_th = moving if frozen_th is None else (frozen_th | moving)

    Ck = C[:k]
    Vk = V[:k]
    if move_x is not None or move_y is not None or frozen_th is not None:
        isx = Ck < n
        isy = (Ck >= n) & (Ck < 2 * n)
        ist = (Ck >= 2 * n) & (Ck < 3 * n)
        if move_x is not None:
            mx = np.zeros(k, np.bool_)
            mx[isx] = move_x[Ck[isx]]
            Ck[mx] = 3 * n
        if move_y is not None:
            my = np.zeros(k, np.bool_)
            my[isy] = move_y[Ck[isy] - n]
            Ck[my] = 3 * n
        if frozen_th is not None:
            mt = np.zeros(k, np.bool_)
            mt[ist] = frozen_th[Ck[ist] - 2 * n]
            Vk[mt] = 0.0

    if center > 0:
        Rr = np.concatenate([R[:k], np.arange(r)])
        Cc = np.concatenate([Ck, nv + np.arange(r)])
        Vv = np.concatenate([Vk, np.ones(r)])
        A = csr_matrix((Vv, (Rr, Cc)), shape=(r, nv + r))
        c = np.zeros(nv + r)
        c[3 * n] = 1.0
        c[3 * n + 1] = M
        c[nv:] = -center / max(r, 1)
        lo = np.full(nv + r, -1.0)
        hi = np.full(nv + r, 1.0)
        lo[nv:] = 0.0
        hi[nv:] = 2.0
    else:
        A = csr_matrix((Vk, (R[:k], Ck)), shape=(r, nv))
        c = np.zeros(nv)
        c[3 * n] = 1.0
        c[3 * n + 1] = M
        lo = np.full(nv, -1.0)
        hi = np.full(nv, 1.0)

    lo[3 * n] = -1.0 if df_lo is None else df_lo / trust
    hi[3 * n] = 1.0 if df_hi is None else df_hi / trust
    lo[3 * n + 1] = 0.0
    hi[3 * n + 1] = np.inf

    if frozen is not None:
        for q in (0, n, 2 * n):
            lo[q:q + n][frozen] = 0.0
            hi[q:q + n][frozen] = 0.0
    if frozen_x is not None:
        lo[0:n][frozen_x] = 0.0
        hi[0:n][frozen_x] = 0.0
    if frozen_y is not None:
        lo[n:2 * n][frozen_y] = 0.0
        hi[n:2 * n][frozen_y] = 0.0
    if frozen_th is not None:
        lo[2 * n:3 * n][frozen_th] = 0.0
        hi[2 * n:3 * n][frozen_th] = 0.0
    if move_x is not None:
        lo[0:n][move_x] = 0.0
        hi[0:n][move_x] = 0.0
    if move_y is not None:
        lo[n:2 * n][move_y] = 0.0
        hi[n:2 * n][move_y] = 0.0

    b = B[:r] / trust
    bounds = np.column_stack([lo, hi])
    res = linprog(
        c,
        A_ub=A,
        b_ub=b,
        bounds=bounds,
        method="highs-ds",
        options=dict(primal_feasibility_tolerance=1e-9, dual_feasibility_tolerance=1e-9),
    )
    if res.status != 0:
        res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="highs")
        if res.status != 0:
            return None
    z = res.x * trust
    if move_x is not None:
        z[:n][move_x] = z[3 * n]
    if move_y is not None:
        z[n:2 * n][move_y] = z[3 * n]
    return z[:n], z[n:2 * n], z[2 * n:3 * n], z[3 * n], z[3 * n + 1], r


def viol(X, Y, T, w, h, f):
    mp, mw = feas(X, Y, T, w + f, h + f)
    return max(0.0, -min(mp, mw))


def squeeze(
    X,
    Y,
    T,
    w,
    h,
    f0,
    trust0=0.02,
    trust_max=0.05,
    trust_min=2e-8,
    max_it=400,
    verbose=False,
    frozen=None,
    f_floor=-1.0,
    center=0.0,
    center_until=1e-4,
    moving=None,
    move_x=None,
    move_y=None,
    frozen_x=None,
    frozen_y=None,
    frozen_th=None,
):
    """Minimise f for a (w+f) x (h+f) rectangle. Returns (f, X, Y, T, violation, iterations)."""
    X = X.copy()
    Y = Y.copy()
    T = T.copy()
    f = float(f0)
    trust = trust0
    it = 0
    v = viol(X, Y, T, w, h, f)
    start = (f, X.copy(), Y.copy(), T.copy(), v)
    lp_kw = dict(
        frozen=frozen,
        moving=moving,
        move_x=move_x,
        move_y=move_y,
        frozen_x=frozen_x,
        frozen_y=frozen_y,
        frozen_th=frozen_th,
    )
    while it < max_it and trust >= trust_min:
        it += 1
        out = lp_step(
            X,
            Y,
            T,
            w,
            h,
            f,
            trust,
            df_lo=max(-trust, f_floor - f),
            center=center if trust >= center_until else 0.0,
            **lp_kw,
        )
        if out is None:
            trust *= 0.5
            continue
        dx, dy, dt, df, t, rows = out
        Xn = X + dx
        Yn = Y + dy
        Tn = T + dt
        fn = f + df
        vn = viol(Xn, Yn, Tn, w, h, fn)
        if vn > max(8.0 * trust * trust, 1e-13) and vn > 0.97 * v + 1e-13:
            trust *= 0.5
            if verbose:
                print(f"  it {it} reject: v {vn:.2e} trust -> {trust:.1e}")
            continue
        progress = f - fn
        X, Y, T, f, v = Xn, Yn, Tn, fn, vn
        if verbose:
            print(f"  it {it} f={f:.9f} df={df:+.2e} v={v:.1e} t={t:.1e} trust={trust:.1e} rows={rows}")
        if progress > 0.3 * trust:
            trust = min(trust_max, trust * 1.5)
        elif v > 4.0 * trust * trust and v > 1e-9:
            pass
        else:
            trust *= 0.5

    for _ in range(12):
        if v <= 1e-13:
            break
        tr = min(0.05, max(4.0 * math.sqrt(v), 1e-7))
        out = lp_step(X, Y, T, w, h, f, tr, df_lo=0.0, df_hi=tr, **lp_kw)
        if out is None:
            break
        dx, dy, dt, df, t, rows = out
        X = X + dx
        Y = Y + dy
        T = T + dt
        f = f + df
        v = viol(X, Y, T, w, h, f)

    if start[4] <= 1e-13 and (v > 1e-12 or f >= start[0]):
        f, X, Y, T, v = start
    return f, X, Y, T, v, it


def box_of(X, Y, T):
    e = 0.5 * (np.abs(np.cos(T)) + np.abs(np.sin(T)))
    return (X - e).min(), (X + e).max(), (Y - e).min(), (Y + e).max()
