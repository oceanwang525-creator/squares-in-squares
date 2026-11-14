"""Batched numpy optimiser for `a*b+g` unit squares in an `(a+f) x (b+f)` box.

Minimise f with all centres AND angles free.  No numba needed.

Engine
------
* penalty: `satb.energy_grad` = the repo's exact SAT penalty
  (`poc/k2p10_fill2/poc/a_overlap.py`) batched over R independent restarts,
  analytic gradients, no per-square python loops;
* descent: normalised-gradient descent with momentum and a *per-restart
  backtracking line search* (one gradient evaluation per iteration).  Adam was
  tried first and diverges/stalls here: normalising every coordinate makes the
  theta components overshoot, whereas the per-restart line search converges to
  E ~ 0 exactly when the box is feasible;
* continuation: every restart carries its own box excess `f_r`, an adaptive
  shrink step and a "best feasible" anchor.  Feasible restarts shrink, jammed
  ones are restored from the anchor with a jitter (parallel basin hopping),
  restarts that never became feasible are re-seeded in a looser box.

`polish.py` then polishes the deepest states with the repo SLP
(`poc/gadget_squeeze/poc/pslp_np.py`) and re-certifies with the exact SAT check.
"""
import math
import time

import numpy as np

import satb


# ------------------------------------------------------------------ inits
def lattice_init(N, a, b, f, rng, tilt=0.0, noise=0.06):
    """Q (R,N,3): k x m lattice filling the (a+f_r) x (b+f_r) boxes, jittered."""
    R = f.shape[0]
    k = int(round(math.sqrt(N * a / max(b, 1.0))))
    k = max(1, min(N, k))
    m = int(math.ceil(N / k))
    W = a + f
    H = b + f
    Q = np.empty((R, N, 3))
    idx = np.arange(N)
    col = idx % k
    row = idx // k
    Q[:, :, 0] = (col[None, :] + 0.5) * (W[:, None] / k)
    Q[:, :, 1] = (row[None, :] + 0.5) * (H[:, None] / m)
    Q[:, :, 0] += rng.normal(0.0, 1.0, (R, N)) * noise * (W / k)[:, None]
    Q[:, :, 1] += rng.normal(0.0, 1.0, (R, N)) * noise * (H / m)[:, None]
    Q[:, :, 2] = tilt + rng.normal(0.0, 0.10, (R, N))
    return Q


def tilted_lattice_init(N, a, b, f, rng, theta=None, noise=0.05):
    """Q (R,N,3): a *genuinely* rotated square lattice (spacing 1), isotropically
    scaled to the box (so it is slightly compressed) -- the standard dense seed."""
    R = f.shape[0]
    W = a + f
    H = b + f
    k = int(round(math.sqrt(N * a / max(b, 1.0))))
    k = max(1, min(N, k))
    m = int(math.ceil(N / k))
    if theta is None:
        theta = rng.uniform(-0.9, 0.9, R)
    Q = np.empty((R, N, 3))
    idx = np.arange(N)
    col = idx % k
    row = idx // k
    th = np.asarray(theta, float)
    c = np.cos(th)[:, None]
    s = np.sin(th)[:, None]
    px = col[None, :] * c - row[None, :] * s          # lattice coords, spacing 1
    py = col[None, :] * s + row[None, :] * c
    e = np.abs(c) + np.abs(s)                          # bbox of one square
    w = px.max(1) - px.min(1) + e[:, 0]
    h = py.max(1) - py.min(1) + e[:, 0]
    sc = np.minimum(W / w, H / h) * rng.uniform(0.995, 1.02, R)
    px = px * sc[:, None]
    py = py * sc[:, None]
    # centre in the box
    px -= px.min(1)[:, None]
    py -= py.min(1)[:, None]
    px += 0.5 * ((W - (px.max(1) - px.min(1)))[:, None])
    py += 0.5 * ((H - (py.max(1) - py.min(1)))[:, None])
    Q[:, :, 0] = px + rng.normal(0.0, 1.0, (R, N)) * noise
    Q[:, :, 1] = py + rng.normal(0.0, 1.0, (R, N)) * noise
    Q[:, :, 2] = th[:, None] + rng.normal(0.0, 0.05, (R, N))
    return Q


def rsa_init(N, a, b, f, rng, trials=8):
    """best-candidate random sequential adsorption, vectorised over restarts."""
    R = f.shape[0]
    W = a + f
    H = b + f
    xx = np.empty((R, N)); yy = np.empty((R, N)); tt = np.empty((R, N))
    hh = np.empty((R, N))
    for k in range(N):
        cx = rng.uniform(0.0, 1.0, (R, trials)) * W[:, None]
        cy = rng.uniform(0.0, 1.0, (R, trials)) * H[:, None]
        ca = rng.normal(0.0, 0.25, (R, trials))
        if k > 0:
            dx = cx[:, :, None] - xx[:, None, :k]
            dy = cy[:, :, None] - yy[:, None, :k]
            e = 0.5 * (np.abs(np.cos(ca)) + np.abs(np.sin(ca)))
            E = np.maximum(e[:, :, None], hh[:, None, :k])
            clr = np.maximum(np.abs(dx), np.abs(dy)) - E
            best = np.argmax(clr.min(2), 1)
            ar = np.arange(R)
            xx[:, k] = cx[ar, best]; yy[:, k] = cy[ar, best]; tt[:, k] = ca[ar, best]
        else:
            xx[:, 0] = cx[:, 0]; yy[:, 0] = cy[:, 0]; tt[:, 0] = ca[:, 0]
        hh[:, k] = 0.5 * (np.abs(np.cos(tt[:, k])) + np.abs(np.sin(tt[:, k])))
    Q = np.empty((R, N, 3))
    Q[:, :, 0] = xx; Q[:, :, 1] = yy; Q[:, :, 2] = tt
    return Q


def seed(N, a, b, f, rng, kind=None):
    """mixed init families; `kind` None -> random choice per row."""
    R = f.shape[0]
    Q = np.empty((R, N, 3))
    which = rng.integers(0, 4, R) if kind is None else np.full(R, kind)
    for w in range(4):
        mk = which == w
        if not mk.any():
            continue
        if w == 0:
            Q[mk] = lattice_init(N, a, b, f[mk], rng, tilt=0.0)
        elif w == 1:
            Q[mk] = lattice_init(N, a, b, f[mk], rng,
                                 tilt=rng.uniform(-0.55, 0.55), noise=0.12)
        elif w == 2:
            Q[mk] = tilted_lattice_init(N, a, b, f[mk], rng)
        else:
            Q[mk] = rsa_init(N, a, b, f[mk], rng)
    return Q


# ------------------------------------------------------------------ engine
class Run:
    def __init__(self, a, b, g, f0, R=1024, seed_=0, mu=0.6, sp0=0.02, st0=0.03,
                 tolE=1e-11, jitter=0.03, verbose=True, f_floor=-10.0):
        self.a, self.b, self.g = a, b, g
        self.N = a * b + g
        self.R = R
        self.rng = np.random.default_rng(seed_)
        self.mu, self.tolE, self.jitter = mu, tolE, jitter
        self.verbose = verbose
        self.f_floor = f_floor
        self.I, self.J = satb.pair_list(self.N)
        rng = self.rng
        self.f = np.asarray(f0, float).copy() if np.ndim(f0) else np.full(R, float(f0))
        self.Q = seed(self.N, a, b, self.f, rng)
        self.sp = np.full(R, sp0)
        self.st = np.full(R, st0)
        self.vx = np.zeros((R, self.N, 2))
        self.vt = np.zeros((R, self.N, 1))
        self.E = np.full(R, np.inf)
        self.G = np.zeros((R, self.N, 3))
        self.best_f = np.full(R, np.inf)
        self.best_Q = self.Q.copy()
        self.delta = np.full(R, 0.0)
        self.archive = []          # (f, Q) deepest feasible states
        self.t0 = time.time()
        self.hist = []

    # ---- one normalised-GD step with per-restart backtracking (1 grad eval)
    def step(self):
        W = self.a + self.f
        H = self.b + self.f
        G = self.G
        gx = G[:, :, :2]
        gt = G[:, :, 2:3]
        nx = np.sqrt((gx * gx).sum((1, 2))) + 1e-30
        nt = np.sqrt((gt * gt).sum((1, 2))) + 1e-30
        self.vx = self.mu * self.vx + (1 - self.mu) * gx / nx[:, None, None]
        self.vt = self.mu * self.vt + (1 - self.mu) * gt / nt[:, None, None]
        Q2 = self.Q.copy()
        Q2[:, :, :2] -= self.sp[:, None, None] * self.vx
        Q2[:, :, 2:] -= self.st[:, None, None] * self.vt
        E2, G2 = satb.energy_grad_flat(Q2, W, H, *self.flat)
        acc = E2 < self.E
        self.Q[acc] = Q2[acc]
        self.E[acc] = E2[acc]
        self.G[acc] = G2[acc]
        self.sp[acc] = np.minimum(0.15, self.sp[acc] * 1.3)
        self.st[acc] = np.minimum(0.25, self.st[acc] * 1.3)
        bad = ~acc
        self.sp[bad] = np.maximum(1e-7, self.sp[bad] * 0.5)
        self.st[bad] = np.maximum(1e-7, self.st[bad] * 0.5)
        self.vx[bad] = 0.0
        self.vt[bad] = 0.0
        return self.E

    def relax(self, iters, margin=1.0):
        W = self.a + self.f
        H = self.b + self.f
        self.flat = satb.neighbours(self.Q, margin)
        self.E, self.G = satb.energy_grad_flat(self.Q, W, H, *self.flat)
        for _ in range(iters):
            self.step()
        # exact (all-pairs) energy for the feasibility gate
        Efull, _ = satb.energy_grad(self.Q, W, H, self.I, self.J, want_grad=False)
        self.E = Efull
        return self.E

    def bank(self):
        ok = self.E <= self.tolE
        if ok.any():
            self.best_f[ok] = np.minimum(self.best_f[ok], self.f[ok])
            self.best_Q[ok] = self.Q[ok]
            fo = self.f[ok]; Qo = self.Q[ok]
            for t in np.argsort(fo)[:4]:
                self.archive.append((float(fo[t]), Qo[t].copy()))
            self.f[ok] -= self.delta[ok]
            self.delta[ok] = np.minimum(0.08, self.delta[ok] * 1.5 + 0.004)
        bad = ~ok
        if bad.any():
            ib = bad.nonzero()[0]
            fb = self.best_f[ib]
            retry = np.isfinite(fb)
            nr = int(retry.sum())
            rng = self.rng
            if nr:
                ii = ib[retry]
                Qb = self.best_Q[ii].copy()
                Qb[:, :, 0] += rng.normal(0.0, self.jitter, (nr, self.N))
                Qb[:, :, 1] += rng.normal(0.0, self.jitter, (nr, self.N))
                Qb[:, :, 2] += rng.normal(0.0, self.jitter, (nr, self.N))
                self.Q[ii] = Qb
                self.delta[ii] = np.maximum(1.5e-3, self.delta[ii] * 0.6)
                self.f[ii] = np.maximum(self.f_floor, fb[retry] - self.delta[ii])
                self.sp[ii] = 0.02; self.st[ii] = 0.03
                self.vx[ii] = 0.0; self.vt[ii] = 0.0
            fresh = ~np.isfinite(fb)
            if fresh.any():
                ii = ib[fresh]
                if np.isfinite(self.best_f).any():
                    anchor = float(np.min(self.best_f))
                else:
                    anchor = float(self.f[ii].max())
                self.f[ii] = anchor + rng.uniform(0.15, 0.45, len(ii))
                self.Q[ii] = seed(self.N, self.a, self.b, self.f[ii], rng)
                self.sp[ii] = 0.02; self.st[ii] = 0.03
                self.vx[ii] = 0.0; self.vt[ii] = 0.0
            self.E[ii] = np.inf
            self.G[ii] = 0.0
        return int(ok.sum())

    def sweep(self, iters, sw=0):
        self.relax(iters)
        nok = self.bank()
        self.hist.append((sw, float(np.min(self.E)), float(np.median(self.E)),
                          float(self.f.min()), float(np.min(self.best_f)), nok))
        if self.verbose:
            s = self.hist[-1]
            print('  sw %2d  E min %.2e med %.2e | f min %.5f best_f %.5f | ok %d/%d %.1fs'
                  % (s[0], s[1], s[2], s[3], s[4], s[5], self.R, time.time() - self.t0),
                  flush=True)
        return nok

    def candidates(self, nmax=200):
        """deepest feasible states: per-restart anchors + archive, deduped by f."""
        cand = []
        ok = np.isfinite(self.best_f)
        if ok.any():
            ib = ok.nonzero()[0]
            for j in ib[np.argsort(self.best_f[ib])][:nmax]:
                cand.append((float(self.best_f[j]), self.best_Q[j].copy()))
        cand.extend(self.archive)
        cand.sort(key=lambda r: r[0])
        out = []
        for f, q in cand:
            if not out or f > out[-1][0] * (1 + 1e-6) + 1e-9:
                out.append((f, q))
            if len(out) >= nmax:
                break
        return out

    def report(self):
        cand = self.candidates(40)
        if cand:
            return dict(f=cand[0][0], Q=cand[0][1], all=cand,
                        best_f_all=float(np.min(self.best_f)))
        return dict(f=float('inf'), Q=None, all=[], best_f_all=float('inf'))
