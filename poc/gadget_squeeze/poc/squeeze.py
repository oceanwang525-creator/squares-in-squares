"""Free-angle squeeze of an N-square pocket into a (w+f) x (h+f) rectangle, minimising f.

A gadget (w, h, +g) is a (w+f) x (h+f) box holding w*h + g unit squares; the composition DP
(poc/dp.py) pays f once per box side, so every 1e-3 shaved off f moves whole families of n.

Pipeline per start (centres AND angles free throughout):

  1. SEEDS (`make_seeds`): the extracted incumbent pocket plus mirror / transpose / rotated
     variants, an upright grid, tilted row-blocks over a family of angles (rows along the
     angle, per-row slides = the "meshed end" degree of freedom the tracked pockets use),
     and random sequential adsorption.

  2. COMPRESS-AND-RELAX (`compress`): drop the seed in a deliberately loose box
     (w+1.4) x (h+1.4) and minimise the repo's exact SAT overlap penalty -- a_overlap.py,
     copied as sat.py: vectorised separating-axis penetration squared, analytic gradients --
     plus a rectangle wall term, with L-BFGS. Then step f down (side continuation),
     rescaling the state and re-relaxing with jittered retries until a relaxation no longer
     reaches zero violation. Each rung is a full global rearrangement, not a polish.

  3. EXACT POLISH (`polish`): pslp_np.py (pocket_squeeze's sequential-LP solver, numba
     stripped) linearises every near-active separating axis and shrinks f inside a trust
     region, ending with an infeasible-path restoration to ~1e-15.

A candidate counts only if an independent SAT check (`violation`, sat.overlap_pairs) reports
max(pair, wall) penetration <= 1e-12 in the (w+f) x (h+f) box and the bounding box really
fits. Every returned f is therefore a realised packing, not a penalty-function near miss.

usage:
  squeeze.py --shape 9,15,3 [--starts 24] [--procs 24] [--iters 2000]
             [--f-start 0.608713] [--out ../data/squeezed.json]
"""
import argparse, json, math, os, sys, time
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sat                    # noqa: E402  repo SAT overlap penalty (copy of k2p10_fill2/poc/a_overlap.py)
import pslp_np as pslp        # noqa: E402  SLP squeeze (numba stripped)

HP = math.pi / 2.0
TOL = 1e-12
EFEAS = 1e-18                 # penalty energy accepted as "feasible enough" for the SLP polish


# ---------------------------------------------------------------- geometry helpers
def extents(T):
    return 0.5 * (np.abs(np.cos(T)) + np.abs(np.sin(T)))


def bbox(q):
    e = extents(q[:, 2])
    return (q[:, 0] - e).min(), (q[:, 0] + e).max(), (q[:, 1] - e).min(), (q[:, 1] + e).max()


def f_of(q, w, h):
    x0, x1, y0, y1 = bbox(q)
    return max(x1 - x0 - w, y1 - y0 - h)


def wall_violation_rect(q, W, H):
    """Per-square penetration of the box walls (sat.wall_violation, rectangle version)."""
    h = extents(q[:, 2])
    lo = np.maximum(0, h[:, None] - q[:, :2])
    hi = np.maximum(0, q[:, :2] + h[:, None] - np.array([W, H]))
    return lo + hi


def violation(q, W, H):
    """max(pair penetration, wall penetration) with the repo's vectorised SAT."""
    P, dep = sat.overlap_pairs(q, tol=0.0)
    v = float(dep.max()) if len(dep) else 0.0
    wv = wall_violation_rect(q, W, H)
    return max(v, float(wv.max()) if wv.size else 0.0)


def place(q, W, H):
    """Translate (and, if needed, isotropically crush) a seed into [0,W] x [0,H]."""
    q = q.copy()
    x0, x1, y0, y1 = bbox(q)
    s = min(1.0, W / max(x1 - x0, 1e-12), H / max(y1 - y0, 1e-12))
    q[:, 0] = (q[:, 0] - x0) * s
    q[:, 1] = (q[:, 1] - y0) * s
    x0, x1, y0, y1 = bbox(q)
    q[:, 0] += (W - (x1 - x0)) / 2 - x0
    q[:, 1] += (H - (y1 - y0)) / 2 - y0
    return q


# ---------------------------------------------------------------- stage 2: penalty relaxation
def penalty_rect(q, W, H, w_wall=1.0):
    """E = sum_pairs depth^2 + w_wall * sum wall^2, with analytic gradient wrt (x, y, theta)."""
    P = sat.pairs(q)
    E = 0.0
    G = np.zeros_like(q)
    if len(P):
        dep, gpi, gpj, dti, dtj = sat.pair_depth(q, P, grad=True)
        m = dep > 0
        if m.any():
            e = dep[m]
            E += float(np.sum(e * e))
            Pm = P[m]
            f = 2 * e
            np.add.at(G[:, 0], Pm[:, 0], f * gpi[m, 0]); np.add.at(G[:, 1], Pm[:, 0], f * gpi[m, 1])
            np.add.at(G[:, 0], Pm[:, 1], f * gpj[m, 0]); np.add.at(G[:, 1], Pm[:, 1], f * gpj[m, 1])
            np.add.at(G[:, 2], Pm[:, 0], f * dti[m]); np.add.at(G[:, 2], Pm[:, 1], f * dtj[m])
    h = extents(q[:, 2])
    lo = np.maximum(0, h[:, None] - q[:, :2])
    hi = np.maximum(0, q[:, :2] + h[:, None] - np.array([W, H]))
    E += w_wall * float(np.sum(lo * lo + hi * hi))
    if w_wall:
        G[:, :2] += w_wall * 2 * (hi - lo)
        c, s = np.cos(q[:, 2]), np.sin(q[:, 2])
        dh = 0.5 * (-np.sign(c) * s + np.sign(s) * c)
        G[:, 2] += w_wall * 2 * dh * (np.sum(lo, 1) + np.sum(hi, 1))
    return E, G


def lbfgs(fun, z0, iters=500, m=10, tol=1e-11, step_cap=0.25, seed=0):
    """Plain L-BFGS with an Armijo backtracking line search and kink restarts.

    scipy's L-BFGS-B costs ~20 ms per iteration in this build regardless of problem size
    (measured: 63 variables, 300 iterations, 6-19 s, E still 2.0), and its dense BFGS costs
    0.3 s/iteration at 636 variables. A numpy two-loop recursion is ~2 ms per iteration at
    that size. The SAT penalty is only piecewise smooth (the separating axis is a min over 4
    branches), so a failed line search restarts the memory from a tiny random perturbation
    instead of stopping: without that the relaxation parks at E~1e-7.
    """
    rng = np.random.default_rng(seed)
    z = np.array(z0, float)
    E, G = fun(z)
    S, Y = [], []
    fails = 0
    for it in range(iters):
        # ---- two-loop recursion
        qv = G.copy()
        al = []
        for s, y in zip(reversed(S), reversed(Y)):
            rho = 1.0 / (y @ s)
            a = rho * (s @ qv)
            qv -= a * y
            al.append(a)
        if S:
            s, y = S[-1], Y[-1]
            qv *= (s @ y) / (y @ y)
            for (s, y), a in zip(zip(S, Y), reversed(al)):
                rho = 1.0 / (y @ s)
                b = rho * (y @ qv)
                qv += s * (a - b)
        else:
            qv *= 1e-2
        d = -qv
        dmax = float(np.abs(d).max())
        if dmax > step_cap:
            d *= step_cap / dmax
        gd = float(G @ d)
        if gd >= 0:
            d = -G
            dmax = float(np.abs(d).max())
            if dmax > step_cap:
                d *= step_cap / dmax
            gd = float(G @ d)
        step = 1.0
        ok = False
        for _ in range(50):
            zn = z + step * d
            En, Gn = fun(zn)
            if En <= E + 1e-4 * step * gd:
                ok = True
                break
            step *= 0.5
        if not ok:
            fails += 1
            if fails > 40 or E < 1e-26:
                break
            # kink: restart the memory and jitter a little
            S, Y = [], []
            z = z + rng.normal(0, 1e-5, z.shape)
            E, G = fun(z)
            continue
        s = zn - z; y = Gn - G
        if s @ y > 1e-16:
            S.append(s); Y.append(y)
            if len(S) > m:
                S.pop(0); Y.pop(0)
        z, E, G = zn, En, Gn
        if np.abs(G).max() < tol and E < 1e-20:
            break
    return z, E


def relax(q, W, H, iters=500, tol=1e-11):
    """L-BFGS on the SAT penalty inside a W x H box. Returns (q, E)."""
    def fun(z):
        E, G = penalty_rect(z.reshape(-1, 3), W, H)
        return E, G.ravel()

    z, E = lbfgs(fun, np.ascontiguousarray(q).ravel(), iters=iters, tol=tol)
    out = z.reshape(-1, 3).copy()
    out[:, 2] = np.mod(out[:, 2], HP)          # square symmetry
    return out, float(E)


# ---------------------------------------------------------------- stage 3: exact SLP polish
def polish(q, w, h, max_it=300, trust0=0.02):
    """pslp squeeze from a state that already (nearly) fits; returns (f, q, viol)."""
    f0 = max(f_of(q, w, h), 0.0)
    qq = place(q, w + f0, h + f0)
    f, X, Y, T, v, it = pslp.squeeze(qq[:, 0].copy(), qq[:, 1].copy(), qq[:, 2].copy(),
                                     float(w), float(h), f0, max_it=max_it, trust0=trust0)
    out = place(np.column_stack([X, Y, T]), w + max(f, 0.0), h + max(f, 0.0))
    return float(f), out, v


def certified(q, w, h, f):
    """Independent SAT check in the (w+f) x (h+f) box, and the side must really fit."""
    W, H = w + f, h + f
    v = violation(q, W, H)
    x0, x1, y0, y1 = bbox(q)
    fits = (x1 - x0 <= W + 1e-9) and (y1 - y0 <= H + 1e-9)
    return (v <= TOL and fits), v


# ---------------------------------------------------------------- side continuation
def compress(seed, w, h, iters=2000, f0=1.4, fmin=0.15, tries=3, step0=0.06, min_step=5e-5,
             rng=None):
    """Loose box -> relax -> shrink f in halving steps with jittered retries. Returns (f, q)."""
    rng = rng or np.random.default_rng()
    n = len(seed)
    q = place(np.asarray(seed, float), w + f0, h + f0)
    q, E = relax(q, w + f0, h + f0, iters)
    if not np.isfinite(E) or E > EFEAS:
        return None
    f = f0
    best = (f, q.copy())
    step = step0
    while step > min_step and f - step > fmin:
        ft = f - step
        sx = (w + ft) / (w + f); sy = (h + ft) / (h + f)
        ok = False
        for t in range(tries):
            Y = q.copy()
            Y[:, 0] *= sx; Y[:, 1] *= sy
            if t:
                sig = 0.01 * t
                Y += rng.normal(0, sig, (n, 3))
            Y, E = relax(Y, w + ft, h + ft, iters)
            if np.isfinite(E) and E < EFEAS:
                ok = True
                break
        if ok:
            q, f = Y, ft
            best = (f, q.copy())
            step = min(step * 1.3, step0)
        else:
            step *= 0.5
    return best


# ---------------------------------------------------------------- stage 1: seeds
def seed_grid(N, w, h, rng):
    cols = min(w, max(1, int(round(math.sqrt(N * w / max(h, 1))))))
    rows = int(math.ceil(N / cols))
    pts = [(c + 0.5, r + 0.5, 0.0) for r in range(rows) for c in range(cols)][:N]
    return np.array(pts, float)


def seed_block(N, w, h, theta, rng, pitch=1.0, jitter=1.0):
    """Tilted lattice of rows along theta; per-row slide = the meshed-end freedom."""
    ct, st = math.cos(theta), math.sin(theta)
    u = np.array([ct, st]); vv = np.array([-st, ct])
    L = max(w, h) + 2.0
    nrow_max = int(L / pitch) + 3
    pts = []
    r = 0
    while len(pts) < N and r < 3 * (w + h):
        shift = rng.uniform(-0.5, 0.5) * jitter
        for c in range(nrow_max):
            if len(pts) >= N:
                break
            if rng.random() < 0.03:
                continue
            p = u * (c * pitch + shift) + vv * (r * pitch)
            pts.append((p[0], p[1], theta))
        r += 1
    while len(pts) < N:
        pts.append((rng.uniform(0, w), rng.uniform(0, h), theta))
    return np.array(pts[:N], float)


def seed_rsa(N, w, h, theta, rng, rmin=1.32):
    q = []
    tries = 0
    while len(q) < N and tries < 300 * N:
        tries += 1
        p = np.array([rng.uniform(0, w), rng.uniform(0, h),
                      theta if rng.random() < 0.6 else rng.uniform(0, HP)])
        if not q:
            q.append(p); continue
        a = np.array(q)
        if np.hypot(a[:, 0] - p[0], a[:, 1] - p[1]).min() > rmin:
            q.append(p)
    while len(q) < N:
        q.append(np.array([rng.uniform(0, w), rng.uniform(0, h), theta]))
    return np.array(q[:N], float)


ANGLES = [45.0, 40.6, 36.87, 33.69, 30.96, 30.0, 26.57, 24.5, 22.62, 21.8, 19.0, 18.43, 16.7, 14.04, 12.53, 9.46]


def make_seeds(N, w, h, pocket=None, n_starts=32, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    if pocket is not None:
        base = np.asarray(pocket, float)
        if len(base) == N:
            out.append(('pocket', base.copy()))
            b = base.copy()
            x0, x1 = b[:, 0].min(), b[:, 0].max()
            b[:, 0] = (x0 + x1) - b[:, 0]
            out.append(('pocket-mx', b))
            out.append(('pocket-t', np.column_stack([base[:, 1], base[:, 0], HP - base[:, 2]])))
            b = base.copy(); b[:, 2] = np.mod(b[:, 2] + 0.02, HP)
            out.append(('pocket-r', b))
    out.append(('grid', seed_grid(N, w, h, rng)))
    for a in ANGLES:
        out.append((f'block{a:.1f}', seed_block(N, w, h, math.radians(a), rng)))
    i = 0
    while len(out) < n_starts:
        a = math.radians(ANGLES[i % len(ANGLES)] + rng.normal(0, 1.5))
        k = i // len(ANGLES)
        if k % 3 == 0:
            out.append((f'blk{i}', seed_block(N, w, h, a, rng, pitch=rng.uniform(0.92, 1.08),
                                              jitter=rng.uniform(0.2, 1.0))))
        elif k % 3 == 1:
            out.append((f'rsa{i}', seed_rsa(N, w, h, a, rng)))
        else:
            out.append((f'blkh{i}', seed_block(N, w, h, a, rng, pitch=rng.uniform(0.98, 1.02),
                                               jitter=0.15)))
        i += 1
    return out[:n_starts]


# ---------------------------------------------------------------- one start
def run_job(job):
    name, q, N, w, h, f_cap, iters, seed = job
    t0 = time.time()
    rng = np.random.default_rng(seed)
    try:
        res = compress(np.asarray(q, float)[:N], w, h, iters=iters, rng=rng)
        if res is None:
            return name, None, time.time() - t0, None
        f, qc = res
        out = None
        # exact polish (SLP) -- cheap for the shapes we care about
        if N <= 1200:
            f2, q2, _ = polish(qc, w, h)
            ok, v = certified(q2, w, h, f2)
            if ok and f2 <= f + 1e-9:
                out = ('slp', f2, q2, v)
        if out is None:
            ok, v = certified(qc, w, h, f)
            if ok:
                out = ('relax', f, qc, v)
        if out is None:
            return name, None, time.time() - t0, None
        return name, out, time.time() - t0, None
    except Exception as e:
        return name, None, time.time() - t0, repr(e)


def squeeze_shape(w, h, g, N=None, pocket=None, starts=32, procs=24, seed=0, iters=2000,
                  verbose=True):
    """Multi-start compress-and-relax. Returns (best f, q, provenance) or (None, None, None)."""
    N = N or (w * h + g)
    seeds = make_seeds(N, w, h, pocket, n_starts=starts, seed=seed)
    jobs = [(nm, q, N, w, h, None, iters, seed * 1000 + i) for i, (nm, q) in enumerate(seeds)]
    best = None
    done = 0
    with Pool(procs) as pool:
        for nm, res, dt, err in pool.imap_unordered(run_job, jobs):
            done += 1
            if err and verbose:
                print(f'    [{nm}] error {err}', flush=True)
            if res:
                kind, f, q, v = res
                if verbose:
                    print(f'    [{nm}] {kind} f={f:.9f} viol={v:.1e} ({dt:.0f}s)', flush=True)
                if best is None or f < best[0] - 1e-12:
                    best = (f, q, f'{nm}/{kind}')
    if verbose:
        if best is None:
            print(f'({w},{h},+{g}) N={N}: no feasible packing found in {done} starts', flush=True)
        else:
            print(f'({w},{h},+{g}) N={N}: best f={best[0]:.9f} from {best[2]}', flush=True)
    return best if best else (None, None, None)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--shape', required=True, help='w,h,g')
    ap.add_argument('--n', type=int, default=None)
    ap.add_argument('--seed-file', default='../data/targets.json')
    ap.add_argument('--seed-key', default=None)
    ap.add_argument('--starts', type=int, default=32)
    ap.add_argument('--procs', type=int, default=24)
    ap.add_argument('--iters', type=int, default=2000)
    ap.add_argument('--f-start', type=float, default=None, help='incumbent f (default: seed file)')
    ap.add_argument('--no-seed', action='store_true')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='../data/squeezed.json')
    a = ap.parse_args()
    w, h, g = (int(x) for x in a.shape.split(','))
    N = a.n or (w * h + g)
    pocket, f_inc = None, a.f_start
    if not a.no_seed and os.path.exists(a.seed_file):
        d = json.load(open(a.seed_file))
        for k, v in d.items():
            if a.seed_key and k != a.seed_key:
                continue
            if (v['w'], v['h'], v['g']) == (w, h, g) or (v['h'], v['w'], v['g']) == (w, h, g):
                sq = np.asarray(v['sq'], float)
                if v['w'] != w:
                    sq = np.column_stack([sq[:, 1], sq[:, 0], HP - sq[:, 2]])
                if len(sq) == N:
                    pocket = sq
                    f_inc = f_inc if f_inc is not None else v['f']
                    print(f'seed pocket from {k} (src n={v.get("src")}) f={v["f"]:.6f}', flush=True)
                break
    print(f'shape ({w},{h},+{g}) N={N} incumbent f='
          f'{f_inc if f_inc is not None else float("nan"):.6f} starts={a.starts}', flush=True)
    t0 = time.time()
    f, q, prov = squeeze_shape(w, h, g, N=N, pocket=pocket, starts=a.starts, procs=a.procs,
                               seed=a.seed, iters=a.iters)
    print(f'wall {time.time()-t0:.0f}s', flush=True)
    if f is not None:
        out = json.load(open(a.out)) if os.path.exists(a.out) else {}
        key = f'{w}x{h}g{g}'
        old = out.get(key, {}).get('f', 1e9)
        gain = (f_inc - f) if f_inc is not None else None
        if f < old:
            out[key] = dict(w=w, h=h, g=g, f=float(f), sq=q.tolist(), src=prov, n=N,
                            prev=None if old > 1e8 else old, incumbent=f_inc)
            json.dump(out, open(a.out, 'w'))
        print(f'{key}: found {f:.9f}; incumbent best '
              f'{min(f_inc, old) if f_inc is not None else old:.9f}; '
              f'{"IMPROVED by %.6f" % gain if gain and gain > 0 else "no improvement"}')
