"""Global stochastic search at mid n. usage: gsearch.py n budget_s seed"""
import sys, math, time, json, numpy as np, pandas as pd
from scipy.optimize import minimize
sys.path.insert(0, '/Users/lucaszhang/squares-in-squares/poc/k2p10_fill2/poc')
from a_overlap import penalty, overlap_pairs, bbox_side, pair_depth, wall_violation
ROOT = '/Users/lucaszhang/squares-in-squares/'
n = int(sys.argv[1]); budget = float(sys.argv[2]); seed = int(sys.argv[3])
rng = np.random.default_rng(seed)
m = pd.read_csv(ROOT + 'poc/pell_corrugation/data/border-scan/merged.csv')
inc = m[m.n == n][['x', 'y', 'angle']].to_numpy(float).copy()
h =lambda a: 0.5 * (np.abs(np.cos(a)) + np.abs(np.sin(a)))
inc[:, 0] -= (inc[:, 0] - h(inc[:, 2])).min(); inc[:, 1] -= (inc[:, 1] - h(inc[:, 2])).min()
Lours = bbox_side(inc)
Lt = Lours - 0.001

def soft(q, L, it=400):
    f = lambda v: penalty(v.reshape(-1, 3), L)
    def fg(v):
        E, G = penalty(v.reshape(-1, 3), L); return E, G.ravel()
    r = minimize(fg, q.ravel(), jac=True, method='L-BFGS-B', options=dict(maxiter=it, ftol=1e-16, gtol=1e-12))
    return r.x.reshape(-1, 3), r.fun

def harden(q):
    """Exact repair: recentre, scale positions until no overlap. returns (q, bbox L) or None."""
    q = q.copy(); s = 1.0
    for _ in range(60):
        q2 = q.copy(); c = q[:, :2].mean(0); q2[:, :2] = c + (q[:, :2] - c) * s
        P, dep = overlap_pairs(q2)
        if len(P) == 0:
            q2[:, 0] -= (q2[:, 0] - h(q2[:, 2])).min(); q2[:, 1] -= (q2[:, 1] - h(q2[:, 2])).min()
            return q2, bbox_side(q2)
        s *= 1 + max(dep.max(), 1e-12) * 0.6
    return None

def metropolis(q, L, steps, T0, T1):
    """single-square translate/rotate moves with local energy via cell list; swap-to-hole move."""
    N = len(q); cs = math.sqrt(2); G = int(L / cs) + 2
    cell = lambda x, y: (min(max(int(x / cs), 0), G - 1), min(max(int(y / cs), 0), G - 1))
    grid = {}
    for i in range(N): grid.setdefault(cell(q[i, 0], q[i, 1]), set()).add(i)
    def loc(i, x, y, a):
        cx, cy = cell(x, y); idx = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                idx.extend(grid.get((cx + dx, cy + dy), ()))
        idx = [j for j in idx if j != i]
        hh = h(a); w = sum(max(0, hh - v) ** 2 + max(0, v + hh - L) ** 2 for v in (x, y))
        if not idx: return w
        J = np.array(idx); qq = np.empty((len(J) + 1, 3)); qq[0] = (x, y, a); qq[1:] = q[J]
        P = np.stack([np.zeros(len(J), int), np.arange(1, len(J) + 1)], 1)
        d = pair_depth(qq, P); return w + float(np.sum(np.maximum(d, 0) ** 2))
    cur = np.array([loc(i, *q[i]) for i in range(N)])
    for t in range(steps):
        T = T0 * (T1 / T0) ** (t / steps)
        i = rng.integers(N)
        if cur[i] == 0 and rng.random() < 0.7: i = rng.integers(N)
        x, y, a = q[i]
        r = rng.random()
        step = 0.3 * (T / T0) ** 0.5 + 0.01
        if r < 0.6: x += rng.normal(0, step); y += rng.normal(0, step)
        elif r < 0.9: a += rng.normal(0, step * 2)
        else: x += rng.normal(0, step); y += rng.normal(0, step); a += rng.normal(0, step * 2)
        e0 = cur[i]; e1 = loc(i, x, y, a)
        # total delta counts pair energy twice (both squares); fine as heuristic
        if e1 <= e0 or rng.random() < math.exp(-(e1 - e0) / T):
            grid[cell(q[i, 0], q[i, 1])].discard(i)
            q[i] = (x, y, a); grid.setdefault(cell(x, y), set()).add(i); cur[i] = e1
    return q

def structured_band(L):
    """square lattice rows with a random tilted band of angle th and random jitter"""
    k = int(L); q = []
    th = rng.uniform(0.05, 0.6)
    b0 = rng.uniform(0.2, 0.7) * L; bw = rng.uniform(0.15, 0.4) * L
    pts = [(i + .5, j + .5) for i in range(k) for j in range(k)]
    sp = L / k
    pts = [(p[0] * sp, p[1] * sp) for p in pts]
    # extras fill randomly
    while len(pts) < n: pts.append((rng.uniform(0.5, L - .5), rng.uniform(0.5, L - .5)))
    pts = pts[:n] if len(pts) > n else pts
    for (x, y) in pts:
        a = th * rng.choice([-1, 1]) if abs(x + y - b0) < bw else rng.normal(0, 0.02)
        q.append((x, y, a))
    return np.array(q) + rng.normal(0, 0.01, (len(q), 3)) * [1, 1, 0]

best = {}; log = []
t0 = time.time(); k = 0
while time.time() - t0 < budget:
    mode = 'abc'[k % 3]; k += 1
    if mode == 'a':
        q = inc.copy(); c = q[:, :2].mean(0); q[:, :2] = c + (q[:, :2] - c) * 1.02
        sel = rng.random(n) < 0.5; q[sel, 2] = rng.uniform(0, math.pi, sel.sum())
        q[:, :2] += (Lt / 2 - c)
        q, E = soft(q, Lt, 200)
    elif mode == 'b':
        q = structured_band(Lt); q, E = soft(q, Lt, 300)
    else:
        q = np.c_[rng.uniform(0.5, Lt - .5, (n, 2)), rng.uniform(0, math.pi, n)]
        q = metropolis(q, Lt * 1.03, 40000, 0.05, 0.0005)
        q, E = soft(q, Lt, 300)
    # a few Metropolis polish + soft rounds if not yet near zero
    for _ in range(2):
        if E < 1e-9: break
        q = metropolis(q, Lt, 15000, 0.002, 0.00005); q, E = soft(q, Lt, 300)
    P, dep = overlap_pairs(q); wv, _ = wall_violation(q, Lt)
    res = harden(q) if E < 0.1 else None
    rec = dict(mode=mode, E=E, maxdep=float(dep.max()) if len(dep) else 0.0, npairs=len(P),
               Lhard=res[1] if res else None, t=time.time() - t0,
               hist=np.histogram(np.mod(q[:, 2], math.pi / 2), bins=6, range=(0, math.pi / 2))[0].tolist())
    log.append(rec); print(json.dumps(rec), flush=True)
    if res and res[1] < best.get(mode, (9e9,))[0]: best[mode] = (res[1], res[0])
    if mode not in best or E < best.get('E' + mode, (9e9,))[0] if False else False: pass
out = dict(n=n, Lours=Lours, Lt=Lt, log=log, best_L={k: v[0] for k, v in best.items()})
json.dump(out, open(ROOT + f'poc/global_search_midn/data/run_{n}_{seed}.json', 'w'))
if best:
    bk = min(best, key=lambda k: best[k][0]); np.save(ROOT + f'poc/global_search_midn/data/best_{n}.npy', best[bk][1])
