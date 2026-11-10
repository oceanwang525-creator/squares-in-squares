import sys, time, math, json, numpy as np
import s2r, incumbent
d, L = incumbent.table()
def inc_z(n):
    g = d[d.n == n]; return np.column_stack([g.x, g.y, g.angle]).ravel().astype(float)

def one(n, mode, seed, p1=48.0, stages=30, sigma=0.05, shrink=0.0, log=print):
    rng = np.random.default_rng(seed); Li = L[n]
    Lend = Li - shrink
    if mode == 'random':
        L0 = Lend*1.08
        z = np.column_stack([rng.uniform(.5, L0-.5, (n, 2)), rng.uniform(-.8, .8, n)]).ravel()
    else:
        z = inc_z(n); q = z.reshape(n, 3)
        q[:, :2] -= q[:, :2].min(0)-0.5; q[:, :2] *= (Li*1.02)/ (q[:, :2].max()+.5)
        L0 = Li*1.02
        q[:, :2] += rng.normal(0, sigma, (n, 2)); q[:, 2] += rng.normal(0, sigma, n)
    t = time.time()
    z = s2r.harden(z, n, L0, Lend, p1=p1, stages=stages, rng=rng)
    th = time.time()-t
    res = s2r.compress(z, n, Lend+0.02)
    if res is None: return dict(n=n, mode=mode, seed=seed, L=None, note='no feasible below ceil+.5', t=time.time()-t)
    zc, Lc = res
    side, reason, rows = s2r.certify_z(zc, n, Lc)
    return dict(n=n, mode=mode, seed=seed, L_rigid=Lc, L_cert=side, reason=reason, inc=Li,
                ceil=math.ceil(math.sqrt(n)), t_harden=th, t=time.time()-t, rows=rows)
if __name__ == '__main__':
    n = int(sys.argv[1]); mode = sys.argv[2]; seed = int(sys.argv[3])
    r = one(n, mode, seed, shrink=float(sys.argv[4]) if len(sys.argv) > 4 else 0.0)
    rows = r.pop('rows', None); print(r)
