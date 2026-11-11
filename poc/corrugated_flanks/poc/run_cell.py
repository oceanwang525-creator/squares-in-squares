import sys, math, time, json, numpy as np
sys.path.insert(0, '/Users/lucaszhang/squares-in-squares/poc/corrugated_flanks/poc')
from cell import Cell, R2
def search(k, m, w, starts, seed, tol=1e-11, step=0.01, dphi=None):
    c = Cell(k, m, w); c.dphi = dphi; rng = np.random.default_rng(seed)
    Hr = w + 2 * R2
    best = Hr + 0.3; bz = None
    for s in range(starts):
        H = Hr + 0.3
        z = c.init(H, rng, rng.choice([0.02, 0.06, 0.15]))
        z, e = c.minimize(z, H)
        if e > tol:
            continue
        ok = None
        while H > 1.0:
            H2 = H - step
            z2, e2 = c.minimize(z, H2)
            if e2 > tol:
                # tiny kick retry
                z3 = z2 + rng.normal(0, 0.01, z2.shape); z3, e3 = c.minimize(z3, H2)
                if e3 > tol: break
                z2, e2 = z3, e3
            z, H = z2, H2
        if H < best: best, bz = H, z.copy()
    return c, best, bz
if __name__ == '__main__':
    k, m, w, starts = map(int, sys.argv[1:5])
    t = time.time()
    c, H, z = search(k, m, w, starts, seed=k * 100 + m * 7 + w)
    Hr = w + 2 * R2
    print(json.dumps(dict(k=k, m=m, w=w, H=round(H, 4), H_rigid=round(Hr, 4), waste=round(c.waste(H), 4), waste_rigid=round(c.waste(Hr), 4),
        per_len=round(c.waste(H) / c.P, 4), per_len_rigid=round(c.waste(Hr) / c.P, 4), sec=round(time.time() - t, 1))), flush=True)
