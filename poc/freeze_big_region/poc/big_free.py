"""Big-free-region freeze/relax with annealed container + exact repair.
usage: big_free.py N R seed secs   (R<=0 => free everything except 4 corner 3x3 blocks)"""
import sys, time, json, math
import numpy as np
from scipy.optimize import minimize
sys.path.insert(0, "poc/k2p10_fill2/poc")
import a_overlap as ao
from squares import core
INC = "poc/pell_corrugation/data/border-scan/merged.csv"

def load(n):
    rows = core.read_submission(INC)[n]
    q = np.array([[float(r["x"]), float(r["y"]), float(r["angle"])] for r in rows])
    h = ao.wall_violation(q, 1e9)[1]
    q[:, 0] -= (q[:, 0] - h).min(); q[:, 1] -= (q[:, 1] - h).min()
    return q

def sidecheck(n, q, s):
    c = q[:, :2].mean(0)
    r = [dict(square_id=str(i), x=repr(float((p[0]-c[0])*s+c[0])), y=repr(float((p[1]-c[1])*s+c[1])),
              angle=repr(float(p[2]))) for i, p in enumerate(q)]
    _, _, side, reason = core.evaluate(n, r)
    return (float(side) if side is not None and reason == "ok" else None), r

LOOSE = 0.3
def run(n, R, seed, secs, sig, adeg, step=2e-5):
    q0 = load(n); N = len(q0)
    h = ao.wall_violation(q0, 1e9)[1]
    L0 = ao.bbox_side(q0)
    off = np.mod(q0[:, 2], np.pi/2); off = np.minimum(off, np.pi/2 - off)
    til = np.where(off > 1e-3)[0]
    from scipy.spatial import cKDTree
    t = cKDTree(q0[:, :2])
    if R > 0:
        fr = set(til.tolist())
        for i in til: fr.update(t.query_ball_point(q0[i, :2], R))
        fr.update(np.where((q0[:, 0] > L0 - 1.6) | (q0[:, 1] > L0 - 1.6))[0].tolist())
    else:
        d = np.minimum(np.abs(q0[:, 0, None] - np.array([0, L0])), 1e9)
        cx = np.minimum(q0[:, 0], L0 - q0[:, 0]); cy = np.minimum(q0[:, 1], L0 - q0[:, 1])
        fr = set(np.where(~((cx < 2) & (cy < 2)) | (q0[:, 0] > L0 - 1.6) | (q0[:, 1] > L0 - 1.6))[0].tolist())
    free = np.array(sorted(fr)); isf = np.zeros(N, bool); isf[free] = True
    P = ao.pairs(q0, margin=0.3)
    P = P[isf[P[:, 0]] | isf[P[:, 1]]]
    rng = np.random.default_rng(seed)
    q = q0.copy()
    q[free, :2] += rng.normal(0, sig, (len(free), 2))
    q[free, 2] += rng.normal(0, math.radians(adeg), len(free))
    nf = len(free)
    def fun(x, L):
        qq = q.copy(); qq[free] = x.reshape(nf, 3)
        E, G = ao.penalty(qq, L, P)
        return E, G[free].ravel()
    def relax(L, it=400):
        r = minimize(fun, q[free].ravel(), args=(L,), jac=True, method="L-BFGS-B",
                     options=dict(maxiter=it, ftol=1e-30, gtol=1e-14, maxcor=20))
        q[free] = r.x.reshape(nf, 3)
        return r.fun
    print(f"n={n} R={R} seed={seed} sig={sig} adeg={adeg} free={nf}/{N} pairs={len(P)} L0={L0:.9f}", flush=True)
    t0 = time.time()
    # start loose (L0+0.05), descend with adaptive step; failure => back off upward with a kick (anneal)
    Lc = L0 + LOOSE; E = relax(Lc, 30000)
    print(f"  jitter-relax at L0+loose: E={E:.2e}", flush=True)
    goodq = q.copy(); goodL = Lc; best = None; st = 5e-3; fails = 0; tried = 0
    while time.time() - t0 < secs and fails < 25:
        Ltry = goodL - st
        E = relax(Ltry, 800)
        tried += 1
        if E < 1e-20:
            goodq = q.copy(); goodL = Ltry; fails = 0; st = min(st * 1.5, 1e-2)
            if goodL < L0 - 1e-9: best = (goodL, q.copy())
        else:
            fails += 1
            q[:] = goodq
            if st > 2e-6:
                st *= 0.5
            else:  # kick and re-relax at a looser L, then continue
                q[free, :2] += rng.normal(0, sig * 0.2, (nf, 2)); q[free, 2] += rng.normal(0, math.radians(adeg) * 0.2, nf)
                relax(goodL + 2e-3, 800); goodq = q.copy(); goodL = goodL + 2e-3; st = 1e-4
    print(f"  lowest feasible-penalty L reached {goodL:.9f} (vs L0 {L0:.9f})", flush=True)
    Lb = L0 if best is None else best[0]
    print(f"  stalled: feasible-penalty L={Lb:.9f} (shrink {L0-Lb:.2e}) tried={tried} t={time.time()-t0:.0f}s", flush=True)
    meas = None
    if best is not None:
        for s in (1+1e-10, 1+1e-9, 1+1e-8, 1+1e-7, 1+1e-6):
            qq = best[1]
            m, r = sidecheck(n, qq, s)
            if m is not None:
                meas = m; print(f"  repaired s={s}: measured L={m:.10f}", flush=True)
                json.dump(r, open(f"poc/freeze_big_region/data_{n}_{R}_{seed}.json", "w"))
                break
    return L0, Lb, meas

if __name__ == "__main__":
    n, R, seed, secs = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
    rng = np.random.default_rng(1000 + seed)
    sig = float(rng.uniform(0.02, 0.1)); adeg = float(rng.uniform(1, 3))
    L0, Lb, meas = run(n, R, seed, secs, sig, adeg)
    print("RESULT", n, R, seed, "incumbent", L0, "penalty-feasible", Lb, "measured", meas, flush=True)
