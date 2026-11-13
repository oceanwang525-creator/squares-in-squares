"""Neighbour deletion + shrink: delete k squares from our n+k packing, scale to L_target, penalty-relax (L-BFGS, all x,y,angle free)."""
import sys, os, math, time, json
import numpy as np, pandas as pd
from scipy.optimize import minimize
sys.path.insert(0, "/mnt/e/squares-poc-regenerate-nov13/poc/k2p10_fill2/poc")
import a_overlap as ao
from scipy.spatial import cKDTree
ROOT = "/mnt/e/squares-poc-regenerate-nov13/poc/"
MERGED = ROOT + "pell_corrugation/data/border-scan/merged.csv"
CMP = pd.read_csv(ROOT + "leader_curve_analysis/docs/per_n_comparison.csv").set_index("n")

def load():
    df = pd.read_csv(MERGED)
    return {n: g[["x", "y", "angle"]].to_numpy(float) for n, g in df.groupby("n")}

def norm(q):
    q = q.copy(); h = 0.5 * (np.abs(np.cos(q[:, 2])) + np.abs(np.sin(q[:, 2])))
    q[:, 0] -= (q[:, 0] - h).min(); q[:, 1] -= (q[:, 1] - h).min()
    return q

def relax(q, L, iters, margin=0.0):
    N = len(q)
    def f(v):
        E, G = ao.penalty(v.reshape(N, 3), L, margin=margin)
        return E, G.ravel()
    r = minimize(f, q.ravel(), jac=True, method="L-BFGS-B", options=dict(maxiter=iters, maxcor=20, ftol=1e-30, gtol=1e-14))
    return r.x.reshape(N, 3), r.fun

def exact_ok(q, L):
    P = ao.pairs(q)
    dep = max((ao.ref_depth(q[i], q[j]) for i, j in P), default=-1)
    wv, _ = ao.wall_violation(q, L)
    return dep, float(wv.max())

def candidates(q, k, rng, ncand):
    N = len(q); L = ao.bbox_side(q)
    t = np.abs(np.sin(2 * q[:, 2])) > 1e-6  # tilted
    P = ao.pairs(q, 0.0); dep = ao.pair_depth(q, P)
    cont = np.zeros(N); m = dep > -1e-6
    np.add.at(cont, P[m, 0], 1); np.add.at(cont, P[m, 1], 1)
    tree = cKDTree(q[:, :2]); S = []
    S.append(np.argsort(cont + rng.random(N) * 0.1)[:k])
    for c in [(0, 0), (L, 0), (0, L), (L, L)]:
        S.append(np.argsort(((q[:, :2] - c) ** 2).sum(1))[:k])
    S.append(np.atleast_1d(tree.query([L / 2, L / 2], k)[1]))
    ti = np.where(t)[0]
    while len(S) < ncand:
        typ = len(S) % 3
        if len(ti) == 0: typ = 2
        if typ == 0: s = ti[rng.choice(len(ti), k, replace=False)]
        elif typ == 1: s = np.atleast_1d(tree.query(q[ti[rng.integers(len(ti))], :2], k)[1])
        else:
            c = q[rng.integers(N), :2]; s = np.atleast_1d(tree.query(c, k)[1])
        S.append(np.asarray(s))
    return [tuple(sorted(set(map(int, s)))) for s in S][:ncand]

def attempt(q_src, dele, Lt, iters):
    keep = np.setdiff1d(np.arange(len(q_src)), dele)
    q = norm(q_src[keep]); Ls = ao.bbox_side(q)
    q[:, :2] *= Lt / Ls
    q, E = relax(q, Lt, iters)
    return q, E

if __name__ == "__main__":
    targets = [int(x) for x in sys.argv[1].split(",")]
    ncand = int(sys.argv[2]); iters = int(sys.argv[3]); budget = float(sys.argv[4])
    out = sys.argv[5]
    Q = load(); rng = np.random.default_rng(0); res = {}
    T0 = time.time()
    for n in targets:
        Lours = float(CMP.L_ours[n]); Llead = float(CMP.L_lb[n]); Lt = Lours - 1e-6
        best = None; log = []
        for k in range(1, 5):
            q_src = Q[n + k]
            for dele in candidates(norm(q_src), k, rng, ncand):
                t0 = time.time(); q, E = attempt(norm(q_src), dele, Lt, iters)
                log.append((k, dele, E, time.time() - t0))
                if best is None or E < best[0]: best = (E, k, dele, q)
            if time.time() - T0 > budget * (targets.index(n) + 1) / len(targets): pass
        E, k, dele, q = best
        # harden best: margin ramp, then exact check
        for mg in (1e-8, 1e-7, 1e-6):
            q, E2 = relax(q, Lt, 400, margin=mg)
        dep, wv = exact_ok(q, Lt)
        ok = dep <= 0 and wv <= 0
        res[n] = dict(n=n, Lours=Lours, Llead=Llead, Ltarget=Lt, best_k=k, best_dele=list(dele), E_resid=E, exact_maxdepth=dep, wall=wv, feasible=bool(ok),
                      E_all=sorted(round(l[2], 12) for l in log)[:5], ncand_tried=len(log), sec=time.time() - T0)
        print(json.dumps(res[n]), flush=True)
        if ok: np.save(out + f"_win{n}.npy", q)
    json.dump(res, open(out + ".json", "w"), indent=1)
