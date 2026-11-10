"""Sequential LP over (x, y, angle) of every square: minimize bounding-box side L.
usage: slp.py n seconds [tp ta]   -> writes data/slp_n{n}.npy
"""
import sys, time, numpy as np, pandas as pd
from pathlib import Path
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree
REPO = Path(__file__).resolve().parents[3]
MERGED = REPO/"poc/pell_corrugation/data/border-scan/merged.csv"

def axes(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, s], [-s, c]])

def satb(pi, pj, k, sg, s1, s2):
    Ai, Aj = axes(pi[2]), axes(pj[2])
    AX = np.vstack([Ai, Aj])
    a = AX[k]; P = Aj if k < 2 else Ai
    d = np.array([pj[0]-pi[0], pj[1]-pi[1]])
    return sg*(a @ d) - 0.5 - 0.5*(s1*(a @ P[0]) + s2*(a @ P[1]))

def sat(pi, pj, k=None, sg=None):
    """pi,pj=(x,y,t). returns (g, k, sg): signed separation along best axis."""
    Ai, Aj = axes(pi[2]), axes(pj[2])
    AX = np.vstack([Ai, Aj])
    d = np.array([pj[0]-pi[0], pj[1]-pi[1]])
    proj = AX @ d
    rad = 0.5*(np.abs(AX @ Ai.T).sum(1) + np.abs(AX @ Aj.T).sum(1))
    if k is None:
        gs = np.abs(proj) - rad
        k = int(np.argmax(gs)); sg = 1.0 if proj[k] >= 0 else -1.0
    return sg*proj[k] - rad[k], k, sg

def ext(t): return 0.5*(abs(np.cos(t))+abs(np.sin(t)))

def L_of(P):
    e = np.array([ext(t) for t in P[:, 2]])
    return max((P[:, 0]+e).max()-(P[:, 0]-e).min(), (P[:, 1]+e).max()-(P[:, 1]-e).min())

def min_sep(P, R=1.7):
    m = 1e9
    for i, j in cKDTree(P[:, :2]).query_pairs(R):
        m = min(m, sat(P[i], P[j])[0])
    return m

def step(P, tp, ta, R=1.7, eps=1e-7):
    n = len(P)
    nv = 3*n+5  # dx dy dt ..., xlo xhi ylo yhi L
    rr, cc, vv, rhs = [], [], [], []
    def add(cols, vals, b):
        r = len(rhs); rr.extend([r]*len(cols)); cc.extend(cols); vv.extend(vals); rhs.append(b)
    for i, j in cKDTree(P[:, :2]).query_pairs(R):
        g, k, sg = sat(P[i], P[j])
        if g > 0.3: continue
        base = np.concatenate([P[i], P[j]])
        cols = [3*i, 3*i+1, 3*i+2, 3*j, 3*j+1, 3*j+2]
        for s1 in (1, -1):
          for s2 in (1, -1):
            g0 = satb(base[:3], base[3:], k, sg, s1, s2)
            grad = np.zeros(6)
            for q in range(6):
                b2 = base.copy(); b2[q] += eps
                grad[q] = (satb(b2[:3], b2[3:], k, sg, s1, s2)-g0)/eps
            add(cols, list(-grad), g0 - 1e-9)
    for i in range(n):
        t = P[i, 2]
        for s1 in (1, -1):
          for s2 in (1, -1):
            e = 0.5*(s1*np.cos(t)+s2*np.sin(t)); de = 0.5*(-s1*np.sin(t)+s2*np.cos(t))
            for q, (lo, hi) in enumerate([(3*n, 3*n+1), (3*n+2, 3*n+3)]):
                add([lo, 3*i+q, 3*i+2], [1, -1, de], P[i, q]-e)
                add([3*i+q, 3*i+2, hi], [1, de, -1], -(P[i, q]+e))
    add([3*n+1, 3*n, 3*n+4], [1, -1, -1], 0)
    add([3*n+3, 3*n+2, 3*n+4], [1, -1, -1], 0)
    # variables for xlo.. are absolute; shift so they are near 0 not needed
    A = coo_matrix((vv, (rr, cc)), shape=(len(rhs), nv)).tocsr()
    c = np.zeros(nv); c[-1] = 1
    bnds = []
    for i in range(n):
        bnds += [(-tp, tp), (-tp, tp), (-ta, ta)]
    bnds += [(None, None)]*4 + [(0, None)]
    res = linprog(c, A_ub=A, b_ub=rhs, bounds=bnds, method="highs",
                  options={"primal_feasibility_tolerance": 1e-10, "dual_feasibility_tolerance": 1e-10, "time_limit": 60})
    if not res.success: return None
    return P + res.x[:3*n].reshape(n, 3)

def load(n):
    g = pd.read_csv(MERGED); g = g[g.n == n].sort_values("square_id")
    return g[["x", "y", "angle"]].to_numpy(float)

def run(P, secs, tp, ta, log=print):
    t0 = time.time(); Lc = L_of(P); it = 0
    while time.time()-t0 < secs and tp > 1e-7:
        Q = step(P, tp, ta)
        if Q is None: tp /= 2; ta /= 2; continue
        ms = min_sep(Q); Lq = L_of(Q)
        it += 1
        if ms >= -1e-10 and Lq < Lc-1e-12:
            log(f"it{it} L={Lq:.10f} dL={Lc-Lq:.2e} tp={tp:.1e} ms={ms:.1e}")
            P, Lc = Q, Lq
        else:
            tp /= 2; ta /= 2
            log(f"it{it} reject Lq={Lq:.10f} ms={ms:.1e} -> tp={tp:.1e}")
    return P, Lc

if __name__ == "__main__":
    n = int(sys.argv[1]); secs = float(sys.argv[2])
    tp = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05
    ta = float(sys.argv[4]) if len(sys.argv) > 4 else 0.02
    P = load(n); print("start L", L_of(P), "minsep", min_sep(P), flush=True)
    P, Lc = run(P, secs, tp, ta, lambda s: print(s, flush=True))
    np.save(REPO/f"poc/leader_poc_k2p4/data/slp_n{n}.npy", P); print("final", Lc, Lc-int(n**.5))
