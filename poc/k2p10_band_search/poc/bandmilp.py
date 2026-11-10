"""Slanted-band candidate pool + MILP max independent set.
Pool: upright unit squares on four corner-anchored lattices + band squares at angle th on a lattice
(columns j=-w/2..w/2 across, free phase) along a line through the container, crossing wall to wall.
"""
import math, sys, time, json, random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree
EPS = 1e-7
def half(a): return 0.5*(abs(math.cos(a))+abs(math.sin(a)))
def corners_arr(P):
    x, y, a = P[:,0], P[:,1], P[:,2]; c, s = np.cos(a), np.sin(a)
    sg = np.array([[1,1],[-1,1],[-1,-1],[1,-1]])*0.5
    X = x[:,None] + sg[None,:,0]*c[:,None] - sg[None,:,1]*s[:,None]
    Y = y[:,None] + sg[None,:,0]*s[:,None] + sg[None,:,1]*c[:,None]
    return X, Y
def overlap_pairs(P, I, J):
    """Vectorised SAT: True where squares I,J overlap by > EPS."""
    X, Y = corners_arr(P)
    ov = np.ones(len(I), bool)
    for src in (I, J):
        a = P[src,2]
        for ax in ((np.cos(a), np.sin(a)), (-np.sin(a), np.cos(a))):
            pa = X[I]*ax[0][:,None] + Y[I]*ax[1][:,None]
            pb = X[J]*ax[0][:,None] + Y[J]*ax[1][:,None]
            sep = np.minimum(pa.max(1), pb.max(1)) - np.maximum(pa.min(1), pb.min(1)) <= EPS
            ov &= ~sep
    return ov
def upright(L):
    m = int(math.floor(L+1e-9)); out = []
    for ax in (0,1):
        for ay in (0,1):
            for i in range(m):
                for j in range(m):
                    out.append((i+.5 if ax==0 else L-i-.5, j+.5 if ay==0 else L-j-.5, 0.0))
    return out
def band(L, th, w, off, phase):
    """w columns of squares tilted th; centre line passes through (L/2+off, L/2); phase=(pu,pv) in [0,1)."""
    c, s = math.cos(th), math.sin(th); h = half(th)
    cx, cy = L/2+off, L/2; out = []
    R = int(L*1.5)+3
    for i in range(-R, R):
        for j in range(w):
            u = i + phase[0]; v = (j-(w-1)/2) + phase[1]
            x = cx + u*c - v*s; y = cy + u*s + v*c
            if h-1e-9 <= x <= L-h+1e-9 and h-1e-9 <= y <= L-h+1e-9: out.append((x, y, th))
    return out
def solve(L, cand, tl=10.0):
    cand = np.unique(np.round(np.array(cand), 9), axis=0)
    tree = cKDTree(cand[:,:2]); pr = tree.query_pairs(math.sqrt(2)+1e-6, output_type='ndarray')
    ov = overlap_pairs(cand, pr[:,0], pr[:,1]); pr = pr[ov]
    e = len(pr); r = np.repeat(np.arange(e), 2)
    A = coo_matrix((np.ones(2*e), (r, pr.reshape(-1))), shape=(e, len(cand))).tocsr()
    res = milp(-np.ones(len(cand)), constraints=LinearConstraint(A, -np.inf, 1), integrality=np.ones(len(cand)),
               bounds=Bounds(0,1), options=dict(time_limit=tl, disp=False))
    if res.x is None: return 0, cand[:0], None
    sel = res.x > .5
    return int(sel.sum()), cand[sel], -res.mip_dual_bound if hasattr(res,'mip_dual_bound') else None
if __name__ == "__main__":
    L = float(sys.argv[1]); t = time.time()
    cand = upright(L) + band(L, math.pi/4, 8, 0.0, (0.3, 0.2))
    print(len(cand)); print(solve(L, cand, 20)[0], time.time()-t)

def pool(L, th, w, off, phase, rc=12.0):
    """Region-restricted pool: SE lattice on the lower-right side of the band line, NW lattice on the other side,
    SW/NE lattices only within rc of their corner."""
    c, s = math.cos(th), math.sin(th); cx, cy = L/2+off, L/2
    m = int(math.floor(L+1e-9)); out = []
    for ax in (0,1):
        for ay in (0,1):
            for i in range(m):
                for j in range(m):
                    x = i+.5 if ax==0 else L-i-.5; y = j+.5 if ay==0 else L-j-.5
                    d = (x-cx)*(-s) + (y-cy)*c
                    if (ax,ay) == (1,0) and d > 0: continue
                    if (ax,ay) == (0,1) and d < 0: continue
                    if (ax,ay) == (0,0) and math.hypot(x, y) > rc: continue
                    if (ax,ay) == (1,1) and math.hypot(L-x, L-y) > rc: continue
                    out.append((x, y, 0.0))
    return out + band(L, th, w, off, phase)
