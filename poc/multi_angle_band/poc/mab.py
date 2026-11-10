"""Multi-angle off-center band: lattice-candidate max-independent-set (MILP).

Candidates: upright unit squares on four corner-anchored lattices, plus tilted
squares in a diagonal strip split into two segments, each segment with its own
tilt angle and lattice phase. HiGHS picks the largest non-overlapping subset
for container side L. A hit needs count >= n at L < incumbent L.
"""
from __future__ import annotations
import math, sys, time, json
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree

EPS = 1e-7

def half(a): return 0.5 * (abs(math.cos(a)) + abs(math.sin(a)))

def corners(x, y, a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[x + .5*c - .5*s, y + .5*s + .5*c], [x - .5*c - .5*s, y - .5*s + .5*c],
                     [x - .5*c + .5*s, y - .5*s - .5*c], [x + .5*c + .5*s, y + .5*s - .5*c]])

def overlap(p, q):
    """True if squares p,q=(x,y,a) overlap by more than EPS (SAT)."""
    A, B = corners(*p), corners(*q)
    for a in (p[2], q[2]):
        for ax in ((math.cos(a), math.sin(a)), (-math.sin(a), math.cos(a))):
            pa, pb = A @ ax, B @ ax
            if min(pa.max(), pb.max()) - max(pa.min(), pb.min()) <= EPS:
                return False
    return True

def candidates(L, bands, upright=True):
    """bands: list of dict(theta, phase=(px,py), t0, t1, c, w). Returns array (m,3)."""
    cand = []
    if upright:
        m = int(math.floor(L + 1e-9))
        for ax in (0, 1):
            for ay in (0, 1):
                for i in range(m):
                    for j in range(m):
                        x = i + .5 if ax == 0 else L - i - .5
                        y = j + .5 if ay == 0 else L - j - .5
                        cand.append((x, y, 0.0))
    for b in bands:
        th = b["theta"]; h = half(th)
        c, s = math.cos(th), math.sin(th)
        R = int(L * 1.5) + 2
        px, py = b["phase"]
        for a in range(-R, R):
            for bb in range(-R, R):
                x = px + a*c - bb*s; y = py + a*s + bb*c
                if x < h - 1e-9 or y < h - 1e-9 or x > L - h + 1e-9 or y > L - h + 1e-9: continue
                t = (x + y) / math.sqrt(2); d = (x - y) / math.sqrt(2)  # along / across diagonal
                if not (b["t0"] <= t < b["t1"]): continue
                if abs(d - b["c"]) > b["w"] / 2: continue
                cand.append((x, y, th))
    arr = np.unique(np.round(np.array(cand), 9), axis=0)
    return arr

def solve(L, cand, tl=10.0):
    m = len(cand)
    tree = cKDTree(cand[:, :2])
    rows, cols = [], []
    e = 0
    for i, j in tree.query_pairs(math.sqrt(2) + 1e-6):
        if overlap(tuple(cand[i]), tuple(cand[j])):
            rows += [e, e]; cols += [i, j]; e += 1
    A = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(e, m)).tocsr()
    res = milp(-np.ones(m), constraints=LinearConstraint(A, -np.inf, 1), integrality=np.ones(m),
               bounds=Bounds(0, 1), options=dict(time_limit=tl, disp=False))
    if res.x is None: return 0, cand[:0]
    sel = res.x > .5
    return int(sel.sum()), cand[sel]

def verify_local(L, sel):
    """Float pairwise + containment check (not the official metric)."""
    tree = cKDTree(sel[:, :2])
    for i, j in tree.query_pairs(math.sqrt(2) + 1e-6):
        if overlap(tuple(sel[i]), tuple(sel[j])): return False
    h = np.array([half(a) for a in sel[:, 2]])
    return bool((sel[:, 0] - h >= -1e-9).all() and (sel[:, 1] - h >= -1e-9).all()
                and (sel[:, 0] + h <= L + 1e-9).all() and (sel[:, 1] + h <= L + 1e-9).all())
