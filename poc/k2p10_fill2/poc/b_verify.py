"""Verify b_overlap against an independent polygon-projection reference, finite-difference gradient, and the incumbent 635."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "/Users/lucaszhang/squares-in-squares/poc/k2p10_fill2/poc")
sys.path.insert(0, "/Users/lucaszhang/squares-in-squares/poc/pell_corrugation/poc")
import b_overlap as bo

def corners(x, y, a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[x + c*dx - s*dy, y + s*dx + c*dy] for dx, dy in ((-.5, -.5), (.5, -.5), (.5, .5), (-.5, .5))])

def ref_gap(p, q):
    A, B = corners(*p), corners(*q)
    best = -np.inf
    for pp in (p, q):
        for k in (0, 1):
            a = np.array([np.cos(pp[2] + k*np.pi/2), np.sin(pp[2] + k*np.pi/2)])
            pa, pb = A @ a, B @ a
            best = max(best, pb.min() - pa.max(), pa.min() - pb.max())
    return best

rng = np.random.default_rng(1)
n = 60
P = np.column_stack([rng.uniform(0, 6, n), rng.uniform(0, 6, n), rng.uniform(-3, 3, n)])
I, J = np.triu_indices(n, 1)
g = bo.pair_gap(P, I, J)
r = np.array([ref_gap(P[i], P[j]) for i, j in zip(I, J)])
print("pair gap max abs err vs polygon reference:", np.abs(g - r).max(), "pairs", len(I))
# energy vs reference over all pairs (cutoff neighbours: pairs beyond sqrt2 never overlap)
E, gr, info = bo.energy(P, 6.0)
Eref = sum(max(0, -x) ** 2 for x in r)
for x, y, a in P:
    e = 0.5*(abs(np.cos(a)) + abs(np.sin(a)))
    Eref += sum(max(0, v) ** 2 for v in (e - x, x + e - 6.0, e - y, y + e - 6.0))
print("energy", E, "ref", Eref, "diff", abs(E - Eref), info)
# gradient finite differences
num = np.zeros_like(P); h = 1e-6
for i in range(n):
    for q in range(3):
        Pp = P.copy(); Pp[i, q] += h; Pm = P.copy(); Pm[i, q] -= h
        num[i, q] = (bo.energy(Pp, 6.0, grad=False)[0] - bo.energy(Pm, 6.0, grad=False)[0]) / (2*h)
print("grad max abs err", np.abs(num - gr).max(), "grad scale", np.abs(gr).max())
# incumbent 635
d = pd.read_csv("/Users/lucaszhang/squares-in-squares/poc/pell_corrugation/data/border-scan/merged.csv")
q = d[d.n == 635].sort_values("square_id")[["x", "y", "angle"]].to_numpy(float).copy()
h_ = 0.5*(np.abs(np.cos(q[:, 2])) + np.abs(np.sin(q[:, 2])))
lo = (q[:, :2] - h_[:, None]).min(0); q[:, :2] -= lo
Lb = (q[:, :2] + h_[:, None]).max()
E, gr, info = bo.energy(q, Lb + 1e-9)
print("incumbent 635: Lbox", Lb, "E", E, info)
import pell_corrugation as pc
print("pc.sat_check", pc.sat_check(q))
I, J = bo.pairs(q); print("min gap (module)", bo.pair_gap(q, I, J).min())
