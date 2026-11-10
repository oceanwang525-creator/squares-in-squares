"""Freeze lattice blocks, relax irregular region with penalty + heavy-ball momentum,
hardening (container shrink + penalty weight), then rigid repair by dilation and
official-metric certification.  Usage: freeze_relax.py N [radius] [band] [seconds]"""
import sys, time, csv, io
import numpy as np
from scipy.spatial import cKDTree
from squares import core

INC = "poc/pell_corrugation/data/border-scan/merged.csv"

def load(n):
    rows = core.read_submission(INC)[n]
    return rows, np.array([[float(r["x"]), float(r["y"]), float(r["angle"])] for r in rows])

def extent(th):
    return 0.5 * (np.abs(np.cos(th)) + np.abs(np.sin(th)))

def pair_pen(a, b):  # a,b: (m,3) -> squared penetration depth (SAT, min over 4 axes)
    d = b[:, :2] - a[:, :2]
    best = np.full(len(a), 1e9)
    for t in (a[:, 2], b[:, 2]):
        for ax in (t, t + np.pi / 2):
            ux, uy = np.cos(ax), np.sin(ax)
            ra = 0.5 * (np.abs(np.cos(a[:, 2] - ax)) + np.abs(np.sin(a[:, 2] - ax)))
            rb = 0.5 * (np.abs(np.cos(b[:, 2] - ax)) + np.abs(np.sin(b[:, 2] - ax)))
            best = np.minimum(best, ra + rb - np.abs(d[:, 0] * ux + d[:, 1] * uy))
    return np.maximum(best, 0) ** 2

def wall_pen(p, L):
    e = extent(p[:, 2])
    return (np.maximum(e - p[:, 0], 0) ** 2 + np.maximum(e - p[:, 1], 0) ** 2 +
            np.maximum(p[:, 0] + e - L, 0) ** 2 + np.maximum(p[:, 1] + e - L, 0) ** 2)

def energy_grad(P, free, pi, pj, L, w):
    E = w * (pair_pen(P[pi], P[pj]).sum() + wall_pen(P[free], L).sum())
    G = np.zeros_like(P); h = 1e-7
    for k in range(3):
        for s, sg in ((h, 1), (-h, -1)):
            A = P[pi].copy(); B = P[pj].copy(); A[:, k] += s
            ga = sg * pair_pen(A, P[pj]) / (2 * h)
            B[:, k] += s; gb = sg * pair_pen(P[pi], B) / (2 * h)
            np.add.at(G[:, k], pi, ga); np.add.at(G[:, k], pj, gb)
            Q = P[free].copy(); Q[:, k] += s
            G[free, k] += sg * wall_pen(Q, L) / (2 * h)
    G *= w
    mask = np.zeros(len(P), bool); mask[free] = True
    G[~mask] = 0
    return E, G

def run(n, radius=2.0, band=1.5, secs=300, seed=0):
    rows, P = load(n)
    ext = extent(P[:, 2])
    lo = min((P[:, 0] - ext).min(), (P[:, 1] - ext).min())
    P[:, :2] -= lo
    L0 = max((P[:, 0] + ext).max(), (P[:, 1] + ext).max())
    off = np.minimum(np.mod(P[:, 2], np.pi / 2), np.pi / 2 - np.mod(P[:, 2], np.pi / 2))
    tilted = np.where(off > 1e-3)[0]
    tree = cKDTree(P[:, :2])
    free = set(tilted.tolist())
    for i in tilted: free.update(tree.query_ball_point(P[i, :2], radius))
    free.update(np.where((P[:, 0] > L0 - band) | (P[:, 1] > L0 - band))[0].tolist())
    free = np.array(sorted(free))
    isf = np.zeros(n, bool); isf[free] = True
    pairs = np.array([(i, j) for i, j in tree.query_pairs(1.6) if isf[i] or isf[j]])
    pi, pj = pairs[:, 0], pairs[:, 1]
    print(f"n={n} L0={L0:.9f} tilted={len(tilted)} free={len(free)}/{n} pairs={len(pairs)}", flush=True)
    rng = np.random.default_rng(seed)
    best = None; t0 = time.time(); V = np.zeros_like(P)
    shrink = 0.0
    for stage in range(400):
        if time.time() - t0 > secs: break
        shrink += 2e-5 if stage else 0.0           # hardness: container shrink
        L = L0 - shrink; w = 1.0 * 1.05 ** stage    # hardness: penalty weight
        lr, mom = 0.02 / (1 + w * 0.02), 0.9
        for it in range(150):                       # heavy-ball
            E, G = energy_grad(P, free, pi, pj, L, w)
            V = mom * V - lr * G; P += V
            if E < 1e-18: break
        print(f"stage {stage} L={L:.6f} w={w:.1f} E={E:.3e}", flush=True)
        if E < 1e-16 * w:                           # near-feasible -> try repair
            ok = repair(n, rows, P, L)
            if ok is not None: best = ok
        elif E > 1e-3 and stage > 3: break
    return best, L0, rows

def fmt(P, s):
    return [dict(square_id=str(i), x=repr(float(p[0] * s)), y=repr(float(p[1] * s)), angle=repr(float(p[2])))
            for i, p in enumerate(P)]

def repair(n, rows, P, L):
    for s in (1.0, 1 + 1e-9, 1 + 1e-8, 1 + 5e-8, 1 + 2e-7, 1 + 1e-6):  # rigid repair: centre dilation
        r = fmt(P, s)
        _, _, side, reason = core.evaluate(n, r)
        if side is not None and reason == "ok":
            print(f"  repaired s={s} measured L={float(side):.10f}", flush=True)
            return float(side), r
    return None

if __name__ == "__main__":
    n = int(sys.argv[1]); rad = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
    band = float(sys.argv[3]) if len(sys.argv) > 3 else 1.5
    secs = float(sys.argv[4]) if len(sys.argv) > 4 else 300
    best, L0, rows = run(n, rad, band, secs)
    inc = float(core.certify(n, rows))
    print("incumbent measured L", inc, "| best repaired", None if best is None else best[0])
