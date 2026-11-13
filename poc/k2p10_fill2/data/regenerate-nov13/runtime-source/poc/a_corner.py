"""Approaches 1+2 attempt at L=25.7070: 633 base + 2 inserted squares; (1) plain relax, (2) corner b-staircases jittered/tilted before relax."""
import sys, time, math, numpy as np
sys.path.insert(0, "/mnt/e/squares-poc-regenerate-nov13/poc/k2p10_fill2/poc")
from a_relax import *
L = 25.7070
q45, L45 = base633(); q0 = q45.copy(); q0[:, :2] *= L / L45
rng = np.random.default_rng(5)
cx = lambda p: np.hypot(*(p[:, :2] - 0.5).T)  # distance to BL corner
dBL = np.hypot(q0[:, 0], q0[:, 1]); dTR = np.hypot(L - q0[:, 0], L - q0[:, 1])
corner = ((dBL < 9) | (dTR < 9)) & (np.abs(q0[:, 2]) < 1e-9)
print("corner squares", corner.sum())
t0 = time.time(); results = []
for mode in ("plain", "tilt45", "tiltpq", "jitter"):
    for tr in range(3):
        q = q0.copy()
        if mode == "tilt45": q[corner, 2] = math.pi / 4 * rng.integers(0, 2, corner.sum())
        if mode == "tiltpq": q[corner, 2] = math.atan(3 / 4) * rng.choice([0, 1], corner.sum()) 
        if mode == "jitter": q[corner, 2] += rng.normal(0, 0.15, corner.sum()); q[corner, :2] += rng.normal(0, 0.05, (corner.sum(), 2))
        pts = [[rng.uniform(1, L - 1), rng.uniform(1, L - 1), rng.uniform(0, 1.57)] for _ in range(2)]
        if mode != "plain":  # put new squares in the corner regions
            pts = [[1.0 + rng.uniform(0, 3), 1.0 + rng.uniform(0, 3), 0.3], [L - 1 - rng.uniform(0, 3), L - 1 - rng.uniform(0, 3), 0.3]]
        qa = np.vstack([q, pts])
        qr, E = relax(qa, L, iters=600, rounds=3)
        mv = max_violation(qr, L); results.append((E, mode))
        print(mode, tr, "E %.5f" % E, "maxdepth %.4f wall %.4f pairs %d" % (mv[0], mv[1], mv[2]), round(time.time() - t0), flush=True)
        if time.time() - t0 > 330: sys.exit()
