"""Audit: wrap every incumbent small-n packing (core) with an upright L-border of
thickness t and compare the resulting (n, L) with the incumbent at that n.
L is the float AABB side of the incumbent rows (screening only; winners must be
re-scored with the official metric)."""
import sys, math
import numpy as np, pandas as pd

CSV = "poc/pell_corrugation/data/border-scan/merged.csv"
df = pd.read_csv(CSV)
L = {}
for n, g in df.groupby("n"):
    x, y, a = g.x.values, g.y.values, g.angle.values
    h = (np.abs(np.cos(a)) + np.abs(np.sin(a))) / 2
    L[n] = max((x + h).max() - (x - h).min(), (y + h).max() - (y - h).min())
best = {}
for nc in range(2, 400):
    k = math.floor(L[nc] + 1e-9)
    for t in range(1, 40):
        n = nc + 2 * t * k + t * t
        if n > 1000: break
        Lt = L[nc] + t
        if Lt < L[n] - 1e-9 and (n not in best or Lt < best[n][0]):
            best[n] = (Lt, nc, t)
print("wins (screen):", len(best))
for n, v in sorted(best.items()): print(n, v, "incumbent", L[n], "delta", v[0] - L[n])
# nearest misses
miss = sorted(((L[nc] + t) - L[nc + 2*t*math.floor(L[nc]+1e-9) + t*t], nc, t)
              for nc in range(2, 400) for t in range(1, 40)
              if nc + 2*t*math.floor(L[nc]+1e-9) + t*t <= 1000)
print("closest misses (dL, core, t):", [(round(d, 6), c, t) for d, c, t in miss[:8]])
pos = [m for m in miss if m[0] > 1e-9]
print("smallest strictly-worse wraps:", [(round(float(d), 6), c, t) for d, c, t in pos[:6]])
print("exact ties (|dL|<1e-9):", sum(1 for m in miss if abs(m[0]) < 1e-9))
