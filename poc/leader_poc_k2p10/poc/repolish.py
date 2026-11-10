"""Relax the n=635/686 incumbent (45deg w=1 chain + staircases) with the Pell polish LP; report L and certify."""
import sys, time, numpy as np, pandas as pd
sys.path.insert(0, "poc/pell_corrugation/poc")
import pell_corrugation as pc
from squares import core
n = int(sys.argv[1]); iters = int(sys.argv[2]); dil = 1.0000002
d = pd.read_csv("poc/pell_corrugation/data/border-scan/merged.csv")
g = d[d.n == n]
q = g[["x", "y", "angle"]].to_numpy(float)
print("incumbent extent", pc.extent(q), flush=True)
t = time.time()
# a small seed squeeze: ask the LP to start from a slightly smaller box budget
res = pc.polish_band(q, pc.extent(q), iterations=iters, position_trust=0.05, angle_trust=0.01)
print("polish", None if res is None else res.L, "time", time.time() - t, flush=True)
if res is not None:
    np.save(f"poc/leader_poc_k2p10/docs/polished_{n}.npy", res.q)
    ok, mn = pc.sat_check(res.q)
    print("sat", ok, mn)
