import sys, math, random, time, json
sys.path.insert(0, __file__.rsplit('/',1)[0])
from mab import *
n, Linc = int(sys.argv[1]), float(sys.argv[2]); budget = float(sys.argv[3]); dL = float(sys.argv[4]) if len(sys.argv)>4 else 0.0
L = Linc + dL
random.seed(n); best = (-1, None); t0 = time.time(); log = []
Tmax = L * math.sqrt(2)
while time.time() - t0 < budget:
    split = random.uniform(.3, .7) * Tmax
    c = random.uniform(-0.15, 0.15) * L; w = random.uniform(1.5, 4.5)
    th1 = math.pi/4 + random.choice([0, 0, random.uniform(-.2, .2)])
    th2 = math.pi/4 + random.choice([0, random.uniform(-.2, .2)])
    bands = [dict(theta=th1, phase=(random.random(), random.random()), t0=0, t1=split, c=c, w=w),
             dict(theta=th2, phase=(random.random(), random.random()), t0=split, t1=Tmax + 1, c=c + random.uniform(-.5,.5), w=w)]
    cand = candidates(L, bands)
    cnt, sel = solve(L, cand, tl=6)
    log.append((cnt, bands))
    if cnt > best[0]:
        best = (cnt, sel, bands); print(f"n={n} L={L:.6f} count={cnt} target={n} cands={len(cand)} t={time.time()-t0:.0f}s", flush=True)
cnt, sel, bands = best
import numpy as np
np.save(f"/mnt/e/squares-poc-regenerate-nov13/poc/multi_angle_band/docs/best_n{n}_dL{dL}.npy", sel)
print("FINAL", n, L, cnt, "valid_local", verify_local(L, sel), "gap_to_n", n - cnt)
