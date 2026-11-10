"""Two thin 45deg bands (or band + pocket) MILP count. python sweep.py n L ncfg seed [tl]
Candidates: upright squares on 4 corner lattices + 45deg squares with centres on a (t,d) grid inside band windows."""
import sys, math, random, time, json
sys.path.insert(0, "/Users/lucaszhang/squares-in-squares/poc/multi_angle_band/poc")
import numpy as np
from multiprocessing import Pool
from mab import candidates, solve, verify_local, half
S2 = math.sqrt(2)

def band_cands(L, th, c, w, step):
    h = half(th); cs, sn = math.cos(th), math.sin(th)
    out = []
    Tm = L * S2
    # axes: t along (1,1)/s2, d along (1,-1)/s2 ; th=pi/4 squares aligned with them
    for t in np.arange(0, Tm, step):
        for d in np.arange(c - w/2, c + w/2 + 1e-9, step):
            x = (t + d)/S2; y = (t - d)/S2
            if x < h-1e-9 or y < h-1e-9 or x > L-h+1e-9 or y > L-h+1e-9: continue
            out.append((x, y, th))
    return out

def run(args):
    n, L, cfg, tl, step = args
    cand = list(map(tuple, candidates(L, [])))
    for (th, c, w) in cfg: cand += band_cands(L, th, c, w, step)
    arr = np.unique(np.round(np.array(cand), 9), axis=0)
    cnt, sel = solve(L, arr, tl=tl)
    return cnt, cfg, sel, len(arr)

if __name__ == "__main__":
    n, L, ncfg, seed = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    tl = float(sys.argv[5]) if len(sys.argv) > 5 else 20; mode = sys.argv[6] if len(sys.argv) > 6 else "two"
    step = 0.1; random.seed(seed); cfgs = []
    P = math.pi/4
    if mode == "one": cfgs = [[(P, 0.0, 1.0)]]
    for _ in range(ncfg if mode != "one" else 0):
        c1 = random.uniform(-6, 6); c2 = c1 + random.uniform(3, 8)
        if c2 > 9: c2 = c1 - random.uniform(3, 8)
        cfgs.append([(P, round(c1, 2), 1.0), (P, round(c2, 2), 1.0)])
    t0 = time.time(); best = (-1,)
    with Pool(8) as p:
        for cnt, cfg, sel, m in p.imap_unordered(run, [(n, L, c, tl, step) for c in cfgs]):
            print(cnt, n, cfg, m, f"{time.time()-t0:.0f}s", flush=True)
            if cnt > best[0]: best = (cnt, cfg, sel)
    np.save(f"/Users/lucaszhang/squares-in-squares/poc/k2p10_two_bands/docs/best_{mode}_n{n}_L{L}.npy", best[2])
    print("BEST", best[0], best[1])
