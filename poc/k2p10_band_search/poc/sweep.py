"""Sweep (theta, w, off, pu) -> MILP count at fixed L. usage: sweep.py n L tag [tl]"""
import sys, math, json, time
sys.path.insert(0, 'poc'); from bandmilp import *
from multiprocessing import Pool
n, L, tag = int(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
THS = [45.0, math.degrees(math.atan2(20,21)), 42.0, 40.0, math.degrees(math.atan2(12,... ) if False else math.atan2(3,4))*0+36.8699, 33.0, 30.0, math.degrees(math.atan2(8,15)), math.degrees(math.atan2(5,12)), math.degrees(math.atan2(7,24))]
WS = [2,3,4,5,6,8,10,13]
def task(args):
    thd, w = args; th = math.radians(thd); best = (0,)
    for off in np.linspace(-1.5, 1.5, 13):
        for pu in (0.0, .2, .4, .6, .8):
            try: cnt, sel, _ = solve(L, pool(L, th, w, off, (pu, 0.0)), 3)
            except Exception as e: continue
            if cnt > best[0]: best = (cnt, thd, w, float(off), pu)
    return best
if __name__ == "__main__":
    t = time.time()
    with Pool(8) as p: res = p.map(task, [(t_, w) for t_ in THS for w in WS], chunksize=1)
    res.sort(reverse=True)
    json.dump(res, open(f"docs/sweep_{tag}.json", "w"))
    print(n, L, "time", round(time.time()-t))
    for r in res[:15]: print(r)
