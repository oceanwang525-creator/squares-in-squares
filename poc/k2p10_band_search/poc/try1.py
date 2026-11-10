import sys, math, time, itertools
sys.path.insert(0, 'poc'); from bandmilp import *
L = float(sys.argv[1]); w = int(sys.argv[2]); th=math.pi/4
t=time.time(); best=0
for pu in np.linspace(0,1,6,endpoint=False):
  for pv in np.linspace(0,1,6,endpoint=False):
    cnt, sel, _ = solve(L, pool(L, th, w, 0.0, (pu,pv)), 5)
    best=max(best,cnt)
print(L,w,best,time.time()-t)
