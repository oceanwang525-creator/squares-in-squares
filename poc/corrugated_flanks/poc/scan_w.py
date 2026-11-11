import sys, math, json, time
sys.path.insert(0,'/Users/lucaszhang/squares-in-squares/poc/corrugated_flanks/poc')
from run_cell import search, R2
for w in (2,3):
  for k,m in ((5,7),(7,10),(12,17)):
    for d in (0.5,0.0):
        if k==12 and w==3: continue
        t=time.time(); c,H,z=search(k,m,w,4,seed=k*31+m+w,dphi=d); Hr=w+2*R2
        print(json.dumps(dict(k=k,m=m,w=w,dphi=d,H=round(H,4),dH=round(H-Hr,4),waste_len=round(c.waste(H)/c.P,4),rigid_len=round(c.waste(Hr)/c.P,4),sec=round(time.time()-t))),flush=True)
