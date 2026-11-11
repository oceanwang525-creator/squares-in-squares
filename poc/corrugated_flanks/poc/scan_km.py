import sys, math, json, time, numpy as np
sys.path.insert(0,'/Users/lucaszhang/squares-in-squares/poc/corrugated_flanks/poc')
from run_cell import search, R2
w=int(sys.argv[1]); st=int(sys.argv[2]); ks=list(map(int,sys.argv[3].split(',')))
for k in ks:
    f=math.floor(k*R2)
    for m in (f-1,f,f+1):
        for d in (0.5,0.0):
            t=time.time(); c,H,z=search(k,m,w,st,seed=k*31+m,dphi=d)
            Hr=w+2*R2
            print(json.dumps(dict(k=k,m=m,w=w,dphi=d,H=round(H,4),dH=round(H-Hr,4),waste_len=round(c.waste(H)/c.P,4),rigid_len=round(c.waste(Hr)/c.P,4),sec=round(time.time()-t))),flush=True)
