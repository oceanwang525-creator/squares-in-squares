import sys, numpy as np, json
sys.path.insert(0,'/Users/lucaszhang/squares-in-squares/poc/corrugated_flanks/poc')
from run_cell import search, R2
k,m,w,st=map(int,sys.argv[1:5])
out=[]
for d in np.linspace(0,1,9)[:-1]:
    c,H,z=search(k,m,w,st,seed=11,dphi=d)
    out.append((round(d,3),round(H,4))); print(out[-1],flush=True)
print('rigid',w+2*R2)
