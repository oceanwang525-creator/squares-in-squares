import sys, numpy as np
sys.path.insert(0,'/Users/lucaszhang/squares-in-squares/poc/corrugated_flanks/poc')
from run_cell import search
import a_overlap as ov
k,m,w,st=map(int,sys.argv[1:5])
c,H,z=search(k,m,w,st,seed=k*100+m*7+w)
Q,nb=c.full(z,H)
# independent check: all pairs within 2 with scalar reference
mx=-9
for i in range(len(Q)):
    for j in range(i+1,len(Q)):
        if np.hypot(*(Q[i,:2]-Q[j,:2]))<1.5: mx=max(mx,ov.ref_depth(Q[i],Q[j]))
print('H',H,'max ref depth (fixed-fixed touching = 0):',mx,'phases',z[-2:]%1)
q=z[:3*c.n].reshape(-1,3); print(np.round(q[np.argsort(q[:,1])],3)[:,:3].tolist())
