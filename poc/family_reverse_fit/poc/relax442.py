"""Shrink the incumbent n=442 (L=21.535534) toward leader L=21.524648 via penalty relaxation. python relax442.py n target_L trials seed"""
import sys, math, time, numpy as np, pandas as pd
sys.path.insert(0,"/Users/lucaszhang/squares-in-squares/poc/k2p10_fill2/poc")
from a_overlap import penalty, pairs, max_violation, bbox_side
from scipy.optimize import minimize
n=int(sys.argv[1]); Lt=float(sys.argv[2]); T=int(sys.argv[3]); rng=np.random.default_rng(int(sys.argv[4]))
df=pd.read_csv("/Users/lucaszhang/squares-in-squares/poc/pell_corrugation/data/border-scan/merged.csv",usecols=["n","x","y","angle"],dtype=str)
df=df[df.n==str(n)]
q=np.c_[df.x.astype(float),df.y.astype(float),df.angle.astype(float)]
print("n",len(q),"bbox",bbox_side(q))
c=np.cos(q[:,2]);s=np.sin(q[:,2]);h=0.5*(abs(c)+abs(s))
q[:,0]-=(q[:,0]-h).min();q[:,1]-=(q[:,1]-h).min()
L0=max((q[:,0]+h).max(),(q[:,1]+h).max());print("L0",L0)
q0=q.copy()
def relax(q,L,it=600):
    N=len(q)
    for r in range(3):
        P=pairs(q,margin=0.6)
        f=lambda v:(lambda E,G:(E,G.ravel()))(*penalty(v.reshape(N,3),L,P=P))
        res=minimize(f,q.ravel(),jac=True,method="L-BFGS-B",options=dict(maxiter=it,ftol=1e-20,gtol=1e-13))
        q=res.x.reshape(N,3)
    return q,res.fun
best=9;t0=time.time()
for tr in range(T):
    qq=q0.copy();qq[:,:2]*=Lt/L0
    qq+=rng.normal(0,[0.02*(tr>0)]*3,qq.shape)
    qr,E=relax(qq,Lt);mv=max(max_violation(qr,Lt)[:2])
    best=min(best,mv);print(tr,"E",E,"maxviol",mv,"t",round(time.time()-t0),flush=True)
    if time.time()-t0>420:break
print("best max violation",best,"(feasible needs <1e-9)")
