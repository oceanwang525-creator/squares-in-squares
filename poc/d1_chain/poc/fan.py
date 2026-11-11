"""Relax incumbent 401 at decreasing L with diamond-level perturbations (fan/rotation/shift). python fan.py seconds seed"""
import sys,time,math,numpy as np
from common import *
from scipy.optimize import minimize
T=float(sys.argv[1]); rng=np.random.default_rng(int(sys.argv[2]))
q0=load(401); h=0.5*(abs(np.cos(q0[:,2]))+abs(np.sin(q0[:,2])))
q0[:,0]-=(q0[:,0]-h).min(); q0[:,1]-=(q0[:,1]-h).min(); L0=Lof(q0)
tilt=np.abs(np.sin(2*q0[:,2]))>0.5  # 45deg squares
cen=q0[tilt,:2]
# two diamonds: split by x+... use kmeans-ish on sign of (x-y) relative
lab=(q0[:,0]-q0[:,1]>np.median(q0[tilt,0]-q0[tilt,1]))
print('tilted',tilt.sum(),'L0',L0)
def relax(q,L,it=500):
    N=len(q)
    for r in range(3):
        P=pairs=ao.pairs(q,margin=0.6)
        f=lambda v:(lambda E,G:(E,G.ravel()))(*ao.penalty(v.reshape(N,3),L,P=P))
        res=minimize(f,q.ravel(),jac=True,method='L-BFGS-B',options=dict(maxiter=it,ftol=1e-22,gtol=1e-14))
        q=res.x.reshape(N,3)
    return q,res.fun
t0=time.time(); res=[]
Ls=[20.5355,20.532,20.529,20.526143]
while time.time()-t0<T:
    L=Ls[rng.integers(len(Ls))]
    q=q0.copy(); q[:,:2]*=L/L0
    mode=rng.integers(3)
    if mode>=1: # rotate each diamond by small delta, and give fan jitter to tilted squares
        for g in (True,False):
            m=tilt&(lab==g)
            if m.sum()==0: continue
            c=q[m,:2].mean(0); d=rng.normal(0,0.03)
            R=np.array([[math.cos(d),-math.sin(d)],[math.sin(d),math.cos(d)]])
            q[m,:2]=(q[m,:2]-c)@R.T+c; q[m,2]+=d
    if mode==2: q[tilt,2]+=rng.normal(0,0.02,tilt.sum())
    q[:,:2]+=rng.normal(0,0.01,(len(q),2))
    qr,E=relax(q,L)
    mv=max(ao.max_violation(qr,L)[:2])
    res.append((L,mode,E,mv)); print('L',L,'mode',mode,'E %.3e'%E,'maxv %.3e'%mv,round(time.time()-t0),flush=True)
