import sys, math, numpy as np, time
sys.path.insert(0,'/Users/lucaszhang/squares-in-squares/poc/k2p10_fill2/poc')
from a_overlap import pairs, pair_depth
from scipy.optimize import linprog
from scipy.sparse import lil_matrix
q=np.load('data/q401.npy'); N=len(q)
def hh(t): return 0.5*(abs(np.cos(t))+abs(np.sin(t)))
q[:,:2]+=0  # centered coords; container [-L/2,L/2] -> shift to [0,L]
L0=20.535534; 
# true L from data
h=hh(q[:,2]); Lc=max(((q[:,:2]+h[:,None]).max()-(q[:,:2]-h[:,None]).min()),0); 
lo=(q[:,:2]-h[:,None]).min(); q[:,:2]-=lo; L=(q[:,:2]+h[:,None]).max(); print('L',L,'min',(q[:,:2]-h[:,None]).min())
def maxviol(q,L):
    P=pairs(q); dep=pair_depth(q,P); h=hh(q[:,2])
    w=max((h-q[:,0]).max(),(h-q[:,1]).max(),(q[:,0]+h-L).max(),(q[:,1]+h-L).max())
    return max(dep.max(),w)
print('start viol',maxviol(q,L))
tr=0.03; mar=1e-9; t0=time.time(); hist=[]
while time.time()-t0<240 and tr>1e-5:
    P=pairs(q,0.2); dep,gpi,gpj,dti,dtj=pair_depth(q,P,True)
    m=dep>-0.15; P=P[m];dep=dep[m];gpi=gpi[m];gpj=gpj[m];dti=dti[m];dtj=dtj[m]
    nv=3*N+1; rows=len(P)+4*N
    A=lil_matrix((rows,nv)); b=np.zeros(rows)
    for r,(i,j) in enumerate(P):
        A[r,3*i:3*i+2]=gpi[r]; A[r,3*i+2]=dti[r]; A[r,3*j:3*j+2]=gpj[r]; A[r,3*j+2]=dtj[r]; b[r]=-dep[r]-mar
    h=hh(q[:,2]); dh=0.5*(-np.sign(np.cos(q[:,2]))*np.sin(q[:,2])+np.sign(np.sin(q[:,2]))*np.cos(q[:,2]))
    r=len(P)
    for i in range(N):
        for ax in (0,1):
            # -x + h <=0 : -dx + dh dth <= x-h
            A[r,3*i+ax]=-1; A[r,3*i+2]=dh[i]; b[r]=q[i,ax]-h[i]-mar; r+=1
            # x + h - L <=0
            A[r,3*i+ax]=1; A[r,3*i+2]=dh[i]; A[r,3*N]=-1; b[r]=L-q[i,ax]-h[i]-mar; r+=1
    c=np.zeros(nv); c[3*N]=1
    bnds=[]
    for i in range(N):
        a=q[i,2]%(math.pi/2); off=min(a,math.pi/2-a)
        ta=min(tr,0.5*off) if off>0.003 else 0.0
        if abs(off-math.pi/4)<0.03: ta=0.0 if abs(off-math.pi/4)<0.003 else min(tr,0.5*abs(off-math.pi/4))
        bnds+=[(-tr,tr),(-tr,tr),(-ta,ta)]
    bnds+=[(-0.05,0.05)]
    res=linprog(c,A_ub=A.tocsr(),b_ub=b,bounds=bnds,method='highs')
    if res.status!=0: tr*=0.5; continue
    d=res.x; qn=q.copy(); qn[:,:2]+=d[:3*N].reshape(N,3)[:,:2]; qn[:,2]+=d[:3*N].reshape(N,3)[:,2]; Ln=L+d[3*N]
    v=maxviol(qn,Ln)
    print(f'tr={tr:.4g} dL={d[3*N]:.3e} L={Ln:.6f} viol={v:.2e}',flush=True)
    if v<=1e-10 or (v<1e-7 and d[3*N]<-1e-9):
        q,L=qn,Ln; hist.append(L)
        if v>1e-10: pass
    else: tr*=0.5
print('final L',L,'viol',maxviol(q,L)); np.save('data/slp401.npy',np.c_[q,np.full(N,L)])
