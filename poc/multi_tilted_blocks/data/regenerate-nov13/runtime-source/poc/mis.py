"""MIS over candidate pool (upright anchored lattices + tilted lattice patches) at fixed L; HiGHS MILP. Usage: mis.py L tlimit nphase"""
import sys,math,time,numpy as np
sys.path.insert(0,"/mnt/e/squares-poc-regenerate-nov13/poc/k2p10_fill2/poc")
from a_overlap import pairs,pair_depth
import scipy.sparse as sp
from scipy.optimize import milp,LinearConstraint,Bounds
L=float(sys.argv[1]); TL=float(sys.argv[2]); NP=int(sys.argv[3])
th0=math.radians(24.1004); th1=math.atan(.5)
cands=[]
# upright: 4 anchors
m=int(L)
for ax in (0,1):
  for ay in (0,1):
    for i in range(m):
      for j in range(m):
        x=i+.5 if ax==0 else L-i-.5; y=j+.5 if ay==0 else L-j-.5
        cands.append((x,y,0.0))
cands=list(dict.fromkeys(cands))
nup=len(cands)
# regions (xlo,xhi,ylo,yhi) for tilted patches: NW, SE, SW-centre, NE-centre
regs=[(0,16,14,L),(12,L,0,16),(0,13,0,13),(L-13,L,L-13,L)]
for th in (th0,-th0,th1,-th1):
  c,s=math.cos(th),math.sin(th)
  for (xl,xh,yl,yh) in regs:
    for pa in range(NP):
      for pb in range(NP):
        for reg_ox in (0,):
          ox=xl+(xh-xl)/2+pa/NP; oy=yl+(yh-yl)/2+pb/NP
          for i in range(-14,15):
            for j in range(-14,15):
              x=ox+i*c-j*s; y=oy+i*s+j*c
              if xl<=x<=xh and yl<=y<=yh: cands.append((x,y,th%(math.pi/2)))
q=np.array(cands); h=.5*(abs(np.cos(q[:,2]))+abs(np.sin(q[:,2])))
ok=(q[:,0]>=h-1e-9)&(q[:,0]<=L-h+1e-9)&(q[:,1]>=h-1e-9)&(q[:,1]<=L-h+1e-9)
q=q[ok]; q=np.unique(np.round(q,9),axis=0)
print("pool",len(q),"upright",int((q[:,2]==0).sum()),flush=True)
P=pairs(q); d=pair_depth(q,P); P=P[d>1e-8]; print("edges",len(P),flush=True)
N=len(q)
# greedy-ish clique cover: use edge constraints
A=sp.coo_matrix((np.ones(2*len(P)),(np.repeat(np.arange(len(P)),2),P.ravel())),shape=(len(P),N)).tocsr()
t=time.time()
r=milp(-np.ones(N),constraints=LinearConstraint(A,-np.inf,1),integrality=np.ones(N),bounds=Bounds(0,1),options=dict(time_limit=TL,disp=False))
print("status",r.status,r.message,"time",time.time()-t)
if r.x is None: print("no sol"); sys.exit()
sel=r.x>.5; print(f"L={L} count={sel.sum()} tilted={(q[sel,2]!=0).sum()} upright={(q[sel,2]==0).sum()} bound={-r.mip_dual_bound if hasattr(r,'mip_dual_bound') else None}")
np.save(f"/mnt/e/squares-poc-regenerate-nov13/poc/multi_tilted_blocks/data/sel_L{L}.npy",q[sel])
