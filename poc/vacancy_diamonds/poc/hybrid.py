"""Peel full rows/cols off a rectangle (where free of blocks) then column/row-slide DP in the core."""
import math, numpy as np, itertools, sys
sys.path.insert(0,'poc')
from colmodel import block_poly, shadow, S2
def colc(x0,polys,y0,y1):
    iv=[]
    for P in polys:
        s=shadow(P,x0,x0+1)
        if s: iv.append(s)
    iv.sort(); cur=y0; n=0
    for lo,hi in iv:
        lo=max(lo,y0)
        if lo>cur: n+=int(math.floor(lo-cur+1e-9))
        cur=max(cur,hi)
    if y1>cur: n+=int(math.floor(y1-cur+1e-9))
    return n
def slide(polys,x0,x1,y0,y1,step):
    K=int(math.floor(x1-x0+1e-9)); 
    if K<=0 or y1-y0<1: return 0
    slack=x1-x0-K; G=max(0,int(round(slack/step))); xs=x0+np.arange(G+1)*step
    tot=None
    for k in range(K):
        c=np.array([colc(k+x,polys,y0,y1) for x in xs])
        tot=c if tot is None else np.maximum.accumulate(tot)+c
    return int(tot.max())
def tr(polys): return [[np.array([p[1],p[0]]) for p in P] for P in polys]
def bbox(P):
    a=np.array(P); return a[:,0].min(),a[:,0].max(),a[:,1].min(),a[:,1].max()
def count(polys,L,step=0.02,maxpeel=9):
    bbs=[bbox(P) for P in polys]
    def free_strip(xa,xb,ya,yb): return all(b[1]<=xa+1e-9 or b[0]>=xb-1e-9 or b[3]<=ya+1e-9 or b[2]>=yb-1e-9 for b in bbs)
    best=(0,None); W=int(L)
    for rowsfirst in (True,False):
      for t in range(maxpeel):
        if t and not all(free_strip(0,L,L-j-1,L-j) for j in range(t)): break
        for b in range(maxpeel):
          if b and not all(free_strip(0,L,j,j+1) for j in range(b)): break
          for term in (0,1):
            # then cols l,r : columns height depends on order
            for l in range(maxpeel):
              for r in range(maxpeel):
                if rowsfirst: ya,yb=b,L-t; ca=0; # rows full width
                x0,x1=l,L-r
                if rowsfirst:
                    nrow=(t+b)*W; ya,yb=b,L-t
                    cy0,cy1=ya,yb; cn=0
                    ok=all(free_strip(j,j+1,ya,yb) for j in range(l)) and all(free_strip(L-j-1,L-j,ya,yb) for j in range(r))
                    if not ok: break
                    cn=(l+r)*int(math.floor(yb-ya+1e-9)); core=(x0,x1,ya,yb); base=nrow+cn
                else:
                    ya,yb=0,L
                    ok=all(free_strip(j,j+1,0,L) for j in range(l)) and all(free_strip(L-j-1,L-j,0,L) for j in range(r))
                    if not ok: break
                    cn=(l+r)*W; ya,yb=b,L-t
                    ok2=all(free_strip(x0,x1,L-j-1,L-j) for j in range(t)) and all(free_strip(x0,x1,j,j+1) for j in range(b))
                    if not ok2: continue
                    base=cn+(t+b)*int(math.floor(x1-x0+1e-9)); core=(x0,x1,ya,yb)
                x0,x1,ya,yb=core
                if term==0: v=slide(polys,x0,x1,ya,yb,step)
                else: v=slide(tr(polys),ya,yb,x0,x1,step)
                if base+v>best[0]: best=(base+v,(rowsfirst,t,b,l,r,term))
    return best
if __name__=='__main__':
    L=20.535534; h=L/2
    P=[block_poly(3+h,-5.99+h,5,5),block_poly(-6+h,3+h,5,5)]
    print(count(P,L,0.05,6))
