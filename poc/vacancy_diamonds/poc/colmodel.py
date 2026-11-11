"""Column-sliding model: tilted a x b blocks (45deg) + upright unit columns each with own x offset,
count = sum over free vertical intervals of floor(len). DP over column x offsets (exact except grid step)."""
import math, numpy as np
S2=math.sqrt(2)
def block_poly(cx,cy,a,b,flip=False):
    d1=np.array([1,1])/S2; d2=np.array([1,-1])/S2
    if flip: d1,d2=d2,d1
    c=np.array([cx,cy]); 
    return [c+s1*a/2*d1+s2*b/2*d2 for s1,s2 in [(-1,-1),(1,-1),(1,1),(-1,1)]]
def shadow(P,x0,x1):
    """y-range of convex polygon P within x in [x0,x1] or None."""
    ys=[]; n=len(P)
    for p in P:
        if x0<=p[0]<=x1: ys.append(p[1])
    for i in range(n):
        p,q=P[i],P[(i+1)%n]
        for xe in (x0,x1):
            if (p[0]-xe)*(q[0]-xe)<0:
                t=(xe-p[0])/(q[0]-p[0]); ys.append(p[1]+t*(q[1]-p[1]))
    return (min(ys),max(ys)) if ys else None
def colcount(x0,polys,L,rows=False):
    iv=[]
    for P in polys:
        s=shadow(P,x0,x0+1)
        if s: iv.append(s)
    iv.sort(); cur=0.0; n=0
    for lo,hi in iv:
        if lo>cur: n+=int(math.floor(lo-cur+1e-9))
        cur=max(cur,hi)
    n+=int(math.floor(L-cur+1e-9)) if L>cur else 0
    return n
def best(polys,L,nblk,step=0.005,transpose=False):
    if transpose: polys=[[np.array([p[1],p[0]]) for p in P] for P in polys]
    K=int(math.floor(L)); slack=L-K
    G=int(round(slack/step)); xs=np.arange(G+1)*step
    # column k left edge = k + xs[g], g nondecreasing
    best_prev=np.zeros(G+1); cnts=[]
    tot=np.zeros(G+1)
    for k in range(K):
        c=np.array([colcount(k+x,polys,L) for x in xs])
        new=np.maximum.accumulate(tot)+c if k>0 else c
        tot=new
    return int(tot.max())+nblk
