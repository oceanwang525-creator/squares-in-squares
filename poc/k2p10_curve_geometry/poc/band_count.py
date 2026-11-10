"""Estimate square count of a rational-slope (p,q) single-band template.
BL staircase: upright cells (i,j) with p*i+q*j<=T1 ; TR staircase: same with T2 ; chain of tilted squares (angle atan(p/q) band dir (q,-p)) width exactly 1 between the tip lines.
L = 2 + (T1+T2+sqrt(p^2+q^2))/(p+q).  Count estimate only; real packing needs MILP/overlap check."""
import math, sys, numpy as np
from math import gcd
def lat(p,q,T,m):
    c=0
    for j in range(m):
        r=T-q*j
        if r<0: break
        c+=min(m,r//p+1)
    return c
def chain(p,q,T1,L):
    s=math.hypot(p,q); h=(p+q)/(2*s)
    # tip line (BL far corner): p x+q y = T1+p+q ; centreline offset +0.5 s
    c=(T1+p+q)+0.5*s            # p x + q y = c
    # centre (x,y) in [h,L-h]^2 ; param x = x0 + tau*q/s ... find extent along direction (q,-p)/s
    # points: x in[h,L-h], y=(c-p x)/q in [h,L-h]
    lo,hi=-1e9,1e9
    if p>0 and q>0:
        # x range from y bounds: x=(c-q y)/p
        xs=[h,L-h,(c-q*(L-h))/p,(c-q*h)/p]
        xl=max(h,(c-q*(L-h))/p); xh=min(L-h,(c-q*h)/p)
    if xh<xl: return 0
    ell=(xh-xl)*s/q
    return int(math.floor(ell+1e-9))+1
def best(K,Lmax,Lmin,target,maxpq=30):
    out=[]
    for p in range(1,maxpq):
      for q in range(p,maxpq):
        if gcd(p,q)!=1: continue
        s=math.hypot(p,q)
        for S in range(0,(p+q)*(K+2)):
            L=2+(S+s)/(p+q)
            if not(Lmin<=L<Lmax): continue
            m=int(math.floor(L+1e-9))
            for T1 in range(0,S+1):
                T2=S-T1
                for (a,b) in ((p,q),):
                    n=lat(p,q,T1,m)+lat(p,q,T2,m)+chain(p,q,T1,L)
                    if n>=target: out.append((L,n,p,q,T1,T2))
    return sorted(out)
if __name__=="__main__":
    K=int(sys.argv[1]); D=int(sys.argv[2]); Lmax=float(sys.argv[3])
    r=best(K,Lmax,K+0.5,K*K+D,maxpq=int(sys.argv[4]) if len(sys.argv)>4 else 20)
    for t in r[:15]: print(t)
    print(len(r))
