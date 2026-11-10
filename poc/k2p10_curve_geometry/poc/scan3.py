"""Multi-row (W rows) rational-slope band estimator; same staircases as band_count."""
import sys, math
from math import gcd
sys.path.insert(0,'poc')
from band_count import lat
def row(p,q,c,L):
    s=math.hypot(p,q); h=(p+q)/(2*s)
    xl=max(h,(c-q*(L-h))/p); xh=min(L-h,(c-q*h)/p)
    if xh<xl: return 0
    return int(math.floor((xh-xl)*s/q+1e-9))+1
K=int(sys.argv[1]); target=K*K+int(sys.argv[2]); Lmax=float(sys.argv[3]); mp_=int(sys.argv[4])
res=[]
for p in range(1,mp_):
  for q in range(p,mp_):
    if gcd(p,q)!=1: continue
    s=math.hypot(p,q)
    for W in (1,2,3,4,5,6):
      for S in range(0,(p+q)*(K+2)):
        L=2+(S+W*s)/(p+q)
        if not(K+0.5<=L<Lmax): continue
        m=int(L+1e-9)
        for T1 in range(S+1):
            n=lat(p,q,T1,m)+lat(p,q,S-T1,m)+sum(row(p,q,T1+p+q+(r+.5)*s,L) for r in range(W))
            if n>=target-2: res.append((-n,L,p,q,W,T1,S-T1))
res.sort()
for t in res[:12]: print(-t[0],"short",target+t[0],"L-K=%.5f"%(t[1]-K),t[2:])
