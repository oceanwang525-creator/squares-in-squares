"""For n=K^2+D: min L over (p,q,W,T1,T2) rational-band estimator templates with count>=n vs ours/leader."""
import sys, math, pandas as pd
from math import gcd
sys.path.insert(0,'poc')
from band_count import lat
def row(p,q,c,L):
    s=math.hypot(p,q); h=(p+q)/(2*s)
    xl=max(h,(c-q*(L-h))/p); xh=min(L-h,(c-q*h)/p)
    if xh<xl: return 0
    return int(math.floor((xh-xl)*s/q+1e-9))+1
d=pd.read_csv('../leader_curve_analysis/docs/per_n_comparison.csv').set_index('n')
def minL(K,D,mp_=12):
    n=K*K+D; best=(9e9,)
    for p in range(1,mp_):
      for q in range(p,mp_):
        if gcd(p,q)!=1: continue
        s=math.hypot(p,q)
        for W in range(1,7):
          for S in range(0,(p+q)*(K+2)):
            L=2+(S+W*s)/(p+q)
            if not(K+0.3<=L<K+1.0) or L>=best[0]: continue
            m=int(L+1e-9)
            for T1 in range(S+1):
                if lat(p,q,T1,m)+lat(p,q,S-T1,m)+sum(row(p,q,T1+p+q+(r+.5)*s,L) for r in range(W))>=n:
                    best=(L,p,q,W,T1,S-T1); break
    return best
for D in (10,):
  for K in range(14,27):
    n=K*K+D; b=minL(K,D)
    print(K,n,"template L-K=%.5f"%(b[0]-K),b[1:],"ours %.5f lead %.5f"%(d.loc[n,'ours_mK'],d.loc[n,'lb_mK']),flush=True)
