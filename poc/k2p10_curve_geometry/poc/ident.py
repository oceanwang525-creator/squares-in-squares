import pandas as pd, math, itertools, sys
from mpmath import mp, mpf, sqrt, findpoly, pslq
mp.dps=40
d=pd.read_csv('../leader_curve_analysis/docs/per_n_comparison.csv')
p=pd.read_csv('../leader_curve_analysis/data/perpuzzle_best.csv',dtype={'L':str})
p=p.set_index('n')
print(p.loc[686].to_dict())
# closed forms L=(m+w*sqrt(a^2+b^2))/(a+b) , m integer, w in 1..4, a,b<=20 ; tol 1e-9
res=[]
for D in (8,9,10,11,12,13):
  for K in range(14,32):
    n=K*K+D; L=float(p.loc[n,'L'])
    hit=[]
    for a in range(0,25):
      for b in range(1,25):
        s=math.hypot(a,b)
        for w in (1,2,3,4):
          for c in (a+b,2*(a+b),(a+b)*3):
            m=L*c-w*s
            if abs(m-round(m))<1e-8 and math.gcd(a,b)==1:
              hit.append((a,b,w,c,int(round(m))))
    team=d[d.n==n].team.iloc[0]
    print(D,K,n,f"{L-K:.9f}",team[:8],hit[:4])
