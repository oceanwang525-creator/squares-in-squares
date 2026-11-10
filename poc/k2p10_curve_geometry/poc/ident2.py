import pandas as pd
from mpmath import mp, mpf, findpoly, identify
mp.dps=30
p=pd.read_csv('../leader_curve_analysis/data/perpuzzle_best.csv',dtype={'L':str}).set_index('n')
for n in (537,739,794,908,969,854,635,686,687,851,910,971,972,911,795):
    x=mpf(p.loc[n,'L'][:32]); K=int(x)
    f=x-K
    r=[findpoly(f,deg,maxcoeff=60,tol=1e-12) for deg in (1,2,3,4)]
    print(n,p.loc[n,'L'][:26],[q for q in r if q])
