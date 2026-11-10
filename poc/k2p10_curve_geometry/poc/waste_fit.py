import pandas as pd, numpy as np
d=pd.read_csv('../leader_curve_analysis/docs/per_n_comparison.csv')
d['Wo']=d.L_ours**2-d.n; d['Wl']=d.L_lb**2-d.n
print("D  slopeOurs  slopeLead (W vs K, K=22..31) ; lead W/ (sqrt2 K) ; lead 1/K-law a,b")
for D in range(8,14):
    s=d[(d.D==D)&(d.K>=22)]
    po=np.polyfit(s.K,s.Wo,1); pl=np.polyfit(s.K,s.Wl,1)
    # lower envelope (monotone staircase floor) for lead
    print(D, "ours W=%.3f K%+.2f"%(po[0],po[1]), "lead W=%.3f K%+.2f"%(pl[0],pl[1]),
      "implied dL/dK-asympt lead=%.3f"%(pl[0]/2))
# all D 4..20 pooled lead slope
for D in (4,6,8,10,12,14,16):
    s=d[(d.D==D)&(d.K>=22)]; print(D, np.polyfit(s.K,s.Wl,1).round(3), np.polyfit(s.K,s.Wo,1).round(3))
