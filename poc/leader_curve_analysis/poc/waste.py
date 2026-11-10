#!/usr/bin/env python3
"""Wasted area W=L^2-n vs K: power-law exponents for ours vs leaders, and reuse of L-K constants."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from compare import build, D as DD
t = build(); t["W_o"]=t.L_ours**2-t.n; t["W_l"]=t.L_lb**2-t.n
rows=[]
for d,g in t[(t.D>0)&(t.K>=14)].groupby("D"):
    if len(g)<10: continue
    po=np.polyfit(np.log(g.K),np.log(g.W_o),1); pl=np.polyfit(np.log(g.K),np.log(g.W_l),1)
    rows.append(dict(D=d,exp_ours=po[0],exp_lb=pl[0],W_o_K30=g[g.K==30].W_o.mean(),W_l_K30=g[g.K==30].W_l.mean()))
f=pd.DataFrame(rows); f.to_csv(DD/"docs/waste_exponents.csv",index=False)
print(f.round(3).to_string()); print(f[["exp_ours","exp_lb"]].describe().round(3))
# overall: all n with K>=14, D>0
g=t[(t.D>0)&(t.K>=14)]
# bin by D/K regime: fit W vs L for rows where D <= K (i.e. L-K <~1)
for lab,col in [("ours","W_o"),("lb","W_l")]:
    p=np.polyfit(np.log(g.K),np.log(g[col]),1); print(lab,"global W~K^%.3f"%p[0])
c=t[(t.team=="Cats and boxes")&t.beats]
r=c.lb_mK.round(5).value_counts(); print("distinct lb L-K among Cats-beat n:",len(r),"reused (>=2):",(r>=2).sum(), "n covered by reused:",r[r>=2].sum())
print(r.head(15))
print("their-L-K reuse also in OUR set:", c.lb_mK.round(5).isin(t.ours_mK.round(5)).sum())
