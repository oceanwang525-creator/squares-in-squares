#!/usr/bin/env python3
"""Family analysis of Cats-and-boxes gaps; reads docs/per_n_comparison.csv, writes docs/*.csv."""
import math, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from compare import build, D
t = build(); t["mass"] = t.gap.clip(lower=0)/np.sqrt(t.n)
t["lb_ratio"] = t.L_lb/np.sqrt(t.n)-1
t["off"] = t.K*(t.K+1)-t.n   # distance below k(k+1)
c = t[(t.team=="Cats and boxes")&t.beats].copy()
print("cats beats", len(c), "mass", c.mass.sum(), "total all mass", t.mass.sum())
print(t[t.beats].groupby("team").mass.agg(["size","sum"]))
# ours flat levels
pd.set_option("display.width",250); pd.set_option("display.max_rows",500)
# by D: slope of L-K vs K for each D (ours, lb)
rows=[]
for d,g in t[(t.D>0)].groupby("D"):
    g=g[g.K>=8]
    if len(g)<6: continue
    cg=g[g.team=="Cats and boxes"]
    def fit(y,x):
        A=np.c_[np.ones_like(x),1/x]; co=np.linalg.lstsq(A,y,rcond=None)[0]; return co
    ao=fit(g.ours_mK.values,g.K.values.astype(float))
    al=fit(g.lb_mK.values,g.K.values.astype(float))
    rows.append(dict(D=d,nK=len(g),n_cats=len(cg),ours_mean=g.ours_mK.mean(),lb_mean=g.lb_mK.mean(),
      ours_first=g.ours_mK.iloc[0],ours_last=g.ours_mK.iloc[-1],lb_first=g.lb_mK.iloc[0],lb_last=g.lb_mK.iloc[-1],
      ours_a=ao[0],ours_b=ao[1],lb_a=al[0],lb_b=al[1],mass=g.mass.sum()))
fd=pd.DataFrame(rows); fd.to_csv(D/"docs/by_D.csv",index=False)
print(fd.sort_values("mass",ascending=False).head(25).round(4).to_string())
# by offset below k(k+1)
rows=[]
for o,g in t[(t.off>=0)&(t.K>=8)&(t.D>0)].groupby("off"):
    rows.append(dict(off=o,cnt=len(g),beats=int(g.beats.sum()),ours_mean=g.ours_mK.mean(),lb_mean=g.lb_mK.mean(),mass=g.mass.sum()))
fo=pd.DataFrame(rows); fo.to_csv(D/"docs/by_offset.csv",index=False)
print(fo.sort_values("mass",ascending=False).head(15).round(4).to_string())
# K-level table: mean ours/lb L-K by K
fk=t[t.D>0].groupby("K").agg(cnt=("n","size"),ours=("ours_mK","mean"),lb=("lb_mK","mean"),mass=("mass","sum"),beats=("beats","sum"))
fk.to_csv(D/"docs/by_K.csv"); print(fk.round(4).to_string())
# top gaps
top=t.sort_values("mass",ascending=False).head(40)[["n","K","D","off","L_ours","L_lb","team","ours_mK","lb_mK","gap","mass"]]
top.to_csv(D/"docs/top_gaps.csv",index=False); print(top.round(5).to_string())
# stock constants
for v in [.707107,.727922,.637029]:
    s=t[(t.ours_mK-v).abs()<2e-6]
    print("const",v,len(s),"beats",int(s.beats.sum()),"mass",s.mass.sum(), "lb_mK mean",s.lb_mK[s.beats].mean())
    s2=s[s.beats]; print(s2.groupby("K").lb_mK.agg(["size","mean"]).round(4).T.to_string())
