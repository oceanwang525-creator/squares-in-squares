#!/usr/bin/env python3
"""Per-D tables (L-K vs K, ours vs leaders) and gap-vs-K fits; writes docs/D_tables.txt, docs/gap_fits.csv."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from compare import build, D as DD
t = build(); t["mass"] = t.gap.clip(lower=0)/np.sqrt(t.n)
out=[]
for d in [1,2,3,4,7,9,10,11,12,16,19,20]:
    g=t[(t.D==d)&(t.K>=10)]
    out.append(f"D={d}\n"+g[["n","K","ours_mK","lb_mK","gap","team"]].round(5).to_string(index=False))
open(DD/"docs/D_tables.txt","w").write("\n\n".join(out))
# fits on Cats-beaten n with K>=12: gap ~ a K^p ; and lb_mK vs ours_mK per-n
c=t[(t.team=="Cats and boxes")&t.beats&(t.K>=12)]
p=np.polyfit(np.log(c.K),np.log(c.gap),1); print("gap ~ K^%.3f"%p[0], "n",len(c), "median gap",c.gap.median())
print("gap/ours_mK  by K:"); print((c.gap/c.ours_mK).groupby(c.K).mean().round(4).to_string())
print("mean gap by K:"); print(c.gap.groupby(c.K).agg(["size","mean"]).round(4).to_string())
# model fits for all D>0, n beating, ours & theirs: L-K = a + b/K, a+b/sqrt K, a + b/K^(4/11)  per-D (rms)
rows=[]
for d,g in t[(t.D>0)&(t.K>=12)].groupby("D"):
    if len(g)<8: continue
    r={"D":d,"n":len(g)}
    for lab,y in [("ours",g.ours_mK.values),("lb",g.lb_mK.values)]:
        K=g.K.values.astype(float)
        for nm,f in [("1/K",1/K),("1/sqrtK",K**-.5),("K^-4/11",K**(-4/11)),("K^-1/2*..",K**-.25)]:
            A=np.c_[np.ones_like(K),f]; co,res,*_=np.linalg.lstsq(A,y,rcond=None)
            r[f"{lab}_{nm}_a"]=co[0]; r[f"{lab}_{nm}_b"]=co[1]; r[f"{lab}_{nm}_rms"]=np.sqrt(np.mean((A@co-y)**2))
    rows.append(r)
f=pd.DataFrame(rows); f.to_csv(DD/"docs/gap_fits.csv",index=False)
cols=[c for c in f.columns if c.endswith("rms")]
print(f[cols].mean().round(5).to_string())
print(f[["D","ours_1/K_a","ours_1/K_b","lb_1/K_a","lb_1/K_b"]].round(3).head(30).to_string())
