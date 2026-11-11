import pandas as pd, numpy as np
d=pd.read_csv('poc/leader_curve_analysis/docs/per_n_comparison.csv')
d['lbmK']=d.L_lb-d.K
d=d[d.n>=2].sort_values('lbmK').reset_index(drop=True)
cl=[];cur=[0]
for i in range(1,len(d)):
    if d.lbmK[i]-d.lbmK[i-1]<1e-8: cur.append(i)
    else: cl.append(cur);cur=[i]
cl.append(cur)
rows=[]
for c in cl:
    if len(c)<2: continue
    s=d.loc[c]
    # skip integer-ish trivial K? keep all
    rows.append(s.assign(cid=len(rows)))
    
out=pd.concat(rows)
out.to_csv('poc/constant_clusters/docs/clusters.csv',index=False)
for cid,g in out.groupby('cid'):
    print(cid,round(g.lbmK.mean(),9),[(r.n,r.K,r.D,r.team[:8],round(r.L_ours-r.L_lb,4)) for r in g.itertuples()])
