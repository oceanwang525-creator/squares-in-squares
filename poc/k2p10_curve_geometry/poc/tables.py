import pandas as pd, numpy as np, math
d = pd.read_csv('../leader_curve_analysis/docs/per_n_comparison.csv')
d['W_ours']=d.L_ours**2-d.n; d['W_lb']=d.L_lb**2-d.n
out=[]
for D in range(8,14):
    s=d[(d.D==D)&(d.K>=10)].sort_values('K')
    out.append(f"=== D={D}\nK  n  ours-K  lead-K  team  Wours  Wlead")
    for _,r in s.iterrows():
        out.append(f"{r.K:2d} {r.n:4d} {r.ours_mK:.5f} {r.lb_mK:.5f} {str(r.team)[:10]:10s} {r.W_ours:6.2f} {r.W_lb:6.2f}")
    s2=s[s.K>=24]
    for name,f in [('a+b/K',lambda K:1/K),('a+b/sqrtK',lambda K:K**-.5),('a+bK',lambda K:K)]:
        for col in ('ours_mK','lb_mK'):
            A=np.c_[np.ones(len(s2)),f(s2.K.values)]
            c,res,*_=np.linalg.lstsq(A,s2[col].values,rcond=None)
            rms=np.sqrt(np.mean((A@c-s2[col].values)**2))
            out.append(f"fit K>=24 {col} {name}: a={c[0]:.4f} b={c[1]:.4f} rms={rms:.4f}")
open('docs/D_tables.txt','w').write("\n".join(out)); print("\n".join(out))
