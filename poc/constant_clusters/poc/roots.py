import pandas as pd, numpy as np, json, math
d=pd.read_csv('poc/leader_curve_analysis/docs/per_n_comparison.csv').set_index('n')
cl=pd.read_csv('poc/constant_clusters/docs/clusters.csv')
cores=[c for c in json.load(open('poc/pell_corrugation/data/border-scan/cores.json')) if 'L_corrugated' in c]
fr=[(c['w'],c['n0'],c['L_corrugated']%1) for c in cores]
ours=d.L_ours; N=sorted(d.index)
# border bound: from ours at n' (floor-side) with t-thick upright L border:
# squares = n' + t*(floor(L'+t)) + t*floor(L')  (strips), L=L'+t
best={}
for n in N:
    b=(ours[n],None)
    for n2 in N:
        if n2>=n: break
        L2=ours[n2]; 
        for t in range(1,int(ours[n])-int(L2)+2):
            cnt=n2+t*math.floor(L2+t+1e-9)+t*math.floor(L2+1e-9)
            if cnt>=n and L2+t<b[0]-1e-9: b=(L2+t,(n2,t,cnt))
    best[n]=b
bb=[(n,ours[n],*best[n]) for n in N if best[n][1] is not None]
print('n where ours-border bound beats ours:',len(bb)); print(bb[:20])
# clusters we lose on
out=[]
for cid,g in cl.groupby('cid'):
    c=g.lbmK.mean()
    if c<0.5 or c>0.99 or g.lbmK.mean()==0: continue
    gap=(g.L_ours-g.L_lb)
    if gap.max()<=1e-9: continue
    g=g.sort_values('n'); r=g.iloc[0]
    sm=d[(d.index<r.n)&(abs(d.ours_mK-c)<2e-3)]
    pm=[f for f in fr if abs(f[2]-c)<2e-3]
    out.append((cid,round(c,9),int(r.n),int(r.K),int(r.D),r.team[:10],len(g),[int(x) for x in g.n],round(float(gap.max()),4),list(sm.index[:5]),pm))
    print(out[-1])
pd.DataFrame(out,columns='cid c root_n root_K root_D team members_cnt members max_gap ours_equal_c_smaller_n pell_frac_match'.split()).to_csv('poc/constant_clusters/docs/lost_clusters.csv',index=False)
