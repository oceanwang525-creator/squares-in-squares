import pandas as pd, numpy as np, json, math
d=pd.read_csv('poc/leader_curve_analysis/docs/per_n_comparison.csv').set_index('n')
cl=pd.read_csv('poc/constant_clusters/docs/clusters.csv')
cl=cl[(cl.lbmK>0.5)&(cl.lbmK<0.99)&(cl.lbmK!=0)]
cores=[c for c in json.load(open('poc/pell_corrugation/data/border-scan/cores.json')) if 'L_corrugated' in c]
print('cores',[(c['w'],c['n0'],round(c['L_corrugated'],6),round(c['L_corrugated']-math.floor(c['L_corrugated']),6)) for c in cores])
for cid,g in cl.groupby('cid'):
    g=g.sort_values('n'); c=g.lbmK.mean()
    print(f'\n== cluster {cid} c={c:.9f}  gap(ours-lb): ', [(r.n,r.K,r.D,round(r.L_ours-r.L_lb,4)) for r in g.itertuples()])
    # pairwise arithmetic
    m=g.n.tolist(); Ks=g.K.tolist()
    for i in range(len(m)-1):
        for j in range(i+1,len(m)):
            t=Ks[j]-Ks[i]; Lc=Ks[i]+c
            print(f'  {m[i]}->{m[j]} t={t} dn={m[j]-m[i]} 2t*floor(Lc)+t^2={2*t*math.floor(Lc)+t*t} 2t(K)+t^2={2*t*Ks[i]+t*t}')
    # ours match: any n' with ours L - (K-t) ~ c
    for r in g.itertuples():
        hits=[]
        for t in range(1,r.K):
            Kp=r.K-t
            for n2,row in d.iterrows():
                if n2<r.n and abs(row.L_ours-t-(r.L_lb-0)+0)<1e-3 and False: pass
            # ours side L_ours(n') ~ L_lb - t
            tgt=r.L_lb-t
            m_=d[(abs(d.L_ours-tgt)<2e-4)&(d.index<r.n)]
            for n2,row in m_.iterrows(): hits.append((t,n2,round(row.L_ours-tgt,7),round(r.n-n2-2*t*math.floor(tgt)-t*t,1)))
        print('  member',r.n,'ours-core hits (t,n_core,Lours-tgt,n_extra over 2t*floor+t^2):',hits[:8])
