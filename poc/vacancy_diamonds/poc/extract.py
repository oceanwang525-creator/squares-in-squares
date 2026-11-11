import pandas as pd, numpy as np, math
d=pd.read_csv('poc/pell_corrugation/data/border-scan/merged.csv')
s=d[d.n==401]; q=s[['x','y','angle']].values
a=np.mod(q[:,2],math.pi/2); a=np.minimum(a,math.pi/2-a)  # dist to upright
tilt=np.abs(a-math.pi/4)<0.02
up=a<0.02
oth=~(tilt|up)
print(len(q),'tilt',tilt.sum(),'upright',up.sum(),'other',oth.sum())
h=0.5*(abs(np.cos(q[:,2]))+abs(np.sin(q[:,2])))
L=max((q[:,:2]+h[:,None]).max(),0)
print('L',L,'min',(q[:,:2]-h[:,None]).min())
np.save('poc/vacancy_diamonds/data/q401.npy',q)
# tilted lattice coords u=(x+y)/√2... rotate 45
T=q[tilt]; u=(T[:,0]+T[:,1])/math.sqrt(2); v=(T[:,0]-T[:,1])/math.sqrt(2)
from scipy.cluster.hierarchy import fcluster,linkage
Z=fcluster(linkage(T[:,:2],'single'),1.2,'distance'); 
for c in np.unique(Z):
    m=Z==c; print('cluster',c,m.sum(),'center',T[m,:2].mean(0).round(2))
    uu=np.sort(np.unique(np.round(u[m]*1,2)));vv=np.sort(np.unique(np.round(v[m],2)))
    print(' u distinct',len(uu),uu[:12]);print(' v distinct',len(vv),vv[:12])
    # row counts along v rows
    rows={}
    for a_,b_ in zip(np.round(u[m]),np.round(v[m])): rows.setdefault(a_,[]).append(b_)
    print(' rows(u)->count',{k:len(r) for k,r in sorted(rows.items())})
    # in-lattice coords relative to cluster min
    uu0=u[m]-u[m].min(); vv0=v[m]-v[m].min()
    print(' u frac range',uu0.max().round(2),'v range',vv0.max().round(2), 'frac u',np.round(uu0%1,2)[:15])
print('other',q[oth].round(3))
