import pandas as pd,numpy as np,collections
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
import scipy.sparse as sp
df=pd.read_csv("/mnt/e/squares-poc-regenerate-nov13/poc/pell_corrugation/data/border-scan/merged.csv")
q=df[df.n==908][["x","y","angle"]].to_numpy()
a=np.degrees(q[:,2])%90; a=np.where(a>89.999,0,a); a=np.where(a<1e-6,0,a)
t=np.where(a>0)[0]; print("tilted",len(t),"upright",908-len(t))
tr=cKDTree(q[t,:2]); P=tr.query_pairs(1.5,output_type="ndarray")
g=sp.coo_matrix((np.ones(len(P)),(P[:,0],P[:,1])),shape=(len(t),)*2)
k,lab=connected_components(g,directed=False)
for c in range(k):
    m=t[lab==c]
    if len(m)<3: continue
    aa=a[m]; cnt=collections.Counter(np.round(aa,1)).most_common(4)
    print(f"cluster {c}: n={len(m)} x[{q[m,0].min():.2f},{q[m,0].max():.2f}] y[{q[m,1].min():.2f},{q[m,1].max():.2f}] centroid=({q[m,0].mean():.2f},{q[m,1].mean():.2f}) angles={cnt}")
print("small clusters:",[ (int((lab==c).sum())) for c in range(k) if (lab==c).sum()<3])
