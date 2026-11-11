import numpy as np, pandas as pd, math, sys
sys.path.insert(0,'/Users/lucaszhang/squares-in-squares/poc/k2p10_fill2/poc')
import a_overlap as ao
MERGED='/Users/lucaszhang/squares-in-squares/poc/pell_corrugation/data/border-scan/merged.csv'
def load(n):
    d=pd.read_csv(MERGED); r=d[d.n==n].sort_values('square_id')
    return r[['x','y','angle']].values.copy()
def extent(q):
    h=0.5*(abs(np.cos(q[:,2]))+abs(np.sin(q[:,2])))
    return (q[:,0]-h).min(),(q[:,1]-h).min(),(q[:,0]+h).max(),(q[:,1]+h).max()
def Lof(q):
    a,b,c,d=extent(q); return max(c-a,d-b)
