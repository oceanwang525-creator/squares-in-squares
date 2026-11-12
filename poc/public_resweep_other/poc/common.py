import sys,json,os,re,hashlib,glob
sys.path.insert(0,'/Users/lucaszhang/squares-in-squares/poc/baseline/poc/src')
import numpy as np,pandas as pd
from decimal import Decimal
R='/Users/lucaszhang/squares-in-squares/'
F=R+'poc/public_resweep_other/'
INC=R+'poc/pell_corrugation/data/border-scan/merged.csv'
ours={r['n']:Decimal(r['L']) for r in json.load(open(R+'poc/pell_corrugation/data/border-scan/merged-literal-official.json'))['results']}
def side_of(x,y,a):
    h2=.5*(np.abs(np.cos(a))+np.abs(np.sin(a)))
    return max((x+h2).max()-(x-h2).min(),(y+h2).max()-(y-h2).min())
def key_of(xs,ys,as_):
    return hashlib.md5((';'.join(f'{a},{b},{c}' for a,b,c in zip(xs,ys,as_))).encode()).hexdigest()
from scipy.spatial import cKDTree
def max_pen(x,y,a):
    """Max SAT penetration depth over close pairs (float). >0 means overlap."""
    n=len(x);xy=np.column_stack([x,y]);u=np.column_stack([np.cos(a),np.sin(a)]);v=np.column_stack([-np.sin(a),np.cos(a)])
    best=-1.0
    for i,j in cKDTree(xy).query_pairs(1.5):
        d=xy[j]-xy[i];sep=-1e9
        for ax in (u[i],v[i],u[j],v[j]):
            r=.5*(abs(ax@u[i])+abs(ax@v[i])+abs(ax@u[j])+abs(ax@v[j]))
            sep=max(sep,abs(ax@d)-r)
        best=max(best,-sep)
    return best
