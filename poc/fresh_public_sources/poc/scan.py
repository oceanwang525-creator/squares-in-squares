"""Prefilter public submission CSVs against the incumbent by bbox side, then exact-check survivors."""
import sys,json,glob,hashlib,os
sys.path.insert(0,'/Users/lucaszhang/squares-in-squares/poc/baseline/poc/src')
import numpy as np,pandas as pd
from decimal import Decimal
R='/Users/lucaszhang/squares-in-squares/'
F=R+'poc/fresh_public_sources/'
inc=json.load(open(R+'poc/pell_corrugation/data/border-scan/merged-literal-official.json'))
ours={r['n']:Decimal(r['L']) for r in inc['results']}
files=sorted(glob.glob(F+'data/src/**/*.csv',recursive=True)+glob.glob(F+'data/kernels/**/*.csv',recursive=True))
seen={};cand={}  # n -> list of (bbox,src,rows-hash)
stats={}
for f in files:
    with open(f) as fh: h=fh.readline().strip()
    if not h.startswith('id,n,square_id,x,y,angle'):continue
    df=pd.read_csv(f,dtype=str)
    src=os.path.relpath(f,F+'data')
    ns=0;nc=0
    for n,g in df.groupby(df['n'].astype(int)):
        if n not in ours or len(g)!=n:
            stats.setdefault(src,{}).setdefault('bad_count',[]).append(int(n)) if len(g)!=n else None
            continue
        key=hashlib.md5((g['x']+','+g['y']+','+g['angle']).str.cat(sep=';').encode()).hexdigest()
        if (n,key) in seen: continue
        seen[(n,key)]=src;ns+=1
        x=g['x'].astype(float).values;y=g['y'].astype(float).values;a=g['angle'].astype(float).values
        h2=.5*(np.abs(np.cos(a))+np.abs(np.sin(a)))
        side=max((x+h2).max()-(x-h2).min(),(y+h2).max()-(y-h2).min())
        if side<float(ours[n])-1e-11:
            cand.setdefault(n,[]).append((float(side),src,g.copy()))
            nc+=1
    stats.setdefault(src,{}).update(new_unique=ns,prefilter_pass=nc)
json.dump(stats,open(F+'data/scan_stats.json','w'),indent=1)
print(json.dumps({s:{k:v for k,v in d.items() if k!='bad_count'} for s,d in stats.items()},indent=0))
print('cand n:',sorted(cand))
import pickle;pickle.dump(cand,open(F+'data/cand.pkl','wb'))
