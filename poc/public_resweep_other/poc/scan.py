"""Harvest every packing (CSV/JSON) under data/src, data/sigma, data/gallery_n; keep bbox < L+1e-9."""
import sys,pickle,collections
sys.path.insert(0,'/Users/lucaszhang/squares-in-squares/poc/public_resweep_other/poc')
from common import *
TOL=1e-9
seen=set();cand=collections.defaultdict(list);stats=collections.defaultdict(lambda:dict(examined=0,unique=0,pass_=0,ns=set()))
def offer(n,src,xs,ys,as_,unit='rad',fl=None,lazy=None):
    # xs,ys,as_ : lists of strings or floats
    if n not in ours or (fl is None and len(xs)!=n):return
    st=stats[src];st['examined']+=1
    xf,yf,af=fl if fl else (np.array([float(v) for v in xs]),np.array([float(v) for v in ys]),np.array([float(v) for v in as_]))
    if not np.isfinite(xf).all() or not np.isfinite(af).all():return
    if unit=='auto':
        if np.abs(af).max()>6.5:unit='deg'
        else:unit='rad'
    if unit=='deg':af=np.deg2rad(af);as_=[repr(float(v)) for v in af]
    k=(n,key_of(xf.tolist(),yf.tolist(),af.tolist()) if fl else key_of(xs,ys,as_))
    if k in seen:return
    seen.add(k);st['unique']+=1
    s=side_of(xf,yf,af)
    if s<float(ours[n])+TOL:
        if lazy:xs,ys,as_=lazy()
        cand[n].append(dict(src=src,bbox=float(s),x=[str(v) for v in xs],y=[str(v) for v in ys],a=[str(v) for v in as_]));st['pass_']+=1;st['ns'].add(n)
def do_csv(f,src):
    with open(f,errors='replace') as fh:h=fh.readline().strip().split(',')
    if not {'x','y','angle'}<=set(h):return
    df=pd.read_csv(f,dtype=str)
    if 'n' in h:
        for n,g in df.groupby(df['n'].astype(int)):
            offer(int(n),src,g['x'].tolist(),g['y'].tolist(),g['angle'].tolist())
    else:
        m=re.findall(r'(\d+)',os.path.basename(f))
        if not m:return
        n=int(m[-1]) if len(m)==1 else int(m[0])
        if len(df)!=n and len(m)>1:
            for c in m:
                if len(df)==int(c):n=int(c)
        offer(n,src,df['x'].tolist(),df['y'].tolist(),df['angle'].tolist())
def is_tri(l,n):
    return isinstance(l,list) and len(l)==n and n>0 and all(isinstance(r,(list,tuple)) and len(r)>=3 and all(isinstance(v,(int,float)) for v in r[:3]) for r in l)
def is_dicts(l,n):
    return isinstance(l,list) and len(l)==n and n>0 and all(isinstance(r,dict) and {'x','y'}<=set(r) for r in l)
def walk(o,src,nhint=None,depth=0):
    if depth>6:return
    if isinstance(o,dict):
        n=None
        for k in ('n','N','puzzle','n_squares'):
            if isinstance(o.get(k),int):n=o[k]
        if n is None:n=nhint
        for k,v in o.items():
            if n is not None and is_tri(v,n):
                offer(n,src,[r[0] for r in v],[r[1] for r in v],[r[2] for r in v],'auto')
            elif n is not None and is_dicts(v,n) and all(('angle' in r or 'theta' in r) for r in v):
                offer(n,src,[r['x'] for r in v],[r['y'] for r in v],[r.get('angle',r.get('theta')) for r in v],'auto')
            else:
                nh=int(k) if isinstance(k,str) and k.isdigit() else None
                walk(v,src,nh if nh else n if not isinstance(v,(list,dict)) else (nh or None),depth+1)
    elif isinstance(o,list):
        for v in o[:5000]:walk(v,src,nhint,depth+1)
def do_json(f,src):
    if os.path.getsize(f)>400e6:print('skip big',f);return
    try:o=json.load(open(f))
    except Exception as e:print('bad json',f,str(e)[:60]);return
    walk(o,src)
def main():
    roots=[(F+'data/src','src'),(F+'data/sigma','sigma'),(F+'data/gallery_n','gallery'),(F+'data/ext','ext')]
    for root,tag in roots:
        for f in sorted(glob.glob(root+'/**/*',recursive=True)):
            if not os.path.isfile(f):continue
            rel=os.path.relpath(f,F+'data')
            if f.endswith('.csv'):
                try:do_csv(f,rel)
                except Exception as e:print('csv err',rel,str(e)[:80])
            elif f.endswith('.json') and not os.path.basename(f).startswith('gate_'):
                do_json(f,rel)
    # compress stats by top-level source dir
    agg=collections.defaultdict(lambda:dict(examined=0,unique=0,pass_=0,ns=set()))
    for s,d in stats.items():
        t='/'.join(s.split('/')[:2]) if s.startswith('src') else '/'.join(s.split('/')[:2])
        a=agg[t];a['examined']+=d['examined'];a['unique']+=d['unique'];a['pass_']+=d['pass_'];a['ns']|=d['ns']
    for t,a in sorted(agg.items()):print(t,a['examined'],a['unique'],a['pass_'],len(a['ns']))
    pickle.dump(dict(cand=dict(cand),agg={t:{**a,'ns':sorted(a['ns'])} for t,a in agg.items()}),open(F+'data/cand.pkl','wb'))
    print('cand n',len(cand),'cands',sum(len(v) for v in cand.values()))
if __name__=='__main__':main()
