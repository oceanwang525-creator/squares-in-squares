"""Collect every per-n candidate from every CSV, dedupe, prefilter by bbox side < ourL+1e-9. Data parsed with pandas only."""
import sys,json,glob,hashlib,os,gzip,pickle,io,collections
import numpy as np,pandas as pd
F='/Users/lucaszhang/squares-in-squares/poc/public_resweep_kernels/'
ours={int(n):float(v['L']) for n,v in json.load(open(F+'docs/our_L.json')).items()}
files=sorted(glob.glob(F+'data/kernels/**/*.csv*',recursive=True)+glob.glob(F+'data/extract/**/*.csv*',recursive=True))
files=[f for f in files if f.endswith(('.csv','.csv.gz'))]
seen=set();cand=collections.defaultdict(list);stats={};ncsv=0;nrec=0;skipped=[]
for f in files:
    src=os.path.relpath(f,F+'data')
    try:
        txt=gzip.open(f,'rt').read() if f.endswith('.gz') else open(f,encoding='utf-8',errors='replace').read()
        head=txt[:400].splitlines()[0].lower().replace(' ','') if txt else ''
        cols=head.split(',')
        if not {'x','y','angle'}<=set(cols):
            skipped.append((src,head[:80]));continue
        df=pd.read_csv(io.StringIO(txt),dtype=str,keep_default_na=False)
        df.columns=[c.strip().lower() for c in df.columns]
    except Exception as e:
        skipped.append((src,repr(e)[:80]));continue
    ncsv+=1
    if 'n' in df.columns: groups=[(int(n),g) for n,g in df.groupby(df['n'].str.strip().astype(float).astype(int),sort=False)] if len(df) else []
    else: groups=[(len(df),df)]
    st=dict(rows=len(df),groups=len(groups),new=0,pass_=0);
    for n,g in groups:
        if n not in ours or len(g)!=n: continue
        key=hashlib.md5(g[['x','y','angle']].apply(lambda c:c.str.strip()).agg(','.join,axis=1).str.cat(sep=';').encode()).hexdigest()
        if (n,key) in seen: continue
        seen.add((n,key));st['new']+=1;nrec+=1
        try:
            x=g['x'].astype(float).values;y=g['y'].astype(float).values;a=g['angle'].astype(float).values
        except Exception: continue
        h=.5*(np.abs(np.cos(a))+np.abs(np.sin(a)))
        side=max((x+h).max()-(x-h).min(),(y+h).max()-(y-h).min())
        # angle in degrees? try both
        h2=.5*(np.abs(np.cos(np.radians(a)))+np.abs(np.sin(np.radians(a))))
        side2=max((x+h2).max()-(x-h2).min(),(y+h2).max()-(y-h2).min())
        for unit,sd in ((('deg',side2),) if np.abs(a).max()>6.5 else (('rad',side),)):
            if sd<ours[n]+1e-9:
                cand[n].append(dict(bbox=float(sd),unit=unit,src=src,rows=[(r.x.strip(),r.y.strip(),r.angle.strip()) for r in g.itertuples()]));st['pass_']+=1
    stats[src]=st
json.dump(dict(csv_scanned=ncsv,unique_packings=nrec,skipped=skipped,stats=stats),open(F+'docs/scan_stats.json','w'))
pickle.dump(dict(cand),open(F+'data/cand.pkl','wb'))
print('csv',ncsv,'skipped',len(skipped),'unique packings',nrec,'cand n',len(cand),'cand total',sum(len(v) for v in cand.values()))
