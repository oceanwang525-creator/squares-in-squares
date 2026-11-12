"""float64-canonical rows (repr strings, so literal == pandas round trip) with minimal uniform center dilation."""
import sys,pickle,time,collections
sys.path.insert(0,'/Users/lucaszhang/squares-in-squares/poc/public_resweep_other/poc')
from common import *
from squares.core import evaluate
bad=set(int(n) for n,v in json.load(open(F+'docs/wins.json')).items() if not v['literal_pandas_ok'])
per=collections.defaultdict(list)
for cf in ['cand','cand_ext','cand_npz']:
    for n,l in pickle.load(open(F+f'data/{cf}.pkl','rb'))['cand'].items():
        if n in bad:
            for cc in l:
                d=cc['bbox']-float(ours[n])
                if d<-1e-14:per[n].append((d,cc))
print(len(bad),sum(len(v) for v in per.values()),flush=True)
LAD=(0,1e-16,3e-16,1e-15,3e-15,1e-14,3e-14,1e-13,3e-13,1e-12,3e-12,1e-11,1e-10)
wins={};log=[];t0=time.time()
for k,n in enumerate(sorted(per)):
    got=0
    for d,cc in sorted(per[n],key=lambda t:t[0]):
        if got>=4:break
        x=np.array([float(s) for s in cc['x']]);y=np.array([float(s) for s in cc['y']]);a=[repr(float(s)) for s in cc['a']]
        for dil in LAD:
            if dil>0 and -d<=dil*float(ours[n]):break
            xs=x*(1+dil);ys=y*(1+dil)
            rows=[dict(square_id=str(i),x=repr(float(xs[i])),y=repr(float(ys[i])),angle=a[i]) for i in range(n)]
            try:_,_,s,r=evaluate(n,rows)
            except Exception as e:s,r=None,repr(e)
            log.append(dict(n=n,src=cc['src'],d=d,dil=dil,reason=r,L=s))
            if r=='ok':
                got+=1
                if Decimal(s)<ours[n] and (n not in wins or Decimal(s)<Decimal(wins[n]['L'])):wins[n]=dict(L=s,rows=rows,src=cc['src']+f'|f64canon dil={dil}')
                break
    if k%10==0:print(k,n,len(wins),round(time.time()-t0),flush=True);pickle.dump(dict(log=log,wins=wins),open(F+'data/ver6.pkl','wb'))
pickle.dump(dict(log=log,wins=wins),open(F+'data/ver6.pkl','wb'));print('DONE',len(wins))
