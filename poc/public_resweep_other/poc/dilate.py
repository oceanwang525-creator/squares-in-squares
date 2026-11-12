"""Failed (touching/overlapping by rounding) candidates with d<-1e-14: tiny uniform dilation of centers, accept if still < ours."""
import sys,pickle,time
sys.path.insert(0,'/Users/lucaszhang/squares-in-squares/poc/public_resweep_other/poc')
from common import *
import mpmath as mp
from squares.core import evaluate
mp.mp.dps=75
pools={'data/ver1.pkl':'data/cand.pkl','data/ver2.pkl':'data/cand_ext.pkl'}
jobs=[]
for vf,cf in pools.items():
    v=pickle.load(open(F+vf,'rb'));c=pickle.load(open(F+cf,'rb'))['cand']
    for l in v['log']:
        if l['reason']!='ok' and l['d']<-1e-14 and l['n']>1:jobs.append((l['d'],l['n'],c[l['n']][l['i']]))
jobs.sort(key=lambda t:t[0]);print(len(jobs),flush=True)
wins={};log=[];t0=time.time()
for k,(d,n,cc) in enumerate(jobs):
    best=None
    xs=[mp.mpf(s) for s in cc['x']];ys=[mp.mpf(s) for s in cc['y']]
    for dil in (1e-17,1e-16,1e-15,3e-15,1e-14,3e-14,1e-13,3e-13,1e-12):
        if -d<=dil*float(ours[n])*1.0:break  # growth would exceed the candidate's advantage
        f=1+mp.mpf(dil)
        rows=[dict(square_id=str(i),x=mp.nstr(xs[i]*f,65),y=mp.nstr(ys[i]*f,65),angle=cc['a'][i]) for i in range(n)]
        try:_,_,s,r=evaluate(n,rows)
        except Exception as e:s,r=None,repr(e)
        log.append(dict(n=n,src=cc['src'],d=d,dil=dil,reason=r,L=s))
        if r=='ok':
            if Decimal(s)<ours[n]:best=dict(L=s,rows=rows,src=cc['src']+f'|dilate{dil}')
            break
    if best and (n not in wins or Decimal(best['L'])<Decimal(wins[n]['L'])):wins[n]=best
    if k%25==0:print(k,len(wins),round(time.time()-t0),flush=True);pickle.dump(dict(log=log,wins=wins),open(F+'data/ver5.pkl','wb'))
pickle.dump(dict(log=log,wins=wins),open(F+'data/ver5.pkl','wb'));print('DONE',len(wins))
