"""Exact-evaluate candidates (bbox - L < thresh), ascending; log + keep best certified strict win per n."""
import sys,pickle,time,json
sys.path.insert(0,'/Users/lucaszhang/squares-in-squares/poc/public_resweep_other/poc')
from common import *
from squares.core import evaluate
src=sys.argv[1]; thresh=float(sys.argv[2]); lo=float(sys.argv[3]); out=sys.argv[4]
c=pickle.load(open(F+src,'rb'))['cand']
items=sorted(((cc['bbox']-float(ours[n]),n,i) for n,l in c.items() for i,cc in enumerate(l)),key=lambda t:t[0])
items=[t for t in items if lo<=t[0]<thresh]
print(len(items),flush=True)
log=[];wins={};t0=time.time()
for k,(d,n,i) in enumerate(items):
    cc=c[n][i]
    rows=[dict(square_id=str(j),x=cc['x'][j],y=cc['y'][j],angle=cc['a'][j]) for j in range(n)]
    try:_,_,s,r=evaluate(n,rows)
    except Exception as e:s,r=None,repr(e)
    better=s is not None and r=='ok' and Decimal(s)<ours[n]
    log.append(dict(n=n,i=i,src=cc['src'],d=d,reason=r,L=s,better=bool(better)))
    if better and (n not in wins or Decimal(s)<Decimal(wins[n]['L'])):
        wins[n]=dict(L=s,src=cc['src'],rows=rows)
    if k%100==0:
        print(k,len(wins),round(time.time()-t0),flush=True)
        pickle.dump(dict(log=log,wins=wins),open(F+out,'wb'))
pickle.dump(dict(log=log,wins=wins),open(F+out,'wb'));print('DONE',len(wins),flush=True)
