"""Re-run the fixed-angle LP repair of the rounded gallery with leaner slack/dilation than the earlier import."""
import sys,pickle,time
sys.path.insert(0,'/Users/lucaszhang/squares-in-squares/poc/public_resweep_other/poc')
from repair import *
from squares.core import evaluate
prov=json.load(open(R+'poc/pell_corrugation/data/border-scan/gallery-repair-provenance.json'))
ns=sorted(int(k) for k in prov if k.isdigit()) if any(k.isdigit() for k in prov) else []
if not ns:
    ns=sorted(int(k) for k in json.load(open(R+'poc/pell_corrugation/data/border-scan/gallery-repair-attempts.json')) if str(k).isdigit())
idx=json.load(open(F+'data/gallery/index.json'))
if len(sys.argv)>1 and sys.argv[1]=='all':
    ns=[r['n'] for r in idx if r['L']<float(ours[r['n']])+2e-6 and r['n']>1 and r['tilt']>0]
print(len(ns),flush=True)
wins={};log=[];t0=time.time()
for n in ns:
    g=json.load(open(F+f'data/gallery_n/g{n}.json'))
    q=np.array(g['s'],float);xy=q[:,:2];ang=np.deg2rad(q[:,2])
    mp.mp.dps=75
    astr=[mp.nstr(mp.mpf(str(v[2]))*mp.pi/180,65) for v in g['s']]
    best=None
    for trust in (0.03,0.1):
        for slack in (1e-10,1e-11,1e-12):
            r=lp(xy,ang,slack,trust=trust,tl=40)
            if r is None:continue
            cen,L=r
            if L>float(ours[n])+1e-7:continue
            for dil in (1e-15,1e-14,1e-13,1e-12,1e-11):
                rows=rows_from(cen,astr,dil)
                try:_,_,s,reason=evaluate(n,rows)
                except Exception as e:s,reason=None,repr(e)
                log.append(dict(n=n,trust=trust,slack=slack,dil=dil,reason=reason,L=s))
                if reason=='ok':
                    if Decimal(s)<ours[n] and (best is None or Decimal(s)<Decimal(best['L'])):best=dict(L=s,rows=rows,src=f'gallery_lean(trust={trust},slack={slack},dil={dil})')
                    break
    if best:wins[n]=best
    print(n,'win' if best else 'no',best and float(ours[n]-Decimal(best['L'])),round(time.time()-t0),flush=True)
    pickle.dump(dict(log=log,wins=wins),open(F+'data/ver4.pkl','wb'))
print('DONE',len(wins))
