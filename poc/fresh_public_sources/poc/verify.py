import sys,json,pickle,time
sys.path.insert(0,'/Users/lucaszhang/squares-in-squares/poc/baseline/poc/src')
from decimal import Decimal
from squares.core import evaluate
R='/Users/lucaszhang/squares-in-squares/';F=R+'poc/fresh_public_sources/'
ours={r['n']:Decimal(r['L']) for r in json.load(open(R+'poc/pell_corrugation/data/border-scan/merged-literal-official.json'))['results']}
cand=pickle.load(open(F+'data/cand.pkl','rb'))
wins={};log={};t0=time.time()
for n in sorted(cand):
    tried=0
    for bb,src,g in sorted(cand[n],key=lambda c:c[0]):
        if tried>=8:break
        tried+=1
        rows=[dict(square_id=str(r.square_id),x=r.x,y=r.y,angle=r.angle) for r in g.itertuples()]
        try:_,_,side,reason=evaluate(n,rows)
        except Exception as e:side,reason=None,repr(e)
        log.setdefault(n,[]).append(dict(src=src,bbox=bb,reason=reason,L=side))
        if reason=='ok' and Decimal(side)<ours[n]:
            wins[n]=dict(src=src,L=side,ours=str(ours[n]),rows=rows);break
    print(n,len(cand[n]),'win' if n in wins else 'no',round(time.time()-t0),flush=True)
    json.dump(log,open(F+'data/verify_log.json','w'))
    pickle.dump(wins,open(F+'data/wins.pkl','wb'))
