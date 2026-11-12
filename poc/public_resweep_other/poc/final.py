"""Assemble best strict win per n from verify passes; re-check literal + pandas round trip; write out/ and docs/."""
import sys,pickle,io,collections,glob
sys.path.insert(0,'/Users/lucaszhang/squares-in-squares/poc/public_resweep_other/poc')
from common import *
from squares.core import evaluate,read_submission
from squares import _official_metric as metric
import csv
vers=[f for f in sorted(glob.glob(F+'data/ver[0-9].pkl'))]
best={};nev=collections.Counter();nfail=collections.Counter();allwins=collections.defaultdict(list)
for f in vers:
    v=pickle.load(open(f,'rb'))
    for l in v['log']:
        nev[l['n']]+=1
        if l['reason']!='ok':nfail[l['n']]+=1
    for n,w in v['wins'].items():
        allwins[n].append((w['L'],w['src'],w['rows']))
        if n not in best or Decimal(w['L'])<Decimal(best[n]['L']):best[n]=w
pool=collections.Counter()
for f in ['cand','cand_ext','cand_npz']:
    for n,l in pickle.load(open(F+f'data/{f}.pkl','rb'))['cand'].items():pool[n]+=len(l)
import pandas as pd
m=read_submission(INC)
def roundtrip(n,rows):
    rows=sorted(rows,key=lambda r:int(r['square_id']))
    t=io.StringIO();w=csv.writer(t,lineterminator='\n');w.writerow(['id','n','square_id','x','y','angle'])
    for r in rows:w.writerow([n*(n-1)//2+int(r['square_id']),n,r['square_id'],r['x'],r['y'],r['angle']])
    txt=t.getvalue();pt=pd.read_csv(io.StringIO(txt)).to_csv(index=False)
    return txt,metric._parse_submission_csv(txt)[n],metric._parse_submission_csv(pt)[n]
chosen={};docs={};dropped={}
for n in sorted(allwins):
    _,_,Lm,rm=evaluate(n,m[n])
    for L,src,rows in sorted(allwins[n],key=lambda t:Decimal(t[0]))[:8]:
        txt,lit,pdr=roundtrip(n,rows)
        _,_,Ll,rl=evaluate(n,lit);_,_,Lp,rp=evaluate(n,pdr)
        if rl==rp=='ok' and Decimal(Ll)<Decimal(Lm) and Decimal(Lp)<Decimal(Lm):
            chosen[n]=(txt,rows)
            docs[n]=dict(new_L=Ll,new_L_pandas=Lp,our_L=Lm,gain=float(Decimal(Lm)-Decimal(Ll)),gain_pandas=float(Decimal(Lm)-Decimal(Lp)),source=src,n_candidates_prefiltered=pool[n],n_exact_evaluated=nev[n],n_failed_exact=nfail[n],n_strict_better_candidates=len(allwins[n]))
            break
    else:dropped[n]=dict(our_L=Lm,candidates=[(t[0],t[1]) for t in allwins[n]][:8])
ns=sorted(chosen)
text=io.StringIO();w=csv.writer(text,lineterminator='\n');w.writerow(['id','n','square_id','x','y','angle'])
for n in ns:
    for r in sorted(chosen[n][1],key=lambda r:int(r['square_id'])):w.writerow([n*(n-1)//2+int(r['square_id']),n,r['square_id'],r['x'],r['y'],r['angle']])
open(F+'out/wins_rows.csv','w').write(text.getvalue())
json.dump(dropped,open(F+'docs/dropped_pandas_unstable.json','w'),indent=1)
bad=sorted(dropped)
json.dump(docs,open(F+'docs/wins.json','w'),indent=1)
print(len(ns),'wins; bad:',bad)
g=sorted(docs.items(),key=lambda t:-t[1]['gain'])
print([(n,'%.3e'%d['gain']) for n,d in g[:10]])
by=collections.Counter(d['source'].split('/')[1] if d['source'].startswith(('src/','ext/')) else d['source'].split('/')[0] for d in docs.values());print(by)
print('sum gain/sqrt(n)',sum(d['gain']/n**.5 for n,d in docs.items()))
