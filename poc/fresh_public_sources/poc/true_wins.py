import sys,pickle,json
sys.path.insert(0,'/Users/lucaszhang/squares-in-squares/poc/baseline/poc/src')
from decimal import Decimal
from squares.core import read_submission,evaluate
R='/Users/lucaszhang/squares-in-squares/';F=R+'poc/fresh_public_sources/'
m=read_submission(R+'poc/pell_corrugation/data/border-scan/merged.csv')
w=pickle.load(open(F+'data/wins.pkl','rb'))
tw={};allr={}
for n,v in sorted(w.items()):
    _,_,s,r=evaluate(n,m[n]); assert r=='ok'
    allr[n]=dict(merged_L=s,cand_L=v['L'],src=v['src'])
    if Decimal(v['L'])<Decimal(s): tw[n]=v; tw[n]['ours']=s
json.dump(allr,open(F+'data/vs_current_merged.json','w'),indent=1)
pickle.dump(tw,open(F+'data/true_wins.pkl','wb'))
print(len(tw),sorted(tw))
