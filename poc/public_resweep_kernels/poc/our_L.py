import sys,json,time
sys.path.insert(0,'/Users/lucaszhang/squares-in-squares/poc/baseline/poc/src')
from squares.core import read_submission,evaluate
m=read_submission('/Users/lucaszhang/squares-in-squares/poc/pell_corrugation/data/border-scan/merged.csv')
out={};t=time.time()
for n in sorted(m):
    _,_,s,r=evaluate(n,m[n]);out[n]=dict(L=s,reason=r)
    if n%100==0:print(n,time.time()-t,flush=True)
json.dump(out,open('/Users/lucaszhang/squares-in-squares/poc/public_resweep_kernels/docs/our_L.json','w'))
print(len(out),sum(v['reason']!='ok' for v in out.values()))
