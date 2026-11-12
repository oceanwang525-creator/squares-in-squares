import sys,json,pickle,glob,csv,collections
sys.path.insert(0,'/Users/lucaszhang/squares-in-squares/poc/baseline/poc/src')
from decimal import Decimal
from squares.core import evaluate,kaggle_roundtrip
F='/Users/lucaszhang/squares-in-squares/poc/public_resweep_kernels/'
res={}
for f in glob.glob(F+'data/wins_*.pkl'): res.update(pickle.load(open(f,'rb')))
logs={}
for f in glob.glob(F+'data/verify_log_*.json'): logs.update({int(k):v for k,v in json.load(open(f)).items()})
wins={};rows_out=[]
for n,v in sorted(res.items()):
    rows=v['rows']
    _,_,L,r=evaluate(n,rows); assert r=='ok' and Decimal(L)<Decimal(v['ours'])
    rt=kaggle_roundtrip({n:rows})[n]
    _,_,L2,r2=evaluate(n,rt)
    src=v['src'].split('/')
    kern=src[1].replace('__','/',1) if src[0]=='kernels' else src[1]
    wins[n]=dict(new_L=L,our_L=v['ours'],gain=str(Decimal(v['ours'])-Decimal(L)),source_kernel=kern,source_file=v['src'],
      version='latest (anonymous output API serves only the latest version)',repaired=v['repaired'],dilation=v.get('dilation'),
      pandas_roundtrip_ok=(r2=='ok'),pandas_roundtrip_L=L2 if r2=='ok' else None,
      candidates_examined=v['examined'],candidates_prefiltered=v['prefilter'])
    base=n*(n-1)//2
    for j,row in enumerate(sorted(rows,key=lambda r:int(r['square_id']))):
        rows_out.append((base+j,n,row['square_id'],row['x'],row['y'],row['angle']))
json.dump(wins,open(F+'docs/wins.json','w'),indent=1)
with open(F+'out/wins_rows.csv','w',newline='') as f:
    w=csv.writer(f);w.writerow(['id','n','square_id','x','y','angle']);w.writerows(rows_out)
by=collections.Counter(v['source_kernel'] for v in wins.values())
print(len(wins),'wins; pandas-roundtrip ok:',sum(v['pandas_roundtrip_ok'] for v in wins.values()),'repaired:',sum(v['repaired'] for v in wins.values()))
print(by.most_common())
print('examined total',sum(v['examined'] for v in logs.values()),'prefilter total',sum(v['prefilter'] for v in logs.values()))
for n,v in sorted(wins.items(),key=lambda z:-Decimal(z[1]['gain']))[:10]: print(n,v['gain'],v['source_kernel'],v['pandas_roundtrip_ok'])
