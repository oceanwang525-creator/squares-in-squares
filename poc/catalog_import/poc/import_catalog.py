# Import coordinate data only; never execute source-repository code.
import os,sys,json,gzip,glob,csv
from pathlib import Path
import yaml,mpmath as mp
mp.mp.dps=80
R=Path(__file__).resolve().parents[3]
J=Path(os.environ.get('JLEVY_CATALOG','/home/kit/jlevy-squares'))
E=Path(os.environ.get('EVAND_CATALOG','/home/kit/evand-catalog'))
sys.path.insert(0,str(R/'poc/baseline/poc/src'))
from squares.core import read_submission,score,evaluate
out=Path(os.environ.get('CATALOG_OUTPUT','/home/kit/catalog-output'));out.mkdir(exist_ok=True)
ROOT=mp.mpf(0)
def num(x):
 if isinstance(x,list):return sum(num(c)*ROOT**i for i,c in enumerate(x))
 s=str(x)
 if '/' in s:
  a,b=s.split('/');return mp.mpf(a)/mp.mpf(b)
 return mp.mpf(s)
def fmt(x):return mp.nstr(x,70)
base=read_submission(R/'poc/pell_corrugation/data/border-scan/merged.csv')
oldfile=R/'poc/pell_corrugation/data/border-scan/merged-literal-official.json'
old={int(z['n']):float(z['L']) for z in json.load(open(oldfile))['results']}
candidates={};bounds={};errors=[]
for p in sorted((J/'packing/witnesses').rglob('*.yaml*')):
 if '.schema.' in str(p):continue
 try:
  text=gzip.open(p,'rt').read() if p.suffix=='.gz' else p.read_text()
  w=yaml.load(text,Loader=yaml.CSafeLoader).get('witness',{})
  n=w.get('n',0)
  if not 1<=n<=1000:continue
  if w.get('scalar',{}).get('kind')=='algebraic-number-field':
   lo,hi=map(num,w['scalar']['isolating_interval']); coeff=list(map(num,w['scalar']['minimal_polynomial']))
   ROOT=mp.findroot(lambda t:mp.polyval(coeff,t),(lo,hi), solver='secant')
  L=float(num(w['side']));bounds[n]=min(bounds.get(n,1e9),L)
  rows=[]
  for i,s in enumerate(w['squares']):
   if 'center' in s:
    x,y=map(num,s['center'])
    if 'basis' in s:angle=mp.atan2(num(s['basis'][1]),num(s['basis'][0]))
    else:angle=num(s.get('angle',0))*mp.pi/180
   else:
    cs=[[num(z) for z in c] for c in s['corners']]
    x=sum(c[0] for c in cs)/4;y=sum(c[1] for c in cs)/4
    angle=mp.atan2(cs[1][1]-cs[0][1],cs[1][0]-cs[0][0])
   rows.append(dict(square_id=str(i),x=fmt(x),y=fmt(y),angle=fmt(angle)))
  if len(rows)!=n:raise ValueError('count')
  candidates.setdefault(n,[]).append(dict(L=L,rows=rows,source=str(p.relative_to(J))))
 except Exception as e:errors.append([str(p),str(e)])
for p in (E/'site/www/data/p').glob('*.json'):
 d=json.load(open(p));n=d['n']
 if not 1<=n<=1000:continue
 L=float(d['s'])
 bounds[n]=min(bounds.get(n,1e9),L)
 rows=[dict(square_id=str(i),x=fmt(num(s[0])),y=fmt(num(s[1])),angle=fmt(num(s[2])*mp.pi/180)) for i,s in enumerate(d['squares'])]
 if len(rows)==n:candidates.setdefault(n,[]).append(dict(L=L,rows=rows,source='evand/'+p.name))
json.dump(dict(coordinate_n=len(candidates),errors=errors,bounds=bounds),open(out/'inventory.json','w'),indent=2)
json.dump(candidates,open(out/'candidates.json','w'))
better={n:sorted([c for c in cs if c['L']<old[n]-1e-12],key=lambda c:c['L']) for n,cs in candidates.items()}
better={n:cs for n,cs in better.items() if cs}
print('INVENTORY',len(candidates),'better',len(better),'errors',errors[:5],flush=True)
# Keep all better alternatives for later repairs, not only the minimum.
json.dump(better,open(out/'better.json','w'))
print([(n,old[n]-cs[0]['L']) for n,cs in better.items()],flush=True)
