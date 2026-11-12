import os,sys,json,csv,hashlib,subprocess,time,multiprocessing as mp,gc
from pathlib import Path
from decimal import Decimal
sys.path.insert(0,str(Path(r'E:\squares-recovery-nov12')/'pandas31'))
import pandas as pd
assert pd.__version__=='3.1.0'
O=Path(r'E:\squares-recovery-nov12');R=O/'post-merge-main';A=O/'audit-pandas31-postmerge';A.mkdir(exist_ok=True)
sys.path.insert(0,str(R/'poc/baseline/poc/src'))
from squares.core import read_submission,score,evaluate
from squares import _official_metric as metric
metric.KAGGLE_CORES=20
C=str(O/'certifier.exe')
def save(p,v):
 p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2));os.replace(tmp,p)
def write(p,ps):
 with open(p,'w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['id','n','square_id','x','y','angle']);w.writeheader();idx=0
  for n,rows in sorted(ps.items()):
   for j,r in enumerate(sorted(rows,key=lambda r:int(r['square_id']))):
    w.writerow(dict(id=idx,n=n,square_id=j,**{k:r[k] for k in ['x','y','angle']}));idx+=1
def cpp(p,d,rt=False):
 if not d.exists():
  subprocess.run([C,str(p),'--threads','20','--details',str(d)]+(['--roundtrip'] if rt else []),check=True)
 return json.loads(d.read_text())
def off(p,d):
 if not d.exists():save(d,score(read_submission(p)))
 q=json.loads(d.read_text());print('OFFICIAL',d.name,q['score'],q['accepted'],q['rejected'],flush=True);return q
def mapping(q):return {z['n']:z for z in q['results']}
def fingerprint(rs):return hashlib.sha256(json.dumps([{k:r[k] for k in ['square_id','x','y','angle']} for r in rs],sort_keys=True).encode()).hexdigest()
def eval_job(job):
 key,n,rows=job
 return key,evaluate(n,rows)
def main():
 inventory=[];paths=[O/p for p in json.loads((O/'manifest11.json').read_text())]
 assert len(paths)==11
 # Fresh baseline replaces obsolete main snapshot while retaining old hashed input.
 fresh=O/'post-merge-main/poc/pell_corrugation/data/border-scan/merged.csv'
 paths=[p for p in paths if p!=O/'sources/main-e2e9811.csv']+[fresh]
 sources=[];seen=set()
 for index,p in enumerate(paths):
  sha=hashlib.sha256(p.read_bytes()).hexdigest()
  with p.open() as f:fields=next(csv.reader(f),[])
  entry=dict(path=str(p),bytes=p.stat().st_size,sha256=sha,fields=fields)
  if not {'n','square_id','x','y','angle'}.issubset(fields):
   entry['status']='auxiliary_no_coordinates';inventory.append(entry);continue
  if sha in seen:entry['status']='duplicate';inventory.append(entry);continue
  seen.add(sha);label=f's{index:02d}';work=A/label;work.mkdir(exist_ok=True)
  # Normalize missing id without numeric conversion, exact strings preserved.
  norm=work/'literal.csv';ps={}
  with p.open() as f:
   for row in csv.DictReader(f):ps.setdefault(int(row['n']),[]).append({k:row[k] for k in ['square_id','x','y','angle']})
  write(norm,ps);pandas=work/'pandas.csv'
  if not pandas.exists():pd.read_csv(norm).to_csv(pandas,index=False)
  qs=[cpp(norm,work/'literal-cpp.json'),cpp(norm,work/'roundtrip-cpp.json',True),cpp(pandas,work/'pandas-cpp.json')]
  entry.update(status='screened',label=label,groups=len(ps),rows=sum(map(len,ps.values())),cpp=[{k:q[k] for k in ['score','accepted','missing','rejected']} for q in qs])
  inventory.append(entry);save(A/'inventory.json',inventory)
  groups=work/'groups';groups.mkdir(exist_ok=True)
  for n,rows in ps.items():
   gp=groups/f'{n}-literal.json'
   if not gp.exists():save(gp,rows)
  del ps;gc.collect()
  pps=read_submission(pandas)
  for n,rows in pps.items():
   gp=groups/f'{n}-pandas.json'
   if not gp.exists():save(gp,rows)
  del pps;gc.collect()
  sources.append(dict(label=label,path=norm,pandas=pandas,groups=groups,maps=list(map(mapping,qs))))
  print('SCREENED',label,p,entry['rows'],entry['cpp'],flush=True)
 assert len(sources)==11,len(sources)
 base=sources[-1]
 bl=off(base['path'],A/'baseline-literal-official.json');bp=off(base['pandas'],A/'baseline-pandas-official.json')
 bm=[mapping(bl),mapping(bp)]
 assert bl['accepted']==1000
 candidates={};jobs={};cache={}
 for n in range(1,1001):
  seen_pairs=set()
  for source in sources:
   if not all(n in m and m[n]['reason']=='ok' for m in source['maps']):continue
   ls=[Decimal(m[n]['L']) for m in source['maps']]
   if ls[0]>Decimal(bm[0][n]['L']):continue
   if bm[1][n]['reason']=='ok' and ls[2]>Decimal(bm[1][n]['L']):continue
   rs=json.loads((source['groups']/f'{n}-literal.json').read_text());prs=json.loads((source['groups']/f'{n}-pandas.json').read_text())
   pair=(fingerprint(rs),fingerprint(prs))
   if pair in seen_pairs:continue
   seen_pairs.add(pair)
   candidates.setdefault(n,[]).append((ls[0],ls[2],source['label'],rs,prs))
   for key,group in zip(pair,[rs,prs]):
    if key not in cache:jobs[key]=(key,n,group)
 # Seed official cache with independent baseline receipts.
 for n in range(1,1001):
  for mode,rep in [('literal',bm[0][n]),('pandas',bm[1][n])]:
   group=json.loads((base['groups']/f'{n}-{mode}.json').read_text());key=fingerprint(group)
   cache[key]=(n,rep['score'],rep['L'],rep['reason']);jobs.pop(key,None)
 print('OFFICIAL unique candidate groups',len(jobs),flush=True)
 with mp.Pool(20) as pool:
  for count,(key,res) in enumerate(pool.imap_unordered(eval_job,jobs.values()),1):
   cache[key]=res
   if count%100==0:print('CANDIDATES',count,'/',len(jobs),flush=True);save(A/'candidate-cache.json',cache)
 save(A/'candidate-cache.json',cache)
 final={};chosen={}
 for n in range(1,1001):
  good=[]
  for _,_,label,rs,prs in candidates.get(n,[]):
   z=cache[fingerprint(rs)];p=cache[fingerprint(prs)]
   if z[3]!='ok' or p[3]!='ok':continue
   if Decimal(z[2])>Decimal(bm[0][n]['L']):continue
   if bm[1][n]['reason']=='ok' and Decimal(p[2])>Decimal(bm[1][n]['L']):continue
   good.append((Decimal(z[2]),Decimal(p[2]),label,rs,z,p))
  assert good,('no valid nonregressive group',n)
  best=min(good,key=lambda a:(a[0],a[1],a[2]))
  final[n]=best[3];chosen[n]=dict(source=best[2],literal_L=best[4][2],pandas_L=best[5][2],baseline_literal_L=bm[0][n]['L'],baseline_pandas_L=bm[1][n]['L'])
 write(A/'final.csv',final);save(A/'chosen.json',chosen)
 pd.read_csv(A/'final.csv').to_csv(A/'final-pandas.csv',index=False)
 reps=[cpp(A/'final.csv',A/'final-literal-cpp.json'),cpp(A/'final.csv',A/'final-roundtrip-cpp.json',True),cpp(A/'final-pandas.csv',A/'final-pandas-cpp.json'),off(A/'final.csv',A/'final-literal-official.json'),off(A/'final-pandas.csv',A/'final-pandas-official.json')]
 for rep in reps:assert rep['accepted']==1000 and rep['missing']==0 and rep['rejected']==0
 for n in range(1,1001):
  assert Decimal(mapping(reps[3])[n]['L'])<=Decimal(bm[0][n]['L'])
  if bm[1][n]['reason']=='ok':assert Decimal(mapping(reps[4])[n]['L'])<=Decimal(bm[1][n]['L'])
 receipt=dict(score_literal=reps[3]['score'],score_pandas=reps[4]['score'],sha256=hashlib.sha256((A/'final.csv').read_bytes()).hexdigest(),bytes=(A/'final.csv').stat().st_size,source_count=len(sources),baseline_pandas_rejects=bp['rejected'],wins=sum(Decimal(v['literal_L'])<Decimal(v['baseline_literal_L']) for v in chosen.values()),all_1000_zero_rejects=True)
 save(A/'COMPLETE.json',receipt);print('COMPLETE',receipt,flush=True)
if __name__=='__main__':main()
