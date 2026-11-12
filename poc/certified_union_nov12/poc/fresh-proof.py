import os,sys,json,csv,hashlib,subprocess,uuid,time
from pathlib import Path
from decimal import Decimal
sys.path.insert(0,str(Path(r'E:\squares-recovery-nov12')/'pandas31'))
import pandas as pd
assert pd.__version__=='3.1.0'
O=Path(r'E:\squares-recovery-nov12');A=O/'audit-pandas31-postmerge';R=O/'post-merge-main';C=O/'certifier.exe'
sys.path.insert(0,str(R/'poc/baseline/poc/src'))
from squares.core import score,read_submission
from squares import _official_metric as metric
metric.KAGGLE_CORES=20
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,indent=2))
def main():
 assert not (O/'SUBMIT-ATTEMPT.json').exists()
 print('CONTINUATION_WAIT',os.getpid(),flush=True)
 # Windows handle wait, not SSH lifetime dependent, exact previously verified runner.
 import ctypes
 h=0
 if h:ctypes.windll.kernel32.WaitForSingleObject(h,0xffffffff);ctypes.windll.kernel32.CloseHandle(h)
 assert (A/'COMPLETE.json').exists(),'Runner exited without successful union'
 rec=json.loads((A/'COMPLETE.json').read_text());f=A/'final.csv';assert sha(f)==rec['sha256']
 raw=O/'opencode-verified-7de2a843.csv';assert sha(raw)=='7de2a8432ca6380678127734fb611cc5c4f090e151d9e804475b2c7ffb4eef05'
 rows=list(csv.DictReader(f.open()));wg=list(csv.DictReader((O/'official-962.csv').open()))
 assert len(wg)==269
 rows=[r for r in rows if int(r['n'])!=269]+wg
 rows.sort(key=lambda r:(int(r['n']),int(r['square_id'])))
 assert len(rows)==500500
 shutil_path=A/'pre-962-final.csv';shutil_path.write_bytes(f.read_bytes())
 with f.open('w',newline='') as out:
  w=csv.DictWriter(out,fieldnames=['id','n','square_id','x','y','angle']);w.writeheader()
  for i,r in enumerate(rows):r['id']=str(i);w.writerow(r)
 del rows
 print('WITNESS269_INTEGRATED',sha(f),flush=True)
 inventory=json.loads((A/'inventory.json').read_text())
 expected={r['path']:r['sha256'] for r in inventory if r['status']=='screened'}
 assert len(expected)==11
 save(A/'expected-source-manifest.json',expected)
 proof_id=str(uuid.uuid4());D=A/('fresh-proof-'+proof_id);D.mkdir()
 (D/'literal.csv').write_bytes(f.read_bytes());pd.read_csv(D/'literal.csv').to_csv(D/'pandas.csv',index=False)
 bindings={}
 for mode,inp,extra in [('literal',D/'literal.csv',[]),('roundtrip',D/'literal.csv',['--roundtrip']),('pandas',D/'pandas.csv',[])]:
  out=D/(mode+'-cpp.json');subprocess.run([str(C),str(inp),'--threads','20','--details',str(out)]+extra,check=True)
  q=json.loads(out.read_text());assert(q['accepted'],q['missing'],q['rejected'])==(1000,0,0)
  bindings[out.name]={'input_sha256':sha(inp),'report_sha256':sha(out),'certifier_sha256':sha(C)}
 final=[]
 for mode in ['literal','pandas']:
  inp=D/(mode+'.csv');q=score(read_submission(inp));save(D/(mode+'-official.json'),q)
  assert(q['accepted'],q['missing'],q['rejected'])==(1000,0,0)
  final.append(q);print('FRESH_OFFICIAL',mode,q['score'],flush=True)
 # Independent fresh baseline actual bytes, not cached.
 baseline=R/'poc/pell_corrugation/data/border-scan/merged.csv'
 pd.read_csv(baseline).to_csv(D/'baseline-pandas.csv',index=False)
 for i,inp in enumerate([baseline,D/'baseline-pandas.csv']):
  q=score(read_submission(inp));save(D/('baseline-'+str(i)+'.json'),q)
  bm={r['n']:r for r in q['results']};fm={r['n']:r for r in final[i]['results']}
  for n in range(1,1001):
   if bm[n]['reason']=='ok':assert Decimal(fm[n]['L'])<=Decimal(bm[n]['L']),(i,n)
  print('BASELINE_NONREGRESSION',i,flush=True)
 assert sha(f)==sha(D/'literal.csv')
 for p,hsh in expected.items():assert sha(p)==hsh
 proof={'run_id':proof_id,'directory':str(D),'final_sha256':sha(f),'bindings':bindings,'source_manifest':expected,'official_sha256':sha(R/'poc/baseline/poc/src/squares/_official_metric.py'),'score_literal':final[0]['score'],'score_pandas':final[1]['score'],'all_1000_zero_rejects':True,'witness269_provenance':json.loads((O/'official-962-local-provenance.json').read_text())}
 save(A/'FRESH-BOUND-PROOF.json',proof);print('FRESH_BOUND_PROOF_COMPLETE',proof,flush=True)
 print('READY_FOR_GUARDED_MERGE_NO_SUBMIT_PERFORMED',flush=True)
if __name__=='__main__':main()
