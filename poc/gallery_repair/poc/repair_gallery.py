"""Fixed-angle sparse SAT LP repair. Floating search proposals, exact-decimal certification gate."""
import os
os.environ['OMP_NUM_THREADS']='1'
import sys,json,csv,subprocess,io,multiprocessing
multiprocessing.set_start_method('fork',force=True)
from pathlib import Path
from decimal import Decimal
import numpy as np,pandas as pd,mpmath as mp
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.optimize import linprog
REPO=Path(os.environ.get('SQUARES_REPO',str(Path(__file__).resolve().parents[3])))
sys.path.insert(0,str(REPO/'poc/baseline/poc/src'))
from squares.core import read_submission,score
R=REPO
INPUT=Path(os.environ.get('GALLERY_INPUT','/home/kit/gallery-input'))
AUDIT=Path(os.environ.get('GALLERY_AUDIT','/home/kit/gallery-audit'))
CERTIFIER=os.environ.get('SQUARES_CERTIFIER','/home/kit/certifier')
O=Path(os.environ.get('GALLERY_REPAIR_OUTPUT','/home/kit/gallery-repair'));O.mkdir(exist_ok=True)
def write(p,ps):
 with open(p,'w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['id','n','square_id','x','y','angle']);w.writeheader();idx=0
  for n,rs in sorted(ps.items()):
   for j,r in enumerate(rs):w.writerow(dict(id=idx,n=n,square_id=j,**{k:r[k] for k in ['x','y','angle']}));idx+=1
def save(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
def repair(n):
 d=json.load(open(INPUT/f'gallery/{n}.json'))
 q=np.array(d['s'],float);xy=q[:,:2];angles=q[:,2]*np.pi/180
 u=np.column_stack([np.cos(angles),np.sin(angles)]);v=np.column_stack([-np.sin(angles),np.cos(angles)])
 pairs=list(cKDTree(xy).query_pairs(1.6))
 rr=[];cc=[];vals=[];rhs=[]
 def add(columns,values,b):
  row=len(rhs);rr.extend([row]*len(columns));cc.extend(columns);vals.extend(values);rhs.append(b)
 for i,j in pairs:
  axes=np.array([u[i],v[i],u[j],v[j]])
  delta=xy[j]-xy[i];proj=axes@delta
  radii=.5*(abs(axes@u[i])+abs(axes@v[i])+abs(axes@u[j])+abs(axes@v[j]))
  k=np.argmax(abs(proj)-radii);a=axes[k]*(1 if proj[k]>=0 else -1)
  add([2*i,2*i+1,2*j,2*j+1],[a[0],a[1],-a[0],-a[1]],-radii[k]-1e-9)
 ext=.5*(abs(np.cos(angles))+abs(np.sin(angles)))
 for i in range(n):
  for k in range(2):
   add([2*i+k],[-1],-ext[i]-1e-9)
   add([2*i+k,2*n],[1,-1],-ext[i]-1e-9)
 A=coo_matrix((vals,(rr,cc)),shape=(len(rhs),2*n+1)).tocsr()
 c=np.zeros(2*n+1);c[-1]=1
 bounds=[(float(z)-.03,float(z)+.03) for z in xy.ravel()]+[(0,None)]
 res=linprog(c,A_ub=A,b_ub=rhs,bounds=bounds,method='highs',options={'dual_feasibility_tolerance':1e-9,'primal_feasibility_tolerance':1e-9,'time_limit':30})
 if not res.success:return n,None,dict(status=res.message)
 centers=res.x[:-1].reshape(n,2)
 # Decimal.from_float captures the exact rational float, not invented precision.
 # High-precision center dilation adds strict separation even after pandas roundtrip.
 mp.mp.dps=75
 rows=[]
 for i in range(n):
  rows.append(dict(x=mp.nstr(mp.mpf(str(Decimal.from_float(float(centers[i,0]))))*mp.mpf('1.0000000001'),65),y=mp.nstr(mp.mpf(str(Decimal.from_float(float(centers[i,1]))))*mp.mpf('1.0000000001'),65),angle=mp.nstr(mp.mpf(str(d['s'][i][2]))*mp.pi/180,65)))
 return n,rows,dict(status='ok',gallery_L=d['L'],LP_L=float(res.fun),box_growth=float(res.fun)-float(d['L']),max_center_shift=float(np.max(abs(centers-xy))))
def main():
 base=read_submission(R/'poc/pell_corrugation/data/border-scan/merged.csv')
 old={r['n']:r for r in json.load(open(R/'poc/pell_corrugation/data/border-scan/merged-literal-official.json'))['results']}
 failed=[r['n'] for r in json.load(open(AUDIT/'gallery-literal-official.json'))['results'] if r['reason']!='ok']
 ps={};meta={}
 for n in failed:
  f=O/f'repair-{n}.json'
  if f.exists():z=json.load(open(f));rows=z['rows'];m=z['meta']
  else:
   _,rows,m=repair(n);save(f,dict(rows=rows,meta=m))
  meta[n]=m
  if rows:ps[n]=rows
  print('repair',n,m,flush=True)
 write(O/'repaired.csv',ps)
 save(O/'repair-metadata.json',meta)
 checks=[]
 for mode in ['literal','pandas']:
  p=O/'repaired.csv'
  if mode=='pandas':
   p=O/'repaired-pandas.csv';pd.read_csv(O/'repaired.csv').to_csv(p,index=False)
  for rt in ([False,True] if mode=='literal' else [False]):
   dest=O/f'{mode}-cpp-rt{rt}.json'
   subprocess.run([CERTIFIER,str(p),'--threads','3','--details',str(dest)]+(['--roundtrip'] if rt else []),check=True)
   checks.append({r['n']:r for r in json.load(open(dest))['results']})
  rep=score(read_submission(p));save(O/f'{mode}-official.json',rep)
  checks.append({r['n']:r for r in rep['results']})
  print('official',mode,rep['accepted'],rep['rejected'],flush=True)
 wins={}
 for n,rows in ps.items():
  if all(c[n]['reason']=='ok' for c in checks):
   new=checks[-1][n]['L'];lit=checks[0][n]['L']
   if Decimal(lit)<Decimal(old[n]['L']) and Decimal(new)<Decimal(old[n]['L']):
    base[n]=rows;wins[n]=dict(old_L=old[n]['L'],new_L=lit,roundtrip_L=new,gain_score=float((Decimal(old[n]['L'])-Decimal(lit))/Decimal(n).sqrt()),**meta[n])
 write(O/'merged.csv',base)
 save(O/'wins.json',wins)
 for mode in ['literal','pandas']:
  p=O/'merged.csv'
  if mode=='pandas':
   p=O/'merged-pandas.csv';pd.read_csv(O/'merged.csv').to_csv(p,index=False)
  for rt in [False,True]:
   dest=O/f'merged-{mode}-cpp-rt{rt}.json'
   subprocess.run([CERTIFIER,str(p),'--threads','3','--details',str(dest)]+(['--roundtrip'] if rt else []),check=True)
   z=json.load(open(dest));assert z['accepted']==1000 and z['rejected']==0
  rep=score(read_submission(p));save(O/f'merged-{mode}-official.json',rep);assert rep['accepted']==1000 and rep['rejected']==0
 save(O/'COMPLETE.json',dict(score=rep['score'],improved=len(wins),attempted=len(failed),LP_success=len(ps),pandas=pd.__version__,top=sorted(wins.items(),key=lambda z:-z[1]['gain_score'])[:20]))
 print('DONE',json.load(open(O/'COMPLETE.json')),flush=True)
if __name__=='__main__':main()
