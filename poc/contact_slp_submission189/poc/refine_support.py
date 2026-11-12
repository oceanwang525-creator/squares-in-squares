import concurrent.futures as cf
import csv, math, os, sys
from collections import defaultdict
from pathlib import Path
os.environ["OPENBLAS_NUM_THREADS"]="1"
os.environ["OMP_NUM_THREADS"]="1"
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from sqk import lp
from sqk.geometry import side,normalize,inflate,min_gap,half_extent
from sqk.submission import export_ready
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'baseline' / 'poc' / 'src'))
from squares.core import evaluate as evaluate_single_puzzle

def step(p, rt, xy, rng, noise):
 n=len(p); L=side(p)
 pa=lp.choose_axes(p,math.sqrt(2)+2*math.sqrt(2)*xy+1e-6)
 i,j=pa.pairs.T
 owner=np.where(pa.owner==0,i,j)
 other=np.where(pa.owner==0,j,i)
 th=p[:,2]
 axis=lp._axis(th[owner],pa.k,pa.sign)
 dx=p[j,0]-p[i,0];dy=p[j,1]-p[i,1]
 ad=-axis[:,1]*dx+axis[:,0]*dy
 pr=axis[:,0]*dx+axis[:,1]*dy
 d=th[other]-th[owner]
 c,s=np.cos(d),np.sin(d)
 rr=[]; cc=[]; vv=[]; bb=[]
 def add(ids,coeff,b):
  row=np.arange(len(b))+len(bb)
  for col,v in zip(ids,coeff):
   rr.extend(row);cc.extend(col);vv.extend(v)
  bb.extend(b)
 for u,v in ((1,1),(1,-1),(-1,1),(-1,-1)):
  sup=.5*(1+u*c+v*s)
  ds=.5*(-u*s+v*c)
  b=pr-sup-5e-11
  keep=b<.02+3*xy+3*rt
  add([z[keep] for z in (i,j,n+i,n+j,2*n+owner,2*n+other)],
      [z[keep] for z in (axis[:,0],-axis[:,0],axis[:,1],-axis[:,1],-ad-ds,ds)],b[keep])
 k=np.arange(n)
 for u,v in ((1,1),(1,-1),(-1,1),(-1,-1)):
  e=.5*(u*np.cos(th)+v*np.sin(th))
  de=.5*(-u*np.sin(th)+v*np.cos(th))
  for ax in (0,1):
   z=p[:,ax]
   for sgn in (-1,1):
    b=(z if sgn<0 else L-z)-e
    keep=b<.01+xy+2*rt
    ids=k[keep]
    add([ids+ax*n,ids+2*n,np.full(len(ids),3*n)],[np.full(len(ids),sgn),de[keep],np.full(len(ids),0 if sgn<0 else -1)],b[keep])
 mat=coo_matrix((vv,(rr,cc)),shape=(len(bb),3*n+1)).tocsr()
 obj=np.zeros(3*n+1);obj[-1]=1
 if noise:obj[2*n:3*n]=rng.normal(0,noise,n)
 res=linprog(obj,A_ub=mat,b_ub=bb,bounds=[(-xy,xy)]*(2*n)+[(-rt,rt)]*n+[(-1,1)],method="highs",options={"primal_feasibility_tolerance":1e-9,"dual_feasibility_tolerance":1e-9,"time_limit":6})
 if not res.success:return None
 q=p+res.x[:-1].reshape(3,n).T
 return q,float(res.x[-1])

def work(n, p):
 p=normalize(p)
 rng=np.random.default_rng(1000+n)
 best=p;rt=.014;xy=.04
 for j in range(22):
  v=step(best,rt,xy,rng,0 if j<9 else 0.0004)
  if v is None:rt*=.5;xy*=.6;continue
  q,pred=v
  if pred>-.000000002 and j<8:
   rt*=.45;xy*=.6
   if rt<5e-7: break
   continue
  if min_gap(q)<-1e-9:
   r=lp.solve(q,trust=.025,margin=1e-11)
   if r.status==0:
    q=r.P
   else:
    q=inflate(q,max(2e-10,-min_gap(q)*1.005))
  if side(q)<side(best)-2e-10 and min_gap(q)>-1e-9:
   best=q
   rt=min(rt*1.1,.025);xy=min(xy*1.1,.06)
  else:rt*=.50;xy*=.60
  if rt<5e-8:break
 if side(best)<side(p)-5e-10:
  try:return n, export_ready(best)
  except ValueError:return
 return None

def main():
 import argparse
 parser=argparse.ArgumentParser(description='Joint center and angle refinement using support constraints')
 parser.add_argument('seed')
 parser.add_argument('output')
 parser.add_argument('n', nargs='+', type=int)
 args=parser.parse_args()
 groups=defaultdict(list)
 with open(args.seed,newline='') as f:
  for r in csv.DictReader(f):groups[int(r['n'])].append(r)
 ps={n:np.array([[float(r[k]) for k in ('x','y','angle')] for r in rows]) for n,rows in groups.items()}
 targets=args.n
 print(len(targets),'SLP targets',flush=True)
 wins={}; gains={}
 with cf.ProcessPoolExecutor(max_workers=3) as ex:
  jobs={ex.submit(work,n,ps[n]):n for n in targets}
  for i,f in enumerate(cf.as_completed(jobs),1):
   try:result=f.result()
   except Exception as e:print(jobs[f],type(e).__name__,str(e),flush=True);continue
   if result is not None:
    n,q=result
    rows=[dict(r,x=repr(x),y=repr(y),angle=repr(a)) for r,(x,y,a) in zip(groups[n],q.tolist())]
    ck=evaluate_single_puzzle(n,rows)
    if ck[-1]=='ok' and float(ck[2])<side(ps[n])-5e-10:
     groups[n]=rows;wins[n]=float(ck[2]);gains[n]=(side(ps[n])-wins[n])/math.sqrt(n)
     print('win',n,wins[n],gains[n],flush=True)
   if i%10==0:print(i,'screened',flush=True)
   if wins and (i%10==0 or i==len(targets)):
    with open(args.output,'w',newline='') as out:
     w=csv.DictWriter(out,fieldnames=['id','n','square_id','x','y','angle']);w.writeheader()
     for rs in groups.values():w.writerows(rs)
 print(len(wins),sum(gains.values()),wins,flush=True)
if __name__=='__main__':main()

