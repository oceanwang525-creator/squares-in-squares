import math
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

from common import arguments, array, convert, groups, write_candidate
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from sqk import lp
from sqk.geometry import inflate, min_gap, normalize, side
from sqk.submission import export_ready


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
    args = arguments('Screen supplied layouts by joint center and angle compaction', extra_seeds=True)
    baseline = groups(args.incumbent)
    sides = {n: side(array(rows)) for n, rows in baseline.items()}
    seeds = defaultdict(list)
    signatures = defaultdict(set)
    for path in [args.incumbent, *args.extra_seeds]:
        for n, rows in groups(path).items():
            if len(rows) != n:
                continue
            points = array(rows)
            signature = tuple(np.round(points, 8).ravel())
            if signature in signatures[n]:
                continue
            signatures[n].add(signature)
            length = side(points)
            if length < sides[n] - 1e-9:
                candidate = [dict(base, **{key: row[key] for key in ('x','y','angle')})
                             for base, row in zip(baseline[n], rows, strict=True)]
                result = write_candidate(n, candidate, baseline[n], args.output, f'seed-{path.stem}')
                if result is not None:
                    print('seed win', n, path.name, result[0][2], result[1][2], flush=True)
            if length < sides[n] + 0.1 and np.any(points[:,2]):
                seeds[n].append((length, path.name, points))
    jobs = [(n,name,p) for n, layouts in seeds.items()
            for length,name,p in sorted(layouts,key=lambda x:x[0])[:5]]
    print(len(jobs), 'jobs', flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = {pool.submit(work,n,p):(n,name) for n,name,p in jobs}
        for task in as_completed(results):
            n, name = results[task]
            outcome = task.result()
            if outcome is None:
                continue
            _, points = outcome
            if side(points) < sides[n]-1e-9:
                rows = convert(baseline[n],points)
                result = write_candidate(n, rows, baseline[n], args.output, f'slp-{Path(name).stem}')
                if result is not None:
                    print('win', n, name, result[0][2], result[1][2], flush=True)


if __name__ == '__main__':
    main()
