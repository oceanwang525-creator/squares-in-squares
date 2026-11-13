"""Try inserting a square into a locally deformable nearby packing."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from time import monotonic

from common import arguments, array, convert, groups, write_candidate
from patch import energy, try_move
import numpy as np
from scipy.spatial import Voronoi, cKDTree
from sqk import lp
from sqk.geometry import min_gap, normalize, side
from sqk.submission import export_ready

def starts(donor, target, count=14):
    donor=normalize(donor)
    donor=donor.copy()
    donor[:,:2]+=.5*max(0,target-side(donor))
    where=donor[:,:2]
    step=.2
    g=np.arange(.5,target-.45,step)
    cells=np.array(np.meshgrid(g,g)).reshape(2,-1).T
    vor=Voronoi(where).vertices
    sites=np.vstack([cells,vor])
    sites=sites[np.all((sites>=.45)&(sites<=target-.45),axis=1)]
    _,ix=cKDTree(where).query(sites,k=12)
    neigh=donor[ix]
    disp=neigh[:,:,:2]-sites[:,None,:]
    dx,dy=disp.transpose(2,0,1)
    c=np.cos(neigh[:,:,2])
    s=np.sin(neigh[:,:,2])
    theirs=np.maximum(np.abs(dx*c+dy*s),np.abs(dy*c-dx*s))
    trial=[]
    for angle in np.arange(-8,9)*np.pi/32:
        c,s=np.cos(angle),np.sin(angle)
        mine=np.maximum(np.abs(dx*c+dy*s),np.abs(dy*c-dx*s))
        rel=neigh[:,:,2]-angle
        threshold=(1+np.abs(np.cos(rel))+np.abs(np.sin(rel)))*.5
        overlap=np.maximum(threshold-np.maximum(mine,theirs),0)
        e=.5*(abs(c)+abs(s))
        wall=np.maximum(e-sites,0)+np.maximum(sites+e-target,0)
        cost=(overlap**2).sum(axis=1)+(wall**2).sum(axis=1)
        k=min(500,len(cost))
        indexes=np.argpartition(cost,k-1)[:k]
        trial.extend((cost[i],np.r_[sites[i],angle]) for i in indexes)
    retained=[]
    for cost,seed in sorted(trial,key=lambda entry:entry[0]):
        if any(np.linalg.norm(seed[:2]-other[:2])<.55 and abs(seed[2]-other[2])<.38 for _,other in retained):
            continue
        retained.append((cost,seed))
        if len(retained)>=count:
            break
    return donor,retained

def solve(job):
    n, donor, target, budget = job
    points, trial = starts(donor,target)
    rng=np.random.default_rng(n+20261112)
    best=None
    incumbent=target
    started=monotonic()
    lowest=1.
    attempts=0
    for _, square in trial:
        if monotonic()-started>budget:
            break
        for perturb in (0.,.045,.12):
            p=np.vstack([points,square])
            q=p.copy()
            if perturb:
                distances=np.linalg.norm(q[:,:2]-square[:2],axis=1)
                nearby=distances<2.4
                q[nearby]+=rng.normal(0,perturb,(np.sum(nearby),3))
            q,e=try_move(p,target-1e-5,q,.42,900)
            attempts+=1
            lowest=min(e,lowest)
            if e>1e-3 or min_gap(q)<-.035:
                continue
            result=lp.solve(q,trust=.09,margin=1e-11)
            if result.status==0 and min_gap(result.P)>-1e-8 and result.L<target-1e-8:
                try:
                    best=export_ready(result.P)
                except ValueError:
                    continue
                target=side(best)
    return n,attempts,lowest,best

def main():
    args = arguments(__doc__)
    baseline=groups(args.incumbent)
    points={n:array(rows) for n,rows in baseline.items()}
    energy(np.array([.5,.5,0.]),np.zeros((0,2),dtype=np.int64),1.)
    selected=(102,103,104,105,106,107,108,109,110,111,123,124,125,126,127,128,129,130,131,132,133)
    jobs=[(n,points[n-1],side(points[n]),8) for n in selected]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for task in as_completed([pool.submit(solve,job) for job in jobs]):
            n,attempts,lowest,p=task.result()
            print(n,attempts,lowest,flush=True)
            if p is None:
                continue
            rows=convert(baseline[n],p)
            result = write_candidate(n, rows, baseline[n], args.output, 'insert')
            if result is not None:
                print('win', n, result[0][2], result[1][2], flush=True)

if __name__=='__main__':
    main()
