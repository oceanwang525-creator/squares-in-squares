"""Perturb tightly packed patches, relax SAT overlap, then compact."""
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from time import monotonic

from common import arguments, array, convert, groups, write_candidate
import numpy as np
from numba import njit
from scipy.optimize import minimize
from sqk import lp
from sqk.geometry import half_extent, inflate, min_gap, neighbor_pairs, normalize, side
from sqk.submission import export_ready

TARGETS = (105,106,107,108,109,110,123,127,128,129,130,131,132)

@njit(nogil=True,cache=True)
def energy(values, pairs, limit):
    points=values.reshape((-1,3))
    count=len(points)
    cos=np.cos(points[:,2])
    sin=np.sin(points[:,2])
    extents=.5*(np.abs(cos)+np.abs(sin))
    gradients=np.zeros((count,3))
    loss=0.
    for k in range(count):
        e=extents[k]
        derivative=.5*(np.sign(sin[k])*cos[k]-np.sign(cos[k])*sin[k])
        for dim in range(2):
            z=points[k,dim]
            left=max(e-z,0)
            right=max(z+e-limit,0)
            gradients[k,dim]+=2*(right-left)
            gradients[k,2]+=2*(left+right)*derivative
            loss+=left*left+right*right
    for i,j in pairs:
        dx=points[j,0]-points[i,0]
        dy=points[j,1]-points[i,1]
        if dx*dx+dy*dy>2:
            continue
        angle=points[j,2]-points[i,2]
        c=math.cos(angle)
        s=math.sin(angle)
        extent=.5*(1+abs(c)+abs(s))
        winner=-1
        best=-1.
        ax=0.
        ay=0.
        sign=0.
        for k in range(4):
            owner=i if k<2 else j
            if k%2:
                u=-sin[owner]
                v=cos[owner]
            else:
                u=cos[owner]
                v=sin[owner]
            projection=dx*u+dy*v
            if abs(projection)>best:
                best=abs(projection)
                winner=owner
                ax=u
                ay=v
                sign=np.sign(projection)
        overlap=extent-best
        if overlap<=0:
            continue
        weight=2*overlap
        loss+=overlap*overlap
        de=.5*(np.sign(s)*c-np.sign(c)*s)
        gradients[i,0]+=weight*sign*ax
        gradients[i,1]+=weight*sign*ay
        gradients[j,0]-=weight*sign*ax
        gradients[j,1]-=weight*sign*ay
        gradients[i,2]-=weight*de
        gradients[j,2]+=weight*de
        gradients[winner,2]-=weight*sign*(dy*ax-dx*ay)
    return loss,gradients.ravel()

def try_move(points,limit,start,restriction=.27,maxiter=440):
    radius=np.array([restriction,restriction,restriction/1.3])
    bounds=np.column_stack([(points-radius).ravel(),(points+radius).ravel()])
    pairs=neighbor_pairs(points,np.sqrt(2)*(1+2*restriction)+1e-8)
    result=minimize(energy,start.ravel(),args=(pairs,limit),jac=True,
                    method='L-BFGS-B',bounds=bounds,
                    options={'maxiter':maxiter,'gtol':1e-10,'ftol':1e-15,'maxls':30})
    return result.x.reshape((-1,3)),result.fun

def solve(job):
    n, seed, points, budget = job
    best=normalize(points)
    incumbent=side(best)
    length=incumbent
    rng=np.random.default_rng(seed)
    ntries=0
    closest=math.inf
    near=best.copy()
    timer=monotonic()+budget
    tilted=np.abs(np.sin(2*points[:,2]))>1e-8
    choices=np.flatnonzero(tilted)
    while monotonic()<timer:
        delta=rng.choice([.00002,.0002,.001,.004,.01])
        start=(near if ntries%6==0 else best).copy()
        limit=length-delta
        start[:,:2]*=limit/length
        style=ntries%5
        if style==0:
            row=rng.choice(choices)
            dist=np.linalg.norm(start[:,:2]-start[row,:2],axis=1)
            mask=dist<rng.uniform(1.5,3.5)
            strength=rng.uniform(.005,.10)
            start[mask]+=rng.normal(0,strength,(np.sum(mask),3))
        elif style==1:
            start[tilted]+=rng.normal(0,rng.uniform(.005,.12),(np.sum(tilted),3))
        elif style==2:
            start+=rng.normal(0,rng.uniform(.005,.035),start.shape)
        elif style==3:
            start[tilted,2]+=rng.uniform(-.10,.10)
        else:
            row=rng.choice(choices)
            dist=np.linalg.norm(start[:,:2]-start[row,:2],axis=1)
            mask=dist<rng.uniform(.5,2.5)
            start[mask]+=rng.normal(0,rng.uniform(.12,.24),3)
        candidate,e=try_move(best,limit,start)
        ntries+=1
        if e<closest and delta>=.0002:
            closest=e
            near=candidate
        if e>1e-3 or min_gap(candidate)<-.025:
            continue
        for tr in (.04,.12):
            result=lp.solve(candidate,trust=tr,margin=1e-11)
            if result.status==0 and min_gap(result.P)>-1e-8:
                if result.L<length-1e-9:
                    length=result.L
                    best=result.P
                    print('working improvement',n,length,flush=True)
                break
    if length<incumbent-1e-8:
        return n,ntries,closest,export_ready(best)
    return n,ntries,closest,None

def main():
    args = arguments(__doc__)
    baseline = groups(args.incumbent)
    # Compile in the parent before starting workers.
    energy(np.array([.5,.5,0.]),np.zeros((0,2),dtype=np.int64),1.)
    jobs=[(n,20261112+n,array(baseline[n]),8) for n in TARGETS]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(solve,job) for job in jobs]):
            n,tries,energy_min,points=future.result()
            print(n,tries,'tries, lowest target overlap',energy_min,flush=True)
            if points is None:
                continue
            rows=convert(baseline[n],points)
            result = write_candidate(n, rows, baseline[n], args.output, 'patch')
            if result is not None:
                print('win', n, result[0][2], result[1][2], flush=True)

if __name__ == '__main__':
    main()
