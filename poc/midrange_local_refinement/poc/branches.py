"""Reconsider close separating axes before a joint center and angle step."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from time import monotonic
from common import arguments, array, convert, groups, write_candidate
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from sqk import lp
from sqk.geometry import axes_of, half_extent, min_gap, neighbor_pairs, normalize, side
from sqk.submission import export_ready

SIGNS = np.array([[1,1],[1,-1],[-1,1],[-1,-1]])
TARGETS = (105,106,107,108,109,110,123,127,128,129,130,131,132)

def branches(points, reach):
    pairs = neighbor_pairs(points, reach)
    i, j = pairs.T
    angles = points[:,2]
    axes = np.concatenate([axes_of(angles[i]), axes_of(angles[j])],axis=1)
    delta = points[j,:2] - points[i,:2]
    signed = np.einsum('nkd,nd->nk', axes, delta)
    difference = angles[j]-angles[i]
    extent = (1+np.abs(np.cos(difference))+np.abs(np.sin(difference)))*.5
    return pairs, np.abs(signed)-extent[:,None], np.where(signed>0,1.,-1.)

def joint_step(points, trans, rot, rng, branch_noise, desired):
    n = len(points)
    box = side(points)
    pairs, gaps, signs = branches(points, np.sqrt(2)*(1+2*trans)+1e-8)
    i, j = pairs.T
    # Only switch axes with nearly equal true gaps. The LP then determines whether
    # those branches can support the local perturbation under the trust region.
    score = gaps + rng.uniform(0,branch_noise,gaps.shape)
    choice = score.argmax(1)
    axis = lp._axis(points[np.where(choice<2,i,j),2], choice%2, signs[np.arange(len(i)),choice])
    owner = np.where(choice<2,i,j)
    other = np.where(choice<2,j,i)
    delta = points[j,:2]-points[i,:2]
    pr = np.sum(axis*delta, axis=1)
    da = axis[:,0]*delta[:,1]-axis[:,1]*delta[:,0]
    angle = points[other,2]-points[owner,2]
    cos,sin = np.cos(angle),np.sin(angle)
    rows=[]; columns=[]; values=[]; rhs=[]
    def add(ids, coefficients, bounds, threshold):
        keep = bounds <= threshold
        row = np.arange(np.sum(keep))+len(rhs)
        for index, coefficient in zip(ids,coefficients):
            rows.extend(row)
            columns.extend(index[keep])
            values.extend(coefficient[keep])
        rhs.extend(bounds[keep])
    for uc,us in SIGNS:
        support = .5*(1+uc*cos+us*sin)
        derivative = .5*(-uc*sin+us*cos)
        add([i,j,n+i,n+j,2*n+owner,2*n+other],
            [axis[:,0],-axis[:,0],axis[:,1],-axis[:,1],-da-derivative,derivative],
            pr-support-2e-11, .02+3*trans+3*rot)
    cos,sin=np.cos(points[:,2]),np.sin(points[:,2])
    k=np.arange(n)
    for uc,us in SIGNS:
        extent=.5*(uc*cos+us*sin)
        derivative=.5*(-uc*sin+us*cos)
        for dim in (0,1):
            for sign in (-1,1):
                wall = (points[:,dim] if sign<0 else box-points[:,dim])-extent
                add([k+dim*n,k+2*n,np.full(n,3*n)],
                    [np.full(n,sign),derivative,np.full(n,0 if sign<0 else -1)],
                    wall, .01+trans+rot)
    objective=np.zeros(3*n+1)
    objective[-1]=1
    objective[2*n:3*n] = desired
    constraints=coo_matrix((values,(rows,columns)),shape=(len(rhs),3*n+1)).tocsr()
    result=linprog(objective,A_ub=constraints,b_ub=rhs,
                   bounds=[(-trans,trans)]*(2*n)+[(-rot,rot)]*n+[(-.2,.15)],
                   options={'primal_feasibility_tolerance':1e-9,
                            'dual_feasibility_tolerance':1e-9})
    if result.success:
        return normalize(points+result.x[:-1].reshape(3,n).T),result.x[-1]
    return points,0.

def solve(job):
    n, points, seed, budget = job
    rng=np.random.default_rng(seed)
    best = normalize(points)
    length = side(best)
    timer = monotonic()+budget
    tries=0
    while monotonic()<timer:
        walk = best
        noise = rng.choice([.00001,.0003,.003,.015,.04])
        rate = rng.choice([.0, .0002, .001, .006])
        desired = rng.normal(0,rate,n)
        trust = rng.choice([.005,.025,.06])
        for iteration in range(4):
            q, prediction = joint_step(walk, trust, trust/3, rng, noise, desired)
            if prediction >= .03:
                break
            if min_gap(q)<-1e-10:
                repaired=lp.solve(q,trust=.035, margin=1e-11, tol=1e-10)
                if repaired.status != 0 or min_gap(repaired.P) < -1e-8:
                    break
                q=repaired.P
            if side(q) < length-1e-8:
                best=q
                length=side(best)
            if side(q) > length+.04 or np.max(np.abs(q-walk))<1e-8:
                break
            walk=q
            noise *= .35
            desired *= .25
            trust *= .7
        tries+=1
    if length<side(points)-1e-8:
        return n, tries, export_ready(best)
    return n,tries,None

def main():
    args = arguments(__doc__)
    baseline = groups(args.incumbent)
    jobs=[(n,array(baseline[n]),20261112+n,9) for n in TARGETS]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(solve,job) for job in jobs]):
            n,tries,points=future.result()
            print(n,tries,'tries',flush=True)
            if points is None:
                continue
            rows=convert(baseline[n],points)
            result = write_candidate(n, rows, baseline[n], args.output, 'branch')
            if result is not None:
                print('win', n, result[0][2], result[1][2], flush=True)

if __name__ == '__main__':
    main()
