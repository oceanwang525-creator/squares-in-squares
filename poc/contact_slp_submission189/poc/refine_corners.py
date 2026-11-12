import csv
import json
import math
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, vstack
from sqk import lp
from sqk.geometry import corners, normalize, side, min_gap
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'baseline' / 'poc' / 'src'))
from squares.core import evaluate as evaluate_single_puzzle
from sqk.submission import export_ready

def perp(v):
    return np.stack((-v[..., 1], v[..., 0]), axis=-1)

def sparse_rows(cols, vals, width):
    return coo_matrix((vals.ravel(), (np.repeat(np.arange(len(cols)), cols.shape[1]), cols.ravel())), shape=(len(cols),width)).tocsr()

def step(p, translation, rotation):
    n=len(p)
    width=3*n+1
    current=side(p)
    c=corners(p)
    edges=c-p[:,None,:2]
    pairs=lp.choose_axes(p,1.51)
    i,j=pairs.pairs.T
    owns_i=pairs.owner==0
    owner=np.where(owns_i,i,j)
    other=np.where(owns_i,j,i)
    axis=lp._axis(p[owner,2],pairs.k,pairs.sign*np.where(owns_i,1,-1))
    offset=p[other,:2]-p[owner,:2]
    vectors=offset[:,None,:]+edges[other]
    gap=np.einsum('md,mkd->mk',axis,vectors)-.5
    ids,k=np.nonzero(gap<.12)
    own,oth=owner[ids],other[ids]
    a=axis[ids]
    angle_own=np.einsum('md,md->m',perp(a),vectors[ids,k])
    angle_oth=np.einsum('md,md->m',a,perp(edges[oth,k]))
    pair_cols=np.column_stack((own,own+n,oth,oth+n,own+2*n,oth+2*n))
    pair_coeff=np.column_stack((a,-a,-angle_own,-angle_oth))
    parts=[sparse_rows(pair_cols,pair_coeff,width)]
    rhs=[gap[ids,k]-1e-10]
    for dim in (0,1):
        for sign in (-1,1):
            upper=sign==1
            slack=(current-c[:,:,dim]) if upper else c[:,:,dim]
            sq,corner=np.nonzero(slack<.12)
            col=np.column_stack((sq+n*dim,sq+2*n,np.full(len(sq),3*n)))
            coeff=np.column_stack((np.full(len(sq),sign),sign*perp(edges)[sq,corner,dim],np.full(len(sq),-int(upper))))
            parts.append(sparse_rows(col,coeff,width))
            rhs.append(slack[sq,corner])
    objective=np.zeros(width)
    objective[-1]=1
    result=linprog(objective,A_ub=vstack(parts).tocsr(),b_ub=np.concatenate(rhs),
        bounds=[(-translation,translation)]*(2*n)+[(-rotation,rotation)]*n+[(-.04,.02)],
        method='highs',options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
    if not result.success or result.x[-1]>-5e-10:
        return None
    return result.x[:-1].reshape(3,n).T

def improve(n,p):
    p=normalize(p)
    original=side(p)
    for factor in (1,.5,.25,.12):
        d=step(p,.014*factor,.004*factor)
        if d is None:
            continue
        for rate in (1,.5):
            q=p+rate*d
            if min_gap(q)<-1e-9:
                fixed=lp.solve(q,trust=.025)
                if fixed.status!=0 or min_gap(fixed.P)<-1e-8:
                    continue
                q=fixed.P
            if side(q)<side(p)-5e-10:
                p=normalize(q)
                break
    if side(p)>=original-1e-9:
        return None
    try:
        return n, export_ready(p,1e-11)
    except ValueError:
        return None

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Joint center and angle refinement using corner contacts')
    parser.add_argument('seed')
    parser.add_argument('output')
    parser.add_argument('n', nargs='+', type=int)
    args = parser.parse_args()
    groups=defaultdict(list)
    with open(args.seed,newline='') as f:
        for r in csv.DictReader(f):
            groups[int(r['n'])].append(r)
    points={n:np.array([[float(r[k]) for k in ('x','y','angle')] for r in rows]) for n,rows in groups.items()}
    sides={n:side(p) for n,p in points.items()}
    total=sum(L/math.sqrt(n)-1 for n,L in sides.items())
    targets=args.n
    print(f'{len(targets)} layouts selected',flush=True)
    wins={}
    with ProcessPoolExecutor(max_workers=2) as pool:
        jobs={pool.submit(improve,n,points[n]):n for n in targets}
        for done,job in enumerate(as_completed(jobs),1):
            try:
                result=job.result()
            except Exception as exc:
                print(jobs[job],str(exc),flush=True)
                continue
            if result:
                n,p=result
                rows=[dict(r,x=repr(x),y=repr(y),angle=repr(a)) for r,(x,y,a) in zip(groups[n],p.tolist())]
                check=evaluate_single_puzzle(n,rows)
                if check[-1]=='ok' and float(check[2])<sides[n]-1e-9:
                    new=float(check[2])
                    wins[n]={'side':new,'gain':(sides[n]-new)/math.sqrt(n)}
                    groups[n]=rows
                    print(n,wins[n],flush=True)
            if done%40==0:
                print(f'{done}/{len(jobs)} finished',flush=True)
    print(json.dumps({'score':total-sum(r['gain'] for r in wins.values()),'wins':len(wins)}),flush=True)
    if wins:
        with open(args.output,'w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['id','n','square_id','x','y','angle'])
            writer.writeheader()
            for rows in groups.values():
                writer.writerows(rows)
        Path(args.output).with_suffix('.json').write_text(json.dumps(wins,indent=2))
if __name__=='__main__':
    main()

