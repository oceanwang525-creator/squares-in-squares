import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed

from portfolio import OUT, TEAM, evaluate_single_puzzle, load, points, to_rows, write
import numpy as np
from algorithms import relax, seeds, step
from sqk import grow, lp
from sqk.geometry import fold_angle, min_gap, neighbor_pairs, normalize, pair_gaps, side
from sqk.submission import export_ready


def perturb(n, p, mode, seed):
    rng = np.random.default_rng(n*10_007 + seed)
    q = p.copy()
    if mode == 'sweep':
        selected = rng.random(n) < .35
        q[selected,:2] += rng.normal(0, .018, (selected.sum(),2))
        q[selected,2] += rng.normal(0, .045, selected.sum())
    elif mode == 'patch':
        core = np.flatnonzero(abs(q[:,2]) > .07)
        center = q[rng.choice(core),:2]
        radius = rng.uniform(1.3, 3.2)
        selected = np.linalg.norm(q[:,:2]-center, axis=1) < radius
        q[selected,:2] += rng.normal(0, rng.uniform(.02,.11), (selected.sum(),2))
        q[selected,2] += rng.normal(0, rng.uniform(.02,.16), selected.sum())
    return q


def shrink(n, p, length, label, seed):
    q = normalize(p)
    q[:,2] = fold_angle(q[:,2])
    q = perturb(n, q, label, seed)
    current = side(q)
    if current > length:
        q[:,:2] *= length / current
    q,e = relax(q, length-.0009, seed, displacement=.44)
    if e > .001 or min_gap(q) < -.05:
        return side(q), e, None
    q = grow.repair(q, trust=.14, iters=9)
    if q is None:
        return current, e, None
    for t in (.02,.008,.003,.001,.0004,.00016,.00005,.000015):
        d = step(q,.035,t)
        if d is None:
            continue
        fixed=lp.solve(normalize(q+d),trust=.05,margin=2e-12)
        if fixed.status==0 and fixed.L<side(q)-1e-10 and min_gap(fixed.P)>-1e-8:
            q = fixed.P
    best = side(q)
    if best >= length - 1e-9:
        return best, e, None
    try:
        return best, e, export_ready(q, 2e-12)
    except ValueError:
        return best, e, None


def deletions(n, p, goal):
    p = normalize(p)
    candidates = []
    for i in range(len(p)):
        q = np.delete(p, i, axis=0)
        target = q.copy()
        target[:,:2] *= goal / side(q)
        pairs = neighbor_pairs(target, 1.46)
        gaps = pair_gaps(target, pairs)
        cost = np.square(np.minimum(gaps, 0)).sum()
        candidates.append((cost, i, q))
    selected=[]
    for cost, i, q in sorted(candidates,key=lambda item:item[0]):
        if all(np.linalg.norm(p[i,:2]-p[j,:2])>.95 for j in selected):
            yield i,q
            selected.append(i)
        if len(selected)>=8:
            break


def main():
    groups = {n:normalize(points(rs)) for n,rs in load(TEAM,lo=167,hi=202).items()}
    lengths = {n:side(p) for n,p in groups.items()}
    targets = (199, 184, 183, 180, 179, 175, 172, 198, 197, 181, 182, 176, 177, 178, 174, 173, 171, 170, 200)
    trials=[]
    wins={}
    with ProcessPoolExecutor(max_workers=2) as pool:
        tasks={}
        for n in targets:
            for seed in range(1, 5):
                if n==184:
                    break
                mode='patch' if seed%2 else 'sweep'
                task = pool.submit(shrink,n,groups[n],lengths[n],mode,seed)
                tasks[task] = (n,mode,seed)
            if n in (199,184,183,180,179,175):
                for seed,(_, q) in enumerate(seeds(groups[n-1], lengths[n]), 100):
                    task=pool.submit(shrink,n,q,lengths[n],'insert',seed)
                    tasks[task]=(n,'insert',seed)
            if n in (199,180,175,172):
                for seed,q in deletions(n,groups[n+1],lengths[n]):
                    task=pool.submit(shrink,n,q,lengths[n],'delete',200+seed)
                    tasks[task]=(n,'delete',200+seed)
        for j, task in enumerate(as_completed(tasks),1):
            n,mode,seed = tasks[task]
            try:
                length, error, p = task.result()
            except Exception as err:
                print('error',n,mode,seed,repr(err),flush=True)
                continue
            trials.append(dict(n=n, mode=mode, seed=seed, side=length, loss=error))
            if p is not None:
                rs=to_rows(n,p)
                ck=evaluate_single_puzzle(n,rs)
                if ck[-1]=='ok' and float(ck[2])<min(lengths[n],wins.get(n,(math.inf,))[0])-1e-9:
                    wins[n] = (float(ck[2]),rs)
                    write(OUT/'restart_wins.csv',{n:rs for n,(_,rs) in wins.items()})
                    print('win',n,mode,seed,ck,flush=True)
            if j % 20 == 0:
                print('trials',j,'/',len(tasks),flush=True)
            (OUT/'restart_trials.json').write_text(json.dumps(trials,indent=2))


if __name__=='__main__':
    main()
