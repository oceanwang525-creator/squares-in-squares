import json
import math
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

from portfolio import END, OUT, START, TEAM, evaluate_single_puzzle, load, points, to_rows, write
import numpy as np
from algorithms import max_window, orient, relax, step
from sqk import grow, lp
from sqk.geometry import fold_angle, half_extent, min_gap, normalize, side
from sqk.submission import export_ready


def optimize(n, p, length, mode):
    p = normalize(p)
    original = side(p)
    if mode == 'incumbent':
        q = lp.compact(p, trust=.05, iters=5)
        for angle in (.01, .004, .0015, .0005, .00015, .00005, .000015, .000004):
            d = step(q, min(.045, max(.002, angle*5)), angle)
            if d is None:
                continue
            solved = lp.solve(normalize(q+d), trust=.04, margin=2e-12)
            if solved.status == 0 and solved.L < side(q) - 1e-10:
                q = solved.P
    else:
        q = p.copy()
        q[:, :2] *= length / original
        q, loss = relax(q, length - .00015, n + round(original*1e6), displacement=.26)
        if loss > 0.0002 or min_gap(q) < -.045:
            return original, loss, None
        q = grow.repair(q, trust=.15, iters=7)
        if q is None:
            return original, loss, None
        q = lp.compact(q, trust=.1, iters=6)
        for angle in (.006, .002, .0006, .0002, .00006):
            d = step(q, .018, angle)
            if d is None:
                continue
            solved = lp.solve(normalize(q+d), trust=.035, margin=2e-12)
            if solved.status == 0 and solved.L < side(q) - 1e-9:
                q = solved.P
        original = side(q)
        if original >= length - 1e-8:
            return original, loss, None
    length_q = side(q)
    if length_q >= length - 1e-9:
        return length_q, None, None
    try:
        return length_q, None, export_ready(q, 2e-12)
    except ValueError:
        return length_q, None, None


def donors(groups, bounds):
    for m, p0 in groups.items():
        targets = [n for n in bounds if m>n and m<=n+28 and side(p0) <= bounds[n]+.68]
        if not targets:
            continue
        for angle in (0., -.016, -.008, -.003, .003, .008, .016):
            c,s = math.cos(angle), math.sin(angle)
            p = np.column_stack((p0[:, :2] @ np.array([[c,s],[-s,c]]), fold_angle(p0[:,2]+angle)))
            h = half_extent(p[:,2])[:,None]
            lo, hi = np.ascontiguousarray(p[:,:2]-h), np.ascontiguousarray(p[:,:2]+h)
            xs, xe, ys = [np.argsort(v) for v in (hi[:,0],lo[:,0],hi[:,1])]
            inverse = np.argsort(ys)
            for n in targets:
                length = bounds[n]
                left, right = length-.001, length + .38
                count, x, y = max_window(lo, hi, xs, xe, ys, inverse, right)
                if count < n:
                    continue
                for _ in range(7):
                    middle = (left+right)/2
                    c0, x0, y0 = max_window(lo, hi, xs, xe, ys, inverse, middle)
                    if c0 >= n:
                        right, x, y = middle, x0, y0
                    else:
                        left = middle
                indices = np.flatnonzero(((lo >= np.array([x,y])-4e-10)&(hi <= np.array([x,y])+right+4e-10)).all(1))
                if len(indices) < n:
                    continue
                if len(indices) > n:
                    center = np.array([x+right/2,y+right/2])
                    distance = np.max(abs(p[indices,:2]-center),axis=1)
                    indices = indices[np.argsort(distance)[:n]]
                q = p[indices]
                if n>167 and (bounds[n].is_integer() and not abs(np.sin(2*q[:,2])).max() > .03):
                    continue
                q1 = orient(q)
                if q1 is not None:
                    q = q1
                yield n, side(q), (m, angle), q


def main():
    rows = load(TEAM, lo=155, hi=230)
    groups = {n: points(rs) for n,rs in rows.items()}
    bounds = {n: side(groups[n]) for n in range(START, END+1) if math.sqrt(n)+1e-7 < side(groups[n])}
    with ProcessPoolExecutor(max_workers=2) as pool:
        jobs = {pool.submit(optimize, n, groups[n], length, 'incumbent'): (n, 'incumbent')
                for n,length in bounds.items() if length % 1 > 1e-8}
        best = defaultdict(list)
        for n,L,source,p in donors(groups,bounds):
            best[n].append((L,source,p))
        for n, choices in sorted(best.items()):
            chosen = []
            for L, source, p in sorted(choices, key=lambda x: x[0]):
                if any(source[0]==m and abs(source[1]-a)<.007 for m,a in chosen):
                    continue
                chosen.append(source)
                jobs[pool.submit(optimize, n, p, bounds[n], 'donor')] = (n, source)
                if len(chosen) >= 5:
                    break
            print('donors', n, f'{min(x[0] for x in choices)-bounds[n]:+.8f}', chosen,flush=True)
        wins = {}
        trials = []
        for k,future in enumerate(as_completed(jobs),1):
            n, source = jobs[future]
            try:
                length, loss, p = future.result()
            except Exception as err:
                print('error',n,source,repr(err),flush=True)
                continue
            trials.append(dict(n=n, source=source, side=length, loss=loss))
            if p is not None:
                rs = to_rows(n,p)
                ck = evaluate_single_puzzle(n,rs)
                if ck[-1]=='ok' and float(ck[2]) < min(bounds[n], wins.get(n,(math.inf,))[0]) - 1e-9:
                    print('win',n,source,ck,flush=True)
                    wins[n] = (float(ck[2]),rs)
                    write(OUT/'recrop_wins.csv',{n:rs for n,(_,rs) in wins.items()})
            if k % 20 == 0:
                print('trials', k, '/', len(jobs), flush=True)
            (OUT/'recrop_trials.json').write_text(json.dumps(trials,indent=2))


if __name__ == '__main__':
    main()
