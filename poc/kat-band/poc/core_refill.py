"""Finite-candidate FTMC upright refill, not a continuous packing certificate.

The source briefing describes, but does not provide code for, fill_rect.
Here staircase anchors are intersections of exact square/upright Minkowski
boundary edges with wall lattice lines, plus boundary vertices. Integer
translations up to depth four form the staircase candidates. MILP optimality
refers ONLY to this enumerated graph, never all possible upright positions.
"""
from dataclasses import dataclass
import time
import numpy as np
from scipy.spatial import cKDTree, ConvexHull
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import coo_matrix
try:
    from .band_polish import sat_check
except ImportError:
    from band_polish import sat_check


@dataclass
class RefillResult:
    uprights: np.ndarray | None
    diagnostics: dict
    x: np.ndarray | None = None


def _array(a, columns):
    a = np.asarray(a, dtype=np.float64)
    if a.size == 0:
        return np.empty((0, columns), dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != columns or not np.isfinite(a).all():
        raise ValueError(f'expected finite (n,{columns}) array')
    return a.copy()


def _valid(q, W, H):
    if not len(q):
        return True
    z = q.astype(np.longdouble)
    half = (np.abs(np.cos(z[:, 2])) + np.abs(np.sin(z[:, 2]))) / 2
    return bool(np.all(z[:, :2] - half[:, None] >= 0) and
                np.all(z[:, :2] + half[:, None] <= [W, H]) and sat_check(q)[0])


def refill(core, W, H, target, *, hints=None, old_uprights=None, depth=4,
           candidate_cap=20000, time_limit=5.0):
    """Return exactly target upright rows (x,y,0), or None with diagnostics.

    core is immutable float64-compatible (n,3), angles in radians, coordinates
    in [0,W] x [0,H]. old_uprights is optional (m,2) incumbent centers.
    hints is an optional (m,2) center array (old_uprights is an alias).
    Returns RefillResult with .uprights, .x (combined core + uprights), and
    .diagnostics. On refusal .uprights and .x are None. Hints are candidates,
    never locked selections. Both outputs are new arrays.
    A successful result is strictly all-pairs screened without overlap slack.
    A time/candidate cap never implies a proof of optimality. Time is checked
    between bounded geometry operations and passed as remaining time to HiGHS;
    it is a cooperative budget, not a hard process-kill deadline.
    """
    start = time.monotonic(); deadline = start + time_limit
    core = _array(core, 3)
    if (not np.isfinite([W,H,time_limit]).all() or W <= 0 or H <= 0 or
            time_limit <= 0 or int(target) != target or target < 0 or
            int(depth) != depth or not 0 <= depth <= 4 or candidate_cap < 1):
        raise ValueError('invalid dimensions, target, depth, cap or time limit')
    target = int(target)
    d = dict(candidate_cap_hit=False, timed_out=False, graph_optimal=False,
             candidates=0, edges=0, selected=0, simplicial=0, components=[])
    def finish(rows=None, reason='ok'):
        d['reason'] = reason; d['elapsed'] = time.monotonic()-start
        combined = None if rows is None else np.concatenate((core, rows))
        return RefillResult(rows, d, combined)
    if not _valid(core, W, H):
        return finish(reason='invalid_core')
    if target == 0:
        return finish(np.empty((0,3), dtype=np.float64))
    if hints is not None and old_uprights is not None:
        raise ValueError("supply hints or old_uprights, not both")
    supplied = hints if hints is not None else old_uprights
    old = np.empty((0,2)) if supplied is None else _array(supplied, 2)
    # Cache tilted axes once; candidate checking is O(core), not O(core**2).
    lc=core.astype(np.longdouble)
    cu=np.column_stack((np.cos(lc[:,2]),np.sin(lc[:,2])))
    cv=np.column_stack((-cu[:,1],cu[:,0]))
    support=(np.abs(cu[:,0])+np.abs(cu[:,1]))/2
    def clear_candidate(x,y):
        delta=np.array([x,y],dtype=np.longdouble)-lc[:,:2]
        best=np.maximum(np.abs(delta[:,0])-support-.5,
                        np.abs(delta[:,1])-support-.5)
        for axis in (cu,cv):
            gap=np.abs(np.sum(delta*axis,axis=1))-.5-(np.abs(axis).sum(axis=1)/2)
            best=np.maximum(best,gap)
        return bool(np.all(best>=0))
    points = []; seen = set()
    def add(x, y):
        if time.monotonic() >= deadline:
            d['timed_out'] = True; return False
        key = (float(x), float(y))
        if key in seen:
            return True
        seen.add(key)
        if not (.5 <= x <= W-.5 and .5 <= y <= H-.5):
            return True
        if not clear_candidate(x,y):
            return True
        if len(points) >= candidate_cap:
            d['candidate_cap_hit'] = True; return False
        points.append(key); return True
    # Incumbent first, so capped searches retain identity recovery candidates.
    for x,y in old:
        if not add(x,y): break
    xs = np.unique(np.r_[np.arange(.5,W,1), W-np.arange(.5,W,1)])
    ys = np.unique(np.r_[np.arange(.5,H,1), H-np.arange(.5,H,1)])
    stopped = d['timed_out'] or d['candidate_cap_hit']
    for x in xs:
        if stopped: break
        for y in ys:
            if not add(x,y): stopped=True; break
    # Forbidden-center polygon = tilted square Minkowski-summed with upright.
    for cx,cy,a in core:
        if stopped: break
        c,s = np.cos(a),np.sin(a)
        local = np.array([[-.5,-.5],[-.5,.5],[.5,.5],[.5,-.5]])
        vertices = local @ np.array([[c,s],[-s,c]]) + [cx,cy]
        sums = (vertices[:,None,:]+local[None,:,:]).reshape(-1,2)
        poly = sums[ConvexHull(sums).vertices]
        anchors = list(poly)
        for p,r in zip(poly,np.roll(poly,-1,axis=0)):
            delta = r-p
            for axis, lattice in ((0,xs),(1,ys)):
                if delta[axis] == 0: continue
                for value in lattice:
                    t = (value-p[axis])/delta[axis]
                    if 0 <= t <= 1: anchors.append(p+t*delta)
        for anchor in anchors:
            if stopped: break
            for ix in range(-depth,depth+1):
                if stopped: break
                for iy in range(-depth,depth+1):
                    if not add(*(anchor+[ix,iy])): stopped=True; break
    d['candidates'] = len(points)
    if d['timed_out']: return finish(reason='generation_timeout')
    if not points: return finish(reason='no_candidates')
    xy = np.asarray(points, dtype=np.float64)
    # L-infinity radius 1 includes exact touching pairs; only strict conflicts.
    neighbors = cKDTree(xy)
    adj = [set() for _ in xy]
    for i in range(len(xy)):
        if time.monotonic() >= deadline:
            d['timed_out']=True; return finish(reason='graph_timeout')
        for j in neighbors.query_ball_point(xy[i], 1., p=np.inf):
            if j > i and np.all(np.abs(xy[i].astype(np.longdouble)-xy[j]) < 1):
                adj[i].add(j); adj[j].add(i); d['edges'] += 1
    # A simplicial vertex belongs to some maximum independent set.
    active=set(range(len(xy))); chosen=[]
    changed=True
    while changed and len(chosen)<target:
        changed=False
        for v in sorted(active):
            if time.monotonic() >= deadline:
                d['timed_out']=True; break
            ns=adj[v]&active
            if all((ns-{u}) <= adj[u] for u in ns):
                chosen.append(v); active.difference_update(ns|{v})
                d['simplicial']+=1; changed=True; break
        if d['timed_out']: break
    optimal=True
    while active and len(chosen)<target and not d['timed_out']:
        seed=min(active); comp={seed}; stack=[seed]
        while stack:
            v=stack.pop()
            fresh=(adj[v]&active)-comp; comp.update(fresh); stack.extend(fresh)
        active-=comp; ids=sorted(comp); index={v:i for i,v in enumerate(ids)}
        edges=[(index[v],index[u]) for v in ids for u in adj[v]&comp if v<u]
        remaining=deadline-time.monotonic()
        if remaining<=0: d['timed_out']=True; break
        rr=[];cc=[]
        for k,(i,j) in enumerate(edges): rr.extend([k,k]); cc.extend([i,j])
        A=coo_matrix((np.ones(len(rr)),(rr,cc)),shape=(len(edges),len(ids))).tocsc()
        res=milp(-np.ones(len(ids)),integrality=np.ones(len(ids)),
                 bounds=Bounds(0,1),constraints=LinearConstraint(A,-np.inf,1),
                 options={'time_limit':remaining})
        d['components'].append({'size':len(ids),'status':int(res.status)})
        optimal &= res.status==0
        if res.status==1: d['timed_out']=True
        if res.x is not None:
            selected=[ids[i] for i,x in enumerate(res.x) if x>.5]
            if all(not (adj[v]&set(selected)) for v in selected): chosen.extend(selected)
    d['selected']=len(chosen)
    d['graph_optimal']=bool(optimal and not active and not d['timed_out'] and
                             not d['candidate_cap_hit'])
    if len(chosen)<target: return finish(reason='insufficient_refill')
    rows=np.column_stack((xy[chosen[:target]],np.zeros(target))).astype(np.float64)
    if not _valid(np.concatenate((core,rows)),W,H):
        return finish(reason='strict_validation_failed')
    return finish(rows)
