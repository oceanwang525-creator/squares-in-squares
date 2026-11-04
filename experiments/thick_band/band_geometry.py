"""Constructive off-center finite tilted bands, with upright triangular corners.

No atlas coordinates or files are read by this module.
angle radians; each row (x,y,angle) is a unit square.
CPU topology discovery only. Arrays can be handed directly to a GPU relaxer.
"""
from dataclasses import dataclass, asdict, replace
import csv
import math
import numpy as np
import mpmath as mp

CORNERS = ((0, 0), (1, 1), (0, 1), (1, 0))


@dataclass(frozen=True)
class BandSpec:
    side: float
    width: int
    length: int
    theta: float = math.pi / 4
    offset: tuple = (0.0, 0.0)
    phase: tuple = (0.0, 0.0)
    # Inward shifts from the four respective box corners, in box axes.
    grid_phases: tuple = ((0., 0.),) * 4
    corner_depths: tuple = ()
    # Remove specified longitudinal cells at the two band ends.
    # Integers are counts trimmed from start/end of each transverse row.
    end_trims: tuple = ()
    # (u,v) translations of each longitudinal layer; permits split/gapped bands.
    layer_shifts: tuple = ()


def rectangle(spec):
    """A width x length lattice, with optional explicit per-row end caps."""
    c, s = math.cos(spec.theta), math.sin(spec.theta)
    u, v = np.array([c, s]), np.array([-s, c])
    origin = np.array([spec.side / 2] * 2) + np.array(spec.offset)
    origin += spec.phase[0] * u + spec.phase[1] * v
    out = []
    for i in range(spec.width):
        lo, hi = spec.end_trims[i] if spec.end_trims else (0, 0)
        for j in range(lo, spec.length - hi):
            xy = origin + (i - (spec.width - 1) / 2) * u
            xy = xy + (j - (spec.length - 1) / 2) * v
            if spec.layer_shifts:
                du, dv = spec.layer_shifts[j]
                xy = xy + float(du)*u + float(dv)*v
            out.append((*xy, spec.theta))
    return np.asarray(out, dtype=float).reshape(-1, 3)


def overlap_matrix(a, b, tol=1e-10):
    """Float64 screening SAT, not an interval certificate.

    True means all four separation axes overlap by more than tol.
    """
    a, b = np.asarray(a), np.asarray(b)
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)), dtype=bool)
    d = a[:, None, :2] - b[None, :, :2]
    ca, sa = np.cos(a[:, 2]), np.sin(a[:, 2])
    cb, sb = np.cos(b[:, 2]), np.sin(b[:, 2])
    ua = np.stack((ca, sa), -1)
    va = np.stack((-sa, ca), -1)
    ub = np.stack((cb, sb), -1)
    vb = np.stack((-sb, cb), -1)
    bad = np.ones((len(a), len(b)), dtype=bool)
    for axis in (ua[:, None], va[:, None], ub[None, :], vb[None, :]):
        radius_a = (np.abs((axis * ua[:, None]).sum(-1)) +
                    np.abs((axis * va[:, None]).sum(-1))) / 2
        radius_b = (np.abs((axis * ub[None, :]).sum(-1)) +
                    np.abs((axis * vb[None, :]).sum(-1))) / 2
        bad &= np.abs((d * axis).sum(-1)) < radius_a + radius_b - tol
    return bad


def triangle(side, corner, depth, phase=(0., 0.)):
    out = []
    for i in range(depth):
        for j in range(depth - i):
            x, y = i + .5 + phase[0], j + .5 + phase[1]
            out.append((side - x if corner[0] else x,
                        side - y if corner[1] else y, 0.))
    return np.asarray(out, dtype=float).reshape(-1, 3)


def inside(p, side, tol=1e-10):
    h = (np.abs(np.cos(p[:, 2])) + np.abs(np.sin(p[:, 2]))) / 2
    return bool(np.all(p[:, :2] >= h[:, None] - tol) and
                np.all(p[:, :2] <= side - h[:, None] + tol))


def generate(spec, discover=True):
    """Build one band and four explicit corner triangles.

    With no corner_depths, discover the largest complete triangle at each
    corner, in order SW,NE,NW,SE. Does not optimize coordinates or search.
    Non-triangular corner regions remain deliberately unfilled.
    """
    p = rectangle(spec)
    if not inside(p, spec.side):
        raise ValueError("band rectangle escapes box")
    depths = []
    for z, corner in enumerate(CORNERS):
        phase = spec.grid_phases[z]
        if spec.corner_depths:
            depth = spec.corner_depths[z]
        elif discover:
            # First rejected shell bounds the maximal complete triangle.
            depth = 0
            for k in range(1, math.floor(spec.side) + 1):
                shell = []
                for i in range(k):
                    x, y = i + .5 + phase[0], k - 1 - i + .5 + phase[1]
                    shell.append((spec.side-x if corner[0] else x,
                                  spec.side-y if corner[1] else y, 0.))
                q = np.asarray(shell)
                if not inside(q, spec.side) or overlap_matrix(q, p).any():
                    break
                depth = k
        else:
            raise ValueError("explicit topology required")
        q = triangle(spec.side, corner, depth, phase)
        if not inside(q, spec.side) or overlap_matrix(q, p).any():
            raise ValueError("infeasible triangle")
        depths.append(depth)
        p = np.concatenate((p, q))
    return p, replace(spec, corner_depths=tuple(depths))


def validate(p, side):
    """Strict-ish float screen. Official metric still required for export."""
    bad = overlap_matrix(p, p)
    np.fill_diagonal(bad, False)
    return dict(n=len(p), inside=inside(p, side),
                overlap_pairs=int(np.count_nonzero(bad) // 2))


def high_precision_rows(spec, dps=65, dilation="1.00000000001",
                        exact_side=None, exact_theta=None,
                        exact_offset=None, exact_layer_shifts=None):
    """Recompute analytic geometry at high precision; never roundtrip float p.

    Decimal strings are recommended for variable-angle parameters.
    For the Stenlund gate pass exact_side=3+4*mp.sqrt(2),
    exact_theta=mp.pi/4, exact_offset=(-mp.mpf('.25'),)*2.
    A small *center-only* dilation gives serialization clearance. This raises
    bounding side slightly; do not claim original ideal L for the CSV.
    """
    with mp.workdps(dps + 15):
        L = mp.mpf(str(spec.side)) if exact_side is None else mp.mpf(exact_side)
        t = mp.mpf(str(spec.theta)) if exact_theta is None else mp.mpf(exact_theta)
        off = spec.offset if exact_offset is None else exact_offset
        ox, oy = [mp.mpf(str(v)) for v in off]
        c, s = mp.cos(t), mp.sin(t)
        ph, pv = [mp.mpf(str(v)) for v in spec.phase]
        cx, cy = L / 2 + ox + ph*c-pv*s, L / 2+oy+ph*s+pv*c
        out = []
        for i in range(spec.width):
            lo, hi = spec.end_trims[i] if spec.end_trims else (0, 0)
            for j in range(lo, spec.length-hi):
                a, b = mp.mpf(i)-mp.mpf(spec.width-1)/2, mp.mpf(j)-mp.mpf(spec.length-1)/2
                shifts = spec.layer_shifts if exact_layer_shifts is None else exact_layer_shifts
                if shifts:
                    du,dv = [mp.mpf(str(v)) for v in shifts[j]]
                    a,b = a+du,b+dv
                out.append((cx+a*c-b*s, cy+a*s+b*c, t))
        if not spec.corner_depths:
            raise ValueError("discover topology first")
        for z, corner in enumerate(CORNERS):
            px, py = [mp.mpf(str(v)) for v in spec.grid_phases[z]]
            depth = spec.corner_depths[z]
            for i in range(depth):
                for j in range(depth-i):
                    x,y=mp.mpf(i)+mp.mpf('.5')+px,mp.mpf(j)+mp.mpf('.5')+py
                    out.append((L-x if corner[0] else x,
                                L-y if corner[1] else y,mp.mpf(0)))
        f = mp.mpf(dilation)
        return [dict(square_id=str(i), x=mp.nstr((x-L/2)*f, dps),
                     y=mp.nstr((y-L/2)*f, dps), angle=mp.nstr(t, dps))
                for i,(x,y,t) in enumerate(out)]


def write_csv(path, puzzles):
    with open(path, "w", newline="") as f:
        w=csv.DictWriter(f,fieldnames=("id","n","square_id","x","y","angle"))
        w.writeheader()
        idx=0
        for n,rows in sorted(puzzles.items()):
            for row in rows:
                w.writerow(dict(id=idx,n=n,**row))
                idx+=1


def enumerate_specs(side, widths=(2, 3), offsets=((-0.25,-0.25),(0.,0.),(.25,.25)),
                    phases=((0.,0.),), angles=(math.pi/4,),
                    length_adjustments=(0,-1), cap_modes=("full","staggered")):
    """Lazy parameter grid for an external GPU driver. Does not run a sweep.

    Length follows rectangle containment, not a hardcoded count. Phase is
    along band lattice u/v, offset is in container x/y. Cap trimming has
    separate per-row endpoints; staggered removes alternating end cells.
    """
    for width in widths:
        for theta in angles:
            c,s=abs(math.cos(theta)),abs(math.sin(theta))
            for offset in offsets:
                for phase in phases:
                    ox=offset[0]+phase[0]*math.cos(theta)-phase[1]*math.sin(theta)
                    oy=offset[1]+phase[0]*math.sin(theta)+phase[1]*math.cos(theta)
                    maxlen=math.floor(min((side-2*abs(ox)-width*c)/max(s,1e-15),
                                          (side-2*abs(oy)-width*s)/max(c,1e-15))+1e-10)
                    for adjustment in length_adjustments:
                        length=maxlen+adjustment
                        if length<1: continue
                        for cap in cap_modes:
                            trims=() if cap=="full" else tuple((i%2,(i+1)%2) for i in range(width))
                            yield BandSpec(side,width,length,theta,offset,phase,end_trims=trims)


def sat_penetrations(a, b):
    """Minimum translation SAT depth for each pair, zero if disjoint.

    Axes are unit length. Positive values mean real overlap, not validity.
    Shape len(a),len(b); intended for a small insertion-site pool.
    """
    a, b = np.asarray(a), np.asarray(b)
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    d = a[:, None, :2] - b[None, :, :2]
    ua = np.stack((np.cos(a[:, 2]), np.sin(a[:, 2])), -1)
    va = np.stack((-ua[:, 1], ua[:, 0]), -1)
    ub = np.stack((np.cos(b[:, 2]), np.sin(b[:, 2])), -1)
    vb = np.stack((-ub[:, 1], ub[:, 0]), -1)
    depth = np.full((len(a), len(b)), np.inf)
    for axis in (ua[:, None], va[:, None], ub[None], vb[None]):
        ra = (np.abs((axis*ua[:, None]).sum(-1))+
              np.abs((axis*va[:, None]).sum(-1)))/2
        rb = (np.abs((axis*ub[None]).sum(-1))+
              np.abs((axis*vb[None]).sum(-1)))/2
        depth = np.minimum(depth,ra+rb-np.abs((d*axis).sum(-1)))
    return np.maximum(depth,0.)


def insertion_sites(spec, shells=2):
    """Small deterministic endpoint/corner collar, NOT a box-wide search.

    Next complete-triangle shells expose gaps next to the existing upright
    regions. One cell collar around the finite tilted band exposes endpoint
    and side vacancies. Three local translations on each axis allow an
    inserted cell to straddle a small gap instead of duplicating a lattice
    center. Sites may overlap: the caller explicitly requests repair seeds.
    """
    if not spec.corner_depths:
        raise ValueError('generate explicit topology before augmentation')
    sites=[]
    for z,corner in enumerate(CORNERS):
        depth=spec.corner_depths[z]
        px,py=spec.grid_phases[z]
        for k in range(depth,depth+shells):
            for i in range(k+1):
                x,y=i+.5+px,k-i+.5+py
                sites.append((spec.side-x if corner[0] else x,
                              spec.side-y if corner[1] else y,0.))
    c,s=math.cos(spec.theta),math.sin(spec.theta)
    u,v=np.array([c,s]),np.array([-s,c])
    origin=np.array([spec.side/2]*2)+np.array(spec.offset)
    origin=origin+spec.phase[0]*u+spec.phase[1]*v
    for i in range(-1,spec.width+1):
        for j in range(-1,spec.length+1):
            if 0<=i<spec.width and 0<=j<spec.length:
                continue
            q=origin+(i-(spec.width-1)/2)*u+(j-(spec.length-1)/2)*v
            sites.extend([(*q,0.),(*q,spec.theta)])
    sites=np.asarray(sites,dtype=float)
    # A fixed local stencil, not an optimization sweep.
    local=[]
    for dx,dy in ((0.,0.),(.25,0.),(-.25,0.),(0.,.25),(0.,-.25)):
        q=sites.copy();q[:,:2]+=(dx,dy);local.append(q)
    candidates=np.concatenate(local)
    h=(np.abs(np.cos(candidates[:,2]))+np.abs(np.sin(candidates[:,2])))/2
    mask=((candidates[:,:2]>=h[:,None]-1e-10)&
          (candidates[:,:2]<=spec.side-h[:,None]+1e-10)).all(axis=1)
    candidates=candidates[mask]
    # Dedup generated sites only, never perturb an existing square.
    _,idx=np.unique(np.round(candidates,11),axis=0,return_index=True)
    return candidates[np.sort(idx)]


def augment_to_target(points, spec, target_n, max_added=8, shells=2,
                      max_depth=None, chunk_size=192):
    """Return an actual target-count seed plus explicit repair diagnostics.

    This is intentionally allowed to create OVERLAPPING seeds. Each added
    unit square is inside the box; it is selected by low SAT penetration
    against all old and already-added cells. Original cells never move.
    The return must pass GPU repair and official certification before use.
    Never label this result as a valid packing merely because len==target_n.

    max_depth optionally refuses the best remaining site if it exceeds an
    overlap-depth limit. There is no hidden fallback to baseline coordinates.
    If overfull, this function refuses to make an arbitrary deletion.
    """
    p=np.asarray(points,dtype=float).copy()
    if p.ndim!=2 or p.shape[1]!=3 or not np.isfinite(p).all():
        raise ValueError('expected finite (count,3) points')
    if target_n<len(p):
        raise ValueError('overfull seed: deletion belongs to caller')
    deficit=target_n-len(p)
    if deficit>max_added:
        raise ValueError('insertion deficit exceeds bounded local repair budget')
    candidates=insertion_sites(spec,shells) if deficit else np.empty((0,3))
    added=[]
    for step in range(deficit):
        best=None
        for start in range(0,len(candidates),chunk_size):
            q=candidates[start:start+chunk_size]
            # Avoid duplicate/near-duplicate centers even if orientation differs.
            dist2=((q[:,None,:2]-p[None,:,:2])**2).sum(-1).min(axis=1)
            sums,maxima,counts=_local_penetration_costs(q,p)
            energy=sums+maxima**2+.001*counts
            energy[dist2<.6**2]=np.inf
            k=int(np.argmin(energy))
            record=(float(energy[k]),start+k,float(maxima[k]),int(counts[k]))
            if best is None or record<best:
                best=record
        if best is None or not np.isfinite(best[0]):
            raise ValueError('no distinct bounded insertion site remains')
        energy,k,depth,count=best
        if max_depth is not None and depth>max_depth:
            raise ValueError('best local insertion exceeds overlap-depth budget')
        row=candidates[k].copy()
        added.append(dict(index=len(p),point=row.tolist(),energy=energy,
                          max_penetration=depth,conflicting_pairs=count))
        p=np.concatenate((p,row[None]))
        candidates=np.delete(candidates,k,axis=0)
    return p,dict(target_n=target_n,initial_count=target_n-deficit,
                  count=len(p),added_count=deficit,requires_repair=any(
                      q['max_penetration']>1e-9 for q in added),
                  certified=False,insertions=added,
                  note='Target-count repair seed, not a certified packing.')


def _paired_penetrations(a,b):
    """SAT for matching pairs; linear memory, unlike all-pairs screening."""
    a,b=np.asarray(a),np.asarray(b)
    ua=np.stack((np.cos(a[:,2]),np.sin(a[:,2])),-1)
    va=np.stack((-ua[:,1],ua[:,0]),-1)
    ub=np.stack((np.cos(b[:,2]),np.sin(b[:,2])),-1)
    vb=np.stack((-ub[:,1],ub[:,0]),-1)
    depth=np.full(len(a),np.inf)
    for axis in (ua,va,ub,vb):
        r=(np.abs((axis*ua).sum(-1))+np.abs((axis*va).sum(-1))+
           np.abs((axis*ub).sum(-1))+np.abs((axis*vb).sum(-1)))/2
        depth=np.minimum(depth,r-np.abs(((a[:,:2]-b[:,:2])*axis).sum(-1)))
    return np.maximum(depth,0.)


def greedy_fill_upright(points,spec,target_n):
    """Fill true nontriangular vacancies on all four corner-grid phases.

    Linear-neighborhood SAT via an integer spatial hash, not an N^2 matrix.
    Original cells never move. It returns at most target_n actual cells and
    refuses positive penetration greater than 1e-10. This is a float screen,
    not an official certificate. No GPU/optimizer is invoked here.
    """
    p=np.asarray(points,dtype=float)
    if target_n<len(p):
        raise ValueError('overfull seed')
    if len(p)==target_n:
        return p.copy(),[]
    sites=[]
    for z,(cx,cy) in enumerate(CORNERS):
        px,py=spec.grid_phases[z]
        for i in range(math.floor(spec.side)+1):
            x=i+.5+px
            if not .5-1e-10<=x<=spec.side-.5+1e-10:continue
            for j in range(math.floor(spec.side)+1):
                y=j+.5+py
                if .5-1e-10<=y<=spec.side-.5+1e-10:
                    sites.append((spec.side-x if cx else x,
                                  spec.side-y if cy else y,0.))
    sites=np.asarray(sites)
    _,idx=np.unique(np.round(sites,11),axis=0,return_index=True)
    sites=sites[np.sort(idx)]
    buckets={}
    for i,row in enumerate(p):
        key=tuple(np.floor(row[:2]).astype(int))
        buckets.setdefault(key,[]).append(i)
    ai=[];bi=[]
    for k,row in enumerate(sites):
        x,y=np.floor(row[:2]).astype(int)
        for dx in range(-2,3):
            for dy in range(-2,3):
                nearby=buckets.get((x+dx,y+dy),())
                ai.extend([k]*len(nearby));bi.extend(nearby)
    rejected=np.zeros(len(sites),dtype=bool)
    if ai:
        ai=np.asarray(ai);bi=np.asarray(bi)
        depth=_paired_penetrations(sites[ai],p[bi])
        rejected[ai[depth>1e-10]]=True
    inserted=[]
    for row in sites[~rejected]:
        if inserted and overlap_matrix(row[None],np.asarray(inserted)).any():
            continue
        inserted.append(row)
        if len(p)+len(inserted)>=target_n:break
    return (np.concatenate((p,np.asarray(inserted).reshape(-1,3))),
            [row.tolist() for row in inserted])


_augment_low_conflict=augment_to_target


def augment_to_target(points,spec,target_n,max_added=8,shells=2,
                      max_depth=None,chunk_size=192):
    """Greedy VALID upright gap fill first, bounded repair insertion second.

    See requires_repair in returned metadata. False means float-screen valid,
    not official certified. True must pass A100 repair before certification.
    """
    initial=len(points)
    p,feasible=greedy_fill_upright(points,spec,target_n)
    p,meta=_augment_low_conflict(p,spec,target_n,max_added,shells,
                                 max_depth,chunk_size)
    meta.update(initial_count=initial,feasible_added_count=len(feasible),
                feasible_insertions=feasible,repair_added_count=meta['added_count'],
                added_count=target_n-initial)
    return p,meta


def _local_penetration_costs(q,p):
    buckets={}
    for i,row in enumerate(p):
        key=tuple(np.floor(row[:2]).astype(int))
        buckets.setdefault(key,[]).append(i)
    ai=[];bi=[]
    for k,row in enumerate(q):
        x,y=np.floor(row[:2]).astype(int)
        for dx in range(-2,3):
            for dy in range(-2,3):
                near=buckets.get((x+dx,y+dy),())
                ai.extend([k]*len(near));bi.extend(near)
    sums=np.zeros(len(q));maxima=np.zeros(len(q));counts=np.zeros(len(q),dtype=int)
    if ai:
        ai=np.asarray(ai);bi=np.asarray(bi)
        d=_paired_penetrations(q[ai],p[bi])
        sums=np.bincount(ai,weights=d*d,minlength=len(q))
        counts=np.bincount(ai[d>1e-9],minlength=len(q))
        np.maximum.at(maxima,ai,d)
    return sums,maxima,counts
