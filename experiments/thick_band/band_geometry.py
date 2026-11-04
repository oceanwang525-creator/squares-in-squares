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
