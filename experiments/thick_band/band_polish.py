"""Local unit-square LP polish. q[:, :] = (center x, center y, radians).
Numerical SAT is a screening check, NOT the official interval certificate.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from itertools import product
import math
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree

SIGNS = tuple(product((-1., 1.), repeat=2))

def _q(q):
    q = np.asarray(q, dtype=float)
    if q.ndim != 2 or q.shape[1] != 3 or len(q) == 0 or not np.isfinite(q).all():
        raise ValueError("q must be a nonempty finite (n,3) array with radians")
    return q

def extent(q):
    q = _q(q)
    h = .5*(np.abs(np.cos(q[:,2]))+np.abs(np.sin(q[:,2])))
    return float(np.max(np.max(q[:,:2]+h[:,None], axis=0) -
                        np.min(q[:,:2]-h[:,None], axis=0)))

def sat_check(q, clearance=0.):
    """Full all-pairs 4-axis disjointness, long-double arithmetic, no overlap tolerance."""
    q = _q(q).astype(np.longdouble)
    c, s = np.cos(q[:,2]), np.sin(q[:,2])
    u = np.column_stack((c,s)); v = np.column_stack((-s,c))
    minimum = np.longdouble(np.inf)
    for i in range(len(q)-1):
        d = q[i+1:,:2]-q[i,:2]
        best = np.full(len(d), -np.inf, dtype=np.longdouble)
        for owner_axes in (False, True):
            for basis in (u,v):
                a = basis[i+1:] if owner_axes else np.broadcast_to(basis[i],d.shape)
                hi = .5*(np.abs(a@u[i])+np.abs(a@v[i]))
                hj = .5*(np.abs(np.sum(a*u[i+1:],axis=1))+
                         np.abs(np.sum(a*v[i+1:],axis=1)))
                g = np.abs(np.sum(d*a,axis=1))-hi-hj
                best = np.maximum(best,g)
        minimum = min(minimum,np.min(best))
        if np.any(best < clearance):
            return False, float(minimum)
    return True, float(minimum)

def chosen_axis(q, i, j, previous=None, hysteresis=1e-9):
    """Pick the strongest SAT separation; retain previous near ties."""
    d = q[j,:2]-q[i,:2]
    choices = []
    for owner in (i,j):
        for k in (0,1):
            phi = q[owner,2]+k*math.pi/2
            a = np.array([math.cos(phi),math.sin(phi)])
            direction = 1 if d@a >= 0 else -1
            a *= direction
            h = sum(.5*(abs(math.cos(q[t,2]-phi))+
                          abs(math.sin(q[t,2]-phi))) for t in (i,j))
            choices.append((float(d@a-h),(owner,k,direction)))
    best = max(choices,key=lambda x:x[0])
    if previous is not None:
        for value,key in choices:
            if key == previous and value >= best[0]-hysteresis:
                return key
    return best[1]

def piece_gap_gradient(q, i, j, axis, signs):
    """Smooth support sign piece and its derivative wrt (xi,yi,ti,xj,yj,tj)."""
    owner,k,direction = axis
    phi = q[owner,2]+k*math.pi/2
    a = direction*np.array([math.cos(phi),math.sin(phi)])
    ap = np.array([-a[1],a[0]])
    # Directed axis reversal does not affect support but changes relative angle.
    phi += 0 if direction == 1 else math.pi
    other = j if owner == i else i
    r = q[other,2]-phi
    s1,s2 = signs
    hp = .5*(-s1*math.sin(r)+s2*math.cos(r))
    d = q[j,:2]-q[i,:2]
    g = float(d@a-.5-.5*(s1*math.cos(r)+s2*math.sin(r)))
    grad = np.zeros(6)
    grad[:2] = -a; grad[3:5] = a
    grad[2 if owner == i else 5] = d@ap+hp
    grad[2 if other == i else 5] = -hp
    return g,grad

@dataclass
class Result:
    q: np.ndarray
    L: float
    history: list = field(default_factory=list)
    accepted: int = 0

def _lp(q, upper, position_trust, angle_trust, clearance, axes):
    n = len(q); free = angle_trust > 0
    stride = 3 if free else 2
    nv = stride*n+3
    rr=[]; cc=[]; vv=[]; rhs=[]
    def row(coeff, b):
        r=len(rhs); rhs.append(float(b))
        for col,val in coeff.items():
            if val:
                rr.append(r);cc.append(col);vv.append(float(val))
    # Box support is max of four smooth sign pieces. All four must hold,
    # including inactive pieces which may become active across an abs kink.
    for i in range(n):
        t=q[i,2]; c=math.cos(t); s=math.sin(t)
        pieces = [(.5*(a*c+b*s),.5*(-a*s+b*c)) for a,b in SIGNS] if free else [
            (.5*(abs(c)+abs(s)),0.)]
        for h,dh in pieces:
            for dim in (0,1):
                lower={stride*i+dim:-1.,stride*n+dim:1.}
                upperrow={stride*i+dim:1.,stride*n+dim:-1.,nv-1:-1.}
                if free:
                    lower[stride*i+2]=dh; upperrow[stride*i+2]=dh
                row(lower,q[i,dim]-h-clearance)
                row(upperrow,-q[i,dim]-h-clearance)
    # A square is contained in its circumcircle for every angle. Pairs farther
    # than this radius cannot collide within the position trust box.
    radius=math.sqrt(2)+2*math.sqrt(2)*position_trust+1e-10
    pairs=sorted(cKDTree(q[:,:2]).query_pairs(radius))
    for i,j in pairs:
        key=(i,j)
        axis=chosen_axis(q,i,j,axes.get(key))
        axes[key]=axis
        if free:
            for signs in SIGNS:
                g,grad=piece_gap_gradient(q,i,j,axis,signs)
                coeff={}
                for k in range(3):
                    coeff[stride*i+k]=-grad[k]
                    coeff[stride*j+k]=-grad[3+k]
                row(coeff,g-clearance)
        else:
            owner,k,direction=axis
            phi=q[owner,2]+k*math.pi/2
            a=direction*np.array([math.cos(phi),math.sin(phi)])
            h=sum(.5*(abs(math.cos(q[t,2]-phi))+abs(math.sin(q[t,2]-phi))) for t in (i,j))
            row({stride*i:a[0],stride*i+1:a[1],
                 stride*j:-a[0],stride*j+1:-a[1]},
                float((q[j,:2]-q[i,:2])@a-h-clearance))
    bounds=[]
    for _ in range(n):
        bounds.extend([(-position_trust,position_trust)]*2)
        if free: bounds.append((-angle_trust,angle_trust))
    bounds.extend([(None,None),(None,None),(1.,upper)])
    objective=np.zeros(nv);objective[-1]=1.
    A=coo_matrix((vv,(rr,cc)),shape=(len(rhs),nv)).tocsr()
    res=linprog(objective,A_ub=A,b_ub=rhs,bounds=bounds,method="highs",
                options={"primal_feasibility_tolerance":1e-9,
                         "dual_feasibility_tolerance":1e-9})
    if not res.success or not np.isfinite(res.x).all():
        return None,str(res.message)
    out=q.copy()
    out[:,:2]+=res.x[:stride*n].reshape(n,stride)[:,:2]
    if free: out[:,2]+=res.x[:stride*n].reshape(n,stride)[:,2]
    return out,str(res.message)

def acceptable(candidate, best_L, clearance=0., validator=None):
    """Reject overlaps, nonfinite data, worsening, or external certificate failure."""
    try:
        valid,_=sat_check(candidate,clearance)
        if not valid or extent(candidate)>best_L: return False
        return validator is None or bool(validator(candidate))
    except (ValueError, FloatingPointError):
        return False

def polish(q, *, free_angles=False, iterations=20, position_trust=.15,
           angle_trust=.025, clearance=2e-9, backtracks=10, validator=None):
    """Alternate linear angle proposals and exact fixed-angle center LP repair.

    validator(q)->bool can enforce the project's interval certificate per update.
    Without it, validity is numerical all-pairs SAT only, not submission authority.
    Invalid input is refused, not repaired silently.
    """
    q=_q(q).copy()
    if position_trust<=0 or angle_trust<0 or clearance<0 or iterations<0:
        raise ValueError("invalid trust, clearance or iteration count")
    if not sat_check(q)[0] or (validator is not None and not validator(q)):
        raise ValueError("initial packing is invalid")
    best=Result(q.copy(),extent(q)); axes={}
    for iteration in range(iterations):
        old=best.L
        # Fixed angles first, an exact LP in the chosen local SAT topology.
        fixed,status=_lp(best.q,old,position_trust,0.,clearance,axes)
        if fixed is not None and acceptable(fixed,old,0.,validator):
            best.q=fixed;best.L=extent(fixed);best.accepted+=1
        if free_angles:
            proposal,status=_lp(best.q,best.L,position_trust,angle_trust,clearance,axes)
            if proposal is not None:
                origin=best.q.copy()
                # Never accept a linearized angle step directly. Repair centers
                # at the proposed angles with exact supports, then check ALL pairs.
                for bt in range(backtracks+1):
                    alpha=2.**(-bt)
                    seed=origin+alpha*(proposal-origin)
                    repaired,_=_lp(seed,best.L,position_trust,0.,clearance,{})
                    if repaired is not None and acceptable(repaired,best.L,0.,validator):
                        best.q=repaired;best.L=extent(repaired);best.accepted+=1
                        break
        best.history.append(dict(iteration=iteration,L=best.L,status=status))
        if old-best.L < 1e-11:
            break
    return best

def fixed_angle_polish(q, **kwargs):
    return polish(q,free_angles=False,**kwargs)

def free_angle_polish(q, **kwargs):
    return polish(q,free_angles=True,**kwargs)
