"""Bounded host LP recovery. Never replaces the preserved search incumbent.

State remains contiguous float64 (n,3). Angle proposals are linearized; every
accepted state has exact-angle center repair and an independent full SAT screen.
LP is host orchestration, not a candidate custom CUDA kernel.
"""
import time
import numpy as np
from band_polish import _lp,extent,sat_check

def recover(seed, *, seconds=20., iterations=12, position_trust=.005,
            angle_trust=.001, clearance=2e-8, target=None):
    q=np.ascontiguousarray(seed,dtype=np.float64).copy()
    end=time.monotonic()+seconds; diagnostics=[];history=[]
    best=None;side=float("inf");radius=angle_trust
    def solve(state,upper,angles):
        remaining=end-time.monotonic()
        if remaining<=0:return None
        result,_=_lp(state,upper,position_trust,angles,clearance,{},
             time_limit=min(5.,remaining),method="highs-ipm" if angles else "highs",diagnostics=diagnostics)
        return result
    # Input can overlap. Obtain a feasible working state, not an incumbent.
    q0=solve(q,extent(q)+4*position_trust,0.)
    if q0 is not None and sat_check(q0)[0]:
        best=q0.copy();side=extent(q0);q=q0
    else:
        return dict(x=None,L=None,valid=False,history=history,diagnostics=diagnostics)
    for it in range(iterations):
        if time.monotonic()>=end or (target is not None and side<=target):break
        previous=side
        # Frozen-angle exact LP first.
        fixed=solve(q,side+1e-10,0.)
        if fixed is not None and sat_check(fixed)[0] and extent(fixed)<side:
            q=fixed;best=q.copy();side=extent(q)
        proposal=solve(q,side+1e-10,radius)
        accepted=False
        if proposal is not None:
            for bt in range(4):
                if time.monotonic()>=end:break
                trial=q+(proposal-q)*(2.**(-bt))
                exact=solve(trial,side+1e-10,0.)
                if exact is not None and sat_check(exact)[0] and extent(exact)<side:
                    q=exact;side=extent(q);best=q.copy();accepted=True;break
        history.append(dict(iteration=it,L=side,angle_trust=radius,accepted=accepted))
        radius=min(angle_trust,radius*1.5) if accepted else radius*.5
        if previous-side<1e-11 and radius<1e-7:break
    return dict(x=best,L=side,valid=best is not None,history=history,diagnostics=diagnostics)
