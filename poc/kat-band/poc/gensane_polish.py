"""Fixed-box contact quench, not event-driven billiards.

The neighbor mask is rebuilt on EVERY closure from circumscribed circles.
Hence even unconstrained LBFGS trial steps cannot silently lose collisions.
External interval certification remains mandatory for exported candidates.
"""
from __future__ import annotations
import math
import time
import torch
from band_relax import DTYPE, pair_penetration, half_extent, canonicalize, expand_to_separate, Config

def collision_loss(x, lengths, clearance=1e-11):
    # Selection detached, but all possibly intersecting pairs are included.
    with torch.no_grad():
        d = torch.cdist(x[..., :2].detach(), x[..., :2].detach())
        mask = torch.triu(d <= math.sqrt(2)+1e-8, diagonal=1)
        ids = mask.nonzero()
    p = pair_penetration(x[ids[:,0],ids[:,1]], x[ids[:,0],ids[:,2]])
    h = half_extent(x)[..., None] + clearance
    wall = torch.relu(h-x[..., :2]).square()+torch.relu(x[..., :2]+h-lengths[:,None,None]).square()
    return torch.relu(p+clearance).square().sum()+10*wall.sum()

def quench(x0, lengths, *, steps=200, seconds=30., clearance=1e-11, lr=1.0):
    """One independent coordinate block per lane, shared LBFGS line search."""
    x=x0.detach().clone().requires_grad_(True)
    lengths=torch.as_tensor(lengths,dtype=DTYPE,device=x.device).expand(len(x))
    end=time.monotonic()+seconds
    opt=torch.optim.LBFGS([x], lr=lr, max_iter=steps, max_eval=steps*2,
        tolerance_grad=1e-13, tolerance_change=1e-20,
        history_size=20, line_search_fn="strong_wolfe")
    evaluations=0
    class Deadline(Exception): pass
    best_loss=float("inf")
    best=x.detach().clone()
    def closure():
        nonlocal evaluations,best_loss,best
        if time.monotonic()>end: raise Deadline()
        opt.zero_grad(set_to_none=True)
        loss=collision_loss(x,lengths,clearance)
        if not torch.isfinite(loss): raise FloatingPointError("nonfinite quench")
        loss.backward()
        evaluations+=1
        value=float(loss.detach())
        if value<best_loss:
            best_loss=value;best=x.detach().clone()
        return loss
    try:opt.step(closure)
    except Deadline:pass
    return best,dict(evaluations=evaluations,loss=best_loss)

def unit_feasible(x, *, clearance=1e-10, max_expand=100):
    return expand_to_separate(x, Config(clearance=clearance,max_expand=max_expand,tile=256,batch_chunk=8))
