"""Gensane-inspired global perturb/inflate and sequential feasible random walks.

Not event-driven billiards. Float64 SAT screening is not a geometric proof.
All coordinates have unit square side at API boundaries; box origin is zero.
"""
from dataclasses import dataclass, asdict
import math
import time
import numpy as np
import torch
try:
    from .band_relax import (Config as GateConfig, all_pair_tiles, pair_penetration,
                             half_extent, validate, boundary_loss)
except ImportError:
    from band_relax import (Config as GateConfig, all_pair_tiles, pair_penetration,
                            half_extent, validate, boundary_loss)

DTYPE = torch.float64

@dataclass
class Config:
    seconds: float = 60.
    seed: int = 42
    max_rounds: int = 100000
    walks_per_round: int = 0  # zero selects five full N-square sweeps
    inflate_every: int = 512
    restart_every: int = 8  # zero disables best-state restarts
    global_xy: float = .015
    global_angle: float = .015
    walk_xy: float = .04
    walk_angle: float = .04
    checkpoint_every: float = 15.
    tile: int = 128
    batch_chunk: int = 4
    polish_steps: int = 0
    polish_shrink: float = .0001


def gate_config(cfg=None):
    cfg = cfg or Config()
    return GateConfig(tile=cfg.tile, batch_chunk=cfg.batch_chunk, overlap_tol=0.)

@torch.no_grad()
def maximal_side(x, lengths, cfg=None):
    """Largest common physical square side s in this FIXED box.

    For each pair, disjointness means s <= max_k |d.axis_k|/r,
    where r=(1+|cos da|+|sin da|)/2. Walls give s<=d_wall/h.
    Thus take the minimum of these bounds, not capped at one. Duplicate centers
    give zero and cannot be recovered by uniform inflation/shrink alone.
    """
    h = half_extent(x)
    distances = torch.minimum(x[..., :2], lengths[:, None, None]-x[..., :2]).amin(-1)
    side = (distances/h).amin(-1)
    for b0,b1,i0,i1,j0,j1 in all_pair_tiles(x,gate_config(cfg)):
        a=x[b0:b1,i0:i1,None,:]; b=x[b0:b1,None,j0:j1,:]
        dt=a[...,2]-b[...,2]
        r=(1+dt.cos().abs()+dt.sin().abs())*.5
        ratio=(r-pair_penetration(a,b))/r
        if i0 == j0:
            mask=torch.arange(i0,i1,device=x.device)[:,None]>=torch.arange(j0,j1,device=x.device)[None,:]
            ratio.masked_fill_(mask[None],torch.inf)
        side[b0:b1]=torch.minimum(side[b0:b1],ratio.amin((1,2)))
    finite=torch.isfinite(x).all((1,2)) & torch.isfinite(lengths) & (lengths>0)
    return torch.where(finite,side,torch.zeros_like(side))

@torch.no_grad()
def inflate(x, lengths, cfg=None):
    """Normalize maximal feasible common side to one; includes strict recheck."""
    s=maximal_side(x,lengths,cfg)
    usable=torch.isfinite(s)&(s>0)
    # Round size downward to protect touching pairs from normalization error.
    safe=torch.where(usable,s*(1-256*torch.finfo(DTYPE).eps),torch.ones_like(s))
    out=x.clone(); out[...,:2]/=safe[:,None,None]
    sides=lengths/safe
    ok,_,_=validate(out,sides,gate_config(cfg))
    return out,sides,usable&ok

@torch.no_grad()
def global_perturb(x, generator, xy, angle):
    """Independent proposals for ALL centers and ALL orientations."""
    noise=torch.randn(x.shape,device=x.device,dtype=DTYPE,generator=generator)
    scale=torch.as_tensor([xy,xy,angle],dtype=DTYPE,device=x.device)
    return x+noise*scale

@torch.no_grad()
def walk_step(x, lengths, generator, xy, angle):
    """One square per independent lane; O(B*N), no conflicting updates.

    Input lanes must be feasible. Failed proposals leave the entire lane intact.
    """
    batch,n,_=x.shape
    lanes=torch.arange(batch,device=x.device)
    ids=torch.randint(n,(batch,),device=x.device,generator=generator)
    noise=torch.randn((batch,3),dtype=DTYPE,device=x.device,generator=generator)
    noise[:,:2]*=xy; noise[:,2:]*=angle
    proposal=x[lanes,ids]+noise
    overlap=pair_penetration(proposal[:,None,:],x)
    overlap[lanes,ids]=-torch.inf
    h=half_extent(proposal)[:,None]
    ok=(overlap<=0).all(-1)&(proposal[:,:2]>=h).all(-1)&(proposal[:,:2]+h<=lengths[:,None]).all(-1)&torch.isfinite(proposal).all(-1)
    out=x.clone(); out[lanes[ok],ids[ok]]=proposal[ok]
    return out,ok


def polish(x, lengths, cfg, deadline):
    """Optional dense all-pair LBFGS fixed-side repair, NOT billiard dynamics.

    No neighbor list can go stale. Memory is quadratic; disabled by default.
    A controller must measure usefulness before enabling on large batches.
    """
    target=lengths*(1-cfg.polish_shrink)
    z=x.detach().clone(); z[...,:2]*=(target/lengths)[:,None,None]
    z.requires_grad_(True)
    optimizer=torch.optim.LBFGS([z],lr=.5,max_iter=1,history_size=5)
    n=x.shape[1]; ids=torch.triu_indices(n,n,1,device=x.device)
    def closure():
        optimizer.zero_grad()
        p=pair_penetration(z[:,ids[0]],z[:,ids[1]])
        loss=torch.relu(p+1e-10).square().sum()+10*boundary_loss(z,target,1e-10).sum()
        loss.backward()
        return loss
    for _ in range(cfg.polish_steps):
        if time.monotonic()>=deadline: break
        optimizer.step(closure)
    return inflate(z.detach(),target,cfg)


def relax(seeds, lengths, *, device='cpu', config=None, checkpoint=None, deadline=None, polish_hook=None):
    """Independent lanes, immutable retain-best; soft monotonic deadline.

    Checkpoint dictionaries own their numpy buffers. Invalid lanes retain L=inf.
    One all-pair sweep/optimizer step may overrun the deadline.
    """
    cfg=config or Config()
    if cfg.restart_every<0 or cfg.seconds<=0 or cfg.walks_per_round<0 or cfg.inflate_every<1 or cfg.max_rounds<0 or cfg.tile<1 or cfg.batch_chunk<1 or cfg.polish_steps<0:
        raise ValueError('invalid budget or counts')
    if not 0<=cfg.polish_shrink<1 or min(cfg.global_xy,cfg.global_angle,cfg.walk_xy,cfg.walk_angle)<0:
        raise ValueError('invalid amplitudes')
    x=torch.as_tensor(seeds,dtype=DTYPE,device=device).clone()
    if x.ndim!=3 or x.shape[-1]!=3 or min(x.shape[:2])<1 or not bool(torch.isfinite(x).all()):
        raise ValueError('finite B,N,3 seeds required')
    lens=torch.as_tensor(lengths,dtype=DTYPE,device=device).expand(len(x)).clone()
    if not bool((torch.isfinite(lens)&(lens>0)).all()): raise ValueError('positive finite box lengths required')
    gen=torch.Generator(device=x.device).manual_seed(cfg.seed)
    start=time.monotonic(); end=min(start+cfg.seconds,deadline if deadline is not None else math.inf)
    valid,_,_=validate(x,lens,gate_config(cfg))
    best=x.clone(); best_l=torch.where(valid,lens,torch.full_like(lens,torch.inf))
    rounds=walks=accepted=0; last=start
    amplitudes=torch.ones(len(x),dtype=DTYPE,device=x.device)
    walk_amplitudes=amplitudes.clone()
    walk_count=cfg.walks_per_round or 5*x.shape[1]
    def retain(y, sides, usable, checked=False):
        nonlocal best,best_l
        strict = usable if checked else validate(y,sides,gate_config(cfg))[0]
        improve=usable&strict&(sides<best_l)
        best[improve]=y[improve].clone(); best_l[improve]=sides[improve].clone()
    def result():
        current_ok=validate(x,lens,gate_config(cfg))[0]
        return dict(current_x=x.detach().cpu().numpy().copy(),current_L=lens.detach().cpu().numpy().copy(),current_float_valid=current_ok.cpu().numpy().copy(),x=best.detach().cpu().numpy().copy(),L=best_l.detach().cpu().numpy().copy(),float_valid=torch.isfinite(best_l).cpu().numpy().copy(),rounds=rounds,steps=walks,accepted=accepted,elapsed=time.monotonic()-start,config=asdict(cfg),certificate='float64 strict all-pair screen only; official metric required')
    if checkpoint: checkpoint(result())
    while rounds<cfg.max_rounds and time.monotonic()<end:
        # All-square global perturb, then shrink OR inflate in the unchanged box.
        old_best=best_l.clone()
        restart=rounds==0 or (cfg.restart_every>0 and rounds%cfg.restart_every==0)
        base=torch.where(torch.isfinite(best_l)[:,None,None],best,x) if restart else x
        box=torch.where(torch.isfinite(best_l),best_l,lens) if restart else lens
        base_box=box.clone()
        # Log-uniform per-lane amplitudes span two decades, plus adaptation.
        amp=amplitudes*torch.exp(torch.rand((len(x),),dtype=DTYPE,device=x.device,generator=gen)*math.log(100)-math.log(10))
        noise=global_perturb(base,gen,cfg.global_xy,cfg.global_angle)-base
        trial,box,ok=inflate(base+noise*amp[:,None,None],box,cfg)
        # Failed lanes revert, including exact duplicate-center perturb failures.
        fallback=base_box
        x=torch.where(ok[:,None,None],trial,base).clone(); lens=torch.where(ok,box,fallback)
        active,_,_=validate(x,lens,gate_config(cfg))
        retain(x,lens,active,checked=True)
        if polish_hook is not None and time.monotonic()<end:
            # Hook receives owned buffers; it may return a repaired proposal.
            # Contract is (x, lengths, best_lengths, deadline) -> (x, lengths).
            hy,hs=polish_hook(x.clone(),lens.clone(),best_l.clone(),end)
            hy=torch.as_tensor(hy,dtype=DTYPE,device=x.device)
            hs=torch.as_tensor(hs,dtype=DTYPE,device=x.device)
            if hy.shape!=x.shape or hs.shape!=lens.shape: raise ValueError("polish hook shape mismatch")
            hok=validate(hy,hs,gate_config(cfg))[0]
            x=torch.where(hok[:,None,None],hy,x); lens=torch.where(hok,hs,lens)
            active=validate(x,lens,gate_config(cfg))[0]
            retain(x,lens,active,checked=True)
        lane_accept=torch.zeros(len(x),dtype=DTYPE,device=x.device)
        for step in range(walk_count):
            if time.monotonic()>=end: break
            proposal,take=walk_step(x,lens,gen,cfg.walk_xy*walk_amplitudes[:,None],cfg.walk_angle*walk_amplitudes[:,None])
            take &= active
            x=torch.where(take[:,None,None],proposal,x)
            lane_accept+=take; accepted+=int(take.sum()); walks+=1
            if checkpoint and time.monotonic()-last>=cfg.checkpoint_every:
                retain(x,lens,active)
                checkpoint(result()); last=time.monotonic()
            if (step+1)%cfg.inflate_every==0:
                y,s,good=inflate(x,lens,cfg)
                good &= active
                x=torch.where(good[:,None,None],y,x); lens=torch.where(good,s,lens)
                retain(x,lens,active,checked=True)
        y,s,good=inflate(x,lens,cfg)
        x=torch.where(good[:,None,None],y,x); lens=torch.where(good,s,lens)
        retain(y,s,good,checked=True)
        if cfg.polish_steps and time.monotonic()<end:
            y,s,good=polish(x,lens,cfg,end); retain(y,s,good,checked=True)
        # Global scale follows strict best improvement, walk scale follows acceptance.
        amplitudes=(amplitudes*torch.where(best_l<old_best,1.1,.8)).clamp(.05,20)
        walk_amplitudes=(walk_amplitudes*torch.where(lane_accept>walk_count*.2,1.1,.8)).clamp(.05,20)
        rounds+=1
        if checkpoint and time.monotonic()-last>=cfg.checkpoint_every:
            checkpoint(result()); last=time.monotonic()
    out=result()
    if checkpoint: checkpoint(out)
    return out
