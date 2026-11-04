"""Float64 unit-square packing heuristic. Angles are radians.

Internal float64 all-pair SAT checks are NOT interval certificates. Every saved
candidate must pass the external official metric after serialization.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import math
import time
from typing import Callable
import numpy as np
import torch

DTYPE = torch.float64

@dataclass
class Config:
    seconds: float = 60.0
    seed: int = 42
    repair_steps: int = 180
    lr: float = 0.006
    shrink: float = 0.001
    perturb_xy: float = 0.012
    perturb_angle: float = 0.012
    collective_xy: float = 0.025
    collective_angle: float = 0.012
    temperature: float = 0.015
    skin: float = 0.6
    rebuild_every: int = 20
    pair_chunk: int = 32768
    tile: int = 128
    batch_chunk: int = 4
    max_step: float = 0.02
    overlap_tol: float = 0.0
    clearance: float = 1e-9
    max_expand: float = 1.025
    checkpoint_every: float = 15.0
    max_rounds: int = 1000000

def pair_penetration(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Signed minimum overlap on the four exact SAT axes, positive = overlap.

    All squares have unit side, so their summed projection radii are equal on
    the four axes: (1 + |cos(dt)| + |sin(dt)|)/2.
    No circular or AABB approximation is used for the actual collision test.
    """
    d = a[..., :2] - b[..., :2]
    ca, sa = a[..., 2].cos(), a[..., 2].sin()
    cb, sb = b[..., 2].cos(), b[..., 2].sin()
    projections = torch.stack((
        (d[..., 0]*ca + d[..., 1]*sa).abs(),
        (-d[..., 0]*sa + d[..., 1]*ca).abs(),
        (d[..., 0]*cb + d[..., 1]*sb).abs(),
        (-d[..., 0]*sb + d[..., 1]*cb).abs()), dim=-1)
    dt = a[..., 2] - b[..., 2]
    radius = (1 + dt.cos().abs() + dt.sin().abs()) * 0.5
    return radius - projections.amax(dim=-1)

def half_extent(x):
    return (x[..., 2].cos().abs() + x[..., 2].sin().abs()) * 0.5

def boundary_loss(x, lengths, clearance=0.0):
    h = half_extent(x)[..., None] + clearance
    p = torch.relu(h - x[..., :2])
    q = torch.relu(x[..., :2] + h - lengths[:, None, None])
    return (p.square() + q.square()).sum(dim=(1, 2))

@torch.no_grad()
def canonicalize(x):
    """Translate into [0,L]^2 and return the actual corner bounding side."""
    y = x.clone()
    h = half_extent(y)
    low = (y[..., :2] - h[..., None]).amin(dim=1)
    high = (y[..., :2] + h[..., None]).amax(dim=1)
    # A bounding box reconstructed by subtraction can be one ulp too tight.
    # Put the layout strictly inside a padded box; never mask this with tolerance.
    magnitude = torch.maximum(low.abs().amax(-1), high.abs().amax(-1))
    pad = 64 * torch.finfo(DTYPE).eps * (1 + magnitude)
    y[..., :2] -= low[:, None, :]
    y[..., :2] += pad[:, None, None]
    side = (high-low).amax(dim=-1) + 2*pad
    side = torch.nextafter(side, torch.full_like(side, torch.inf))
    return y, side

def all_pair_tiles(x, cfg):
    """Enumerate every unordered pair, without a B*N*N allocation."""
    batch, n, _ = x.shape
    for b0 in range(0, batch, cfg.batch_chunk):
        b1 = min(batch, b0 + cfg.batch_chunk)
        for i0 in range(0, n, cfg.tile):
            i1 = min(n, i0 + cfg.tile)
            for j0 in range(i0, n, cfg.tile):
                j1 = min(n, j0 + cfg.tile)
                yield b0, b1, i0, i1, j0, j1

@torch.no_grad()
def validate(x, lengths, cfg=None):
    """All pairs and every boundary; returns float_valid, worst overlap, wall.

    float_valid is only a numerical screen, not a proof or official certificate.
    NaN/infinity always invalid, including single-square fixtures.
    """
    cfg = cfg or Config()
    batch = x.shape[0]
    worst = torch.full((batch,), -torch.inf, dtype=DTYPE, device=x.device)
    for b0, b1, i0, i1, j0, j1 in all_pair_tiles(x, cfg):
        p = pair_penetration(x[b0:b1, i0:i1, None, :],
                             x[b0:b1, None, j0:j1, :])
        if i0 == j0:
            mask = torch.arange(i0,i1,device=x.device)[:,None] >= torch.arange(j0,j1,device=x.device)[None,:]
            p.masked_fill_(mask[None], -torch.inf)
        worst[b0:b1] = torch.maximum(worst[b0:b1], p.amax(dim=(1,2)))
    h = half_extent(x)[..., None]
    wall = torch.maximum((h-x[...,:2]).amax(dim=(1,2)),
                          (x[...,:2]+h-lengths[:,None,None]).amax(dim=(1,2)))
    finite = torch.isfinite(x).all(dim=(1,2)) & torch.isfinite(lengths) & (lengths > 0)
    valid = finite & (worst <= cfg.overlap_tol) & (wall <= cfg.overlap_tol)
    return valid, worst, wall

@torch.no_grad()
def near_pairs(x, cfg):
    """Angle-independent circumscribed-circle neighbor list with skin.

    Rebuilt when any center moves skin/2 or on rebuild_every. Rotations cannot
    cause a missed pair because every unit square fits a radius sqrt(0.5) disk.
    Building uses bounded Bchunk*tile*tile temporary tensors.
    """
    parts = []
    radius2 = (math.sqrt(2) + cfg.skin)**2
    for b0,b1,i0,i1,j0,j1 in all_pair_tiles(x,cfg):
        d = x[b0:b1,i0:i1,None,:2] - x[b0:b1,None,j0:j1,:2]
        mask = d.square().sum(-1) <= radius2
        if i0 == j0:
            mask &= (torch.arange(i0,i1,device=x.device)[:,None] <
                     torch.arange(j0,j1,device=x.device)[None,:])[None]
        inds = mask.nonzero()
        inds += torch.tensor([b0,i0,j0],device=x.device)
        parts.append(inds)
    return torch.cat(parts,dim=0) if parts else torch.empty((0,3),dtype=torch.long,device=x.device)

@torch.no_grad()
def expand_to_separate(x, cfg):
    """Small uniform center dilation to create actual SAT clearance.

    Checks ALL pairs, not only a stale sparse list. Identical centers cannot be
    fixed by dilation and are explicitly flagged. Expansion is capped.
    """
    scale = torch.ones(x.shape[0],device=x.device,dtype=DTYPE)
    for b0,b1,i0,i1,j0,j1 in all_pair_tiles(x,cfg):
        a=x[b0:b1,i0:i1,None,:]
        b=x[b0:b1,None,j0:j1,:]
        dt=a[...,2]-b[...,2]
        radius=(1+dt.cos().abs()+dt.sin().abs())*0.5
        proj=radius-pair_penetration(a,b)
        factor=(radius+cfg.clearance)/proj.clamp_min(1e-30)
        if i0 == j0:
            mask=torch.arange(i0,i1,device=x.device)[:,None] >= torch.arange(j0,j1,device=x.device)[None,:]
            factor.masked_fill_(mask[None],1.0)
        scale[b0:b1]=torch.maximum(scale[b0:b1],factor.amax(dim=(1,2)))
    # Round outward, including when clearance=0. The cap applies to the
    # actual applied factor, not its downward-rounded predecessor.
    scale = torch.nextafter(
        scale * (1 + 64*torch.finfo(DTYPE).eps),
        torch.full_like(scale, torch.inf))
    usable=torch.isfinite(scale) & (scale <= cfg.max_expand)
    out=x.clone()
    # Use the true required factor only when bounded. Unusable candidates remain
    # unexpanded and are later rejected; never return giant poisoned coordinates.
    factor=torch.where(usable,scale,torch.ones_like(scale))
    out[...,:2] *= factor[:,None,None]
    out,lengths=canonicalize(out)
    _, overlap, wall = validate(out, lengths, cfg)
    # Strict post-transform verification is independent of any caller-selected
    # screening tolerance. Even tiny positive residuals make this unusable.
    usable &= torch.isfinite(out).all(dim=(1,2)) & (overlap <= 0) & (wall <= 0)
    return out,lengths,usable

def repair(x0, lengths, cfg, deadline):
    x=x0.detach().clone().requires_grad_(True)
    opt=torch.optim.Adam([x],lr=cfg.lr)
    reference=None
    pairs=None
    steps=0
    for step in range(cfg.repair_steps):
        if time.monotonic() >= deadline:
            break
        rebuild=reference is None or step % cfg.rebuild_every == 0
        if not rebuild:
            # scalar sync also keeps the soft wall budget responsive on CUDA
            rebuild=bool(((x.detach()[...,:2]-reference).norm(dim=-1).amax() >= cfg.skin*0.49).item())
        if rebuild:
            pairs=near_pairs(x.detach(),cfg)
            reference=x.detach()[...,:2].clone()
        opt.zero_grad(set_to_none=True)
        # Sum (not mean) per lane so results are invariant to batch size.
        (10.0*boundary_loss(x,lengths,cfg.clearance).sum()).backward()
        for k in range(0,len(pairs),cfg.pair_chunk):
            ids=pairs[k:k+cfg.pair_chunk]
            p=pair_penetration(x[ids[:,0],ids[:,1]],x[ids[:,0],ids[:,2]])
            # Each chunk has its own graph, so dense invalid seeds do not retain
            # O(B*N^2) autograd intermediates.
            torch.relu(p+cfg.clearance).square().sum().backward()
        before=x.detach().clone()
        opt.step()
        with torch.no_grad():
            x.copy_(before+(x-before).clamp(-cfg.max_step,cfg.max_step))
            x[...,2].copy_(torch.remainder(x[...,2]+math.pi/4,math.pi/2)-math.pi/4)
        steps += 1
    return x.detach(),steps

@torch.no_grad()
def perturb(x, lengths, band_mask, temp, generator, cfg):
    out=x.clone()
    def randn(shape):
        return torch.randn(shape,dtype=DTYPE,device=x.device,generator=generator)
    out[...,:2] += randn(out[...,:2].shape)*cfg.perturb_xy*temp
    out[...,2] += randn(out[...,2].shape)*cfg.perturb_angle*temp
    # Collective rigid-ish strip motion plus independent angles. The mask is
    # seed-supplied when available; otherwise use non-upright squares.
    shift=randn((len(x),1,2))*cfg.collective_xy*temp
    angle=randn((len(x),1))*cfg.collective_angle*temp
    weight=band_mask.to(DTYPE)
    center=(out[...,:2]*weight[...,None]).sum(1)/weight.sum(1).clamp_min(1)[:,None]
    d=out[...,:2]-center[:,None,:]
    c,s=angle.cos(),angle.sin()
    rot=torch.stack((d[...,0]*c-d[...,1]*s,d[...,0]*s+d[...,1]*c),-1)
    out[...,:2] += weight[...,None]*(rot-d+shift)
    out[...,2] += weight*angle
    new_lengths=lengths*(1-cfg.shrink)
    out[...,:2] *= (new_lengths/lengths)[:,None,None]
    return out,new_lengths

def relax(seeds: np.ndarray, lengths, *, device="cuda", config=None,
          band_mask=None, checkpoint: Callable | None=None):
    """B,N,3 seeds, B lengths. Output dict includes x,L,float_valid,diagnostics.

    All state is Float64. Invalid initial lanes have L=inf until a repaired
    all-pair-valid candidate is found. The caller owns GPU allocation/lifetime.
    Walltime is a soft deadline (one SAT sweep / kernel can overrun it).
    """
    cfg=config or Config()
    if cfg.seconds <= 0 or cfg.skin <= 0 or cfg.tile < 1 or cfg.batch_chunk < 1 or cfg.pair_chunk < 1:
        raise ValueError("positive time, skin, and chunk sizes required")
    if cfg.repair_steps < 0 or cfg.rebuild_every < 1 or cfg.max_rounds < 0:
        raise ValueError("invalid step counts")
    if not (0 <= cfg.shrink < 1) or cfg.max_expand < 1 or cfg.overlap_tol < 0:
        raise ValueError("invalid shrink, expansion, or tolerance")
    dev=torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    x=torch.as_tensor(np.asarray(seeds),device=dev,dtype=DTYPE).clone()
    if x.ndim != 3 or x.shape[-1] != 3 or min(x.shape[:2]) < 1:
        raise ValueError("seeds must have shape B,N,3 with nonempty B,N")
    if not bool(torch.isfinite(x).all()):
        raise ValueError("seeds must be finite")
    lens=torch.as_tensor(lengths,device=dev,dtype=DTYPE).expand(x.shape[0]).clone()
    if not bool((torch.isfinite(lens)&(lens>0)).all()):
        raise ValueError("lengths must be finite positive scalar or B vector")
    if band_mask is None:
        band=(torch.sin(2*x[...,2]).abs()>0.1)
    else:
        band=torch.as_tensor(band_mask,device=dev,dtype=torch.bool)
        if band.shape != x.shape[:2]:
            raise ValueError("band_mask must have shape B,N")
    start=time.monotonic()
    deadline=start+cfg.seconds
    gen=torch.Generator(device=dev).manual_seed(cfg.seed)
    valid,_,_=validate(x,lens,cfg)
    best=x.clone()
    best_l=torch.where(valid,lens,torch.full_like(lens,torch.inf))
    best_valid=valid.clone()
    current=x.clone()
    current_l=lens.clone()
    rounds=steps=accepted=0
    last_checkpoint=start
    def result():
        ok,overlap,wall=validate(best,torch.where(best_valid,best_l,lens),cfg)
        return dict(x=best.cpu().numpy(),L=best_l.cpu().numpy(),
            float_valid=(ok&best_valid).cpu().numpy(),
            worst_overlap=overlap.cpu().numpy(),worst_wall=wall.cpu().numpy(),
            rounds=rounds,steps=steps,accepted=accepted,
            elapsed=time.monotonic()-start,config=asdict(cfg),
            certificate="float64 all-pair SAT screen only; official metric required")
    if checkpoint:
        checkpoint(result())
    while time.monotonic()<deadline and rounds<cfg.max_rounds:
        fraction=max(0.0,1-(time.monotonic()-start)/cfg.seconds)
        temp=max(0.08,fraction)
        # Alternate annealed walk and restart from immutable best state.
        base=current
        base_l=current_l
        if rounds % 3 == 0:
            base=torch.where(best_valid[:,None,None],best,current)
            base_l=torch.where(best_valid,best_l,current_l)
        proposal,target=perturb(base,base_l,band,temp,gen,cfg)
        proposal,used=repair(proposal,target,cfg,deadline)
        steps += used
        proposal,side,expand_ok=expand_to_separate(proposal,cfg)
        ok,_,_=validate(proposal,side,cfg)
        ok &= expand_ok
        improve=ok & (side < best_l)
        best[improve]=proposal[improve]
        best_l[improve]=side[improve]
        best_valid |= improve
        delta=side-current_l
        prob=torch.exp((-delta/(cfg.temperature*temp+1e-15)).clamp(max=0))
        take=ok & ((delta<0)|(torch.rand(side.shape,device=dev,dtype=DTYPE,generator=gen)<prob))
        current[take]=proposal[take]
        current_l[take]=side[take]
        accepted += int(take.sum().item())
        rounds += 1
        if checkpoint and time.monotonic()-last_checkpoint >= cfg.checkpoint_every:
            checkpoint(result())
            last_checkpoint=time.monotonic()
    out=result()
    if checkpoint:
        checkpoint(out)
    return out
