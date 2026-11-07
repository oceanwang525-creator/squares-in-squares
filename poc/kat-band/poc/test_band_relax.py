import math
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
from band_relax import (Config, pair_penetration, boundary_loss, validate,
                        near_pairs, expand_to_separate, perturb, repair, relax)
from run_relax import save_checkpoint

torch.set_num_threads(1)
D=torch.float64

def tensor(x):
    return torch.tensor(x,dtype=D)

def grid():
    return np.array([[[.5,.5,0],[1.5,.5,0],[.5,1.5,0],[1.5,1.5,0]]])

def independent_polygon_sat(a,b):
    # Deliberately different corners-based reference, four axes.
    def corners(s):
        t=s[2]
        R=np.array([[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]])
        return np.array([[-.5,-.5],[.5,-.5],[.5,.5],[-.5,.5]])@R.T+s[:2]
    ca,cb=corners(a),corners(b)
    axes=[]
    for s in (a,b):
        axes.extend([[np.cos(s[2]),np.sin(s[2])],[-np.sin(s[2]),np.cos(s[2])]])
    return all(min((ca@u).max(),(cb@u).max()) >
               max((ca@u).min(),(cb@u).min()) for u in axes)

def test_sat_random_polygon_reference():
    rng=np.random.default_rng(48)
    a=rng.normal(size=(250,3))
    b=rng.normal(size=(250,3))
    got=(pair_penetration(tensor(a),tensor(b))>0).numpy()
    want=np.array([independent_polygon_sat(x,y) for x,y in zip(a,b)])
    assert np.array_equal(got,want)

def test_four_axes_and_boundaries():
    x=tensor(grid())
    ok,p,w=validate(x,tensor([2]))
    assert ok.tolist()==[True] and p.item()==0 and w.item()==0
    bad=x.clone();bad[0,1,0]-=.03
    assert not validate(bad,tensor([2]))[0].item()
    assert not validate(x,tensor([1.999]))[0].item()
    # Rotated single-square containment.
    r=tensor([[[math.sqrt(.5),math.sqrt(.5),math.pi/4]]])
    assert validate(r,tensor([math.sqrt(2)]))[0].item()

def test_gradient_finite_difference_away_kinks():
    x=tensor([[[.28,.46,.19],[.88,.73,-.31]]]).requires_grad_(True)
    def loss(z):
        return torch.relu(pair_penetration(z[:,0],z[:,1])).square().sum()+boundary_loss(z,tensor([1.4])).sum()
    loss(x).backward()
    numeric=torch.empty_like(x)
    for index in np.ndindex(*x.shape):
        a=x.detach().clone(); b=x.detach().clone()
        a[index]+=1e-6;b[index]-=1e-6
        numeric[index]=(loss(a)-loss(b))/(2e-6)
    torch.testing.assert_close(x.grad,numeric,rtol=2e-5,atol=2e-7)

def test_neighbor_list_matches_all_possible_collisions():
    rng=np.random.default_rng(44)
    x=tensor(rng.uniform(-3,3,size=(3,35,3)))
    cfg=Config(tile=9,batch_chunk=2,skin=.6)
    ids=near_pairs(x,cfg).tolist()
    assert len(ids)==len(set(map(tuple,ids)))
    have=set(map(tuple,ids))
    for b in range(3):
        for i in range(35):
            for j in range(i+1,35):
                if pair_penetration(x[b,i],x[b,j])>=0:
                    assert (b,i,j) in have

def test_stale_list_displacement_guard_geometry():
    # pair starts beyond list radius; after approaching enough to collide,
    # at least one particle must have crossed skin/2.
    cfg=Config(skin=.6)
    x=tensor([[[0,0,0],[2.1,0,0]]])
    assert len(near_pairs(x,cfg))==0
    moved=x.clone();moved[0,0,0]+=.56;moved[0,1,0]-=.56
    assert (moved[...,:2]-x[...,:2]).norm(dim=-1).amax() > cfg.skin/2
    assert len(near_pairs(moved,cfg))==1

def test_uniform_expansion_and_coincident_rejection():
    x=tensor([[[.5,.5,.1],[1.45,.5,.1]]])
    cfg=Config(max_expand=1.5)
    y,L,usable=expand_to_separate(x,cfg)
    assert usable.item()
    assert validate(y,L,cfg)[0].item()
    assert validate(y,L,cfg)[1].item() < 0
    same=tensor([[[.5,.5,0],[.5,.5,0]]])
    y,L,usable=expand_to_separate(same,cfg)
    assert not usable.item()
    assert not validate(y,L,cfg)[0].item()

def test_initial_valid_and_invalid_lanes_no_fake_best():
    x=np.concatenate([grid(),grid()])
    x[1,1,0]-=.1
    r=relax(x,[2,2],device="cpu",config=Config(max_rounds=0))
    assert r["float_valid"].tolist()==[True,False]
    assert r["L"][0]==2 and np.isinf(r["L"][1])
    assert "official metric required" in r["certificate"]

def test_deterministic_and_best_retention(tmp_path):
    cfg=Config(seconds=10,max_rounds=3,repair_steps=12,seed=91,pair_chunk=2)
    a=relax(grid(),[2],device="cpu",config=cfg)
    b=relax(grid(),[2],device="cpu",config=cfg)
    np.testing.assert_array_equal(a["x"],b["x"])
    np.testing.assert_array_equal(a["L"],b["L"])
    assert a["float_valid"].all() and a["L"][0]<=2
    path=tmp_path/"checkpoint.npz"
    save_checkpoint(path,a)
    with np.load(path,allow_pickle=False) as z:
        np.testing.assert_array_equal(z["x"],a["x"])
        assert "official metric" in str(z["metadata"])

def test_invalid_seed_repaired_tiny():
    x=np.array([[[.5,.5,.12],[1.48,.53,-.08]]])
    r=relax(x,[2.2],device="cpu",config=Config(seconds=10,max_rounds=2,repair_steps=80,max_expand=1.15))
    assert r["float_valid"].all()
    assert np.isfinite(r["L"]).all()

def test_global_and_collective_perturbations():
    x=tensor(grid())
    x[...,2]=.2
    gen=torch.Generator().manual_seed(9)
    y,L=perturb(x,tensor([2.5]),torch.ones((1,4),dtype=torch.bool),1,gen,Config())
    assert torch.unique(y[...,2]).numel()==4
    assert L.item()<2.5
    assert not torch.equal(y[...,:2],x[...,:2])

def test_nan_and_single_square():
    x=tensor([[[.5,.5,0]]])
    assert validate(x,tensor([1]))[0].item()
    x[0,0,2]=float("nan")
    assert not validate(x,tensor([1]))[0].item()
    with pytest.raises(ValueError):
        relax(x.numpy(),[1],device="cpu")

def test_main_driver_refuses_cpu(tmp_path):
    r=subprocess.run([sys.executable,str(Path(__file__).with_name("run_relax.py")),"unused.npz",
                      "--output",str(tmp_path/"out.npz"),"--device","cpu"],capture_output=True,text=True)
    assert r.returncode!=0 and "requires CUDA" in r.stderr

def test_initial_checkpoint_written(tmp_path):
    checkpoints=[]
    r=relax(grid(),[2],device="cpu",config=Config(max_rounds=0),
            checkpoint=lambda r:checkpoints.append(r))
    assert len(checkpoints)==2
    assert all(z["float_valid"].all() for z in checkpoints)

def test_zero_clearance_still_strict_and_expansion_cap():
    cfg=Config(clearance=0,max_expand=1.5,overlap_tol=1e-5)
    x=tensor([[[.5,.5,.123],[1.42,.51,-.081]]])
    y,L,usable=expand_to_separate(x,cfg)
    assert usable.item()
    _,overlap,wall=validate(y,L,cfg)
    assert overlap.item()<=0 and wall.item()<=0
    # A cap of exactly one cannot permit even an outward-rounded expansion.
    _,_,usable=expand_to_separate(x,Config(clearance=0,max_expand=1))
    assert not usable.item()

def test_canonical_bounds_roundoff_stress():
    from band_relax import canonicalize
    rng=np.random.default_rng(230)
    for _ in range(20):
        x=tensor(rng.normal(size=(4,3,3))*10)
        y,L=canonicalize(x)
        assert (validate(y,L)[2]<=0).all()
