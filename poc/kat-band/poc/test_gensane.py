import math
import numpy as np
import torch
try:
    from . import gensane as g
except ImportError:
    import gensane as g


def independent_valid(x,L,side=1.):
    # Independent corner construction and separating projection intervals.
    corners=[]
    for cx,cy,a in x:
        c,s=math.cos(a),math.sin(a)
        q=np.array([[u*c-v*s,u*s+v*c] for u,v in [(-.5,-.5),(-.5,.5),(.5,.5),(.5,-.5)]])*side+[cx,cy]
        if q.min() < -1e-12 or q.max()>L+1e-12: return False
        corners.append(q)
    for i in range(len(x)):
        for j in range(i):
            axes=[]
            for k in (i,j):
                for edge in (corners[k][1]-corners[k][0],corners[k][2]-corners[k][1]):
                    axes.append(np.array([-edge[1],edge[0]]))
            if not any((corners[i]@a).max()<=(corners[j]@a).min()+1e-12 or (corners[j]@a).max()<=(corners[i]@a).min()+1e-12 for a in axes): return False
    return True


def fixture():
    return torch.tensor([[[.6,.6,0],[1.8,.6,0],[.6,1.8,0],[1.8,1.8,0]]],dtype=torch.float64),torch.tensor([2.4],dtype=torch.float64)


def test_inflate_above_one_and_maximality():
    x,L=fixture(); s=g.maximal_side(x,L)
    assert abs(s.item()-1.2)<1e-14
    assert independent_valid(x[0].numpy(),L.item(),s.item()*(1-1e-12))
    assert not independent_valid(x[0].numpy(),L.item(),s.item()*(1+1e-7))
    y,l,ok=g.inflate(x,L)
    assert ok.all() and l.item()<2.000000001
    assert independent_valid(y[0].numpy(),l.item())


def test_rotated_shrink_and_duplicates():
    x=torch.tensor([[[.6,.6,.3],[1.,1.,-.2]]],dtype=torch.float64); L=torch.tensor([2.],dtype=torch.float64)
    s=g.maximal_side(x,L).item(); assert 0<s<1
    assert independent_valid(x[0].numpy(),2.,s*(1-1e-12))
    assert not independent_valid(x[0].numpy(),2.,s*(1+1e-7))
    y,l,ok=g.inflate(x,L); assert ok.all() and independent_valid(y[0].numpy(),l.item())
    x[:,1,:2]=x[:,0,:2]
    assert g.maximal_side(x,L).item()==0
    assert not g.inflate(x,L)[2].any()


def test_random_walk_and_all_perturb():
    x,L=fixture(); x=x.repeat(3,1,1); L=L.repeat(3)
    gen=torch.Generator().manual_seed(38)
    y=g.global_perturb(x,gen,.02,.03)
    assert torch.all(y!=x) and torch.equal(x,fixture()[0].repeat(3,1,1))
    accepted=0
    for _ in range(100):
        old=x.clone(); x,ok=g.walk_step(x,L,gen,.08,.08)
        assert ((x!=old).any(-1).sum(-1)<=1).all()
        accepted+=int(ok.sum())
        assert all(independent_valid(a.numpy(),b.item()) for a,b in zip(x,L))
    assert accepted>0 and x.dtype==torch.float64


def test_best_immutable_monotonic_deadline():
    x,L=fixture(); snapshots=[]
    cfg=g.Config(seconds=10,max_rounds=3,walks_per_round=10,inflate_every=5,checkpoint_every=0)
    out=g.relax(x,L,config=cfg,checkpoint=snapshots.append)
    for a,b in zip(snapshots,snapshots[1:]): assert np.all(b['L']<=a['L'])
    saved=out['x'].copy(); snapshots[0]['x'].fill(999)
    assert np.array_equal(out['x'],saved)
    assert out['float_valid'].all() and independent_valid(out['x'][0],out['L'][0])
    assert out['rounds']==3
    import time
    out=g.relax(x,L,config=cfg,deadline=time.monotonic()-1)
    assert out['rounds']==0


def test_invalid_initial_never_claimed_valid():
    x=torch.zeros((1,2,3),dtype=torch.float64)
    out=g.relax(x,2.,config=g.Config(max_rounds=0))
    assert not out['float_valid'].any() and np.isinf(out['L']).all()


def test_hook_rejects_collision_and_can_mutate_owned_arguments():
    x,L=fixture(); original=x.clone(); calls=[]
    def hook(y,s,best,deadline):
        calls.append(1); y.zero_(); best.zero_()
        return y,s
    out=g.relax(x,L,config=g.Config(seconds=5,max_rounds=2,walks_per_round=3,restart_every=0),polish_hook=hook)
    assert len(calls)==2 and torch.equal(x,original)
    assert out['float_valid'].all() and independent_valid(out['x'][0],out['L'][0])


def test_tiled_max_side_matches_dense():
    x,L=fixture(); x=x.repeat(2,1,1)
    x[1,:,2]=torch.tensor([.2,-.1,.3,.4])
    L=L.repeat(2)
    assert torch.equal(g.maximal_side(x,L,g.Config(tile=2,batch_chunk=1)),g.maximal_side(x,L,g.Config(tile=1024,batch_chunk=4)))


def test_current_result_is_independent_and_feasible():
    x,L=fixture()
    out=g.relax(x,L,config=g.Config(seconds=5,max_rounds=2,walks_per_round=8,restart_every=0))
    assert out['current_float_valid'].all()
    assert independent_valid(out['current_x'][0],out['current_L'][0])
    assert np.all(out['L']<=out['current_L'])
    saved=out['x'].copy(); out['current_x'].fill(999); out['current_L'].fill(999)
    assert np.array_equal(out['x'],saved)
