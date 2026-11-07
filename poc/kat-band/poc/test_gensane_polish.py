import numpy as np
import torch
from gensane_polish import quench,unit_feasible,collision_loss
from band_relax import validate
torch.set_num_threads(1)

def test_quench_tiny_grid_recovers():
    q=torch.tensor([[[.5,.5,0],[1.5,.5,0],[.5,1.5,0],[1.5,1.5,0]]],dtype=torch.float64)
    z=q+torch.randn(q.shape,dtype=q.dtype,generator=torch.Generator().manual_seed(2))*.001
    y,meta=quench(z,[2],steps=600,seconds=10)
    y,L,ok=unit_feasible(y)
    assert ok.all() and float(L[0])<2+1e-7
    assert meta["evaluations"]>0

def test_loss_includes_pairs_newly_entering_radius():
    x=torch.tensor([[[.5,.5,0],[3,.5,0]]],dtype=torch.float64,requires_grad=True)
    assert collision_loss(x,torch.tensor([4.])).item()<1e-15
    with torch.no_grad():x[0,1,0]=1.4
    loss=collision_loss(x,torch.tensor([4.]))
    assert loss.item()>.009
    loss.backward()
    assert x.grad[0,1,0]<0

def test_deadline_still_returns_owned_state():
    x=torch.tensor([[[.5,.5,0],[1.4,.5,0]]],dtype=torch.float64)
    y,meta=quench(x,[2],seconds=0)
    assert y.data_ptr()!=x.data_ptr()
    assert torch.equal(y,x)
