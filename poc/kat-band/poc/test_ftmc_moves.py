import numpy as np
from ftmc_moves import split,mutate

def test_split_does_not_snap_near_upright():
    q=np.array([[.5,.5,0.],[2,2,1e-7],[3,3,np.pi/2],[4,4,.4]])
    a,u=split(q)
    assert len(a)==2 and len(u)==2
    assert a.flags.c_contiguous and a.dtype==np.float64

def test_replay_and_incumbent_immutability():
    q=np.array([[.5,.5,0.],[2,2,.4],[3.2,2.4,.4]])
    old=q.copy()
    for mode in ['angle','shear','slide','taper','identity']:
        a,u,m=mutate(q,5,seed=42,mode=mode)
        b,_,_=mutate(q,5,seed=42,mode=mode)
        np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(q,old)
        np.testing.assert_array_equal(u,q[:1])
