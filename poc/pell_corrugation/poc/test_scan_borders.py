"""Fast scan invariants, independent of the expensive search."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import pell_corrugation as p
def test_border_counts_and_spacing():
    core=np.array([[.5,.5,0],[1.5,.5,0],[.5,1.5,0]])
    for t in range(1,5):
        wrapped=p.add_l_border(core,t,2+t)
        assert len(wrapped)==3+4*t+t*t
        assert p.sat_check(wrapped)[0]
        assert abs(p.extent(wrapped)-(2+t+2e-7))<1e-12
def test_prefix_deletions_preserve_feasibility():
    core=np.array([[.5,.5,0],[1.5,.5,0],[.5,1.5,0]])
    wrapped=p.add_l_border(core,2,4)
    for axis in (0,1):
        order=np.argsort(wrapped[:,axis],kind="stable")
        for ids in (order,order[::-1]):
            for n in range(1,len(ids)+1):
                q=wrapped[ids[:n]]
                assert p.sat_check(q)[0]
                assert p.extent(q)<=p.extent(wrapped)+1e-12
