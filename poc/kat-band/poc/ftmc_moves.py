"""FTMC core mutations. Host proposal generation; no incumbent writes.

Contiguous fp64 center/angle arrays, explicit RNG seed, replayable mutation
metadata. Uprights are discarded, never jittered or silently snapped.
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

def split(q, tolerance=1e-9):
    q=np.asarray(q,dtype=np.float64)
    delta=np.abs((q[:,2]+np.pi/4)%(np.pi/2)-np.pi/4)
    mask=delta>tolerance
    return np.ascontiguousarray(q[mask]),np.ascontiguousarray(q[~mask])

def components(core):
    if not len(core):return []
    p=np.array(sorted(cKDTree(core[:,:2]).query_pairs(1.5)),dtype=int).reshape(-1,2)
    g=coo_matrix((np.ones(2*len(p)),(np.r_[p[:,0],p[:,1]],np.r_[p[:,1],p[:,0]])),
                 shape=(len(core),len(core)))
    n,labels=connected_components(g)
    return [np.flatnonzero(labels==i) for i in range(n)]

def mutate(q,side,*,seed,mode="angle",amplitude=.001,shrink=0.):
    core,upright=split(q);out=core.copy();rng=np.random.default_rng(seed)
    groups=components(core)
    selected=[] if not groups else groups[int(rng.integers(len(groups)))]
    if len(selected):
        c=out[selected,:2].mean(0);xy=out[selected,:2]-c
        if mode=="angle":
            angle=float(rng.uniform(-amplitude,amplitude));cs=np.cos(angle);sn=np.sin(angle)
            out[selected,:2]=xy@np.array([[cs,sn],[-sn,cs]])+c
            out[selected,2]+=angle
        elif mode=="shear":
            shear=float(rng.uniform(-amplitude,amplitude))
            out[selected,0]+=shear*xy[:,1]
        elif mode=="slide":
            out[selected,:2]+=rng.uniform(-amplitude,amplitude,2)
        elif mode=="taper":
            # Convert one adjacent upright to tilted and remove an opposite
            # tip tilted cell. Refill decides actual upright count afterward.
            if len(upright):
                axis=np.linalg.svd(xy,full_matrices=False)[2][0]
                projection=xy@axis
                tip=int(selected[np.argmin(projection)])
                end=int(selected[np.argmax(projection)])
                distance=np.linalg.norm(upright[:,:2]-out[end,:2],axis=1)
                added=upright[int(np.argmin(distance))].copy()
                added[2]=out[end,2]
                out=np.vstack((np.delete(out,tip,axis=0),added))
        elif mode!="identity":raise ValueError("unknown mutation")
    target=side-shrink
    if shrink:out[:,:2]=target/side*(out[:,:2]-side/2)+target/2
    return np.ascontiguousarray(out),upright.copy(),dict(seed=seed,mode=mode,
        amplitude=amplitude,shrink=shrink,core_count=len(out),groups=len(groups),target=target)
