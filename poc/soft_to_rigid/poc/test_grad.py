import numpy as np, s2r
n=6; rng=np.random.default_rng(0)
z=np.column_stack([rng.uniform(.3,2.2,(n,2)),rng.uniform(-1,1,n)]).ravel()
for name,fn,args in (('soft',s2r.soft_fg,(n,2.5,6.0,s2r.shape_pts(6.0))),('rigid',s2r.rigid_fg,(n,2.5))):
    f,g=fn(z,*args); gn=np.zeros_like(z)
    for i in range(len(z)):
        e=np.zeros_like(z); e[i]=1e-6
        gn[i]=(fn(z+e,*args)[0]-fn(z-e,*args)[0])/2e-6
    print(name,f,np.abs(g-gn).max(),np.abs(g).max())
