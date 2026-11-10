"""Basin hopping: dilate + jitter tilted angles, then SLP. usage: hop.py n seed secs sigma dil"""
import sys, numpy as np
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from slp import *
n=int(sys.argv[1]); seed=int(sys.argv[2]); secs=float(sys.argv[3]); sig=float(sys.argv[4]); dil=float(sys.argv[5])
rng=np.random.default_rng(seed)
P=load(n).copy(); L0=L_of(P)
til=np.abs(np.sin(2*P[:,2]))>0.05
c=P[:,:2].mean(0)
P[:,:2]=c+(P[:,:2]-c)*dil
P[til,2]+=rng.normal(0,sig,til.sum())
P[til,:2]+=rng.normal(0,0.002,(til.sum(),2))
print("start",L_of(P),min_sep(P),flush=True)
if min_sep(P)<0:
    print("infeasible start");sys.exit()
P,Lc=run(P,secs,0.05,0.02,lambda s:None)
print("RESULT",n,seed,sig,dil,Lc,Lc-int(n**.5),"inc",L0,flush=True)
np.save(f"data/hop_{n}_{seed}.npy",P)
