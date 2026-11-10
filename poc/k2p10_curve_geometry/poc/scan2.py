import sys; sys.path.insert(0,'poc')
from band_count import *
K=int(sys.argv[1]); target=K*K+int(sys.argv[2]); Lmax=float(sys.argv[3]); mp_=int(sys.argv[4])
r=best(K,Lmax,K+0.5,target-3,maxpq=mp_)
r.sort(key=lambda t:(-t[1],t[0]))
for t in r[:12]: print(t, "L-K=%.5f"%(t[0]-K), "short=",target-t[1])
