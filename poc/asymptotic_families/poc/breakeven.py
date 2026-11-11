import math, sys
sys.path.insert(0,'.')
sys.path.insert(0,'poc')
from strip import strip_count
f=0.7071067812
print('s  best_j gain  N*m-j*floor  tiltedwaste  0.8s+5   trivialwaste(L^2-floor^2)')
first=None
for S in [25,32,50,100,200,400,800,1600,3200,6400,12800,25600]:
    L=S+f; best=(-1e9,0,None)
    for j in range(1,min(S,int(6*S**0.8))+1) if S<=3200 else range(1,S+1,max(1,S//400)):
        r=strip_count(L,j)
        if r: 
            g=r[2]-j*S
            if g>best[0]: best=(g,j,r)
    g,j,r=best
    n=S*S+g
    print(S,j,g,round(L*L-n,1),round(0.8*L+5,1),round(L*L-S*S,1))
