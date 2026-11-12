import numpy as np, sys, time
from multiprocessing import Pool
from bandgen import build
R2=np.sqrt(2)
def cap(args):
    L,m,p,du,dv=args
    uc=R2*L/2
    P=build(L,m,dv,uc-(p-1)/2+du,np.pi/4,-1e-12,0,p)
    return len(P),args
def scan(n,Ls,ms,offs):
    jobs=[(L,m,p,du,dv) for L in Ls for m in ms for p in ms if p>=m for du in offs for dv in offs]
    best={}
    with Pool(8) as pool:
        for c,a in pool.imap_unordered(cap,jobs,chunksize=64):
            L=a[0]
            if c>best.get(L,(0,))[0]: best[L]=(c,a)
    return best
if __name__=='__main__':
    n=int(sys.argv[1]); L0,L1,steps=float(sys.argv[2]),float(sys.argv[3]),int(sys.argv[4])
    ms=range(int(sys.argv[5]),int(sys.argv[6]))
    offs=[0,0.5,-0.5,0.25,-0.25]
    t=time.time()
    best=scan(n,np.linspace(L0,L1,steps),ms,offs)
    for L in sorted(best): print(f'L={L:.6f} cap={best[L][0]} need={n} {"OK" if best[L][0]>=n else ""} m,p,du,dv={best[L][1][1:]}')
    print('time',round(time.time()-t,1))
