"""Search for free spots (unit square, any angle) in a built packing inside a container of side Lc."""
import sys, math, numpy as np
sys.path.insert(0, "/Users/lucaszhang/squares-in-squares/poc/k2p10_out_of_phase/poc")
from build import build
def corners(x,y,a):
    c,s=math.cos(a),math.sin(a); h=0.5
    return np.array([[x+c*dx-s*dy, y+s*dx+c*dy] for dx,dy in ((-h,-h),(h,-h),(h,h),(-h,h))])
def overlap(c1,c2,a1,a2,eps=1e-9):
    for a in (a1,a2):
        for k in (0,1):
            ax=np.array([math.cos(a+k*math.pi/2), math.sin(a+k*math.pi/2)])
            p1=c1@ax; p2=c2@ax
            if p1.max()<=p2.min()+eps or p2.max()<=p1.min()+eps: return False
    return True
def free_spots(q, Lc, step=0.05, angs=range(0,90,5)):
    cs=[corners(*r) for r in q]; cen=q[:,:2]
    found=[]
    for ad in angs:
        a=math.radians(ad)
        for x in np.arange(0.5,Lc-0.5+1e-9,step):
            for y in np.arange(0.5,Lc-0.5+1e-9,step):
                c=corners(x,y,a)
                if c.min()<-1e-9 or c.max()>Lc+1e-9: continue
                idx=np.where(np.abs(cen-[x,y]).max(1)<1.5)[0]
                if all(not overlap(c,cs[i],a,q[i,2]) for i in idx): found.append((x,y,ad))
    return found
if __name__=="__main__":
    t,w,a1,a2,b1,b2,ell,K=map(int,sys.argv[1:9]); Lc=float(sys.argv[9])
    q,L=build(t,w,a1,a2,b1,b2,ell,K)
    f=free_spots(q,Lc,step=0.1,angs=[0,45])
    print(len(q),L,"free spots",len(f),f[:10])
