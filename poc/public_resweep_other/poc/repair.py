"""Fixed-angle SAT-guided LP repair (same method as poc/gallery_repair, parametrized slack)."""
import os
os.environ['OMP_NUM_THREADS']='1'
import sys
sys.path.insert(0,'/Users/lucaszhang/squares-in-squares/poc/public_resweep_other/poc')
from common import *
import mpmath as mp
from scipy.sparse import coo_matrix
from scipy.optimize import linprog
def lp(xy,ang,slack,trust=0.03,tl=30):
    n=len(ang)
    e0=.5*(abs(np.cos(ang))+abs(np.sin(ang)))
    xy=xy-np.array([(xy[:,0]-e0).min(),(xy[:,1]-e0).min()])+1e-6
    u=np.column_stack([np.cos(ang),np.sin(ang)]);v=np.column_stack([-np.sin(ang),np.cos(ang)])
    rr=[];cc=[];vals=[];rhs=[]
    def add(cols,vs,b):
        row=len(rhs);rr.extend([row]*len(cols));cc.extend(cols);vals.extend(vs);rhs.append(b)
    for i,j in cKDTree(xy).query_pairs(1.6):
        axes=np.array([u[i],v[i],u[j],v[j]]);delta=xy[j]-xy[i];proj=axes@delta
        radii=.5*(abs(axes@u[i])+abs(axes@v[i])+abs(axes@u[j])+abs(axes@v[j]))
        k=np.argmax(abs(proj)-radii);a=axes[k]*(1 if proj[k]>=0 else -1)
        add([2*i,2*i+1,2*j,2*j+1],[a[0],a[1],-a[0],-a[1]],-radii[k]-slack)
    ext=.5*(abs(np.cos(ang))+abs(np.sin(ang)))
    for i in range(n):
        for k in range(2):
            add([2*i+k],[-1],-ext[i]-slack);add([2*i+k,2*n],[1,-1],-ext[i]-slack)
    A=coo_matrix((vals,(rr,cc)),shape=(len(rhs),2*n+1)).tocsr()
    c=np.zeros(2*n+1);c[-1]=1
    bounds=[(float(z)-trust,float(z)+trust) for z in xy.ravel()]+[(0,None)]
    res=linprog(c,A_ub=A,b_ub=rhs,bounds=bounds,method='highs',options={'dual_feasibility_tolerance':1e-9,'primal_feasibility_tolerance':1e-10,'time_limit':tl})
    if not res.success:return None
    return res.x[:-1].reshape(n,2),float(res.fun)
def rows_from(centers,ang_strs,dil):
    mp.mp.dps=75
    f=1+mp.mpf(dil)
    return [dict(square_id=str(i),x=mp.nstr(mp.mpf(str(Decimal.from_float(float(centers[i,0]))))*f,65),y=mp.nstr(mp.mpf(str(Decimal.from_float(float(centers[i,1]))))*f,65),angle=ang_strs[i]) for i in range(len(centers))]
