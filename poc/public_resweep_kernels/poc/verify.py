"""Exact-evaluate every prefiltered candidate (literal), repair overlapping ones (fixed-angle SAT LP), keep best strict win per n."""
import os
os.environ['OMP_NUM_THREADS']='1'
import sys,json,pickle,time,math
sys.path.insert(0,'/Users/lucaszhang/squares-in-squares/poc/baseline/poc/src')
from decimal import Decimal
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.optimize import linprog
import mpmath as mp
from squares.core import evaluate,read_submission
F='/Users/lucaszhang/squares-in-squares/poc/public_resweep_kernels/'
ours={int(n):Decimal(v['L']) for n,v in json.load(open(F+'docs/our_L.json')).items()}
m=read_submission('/Users/lucaszhang/squares-in-squares/poc/pell_corrugation/data/border-scan/merged.csv')
okeys={n:tuple((r['x'],r['y'],r['angle']) for r in rs) for n,rs in m.items()}
cand=pickle.load(open(F+'data/cand.pkl','rb'))
ns=sorted(cand)
if len(sys.argv)>1: ns=[n for n in ns if n%int(sys.argv[1])==int(sys.argv[2])]
mp.mp.dps=60
def torad(rows,unit):
    if unit=='rad':return rows
    return [(x,y,mp.nstr(mp.mpf(a)*mp.pi/180,50)) for x,y,a in rows]
def ev(n,rows):
    R=[dict(square_id=str(i),x=x,y=y,angle=a) for i,(x,y,a) in enumerate(rows)]
    try:_,_,s,r=evaluate(n,R)
    except Exception as e:s,r=None,repr(e)
    return s,r,R
def repair(n,rows,L0):
    x=np.array([float(r[0]) for r in rows]);y=np.array([float(r[1]) for r in rows]);th=np.array([float(r[2]) for r in rows])
    xy=np.column_stack([x,y]);u=np.column_stack([np.cos(th),np.sin(th)]);v=np.column_stack([-np.sin(th),np.cos(th)])
    pairs=list(cKDTree(xy).query_pairs(1.6))
    rr=[];cc=[];vals=[];rhs=[]
    def add(c,vl,b):
        r=len(rhs);rr.extend([r]*len(c));cc.extend(c);vals.extend(vl);rhs.append(b)
    mg=1e-9
    for i,j in pairs:
        axes=np.array([u[i],v[i],u[j],v[j]]);proj=axes@(xy[j]-xy[i])
        rad=.5*(abs(axes@u[i])+abs(axes@v[i])+abs(axes@u[j])+abs(axes@v[j]))
        k=np.argmax(abs(proj)-rad);a=axes[k]*(1 if proj[k]>=0 else -1)
        add([2*i,2*i+1,2*j,2*j+1],[a[0],a[1],-a[0],-a[1]],-rad[k]-mg)
    ext=.5*(abs(np.cos(th))+abs(np.sin(th)))
    for i in range(n):
        for k in range(2):
            add([2*i+k],[-1],-ext[i]-mg);add([2*i+k,2*n],[1,-1],-ext[i]-mg)
    A=coo_matrix((vals,(rr,cc)),shape=(len(rhs),2*n+1)).tocsr()
    c=np.zeros(2*n+1);c[-1]=1
    b=[(float(z)-.01,float(z)+.01) for z in xy.ravel()]+[(0,None)]
    res=linprog(c,A_ub=A,b_ub=rhs,bounds=b,method='highs',options={'dual_feasibility_tolerance':1e-9,'primal_feasibility_tolerance':1e-9,'time_limit':60})
    if not res.success:return None
    return res.x[:-1].reshape(n,2),float(res.fun)
res={};log={};t0=time.time()
for n in ns:
    best=None;L0=ours[n];lg=[];tried=0
    allc=sorted(cand[n],key=lambda c:c['bbox']);tot=len(allc)
    for c in [c for c in allc if c['bbox']<float(L0)+1e-13]:
        rows=torad(c['rows'],c['unit'])
        if tuple(c['rows'])==okeys[n] and c['unit']=='rad':continue
        tried+=1
        s,r,R=ev(n,rows)
        rec=dict(src=c['src'],bbox=c['bbox'],unit=c['unit'],reason=r,L=s,repaired=False)
        if r=='ok' and Decimal(s)<L0 and (best is None or Decimal(s)<Decimal(best['L'])):
            best=dict(src=c['src'],L=s,rows=R,repaired=False,pandas_ok=None)
        elif r!='ok' and c['bbox']<float(L0)+1e-9 and (best is None):
            # repair only if still potentially better
            out=repair(n,rows,L0)
            if out is not None:
                cen,lp=out
                rec['lp_L']=lp
                if lp<float(L0)+1e-9:
                    th=[mp.mpf(r_[2]) for r_ in rows]
                    cx,cy=cen[:,0].mean(),cen[:,1].mean()
                    for dil in ['1','1.000000000001','1.00000000001','1.0000000001','1.000000001']:
                        d=mp.mpf(dil)
                        R2=[(mp.nstr(mp.mpf(Decimal(float(cen[i,0])-cx).__str__())*d+mp.mpf(repr(cx)),50),mp.nstr(mp.mpf(Decimal(float(cen[i,1])-cy).__str__())*d+mp.mpf(repr(cy)),50),rows[i][2]) for i in range(n)]
                        s2,r2,RR=ev(n,R2)
                        if r2=='ok' and Decimal(s2)<L0:
                            best=dict(src=c['src'],L=s2,rows=RR,repaired=True,dilation=dil,pandas_ok=None);rec['repaired']=True;rec['L']=s2;rec['reason']='ok(repaired)';break
        lg.append(rec)
    log[n]=dict(examined=tried,prefilter=tot,recs=lg[:40])
    if best:res[n]=dict(best,ours=str(L0),examined=tried,prefilter=tot)
    print(n,tried,'WIN' if best else '-',round(time.time()-t0),flush=True)
    tag=(sys.argv[2] if len(sys.argv)>2 else 'all')
    pickle.dump(res,open(F+f'data/wins_{tag}.pkl','wb'));json.dump(log,open(F+f'data/verify_log_{tag}.json','w'))
