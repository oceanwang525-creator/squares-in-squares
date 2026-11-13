import os
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[v]="1"
import sys, math, json, time, numpy as np
from decimal import Decimal
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree
sys.path.insert(0,'/mnt/e/squares-poc-regenerate-nov13/poc/baseline/poc/src')
from squares import core
from squares import _official_metric as metric
metric.KAGGLE_CORES=1
L=19.6708567970218
C=[]
# grid
for i in range(19):
    for j in range(19): C.append((i+0.5,j+0.5,0.0))
# margin strips top/right, several angles
for deg in (15,25,35,45,55,65,75):
    th=math.radians(deg); e=0.5*(abs(math.cos(th))+abs(math.sin(th)))
    xs=np.arange(e,L-e+1e-9,0.3)
    for x in xs:
        for y in xs:
            if x>16.0 or y>16.0:
                C.append((float(x),float(y),th))
C=np.array(C)
print('candidates',len(C),flush=True)
tree=cKDTree(C[:,:2]); pairs=tree.query_pairs(1.45,output_type='ndarray')
i=pairs[:,0]; j=pairs[:,1]
dx=C[j,0]-C[i,0]; dy=C[j,1]-C[i,1]
ti=C[i,2]; tj=C[j,2]
ci=np.cos(ti); si=np.sin(ti); cj=np.cos(tj); sj=np.sin(tj)
dt=tj-ti; rad=0.5*(1+np.abs(np.cos(dt))+np.abs(np.sin(dt)))
p=np.abs(ci*dx+si*dy); p=np.maximum(p,np.abs(-si*dx+ci*dy))
p=np.maximum(p,np.abs(cj*dx+sj*dy)); p=np.maximum(p,np.abs(-sj*dx+cj*dy))
E=pairs[p<rad-1e-9]
print('edges',len(E),flush=True)
rows=np.repeat(np.arange(len(E)),2)
cols=np.empty(2*len(E),dtype=np.int64); cols[0::2]=E[:,0]; cols[1::2]=E[:,1]
A=coo_matrix((np.ones(2*len(E)),(rows,cols)),shape=(len(E),len(C))).tocsr()
t0=time.time()
res=milp(c=-np.ones(len(C)),constraints=LinearConstraint(A,-np.inf,1.0),
         integrality=np.ones(len(C)),bounds=Bounds(0,1),
         options={'time_limit':300.0,'presolve':True,'mip_rel_gap':0.0})
print('status',res.status,res.message,'time',time.time()-t0,flush=True)
if res.x is None: sys.exit()
sel=np.where(res.x>0.5)[0]
print('selected',len(sel),flush=True)
# take 367 if possible
if len(sel)>=367:
    sel=sel[:367]
    rows=[dict(square_id=str(k),x=repr(float(C[c,0])),y=repr(float(C[c,1])),angle=repr(float(C[c,2]))) for k,c in enumerate(sel)]
    for eps in (0.0,1e-12,1e-11,1e-10,1e-9):
        a2=np.array([[float(r['x']),float(r['y']),float(r['angle'])] for r in rows])
        if eps>0:
            em=0.5*(np.abs(np.cos(a2[:,2]))+np.abs(np.sin(a2[:,2])))
            cx=((a2[:,0]-em).min()+(a2[:,0]+em).max())/2; cy=((a2[:,1]-em).min()+(a2[:,1]+em).max())/2
            a2[:,0]=cx+(a2[:,0]-cx)*(1+eps); a2[:,1]=cy+(a2[:,1]-cy)*(1+eps)
        rr=[dict(square_id=str(i),x=repr(float(a2[i,0])),y=repr(float(a2[i,1])),angle=repr(float(a2[i,2]))) for i in range(367)]
        try:
            lit=Decimal(core.certify(367,rr)); tr=core.kaggle_roundtrip({367:rr}); pan=Decimal(core.certify(367,tr[367]))
            print('CERT',lit,pan,'eps',eps,flush=True)
            import csv
            ours=Decimal(core.certify(367,[r for r in csv.DictReader(open('/mnt/e/squares-poc-regenerate-nov13/poc/pell_corrugation/data/border-scan/merged.csv')) if int(r['n'])==367]))
            print('ours',ours,flush=True)
            if lit<ours and pan<ours:
                json.dump({'367':dict(old_L=str(ours),new_L=str(max(lit,pan)),literal_L=str(lit),pandas_L=str(pan),rows=rr)},open('/mnt/e/squares-poc-regenerate-nov13/poc/setpack367/data/regenerate-nov13/setpack367_win.json','w'),indent=1)
                print('WIN',flush=True)
            break
        except Exception as ex:
            print('cert eps',eps,'fail',str(ex)[:80],flush=True)
