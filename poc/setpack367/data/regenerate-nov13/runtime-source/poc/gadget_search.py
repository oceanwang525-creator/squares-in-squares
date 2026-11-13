import os
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[v]="1"
import sys, math, json, numpy as np, torch, time
from decimal import Decimal
sys.path.insert(0,'/mnt/e/squares-poc-regenerate-nov13/poc/baseline/poc/src')
from squares import core
from squares import _official_metric as metric
metric.KAGGLE_CORES=1
torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
DELTA=0.6708567970218
def run(args):
    m,seed,steps=args
    N=m*m+6; L=m+DELTA
    np.random.seed(seed); torch.manual_seed(seed)
    # seed: m x m grid + 6 random extras, plus noise
    xs=[]; ys=[]; ts=[]
    for i in range(m):
        for j in range(m):
            xs.append(i+0.5); ys.append(j+0.5); ts.append(0.0)
    for k in range(6):
        xs.append(np.random.uniform(0.6,L-0.6)); ys.append(np.random.uniform(0.6,L-0.6)); ts.append(np.random.uniform(-0.8,0.8))
    xs=np.array(xs)+np.random.normal(0,0.08,N)
    ys=np.array(ys)+np.random.normal(0,0.08,N)
    ts=np.array(ts)+np.random.normal(0,0.08,N)
    c=torch.nn.Parameter(torch.tensor(np.column_stack([xs,ys]))); th=torch.nn.Parameter(torch.tensor(ts))
    opt=torch.optim.Adam([c,th],lr=1.2e-3)
    def energy():
        u=torch.stack([torch.cos(th),torch.sin(th)],1); v=torch.stack([-torch.sin(th),torch.cos(th)],1)
        delta=c[None,:,:]-c[:,None,:]
        axes=torch.stack([u[:,None,:].expand(N,N,2),v[:,None,:].expand(N,N,2),
                          u[None,:,:].expand(N,N,2),v[None,:,:].expand(N,N,2)],2)
        proj=(axes*delta[:,:,None,:]).sum(-1)
        rad=0.5*((axes*u[:,None,None,:]).sum(-1).abs()+(axes*v[:,None,None,:]).sum(-1).abs()
                 +(axes*u[None,:,None,:]).sum(-1).abs()+(axes*v[None,:,None,:]).sum(-1).abs())
        pen=(rad-proj.abs()).min(dim=-1).values
        iu=torch.triu_indices(N,N,1)
        viol=pen[iu].clamp(min=0)
        ep=0.5*(torch.cos(th).abs()+torch.sin(th).abs())
        b=torch.cat([ep-c[:,0],c[:,0]+ep-L,ep-c[:,1],c[:,1]+ep-L]).clamp(min=0)
        return (viol**2).sum()+5.0*(b**2).sum(), viol.max(), b.max()
    for it in range(steps):
        opt.zero_grad(); loss,p,w=energy(); loss.backward(); opt.step()
        if it%2000==0 and it>0:
            for g in opt.param_groups: g['lr']*=0.7
    with torch.no_grad(): loss,p,w=energy()
    if p.item()>1e-8 or w.item()>1e-8: return None
    cn=c.numpy(); tn=th.numpy()
    e=0.5*(np.abs(np.cos(tn))+np.abs(np.sin(tn)))
    Latt=float(max((cn[:,0]+e).max()-(cn[:,0]-e).min(),(cn[:,1]+e).max()-(cn[:,1]-e).min()))
    if Latt>L+1e-8: return None
    cand=[dict(square_id=str(i),x=repr(float(cn[i,0])),y=repr(float(cn[i,1])),angle=repr(float(tn[i]))) for i in range(N)]
    for eps in (0.0,1e-12,1e-11,1e-10,1e-9):
        a2=np.array([[float(r['x']),float(r['y']),float(r['angle'])] for r in cand])
        if eps>0:
            em=0.5*(np.abs(np.cos(a2[:,2]))+np.abs(np.sin(a2[:,2])))
            cx=((a2[:,0]-em).min()+(a2[:,0]+em).max())/2; cy=((a2[:,1]-em).min()+(a2[:,1]+em).max())/2
            a2[:,0]=cx+(a2[:,0]-cx)*(1+eps); a2[:,1]=cy+(a2[:,1]-cy)*(1+eps)
        rr=[dict(square_id=str(i),x=repr(float(a2[i,0])),y=repr(float(a2[i,1])),angle=repr(float(a2[i,2]))) for i in range(N)]
        try:
            lit=Decimal(core.certify(N,rr)); tr=core.kaggle_roundtrip({N:rr}); pan=Decimal(core.certify(N,tr[N]))
        except Exception: continue
        np.save(f'/mnt/e/squares-poc-regenerate-nov13/poc/setpack367/data/regenerate-nov13/gadget_{m}.npy',a2)
        return (m,N,str(lit),str(pan),eps)
    return None
if __name__=='__main__':
    import multiprocessing as mp
    ms=[8]
    tasks=[(m,s,3000) for m in ms for s in range(16)]
    print('tasks',len(tasks),flush=True)
    with mp.get_context('fork').Pool(32) as pool:
        for m,N,lit,pan,eps in [r for r in pool.imap_unordered(run,tasks,chunksize=1) if r is not None]:
            print('GADGET m',m,'N',N,'L',lit,pan,'eps',eps,flush=True)
    print('done')
