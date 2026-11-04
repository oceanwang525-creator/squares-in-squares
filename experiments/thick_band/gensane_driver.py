"""A100-only measured recovery gate and bounded perturb/walk/quench search.

Run in a staged directory containing baseline.json. The outer owner MUST stop
the Colab assignment even on interpreter failure. No submission or push.
"""
import os,sys,time,json,traceback,tarfile
from pathlib import Path
os.environ.setdefault("OMP_NUM_THREADS","2")
import numpy as np
import torch
from band_relax import validate, Config as GateConfig,canonicalize
from gensane_polish import quench,unit_feasible,collision_loss
from gensane import relax,Config
from gensane_seeds import build_portfolio
D=Path(__file__).resolve().parent
OUT=D/"gensane-results";OUT.mkdir(exist_ok=True)
torch.set_num_threads(2)
GATE=GateConfig(tile=1024,batch_chunk=4)
def log(event,**values):
    row=dict(event=event,time=time.time(),**values)
    with (OUT/"events.jsonl").open("a") as f:f.write(json.dumps(row,allow_nan=False)+"\n")
    print(json.dumps(row,allow_nan=False),flush=True)
def save(name,x,L,**meta):
    p=OUT/(name+".npz");tmp=p.with_suffix(".tmp.npz")
    np.savez_compressed(tmp,x=np.asarray(x),L=np.asarray(L),metadata=json.dumps(meta))
    os.replace(tmp,p)
def main():
    assert torch.cuda.is_available() and "A100" in torch.cuda.get_device_name()
    torch.set_default_dtype(torch.float64)
    start=time.monotonic();hard=start+float(os.getenv("SEARCH_SECONDS","7200"))
    raw=json.loads((D/"baseline.json").read_text())
    targets=[908,909,910,911,971,972,973]
    baselines={}
    for n in targets:
        q=torch.tensor(raw[str(n)],device="cuda",dtype=torch.float64)[None]
        q,L=canonicalize(q)
        # preserve source side as objective, but positive float contacts must not
        # knock the published baseline out of an immutable incumbent.
        original=float(L[0]);q,padded,ok=unit_feasible(q,clearance=2e-11,max_expand=1.000001)
        if not bool(ok.all()):raise RuntimeError("baseline float repair failed")
        baselines[n]=(q[0].cpu().numpy(),float(padded[0]),original)
        save(f"n{n}-best",q.cpu().numpy(),padded.cpu().numpy(),source="safe baseline screen",n=n)
    log("hardware",gpu=torch.cuda.get_device_name(),dtype=str(torch.get_default_dtype()),
        budget_seconds=hard-start,memory_gb=torch.cuda.get_device_properties(0).total_memory/1e9)
    # Recovery is a mandatory experiment before spending hours searching.
    q,L,_=baselines[908];report=[]
    rng=np.random.default_rng(27011)
    for sigma in [1e-4,1e-3,.005,.02]:
        z=np.repeat(q[None],2,0)+rng.normal(0,sigma,(2,len(q),3))
        z=torch.tensor(z,device="cuda")
        t=time.monotonic()
        repaired,info=quench(z,[L,L],steps=1600,seconds=35,clearance=2e-11)
        repaired,side,ok=unit_feasible(repaired,clearance=2e-11,max_expand=5)
        excess=(side-L).cpu().numpy()
        row=dict(sigma=sigma,seconds=time.monotonic()-t,loss=info["loss"],
            evaluations=info["evaluations"],valid=ok.cpu().tolist(),
            excess=excess.tolist(),recover_1e7=(ok&(side<L+1e-7)).cpu().tolist())
        report.append(row);log("recovery",**row)
    (OUT/"recovery.json").write_text(json.dumps(report,indent=2))
    # Fail fast if even a tiny global perturbation cannot be re-jammed.
    if not all(report[0]["recover_1e7"]):
        log("gate_failed",reason="tiny perturb recovery worse than 1e-7; long run forbidden")
        return
    log("gate_passed",recovered=sum(sum(r["recover_1e7"]) for r in report),trials=8)
    incumbents={n:(q.copy(),L) for n,(q,L,_) in baselines.items()}
    cycle=0
    while time.monotonic()<hard-40:
        n=targets[cycle%len(targets)];q,L=incumbents[n]
        t=time.monotonic()
        seeds,metadata=build_portfolio(q,L,count=12,rng=4200+cycle)
        # Seed candidates can overlap. Quench before trying maximal uniform
        # scaling, which cannot cure exact coincident centers.
        seed_t=torch.tensor(seeds,device="cuda")
        target=torch.full((len(seeds),),L,device="cuda")
        seed_t,info=quench(seed_t,target,steps=900,seconds=min(55,hard-time.monotonic()-10),clearance=2e-11)
        y,sides,ok=unit_feasible(seed_t,clearance=2e-11,max_expand=100)
        valid,_,_=validate(y,sides,GATE)
        # Lane zero is always restored exactly so experimental failures cannot
        # erase the immutable incumbent.
        y[0]=torch.tensor(q,device="cuda");sides[0]=L
        def hook(x,box,best,end):
            remaining=min(25,end-time.monotonic())
            if remaining<1:return x,box
            # Direct fixed-baseline target allows crossing contact topology;
            # acceptance still needs strict all-pair postrepair projection.
            shrink=[1e-5,1e-4,.001,.003][cycle%4]
            lens=torch.full_like(box,L*(1-shrink))
            z=x.clone()
            z[...,:2]*=(lens/box)[:,None,None]
            z,_=quench(z,lens,steps=700,seconds=remaining,clearance=2e-11)
            z,ll,usable=unit_feasible(z,clearance=2e-11,max_expand=100)
            z=torch.where(usable[:,None,None],z,x)
            return z,torch.where(usable,ll,box)
        r=relax(y.cpu().numpy(),sides.cpu().numpy(),device="cuda",
            config=Config(seconds=min(70,hard-time.monotonic()-5),seed=91+cycle,
              walks_per_round=2048,inflate_every=512,restart_every=0,
              global_xy=[.001,.01,.03][cycle%3],global_angle=[.001,.01,.03][cycle%3],
              tile=1024,batch_chunk=4,checkpoint_every=30),polish_hook=hook)
        good=np.isfinite(r["L"])&r["float_valid"]
        chosen=int(np.argmin(r["L"]))
        new=float(r["L"][chosen])
        if good[chosen] and new<L:
            incumbents[n]=(r["x"][chosen].copy(),new)
            save(f"n{n}-best",r["x"][chosen:chosen+1],[new],n=n,cycle=cycle,source="new float screen")
            log("prospective_win",n=n,old_L=L,new_L=new,delta=L-new,
                needs_official_certificate=True)
        save(f"n{n}-cycle",r["x"],r["L"],n=n,cycle=cycle,metadata=metadata,
             rounds=r["rounds"],steps=r["steps"],accepted=r["accepted"])
        log("cycle",n=n,cycle=cycle,seconds=time.monotonic()-t,
            best_L=incumbents[n][1],rounds=r["rounds"],steps=r["steps"],accepted=r["accepted"])
        cycle+=1
    log("finished",cycles=cycle,seconds=time.monotonic()-start,
        best={str(n):v[1] for n,v in incumbents.items()})
if __name__=="__main__":
    try:main()
    except BaseException:
        (OUT/"ERROR.txt").write_text(traceback.format_exc());traceback.print_exc();raise
    finally:
        with tarfile.open(D/"gensane-results.tar.gz","w:gz") as f:f.add(OUT,arcname="gensane-results")
