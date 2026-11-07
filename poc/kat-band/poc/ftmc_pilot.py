"""One bounded FTMC pilot, no submission, no incumbent replacement.

MILP/LP stay host-side. SAT/contact quench on contiguous batched fp64 chains.
All prospective improvements require independent official interval certification.
"""
import os,time,json,tarfile,traceback
from pathlib import Path
import numpy as np
import torch
from band_relax import canonicalize,half_extent
from band_polish import extent,sat_check
from gensane_polish import quench,unit_feasible
from gensane_recovery import recover
from ftmc_moves import split,mutate
from core_refill import refill
from ribbon_archive import RibbonArchive
D=Path(__file__).resolve().parent;O=D/"ftmc-results";O.mkdir(exist_ok=True)
torch.set_num_threads(2)
def log(event,**k):
    row=dict(event=event,time=time.time(),**k)
    print(json.dumps(row,allow_nan=False),flush=True)
    with (O/"events.jsonl").open("a") as f:f.write(json.dumps(row,allow_nan=False)+"\n")
def main():
    assert torch.cuda.is_available() and "A100" in torch.cuda.get_device_name()
    raw=json.loads((D/"baseline.json").read_text())
    start=time.monotonic();end=start+float(os.getenv("FTMC_SECONDS","420"))
    torch.set_default_dtype(torch.float64)
    log("hardware",gpu=torch.cuda.get_device_name(),budget=end-start)
    archive=RibbonArchive(O/"ribbon-archive")
    for n in [911,973,908]:
        q0=np.array(raw[str(n)],dtype=np.float64)
        reference=extent(q0)
        t,L=canonicalize(torch.tensor(q0)[None])
        q=t[0].numpy()
        # One-ulp/near-contact conversion needs outward screen, never slack.
        safe,sL,ok=unit_feasible(t,clearance=2e-11,max_expand=1.000001)
        if not bool(ok[0]):raise ValueError("baseline float repair")
        q=safe[0].numpy();side=float(sL[0])
        core,u=split(q)
        archive_args=dict(structural_identity=f"baseline-{n}-core-order-v1",
                          outer_bounds=[0,0,reference,reference])
        archive.offer_components(core,**archive_args,contact_metadata={"source":"baseline"})
        identity=refill(core,side,side,len(u),hints=u[:,:2],candidate_cap=6000,time_limit=12)
        log("identity",n=n,diagnostics=identity.diagnostics,
            passed=identity.x is not None,reference_L=reference)
        if identity.x is None:continue
        # Strict identity screening is prerequisite to mutations on this n.
        assert len(identity.x)==n and sat_check(identity.x)[0]
        for lane in range(8):
            if time.monotonic()>end-25:break
            mode=["angle","shear","slide","taper"][lane%4]
            amplitude=[.0001,.001][lane//4]
            # Loosen SEARCH BOX only, not SAT/contact tolerance. Score remains
            # against original immutable reference, never this working side.
            work=side+[.005,.015][lane//4]
            c,hints,meta=mutate(q,side,seed=100*n+lane,mode=mode,amplitude=amplitude)
            c[:,:2]+=(work-side)/2
            h=hints[:,:2]+(work-side)/2
            x=torch.tensor(np.ascontiguousarray(c)[None],device="cuda")
            t0=time.monotonic()
            x,qi=quench(x,[work],steps=900,seconds=min(10,end-time.monotonic()-10),clearance=2e-10)
            anchor=x.detach().mean(dim=1)[:,:2]
            x,cL,valid=unit_feasible(x,clearance=2e-10,max_expand=1.01)
            # Full-layout canonicalization would move the isolated band into
            # a corner and invalidate all refill hints. Restore its position.
            x[:,:,:2]+=anchor[:,None,:]-x.mean(dim=1)[:,None,:2]
            hh=half_extent(x)[:,:,None]
            valid &= ((x[:,:,:2]-hh)>=0).all(dim=(1,2))
            valid &= ((x[:,:,:2]+hh)<=work).all(dim=(1,2))
            if not bool(valid[0]) or float(cL[0])>work:
                log("core_rejected",n=n,lane=lane,side=float(cL[0]),work=work)
                continue
            c=x[0].cpu().numpy()
            archive.offer_components(c,**archive_args,contact_metadata={"stage":"core","lane":lane})
            filled=refill(c,work,work,n-len(c),hints=h,candidate_cap=6000,time_limit=10)
            log("refill",n=n,lane=lane,meta=meta,diagnostics=filled.diagnostics)
            if filled.x is None:continue
            seed=np.ascontiguousarray(filled.x)
            assert len(seed)==n and sat_check(seed)[0]
            # Exact lattice seed then full contact welding, then host SLP.
            z=torch.tensor(seed[None],device="cuda")
            z,info=quench(z,[reference-1e-5],steps=1200,
                seconds=min(12,end-time.monotonic()-4),clearance=2e-9)
            z,zL,valid=unit_feasible(z,clearance=2e-9,max_expand=1.03)
            if not bool(valid[0]):continue
            candidate=z[0].cpu().numpy()
            r=recover(candidate,seconds=min(20,max(.1,end-time.monotonic()-2)),
                 iterations=6,position_trust=.02,angle_trust=.001,
                 target=reference-1e-8)
            if r["x"] is not None and r["L"]<extent(candidate):candidate=r["x"]
            # Refill keeps core rows first, and LP preserves row identities.
            archive.offer_components(candidate[:len(core)],**archive_args,
                contact_metadata={"stage":"post-LP","lane":lane})
            L=extent(candidate);screen=sat_check(candidate)[0]
            np.savez_compressed(O/f"n{n}-lane{lane}.npz",x=candidate,L=L,
                reference_L=reference,meta=json.dumps(meta))
            log("candidate",n=n,lane=lane,L=L,reference_L=reference,
                delta=reference-L,screen=screen,seconds=time.monotonic()-t0,
                prospective_win=bool(screen and L<reference-1e-9),
                needs_official_certificate=True)
    log("finished",seconds=time.monotonic()-start)
if __name__=="__main__":
    try:main()
    except BaseException:
        (O/"ERROR.txt").write_text(traceback.format_exc());raise
    finally:
        with tarfile.open(D/"ftmc-results.tar.gz","w:gz") as f:f.add(O,arcname="ftmc-results")
