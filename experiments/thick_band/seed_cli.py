"""Portable CPU-light planning / single-seed emitter for a Colab A100 driver.

No automatic CPU sweep. 'plan' uses only closed-form 45-degree capacity math.
'emit' constructs exactly ONE selected seed, returning .npz and metadata.
Main parameter generation and relaxation belong in the external A100 driver.
"""
import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
import numpy as np
from band_geometry import BandSpec, generate, augment_to_target

TARGETS = {908:30.637029559612035,909:30.656854249492582,
           910:30.690960120646093,911:30.707106781186727,
           971:31.67877987534328,972:31.692388155428386,
           973:31.70710678118685}


def rigid_capacity(side,width,length,offset=(-.25,-.25)):
    """Exact separating-halfplane capacity at theta=pi/4, zero phases.

    Complete triangle depth d has extreme upright vertex sum d+1.
    Band full extent is width/sqrt(2) along x+y and length/sqrt(2)
    along y-x. Thus d=floor(L - relevant_extent +/- offset - 1).
    Endpoint triangle depths are also capped by adjacent upright triangles.
    Axis-aligned corner grids require adjacent depths summing to at most floor(L).
    """
    ox,oy=offset
    d=[max(0,math.floor(v+1e-10)) for v in
       (side+ox+oy-width/math.sqrt(2)-1,
        side-ox-oy-width/math.sqrt(2)-1,
        side+oy-ox-length/math.sqrt(2)-1,
        side-oy+ox-length/math.sqrt(2)-1)]
    cap=max(0,math.floor(side+1e-10)-max(d[0],d[1]))
    d[2],d[3]=min(d[2],cap),min(d[3],cap)
    return width*length+sum(k*(k+1)//2 for k in d),d


def make_plan(targets, widths, margin=1e-6):
    out=[]
    for n in targets:
        L=TARGETS[n]-margin
        m=math.floor(L)
        entries=[]
        for w in widths:
            maxlen=math.floor(math.sqrt(2)*(L-.5)-w+1e-10)
            counts=[]
            for length in range(max(1,maxlen-3),maxlen+1):
                count,depths=rigid_capacity(L,w,length)
                counts.append(dict(length=length,capacity=count,
                                   deficit=n-count,corner_depths=depths))
            entries.append(dict(width=w,length_range=[max(1,maxlen-3),maxlen],
                                offset_range=[-.5,.5],phase_range=[-.5,.5],
                                angle_range=[math.pi/4-.03,math.pi/4+.03],
                                baseline_offset_capacity=counts))
        out.append(dict(n=n,side=L,base_grid=m*m,extras=n-m*m,
                        parameter_ranges=entries))
    return dict(version=1,targets=out,
                note="Capacity values are analytic rigid-family seeds, not claimed wins. "
                     "Negative deficit means a valid subpacking may be selected. "
                     "Positive deficit requires improved cap topology or GPU relaxation/insertion. "
                     "No target-count CSV is fabricated by duplicating or overlapping squares.")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sp=p.add_subparsers(dest="command",required=True)
    plan=sp.add_parser("plan")
    plan.add_argument("--targets",default="908,909,910,911,971,972,973")
    plan.add_argument("--widths",default="2,3,4,6,8,10,12")
    plan.add_argument("--margin",type=float,default=1e-6)
    plan.add_argument("--out",default="target-plan.json")
    emit=sp.add_parser("emit")
    emit.add_argument("--n",type=int,required=True,choices=tuple(TARGETS))
    emit.add_argument("--width",type=int,required=True)
    emit.add_argument("--length",type=int,required=True)
    emit.add_argument("--side",type=float)
    emit.add_argument("--angle",type=float,default=math.pi/4)
    emit.add_argument("--offset",type=float,nargs=2,default=(-.25,-.25))
    emit.add_argument("--phase",type=float,nargs=2,default=(0.,0.))
    emit.add_argument("--corner-depths",type=int,nargs=4)
    emit.add_argument("--cap",choices=("full","staggered"),default="full")
    emit.add_argument("--out",required=True)
    emit.add_argument("--augment",action="store_true",help="fill valid gaps then insert repair cells to reach n")
    a=p.parse_args()
    if a.command=="plan":
        plan=make_plan([int(n) for n in a.targets.split(",")],
                       [int(w) for w in a.widths.split(",")],a.margin)
        Path(a.out).write_text(json.dumps(plan,indent=2))
    else:
        L=a.side if a.side is not None else TARGETS[a.n]-1e-6
        trims=() if a.cap=="full" else tuple((i%2,(i+1)%2) for i in range(a.width))
        spec=BandSpec(L,a.width,a.length,a.angle,tuple(a.offset),tuple(a.phase),
                      corner_depths=tuple(a.corner_depths or ()),end_trims=trims)
        points,spec=generate(spec)
        augmentation=None
        if a.augment:
            points,augmentation=augment_to_target(points,spec,a.n)
        # Preserve every constructed square in overfull seeds. Driver chooses
        # deletion, target completion, perturbation, and GPU relaxation.
        np.savez(a.out,points=points,side=L,target_n=a.n)
        meta=dict(spec=asdict(spec),count=len(points),target_n=a.n,
                  deficit=a.n-len(points),
                  certified=False,screen="construction float SAT only")
        if augmentation is not None:meta["augmentation"]=augmentation
        Path(str(a.out)+".json").write_text(json.dumps(meta,indent=2))


if __name__=="__main__":
    main()
