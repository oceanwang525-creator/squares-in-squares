"""Standalone CUDA-only bounded search; CPU use is limited to unit tests."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
import torch
from band_relax import Config, relax

def save_checkpoint(path, result):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+".tmp")
    metadata={k:v for k,v in result.items() if not isinstance(v,np.ndarray)}
    with tmp.open("wb") as f:
        np.savez_compressed(f,**{k:v for k,v in result.items() if isinstance(v,np.ndarray)},
                            metadata=np.array(json.dumps(metadata)))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp,path)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("input",help="NPZ with x[B,N,3], L[B] or scalar, optional band_mask[B,N]")
    p.add_argument("--output",required=True)
    p.add_argument("--seconds",type=float,default=60)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--repair-steps",type=int,default=180)
    p.add_argument("--checkpoint-every",type=float,default=15)
    p.add_argument("--device",default="cuda")
    args=p.parse_args()
    dev=torch.device(args.device)
    if dev.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit("Main search requires CUDA. CPU only permitted in tiny unit tests.")
    print(json.dumps({"gpu":torch.cuda.get_device_name(dev),"torch":torch.__version__}),flush=True)
    with np.load(args.input,allow_pickle=False) as z:
        x=z["x"].copy()
        lengths=z["L"].copy()
        mask=z["band_mask"].copy() if "band_mask" in z else None
    cfg=Config(seconds=args.seconds,seed=args.seed,repair_steps=args.repair_steps,
               checkpoint_every=args.checkpoint_every)
    result=relax(x,lengths,device=dev,config=cfg,band_mask=mask,
                 checkpoint=lambda r:save_checkpoint(args.output,r))
    print(json.dumps({"L":result["L"].tolist(),
                      "float_valid":result["float_valid"].tolist(),
                      "rounds":result["rounds"],"steps":result["steps"],
                      "elapsed":result["elapsed"],
                      "certificate":result["certificate"]}),flush=True)

if __name__=="__main__":
    main()
