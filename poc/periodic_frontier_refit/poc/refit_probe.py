#!/usr/bin/env python3
"""Quick probe for contact-fitted periodic-block refits."""
from __future__ import annotations
import argparse, collections, json, subprocess, sys, time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[3]
for rel in ("poc/baseline/poc/src", "poc/hydrogen_brainstorm/poc", "poc/swarm_homology/poc",
            "poc/swarm_highd/poc", "poc/swarm_new159/poc", "poc/rectangle_gadgets/poc",
            "poc/swarm_brickslp/poc"):
    p = str(REPO / rel)
    if p not in sys.path: sys.path.insert(0, p)
import squares.core as core
from geom_utils import sat_max_overlap_and_bbox
from joint_slp import joint_xytheta_slp
from brick_slp import rect_bbox, relax_rect_lbfgs

def read_baseline(tree, filename):
    return core.metric._parse_submission_csv(subprocess.check_output(["git", "show", f"{tree}:{filename}"], text=True))

def components(x, y, th):
    non = np.minimum(np.abs(np.cos(th)), np.abs(np.sin(th))) > 1e-5
    ids = np.flatnonzero(non); parent = {i: i for i in ids}
    def find(a):
        while parent[a] != a: parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for p, i in enumerate(ids):
        for j in ids[p+1:]:
            if (x[i]-x[j])**2 + (y[i]-y[j])**2 < 1.9**2:
                a,b = find(i), find(j)
                if a != b: parent[b] = a
    g = collections.defaultdict(list)
    for i in ids: g[find(i)].append(i)
    return sorted(g.values(), key=len, reverse=True)

def joint_probe(rows):
    x = np.array([float(r["x"]) for r in rows]); y = np.array([float(r["y"]) for r in rows])
    th = np.array([float(r["angle"]) for r in rows]); t0 = time.perf_counter()
    nx,ny,nth,_,ok = joint_xytheta_slp(x,y,th,rounds=8,tr_th0=.01,tr_xy0=.03,margin=1e-9,tol=1e-9,patience=2)
    pen,side = sat_max_overlap_and_bbox(nx,ny,nth)
    return {"feasible": bool(ok and pen <= 1e-9), "side": side if pen <= 1e-9 else None,
            "penetration": float(pen), "seconds": time.perf_counter()-t0}

def patch_probe(rows, shrink):
    x = np.array([float(r["x"]) for r in rows]); y = np.array([float(r["y"]) for r in rows])
    th = np.array([float(r["angle"]) for r in rows]); ans=[]
    for ids in components(x,y,th):
        g = np.array(ids); px,py,w,h = rect_bbox(x[g],y[g],th[g])
        rx,ry,rth,_ = relax_rect_lbfgs(px,py,th[g],w*(1-shrink),h*(1-shrink),maxiter=3000)
        pen,_ = sat_max_overlap_and_bbox(rx,ry,rth)
        ans.append({"squares":len(g),"original_w":float(w),"original_h":float(h),
                    "isolated_feasible":bool(pen<=1e-10),"isolated_pen":float(pen)})
    return ans

p = argparse.ArgumentParser(); p.add_argument("--baseline-tree", default="238087b9787a14b8ad597d697a7319d604d74b5f")
p.add_argument("--baseline-file", default="merged-22.15325720501.csv"); p.add_argument("--targets", default="380,701,846,979")
p.add_argument("--patch-shrink", type=float, default=.05)
p.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "docs" / "quick_summary.json")
a = p.parse_args(); puzzles = read_baseline(a.baseline_tree,a.baseline_file); report={}
for tok in a.targets.split(","):
    n=int(tok); rows=puzzles[n]; _,_,base,reason=core.evaluate(n,rows)
    report[n]={"baseline_side":base,"baseline_reason":reason,"joint_xytheta":joint_probe(rows),
               "periodic_patch_isolated":patch_probe(rows,a.patch_shrink)}
    print(n,json.dumps(report[n],separators=(",",":")))
a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(report,indent=2)+"\n")
