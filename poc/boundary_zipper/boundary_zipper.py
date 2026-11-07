"""Bounded boundary-connected cardinality augmentation, CPU proposal only.

A full movable perimeter connects a two-cell zigzag seam to both endpoint caps.
Independent displacement labels and opposing wall residue phases replace a
binary rigid-band shift. A finite conflict MILP asks for removed+1 squares in
the side of donor n-1. It is not a continuous infeasibility proof.
"""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import argparse
import csv
import io
import json
import math
import time
from decimal import Decimal
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree
from squares.core import certify, grid
from squares._official_metric import _parse_submission_csv

HERE = Path(__file__).resolve().parent
DEFAULTS = [Path("/home/kit/out/public-20261104/merged-material-candidate.csv"),
            HERE.parent / "shared/merged-535.csv"]

def load(path, ns):
    result = {n: [] for n in ns}
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            n = int(row["n"])
            if n in result:
                result[n].append({k: row[k] for k in ("square_id", "x", "y", "angle")})
    for n, rows in result.items():
        if len(rows) != n:
            raise ValueError(f"missing or wrong source count for {n}")
    return result

def array(rows):
    return np.array([[float(r[k]) for k in ("x", "y", "angle")] for r in rows])

def bounds(q):
    h = .5 * (np.abs(np.cos(q[:, 2])) + np.abs(np.sin(q[:, 2])))
    return (q[:, :2] - h[:, None]).min(0), (q[:, :2] + h[:, None]).max(0)

def overlap(a, b):
    """Permissive proposal screening only; all exports independently certified."""
    if len(b) == 0:
        return np.zeros(0, dtype=bool)
    u = np.array([math.cos(a[2]), math.sin(a[2])])
    v = np.array([-u[1], u[0]])
    bu = np.column_stack((np.cos(b[:, 2]), np.sin(b[:, 2])))
    bv = np.column_stack((-bu[:, 1], bu[:, 0]))
    delta = b[:, :2] - a[:2]
    hit = np.ones(len(b), dtype=bool)
    for axis in (np.broadcast_to(u, bu.shape), np.broadcast_to(v, bu.shape), bu, bv):
        ra = .5 * (np.abs(axis @ u) + np.abs(axis @ v))
        rb = .5 * (np.abs(np.sum(axis * bu, 1)) + np.abs(np.sum(axis * bv, 1)))
        hit &= np.abs(np.sum(delta * axis, 1)) < ra + rb - 2e-9
    return hit

def seam_mask(q, lo, hi, variant):
    # Whole perimeter is free, including every frozen extremizer from prior LNS.
    # The stair seam runs continuously from bottom to top and joins the shell.
    p = q[:, :2] - lo
    W, H = hi - lo
    shell = np.min(np.column_stack((p[:, 0], W-p[:, 0], p[:, 1], H-p[:, 1])), axis=1) < 1.65
    center = W * (.40 if variant == 0 else .60)
    step = np.where((np.floor(p[:, 1] / 2).astype(int) % 2) == 0, -.45, .45)
    seam = np.abs(p[:, 0] - center - step) < 1.65
    return shell | seam

def connected_cohort(q, mask):
    """Retain only components linked to the shell, via conservative contact graph.

    Distance <=2.25 joins touching unit squares at arbitrary orientations.
    This establishes boundary connection of the evacuation, not feasible moves.
    """
    lo, hi = bounds(q)
    p = q[:, :2]
    boundary = np.min(np.column_stack((p[:,0]-lo[0],hi[0]-p[:,0],
                                      p[:,1]-lo[1],hi[1]-p[:,1])),axis=1) < 1.65
    ids = np.flatnonzero(mask)
    tree = cKDTree(p[ids])
    seen = set(np.flatnonzero(boundary[ids]).tolist())
    stack = list(seen)
    while stack:
        i = stack.pop()
        for j in tree.query_ball_point(p[ids[i]], 2.25):
            if j not in seen:
                seen.add(j)
                stack.append(j)
    out = np.zeros(len(q), bool)
    out[ids[sorted(seen)]] = True
    return out

def propose(rows, variant, solve_seconds, deadline, cap=12000):
    begin = time.monotonic()
    q = array(rows)
    lo, hi = bounds(q)
    mask = connected_cohort(q, seam_mask(q, lo, hi, variant))
    old, frozen = q[mask], q[~mask]
    residues = sorted(set([round(float((hi-lo)[0] % 1), 8),
                          round(math.sqrt(2)-1, 8), .5]))
    raw = [r.copy() for r in old]
    # Every square may choose its own displacement. Original states are optional.
    for r in old:
        for delta in residues:
            for axis in (0, 1):
                for sign in (-1, 1):
                    z = r.copy()
                    z[axis] += sign * delta
                    raw.append(z)
        # Rotation replacement states can exchange a seam/cap role.
        for angle in (0., math.pi/4):
            z = r.copy(); z[2] = angle; raw.append(z)
    # Opposing wall phases keep the exact leftover residue in the endpoint cap.
    # Enumerate wall-aligned upright cap chains and diagonal zipper teeth.
    for angle in (0., math.pi/4):
        h = .5*(abs(math.cos(angle))+abs(math.sin(angle)))
        u = np.array([math.cos(angle), math.sin(angle)])
        v = np.array([-u[1], u[0]])
        for sx, sy in ((0,0),(1,0),(0,1),(1,1)):
            anchor = np.where([sx,sy], hi-h, lo+h)
            for i in range(-2, int(math.ceil(max(hi-lo)))+2):
                for j in range(-2, 3):
                    xy = anchor + i*u + j*v
                    raw.append(np.r_[xy, angle])
                    raw.append(np.r_[xy[::-1] + (lo-lo[::-1]), angle])
    candidates = []
    seen = set()
    ftree = cKDTree(frozen[:, :2]) if len(frozen) else None
    for z in raw:
        if time.monotonic() > deadline:
            return None, dict(reason="generation_budget", removed=len(old))
        key = tuple(np.round(z, 9))
        if key in seen:
            continue
        seen.add(key)
        h = .5*(abs(math.cos(z[2]))+abs(math.sin(z[2])))
        if np.any(z[:2]-h < lo-3e-9) or np.any(z[:2]+h > hi+3e-9):
            continue
        near = ftree.query_ball_point(z[:2], 1.415) if ftree else []
        if near and overlap(z, frozen[near]).any():
            continue
        candidates.append(z)
        if len(candidates) >= cap:
            break
    c = np.array(candidates)
    if len(c) == 0:
        return None, dict(reason="no_candidates", removed=len(old))
    tree = cKDTree(c[:, :2])
    edges = []
    for i in range(len(c)):
        near = np.array([j for j in tree.query_ball_point(c[i,:2], 1.415) if j > i], dtype=int)
        if len(near):
            edges.extend((i,int(j)) for j in near[overlap(c[i],c[near])])
    m = len(c)
    # Exactly old+1; no copying target incumbent, no same-count success.
    rr, cc = [], []
    for k, (i,j) in enumerate(edges):
        rr.extend([k,k]); cc.extend([i,j])
    rr.extend([len(edges)]*m); cc.extend(range(m))
    A = coo_matrix((np.ones(len(rr)), (rr,cc)),
                   shape=(len(edges)+1,m)).tocsc()
    lower = np.full(len(edges)+1, -np.inf)
    upper = np.ones(len(edges)+1)
    lower[-1] = upper[-1] = len(old)+1
    remaining = min(solve_seconds, deadline-time.monotonic())
    candidate_keys = {tuple(np.round(v,9)) for v in c}
    identity_retained = sum(tuple(np.round(z,9)) in candidate_keys for z in old)
    identity_conflicts = sum(i < len(old) and j < len(old) for i,j in edges)
    assert identity_retained == len(old) and identity_conflicts == 0, "donor recovery control failed"
    info = dict(removed=len(old), frozen=len(frozen), candidates=m, edges=len(edges),
                target_refill=len(old)+1, residues=residues, cap_hit=len(c)>=cap,
                all_four_extrema_free=True, boundary_connected=True,
                identity_candidates_retained=identity_retained,
                identity_conflicts=identity_conflicts)
    if remaining <= 0:
        return None, dict(info, reason="graph_budget")
    # Small preference for later states, but count is an equality constraint.
    result = milp(np.linspace(0., -.001, m), integrality=np.ones(m),
                  bounds=Bounds(0,1), constraints=LinearConstraint(A,lower,upper),
                  options=dict(time_limit=remaining, mip_rel_gap=0.))
    info.update(status=int(result.status), message=str(result.message),
                elapsed=time.monotonic()-begin)
    if result.x is None:
        return None, dict(info, reason="no_plus_one_solution")
    chosen = c[result.x > .5]
    if len(chosen) != len(old)+1:
        return None, dict(info, reason="wrong_count")
    out = np.vstack((frozen, chosen))
    # Protect touching imported coordinates through real serialization.
    # All squares move outward slightly, only after search; no unchanged-L claim.
    out[:, :2] = (out[:, :2]-lo)*1.00000002
    emitted = [dict(square_id=str(i), x=format(z[0],".17g"),
                    y=format(z[1],".17g"), angle=format(z[2],".17g"))
               for i,z in enumerate(out)]
    return emitted, dict(info, reason="uncertified_plus_one_proposal",
                         inserted=len(chosen))

def dump_csv(path, puzzles):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("id","n","square_id","x","y","angle"))
        writer.writeheader()
        index = 0
        for n, rows in sorted(puzzles.items()):
            assert len(rows) == n
            for row in rows:
                writer.writerow(dict(row, id=index, n=n))
                index += 1

def check_file(path):
    """Use unchanged core.certify and actual file/pandas parser in BOTH modes."""
    literal = _parse_submission_csv(path.read_text())
    transported_text = pd.read_csv(path).to_csv(index=False)
    path.with_suffix(".pandas.csv").write_text(transported_text)
    transported = _parse_submission_csv(transported_text)
    result = {}
    for n in literal:
        result[n] = {}
        for mode, puzzles in (("literal",literal),("pandas",transported)):
            try:
                result[n][mode] = dict(ok=True, L=certify(n,puzzles[n]))
            except ValueError as error:
                result[n][mode] = dict(ok=False, error=str(error))
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=next((p for p in DEFAULTS if p.exists()),DEFAULTS[0]))
    parser.add_argument("--outdir", type=Path, default=HERE)
    parser.add_argument("--targets", default="908,910,972")
    parser.add_argument("--seconds", type=float, default=95.)
    parser.add_argument("--solve-seconds", type=float, default=9.)
    args = parser.parse_args()
    targets = list(map(int,args.targets.split(",")))
    if len(set(targets)) != len(targets) or not all(2<=n<=1000 for n in targets):
        parser.error("targets must be distinct numbers in 2..1000")
    args.outdir.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    data = load(args.baseline, set(targets)|{n-1 for n in targets})
    deadline = time.monotonic()+args.seconds
    logs = []
    output = {}
    summary = {}
    # Always construct a valid exact-count fallback. It is not an incumbent replay.
    for n in targets:
        output[n] = grid(n)
        summary[n] = dict(output_kind="constructive_grid_fallback",
                          augmentation_succeeded=False, exact_count=n)
    for n in targets:
        for variant in range(2):
            if time.monotonic() >= deadline:
                logs.append(dict(n=n,variant=variant,reason="overall_search_budget")); continue
            candidate, info = propose(data[n-1],variant,args.solve_seconds,deadline)
            info.update(n=n, donor=n-1, variant=variant)
            if candidate is not None:
                path=args.outdir/f"proposal-n{n}-v{variant}.csv"
                dump_csv(path,{n:candidate})
                evidence=check_file(path)
                info["certification"]=evidence[n]
                if all(evidence[n][mode]["ok"] for mode in ("literal","pandas")):
                    newL=Decimal(evidence[n]["literal"]["L"])
                    if summary[n]["output_kind"]=="constructive_grid_fallback" or newL<Decimal(summary[n]["candidate_L"]):
                        output[n]=candidate
                        summary[n].update(output_kind="certified_boundary_zipper_augmentation",
                                          augmentation_succeeded=True,candidate_L=str(newL))
            print(json.dumps(info),flush=True)
            logs.append(info)
    dump_csv(args.outdir/"baseline-targets.csv",{n:data[n] for n in targets})
    baseline_evidence=check_file(args.outdir/"baseline-targets.csv")
    dump_csv(args.outdir/"output.csv",output)
    output_evidence=check_file(args.outdir/"output.csv")
    for n in targets:
        summary[n]["baseline"]=baseline_evidence[n]
        summary[n]["output"]=output_evidence[n]
        comparisons={}
        for mode in ("literal","pandas"):
            b=baseline_evidence[n][mode]; o=output_evidence[n][mode]
            if b["ok"] and o["ok"]:
                delta=Decimal(o["L"])-Decimal(b["L"])
                comparisons[mode]=dict(delta_L=str(delta), strict_win=delta<0)
        summary[n]["comparison"]=comparisons
        summary[n]["verified_win"]=bool(len(comparisons)==2 and
                            all(c["strict_win"] for c in comparisons.values()))
    report=dict(baseline=str(args.baseline), targets=targets, trials=logs,
                summary=summary, elapsed_seconds=time.monotonic()-start,
                search_budget_seconds=args.seconds, pandas_version=pd.__version__,
                note="Partial CSV only. Missing n not scored. No GPU, C++ check, submission or git operations.")
    (args.outdir/"results.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(elapsed=report["elapsed_seconds"],summary=summary)),flush=True)

if __name__=="__main__":
    main()
