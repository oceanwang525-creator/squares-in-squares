"""CPU split-angle finite ribbon with explicit caps and corner-grid phases.

A proposal generator only. All exports use unchanged official certificates.
No baseline coordinates are used in any generated packing.
"""
from __future__ import annotations
import argparse
import csv
import io
import json
import math
import sys
import time
from decimal import Decimal
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
from squares.core import certify
from squares import _official_metric as metric

def paired_overlap(a, b, tol=1e-10):
    if not len(a):
        return np.zeros(0, dtype=bool)
    ua = np.column_stack((np.cos(a[:, 2]), np.sin(a[:, 2])))
    va = np.column_stack((-ua[:, 1], ua[:, 0]))
    ub = np.column_stack((np.cos(b[:, 2]), np.sin(b[:, 2])))
    vb = np.column_stack((-ub[:, 1], ub[:, 0]))
    delta = a[:, :2] - b[:, :2]
    bad = np.ones(len(a), dtype=bool)
    for axis in (ua, va, ub, vb):
        r = sum(np.abs((axis*v).sum(1)) for v in (ua, va, ub, vb))/2
        bad &= np.abs((delta*axis).sum(1)) < r-tol
    return bad

def conflicts(q):
    pairs = np.array(list(cKDTree(q[:, :2]).query_pairs(1.415)), dtype=int).reshape(-1, 2)
    if not len(pairs):
        return pairs
    return pairs[paired_overlap(q[pairs[:, 0]], q[pairs[:, 1]])]

def inside(q, side):
    h = (np.abs(np.cos(q[:, 2]))+np.abs(np.sin(q[:, 2])))/2
    return ((q[:, :2] >= h[:, None]-1e-10) &
            (q[:, :2] <= side-h[:, None]+1e-10)).all(1)

def fill_corners(q, side, offset=0.0):
    """Four corner-anchored phases. Each entire phase is internally disjoint."""
    additions = []
    k = math.floor(side)
    xx, yy = np.meshgrid(np.arange(k)+.5, np.arange(k)+.5)
    raw = np.column_stack((xx.ravel(), yy.ravel(), np.zeros(k*k)))
    for cx, cy in ((0,0),(1,1),(0,1),(1,0)):
        sites = raw.copy()
        # Phase offset is applied only in the second-level opposite corners.
        phase = offset if cx != cy else 0.0
        sites[:, :2] += phase
        if cx:
            sites[:, 0] = side-sites[:, 0]
        if cy:
            sites[:, 1] = side-sites[:, 1]
        sites = sites[inside(sites, side)]
        tree = cKDTree(q[:, :2])
        near = tree.query_ball_point(sites[:, :2], 1.415)
        counts = np.array([len(v) for v in near])
        ai = np.repeat(np.arange(len(sites)), counts)
        bi = np.array([i for ids in near for i in ids], dtype=int)
        bad = np.zeros(len(sites), dtype=bool)
        if len(ai):
            bad[ai[paired_overlap(sites[ai], q[bi])]] = True
        good = sites[~bad]
        additions.append(dict(corner=[cx,cy], phase=phase, count=len(good)))
        q = np.concatenate((q, good))
    return q, additions

def hinged(side, width, delta, slide, offset, cap):
    """Two independent oblique subbands meet at a sliding angular hinge.

    The first has angle pi/4-delta, the second pi/4+delta.
    Longitudinal centers follow each subband's own frame, not a straight
    ribbon with changed square orientations. Explicit asymmetric end caps
    and a transverse hinge slide are geometry parameters.
    """
    out, labels = [], []
    length = math.ceil(side*math.sqrt(2))+2
    for half in (-1, 1):
        theta = math.pi/4 + half*delta
        u = np.array([math.cos(theta), math.sin(theta)])
        v = np.array([-math.sin(theta), math.cos(theta)])
        origin = np.array([side/2+offset, side/2-offset])
        origin += half*slide*u/2
        for i in range(width):
            # A nonrectangular endpoint cap, independent at each end.
            trim = ((i + (half > 0)) % 2)*cap
            for j in range(length//2-trim):
                d = half*(j+.5)
                xy = origin+(i-(width-1)/2)*u+d*v
                out.append((*xy, theta))
                labels.append(half)
    q = np.array(out)
    labels = np.array(labels)
    keep = inside(q, side)
    q, labels = q[keep], labels[keep]
    # Invalid hinge intersections are explicit finite cap deletions, never
    # hidden overlap repair. Keep the lower-degree endpoint deterministically.
    edges = conflicts(q)
    active = np.ones(len(q), dtype=bool)
    removed = []
    while len(edges):
        degree = np.bincount(edges.ravel(), minlength=len(q))
        k = int(np.argmax(degree))
        removed.append(k)
        active[k] = False
        edges = edges[(edges != k).all(1)]
    q, labels = q[active], labels[active]
    return q, dict(width=width, delta=delta, slide=slide, offset=offset,
                   end_cap=cap, hinge_removed_indices=removed,
                   half_counts=[int(sum(labels == -1)), int(sum(labels == 1))],
                   rotated_count=len(q))

def atom17():
    """Source Hamalainen17 alternate, finite.py / square-17_r2.svg."""
    z = math.sqrt(2)
    L = (7+5*z)/3
    a = [(.5,.5,0),(L-.5,.5,0),
         (L-.5,L-.5,0),(L-1.5,L-.5,0),(L-.5,L-1.5,0),
         (.5,L-.5,0),(1.5,L-.5,0),(.5,L-1.5,0),(.5,L-2.5,0)]
    for x,y,m in [((7+2*z)/6,.5+2*z,3),(1,(2+4*z)/3,3),((5-2*z)/3,z,2)]:
        for j in range(m):
            u,v=j+.5,.5
            a.append((x+(u+v)/z,y+(-u+v)/z,-math.pi/4))
    return np.array(a)

def atom37():
    """Source Cantrell37 local geometry, finite.py / square-37.svg."""
    L = 6.59861960924436011617837046268
    th = math.radians(47.9133925257940968951470647263)
    x2, y2 = .7022366221116755433, 2.3296961491127976962
    r1, r2 = .07187936019410241754, .92812063980589758246
    a = [(L-i-.5,j+.5,0) for i in range(3) for j in range(3-i)]
    block = [(.5,.5,0),(1.5,.5,0),(.5,1.5,0)]
    local = [(.5,.5),(1.5,r1+.5),(2.5,r1+.5),
             (r2+.5,1+r1+.5),(r2+1.5,1+r1+.5),(-r1+.5,1.5)]
    for u,v in local:
        block.append((x2+u*math.cos(th)+v*math.sin(th),
                      y2-u*math.sin(th)+v*math.cos(th),-th))
    a += block
    a += [(L-y,L-x,math.pi/2-t) for x,y,t in block]
    a += [(i+.5,L-j-.5,0) for i in range(4) for j in range(4-i)]
    for j in range(3):
        u,v = .5,j+.5
        a.append((2+(u+v)/math.sqrt(2),
                  L-(2+1/math.sqrt(2)+(-u+v)/math.sqrt(2)),math.pi/4))
    assert len(a) == 37
    return np.array(a)

def fallback(n):
    """Explicit rectangular/corner grammar, not incumbent replay.

    Independently sliding interlocking17/37 atoms occupy opposite corners.
    Upright cells complete four corner phases. Atom clearance is explicit.
    """
    side = math.ceil(math.sqrt(n))
    a = atom17()
    b = atom37()
    a[:, :2] += (.125, .0)
    b[:, :2] = side-b[:, :2]
    b[:, 0] -= .125
    q = np.concatenate((a,b))
    p, caps = fill_corners(q, side, .125)
    if len(p) < n or len(conflicts(p)):
        raise RuntimeError("constructive corner fallback failed")
    return p[:n], dict(kind="valid_constructive_fallback", atoms=["Hamalainen17","Cantrell37"],
                       independent_slides=[.125,-.125], corner_caps=caps,
                       constructive_count_before_deletion=len(p),
                       rotated_count=int(np.count_nonzero(q[:,2])), side=side)

def rows(q):
    # Center-only dilation is genuine output geometry, not a metric tolerance.
    q = q.copy()
    q[:, :2] -= q[:, :2].mean(0)
    q[:, :2] *= 1.00000002
    return [dict(square_id=str(i), x=format(x,'.17g'), y=format(y,'.17g'),
                 angle=format(a,'.17g')) for i,(x,y,a) in enumerate(q)]

def csv_text(puzzles):
    f = io.StringIO()
    w = csv.DictWriter(f, fieldnames=("id","n","square_id","x","y","angle"))
    w.writeheader()
    idx = 0
    for n, rs in sorted(puzzles.items()):
        for r in rs:
            w.writerow(dict(id=idx,n=n,**{k:r[k] for k in ('square_id','x','y','angle')}))
            idx += 1
    return f.getvalue()

def certify_transport(text, path=None):
    literal = metric._parse_submission_csv(text)
    transported_text = pd.read_csv(io.StringIO(text)).to_csv(index=False)
    transported = metric._parse_submission_csv(transported_text)
    if path:
        path.write_text(transported_text)
    result = {}
    for n in literal:
        entry = {}
        for mode, puzzles in (("literal", literal), ("actual_pandas", transported)):
            try:
                entry[mode] = dict(ok=True, L=certify(n, puzzles[n]))
            except ValueError as e:
                entry[mode] = dict(ok=False, error=str(e))
        result[str(n)] = entry
    return result

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", type=Path, default=Path("/home/kit/out/public-20261104/merged-material-candidate.csv"))
    ap.add_argument("--out", type=Path, default=HERE)
    ap.add_argument("--seconds", type=float, default=100)
    ap.add_argument("--targets", nargs="+", type=int, default=[908,910,972])
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    wanted = set(args.targets)
    baseline = {}
    with args.baseline.open() as f:
        for r in csv.DictReader(f):
            n = int(r["n"])
            if n in wanted:
                baseline.setdefault(n, []).append(r)
    # Certify the true comparator instead of using float-measured incumbents.
    base_cert = certify_transport(csv_text(baseline))
    candidates, records, attempts = {}, {}, []
    # Search group shares geometry between908/910. High to low integer side
    # remains a construction search, not a frozen-atlas coordinate polish.
    sides = sorted(set([math.ceil(math.sqrt(n)) for n in wanted] +
                       [float(base_cert[str(n)]["literal"]["L"])-.00001
                        for n in wanted if base_cert[str(n)]["literal"]["ok"]]))
    best_side = {n:float("inf") for n in wanted}
    best_counts = {n:0 for n in wanted}
    for side in sides:
        for width in (2,3,5,7):
            for delta,slide,offset,cap in (
                (.006,0.,0.,0),(.015,.12,.25,1),(.03,-.12,-.25,1),
                (.06,.25,0.,2),(.12,-.25,.125,1),(.001,.05,-.125,0)):
                if time.monotonic()-start > args.seconds:
                    break
                q, meta = hinged(side,width,delta,slide,offset,cap)
                p, caps = fill_corners(q,side,.125 if cap else 0.)
                assert not len(conflicts(p)), "screening bug"
                entry = dict(side=side,count=len(p),**meta,corner_caps=caps)
                attempts.append(entry)
                for n in wanted:
                    best_counts[n] = max(best_counts[n],len(p) if side<math.ceil(math.sqrt(n)) else 0)
                    if len(p)>=n and side<best_side[n] and min(meta["half_counts"])>0:
                        candidates[n] = rows(p[:n])
                        records[n] = dict(kind="constructed_hinged_band",
                                          constructive_count_before_deletion=len(p),**entry)
                        best_side[n] = side
            if time.monotonic()-start > args.seconds:
                break
        if time.monotonic()-start > args.seconds:
            break
    # Preserve failed finite-count attempts independently from emitted fallback.
    search_failures = {str(n):dict(best_count_below_grid_side=best_counts[n],
                                    best_count_any_side=max(a["count"] for a in attempts),
                                    found_hinged_candidate=n in candidates)
                       for n in wanted}
    for n in wanted:
        if n not in candidates:
            q, meta = fallback(n)
            candidates[n], records[n] = rows(q), meta
    text = csv_text(candidates)
    checks = certify_transport(text, args.out/"output-pandas.csv")
    # A screened candidate can fail official arithmetic. Preserve it, then
    # explicitly replace with the certified constructive corner fallback.
    rejected = {}
    for n in wanted:
        if not all(c["ok"] for c in checks[str(n)].values()):
            rejected[str(n)] = dict(metadata=records[n],certification=checks[str(n)])
            (args.out/f"rejected-{n}.csv").write_text(csv_text({n:candidates[n]}))
            q, meta = fallback(n)
            candidates[n], records[n] = rows(q), meta
    text = csv_text(candidates)
    checks = certify_transport(text,args.out/"output-pandas.csv")
    (args.out/"output.csv").write_text(text)
    comparisons = {}
    for n in wanted:
        c, b = checks[str(n)], base_cert[str(n)]
        delta = {mode:str(Decimal(c[mode]["L"])-Decimal(b[mode]["L"]))
                 for mode in c if c[mode]["ok"] and b[mode]["ok"]}
        comparisons[str(n)] = dict(construction=records[n],certificates=c,
                                    incumbent=b,delta_L=delta,
                                    improvement=all(Decimal(v)<Decimal("-1e-8") for v in delta.values()) and len(delta)==2)
    result = dict(elapsed_seconds=time.monotonic()-start,baseline=str(args.baseline),
                  pandas_version=pd.__version__,search_attempts=attempts,
                  search_failures=search_failures,official_rejections=rejected,
                  comparisons=comparisons,
                  meaningful_wins=sum(c["improvement"] for c in comparisons.values()))
    (args.out/"results.json").write_text(json.dumps(result,indent=2))
    report = f"""Hinged finite bands, CPU POC

This is a newly generated split-angle construction, not an incumbent replay.
Two subbands follow independently angled lattice frames, with an actual
transverse slide, asymmetric explicit endpoint caps, finite junction deletions
and four independently corner-anchored upright fill phases. This is not the
earlier straight band library or a free-angle atlas relaxation.

The alternate fallback is explicitly a two-level corner grammar with sourced
Hamalainen17/Cantrell37 interlocking atoms at opposite corners, independent plus/minus
0.125 slides and four upright caps. It is constructed, not a source-row replay.
Neither fallback nor a screened count alone constitutes an improvement.

The small17 source atom is reconstructed analytically from
https://kingbird.myphotos.cc/packing/square-17_r2.svg
and https://kingbird.myphotos.cc/packing/square-37.svg, following prior
child_finite/finite.py. Cantrell37 is retained whole. This experiment does not
claim reproducing the better Bidwell17 benchmark. The stricter17/37 gate remains
unpassed, so these bounded large cases are exploratory only, not evidence to
scale this family.

Elapsed CPU wall time was {result['elapsed_seconds']:.3f} seconds.
Generated {len(attempts)} finite configurations. Meaningful certified wins
{result['meaningful_wins']}. Every failed target count is retained in
results.json search_attempts/search_failures. Official rejections are preserved
separately and never counted as wins.

"""
    for n,c in sorted(comparisons.items(),key=lambda v:int(v[0])):
        report += f"n{n} emitted {c['construction']['kind']}. Literal L {c['certificates']['literal'].get('L')}. Actual pandas L {c['certificates']['actual_pandas'].get('L')}. Incumbent literal L {c['incumbent']['literal'].get('L')}. Delta L {c['delta_L']}.\n\n"
    report += """Validation invokes unchanged squares.core.certify on the literal CSV
parsed with squares._official_metric._parse_submission_csv and on actual
pandas.read_csv(...).to_csv(index=False) text parsed by that same parser.
output-pandas.csv preserves the actual serialized bytes. No custom metric.
Lengths include the center-only 1.00000002 numerical-clearance dilation.
No win is claimed from float screening or numerical dust.

Reference pandas version is 3.1.0. The verified interpreter was
/home/kit/band-relax/.venv/bin/python. Run with PYTHONDONTWRITEBYTECODE=1
PYTHONPATH=src and that interpreter, followed by
poc/hinged_band/hinged_band.py --baseline YOUR_BASELINE.csv --seconds 30.
Outputs default to the script directory and --out overrides that. CPU only.
No GPU, git/index operations, shared edits, push or submission occurred.
"""
    (args.out/"hinged_band.md").write_text(report)
    print(json.dumps(dict(elapsed=result["elapsed_seconds"],attempts=len(attempts),
                          comparisons=comparisons),indent=2))
    assert all(v["ok"] for c in checks.values() for v in c.values())

if __name__ == "__main__":
    main()
