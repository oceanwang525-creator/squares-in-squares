"""Search repeatable cuts and local compaction for the team's n=134–166 layouts."""
import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

OUT = Path(__file__).resolve().parent
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from scipy.spatial import Voronoi, cKDTree
from sqk import grow, lp
from sqk.geometry import bbox, half_extent, min_gap, normalize, side
from sqk.submission import export_ready
from squares.core import evaluate as evaluate_single_puzzle

from relax import relax

START, STOP = 134, 166
FIELDS = ("id", "n", "square_id", "x", "y", "angle")


def read_geometries(source, start=75, stop=245):
    groups = defaultdict(list)
    with source.open(encoding="utf-8-sig", newline="") as inp:
        for row in csv.DictReader(inp):
            n = int(row["n"])
            if n < start:
                continue
            if n > stop:
                break
            groups[n].append(row)
    return groups


def array(rows):
    return np.asarray([[float(row[key]) for key in ("x", "y", "angle")] for row in rows])


def cut_positions(coordinate, width, extent):
    crossings = np.unique(np.round(np.concatenate((coordinate, coordinate + width)), 8))
    cuts = (crossings[:-1] + crossings[1:]) * .5
    return [c for c in cuts if width + .35 < c < extent - .35]


def quick_gap(p):
    pairs = cKDTree(p[:,:2]).query_pairs(math.sqrt(2), output_type="ndarray")
    if not len(pairs):
        return math.inf
    a, b = p[pairs[:,0]], p[pairs[:,1]]
    dx, dy = (b[:,:2]-a[:,:2]).T
    t1, t2 = a[:,2], b[:,2]
    c1, s1, c2, s2 = np.cos(t1), np.sin(t1), np.cos(t2), np.sin(t2)
    support = (1+abs(np.cos(t2-t1))+abs(np.sin(t2-t1)))/2
    projection = np.maximum.reduce([abs(c1*dx+s1*dy), abs(c1*dy-s1*dx),
                                    abs(c2*dx+s2*dy), abs(c2*dy-s2*dx)])
    return float((projection-support).min())


def structural_candidates(donors, thresholds):
    ranked = defaultdict(dict)
    seen_donors = set()
    for source, groups in donors:
        for m, geometry in groups.items():
            p = normalize(array(geometry))
            if not np.any(abs(np.sin(2 * p[:, 2])) > 1e-4):
                continue
            identity = (m, np.round(p, 7).tobytes())
            if identity in seen_donors:
                continue
            seen_donors.add(identity)
            old = side(p)
            donor_tag = (source.name, m)
            for width in (1.0, .94, 1.06):
                for shrink in (True, False):
                    lo, hi = (min(thresholds), max(thresholds))
                    if shrink and (m <= lo or m > hi + 35):
                        continue
                    if not shrink and (m < lo - 35 or m >= hi):
                        continue
                    if not shrink and width != 1.0:
                        continue
                    target = old - width if shrink else old + width
                    if all(target > L + .085 or abs(m-n) < 13 or abs(m-n) > 33 for n, L in thresholds.items()):
                        continue
                    xcuts, ycuts = (np.asarray(cut_positions(p[:, dim], width, old)) for dim in (0,1))
                    cols = (p[None,:,0] > xcuts[:,None]-width) & (p[None,:,0] <= xcuts[:,None])
                    rows = (p[None,:,1] > ycuts[:,None]-width) & (p[None,:,1] <= ycuts[:,None])
                    nc, nr = cols.sum(1), rows.sum(1)
                    # The sums are small integers, so float32 BLAS preserves the
                    # exact counts while avoiding a slow integer matrix product.
                    shared = (cols.astype(np.float32) @ rows.T.astype(np.float32)).astype(np.int16)
                    counts = m-nc[:,None]-nr+shared if shrink else m+nc[:,None]+nr+shared
                    wanted = [n for n, L in thresholds.items() if target <= L+.085 and abs(m-n) <= 33]
                    seed = 7919*m + round(100*width) + int(shrink)
                    rng = np.random.default_rng(seed)
                    hits = []
                    for target_n in wanted:
                        matching = np.argwhere(counts == target_n)
                        if len(matching) > 48:
                            matching = matching[rng.choice(len(matching), 48, replace=False)]
                        hits.extend(matching)
                    for j, k in hits:
                        xcut, ycut, col, row = xcuts[j], ycuts[k], cols[j], rows[k]
                        n = counts[j,k]
                        signature = (donor_tag, width, shrink, np.packbits(p[:,0]>xcut).tobytes(),
                                     np.packbits(p[:,1]>ycut).tobytes(), np.packbits(col).tobytes(), np.packbits(row).tobytes())
                        if signature in ranked[n]:
                            continue
                        if shrink:
                            q = p[~(row | col)].copy()
                            q[q[:, 0] > xcut, 0] -= width
                            q[q[:, 1] > ycut, 1] -= width
                        else:
                            q = grow.insert_slabs(p, ycut, xcut)
                        excess = side(q) - thresholds[n]
                        if excess > .105:
                            continue
                        gap = quick_gap(q)
                        if gap < -.19:
                            continue
                        priority = max(0, excess) + .45 * max(0, -gap)
                        ranked[n][signature] = (priority, n, f"{source.name} n={m}, {'remove' if shrink else 'insert'} {width:.2f}", q)
    return [job for options in ranked.values() for job in sorted(options.values(), key=lambda j:j[0])[:12]]


def deletion_candidates(donors, limits):
    from itertools import combinations

    seen = set()
    options = defaultdict(list)
    for source, groups in donors:
        for n, limit in limits.items():
            if n == round(math.sqrt(n)) ** 2:
                continue
            for m in (n+1, n+2):
                p = normalize(array(groups[m]))
                theta = (p[:,2] + math.pi/4) % (math.pi/2) - math.pi/4
                if not np.any(np.abs(theta) > .001) or side(p) > limit + .12:
                    continue
                signature = (n, np.round(p, 7).tobytes())
                if signature in seen:
                    continue
                seen.add(signature)
                h = half_extent(p[:,2])[:,None]
                left, right = p[:,:2]-h, p[:,:2]+h
                edge_gap = np.minimum(left, right.max(0)-right)
                importance = np.min(edge_gap, axis=1) + np.where(abs(theta)<.01,.18,0)
                if m == n+1:
                    removed = [(i,) for i in np.argsort(importance)[:60]]
                else:
                    removed = combinations(np.argsort(importance)[:9], 2)
                label = f"{source.name} n={m}, delete"
                for ids in removed:
                    q = np.delete(p, ids, axis=0)
                    if side(q) < limit + .065:
                        options[n].append((max(0., side(q)-limit),n,f"{label} {ids}",q))
    return [job for choices in options.values() for job in sorted(choices,key=lambda j:j[0])[:75]]


def rotated_diamond_candidates(groups):
    """Vary the rhombus orientation in the 148-square Goebel layout with one vacancy."""
    donor = normalize(array(groups[148]))
    h = side(donor)/2
    tilted = abs(np.sin(2*donor[:,2])) > .05
    seen = set()
    options = []
    for index, row in enumerate(donor):
        orbit = tuple(sorted(np.round(np.abs(row[:2]-h), 5)))
        if orbit in seen:
            continue
        seen.add(orbit)
        for rotation in (0., .003, .01, .026, .045, -.01):
            p = donor.copy()
            c, s = math.cos(rotation), math.sin(rotation)
            p[tilted,:2] = (p[tilted,:2]-h) @ np.array([[c,s],[-s,c]])+h
            p[tilted,2] += rotation
            options.append((abs(rotation),147,f"n148, vacancy {index}, diamond rotation {rotation}",np.delete(p,index,axis=0)))
    return sorted(options, key=lambda j:j[0])


def insertion_candidates(groups, limits):
    """Sample low-overlap places to add a square to the n=146 layout."""
    p = normalize(array(groups[146]))
    length = limits[147]-.002
    _, _, xspan, yspan = bbox(p)
    p[:,:2] += (length - np.array([xspan,yspan]))/2
    xy = p[:,:2]
    tree = cKDTree(xy)
    coordinates = np.linspace(.5,length-.5,81)
    x,y = np.meshgrid(coordinates, coordinates)
    centers = np.vstack((np.column_stack((x.ravel(),y.ravel())),Voronoi(xy).vertices))
    centers = centers[(centers>=.5).all(1)&(centers<=length-.5).all(1)]
    _, ids = tree.query(centers,k=10)
    near = p[ids]
    dx, dy = np.moveaxis(centers[:,None,:]-near[:,:,:2],-1,0)
    c, s = np.cos(near[:,:,2]), np.sin(near[:,:,2])
    own_projection = np.maximum(abs(c*dx+s*dy),abs(c*dy-s*dx))
    options = []
    for angle in np.linspace(-math.pi/4,math.pi/4,13)[:-1]:
        c, s = math.cos(angle), math.sin(angle)
        angle_delta = near[:,:,2]-angle
        support = (1+abs(np.cos(angle_delta))+abs(np.sin(angle_delta)))/2
        projection = np.maximum(own_projection,np.maximum(abs(c*dx+s*dy),abs(c*dy-s*dx)))
        loss = (np.maximum(support-projection,0)**2).sum(1)
        e = (abs(c)+abs(s))/2
        loss += (np.maximum(e-centers,0)**2 + np.maximum(centers+e-length,0)**2).sum(1)
        picked = []
        for idx in np.argsort(loss):
            location = centers[idx]
            if any(np.linalg.norm(location-other)<.53 for other in picked):
                continue
            picked.append(location)
            label = f"n146, insert at {location[0]:.6f},{location[1]:.6f}, angle {angle:.6f}"
            options.append((loss[idx],147,label,np.vstack((p,[*location,angle]))))
            if len(picked) == 7:
                break
    return sorted(options,key=lambda job:job[0])


def process(n, source, p, limit, mode, contact_refiner):
    sys.path.insert(0, str(contact_refiner))
    from refine_corners import improve

    if mode == "cuts":
        q = grow.repair(p, trust=.22, iters=7)
        if q is None or side(q) > limit + .035:
            return None
    elif mode in ("diamond", "insertion"):
        length = limit - (.003 if mode == "diamond" else .002)
        q = p
        if mode == "diamond":
            q = normalize(p)
            q[:,:2] *= length/side(q)
        seed = sum(ord(c)*(i+1) for i,c in enumerate(source))
        q, energy = relax(q, length, seed, displacement=.22 if mode == "diamond" else .35)
        if energy > (.002 if mode == "diamond" else .012) or min_gap(q) < (-.035 if mode == "diamond" else -.055):
            return None
        q = grow.repair(q, trust=.15, iters=7)
        if q is None or side(q) > limit + .015:
            return None
    else:
        q = p
    best = lp.compact(q, trust=.12, iters=8)
    if mode == "deletion" and side(best) >= limit-1e-9 and side(best) < limit+.009:
        rng = np.random.default_rng(sum(ord(c)*(i+1) for i,c in enumerate(source)))
        for _ in range(2):
            seed = best.copy()
            seed[:,2] += np.where(abs(np.sin(2*seed[:,2]))>.01, rng.normal(0,.006,n),0)
            trial = lp.solve(seed, trust=.09, margin=1e-11)
            if trial.status == 0 and min_gap(trial.P) >= -1e-9 and side(trial.P) < side(best):
                best = trial.P
    if side(best) < limit + .02:
        candidate = improve(n, best)
        if candidate is not None and side(candidate[1]) < side(best):
            best = candidate[1]
    if side(best) >= limit - 1e-9:
        return None
    try:
        return n, source, export_ready(best, 1e-11)
    except ValueError:
        return None


def official(n, rows):
    return evaluate_single_puzzle(n, rows)


def as_rows(n, points):
    return [{"id": n*(n-1)//2+i, "n": n, "square_id": str(i), "x": repr(x), "y": repr(y), "angle": repr(a)}
            for i, (x, y, a) in enumerate(points.tolist())]


def certify(n, rows):
    literal = official(n, rows)
    if literal[-1] != "ok":
        return None
    from io import StringIO
    raw = StringIO()
    writer = csv.DictWriter(raw, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    raw.seek(0)
    parsed = pd.read_csv(raw)
    raw = StringIO(parsed.to_csv(index=False))
    roundtrip = official(n, list(csv.DictReader(raw)))
    return (literal, roundtrip) if roundtrip[-1] == "ok" else None


def save(output, wins):
    with output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        for n in sorted(wins):
            writer.writerows(wins[n][0])
    output.with_suffix(".json").write_text(json.dumps({n: wins[n][1] for n in sorted(wins)}, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("compact", "cuts", "deletion", "diamond", "insertion"))
    parser.add_argument("--baseline", type=Path, default=OUT.parent/"pell_corrugation/data/border-scan/merged.csv")
    parser.add_argument("--contact-refiner", type=Path, default=OUT.parent/"contact_slp_submission189/poc")
    parser.add_argument("--donor", action="append", type=Path, default=[])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    mode = args.mode
    output = args.output or (OUT/mode)
    team = read_geometries(args.baseline)
    metrics = []
    for name in ("merged-literal-official.json", "merged-pandas-official.json"):
        data = json.loads(args.baseline.with_name(name).read_text(encoding="utf-8-sig"))["results"]
        metrics.append({r["n"]: (r["n"], r["score"], r["L"], r["reason"]) for r in data})
    baseline = {n: tuple(mode[n] for mode in metrics) for n in range(START, STOP + 1)}
    limits = {n: min(float(mode[2]) for mode in check) for n, check in baseline.items()}
    donors = [(args.baseline, team), *((source, read_geometries(source)) for source in args.donor)]
    if mode == "cuts":
        options = structural_candidates(donors, limits)
    elif mode == "deletion":
        options = deletion_candidates(donors, limits)
    elif mode == "diamond":
        options = rotated_diamond_candidates(team)
    elif mode == "insertion":
        options = insertion_candidates(team, limits)
    else:
        options = []
        seen = set()
        for source, groups in donors:
            for n in limits:
                p = array(groups[n])
                signature = (n, np.round(normalize(p), 7).tobytes())
                if signature in seen or side(p) > limits[n] + .02 or not np.any(abs(np.sin(2*p[:,2])) > 1e-4):
                    continue
                seen.add(signature)
                options.append((0., n, source.name, p))
    print(f"{mode}: {len(options)} candidates", flush=True)
    wins = {}
    with ProcessPoolExecutor(max_workers=2) as pool:
        tasks = [pool.submit(process, n, source, p, limits[n], mode, args.contact_refiner.resolve()) for _,n,source,p in options]
        for i, future in enumerate(as_completed(tasks), 1):
            result = future.result()
            if result:
                n, source, p = result
                rows = as_rows(n, p)
                check = certify(n, rows)
                if check and all(float(new[2]) < float(old[2])-1e-9 for new, old in zip(check, baseline[n])):
                    record = {
                        "source": source,
                        "literal_old": baseline[n][0][2], "pandas_old": baseline[n][1][2],
                        "literal_new": check[0][2], "pandas_new": check[1][2],
                        "literal_gain": baseline[n][0][1]-check[0][1],
                        "pandas_gain": baseline[n][1][1]-check[1][1],
                    }
                    if n not in wins or record["pandas_gain"] > wins[n][1]["pandas_gain"]:
                        wins[n] = rows, record
                        save(output, wins)
                        print(n, record, flush=True)
            if i % 25 == 0:
                print(f"{i}/{len(tasks)}, {len(wins)} wins", flush=True)
    summary = {"candidates": len(tasks), "wins": len(wins), "pandas": pd.__version__,
               "gain": math.fsum(r[1]["pandas_gain"] for r in wins.values())}
    output.with_name(output.name + "-summary").with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
