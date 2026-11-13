"""Continue the strongest pinwheel growth seeds with exact corner constraints."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import itertools
import json
import math
from pathlib import Path

from search import (
    MARGIN, REACH, certify, compact, insert_layer, minimum_gap, normalize,
    points_from, sat, settle, side, touching_pairs, write_csv,
)

import casadi as ca
import numpy as np

CORNERS = np.array([[-.5, -.5], [-.5, .5], [.5, -.5], [.5, .5]])


def corner_positions(x, y, cos_angle, sin_angle, indices):
    square = np.repeat(indices, 4).tolist()
    corner = np.tile(CORNERS, (len(indices), 1))
    a, b = ca.DM(corner[:, 0]), ca.DM(corner[:, 1])
    return (
        x[square] + a*cos_angle[square] - b*sin_angle[square],
        y[square] + a*sin_angle[square] + b*cos_angle[square],
    )


def ipopt_step(points, trust, angle_trust, max_iterations, cpu_seconds):
    p = normalize(points)
    n = len(p)
    length = side(p)
    left, right = touching_pairs(p, REACH + 2*math.sqrt(2.)*trust + 1e-6)
    _, direction, winner, _, _ = sat(p, left, right)
    owner = np.where(winner < 2, left, right)
    other = np.where(winner < 2, right, left)
    outward = direction * np.where(winner < 2, 1., -1.)[:, None]
    orthogonal = (winner % 2).astype(float)
    axis_angle = p[owner, 2] + orthogonal*(math.pi/2)
    sign = np.sign(outward[:, 0]*np.cos(axis_angle) + outward[:, 1]*np.sin(axis_angle))
    x, y, theta = (ca.MX.sym(name, n) for name in ("x", "y", "theta"))
    box = ca.MX.sym("box")
    c, s = ca.cos(theta), ca.sin(theta)
    px, py = corner_positions(x, y, c, s, other)
    idx = np.repeat(owner, 4).tolist()
    along = ca.DM(np.repeat(1.-orthogonal, 4))
    across = ca.DM(np.repeat(orthogonal, 4))
    ux = along*c[idx] - across*s[idx]
    uy = along*s[idx] + across*c[idx]
    separation = ca.DM(np.repeat(sign, 4))*((px-x[idx])*ux + (py-y[idx])*uy) - .5
    vx, vy = corner_positions(x, y, c, s, np.arange(n))
    constraint = ca.vertcat(separation, vx, vy, box-vx, box-vy)
    model = {"x": ca.vertcat(x, y, theta, box), "f": box, "g": constraint}
    solver = ca.nlpsol("pinwheel", "ipopt", model, {
        "print_time": False, "ipopt.print_level": 0, "ipopt.sb": "yes",
        "ipopt.max_iter": max_iterations, "ipopt.max_cpu_time": cpu_seconds,
        "ipopt.tol": 1e-9, "ipopt.constr_viol_tol": 2e-10, "ipopt.mu_strategy": "adaptive",
    })
    start = np.r_[p.T.ravel(), length]
    radius = np.r_[np.repeat(trust, 2*n), np.repeat(angle_trust, n), 2*trust]
    result = solver(x0=start, lbx=start-radius, ubx=start+radius, lbg=MARGIN, ubg=ca.inf)
    q = np.asarray(result["x"]).ravel()[:-1].reshape(3, n).T
    info = solver.stats()
    return normalize(q), {"status": info["return_status"], "iterations": info["iter_count"]}


def merit(q):
    return side(q) + 40*max(0., -minimum_gap(q))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--incumbent", type=Path, required=True)
    parser.add_argument("--scan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target", type=int, default=184)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=180)
    parser.add_argument("--cpu-seconds", type=float, default=12.)
    args = parser.parse_args()
    scan = json.loads(args.scan.read_text(encoding="utf-8"))
    incumbent_hash = hashlib.sha256(args.incumbent.read_bytes()).hexdigest()
    if incumbent_hash != scan["incumbent_sha256"]:
        parser.error("The scan and incumbent must refer to the same file")
    family = next(f for f in scan["families"] if f["target_n"] == args.target)
    donor_n, n = family["donor_n"], args.target
    with args.incumbent.open(encoding="utf-8", newline="") as f:
        groups = {k: list(rows) for k, rows in itertools.groupby(
            itertools.takewhile(lambda row: int(row["n"]) <= n, csv.DictReader(f)),
            key=lambda row: int(row["n"])) if k in (donor_n, n)}
    p = normalize(points_from(groups[donor_n]))
    p[:, 2] = (p[:, 2] + math.pi/4) % (math.pi/2) - math.pi/4
    trials = sorted((t for t in family["trials"] if t["seed_count"] == n),
                    key=lambda t: t["side_float"] + 40*max(0., -t["minimum_gap_float"]))
    output = {"target_n": n, "donor_n": donor_n, "incumbent_sha256": incumbent_hash,
              "incumbent_side": family["incumbent_side_float"], "trials": [], "wins": []}
    target = family["incumbent_side_float"]
    args.out.mkdir(parents=True, exist_ok=True)
    for trial in trials[:args.seeds]:
        allowance, clearance = trial["copy_overlap_limit"], trial["shift_clearance"]
        layer = insert_layer(p, trial["y"], 1, allowance, clearance)
        seed = insert_layer(layer, trial["x"], 0, allowance, clearance)
        assert len(seed) == n
        q = compact(settle(seed, target-.00005, scan["relax_iters"]), scan["lp_rounds"])
        result = {"seed": trial, "rounds": []}
        for r in range(args.rounds):
            candidate, info = ipopt_step(q, .20 if r == 0 else .10,
                                         .08 if r == 0 else .04, args.iterations, args.cpu_seconds)
            info.update(side_float=side(candidate), minimum_gap_float=minimum_gap(candidate))
            print(n, trial["x"], trial["y"], r+1, info, flush=True)
            result["rounds"].append(info)
            if merit(candidate) < merit(q):
                q = candidate
        if side(q) < target-1e-9 and minimum_gap(q) >= -1e-7:
            import pandas as pd
            frame = pd.read_csv(args.incumbent)
            transported = list(csv.DictReader(io.StringIO(frame[frame.n == n].to_csv(index=False))))
            checked = certify(q, n, groups[n], transported)
            if checked:
                rows, details = checked
                output["wins"].append(details)
                write_csv(args.out/f"ipopt_rows_{len(output['wins'])}.csv", rows)
        output["trials"].append(result)
        (args.out/"ipopt.json").write_text(json.dumps(output, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
