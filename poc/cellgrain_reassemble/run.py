#!/usr/bin/env python3
"""Whole-cell mixed-angle search and strict certification."""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import multiprocessing as mp
import os
import sys
import tempfile
import time
from decimal import Decimal
from pathlib import Path

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(v, "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "poc/swarm_highd/poc"))
sys.path.insert(0, str(ROOT / "poc/swarm_brickslp/poc"))
sys.path.insert(0, str(ROOT / "poc/baseline/poc/src"))

import brick_slp as bs  # noqa: E402
from cpp_wrap import cpp_eval_sat_pen_bbox  # noqa: E402
import utils as u  # noqa: E402
from squares.core import certify, kaggle_roundtrip, read_submission  # noqa: E402

INCUMBENT = ROOT / "poc/pell_corrugation/data/border-scan/merged.csv"
HERE = Path(__file__).resolve().parent


def arrays(rows):
    a = np.array([[float(r["x"]), float(r["y"]), float(r["angle"])] for r in rows])
    return a[:, 0], a[:, 1], a[:, 2]


def extrema(x, y, th):
    e = 0.5 * (np.abs(np.cos(th)) + np.abs(np.sin(th)))
    return float((x + e).max() - (x - e).min()), float((y + e).max() - (y - e).min())


def certify_rows(n, rows):
    transported = kaggle_roundtrip({n: rows})[n]
    lit = Decimal(certify(n, rows))
    pan = Decimal(certify(n, transported))
    return lit, pan


def to_rows(n, x, y, th):
    return [dict(square_id=str(i), x=repr(float(x[i])), y=repr(float(y[i])),
                 angle=repr(float(th[i]))) for i in range(n)]


def exact_gate(n, x, y, th, inc_lit, inc_pan):
    rows = to_rows(n, x, y, th)
    try:
        lit, pan = certify_rows(n, rows)
    except Exception:
        return None
    if lit >= inc_lit or pan >= inc_pan:
        return None
    d = min(inc_lit - lit, inc_pan - pan)
    return dict(lit=str(lit), pan=str(pan), dL=str(d), rows=rows)


def polish(n, x, y, th, k):
    pen, _ = cpp_eval_sat_pen_bbox(x, y, th)
    if pen > 0.02:
        return x, y, th, None, float(pen)
    lx, ly = extrema(x, y, th)
    f0 = max(lx, ly) - k
    for rounds, tt, tx, tol in ((8, 0.014, 0.060, 1e-10), (6, 0.003, 0.012, 1e-11)):
        x2, y2, t2, f2, ok = bs.joint_xytheta_slp_rect(
            x, y, th, k, k, rounds=rounds, tr_th0=tt, tr_xy0=tx, tol=tol)
        if ok and f2 < f0:
            pen2, _ = cpp_eval_sat_pen_bbox(x2, y2, t2)
            if pen2 <= 1e-9:
                x, y, th, f0 = x2, y2, t2, f2
            else:
                x3, y3, f3, ok3 = bs.fixed_angle_lp_tighten_rect(
                    x2, y2, t2, k, k, rounds=5, margin=5e-7)
                pen3, _ = cpp_eval_sat_pen_bbox(x3, y3, t2)
                if ok3 and pen3 <= 1e-9 and f3 < f0:
                    x, y, th, f0 = x3, y3, t2, f3
                else:
                    x4, y4, t4, _ = bs.relax_rect_lbfgs(
                        x2, y2, t2, k + max(f2, f0) + 0.002, k + max(f2, f0) + 0.002,
                        maxiter=420)
                    pen4, _ = cpp_eval_sat_pen_bbox(x4, y4, t4)
                    if pen4 <= 1e-9:
                        x, y, th = x4, y4, t4
    x, y, w, h = bs.rect_bbox(x, y, th)
    f = max(w - k, h - k)
    return x, y, th, f, float(pen)


def seed_task(task):
    n, x0, y0, t0, f0, mode, seed = task
    rng = np.random.default_rng(seed)
    x, y, th = x0.copy(), y0.copy(), t0.copy()
    expand = 0.0 if mode == "direct" else 0.018
    if expand:
        x += expand / 2
        y += expand / 2
    if mode == "full":
        x += rng.uniform(-0.06, 0.06, n)
        y += rng.uniform(-0.06, 0.06, n)
        th += rng.uniform(-0.04, 0.04, n)
    elif mode.startswith("pocket"):
        scale = 0.8 + 0.4 * (mode == "pocket2")
        cx = rng.uniform(0.8, math.isqrt(n) + f0 + expand - 0.8)
        cy = rng.uniform(0.8, math.isqrt(n) + f0 + expand - 0.8)
        rad = rng.uniform(2.5, 6.0) * scale
        near = (x - cx) ** 2 + (y - cy) ** 2 <= rad * rad
        count = int(near.sum())
        if count > 2:
            x[near] += rng.uniform(-0.12, 0.12, count)
            y[near] += rng.uniform(-0.12, 0.12, count)
            th[near] += rng.uniform(-0.20, 0.20, count)
    elif mode == "tilt":
        deg = np.degrees(th) % 90.0
        tilt = np.minimum(deg, 90.0 - deg) > 3.0
        count = int(tilt.sum())
        if count > 2:
            x[tilt] += rng.uniform(-0.04, 0.04, count)
            y[tilt] += rng.uniform(-0.04, 0.04, count)
            th[tilt] += rng.uniform(-0.06, 0.06, count)

    pen, _ = cpp_eval_sat_pen_bbox(x, y, th)
    if pen > 1e-7:
        x, y, th, _ = u.relax_lbfgs(x, y, th, math.isqrt(n) + f0 + expand + 0.006, maxiter=260)
        pen, _ = cpp_eval_sat_pen_bbox(x, y, th)
    if pen > 0.05:
        return None, {"mode": mode, "reason": f"unrepaired pen={pen:.4g}"}

    if mode != "direct":
        for ft in (f0 + 0.005, f0 - 0.0006):
            x, y, th, energy = bs.relax_rect_lbfgs(
                x, y, th, math.isqrt(n) + ft, math.isqrt(n) + ft, maxiter=380)
            if energy > 3e-4:
                break
    k = math.isqrt(n)
    x, y, th, f, pen = polish(n, x, y, th, k)
    return (x, y, th, f, pen), {"mode": mode, "f": f, "pen": pen}


def write_final(win):
    import mpmath as mp
    out = io.StringIO(newline="")
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(["id", "x", "y", "deg"])
    with mp.workdps(95):
        for n, rows in sorted(win.items()):
            for row in sorted(rows, key=lambda r: int(r["square_id"])):
                deg = mp.nstr(mp.mpf(row["angle"]) * 180 / mp.pi, 75)
                writer.writerow([f"{n:03}_{row['square_id']}", "s" + row["x"],
                                 "s" + row["y"], "s" + deg])
    fd, name = tempfile.mkstemp(dir=HERE, prefix=".final.")
    with os.fdopen(fd, "w") as f:
        f.write(out.getvalue())
        f.flush()
        os.fsync(f.fileno())
    os.replace(name, HERE / "final.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", required=True)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--final", action="store_true")
    args = ap.parse_args()
    all_groups = read_submission(INCUMBENT)
    report = {}
    t0 = time.time()
    for n in [int(v) for v in args.n.split(",")]:
        rows = all_groups[n]
        x0, y0, th0 = arrays(rows)
        f0 = max(extrema(x0, y0, th0)) - math.isqrt(n)
        inc_lit, inc_pan = certify_rows(n, rows)
        tasks = [(n, x0, y0, th0, f0, "direct", 90000 + n)]
        modes = ["full", "pocket", "tilt", "pocket2"]
        for i in range(args.seeds):
            tasks.append((n, x0, y0, th0, f0, modes[i % len(modes)], 91000 + 17 * n + i))
        candidates, infos = [], []
        with mp.get_context("fork").Pool(min(args.jobs, len(tasks))) as pool:
            for result, info in pool.imap_unordered(seed_task, tasks):
                infos.append(info)
                if result is not None:
                    candidates.append(result)
        best = None
        for result in candidates:
            x, y, th, _, _ = result
            win = exact_gate(n, x, y, th, inc_lit, inc_pan)
            if win is not None and (best is None or Decimal(win["dL"]) < Decimal(best["dL"])):
                best = win
        report[str(n)] = {
            "incumbent_literal": str(inc_lit), "incumbent_pandas": str(inc_pan),
            "f0": f0, "candidate_count": len(candidates),
            "infos": infos,
            "win": None if best is None else {k: v for k, v in best.items() if k != "rows"},
        }
        print(json.dumps(report[str(n)], indent=2), flush=True)
        if best and args.final:
            ax, ay, at = arrays(best["rows"])
            rewin = exact_gate(n, ax, ay, at, inc_lit, inc_pan)
            if rewin is not None:
                write_final({n: rewin["rows"]})
    (HERE / "data").mkdir(exist_ok=True)
    (HERE / "data/report.json").write_text(json.dumps(report, indent=2))
    print("elapsed", round(time.time() - t0, 2), flush=True)


if __name__ == "__main__":
    main()
