#!/usr/bin/env python3
"""Frontier-target probe #2: allow a tiny *uniform expansion* as the repair.

`swarm_lpslack.optimize_group` only ever *shrinks* its final configuration
(`try_shrink_and_certify`), and only when every pair already has a positive
separating gap `> 1e-12`.  Started from a frontier-folded row the SLP+LP loop
routinely ends with `min_sat_gap` at ~1e-12..-1e-9, i.e. a configuration that is
float-feasible and ~1e-2 *below* our certified row but has unresolvable
micro-overlaps; with a shrink-only repair nothing certifies.

This driver imports `lpslack.py` unchanged and only replaces
`try_shrink_and_certify` with a version that first tries the raw candidate and
then a ladder of small *expansions* `(1+d)` about the bbox centre, d = 1e-12 ..
1e-3, certifying each in both transports.  An expansion of `d` costs `d*L` of
side, so even d = 1e-4 is negligible against a 1e-2 gain.

usage: frontier_repair.py ROWS.csv --report REPORT.json --n-list 442,492 \
           --out OUT.json [--jobs 4]
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = "1"

import argparse
import csv
import importlib.util
import json
import multiprocessing as mp
import time
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]

SPEC = importlib.util.spec_from_file_location(
    "lpslack_mod", REPO / "poc/swarm_lpslack/poc/lpslack.py")
lpslack = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lpslack)

LADDER = (0.0, 1e-12, 1e-10, 1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2)


def scale_rows(rows: list[dict[str, str]], factor: float) -> list[dict[str, str]]:
    """Uniform scale about the bbox centre with 80-digit decimals."""
    xs0 = xs1 = ys0 = ys1 = None
    for r in rows:
        a = float(r["angle"])
        h = 0.5 * (float(abs(np.cos(a))) + float(abs(np.sin(a))))
        x, y = float(r["x"]), float(r["y"])
        xs0 = x - h if xs0 is None else min(xs0, x - h)
        xs1 = x + h if xs1 is None else max(xs1, x + h)
        ys0 = y - h if ys0 is None else min(ys0, y - h)
        ys1 = y + h if ys1 is None else max(ys1, y + h)
    cx, cy = 0.5 * (xs0 + xs1), 0.5 * (ys0 + ys1)
    with localcontext() as ctx:
        ctx.prec = 80
        q = Decimal("1e-55")
        s = Decimal(repr(factor))
        dcx, dcy = Decimal(repr(cx)), Decimal(repr(cy))
        out = []
        for r in rows:
            x = dcx + s * (Decimal(r["x"]) - dcx)
            y = dcy + s * (Decimal(r["y"]) - dcy)
            out.append(dict(square_id=r["square_id"], x=format(x.quantize(q), "f"),
                            y=format(y.quantize(q), "f"), angle=r["angle"]))
        return out


def repaired_certify(n, centers, angles, old_L):
    rows = lpslack.rows_from_arrays(centers, angles)
    gaps = lpslack.sat_gaps(centers, angles, radius=lpslack.SHRINK_RADIUS)[1]
    info = dict(min_gap=float(gaps.min()) if len(gaps) else None,
                float_L=lpslack.exact_side(centers, angles), tried=[])
    for d in LADDER:
        cand = rows if d == 0.0 else scale_rows(rows, 1.0 + d)
        res = lpslack.certify_pair(n, cand)
        if res is None:
            info["tried"].append((d, "certify-error"))
            continue
        lit, pan = res
        info["tried"].append((d, str(max(lit, pan))))
        if lit < old_L and pan < old_L:
            print(f"  n={n}: REPAIR d={d:g} -> {max(lit, pan):.9f} "
                  f"(min_gap {info['min_gap']:.2e})", flush=True)
            return lit, pan, cand, info
    return None, None, None, info


def task(args):
    n, rows, old_L, outer_iters, fixed_rounds, tr_sched = args
    captured = {}
    orig = lpslack.try_shrink_and_certify

    def patched(nn, centers, angles, old):
        out = orig(nn, centers, angles, old)
        if out is not None:
            captured["info"] = dict(min_gap=None, mode="shrink")
            return out
        lit, pan, cand, info = repaired_certify(nn, centers, angles, old)
        captured["info"] = info
        if lit is None:
            return None
        return lit, pan, cand
    lpslack.try_shrink_and_certify = patched
    try:
        res = lpslack.optimize_group(n, rows, old_L, outer_iters=outer_iters,
                                     fixed_rounds=fixed_rounds,
                                     tr_th_schedule=tr_sched)
    finally:
        lpslack.try_shrink_and_certify = orig
    res["repair"] = captured.get("info")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("rows")
    ap.add_argument("--report", required=True)
    ap.add_argument("--n-list", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--outer-iters", type=int, default=8)
    ap.add_argument("--fixed-rounds", type=int, default=8)
    ap.add_argument("--tr-sched", default="5e-4,5e-5")
    a = ap.parse_args()

    report = json.loads(Path(a.report).read_text(encoding="utf-8"))
    gate = {int(r["n"]): Decimal(r["L"]) for r in report["results"] if r.get("L")}
    groups: dict[int, list[dict[str, str]]] = {}
    with open(a.rows, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            groups.setdefault(int(row["n"]), []).append(
                dict(square_id=row["square_id"], x=row["x"], y=row["y"],
                     angle=row["angle"]))
    for n in groups:
        groups[n].sort(key=lambda r: int(r["square_id"]))
    targets = [int(x) for x in a.n_list.split(",") if x.strip()]
    tr = tuple(float(x) for x in a.tr_sched.split(",") if x.strip())
    tasks = [(n, groups[n], str(gate[n]), a.outer_iters, a.fixed_rounds, tr)
             for n in sorted(targets, reverse=True)]
    t0 = time.time()
    out = []
    with mp.get_context("fork").Pool(min(a.jobs, len(tasks))) as pool:
        for res in pool.imap_unordered(task, tasks, chunksize=1):
            out.append(res)
            print(f"[{len(out)}/{len(tasks)}] n={res['n']} status={res['status']} "
                  f"old={res['old_L'][:18]} new={str(res.get('new_L', res.get('best_float_L')))[:18]} "
                  f"repair={res.get('repair')}", flush=True)
    out.sort(key=lambda r: r["n"])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {a.out} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    mp.freeze_support()
    main()
