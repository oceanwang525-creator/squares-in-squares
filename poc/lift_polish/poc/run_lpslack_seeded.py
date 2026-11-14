#!/usr/bin/env python3
"""Run `swarm_lpslack`'s alternating optimiser on an *arbitrary* row set.

`poc/swarm_lpslack/poc/lpslack.py` has a hard-wired portfolio loader: it always
starts from the best certified candidate it can find in `merged.csv` plus every
`poc/*/final.csv`.  That is the right behaviour for banked rows, but not for
(a) re-polishing a row this folder just improved (iteration 2+), or
(b) starting from a *shrunk* copy of a row (frontier-target probe).

This driver imports that module unchanged and calls `optimize_group` directly
with the rows and the gate side we choose; the geometric model, shrink and
double certification are exactly the proven ones.

usage:
  run_lpslack_seeded.py --merged BASE.csv --report REPORT.json \
      --n-list 442,485 --out OUT.json [--jobs 7] [--outer-iters 8]
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
import sys
import time
from decimal import Decimal
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "poc/baseline/poc/src"))
sys.path.insert(0, str(REPO / "poc/auto_union/poc"))

SPEC = importlib.util.spec_from_file_location(
    "lpslack_mod", REPO / "poc/swarm_lpslack/poc/lpslack.py")
lpslack = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lpslack)


def _task(args):
    n, rows, gate_L, outer_iters, fixed_rounds, tr_sched = args
    return lpslack.optimize_group(
        n, rows, gate_L, outer_iters=outer_iters, fixed_rounds=fixed_rounds,
        tr_th_schedule=tr_sched)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--n-list", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=7)
    ap.add_argument("--outer-iters", type=int, default=8)
    ap.add_argument("--fixed-rounds", type=int, default=8)
    ap.add_argument("--tr-sched", default="5e-4,5e-5")
    args = ap.parse_args()

    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    gate = {int(r["n"]): Decimal(r["L"]) for r in report["results"]
            if r["L"] is not None}
    groups: dict[int, list[dict[str, str]]] = {}
    with open(args.merged, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            groups.setdefault(int(row["n"]), []).append(
                {"square_id": row["square_id"], "x": row["x"], "y": row["y"],
                 "angle": row["angle"]})
    for n in groups:
        groups[n].sort(key=lambda r: int(r["square_id"]))
    targets = [int(x) for x in args.n_list.split(",") if x.strip()] or sorted(groups)
    tr_sched = tuple(float(x) for x in args.tr_sched.split(",") if x.strip())

    tasks = [(n, groups[n], str(gate[n]), args.outer_iters, args.fixed_rounds,
              tr_sched) for n in sorted(targets, reverse=True)]
    t0 = time.time()
    results = []
    with mp.get_context("fork").Pool(min(args.jobs, len(tasks))) as pool:
        for res in pool.imap_unordered(_task, tasks, chunksize=1):
            results.append(res)
            print(f"[{len(results)}/{len(tasks)}] n={res['n']} status={res['status']} "
                  f"old={res['old_L'][:17]} "
                  f"new={str(res.get('new_L', res.get('best_float_L')))[:17]} "
                  f"dL={res.get('dL', '0')[:12]} t={res['elapsed_s']}s", flush=True)
    results.sort(key=lambda r: r["n"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"wrote {args.out} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    mp.freeze_support()
    main()
