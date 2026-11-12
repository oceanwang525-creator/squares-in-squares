#!/usr/bin/env python3
"""Certify harvested witnesses, repair pandas transport, and merge winners.

A candidate replaces the incumbent only when its certified L is strictly
smaller. If the exact rational witness loses precision in float64 transport, a
uniform dilation about the bbox centre is tried on a small ladder; the smallest
dilation that passes both literal and pandas certification is kept.

Parallelism
- one process per ``n`` (``multiprocessing`` pool, default 32 workers);
- BLAS/OpenMP thread counts pinned to 1 before numpy import;
- ``imap_unordered`` with the largest ``n`` first;
- the merged CSV is written in sorted ``n`` order, so the output is
  deterministic regardless of completion order.
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = "1"

import argparse
import csv
import json
import multiprocessing as mp
import sys
import time
from decimal import Decimal
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "poc" / "baseline" / "poc" / "src"))
from squares import core  # noqa: E402

DILATIONS = (1e-15, 1e-14, 1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8)


def dilate(rows, epsilon):
    x = np.array([float(row["x"]) for row in rows])
    y = np.array([float(row["y"]) for row in rows])
    angle = np.array([float(row["angle"]) for row in rows])
    half = 0.5 * (np.abs(np.cos(angle)) + np.abs(np.sin(angle)))
    cx = ((x - half).min() + (x + half).max()) / 2
    cy = ((y - half).min() + (y + half).max()) / 2
    x = cx + (x - cx) * (1 + epsilon)
    y = cy + (y - cy) * (1 + epsilon)
    return [dict(square_id=str(i), x=repr(float(x[i])), y=repr(float(y[i])),
                 angle=repr(float(angle[i]))) for i in range(len(rows))]


def pandas_ok(n, rows):
    transported = core.kaggle_roundtrip({n: rows})
    return Decimal(core.certify(n, transported[n]))


def certify_group(task):
    """Worker: pick the best transport-stable strictly-better row set for one n."""
    n, ours_literal, entries = task
    ours = Decimal(ours_literal)
    for entry in sorted(entries, key=lambda item: item["float_side"]):
        rows = entry["rows"]
        try:
            literal = Decimal(core.certify(n, rows))
        except Exception as error:  # noqa: BLE001
            print("certify failed", n, entry["source"], error, flush=True)
            continue
        if literal >= ours:
            print("not strict", n, entry["source"], literal, flush=True)
            continue
        chosen = None
        try:
            transported = pandas_ok(n, rows)
            if transported < ours:
                chosen = (rows, literal, transported, None)
        except Exception:  # noqa: BLE001
            pass
        if chosen is None:
            for epsilon in DILATIONS:
                trial = dilate(rows, epsilon)
                try:
                    trial_literal = Decimal(core.certify(n, trial))
                    trial_pandas = pandas_ok(n, trial)
                except Exception:  # noqa: BLE001
                    continue
                if trial_literal < ours and trial_pandas < ours:
                    chosen = (trial, trial_literal, trial_pandas, epsilon)
                    break
        if chosen is None:
            print("pandas repair failed", n, entry["source"], flush=True)
            continue
        rows, literal, transported, epsilon = chosen
        certified = max(literal, transported)
        return n, dict(source=entry["source"], dilation=epsilon,
                       old_L=str(ours), new_L=str(certified),
                       literal_L=str(literal), pandas_L=str(transported)), rows
    return n, None, None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--incumbent", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--wins", required=True)
    parser.add_argument("--jobs", type=int, default=32)
    args = parser.parse_args()

    incumbent = Path(args.incumbent)
    report = json.load(open(incumbent.parent / "merged-literal-official.json",
                            encoding="utf-8"))
    ours = {int(row["n"]): Decimal(row["L"]) for row in report["results"]
            if row["L"] is not None}
    candidates = json.load(open(args.candidates, encoding="utf-8"))

    tasks = [(int(n_text), str(ours[int(n_text)]), entries)
             for n_text, entries in sorted(candidates.items(),
                                           key=lambda item: int(item[0]),
                                           reverse=True)]

    wins: dict[int, dict] = {}
    replacement_rows: dict[int, list[dict]] = {}
    t0 = time.time()
    done = 0
    jobs = max(1, min(args.jobs, os.cpu_count() or 1))
    with mp.get_context("fork").Pool(jobs) as pool:
        for n, win, rows in pool.imap_unordered(certify_group, tasks, chunksize=1):
            done += 1
            if win is not None:
                wins[n] = win
                replacement_rows[n] = rows
                print(f"WIN n={n} {win['source']} {win['old_L']} -> {win['new_L']}",
                      flush=True)
            if done % 25 == 0 or done == len(tasks):
                rate = done / max(time.time() - t0, 1e-9) * 60.0
                print(f"{done}/{len(tasks)} groups, {rate:.0f} groups/min, "
                      f"wins={len(wins)}", flush=True)

    groups: dict[int, list[dict]] = {}
    with incumbent.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            groups.setdefault(int(row["n"]), []).append(row)
    for n, rows in replacement_rows.items():
        groups[n] = [dict(n=str(n), square_id=row["square_id"], x=row["x"],
                          y=row["y"], angle=row["angle"]) for row in rows]

    with open(args.output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "n", "square_id", "x", "y", "angle"])
        identifier = 0
        for n in range(1, 1001):
            for row in groups[n]:
                writer.writerow([identifier, n, row["square_id"],
                                 row["x"], row["y"], row["angle"]])
                identifier += 1

    ordered_wins = {n: wins[n] for n in sorted(wins)}
    reduction = sum(
        (Decimal(entry["old_L"]) - Decimal(entry["new_L"])) / Decimal(n).sqrt()
        for n, entry in ordered_wins.items()
    )
    payload = dict(wins={str(k): v for k, v in ordered_wins.items()},
                   certified_score_reduction=str(reduction))
    Path(args.wins).write_text(json.dumps(payload, indent=1) + "\n",
                               encoding="utf-8")
    print(json.dumps(dict(wins=len(ordered_wins), jobs=jobs,
                          seconds=round(time.time() - t0, 1),
                          certified_score_reduction=str(reduction)), indent=1))


if __name__ == "__main__":
    mp.freeze_support()
    main()
