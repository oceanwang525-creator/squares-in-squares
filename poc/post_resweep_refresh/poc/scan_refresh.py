#!/usr/bin/env python3
"""Prefilter refreshed public CSVs by floating bounding side.

Every submission-shaped CSV under --inputs is grouped by n. A group is kept
only when its floating bbox side is strictly below the incumbent certified L
by more than 1e-10. Exact certification happens in merge_refresh.py.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from decimal import Decimal

import numpy as np

COLUMNS = ("id", "n", "square_id", "x", "y", "angle")
COLUMNS_NO_ID = ("n", "square_id", "x", "y", "angle")


def read_groups(path: str):
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            header = tuple(column.strip() for column in next(reader))
        except StopIteration:
            return None, None
        if header == COLUMNS:
            indices = (1, 3, 4, 5)
        elif header == COLUMNS_NO_ID:
            indices = (0, 2, 3, 4)
        else:
            return None, None
        groups: dict[int, list[list[str]]] = {}
        ni, xi, yi, ai = indices
        try:
            for row in reader:
                groups.setdefault(int(row[ni]), []).append(row)
        except (IndexError, ValueError):
            return None, None
    return indices, groups


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--incumbent-report", required=True)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--candidates", required=True)
    args = parser.parse_args()

    report = json.load(open(args.incumbent_report, encoding="utf-8"))
    ours = {int(row["n"]): Decimal(row["L"]) for row in report["results"]
            if row["L"] is not None}

    files = sorted(glob.glob(os.path.join(args.inputs, "**", "*.csv"),
                             recursive=True))
    candidates: dict[int, list[tuple[float, str, list[dict[str, str]]]]] = {}
    for name in files:
        indices, groups = read_groups(name)
        if groups is None:
            continue
        ni, xi, yi, ai = indices
        relative = os.path.relpath(name, args.inputs)
        wins = 0
        for n, rows in groups.items():
            if n not in ours or len(rows) != n:
                continue
            try:
                x = np.array([float(row[xi]) for row in rows])
                y = np.array([float(row[yi]) for row in rows])
                angle = np.array([float(row[ai]) for row in rows])
            except (IndexError, ValueError):
                continue
            half = 0.5 * (np.abs(np.cos(angle)) + np.abs(np.sin(angle)))
            side = float(max((x + half).max() - (x - half).min(),
                             (y + half).max() - (y - half).min()))
            if side < float(ours[n]) - 1e-10:
                payload = [
                    dict(square_id=str(i), x=repr(float(x[i])),
                         y=repr(float(y[i])), angle=repr(float(angle[i])))
                    for i in range(n)
                ]
                candidates.setdefault(n, []).append((side, relative, payload))
                wins += 1
        print(f"{relative}: {wins} candidate n", flush=True)

    serializable = {
        str(n): [(side, source, rows) for side, source, rows in entries]
        for n, entries in candidates.items()
    }
    with open(args.candidates, "w", encoding="utf-8") as handle:
        json.dump(serializable, handle, indent=1)
    print(f"candidate n: {len(candidates)}; wrote {args.candidates}")


if __name__ == "__main__":
    main()
