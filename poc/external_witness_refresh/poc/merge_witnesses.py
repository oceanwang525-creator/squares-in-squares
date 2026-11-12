#!/usr/bin/env python3
"""Certify harvested witnesses, repair pandas transport, and merge winners.

A candidate replaces the incumbent only when its certified L is strictly
smaller. If the exact rational witness loses precision in float64 transport,
a uniform dilation about the bbox centre is tried on a small ladder; the
smallest dilation that passes both literal and pandas certification is kept.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--incumbent", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--wins", required=True)
    args = parser.parse_args()

    incumbent = Path(args.incumbent)
    report = json.load(open(incumbent.parent / "merged-literal-official.json",
                            encoding="utf-8"))
    ours = {int(row["n"]): Decimal(row["L"]) for row in report["results"]
            if row["L"] is not None}
    candidates = json.load(open(args.candidates, encoding="utf-8"))

    wins: dict[int, dict] = {}
    replacement_rows: dict[int, list[dict]] = {}
    for n_text, entries in sorted(candidates.items(), key=lambda item: int(item[0])):
        n = int(n_text)
        for entry in sorted(entries, key=lambda item: item["float_side"]):
            rows = entry["rows"]
            try:
                literal = Decimal(core.certify(n, rows))
            except Exception as error:  # noqa: BLE001
                print("certify failed", n, entry["source"], error)
                continue
            if literal >= ours[n]:
                print("not strict", n, entry["source"], literal)
                continue
            chosen = None
            try:
                transported = pandas_ok(n, rows)
                if transported < ours[n]:
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
                    if trial_literal < ours[n] and trial_pandas < ours[n]:
                        chosen = (trial, trial_literal, trial_pandas, epsilon)
                        break
            if chosen is None:
                print("pandas repair failed", n, entry["source"])
                continue
            rows, literal, transported, epsilon = chosen
            certified = max(literal, transported)
            wins[n] = dict(source=entry["source"], dilation=epsilon,
                           old_L=str(ours[n]), new_L=str(certified),
                           literal_L=str(literal), pandas_L=str(transported))
            replacement_rows[n] = rows
            break

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

    reduction = sum(
        (Decimal(entry["old_L"]) - Decimal(entry["new_L"])) / Decimal(n).sqrt()
        for n, entry in wins.items()
    )
    payload = dict(wins={str(k): v for k, v in wins.items()},
                   certified_score_reduction=str(reduction))
    Path(args.wins).write_text(json.dumps(payload, indent=1) + "\n",
                               encoding="utf-8")
    print(json.dumps(dict(wins=len(wins),
                          certified_score_reduction=str(reduction)), indent=1))


if __name__ == "__main__":
    main()
