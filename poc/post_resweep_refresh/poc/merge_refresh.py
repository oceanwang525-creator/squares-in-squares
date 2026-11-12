#!/usr/bin/env python3
"""Certify refreshed candidates and replace only strict per-n winners.

Unchanged groups keep their original CSV strings; the output is otherwise
byte-identical in layout to the incumbent.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "poc" / "baseline" / "poc" / "src"))
from squares import core  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--incumbent", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--wins", required=True)
    args = parser.parse_args()

    incumbent = Path(args.incumbent)
    puzzles = core.read_submission(str(incumbent))
    report_path = incumbent.parent / "merged-literal-official.json"
    report = json.load(open(report_path, encoding="utf-8"))
    ours = {int(row["n"]): Decimal(row["L"]) for row in report["results"]
            if row["L"] is not None}

    candidates = json.load(open(args.candidates, encoding="utf-8"))
    wins: dict[int, dict] = {}
    for n_text, entries in sorted(candidates.items(), key=lambda item: int(item[0])):
        n = int(n_text)
        for side, source, rows in sorted(entries, key=lambda entry: entry[0]):
            try:
                certified = Decimal(core.certify(n, rows))
            except Exception as error:  # noqa: BLE001
                print("certify failed", n, source, error)
                continue
            if certified < ours[n]:
                wins[n] = dict(source=source, float_side=side,
                               old_L=str(ours[n]), new_L=str(certified))
                puzzles[n] = rows
                break
            print("not strict", n, source, certified)

    groups: dict[int, list[dict]] = {}
    with incumbent.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            groups.setdefault(int(row["n"]), []).append(row)
    for n, rows in puzzles.items():
        if n in wins:
            groups[n] = [
                dict(n=str(n), square_id=row["square_id"], x=row["x"],
                     y=row["y"], angle=row["angle"])
                for row in rows
            ]

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
