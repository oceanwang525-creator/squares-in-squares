#!/usr/bin/env python3
"""Prefilter public CSVs by bbox, certify strict candidates, merge best per n.

Only candidates whose floating bounding side is below the incumbent are sent
to the official interval certifier. A candidate replaces the incumbent only
when its certified 50-digit L is strictly smaller.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import sys
from decimal import Decimal
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "poc" / "baseline" / "poc" / "src"))
from squares import core  # noqa: E402

COLUMNS = ("id", "n", "square_id", "x", "y", "angle")


def incumbent_sides(path: Path, report: Path | None) -> dict[int, Decimal]:
    if report and report.exists():
        data = json.loads(report.read_text(encoding="utf-8"))
        return {int(row["n"]): Decimal(row["L"]) for row in data["results"]
                if row["L"] is not None}
    return {n: Decimal(core.certify(n, rows))
            for n, rows in core.read_submission(path).items()}


def candidate_rows(path: Path) -> dict[int, list[dict[str, str]]]:
    groups: dict[int, list[dict[str, str]]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != COLUMNS:
            return {}
        for row in reader:
            try:
                n = int(row["n"])
            except (TypeError, ValueError):
                return {}
            groups.setdefault(n, []).append(dict(
                square_id=row["square_id"], x=row["x"],
                y=row["y"], angle=row["angle"],
            ))
    return groups


def float_side(rows: list[dict[str, str]]) -> float | None:
    try:
        x = np.array([float(r["x"]) for r in rows])
        y = np.array([float(r["y"]) for r in rows])
        a = np.array([float(r["angle"]) for r in rows])
    except (TypeError, ValueError):
        return None
    h = 0.5 * (np.abs(np.cos(a)) + np.abs(np.sin(a)))
    return float(max((x + h).max() - (x - h).min(),
                     (y + h).max() - (y - h).min()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--incumbent", required=True)
    parser.add_argument("--incumbent-report")
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--wins", required=True)
    args = parser.parse_args()

    incumbent = Path(args.incumbent)
    puzzles = core.read_submission(incumbent)
    ours = incumbent_sides(
        incumbent,
        Path(args.incumbent_report) if args.incumbent_report else None,
    )
    files = sorted(glob.glob(os.path.join(args.inputs, "**", "*.csv"),
                             recursive=True))
    candidates: dict[int, list[tuple[float, str, list[dict[str, str]]]]] = {}
    for name in files:
        path = Path(name)
        groups = candidate_rows(path)
        for n, rows in groups.items():
            if n not in ours or len(rows) != n:
                continue
            side = float_side(rows)
            if side is not None and side < float(ours[n]) - 1e-10:
                rel = str(path.relative_to(args.inputs))
                candidates.setdefault(n, []).append((side, rel, rows))

    wins: dict[int, dict] = {}
    rejected: dict[str, list[dict]] = {}
    for n in sorted(candidates):
        for side, source, rows in sorted(candidates[n], key=lambda z: z[0]):
            try:
                side_exact = Decimal(core.certify(n, rows))
            except Exception as error:  # noqa: BLE001
                rejected.setdefault(str(n), []).append(
                    dict(source=source, error=str(error)[:200]))
                continue
            if side_exact < ours[n]:
                wins[n] = dict(source=source, old_L=str(ours[n]),
                               new_L=str(side_exact), float_side=side)
                puzzles[n] = rows
                break
            rejected.setdefault(str(n), []).append(
                dict(source=source, error="certified L not lower",
                     L=str(side_exact)))

    core.write_submission(args.output, puzzles, validate=False)
    reduction = sum(
        (Decimal(v["old_L"]) - Decimal(v["new_L"])) / Decimal(n).sqrt()
        for n, v in wins.items()
    )
    payload = dict(wins={str(k): v for k, v in sorted(wins.items())},
                   rejected=rejected,
                   certified_score_reduction=str(reduction))
    Path(args.wins).write_text(json.dumps(payload, indent=1) + "\n",
                               encoding="utf-8")
    print(json.dumps(dict(candidates=len(candidates), wins=len(wins),
                          rejected=len(rejected),
                          certified_score_reduction=str(reduction)),
                     indent=1))


if __name__ == "__main__":
    main()
