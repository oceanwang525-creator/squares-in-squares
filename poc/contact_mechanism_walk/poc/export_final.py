#!/usr/bin/env python3
"""Export only complete private groups that beat the current merge artifact."""
from __future__ import annotations

import argparse
import json
import sys
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
UNION_PATH = ROOT / "poc" / "auto_union" / "poc" / "union.py"


def load_union():
    sys.path.insert(0, str(UNION_PATH.parent))
    import union

    return union


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--best", type=Path, required=True)
    parser.add_argument("--incumbent", type=Path, required=True)
    parser.add_argument("--origin", type=Path, required=True)
    parser.add_argument("--carry", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    union = load_union()
    best, best_transport = union.load(args.best)
    incumbent, incumbent_transport = union.load(args.incumbent)
    origin, origin_transport = union.load(args.origin)
    carried, carried_transport = union.load(args.carry)
    if set(best) != set(range(1, 1001)):
        raise ValueError("Merge artifact must contain all n=1..1000")
    if set(incumbent) != set(best):
        raise ValueError("Private incumbent must contain all n=1..1000")
    if set(origin) != set(best):
        raise ValueError("Origin incumbent must contain all n=1..1000")

    changed_ns = {n for n in incumbent if incumbent[n] != origin[n]}
    candidate_ns = set(carried) | changed_ns

    sources = dict(carried)
    for n in candidate_ns - set(carried):
        sources[n] = incumbent[n]
    candidate_tasks = [(n, sources[n], (
        incumbent_transport[n] if n in incumbent_transport and n not in carried
        else carried_transport[n]
    )) for n in sorted(sources)]
    baseline_tasks = [(n, best[n], best_transport[n]) for n in sorted(sources)]
    cache: dict[str, tuple[str, str] | None] = {}
    union.certify_many(candidate_tasks + baseline_tasks, 4, cache)

    groups = {}
    details = {}
    for n, rows, transported in candidate_tasks:
        candidate = cache[union.key((n, rows, transported))]
        baseline = cache[union.key((n, best[n], best_transport[n]))]
        if candidate is None:
            raise ValueError(f"Uncertified private group n={n}")
        if baseline is None:
            raise ValueError(f"Uncertified merge group n={n}")
        candidate_sides = tuple(map(Decimal, candidate))
        baseline_sides = tuple(map(Decimal, baseline))
        if all(a < b for a, b in zip(candidate_sides, baseline_sides)):
            groups[n] = rows
            details[n] = {
                "before": [str(value) for value in baseline_sides],
                "after": [str(value) for value in candidate_sides],
                "dL": str(candidate_sides[0] - baseline_sides[0]),
            }

    args.out.write_text(union.final_text(groups), encoding="utf-8")
    literal, transported = union.load(args.out)
    exported_tasks = [(n, literal[n], transported[n]) for n in sorted(literal)]
    union.certify_many(exported_tasks, 4, cache)
    for n, rows, transported in exported_tasks:
        result = cache[union.key((n, rows, transported))]
        if result is None:
            raise ValueError(f"Exported file failed certification for n={n}")
        if tuple(map(Decimal, result)) != tuple(map(Decimal, details[n]["after"])):
            raise ValueError(f"Exported L changed during transport for n={n}")

    report = {
        "count": len(groups),
        "ns": sorted(groups),
        "details": {str(n): details[n] for n in sorted(details)},
        "sha256": union.digest(args.out.read_bytes()),
    }
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("count", "ns", "sha256")}, indent=2))


if __name__ == "__main__":
    main()
