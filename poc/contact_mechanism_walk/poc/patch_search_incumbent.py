#!/usr/bin/env python3
"""Apply certified final.csv groups to a private legacy-format search incumbent."""
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
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--wins", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    union = load_union()
    best, transported_base = union.load(args.base)
    improvements = {}
    for path in args.wins:
        literal, transported = union.load(path)
        for n, rows in literal.items():
            current = union.check((n, best[n], transported_base[n]))
            candidate = union.check((n, rows, transported[n]))
            if current is None:
                raise ValueError(f"base group n={n} is not certified")
            if candidate is None:
                raise ValueError(f"candidate group n={n} in {path} is not certified")
            old = tuple(map(Decimal, current))
            new = tuple(map(Decimal, candidate))
            if new[0] < old[0] and new[1] < old[1]:
                best[n], transported_base[n] = rows, transported[n]
                improvements[n] = {
                    "path": str(path),
                    "before": [str(value) for value in old],
                    "after": [str(value) for value in new],
                }

    args.out.write_text(union.csv_text(best), encoding="utf-8")
    report = {
        "improved": sorted(improvements),
        "details": {str(n): row for n, row in sorted(improvements.items())},
        "sha256": union.digest(args.out.read_bytes()),
    }
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
