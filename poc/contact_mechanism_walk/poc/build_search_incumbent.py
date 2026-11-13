#!/usr/bin/env python3
"""Build a private search incumbent from the upstream merge and our POC wins."""
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
    parser.add_argument(
        "--wins",
        type=Path,
        nargs="+",
        default=[
            ROOT / "poc" / "guestdata_atlas" / "final.csv",
            ROOT / "poc" / "contact_mechanism_walk" / "final.csv",
        ],
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--proof", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    union = load_union()
    best, sides = union.baseline_checks(args.base, args.workers, {})
    original_sides = dict(sides)
    sources: dict[int, str] = {}
    for path in args.wins:
        literal, transported = union.load(path)
        tasks = [(n, rows, transported[n]) for n, rows in literal.items()]
        cache: dict[str, tuple[str, str] | None] = {}
        union.certify_many(tasks, args.workers, cache)
        for n, rows, transported_rows in tasks:
            result = cache[union.key((n, rows, transported_rows))]
            if result is None:
                raise ValueError(f"uncertified group n={n} in {path}")
            pair = tuple(map(Decimal, result))
            if pair[0] < sides[n][0] and pair[1] < sides[n][1]:
                best[n], sides[n] = rows, pair
                sources[n] = str(path.relative_to(ROOT))

    args.out.write_text(union.csv_text(best), encoding="utf-8")
    reports = union.full_gate(args.out)
    wins = [n for n in sorted(sides) if sides[n][0] < original_sides[n][0]]
    proof = {
        "groups": len(wins),
        "ns": wins,
        "sources": {str(n): sources[n] for n in wins},
        "scores": [report["score"] for report in reports],
        "rows": sum(len(best[n]) for n in best),
        "sha256": union.digest(args.out.read_bytes()),
    }
    args.proof.write_text(json.dumps(proof, indent=2), encoding="utf-8")
    print(json.dumps({key: proof[key] for key in ("groups", "scores", "rows", "sha256")}, indent=2))


if __name__ == "__main__":
    main()
