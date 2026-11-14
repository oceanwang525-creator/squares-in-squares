"""Recompute the official per-n L (50 decimal digits) for a submission CSV.

Used to pin the *fresh main* reference before any rotation claim is made.
Writes {"total": ..., "L": {n: "..."} } and refuses to continue silently when
the total does not match the expected value.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "poc/baseline/poc/src"))
from squares import core  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("submission", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--pandas", action="store_true",
                    help="score the pandas read_csv/to_csv transport instead")
    args = ap.parse_args()

    puzzles = core.read_submission(args.submission)
    if len(puzzles) != 1000:
        raise SystemExit(f"expected 1000 groups, found {len(puzzles)}")
    if args.pandas:
        puzzles = core.kaggle_roundtrip(puzzles)
    report = core.score(puzzles)
    if report["accepted"] != 1000:
        raise SystemExit(f"accepted={report['accepted']} missing={report['missing']} "
                         f"rejected={report['rejected']}")
    total = math.fsum(r["score"] for r in report["results"])
    payload = dict(source=str(args.submission), pandas=args.pandas,
                   total=repr(total),
                   L={str(r["n"]): r["L"] for r in report["results"]})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(f"total={total!r} accepted={report['accepted']} -> {args.output}")


if __name__ == "__main__":
    main()
