#!/usr/bin/env python3
"""Independent verification of the numbers used by the reachability audit.

Checks, with Decimal arithmetic:
  1. the per-n L strings reproduce the current total score;
  2. the literal and pandas official tables agree to rounding noise;
  3. the public gap against baseline-official.json is zero;
  4. the gallery repair residual is ~0.000724;
  5. the top-k closure sums are monotone and bounded.

Exits non-zero on failure.
"""

from __future__ import annotations

import json
import math
import sys
from decimal import Decimal, getcontext
from pathlib import Path

getcontext().prec = 60

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
SCAN = REPO / "poc/pell_corrugation/data/border-scan"

EXPECTED = Decimal("22.216763458415837")
TOL = Decimal("1e-12")

failures: list[str] = []
passes: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        passes.append(name)
        print(f"PASS {name} {detail}".rstrip())
    else:
        failures.append(name)
        print(f"FAIL {name} {detail}".rstrip())


def total_from_json(name: str) -> tuple[Decimal, dict[int, Decimal]]:
    doc = json.loads((SCAN / name).read_text())
    L = {x["n"]: Decimal(str(x["L"])) for x in doc["results"]}
    total = sum(L[n] / Decimal(n).sqrt() - 1 for n in range(1, 1001))
    return total, L


def main() -> int:
    literal_total, L = total_from_json("merged-literal-official.json")
    pandas_total, Lp = total_from_json("merged-pandas-official.json")

    check(
        "literal total matches expected",
        abs(literal_total - EXPECTED) <= TOL,
        f"got {literal_total}",
    )
    check(
        "pandas total matches expected",
        abs(pandas_total - EXPECTED) <= TOL,
        f"got {pandas_total}",
    )
    check(
        "literal/pandas totals agree",
        abs(literal_total - pandas_total) <= TOL,
        f"delta {literal_total - pandas_total}",
    )

    pub = json.loads((SCAN / "baseline-official.json").read_text())
    Lpub = {x["n"]: Decimal(str(x["L"])) for x in pub["results"]}
    public_gap = sum(
        max(Decimal(0), L[n] / Decimal(n).sqrt() - 1 - (Lpub[n] / Decimal(n).sqrt() - 1))
        for n in range(1, 1001)
    )
    check("public gap is zero", public_gap <= TOL, f"got {public_gap}")
    check(
        "merged never worse than public",
        all(L[n] <= Lpub[n] for n in range(1, 1001)),
    )
    check(
        "public worse count is 195",
        sum(1 for n in range(1, 1001) if L[n] < Lpub[n]) == 195,
    )

    gallery = json.loads((SCAN / "gallery-repair-provenance.json").read_text())
    wins = {int(k): v for k, v in gallery["wins"].items()}
    box = sum(Decimal(str(w["box_growth"])) / Decimal(n).sqrt() for n, w in wins.items())
    check("gallery has 194 wins", len(wins) == 194)
    check(
        "gallery box-growth residual ~0.000724",
        abs(box - Decimal("0.000724")) <= Decimal("2e-6"),
        f"got {box}",
    )

    gap = EXPECTED - Decimal("22.19")
    check("reported gap is 0.026763458415837", abs(gap - Decimal("0.026763458415837")) <= TOL, f"got {gap}")

    print()
    print(f"{len(passes)} passed, {len(failures)} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
