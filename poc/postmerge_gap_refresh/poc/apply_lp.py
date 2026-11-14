#!/usr/bin/env python3
"""Merge certified fixed-angle LP wins into the gap-refresh final."""

from __future__ import annotations

import json
import sys
from decimal import Decimal
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
POC = ROOT / "poc" / "postmerge_gap_refresh"
sys.path.insert(0, str(ROOT / "poc" / "baseline" / "poc" / "src"))
sys.path.insert(0, str(ROOT / "poc" / "auto_union" / "poc"))
from squares import core  # noqa: E402
import union as auto_union  # noqa: E402


def main() -> None:
    final = POC / "final.csv"
    literal, _pandas = auto_union.load(final)
    merged = {}

    sources = [
        POC / "docs" / "lp-wins.json",
        POC / "docs" / "full-lp-wins.json",
    ]
    for source in sources:
        if not source.exists():
            continue
        wins = json.loads(source.read_text(encoding="utf-8"))
        for key, row in wins.items():
            n = int(key)
            rebuilt = [
                {
                    "square_id": str(index),
                    "x": item["x"],
                    "y": item["y"],
                    "angle": item["angle"],
                }
                for index, item in enumerate(row["rows"])
            ]
            if len(rebuilt) != n:
                raise ValueError(f"Malformed LP win n={key}")
            literal[n] = rebuilt
            merged[n] = row

    final.write_text(auto_union.final_text(literal), encoding="utf-8")
    literal_groups, pandas_groups = auto_union.load(final)
    if set(literal_groups) != set(pandas_groups):
        raise ValueError("Literal and pandas groups disagree")
    for n in sorted(literal_groups):
        if len(literal_groups[n]) != n or len(pandas_groups[n]) != n:
            raise ValueError(f"Incomplete final group n={n}")
        core.certify(n, literal_groups[n])
        core.certify(n, pandas_groups[n])

    summary = {
        "sources": [str(source.relative_to(ROOT)) for source in sources if source.exists()],
        "replaced_groups": sorted(merged),
        "replacement_count": len(merged),
        "accepted_groups": sorted(literal_groups),
        "accepted_count": len(literal_groups),
        "lp_score_reduction": str(
            sum(
                (Decimal(row["old_L"]) - Decimal(row["new_L"]))
                / Decimal(n).sqrt()
                for n, row in merged.items()
            )
        ),
    }
    (POC / "docs" / "lp-merge.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
