#!/usr/bin/env python3
"""Verify the carried H75 rows against the tracked incumbent."""
from __future__ import annotations

import json
import sys
from decimal import Decimal
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
POC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "poc" / "auto_union" / "poc"))
import union as auto_union  # noqa: E402


def main() -> None:
    final = POC / "final.csv"
    assert final.read_text(encoding="utf-8").splitlines()[0] == "id,x,y,deg"

    base_literal, base_pandas = auto_union.load(auto_union.BASE)
    literal, pandas = auto_union.load(final)
    assert set(literal) == set(pandas)
    assert len(literal) == 69
    assert {740, 861}.isdisjoint(literal)

    wins = []
    total_ds = Decimal(0)
    for n in sorted(literal):
        assert len(literal[n]) == n
        assert [int(r["square_id"]) for r in literal[n]] == list(range(n))
        before = auto_union.check((n, base_literal[n], base_pandas[n]))
        after = auto_union.check((n, literal[n], pandas[n]))
        assert before is not None and after is not None
        before_dec = tuple(map(Decimal, before))
        after_dec = tuple(map(Decimal, after))
        assert all(a < b for a, b in zip(after_dec, before_dec))
        worst = max(after_dec)
        dL = max(before_dec) - worst
        ds = dL / Decimal(n).sqrt()
        total_ds += ds
        wins.append(
            {
                "n": n,
                "before_literal": str(before_dec[0]),
                "before_pandas": str(before_dec[1]),
                "after_literal": str(after_dec[0]),
                "after_pandas": str(after_dec[1]),
                "dL": str(dL),
                "ds": str(ds),
            }
        )

    summary = {
        "count": len(wins),
        "total_ds": str(total_ds),
        "score_before": "22.15325720501",
        "score_after": str(Decimal("22.15325720501") - total_ds),
        "wins": wins,
    }
    docs = POC / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "verify_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
