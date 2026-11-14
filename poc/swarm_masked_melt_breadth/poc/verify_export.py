#!/usr/bin/env python3
"""Re-read final.csv and certify every exported group in both transport modes."""
from __future__ import annotations

from decimal import Decimal
import importlib.util
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
UNION_PATH = REPO_ROOT / "poc" / "auto_union" / "poc" / "union.py"
SPEC = importlib.util.spec_from_file_location("auto_union_verify", UNION_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {UNION_PATH}")
union = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = union
SPEC.loader.exec_module(union)


def main() -> None:
    baseline_path = REPO_ROOT / "poc/pell_corrugation/data/border-scan/merged.csv"
    final_path = Path(__file__).resolve().parents[1] / "final.csv"
    baseline_literal, baseline_transported = union.load(baseline_path)
    literal, transported = union.load(final_path)

    rows = []
    for n in sorted(literal):
        if set(literal) != set(transported):
            raise ValueError("literal and pandas transports found different groups")
        result = union.check((n, literal[n], transported[n]))
        if result is None:
            raise ValueError(f"n={n}: transport certification failed")
        baseline_result = union.check(
            (n, baseline_literal[n], baseline_transported[n])
        )
        if baseline_result is None:
            raise ValueError(f"n={n}: baseline certification failed")
        before_literal, before_pandas = map(Decimal, baseline_result)
        after_literal, after_pandas = map(Decimal, result)
        d_literal = after_literal - before_literal
        d_pandas = after_pandas - before_pandas
        if d_literal >= 0 or d_pandas >= 0:
            raise ValueError(f"n={n}: not a strict win in both transports")
        rows.append(
            {
                "n": n,
                "before_literal": str(before_literal),
                "before_pandas": str(before_pandas),
                "after_literal": str(after_literal),
                "after_pandas": str(after_pandas),
                "dL_literal": str(d_literal),
                "dL_pandas": str(d_pandas),
            }
        )

    print(json.dumps({"rows": rows, "count": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
