#!/usr/bin/env python3
"""Certify H131 exports against both upstream and H128 in two transports."""
from __future__ import annotations

from decimal import Decimal
import importlib.util
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
UNION_PATH = REPO_ROOT / "poc" / "auto_union" / "poc" / "union.py"
SPEC = importlib.util.spec_from_file_location("auto_union_verify_h131", UNION_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {UNION_PATH}")
union = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = union
SPEC.loader.exec_module(union)


def certified_pair(path: Path, n: int) -> tuple[Decimal, Decimal]:
    literal, transported = union.load(path)
    if n not in literal or n not in transported:
        raise ValueError(f"{path}: missing n={n}")
    result = union.check((n, literal[n], transported[n]))
    if result is None:
        raise ValueError(f"{path}: n={n} transport certification failed")
    return tuple(map(Decimal, result))


def main() -> None:
    upstream_path = (
        REPO_ROOT / "poc/pell_corrugation/data/border-scan/merged.csv"
    )
    h128_path = REPO_ROOT / "poc/swarm_masked_melt_bootstrap"
    final_path = Path(__file__).resolve().parents[1] / "final.csv"
    literal, _ = union.load(final_path)

    rows = []
    for n in sorted(literal):
        upstream = certified_pair(upstream_path, n)
        h128 = certified_pair(h128_path / "data" / f"win_n{n}.csv", n)
        after = certified_pair(final_path, n)
        d_up = tuple(after[i] - upstream[i] for i in range(2))
        d_h128 = tuple(after[i] - h128[i] for i in range(2))
        if any(value >= 0 for value in d_up + d_h128):
            raise ValueError(f"n={n}: not a strict dual win")
        rows.append(
            {
                "n": n,
                "upstream_literal": str(upstream[0]),
                "upstream_pandas": str(upstream[1]),
                "h128_literal": str(h128[0]),
                "h128_pandas": str(h128[1]),
                "after_literal": str(after[0]),
                "after_pandas": str(after[1]),
                "dL_upstream_literal": str(d_up[0]),
                "dL_upstream_pandas": str(d_up[1]),
                "dL_h128_literal": str(d_h128[0]),
                "dL_h128_pandas": str(d_h128[1]),
            }
        )

    print(json.dumps({"rows": rows, "count": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
