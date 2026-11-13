#!/usr/bin/env python3
"""Gate final rows and certify the full union with both official transports."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from decimal import Decimal
from pathlib import Path

import pandas as pd


POC_DIR = Path(__file__).resolve().parents[1]
ROOT = POC_DIR.parents[1]
sys.path.insert(0, str(ROOT / "poc" / "claude-midn" / "poc"))
sys.path.insert(0, str(ROOT / "poc" / "baseline" / "poc" / "src"))

import rows as rowio  # noqa: E402
from squares import core  # noqa: E402


def report_sides(report: dict) -> dict[int, Decimal]:
    return {
        int(item["n"]): Decimal(item["L"])
        for item in report["results"]
        if item["L"] is not None
    }


def require_complete(report: dict, label: str) -> None:
    state = (
        report["accepted"],
        report["missing"],
        report["rejected"],
    )
    if state != (1000, 0, 0):
        raise SystemExit(
            f"{label}: accepted={state[0]} missing={state[1]} rejected={state[2]}"
        )


def canonical_rows(rows: list[dict[str, str]]) -> set[tuple[Decimal, Decimal, Decimal]]:
    return {
        (Decimal(row["x"]), Decimal(row["y"]), Decimal(row["angle"]))
        for row in rows
    }


def is_grid_fallback(n: int, rows: list[dict[str, str]]) -> bool:
    return canonical_rows(rows) == canonical_rows(core.grid(n))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--incumbent",
        type=Path,
        default=ROOT
        / "poc"
        / "pell_corrugation"
        / "data"
        / "border-scan"
        / "merged.csv",
    )
    parser.add_argument("--final", type=Path, default=POC_DIR / "final.csv")
    parser.add_argument("--full-out", type=Path, required=True)
    parser.add_argument("--docs", type=Path, default=POC_DIR / "docs")
    args = parser.parse_args()

    incumbent = rowio.read_groups(args.incumbent)
    final = rowio.read_groups(args.final)
    if sorted(incumbent) != list(range(1, 1001)):
        raise SystemExit("incumbent must contain every complete group n=1..1000")

    gate: dict[str, dict[str, str]] = {}
    for n, candidate in sorted(final.items()):
        old = rowio.both(n, incumbent[n])
        new = rowio.both(n, candidate)
        if new is None:
            raise SystemExit(f"n={n}: rejected by the official metric")
        if not (new[0] < old[0] and new[1] < old[1]):
            raise SystemExit(
                f"n={n}: not a strict win "
                f"(dL literal={old[0] - new[0]}, pandas={old[1] - new[1]})"
            )
        if is_grid_fallback(n, candidate):
            raise SystemExit(f"n={n}: final rows are the grid fallback")
        gate[str(n)] = {
            "old_literal": str(old[0]),
            "new_literal": str(new[0]),
            "old_pandas": str(old[1]),
            "new_pandas": str(new[1]),
        }

    union = dict(incumbent)
    union.update(final)
    args.full_out.parent.mkdir(parents=True, exist_ok=True)
    rowio.write_full(args.full_out, union)

    literal = core.score(core.read_submission(args.full_out), max_n=1000)
    require_complete(literal, "literal")
    with tempfile.TemporaryDirectory() as directory:
        roundtrip_path = Path(directory) / "roundtrip.csv"
        pd.read_csv(args.full_out).to_csv(roundtrip_path, index=False)
        transported = core.score(
            core.read_submission(roundtrip_path), max_n=1000
        )
    require_complete(transported, "pandas round trip")

    old_sides = {
        n: Decimal(gate[str(n)]["old_literal"]) for n in sorted(final)
    }
    old_pandas = {
        n: Decimal(gate[str(n)]["old_pandas"]) for n in sorted(final)
    }
    literal_sides = report_sides(literal)
    pandas_sides = report_sides(transported)
    for n in sorted(final):
        if (
            literal_sides[n] != Decimal(gate[str(n)]["new_literal"])
            or pandas_sides[n] != Decimal(gate[str(n)]["new_pandas"])
        ):
            raise SystemExit(f"n={n}: full-file score disagrees with group gate")
        if literal_sides[n] >= old_sides[n] or pandas_sides[n] >= old_pandas[n]:
            raise SystemExit(f"n={n}: full-file gate is not a strict win")

    args.docs.mkdir(parents=True, exist_ok=True)
    (args.docs / "FINAL_GATE.json").write_text(
        json.dumps(gate, indent=1) + "\n", encoding="utf-8"
    )
    (args.docs / "official-literal.json").write_text(
        json.dumps(literal, indent=1) + "\n", encoding="utf-8"
    )
    (args.docs / "official-pandas.json").write_text(
        json.dumps(transported, indent=1) + "\n", encoding="utf-8"
    )
    print(
        f"gate={len(gate)}/{len(gate)} "
        f"literal={literal['score']:.15f} {literal['accepted']}/1000 "
        f"pandas={transported['score']:.15f} {transported['accepted']}/1000"
    )


if __name__ == "__main__":
    main()
