#!/usr/bin/env python3
"""Merge PR #5 candidates into the current local portfolio with a strict gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import tempfile
from decimal import Decimal
from pathlib import Path

import pandas as pd


POC_DIR = Path(__file__).resolve().parents[1]
ROOT = POC_DIR.parents[1]
sys.path.insert(0, str(ROOT / "poc" / "baseline" / "poc" / "src"))

from squares import core  # noqa: E402

COLUMNS = ("n", "square_id", "x", "y", "angle")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_groups(path: Path) -> dict[int, list[dict]]:
    groups: dict[int, list[dict]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            n = int(row["n"])
            groups.setdefault(n, []).append(
                {
                    "n": str(n),
                    "square_id": str(row["square_id"]),
                    "x": row["x"],
                    "y": row["y"],
                    "angle": row["angle"],
                }
            )
    for n, rows in groups.items():
        rows.sort(key=lambda row: int(row["square_id"]))
        if len(rows) != n:
            raise ValueError(f"n={n}: expected {n} rows, got {len(rows)}")
    return groups


def certify_both(n: int, rows: list[dict]) -> tuple[Decimal, Decimal]:
    literal = Decimal(core.certify(n, rows))
    transported = core.kaggle_roundtrip({n: rows})[n]
    pandas_l = Decimal(core.certify(n, transported))
    return literal, pandas_l


def write_full(path: Path, groups: dict[int, list[dict]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("id", "n", "square_id", "x", "y", "angle"))
        identifier = 0
        for n in range(1, 1001):
            for row in groups[n]:
                writer.writerow(
                    (
                        identifier,
                        n,
                        row["square_id"],
                        row["x"],
                        row["y"],
                        row["angle"],
                    )
                )
                identifier += 1


def write_changed_rows(
    path: Path,
    groups: dict[int, list[dict]],
    changed_ns: set[int],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        for n in sorted(changed_ns):
            writer.writerows(groups[n])


def report_sides(report: dict) -> dict[int, Decimal]:
    return {
        int(row["n"]): Decimal(row["L"])
        for row in report["results"]
        if row["L"] is not None
    }


def require_all(report: dict, label: str) -> None:
    if (
        report["accepted"] != 1000
        or report["missing"] != 0
        or report["rejected"] != 0
    ):
        raise SystemExit(
            f"{label}: accepted={report['accepted']} "
            f"missing={report['missing']} rejected={report['rejected']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--incumbent",
        type=Path,
        default=ROOT / "poc" / "dual-pocket" / "data" / "submission.csv",
    )
    parser.add_argument(
        "--candidate-rows",
        type=Path,
        default=POC_DIR / "data" / "pr5-winning-rows.csv",
    )
    parser.add_argument(
        "--baseline-report",
        type=Path,
        default=ROOT
        / "poc"
        / "pell_corrugation"
        / "data"
        / "border-scan"
        / "merged-literal-official.json",
    )
    parser.add_argument(
        "--baseline-pandas-report",
        type=Path,
        default=ROOT
        / "poc"
        / "pell_corrugation"
        / "data"
        / "border-scan"
        / "merged-pandas-official.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    groups = read_groups(args.incumbent)
    if sorted(groups) != list(range(1, 1001)):
        raise SystemExit("incumbent must contain exactly n=1..1000")
    candidates = read_groups(args.candidate_rows)

    accepted: dict[int, dict] = {}
    for n, candidate_rows in sorted(candidates.items()):
        old_literal, old_pandas = certify_both(n, groups[n])
        new_literal, new_pandas = certify_both(n, candidate_rows)
        if new_literal < old_literal and new_pandas < old_pandas:
            groups[n] = candidate_rows
            accepted[n] = {
                "old_literal": str(old_literal),
                "new_literal": str(new_literal),
                "old_pandas": str(old_pandas),
                "new_pandas": str(new_pandas),
            }

    write_full(args.output, groups)
    literal = core.score(core.read_submission(args.output), max_n=1000)
    require_all(literal, "literal")
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, encoding="utf-8"
    ) as handle:
        temporary = Path(handle.name)
    try:
        pd.read_csv(args.output).to_csv(temporary, index=False)
        transported = core.score(core.read_submission(temporary), max_n=1000)
        require_all(transported, "pandas round trip")
    finally:
        temporary.unlink(missing_ok=True)

    baseline_literal = report_sides(
        json.loads(args.baseline_report.read_text(encoding="utf-8"))
    )
    baseline_pandas = report_sides(
        json.loads(args.baseline_pandas_report.read_text(encoding="utf-8"))
    )
    literal_sides = report_sides(literal)
    pandas_sides = report_sides(transported)

    wins: list[int] = []
    for n in range(1, 1001):
        if (
            literal_sides[n] > baseline_literal[n]
            or pandas_sides[n] > baseline_pandas[n]
        ):
            raise SystemExit(f"regression at n={n}")
        if (
            literal_sides[n] < baseline_literal[n]
            and pandas_sides[n] < baseline_pandas[n]
        ):
            wins.append(n)

    changed_rows = POC_DIR / "data" / "changed-rows.csv"
    write_changed_rows(changed_rows, groups, set(wins))
    summary = {
        "literal_score": literal["score"],
        "pandas_score": transported["score"],
        "accepted_literal": literal["accepted"],
        "accepted_pandas": transported["accepted"],
        "strict_wins": len(wins),
        "strict_regressions": 0,
        "winning_n": wins,
        "pr5_complement_n": sorted(accepted),
        "output_sha256": sha256(args.output),
        "changed_rows_sha256": sha256(changed_rows),
        "candidate_wins": accepted,
    }
    summary_path = POC_DIR / "data" / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (POC_DIR / "docs" / "literal.json").write_text(
        json.dumps(literal, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (POC_DIR / "docs" / "pandas.json").write_text(
        json.dumps(transported, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"pr5_complement={len(accepted)} "
        f"literal={literal['score']:.15f} "
        f"pandas={transported['score']:.15f} "
        f"wins={len(wins)} losses=0"
    )


if __name__ == "__main__":
    main()
