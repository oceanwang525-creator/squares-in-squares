#!/usr/bin/env python3
"""Certify the complete POC twice and list strict improvements over the baseline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import tempfile
from decimal import Decimal, InvalidOperation
from pathlib import Path


HERE = Path(__file__).resolve().parent
POC_DIR = HERE.parent
REPO_ROOT = POC_DIR.parents[1]
BASELINE_SRC = REPO_ROOT / "poc" / "baseline" / "poc" / "src"
sys.path.insert(0, str(BASELINE_SRC))

import pandas as pd  # noqa: E402

from squares.core import read_submission, score  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path, default=POC_DIR / "data" / "submission.csv"
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=REPO_ROOT
        / "poc"
        / "pell_corrugation"
        / "data"
        / "border-scan"
        / "merged.csv",
    )
    parser.add_argument(
        "--baseline-report",
        type=Path,
        default=REPO_ROOT
        / "poc"
        / "pell_corrugation"
        / "data"
        / "border-scan"
        / "merged-literal-official.json",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=POC_DIR / "data"
    )
    parser.add_argument(
        "--docs-dir", type=Path, default=POC_DIR / "docs"
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.docs_dir.mkdir(parents=True, exist_ok=True)

    literal_puzzles = read_submission(args.input)
    literal = score(literal_puzzles, max_n=1000)
    require_all(literal, "literal")
    literal_report = {
        "mode": "literal decimal strings",
        "input": str(args.input.resolve()),
        "sha256": sha256(args.input),
        "pandas_version": pd.__version__,
        **literal,
    }
    write_json(args.out_dir / "certification-literal.json", literal_report)

    frame = pd.read_csv(args.input)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        ) as handle:
            temporary = Path(handle.name)
        frame.to_csv(temporary, index=False)
        transported_puzzles = read_submission(temporary)
        transported = score(transported_puzzles, max_n=1000)
        require_all(transported, "pandas round trip")
        transported_report = {
            "mode": "pandas read_csv then to_csv",
            "input": str(args.input.resolve()),
            "source_sha256": sha256(args.input),
            "roundtrip_sha256": sha256(temporary),
            "pandas_version": pd.__version__,
            **transported,
        }
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    write_json(
        args.out_dir / "certification-pandas.json", transported_report
    )

    baseline_report = json.loads(args.baseline_report.read_text(encoding="utf-8"))
    baseline_sides = {
        int(row["n"]): Decimal(row["L"])
        for row in baseline_report["results"]
        if row["reason"] == "ok"
    }
    our_sides = {
        int(row["n"]): Decimal(row["L"])
        for row in literal["results"]
        if row["reason"] == "ok"
    }
    if set(our_sides) != set(range(1, 1001)):
        raise SystemExit("literal report does not contain all 1,000 certified L values")

    wins = []
    losses = []
    equal = []
    for n in range(1, 1001):
        ours, baseline = our_sides[n], baseline_sides[n]
        if ours < baseline:
            delta = baseline - ours
            wins.append(
                {
                    "n": n,
                    "ours_L": format(ours, "f"),
                    "baseline_L": format(baseline, "f"),
                    "delta_L": format(delta, "f"),
                    "score_gain": float(delta) / math.sqrt(n),
                }
            )
        elif ours > baseline:
            delta = ours - baseline
            losses.append(
                {
                    "n": n,
                    "ours_L": format(ours, "f"),
                    "baseline_L": format(baseline, "f"),
                    "delta_L": format(delta, "f"),
                    "score_loss": float(delta) / math.sqrt(n),
                }
            )
        else:
            equal.append(n)
    if not wins:
        raise SystemExit("no strict improvements over the baseline")
    total_gain = sum(row["score_gain"] for row in wins)
    total_loss = sum(row["score_loss"] for row in losses)
    net_score_gain = baseline_report["score"] - literal["score"]
    summary = {
        "input": str(args.input.resolve()),
        "input_sha256": sha256(args.input),
        "baseline": str(args.baseline.resolve()),
        "baseline_sha256": sha256(args.baseline),
        "baseline_report": str(args.baseline_report.resolve()),
        "baseline_score": baseline_report["score"],
        "literal_score": literal["score"],
        "pandas_score": transported["score"],
        "wins": len(wins),
        "losses": len(losses),
        "equal": len(equal),
        "win_score_gain": total_gain,
        "loss_score": total_loss,
        "net_score_gain": net_score_gain,
        "wins_detail": wins,
        "losses_detail": losses,
    }
    write_json(args.out_dir / "wins.json", summary)

    with (args.out_dir / "wins.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("n", "ours_L", "baseline_L", "delta_L", "score_gain"),
        )
        writer.writeheader()
        writer.writerows(wins)

    lines = [
        "# Strict wins",
        "",
        f"- literal official score: `{literal['score']:.15f}`",
        f"- pandas-roundtrip official score: `{transported['score']:.15f}`",
        f"- baseline official score: `{baseline_report['score']:.15f}`",
        f"- strict wins: **{len(wins)}**",
        f"- tiny regressions: **{len(losses)}**",
        f"- regression score: **{total_loss:.15e}**",
        f"- net official score gain: **{net_score_gain:.15f}**",
        "",
        "| n | ours L | baseline L | delta L | score gain |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    lines.extend(
        f"| {row['n']} | {row['ours_L']} | {row['baseline_L']} | "
        f"{row['delta_L']} | {row['score_gain']:.12g} |"
        for row in wins
    )
    (args.docs_dir / "WINS.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(
        f"literal={literal['score']:.15f} "
        f"pandas={transported['score']:.15f} "
        f"wins={len(wins)} losses={len(losses)} "
        f"net_gain={net_score_gain:.15f}"
    )


if __name__ == "__main__":
    main()
