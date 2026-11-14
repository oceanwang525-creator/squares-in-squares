#!/usr/bin/env python3
"""Rebase this POC onto upstream/merge and certify the complete union."""

from __future__ import annotations

import argparse
import io
import json
import re
import subprocess
import sys
import tempfile
from decimal import Decimal
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
POC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "poc" / "auto_union" / "poc"))
import union as auto_union  # noqa: E402


def source_text() -> tuple[str, str]:
    tree = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", "upstream/merge"],
        cwd=ROOT,
        text=True,
    ).splitlines()
    files = [name for name in tree if re.fullmatch(r"merged-[0-9.]+\.csv", name)]
    if len(files) != 1:
        raise RuntimeError(f"Expected one upstream merge CSV, found {files}")
    text = subprocess.check_output(
        ["git", "show", f"upstream/merge:{files[0]}"],
        cwd=ROOT,
        text=True,
    )
    return files[0], text


def score_mode(name: str, groups: dict[int, list[dict[str, str]]]) -> dict:
    result = auto_union.score(groups)
    if (
        result["accepted"] != 1000
        or result["missing"]
        or result["rejected"]
    ):
        raise RuntimeError(f"{name} failed official certification")
    print(f"{name} score={result['score']:.17g}", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=7)
    parser.add_argument("--skip-base-score", action="store_true")
    args = parser.parse_args()
    auto_union.metric.KAGGLE_CORES = max(1, min(args.workers, 8))

    base_name, base_text = source_text()
    base_literal = auto_union.parse_text(base_text)
    base_pandas = auto_union.parse_text(
        pd.read_csv(io.StringIO(base_text)).to_csv(index=False)
    )
    candidate_literal, candidate_pandas = auto_union.load(POC / "final.csv")
    if set(candidate_literal) != set(candidate_pandas):
        raise RuntimeError("Candidate transports disagree")

    tasks: dict[
        tuple[str, int],
        tuple[int, list[dict[str, str]], list[dict[str, str]]],
    ] = {}
    for n in sorted(candidate_literal):
        if len(candidate_literal[n]) != n or len(candidate_pandas[n]) != n:
            raise RuntimeError(f"Incomplete candidate group n={n}")
        if n not in base_literal or n not in base_pandas:
            raise RuntimeError(f"Missing baseline group n={n}")
        tasks[("base", n)] = (n, base_literal[n], base_pandas[n])
        tasks[("candidate", n)] = (
            n,
            candidate_literal[n],
            candidate_pandas[n],
        )

    cache: dict[str, str | None] = {}
    unique_tasks = {
        auto_union.key(task): task
        for task in tasks.values()
    }
    auto_union.certify_many(unique_tasks.values(), args.workers, cache)
    for task in tasks.values():
        if cache[auto_union.key(task)] is None:
            raise RuntimeError(f"Certification rejected group n={task[0]}")

    wins: list[dict[str, str | int]] = []
    strict: dict[int, list[dict[str, str]]] = {}
    for n in sorted(candidate_literal):
        old = tuple(
            map(Decimal, cache[auto_union.key(tasks[("base", n)])])
        )
        new = tuple(
            map(Decimal, cache[auto_union.key(tasks[("candidate", n)])])
        )
        if new[0] < old[0] and new[1] < old[1]:
            strict[n] = candidate_literal[n]
            wins.append(
                {
                    "n": n,
                    "old_L": str(max(old)),
                    "new_L": str(max(new)),
                    "dL": str(max(old) - max(new)),
                }
            )
        else:
            print(f"DROP n={n}: old={max(old)} new={max(new)}", flush=True)

    (POC / "final.csv").write_text(
        auto_union.final_text(strict),
        encoding="utf-8",
    )
    literal, pandas = auto_union.load(POC / "final.csv")
    if set(literal) != set(strict) or set(pandas) != set(strict):
        raise RuntimeError("Pruned final.csv changed groups during transport")
    for n in strict:
        if len(literal[n]) != n or len(pandas[n]) != n:
            raise RuntimeError(f"Incomplete exported group n={n}")

    merged = dict(base_literal)
    merged.update(literal)
    with tempfile.TemporaryDirectory(prefix="postmerge-gap-refresh-") as td:
        union_path = Path(td) / "merged.csv"
        union_path.write_text(auto_union.csv_text(merged), encoding="utf-8")
        union_literal, union_pandas = auto_union.load(union_path)
        after_literal = score_mode("union-literal", union_literal)
        after_pandas = score_mode("union-pandas", union_pandas)

    if args.skip_base_score:
        match = re.fullmatch(r"merged-([0-9.]+)\.csv", base_name)
        if not match:
            raise RuntimeError(f"Cannot parse baseline score from {base_name}")
        before = {
            "literal": float(match.group(1)),
            "pandas": float(match.group(1)),
        }
    else:
        before_literal = score_mode("base-literal", base_literal)
        before_pandas = score_mode("base-pandas", base_pandas)
        before = {
            "literal": before_literal["score"],
            "pandas": before_pandas["score"],
        }

    docs = POC / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    gate = {
        "source_file": base_name,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "upstream/merge"],
            cwd=ROOT,
            text=True,
        ).strip(),
        "before_literal": before["literal"],
        "before_pandas": before["pandas"],
        "literal": after_literal["score"],
        "pandas": after_pandas["score"],
        "reduction_literal": Decimal(str(before["literal"]))
        - Decimal(str(after_literal["score"])),
        "reduction_pandas": Decimal(str(before["pandas"]))
        - Decimal(str(after_pandas["score"])),
        "accepted_literal": after_literal["accepted"],
        "accepted_pandas": after_pandas["accepted"],
        "wins": sorted(strict),
        "win_count": len(strict),
    }
    (docs / "final-gate.json").write_text(
        json.dumps(gate, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    (docs / "union-official.json").write_text(
        json.dumps(
            {
                "before": {"literal": before_literal, "pandas": before_pandas}
                if not args.skip_base_score
                else before,
                "after": {"literal": after_literal, "pandas": after_pandas},
                "wins": wins,
            },
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    table = ["| n | upstream L | final L | dL |", "|---:|---:|---:|---:|"]
    for row in wins:
        table.append(
            f"| {row['n']} | `{row['old_L']}` | `{row['new_L']}` | "
            f"`{Decimal(str(row['dL'])):.27E}` |"
        )
    (docs / "per-n.md").write_text("\n".join(table) + "\n", encoding="utf-8")

    print(json.dumps(gate, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
