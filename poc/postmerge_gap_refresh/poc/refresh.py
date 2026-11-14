#!/usr/bin/env python3
"""Reopen certified positive contact gaps in the current upstream merge."""

from __future__ import annotations

import csv
import io
import json
import math
import subprocess
import sys
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "poc" / "baseline" / "poc" / "src"))
sys.path.insert(0, str(ROOT / "poc" / "auto_union" / "poc"))
from squares import core  # noqa: E402
import union as auto_union  # noqa: E402


def upstream_merge() -> tuple[str, str]:
    tree = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", "upstream/merge"],
        cwd=ROOT,
        text=True,
    ).splitlines()
    files = [name for name in tree if name.startswith("merged-") and name.endswith(".csv")]
    if len(files) != 1:
        raise RuntimeError(f"Expected one upstream merge artifact, found {files}")
    text = subprocess.check_output(
        ["git", "show", f"upstream/merge:{files[0]}"],
        cwd=ROOT,
        text=True,
    )
    return files[0], text


def groups_from_canonical(text: str) -> dict[int, list[dict[str, str]]]:
    groups: dict[int, list[dict[str, str]]] = {}
    for row in csv.DictReader(io.StringIO(text)):
        groups.setdefault(int(row["n"]), []).append(
            {
                "square_id": row["square_id"],
                "x": row["x"],
                "y": row["y"],
                "angle": row["angle"],
            }
        )
    return groups


def pair_limit(rows: list[dict[str, str]]) -> tuple[float, float]:
    points = np.array(
        [[float(row["x"]), float(row["y"]), float(row["angle"])] for row in rows]
    )
    pairs = cKDTree(points[:, :2]).query_pairs(2.0, output_type="ndarray")
    if len(pairs) == 0:
        return 1.0, math.inf

    i, j = pairs.T
    dx = points[j, 0] - points[i, 0]
    dy = points[j, 1] - points[i, 1]
    ai, aj = points[i, 2], points[j, 2]
    ca, sa = np.cos(ai), np.sin(ai)
    cb, sb = np.cos(aj), np.sin(aj)
    axes_x = np.stack((ca, -sa, cb, -sb), axis=1)
    axes_y = np.stack((sa, ca, sb, cb), axis=1)
    projection = dx[:, None] * axes_x + dy[:, None] * axes_y
    cosine = ca * cb + sa * sb
    sine = ca * sb - sa * cb
    radius = 0.5 * (1.0 + np.abs(cosine) + np.abs(sine))
    gap = (np.abs(projection) - radius[:, None]).max(axis=1)
    distance = np.hypot(dx, dy)
    return float((gap / distance).min()), float(gap.min())


def bbox_center(rows: list[dict[str, str]]) -> tuple[Decimal, Decimal]:
    xmin = xmax = ymin = ymax = None
    for row in rows:
        angle = float(row["angle"])
        half = 0.5 * (abs(math.cos(angle)) + abs(math.sin(angle)))
        x, y = float(row["x"]), float(row["y"])
        xmin = x - half if xmin is None else min(xmin, x - half)
        xmax = x + half if xmax is None else max(xmax, x + half)
        ymin = y - half if ymin is None else min(ymin, y - half)
        ymax = y + half if ymax is None else max(ymax, y + half)
    return (
        Decimal(repr((xmin + xmax) / 2.0)),
        Decimal(repr((ymin + ymax) / 2.0)),
    )


def shrink(
    rows: list[dict[str, str]], ratio: float, factor: float
) -> list[dict[str, str]]:
    center_x, center_y = bbox_center(rows)
    with localcontext() as context:
        context.prec = 80
        scale = Decimal(1) - Decimal(repr(factor * ratio))
        quantum = Decimal("1e-55")
        output = []
        for row in rows:
            x = center_x + scale * (Decimal(row["x"]) - center_x)
            y = center_y + scale * (Decimal(row["y"]) - center_y)
            output.append(
                {
                    "square_id": row["square_id"],
                    "x": format(x.quantize(quantum), "f"),
                    "y": format(y.quantize(quantum), "f"),
                    "angle": row["angle"],
                }
            )
        return output


def certify_pair(n: int, rows: list[dict[str, str]]) -> tuple[str, str] | None:
    try:
        literal = core.certify(n, rows)
        transported = core.kaggle_roundtrip({n: rows})[n]
        pandas = core.certify(n, transported)
    except ValueError:
        return None
    return literal, pandas


def main() -> None:
    source_name, source_text = upstream_merge()
    source = groups_from_canonical(source_text)
    if set(source) != set(range(1, 1001)):
        raise RuntimeError("Upstream merge must contain all 1000 groups")

    accepted: dict[int, list[dict[str, str]]] = {}
    report = {
        "source": "upstream/merge",
        "source_file": source_name,
        "factor": 0.95,
        "candidates": [],
        "wins": [],
    }
    for n in range(2, 1001):
        ratio, raw_gap = pair_limit(source[n])
        predicted = ratio * (
            max(float(row["x"]) for row in source[n])
            - min(float(row["x"]) for row in source[n])
        )
        if predicted <= 1e-11:
            continue
        old_literal = core.certify(n, source[n])
        old_pandas = core.certify(n, core.kaggle_roundtrip({n: source[n]})[n])
        winner = None
        for factor in (0.95, 0.80, 0.60, 0.40, 0.20):
            candidate = shrink(source[n], ratio, factor)
            lengths = certify_pair(n, candidate)
            if lengths is None:
                continue
            literal, pandas = lengths
            if Decimal(literal) < Decimal(old_literal) and Decimal(pandas) < Decimal(old_pandas):
                winner = {
                    "n": n,
                    "factor": factor,
                    "ratio": ratio,
                    "raw_min_gap": raw_gap,
                    "old_L": str(max(Decimal(old_literal), Decimal(old_pandas))),
                    "new_L": str(max(Decimal(literal), Decimal(pandas))),
                    "literal_L": literal,
                    "pandas_L": pandas,
                }
                accepted[n] = candidate
                break
        report["candidates"].append(
            {
                "n": n,
                "ratio": ratio,
                "raw_min_gap": raw_gap,
                "winner": winner,
            }
        )
        if winner is not None:
            report["wins"].append(winner)
            print(
                f"WIN n={n} dL={Decimal(winner['old_L']) - Decimal(winner['new_L']):.3e}",
                flush=True,
            )

    final = ROOT / "poc" / "postmerge_gap_refresh" / "final.csv"
    final.write_text(auto_union.final_text(accepted), encoding="utf-8")

    literal_groups, pandas_groups = auto_union.load(final)
    for n in sorted(accepted):
        if len(literal_groups[n]) != n or len(pandas_groups[n]) != n:
            raise RuntimeError(f"Incomplete exported group n={n}")
        core.certify(n, literal_groups[n])
        core.certify(n, pandas_groups[n])

    report["accepted_groups"] = sorted(accepted)
    report["score_gain"] = str(
        sum(
            (Decimal(item["old_L"]) - Decimal(item["new_L"])) / Decimal(item["n"]).sqrt()
            for item in report["wins"]
        )
    )
    docs = ROOT / "poc" / "postmerge_gap_refresh" / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "results.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "accepted": len(accepted),
                "score_gain": report["score_gain"],
                "wins": report["accepted_groups"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
