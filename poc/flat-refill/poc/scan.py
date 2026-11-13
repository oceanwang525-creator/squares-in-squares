#!/usr/bin/env python3
"""Try shrinking flat tilted-pocket rows after exact upright refill."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "poc" / "baseline" / "poc" / "src"))
from squares import core  # noqa: E402

DELTAS = (
    0.0,
    1e-10,
    2e-10,
    5e-10,
    1e-9,
    2e-9,
    5e-9,
    1e-8,
    2e-8,
    5e-8,
    1e-7,
)
DILATIONS = (1e-12, 2e-12, 5e-12, 1e-11, 2e-11, 5e-11, 1e-10, 5e-10, 1e-9)


def load_rows(frame: pd.DataFrame, n: int) -> list[dict]:
    group = frame[frame.n == n].sort_values("square_id")
    return [
        {
            "n": str(n),
            "square_id": str(row.square_id),
            "x": repr(float(row.x)),
            "y": repr(float(row.y)),
            "angle": repr(float(row.angle)),
        }
        for row in group.itertuples()
    ]


def normalized_tilted(rows: list[dict]):
    angle = np.array([float(row["angle"]) for row in rows])
    x = np.array([float(row["x"]) for row in rows])
    y = np.array([float(row["y"]) for row in rows])
    extent = 0.5 * (np.abs(np.cos(angle)) + np.abs(np.sin(angle)))
    x0 = float((x - extent).min())
    y0 = float((y - extent).min())
    side = max(
        float((x + extent).max() - x0),
        float((y + extent).max() - y0),
    )
    degree = np.degrees(angle) % 90.0
    tilted = np.minimum(degree, 90.0 - degree) > 1e-7
    pocket = np.column_stack((x[tilted] - x0, y[tilted] - y0, degree[tilted]))
    return side, pocket


def certify_candidate(n: int, filled: np.ndarray):
    points = np.asarray(filled[:n], dtype=float).copy()
    points[:, 2] = np.radians(points[:, 2])
    x, y, angle = points.T
    extent = 0.5 * (np.abs(np.cos(angle)) + np.abs(np.sin(angle)))
    cx = ((x - extent).min() + (x + extent).max()) / 2.0
    cy = ((y - extent).min() + (y + extent).max()) / 2.0
    results = []
    for dilation in DILATIONS:
        xx = cx + (x - cx) * (1.0 + dilation)
        yy = cy + (y - cy) * (1.0 + dilation)
        rows = [
            {
                "n": str(n),
                "square_id": str(i),
                "x": repr(float(xx[i])),
                "y": repr(float(yy[i])),
                "angle": repr(float(angle[i])),
            }
            for i in range(n)
        ]
        try:
            literal = Decimal(core.certify(n, rows))
            transported = core.kaggle_roundtrip({n: rows})[n]
            pandas_l = Decimal(core.certify(n, transported))
        except Exception:
            continue
        results.append(
            {
                "dilation": dilation,
                "literal": literal,
                "pandas": pandas_l,
                "rows": rows,
            }
        )
        break
    return results[0] if results else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", required=True, type=Path)
    parser.add_argument("--fillfast-dir", required=True, type=Path)
    parser.add_argument("--targets", default="627,678,843,902,963")
    parser.add_argument("--summary", required=True, type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.fillfast_dir))
    import fillfast

    frame = pd.read_csv(args.submission)
    targets = [int(value) for value in args.targets.split(",") if value]
    report = []
    wins = {}
    for n in targets:
        rows = load_rows(frame, n)
        current = Decimal(core.certify(n, rows))
        side, pocket = normalized_tilted(rows)
        target = {
            "n": n,
            "current_literal": str(current),
            "side": side,
            "tilted": len(pocket),
            "attempts": [],
        }
        for delta in DELTAS:
            candidate_side = side - delta
            filled = fillfast.fill(
                candidate_side, pocket, mode="exact", time_limit=20
            )
            attempt = {
                "delta": delta,
                "count": len(filled),
                "certified": None,
            }
            if len(filled) >= n:
                cert = certify_candidate(n, filled)
                if cert is not None:
                    attempt["certified"] = {
                        "dilation": cert["dilation"],
                        "literal": str(cert["literal"]),
                        "pandas": str(cert["pandas"]),
                    }
                    if cert["literal"] < current and cert["pandas"] < current:
                        wins[n] = cert["rows"]
                        break
            target["attempts"].append(attempt)
        report.append(target)

    rows_out: list[dict] = []
    for n in sorted(wins):
        rows_out.extend(wins[n])
    args.summary.write_text(
        json.dumps(
            {"wins": sorted(wins), "targets": report},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    if rows_out:
        rows_path = args.summary.with_name("winning-rows.csv")
        with rows_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=("n", "square_id", "x", "y", "angle"),
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows_out)
    print(f"targets={len(targets)} wins={len(wins)} {sorted(wins)}")


if __name__ == "__main__":
    main()
