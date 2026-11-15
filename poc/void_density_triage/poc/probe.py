#!/usr/bin/env python3
"""Screen 1-square insertions into canonical n-1 packings."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "poc/baseline/poc/src"))

from squares.core import certify, kaggle_roundtrip, read_submission  # noqa: E402


def solve_targets(args):
    puzzles = read_submission(ROOT / "poc/pell_corrugation/data/border-scan/merged.csv")
    docs = ROOT / "poc/void_density_triage/docs"
    docs.mkdir(parents=True, exist_ok=True)
    output = Path(args.out)
    if not output.is_absolute():
        output = ROOT / output
    records = []
    wins = []

    for n in [int(v) for v in args.targets.split(",")]:
        if n <= 1:
            raise ValueError("target must be at least 2")
        incumbent_n = puzzles[n]
        source_offset = args.source_offset
        source_n = puzzles[n - source_offset]
        incumbent_L = float(certify(n, incumbent_n))
        source_L = float(certify(n - source_offset, source_n))
        target_L = incumbent_L - args.safety
        if source_L >= target_L:
            records.append({
                "n": n,
                "incumbent_L": repr(incumbent_L),
                "target_L": repr(target_L),
                "source_n": n - source_offset,
                "source_L": repr(source_L),
                "status": "source-not-smaller",
                "found": None,
            })
            print(f"n={n}: source L is not below target", flush=True)
            continue

        xs = np.array([float(r["x"]) for r in source_n], dtype=np.float64)
        ys = np.array([float(r["y"]) for r in source_n], dtype=np.float64)
        aa = np.array([float(r["angle"]) for r in source_n], dtype=np.float64)
        # The stored source box is already below target_L. Translate it to the
        # bottom-left with a small positive wall margin and use the extra side
        # only as insertion clearance.
        support = 0.5 * (np.abs(np.cos(aa)) + np.abs(np.sin(aa)))
        xmin = float(np.min(xs - support))
        ymin = float(np.min(ys - support))
        xmax = float(np.max(xs + support))
        ymax = float(np.max(ys + support))
        width = xmax - xmin
        height = ymax - ymin
        # Equal translation keeps every center distance and every contact exact.
        ox = 0.5 * (target_L - width) - xmin
        oy = 0.5 * (target_L - height) - ymin
        xs = xs + ox
        ys = ys + oy
        tree = cKDTree(np.column_stack([xs, ys]))

        modes = []
        for a in aa:
            q = round(a / math.pi * 4) / 4 * math.pi
            if all(abs(q - b) > 1e-8 for b in modes):
                modes.append(q)
        angles = list(modes[:8])
        for deg in (0, 10, 15, 20, 22.62, 25, 28.07, 30, 32, 35, 36.87,
                    40, 42, 45, 48, 52, 55):
            for sign in (1, -1):
                q = math.radians(sign * deg)
                if all(abs(q - b) > 1e-8 for b in angles):
                    angles.append(q)

        step = args.grid_step
        found = None
        coarse_tested = 0
        for angle in angles:
            r = 0.5 * (abs(math.cos(angle)) + abs(math.sin(angle)))
            vals = np.arange(r, target_L - r + 1e-12, step)
            if len(vals) < 2:
                continue
            gx, gy = np.meshgrid(vals, vals, indexing="xy")
            pts = np.column_stack([gx.ravel(), gy.ravel()])
            coarse_tested += len(pts)
            k = min(32, len(xs))
            dist, ii = tree.query(pts, k=k, workers=args.workers)
            if k == 1:
                dist = dist[:, None]
                ii = ii[:, None]
            # Ignore neighbors far enough that no pair of unit squares can
            # overlap. The remaining exact SAT work is vectorized.
            relevant = dist < (math.sqrt(2.0) + 2.0 * step)
            candidate_ok = np.ones(len(pts), dtype=bool)
            for j in range(k):
                active = relevant[:, j]
                if not np.any(active):
                    continue
                p = pts[active]
                q = ii[active, j]
                dx = xs[q] - p[:, 0]
                dy = ys[q] - p[:, 1]
                axis_a = aa[q]
                axis_b = angle
                sep = np.zeros(len(p), dtype=bool)
                for axis in (axis_a, axis_a + 0.5 * math.pi,
                             axis_b, axis_b + 0.5 * math.pi):
                    ux = np.cos(axis)
                    uy = np.sin(axis)
                    proj = np.abs(dx * ux + dy * uy)
                    ra = 0.5 * (np.abs(np.cos(axis_a - axis)) +
                                np.abs(np.sin(axis_a - axis)))
                    rb = 0.5 * (np.abs(np.cos(axis_b - axis)) +
                                np.abs(np.sin(axis_b - axis)))
                    sep |= proj >= ra + rb - 2e-12
                candidate_ok[active] &= sep
            idx = np.flatnonzero(candidate_ok)
            if idx.size:
                # Prefer points away from walls when several coarse hits exist.
                x = pts[idx, 0]
                y = pts[idx, 1]
                order = np.argsort(-np.minimum.reduce([
                    x - r, target_L - r - x, y - r, target_L - r - y,
                ]))[:args.top]
                for j in order:
                    found = {
                        "x": float(x[j]),
                        "y": float(y[j]),
                        "angle": float(angle),
                        "coarse_clearance": float(min(
                            x[j] - r, target_L - r - x[j],
                            y[j] - r, target_L - r - y[j],
                        )),
                    }
                    break
            if found is not None:
                break

        status = "coarse-hit" if found else "no-insertion"
        record = {
            "n": n,
            "incumbent_L": repr(incumbent_L),
            "target_L": repr(target_L),
            "source_n": n - source_offset,
            "source_L": repr(source_L),
            "angles": len(angles),
            "coarse_points_tested": coarse_tested,
            "found": found,
            "status": status,
        }
        if found and len(source_n) + 1 == n:
            rows = []
            for i, (x, y, a) in enumerate(zip(xs, ys, aa)):
                rows.append({
                    "square_id": str(i),
                    "x": format(float(x), ".17g"),
                    "y": format(float(y), ".17g"),
                    "angle": format(float(a), ".17g"),
                })
            rows.append({
                "square_id": str(len(rows)),
                "x": format(found["x"], ".17g"),
                "y": format(found["y"], ".17g"),
                "angle": format(found["angle"], ".17g"),
            })
            try:
                candidate_L = float(certify(n, rows))
                transported = kaggle_roundtrip({n: rows})
                transported_L = float(certify(n, transported[n]))
                record["literal_L"] = repr(candidate_L)
                record["pandas_L"] = repr(transported_L)
                if candidate_L < incumbent_L - 1e-9 and transported_L < incumbent_L - 1e-9:
                    record["status"] = "certified-win"
                    wins.append((n, rows, candidate_L, transported_L))
                else:
                    record["status"] = "coarse-hit-not-strict"
            except Exception as exc:
                record["status"] = "coarse-hit-cert-failed"
                record["cert_error"] = f"{type(exc).__name__}: {exc}"
        elif found:
            record["status"] = "partial-first-hit"
        records.append(record)
        print(f"n={n}: {record['status']} target_L={target_L:.12g}", flush=True)

    if wins:
        with (ROOT / "poc/void_density_triage/final.csv").open(
            "w", newline="", encoding="utf-8"
        ) as fh:
            writer = csv.writer(fh)
            writer.writerow(["id", "x", "y", "deg"])
            for n, rows, _, _ in wins:
                for i, row in enumerate(rows):
                    writer.writerow([
                        f"{n}_{i}",
                        row["x"],
                        row["y"],
                        format(math.degrees(float(row["angle"])), ".17g"),
                    ])

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output} ({len(wins)} certified wins)", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", default="803,328,791")
    parser.add_argument("--grid-step", type=float, default=0.05)
    parser.add_argument("--safety", type=float, default=1e-7)
    parser.add_argument("--source-offset", type=int, default=1)
    parser.add_argument("--top", type=int, default=4)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument(
        "--out", default="poc/void_density_triage/docs/results.json"
    )
    solve_targets(parser.parse_args())


if __name__ == "__main__":
    main()
