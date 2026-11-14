#!/usr/bin/env python3
"""Bounded ownership-release probe for neighboring high-gap groups.

The donor is the trusted incumbent group with one fewer square.  A local set of
donor squares is released, the rest remains rigid, and the released seam is
refilled by the exact upright MILP used by poc/highn-mondrian/poc/fillfast.py.
This is deliberately a quick structural screen, not a global search.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "poc/baseline/poc/src"))
sys.path.insert(0, str(ROOT / "poc/highn-mondrian/poc"))

from squares import _official_metric as metric  # noqa: E402
import fillfast  # noqa: E402


def target_sides() -> dict[int, float]:
    path = ROOT / "poc/frontier_nov12/docs/per_n_gap.csv"
    with path.open() as f:
        return {int(r["n"]): float(r["L_frontier"]) for r in csv.DictReader(f)}


def incumbent() -> dict[int, list[dict[str, str]]]:
    path = ROOT / "poc/pell_corrugation/data/border-scan/merged.csv"
    return metric._parse_submission_csv(path.read_text())


def normalize(rows: list[dict[str, str]], side: float) -> np.ndarray:
    q = np.array([[float(r["x"]), float(r["y"]), float(r["angle"])] for r in rows])
    q[:, :2] -= q[:, :2].min(axis=0)
    q[:, :2] += (side - np.ptp(q[:, :2], axis=0)) / 2
    return q


def rectangular_windows(q: np.ndarray, kind: str, limit: int) -> list[tuple[int, ...]]:
    upright = np.where(np.abs(q[:, 2]) <= 1e-8)[0]
    points = q[upright, :2]
    # Center-first ordering makes the bounded scan deterministic and prefers
    # seams away from the outer wall, where ownership exchange is likelier.
    center = points.mean(axis=0)
    order = upright[np.argsort(np.linalg.norm(points - center, axis=1))]
    wins: list[tuple[int, ...]] = []
    if kind == "1x1":
        return [(int(i),) for i in order[:limit]]
    if kind == "pairs":
        out = []
        for a, b in itertools.combinations(order, 2):
            if np.linalg.norm(q[a, :2] - q[b, :2]) <= 1.3:
                out.append((int(a), int(b)))
            if len(out) >= limit:
                break
        return out
    if kind in {"2x2", "3x3"}:
        width = 2 if kind == "2x2" else 3
        # Use nearest neighbours around each ordered seed and compact their
        # coordinates into a deterministic local window.
        for seed in order:
            d = np.linalg.norm(points - q[seed, :2], axis=1)
            near = upright[np.argsort(d)[: max(8, width * width * 3)]]
            xs = sorted(set(np.round(q[near, 0], 9)))
            ys = sorted(set(np.round(q[near, 1], 9)))
            for x0, y0 in itertools.product(xs[:-width + 1], ys[:-width + 1]):
                block = tuple(
                    int(i) for i in near
                    if x0 - 1e-7 <= q[i, 0] <= x0 + width - 1 + 1e-7
                    and y0 - 1e-7 <= q[i, 1] <= y0 + width - 1 + 1e-7
                )
                if len(block) >= width * width - 1 and block not in wins:
                    wins.append(block)
            if len(wins) >= limit:
                break
        return wins[:limit]
    raise ValueError(kind)


def run(args: argparse.Namespace) -> dict:
    sides = target_sides()
    base = incumbent()
    results: list[dict] = []
    for n in args.targets:
        donor_n = n - 1
        side = sides[n]
        q = normalize(base[donor_n], side)
        for kind in args.windows:
            release_sets = rectangular_windows(q, kind, args.limit)
            best_count = donor_n
            best_release: list[int] = []
            started = time.process_time()
            for release in release_sets:
                keep = np.ones(len(q), dtype=bool)
                keep[list(release)] = False
                filled = fillfast.fill(side, q[keep], "exact", 0.0, args.milp_seconds)
                count = len(filled)
                if count > best_count:
                    best_count = count
                    best_release = list(release)
                if count >= n and args.stop_on_target:
                    break
            results.append(dict(n=n, donor=donor_n, window=kind, probes=len(release_sets),
                                best_count=best_count, short=max(0, n - best_count),
                                best_release=best_release,
                                seconds=time.process_time() - started))
            print(results[-1], flush=True)
    out = {"targets": args.targets, "windows": args.windows, "results": results}
    path = ROOT / args.output
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--targets", type=int, nargs="+", default=[682, 744, 844, 877, 979])
    p.add_argument("--windows", nargs="+", default=["1x1"], choices=["1x1", "pairs", "2x2", "3x3"])
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--milp-seconds", type=float, default=5.0)
    p.add_argument("--stop-on-target", action="store_true")
    p.add_argument("--output", default="poc/corner_lattice_borrow/data/scan.json")
    run(p.parse_args())


if __name__ == "__main__":
    main()
