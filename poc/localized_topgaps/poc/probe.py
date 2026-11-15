#!/usr/bin/env python3
"""Apply the H21 localized donor mixture to unresolved top-gap rows."""
from __future__ import annotations

import argparse
import importlib.util
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
H21_PATH = ROOT / "poc/localized_donor_weights/poc/probe.py"
OUT = ROOT / "poc/localized_topgaps"
FINAL = OUT / "final.csv"
DOCS = OUT / "docs"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


h21 = load_module("localized_topgaps_h21", H21_PATH)
union = h21.union


def load_best_groups() -> dict[int, np.ndarray]:
    """Use every complete final.csv as a source and an incumbent lower bound."""
    groups = h21.load_improved_groups()
    best = {n: float(h21.h16.side(q)) for n, q in groups.items()}
    for path in sorted((ROOT / "poc").glob("*/final.csv")):
        if path.resolve() == FINAL.resolve():
            continue
        try:
            parsed = union.parse_text(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for n, rows in parsed.items():
            if len(rows) != n:
                continue
            ordered = sorted(rows, key=lambda row: int(row["square_id"]))
            q = np.asarray(
                [
                    (float(r["x"]), float(r["y"]), float(r["angle"]))
                    for r in ordered
                ],
                dtype=np.float64,
            )
            side = float(h21.h16.side(q))
            if n not in groups or side < best[n]:
                groups[n] = q
                best[n] = side
    return groups


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", default="969,861,747,804,848")
    parser.add_argument("--seconds-per-target", type=float, default=120)
    parser.add_argument("--repair-iterations", type=int, default=8000)
    parser.add_argument("--top-starts", type=int, default=32)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--donor-offsets", default="1,2,3,4")
    parser.add_argument("--epsilons", default="0.0125,0.018,0.025")
    parser.add_argument("--alphas", default="0.04,0.08,0.15,0.25")
    parser.add_argument("--deltas", default="1e-5,1e-4,3e-4,6e-4,1e-3")
    parser.add_argument("--bases", default="0.25,0.4,0.5,0.65,0.75")
    parser.add_argument("--strengths", default="0.75,1.5,2.5")
    parser.add_argument("--patterns", default="x,y,d+,d-,quadrant")
    parser.add_argument("--out", default=str(DOCS / "results.json"))
    args = parser.parse_args()

    targets = [int(x) for x in args.targets.split(",") if x]
    groups = load_best_groups()
    offsets = tuple(int(x) for x in args.donor_offsets.split(",") if x)
    epsilons = tuple(float(x) for x in args.epsilons.split(",") if x)
    alphas = tuple(float(x) for x in args.alphas.split(",") if x)
    deltas = tuple(float(x) for x in args.deltas.split(",") if x)
    bases = tuple(float(x) for x in args.bases.split(",") if x)
    strengths = tuple(float(x) for x in args.strengths.split(",") if x)
    patterns = tuple(x for x in args.patterns.split(",") if x)

    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            results = list(
                pool.map(
                    h21.probe_target,
                    targets,
                    [groups] * len(targets),
                    [args.seconds_per_target] * len(targets),
                    [args.repair_iterations] * len(targets),
                    [args.top_starts] * len(targets),
                    [offsets] * len(targets),
                    [epsilons] * len(targets),
                    [alphas] * len(targets),
                    [deltas] * len(targets),
                    [bases] * len(targets),
                    [strengths] * len(targets),
                    [patterns] * len(targets),
                )
            )
    else:
        results = [
            h21.probe_target(
                n, groups, args.seconds_per_target, args.repair_iterations,
                args.top_starts, offsets, epsilons, alphas, deltas, bases,
                strengths, patterns,
            )
            for n in targets
        ]

    existing = union.parse_text(FINAL.read_text(encoding="utf-8"))
    wins = {}
    for result in results:
        item = result["certified"]
        if item is None:
            continue
        n = result["n"]
        existing[n] = union.parse_text(item["final_text"])[n]
        wins[str(n)] = {
            "literal_side": item["literal_side"],
            "pandas_side": item["pandas_side"],
        }

    FINAL.write_text(union.final_text(existing), encoding="utf-8")
    payload = {
        "method": "H21 position-localized two-donor transport on top gaps",
        "targets": targets,
        "results": [
            {k: v for k, v in result.items() if k != "certified"}
            for result in results
        ],
        "wins": wins,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"targets": targets, "wins": wins}, indent=2))


if __name__ == "__main__":
    main()
