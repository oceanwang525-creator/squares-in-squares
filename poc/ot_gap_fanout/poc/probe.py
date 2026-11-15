#!/usr/bin/env python3
"""Run the proven multi-donor transport pipeline over a target list."""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = ROOT / "poc/ot_gap_fanout"
DOCS = OUTPUT_DIR / "docs"
FINAL = OUTPUT_DIR / "final.csv"
BASE_PROBE = ROOT / "poc/ot_multidonor_barycenter/poc/probe.py"


def load_base():
    spec = importlib.util.spec_from_file_location("ot_fanout_base", BASE_PROBE)
    if spec is None or spec.loader is None:
        raise ImportError(BASE_PROBE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_one(n: int, seconds: float, iterations: int, top_starts: int) -> dict:
    base = load_base()
    groups = base.load_groups()
    result = base.probe_target(n, groups, seconds, iterations, top_starts)
    DOCS.mkdir(parents=True, exist_ok=True)
    checkpoint = DOCS / f"result-{n}.json"
    checkpoint.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    certified = result.pop("certified", None)
    if certified:
        (DOCS / f"win-{n}.txt").write_text(
            certified["final_text"], encoding="utf-8"
        )
        result["certified"] = {
            "literal_side": certified["literal_side"],
            "pandas_side": certified["pandas_side"],
        }
    return result


def run_child(n: int, seconds: float, iterations: int, top_starts: int) -> int:
    result = run_one(n, seconds, iterations, top_starts)
    print(json.dumps({"n": n, **result}, indent=2))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", default="")
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--seconds-per-target", type=float, default=75.0)
    parser.add_argument("--repair-iterations", type=int, default=5000)
    parser.add_argument("--top-starts", type=int, default=18)
    parser.add_argument("--one", type=int)
    args = parser.parse_args()
    if args.one is not None:
        raise SystemExit(
            run_child(
                args.one,
                args.seconds_per_target,
                args.repair_iterations,
                args.top_starts,
            )
        )

    targets = [int(item) for item in args.targets.split(",") if item]
    if not targets:
        parser.error("--targets is required")
    DOCS.mkdir(parents=True, exist_ok=True)
    for path in DOCS.glob("result-*.json"):
        path.unlink()
    for path in DOCS.glob("win-*.txt"):
        path.unlink()

    started = time.monotonic()

    def launch(n: int) -> tuple[int, str]:
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--one",
            str(n),
            "--seconds-per-target",
            str(args.seconds_per_target),
            "--repair-iterations",
            str(args.repair_iterations),
            "--top-starts",
            str(args.top_starts),
        ]
        completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        return n, completed.stdout + completed.stderr

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        logs = list(pool.map(launch, targets))

    base = load_base()
    existing = base.union.parse_text(FINAL.read_text(encoding="utf-8"))
    wins: dict[str, dict[str, str]] = {}
    for n in targets:
        win_path = DOCS / f"win-{n}.txt"
        if win_path.exists():
            parsed = base.union.parse_text(win_path.read_text(encoding="utf-8"))
            existing[n] = parsed[n]
            result = json.loads((DOCS / f"result-{n}.json").read_text(encoding="utf-8"))
            wins[str(n)] = result["certified"]
    FINAL.write_text(base.union.final_text(existing), encoding="utf-8")

    payload = {
        "method": "multi-donor Sinkhorn transport fanout plus fixed-angle certification",
        "targets": targets,
        "wins": wins,
        "elapsed_seconds": time.monotonic() - started,
        "process_logs": {str(n): output for n, output in logs},
    }
    (DOCS / "results.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"wins": wins, "elapsed_seconds": payload["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
