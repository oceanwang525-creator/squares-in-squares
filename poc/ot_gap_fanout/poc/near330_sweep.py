#!/usr/bin/env python3
"""Parallel fine sweep around the n=330 transport near miss."""
from __future__ import annotations

import itertools
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = Path(__file__).with_name("near330.py")
OUT = ROOT / "poc/ot_gap_fanout/docs/near330-grid"


def run_one(params: tuple[float, float, float, str]) -> dict:
    alpha, delta, epsilon, schedule = params
    label = (
        f"a{alpha:.5f}_d{delta:.2e}_e{epsilon:.4f}_"
        + schedule.replace(",", "_").replace(":", "")
    ).replace("-", "m")
    path = OUT / f"{label}.json"
    env = os.environ.copy()
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        env[key] = "1"
    command = [
        sys.executable,
        str(SCRIPT),
        "--iterations",
        "20000",
        "--alpha",
        str(alpha),
        "--delta",
        str(delta),
        "--epsilon",
        str(epsilon),
        f"--schedule={schedule}",
        "--skip-contractions",
        "--out",
        str(path),
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )
    if completed.returncode:
        return {
            "params": [alpha, delta, epsilon, schedule],
            "error": completed.stdout + completed.stderr,
        }
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "params": [alpha, delta, epsilon, schedule],
        "path": str(path.relative_to(ROOT)),
        "tightened_side": data["tightened_side"],
        "stage_violation": data["stages"][-1]["violation"],
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base = (0.05, 7.5e-7, 0.0185, "-1:2.2,1:1")
    grid = {base}
    grid.update(
        (alpha, base[1], base[2], base[3])
        for alpha in (0.0498, 0.0499, 0.0501, 0.0502)
    )
    grid.update(
        (base[0], delta, base[2], base[3])
        for delta in (7.48e-7, 7.49e-7, 7.51e-7, 7.52e-7)
    )
    grid.update(
        (base[0], base[1], epsilon, base[3])
        for epsilon in (0.01848, 0.01849, 0.01851, 0.01852)
    )
    grid.update(
        (base[0], base[1], base[2], f"-1:{weight},1:1")
        for weight in (2.17, 2.18, 2.19, 2.21, 2.22, 2.23)
    )
    grid = list(grid)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(run_one, grid))
    errors = [row for row in results if "tightened_side" not in row]
    results = [row for row in results if "tightened_side" in row]
    results.sort(key=lambda row: row["tightened_side"])
    report = {
        "incumbent": 18.692388155425192,
        "runs": len(results),
        "best": results[:20],
        "errors": errors[:10],
    }
    (ROOT / "poc/ot_gap_fanout/docs/near330-grid.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["best"][:10], indent=2))


if __name__ == "__main__":
    main()
