"""Run the bounded donor-slice composite search and certify the outputs."""
from __future__ import annotations

import csv
import json
import math
import sys
import time
from decimal import Decimal, localcontext
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "poc" / "baseline" / "poc" / "src"))

import pandas as pd

import certify as certify_module
from composite import build_rows, rows_side, search_target
from crop import load_portfolio

DATA = ROOT / "poc" / "donor_slices" / "data"
PORTFOLIO = ROOT / "poc" / "pell_corrugation" / "data" / "border-scan" / "merged.csv"
INCUMBENT_JSON = (
    ROOT / "poc" / "pell_corrugation" / "data" / "border-scan"
    / "merged-literal-official.json"
)
TARGETS = (645, 925, 987)
PRIOR_CAPACITY = {645: 640, 925: 920, 987: 961}


def _placement_record(placement):
    if placement is None:
        return None
    crop = placement.crop
    return {
        "donor_n": placement.donor_n,
        "capacity": crop.capacity,
        "total_capacity": placement.total_capacity,
        "side": format(placement.side, ".17g"),
        "crop_width": format(placement.width, ".17g"),
        "crop_height": format(placement.height, ".17g"),
        "crop_cut": list(crop.cut),
        "orientation": placement.orientation,
        "grid_right": [placement.columns_right, placement.rows_right],
        "grid_top": [placement.columns_top, placement.rows_top],
        "grid_capacity": placement.grid_capacity,
    }


def _write_csv(path: Path, target: int, rows) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "n", "square_id", "x", "y", "angle"])
        for index, (square_id, x, y, angle) in enumerate(rows):
            writer.writerow([index, target, square_id, x, y, angle])


def main() -> None:
    started = time.monotonic()
    DATA.mkdir(parents=True, exist_ok=True)
    packings = load_portfolio(PORTFOLIO)
    official = {
        entry["n"]: entry
        for entry in json.loads(INCUMBENT_JSON.read_text(encoding="utf-8"))["results"]
    }

    report = {
        "portfolio": str(PORTFOLIO.relative_to(ROOT)),
        "incumbent_source": str(INCUMBENT_JSON.relative_to(ROOT)),
        "targets": {},
        "prior_capacity": {str(n): PRIOR_CAPACITY[n] for n in TARGETS},
        "pandas_version": pd.__version__,
    }

    for target in TARGETS:
        incumbent_text = official[target]["L"]
        incumbent = float(incumbent_text)
        ceiling = math.ceil(math.sqrt(target))
        min_side = max(1, ceiling - 10)

        below, match = search_target(
            packings,
            target,
            incumbent,
            min_side,
            ceiling,
            min_side,
            ceiling,
        )
        if match is None:
            raise RuntimeError(f"n={target}: no capacity-matched composite found")

        rows = build_rows(packings, match, target)
        path = DATA / f"composite-{target}.csv"
        _write_csv(path, target, rows)
        certification = certify_module.certify_file(path, target)
        measured = certification["literal_L"]

        with localcontext() as context:
            context.prec = 80
            incumbent_dec = Decimal(incumbent_text)
            measured_dec = Decimal(measured)
            delta_l = measured_dec - incumbent_dec
            root = Decimal(target).sqrt()
            incumbent_score = incumbent_dec / root - Decimal(1)
            measured_score = measured_dec / root - Decimal(1)
            delta_score = measured_score - incumbent_score

        record = {
            "incumbent_L": incumbent_text,
            "incumbent_score": format(incumbent_score, ".50g"),
            "candidate_csv": path.name,
            "candidate_rows": len(rows),
            "candidate_capacity": match.total_capacity,
            "capacity_match": match.total_capacity == target,
            "below_incumbent": _placement_record(below),
            "capacity_composite": _placement_record(match),
            "measured_side_from_rows": format(rows_side(rows), ".17g"),
            "certification": certification,
            "measured_L": measured,
            "measured_score": format(measured_score, ".50g"),
            "delta_L": str(delta_l),
            "delta_score": format(delta_score, ".50g"),
            "strict_win": delta_l < 0,
            "shortfall": (
                target - below.total_capacity if below is not None else None
            ),
            "prior_capacity": PRIOR_CAPACITY[target],
            "capacity_deviation": (
                below.total_capacity - PRIOR_CAPACITY[target]
                if below is not None
                else None
            ),
        }
        report["targets"][str(target)] = record
        print(
            f"n={target} incumbent={incumbent_text} candidate={measured} "
            f"delta_L={delta_l} capacity={match.total_capacity} "
            f"below={below.total_capacity if below else None}",
            flush=True,
        )

    report["elapsed_seconds"] = time.monotonic() - started
    report["summary"] = {
        "wins": sum(
            1 for record in report["targets"].values() if record["strict_win"]
        ),
        "note": (
            "Every target emits a certified composite, but all candidate sides "
            "exceed the incumbent. The donor slice is a negative result."
        ),
        "measured_below_incumbent_capacity": {
            n: record["below_incumbent"]["total_capacity"]
            if record["below_incumbent"] is not None
            else None
            for n, record in report["targets"].items()
        },
        "prior_below_incumbent_capacity": {
            str(n): PRIOR_CAPACITY[n] for n in TARGETS
        },
    }
    (DATA / "results.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
