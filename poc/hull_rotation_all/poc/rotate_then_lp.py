"""Step 3 of the C3 audit: rotate, then re-shrink with the fixed-angle box LP.

``poc/alln_fixed_lp`` shows that for *fixed* square angles the smallest box side
is a linear program in the centres.  A rigid rotation of the whole packing
rotates every separating axis with it, so the LP is equivariant: the optimum of
the rotated instance is the rotation of the optimum of the original instance and
the optimal side is identical.  This script checks that claim empirically (it
would only fail if the axis-selection heuristic in ``build_lp`` changed
discretely), by running the unchanged ``alln_fixed_lp`` machinery at several
rigid rotations of a few of the largest rows.

The LP functions are imported read-only from that folder; nothing there is
modified.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "poc/baseline/poc/src"))
from squares import core  # noqa: E402


def load_module():
    path = REPO / "poc/alln_fixed_lp/poc/alln_fixed_lp.py"
    spec = importlib.util.spec_from_file_location("alln_fixed_lp", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["alln_fixed_lp"] = module
    spec.loader.exec_module(module)
    return module


def read_groups(path: Path) -> dict[int, list[dict]]:
    groups: dict[int, list[dict]] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            groups.setdefault(int(row["n"]), []).append(row)
    for n, rows in groups.items():
        rows.sort(key=lambda r: int(r["square_id"]))
    return groups


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--incumbent", required=True, type=Path)
    ap.add_argument("--audit", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--ns", default="492,791,908")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--time-limit", type=float, default=60.0)
    ap.add_argument("--extra-deg", default="5,22.5,45")
    ap.add_argument("--certify", action="store_true",
                    help="certify every LP solution with the official metric")
    args = ap.parse_args()

    alp = load_module()
    audit = json.loads(args.audit.read_text(encoding="utf-8"))["rows"]
    groups = read_groups(args.incumbent)
    out = {}
    for n in [int(x) for x in args.ns.split(",") if x.strip()]:
        angles = np.array([float(r["angle"]) for r in groups[n]])
        centers = np.array([[float(r["x"]), float(r["y"])] for r in groups[n]])
        thetas = [0.0, audit[str(n)]["theta"]]
        thetas += [math.radians(float(d)) for d in args.extra_deg.split(",") if d.strip()]
        entry = dict(base_side=alp.exact_side(n, centers, angles), trials=[])
        for theta in thetas:
            c, s = math.cos(theta), math.sin(theta)
            rot = np.array([[c, -s], [s, c]])
            moved = centers @ rot.T
            result = alp.refine(n, moved, angles + theta, args.rounds, args.time_limit)
            side = None if result is None else result[0]
            trial = dict(theta=theta, theta_deg=math.degrees(theta), lp_side=side)
            if args.certify and result is not None:
                cand = alp.to_rows(n, result[1], angles + theta)
                try:
                    trial["literal_L"] = core.certify(n, cand)
                    trial["literal_ok"] = True
                except Exception as error:  # noqa: BLE001
                    trial["literal_ok"] = False
                    trial["literal_error"] = str(error)[:160]
                if trial["literal_ok"]:
                    transported = core.kaggle_roundtrip({n: cand})
                    trial["pandas_L"] = core.certify(n, transported[n])
            entry["trials"].append(trial)
            print(f"n={n} theta={math.degrees(theta):8.4f} lp_side={side} "
                  f"certified={trial.get('literal_ok')} "
                  f"{trial.get('literal_error', trial.get('literal_L', ''))}", flush=True)
        out[str(n)] = entry
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
