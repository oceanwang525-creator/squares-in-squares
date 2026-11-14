"""Independent brute-force cross-check of the rotation audit on selected rows.

Deliberately implemented differently from ``audit.py``: it rotates the *centres*
by ``R(t)``, adds ``t`` to the square angles, and forms the extents from the
rotated angles (no convex hull, no event/crossing enumeration).  A 20001-point
sweep of ``[0, pi/2]`` must never go below the audit's analytic minimum, and the
minimum of the sweep should sit at ``t = 0`` with value ``L`` for these rows.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def read_groups(path: Path) -> dict[int, list[dict]]:
    groups: dict[int, list[dict]] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            groups.setdefault(int(row["n"]), []).append(row)
    for n, rows in groups.items():
        rows.sort(key=lambda r: int(r["square_id"]))
    return groups


def side_at(p: np.ndarray, t: float) -> float:
    c, s = math.cos(t), math.sin(t)
    x = p[:, 0] * c - p[:, 1] * s
    y = p[:, 0] * s + p[:, 1] * c
    e = 0.5 * (np.abs(np.cos(p[:, 2] + t)) + np.abs(np.sin(p[:, 2] + t)))
    return float(max((x + e).max() - (x - e).min(), (y + e).max() - (y - e).min()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("incumbent", type=Path)
    ap.add_argument("audit", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--ns", default="798,499,585,542,744,746,936,791,908,969,"
                                    "803,861,492,328,365,367,902,963,1000")
    ap.add_argument("--samples", type=int, default=20001)
    args = ap.parse_args()

    audit = json.loads(args.audit.read_text(encoding="utf-8"))["rows"]
    groups = read_groups(args.incumbent)
    grid = np.linspace(0.0, math.pi / 2, args.samples)
    out = {}
    for n in [int(x) for x in args.ns.split(",") if x.strip()]:
        p = np.array([[float(r["x"]), float(r["y"]), float(r["angle"])]
                      for r in groups[n]])
        values = np.array([side_at(p, t) for t in grid])
        i = int(values.argmin())
        out[str(n)] = dict(side_axis=side_at(p, 0.0),
                           audit_side_min=audit[str(n)].get("hp", audit[str(n)])["side_min"],
                           brute_min=float(values[i]), brute_argmin=float(grid[i]),
                           consistent=audit_min_ok(audit, n, float(values[i])),
                           brute_gain=side_at(p, 0.0) - float(values[i]))
        print(f"n={n} L={out[str(n)]['side_axis']:.12f} "
              f"brute_min={values[i]:.12f} at {math.degrees(grid[i]):.6f} deg "
              f"audit={out[str(n)]['audit_side_min']:.12f}", flush=True)
    args.output.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")



def audit_min_ok(audit: dict, n: int, brute: float) -> bool:
    rec = audit[str(n)]
    return rec.get("hp", rec)["side_min"] <= brute + 1e-12


if __name__ == "__main__":
    main()
