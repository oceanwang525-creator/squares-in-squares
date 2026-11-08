"""Slope Band POC: Variable-Slope Band Construction & Optimization Driver.

Top-level CLI and programmatic interface for variable-slope band optimization
and pocket construction around theta = 24.10 degrees (0.420631 rad).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

from variable_slope_optimizer import (
    HERE,
    PORTFOLIO_BASELINE,
    PUBLIC_TARGET_INCUMBENTS,
    certify_packing,
    extent,
    generate_variable_slope_target,
    load_incumbent_coordinates,
    normalize_box,
    optimize_variable_slope_band,
    sat_check,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Variable-Slope Band Optimizer CLI")
    parser.add_argument("--n", type=int, default=908, help="Target n (default: 908)")
    parser.add_argument("--iterations", type=int, default=25, help="LP iterations")
    parser.add_argument("--theta-deg", type=float, default=24.1004, help="Initial band angle (deg)")
    parser.add_argument("--free-angle", action="store_true", help="Enable free angle LP trust region")
    parser.add_argument("--out", type=Path, default=HERE.parent / "data", help="Output directory")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    theta_rad = math.radians(args.theta_deg)
    print(f"Loading or generating layout for n={args.n} at theta={args.theta_deg:.4f} deg ({theta_rad:.6f} rad)...")

    coords, L_init = load_incumbent_coordinates(args.n)
    if coords is None:
        coords, L_init = generate_variable_slope_target(args.n, theta_rad=theta_rad)

    coords = normalize_box(coords)
    L_init = extent(coords)
    sat_ok, clearance = sat_check(coords)
    print(f"Initial extent L={L_init:.12f}, SAT clearance={clearance:.6e}")

    angle_trust = 0.005 if args.free_angle else 0.0
    result = optimize_variable_slope_band(
        coords,
        L_init=L_init,
        iterations=args.iterations,
        position_trust=0.08,
        angle_trust=angle_trust,
    )

    print(f"Optimization finished: L={result.L:.12f}, accepted={result.accepted_steps}")
    checks = certify_packing(result.q, args.out / f"packing_{args.n}.csv")
    res_data = {
        "n": args.n,
        "initial_L": L_init,
        "optimized_L": result.L,
        "accepted_steps": result.accepted_steps,
        "feasible": result.feasible,
        "clearance": result.clearance,
        "certificates": checks,
    }
    (args.out / "results.json").write_text(json.dumps(res_data, indent=2))
    print(json.dumps(res_data, indent=2))


if __name__ == "__main__":
    main()
