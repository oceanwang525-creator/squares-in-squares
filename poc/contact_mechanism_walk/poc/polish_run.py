#!/usr/bin/env python3
"""Iteratively polish the best certified candidate in a mechanism-walk run."""
from __future__ import annotations

import argparse
import json
import os
import sys
from decimal import Decimal
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import mechanism_walk as mw  # noqa: E402


def rows_to_arrays(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.array([float(row["x"].lstrip("s")) for row in rows], dtype=np.float64)
    y = np.array([float(row["y"].lstrip("s")) for row in rows], dtype=np.float64)
    theta = np.array(
        [float(row["angle"].lstrip("s")) for row in rows], dtype=np.float64
    )
    nx, ny, _ = mw.normalize(x, y, theta)
    return nx, ny, theta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--target", type=int, required=True)
    parser.add_argument("--passes", type=int, default=5)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    result = json.loads(args.run.read_text(encoding="utf-8"))
    best = result["best"]
    if best is None:
        raise ValueError(f"run has no certified candidate for n={args.target}")

    incumbent = Decimal(str(result["incumbent_L"]))
    x, y, theta = rows_to_arrays(best["rows"])
    best_rows = best["rows"]
    best_l = max(Decimal(best["L_lit"]), Decimal(best["L_pandas"]))
    print(
        f"start L={best_l} dL={float(best_l - incumbent):+.9e}",
        flush=True,
    )

    schedules = (
        (12, 0.002, 0.003),
        (16, 0.001, 0.0015),
        (20, 0.0005, 0.0007),
        (24, 0.0002, 0.0003),
    )
    margins = (1e-8, 5e-9, 2e-9, 1e-9, 5e-10)

    for pass_index in range(1, args.passes + 1):
        pass_best = None
        for rounds, tr_theta, tr_xy in schedules:
            sx, sy, st, _, ok = mw.su.joint_xytheta_slp(
                x,
                y,
                theta,
                rounds=rounds,
                tr_th0=tr_theta,
                tr_xy0=tr_xy,
                margin=1e-9,
            )
            if not ok:
                continue

            for margin in margins:
                fx, fy, _, fixed_ok = mw.su.fixed_angle_lp_tighten(
                    sx, sy, st, rounds=20, margin=margin
                )
                if not fixed_ok:
                    continue
                verdict = mw.su.verify_official(args.target, fx, fy, st)
                if not verdict.get("ok"):
                    continue
                candidate_l = max(
                    Decimal(verdict["L_lit"]), Decimal(verdict["L_pandas"])
                )
                if pass_best is None or candidate_l < pass_best[0]:
                    pass_best = (
                        candidate_l,
                        fx,
                        fy,
                        st,
                        verdict["rows"],
                        (rounds, tr_theta, tr_xy, margin),
                    )

        if pass_best is None:
            print(f"pass {pass_index}: no certified candidate", flush=True)
            continue

        candidate_l, cx, cy, ct, rows, settings = pass_best
        print(
            f"pass {pass_index}: L={candidate_l} "
            f"dL={float(candidate_l - incumbent):+.9e} settings={settings}",
            flush=True,
        )
        if candidate_l < best_l:
            best_l = candidate_l
            best_rows = rows
            x, y, _ = mw.normalize(cx.copy(), cy.copy(), ct.copy())
            theta = ct.copy()

    payload = {
        "n": args.target,
        "incumbent_L": str(incumbent),
        "L": str(best_l),
        "dL": float(best_l - incumbent),
        "rows": best_rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"best L={best_l} dL={float(best_l - incumbent):+.9e}", flush=True)


if __name__ == "__main__":
    main()
