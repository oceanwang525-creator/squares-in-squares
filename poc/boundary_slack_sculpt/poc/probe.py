#!/usr/bin/env python3
"""Coherent wall-profile slack waves with exact-event re-jamming."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
import tempfile
import time
from decimal import Decimal
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_DISABLE_CACHING", "1")
os.environ["NUMBA_CACHE_DIR"] = tempfile.mkdtemp(prefix="boundary-slack-numba-")

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "poc/pell_corrugation/data/border-scan/merged.csv"
BASELINE = ROOT / "poc/baseline/poc/src"
GEOM = ROOT / "poc/hydrogen_brainstorm/poc/geom_utils.py"
FINAL = ROOT / "poc/boundary_slack_sculpt/final.csv"
if str(BASELINE) not in sys.path:
    sys.path.insert(0, str(BASELINE))

import squares.core as core  # noqa: E402
from squares import _official_metric as metric  # noqa: E402


def load_geom():
    spec = importlib.util.spec_from_file_location("boundary_geom", GEOM)
    module = importlib.util.module_from_spec(spec)
    sys.modules["boundary_geom"] = module
    spec.loader.exec_module(module)
    return module


g = load_geom()


def load_group(n: int):
    rows = []
    with BASE.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(row["n"]) == n:
                rows.append(row)
    if len(rows) != n:
        raise ValueError(f"n={n}: expected {n} incumbent rows, got {len(rows)}")
    x = np.array([float(r["x"]) for r in rows], dtype=np.float64)
    y = np.array([float(r["y"]) for r in rows], dtype=np.float64)
    th = np.array([float(r["angle"]) for r in rows], dtype=np.float64)
    return x, y, th


def extents(th: np.ndarray) -> np.ndarray:
    return 0.5 * (np.abs(np.cos(th)) + np.abs(np.sin(th)))


def normalize(x: np.ndarray, y: np.ndarray, th: np.ndarray):
    h = extents(th)
    x = x - np.min(x - h) + 1e-12
    y = y - np.min(y - h) + 1e-12
    side = float(max(np.max(x + h), np.max(y + h)))
    return x, y, side


def adjacency(x: np.ndarray, y: np.ndarray, th: np.ndarray):
    h = extents(th)
    neighbors: list[set[int]] = [set() for _ in range(len(x))]
    tree = cKDTree(np.column_stack([x, y]))
    for i, js in enumerate(tree.query_ball_point(np.column_stack([x, y]), 1.55)):
        for raw_j in js:
            j = int(raw_j)
            if i >= j:
                continue
            dx, dy = x[j] - x[i], y[j] - y[i]
            ci, si = np.cos(th[i]), np.sin(th[i])
            cj, sj = np.cos(th[j]), np.sin(th[j])
            rad = 0.5 + 0.5 * (
                abs(ci * cj + si * sj) + abs(ci * sj - si * cj)
            )
            projections = (
                dx * ci + dy * si,
                -dx * si + dy * ci,
                dx * cj + dy * sj,
                -dx * sj + dy * cj,
            )
            gap = max(abs(v) for v in projections) - rad
            if gap <= 2e-5:
                neighbors[i].add(j)
                neighbors[j].add(i)
    return neighbors


def wall_profiles(x: np.ndarray, y: np.ndarray, th: np.ndarray):
    h = extents(th)
    side = float(max(np.max(x + h), np.max(y + h)))
    walls = {
        "left": (np.flatnonzero(x - h <= 2e-5), "y"),
        "right": (np.flatnonzero(side - x - h <= 2e-5), "y"),
        "bottom": (np.flatnonzero(y - h <= 2e-5), "x"),
        "top": (np.flatnonzero(side - y - h <= 2e-5), "x"),
    }
    result = {}
    for name, (idx, axis) in walls.items():
        coord = y if axis == "y" else x
        order = idx[np.argsort(coord[idx])] if len(idx) else idx
        result[name] = [int(v) for v in order]
    return result, side


def expand_chain(chain: list[int], neighbors, hops: int = 1) -> np.ndarray:
    frozen = set()
    active = set(chain)
    for _ in range(hops):
        frontier = set()
        for i in active:
            frontier.update(neighbors[i])
        active.update(frontier)
        frozen.update(frontier)
    return np.array(sorted(active), dtype=np.int64)


def wave_seed(
    x: np.ndarray,
    y: np.ndarray,
    th: np.ndarray,
    wall: str,
    chain: list[int],
    amp: float,
    phase: float,
    rotation: float,
):
    nx, ny, nth = x.copy(), y.copy(), th.copy()
    if not chain:
        return nx, ny, nth
    axis_coord = y if wall in ("left", "right") else x
    values = axis_coord[chain]
    span = float(np.ptp(values))
    if span < 1e-12:
        fractions = np.arange(len(chain), dtype=np.float64) / max(1, len(chain) - 1)
    else:
        fractions = (values - values.min()) / span
    wave = np.sin(2 * np.pi * (fractions + phase))
    inward = {
        "left": np.array([1.0, 0.0]),
        "right": np.array([-1.0, 0.0]),
        "bottom": np.array([0.0, 1.0]),
        "top": np.array([0.0, -1.0]),
    }[wall]
    tangent = np.array([-inward[1], inward[0]])
    for i, w in zip(chain, wave):
        shift = inward * amp + tangent * (0.35 * amp * float(w))
        nx[i] += shift[0]
        ny[i] += shift[1]
        nth[i] += rotation * float(w)
    return nx, ny, nth


def wall_ownership(x, y, th, side):
    h = extents(th)
    out = {}
    for name in ("left", "right", "bottom", "top"):
        if name == "left":
            gap = x - h
        elif name == "right":
            gap = side - x - h
        elif name == "bottom":
            gap = y - h
        else:
            gap = side - y - h
        out[name] = set(int(i) for i in np.flatnonzero(np.abs(gap) <= 2e-5))
    return out


def ownership_delta(before, after):
    return {
        wall: {
            "left": sorted(before[wall] - after[wall]),
            "gained": sorted(after[wall] - before[wall]),
        }
        for wall in before
    }


def export_win(n: int, verdict: dict):
    rows = verdict["rows"]
    core.write_submission(FINAL, {n: rows}, validate=True)
    parsed = core.read_submission(FINAL)
    lit = core.certify(n, parsed[n])
    frame = pd.read_csv(FINAL)
    transported = metric._parse_submission_csv(frame.to_csv(index=False))
    pandas_side = core.certify(n, transported[n])
    return {"literal": lit, "pandas": pandas_side}


def probe_target(n: int, side_delta: float, amps: list[float], maxiter: int):
    x0, y0, th0 = load_group(n)
    x0, y0, incumbent = normalize(x0, y0, th0)
    neighbors = adjacency(x0, y0, th0)
    profiles, measured = wall_profiles(x0, y0, th0)
    if abs(measured - incumbent) > 1e-8:
        raise ValueError("wall profile side disagrees with normalized side")
    target_side = incumbent - side_delta
    before_owner = wall_ownership(x0, y0, th0, incumbent)
    records = []
    best = {
        "residual": float("inf"),
        "local_side": float("inf"),
        "lp_side": float("inf"),
        "seed": None,
    }

    for wall, chain in profiles.items():
        active = expand_chain(chain, neighbors, hops=1)
        free = np.zeros(n, dtype=bool)
        free[active] = True
        for amp in amps:
            for phase in (0.0, 0.25, 0.5):
                for rotation in (-0.0025, 0.0, 0.0025):
                    sx, sy, sth = wave_seed(
                        x0, y0, th0, wall, chain, amp, phase, rotation
                    )
                    rx, ry, rth, residual = g.relax_lbfgs(
                        sx, sy, sth, target_side, free_mask=free,
                        maxiter=maxiter, w_wall=8.0, w_pair=1.0,
                    )
                    if residual < 1.0e-5:
                        gx, gy, gth, global_residual = g.relax_lbfgs(
                            rx, ry, rth, target_side, free_mask=None,
                            maxiter=min(260, maxiter + 80),
                            w_wall=8.0, w_pair=1.0,
                        )
                        if global_residual < residual:
                            rx, ry, rth, residual = gx, gy, gth, global_residual
                    local_side = float(
                        max(np.max(rx + extents(rth)), np.max(ry + extents(rth)))
                    )
                    lp_side = float("inf")
                    lp_ok = False
                    if residual < 3e-4:
                        tx, ty, lp_side, lp_ok = g.fixed_angle_lp_tighten(
                            rx, ry, rth, rounds=4, margin=1.0e-9
                        )
                    else:
                        tx, ty = rx, ry
                    after_owner = wall_ownership(rx, ry, rth, local_side)
                    rec = {
                        "wall": wall,
                        "chain": len(chain),
                        "active": int(active.size),
                        "amp": amp,
                        "phase": phase,
                        "rotation": rotation,
                        "residual": residual,
                        "local_side": local_side,
                        "lp_side": lp_side,
                        "lp_feasible": lp_ok,
                        "ownership_delta": ownership_delta(before_owner, after_owner),
                    }
                    records.append(rec)
                    rank = (residual, lp_side, local_side)
                    best_rank = (best["residual"], best["lp_side"], best["local_side"])
                    if rank < best_rank:
                        best = {
                            "residual": residual,
                            "local_side": local_side,
                            "lp_side": lp_side,
                            "seed": rec,
                        }
                    if lp_ok and lp_side < incumbent - 1.0e-9:
                        verdict = g.verify_official(n, tx, ty, rth)
                        if verdict["ok"]:
                            certified = float(
                                max(Decimal(verdict["L_lit"]), Decimal(verdict["L_pandas"]))
                            )
                            if certified < incumbent - 1.0e-9:
                                checked = export_win(n, verdict)
                                return {
                                    "n": n,
                                    "incumbent_L": str(incumbent),
                                    "target_L": str(target_side),
                                    "wall_contacts": {
                                        k: len(v) for k, v in profiles.items()
                                    },
                                    "attempts": len(records),
                                    "best": best,
                                    "win": {
                                        "L_lit": verdict["L_lit"],
                                        "L_pandas": verdict["L_pandas"],
                                        "dL": certified - incumbent,
                                        "seed": rec,
                                        "file_check": checked,
                                    },
                                }

    return {
        "n": n,
        "incumbent_L": str(incumbent),
        "target_L": str(target_side),
        "wall_contacts": {k: len(v) for k, v in profiles.items()},
        "attempts": len(records),
        "best": best,
        "win": None,
        "records": records,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", default="803,908,969")
    parser.add_argument("--side-delta", type=float, default=1e-4)
    parser.add_argument("--maxiter", type=int, default=180)
    parser.add_argument(
        "--out", type=Path,
        default=ROOT / "poc/boundary_slack_sculpt/docs/results.json",
    )
    args = parser.parse_args()
    targets = [int(v) for v in args.targets.split(",") if v]
    amps = [0.004, 0.010, 0.020, 0.035]
    results = []
    for n in targets:
        started = time.time()
        result = probe_target(n, args.side_delta, amps, args.maxiter)
        result["elapsed_seconds"] = time.time() - started
        results.append(result)
        best = result["best"]
        print(
            f"n={n} attempts={result['attempts']} residual={best['residual']:.6g} "
            f"local={best['local_side']:.12g} lp={best['lp_side']:.12g} "
            f"win={result['win'] is not None}",
            flush=True,
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"targets": results}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
