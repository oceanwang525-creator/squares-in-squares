#!/usr/bin/env python3
"""Walk soft contact-network mechanisms, re-jam, and compress the result."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
import warnings
from decimal import Decimal
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds

ROOT = Path(__file__).resolve().parents[3]
SWARM = ROOT / "poc" / "swarm_cascades" / "poc"
AUTO_UNION = ROOT / "poc" / "auto_union" / "poc" / "union.py"
if str(SWARM) not in sys.path:
    sys.path.insert(0, str(SWARM))

import slp_utils as su  # noqa: E402


def _load_union_module():
    spec = importlib.util.spec_from_file_location("mechanism_union", AUTO_UNION)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def half_extent(theta: np.ndarray) -> np.ndarray:
    return 0.5 * (np.abs(np.cos(theta)) + np.abs(np.sin(theta)))


def half_extent_derivative(theta: np.ndarray) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return 0.5 * (-np.sign(c) * s + np.sign(s) * c)


def _axis_half_extent(theta: np.ndarray, ax: float, ay: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    along_cos = ax * c + ay * s
    along_sin = -ax * s + ay * c
    return 0.5 * (np.abs(along_cos) + np.abs(along_sin))


def _axis_half_extent_derivative(
    theta: np.ndarray, ax: float, ay: float
) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    along_cos = ax * c + ay * s
    along_sin = -ax * s + ay * c
    return 0.5 * (
        np.sign(along_cos) * along_sin - np.sign(along_sin) * along_cos
    )


def normalize(
    x: np.ndarray, y: np.ndarray, theta: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    h = half_extent(theta)
    x = np.asarray(x, dtype=np.float64) - np.min(np.asarray(x) - h)
    y = np.asarray(y, dtype=np.float64) - np.min(np.asarray(y) - h)
    return x, y, float(max(np.max(x + h), np.max(y + h)))


def contact_jacobian(
    x: np.ndarray,
    y: np.ndarray,
    theta: np.ndarray,
    contact_tol: float = 1e-5,
) -> tuple[coo_matrix, dict]:
    """Linearize active contacts in center and rotation coordinates."""
    n = len(x)
    c, s = np.cos(theta), np.sin(theta)
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    pair_gaps: list[float] = []
    row = 0

    for i in range(n):
        for j in range(i + 1, n):
            dx, dy = x[j] - x[i], y[j] - y[i]
            if abs(dx) > 1.5 or abs(dy) > 1.5:
                continue
            rot = (
                (c[i], s[i]),
                (-s[i], c[i]),
            )
            others = (
                (c[j], s[j]),
                (-s[j], c[j]),
            )
            cth = c[i] * c[j] + s[i] * s[j]
            sth = c[i] * s[j] - s[i] * c[j]
            radius = 0.5 * (1.0 + abs(cth) + abs(sth))
            best: tuple[float, float, float, float, int] | None = None
            for owner, axis in ((i, rot[0]), (i, rot[1]), (j, others[0]), (j, others[1])):
                ax, ay = axis
                projection = dx * ax + dy * ay
                gap = abs(projection) - radius
                # Non-overlap needs every axis gap >= 0. The controlling
                # contact is therefore the least separated (largest-gap) axis.
                if best is None or gap > best[0]:
                    best = (gap, ax, ay, projection, owner)
            assert best is not None
            gap, ax, ay, projection, owner = best
            if gap > contact_tol:
                continue
            sign = 1.0 if projection >= 0.0 else -1.0
            radius_derivative = _axis_half_extent_derivative(theta, ax, ay)
            rows.extend((row, row, row, row, row, row))
            cols.extend(
                (i, n + i, 2 * n + i, j, n + j, 2 * n + j)
            )
            vals.extend(
                (
                    -sign * ax,
                    -sign * ay,
                    -radius_derivative[i],
                    sign * ax,
                    sign * ay,
                    -radius_derivative[j],
                )
            )
            # When the selected axis rotates with its owning square, its
            # orientation derivative adds one term to both the projection and
            # the relative-orientation half extent.
            if owner == i:
                vals[2] += radius_derivative[j]
                if abs(projection) > 1e-15:
                    vals[2] += sign * (-dx * ay + dy * ax)
            else:
                vals[5] += radius_derivative[i]
                if abs(projection) > 1e-15:
                    vals[5] += sign * (-dx * ay + dy * ax)
            pair_gaps.append(float(gap))
            row += 1

    h = half_extent(theta)
    wall_gaps: list[float] = []
    for axis, coord, extent in ((0, x, h), (1, y, h)):
        gap = coord - extent
        mask = gap <= contact_tol
        for i in np.flatnonzero(mask):
            rows.append(row)
            cols.append(axis * n + int(i))
            vals.append(-1.0)
            rows.append(row)
            cols.append(2 * n + int(i))
            vals.append(-float(half_extent_derivative(theta)[i]))
            wall_gaps.append(float(gap[i]))
            row += 1

    # Upper/right walls use the measured box side.
    side = max(
        float(np.max(x + h)),
        float(np.max(y + h)),
        float(np.max(x - h)),
        float(np.max(y - h)),
    )
    for axis, coord, extent in ((0, x, h), (1, y, h)):
        gap = side - coord - extent
        mask = gap <= contact_tol
        for i in np.flatnonzero(mask):
            rows.append(row)
            cols.append(axis * n + int(i))
            vals.append(-1.0)
            rows.append(row)
            cols.append(2 * n + int(i))
            vals.append(-float(half_extent_derivative(theta)[i]))
            wall_gaps.append(float(gap[i]))
            row += 1

    jac = coo_matrix((vals, (rows, cols)), shape=(row, 3 * n)).tocsr()
    return jac, {
        "constraints": int(row),
        "pair_contacts": len(pair_gaps),
        "wall_contacts": len(wall_gaps),
        "max_pair_gap": max(pair_gaps, default=None),
        "max_wall_gap": max(wall_gaps, default=None),
        "side": side,
    }


def soft_modes(
    jac: coo_matrix, k: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    """Return smallest right singular vectors and their singular values."""
    k = max(1, min(k, min(jac.shape) - 1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        _, singular, vt = svds(
            jac, k=k, which="SM", solver="lobpcg", maxiter=5000
        )
    order = np.argsort(singular)
    return singular[order], vt[order]


def _translation_fraction(mode: np.ndarray, n: int) -> float:
    vx, vy = mode[:n], mode[n : 2 * n]
    # A rigid translation has no centered variation.
    return float(math.sqrt(np.var(vx) + np.var(vy)))


def remove_rigid_motion(
    mode: np.ndarray, x: np.ndarray, y: np.ndarray
) -> np.ndarray:
    """Project out rigid translation and whole-layout rotation from a mode."""
    n = len(x)
    projected = np.asarray(mode, dtype=np.float64).copy()
    projected[:n] -= np.mean(projected[:n])
    projected[n : 2 * n] -= np.mean(projected[n : 2 * n])

    cx, cy = x - np.mean(x), y - np.mean(y)
    rigid = np.concatenate((-cy, cx, np.ones(n)))
    rigid /= np.linalg.norm(rigid)
    projected -= rigid * float(np.dot(projected, rigid))
    return projected


def build_walk_seeds(
    x: np.ndarray,
    y: np.ndarray,
    theta: np.ndarray,
    modes: np.ndarray,
    amplitudes: list[float],
    min_variation: float = 1e-4,
) -> tuple[list[tuple[np.ndarray, np.ndarray, np.ndarray]], list[str]]:
    n = len(x)
    xs: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    labels: list[str] = []
    for m, mode in enumerate(modes):
        oriented = remove_rigid_motion(mode, x, y)
        norm = float(np.linalg.norm(oriented))
        if norm < 1e-10:
            continue
        oriented /= norm
        if _translation_fraction(oriented, n) < min_variation:
            continue
        # Orient each mode deterministically.
        pivot = int(np.argmax(np.abs(oriented)))
        oriented = oriented if oriented[pivot] >= 0.0 else -oriented
        for amp in amplitudes:
            for sign, sign_label in ((1.0, "p"), (-1.0, "m")):
                dx = sign * amp * oriented[:n]
                dy = sign * amp * oriented[n : 2 * n]
                dtheta = sign * amp * oriented[2 * n :]
                walk_theta = theta + dtheta
                walk_x, walk_y, _ = normalize(x + dx, y + dy, walk_theta)
                xs.append((walk_x, walk_y, walk_theta))
                labels.append(f"mode{m}_{sign_label}_{amp:.4g}")
    return xs, labels


def load_target(n: int, incumbent: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    frame = pd.read_csv(incumbent, usecols=["n", "x", "y", "angle"])
    group = frame[frame["n"] == n]
    if len(group) != n:
        raise ValueError(f"expected {n} rows for n={n}, found {len(group)}")
    x = group["x"].to_numpy(dtype=np.float64)
    y = group["y"].to_numpy(dtype=np.float64)
    theta = group["angle"].to_numpy(dtype=np.float64)
    x, y, side = normalize(x, y, theta)
    return x, y, theta, side


def repair_and_polish(
    n: int,
    seeds: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    labels: list[str],
    incumbent_side: float,
    threads: int,
    top_k: int,
    slp_rounds: int,
) -> list[dict]:
    if not seeds:
        return []
    schedules = []
    for seed_idx in range(len(seeds)):
        # Alternate a shallow topology-preserving path and a deeper re-jam path.
        if seed_idx % 2 == 0:
            schedules.append(
                [incumbent_side + 0.012, incumbent_side + 0.003, incumbent_side - 2e-4]
            )
        else:
            schedules.append(
                [incumbent_side + 0.025, incumbent_side + 0.006, incumbent_side - 5e-4]
            )

    relaxed = su.cpp_batch_melt(
        seeds,
        schedules,
        sym_mode=1,
        a_start=0.965,
        n_melt_stages=3,
        seam_masks=None,
        seam_iters=0,
        full_iters=260,
        n_ruin_iters=0,
        n_threads=min(4, max(1, threads)),
        base_seed=90210 + n,
    )
    order = sorted(
        range(len(relaxed)),
        key=lambda i: (
            0 if relaxed[i]["best_L_feas"] is not None else 1,
            relaxed[i]["best_L_feas"]
            if relaxed[i]["best_L_feas"] is not None
            else relaxed[i]["pen"],
        ),
    )[:top_k]

    reports: list[dict] = []
    for i in order:
        row = relaxed[i]
        x, y, _ = normalize(row["x"], row["y"], row["th"])
        theta = np.asarray(row["th"], dtype=np.float64).copy()
        sx, sy, st, slp, ok = su.joint_xytheta_slp(
            x,
            y,
            theta,
            rounds=slp_rounds,
            tr_th0=0.025,
            tr_xy0=0.10,
            margin=1e-8,
        )
        if not ok:
            reports.append(
                {
                    "label": labels[i],
                    "melt_L": row["best_L_feas"],
                    "melt_pen": row["pen"],
                    "slp_ok": False,
                }
            )
            continue

        candidates = []
        for margin in (1e-7, 2e-8, 5e-9):
            fx, fy, fl, fixed_ok = su.fixed_angle_lp_tighten(
                sx, sy, st, rounds=5, margin=margin
            )
            if not fixed_ok:
                continue
            verdict = su.verify_official(n, fx, fy, st)
            if verdict.get("ok"):
                candidates.append(verdict)
        if not candidates:
            verdict = su.verify_official(n, sx, sy, st)
            if verdict.get("ok"):
                candidates.append(verdict)
        if not candidates:
            reports.append(
                {
                    "label": labels[i],
                    "melt_L": row["best_L_feas"],
                    "melt_pen": row["pen"],
                    "slp_L": slp,
                    "slp_ok": True,
                    "certified": False,
                }
            )
            continue
        best = min(candidates, key=lambda v: Decimal(v["L_lit"]))
        L = float(max(Decimal(best["L_lit"]), Decimal(best["L_pandas"])))
        reports.append(
            {
                "label": labels[i],
                "melt_L": row["best_L_feas"],
                "melt_pen": row["pen"],
                "slp_L": slp,
                "certified_L": L,
                "dL": L - incumbent_side,
                "L_lit": best["L_lit"],
                "L_pandas": best["L_pandas"],
                "rows": best["rows"],
            }
        )
    return reports


def write_final(wins: dict[int, dict], final_path: Path) -> None:
    union = _load_union_module()
    groups = {
        n: [
            {
                "square_id": row["square_id"],
                "x": row["x"],
                "y": row["y"],
                "angle": row["angle"],
            }
            for row in win["rows"]
        ]
        for n, win in sorted(wins.items())
    }
    text = union.final_text(groups)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    final_path.write_text(text, encoding="utf-8")
    literal, transported = union.load(final_path)
    for n, rows in groups.items():
        result = union.check((n, literal[n], transported[n]))
        if result is None:
            raise ValueError(f"exported final failed dual certification for n={n}")


def run_target(
    n: int,
    incumbent: Path,
    modes_per_target: int,
    threads: int,
    top_k: int,
    slp_rounds: int,
) -> dict:
    started = time.time()
    x, y, theta, side = load_target(n, incumbent)
    jac, graph = contact_jacobian(x, y, theta)
    singular, modes = soft_modes(jac, k=modes_per_target)
    amplitudes = [0.02, 0.06, 0.14, 0.30, 0.60]
    seeds, labels = build_walk_seeds(x, y, theta, modes, amplitudes)
    reports = repair_and_polish(
        n, seeds, labels, side, threads, top_k, slp_rounds
    )
    best = min(
        (row for row in reports if row.get("certified_L") is not None),
        key=lambda row: row["certified_L"],
        default=None,
    )
    return {
        "n": n,
        "incumbent_L": side,
        "graph": graph,
        "singular_values": [float(value) for value in singular],
        "seed_count": len(seeds),
        "best": best,
        "results": reports,
        "seconds": round(time.time() - started, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=int, nargs="+", default=[971, 911, 367])
    parser.add_argument(
        "--incumbent",
        type=Path,
        default=ROOT / "poc/pell_corrugation/data/border-scan/merged.csv",
    )
    parser.add_argument("--modes", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--slp-rounds", type=int, default=4)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parents[1] / "docs")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    wins: dict[int, dict] = {}
    all_results: dict[str, dict] = {}
    for n in args.targets:
        result = run_target(
            n,
            args.incumbent,
            args.modes,
            args.threads,
            args.top_k,
            args.slp_rounds,
        )
        all_results[str(n)] = result
        (args.out_dir / f"run_n{n}.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        best = result["best"]
        if best is not None and best["dL"] <= -1e-9:
            wins[n] = best
            print(
                f"WIN n={n} {best['label']} "
                f"L={best['certified_L']:.15f} dL={best['dL']:+.6e}",
                flush=True,
            )
        else:
            print(
                f"MISS n={n} seeds={result['seed_count']} "
                f"best_dL={None if best is None else best['dL']}",
                flush=True,
            )
        # Keep an incremental checkpoint even if a long scale-up is interrupted.
        write_final(wins, Path(__file__).resolve().parents[1] / "final.csv")
    (args.out_dir / "summary.json").write_text(
        json.dumps(all_results, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
