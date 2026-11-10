"""Free-angle coordinate-descent re-polish of repaired gallery packings.

The repaired layouts in merged.csv came from a fixed-angle LP: gallery angles
stayed at two decimal degrees and only centers moved inside +/-0.03.  This
script keeps that trust region and topology but gives each tilted square a
small angle ladder.  After every trial angle it rebuilds the separating-axis
pair set and sign pattern and re-solves the same box-side LP.  A move is kept
only when the LP side strictly decreases.

Outputs go under poc/free_angle_repolish/data/.  Certification uses the
unchanged official literal and actual pandas 3.1.0 transport checks.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from decimal import Decimal
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import mpmath as mp
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "poc/baseline/poc/src"))
from squares.core import certify, read_submission  # noqa: E402

DATA = REPO / "poc/pell_corrugation/data/border-scan"
MERGED = DATA / "merged.csv"
OFFICIAL = DATA / "merged-literal-official.json"
PROVENANCE = DATA / "gallery-repair-provenance.json"
PAIR_RADIUS = 1.6
SAFETY = mp.mpf("1.0000000001")


def solve_lp(angles, centers, origin, trust):
    """Minimize box side over centers with the given fixed angles.

    Mirrors repair_gallery.py: one active separating axis per contacting pair
    plus the two wall extents per center coordinate.  Returns (L, centers) or
    None when the LP is infeasible.
    """
    n = len(angles)
    u = np.column_stack([np.cos(angles), np.sin(angles)])
    v = np.column_stack([-np.sin(angles), np.cos(angles)])
    pairs = list(cKDTree(centers).query_pairs(PAIR_RADIUS))
    rr: list[int] = []
    cc: list[int] = []
    vals: list[float] = []
    rhs: list[float] = []

    def add(columns, values, b):
        row = len(rhs)
        rr.extend([row] * len(columns))
        cc.extend(columns)
        vals.extend(values)
        rhs.append(b)

    for i, j in pairs:
        axes = np.array([u[i], v[i], u[j], v[j]])
        delta = centers[j] - centers[i]
        proj = axes @ delta
        radii = 0.5 * (
            np.abs(axes @ u[i])
            + np.abs(axes @ v[i])
            + np.abs(axes @ u[j])
            + np.abs(axes @ v[j])
        )
        k = int(np.argmax(np.abs(proj) - radii))
        a = axes[k] * (1 if proj[k] >= 0 else -1)
        add(
            [2 * i, 2 * i + 1, 2 * j, 2 * j + 1],
            [a[0], a[1], -a[0], -a[1]],
            -radii[k] - 1e-9,
        )

    ext = 0.5 * (np.abs(np.cos(angles)) + np.abs(np.sin(angles)))
    for i in range(n):
        for k in range(2):
            add([2 * i + k], [-1], -ext[i] - 1e-9)
            add([2 * i + k, 2 * n], [1, -1], -ext[i] - 1e-9)

    A = coo_matrix((vals, (rr, cc)), shape=(len(rhs), 2 * n + 1)).tocsr()
    c = np.zeros(2 * n + 1)
    c[-1] = 1
    bounds = [(float(z) - trust, float(z) + trust) for z in origin.ravel()]
    bounds.append((0, None))
    res = linprog(
        c,
        A_ub=A,
        b_ub=rhs,
        bounds=bounds,
        method="highs",
        options={
            "dual_feasibility_tolerance": 1e-9,
            "primal_feasibility_tolerance": 1e-9,
            "time_limit": 30,
        },
    )
    if not res.success:
        return None
    return float(res.fun), res.x[:-1].reshape(n, 2)


def repolish(angles, centers, origin, passes, ladder, trust, seed=0):
    """Coordinate descent over tilted-square angles, re-solving the LP each trial."""
    out = solve_lp(angles, centers, origin, trust)
    if out is None:
        raise RuntimeError("baseline LP infeasible")
    best_L, centers = out
    angles = angles.copy()
    tilted = np.flatnonzero(np.abs(angles) > 1e-12)
    rng = np.random.default_rng(seed)
    history = []

    for p in range(passes):
        improved = False
        order = rng.permutation(tilted)
        for i in order:
            trial = None
            for d in ladder:
                trial_angles = angles.copy()
                trial_angles[i] += d
                solved = solve_lp(trial_angles, centers, origin, trust)
                if solved is None:
                    continue
                L2, c2 = solved
                if L2 < best_L - 1e-9 and (trial is None or L2 < trial[0]):
                    trial = (L2, trial_angles, c2)
            if trial is not None:
                best_L, angles, centers = trial
                improved = True
        history.append({"pass": p + 1, "LP_L": best_L, "improved": improved})
        if not improved:
            break

    return angles, centers, best_L, history


def emit_rows(angles, centers):
    """Exact-decimal emission with the same center dilation as repair_gallery.py."""
    mp.mp.dps = 75
    rows = []
    for i in range(len(angles)):
        x = mp.mpf(str(Decimal.from_float(float(centers[i, 0])))) * SAFETY
        y = mp.mpf(str(Decimal.from_float(float(centers[i, 1])))) * SAFETY
        a = mp.mpf(str(Decimal.from_float(float(angles[i]))))
        rows.append(
            {
                "x": mp.nstr(x, 65),
                "y": mp.nstr(y, 65),
                "angle": mp.nstr(a, 65),
            }
        )
    return rows


def write_candidate(path, n, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "n", "square_id", "x", "y", "angle"])
        w.writeheader()
        for j, r in enumerate(rows):
            w.writerow(
                {"id": j, "n": n, "square_id": j, "x": r["x"], "y": r["y"], "angle": r["angle"]}
            )


def certify_file(path, n):
    return certify(n, read_submission(path)[n])


def write_summary(results, outdir):
    rows = []
    for n, p in sorted(results["puzzles"].items(), key=lambda kv: int(kv[0])):
        rows.append(
            {
                "n": int(n),
                "old_L": p["old_L"],
                "candidate_literal_L": p["candidate_literal_L"],
                "candidate_pandas_L": p["candidate_pandas_L"],
                "literal_strictly_better": p["literal_strictly_better"],
                "pandas_strictly_better": p["pandas_strictly_better"],
                "win": p["win"],
                "score_delta_literal": p["score_delta_literal"],
                "score_delta_pandas": p["score_delta_pandas"],
                "max_angle_shift": p["max_angle_shift"],
            }
        )
    pd.DataFrame(rows).to_csv(outdir / "measured.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="490,535,631,293,932")
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--ladder", default="0.002,0.005,0.01,0.02")
    ap.add_argument("--trust", type=float, default=0.03)
    ap.add_argument(
        "--output-dir", default=str(REPO / "poc/free_angle_repolish/data")
    )
    ap.add_argument(
        "--summarize-only",
        action="store_true",
        help="read results.json and rewrite measured.csv without re-solving",
    )
    args = ap.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    if args.summarize_only:
        results = json.load(open(outdir / "results.json"))
        write_summary(results, outdir)
        print(f"wrote {outdir / 'measured.csv'}", flush=True)
        return

    targets = [int(z) for z in args.targets.split(",")]
    ladder = [float(z) for z in args.ladder.split(",")]
    ladder = sorted(set(ladder + [-z for z in ladder]))

    df = pd.read_csv(MERGED)
    old = {r["n"]: r for r in json.load(open(OFFICIAL))["results"]}
    prov = json.load(open(PROVENANCE))
    ceiling = sum(
        float(v["box_growth"]) / np.sqrt(int(k)) for k, v in prov["wins"].items()
    )

    results = {
        "targets": targets,
        "passes": args.passes,
        "ladder": ladder,
        "trust": args.trust,
        "ceiling_box_growth_weighted": ceiling,
        "improved_wins": 0,
        "puzzles": {},
    }

    for n in targets:
        g = df[df.n == n].sort_values("square_id")
        if len(g) != n:
            raise RuntimeError(f"n={n}: expected {n} rows, found {len(g)}")
        origin = g[["x", "y"]].to_numpy(float)
        angles0 = g["angle"].to_numpy(float)
        t0 = time.time()
        angles, centers, lp_L, history = repolish(
            angles0, origin.copy(), origin, args.passes, ladder, args.trust
        )
        rows = emit_rows(angles, centers)
        csv_path = outdir / f"candidate-{n}.csv"
        write_candidate(csv_path, n, rows)
        pandas_path = outdir / f"candidate-{n}-pandas.csv"
        pd.read_csv(csv_path).to_csv(pandas_path, index=False)

        lit_L = certify_file(csv_path, n)
        pan_L = certify_file(pandas_path, n)
        old_L = old[n]["L"]
        lit_delta = (Decimal(old_L) - Decimal(lit_L)) / Decimal(n).sqrt()
        pan_delta = (Decimal(old_L) - Decimal(pan_L)) / Decimal(n).sqrt()
        win = Decimal(lit_L) < Decimal(old_L) and Decimal(pan_L) < Decimal(old_L)
        results["improved_wins"] += int(win)
        results["puzzles"][str(n)] = {
            "old_L": old_L,
            "candidate_literal_L": lit_L,
            "candidate_pandas_L": pan_L,
            "literal_strictly_better": Decimal(lit_L) < Decimal(old_L),
            "pandas_strictly_better": Decimal(pan_L) < Decimal(old_L),
            "win": win,
            "score_delta_literal": str(lit_delta),
            "score_delta_pandas": str(pan_delta),
            "LP_L": lp_L,
            "LP_L_baseline": float(
                solve_lp(angles0, origin, origin, args.trust)[0]
            ),
            "max_center_shift": float(np.max(np.abs(centers - origin))),
            "max_angle_shift": float(np.max(np.abs(angles - angles0))),
            "passes_run": len(history),
            "history": history,
            "seconds": round(time.time() - t0, 3),
        }
        print(
            f"n={n} win={win} old={old_L[:24]} lit={lit_L[:24]} "
            f"pan={pan_L[:24]} dscore={lit_delta:.3e} sec={time.time()-t0:.1f}",
            flush=True,
        )

    (outdir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    write_summary(results, outdir)
    print(
        f"wins={results['improved_wins']}/{len(targets)} "
        f"ceiling={ceiling:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
