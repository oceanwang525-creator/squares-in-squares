"""Audit hypothesis C3 for every incumbent group: a rigid rotation of the whole
packing changes (width, height) and may lower the official side
``L = max(width, height)`` below its axis-aligned value.

Method
------
For one group let ``V`` be the convex hull of the 4n square corners.  For a
rotation angle ``t``

    width(t)  = max_i (x_i cos t + y_i sin t)  - min_i (...)
    height(t) = max_i (-x_i sin t + y_i cos t) - min_i (...)

so on a ``t``-interval where the four active (argmax/argmin) hull vertices are
fixed both width and height are of the form ``A cos t + B sin t``: width stays
positive and therefore concave on that interval.  The interval boundaries are
exactly the hull edge normal directions (mod pi/2 covers the height events via
height(t) = width(t + pi/2)).  On such an interval max(width, height) is a
single concave branch between two consecutive zeros of width - height, so its
minimum sits in {interval endpoints} union {width == height crossings}, which
is a finite candidate set.  The audit evaluates all of them.

Every reported value is cross-checked against a dense 1441-point sweep of
[0, pi/2]; the analytic minimum must be <= the dense minimum, otherwise the
row is flagged as inconsistent.

Float64 screening only: the official 60-digit metric is applied afterwards, and
only to groups that this audit flags as candidates.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import mpmath as mp
from scipy.spatial import ConvexHull

HALF_PI = math.pi / 2
DENSE = 1441


def read_groups(path: Path) -> dict[int, list[dict]]:
    groups: dict[int, list[dict]] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            groups.setdefault(int(row["n"]), []).append(row)
    for n, rows in groups.items():
        rows.sort(key=lambda r: int(r["square_id"]))
        if len(rows) != n:
            raise ValueError(f"group n={n} has {len(rows)} rows")
    return groups


def corners(rows: list[dict]) -> np.ndarray:
    p = np.array([[float(r["x"]), float(r["y"]), float(r["angle"])] for r in rows])
    u = np.column_stack([np.cos(p[:, 2]), np.sin(p[:, 2])])
    v = np.column_stack([-np.sin(p[:, 2]), np.cos(p[:, 2])])
    c = p[:, :2]
    return np.concatenate([c + 0.5 * s * u + 0.5 * t * v
                           for s in (-1, 1) for t in (-1, 1)])


def wh(points: np.ndarray, thetas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """width/height of the point set at each angle (vectorised)."""
    c, s = np.cos(thetas)[:, None], np.sin(thetas)[:, None]
    x = points[:, 0] * c + points[:, 1] * s
    y = -points[:, 0] * s + points[:, 1] * c
    return x.max(1) - x.min(1), y.max(1) - y.min(1)


def hull(points: np.ndarray) -> np.ndarray:
    if len(points) < 4:
        return points
    try:
        return points[ConvexHull(points).vertices]
    except Exception:  # noqa: BLE001 - degenerate/collinear input
        return np.unique(points, axis=0)


def candidates(H: np.ndarray) -> np.ndarray:
    """Angles in [0, pi/2] that can host the minimum of max(width, height)."""
    edges = np.roll(H, -1, axis=0) - H
    # outward normal of a CCW hull edge (dx, dy) is (dy, -dx)
    normals = np.mod(np.arctan2(-edges[:, 0], edges[:, 1]), HALF_PI)
    bounds = np.unique(np.r_[0.0, np.round(normals, 14), HALF_PI])
    out = [0.0, HALF_PI]
    for a, b in zip(bounds[:-1], bounds[1:]):
        mid = 0.5 * (a + b)
        x = H[:, 0] * math.cos(mid) + H[:, 1] * math.sin(mid)
        y = -H[:, 0] * math.sin(mid) + H[:, 1] * math.cos(mid)
        ax = H[int(x.argmax()), 0] - H[int(x.argmin()), 0]
        bx = H[int(x.argmax()), 1] - H[int(x.argmin()), 1]
        hy = H[int(y.argmax()), 1] - H[int(y.argmin()), 1]
        hx = H[int(y.argmax()), 0] - H[int(y.argmin()), 0]
        # width - height = (ax - hy) cos + (bx + hx) sin
        A, B = ax - hy, bx + hx
        if A == 0.0 and B == 0.0:
            continue
        root = math.atan2(-A, B) % math.pi
        for r in (root, root - math.pi, root + math.pi):
            if a < r < b:
                out.append(r)
    t = np.array(sorted(set(np.round(out, 15))))
    return t[(t >= -1e-15) & (t <= HALF_PI + 1e-15)]


def hp_sides(rows: list[dict], thetas: list[float]) -> list[float]:
    """max(width, height) at each theta in 50-digit arithmetic.

    The float64 candidate sweep is only a screen: near a kink, ``x cos - y sin``
    with a tiny ``cos`` loses ~1e-11 relative accuracy on coordinates ~30, i.e.
    ~1e-10 absolute, which is the same size as the reductions that survive it.
    Every promising row is therefore re-evaluated exactly here.
    """
    with mp.workdps(50):
        cs = []
        for row in rows:
            x, y, a = (mp.mpf(row[k]) for k in ("x", "y", "angle"))
            ux, uy = mp.cos(a) / 2, mp.sin(a) / 2
            for s in (-1, 1):
                for t in (-1, 1):
                    cs.append((x + s * ux - t * uy, y + s * uy + t * ux))
        out = []
        for theta in thetas:
            c, s = mp.cos(mp.mpf(theta)), mp.sin(mp.mpf(theta))
            px = [x * c - y * s for x, y in cs]
            py = [x * s + y * c for x, y in cs]
            out.append(float(max(max(px) - min(px), max(py) - min(py))))
        return out


def audit_group(points: np.ndarray) -> dict:
    H = hull(points)
    t = candidates(H)
    w, h = wh(H, t)
    side = np.maximum(w, h)
    i = int(side.argmin())
    # dense cross-check
    grid = np.linspace(0.0, HALF_PI, DENSE)
    gw, gh = wh(H, grid)
    gside = np.maximum(gw, gh)
    j = int(gside.argmin())
    axis_w, axis_h = wh(H, np.array([0.0]))
    return dict(hull_vertices=int(len(H)), n_points=int(len(points)),
                side_axis=float(max(axis_w[0], axis_h[0])),
                width_axis=float(axis_w[0]), height_axis=float(axis_h[0]),
                side_min=float(side[i]), theta=float(t[i]),
                theta_dense=float(grid[j]), side_dense=float(gside[j]),
                candidate_thetas=[float(x) for x in t],
                candidate_sides=float_sides(side),
                wide=float(max(axis_w[0] / max(axis_h[0], 1e-300),
                               axis_h[0] / max(axis_w[0], 1e-300))))


def float_sides(side: np.ndarray) -> list[float]:
    return [float(x) for x in side]


def reduction_of(rec: dict) -> float:
    """Best known reduction for one group: 50-digit value when re-evaluated."""
    return rec["hp"]["reduction"] if "hp" in rec else rec["reduction"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("incumbent", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--official", type=Path, default=None,
                    help="per-n official L JSON from official_L.py (fresh main); "
                         "omit to screen against the float64 axis-aligned side")
    ap.add_argument("--threshold", type=float, default=1e-9,
                    help="report rows whose reduction exceeds this")
    args = ap.parse_args()

    if args.official is None:
        official = dict(total=None, L={})
    else:
        official = json.loads(args.official.read_text(encoding="utf-8"))
    ref = {int(k): float(v) for k, v in official["L"].items()}
    groups = read_groups(args.incumbent)
    rows, inconsistent = {}, []
    for n in sorted(groups):
        rec = audit_group(corners(groups[n]))
        rec["L_official"] = ref.get(n, rec["side_axis"])
        rec["drift_axis"] = (None if n not in ref else rec["side_axis"] - ref[n])
        rec["reduction"] = rec["side_axis"] - rec["side_min"]
        rec["score_gain"] = rec["reduction"] / math.sqrt(n)
        if rec["side_min"] > rec["side_dense"] + 1e-12:
            inconsistent.append(n)
        # 50-digit re-evaluation of every row that survives the float screen.
        if rec["reduction"] > 1e-13:
            order = np.argsort(rec["candidate_sides"])[:6]
            thetas = [0.0] + [rec["candidate_thetas"][int(i)] for i in order]
            sides = hp_sides(groups[n], thetas)
            k = int(np.argmin(sides[1:])) + 1
            rec["hp"] = dict(side_axis=sides[0], side_min=sides[k],
                             theta=thetas[k],
                             reduction=sides[0] - sides[k],
                             checked=len(thetas) - 1)
        rows[str(n)] = rec

    cand = {n: r for n, r in rows.items() if reduction_of(r) > args.threshold}
    positive = [reduction_of(r) for r in rows.values()]
    summary = dict(source=str(args.incumbent), official=str(args.official),
                   official_total=official["total"], groups=len(rows),
                   inconsistent_rows=inconsistent,
                   rows_with_reduction_gt_1e12=sum(x > 1e-12 for x in positive),
                   rows_with_reduction_gt_1e9=sum(x > 1e-9 for x in positive),
                   rows_with_reduction_gt_1e6=sum(x > 1e-6 for x in positive),
                   rows_with_reduction_gt_1e4=sum(x > 1e-4 for x in positive),
                   max_reduction=max(positive),
                   max_reduction_n=int(max(rows, key=lambda k: reduction_of(rows[k]))),
                   total_score_gain_if_all_taken=float(sum(
                       reduction_of(r) / math.sqrt(int(n)) for n, r in rows.items()
                       if reduction_of(r) > 0)),
                   hp_checked=sum(1 for r in rows.values() if "hp" in r),
                   candidates={n: dict(
                       L_official=r["L_official"],
                       side_min=r.get("hp", r)["side_min"],
                       reduction=reduction_of(r),
                       score_gain=reduction_of(r) / math.sqrt(int(n)),
                       theta_deg=math.degrees(r.get("hp", r)["theta"]),
                       theta=r.get("hp", r)["theta"], side_dense=r["side_dense"],
                       wide=r["wide"], hull_vertices=r["hull_vertices"])
                       for n, r in sorted(cand.items(), key=lambda kv: -reduction_of(kv[1]))})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(summary=summary, rows=rows), indent=1) + "\n",
                           encoding="utf-8")
    top = sorted(rows.items(), key=lambda kv: -reduction_of(kv[1]))[:15]
    brief = {k: v for k, v in summary.items() if k != "candidates"}
    brief["top"] = [[n, round(reduction_of(r), 12),
                     round(math.degrees(r.get("hp", r)["theta"]), 6)] for n, r in top]
    print(json.dumps(brief, indent=1), flush=True)
    print({k: len(v) for k, v in [("candidates", summary["candidates"])]}, flush=True)
    if len(cand) > 12:
        print("candidate ns:", ",".join(cand), flush=True)


if __name__ == "__main__":
    main()
