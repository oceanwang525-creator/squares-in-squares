"""Structural explanation of the C3 audit: how far are the four bounding-box
corners from the nearest square corner?

If the hull touches the four corners of its own L x L box, then for any rotation
t the rotated bbox of those four points alone is L(|cos t| + |sin t|) on both
axes, hence ``max(width, height) >= L`` and no rigid rotation can reduce the
official side.  With a corner slack ``g`` (largest distance from a bbox corner
to the nearest square corner) the same argument gives the bound

    max(width(t), height(t)) >= L - sqrt(2) g     for every t,

so ``sqrt(2) g`` is a certified ceiling on any gain a rigid rotation can find.

Also re-derives width(t), height(t) by dense sampling, independently of
``audit.py``'s event/crossing candidate set, as a cross-check.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

HALF_PI = math.pi / 2


def read_groups(path: Path) -> dict[int, list[dict]]:
    groups: dict[int, list[dict]] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            groups.setdefault(int(row["n"]), []).append(row)
    for n, rows in groups.items():
        rows.sort(key=lambda r: int(r["square_id"]))
    return groups


def corners(rows: list[dict]) -> np.ndarray:
    p = np.array([[float(r["x"]), float(r["y"]), float(r["angle"])] for r in rows])
    u = np.column_stack([np.cos(p[:, 2]), np.sin(p[:, 2])])
    v = np.column_stack([-np.sin(p[:, 2]), np.cos(p[:, 2])])
    c = p[:, :2]
    return np.concatenate([c + 0.5 * s * u + 0.5 * t * v
                           for s in (-1, 1) for t in (-1, 1)])


def audit(points: np.ndarray, samples: int = 2881) -> dict:
    lo = points.min(0)
    hi = points.max(0)
    side = float(max(hi - lo))
    box = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [lo[0], hi[1]], [hi[0], hi[1]]])
    dist = np.sqrt(((box[:, None, :] - points[None, :, :]) ** 2).sum(-1))
    gap = float(dist.min(1).max())
    gaps = [float(x) for x in dist.min(1)]

    grid = np.linspace(0.0, HALF_PI, samples)
    c, s = np.cos(grid)[:, None], np.sin(grid)[:, None]
    x = points[:, 0] * c + points[:, 1] * s
    y = -points[:, 0] * s + points[:, 1] * c
    dense = np.maximum(x.max(1) - x.min(1), y.max(1) - y.min(1))
    i = int(dense.argmin())
    span = wall_spans(points, lo, hi)
    return dict(side=side, gap=gap, gaps=gaps, bound=math.sqrt(2) * gap,
                dense_min=float(dense[i]), dense_theta=float(grid[i]),
                dense_gain=side - float(dense[i]), **span)


def wall_spans(points: np.ndarray, lo: np.ndarray, hi: np.ndarray,
               tol: float = 1e-12) -> dict:
    """How much of each box wall the packing touches, and the kink certificate.

    Coordinates are measured from the lower-left box corner.  With right-wall
    y-span [r1, r2], left-wall [l1, l2], bottom-wall x-span [c1, c2] and
    top-wall [d1, d2], for a rotation t (>0) the chords give

        width (t) >= Lx cos t + (r2 - l1) sin t
        width (-t) >= Lx cos t + (l2 - r1) sin t
        height(t) >= Ly cos t + (c2 - d1) sin t
        height(-t) >= Ly cos t + (d2 - c1) sin t

    so if all four brackets are positive, width and height both increase with
    |t| at t = 0 and the axis-aligned side is a strict local minimum of
    max(width, height): a rotation cannot lower L.  (The exhaustive event /
    crossing enumeration in audit.py then handles the rest of the range.)
    """
    right = points[points[:, 0] > hi[0] - tol][:, 1] - lo[1]
    left = points[points[:, 0] < lo[0] + tol][:, 1] - lo[1]
    top = points[points[:, 1] > hi[1] - tol][:, 0] - lo[0]
    bottom = points[points[:, 1] < lo[1] + tol][:, 0] - lo[0]
    r = (float(right.min()), float(right.max())) if len(right) else (0.0, 0.0)
    l = (float(left.min()), float(left.max())) if len(left) else (0.0, 0.0)
    d = (float(top.min()), float(top.max())) if len(top) else (0.0, 0.0)
    b = (float(bottom.min()), float(bottom.max())) if len(bottom) else (0.0, 0.0)
    conds = [r[1] - l[0], l[1] - r[0], b[1] - d[0], d[1] - b[0]]
    return dict(Lx=float(hi[0] - lo[0]), Ly=float(hi[1] - lo[1]),
                right=r, left=l, top=d, bottom=b,
                kink=conds, kink_certified=all(x > 0 for x in conds),
                wall_points=[len(right), len(left), len(bottom), len(top)])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("incumbent", type=Path)
    ap.add_argument("audit", type=Path, help="audit-screen.json from audit.py")
    ap.add_argument("output", type=Path)
    args = ap.parse_args()

    audit_rows = json.loads(args.audit.read_text(encoding="utf-8"))["rows"]
    out, viol, dense_viol = {}, [], []
    for n, rows in sorted(read_groups(args.incumbent).items()):
        rec = audit(corners(rows))
        rec["reduction"] = audit_rows[str(n)]["reduction"]
        rec["theta"] = audit_rows[str(n)]["theta"]
        if rec["reduction"] > rec["bound"] + 1e-12:
            viol.append(n)
        if rec["reduction"] > rec["dense_gain"] + 1e-12 and rec["dense_gain"] >= 0:
            dense_viol.append(n)
        out[str(n)] = rec
    gaps = sorted((r["gap"] for r in out.values()), reverse=True)
    gains = sorted((r["reduction"] for r in out.values()), reverse=True)
    kink = [r["kink_certified"] for r in out.values()]
    square = [abs(r["Lx"] - r["Ly"]) < 1e-12 for r in out.values()]
    summary = dict(groups=len(out),
                   corner_gap_max=gaps[0],
                   corner_gap_median=gaps[len(gaps) // 2],
                   corner_gap_min=gaps[-1],
                   rows_gap_gt_1e9=sum(g > 1e-9 for g in gaps),
                   rows_gap_gt_1e6=sum(g > 1e-6 for g in gaps),
                   rows_gap_gt_1e3=sum(g > 1e-3 for g in gaps),
                   rows_square_axis_bbox=sum(square),
                   rows_nonsquare_axis_bbox=len(square) - sum(square),
                   rows_kink_certified=sum(kink),
                   rows_kink_uncertified=[n for n, r in out.items()
                                          if not r["kink_certified"]],
                   bound_violations=viol,
                   dense_violations=dense_viol,
                   reduction_max=gains[0],
                   bound_beats_reduction_all=all(r["reduction"] <= r["bound"] + 1e-12
                                                 for r in out.values()),
                   top=[(n, out[n]["reduction"], out[n]["gap"], out[n]["bound"])
                        for n in sorted(out, key=lambda k: -out[k]["reduction"])[:10]])
    args.output.write_text(json.dumps(dict(summary=summary, rows=out), indent=1) + "\n",
                           encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
