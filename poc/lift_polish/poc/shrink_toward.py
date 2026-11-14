#!/usr/bin/env python3
"""Shrink every group of a CSV uniformly toward its bbox centre.

Purpose: create a *restart* for the fixed-angle LP.  Scaling a tight packing by
``f < 1`` about its bbox centre keeps the combinatorial layout but turns every
contact into a small overlap, which forces the LP to re-derive its separating
axes from a slightly different configuration (a tie-break the frozen argmax
choice never sees).  With ``--target`` the factor is ``target_L / L(n)`` so the
packing is pulled toward a goal side (e.g. the `frontier` column of
`data/gap_vs_frontier.csv`); the extra ``--eps`` inset makes the pull strictly
infeasible so the LP has to work.

usage:
  shrink_toward.py BASE.csv --out OUT.csv --factor 0.999 [--eps 1e-9]
  shrink_toward.py BASE.csv --out OUT.csv --gap gap_vs_frontier.csv --target frontier
"""
from __future__ import annotations

import argparse
import csv
import math
from decimal import Decimal, localcontext
from pathlib import Path


def bbox(rows):
    xs0 = xs1 = ys0 = ys1 = None
    for r in rows:
        a = float(r["angle"])
        h = 0.5 * (abs(math.cos(a)) + abs(math.sin(a)))
        x, y = float(r["x"]), float(r["y"])
        xs0 = x - h if xs0 is None else min(xs0, x - h)
        xs1 = x + h if xs1 is None else max(xs1, x + h)
        ys0 = y - h if ys0 is None else min(ys0, y - h)
        ys1 = y + h if ys1 is None else max(ys1, y + h)
    return xs0, xs1, ys0, ys1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--out", required=True)
    ap.add_argument("--factor", type=float, default=None)
    ap.add_argument("--eps", type=float, default=1e-9)
    ap.add_argument("--gap", default=None, help="gap_vs_frontier.csv for per-n targets")
    ap.add_argument("--target", default="frontier", choices=["frontier", "leader", "ours"])
    args = ap.parse_args()

    targets: dict[int, float] = {}
    if args.gap:
        with open(args.gap, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                targets[int(row["n"])] = float(row[args.target])

    groups: dict[int, list[dict[str, str]]] = {}
    with open(args.src, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            groups.setdefault(int(row["n"]), []).append(row)

    out_rows = []
    for n in sorted(groups):
        rows = sorted(groups[n], key=lambda r: int(r["square_id"]))
        xs0, xs1, ys0, ys1 = bbox(rows)
        L = max(xs1 - xs0, ys1 - ys0)
        if n in targets:
            f = targets[n] / L * (1.0 - args.eps)
        else:
            f = args.factor * (1.0 - args.eps)
        cx, cy = 0.5 * (xs0 + xs1), 0.5 * (ys0 + ys1)
        with localcontext() as ctx:
            ctx.prec = 80
            q = Decimal("1e-55")
            scale = Decimal(f)
            dcx, dcy = Decimal(repr(cx)), Decimal(repr(cy))
            for r in rows:
                x = dcx + scale * (Decimal(r["x"]) - dcx)
                y = dcy + scale * (Decimal(r["y"]) - dcy)
                out_rows.append(dict(id=r.get("id", "0"), n=n, square_id=r["square_id"],
                                     x=format(x.quantize(q), "f"),
                                     y=format(y.quantize(q), "f"),
                                     angle=r["angle"]))
        print(f"n={n}: L={L:.12f} factor={f:.9f}" +
              (f" target={targets[n]:.9f} residual={L - targets[n]:.9f}" if n in targets else ""))

    with Path(args.out).open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["id", "n", "square_id", "x", "y", "angle"])
        for r in out_rows:
            w.writerow([r["id"], r["n"], r["square_id"], r["x"], r["y"], r["angle"]])
    print(f"wrote {args.out} ({len(out_rows)} rows)")


if __name__ == "__main__":
    main()
