"""Extract, certify (literal + pandas transport) and compare a candidate packing.

Usage:
  python verify.py --n 901 --theta 45 --m 17 --w 17 --bx 6.5 --by 6.5 --t 28 --tp 28
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from decimal import Decimal

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "poc", "baseline", "poc", "src"))
sys.path.insert(0, HERE)

from squares.core import certify, kaggle_roundtrip, score  # noqa: E402
from construct2 import build2  # noqa: E402
from geom import side  # noqa: E402

INCUMBENT = os.path.join(REPO, "poc", "pell_corrugation", "data", "border-scan",
                         "merged.csv")


def incumbent_rows(n: int) -> list[dict]:
    rows = []
    with open(INCUMBENT, newline="") as f:
        for r in csv.DictReader(f):
            if int(r["n"]) == n:
                rows.append(dict(square_id=r["square_id"], x=r["x"], y=r["y"],
                                 angle=r["angle"]))
    rows.sort(key=lambda r: int(r["square_id"]))
    return rows


def incumbent_L(n: int) -> str:
    return certify(n, incumbent_rows(n))


def to_rows(P: np.ndarray, L: float, drop: int = 0) -> list[dict]:
    """Translate to the box origin and format as decimal strings."""
    x0 = P[:, 0].min() - 0.5 * (abs(math.cos(P[0, 2])) + abs(math.sin(P[0, 2])))
    # cheap: shift so the bbox min is >= 0 using the true per-square extents
    h = 0.5 * (np.abs(np.cos(P[:, 2])) + np.abs(np.sin(P[:, 2])))
    dx = (P[:, 0] - h).min()
    dy = (P[:, 1] - h).min()
    Q = P.copy()
    Q[:, 0] -= dx
    Q[:, 1] -= dy
    rows = [dict(square_id=str(i), x=repr(float(q[0])), y=repr(float(q[1])),
                 angle=repr(float(q[2]))) for i, q in enumerate(Q)]
    if drop:
        rows = rows[drop:]
        for i, r in enumerate(rows):
            r["square_id"] = str(i)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--L", type=float, required=True)
    ap.add_argument("--theta", type=float, required=True)
    ap.add_argument("--m", type=int, required=True)
    ap.add_argument("--w", type=int, required=True)
    ap.add_argument("--bx", type=float, required=True)
    ap.add_argument("--by", type=float, required=True)
    ap.add_argument("--t", type=int, default=None)
    ap.add_argument("--tp", type=int, default=None)
    ap.add_argument("--v1", action="store_true", help="use the single-slab family")
    a = ap.parse_args()

    if a.v1:
        from construct import build
        cnt, P = build(math.radians(a.theta), a.m, a.w, a.bx, a.by, a.L, caps=True)
    else:
        cnt, P = build2(math.radians(a.theta), a.m, a.w, a.bx, a.by, a.L,
                        a.t, a.tp, caps=True)
    h = 0.5 * (np.abs(np.cos(P[:, 2])) + np.abs(np.sin(P[:, 2])))
    w_box = (P[:, 0] + h).max() - (P[:, 0] - h).min()
    h_box = (P[:, 1] + h).max() - (P[:, 1] - h).min()
    print(f"n={a.n} L={a.L:.12f} count={cnt} bbox=({w_box:.12f},{h_box:.12f}) "
          f"max={max(w_box, h_box):.12f}")
    if cnt != a.n:
        print("count != n: nothing to certify")
        return
    rows = to_rows(P, a.L)
    inc = incumbent_L(a.n)
    try:
        Llit = certify(a.n, rows)
    except ValueError as e:
        print("literal certification FAILED:", e)
        return
    tr = kaggle_roundtrip({a.n: rows})[a.n]
    Lpan = certify(a.n, tr)
    print(f"  incumbent L = {inc}")
    print(f"  literal  L = {Llit}")
    print(f"  pandas   L = {Lpan}")
    dlit = Decimal(Llit) - Decimal(inc)
    dpan = Decimal(Lpan) - Decimal(inc)
    print(f"  dL literal  = {dlit}")
    print(f"  dL pandas   = {dpan}")
    print("  STRICT WIN" if dlit <= Decimal('-1e-9') and dpan <= Decimal('-1e-9')
          else "  no strict win")
    out = os.path.join(HERE, "..", "docs")
    os.makedirs(out, exist_ok=True)
    import json
    with open(os.path.join(out, f"candidate_{a.n}.json"), "w") as f:
        json.dump(dict(n=a.n, L_target=a.L, params=vars(a), count=cnt,
                       incumbent_L=inc, literal_L=Llit, pandas_L=Lpan,
                       dL_literal=str(dlit), dL_pandas=str(dpan),
                       rows=rows), f, indent=1)


if __name__ == "__main__":
    main()
