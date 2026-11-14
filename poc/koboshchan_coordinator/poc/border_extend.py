#!/usr/bin/env python3
"""Coordinator tool: propagate a certified packing by an axis-aligned L-border.

A packing of n unit squares with a square-tight bounding box of side L = K + c
is kept byte-identical. Outside its actual bbox we append a new column of
m = ceil(L) unit squares (x in [x1+GAP, x1+GAP+1]) and a new row of m-1 squares
(y in [y1+GAP, y1+GAP+1], corner excluded). Result: n' = n + 2m - 1 squares at
side L + 1 + GAP. All border coordinates are exact 40-decimal values so that
consecutive squares differ by exactly 1 and cannot be read as overlapping at
any escalation precision.
"""
from __future__ import annotations
import argparse, sys
from decimal import Decimal, getcontext
from pathlib import Path
import mpmath as mp

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "poc/baseline/poc/src"))
from squares.core import certify, kaggle_roundtrip, read_submission  # noqa: E402

getcontext().prec = 120
Q = Decimal(1).scaleb(-40)
GAP = Decimal("1e-9")
SP = Decimal(1) + Decimal("1e-12")  # border pitch; >1 keeps every pair strictly separated
HALF = Decimal("0.5")


def bbox(rows):
    xs0, xs1, ys0, ys1 = [], [], [], []
    with mp.workdps(90):
        for r in rows:
            th = mp.mpf(r["angle"]); c = abs(mp.cos(th)); s = abs(mp.sin(th))
            e = (c + s) / 2
            x = mp.mpf(r["x"]); y = mp.mpf(r["y"])
            xs0.append(x - e); xs1.append(x + e); ys0.append(y - e); ys1.append(y + e)
        return min(xs0), max(xs1), min(ys0), max(ys1)


def dec(v_mp):
    return Decimal(mp.nstr(v_mp, 60))


def fmt(d):
    return format(d.quantize(Q), "f")


def border_extend(rows):
    with mp.workdps(90):
        x0, x1, y0, y1 = bbox(rows)
        W = x1 - x0; H = y1 - y0; L = max(W, H)
        if abs(W - H) > mp.mpf("1e-12"):
            raise ValueError(f"not square-tight: W={mp.nstr(W,20)} H={mp.nstr(H,20)}")
        m = int(mp.ceil(L))
        colx = (dec(x1) + GAP + HALF).quantize(Q)
        rowy = (dec(y1) + GAP + HALF).quantize(Q)
        ybase = (dec(y0) + HALF).quantize(Q)
        xbase = (dec(x0) + HALF).quantize(Q)
    out = [dict(r) for r in rows]
    sid = max(int(r["square_id"]) for r in rows) + 1
    for j in range(m):  # right column, x in [x1+GAP, x1+GAP+1]
        out.append(dict(square_id=str(sid), x=fmt(colx),
                        y=fmt(ybase + j * SP), angle="0"))
        sid += 1
    for i in range(m - 1):  # top row, y in [y1+GAP, y1+GAP+1], corner excluded
        out.append(dict(square_id=str(sid), x=fmt(xbase + i * SP),
                        y=fmt(rowy), angle="0"))
        sid += 1
    out = [dict(r, square_id=str(i)) for i, r in enumerate(out)]
    return len(out), out, L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", default="poc/pell_corrugation/data/border-scan/merged.csv")
    ap.add_argument("--n", type=int, required=True)
    a = ap.parse_args()
    puzzles = read_submission(a.merged)
    rows = puzzles[a.n]
    inc_n = certify(a.n, rows)
    n2, rows2, L = border_extend(rows)
    s_lit = certify(n2, rows2)
    s_pan = certify(n2, kaggle_roundtrip({n2: rows2})[n2])
    inc2 = certify(n2, puzzles[n2]) if n2 in puzzles else None
    print(f"n={a.n} L={inc_n}")
    print(f"extended n'={n2} literal={s_lit} pandas={s_pan} incumbent={inc2}")
    if inc2 is not None:
        from decimal import Decimal as D
        print(f"dL(literal)={D(s_lit)-D(inc2)} dL(pandas)={D(s_pan)-D(inc2)}")


if __name__ == "__main__":
    main()
