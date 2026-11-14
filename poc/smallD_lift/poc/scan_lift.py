"""Exhaustive search over corner lifts of the tracked certified rows.

The lift
--------
Take any packing (a "donor") of `p` unit squares whose axis-aligned bounding box
is `W x H`, put it in the bottom-left corner of a box of side `L`, and fill the
rest with upright unit squares on the lattice anchored at that corner:

    right arm :  j columns at x = W, W+1, ..., W+j-1, each holding K squares
                 (y = 0..K-1)
    top arm   :  i rows    at y = H, H+1, ..., H+i-1, each holding C = floor(W)
                 squares (x = 0..C-1)

Nothing overlaps: the donor lives in `[0,W] x [0,H]`, the right arm in
`[W, W+j] x [0, K]`, the top arm in `[0, C] x [H, H+i]`, and `C <= W`.  Box side

    L = max(W + j, H + i, K).

It holds `j*K + i*C + p` squares, and deleting squares is always legal, so a
lift of the target `n` exists as soon as `j*K + i*C + p >= n` (with `K` = the
integer part of the side, i.e. `K = floor(L)`; we only use `L < K+1`).

For a target `n = K^2 + D` this is *exactly* the "corner gadget" picture: with
`j = i = K - m`, `C = m` and `p = m^2 + D` the box shrinks to
`L = K + max(W - m, H - m)`, so a donation whose box excess `delta = max(W-m,
H-m)` beats the tracked row's own excess `L(n) - K` is a certified win.

This script enumerates every (target n, donor p, j, i) with `L < L_tracked(n)`
and can build + certify the winners with the official metric in both the
literal-decimal and pandas transports.

    python scan_lift.py                  # enumerate candidates
    python scan_lift.py --certify 30     # build + certify the best 30
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "poc" / "baseline" / "poc" / "src"))

import mpmath as mp  # noqa: E402
import numpy as np  # noqa: E402

INCUMBENT = ROOT / "poc" / "pell_corrugation" / "data" / "border-scan" / "merged.csv"
DATA = HERE.parent / "data"
# Clearances.  Exact tangency is *rejected* by the interval certifier: the
# decimal re-emission of a coordinate moves it by ~1e-60, so two squares that
# touch exactly in the source file can overlap by ~1e-60 in the lift.  A 1e-12
# clearance costs 1e-12 of side and removes the problem.  Values are strings and
# are converted to mpf inside the 70-digit contexts below.
PAD = "1e-12"


# --------------------------------------------------------------------------- #
# exact geometry
# --------------------------------------------------------------------------- #
def load_rows(path=INCUMBENT):
    """{n: [(x_str, y_str, angle_str), ...]} with the original decimal strings."""
    groups = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            groups.setdefault(int(r["n"]), []).append(
                (int(r["square_id"]), r["x"], r["y"], r["angle"]))
    for rows in groups.values():
        rows.sort(key=lambda t: t[0])
    return {n: [t[1:] for t in rows] for n, rows in groups.items()}


def bbox_mp(rows, dps=60):
    """Exact decimal bounding box (W, H, x0, y0) of a row list."""
    with mp.workdps(dps):
        xs, ys = [], []
        for x, y, a in rows:
            cx, cy, ang = mp.mpf(x), mp.mpf(y), mp.mpf(a)
            c, s = mp.cos(ang), mp.sin(ang)
            h = mp.mpf(1) / 2
            for dx, dy in ((h, h), (h, -h), (-h, h), (-h, -h)):
                xs.append(cx + c * dx - s * dy)
                ys.append(cy + s * dx + c * dy)
        return max(xs) - min(xs), max(ys) - min(ys), min(xs), min(ys)


def float_bboxes(rows_by_n):
    out = {}
    for n, rows in rows_by_n.items():
        q = np.array([(float(x), float(y), float(t)) for x, y, t in rows])
        h = 0.5 * (np.abs(np.cos(q[:, 2])) + np.abs(np.sin(q[:, 2])))
        out[n] = (float((q[:, 0] + h).max() - (q[:, 0] - h).min()),
                  float((q[:, 1] + h).max() - (q[:, 1] - h).min()))
    return out


def tracked_sides(rows_by_n, bboxes):
    return {n: max(wh[0], wh[1]) for n, wh in bboxes.items()}


# --------------------------------------------------------------------------- #
# lift construction
# --------------------------------------------------------------------------- #
def build_lift(donor_rows, K, j, i, n_target):
    """Rows of a packing of exactly `n_target` squares in side max(W+j,H+i,K).

    The donor keeps its *original decimal strings* (untranslated) so its exact
    internal contact geometry is preserved bit for bit; the arms are placed
    relative to the exactly measured bounding box (x0,y0)-(x1,y1) instead.
    """
    W, H, x0, y0 = bbox_mp(donor_rows)
    C = int(mp.floor(W))
    with mp.workdps(70):
        pad = mp.mpf(PAD)
        x1, y1 = x0 + W, y0 + H
        out = [(mp.mpf(x), mp.mpf(y), mp.mpf(t)) for x, y, t in donor_rows]
        arm = []
        for c in range(j):
            for r in range(K):
                arm.append((x1 + pad + c * (1 + pad) + mp.mpf(1) / 2,
                            y0 + r * (1 + pad) + mp.mpf(1) / 2, mp.mpf(0)))
        for c in range(C):
            for r in range(i):
                arm.append((x0 + c * (1 + pad) + mp.mpf(1) / 2,
                            y1 + pad + r * (1 + pad) + mp.mpf(1) / 2, mp.mpf(0)))
        keep = n_target - len(out)
        if keep < 0 or keep > len(arm):
            raise ValueError(f"count mismatch: donor {len(out)} + arm {len(arm)}"
                             f" cannot make {n_target}")
        out += arm[:keep]
        rows = [dict(square_id=str(k), x=mp.nstr(v[0], 70),
                     y=mp.nstr(v[1], 70), angle=mp.nstr(v[2], 70))
                for k, v in enumerate(out)]
    assert len(rows) == n_target
    return rows


def certify(n, rows):
    from squares.core import certify as _cert, kaggle_roundtrip
    return _cert(n, rows), _cert(n, kaggle_roundtrip({n: rows})[n])


# --------------------------------------------------------------------------- #
# enumeration
# --------------------------------------------------------------------------- #
def scan(rows_by_n, bboxes, sides, nmax=1000):
    """All (n, donor, j, i) lifts with L < L_tracked(n) - 1e-9."""
    counts = np.array(sorted(bboxes), dtype=float)          # donor sizes p
    P = counts.astype(int)
    W = np.array([bboxes[p][0] for p in P])
    H = np.array([bboxes[p][1] for p in P])
    C = np.floor(W)
    hits = []
    for n in sorted(sides):
        L0 = sides[n]
        K = math.isqrt(n)
        if K < 2:
            continue
        D = n - K * K
        need = n - P                                        # squares still to place
        # j right-arm columns: j <= L0 - W  and  j*K <= need + 40 (loose bound)
        jmax = int(min(L0 - 0, K) + 1)
        for j in range(0, jmax + 1):
            rest = need - j * K
            with np.errstate(divide="ignore", invalid="ignore"):
                i = np.ceil(rest / C)
            i = np.where(np.asarray(i) < 0, 0, i)
            i = np.nan_to_num(i, nan=0.0, posinf=1e9)
            L = np.maximum.reduce([W + j, H + i, np.full(len(P), float(K))])
            ok = L < L0 - 1e-9
            if not ok.any():
                continue
            for idx in np.nonzero(ok)[0]:
                hits.append(dict(n=n, K=K, D=D, donor=int(P[idx]), j=j,
                                 i=int(i[idx]), C=int(C[idx]),
                                 delta=float(L[idx]) - K,
                                 c_target=L0 - K, dL=float(L[idx]) - L0,
                                 gain=(float(L[idx]) - L0) / math.sqrt(n)))
    # keep the best (n, donor) pair
    best = {}
    for h in hits:
        key = (h["n"], h["donor"])
        if key not in best or h["dL"] < best[key]["dL"]:
            best[key] = h
    out = sorted(best.values(), key=lambda h: h["dL"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--certify", type=int, default=0)
    ap.add_argument("--out", default=str(DATA / "lift_candidates.json"))
    ap.add_argument("--gramm", type=int, default=0,
                    help="certify this many classic `m^2+D` corner gadgets too")
    a = ap.parse_args()

    rows_by_n = load_rows()
    bboxes = float_bboxes(rows_by_n)
    sides = tracked_sides(rows_by_n, bboxes)
    json.dump({str(n): wh for n, wh in bboxes.items()},
              open(DATA / "bbox_tracked.json", "w"), indent=0)
    hits = scan(rows_by_n, bboxes, sides)
    json.dump(hits, open(a.out, "w"), indent=1)
    print(f"{len(hits)} lift candidates with dL < -1e-9")
    print(f"{'target n':>8} {'K':>2} {'D':>2} {'donor':>5} {'j':>3} {'i':>3} "
          f"{'C':>3} {'delta':>10} {'c_target':>10} {'dL':>12} {'score gain':>11}")
    for h in hits[:40]:
        print(f"{h['n']:>8} {h['K']:>2} {h['D']:>2} {h['donor']:>5} {h['j']:>3} "
              f"{h['i']:>3} {h['C']:>3} {h['delta']:>10.6f} {h['c_target']:>10.6f} "
              f"{h['dL']:>12.3e} {h['gain']:>11.3e}")

    if a.certify:
        from squares.core import certify as _cert
        print("\n--- exact certification (literal + pandas) ---")
        wins, seen = [], set()
        for h in hits:
            if len(wins) >= a.certify:
                break
            key = (h["n"], h["donor"])
            if key in seen:
                continue
            seen.add(key)
            inc_rows = [dict(square_id=str(k), x=x, y=y, angle=t)
                        for k, (x, y, t) in enumerate(rows_by_n[h["n"]])]
            incL = mp.mpf(_cert(h["n"], inc_rows))
            try:
                rows = build_lift(rows_by_n[h["donor"]], h["K"], h["j"], h["i"],
                                  h["n"])
                lit, pan = certify(h["n"], rows)
            except Exception as e:
                print(f"n={h['n']:>4} donor={h['donor']:>4} j={h['j']} i={h['i']}:"
                      f" FAILED {type(e).__name__}: {e}")
                continue
            d_lit = mp.mpf(lit) - incL
            d_pan = mp.mpf(pan) - incL
            ok = (d_lit <= mp.mpf("-1e-9")) and (d_pan <= mp.mpf("-1e-9"))
            print(f"n={h['n']:>4} donor={h['donor']:>4} j={h['j']} i={h['i']}: "
                  f"L={mp.nstr(mp.mpf(lit), 22)} inc={mp.nstr(incL, 22)} "
                  f"dL_lit={mp.nstr(d_lit, 6)} dL_pandas={mp.nstr(d_pan, 6)} "
                  f"{'WIN' if ok else 'no'}")
            if ok:
                wins.append(dict(n=h["n"], donor=h["donor"], j=h["j"], i=h["i"],
                                 L_lit=lit, L_pandas=pan,
                                 dL_lit=mp.nstr(d_lit, 10),
                                 dL_pandas=mp.nstr(d_pan, 10),
                                 score_gain=mp.nstr(d_lit / mp.sqrt(h["n"]), 6)))
        json.dump(wins, open(DATA / "certified_wins.json", "w"), indent=1)
        import pickle
        with open(DATA / "wins_rows.pkl", "wb") as f:
            pickle.dump({w["n"]: build_lift(rows_by_n[w["donor"]], int(math.isqrt(w["n"])),
                                            w["j"], w["i"], w["n"]) for w in wins}, f)
        print(f"\n{len(wins)} strictly certified wins")
        tot = sum(mp.mpf(w["dL_lit"]) / mp.sqrt(w["n"]) for w in wins)
        print(f"total score gain {mp.nstr(tot, 8)}")


if __name__ == "__main__":
    main()
