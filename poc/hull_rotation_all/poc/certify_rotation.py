"""Emit the audited rotations and certify them with the official metric.

For each audited group the packing is rotated rigidly by ``theta`` (centres and
angles), the bounding box is translated to the origin, and the centres are
optionally dilated about the box centre by ``1 + delta`` so that the rounded
decimal literals still separate every contact by more than float64 transport
error.  Every variant is certified twice:

* literal  -- the decimal strings straight into ``squares.core.certify``;
* pandas   -- the same strings through ``squares.core.kaggle_roundtrip``
              (actual pandas ``read_csv``/``to_csv`` transport).

A trial counts only if it is strictly below the *fresh main* incumbent L in
both modes.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from decimal import Decimal
from pathlib import Path

import mpmath as mp

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "poc/baseline/poc/src"))
from squares import core  # noqa: E402

DPS = 60
OUT_DECIMALS = 40


def fmt(value) -> str:
    """Plain fixed-point decimal literal with 40 decimals (well inside the
    metric's 100-character literal cap and far above float64 precision)."""
    with mp.workdps(DPS + 40):
        return f"{mp.mpf(value):.{OUT_DECIMALS}f}"


def read_groups(path: Path) -> dict[int, list[dict]]:
    groups: dict[int, list[dict]] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            groups.setdefault(int(row["n"]), []).append(row)
    for n, rows in groups.items():
        rows.sort(key=lambda r: int(r["square_id"]))
    return groups


def rotate(rows: list[dict], theta: float, delta: float) -> list[dict]:
    """Rigid rotation, then a uniform dilation about the bbox centre."""
    with mp.workdps(DPS):
        c, s = mp.cos(mp.mpf(theta)), mp.sin(mp.mpf(theta))
        scale = mp.mpf(1) + mp.mpf(delta)
        xs, ys, angs = [], [], []
        for row in rows:
            x, y, a = (mp.mpf(row[k]) for k in ("x", "y", "angle"))
            angs.append(a + mp.mpf(theta))
            xs.append(x * c - y * s)
            ys.append(x * s + y * c)
        # half extents in the rotated frame
        ext = [0.5 * (abs(mp.cos(a)) + abs(mp.sin(a))) for a in angs]
        lo = [min(x - e for x, e in zip(xs, ext)), min(y - e for y, e in zip(ys, ext))]
        hi = [max(x + e for x, e in zip(xs, ext)), max(y + e for y, e in zip(ys, ext))]
        mid = [0.5 * (l + h) for l, h in zip(lo, hi)]
        ox = lo[0]
        oy = lo[1]
        out = []
        for i, (x, y) in enumerate(zip(xs, ys)):
            xx = mid[0] + (x - mid[0]) * scale - ox
            yy = mid[1] + (y - mid[1]) * scale - oy
            out.append(dict(square_id=str(i), x=fmt(xx), y=fmt(yy),
                            angle=fmt(angs[i])))
        length = max(hi[0] - lo[0], hi[1] - lo[1]) * scale
        return out, float(length)


def plain(rows: list[dict]) -> list[dict]:
    """Normalise every coordinate to a plain fixed-point decimal literal."""
    return [dict(row, **{k: fmt(row[k]) for k in ("x", "y", "angle")})
            for row in rows]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--incumbent", required=True, type=Path)
    ap.add_argument("--audit", required=True, type=Path)
    ap.add_argument("--literal", required=True, type=Path,
                    help="fresh main official L, literal mode")
    ap.add_argument("--pandas", required=True, type=Path,
                    help="fresh main official L, pandas mode")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--ns", default="", help="comma list; default = every row with "
                                            "reduction > --min-reduction")
    ap.add_argument("--min-reduction", type=float, default=1e-12)
    ap.add_argument("--deltas", default="0,1e-14,1e-13,1e-12,1e-11,1e-10")
    ap.add_argument("--extra-theta", default="",
                    help="comma list of extra absolute angles (radians) to try")
    ap.add_argument("--top-candidates", type=int, default=0,
                    help="also try the K best float candidate angles of each row")
    args = ap.parse_args()

    audit = json.loads(args.audit.read_text(encoding="utf-8"))["rows"]
    inc_lit = {int(k): Decimal(v) for k, v in
               json.loads(args.literal.read_text(encoding="utf-8"))["L"].items()}
    inc_pan = {int(k): Decimal(v) for k, v in
               json.loads(args.pandas.read_text(encoding="utf-8"))["L"].items()}
    groups = read_groups(args.incumbent)
    deltas = [float(x) for x in args.deltas.split(",") if x.strip()]
    extra = [float(x) for x in args.extra_theta.split(",") if x.strip()]

    if args.ns.strip():
        ns = [int(x) for x in args.ns.split(",") if x.strip()]
    else:
        ns = sorted((int(k) for k, r in audit.items()
                     if r["reduction"] > args.min_reduction),
                    key=lambda n: -audit[str(n)]["reduction"])

    results, best = [], {}
    for n in ns:
        thetas = [audit[str(n)]["theta"]] + extra
        if args.top_candidates:
            order = sorted(range(len(audit[str(n)]["candidate_sides"])),
                           key=lambda i: audit[str(n)]["candidate_sides"][i])
            for i in order[:args.top_candidates]:
                thetas.append(audit[str(n)]["candidate_thetas"][i])
        uniq = []
        for t in thetas:
            if all(abs(t - u) > 1e-15 for u in uniq):
                uniq.append(t)
        thetas = uniq
        for theta in thetas:
            entry = dict(n=n, theta=theta, theta_deg=math.degrees(theta),
                         audit_reduction=audit[str(n)]["reduction"],
                         incumbent_literal=str(inc_lit[n]),
                         incumbent_pandas=str(inc_pan[n]), trials=[])
            for delta in deltas:
                rows, predicted = rotate(groups[n], theta, delta)
                rows = plain(rows)
                trial = dict(delta=delta, predicted_L=predicted)
                try:
                    lit = Decimal(core.certify(n, rows))
                except Exception as error:  # noqa: BLE001
                    trial.update(status="literal-fail", error=str(error)[:200])
                    entry["trials"].append(trial)
                    continue
                trial["literal"] = str(lit)
                trial["dL_literal"] = float(lit - inc_lit[n])
                try:
                    transported = core.kaggle_roundtrip({n: rows})
                    pan = Decimal(core.certify(n, transported[n]))
                    trial["pandas"] = str(pan)
                    trial["dL_pandas"] = float(pan - inc_pan[n])
                    trial["status"] = ("WIN" if lit < inc_lit[n] and pan < inc_pan[n]
                                       else "not-strict")
                except Exception as error:  # noqa: BLE001
                    trial.update(status="pandas-fail", error=str(error)[:200])
                entry["trials"].append(trial)
                dpan = trial.get("dL_pandas")
                tail = "" if dpan is None else f" dL_pan={dpan:+.3e}"
                print(f"n={n} theta={math.degrees(theta):.9f} delta={delta:g} "
                      f"{trial['status']} dL_lit={trial['dL_literal']:+.3e}{tail}",
                      flush=True)
                if trial["status"] == "WIN" and (n not in best or
                                                 trial["dL_literal"] < best[n]["dL_literal"]):
                    best[n] = dict(theta=theta, delta=delta, rows=rows,
                                   dL_literal=trial["dL_literal"],
                                   dL_pandas=trial["dL_pandas"],
                                   literal=trial["literal"], pandas=trial["pandas"])
            results.append(entry)
    payload = dict(incumbent=str(args.incumbent), rows=results,
                   wins={str(n): dict(theta=v["theta"], theta_deg=math.degrees(v["theta"]),
                                      delta=v["delta"], literal=v["literal"],
                                      pandas=v["pandas"], dL_literal=v["dL_literal"],
                                      dL_pandas=v["dL_pandas"],
                                      rows=v["rows"])
                         for n, v in sorted(best.items())})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: [n, v["dL_literal"], v["dL_pandas"]]
                      for k, v in payload["wins"].items()}, indent=1), flush=True)


if __name__ == "__main__":
    main()
