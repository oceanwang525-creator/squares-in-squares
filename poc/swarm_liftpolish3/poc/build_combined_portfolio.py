#!/usr/bin/env python3
"""Build a unified canonical incumbent CSV (merged.csv + all certified poc/*/final.csv)."""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import sys
from decimal import Decimal
from pathlib import Path

import mpmath as mp
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "poc/baseline/poc/src"))
sys.path.insert(0, str(REPO / "poc/auto_union/poc"))
import union as auto_union  # noqa: E402
from squares.core import certify, kaggle_roundtrip  # noqa: E402


def _strip_s(v: str) -> str:
    return v[1:] if v.startswith("s") else v


def fast_float_side(rows: list[dict[str, str]], is_deg: bool) -> float:
    x = np.array([float(_strip_s(r["x"])) for r in rows], dtype=np.float64)
    y = np.array([float(_strip_s(r["y"])) for r in rows], dtype=np.float64)
    a_raw = np.array([float(_strip_s(r["deg" if is_deg else "angle"])) for r in rows], dtype=np.float64)
    a = np.radians(a_raw) if is_deg else a_raw
    e = 0.5 * (np.abs(np.cos(a)) + np.abs(np.sin(a)))
    return float(max((x + e).max() - (x - e).min(), (y + e).max() - (y - e).min()))


def parse_final_csv_fast(path: Path) -> dict[int, list[dict[str, str]]]:
    """Read id,x,y,deg rows grouped by n without slow mpmath conversion yet."""
    by: dict[int, list[dict[str, str]]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["id", "x", "y", "deg"]:
            return {}
        for r in reader:
            parts = r["id"].split("_")
            if len(parts) != 2:
                continue
            n, sid = int(parts[0]), parts[1]
            by.setdefault(n, []).append({"square_id": sid, "x": r["x"], "y": r["y"], "deg": r["deg"]})
    return {n: rows for n, rows in by.items() if len(rows) == n}


def convert_deg_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out = []
    with mp.workdps(95):
        for r in sorted(rows, key=lambda x: int(x["square_id"])):
            deg_s = _strip_s(r["deg"])
            ang = mp.nstr(mp.mpf(deg_s) * mp.pi / 180, 75)
            out.append({
                "square_id": str(r["square_id"]),
                "x": _strip_s(r["x"]),
                "y": _strip_s(r["y"]),
                "angle": ang,
            })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-csv", default=str(REPO / "poc/swarm_liftpolish3/data/combined_incumbent.csv"))
    ap.add_argument("--summary-json", default=str(REPO / "poc/swarm_liftpolish3/data/combined_summary.json"))
    ap.add_argument("--extra-final", nargs="*", default=[])
    args = ap.parse_args()

    merged_path = REPO / "poc/pell_corrugation/data/border-scan/merged.csv"
    base: dict[int, list[dict[str, str]]] = {}
    with merged_path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            base.setdefault(int(r["n"]), []).append({
                "square_id": r["square_id"],
                "x": r["x"],
                "y": r["y"],
                "angle": r["angle"],
            })

    base_L = {n: fast_float_side(rows, is_deg=False) for n, rows in base.items()}
    cands: dict[int, list[tuple[float, str, list[dict[str, str]]]]] = {
        n: [(base_L[n], "merged.csv", rows)] for n, rows in base.items()
    }

    final_paths = sorted((REPO / "poc").glob("*/final.csv")) + [Path(p) for p in args.extra_final]
    for p in final_paths:
        if not p.exists():
            continue
        parsed = parse_final_csv_fast(p)
        rel = str(p.relative_to(REPO)) if p.is_relative_to(REPO) else str(p)
        for n, rows in parsed.items():
            fL = fast_float_side(rows, is_deg=True)
            if fL < base_L[n] - 1e-12:
                cands[n].append((fL, rel, rows))

    best_rows: dict[int, list[dict[str, str]]] = {}
    improved: list[dict] = []
    for n in sorted(base):
        ranked = sorted(cands[n], key=lambda t: t[0])
        if len(ranked) == 1:
            best_rows[n] = base[n]
            continue
        # Certify top candidates and baseline
        base_lit = Decimal(certify(n, base[n]))
        best_dec = base_lit
        best_src = "merged.csv"
        best_r = base[n]
        for fL, src, raw in ranked:
            if src == "merged.csv":
                continue
            conv = convert_deg_rows(raw)
            try:
                text = auto_union.final_text({n: conv})
                lit_g = auto_union.parse_text(text)[n]
                pan_g = auto_union.parse_text(pd.read_csv(io.StringIO(text)).to_csv(index=False))[n]
                res = auto_union.check((n, lit_g, pan_g))
                if res is None:
                    continue
                L_worst = max(Decimal(res[0]), Decimal(res[1]))
                if L_worst < best_dec - Decimal("1e-12"):
                    best_dec = L_worst
                    best_src = src
                    best_r = lit_g
            except Exception:
                continue
        best_rows[n] = best_r
        if best_src != "merged.csv":
            dL = float(best_dec - base_lit)
            improved.append({
                "n": n,
                "merged_L": str(base_lit),
                "best_L": str(best_dec),
                "dL": dL,
                "source": best_src,
            })
            print(f"n={n:4d}: {str(base_lit)[:16]} -> {str(best_dec)[:16]} (dL={dL:+.6e}) [{best_src}]")

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(auto_union.csv_text(best_rows), encoding="utf-8")
    Path(args.summary_json).write_text(json.dumps(improved, indent=2), encoding="utf-8")
    print(f"Wrote {out_path} ({len(improved)} groups improved over merged.csv)")


if __name__ == "__main__":
    main()
