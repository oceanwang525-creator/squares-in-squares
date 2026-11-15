#!/usr/bin/env python3
"""Export Round 2 dual-certified strict wins (literal == pandas) against full origin/main portfolio."""
from __future__ import annotations

import csv
import io
import json
import sys
from decimal import Decimal
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "poc" / "auto_union" / "poc"))
sys.path.insert(0, str(REPO / "poc" / "baseline" / "poc" / "src"))

import union as auto_union  # noqa: E402
from squares.core import certify  # noqa: E402


def load_portfolio_incumbents() -> dict[int, tuple[Decimal, str]]:
    """Load min L across merged.csv and all other poc/*/final.csv files (excluding our own)."""
    best: dict[int, tuple[Decimal, str]] = {}

    merged = REPO / "poc" / "pell_corrugation" / "data" / "border-scan" / "merged.csv"
    by: dict[int, list[dict[str, str]]] = {}
    with merged.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            by.setdefault(int(r["n"]), []).append({
                "square_id": r["square_id"],
                "x": r["x"],
                "y": r["y"],
                "angle": r["angle"],
            })
    for n, rows in by.items():
        L = Decimal(certify(n, rows))
        best[n] = (L, "merged.csv")

    for fp in sorted((REPO / "poc").glob("*/final.csv")):
        if fp.parent.name == "swarm_liftpolish3":
            continue
        txt = fp.read_text(encoding="utf-8")
        try:
            parsed = auto_union.parse_text(txt)
        except Exception:
            continue
        for n, rows in parsed.items():
            try:
                L = Decimal(certify(n, rows))
            except Exception:
                continue
            if n not in best or L < best[n][0]:
                best[n] = (L, str(fp.relative_to(REPO)))
    return best


def dual_certify(n: int, rows: list[dict[str, str]]) -> tuple[Decimal, Decimal, list[dict[str, str]]] | None:
    try:
        text = auto_union.final_text({n: rows})
        lit = auto_union.parse_text(text)[n]
        pan = auto_union.parse_text(pd.read_csv(io.StringIO(text)).to_csv(index=False))[n]
        res = auto_union.check((n, lit, pan))
        if res is None:
            return None
        return Decimal(res[0]), Decimal(res[1]), lit
    except Exception:
        return None


def main() -> None:
    inc = load_portfolio_incumbents()
    wins_dir = REPO / "poc" / "swarm_liftpolish3" / "data" / "wins_r2"
    export_groups: dict[int, list[dict[str, str]]] = {}
    records = []

    for wp in sorted(wins_dir.glob("win_n*.json")):
        d = json.loads(wp.read_text(encoding="utf-8"))
        n = int(d["n"])
        rows = d["rows"]
        cert = dual_certify(n, rows)
        if cert is None:
            print(f"n={n:3d} CERT_FAIL")
            continue
        L_lit, L_pan, lit_rows = cert
        if L_lit != L_pan:
            print(f"n={n:3d} LIT_PAN_MISMATCH {L_lit} != {L_pan}")
            continue
        L_inc, src = inc[n]
        dL = L_lit - L_inc
        status = "WIN" if dL <= Decimal("-1e-9") else ("TINY" if dL < 0 else "SUPERSEDED")
        print(f"n={n:3d} {status:10s} L_new={str(L_lit)[:26]} vs {str(L_inc)[:26]} ({src}) dL={float(dL):+.6e}")
        records.append({
            "n": n,
            "status": status,
            "label": d.get("label", ""),
            "L_lit": str(L_lit),
            "L_pan": str(L_pan),
            "L_inc": str(L_inc),
            "inc_src": src,
            "dL": float(dL),
            "ds": float((L_lit - L_inc) / n),
            "ds_L2n": float((L_lit * L_lit - L_inc * L_inc) / n),
        })
        if dL <= Decimal("-1e-9"):
            export_groups[n] = lit_rows

    out_csv = REPO / "poc" / "swarm_liftpolish3" / "final.csv"
    out_csv.write_text(auto_union.final_text(export_groups), encoding="utf-8")
    out_json = REPO / "poc" / "swarm_liftpolish3" / "docs" / "r2_verification.json"
    out_json.write_text(json.dumps({"wins": [r for r in records if r["status"] == "WIN"], "all": records}, indent=2), encoding="utf-8")
    tot_ds = sum(r["ds"] for r in records if r["status"] == "WIN")
    tot_l2n = sum(r["ds_L2n"] for r in records if r["status"] == "WIN")
    print(f"\nExported {len(export_groups)} strict wins to {out_csv} (total ds={tot_ds:+.6e}, ds_L2n={tot_l2n:+.6e})")


if __name__ == "__main__":
    main()
