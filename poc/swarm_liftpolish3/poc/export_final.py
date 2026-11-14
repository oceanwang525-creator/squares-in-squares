#!/usr/bin/env python3
"""Collect all candidate wins in poc/swarm_liftpolish3/data/, verify against global incumbent, export final.csv."""
from __future__ import annotations

import argparse
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


def load_incumbent_sides(csv_path: Path) -> dict[int, Decimal]:
    by: dict[int, list[dict[str, str]]] = {}
    with csv_path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            by.setdefault(int(r["n"]), []).append({
                "square_id": r["square_id"],
                "x": r["x"],
                "y": r["y"],
                "angle": r["angle"],
            })
    return {n: Decimal(certify(n, rows)) for n, rows in by.items()}


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--incumbent-csv", default=str(REPO / "poc/swarm_liftpolish3/data/combined_incumbent.csv"))
    ap.add_argument("--out-csv", default=str(REPO / "poc/swarm_liftpolish3/final.csv"))
    ap.add_argument("--report-json", default=str(REPO / "poc/swarm_liftpolish3/docs/verification.json"))
    args = ap.parse_args()

    data_dir = REPO / "poc/swarm_liftpolish3/data"
    candidates: dict[int, list[tuple[str, list[dict[str, str]]]]] = {}

    for jf in sorted(data_dir.glob("*lpslack*.json")):
        try:
            items = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(items, list):
            for rec in items:
                if rec.get("status") in ("WIN", "TINY_WIN") and "rows" in rec:
                    candidates.setdefault(int(rec["n"]), []).append((jf.name, rec["rows"]))

    wins_dir = data_dir / "wins"
    if wins_dir.exists():
        for wp in sorted(wins_dir.glob("win_n*.json")):
            d = json.loads(wp.read_text(encoding="utf-8"))
            candidates.setdefault(int(d["n"]), []).append((f"wins/{wp.name}", d["rows"]))

    for subdir in ("built_partb", "built_partb_prop"):
        sd = data_dir / subdir
        if sd.exists():
            for np_path in sorted(sd.glob("n*.json")):
                n = int(np_path.stem[1:])
                rows = json.loads(np_path.read_text(encoding="utf-8"))
                candidates.setdefault(n, []).append((f"{subdir}/{np_path.name}", rows))

    if not candidates:
        Path(args.out_csv).write_text("id,x,y,deg\n", encoding="utf-8")
        Path(args.report_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report_json).write_text(json.dumps({"wins": []}, indent=2), encoding="utf-8")
        print("No candidates found; wrote header-only final.csv")
        return

    # Only compute incumbent side for candidate n values (much faster!)
    need_ns = set(candidates.keys())
    by: dict[int, list[dict[str, str]]] = {}
    with Path(args.incumbent_csv).open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            n = int(r["n"])
            if n in need_ns:
                by.setdefault(n, []).append({
                    "square_id": r["square_id"],
                    "x": r["x"],
                    "y": r["y"],
                    "angle": r["angle"],
                })

    exported: dict[int, list[dict[str, str]]] = {}
    win_records = []
    for n in sorted(candidates):
        inc_rows = sorted(by[n], key=lambda r: int(r["square_id"]))
        inc_res = dual_certify(n, inc_rows)
        if inc_res is None:
            inc_L = Decimal(certify(n, inc_rows))
        else:
            inc_L = max(inc_res[0], inc_res[1])

        best_L = inc_L
        best_lit = None
        best_pan = None
        best_rows = None
        best_src = None
        for src, rows in candidates[n]:
            res = dual_certify(n, rows)
            if res is None:
                continue
            L_lit, L_pan, clean_rows = res
            L_worst = max(L_lit, L_pan)
            if L_worst < best_L:
                best_L = L_worst
                best_lit = L_lit
                best_pan = L_pan
                best_rows = clean_rows
                best_src = src

        if best_rows is not None and (best_L - inc_L) <= Decimal("-1e-9"):
            dL = best_L - inc_L
            exported[n] = best_rows
            win_records.append({
                "n": n,
                "source": best_src,
                "incumbent_L": str(inc_L),
                "new_L_literal": str(best_lit),
                "new_L_pandas": str(best_pan),
                "dL": str(dL),
            })
            print(f"WIN n={n:4d} [{best_src}]: {str(inc_L)[:18]} -> {str(best_L)[:18]} (dL={float(dL):+.6e})")
        else:
            print(f"SKIP n={n:4d}: best_dL={float(best_L - inc_L):+.6e}")

    out_text = auto_union.final_text(exported) if exported else "id,x,y,deg\n"
    Path(args.out_csv).write_text(out_text, encoding="utf-8")

    # Self-verify exported file round-trip
    if exported:
        lit_g, pan_g = auto_union.load(Path(args.out_csv))
        for n in exported:
            res = auto_union.check((n, lit_g[n], pan_g[n]))
            if res is None:
                raise RuntimeError(f"Final CSV self-check failed for n={n}")

    Path(args.report_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_json).write_text(json.dumps({"wins": win_records}, indent=2), encoding="utf-8")
    print(f"Exported {len(exported)} strict win(s) to {args.out_csv}")


if __name__ == "__main__":
    main()
