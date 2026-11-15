#!/usr/bin/env python3
"""Dual-certify (literal == pandas) all wins in poc/swarm_vstrip852/data/wins/ against origin/main and export final.csv."""
from __future__ import annotations

import io
import json
import os
import sys
from decimal import Decimal
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vstrip_lib as vl  # noqa: E402
import gstep_lib as gl  # noqa: E402
import union as auto_union  # noqa: E402


def main() -> None:
    wins_dir = vl.DATA_DIR / "wins"
    cand_files = sorted(wins_dir.glob("win_n*.json")) if wins_dir.exists() else []
    # Also include root data/win_n*.json if any
    for p in sorted(vl.DATA_DIR.glob("win_n*.json")):
        cand_files.append(p)

    best_by_n: dict[int, dict] = {}
    for p in cand_files:
        d = json.loads(p.read_text(encoding="utf-8"))
        n = int(d["n"])
        L_worst = Decimal(max(d["L_lit"], d["L_pd"]))
        if n not in best_by_n or L_worst < Decimal(max(best_by_n[n]["L_lit"], best_by_n[n]["L_pd"])):
            best_by_n[n] = d

    if not best_by_n:
        vl.FINAL_CSV.write_text("id,x,y,deg\n", encoding="utf-8")
        print("No wins found; wrote header-only final.csv")
        return

    frontier = gl.fast_load_true_frontier(set(best_by_n.keys()), certify=True)
    accepted: dict[int, list[dict]] = {}
    records: list[dict] = []

    for n in sorted(best_by_n):
        d = best_by_n[n]
        rows = d["rows"]
        inc_dec = frontier[n]["L_dec"]
        text = auto_union.final_text({n: rows})
        lit = auto_union.parse_text(text)
        pd_t = auto_union.parse_text(pd.read_csv(io.StringIO(text)).to_csv(index=False))
        res = auto_union.check((n, lit[n], pd_t[n]))
        if res is None:
            print(f"[SKIP] n={n} failed dual check", flush=True)
            continue
        L_lit, L_pd = Decimal(res[0]), Decimal(res[1])
        L_worst = max(L_lit, L_pd)
        dL = L_worst - inc_dec
        if dL <= Decimal("-1e-9"):
            accepted[n] = rows
            rec = {
                "n": n,
                "label": d.get("label", ""),
                "source": frontier[n]["source"],
                "L_inc": str(inc_dec),
                "L_lit": str(L_lit),
                "L_pd": str(L_pd),
                "dL": float(dL),
            }
            records.append(rec)
            print(f"[PASS] n={n}: {str(inc_dec)[:18]} -> {str(L_worst)[:18]} (dL={float(dL):+.6e}) [{d.get('label','')}]", flush=True)
        else:
            print(f"[NO_WIN] n={n}: dL={float(dL):+.6e} >= -1e-9", flush=True)

    text_out = auto_union.final_text(accepted)
    auto_union.atomic(vl.FINAL_CSV, text_out.encode("utf-8"))
    os.chmod(vl.FINAL_CSV, 0o644)
    vl.DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (vl.DOCS_DIR / "verification.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"Exported {len(accepted)} strict wins to {vl.FINAL_CSV}", flush=True)


if __name__ == "__main__":
    main()
