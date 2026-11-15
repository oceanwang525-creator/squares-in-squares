#!/usr/bin/env python3
"""Dual-certify poc/swarm_armslack2/final.csv (literal + pandas) against the live portfolio."""
from __future__ import annotations

import json
import sys
from decimal import Decimal
from pathlib import Path

POC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(POC_DIR))
import portfolio as pf  # noqa: E402
import union as auto_union  # noqa: E402


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else pf.FINAL_CSV
    lit, pdt = auto_union.load(path)
    ns = sorted(lit)
    port = pf.load_live_portfolio(ns)
    out = []
    tot_ds = 0.0
    wins = 0
    for n in ns:
        res = auto_union.check((n, lit[n], pdt[n]))
        if res is None:
            print(f"n={n}: INVALID")
            out.append(dict(n=n, ok=False))
            continue
        a, b = map(Decimal, res)
        inc = port[n]
        strict = a < inc["L_lit"] and b < inc["L_pandas"]
        dL = float(max(a, b) - inc["L_dec"])
        ds = float((max(a, b) ** 2 - inc["L_dec"] ** 2) / n)
        if strict:
            wins += 1
            tot_ds += ds
        print(
            f"n={n}: lit={str(a)[:22]} pd={str(b)[:22]} inc={str(inc['L_dec'])[:22]} ({inc['source']}) "
            f"dL={dL:+.6e} ds_L2n={ds:+.6e} strict={strict} lit==pd={a == b}"
        )
        out.append(
            dict(
                n=n,
                L_lit=str(a),
                L_pandas=str(b),
                L_inc=str(inc["L_dec"]),
                inc_source=inc["source"],
                dL=dL,
                ds_L2n=ds,
                strict=strict,
            )
        )
    print(f"WINS={wins}/{len(ns)} total_ds_L2n={tot_ds:+.6e}")
    pf.DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (pf.DOCS_DIR / "verify_final.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
