#!/usr/bin/env python3
"""Verify saved win_n*.json candidates against live origin/main portfolio and rebuild final.csv."""
from __future__ import annotations

import argparse
import io
import json
import sys
from decimal import Decimal
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
POC_DIR = Path(__file__).resolve().parent
UNION_POC = REPO_ROOT / "poc" / "auto_union" / "poc"
BASELINE_SRC = REPO_ROOT / "poc" / "baseline" / "poc" / "src"

for _p in (POC_DIR, UNION_POC, BASELINE_SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import portfolio as pf  # noqa: E402
import union as auto_union  # noqa: E402
from squares.core import certify, kaggle_roundtrip  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify win_n*.json vs live portfolio and rebuild final.csv")
    ap.add_argument("--prune", action="store_true", help="Delete non-strict wins relative to live portfolio")
    args = ap.parse_args()

    win_files = sorted(pf.DATA_DIR.glob("win_n*.json"))
    print(f"Found {len(win_files)} win JSON files in {pf.DATA_DIR}")
    if not win_files:
        pf.rebuild_final_csv()
        return 0

    win_data: dict[int, tuple[Path, dict]] = {}
    for wf in win_files:
        d = json.loads(wf.read_text(encoding="utf-8"))
        win_data[int(d["n"])] = (wf, d)

    live = pf.load_live_portfolio(list(win_data.keys()))

    kept: list[int] = []
    pruned: list[int] = []
    for n in sorted(win_data.keys()):
        wf, d = win_data[n]
        rows = d["rows"]
        text = auto_union.final_text({n: rows})
        lit_g = auto_union.parse_text(text)
        pd_text = pd.read_csv(io.StringIO(text)).to_csv(index=False)
        pd_g = auto_union.parse_text(pd_text)
        res = auto_union.check((n, lit_g[n], pd_g[n]))
        if res is None:
            raise RuntimeError(f"auto_union.check failed for n={n}")
        l_lit, l_pd = Decimal(res[0]), Decimal(res[1])
        l_worst = max(l_lit, l_pd)

        kr = kaggle_roundtrip(lit_g)
        l_kr = Decimal(certify(n, kr[n]))

        inc = live[n]
        inc_lit = inc["L_lit"]
        inc_pd = inc["L_pandas"]
        inc_worst = inc["L_dec"]
        strict = (l_lit < inc_lit) and (l_pd < inc_pd) and (l_worst < inc_worst) and (l_kr < inc_worst)
        dL = float(l_worst - inc_worst)
        print(
            f"n={n}: strict={strict} lit==pd={l_lit == l_pd} kr==lit={l_kr == l_lit} "
            f"cand={l_worst} inc={inc_worst} ({inc['source']}) dL={dL:+.6e} "
            f"track={d.get('best_track')}"
        )
        if strict:
            d["L_lit"] = res[0]
            d["L_pandas"] = res[1]
            d["L_dec_str"] = str(l_worst)
            d["inc_L_vs_main"] = str(inc_worst)
            d["inc_src_vs_main"] = inc["source"]
            wf.write_text(json.dumps(d, indent=2), encoding="utf-8")
            kept.append(n)
        else:
            pruned.append(n)
            if args.prune:
                wf.unlink()

    if pruned and not args.prune:
        raise RuntimeError(f"Non-strict wins found (pass --prune to remove): {pruned}")

    count = pf.rebuild_final_csv()
    print(f"\nKept: {kept}, Pruned: {pruned}, Rebuilt final.csv with {count} groups")
    print(f"final.csv bytes: {pf.FINAL_CSV.stat().st_size}")

    lit_all, pd_all = auto_union.load(pf.FINAL_CSV)
    assert sorted(lit_all.keys()) == sorted(kept)
    kr_all = kaggle_roundtrip(lit_all)
    for n in kept:
        res = auto_union.check((n, lit_all[n], pd_all[n]))
        if res is None or Decimal(res[0]) != Decimal(res[1]):
            raise RuntimeError(f"final.csv dual-check mismatch for n={n}: {res}")
        l_kr = Decimal(certify(n, kr_all[n]))
        assert l_kr == Decimal(res[0])
        print(f"final.csv dual-check n={n}: literal==pandas==kaggle_roundtrip=={res[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
