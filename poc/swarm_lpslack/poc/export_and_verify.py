#!/usr/bin/env python3
"""Export strict certified wins to poc/swarm_lpslack/final.csv and verify against origin/main portfolio."""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from decimal import Decimal
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
POC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "poc" / "baseline" / "poc" / "src"))
sys.path.insert(0, str(REPO / "poc" / "auto_union" / "poc"))
sys.path.insert(0, str(POC / "poc"))

from squares import core  # noqa: E402
import union as auto_union  # noqa: E402
import lpslack  # noqa: E402


def _cert_candidate(item: tuple[int, list[dict[str, str]], str]) -> tuple[int, str | None, str | None]:
    n, cand_rows, inc_L_str = item
    inc_L = Decimal(inc_L_str)
    res = lpslack.certify_pair(n, cand_rows)
    if res is None:
        return n, None, None
    lit, pan = res
    return n, str(lit), str(pan)


def _verify_exported(item: tuple[int, list[dict[str, str]], list[dict[str, str]], str]) -> dict:
    n, lit_rows, pan_rows, inc_L_str = item
    inc_L = Decimal(inc_L_str)
    l1 = Decimal(core.certify(n, lit_rows))
    l2 = Decimal(core.certify(n, pan_rows))
    k_rt = core.kaggle_roundtrip({n: lit_rows})[n]
    l3 = Decimal(core.certify(n, k_rt))
    worst_L = max(l1, l2, l3)
    dL = inc_L - worst_L
    ds = dL / Decimal(n).sqrt()
    return {
        "n": n,
        "incumbent_L": str(inc_L),
        "new_L": str(worst_L),
        "literal_L": str(l1),
        "pandas_final_L": str(l2),
        "kaggle_roundtrip_L": str(l3),
        "dL": str(dL),
        "ds": str(ds),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", help="Result JSON files")
    parser.add_argument("--min-dl", default="1e-9")
    parser.add_argument("--jobs", type=int, default=16)
    args = parser.parse_args()

    min_dl = Decimal(args.min_dl)
    best_by_n: dict[int, dict] = {}
    for inp in args.inputs:
        data = json.loads(Path(inp).read_text(encoding="utf-8"))
        for row in data:
            if row.get("status") not in ("WIN", "TINY_WIN") or "rows" not in row:
                continue
            n = int(row["n"])
            new_L = Decimal(row["new_L"])
            if n not in best_by_n or new_L < Decimal(best_by_n[n]["new_L"]):
                best_by_n[n] = row

    if not best_by_n:
        (POC / "final.csv").write_text("id,x,y,deg\n", encoding="utf-8")
        print("No wins; wrote header-only final.csv")
        return

    portfolio = lpslack.load_targets_from_portfolio(set(best_by_n), jobs=args.jobs)
    cand_tasks = []
    cand_map: dict[int, list[dict[str, str]]] = {}
    for n, row in sorted(best_by_n.items()):
        _, inc_L = portfolio[n]
        cand_rows = [
            {
                "square_id": str(r["square_id"]),
                "x": str(r["x"]),
                "y": str(r["y"]),
                "angle": str(r["angle"]),
            }
            for r in row["rows"]
        ]
        cand_map[n] = cand_rows
        cand_tasks.append((n, cand_rows, str(inc_L)))

    exported: dict[int, list[dict[str, str]]] = {}
    with mp.get_context("fork").Pool(min(args.jobs, len(cand_tasks))) as pool:
        for n, lit_s, pan_s in pool.imap_unordered(_cert_candidate, cand_tasks, chunksize=1):
            if lit_s is None:
                print(f"SKIP n={n}: failed certification", flush=True)
                continue
            _, inc_L = portfolio[n]
            lit, pan = Decimal(lit_s), Decimal(pan_s)
            cand_L = max(lit, pan)
            if inc_L - cand_L >= min_dl and lit < inc_L and pan < inc_L:
                exported[n] = cand_map[n]
            else:
                print(f"SKIP n={n}: dL={inc_L - cand_L} < {min_dl}", flush=True)

    final_path = POC / "final.csv"
    final_path.write_text(auto_union.final_text(exported), encoding="utf-8")

    # Reload from final.csv and re-verify both literal and pandas + kaggle_roundtrip
    lit_groups, pan_groups = auto_union.load(final_path)
    assert set(lit_groups) == set(exported) and set(pan_groups) == set(exported)

    ver_tasks = [(n, lit_groups[n], pan_groups[n], str(portfolio[n][1])) for n in sorted(exported)]
    verified = []
    with mp.get_context("fork").Pool(min(args.jobs, len(ver_tasks))) as pool:
        for row in pool.imap_unordered(_verify_exported, ver_tasks, chunksize=1):
            assert Decimal(row["dL"]) >= min_dl, f"n={row['n']} failed post-export check: dL={row['dL']}"
            verified.append(row)

    verified.sort(key=lambda r: r["n"])
    total_ds = sum(Decimal(r["ds"]) for r in verified)
    for r in verified:
        print(
            f"VERIFIED n={r['n']} inc={Decimal(r['incumbent_L']):.12f} "
            f"new={Decimal(r['new_L']):.12f} dL=-{Decimal(r['dL']):.6e} ds=-{Decimal(r['ds']):.6e}",
            flush=True,
        )

    summary = {
        "count": len(verified),
        "total_ds": str(total_ds),
        "wins": verified,
    }
    (POC / "docs").mkdir(parents=True, exist_ok=True)
    (POC / "docs" / "verify_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"TOTAL: {len(verified)} wins, total_ds=-{total_ds:.6e}")


if __name__ == "__main__":
    main()
