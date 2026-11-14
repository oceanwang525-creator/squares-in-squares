#!/usr/bin/env python3
"""Assemble an incumbent-format CSV from a base CSV plus one or more wins.json.

Each ``wins.json`` (as written by ``alln_fixed_lp``, ``lip_slp`` and
``swarm_lpslack``) maps ``"n" -> {"rows": [...]}`` with 50-digit decimal
strings, radians.  For every ``n`` present in ``--base`` we keep the candidate
(base rows or any win row set) with the smallest official side, certified in
**both** literal-decimal and pandas transport; the reported side is
``max(literal, pandas)`` so a later gate can never be fooled by the friendlier
of the two transports.

usage: assemble.py --base BASE.csv --wins W1.json [W2.json ...] --out OUT.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from decimal import Decimal
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "poc/baseline/poc/src"))
from squares import core  # noqa: E402


def read_groups(path: Path) -> dict[int, list[dict[str, str]]]:
    groups: dict[int, list[dict[str, str]]] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            groups.setdefault(int(row["n"]), []).append(
                {"square_id": row["square_id"], "x": row["x"], "y": row["y"],
                 "angle": row["angle"]})
    for n in groups:
        groups[n].sort(key=lambda r: int(r["square_id"]))
    return groups


def certified(n: int, rows: list[dict[str, str]]):
    try:
        lit = Decimal(core.certify(n, rows))
        pan = Decimal(core.certify(n, core.kaggle_roundtrip({n: rows})[n]))
    except Exception as exc:  # noqa: BLE001
        return None, None, str(exc)[:120]
    return lit, pan, None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--wins", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    groups = read_groups(Path(args.base))
    cands: dict[int, list[tuple[str, list[dict[str, str]]]]] = {
        n: [("base", rows)] for n, rows in groups.items()}
    for path in args.wins:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(raw, list):          # swarm_lpslack results list
            wins = {str(e["n"]): e for e in raw if isinstance(e, dict) and "n" in e}
        else:
            wins = raw
        for key, entry in wins.items():
            n = int(key)
            if n not in cands or "rows" not in entry:
                continue
            cands[n].append((Path(path).stem, entry["rows"]))

    chosen: dict[int, list[dict[str, str]]] = {}
    table = []
    for n in sorted(cands):
        best = None
        for tag, rows in cands[n]:
            if len(rows) != n:
                table.append(dict(n=n, tag=tag, status="bad-size"))
                continue
            lit, pan, err = certified(n, rows)
            if err is not None:
                table.append(dict(n=n, tag=tag, status="certify-fail", error=err))
                continue
            side = max(lit, pan)
            table.append(dict(n=n, tag=tag, status="ok", literal=str(lit),
                              pandas=str(pan), side=str(side)))
            if best is None or side < best[0]:
                best = (side, rows, tag)
        if best is not None:
            chosen[n] = best[1]
        print(f"n={n}: best={best[2] if best else '-'} side={best[0] if best else '-'}")

    gid = 0
    with Path(args.out).open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["id", "n", "square_id", "x", "y", "angle"])
        for n in sorted(chosen):
            for r in chosen[n]:
                w.writerow([gid, n, r["square_id"], r["x"], r["y"], r["angle"]])
                gid += 1
    Path(args.out).with_suffix(".detail.json").write_text(
        json.dumps(table, indent=1), encoding="utf-8")
    print(f"wrote {args.out} ({len(chosen)} groups)")


if __name__ == "__main__":
    main()
