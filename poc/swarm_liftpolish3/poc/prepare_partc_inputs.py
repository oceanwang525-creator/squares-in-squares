#!/usr/bin/env python3
"""Prepare Part C inputs on origin/main (1f337f9): overlay wins_r2 + convert cleanpeel2/gstep bricks + list target rows to scan."""
from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from decimal import Decimal
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "poc" / "auto_union" / "poc"))
sys.path.insert(0, str(REPO / "poc" / "swarm_lpslack" / "poc"))
sys.path.insert(0, str(REPO / "poc" / "baseline" / "poc" / "src"))

import lpslack  # noqa: E402
import union as auto_union  # noqa: E402
from squares.core import certify  # noqa: E402


def collect_brick_pockets(donors: dict[int, list[dict[str, str]]], globs: list[str], start_src: int = 2000) -> list[dict]:
    pockets = []
    seen_files: list[Path] = []
    for pat in globs:
        seen_files.extend(sorted(Path(p) for p in glob.glob(str(REPO / pat))))
    for idx, bp in enumerate(seen_files):
        d = json.loads(bp.read_text(encoding="utf-8"))
        if not all(k in d for k in ("x", "y", "th")):
            continue
        src = start_src + idx
        x = np.array(d["x"], dtype=np.float64)
        y = np.array(d["y"], dtype=np.float64)
        th = np.array(d["th"], dtype=np.float64)
        e = 0.5 * (np.abs(np.cos(th)) + np.abs(np.sin(th)))
        x -= float((x - e).min())
        y -= float((y - e).min())
        cnt = len(x)
        centers = np.column_stack([x, y])
        shrunk = lpslack.try_shrink_and_certify(cnt, centers, th, Decimal("999.0"))
        if shrunk is not None:
            _, _, srows = shrunk
        else:
            srows = [
                {"square_id": str(i), "x": repr(float(x[i])), "y": repr(float(y[i])), "angle": repr(float(th[i]))}
                for i in range(cnt)
            ]
            if lpslack.certify_pair(cnt, srows) is None:
                continue
        donors[src] = srows
        xs = np.array([float(r["x"]) for r in srows])
        ys = np.array([float(r["y"]) for r in srows])
        ts = np.array([float(r["angle"]) for r in srows])
        es = 0.5 * (np.abs(np.cos(ts)) + np.abs(np.sin(ts)))
        W = float((xs + es).max() - (xs - es).min())
        H = float((ys + es).max() - (ys - es).min())
        w = int(W + 1e-6)
        h = int(H + 1e-6)
        g = cnt - w * h
        f = max(W - w, H - h)
        if g >= 1:
            pockets.append({
                "src": src, "w": w, "h": h, "g": g, "f": f, "cnt": cnt,
                "x0": 0.0, "x1": W, "y0": 0.0, "y1": H,
            })
    return pockets


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged-csv", default=str(REPO / "poc/pell_corrugation/data/border-scan/merged.csv"))
    ap.add_argument("--wins-dir", default=str(REPO / "poc/swarm_liftpolish3/data/wins_r2"))
    ap.add_argument("--out-incumbent-csv", default=str(REPO / "poc/swarm_liftpolish3/data/partc_incumbent.csv"))
    ap.add_argument("--out-donors-csv", default=str(REPO / "poc/swarm_liftpolish3/data/partc_donors.csv"))
    ap.add_argument("--out-bricks-json", default=str(REPO / "poc/swarm_liftpolish3/data/partc_bricks.json"))
    ap.add_argument("--out-rows-txt", default=str(REPO / "poc/swarm_liftpolish3/data/partc_scan_rows.txt"))
    args = ap.parse_args()

    groups: dict[int, list[dict[str, str]]] = {}
    with Path(args.merged_csv).open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            groups.setdefault(int(r["n"]), []).append({
                "square_id": r["square_id"],
                "x": r["x"],
                "y": r["y"],
                "angle": r["angle"],
            })

    wdir = Path(args.wins_dir)
    win_ns: set[int] = set()
    if wdir.exists():
        for wp in sorted(wdir.glob("win_n*.json")):
            d = json.loads(wp.read_text(encoding="utf-8"))
            n = int(d["n"])
            rows = d["rows"]
            old_L = Decimal(certify(n, groups[n]))
            res = lpslack.certify_pair(n, rows)
            if res is not None and max(res[0], res[1]) < old_L - Decimal("1e-12"):
                groups[n] = rows
                win_ns.add(n)
                print(f"[overlay r2] n={n}: {str(old_L)[:16]} -> {str(max(res[0], res[1]))[:16]}")

    Path(args.out_incumbent_csv).write_text(auto_union.csv_text(groups), encoding="utf-8")

    donors = dict(groups)
    brick_pockets = collect_brick_pockets(
        donors,
        [
            "poc/swarm_gstep/data/brick_*.json",
            "poc/swarm_cleanpeel2/data/brick_*.json",
        ],
        start_src=2000,
    )
    Path(args.out_donors_csv).write_text(auto_union.csv_text(donors), encoding="utf-8")
    Path(args.out_bricks_json).write_text(json.dumps(brick_pockets, indent=0), encoding="utf-8")

    scan_ns: set[int] = set(win_ns)
    for final_rel in (
        "poc/public_watch_nov14/final.csv",
        "poc/row_ratchet_980/final.csv",
        "poc/swarm_corrcasc/final.csv",
        "poc/swarm_gstep/final.csv",
        "poc/swarm_band3/final.csv",
        "poc/swarm_cleanpeel2/final.csv",
        "poc/swarm_interlock2/final.csv",
        "poc/swarm_softseamdp/final.csv",
        "poc/swarm_liftpolish3/final.csv",
    ):
        fp = REPO / final_rel
        if fp.exists():
            parsed = auto_union.parse_text(fp.read_text(encoding="utf-8"))
            scan_ns.update(n for n in parsed if n >= 200)

    rows_str = ",".join(str(n) for n in sorted(scan_ns))
    Path(args.out_rows_txt).write_text(rows_str, encoding="utf-8")
    print(f"Prepared Part C: {len(brick_pockets)} bricks, {len(scan_ns)} rows to scan")


if __name__ == "__main__":
    main()
