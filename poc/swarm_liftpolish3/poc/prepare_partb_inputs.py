#!/usr/bin/env python3
"""Prepare Part B inputs: overlay wins onto combined_incumbent.csv and append gstep brick synthetic donors."""
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--incumbent-csv", default=str(REPO / "poc/swarm_liftpolish3/data/combined_incumbent.csv"))
    ap.add_argument("--wins-dir", default=str(REPO / "poc/swarm_liftpolish3/data/wins"))
    ap.add_argument("--lpslack-json", default=str(REPO / "poc/swarm_liftpolish3/data/partA_lpslack.json"))
    ap.add_argument("--out-incumbent-csv", default=str(REPO / "poc/swarm_liftpolish3/data/partb_incumbent.csv"))
    ap.add_argument("--out-donors-csv", default=str(REPO / "poc/swarm_liftpolish3/data/partb_donors.csv"))
    ap.add_argument("--out-bricks-json", default=str(REPO / "poc/swarm_liftpolish3/data/gstep_pockets.json"))
    args = ap.parse_args()

    # Load incumbent
    groups: dict[int, list[dict[str, str]]] = {}
    with Path(args.incumbent_csv).open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            groups.setdefault(int(r["n"]), []).append({
                "square_id": r["square_id"],
                "x": r["x"],
                "y": r["y"],
                "angle": r["angle"],
            })

    # Collect Part A wins from lpslack JSON and wins_dir
    cand_wins: dict[int, list[dict[str, str]]] = {}
    lp_path = Path(args.lpslack_json)
    if lp_path.exists():
        for rec in json.loads(lp_path.read_text(encoding="utf-8")):
            if rec.get("status") in ("WIN", "TINY_WIN") and "rows" in rec:
                cand_wins[int(rec["n"])] = rec["rows"]

    wdir = Path(args.wins_dir)
    if wdir.exists():
        for wp in sorted(wdir.glob("win_n*.json")):
            d = json.loads(wp.read_text(encoding="utf-8"))
            cand_wins[int(d["n"])] = d["rows"]

    for n, rows in sorted(cand_wins.items()):
        old_L = Decimal(certify(n, groups[n]))
        res = lpslack.certify_pair(n, rows)
        if res is not None:
            new_L = max(res[0], res[1])
            if new_L < old_L - Decimal("1e-12"):
                print(f"[overlay] n={n}: {str(old_L)[:16]} -> {str(new_L)[:16]} (dL={float(new_L - old_L):+.6e})")
                groups[n] = rows

    Path(args.out_incumbent_csv).write_text(auto_union.csv_text(groups), encoding="utf-8")

    # Add gstep bricks as synthetic donors src = 2000 + idx
    donors = dict(groups)
    brick_pockets = []
    for idx, bp in enumerate(sorted(glob.glob(str(REPO / "poc/swarm_gstep/data/brick_*.json")))):
        d = json.loads(Path(bp).read_text(encoding="utf-8"))
        src = 2000 + idx
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
            # Verify raw
            if lpslack.certify_pair(cnt, srows) is None:
                print(f"[skip brick] {d['name']} failed certification")
                continue
        donors[src] = srows
        # Measure exact bbox in Row-normalized coordinates (min lx == 0, min ly == 0)
        xs = np.array([float(r["x"]) for r in srows])
        ys = np.array([float(r["y"]) for r in srows])
        ts = np.array([float(r["angle"]) for r in srows])
        es = 0.5 * (np.abs(np.cos(ts)) + np.abs(np.sin(ts)))
        x0_raw, x1_raw = float((xs - es).min()), float((xs + es).max())
        y0_raw, y1_raw = float((ys - es).min()), float((ys + es).max())
        W, H = x1_raw - x0_raw, y1_raw - y0_raw
        w = int(W + 1e-6)
        h = int(H + 1e-6)
        g = cnt - w * h
        f = max(W - w, H - h)
        if g >= 1:
            brick_pockets.append({
                "src": src,
                "w": w,
                "h": h,
                "g": g,
                "f": f,
                "cnt": cnt,
                "x0": 0.0,
                "x1": W,
                "y0": 0.0,
                "y1": H,
            })

    Path(args.out_donors_csv).write_text(auto_union.csv_text(donors), encoding="utf-8")
    Path(args.out_bricks_json).write_text(json.dumps(brick_pockets, indent=0), encoding="utf-8")
    print(f"Wrote {args.out_incumbent_csv} (1000 groups), {args.out_donors_csv} ({len(donors)} groups), {len(brick_pockets)} brick pockets")


if __name__ == "__main__":
    main()
