#!/usr/bin/env python3
"""Merge pocket libraries, run Pareto prune, and sanitize against donors_all.csv."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "poc" / "pocket_lift_scan3" / "poc"))
from scan_pockets import Row, load_rows  # noqa: E402

sys.path.insert(0, str(REPO / "poc" / "swarm_softseamdp" / "poc"))
import dp_soft  # noqa: E402

TOL = 1e-9


def sanitize_records(records: list[dict], csv_path: str) -> list[dict]:
    rows = load_rows(csv_path)
    cache = {}
    out = []
    dropped = 0
    fixed = 0
    for g in records:
        n = int(g["src"])
        if n not in rows:
            continue
        if n not in cache:
            cache[n] = Row(n, rows[n])
        row = cache[n]
        m = (
            (row.lx >= g["x0"] - TOL)
            & (row.hx <= g["x1"] + TOL)
            & (row.ly >= g["y0"] - TOL)
            & (row.hy <= g["y1"] + TOL)
        )
        k = int(m.sum())
        if k < 2:
            dropped += 1
            continue
        X0, X1 = float(row.lx[m].min()), float(row.hx[m].max())
        Y0, Y1 = float(row.ly[m].min()), float(row.hy[m].max())
        W, H = X1 - X0, Y1 - Y0
        w = int(W + 1e-6)
        h = int(H + 1e-6)
        gg = k - w * h
        if gg < 1 or w < 1 or h < 1:
            dropped += 1
            continue
        f = max(W - w, H - h)
        rec = dict(g)
        rec.update(w=w, h=h, g=gg, f=f, cnt=k, x0=X0, x1=X1, y0=Y0, y1=Y1)
        if gg != g["g"] or abs(f - g["f"]) > 1e-12:
            fixed += 1
        out.append(rec)
    print(f"Sanitize: {len(records)} -> {len(out)} kept ({dropped} dropped, {fixed} re-measured)")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("inputs", nargs="+")
    args = ap.parse_args()

    all_recs = []
    for p in args.inputs:
        if Path(p).exists():
            recs = json.loads(Path(p).read_text())
            print(f"  loaded {len(recs)} from {p}")
            all_recs.extend(recs)

    # First sanitize everything against donors_all.csv
    sane = sanitize_records(all_recs, args.csv)

    # Keep best f per (w, h, g) oriented, then pareto prune
    best: dict[tuple[int, int, int], dict] = {}
    for g in sane:
        gw, gh, gg, gf = int(g["w"]), int(g["h"]), int(g["g"]), float(g["f"])
        for w, h in ((gw, gh), (gh, gw)):
            k = (w, h, gg)
            if k not in best or gf < best[k]["f"] - 1e-12:
                rec = dict(g)
                rec["w"], rec["h"] = w, h
                rec["nw"], rec["nh"] = gw, gh
                best[k] = rec

    items = dp_soft.pareto([dict(w=v["w"], h=v["h"], g=v["g"], f=v["f"]) for v in best.values()])
    keep = {(it["w"], it["h"], it["g"]) for it in items}

    surv = {}
    for k, v in best.items():
        if k not in keep:
            continue
        nk = (int(v["src"]), float(v["x0"]), float(v["x1"]), float(v["y0"]), float(v["y1"]), int(v["g"]))
        if nk not in surv:
            rec = {kk: v[kk] for kk in ("src", "w", "h", "g", "f", "cnt", "x0", "x1", "y0", "y1")}
            rec["w"], rec["h"] = v["nw"], v["nh"]
            surv[nk] = rec

    out_list = sorted(surv.values(), key=lambda v: (v["g"], v["f"]))
    Path(args.out).write_text(json.dumps(out_list, indent=0))
    print(f"Merged Pareto sane library: {len(out_list)} pocket records -> {args.out}")


if __name__ == "__main__":
    main()
