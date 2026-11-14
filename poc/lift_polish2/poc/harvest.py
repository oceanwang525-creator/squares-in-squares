#!/usr/bin/env python3
"""Find every n where a portfolio `final.csv` / built group beats fresh main.

Fresh main (`merged.csv`) is the scored submission, so any *previously certified*
group that is strictly smaller than main's group for the same n is a real,
certifiable win against main — regardless of how it was found.  This script ranks
every candidate group by its exact float bbox side and picks the best per n.

Sources (read-only):
  * `poc/pell_corrugation/data/border-scan/merged.csv`
  * every `poc/*/final.csv`                                   (`id,x,y,deg`)
  * every `poc/*/data/built*/n<k>.json`                        (radians)

usage: harvest.py OUT.csv --prov OUT.json [--margin 1e-7]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def side_of(df: pd.DataFrame) -> float:
    a = df["angle"].to_numpy(np.float64)
    x = df["x"].to_numpy(np.float64)
    y = df["y"].to_numpy(np.float64)
    e = 0.5 * (np.abs(np.cos(a)) + np.abs(np.sin(a)))
    return float(max((x + e).max() - (x - e).min(), (y + e).max() - (y - e).min()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--prov")
    ap.add_argument("--margin", type=float, default=1e-7)
    ap.add_argument("--exclude", default="poc/lift_polish2")
    a = ap.parse_args()

    main_df = pd.read_csv("poc/pell_corrugation/data/border-scan/merged.csv",
                          dtype={"square_id": str, "x": str, "y": str, "angle": str},
                          usecols=["n", "square_id", "x", "y", "angle"])
    main_df["angle"] = main_df["angle"].astype(np.float64)
    base_side = {}
    for n, g in main_df.groupby("n", sort=False):
        base_side[int(n)] = side_of(g)
    print(f"fresh main: {len(base_side)} groups")

    best: dict[int, tuple[float, str, pd.DataFrame]] = {}
    for n, s in base_side.items():
        g = main_df[main_df["n"] == n]
        if len(g) == n:
            best[n] = (s, "merged.csv", g)

    # every poc/*/final.csv
    files = sorted(glob.glob("poc/*/final.csv"))
    for path in files:
        if a.exclude and path.startswith(a.exclude):
            continue
        try:
            df = pd.read_csv(path, dtype=str, usecols=["id", "x", "y", "deg"])
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {path}: {exc}")
            continue
        if df.empty:
            continue
        df["n"] = df["id"].str.split("_").str[0].astype(int)
        df["x"] = df["x"].str.lstrip("s").astype(np.float64)
        df["y"] = df["y"].str.lstrip("s").astype(np.float64)
        df["angle"] = np.radians(df["deg"].str.lstrip("s").astype(np.float64))
        for n, g in df.groupby("n", sort=False):
            n = int(n)
            if len(g) != n or n not in best:
                continue
            s = side_of(g)
            if s < best[n][0] - 1e-10:
                best[n] = (s, path, g[["x", "y", "angle"]].assign(square_id=g["id"].str.split("_").str[1]))
    print(f"scanned {len(files)} final.csv files")

    # every built*/n<k>.json
    nbuilt = 0
    for path in sorted(glob.glob("poc/*/data/built*/n*.json")):
        if a.exclude and path.startswith(a.exclude):
            continue
        stem = Path(path).stem
        if not stem.startswith("n") or not stem[1:].isdigit():
            continue
        n = int(stem[1:])
        if n not in best:
            continue
        rows = json.loads(Path(path).read_text(encoding="utf-8"))
        if len(rows) != n:
            continue
        df = pd.DataFrame(rows).astype({"x": float, "y": float, "angle": float})
        s = side_of(df)
        if s < best[n][0] - 1e-10:
            best[n] = (s, path, df)
        nbuilt += 1
    print(f"scanned {nbuilt} built groups")

    wins = []
    for n in sorted(best):
        s, src, g = best[n]
        d = s - base_side[n]
        if d < -a.margin:
            wins.append((n, d, src, s))
    wins.sort(key=lambda t: t[1] / math.sqrt(t[0]))
    total = sum(d / math.sqrt(n) for n, d, _, _ in wins)
    print(f"{len(wins)} n with a materially smaller known group; nominal score gain {total:.6e}")
    for n, d, src, s in wins:
        print(f"  n={n}: main {base_side[n]:.9f} -> {s:.9f}  dL {d:+.3e}  {src}")

    out_rows = []
    for n, d, src, s in wins:
        _, _, g = best[n]
        g = g.sort_values("square_id", key=lambda c: c.astype(int))
        for sid, x, y, ang in zip(g["square_id"], g["x"], g["y"], g["angle"]):
            out_rows.append(dict(n=n, square_id=str(sid), x=repr(float(x)),
                                 y=repr(float(y)), angle=repr(float(ang))))
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        import csv
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["id", "n", "square_id", "x", "y", "angle"])
        for i, r in enumerate(out_rows):
            w.writerow([i, r["n"], r["square_id"], r["x"], r["y"], r["angle"]])
    print(f"wrote {a.out}: {len(wins)} groups, {len(out_rows)} rows")
    if a.prov:
        Path(a.prov).write_text(json.dumps(
            {str(n): dict(dL_float=d, source=src, L=float(s)) for n, d, src, s in wins},
            indent=1), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
