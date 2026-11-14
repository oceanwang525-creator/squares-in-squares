#!/usr/bin/env python3
"""Build the best-known start row for each target n from every available source.

Sources scanned (read-only):
  * `poc/pell_corrugation/data/border-scan/merged.csv`   (fresh main)
  * every `poc/*/final.csv`                              (`id,x,y,deg`, `s`-prefixed)
  * every `poc/*/data/built*/n<k>.json`                  (radians, decimal strings)

Ranking is the exact float bbox side L = max(W, H) of the group (translation
invariant), so only *relative* order matters here; the official 50-digit sides of
the chosen groups are produced afterwards by `poc/faultline_dp2/poc/mkreport.py`.

usage: mk_start.py OUT.csv [--n 442,447,...] [--prov OUT.json]
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import math
from pathlib import Path


def side(rows) -> float:
    xs0 = xs1 = ys0 = ys1 = None
    for x, y, a in rows:
        h = 0.5 * (abs(math.cos(a)) + abs(math.sin(a)))
        xs0 = x - h if xs0 is None else min(xs0, x - h)
        xs1 = x + h if xs1 is None else max(xs1, x + h)
        ys0 = y - h if ys0 is None else min(ys0, y - h)
        ys1 = y + h if ys1 is None else max(ys1, y + h)
    return max(xs1 - xs0, ys1 - ys0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--n", default="442,447,485,492,530,577,740,743,798")
    ap.add_argument("--prov")
    a = ap.parse_args()
    ns = {int(x) for x in a.n.split(",") if x.strip()}

    cands: dict[int, list[tuple[float, str, list[dict[str, str]]]]] = collections.defaultdict(list)

    def add(n: int, tag: str, rows: list[dict[str, str]], triples) -> None:
        if len(rows) != n:
            return
        cands[n].append((side(triples), tag, rows))

    # fresh main
    g: dict[int, list[dict[str, str]]] = collections.defaultdict(list)
    with open("poc/pell_corrugation/data/border-scan/merged.csv", newline="",
              encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            n = int(r["n"])
            if n in ns:
                g[n].append(dict(square_id=r["square_id"], x=r["x"], y=r["y"],
                                 angle=r["angle"]))
    for n, rows in g.items():
        rows.sort(key=lambda r: int(r["square_id"]))
        add(n, "merged.csv", rows, [(float(r["x"]), float(r["y"]), float(r["angle"]))
                                    for r in rows])

    # every poc/*/final.csv (deg)
    for path in sorted(glob.glob("poc/*/final.csv")):
        gg: dict[int, list[tuple[str, str, str, float]]] = collections.defaultdict(list)
        try:
            with open(path, newline="", encoding="utf-8") as fh:
                for r in csv.DictReader(fh):
                    gid = r["id"].split("_")[0]
                    try:
                        n = int(gid)
                    except ValueError:
                        continue
                    if n in ns:
                        gg[n].append((r["id"].split("_")[1], r["x"].lstrip("s"),
                                      r["y"].lstrip("s"), r["deg"].lstrip("s")))
        except Exception:  # noqa: BLE001
            continue
        for n, rows in gg.items():
            rows.sort(key=lambda t: int(t[0]))
            recs = [dict(square_id=sid, x=x, y=y,
                         angle=repr(math.radians(float(deg)))) for sid, x, y, deg in rows]
            add(n, path, recs, [(float(x), float(y), math.radians(float(deg)))
                                for _, x, y, deg in rows])

    # every built*/n<k>.json (radians)
    for path in sorted(glob.glob("poc/*/data/built*/n*.json")):
        stem = Path(path).stem
        if not stem.startswith("n") or not stem[1:].isdigit():
            continue
        n = int(stem[1:])
        if n not in ns:
            continue
        rows = json.loads(Path(path).read_text(encoding="utf-8"))
        rows.sort(key=lambda r: int(r["square_id"]))
        add(n, path, [dict(square_id=r["square_id"], x=str(r["x"]), y=str(r["y"]),
                           angle=str(r["angle"])) for r in rows],
            [(float(r["x"]), float(r["y"]), float(r["angle"])) for r in rows])

    prov = {}
    out = []
    for n in sorted(ns):
        lst = sorted(cands[n], key=lambda t: t[0])
        best = lst[0]
        prov[n] = dict(L=best[0], source=best[1],
                       runners_up=[(round(s, 9), t) for s, t, _ in lst[1:4]])
        print(f"n={n}: best {best[0]:.12f} from {best[1]}")
        for s, t, _ in lst[1:4]:
            print(f"        next {s:.12f} {t}")
        for r in best[2]:
            out.append((n, r))

    with open(a.out, "w", newline="", encoding="utf-8") as fo:
        w = csv.writer(fo, lineterminator="\n")
        w.writerow(["id", "n", "square_id", "x", "y", "angle"])
        for i, (n, r) in enumerate(out):
            w.writerow([i, n, r["square_id"], r["x"], r["y"], r["angle"]])
    print(f"wrote {a.out}: {len(out)} rows for {len(ns)} groups")
    if a.prov:
        Path(a.prov).write_text(json.dumps({str(k): v for k, v in prov.items()}, indent=1),
                                encoding="utf-8")


if __name__ == "__main__":
    main()
