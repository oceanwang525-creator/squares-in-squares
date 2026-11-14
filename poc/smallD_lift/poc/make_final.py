"""Write `poc/smallD_lift/final.csv` in the repository's required format.

Header `id,x,y,deg`; `deg` in degrees; every decimal prefixed with `s` so
pandas keeps it a string; ids `{n:03}_{square_id}`; complete groups only.

The file is then re-read from disk, degrees are converted back to radians and
every group is re-certified with the official metric in both transports -- a
win is only reported if the *written* file reproduces it.

    python make_final.py            # only strictly certified winning groups
    python make_final.py --all      # every group of the closure library
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "poc" / "baseline" / "poc" / "src"))

import mpmath as mp  # noqa: E402

DATA = HERE.parent / "data"
OUT = HERE.parent / "final.csv"


def write_final(path, groups):
    """groups: {n: [(x_str, y_str, angle_rad_str), ...]} in sid order."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["id", "x", "y", "deg"])
        with mp.workdps(95):
            for n, rows in sorted(groups.items()):
                assert len(rows) == n, (n, len(rows))
                for sid, (x, y, ang) in enumerate(rows):
                    deg = mp.nstr(mp.mpf(ang) * 180 / mp.pi, 75)
                    w.writerow([f"{n:03}_{sid}", "s" + x, "s" + y, "s" + deg])


def verify(path, expected):
    """Re-read final.csv, convert deg -> rad, certify both transports."""
    from squares.core import certify, kaggle_roundtrip
    groups = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            n = int(r["id"].split("_")[0])
            sid = int(r["id"].split("_")[1])
            groups.setdefault(n, []).append(
                (sid, r["x"].lstrip("s"), r["y"].lstrip("s"), r["deg"].lstrip("s")))
    ok = True
    with mp.workdps(95):
        for n, rows in sorted(groups.items()):
            rows.sort()
            assert [t[0] for t in rows] == list(range(n)), f"n={n}: ids not 0..n-1"
            sub = [dict(square_id=str(sid), x=x, y=y,
                        angle=mp.nstr(mp.mpf(d) * mp.pi / 180, 60))
                   for sid, x, y, d in rows]
            lit = mp.mpf(certify(n, sub))
            pan = mp.mpf(certify(n, kaggle_roundtrip({n: sub})[n]))
            exp = mp.mpf(expected[str(n)])
            d_lit, d_pan = lit - exp, pan - exp
            good = d_lit <= mp.mpf("-1e-9") and d_pan <= mp.mpf("-1e-9")
            ok &= good
            print(f"n={n:>4} certified_from_file={mp.nstr(lit, 24)} "
                  f"incumbent={mp.nstr(exp, 24)} dL_lit={mp.nstr(d_lit, 6)} "
                  f"dL_pandas={mp.nstr(d_pan, 6)} {'WIN' if good else 'NO WIN'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()

    wins = json.load(open(DATA / "closure_wins.json"))
    rows = pickle.load(open(DATA / "closure_wins.pkl", "rb"))
    best = {}
    for w in wins:                       # keep the best lift per n
        n = int(w["n"])
        if n not in best or mp.mpf(w["dL_lit"]) < mp.mpf(best[n]["dL_lit"]):
            best[n] = w
    groups = {n: rows[n] for n in best}
    expected = {str(n): json.load(open(DATA / "incumbent_side.json"))[str(n)]
                for n in groups} if (DATA / "incumbent_side.json").exists() else None

    inc = {}
    from squares.core import certify
    tracked = pickle.load(open(DATA / "tracked_rows.pkl", "rb"))
    for n in groups:
        inc[str(n)] = certify(n, [dict(square_id=str(k), x=x, y=y, angle=t)
                                  for k, (x, y, t) in enumerate(tracked[n])])
    json.dump(inc, open(DATA / "incumbent_side.json", "w"), indent=1)

    write_final(OUT, groups)
    print(f"wrote {OUT} with {len(groups)} groups, "
          f"{sum(len(r) for r in groups.values())} squares")
    ok = verify(OUT, inc)
    print("ALL WINNING GROUPS REPRODUCE" if ok else "MISMATCH -- do not claim")


if __name__ == "__main__":
    main()
