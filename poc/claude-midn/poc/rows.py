#!/usr/bin/env python3
"""Row-level helpers: extract puzzle groups, splice them into a full file, gate them.

  rows.py extract FULL.csv N[,N...] OUT_ROWS.csv
  rows.py splice  BASE_FULL.csv ROWS.csv OUT_FULL.csv
  rows.py gate    INCUMBENT_FULL.csv ROWS.csv OUT_ACCEPTED_ROWS.csv [LO HI]

`gate` keeps a group only if n is inside [LO, HI] (default 200..599) and its
official L is strictly below the incumbent group's L in BOTH transports
(literal strings, and after the pandas float round trip).  Strings are never
re-formatted: accepted rows are written exactly as they were read.
"""
import csv
import json
import sys
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "poc" / "baseline" / "poc" / "src"))
from squares import core  # noqa: E402

COLS = ("n", "square_id", "x", "y", "angle")


def read_groups(path, only=None):
    groups = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            n = int(row["n"])
            if only is not None and n not in only:
                continue
            groups.setdefault(n, []).append({k: row[k] for k in COLS})
    for n, rows in groups.items():
        rows.sort(key=lambda r: int(r["square_id"]))
        assert len(rows) == n, (n, len(rows))
    return groups


def write_rows(path, groups):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS, lineterminator="\n")
        w.writeheader()
        for n in sorted(groups):
            w.writerows(groups[n])


def write_full(path, groups):
    assert sorted(groups) == list(range(1, 1001))
    with open(path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(("id", "n", "square_id", "x", "y", "angle"))
        i = 0
        for n in range(1, 1001):
            for r in groups[n]:
                w.writerow((i, n, r["square_id"], r["x"], r["y"], r["angle"]))
                i += 1


def both(n, rows):
    """(literal L, pandas L) as Decimals, or None if either transport is rejected."""
    try:
        lit = Decimal(core.certify(n, rows))
        pan = Decimal(core.certify(n, core.kaggle_roundtrip({n: rows})[n]))
    except ValueError:
        return None
    return lit, pan


def main():
    cmd = sys.argv[1]
    if cmd == "extract":
        ns = {int(v) for v in sys.argv[3].split(",")}
        write_rows(sys.argv[4], read_groups(sys.argv[2], ns))
    elif cmd == "splice":
        g = read_groups(sys.argv[2])
        g.update(read_groups(sys.argv[3]))
        write_full(sys.argv[4], g)
    elif cmd == "gate":
        cand = read_groups(sys.argv[3])
        lo, hi = (int(sys.argv[5]), int(sys.argv[6])) if len(sys.argv) > 6 else (200, 599)
        inc = read_groups(sys.argv[2], set(cand))
        keep, report = {}, {}
        for n in sorted(cand):
            if not lo <= n <= hi:
                report[n] = "out of range"
                continue
            old, new = both(n, inc[n]), both(n, cand[n])
            if new is None:
                report[n] = "rejected by official metric"
            elif new[0] < old[0] and new[1] < old[1]:
                keep[n] = cand[n]
                report[n] = dict(old_literal=str(old[0]), new_literal=str(new[0]),
                                 old_pandas=str(old[1]), new_pandas=str(new[1]))
            else:
                report[n] = f"not strictly lower (dL literal {old[0]-new[0]:.3e}, pandas {old[1]-new[1]:.3e})"
        write_rows(sys.argv[4], keep)
        print(json.dumps(report, indent=1))
        print(f"accepted {len(keep)}/{len(cand)}", file=sys.stderr)


if __name__ == "__main__":
    main()
