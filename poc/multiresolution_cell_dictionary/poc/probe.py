#!/usr/bin/env python3
"""Exact multiresolution band-cell dictionary probe.

The search uses rigid local cells cut from in-repository packings.  A cell is
one horizontal band's complete, fully-contained square set, normalized to its
occupied bounding box.  Since candidate cells are placed into disjoint
vertical intervals, the coarse DP result is also an exact non-overlap
construction when each width fits the target box.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import io
import json
import math
import sys
import tempfile
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import mpmath as mp

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "poc/baseline/poc/src"))

from squares import _official_metric as metric  # noqa: E402
from squares.core import certify, read_submission  # noqa: E402


def load_union():
    path = ROOT / "poc/auto_union/poc/union.py"
    spec = importlib.util.spec_from_file_location("auto_union_probe", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


UNION = load_union()
BASE = ROOT / "poc/pell_corrugation/data/border-scan/merged.csv"


@dataclass
class Square:
    x: float
    y: float
    angle: float


@dataclass
class Row:
    count: int
    width: float
    height: float
    source: int
    transform: int
    band: int
    squares: list[Square]


def box(squares: list[Square]) -> tuple[float, float, float, float]:
    left = bottom = math.inf
    right = top = -math.inf
    for sq in squares:
        h = (abs(math.cos(sq.angle)) + abs(math.sin(sq.angle))) / 2.0
        left = min(left, sq.x - h)
        bottom = min(bottom, sq.y - h)
        right = max(right, sq.x + h)
        top = max(top, sq.y + h)
    return left, bottom, right, top


def incumbent_groups() -> tuple[dict[int, list[dict[str, str]]], dict[int, list[Square]]]:
    official = read_submission(BASE)
    parsed: dict[int, list[Square]] = {}
    for n, rows in official.items():
        parsed[n] = [
            Square(float(r["x"]), float(r["y"]), float(r["angle"]))
            for r in rows
        ]
    return official, parsed


def dihedral(squares: list[Square], side: float, code: int) -> list[Square]:
    """Apply reflection then 0/90/180/270 degree rotation about box centre."""
    reflected = code >= 4
    turns = code % 4
    out: list[Square] = []
    c = side / 2.0
    for sq in squares:
        x, y, a = sq.x, sq.y, sq.angle
        if reflected:
            x = side - x
            a = -a
        dx, dy = x - c, y - c
        if turns == 1:
            dx, dy = -dy, dx
        elif turns == 2:
            dx, dy = -dx, -dy
        elif turns == 3:
            dx, dy = dy, -dx
        out.append(Square(c + dx, c + dy, a + turns * math.pi / 2.0))
    return out


def band_cells(
    sources: dict[int, list[Square]],
    bands: int,
    target_side: float,
) -> list[dict[int, Row]]:
    """Return shortest representative cells for each (band, exact count)."""
    cells: list[dict[int, Row]] = [dict() for _ in range(bands)]
    eps = 2.0e-9
    for source_n, original in sources.items():
        if not 600 <= source_n <= 1000:
            continue
        left, bottom, right, top = box(original)
        source_side = max(right - left, top - bottom)
        if source_side <= 0:
            continue
        for transform in range(8):
            transformed = dihedral(original, source_side, transform)
            for band in range(bands):
                lo = band * source_side / bands
                hi = (band + 1) * source_side / bands
                selected = []
                for sq in transformed:
                    h = (abs(math.cos(sq.angle)) + abs(math.sin(sq.angle))) / 2.0
                    if sq.y - h >= lo - eps and sq.y + h <= hi + eps:
                        selected.append(sq)
                if not selected:
                    continue
                l, b, r, t = box(selected)
                width, height = r - l, t - b
                if width > target_side + eps or height > target_side + eps:
                    continue
                normalized = [
                    Square(sq.x - l, sq.y - b, sq.angle) for sq in selected
                ]
                row = Row(len(selected), width, height, source_n,
                          transform, band, normalized)
                old = cells[band].get(row.count)
                if old is None or (row.height, row.width) < (old.height - eps,
                                                              old.width - eps):
                    cells[band][row.count] = row
    return cells


def assemble(
    cells: list[dict[int, Row]],
    target_n: int,
    target_side: float,
) -> tuple[bool, float, int, list[Row]]:
    """DP on exact count -> minimum stacked height."""
    # state[count] = (height, path)
    state: dict[int, tuple[float, list[Row]]] = {0: (0.0, [])}
    for cells_for_band in cells:
        nxt: dict[int, tuple[float, list[Row]]] = {}
        for old_count, (old_height, path) in state.items():
            for count, row in cells_for_band.items():
                new_count = old_count + count
                if new_count > target_n + 160:
                    continue
                new_height = old_height + row.height
                if new_height > target_side:
                    continue
                old = nxt.get(new_count)
                if old is None or new_height < old[0] - 2.0e-12:
                    nxt[new_count] = (new_height, path + [row])
        state = nxt
        if not state:
            return False, 0.0, 0, []
    usable = [
        (height, count, path)
        for count, (height, path) in state.items()
        if count >= target_n and height <= target_side
    ]
    if usable:
        height, count, path = min(usable)
        return True, height, count, path
    # Keep the closest failed cover for a diagnostic: largest count, then
    # smallest stacked height.
    count, (height, path) = max(
        state.items(), key=lambda item: (item[0], -item[1][0])
    )
    return False, height, count, path


def realize(path: list[Row], target_n: int,
            target_side: float) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    y_offset = 0.0
    for row in path:
        x_offset = (target_side - row.width) / 2.0
        assert x_offset >= -1.0e-8
        for sq in row.squares:
            x = sq.x + max(0.0, x_offset)
            y = sq.y + y_offset
            rows.append({
                "square_id": str(len(rows)),
                "x": format(x, ".17g"),
                "y": format(y, ".17g"),
                "angle": mp.nstr(mp.mpf(sq.angle), 75),
            })
        y_offset += row.height
    if len(rows) < target_n:
        raise ValueError("assembled count unexpectedly short")
    rows = rows[:target_n]
    # Renumber only after truncation of surplus boundary cells.
    for i, row in enumerate(rows):
        row["square_id"] = str(i)
    return rows


def certified_sides(n: int, rows: list[dict[str, str]]) -> tuple[str, str]:
    text = UNION.final_text({n: rows})
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "final.csv"
        path.write_text(text, encoding="utf-8")
        literal, transported = UNION.load(path)
        if set(literal) != {n} or set(transported) != {n}:
            raise ValueError("candidate parse did not preserve its group")
        return (certify(n, literal[n]), certify(n, transported[n]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="803,791,908")
    ap.add_argument("--delta", type=float, default=1.0e-4)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--final", type=Path,
                    default=ROOT / "poc/multiresolution_cell_dictionary/final.csv")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.final.parent.mkdir(parents=True, exist_ok=True)

    official, sources = incumbent_groups()
    report = {
        "method": "band-cell dictionary + exact count/height DP",
        "incumbent": str(BASE.relative_to(ROOT)),
        "targets": [],
        "wins": {},
    }
    best_rows: dict[int, list[dict[str, str]]] = {}
    best_sides: dict[int, tuple[Decimal, Decimal]] = {}

    for raw_n in args.targets.split(","):
        n = int(raw_n)
        incumbent = official[n]
        literal_inc = Decimal(certify(n, incumbent))
        transported_inc = Decimal(certify(n, UNION.load(BASE)[1][n]))
        limit = min(literal_inc, transported_inc)
        target_side = float(limit) - args.delta
        target_entry = {
            "n": n,
            "incumbent_literal": str(literal_inc),
            "incumbent_transported": str(transported_inc),
            "target_side": target_side,
            "runs": [],
        }
        print("target", n, "inc", limit, "probe", target_side, flush=True)

        for bands in (8, 10, 12):
            cells = band_cells(sources, bands, target_side)
            counts = [{count: row.width for count, row in d.items()}
                      for d in cells]
            found, height, count, path = assemble(cells, n, target_side)
            run = {
                "bands": bands,
                "cell_counts_per_band": [len(d) for d in cells],
                "cell_widths": counts,
                "found": found,
                "best_count": count,
                "best_stacked_height": height,
            }
            if path:
                run["sources"] = [
                    dict(source=r.source, transform=r.transform,
                         band=r.band, count=r.count,
                         width=r.width, height=r.height)
                    for r in path
                ]
            if found:
                rows = realize(path, n, target_side)
                literal, transported = certified_sides(n, rows)
                run.update({
                    "literal_side": literal,
                    "transported_side": transported,
                })
                pair = (Decimal(literal), Decimal(transported))
                if pair[0] < literal_inc and pair[1] < transported_inc:
                    best_rows[n] = rows
                    best_sides[n] = pair
                    run["strict_win"] = True
                    report["wins"][str(n)] = {
                        "literal_side": literal,
                        "pandas_side": transported,
                    }
                else:
                    run["strict_win"] = False
            target_entry["runs"].append(run)
            print(" ", bands, "bands found", found,
                  "count", count, "height", height, flush=True)
        report["targets"].append(target_entry)

    if best_rows:
        args.final.write_text(UNION.final_text(best_rows), encoding="utf-8")
    else:
        args.final.write_text("id,x,y,deg\n", encoding="utf-8")
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"wins": report["wins"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
