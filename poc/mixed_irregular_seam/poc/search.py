#!/usr/bin/env python3
"""Interleaving donor crops with an exact upright refill."""

from __future__ import annotations

import argparse
import csv
import heapq
import importlib.util
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from decimal import Decimal
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
HIGHN = ROOT / "poc/highn-mondrian/poc"
sys.path.insert(0, str(HIGHN))
sys.path.insert(0, str(ROOT / "poc/baseline/poc/src"))

import fillfast  # type: ignore  # noqa: E402
import scan  # type: ignore  # noqa: E402
import seamscan  # type: ignore  # noqa: E402
from squares import core  # type: ignore  # noqa: E402


EPS = 1e-9
PHASES = (0.0, 0.125, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0)


def load_first_party_final_text():
    path = ROOT / "poc/auto_union/poc/union.py"
    spec = importlib.util.spec_from_file_location("mixed_irregular_union", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.final_text


def transformed_shapes(crop):
    if crop in _SHAPE_CACHE:
        return _SHAPE_CACHE[crop]
    rows = SEARCH_PACKINGS[crop.donor_n]
    selected = [rows[index] for index in crop.indices]

    def off_axis(row):
        angle = float(row["angle"])
        quarter = round(angle / (math.pi / 2.0))
        return abs(angle - quarter * math.pi / 2.0) > 1e-5

    if crop.corner != "pocket":
        tilted = [row for row in selected if off_axis(row)]
        if len(tilted) >= 4:
            selected = tilted

    vals = []
    for row in selected:
        x, y, angle = float(row["x"]), float(row["y"]), float(row["angle"])
        extent = 0.5 * (abs(math.cos(angle)) + abs(math.sin(angle)))
        vals.append((x, y, angle, extent))
    x0 = min(x - extent for x, _, _, extent in vals)
    y0 = min(y - extent for _, y, _, extent in vals)
    x1 = max(x + extent for x, _, _, extent in vals)
    y1 = max(y + extent for _, y, _, extent in vals)
    width, height = x1 - x0, y1 - y0
    piece = [(x - x0, y - y0, angle) for x, y, angle, _ in vals]
    out = []
    seen = set()
    for mirror in (False, True):
        for rotations in range(4):
            w, h, shape = seamscan.transformed_piece(
                width, height, piece, mirror, rotations
            )
            key = tuple(
                (round(x, 8), round(y, 8), round(a, 8))
                for x, y, a in sorted(shape)
            )
            if key in seen:
                continue
            seen.add(key)
            out.append((float(w), float(h), shape, key))
    _SHAPE_CACHE[crop] = out
    return out


def touched_cells(side, x0, y0, width, height):
    cells = set()
    first_x = max(0, math.floor(x0 + EPS))
    last_x = min(math.floor(side + EPS) - 1, math.ceil(x0 + width - EPS) - 1)
    first_y = max(0, math.floor(y0 + EPS))
    last_y = min(math.floor(side + EPS) - 1, math.ceil(y0 + height - EPS) - 1)
    for i in range(first_x, last_x + 1):
        for j in range(first_y, last_y + 1):
            cells.add((i, j))
    return cells


def square_penetration(a, b):
    """Boolean SAT overlap with exact touching allowed."""
    ax, ay, aa = a
    bx, by, ba = b
    dx, dy = bx - ax, by - ay
    ra = 0.5 * (abs(math.cos(aa)) + abs(math.sin(aa)))
    rb = 0.5 * (abs(math.cos(ba)) + abs(math.sin(ba)))
    if abs(dx) >= ra + rb - 1e-10 or abs(dy) >= ra + rb - 1e-10:
        return False
    ux, uy = math.cos(aa), math.sin(aa)
    vx, vy = -ux, uy
    da = dx * ux + dy * uy
    db = dx * vx + dy * vy
    if abs(da) >= ra + rb - 1e-10 or abs(db) >= ra + rb - 1e-10:
        return False
    ux, uy = math.cos(ba), math.sin(ba)
    vx, vy = -ux, uy
    da = dx * ux + dy * uy
    db = dx * vx + dy * vy
    return not (
        abs(da) >= ra + rb - 1e-10
        or abs(db) >= ra + rb - 1e-10
    )


def shapes_overlap(place_a, place_b):
    for a in place_a:
        for b in place_b:
            if square_penetration(a, b):
                return True
    return False


def translate(shape, x0, y0):
    return [(x + x0, y + y0, angle) for x, y, angle in shape]


def candidate_phases(width, height, side):
    mx, my = max(0.0, side - width), max(0.0, side - height)
    xs = [p for p in PHASES if p <= mx + EPS]
    ys = [p for p in PHASES if p <= my + EPS]
    if mx not in xs:
        xs.append(mx)
    if my not in ys:
        ys.append(my)
    return xs, ys


def solve_pair(task):
    (
        target,
        side,
        margin,
        milp_seconds,
        crop_a,
        crop_b,
    ) = task
    K = math.floor(side + EPS)
    variants_a = transformed_shapes(crop_a)
    variants_b = (
        transformed_shapes(crop_b)
        if crop_b is not None
        else [(0.0, 0.0, [], ())]
    )
    best_capacity = 0
    best_record = None
    max_bound = -1
    tried = 0

    for wa, ha, shape_a, key_a in variants_a:
        for wb, hb, shape_b, key_b in variants_b:
            if wa > side + EPS or ha > side + EPS or wb > side + EPS or hb > side + EPS:
                continue
            phases_ax, phases_ay = candidate_phases(wa, ha, side)
            phases_bx, phases_by = candidate_phases(wb, hb, side)
            local = []
            for ax in phases_ax:
                for ay in phases_ay:
                    cells_a = touched_cells(side, ax, ay, wa, ha)
                    place_a = translate(shape_a, ax, ay)
                    for bx in phases_bx:
                        for by in phases_by:
                            xb, yb = side - wb - bx, side - hb - by
                            cells = cells_a | touched_cells(side, xb, yb, wb, hb)
                            bound = K * K + len(shape_a) + len(shape_b) - len(cells)
                            max_bound = max(max_bound, int(bound))
                            if bound < target:
                                continue
                            local.append((bound, ax, ay, xb, yb, place_a, shape_b))
            local.sort(key=lambda item: item[0], reverse=True)

            for bound, ax, ay, xb, yb, place_a, shape_b in local[:8]:
                place_b = translate(shape_b, xb, yb)
                if shapes_overlap(place_a, place_b):
                    continue
                tried += 1
                fixed = np.asarray(place_a + place_b, dtype=float)
                filled = fillfast.fill(
                    side, fixed, mode="exact", time_limit=milp_seconds
                )
                capacity = int(len(filled))
                if capacity > best_capacity:
                    best_capacity = capacity
                    best_record = {
                        "capacity": capacity,
                        "bound": int(bound),
                        "short": int(target - capacity),
                        "donor_a": crop_a.donor_n,
                        "donor_b": None if crop_b is None else crop_b.donor_n,
                        "crop_a": [crop_a.width, crop_a.height, crop_a.capacity],
                        "crop_b": (
                            None
                            if crop_b is None
                            else [crop_b.width, crop_b.height, crop_b.capacity]
                        ),
                        "variant_a": key_a,
                        "variant_b": key_b,
                        "placement": [float(ax), float(ay), float(xb), float(yb)],
                        "fixed": int(len(fixed)),
                    }
                if capacity >= target:
                    rows = [
                        {
                            "square_id": str(index),
                            "x": format(float(x), ".17g"),
                            "y": format(float(y), ".17g"),
                            "angle": format(float(angle), ".17g"),
                        }
                        for index, (x, y, angle) in enumerate(filled[:target])
                    ]
                    literal, pandas_side = scan.certify_both(rows)
                    win = (
                        Decimal(literal) < Decimal(side)
                        and Decimal(pandas_side) < Decimal(side)
                    )
                    best_record.update(
                        certified_literal_L=literal,
                        certified_pandas_L=pandas_side,
                        strict_win=win,
                    )
                    if win:
                        return {
                            "target": target,
                            "win": True,
                            "rows": rows,
                            "record": best_record,
                            "tried": tried,
                        }
    return {
        "target": target,
        "win": False,
        "rows": None,
        "record": best_record,
        "max_bound": int(max_bound),
        "tried": tried,
        "margin": margin,
    }


def build_pairs(packings, target, side, keep, pair_limit, min_crop, max_crop):
    bottom, top = seamscan.top_crops(
        packings, side, min_crop, max_crop, keep
    )
    heap = []
    serial = 0
    for gain_a, crop_a in bottom:
        for gain_b, crop_b in top:
            bound = gain_a + gain_b
            item = (int(bound), serial, crop_a, crop_b)
            serial += 1
            if len(heap) < pair_limit:
                heapq.heappush(heap, item)
            elif bound > heap[0][0]:
                heapq.heapreplace(heap, item)
    return [
        (crop_a, crop_b)
        for _, _, crop_a, crop_b in sorted(heap, reverse=True)
    ]


def load_pocket_crops(packings, path, target, side, keep, max_side):
    raw = json.loads(path.read_text(encoding="utf-8"))
    crops = []
    seen = set()
    for pocket in raw:
        donor_n = int(pocket["src"])
        if donor_n < target or donor_n not in packings:
            continue
        rows = packings[donor_n]
        x0 = float(pocket["x0"]) - 1e-8
        x1 = float(pocket["x1"]) + 1e-8
        y0 = float(pocket["y0"]) - 1e-8
        y1 = float(pocket["y1"]) + 1e-8
        indices = []
        for index, row in enumerate(rows):
            angle = float(row["angle"])
            extent = 0.5 * (abs(math.cos(angle)) + abs(math.sin(angle)))
            x, y = float(row["x"]), float(row["y"])
            if (
                x - extent >= x0
                and x + extent <= x1
                and y - extent >= y0
                and y + extent <= y1
            ):
                indices.append(index)
        if len(indices) < 8:
            continue
        selected = [rows[index] for index in indices]
        boxes_min_x = []
        boxes_max_x = []
        boxes_min_y = []
        boxes_max_y = []
        for row in selected:
            angle = float(row["angle"])
            extent = 0.5 * (abs(math.cos(angle)) + abs(math.sin(angle)))
            x, y = float(row["x"]), float(row["y"])
            boxes_min_x.append(x - extent)
            boxes_max_x.append(x + extent)
            boxes_min_y.append(y - extent)
            boxes_max_y.append(y + extent)
        width = max(boxes_max_x) - min(boxes_min_x)
        height = max(boxes_max_y) - min(boxes_min_y)
        if width > min(side, max_side) + EPS or height > min(side, max_side) + EPS:
            continue
        key = (
            donor_n,
            len(indices),
            round(width, 8),
            round(height, 8),
            tuple(indices),
        )
        if key in seen:
            continue
        seen.add(key)
        crops.append(
            scan.Crop(
                donor_n=donor_n,
                corner="pocket",
                capacity=len(indices),
                width=float(width),
                height=float(height),
                cut=(
                    int(math.floor(min(boxes_min_x))),
                    int(math.ceil(max(boxes_max_x))),
                    int(math.floor(min(boxes_min_y))),
                    int(math.ceil(max(boxes_max_y))),
                ),
                indices=tuple(indices),
            )
        )

    def density_gain(crop):
        return crop.capacity - math.floor(crop.width + EPS) * math.floor(
            crop.height + EPS
        )

    crops.sort(key=lambda crop: (density_gain(crop), crop.capacity), reverse=True)
    return crops[:keep]


SEARCH_PACKINGS = {}
_SHAPE_CACHE = {}
_SCRIPT_MARGIN = "1e-9"


def init_worker(packings):
    global SEARCH_PACKINGS
    SEARCH_PACKINGS = packings
    _SHAPE_CACHE.clear()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--submission",
        type=Path,
        default=ROOT / "poc/pell_corrugation/data/border-scan/merged.csv",
    )
    parser.add_argument("--targets", default="367,791,969,974")
    parser.add_argument("--min-crop", type=int, default=4)
    parser.add_argument("--max-crop", type=int, default=18)
    parser.add_argument("--keep-crops", type=int, default=1600)
    parser.add_argument("--pair-limit", type=int, default=8000)
    parser.add_argument(
        "--pockets",
        type=Path,
        default=ROOT / "poc/pocket_lift_scan2/data/pockets_pareto3.json",
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--milp-seconds", type=int, default=3)
    parser.add_argument("--margin", default="1e-9")
    parser.add_argument(
        "--summary",
        type=Path,
        default=ROOT / "poc/mixed_irregular_seam/docs/quick.json",
    )
    parser.add_argument(
        "--final",
        type=Path,
        default=ROOT / "poc/mixed_irregular_seam/final.csv",
    )
    args = parser.parse_args()

    global SEARCH_PACKINGS
    SEARCH_PACKINGS = scan.read_groups(args.submission)
    targets = [int(value) for value in args.targets.split(",") if value]
    report = {"submission": str(args.submission), "targets": {}, "wins": {}}
    accepted = {}
    final_text = load_first_party_final_text()

    for target in targets:
        incumbent = Decimal(core.certify(target, SEARCH_PACKINGS[target]))
        side = float(incumbent - Decimal(args.margin))
        if args.pockets and args.pockets.exists():
            crops = load_pocket_crops(
                SEARCH_PACKINGS,
                args.pockets,
                target,
                side,
                args.keep_crops,
                args.max_crop,
            )
            pairs = []
            horizon = max(1, int(math.sqrt(args.pair_limit)))
            pairs.extend((crop, None) for crop in crops[:horizon])
            for crop_a in crops[:horizon]:
                for crop_b in crops[:horizon]:
                    pairs.append((crop_a, crop_b))
        else:
            pairs = build_pairs(
                SEARCH_PACKINGS,
                target,
                side,
                args.keep_crops,
                args.pair_limit,
                args.min_crop,
                args.max_crop,
            )
        tasks = [
            (target, side, args.margin, args.milp_seconds, crop_a, crop_b)
            for crop_a, crop_b in pairs
        ]
        needed = {
            donor_n: SEARCH_PACKINGS[donor_n]
            for task in tasks
            for donor_n in (
                task[4].donor_n,
                None if task[5] is None else task[5].donor_n,
            )
            if donor_n is not None
        }
        best = None
        best_bound = -1
        winner = None
        print(
            json.dumps(
                {
                    "target": target,
                    "incumbent": str(incumbent),
                    "probe_side": side,
                    "pairs": len(tasks),
                }
            ),
            flush=True,
        )
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=init_worker,
            initargs=(needed,),
        ) as pool:
            futures = [pool.submit(solve_pair, task) for task in tasks]
            for done, future in enumerate(as_completed(futures), 1):
                result = future.result()
                if result["record"] is not None:
                    if best is None or result["record"]["capacity"] > best["capacity"]:
                        best = dict(result["record"])
                        best["tried_in_pair"] = result["tried"]
                best_bound = max(best_bound, int(result.get("max_bound", -1)))
                if result["win"]:
                    winner = result
                    for pending in futures:
                        pending.cancel()
                    break
                if done % max(1, len(tasks) // 10) == 0:
                    print(
                        json.dumps(
                            {
                                "target": target,
                                "done": done,
                                "best_capacity": None if best is None else best["capacity"],
                                "best_bound": best_bound,
                            }
                        ),
                        flush=True,
                    )
        report["targets"][str(target)] = {
            "incumbent": str(incumbent),
            "probe_side": args.margin,
            "pairs": len(tasks),
            "best": best,
            "best_bound": best_bound,
            "win": winner is not None,
        }
        if winner is not None:
            accepted[target] = winner["rows"]
            report["targets"][str(target)]["winner"] = winner["record"]
            report["wins"][str(target)] = winner["record"]
            print(f"WIN {target}: {winner['record']}", flush=True)
        else:
            print(f"MISS {target}: best={best}", flush=True)

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    args.final.parent.mkdir(parents=True, exist_ok=True)
    args.final.write_text(final_text(accepted), encoding="utf-8")
    print(json.dumps({"wins": sorted(accepted), "groups": len(accepted)}), flush=True)


if __name__ == "__main__":
    main()
