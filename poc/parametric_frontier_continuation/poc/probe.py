#!/usr/bin/env python3
"""Screen a shared rational-band family across adjacent K.

This is a capacity model, not a certification path.  The template is the
existing two-upright-staircase plus rational tilted-band construction.  The
new question is whether one (p,q,W) tuple succeeds on several adjacent K.
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from math import gcd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CANON = ROOT / "poc/pell_corrugation/data/border-scan/merged.csv"
OUT = ROOT / "poc/parametric_frontier_continuation/docs/probe.json"


def incumbent_sides() -> dict[int, float]:
    boxes: dict[int, list[float]] = {}
    with CANON.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            n = int(row["n"])
            x, y, angle = (float(row[k]) for k in ("x", "y", "angle"))
            extent = (abs(math.cos(angle)) + abs(math.sin(angle))) / 2
            box = boxes.setdefault(n, [math.inf, math.inf, -math.inf, -math.inf])
            box[0] = min(box[0], x - extent)
            box[1] = min(box[1], y - extent)
            box[2] = max(box[2], x + extent)
            box[3] = max(box[3], y + extent)
    return {
        n: max(box[2] - box[0], box[3] - box[1])
        for n, box in boxes.items()
    }


def upright_count(p: int, q: int, limit: int, side: int) -> int:
    total = 0
    for j in range(side):
        remaining = limit - q * j
        if remaining < 0:
            break
        total += min(side, remaining // p + 1)
    return total


def band_row_count(p: int, q: int, offset: float, side: float) -> int:
    norm = math.hypot(p, q)
    half = (p + q) / (2 * norm)
    x_lo = max(half, (offset - q * (side - half)) / p)
    x_hi = min(side - half, (offset - q * half) / p)
    if x_hi < x_lo:
        return 0
    return math.floor((x_hi - x_lo) * norm / q + 1e-10) + 1


def best_side(
    p: int, q: int, width: int, target: int, incumbent: float
) -> tuple[int, float, int, int]:
    norm = math.hypot(p, q)
    period = p + q
    # L = 2 + (T1+T2+W*norm)/period. Explore every possible L grid step
    # between the current side and a conservative lower bound.
    s_min = target
    s_max = int(math.ceil((incumbent - 2.0) * period - width * norm))
    best: tuple[int, float, int, int] | None = None
    for total in range(max(0, s_max - 2 * period), s_max):
        side = 2.0 + (total + width * norm) / period
        if not (math.isqrt(target) + 0.48 <= side < incumbent - 1e-10):
            continue
        side_floor = int(math.floor(side + 1e-10))
        for t1 in range(total + 1):
            t2 = total - t1
            count = (
                upright_count(p, q, t1, side_floor)
                + upright_count(p, q, t2, side_floor)
            )
            for row in range(width):
                offset = t1 + period + (row + 0.5) * norm
                count += band_row_count(p, q, offset, side)
            if best is None or count > best[0] or (
                count == best[0] and side < best[1]
            ):
                best = (count, side, t1, t2)
    return best if best is not None else (0, incumbent, 0, 0)


def main() -> None:
    sides = incumbent_sides()
    rows: dict[str, dict] = {}
    families = ((7, range(24, 32), 18), (8, range(24, 32), 18),
                (4, range(18, 20), 31))
    for d, ks, max_factor in families:
        for k in ks:
            n = k * k + d
            if n > 1000:
                continue
            target = n
            incumbent = sides[n]
            records = []
            for p in range(1, max_factor):
                for q in range(p, max_factor):
                    if gcd(p, q) != 1:
                        continue
                    for width in range(1, 7):
                        count, side, t1, t2 = best_side(
                            p, q, width, target, incumbent
                        )
                        records.append(
                            {
                                "p": p,
                                "q": q,
                                "width": width,
                                "count": count,
                                "deficit": target - count,
                                "L": side,
                                "dL": side - incumbent,
                                "T1": t1,
                                "T2": t2,
                            }
                        )
            records.sort(key=lambda item: (item["deficit"], item["L"]))
            hits = sum(item["deficit"] <= 0 for item in records)
            rows[str(n)] = {
                "K": k,
                "D": d,
                "incumbent_L": incumbent,
                "hits": hits,
                "best": records[:20],
            }
            print(
                f"D={d} K={k} n={n} hits={hits} "
                f"best_count={records[0]['count']} "
                f"deficit={records[0]['deficit']}",
                flush=True,
            )

    tuple_sets: dict[tuple[int, int, int], set[int]] = defaultdict(set)
    for key, record in rows.items():
        for item in record["best"]:
            if item["deficit"] <= 0:
                tuple_sets[(item["p"], item["q"], item["width"])].add(int(key))
    shared = []
    for params, hits in tuple_sets.items():
        if len(hits) >= 2:
            shared.append({"params": params, "hits": sorted(hits)})
    shared.sort(key=lambda item: (-len(item["hits"]), item["params"]))

    payload = {
        "method": "two-staircase rational-band capacity screening",
        "warning": "capacity only; no candidate is a certified geometry",
        "rows": rows,
        "shared_tuple": shared[:80],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"shared tuples: {len(shared)}; wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
