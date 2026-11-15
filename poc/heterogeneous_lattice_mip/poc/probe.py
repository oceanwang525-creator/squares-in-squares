#!/usr/bin/env python3
"""Mixed-orientation lattice candidate MILP probe."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from collections import Counter
from decimal import Decimal
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import mpmath as mp
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree

POC = Path(__file__).resolve().parents[1]
ROOT = POC.parents[1]
INCUMBENT = ROOT / "poc/pell_corrugation/data/border-scan/merged.csv"
sys.path.insert(0, str(ROOT / "poc/baseline/poc/src"))

from squares import core  # noqa: E402

EPS = 2e-9
GAP = 2.5e-8


def extent(angle: float) -> float:
    return 0.5 * (abs(math.cos(angle)) + abs(math.sin(angle)))


def read_incumbent_groups(path: Path) -> dict[int, list[dict[str, str]]]:
    groups: dict[int, list[dict[str, str]]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            groups.setdefault(int(row["n"]), []).append(row)
    return groups


def packing_side(rows: list[dict[str, str]]) -> float:
    widths = []
    heights = []
    lows = []
    for row in rows:
        x = float(row["x"])
        y = float(row["y"])
        a = float(row["angle"])
        h = extent(a)
        lows.append((x - h, y - h))
        widths.append(x + h)
        heights.append(y + h)
    xmin = min(q[0] for q in lows)
    ymin = min(q[1] for q in lows)
    return max(max(widths) - xmin, max(heights) - ymin)


def add_block(
    out: list[tuple[float, float, float]],
    side: float,
    angle: float,
    u_count: int,
    v_count: int,
    corner: str,
    phase: tuple[float, float],
) -> None:
    h = extent(angle)
    c = math.cos(angle)
    s = math.sin(angle)
    step = 1.0 + GAP
    local_x = [
        step * (i * c - j * s)
        for i in (0, u_count - 1)
        for j in (0, v_count - 1)
    ]
    local_y = [
        step * (i * s + j * c)
        for i in (0, u_count - 1)
        for j in (0, v_count - 1)
    ]
    anchor_x = h - min(local_x)
    anchor_y = h - min(local_y)
    pts = []
    for i in range(u_count):
        for j in range(v_count):
            x = anchor_x + step * (i * c - j * s)
            y = anchor_y + step * (i * s + j * c)
            pts.append((x, y))

    # The un-flipped block is corner-anchored at the bottom-left.  Reflect it
    # through the box for the other three corners and reverse its ordering so
    # the rows are still generated from a wall.
    transformed = []
    for x, y in pts:
        if corner in ("tl", "br"):
            x = side - x
        if corner in ("tr", "br"):
            y = side - y
        x += phase[0]
        y += phase[1]
        transformed.append((x, y))

    for x, y in transformed:
        if x - h < -1e-8 or y - h < -1e-8 or x + h > side + 1e-8 or y + h > side + 1e-8:
            return
        out.append((x, y, angle))


def upright_candidates(
    side: float, fractions: list[float]
) -> list[tuple[float, float, float]]:
    k = int(math.floor(side + 1e-9))
    frac = side - k
    offsets = sorted({0.0, frac / 2.0, frac, 0.5 * frac + 0.125 * frac})
    candidates: list[tuple[float, float, float]] = []
    for ox in offsets:
        for oy in offsets:
            for i in range(k):
                x = ox + (i + 0.5) * (1.0 + GAP)
                if x + 0.5 > side + 1e-9:
                    continue
                for j in range(k):
                    y = oy + (j + 0.5) * (1.0 + GAP)
                    if y + 0.5 <= side + 1e-9:
                        candidates.append((x, y, 0.0))
    return candidates


def dominant_angles(rows: list[dict[str, str]]) -> list[float]:
    folded = []
    for row in rows:
        angle = float(row["angle"]) % (math.pi / 2.0)
        if 0.025 < angle < math.pi / 2.0 - 0.025:
            folded.append(angle)
    bins = Counter(round(a / math.radians(1.5)) for a in folded)
    ranked = [rad * math.radians(1.5) for rad, _ in bins.most_common(5)]
    rational = [
        math.atan2(1, 2),
        math.atan2(1, 3),
        math.atan2(2, 3),
        math.atan2(3, 4),
        math.radians(24.1),
        math.radians(24.8),
        math.radians(33.7),
        math.radians(41.97),
        math.pi / 4.0,
    ]
    return ranked + rational


def build_candidate_pool(
    side: float,
    incumbent: list[dict[str, str]],
    rng: random.Random,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    candidates = upright_candidates(side, [])
    blocks = []
    angles = dominant_angles(incumbent)
    # Four independent local lattices leave a heterogeneous mix of boundary
    # ownership and phases while keeping the MILP tractable.
    for corner in ("bl", "br", "tl", "tr"):
        base_angle = rng.choice(angles)
        angle = base_angle + rng.uniform(-0.025, 0.025)
        sign = rng.choice((-1.0, 1.0))
        angle = sign * angle
        u = rng.randint(9, 18)
        v = rng.randint(9, 18)
        phase = (rng.uniform(0.0, 0.12), rng.uniform(0.0, 0.12))
        add_block(candidates, side, angle, u, v, corner, phase)
        blocks.append(
            dict(angle=angle, u=u, v=v, corner=corner, phase=phase)
        )
    # One broad, low-angle block makes the result capable of selecting an
    # oblique middle seam rather than only four corner conventions.
    mid_angle = rng.choice(angles) * rng.choice((-1.0, 1.0))
    add_block(
        candidates,
        side,
        mid_angle,
        rng.randint(14, 23),
        rng.randint(8, 15),
        "bl",
        (rng.uniform(0.35, 0.65) * side, rng.uniform(0.0, 0.08)),
    )
    blocks.append(dict(angle=mid_angle, mid=True))
    arr = np.asarray(candidates, dtype=np.float64)
    h = 0.5 * (np.abs(np.cos(arr[:, 2])) + np.abs(np.sin(arr[:, 2])))
    keep = (
        (arr[:, 0] >= h - 1e-9)
        & (arr[:, 1] >= h - 1e-9)
        & (arr[:, 0] <= side - h + 1e-9)
        & (arr[:, 1] <= side - h + 1e-9)
    )
    arr = arr[keep]
    arr = np.unique(np.round(arr, 10), axis=0)
    return arr, blocks


def overlap_conflicts(candidates: np.ndarray) -> np.ndarray:
    tree = cKDTree(candidates[:, :2])
    pairs = tree.query_pairs(math.sqrt(2.0) + 2e-7, output_type="ndarray")
    if len(pairs) == 0:
        return np.empty((0, 2), dtype=np.int64)
    i, j = pairs[:, 0], pairs[:, 1]
    dx = candidates[i, 0] - candidates[j, 0]
    dy = candidates[i, 1] - candidates[j, 1]
    ai = candidates[i, 2]
    aj = candidates[j, 2]
    ci, si = np.cos(ai), np.sin(ai)
    p1 = np.abs(dx * ci + dy * si)
    p2 = np.abs(-dx * si + dy * ci)
    radius = 0.5 * (
        1.0 + np.abs(np.cos(ai - aj)) + np.abs(np.sin(ai - aj))
    )
    conflict = (p1 < radius - EPS) & (p2 < radius - EPS)
    return pairs[conflict]


def greedy_independent(candidates: np.ndarray, conflicts: np.ndarray) -> np.ndarray:
    n = len(candidates)
    adjacency = [set() for _ in range(n)]
    for i, j in conflicts:
        adjacency[int(i)].add(int(j))
        adjacency[int(j)].add(int(i))
    remaining = set(range(n))
    selected = []
    degree = [len(adjacency[i]) for i in range(n)]
    while remaining:
        # Select the least-conflicted surviving candidate.  This is a standard
        # fast independent-set heuristic and also supplies a lower bound for
        # the MILP probe.
        q = min(remaining, key=lambda idx: (degree[idx], idx))
        selected.append(q)
        remaining.discard(q)
        for neighbor in adjacency[q]:
            remaining.discard(neighbor)
        for idx in remaining:
            degree[idx] = sum(neighbor in remaining for neighbor in adjacency[idx])
    return np.asarray(selected, dtype=np.int64)


def fast_greedy(
    candidates: np.ndarray, conflicts: np.ndarray
) -> np.ndarray:
    """One-pass minimum-degree greedy used to triage heterogeneous block pairs."""
    n = len(candidates)
    adjacency = [set() for _ in range(n)]
    for i, j in conflicts:
        adjacency[int(i)].add(int(j))
        adjacency[int(j)].add(int(i))
    order = sorted(range(n), key=lambda idx: (len(adjacency[idx]), idx))
    banned = np.zeros(n, dtype=bool)
    selected = []
    for idx in order:
        if banned[idx]:
            continue
        selected.append(idx)
        for neighbor in adjacency[idx]:
            banned[neighbor] = True
    return np.asarray(selected, dtype=np.int64)


def block_squares(
    side: float,
    angle: float,
    u_count: int,
    v_count: int,
    corner: str,
    phase: tuple[float, float],
) -> np.ndarray:
    out: list[tuple[float, float, float]] = []
    add_block(out, side, angle, u_count, v_count, corner, phase)
    if len(out) != u_count * v_count:
        return np.empty((0, 3), dtype=np.float64)
    arr = np.asarray(out, dtype=np.float64)
    if len(overlap_conflicts(arr)):
        return np.empty((0, 3), dtype=np.float64)
    return arr


def block_bank(
    side: float, incumbent: list[dict[str, str]], rng: random.Random
) -> list[np.ndarray]:
    angles = dominant_angles(incumbent)
    bank = []
    for corner in ("bl", "br", "tl", "tr"):
        for angle in angles[:8]:
            for magnitude in (8, 11, 14, 17):
                for aspect in (-1, 0, 1):
                    u = magnitude
                    v = max(4, magnitude + aspect)
                    block = block_squares(
                        side,
                        angle,
                        u,
                        v,
                        corner,
                        (rng.uniform(0.0, 0.06), rng.uniform(0.0, 0.06)),
                    )
                    if len(block):
                        bank.append(block)
    return bank


def filter_upright(
    side: float, fixed: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    frac = side - math.floor(side)
    ups = np.asarray(upright_candidates(side, []), dtype=np.float64)
    # Keep the wall, mid-fraction, and far-fraction grids only.  The candidate
    # generator intentionally emits a few more phases for the global MIS test.
    base = np.mod(ups[:, :2] - 0.5, 1.0 + GAP)
    keep = (
        (np.linalg.norm(base - 0.0, axis=1) < 1e-7)
        | (np.linalg.norm(base - frac / 2.0, axis=1) < 1e-7)
        | (np.linalg.norm(base - frac, axis=1) < 1e-7)
    )
    ups = ups[keep]
    if len(fixed) and len(ups):
        h_fixed = 0.5 * (
            np.abs(np.cos(fixed[:, 2])) + np.abs(np.sin(fixed[:, 2]))
        )
        dx = ups[:, None, 0] - fixed[None, :, 0]
        dy = ups[:, None, 1] - fixed[None, :, 1]
        reach = 0.5 + h_fixed[None, :] + 2e-7
        near = (np.abs(dx) < reach) & (np.abs(dy) < reach)
        blocked = np.zeros(len(ups), dtype=bool)
        rows, cols = np.where(near)
        if len(rows):
            ai = ups[rows, 2]
            aj = fixed[cols, 2]
            ci, si = np.cos(ai), np.sin(ai)
            p1 = np.abs(dx[rows, cols] * ci + dy[rows, cols] * si)
            p2 = np.abs(-dx[rows, cols] * si + dy[rows, cols] * ci)
            radius = 0.5 * (
                1.0
                + np.abs(np.cos(ai - aj))
                + np.abs(np.sin(ai - aj))
            )
            hit = (p1 < radius - EPS) & (p2 < radius - EPS)
            blocked[rows[hit]] = True
        ups = ups[~blocked]
    if not len(ups):
        return ups, np.empty((0, 2), dtype=np.int64)
    return ups, overlap_conflicts(ups)


def blocks_nonoverlap(blocks: list[np.ndarray]) -> bool:
    for i in range(len(blocks)):
        for j in range(i + 1, len(blocks)):
            together = np.vstack((blocks[i], blocks[j]))
            if len(overlap_conflicts(together)):
                return False
    return True


def heterogeneous_block_probe(
    side: float,
    n: int,
    rows: list[dict[str, str]],
    incumbent_side: Decimal,
    seconds: float,
    rng: random.Random,
) -> dict[str, object]:
    bank = block_bank(side, rows, rng)
    started = time.time()
    attempts = 0
    best = dict(count=-1, grid=0, blocks=0)
    while time.time() - started < seconds and bank:
        attempts += 1
        count = rng.choices([1, 2, 3, 4], weights=[20, 45, 25, 10])[0]
        blocks = rng.sample(bank, min(count, len(bank)))
        if not blocks_nonoverlap(blocks):
            continue
        fixed = np.vstack(blocks)
        ups, grid_conflicts = filter_upright(side, fixed)
        chosen = fast_greedy(ups, grid_conflicts)
        total = len(fixed) + len(chosen)
        if total > best["count"]:
            best = dict(count=total, grid=len(chosen), blocks=len(fixed))
        if attempts % 25 == 0:
            print(
                f"  block-probe n={n} tries={attempts} best={best['count']} "
                f"blocks={best['blocks']} grid={best['grid']} pool={len(ups)}",
                flush=True,
            )
        if total < n:
            continue
        mip_count, exact_sel, _ = solve_milp(
            ups, grid_conflicts, min(8.0, max(0.5, seconds / 5.0))
        )
        exact_total = len(fixed) + mip_count
        if exact_total > best["count"]:
            best = dict(count=exact_total, grid=mip_count, blocks=len(fixed))
        if exact_total >= n:
            combined = np.vstack((fixed, ups[exact_sel]))
            selected_rows = rows_from_selection(
                combined, np.arange(len(combined), dtype=np.int64), n
            )
            cert = dual_certify(n, selected_rows, incumbent_side)
            if cert["strict_win"]:
                return dict(
                    win=selected_rows,
                    cert=cert,
                    attempts=attempts,
                    best=best,
                    elapsed=time.time() - started,
                )
    return dict(
        win=None,
        attempts=attempts,
        best=best,
        elapsed=time.time() - started,
    )


def solve_milp(
    candidates: np.ndarray, conflicts: np.ndarray, seconds: float
) -> tuple[int, np.ndarray, str]:
    if len(conflicts) == 0:
        selected = np.arange(len(candidates), dtype=np.int64)
        return len(selected), selected, "no conflicts"
    rows = np.repeat(np.arange(len(conflicts)), 2)
    cols = conflicts.ravel()
    matrix = coo_matrix(
        (np.ones(len(rows)), (rows, cols)),
        shape=(len(conflicts), len(candidates)),
    ).tocsr()
    result = milp(
        -np.ones(len(candidates)),
        constraints=LinearConstraint(matrix, -np.inf, 1.0),
        integrality=np.ones(len(candidates)),
        bounds=Bounds(0.0, 1.0),
        options=dict(time_limit=max(0.1, seconds), disp=False),
    )
    if result.x is None:
        return 0, np.empty(0, dtype=np.int64), result.message
    selected = np.flatnonzero(result.x > 0.5)
    return len(selected), selected, result.message


def verify_local(candidates: np.ndarray, selected: np.ndarray, side: float) -> bool:
    if len(selected) == 0:
        return False
    chosen = candidates[selected]
    h = 0.5 * (np.abs(np.cos(chosen[:, 2])) + np.abs(np.sin(chosen[:, 2])))
    if (
        (chosen[:, 0] - h < -1e-8).any()
        or (chosen[:, 1] - h < -1e-8).any()
        or (chosen[:, 0] + h > side + 1e-8).any()
        or (chosen[:, 1] + h > side + 1e-8).any()
    ):
        return False
    pairs = overlap_conflicts(chosen)
    return len(pairs) == 0


def rows_from_selection(
    candidates: np.ndarray, selected: np.ndarray, n: int
) -> list[dict[str, str]]:
    chosen = candidates[selected]
    chosen = chosen[np.lexsort((chosen[:, 0], chosen[:, 1], chosen[:, 2]))][:n]
    return [
        dict(
            square_id=str(i),
            x=repr(float(row[0])),
            y=repr(float(row[1])),
            angle=repr(float(row[2])),
        )
        for i, row in enumerate(chosen)
    ]


def dual_certify(
    n: int, rows: list[dict[str, str]], incumbent_side: Decimal
) -> dict[str, object]:
    try:
        literal = core.certify(n, rows)
        transported = core.kaggle_roundtrip({n: rows})[n]
        pandas = core.certify(n, transported)
        win = Decimal(literal) < incumbent_side and Decimal(pandas) < incumbent_side
        return dict(
            ok=True,
            literal=literal,
            pandas=pandas,
            strict_win=win,
            reason="ok",
        )
    except Exception as exc:
        return dict(ok=False, strict_win=False, reason=f"{type(exc).__name__}: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", default="803,791,908")
    parser.add_argument("--seconds", type=float, default=900.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--delta", type=float, default=0.0002)
    args = parser.parse_args()
    targets = [int(x) for x in args.targets.split(",") if x]
    groups = read_incumbent_groups(INCUMBENT)
    results: dict[str, object] = {
        "targets": targets,
        "seed": args.seed,
        "delta": args.delta,
        "started_unix": time.time(),
        "attempts": {},
    }
    won = False
    wins = {}
    for n in targets:
        rows = groups[n]
        incumbent_side = Decimal(core.certify(n, rows))
        side = float(incumbent_side) - args.delta
        rng = random.Random(args.seed + 1009 * n)
        began = time.time()
        block_result = heterogeneous_block_probe(
            side, n, rows, incumbent_side, args.seconds, rng
        )
        if block_result["win"] is not None:
            won = True
            wins[str(n)] = block_result["win"]
            results["attempts"][str(n)] = dict(
                winner=block_result["cert"],
                incumbent=incumbent_side,
                side=side,
                attempts=block_result["attempts"],
                best=block_result["best"],
                elapsed=block_result["elapsed"],
            )
            continue
        print(
            f"n={n} heterogeneous-block miss attempts={block_result['attempts']} "
            f"best={block_result['best']} elapsed={block_result['elapsed']:.1f}s",
            flush=True,
        )
        best_count = -1
        best_sel = np.empty(0, dtype=np.int64)
        best_pool = None
        best_blocks = []
        attempt = 0
        while time.time() - began < max(20.0, args.seconds / 3.0):
            attempt += 1
            pool, blocks = build_candidate_pool(side, rows, rng)
            conflicts = overlap_conflicts(pool)
            greedy = greedy_independent(pool, conflicts)
            if len(greedy) > best_count:
                best_count, best_sel, best_pool, best_blocks = (
                    len(greedy),
                    greedy,
                    pool,
                    blocks,
                )
            remaining = args.seconds - (time.time() - began)
            if len(greedy) >= n:
                mip_seconds = min(25.0, max(1.0, remaining / 3.0))
            else:
                mip_seconds = min(8.0, max(0.5, remaining / 6.0))
            count, selected, message = solve_milp(pool, conflicts, mip_seconds)
            if count > best_count:
                best_count, best_sel, best_pool, best_blocks = (
                    count,
                    selected,
                    pool,
                    blocks,
                )
            print(
                f"n={n} side={side:.9f} attempt={attempt} pool={len(pool)} "
                f"conflicts={len(conflicts)} greedy={len(greedy)} milp={count} "
                f"best={best_count} elapsed={time.time()-began:.1f}s",
                flush=True,
            )
            if best_count >= n:
                candidate_rows = rows_from_selection(best_pool, best_sel, n)
                cert = dual_certify(n, candidate_rows, incumbent_side)
                if cert["strict_win"]:
                    won = True
                    wins[str(n)] = candidate_rows
                    results["attempts"][str(n)] = dict(
                        winner=cert,
                        incumbent=incumbent_side,
                        side=side,
                        attempts=attempt,
                        blocks=best_blocks,
                    )
                    break
            if remaining <= 0:
                break
        if str(n) not in results["attempts"]:
            results["attempts"][str(n)] = dict(
                incumbent=incumbent_side,
                side=side,
                best_count=best_count,
                attempts=attempt,
                blocks=best_blocks,
                block_probe=block_result["best"],
                block_probe_attempts=block_result["attempts"],
                block_probe_elapsed=block_result["elapsed"],
                win=False,
            )
    results["finished_unix"] = time.time()
    results["won"] = won

    final_path = POC / "final.csv"
    if wins:
        with final_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["id", "x", "y", "deg"])
            with mp.workdps(95):
                for n, rows in sorted(wins.items()):
                    for row in sorted(rows, key=lambda q: int(q["square_id"])):
                        writer.writerow(
                            [
                                f"{int(n):03}_{row['square_id']}",
                                "s" + row["x"],
                                "s" + row["y"],
                                "s" + mp.nstr(mp.mpf(row["angle"]) * 180 / mp.pi, 75),
                            ]
                        )
    else:
        final_path.write_text("id,x,y,deg\n", encoding="utf-8")

    docs = POC / "docs"
    docs.mkdir(exist_ok=True)
    (docs / "results.json").write_text(
        json.dumps(results, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print("RESULT", "WIN" if won else "MISS", json.dumps(results, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
