#!/usr/bin/env python3
"""Low-width contact-braid reset with gentle masked ILS and certified polish."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import multiprocessing as mp
import os
import sys
import time
from collections import deque
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
POC = Path(__file__).resolve().parents[1]
ANGLE_POC = ROOT / "poc" / "swarm_angles" / "poc"
AUTO_UNION = ROOT / "poc" / "auto_union" / "poc" / "union.py"
BASE = ROOT / "poc" / "pell_corrugation" / "data" / "border-scan" / "merged.csv"
GAP = ROOT / "poc" / "frontier_nov12" / "docs" / "per_n_gap.csv"
FINAL = POC / "final.csv"
DOCS = POC / "docs"

if str(ANGLE_POC) not in sys.path:
    sys.path.insert(0, str(ANGLE_POC))

import idea21_seam_cross as i21  # noqa: E402
from idea19_run import cpp_expand_sym, full_to_free_sym, run_cpp_batch_ils  # noqa: E402
from idea18_run import get_cpp_lib  # noqa: E402


def load_union():
    spec = importlib.util.spec_from_file_location("contactbraid_union", AUTO_UNION)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_group(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    frame = pd.read_csv(BASE)
    sub = frame[frame["n"] == n]
    if len(sub) != n:
        raise ValueError(f"expected {n} rows for n={n}, found {len(sub)}")
    return i21._norm(sub)


def shortest_wall_path(
    x: np.ndarray, y: np.ndarray, th: np.ndarray, side: float, direction: str
) -> list[int]:
    n = len(x)
    h = 0.5 * (np.abs(np.cos(th)) + np.abs(np.sin(th)))
    adj: list[list[int]] = [[] for _ in range(n)]
    c, s = np.cos(th), np.sin(th)
    for i in range(n):
        for j in range(i + 1, n):
            dx, dy = x[j] - x[i], y[j] - y[i]
            if abs(dx) > 1.5 or abs(dy) > 1.5:
                continue
            best = -math.inf
            for ax, ay in (
                (c[i], s[i]), (-s[i], c[i]), (c[j], s[j]), (-s[j], c[j])
            ):
                projection = dx * ax + dy * ay
                radius = 0.5 * (
                    1.0 + abs(ax * c[i] + ay * s[i]) + abs(ax * c[j] + ay * s[j])
                )
                best = max(best, abs(projection) - radius)
            if best <= 2e-6:
                adj[i].append(j)
                adj[j].append(i)

    if direction == "lr":
        starts = [i for i in range(n) if x[i] - h[i] <= 1e-5]
        done = lambda i: side - x[i] - h[i] <= 1e-5
    else:
        starts = [i for i in range(n) if y[i] - h[i] <= 1e-5]
        done = lambda i: side - y[i] - h[i] <= 1e-5
    if not starts:
        return []
    previous = {i: None for i in starts}
    queue = deque(starts)
    while queue:
        i = queue.popleft()
        if done(i):
            path = []
            while i is not None:
                path.append(i)
                i = previous[i]
            return path[::-1]
        for j in adj[i]:
            if j not in previous:
                previous[j] = i
                queue.append(j)
    return []


def braid_seeds(
    n: int,
    x0: np.ndarray,
    y0: np.ndarray,
    th0: np.ndarray,
    side: float,
    target_side: float | None,
    quick: bool,
) -> tuple[list[np.ndarray], list[list[float]], list[str], dict]:
    rng = np.random.default_rng(778_000 + n)
    paths = {
        "lr": shortest_wall_path(x0, y0, th0, side, "lr"),
        "bt": shortest_wall_path(x0, y0, th0, side, "bt"),
    }
    directions = [d for d in ("lr", "bt") if paths[d]]
    if not directions:
        raise ValueError("no wall-to-wall contact path")

    # The frontier table supplies only a side-side diagnostic; it is never used
    # to build coordinates. Aim the continuation just below the incumbent so a
    # candidate must pass the unchanged certification before export.
    floor = max(side - 0.008, (target_side + 2e-4) if target_side else side - 0.008)
    schedules: list[list[float]] = []
    qseeds: list[np.ndarray] = []
    labels: list[str] = []

    configs = [
        ("thin", 3.1, 0.006, 0.035, 0.045, 0.94),
        ("mid", 4.7, 0.016, 0.105, 0.160, 0.94),
        ("wide", 6.4, 0.030, 0.210, 0.310, 0.94),
        ("relax", 4.1, 0.045, 0.150, 0.230, 0.90),
    ] if not quick else [
        ("thin", 3.1, 0.006, 0.035, 0.045, 0.94),
        ("mid", 4.7, 0.016, 0.105, 0.160, 0.94),
        ("relax", 4.1, 0.045, 0.150, 0.230, 0.90),
    ]

    for direction in directions:
        path = paths[direction]
        px, py = x0[path], y0[path]
        for name, radius, slack, jitter, angle, astart in configs:
            variants = 2 if quick else 3
            for variant in range(variants):
                l0 = side + slack
                scale = l0 / side
                x, y, th = x0 * scale, y0 * scale, th0.copy()
                dist2 = (x[:, None] - px[None, :]) ** 2 + (y[:, None] - py[None, :]) ** 2
                active = np.min(dist2, axis=1) <= radius * radius
                if variant == 0:
                    # A coherent cross-path shift opens the braid without
                    # injecting independent noise into the frozen crystal.
                    if direction == "lr":
                        y[active] += 0.018
                    else:
                        x[active] += 0.018
                else:
                    count = int(active.sum())
                    x[active] += rng.uniform(-jitter, jitter, count)
                    y[active] += rng.uniform(-jitter, jitter, count)
                    th[active] += rng.uniform(-angle, angle, count)
                qseeds.append(full_to_free_sym(x, y, th, l0))
                schedules.append([
                    l0, side + 0.003, side - 1e-4,
                    side - 0.003, floor,
                ][: 4 if quick else 5])
                labels.append(f"braid_{direction}_{name}_v{variant}_r{radius:g}")
    return qseeds, schedules, labels, {
        "lr_path": paths["lr"], "bt_path": paths["bt"],
        "path_lengths": {k: len(v) for k, v in paths.items()},
    }


def run_target(n: int, threads: int, quick: bool) -> dict:
    started = time.time()
    x, y, th, side = load_group(n)
    gap_frame = pd.read_csv(GAP).set_index("n")
    target_side = float(gap_frame.loc[n, "L_frontier"])
    qs, schedules, labels, graph = braid_seeds(
        n, x, y, th, side, target_side, quick
    )
    ils_iters = [4, 6, 8, 10] if quick else [6, 9, 13, 16, 18]
    print(
        f"=== n={n} side={side:.12f} target={target_side:.12f} "
        f"paths={graph['path_lengths']} seeds={len(qs)} ===",
        flush=True,
    )
    qb, qf, lf, eo, po, lb = run_cpp_batch_ils(
        n, 1, qs, schedules, ils_iters, a_start=0.94, n_melt_stages=4,
        n_threads=threads, base_seed=778_100 + n,
    )
    feasible = int(np.sum(lf < 1e9))
    print(
        f"n={n} ILS feasible={feasible}/{len(qs)} "
        f"best_feasible={float(np.min(lf)) if feasible else None} "
        f"min_pen={float(np.min(po)):.4g}",
        flush=True,
    )

    rank = lambda i: (
        0 if lf[i] < 1e9 else 1,
        float(lf[i]) if lf[i] < 1e9 else float(po[i]),
    )
    chosen = sorted(range(len(qs)), key=rank)[: (5 if quick else 8)]
    tasks = []
    for idx in chosen:
        use_feasible = lf[idx] < 1e9
        ref_l = float(lf[idx]) if use_feasible else schedules[idx][-1]
        cx, cy, cth = cpp_expand_sym(n, 1, ref_l, qf[idx] if use_feasible else qb[idx])
        tasks.append((
            idx, labels[idx], "braid", n, cx, cy, cth,
            float(eo[idx]), 0.0 if use_feasible else float(po[idx]),
            float(lf[idx]) if use_feasible else float(lb[idx]), side,
        ))

    # The trusted polish pool may start before its first C++ call. Compile the
    # helper once here to avoid a startup race on the shared temporary library.
    get_cpp_lib()
    with mp.Pool(processes=min(threads, len(tasks))) as pool:
        polished = pool.map(i21._fast_polish_repair_worker, tasks)
    polished.sort(key=lambda r: (
        0 if r["verify"].get("ok") else 1,
        float(r["verify"].get("L_float", 1e9)) if r["verify"].get("ok") else r["pen_fin"],
    ))
    win = None
    for row in polished:
        if not row["verify"].get("ok"):
            continue
        dL = float(row["verify"]["L_float"]) - side
        if dL <= -1e-9 and (win is None or float(row["verify"]["L_float"]) < float(win["verify"]["L_float"])):
            win = row
    result = {
        "n": n, "side": side, "target_side": target_side, "graph": graph,
        "ils_feasible": feasible, "min_pen": float(np.min(po)),
        "seconds": round(time.time() - started, 2),
        "top_polished": [{k: v for k, v in row.items() if k != "rows"} for row in polished],
        "win": win,
    }
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / f"run_n{n}.json").write_text(json.dumps(result, indent=2) + "\n")
    if win is None:
        print(f"MISS n={n}", flush=True)
    else:
        print(
            f"CANDIDATE n={n} L={win['verify']['L_float']} dL="
            f"{float(win['verify']['L_float']) - side:+.6e}",
            flush=True,
        )
    return result


def certify_and_append(wins: dict[int, dict]) -> list[dict]:
    union = load_union()
    base_literal, base_pandas = union.load(BASE)
    groups = {}
    certified = []
    for n, row in sorted(wins.items()):
        candidate = union.final_text({n: [
            {
                "square_id": item["square_id"], "x": item["x"],
                "y": item["y"], "angle": item["angle"],
            } for item in row["verify"]["rows"]
        ]})
        temporary = DOCS / f"candidate_n{n}.csv"
        temporary.write_text(candidate)
        literal, transported = union.load(temporary)
        before = union.check((n, base_literal[n], base_pandas[n]))
        after = union.check((n, literal[n], transported[n]))
        if before is None or after is None:
            raise ValueError(f"failed dual certification n={n}")
        before_dec = tuple(map(float, before))
        after_dec = tuple(map(float, after))
        dL = min(before_dec) - max(after_dec)
        if not (max(after_dec) < min(before_dec) and dL <= -1e-9):
            print(f"REJECT subthreshold n={n} dL={dL:.6e}", flush=True)
            continue
        groups[n] = [
            {
                "square_id": item["square_id"], "x": item["x"],
                "y": item["y"], "angle": item["angle"],
            } for item in row["verify"]["rows"]
        ]
        certified.append({
            "n": n, "before": list(before_dec), "after": list(after_dec),
            "dL": dL, "label": row["label"],
        })
    if not groups:
        return certified
    existing = union.parse_text(FINAL.read_text()) if FINAL.exists() else {}
    groups.update(existing)
    candidate_text = union.final_text(groups)
    temporary = DOCS / "final_next.csv"
    temporary.write_text(candidate_text)
    literal, transported = union.load(temporary)
    for n in sorted(literal):
        before = union.check((n, base_literal[n], base_pandas[n]))
        after = union.check((n, literal[n], transported[n]))
        if before is None or after is None or not all(
            float(a) < float(b) for a, b in zip(after, before)
        ):
            raise ValueError(f"final regression n={n}")
    FINAL.write_text(temporary.read_text())
    return certified


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=int, nargs="+", default=[592, 813, 928])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    all_results = {}
    pending = {}
    for n in args.targets:
        result = run_target(n, args.threads, args.quick)
        all_results[str(n)] = result
        if result["win"] is not None:
            pending[n] = result["win"]
            certified = certify_and_append(pending)
            print("APPENDED", [(r["n"], f"{r['dL']:+.6e}") for r in certified], flush=True)
    (DOCS / "summary.json").write_text(json.dumps(all_results, indent=2) + "\n")


if __name__ == "__main__":
    main()
