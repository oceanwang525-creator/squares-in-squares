#!/usr/bin/env python3
"""Run the free-angle SLP with seeded noise on its angle objective."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from collections import Counter
from decimal import Decimal
from pathlib import Path

import numpy as np


POC_DIR = Path(__file__).resolve().parents[1]
ROOT = POC_DIR.parents[1]
sys.path.insert(0, str(ROOT / "poc" / "claude-midn" / "poc"))

import free_slp as slp  # noqa: E402
import rows as rowio  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", required=True, type=Path)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--jobs", type=int, default=10)
    parser.add_argument("--out-rows", required=True, type=Path)
    parser.add_argument("--out-report", required=True, type=Path)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--trust", type=float, default=0.03)
    parser.add_argument("--noise", type=float, default=1e-6)
    parser.add_argument("--campaign-seed", type=int, default=1)
    args = parser.parse_args()

    if "-" in args.spec:
        lo, hi = map(int, args.spec.split("-"))
        ns = list(range(lo, hi + 1))
    else:
        ns = [int(value) for value in args.spec.split(",")]

    def randomized_refine(c, th, iters, trust0, seed=0):
        rng = np.random.default_rng(seed)
        cur, gap0 = slp.measure(c, th)
        base = cur
        xy, rt = trust0, trust0 * 0.5
        log = []
        for iteration in range(iters):
            progress = iteration / max(1, 0.8 * iters)
            noise = args.noise * max(0.02, 1.0 - progress)
            out = slp.slp_step(c, th, xy, rt, rng, noise)
            if out is None:
                xy *= 0.3
                rt *= 0.3
                if xy < 1e-9:
                    break
                continue
            dc, dth, pred = out
            if pred > -1e-13:
                log.append((iteration, xy, pred, "rigid"))
                if xy > 2e-5:
                    xy, rt = 1e-5, 5e-6
                    continue
                break
            accepted = False
            for alpha in (1.0, 0.5, 0.25, 0.125):
                q, t = c + alpha * dc, th + alpha * dth
                side, gap = slp.measure(q, t)
                if gap < 0 and slp.REPAIR == "lp":
                    e = 0.5 * (np.abs(np.cos(t)) + np.abs(np.sin(t)))
                    for _ in range(3):
                        fix = slp.TL.lp_round(
                            q, t, e, max(4 * xy, 1e-6), repair=True
                        )
                        if fix is None:
                            break
                        q = fix[0]
                        side, gap = slp.measure(q, t)
                        if gap >= -1e-14:
                            break
                if gap < 0:
                    mid = 0.5 * (q.min(0) + q.max(0))
                    q = mid + (q - mid) * (1.0 + 1.05 * (-gap) + 1e-13)
                    side, gap = slp.measure(q, t)
                if gap >= 0 and side < cur - 1e-13:
                    c, th, cur, accepted = q, t, side, True
                    break
            log.append((iteration, xy, pred, "ok" if accepted else "reject"))
            if accepted:
                xy = min(xy * 1.5, 0.05)
                rt = min(rt * 1.5, 0.025)
            else:
                xy *= 0.3
                rt *= 0.3
                if xy < 1e-9:
                    break
        return c, th, base, cur, log

    slp.refine = randomized_refine
    groups = rowio.read_groups(args.submission, set(ns))
    tasks = [
        (n, groups[n], args.iterations, args.trust)
        for n in sorted(ns, reverse=True)
    ]
    wins = {}
    report = []
    with mp.get_context("fork").Pool(args.jobs) as pool:
        for n, info, candidate in pool.imap_unordered(slp.work, tasks):
            report.append(info)
            if candidate:
                wins[n] = candidate
                delta = Decimal(info["new_literal"]) - Decimal(info["old_literal"])
                print(f"WIN n={n} dL={float(delta):.3e} steps={info['steps']}", flush=True)

    report.sort(key=lambda item: item["n"])
    args.out_rows.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    rowio.write_rows(args.out_rows, wins)
    args.out_report.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    gain = sum(
        (Decimal(item["old_literal"]) - Decimal(item["new_literal"]))
        / Decimal(item["n"]).sqrt()
        for item in report
        if item["status"] == "WIN"
    )
    print(Counter(item["status"] for item in report), "score gain", gain)


if __name__ == "__main__":
    main()
