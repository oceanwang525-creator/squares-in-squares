#!/usr/bin/env python3
"""Periodic strip-cell capacity study.

A cell contains ``m`` unit squares in a strip of width ``W`` and period ``P``
along ``x``.  Squares are periodic images of each other at ``x +/- P``; hard
walls at ``y=0`` and ``y=W``.  The optimizer is a fixed-``W`` SAT-penalty Adam
descent, warm-started while ``W`` is reduced.  The reported number is the
smallest ``W`` at which a cell is overlap-free.

This is a capacity study, not a certified n-packing.  The arm of the incumbent
large-`n` packings has linear density ``167 / 23.664 = 7.057`` squares per unit
length at width ``7.664``; the question is whether any periodic cell reaches
the same density in a smaller width.
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import sys

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = "1"

import numpy as np
import torch

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

MARGIN = 1e-9


def extents(theta):
    return 0.5 * (torch.abs(torch.cos(theta)) + torch.abs(torch.sin(theta)))


def energy(x, y, th, W, P):
    m = len(x)
    u = torch.stack([torch.cos(th), torch.sin(th)], 1)
    v = torch.stack([-torch.sin(th), torch.cos(th)], 1)
    # pairwise, three periodic images in x
    xi = x[:, None]
    yi = y[:, None]
    dx0 = x[None, :] - xi
    dy = y[None, :] - yi
    best = torch.zeros((m, m))
    ui = u[:, None, :].expand(m, m, 2)
    vi = v[:, None, :].expand(m, m, 2)
    uj = u[None, :, :].expand(m, m, 2)
    vj = v[None, :, :].expand(m, m, 2)
    for shift in (-P, 0.0, P):
        dx = dx0 + shift
        axes = torch.stack([ui, vi, uj, vj], 2)
        delta = torch.stack([dx, dy], -1)
        proj = (axes * delta[:, :, None, :]).sum(-1)
        rad = 0.5 * (
            (axes * ui[:, :, None, :]).sum(-1).abs()
            + (axes * vi[:, :, None, :]).sum(-1).abs()
            + (axes * uj[:, :, None, :]).sum(-1).abs()
            + (axes * vj[:, :, None, :]).sum(-1).abs()
        )
        pen = (rad - proj.abs()).min(dim=-1).values.clamp(min=0)
        best = torch.maximum(best, pen)
    mask = torch.triu(torch.ones((m, m), dtype=torch.bool), diagonal=1)
    e = extents(th)
    wall = torch.cat([e - y, y + e - W]).clamp(min=0)
    return (best[mask] ** 2).sum() + 10.0 * (wall ** 2).sum(), best[mask].max(), wall.max()


def run_task(args):
    m, P, W0, Wmin, seed, stages, steps = args
    np.random.seed(seed)
    torch.manual_seed(seed)
    # seed: upright grid, cols per period, pitch 1
    cols = max(1, int(round(P)))
    rows = int(math.ceil(m / cols))
    xs = [(k % cols) + 0.5 for k in range(m)]
    ys = [(k // cols) + 0.5 for k in range(m)]
    x = torch.tensor(xs, dtype=torch.float64)
    y = torch.tensor(ys, dtype=torch.float64)
    th = torch.tensor(np.random.uniform(-0.25, 0.25, m))
    x = torch.nn.Parameter(x)
    y = torch.nn.Parameter(y)
    th = torch.nn.Parameter(th)
    opt = torch.optim.Adam([x, y, th], lr=6e-4)
    W = W0
    last = None
    for stage in range(stages):
        W = max(Wmin, W0 - stage * 0.03)
        for _ in range(steps):
            opt.zero_grad()
            loss, pv, wv = energy(x, y, th, W, P)
            loss.backward()
            opt.step()
        with torch.no_grad():
            loss, pv, wv = energy(x, y, th, W, P)
        if pv.item() < 2e-9 and wv.item() < 2e-9:
            last = (W, float(loss), float(pv), float(wv))
        else:
            break
    return (m, P, seed, last)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--density", type=float, default=167.0 / 23.664318325447745)
    parser.add_argument("--ms", default="14")
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument("--jobs", type=int, default=32)
    parser.add_argument("--stages", type=int, default=45)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    ms = [int(v) for v in args.ms.split(",")]
    tasks = []
    for m in ms:
        P = m / args.density
        for seed in range(args.seeds):
            tasks.append((m, P, 8.0, 6.6, seed, args.stages, args.steps))
    results = []
    with mp.get_context("fork").Pool(max(1, min(args.jobs, os.cpu_count() or 1))) as pool:
        for res in pool.imap_unordered(run_task, tasks, chunksize=1):
            results.append(res)
            m, P, seed, last = res
            if last is not None:
                print(f"m={m} P={P:.4f} seed={seed} Wmin={last[0]:.4f} "
                      f"density={m / (P * last[0]):.5f}", flush=True)
            else:
                print(f"m={m} P={P:.4f} seed={seed} no feasible cell", flush=True)
    json.dump(results, open(args.output, "w"), indent=1)
    best = {}
    for m, P, seed, last in results:
        if last is not None and (m not in best or last[0] < best[m][0]):
            best[m] = (last[0], P, seed)
    print(json.dumps({str(k): dict(Wmin=v[0], P=v[1], seed=v[2],
                                   density=k / (v[1] * v[0]))
                      for k, v in sorted(best.items())}, indent=1))


if __name__ == "__main__":
    mp.freeze_support()
    main()
