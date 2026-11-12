#!/usr/bin/env python3
"""Replica-exchange parallel tempering for hard-square compression.

R replicas with a geometric temperature ladder run Metropolis at a fixed box
side L.  Every batch, adjacent replicas attempt to exchange configurations with
the standard PT acceptance exp((1/T_i - 1/T_j)(E_j - E_i)).  Moves: single
translation+rotation, square swap, fault-line row shift, and a void-targeted
relocation (rejection-sample a free unit-square site, move a random square
there).
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys
import time
from decimal import Decimal

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np
from numba import njit

ROOT = "/mnt/e/squares-in-squares"
sys.path.insert(0, ROOT + "/poc/baseline/poc/src")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from squares import core  # noqa: E402
from squares import _official_metric as metric  # noqa: E402
from hs_sa import _pen, _wall, energy_all  # noqa: E402

metric.KAGGLE_CORES = 1


@njit
def mc_fixed(x, y, th, L, T, steps, seed, p_swap, p_row, p_void,
             sig, sig_row):
    n = len(x)
    np.random.seed(seed)
    ux = np.empty(n); uy = np.empty(n)
    vx = np.empty(n); vy = np.empty(n)
    e = np.empty(n)
    for i in range(n):
        ux[i] = math.cos(th[i]); uy[i] = math.sin(th[i])
        vx[i] = -uy[i]; vy[i] = ux[i]
        e[i] = 0.5 * (abs(ux[i]) + abs(uy[i]))
    E = energy_all(x, y, th, L)
    bestE = E
    bx = x.copy(); by = y.copy(); bt = th.copy()
    for step in range(steps):
        r = np.random.random()
        if r < p_void:
            i = np.random.randint(n)
            found = False
            px = 0.0; py = 0.0; pa = 0.0
            for _try in range(24):
                px = 0.5 + (L - 1.0) * np.random.random()
                py = 0.5 + (L - 1.0) * np.random.random()
                pa = np.random.uniform(-0.7853981633974483, 0.7853981633974483)
                nux = math.cos(pa); nuy = math.sin(pa)
                nvx = -nuy; nvy = nux
                free = True
                for k in range(n):
                    if k == i:
                        continue
                    if _pen(px, py, nux, nuy, nvx, nvy,
                            x[k], y[k], ux[k], uy[k], vx[k], vy[k]) > 0.0:
                        free = False
                        break
                if free:
                    found = True
                    break
            if found:
                nux = math.cos(pa); nuy = math.sin(pa)
                nvx = -nuy; nvy = nux
                ne = 0.5 * (abs(nux) + abs(nuy))
                old = 5.0 * _wall(x[i], y[i], e[i], L)
                new = 5.0 * _wall(px, py, ne, L)
                for k in range(n):
                    if k == i:
                        continue
                    old += _pen(x[i], y[i], ux[i], uy[i], vx[i], vy[i],
                                x[k], y[k], ux[k], uy[k], vx[k], vy[k])
                    new += _pen(px, py, nux, nuy, nvx, nvy,
                                x[k], y[k], ux[k], uy[k], vx[k], vy[k])
                dE = new - old
                if dE <= 0.0 or np.random.random() < math.exp(-dE / T):
                    x[i] = px; y[i] = py; th[i] = pa
                    ux[i] = nux; uy[i] = nuy; vx[i] = nvx; vy[i] = nvy
                    e[i] = ne
                    E += dE
        elif r < p_void + p_row:
            axis = np.random.randint(2)
            cut = L * (0.2 + 0.6 * np.random.random())
            dx = np.random.normal(0.0, sig_row)
            dy = np.random.normal(0.0, sig_row)
            ox = x.copy(); oy = y.copy()
            for i in range(n):
                if (x[i] > cut) if axis == 0 else (y[i] > cut):
                    x[i] += dx; y[i] += dy
            En = energy_all(x, y, th, L)
            dE = En - E
            if dE <= 0.0 or np.random.random() < math.exp(-dE / T):
                E = En
            else:
                x[:] = ox; y[:] = oy
        elif r < p_void + p_row + p_swap:
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i == j:
                continue
            old = 5.0 * (_wall(x[i], y[i], e[i], L)
                         + _wall(x[j], y[j], e[j], L))
            new = 5.0 * (_wall(x[j], y[j], e[j], L)
                         + _wall(x[i], y[i], e[i], L))
            for k in range(n):
                if k == i or k == j:
                    continue
                old += _pen(x[i], y[i], ux[i], uy[i], vx[i], vy[i],
                            x[k], y[k], ux[k], uy[k], vx[k], vy[k])
                old += _pen(x[j], y[j], ux[j], uy[j], vx[j], vy[j],
                            x[k], y[k], ux[k], uy[k], vx[k], vy[k])
                new += _pen(x[j], y[j], ux[j], uy[j], vx[j], vy[j],
                            x[k], y[k], ux[k], uy[k], vx[k], vy[k])
                new += _pen(x[i], y[i], ux[i], uy[i], vx[i], vy[i],
                            x[k], y[k], ux[k], uy[k], vx[k], vy[k])
            dE = new - old
            if dE <= 0.0 or np.random.random() < math.exp(-dE / T):
                x[i], x[j] = x[j], x[i]
                y[i], y[j] = y[j], y[i]
                th[i], th[j] = th[j], th[i]
                ux[i], ux[j] = ux[j], ux[i]
                uy[i], uy[j] = uy[j], uy[i]
                vx[i], vx[j] = vx[j], vx[i]
                vy[i], vy[j] = vy[j], vy[i]
                e[i], e[j] = e[j], e[i]
                E += dE
        else:
            i = np.random.randint(n)
            dth = np.random.normal(0.0, sig)
            dx = np.random.normal(0.0, sig)
            dy = np.random.normal(0.0, sig)
            nux = math.cos(th[i] + dth); nuy = math.sin(th[i] + dth)
            nvx = -nuy; nvy = nux
            ne = 0.5 * (abs(nux) + abs(nuy))
            old = 5.0 * _wall(x[i], y[i], e[i], L)
            new = 5.0 * _wall(x[i] + dx, y[i] + dy, ne, L)
            for k in range(n):
                if k == i:
                    continue
                old += _pen(x[i], y[i], ux[i], uy[i], vx[i], vy[i],
                            x[k], y[k], ux[k], uy[k], vx[k], vy[k])
                new += _pen(x[i] + dx, y[i] + dy, nux, nuy, nvx, nvy,
                            x[k], y[k], ux[k], uy[k], vx[k], vy[k])
            dE = new - old
            if dE <= 0.0 or np.random.random() < math.exp(-dE / T):
                x[i] += dx; y[i] += dy; th[i] += dth
                ux[i] = nux; uy[i] = nuy; vx[i] = nvx; vy[i] = nvy; e[i] = ne
                E += dE
        if E < bestE:
            bestE = E
            bx[:] = x; by[:] = y; bt[:] = th
            if bestE < 1e-22:
                break
    return E, x, y, th, bestE, bx, by, bt


def ladder(args):
    n, delta, seed, rounds, batch, nrep = args
    rows = [r for r in csv.DictReader(open(
        ROOT + "/poc/pell_corrugation/data/border-scan/merged.csv"))
        if int(r["n"]) == n]
    x0 = np.array([float(r["x"]) for r in rows])
    y0 = np.array([float(r["y"]) for r in rows])
    th0 = np.array([float(r["angle"]) for r in rows])
    e0 = 0.5 * (np.abs(np.cos(th0)) + np.abs(np.sin(th0)))
    x0 -= (x0 - e0).min(); y0 -= (y0 - e0).min()
    L0 = float(max((x0 + e0).max() - (x0 - e0).min(),
                   (y0 + e0).max() - (y0 - e0).min()))
    ours = Decimal(core.certify(n, rows))
    L = L0 - delta
    s = L / L0
    xs = []
    ys = []
    ts = []
    rng = np.random.default_rng(seed)
    for r in range(nrep):
        xx = x0 * s; yy = y0 * s
        if r > 0:
            xx = xx + rng.normal(0.0, 0.002, n)
            yy = yy + rng.normal(0.0, 0.002, n)
        tt = th0.copy()
        if r > 0:
            tt = tt + rng.normal(0.0, 0.002, n)
        em = 0.5 * (np.abs(np.cos(tt)) + np.abs(np.sin(tt)))
        xx -= (xx - em).min(); yy -= (yy - em).min()
        xs.append(xx); ys.append(yy); ts.append(tt)
    temps = np.geomspace(2e-6, 5e-3, nrep)
    Es = [energy_all(xs[r], ys[r], ts[r], L) for r in range(nrep)]
    bestE = min(Es)
    best = None
    if bestE < 1e-18:
        r = int(np.argmin(Es)); best = (xs[r], ys[r], ts[r])
    t0 = time.time()
    for rnd in range(rounds):
        for r in range(nrep):
            E, x, y, t, bE, bx, by, bt = mc_fixed(
                xs[r], ys[r], ts[r], L, temps[r], batch, seed + 1000 * r + rnd,
                0.10, 0.001, 0.002, 0.02, 0.03)
            xs[r] = x; ys[r] = y; ts[r] = t; Es[r] = E
            if bE < bestE:
                bestE = bE
                best = (bx.copy(), by.copy(), bt.copy())
        for r in range(nrep - 1):
            d = (1.0 / temps[r] - 1.0 / temps[r + 1]) * (Es[r + 1] - Es[r])
            if d >= 0.0 or np.random.random() < math.exp(d):
                xs[r], xs[r + 1] = xs[r + 1], xs[r]
                ys[r], ys[r + 1] = ys[r + 1], ys[r]
                ts[r], ts[r + 1] = ts[r + 1], ts[r]
                Es[r], Es[r + 1] = Es[r + 1], Es[r]
        be = min(Es)
        if be < bestE:
            bestE = be
            r = int(np.argmin(Es))
            best = (xs[r].copy(), ys[r].copy(), ts[r].copy())
        if bestE < 1e-18:
            break
    if bestE > 1e-16 or best is None:
        return (n, seed, "fail", L, bestE, time.time() - t0, None)
    bx, by, bt = best
    for eps in (0.0, 1e-12, 1e-11, 1e-10, 1e-9):
        xx = bx.copy(); yy = by.copy()
        if eps > 0:
            em = 0.5 * (np.abs(np.cos(bt)) + np.abs(np.sin(bt)))
            cx = ((xx - em).min() + (xx + em).max()) / 2
            cy = ((yy - em).min() + (yy + em).max()) / 2
            xx = cx + (xx - cx) * (1 + eps)
            yy = cy + (yy - cy) * (1 + eps)
        cand = [dict(square_id=str(i), x=repr(float(xx[i])),
                     y=repr(float(yy[i])), angle=repr(float(bt[i])))
                for i in range(n)]
        try:
            lit = Decimal(core.certify(n, cand))
            tr = core.kaggle_roundtrip({n: cand})
            pan = Decimal(core.certify(n, tr[n]))
        except Exception:
            continue
        return (n, seed, "cert", L, bestE, time.time() - t0,
                dict(old_L=str(ours), new_L=str(max(lit, pan)),
                     literal_L=str(lit), pandas_L=str(pan), eps=eps,
                     rows=cand))
    return (n, seed, "certfail", L, bestE, time.time() - t0, None)


if __name__ == "__main__":
    import multiprocessing as mp
    n = int(sys.argv[1])
    delta = float(sys.argv[2])
    rounds = int(sys.argv[3])
    batch = int(sys.argv[4])
    seeds = int(sys.argv[5])
    nrep = int(sys.argv[6]) if len(sys.argv) > 6 else 8
    tasks = [(n, delta, s, rounds, batch, nrep) for s in range(seeds)]
    wins = {}
    with mp.get_context("fork").Pool(32) as pool:
        for res in pool.imap_unordered(ladder, tasks, chunksize=1):
            n2, seed, status, L, E, secs, win = res
            print(f"n={n2} seed={seed} {status} L={L:.9f} E={E:.2e} "
                  f"{secs:.1f}s", flush=True)
            if win is not None and Decimal(win["new_L"]) < Decimal(win["old_L"]):
                wins[str(n2)] = win
                print("WIN", n2, win["old_L"], "->", win["new_L"], flush=True)
    json.dump(wins, open("/tmp/opencode/pt_wins.json", "w"), indent=1)
    print("wins", len(wins))
