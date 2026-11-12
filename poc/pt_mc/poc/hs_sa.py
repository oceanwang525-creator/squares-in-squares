#!/usr/bin/env python3
"""Hard-square simulated annealing with swap and row-shift topology moves.

Energy = squared SAT penetration + 5 * squared wall violation.  Moves:
single translation+rotation, swap of two squares, and a fault-line row shift.
Compression is adaptive: scale to L-delta, anneal at fixed L until E~0, repeat.
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
from squares import core  # noqa: E402
from squares import _official_metric as metric  # noqa: E402

metric.KAGGLE_CORES = 1


@njit
def _wall(x, y, e, L):
    v = 0.0
    a = e - x
    if a > 0.0:
        v += a * a
    a = x + e - L
    if a > 0.0:
        v += a * a
    a = e - y
    if a > 0.0:
        v += a * a
    a = y + e - L
    if a > 0.0:
        v += a * a
    return v


@njit
def _pen(xi, yi, uxi, uyi, vxi, vyi,
         xj, yj, uxj, uyj, vxj, vyj):
    dx = xj - xi
    dy = yj - yi
    dot = uxi * uxj + uyi * uyj
    if dot < 0.0:
        dot = -dot
    cro = uxi * uyj - uyi * uxj
    if cro < 0.0:
        cro = -cro
    rad = 0.5 * (1.0 + dot + cro)
    p = abs(uxi * dx + uyi * dy)
    q = abs(vxi * dx + vyi * dy)
    if q > p:
        p = q
    q = abs(uxj * dx + uyj * dy)
    if q > p:
        p = q
    q = abs(vxj * dx + vyj * dy)
    if q > p:
        p = q
    pen = rad - p
    if pen > 0.0:
        return pen * pen
    return 0.0


@njit
def energy_all(x, y, th, L):
    n = len(x)
    ux = np.empty(n); uy = np.empty(n)
    vx = np.empty(n); vy = np.empty(n)
    e = np.empty(n)
    E = 0.0
    for i in range(n):
        ux[i] = math.cos(th[i]); uy[i] = math.sin(th[i])
        vx[i] = -uy[i]; vy[i] = ux[i]
        e[i] = 0.5 * (abs(ux[i]) + abs(uy[i]))
        E += 5.0 * _wall(x[i], y[i], e[i], L)
    for i in range(n):
        for j in range(i + 1, n):
            E += _pen(x[i], y[i], ux[i], uy[i], vx[i], vy[i],
                      x[j], y[j], ux[j], uy[j], vx[j], vy[j])
    return E


@njit
def sa(x, y, th, L, steps, seed, T0, Tf, p_swap, p_row, sig0, sig_row):
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
        frac = step / steps
        T = T0 * (Tf / T0) ** frac
        sig = sig0 * (1.0 - 0.95 * frac) + 2e-4
        r = np.random.random()
        if r < p_row:
            # fault-line shift
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
        elif r < p_row + p_swap:
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
    return bestE, bx, by, bt


@njit
def compress(x, y, th, L0, Ltarget, delta, steps, seed, T0, Tf,
             p_swap, p_row, sig0, sig_row, max_success=20):
    L = L0
    tries = 0
    ok_steps = 0
    while L > Ltarget + 1e-12 and tries < 80 and ok_steps < max_success:
        Ln = max(Ltarget, L - delta)
        s = Ln / L
        ox = x.copy(); oy = y.copy(); oth = th.copy()
        x *= s; y *= s
        e = np.empty(len(x))
        for i in range(len(x)):
            e[i] = 0.5 * (abs(math.cos(th[i])) + abs(math.sin(th[i])))
        xmin = 1e9
        for i in range(len(x)):
            if x[i] - e[i] < xmin:
                xmin = x[i] - e[i]
        ymin = 1e9
        for i in range(len(x)):
            if y[i] - e[i] < ymin:
                ymin = y[i] - e[i]
        x -= xmin; y -= ymin
        E, bx, by, bt = sa(x, y, th, Ln, steps, seed + tries, T0, Tf,
                           p_swap, p_row, sig0, sig_row)
        if E < 1e-18:
            x[:] = bx; y[:] = by; th[:] = bt
            L = Ln
            ok_steps += 1
            delta = min(delta * 1.3, 0.004)
            tries = 0
        else:
            x[:] = ox; y[:] = oy; th[:] = oth
            delta *= 0.5
            tries += 1
            if delta < 2e-5:
                break
    return L, x, y, th, energy_all(x, y, th, L)


def worker(args):
    n, delta, seed, steps = args
    rows = [r for r in csv.DictReader(open(
        ROOT + "/poc/pell_corrugation/data/border-scan/merged.csv"))
        if int(r["n"]) == n]
    x = np.array([float(r["x"]) for r in rows])
    y = np.array([float(r["y"]) for r in rows])
    th = np.array([float(r["angle"]) for r in rows])
    e = 0.5 * (np.abs(np.cos(th)) + np.abs(np.sin(th)))
    x -= (x - e).min(); y -= (y - e).min()
    L0 = float(max((x + e).max() - (x - e).min(),
                   (y + e).max() - (y - e).min()))
    ours = Decimal(core.certify(n, rows))
    target = TARGETS[n]
    t0 = time.time()
    L, x2, y2, t2, E = compress(x, y, th, L0, target - 1e-9, delta, steps,
                               seed, 3e-4, 1e-10, 0.12, 0.002, 0.02, 0.03)
    if E > 1e-16 or L > target + 1e-8:
        return (n, seed, "fail", float(L), float(E), time.time() - t0, None)
    for eps in (0.0, 1e-12, 1e-11, 1e-10, 1e-9):
        xx = x2.copy(); yy = y2.copy()
        if eps > 0:
            em = 0.5 * (np.abs(np.cos(t2)) + np.abs(np.sin(t2)))
            cx = ((xx - em).min() + (xx + em).max()) / 2
            cy = ((yy - em).min() + (yy + em).max()) / 2
            xx = cx + (xx - cx) * (1 + eps)
            yy = cy + (yy - cy) * (1 + eps)
        cand = [dict(square_id=str(i), x=repr(float(xx[i])),
                     y=repr(float(yy[i])), angle=repr(float(t2[i])))
                for i in range(n)]
        try:
            lit = Decimal(core.certify(n, cand))
            tr = core.kaggle_roundtrip({n: cand})
            pan = Decimal(core.certify(n, tr[n]))
        except Exception:
            continue
        return (n, seed, "cert", float(L), float(E), time.time() - t0,
                dict(old_L=str(ours), new_L=str(max(lit, pan)),
                     literal_L=str(lit), pandas_L=str(pan), eps=eps,
                     rows=cand))
    return (n, seed, "certfail", float(L), float(E), time.time() - t0, None)


TARGETS = {
    911: 30.6609125615235,
    367: 19.6708567970218,
    910: 30.6464577019472,
    972: 31.6514012986526,
    791: 28.6058845085404,
}


if __name__ == "__main__":
    import multiprocessing as mp
    ns = [int(v) for v in sys.argv[1].split(",")]
    delta = float(sys.argv[2]) if len(sys.argv) > 2 else 0.002
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 120000
    seeds = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    tasks = [(n, delta, s, steps) for n in ns for s in range(seeds)]
    wins = {}
    with mp.get_context("fork").Pool(32) as pool:
        for res in pool.imap_unordered(worker, tasks, chunksize=1):
            n, seed, status, L, E, secs, win = res
            print(f"n={n} seed={seed} {status} L={L:.9f} E={E:.2e} "
                  f"{secs:.1f}s", flush=True)
            if win is not None and Decimal(win["new_L"]) < Decimal(win["old_L"]):
                wins[str(n)] = win
                print("WIN", n, win["old_L"], "->", win["new_L"], flush=True)
    json.dump(wins, open("/tmp/opencode/hs_sa_wins.json", "w"), indent=1)
    print("wins", len(wins))
