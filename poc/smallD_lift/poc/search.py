"""Multi-start global optimiser: pack N unit squares into the smallest square.

Task
----
Find the smallest axis-aligned box side `L` holding `N` unit squares (free
centres, free angles) and *certify* it with the repository's official metric
(`squares.core.certify`, 50-digit decimal).

Method
------
Exact penalty (separating-axis penetration, no discretisation):

    E(q, L) = sum_{i<j} max(depth_ij + m, 0)^2 + w_wall * sum_i wall_i^2

`depth_ij` is the exact SAT penetration depth of two unit squares and
`q` is `(x, y, theta)` per square.  `E == 0` means the configuration is a valid
packing of the box with pairwise clearance `m` (> 0 keeps the float solution
provably legal under the official decimal metric).

Search = basin hopping with a shrink continuation:

  1. seeds: rotated unit lattices (16 angles, random phase), upright grids,
     greedy hard-square insertion, uniform random, optionally the incumbent;
  2. repair: Adam on `E` with a decreasing lr, then L-BFGS-B polishing;
  3. compression: `L -> L - step` continuation; accept on feasibility and grow
     `step`, otherwise halve it and (once `step` is small) retry with a deeper
     repair;
  4. refinement: bisection between the best feasible side and the smallest
     infeasible side found;
  5. every new feasible state is certified exactly; the best *certified* side
     is reported.

Usage
-----
    python search.py --n 71 --side 9.0 --target 8.6059 --seconds 60
    python search.py --n 71 --sides 9.0,9.4 --starts 20 --out rows.csv
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

# Single-threaded BLAS: the arrays here are tiny (N <= ~110), so OpenMP thread
# spinning costs far more than it saves.  Must be set before numpy is imported.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "poc" / "baseline" / "poc" / "src"))

import overlap as ov  # noqa: E402

MARGIN = 0.0    # pairwise clearance demanded *during* the search
TOL = 1e-18     # "E == 0" (each pair depth < 1e-9 at margin 0)


# --------------------------------------------------------------------------- #
# optimisers on the fixed-L penalty
# --------------------------------------------------------------------------- #
def adam(q, L, steps, lr, margin=0.0, w_wall=3.0):
    q = q.copy()
    mt = np.zeros_like(q)
    vt = np.zeros_like(q)
    b1, b2 = 0.9, 0.999
    for k in range(1, steps + 1):
        E, G = ov.penalty(q, L, w_wall=w_wall, margin=margin)
        mt = b1 * mt + (1 - b1) * G
        vt = b2 * vt + (1 - b2) * G * G
        mh = mt / (1 - b1 ** k)
        vh = vt / (1 - b2 ** k)
        q -= (lr * math.sqrt(1 - b2 ** k) / (1 - b1 ** k)) * mh / (np.sqrt(vh) + 1e-12)
    return q


def lbfgs(q, L, margin=0.0, w_wall=3.0, maxiter=300):
    from scipy.optimize import minimize

    shape = q.shape

    def fun(v):
        E, G = ov.penalty(v.reshape(shape), L, w_wall=w_wall, margin=margin)
        return E, G.ravel()

    res = minimize(fun, q.ravel(), jac=True, method="L-BFGS-B",
                   options=dict(maxiter=maxiter, maxcor=20,
                                ftol=1e-16, gtol=1e-14))
    return res.x.reshape(shape)


def energy(q, L, margin=0.0):
    return ov.penalty(q, L, margin=margin, grad=False)


def polish(q, L, margin=0.0, rounds=2, steps=200, lr=0.004):
    """Cheap Adam + L-BFGS polish at a fixed side."""
    q = q.copy()
    if energy(q, L, margin) < TOL:
        return q                     # already feasible: Adam would kick it out
    for _ in range(rounds):
        q = adam(q, L, steps, lr, margin=margin)
        q = lbfgs(q, L, margin=margin, maxiter=200)
        if energy(q, L, margin) < TOL:
            return q
    return q


def repair(q, L, margin=0.0, deep=False):
    """Bring `q` to (or towards) a feasible packing of a side-`L` box.

    NB: Adam's step length is ~lr *regardless* of the gradient magnitude, so an
    already-feasible state must be returned untouched instead of being relaxed.
    """
    q = q.copy()
    if energy(q, L, margin) < TOL:
        return q
    if deep:
        sched = ((0.03, 400), (0.01, 400), (0.003, 400), (0.001, 300))
    else:
        sched = ((0.02, 250), (0.004, 200))
    for lr, steps in sched:
        q = adam(q, L, steps, lr, margin=margin)
        if energy(q, L, margin) < TOL:
            break
    q = lbfgs(q, L, margin=margin, maxiter=400 if deep else 200)
    if energy(q, L, margin) > TOL:
        q = lbfgs(q, L, margin=margin, maxiter=800)
    return q


def feasible(q, L, margin=MARGIN):
    """E == 0 and (float) pairwise / wall clearance margins hold."""
    if energy(q, L, margin) > TOL:
        return False
    od, wv = ov.violation(q, L)
    return od < -0.5 * margin and wv <= 1e-12


# --------------------------------------------------------------------------- #
# seeds
# --------------------------------------------------------------------------- #
def seed_lattice(N, L, rng, angle):
    """`angle`-rotated unit lattice with a random rigid offset."""
    c, s = math.cos(angle), math.sin(angle)
    r = 0.5 * (abs(c) + abs(s))
    nmax = int(L) + 3
    ox, oy = rng.uniform(0, L), rng.uniform(0, L)
    pts = []
    for i in range(-nmax, nmax + 1):
        for j in range(-nmax, nmax + 1):
            u, v = float(i), float(j)
            x = ox + u * c - v * s
            y = oy + u * s + v * c
            if r < x < L - r and r < y < L - r:
                pts.append((x, y))
    if len(pts) < N:
        return None
    idx = rng.permutation(len(pts))[:N]
    return np.array([(pts[i][0], pts[i][1], angle) for i in idx], float)


def seed_greedy(N, L, rng, margin=MARGIN, tries=200000):
    """Sequential hard-square insertion at random positions and angles."""
    q = np.zeros((0, 3))
    for _ in range(tries):
        if len(q) == N:
            break
        cand = np.array([[rng.uniform(0, L), rng.uniform(0, L),
                          rng.uniform(0, math.pi / 2)]])
        h = ov.radii(cand)[0]
        if not (h < cand[0, 0] < L - h and h < cand[0, 1] < L - h):
            continue
        if len(q):
            trial = np.vstack([q, cand])
            k = len(q)
            P = np.stack([np.arange(k), np.full(k, k)], 1)
            if ov.pair_depth(trial, P).max() > -margin:
                continue
        q = np.vstack([q, cand])
    return q if len(q) == N else None


def seed_perturb(q, rng, sigma):
    r = q.copy()
    r[:, :2] += rng.normal(0.0, sigma, (len(r), 2))
    r[:, 2] += rng.normal(0.0, 0.3 * sigma, len(r))
    return r


def make_seed(N, L, rng, kind):
    if kind == "grid":
        return seed_lattice(N, L, rng, 0.0)
    if kind == "lattice":
        a = rng.choice([8, 10, 12, 15, 18.43, 22.5, 26.57, 30, 33.69, 36.87,
                        40, 45, 50, 55, 60, 63])
        return seed_lattice(N, L, rng, math.radians(a))
    if kind == "greedy":
        return seed_greedy(N, L, rng)
    pad = 0.5
    return np.c_[rng.uniform(pad, L - pad, (N, 2)), rng.uniform(0, math.pi, N)]


# --------------------------------------------------------------------------- #
# exact certification
# --------------------------------------------------------------------------- #
def certify_side(q):
    """Official-metric certified side of `q` (raises if it is not a packing)."""
    from squares.core import certify

    q = ov.normalise(q)
    N = len(q)
    rows = [dict(square_id=str(i), x=repr(float(q[i, 0])),
                 y=repr(float(q[i, 1])), angle=repr(float(q[i, 2])))
            for i in range(N)]
    return float(certify(N, rows)), rows


def dilate(q, eps):
    """Scale the centres about the bbox lower-left corner by `1+eps`.

    Squares keep unit size while every centre distance grows by a factor
    `1+eps`, so a *touching* pair separates by `eps*d_ij > 0` and the box grows
    by `eps*L`.  This is the cheapest guaranteed way to turn a float solution
    with exact tangencies into one the exact decimal metric accepts.
    """
    r = ov.normalise(q).copy()
    r[:, 0] *= 1.0 + eps
    r[:, 1] *= 1.0 + eps
    return r


def certify_best(q):
    """Certify with the official metric, dilating if exact tangency is rejected.

    The search runs at clearance 0, so a jammed optimum has exact contacts
    (and float-level overlaps of ~1e-16) that the exact decimal metric rejects.
    A `1+eps` dilation costs only `eps*L` of side.
    """
    for eps in (0.0, 1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7):
        try:
            return certify_side(q if eps == 0.0 else dilate(q, eps))
        except Exception:
            continue
    return None


# --------------------------------------------------------------------------- #
# shrink continuation
# --------------------------------------------------------------------------- #
def compress(q, L0, deadline, target=None, shrink=0.01, log=None,
             min_step=3e-4):
    """Shrink a feasible configuration at side <= L0 as far as time allows."""
    q = q.copy()
    L = min(L0, ov.side(q))
    best_q, best_L = q.copy(), ov.side(q)
    Lfail = math.inf
    step = shrink * L
    while time.time() < deadline:
        Ltry = L - step
        if target is not None:
            Ltry = max(Ltry, target - 2 * MARGIN)
        if Ltry >= Lfail - 1e-13:
            step *= 0.5
            if step < min_step * L:
                break
            continue
        r = polish(q, Ltry, margin=MARGIN)
        if energy(r, Ltry, margin=MARGIN) > TOL and step < 2e-3 * L:
            r = repair(q, Ltry, margin=MARGIN, deep=True)
        if energy(r, Ltry, margin=MARGIN) < TOL:
            q = r
            L = min(Ltry, ov.side(q))
            if ov.side(q) < best_L - 1e-13:
                best_q, best_L = q.copy(), ov.side(q)
                if log:
                    log(f"      feasible {best_L:.9f}")
            step *= 1.3
        else:
            Lfail = min(Lfail, Ltry)
            step *= 0.55
            if step < min_step * L:
                break
        if target is not None and best_L <= target:
            break
    return best_q, best_L, Lfail


def refine(q, Lk, Lf, margin=MARGIN, rounds=10, deadline=None):
    """Bisect between a feasible side Lf and an infeasible side Lk."""
    for _ in range(rounds):
        if deadline is not None and time.time() > deadline:
            break
        if Lk - Lf < 1e-12:
            break
        mid = 0.5 * (Lf + Lk)
        r = repair(q, mid, margin=margin, deep=True)
        if energy(r, mid, margin) < TOL:
            q = r
            Lf = min(mid, ov.side(r))
        else:
            Lk = mid
    return q, Lf, Lk


# --------------------------------------------------------------------------- #
# multi-start driver
# --------------------------------------------------------------------------- #
def search_one(N, start_side, deadline, target=None, n_starts=None,
               incumbent=None, seed=0, verbose=True, refine_rounds=10):
    rng = np.random.default_rng(seed)
    log = (lambda *a: print(*a, flush=True)) if verbose else None
    kinds = ["lattice", "grid", "lattice", "greedy", "random"]
    best_q = None
    best_side, best_rows = math.inf, None
    n_feas = 0
    it = 0
    while time.time() < deadline:
        if n_starts is not None and it >= n_starts:
            break
        it += 1
        L0 = start_side * (1.0 + (0.06 if it > 1 else 0.0))
        if it == 1 and incumbent is not None:
            q = incumbent.copy()
            L0 = ov.side(q) * (1 + 1e-7)
        elif best_q is not None and it % 3 == 0:
            q = seed_perturb(best_q, rng, sigma=10 ** rng.uniform(-3.0, -1.3) * N ** 0.1)
            L0 = ov.side(q)
        else:
            q = None
            for kind in [kinds[it % len(kinds)]] + kinds:
                q = make_seed(N, L0, rng, kind)
                if q is not None:
                    break
            if q is None:
                continue
        q = repair(q, L0, margin=MARGIN, deep=True)
        if energy(q, L0, margin=MARGIN) > TOL:
            if log:
                log(f"  start {it}: seed infeasible at {L0:.6f}")
            continue
        n_feas += 1
        q, side, Lfail = compress(q, ov.side(q), deadline, target=target, log=log)
        if math.isfinite(Lfail) and Lfail - side > 1e-9:
            q, side, _ = refine(q, Lfail, side, deadline=deadline,
                                rounds=refine_rounds)
        cert = certify_best(q)
        if cert is None:
            if log:
                log(f"  start {it}: float feasible {side:.9f} but certify failed")
            continue
        cl, rows = cert
        if best_q is None or side < ov.side(best_q):
            best_q = q.copy()
        if cl < best_side:
            best_side, best_rows = cl, rows
            if log:
                log(f"  start {it}: certified side {cl:.12f} (started {L0:.6f})")
        if target is not None and best_side <= target:
            if log:
                log(f"  TARGET reached: {best_side:.12f} <= {target}")
            break
    return best_side, best_rows, dict(starts=it, feasible_starts=n_feas)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--target", type=float, default=None)
    ap.add_argument("--side", type=float, default=None)
    ap.add_argument("--sides", type=str, default=None)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--starts", type=int, default=None)
    ap.add_argument("--refine", type=int, default=10)
    ap.add_argument("--incumbent", type=str, default=None,
                    help="CSV with columns n,square_id,x,y,angle")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)

    start = a.side if a.side else math.sqrt(a.n) * 1.4
    sides = [float(x) for x in a.sides.split(",")] if a.sides else [start]
    inc = None
    if a.incumbent:
        import csv
        rows = [r for r in csv.DictReader(open(a.incumbent))
                if int(r["n"]) == a.n]
        if len(rows) == a.n:
            rows.sort(key=lambda r: int(r["square_id"]))
            inc = ov.normalise(np.array([(float(r["x"]), float(r["y"]),
                                          float(r["angle"])) for r in rows]))

    deadline = time.time() + a.seconds
    best = (math.inf, None, None)
    for i, s in enumerate(sides):
        d = (deadline if i == len(sides) - 1
             else time.time() + a.seconds / len(sides))
        side, rows, stats = search_one(a.n, s, d, target=a.target,
                                       n_starts=a.starts, incumbent=inc,
                                       seed=a.seed + i, verbose=not a.quiet,
                                       refine_rounds=a.refine)
        if side < best[0]:
            best = (side, rows, stats)
    side, rows, stats = best
    print(json.dumps(dict(n=a.n, certified_side=side, target=a.target,
                          stats=stats)))
    if a.out and rows is not None:
        import csv as _csv
        with open(a.out, "w", newline="") as f:
            w = _csv.writer(f, lineterminator="\n")
            w.writerow(["n", "square_id", "x", "y", "angle"])
            for i, r in enumerate(rows):
                w.writerow([a.n, i, r["x"], r["y"], r["angle"]])
    return 0


if __name__ == "__main__":
    sys.exit(main())
