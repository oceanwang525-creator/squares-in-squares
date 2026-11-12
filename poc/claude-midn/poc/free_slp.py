#!/usr/bin/env python3
"""Kink-free free-angle SLP (corner formulation) with a dilation-repair line search.

Why another SLP.  `poc/free_angle_slp` had sign errors; its replacement
`poc/lip_slp` bounds the support growth by LIP*|dtheta| on every row, so in that
model a rotation can only ever *tighten* a constraint - it is a fixed-angle LP
in disguise and cannot detect a first-order rotational escape (for example a
whole tilted block turning rigidly).  Here every nearby pair gets the rows

    a(theta_w) . (c_o + R(theta_o) w_k - c_w) >= 1/2      k = 4 corners of "other"

for the currently best separating face normal `a` of the "owner" square, and the
walls are written on the 4 corners of each square.  Each row is smooth in
(x, y, theta), so aligned contacts need no |dtheta| kink variable, a rigid
joint rotation of an aligned block is free, and a relative rotation is charged
exactly at first order.

Step control.  LP variables are scaled by the trust radius.  The proposed step
is tried at alpha = 1, 1/2, 1/4, 1/8; any second-order overlap is repaired by an
exact fixed-angle LP in the centres (fallback: uniform dilation about the
centre), and the step is kept only if the *measured* side after repair is lower.  A final margin-tight fixed-angle LP removes the
dilation slack, then the unchanged official metric gates both transports.

usage: free_slp.py FULL.csv N_LIST|LO-HI JOBS OUT_ROWS.csv REPORT.json [ITERS] [TRUST0]
"""
import json
import multiprocessing as mp
import os
import sys
from collections import Counter
from decimal import Decimal
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rows as R  # noqa: E402
import tight_lp as TL  # noqa: E402

MARGIN = 5e-13
REPAIR = os.environ.get("SLP_REPAIR", "lp")   # "lp" (exact fixed-angle LP) or "dilate"
W = 0.5 * np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]], float)


def perp(v):
    return np.stack([-v[..., 1], v[..., 0]], -1)


def corners_rel(th):
    """(n, 4, 2): R(theta) w_k."""
    c, s = np.cos(th), np.sin(th)
    return np.stack([c[:, None] * W[None, :, 0] - s[:, None] * W[None, :, 1],
                     s[:, None] * W[None, :, 0] + c[:, None] * W[None, :, 1]], -1)


def measure(c, th):
    e = 0.5 * (np.abs(np.cos(th)) + np.abs(np.sin(th)))
    return TL.side(c, e), TL.pair_axes(c, th)[3].min()


def slp_step(c, th, xy, rt, rng=None, noise=0.0, want_duals=False):
    n = len(c)
    row_a, row_b = [], []                                   # squares involved in each row (for dual mass)
    u = np.c_[np.cos(th), np.sin(th)]
    v = perp(u)
    pr = np.array(sorted(cKDTree(c).query_pairs(1.6 + 3 * xy)))
    i, j = pr[:, 0], pr[:, 1]
    d = c[j] - c[i]
    cand = [(u[i], i, j), (v[i], i, j), (u[j], j, i), (v[j], j, i)]
    best = np.full(len(pr), -np.inf)
    ax = np.zeros((len(pr), 2)); own = np.zeros(len(pr), int); oth = np.zeros(len(pr), int)
    for a, w_, o_ in cand:
        proj = np.einsum("ij,ij->i", a, d)
        rad = 0.5 * (np.abs(np.einsum("ij,ij->i", a, u[i])) + np.abs(np.einsum("ij,ij->i", a, v[i]))
                     + np.abs(np.einsum("ij,ij->i", a, u[j])) + np.abs(np.einsum("ij,ij->i", a, v[j])))
        gap = np.abs(proj) - rad
        b = gap > best
        best[b] = gap[b]
        sgn = np.sign(c[o_] - c[w_]).copy()
        sg = np.sign(np.einsum("ij,ij->i", a, c[o_] - c[w_]))
        ax[b] = (a * sg[:, None])[b]; own[b] = w_[b]; oth[b] = o_[b]
    rel = corners_rel(th)                                   # (n,4,2)
    U = xy
    rr, cc, vv, rhs = [], [], [], []
    row = 0
    for k in range(4):
        rk = rel[oth, k]                                    # (p,2)
        vec = c[oth] - c[own] + rk
        g = np.einsum("ij,ij->i", ax, vec) - 0.5
        keep = g < 2.9 * xy + 3.0 * rt
        m = np.minimum(MARGIN, np.maximum(g, 0.0))
        idx = np.flatnonzero(keep); p = len(idx); r_ = row + np.arange(p)
        a_ = ax[idx]
        Aw = np.einsum("ij,ij->i", perp(a_), vec[idx])
        Ao = np.einsum("ij,ij->i", a_, perp(rk[idx]))
        o_, w_ = oth[idx], own[idx]
        rr += [r_] * 6
        cc += [2 * o_, 2 * o_ + 1, 2 * w_, 2 * w_ + 1, 2 * n + w_, 2 * n + o_]
        vv += [-a_[:, 0], -a_[:, 1], a_[:, 0], a_[:, 1], -Aw, -Ao]
        rhs.append(np.maximum(g[idx] - m[idx], 0.0) / U)
        row_a.append(o_); row_b.append(w_)
        row += p
    e = 0.5 * (np.abs(np.cos(th)) + np.abs(np.sin(th)))
    L = TL.side(c, e)
    OX, DL = 3 * n, 3 * n + 2
    for axis in (0, 1):
        lo0 = (c[:, axis] - e).min()
        for k in range(4):
            pos = c[:, axis] + rel[:, k, axis]
            drot = perp(rel[:, k])[:, axis]
            s_lo = pos - lo0
            idx = np.flatnonzero(s_lo < xy + 0.8 * rt + 1e-9); p = len(idx); r_ = row + np.arange(p)
            rr += [r_] * 3; cc += [2 * idx + axis, 2 * n + idx, np.full(p, OX + axis)]
            vv += [-np.ones(p), -drot[idx], np.ones(p)]
            rhs.append(np.maximum(s_lo[idx], 0.0) / U); row += p
            row_a.append(idx); row_b.append(idx)
            s_hi = lo0 + L - pos
            idx = np.flatnonzero(s_hi < xy + 0.8 * rt + 1e-9); p = len(idx); r_ = row + np.arange(p)
            rr += [r_] * 4; cc += [2 * idx + axis, 2 * n + idx, np.full(p, OX + axis), np.full(p, DL)]
            vv += [np.ones(p), drot[idx], -np.ones(p), -np.ones(p)]
            rhs.append(np.maximum(s_hi[idx], 0.0) / U); row += p
            row_a.append(idx); row_b.append(idx)
    A = coo_matrix((np.concatenate(vv), (np.concatenate(rr), np.concatenate(cc))), shape=(row, 3 * n + 3)).tocsr()
    obj = np.zeros(3 * n + 3); obj[DL] = 1.0
    if noise and rng is not None:
        obj[2 * n:3 * n] = rng.normal(0, noise, n)
    bounds = [(-1, 1)] * (2 * n) + [(-rt / U, rt / U)] * n + [(None, None)] * 2 + [(None, 0)]
    res = linprog(obj, A_ub=A, b_ub=np.concatenate(rhs), bounds=bounds, method="highs-ds", options={"time_limit": 60})
    if not res.success:
        return None
    x = res.x * U
    if want_duals:
        y = np.abs(res.ineqlin.marginals)
        mass = np.zeros(n)
        np.add.at(mass, np.concatenate(row_a), y)
        np.add.at(mass, np.concatenate(row_b), y)
        return x[:2 * n].reshape(n, 2), x[2 * n:3 * n], float(x[DL]), mass
    return x[:2 * n].reshape(n, 2), x[2 * n:3 * n], float(x[DL])


def refine(c, th, iters, trust0, seed=0):
    rng = np.random.default_rng(seed)
    cur, gap0 = measure(c, th)
    base = cur
    xy, rt = trust0, trust0 * 0.5
    log = []
    for it in range(iters):
        out = slp_step(c, th, xy, rt, rng, 0.0)
        if out is None:
            xy *= 0.3; rt *= 0.3
            if xy < 1e-9: break
            continue
        dc, dth, pred = out
        if pred > -1e-13:
            log.append((it, xy, pred, "rigid"))
            if xy > 2e-5:                                   # retry once with a small trust region
                xy, rt = 1e-5, 5e-6
                continue
            break
        ok = False
        for alpha in (1.0, 0.5, 0.25, 0.125):
            q, t = c + alpha * dc, th + alpha * dth
            s, g = measure(q, t)
            if g < 0 and REPAIR == "lp":
                # exact repair: with the new angles frozen, the fixed-angle LP is exact in
                # the centres, so it removes the second-order overlap at the least cost in L
                e = 0.5 * (np.abs(np.cos(t)) + np.abs(np.sin(t)))
                for _ in range(3):
                    fix = TL.lp_round(q, t, e, max(4 * xy, 1e-6), repair=True)
                    if fix is None:
                        break
                    q = fix[0]
                    s, g = measure(q, t)
                    if g >= -1e-14:
                        break
            if g < 0:
                mid = 0.5 * (q.min(0) + q.max(0))
                q = mid + (q - mid) * (1.0 + 1.05 * (-g) + 1e-13)
                s, g = measure(q, t)
            if g >= 0 and s < cur - 1e-13:
                c, th, cur, ok = q, t, s, True
                break
        log.append((it, xy, pred, "ok" if ok else "reject"))
        if ok:
            xy = min(xy * 1.5, 0.05); rt = min(rt * 1.5, 0.025)
        else:
            xy *= 0.3; rt *= 0.3
            if xy < 1e-9: break
    return c, th, base, cur, log


def work(task):
    n, grp, iters, trust0 = task
    c0 = np.array([[float(r["x"]), float(r["y"])] for r in grp])
    th0 = np.array([float(r["angle"]) for r in grp])
    try:
        c, th, base, cur, log = refine(c0, th0, iters, trust0, seed=n)
    except Exception as err:  # noqa: BLE001
        return n, dict(n=n, status="error", error=repr(err)[:200]), None
    info = dict(n=n, base=base, slp=cur, delta=cur - base, steps=sum(1 for l in log if l[3] == "ok"),
                first_pred=log[0][2] if log else None)
    if cur >= base - 1e-12:
        info["status"] = "no-gain"
        return n, info, None
    e = 0.5 * (np.abs(np.cos(th)) + np.abs(np.sin(th)))
    for _ in range(6):                                      # squeeze out the dilation slack at fixed angles
        out = TL.lp_round(c, th, e, 1e-5)
        if out is None or TL.pair_axes(out[0], th)[3].min() < -1e-12 or TL.side(out[0], e) >= TL.side(c, e) - 1e-14:
            break
        c = out[0]
    info["polished"] = TL.side(c, e)
    old = R.both(n, grp)
    mid = 0.5 * (c.min(0) + c.max(0))
    new = None
    for dil in (0.0, 2e-14, 2e-13, 2e-12, 2e-11, 2e-10):
        cd = mid + (c - mid) * (1.0 + dil)
        cand = [dict(n=str(n), square_id=r["square_id"], x=repr(float(cd[t, 0])), y=repr(float(cd[t, 1])),
                     angle=repr(float(th[t]))) for t, r in enumerate(grp)]
        new = R.both(n, cand)
        if new is not None:
            info["dilation"] = dil
            break
    if new is None:
        info["status"] = "official-reject"
        return n, info, None
    if not (new[0] < old[0] and new[1] < old[1]):
        info["status"] = "not-strict"
        return n, info, None
    info.update(status="WIN", old_literal=str(old[0]), new_literal=str(new[0]),
                old_pandas=str(old[1]), new_pandas=str(new[1]))
    return n, info, cand


def main():
    path, spec, jobs, out_rows, out_rep = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]
    iters = int(sys.argv[6]) if len(sys.argv) > 6 else 30
    trust0 = float(sys.argv[7]) if len(sys.argv) > 7 else 2e-3
    if "-" in spec:
        lo, hi = map(int, spec.split("-")); ns = list(range(lo, hi + 1))
    else:
        ns = [int(v) for v in spec.split(",")]
    groups = R.read_groups(path, set(ns))
    wins, rep = {}, []
    with mp.get_context("fork").Pool(jobs) as pool:
        for n, info, cand in pool.imap_unordered(work, [(n, groups[n], iters, trust0) for n in sorted(ns, reverse=True)]):
            rep.append(info)
            if cand:
                wins[n] = cand
                print(f"WIN n={n} dL={float(Decimal(info['new_literal']) - Decimal(info['old_literal'])):.3e} steps={info['steps']}", flush=True)
    rep.sort(key=lambda r: r["n"])
    R.write_rows(out_rows, wins)
    json.dump(rep, open(out_rep, "w"), indent=1)
    gain = sum((Decimal(r["old_literal"]) - Decimal(r["new_literal"])) / Decimal(r["n"]).sqrt() for r in rep if r["status"] == "WIN")
    print(Counter(r["status"] for r in rep), "score gain", gain)


if __name__ == "__main__":
    main()
