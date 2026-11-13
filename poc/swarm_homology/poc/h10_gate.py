#!/usr/bin/env python3
"""H10 gate: 4-block (BL, TR, A, B) modular homology transplant + seam masked melt + joint SLP.

Targets: 910 (<- win_n971), 922 (<- win_n862), 863 (<- win_n862 / m923), 805 (<- m749 + outer cap).
"""
from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import utils as u  # noqa: E402
from cpp_wrap import cpp_eval_sat_pen_bbox, cpp_expand_sym, full_to_free_sym, run_cpp_masked_melt  # noqa: E402
from joint_slp import joint_xytheta_slp  # noqa: E402

REPO = HERE.parents[2]
MERGED = REPO / "poc" / "pell_corrugation" / "data" / "border-scan" / "merged.csv"
DATA = HERE.parent / "data"
DOCS = HERE.parent / "docs"


def hx_of(th: np.ndarray) -> np.ndarray:
    return 0.5 * (np.abs(np.cos(th)) + np.abs(np.sin(th)))


def norm(x, y, th):
    x = np.asarray(x, float).copy()
    y = np.asarray(y, float).copy()
    th = np.asarray(th, float).copy()
    h = hx_of(th)
    x -= (x - h).min()
    y -= (y - h).min()
    return x, y, th, float(max((x + h).max(), (y + h).max()))


def load_merged(ns: list[int]) -> dict[int, tuple]:
    df = pd.read_csv(MERGED)
    out = {}
    for n in ns:
        s = df[df["n"] == n]
        out[n] = norm(s["x"].to_numpy(), s["y"].to_numpy(), s["angle"].to_numpy())
    return out


def load_win_from_git(rev: str, path: str, n_filter: int | None = None):
    txt = subprocess.check_output(["git", "-C", str(REPO), "show", f"{rev}:{path}"], text=True)
    d = pd.read_csv(io.StringIO(txt))
    if n_filter is not None:
        d = d[d["id"].str.startswith(f"{n_filter}_")]
    return norm(d["x"].to_numpy(), d["y"].to_numpy(), np.radians(d["deg"].to_numpy()))


def grid(L: float, k: int):
    g = L - k + 0.5 + np.arange(k)
    gx, gy = np.meshgrid(g, g)
    return gx.ravel(), gy.ravel(), np.zeros(k * k)


def overlap_pairs(X: np.ndarray, Y: np.ndarray, T: np.ndarray, tol: float = 1e-9):
    n = len(X)
    c, s = np.cos(T), np.sin(T)
    out = []
    order = np.argsort(X)
    Xs = X[order]
    for a in range(n):
        i = int(order[a])
        b = a + 1
        while b < n and Xs[b] - X[i] < 1.5:
            j = int(order[b])
            b += 1
            dx, dy = X[j] - X[i], Y[j] - Y[i]
            if abs(dy) > 1.5:
                continue
            rad = 0.5 * (1.0 + abs(c[i] * c[j] + s[i] * s[j]) + abs(c[i] * s[j] - s[i] * c[j]))
            pen = min(
                rad - abs(dx * c[i] + dy * s[i]),
                rad - abs(-dx * s[i] + dy * c[i]),
                rad - abs(dx * c[j] + dy * s[j]),
                rad - abs(-dx * s[j] + dy * c[j]),
            )
            if pen > tol:
                out.append((float(pen), i, j))
    return out


def clearance_points(x: np.ndarray, y: np.ndarray, th: np.ndarray, L: float, n_pts: int, step: float = 0.12):
    g = np.arange(0.5, L - 0.5 + 1e-9, step)
    gx, gy = np.meshgrid(g, g, indexing="ij")
    px, py = gx.ravel(), gy.ravel()
    clr = np.minimum(np.minimum(px, L - px), np.minimum(py, L - py)) - 0.5
    c, s = np.cos(th), np.sin(th)
    for a in range(0, len(px), 4096):
        dx = px[a:a + 4096, None] - x[None, :]
        dy = py[a:a + 4096, None] - y[None, :]
        d = np.maximum(np.abs(c * dx + s * dy), np.abs(-s * dx + c * dy)) - 0.5
        clr[a:a + 4096] = np.minimum(clr[a:a + 4096], d.min(axis=1))
    order = np.argsort(-clr)
    out = []
    for i in order:
        if len(out) >= n_pts:
            break
        if all((px[i] - q[0]) ** 2 + (py[i] - q[1]) ** 2 > 0.85 ** 2 for q in out):
            out.append((float(px[i]), float(py[i]), float(clr[i])))
    return out


def seeds_910(W971):
    x, y, t, L971 = W971
    Lt = L971 - 1.0
    keep = (x <= 30.5) & (y <= 30.5)
    x0, y0, t0 = x[keep], y[keep], t[keep]
    assert len(x0) == 910, len(x0)
    out = [("910_971crop_raw", x0, y0, t0, Lt, np.array([[3.5, 29.5]]))]
    # Variant 2: move the single protruding square (idx with max y+hx) into the 3.662 gap around (3.8, 30.162) upright
    hx = hx_of(t0)
    idx_p = int(np.argmax(y0 + hx))
    for k, gx_try in enumerate((2.5, 3.8, 5.1)):
        x1, y1, t1 = x0.copy(), y0.copy(), t0.copy()
        x1[idx_p], y1[idx_p], t1[idx_p] = gx_try, Lt - 0.5, 0.0
        out.append((f"910_971crop_tuck{k}", x1, y1, t1, Lt, np.array([[3.5, 29.5]])))
    return out


def seeds_922(M, W862):
    x9, y9, t9, L9 = M[922]
    x8, y8, t8, L8 = W862
    Lt = L8 + 1.0
    bl = (x9 < 9) & (y9 < 9)
    a = (x8 < 9) & (y8 > 8)
    b = (x9 > 9) & (y9 < 9)
    gx, gy, gt = grid(Lt, 22)
    X1 = np.concatenate([x9[bl], x8[a], gx, x9[b] - (L9 - Lt)])
    Y1 = np.concatenate([y9[bl], y8[a] + 1.0, gy, y9[b]])
    T1 = np.concatenate([t9[bl], t8[a], gt, t9[b]])
    # Seed 2: flipped w862(x<9, y>8) for B
    xBf = (Lt + 8.0) - y8[a]
    yBf = (Lt - 22.0) - x8[a]
    tBf = 0.5 * np.pi - t8[a]
    X2 = np.concatenate([x9[bl], x8[a], gx, xBf])
    Y2 = np.concatenate([y9[bl], y8[a] + 1.0, gy, yBf])
    T2 = np.concatenate([t9[bl], t8[a], gt, tBf])
    return [
        ("922_w862arm_m922B", X1, Y1, T1, Lt, np.array([[8.5, 8.33], [15.0, 8.5]])),
        ("922_w862arm_flipB", X2, Y2, T2, Lt, np.array([[8.5, 8.33], [15.0, 8.5]])),
    ]


def seeds_863(M, W862):
    out = []
    for name, (x, y, t, L) in (("863a_w862sym", W862), ("863b_m862sym", M[862])):
        bl = (x < 9) & (y < 9)
        a = (x < 9) & (y > 9)
        tr = (x > 9) & (y > 9)
        X = np.concatenate([x[bl], x[a], y[a], x[tr]])
        Y = np.concatenate([y[bl], y[a], x[a], y[tr]])
        T = np.concatenate([t[bl], t[a], 0.5 * np.pi - t[a], t[tr]])
        if len(X) == 864:
            ov = overlap_pairs(X, Y, T)
            cnt = np.zeros(len(X))
            for p, i, j in ov:
                cnt[i] += p
                cnt[j] += p
            k = int(np.argmax(cnt))
            keep = np.ones(len(X), bool)
            keep[k] = False
            X, Y, T = X[keep], Y[keep], T[keep]
        out.append((name, X, Y, T, L, np.array([[8.5, 8.33]])))

    # 863c/d: strip 923 outer border -> 864 -> drop 1 or 2 protruding squares + 0 or 1 void insert
    x23, y23, t23, L23 = M[923]
    Lt = L23 - 1.0
    keep = np.ones(len(x23), bool)
    keep[(x23 < 9) & (y23 > 29.8)] = False
    keep[(y23 < 9) & (x23 > 29.8)] = False
    keep[(x23 > 9) & (y23 > 9) & ((x23 > 29.8) | (y23 > 29.8))] = False
    hx23 = hx_of(t23)
    protrude = np.maximum(x23 + hx23, y23 + hx23)
    idx_k = np.where(keep)[0]
    order_p = idx_k[np.argsort(-protrude[idx_k])]
    # 863c: drop worst protruding (1)
    k1 = keep.copy()
    k1[order_p[0]] = False
    out.append(("863c_923strip1", x23[k1], y23[k1], t23[k1], Lt, np.array([[29.0, 5.4], [3.5, 29.0]])))
    # 863d: drop both protruding (2) -> 862 feasible at Lt, insert 1 in top void
    k2 = keep.copy()
    k2[order_p[:2]] = False
    X862, Y862, T862 = x23[k2], y23[k2], t23[k2]
    vp = clearance_points(X862, Y862, T862, Lt, 3)
    for s, (vx, vy, _) in enumerate(vp[:2]):
        out.append((
            f"863d_923strip2_v{s}",
            np.append(X862, vx),
            np.append(Y862, vy),
            np.append(T862, 0.0),
            Lt,
            np.array([[vx, vy]]),
        ))
    return out


def seeds_805(M):
    x7, y7, t7, L7 = M[749]
    Lt = L7 + 1.0
    bl7 = (x7 < 9) & (y7 < 9)
    a7 = (x7 < 9) & (y7 > 9)
    b7 = (x7 > 9) & (y7 < 9)
    capA_x = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5])
    capA_y = np.full(8, Lt - 0.5)
    capB_x = np.full(8, Lt - 0.5)
    capB_y = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5])
    gx20, gy20, gt20 = grid(Lt, 20)
    X804 = np.concatenate([x7[bl7], x7[a7], capA_x, x7[b7], capB_x, gx20])
    Y804 = np.concatenate([y7[bl7], y7[a7], capA_y, y7[b7], capB_y, gy20])
    T804 = np.concatenate([t7[bl7], t7[a7], np.zeros(8), t7[b7], np.zeros(8), gt20])
    vp = clearance_points(X804, Y804, T804, Lt, 4)
    out = []
    for s, (vx, vy, _) in enumerate(vp[:3]):
        out.append((
            f"805_749cap_v{s}",
            np.append(X804, vx),
            np.append(Y804, vy),
            np.append(T804, 0.4 * (s % 2)),
            Lt,
            np.array([[vx, vy], [8.5, 8.33]]),
        ))
    return out


def build_jobs(name, X, Y, T, Lt, extra_centers, L_inc, rng, n_seeds=8):
    n = len(X)
    pen, bb = u.sat_max_overlap_and_bbox(X, Y, T)
    ov = overlap_pairs(X, Y, T)
    hot = np.zeros(n, bool)
    for _, i, j in ov:
        hot[i] = hot[j] = True
    pts = [np.c_[X[hot], Y[hot]]] if hot.any() else []
    if len(extra_centers):
        pts.append(extra_centers)
    centers = np.vstack(pts) if pts else np.array([[Lt / 2, Lt / 2]])

    # CRITICAL: do NOT stretch X, Y to L0! Keep unperturbed exterior at Lt scale centered inside [0, L0]^2.
    # That way when L shrinks from L0 down to Lt (or L_inc), frozen squares stay inside [0, L]^2.
    L_ref = max(Lt, bb if bb < Lt + 0.005 else Lt)
    L0 = L_ref + 0.032
    off = 0.5 * (L0 - L_ref)
    xb = X + off
    yb = Y + off
    h = hx_of(T)
    xb = np.clip(xb, h + 1e-4, L0 - h - 1e-4)
    yb = np.clip(yb, h + 1e-4, L0 - h - 1e-4)

    # Two L schedules: tight Lt, and slightly relaxed below L_inc
    L_mid = min(L_inc - 2e-4, max(Lt + 0.002, 0.5 * (Lt + L_inc))) if Lt < L_inc - 1e-3 else Lt + 0.004
    sched1 = [L0, max(L_inc, Lt) + 0.006, L_mid, Lt]
    sched2 = [L0, max(L_inc, Lt) + 0.004, min(L_inc - 5e-4, Lt + 0.003) if Lt < L_inc - 1e-3 else Lt + 0.001,
              min(L_inc - 1e-3, Lt + 0.001) if Lt < L_inc - 2e-3 else Lt]

    q, m, ls, lab = [], [], [], []
    for s in range(n_seeds):
        R = float(rng.uniform(4.5, 7.2))
        d2 = ((xb[:, None] - (centers[None, :, 0] + off)) ** 2 + (yb[:, None] - (centers[None, :, 1] + off)) ** 2).min(axis=1)
        near = d2 <= R * R
        # Also ensure protruding squares (if any) are always active
        near |= (xb - h < off) | (xb + h > L0 - off) | (yb - h < off) | (yb + h > L0 - off)
        x2, y2, t2 = xb.copy(), yb.copy(), T.copy()
        k = int(near.sum())
        jit = 0.0 if s == 0 else 0.09
        if jit > 0:
            x2[near] += rng.uniform(-jit, jit, k)
            y2[near] += rng.uniform(-jit, jit, k)
            t2[near] += rng.uniform(-0.22, 0.22, k)
        q.append(full_to_free_sym(x2, y2, t2, L0))
        mm = near.astype(np.uint8)
        m.append(np.concatenate([mm, mm, mm]))
        ls.append(sched1 if s % 2 == 0 else sched2)
        lab.append(f"{name}_R{R:.1f}_s{s}")
    return q, m, ls, lab, pen, len(ov)


def polish(n: int, x: np.ndarray, y: np.ndarray, th: np.ndarray, L_inc: float):
    pen, bb = cpp_eval_sat_pen_bbox(x, y, th)
    if pen > 1e-7 and pen < 0.05:
        x, y, th, pen = u.relax_lbfgs(x, y, th, bb + max(0.002, 2.0 * pen), maxiter=220, w_wall=4.0)
    pen, bb = cpp_eval_sat_pen_bbox(x, y, th)
    if pen > 0.02:
        return None
    xs, ys, ts, Ls, ok = joint_xytheta_slp(x, y, th, rounds=10, tr_th0=0.02, tr_xy0=0.09, margin=1e-8, tol=1e-8, patience=2)
    if not ok:
        return None
    if Ls < L_inc + 5e-4:
        xs2, ys2, ts2, Ls2, ok2 = joint_xytheta_slp(xs, ys, ts, rounds=12, tr_th0=0.006, tr_xy0=0.025, margin=1e-8, tol=1e-11, patience=2)
        if ok2 and Ls2 < Ls:
            xs, ys, ts, Ls = xs2, ys2, ts2, Ls2
    best = None
    for mlp in (1e-7, 2e-8, 5e-9):
        xf, yf, Lf, okf = u.fixed_angle_lp_tighten(xs, ys, ts, rounds=5, margin=mlp)
        if not okf:
            continue
        v = u.verify_official(n, xf, yf, ts)
        if v.get("ok") and (best is None or v["L_float"] < best["L_float"]):
            best = v
    if best is None:
        v = u.verify_official(n, xs, ys, ts)
        best = v if v.get("ok") else None
    return best


def run_target(n: int, seeds, L_inc: float, threads: int):
    rng = np.random.default_rng(1010 + n)
    t0 = time.time()
    allq, allm, alls, alll, info = [], [], [], [], {}
    for name, X, Y, T, Lt, ctrs in seeds:
        q, m, ls, lab, pen, nov = build_jobs(name, X, Y, T, Lt, ctrs, L_inc, rng, n_seeds=8)
        info[name] = {"seed_pen": float(pen), "n_overlaps": nov, "L_seed": float(Lt)}
        print(f"[{n}] seed {name}: pen={pen:.3e} overlaps={nov} L_seed={Lt:.9f}", flush=True)
        allq += q
        allm += m
        alls += ls
        alll += lab

    qb, qf, lf, eo, po, lb = run_cpp_masked_melt(
        n, 1, allq, allm, alls, [5, 8, 12, 15],
        a_start=0.78, n_melt_stages=4, masked_iters=180, full_iters=200,
        n_threads=threads, base_seed=91000 + n,
    )
    n_feas = int((lf < 1e9).sum())
    print(f"[{n}] melt done {time.time()-t0:.1f}s feas={n_feas}/{len(lf)} "
          f"best_Lf={float(lf.min()):.9f} min_pen={float(po.min()):.2e}", flush=True)

    order = sorted(range(len(alll)), key=lambda i: (0 if lf[i] < 1e9 else 1, lf[i] if lf[i] < 1e9 else po[i]))[:6]
    res = []
    best = None
    for i in order:
        use_f = lf[i] < 1e9
        Lr = float(lf[i]) if use_f else alls[i][-1]
        x, y, th = cpp_expand_sym(n, 1, Lr, qf[i] if use_f else qb[i])
        v = polish(n, x, y, th, L_inc)
        Lv = None if v is None else v["L_float"]
        dl = None if Lv is None else Lv - L_inc
        print(f"[{n}]   {alll[i]}: cppLf={lf[i]:.9f} pen={po[i]:.2e} -> polished L={Lv} dL={dl}", flush=True)
        res.append({"label": alll[i], "cpp_Lf": float(lf[i]), "pen": float(po[i]), "L": Lv, "dL": dl})
        if v is not None and (best is None or Lv < best[0]):
            best = (Lv, alll[i], v)

    out = {
        "n": n, "L_inc": L_inc, "seeds": info, "polished": res, "dt": round(time.time() - t0, 2),
        "best_L": None if best is None else best[0],
        "best_dL": None if best is None else best[0] - L_inc,
        "best_label": None if best is None else best[1],
    }
    if best is not None and best[0] - L_inc <= -1e-9:
        DATA.mkdir(parents=True, exist_ok=True)
        u.write_final_csv_from_wins({n: best[2]}, DATA / f"win_n{n}.csv")
        print(f"[{n}] *** REAL WIN *** {best[1]} L={best[0]:.15f} dL={best[0]-L_inc:+.6e}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", type=int, nargs="+", default=[910, 922, 863, 805])
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    M = load_merged([749, 805, 862, 863, 910, 922, 923])
    W862 = load_win_from_git("2493969", "poc/swarm_highd/data/win_n862.csv")
    W971 = load_win_from_git("50badc8", "poc/swarm_inflation/final.csv", n_filter=971)

    DOCS.mkdir(parents=True, exist_ok=True)
    all_out = {}
    for n in a.targets:
        L_inc = M[n][3]
        if n == 910:
            seeds = seeds_910(W971)
        elif n == 922:
            seeds = seeds_922(M, W862)
        elif n == 863:
            seeds = seeds_863(M, W862)
        elif n == 805:
            seeds = seeds_805(M)
        else:
            raise ValueError(n)
        res = run_target(n, seeds, L_inc, min(4, max(1, a.threads)))
        all_out[n] = res
        (DOCS / f"gate_{n}.json").write_text(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    main()
