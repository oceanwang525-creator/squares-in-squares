#!/usr/bin/env python3
"""Idea 1: Multi-Track High-D Basin Transfer + Pocket Melt + Joint (x, y, dtheta) SLP.

Tracks:
  A. surplus_peel: Multi-step (c=1..8) wall-split & interior cross-cut surplus peels (rem in [n, n+4]).
  B. del_n1_n2: High-contact / boundary-pin square deletion from merged[n+1], merged[n+2] + local pocket melt.
  C. inc_melt: 6-mode incumbent pocket melts (pin-teleport, boundary-pin, seam-unlock, void-collapse,
     wall-slack drift, upright-block shift) + direct joint_xytheta_slp on incumbent.
  D. l_corridor_and_m1: Multi-angle sheared L-corridor synthesis (P=5..7, 18..27 deg) & (n-1)+1 void insert.
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import utils as u
from cpp_wrap import cpp_eval_sat_pen_bbox, cpp_expand_sym, full_to_free_sym, run_cpp_batch_ils
from joint_slp import joint_xytheta_slp

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
MERGED_CSV = REPO_ROOT / "poc" / "pell_corrugation" / "data" / "border-scan" / "merged.csv"
OUT_JSON = HERE.parent / "docs" / "idea1_results.json"
FINAL_CSV = HERE.parent / "final.csv"
DATA_DIR = HERE.parent / "data"

OWNED_TARGETS = {269, 918, 545, 237, 919, 862, 805, 921, 698, 703, 639, 981, 860, 339}


def _norm(sub: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    x = sub["x"].to_numpy(np.float64).copy()
    y = sub["y"].to_numpy(np.float64).copy()
    th = sub["angle"].to_numpy(np.float64).copy()
    hx = 0.5 * (np.abs(np.cos(th)) + np.abs(np.sin(th)))
    x -= float((x - hx).min())
    y -= float((y - hx).min())
    L = float(max((x + hx).max(), (y + hx).max()))
    return x, y, th, L


def clearance_points(x: np.ndarray, y: np.ndarray, th: np.ndarray, L: float, n_pts: int, step: float = 0.12) -> list[tuple[float, float, float]]:
    """Return top n_pts (px, py, clearance) maxima to all squares and walls."""
    g = np.arange(0.5, L - 0.5 + 1e-9, step)
    gx, gy = np.meshgrid(g, g, indexing="ij")
    px, py = gx.ravel(), gy.ravel()
    clr = np.minimum(np.minimum(px, L - px), np.minimum(py, L - py)) - 0.5
    c, s = np.cos(th), np.sin(th)
    for a in range(0, len(px), 4096):
        dx = px[a : a + 4096, None] - x[None, :]
        dy = py[a : a + 4096, None] - y[None, :]
        d = np.maximum(
            np.abs(c[None, :] * dx + s[None, :] * dy),
            np.abs(-s[None, :] * dx + c[None, :] * dy),
        ) - 0.5
        clr[a : a + 4096] = np.minimum(clr[a : a + 4096], d.min(axis=1))
    order = np.argsort(-clr)
    out: list[tuple[float, float, float]] = []
    for i in order:
        if len(out) >= n_pts:
            break
        if all((px[i] - q[0]) ** 2 + (py[i] - q[1]) ** 2 > 0.85 ** 2 for q in out):
            out.append((float(px[i]), float(py[i]), float(clr[i])))
    return out


def donor_tilt(th: np.ndarray) -> float:
    deg = np.degrees(th) % 90.0
    t = deg[(deg > 1.0) & (deg < 89.0)]
    return math.radians(float(np.median(t))) if len(t) else math.radians(22.5)


def _peel_mask_split(x: np.ndarray, y: np.ndarray, L: float, cL: int, cR: int, cB: int, cT: int) -> np.ndarray:
    keep = np.ones(len(x), dtype=bool)
    if cL > 0:
        keep &= x >= cL + 0.05
    if cR > 0:
        keep &= x <= L - cR - 0.05
    if cB > 0:
        keep &= y >= cB + 0.05
    if cT > 0:
        keep &= y <= L - cT - 0.05
    return keep


def collect_surplus_peels(n: int, L_inc: float, mats: dict[int, tuple]) -> list[dict]:
    """Collect multi-step wall-split & interior cross-cut surplus peels (rem in [n, n+4])."""
    k = math.isqrt(n)
    cands: list[dict] = []
    seen = set()

    for n2, (x, y, th, L) in mats.items():
        if n2 <= n:
            continue
        c = round(L - L_inc)
        if c < 1 or c > 8:
            continue
        L_box = L - c
        if not (L_inc - 0.16 <= L_box <= L_inc - 0.0004):
            continue
        hx = 0.5 * (np.abs(np.cos(th)) + np.abs(np.sin(th)))
        for cL in range(c + 1):
            cR = c - cL
            for cB in range(c + 1):
                cT = c - cB
                keep = _peel_mask_split(x, y, L, cL, cR, cB, cT)
                nr = int(keep.sum())
                if not (n <= nr <= n + 4):
                    continue
                xs = x[keep] - cL
                ys = y[keep] - cB
                ts = th[keep]
                h2 = hx[keep]
                oh = np.maximum.reduce([
                    np.zeros(nr), h2 - xs, xs + h2 - L_box, h2 - ys, ys + h2 - L_box,
                ])
                surp = nr - n
                max_oh = float(oh.max())
                max_oh_rem = float(np.partition(oh, -(surp + 1))[-(surp + 1)]) if surp > 0 else max_oh
                if max_oh_rem >= 0.28:
                    continue
                key = (n2, nr, round(max_oh_rem, 4), round(L_box, 6))
                if key in seen:
                    continue
                seen.add(key)
                cands.append({
                    "src": f"s{n2}_{cL}{cR}_{cB}{cT}",
                    "x": xs, "y": ys, "th": ts, "oh": oh,
                    "L": L_box, "surplus": surp, "max_oh": max_oh, "max_oh_rem": max_oh_rem,
                })

        if c == 1 and n2 <= n + 3 * k:
            for xc in np.linspace(2.5, L - 2.5, 5):
                kx = (x < xc - 0.5) | (x > xc + 0.5)
                for yc in np.linspace(2.5, L - 2.5, 5):
                    keep = kx & ((y < yc - 0.5) | (y > yc + 0.5))
                    nr = int(keep.sum())
                    if not (n <= nr <= n + 4):
                        continue
                    xs = np.where(x[keep] > xc + 0.5, x[keep] - 1.0, x[keep])
                    ys = np.where(y[keep] > yc + 0.5, y[keep] - 1.0, y[keep])
                    ts = th[keep]
                    h2 = hx[keep]
                    seam_d = np.minimum(np.abs(xs - (xc - 0.5)), np.abs(ys - (yc - 0.5)))
                    oh = np.maximum(0.0, 0.75 - seam_d) + np.maximum.reduce([
                        np.zeros(nr), h2 - xs, xs + h2 - L_box, h2 - ys, ys + h2 - L_box,
                    ])
                    surp = nr - n
                    max_oh = float(oh.max())
                    max_oh_rem = float(np.partition(oh, -(surp + 1))[-(surp + 1)]) if surp > 0 else max_oh
                    if max_oh_rem < 0.45:
                        key = (n2, nr, round(L_box, 6), round(max_oh_rem, 3))
                        if key not in seen:
                            seen.add(key)
                            cands.append({
                                "src": f"x{n2}_{int(xc)}_{int(yc)}",
                                "x": xs, "y": ys, "th": ts, "oh": oh,
                                "L": L_box, "surplus": surp, "max_oh": max_oh, "max_oh_rem": max_oh_rem,
                            })

    cands.sort(key=lambda r: (r["max_oh_rem"], r["surplus"], r["L"]))
    return cands


def _surplus_drop_seeds(don: dict, n: int, L0: float, n_seeds: int, rng: np.random.Generator) -> list[np.ndarray]:
    x0, y0, th0, oh0 = don["x"], don["y"], don["th"], don["oh"]
    L_box = don["L"]
    surplus = don["surplus"]
    hx0 = 0.5 * (np.abs(np.cos(th0)) + np.abs(np.sin(th0)))

    rank = np.argsort(-oh0)
    pool_idx = rank[: max(surplus + 5, min(len(rank), 12))]
    seeds = []
    sc = L0 / L_box
    for s in range(n_seeds):
        if surplus == 0:
            keep = np.ones(len(x0), dtype=bool)
        elif s == 0:
            drop = rank[:surplus]
            keep = np.ones(len(x0), dtype=bool)
            keep[drop] = False
        else:
            drop = rng.choice(pool_idx, size=surplus, replace=False)
            keep = np.ones(len(x0), dtype=bool)
            keep[drop] = False

        x = x0[keep] * sc
        y = y0[keep] * sc
        th = th0[keep].copy()
        hx = hx0[keep]
        oh = oh0[keep]
        x = np.clip(x, hx + 0.005, L0 - hx - 0.005)
        y = np.clip(y, hx + 0.005, L0 - hx - 0.005)
        hot = np.where(oh > 0.01)[0]
        if len(hot) > 0 and s > 0:
            cx, cy = float(x[hot[s % len(hot)]]), float(y[hot[s % len(hot)]])
            near = ((x - cx) ** 2 + (y - cy) ** 2) <= 4.0 ** 2
            nm = int(near.sum())
            x[near] = np.clip(x[near] + rng.uniform(-0.12, 0.12, nm), hx[near] + 0.005, L0 - hx[near] - 0.005)
            y[near] = np.clip(y[near] + rng.uniform(-0.12, 0.12, nm), hx[near] + 0.005, L0 - hx[near] - 0.005)
            th[near] += rng.uniform(-0.10, 0.10, nm)
        seeds.append(full_to_free_sym(x, y, th, L0))
    return seeds


def _del_n1_n2_seeds(mats: dict, n: int, L_inc: float, L0: float, n_seeds: int, rng: np.random.Generator) -> list[tuple[str, np.ndarray]]:
    """Delete 1 or 2 high-contact / boundary-tight squares from merged[n+1] and merged[n+2] + local melt."""
    seeds: list[tuple[str, np.ndarray]] = []
    for dn in (1, 2):
        ns = n + dn
        if ns not in mats:
            continue
        x0, y0, th0, L_src = mats[ns]
        if L_src > L_inc + 0.06:
            continue
        sc = L0 / L_src
        xb, yb, tb = x0 * sc, y0 * sc, th0.copy()
        hx = 0.5 * (np.abs(np.cos(tb)) + np.abs(np.sin(tb)))
        # Score squares by boundary tightness + tilted pocket membership
        w_margin = np.minimum.reduce([xb - hx, L0 - xb - hx, yb - hx, L0 - yb - hx])
        deg = np.degrees(tb) % 90.0
        is_tilt = (deg > 2.0) & (deg < 88.0)
        # Prefer boundary-tight or tilted-pocket-end squares
        score = w_margin - 0.05 * is_tilt.astype(np.float64)
        cand_idx = np.argsort(score)[:16]
        for j in range(n_seeds // 2):
            if dn == 1:
                drop = [int(cand_idx[j % len(cand_idx)])]
            else:
                if j == 0:
                    drop = [int(cand_idx[0]), int(cand_idx[1])]
                else:
                    drop = list(rng.choice(cand_idx[:12], size=2, replace=False))
            cx, cy = float(xb[drop[0]]), float(yb[drop[0]])
            keep = np.ones(ns, dtype=bool)
            keep[drop] = False
            x, y, th = xb[keep].copy(), yb[keep].copy(), tb[keep].copy()
            near = ((x - cx) ** 2 + (y - cy) ** 2) <= 4.8 ** 2
            nm = int(near.sum())
            x[near] += rng.uniform(-0.14, 0.14, nm)
            y[near] += rng.uniform(-0.14, 0.14, nm)
            th[near] += rng.uniform(-0.14, 0.14, nm)
            seeds.append((f"del{ns}_{j}", full_to_free_sym(x, y, th, L0)))
    return seeds


def _incumbent_bottleneck_seeds(
    x_inc: np.ndarray, y_inc: np.ndarray, th_inc: np.ndarray, L_inc: float, L0: float,
    n_seeds: int, rng: np.random.Generator,
) -> list[np.ndarray]:
    """6-mode pocket melts: pin-teleport, wall-pin, seam-unlock, void-collapse, wall-slack drift, upright-block shift."""
    n = len(x_inc)
    sc = L0 / L_inc
    xb, yb, tb = x_inc * sc, y_inc * sc, th_inc.copy()
    hx = 0.5 * (np.abs(np.cos(tb)) + np.abs(np.sin(tb)))
    margin = np.minimum.reduce([xb - hx, L0 - xb - hx, yb - hx, L0 - yb - hx])
    tight_idx = np.argsort(margin)[:20]
    voids = clearance_points(xb, yb, tb, L0, n_pts=8, step=0.14)
    tilt = donor_tilt(tb)

    deg = np.degrees(tb) % 90.0
    deg_m = np.minimum(deg, 90.0 - deg)
    up_mask = deg_m < 1.0
    tilted_mask = ~up_mask
    tilted_idx = np.where(tilted_mask)[0]

    # Identify seam squares (upright adjacent to tilted, or tilted adjacent to upright)
    seam_idx = np.array([], dtype=int)
    if np.any(up_mask) and np.any(tilted_mask):
        dx_ut = xb[tilted_mask, None] - xb[None, up_mask]
        dy_ut = yb[tilted_mask, None] - yb[None, up_mask]
        d_ut = np.hypot(dx_ut, dy_ut)
        close_t = tilted_idx[np.where(d_ut.min(axis=1) < 1.35)[0]]
        seam_idx = close_t if len(close_t) > 0 else tilted_idx

    mL = xb - hx - float((xb - hx).min())
    mR = float((xb + hx).max()) - (xb + hx)
    mB = yb - hx - float((yb - hx).min())
    mT = float((yb + hx).max()) - (yb + hx)
    cL, cR = int(np.sum(mL < 5e-4)), int(np.sum(mR < 5e-4))
    cB, cT = int(np.sum(mB < 5e-4)), int(np.sum(mT < 5e-4))
    dx_sign = 1.0 if cR <= cL else -1.0
    dy_sign = 1.0 if cT <= cB else -1.0

    seeds = []
    for s in range(n_seeds):
        x, y, th = xb.copy(), yb.copy(), tb.copy()
        mode = s % 6
        if mode == 0 and len(voids) > 0:
            k_move = 1 if (s // 6) % 2 == 0 else min(2, len(voids))
            picks = rng.choice(tight_idx[:12], size=k_move, replace=False)
            for j, pi in enumerate(picks):
                vx, vy, _ = voids[(s + j) % len(voids)]
                x[pi] = vx + float(rng.uniform(-0.08, 0.08))
                y[pi] = vy + float(rng.uniform(-0.08, 0.08))
                th[pi] = [0.0, math.pi / 4, tilt][(s + j) % 3] + float(rng.uniform(-0.08, 0.08))
                near = ((x - vx) ** 2 + (y - vy) ** 2) <= 4.5 ** 2
                nm = int(near.sum())
                x[near] += rng.uniform(-0.12, 0.12, nm)
                y[near] += rng.uniform(-0.12, 0.12, nm)
                th[near] += rng.uniform(-0.10, 0.10, nm)
        elif mode == 1:
            pi = int(tight_idx[(s // 6) % len(tight_idx)])
            cx, cy = float(x[pi]), float(y[pi])
            rad = float(rng.uniform(4.5, 7.2))
            near = ((x - cx) ** 2 + (y - cy) ** 2) <= rad * rad
            nm = int(near.sum())
            x[near] += rng.uniform(-0.14, 0.14, nm)
            y[near] += rng.uniform(-0.14, 0.14, nm)
            th[near] += rng.uniform(-0.16, 0.16, nm)
        elif mode == 2:
            pool_s = seam_idx if len(seam_idx) > 0 else tilted_idx
            if len(pool_s) > 0:
                pi = int(pool_s[(s * 7) % len(pool_s)])
                cx, cy = float(x[pi]), float(y[pi])
                near = ((x - cx) ** 2 + (y - cy) ** 2) <= float(rng.uniform(5.0, 7.8)) ** 2
            else:
                near = np.ones(n, dtype=bool)
            nm = int(near.sum())
            x[near] += rng.uniform(-0.14, 0.14, nm)
            y[near] += rng.uniform(-0.14, 0.14, nm)
            th[near] += rng.uniform(-0.16, 0.16, nm)
        elif mode == 3 and len(voids) > 0:
            vx, vy, _ = voids[(s // 6) % len(voids)]
            rad = float(rng.uniform(4.8, 7.8))
            near = ((x - vx) ** 2 + (y - vy) ** 2) <= rad * rad
            nm = int(near.sum())
            pull = float(rng.uniform(0.03, 0.08))
            x[near] += pull * (vx - x[near]) + rng.uniform(-0.12, 0.12, nm)
            y[near] += pull * (vy - y[near]) + rng.uniform(-0.12, 0.12, nm)
            th[near] += rng.uniform(-0.16, 0.16, nm)
        elif mode == 4:
            drift = float(rng.uniform(0.03, 0.10))
            cx, cy = (x[tight_idx[0]], y[tight_idx[0]]) if len(voids) == 0 else (
                0.5 * (float(x[tight_idx[s % 6]]) + voids[0][0]),
                0.5 * (float(y[tight_idx[s % 6]]) + voids[0][1]),
            )
            half_x = (x < L0 * 0.65) if dx_sign > 0 else (x > L0 * 0.35)
            half_y = (y < L0 * 0.65) if dy_sign > 0 else (y > L0 * 0.35)
            x[half_x] += dx_sign * drift
            y[half_y] += dy_sign * 0.6 * drift
            near = ((x - cx) ** 2 + (y - cy) ** 2) <= float(rng.uniform(5.2, 8.2)) ** 2
            nm = int(near.sum())
            x[near] += rng.uniform(-0.13, 0.13, nm)
            y[near] += rng.uniform(-0.13, 0.13, nm)
            th[near] += rng.uniform(-0.14, 0.14, nm)
        else:
            # Mode 5: Rigid upright-core shift toward weak walls + tilted corridor angle unlock
            drift = float(rng.uniform(0.04, 0.11))
            if np.any(up_mask):
                x[up_mask] += dx_sign * drift
                y[up_mask] += dy_sign * drift
            if len(tilted_idx) > 0:
                nt = len(tilted_idx)
                x[tilted_idx] += rng.uniform(-0.12, 0.12, nt)
                y[tilted_idx] += rng.uniform(-0.12, 0.12, nt)
                th[tilted_idx] += rng.uniform(-0.15, 0.15, nt)
        seeds.append(full_to_free_sym(x, y, th, L0))
    return seeds


def _corridor_and_m1_seeds(
    mats: dict, n: int, L_inc: float, L0: float, n_seeds: int, rng: np.random.Generator
) -> list[tuple[str, np.ndarray]]:
    """Multi-angle sheared L-corridor synthesis (P=5..7) + (n-1)+1 void insertions."""
    seeds: list[tuple[str, np.ndarray]] = []
    k = math.isqrt(n)

    # 1. Parametric (k-P) x (k-P) upright core + tilted L-corridor
    for P in (5, 6, 7):
        ku = k - P
        if ku < 8:
            continue
        rem = n - ku * ku
        # Build upright block in bottom-left [0, ku]^2
        gx, gy = np.meshgrid(np.arange(ku) + 0.5, np.arange(ku) + 0.5, indexing="ij")
        ux, uy = gx.ravel(), gy.ravel()
        uth = np.zeros(ku * ku, dtype=np.float64)

        for deg_val in (18.4, 21.8, 24.5, 26.6):
            ang = math.radians(deg_val)
            # Sample candidate centers in the L-strip [0, L0]^2 \ [0, ku]^2
            step = 0.88
            pts = []
            for px in np.arange(0.55, L0 - 0.55, step):
                for py in np.arange(0.55, L0 - 0.55, step):
                    if px < ku + 0.45 and py < ku + 0.45:
                        continue
                    pts.append((float(px), float(py)))
            if len(pts) < rem:
                continue
            pts_arr = np.array(pts, dtype=np.float64)
            # Keep `rem` closest to outer boundary or evenly spaced
            if len(pts_arr) > rem:
                pick = np.linspace(0, len(pts_arr) - 1, rem, dtype=int)
                pts_arr = pts_arr[pick]
            tx = pts_arr[:, 0] + rng.uniform(-0.10, 0.10, rem)
            ty = pts_arr[:, 1] + rng.uniform(-0.10, 0.10, rem)
            # Alternate angle in top vs right arm
            tth = np.where(tx < ku + 0.5, ang, math.pi / 2 - ang) + rng.uniform(-0.08, 0.08, rem)
            x = np.concatenate([ux, tx])
            y = np.concatenate([uy, ty])
            th = np.concatenate([uth, tth])
            seeds.append((f"lcorr_P{P}_a{int(round(deg_val))}", full_to_free_sym(x, y, th, L0)))

    # 2. (n-1)+1 void insertion
    if (n - 1) in mats:
        x1, y1, th1, L1 = mats[n - 1]
        if L1 < L_inc:
            sc = (L0 - 1.0) / max(1e-9, L1 - 1.0)
            xb = 0.5 + (x1 - 0.5) * sc
            yb = 0.5 + (y1 - 0.5) * sc
            pts = clearance_points(xb, yb, th1, L0, n_pts=6, step=0.12)
            tilt = donor_tilt(th1)
            angs = [0.0, math.pi / 4, tilt]
            for s in range(min(n_seeds, 12)):
                px, py, _ = pts[s % len(pts)]
                xs = np.append(xb, px + float(rng.normal(0, 0.05)))
                ys = np.append(yb, py + float(rng.normal(0, 0.05)))
                ts = np.append(th1, angs[(s // max(1, len(pts))) % len(angs)] + float(rng.normal(0, 0.04)))
                near = ((xs - px) ** 2 + (ys - py) ** 2) <= 3.8 ** 2
                nm = int(near.sum())
                xs[near] += rng.uniform(-0.10, 0.10, nm)
                ys[near] += rng.uniform(-0.10, 0.10, nm)
                seeds.append((f"m1_{s}", full_to_free_sym(xs, ys, ts, L0)))
    return seeds


def _polish_and_repair_worker(args: tuple) -> dict:
    """Finish a C++ basin with joint_xytheta_slp + multi-margin LP tighten + exact verification."""
    idx, label, fam, n, x, y, th, cpp_E, cpp_pen, cpp_L = args
    t0 = time.time()

    hx = 0.5 * (np.abs(np.cos(th)) + np.abs(np.sin(th)))
    x = x - float((x - hx).min())
    y = y - float((y - hx).min())

    if cpp_pen > 1e-7 and cpp_pen < 0.05:
        L_try = cpp_L + max(0.002, 1.8 * cpp_pen)
        xr, yr, thr, _ = u.relax_lbfgs(x, y, th, L_try, maxiter=240, w_wall=4.0, w_pair=1.0, side=1.0)
        pen_r, L_r = cpp_eval_sat_pen_bbox(xr, yr, thr, side=1.0)
        if pen_r <= cpp_pen + 1e-6:
            x, y, th, cpp_pen, cpp_L = xr, yr, thr, pen_r, L_r

    pen_pre, L_pre = cpp_eval_sat_pen_bbox(x, y, th, side=1.0)
    ok_slp, L_slp = False, L_pre
    x_f, y_f, th_f = x.copy(), y.copy(), th.copy()

    if pen_pre < 0.04:
        xs, ys, ts, Ls, ok_s = joint_xytheta_slp(x, y, th, rounds=18, tr_th0=0.03, tr_xy0=0.12, margin=1e-8)
        if ok_s:
            x_f, y_f, th_f, L_slp, ok_slp = xs, ys, ts, Ls, True

    ver: dict = {"ok": False}
    for m_lp in (1e-9, 5e-8, 2e-7, 1e-6):
        xf2, yf2, L_lp2, ok_lp2 = u.fixed_angle_lp_tighten(x_f, y_f, th_f, rounds=6, margin=m_lp)
        if ok_lp2 and (m_lp == 1e-9 and L_lp2 < L_slp or m_lp > 1e-9 and L_lp2 < L_slp + 5e-6):
            cand_ver = u.verify_official(n, xf2, yf2, th_f)
            if cand_ver.get("ok"):
                x_f, y_f, L_slp, ok_slp, ver = xf2, yf2, L_lp2, True, cand_ver
                break
            ver = cand_ver

    pen_fin, L_fin = cpp_eval_sat_pen_bbox(x_f, y_f, th_f, side=1.0)
    if not ver.get("ok") and pen_fin < 1e-6:
        ver = u.verify_official(n, x_f, y_f, th_f)
    if not ver.get("ok") and pen_fin < 5e-6:
        for eps in (2e-9, 1e-8, 5e-8, 2e-7, 1e-6, 5e-6):
            sc = 1.0 + eps
            cand_ver = u.verify_official(n, x_f * sc, y_f * sc, th_f)
            if cand_ver.get("ok"):
                x_f, y_f = x_f * sc, y_f * sc
                pen_fin, L_fin = cpp_eval_sat_pen_bbox(x_f, y_f, th_f, side=1.0)
                ver = cand_ver
                break

    ang_deg = np.degrees(th_f) % 90.0
    n45 = int(np.sum(np.abs(ang_deg - 45.0) < 0.2))
    n0 = int(np.sum((ang_deg < 0.2) | (ang_deg > 89.8)))
    return {
        "idx": idx, "label": label, "family": fam, "n": n,
        "cpp_E": float(cpp_E), "cpp_pen": float(cpp_pen), "cpp_L": float(cpp_L),
        "ok_slp": bool(ok_slp), "pen_fin": float(pen_fin), "L_fin": float(L_fin),
        "n0": n0, "n45": n45, "non45_non0": int(n - n0 - n45),
        "verify": {k: v for k, v in ver.items() if k != "rows"},
        "rows": ver.get("rows"),
        "dt": round(time.time() - t0, 2),
    }


def _select_polish_indices(fams: list[str], lf: np.ndarray, po: np.ndarray, k_keep: int) -> list[int]:
    score = lambda i: (0 if lf[i] < 1e9 else 1, float(lf[i]) if lf[i] < 1e9 else float(po[i]))
    all_fams = sorted(set(fams))
    chosen: list[int] = []
    seen: set[int] = set()
    per_fam_quota = max(2, k_keep // max(1, len(all_fams)))
    for fam in all_fams:
        g = sorted([i for i, f in enumerate(fams) if f == fam], key=score)
        for i in g[:per_fam_quota]:
            if i not in seen:
                chosen.append(i)
                seen.add(i)
    for i in sorted(range(len(fams)), key=score):
        if len(chosen) >= k_keep:
            break
        if i not in seen:
            chosen.append(i)
            seen.add(i)
    return chosen


def solve_target(
    n: int, mats: dict, inc_front: dict, n_threads: int, quick: bool, max_donors: int
) -> dict:
    assert n in OWNED_TARGETS, f"n={n} is not in OWNED_TARGETS!"
    t0 = time.time()
    rng = np.random.default_rng(100003 + n * 17)
    x_inc, y_inc, th_inc, L_inc = mats[n]
    L_front = inc_front[n]["L_front"]
    print(f"\n=== Target n={n} (L_inc={L_inc:.9f}, L_front={L_front:.9f}, dL={L_inc - L_front:+.6f}) ===", flush=True)

    peels = collect_surplus_peels(n, L_inc, mats)[:max_donors]
    for p in peels[:6]:
        print(f"  peel {p['src']}: surp={p['surplus']} L={p['L']:.6f} (dL={p['L']-L_inc:+.5f}) oh_rem={p['max_oh_rem']:.4f}", flush=True)

    q_list: list[np.ndarray] = []
    sched_list: list[list[float]] = []
    labels: list[str] = []
    fams: list[str] = []

    # Track A: surplus peels
    n_peel_seeds = 4 if quick else 8
    for p in peels:
        L0_p = max(L_inc + 0.045, p["L"] + min(0.10, p["max_oh"]) + 0.015)
        L_mid = min(L_inc - 0.006, 0.5 * (L_inc + p["L"]))
        sched = [L0_p, L_inc + 0.012, L_inc - 2e-4, L_inc - 0.0025, L_mid]
        for s_i, q in enumerate(_surplus_drop_seeds(p, n, L0_p, n_peel_seeds, rng)):
            q_list.append(q)
            sched_list.append(sched)
            labels.append(f"{p['src']}_s{s_i}")
            fams.append("surplus_peel")

    # Track B: (n+1)->n & (n+2)->n deletion
    L0_del = L_inc + 0.032
    sched_del = [L0_del, L_inc + 0.010, L_inc - 2e-4, L_inc - 0.0025, L_inc - 0.007]
    for lbl, q in _del_n1_n2_seeds(mats, n, L_inc, L0_del, 8 if quick else 16, rng):
        q_list.append(q)
        sched_list.append(sched_del)
        labels.append(lbl)
        fams.append("del_n1_n2")

    # Track C: 6-mode incumbent pocket melts
    n_inc_seeds = 18 if quick else 36
    L0_inc = L_inc + 0.025
    sched_inc = [L0_inc, L_inc + 0.008, L_inc - 1.5e-4, L_inc - 0.0015, L_inc - 0.005]
    for s_i, q in enumerate(_incumbent_bottleneck_seeds(x_inc, y_inc, th_inc, L_inc, L0_inc, n_inc_seeds, rng)):
        q_list.append(q)
        sched_list.append(sched_inc)
        labels.append(f"inc_melt_{s_i}")
        fams.append("inc_melt")

    # Track D: multi-angle L-corridor synthesis & (n-1)+1
    L0_corr = L_inc + 0.045
    sched_corr = [L0_corr, L_inc + 0.015, L_inc - 2e-4, L_inc - 0.0025, L_inc - 0.006]
    for lbl, q in _corridor_and_m1_seeds(mats, n, L_inc, L0_corr, 8 if quick else 12, rng):
        q_list.append(q)
        sched_list.append(sched_corr)
        labels.append(lbl)
        fams.append("l_corr_m1")

    ils_iters = [6, 12, 16, 12, 10] if quick else [10, 18, 22, 18, 12]
    qf, qb, lf, eo, po, lb = run_cpp_batch_ils(
        n, 1, q_list, sched_list, ils_iters,
        a_start=0.82, n_melt_stages=4, n_threads=n_threads, base_seed=71000 + n,
    )
    n_feas = int(np.sum(lf < 1e9))
    best_feas = float(lf.min()) if n_feas > 0 else None
    print(
        f"  C++ ILS ({len(labels)} starts, {round(time.time()-t0, 1)}s): feas={n_feas}, best_Lf={best_feas}, min_pen={po.min():.5f}",
        flush=True,
    )

    k_keep = 16 if quick else 24
    sel = _select_polish_indices(fams, lf, po, k_keep)
    tasks = []
    for i in sel:
        use_f = lf[i] < 1e9
        x, y, th = cpp_expand_sym(n, 1, float(lf[i] if use_f else sched_list[i][-1]), qb[i] if use_f else qf[i])
        tasks.append((
            i, labels[i], fams[i], n, x, y, th,
            float(eo[i]), 0.0 if use_f else float(po[i]), float(lf[i]) if use_f else float(lb[i]),
        ))

    # Also include raw incumbent -> direct joint_xytheta_slp polish!
    tasks.append((
        -1, "raw_inc_slp", "inc_direct", n, x_inc.copy(), y_inc.copy(), th_inc.copy(),
        0.0, 0.0, L_inc,
    ))

    with mp.Pool(processes=min(4, max(1, n_threads), len(tasks))) as pool:
        polished = pool.map(_polish_and_repair_worker, tasks)

    polished.sort(
        key=lambda r: (
            0 if r["verify"].get("ok") else 1,
            float(r["verify"].get("L_float", 1e9)) if r["verify"].get("ok") else r["pen_fin"],
        )
    )
    best_valid, best_win = None, None
    for r in polished:
        if r["verify"].get("ok"):
            Lf = float(r["verify"]["L_float"])
            if best_valid is None or Lf < float(best_valid["verify"]["L_float"]):
                best_valid = r
            if Lf < L_inc - 1e-11 and (best_win is None or Lf < float(best_win["verify"]["L_float"])):
                best_win = r

    if best_valid is not None:
        v = best_valid["verify"]
        print(
            f"  BEST VALID n={n}: {best_valid['label']} ({best_valid['family']}) "
            f"L_lit={v['L_lit'][:16]} (inc={L_inc:.9f}, dL={float(v['L_float']) - L_inc:+.3e}, win={best_win is not None})",
            flush=True,
        )
    else:
        print(f"  NO VALID POLISHED for n={n}", flush=True)

    return {
        "n": n, "L_inc": L_inc, "L_front": L_front, "dt_s": round(time.time() - t0, 2),
        "peels": [{"src": p["src"], "surplus": p["surplus"], "L": p["L"], "max_oh_rem": p["max_oh_rem"]} for p in peels],
        "n_starts": len(labels), "n_feas": n_feas, "best_cpp_Lf": best_feas,
        "top_polished": [{k: v for k, v in r.items() if k != "rows"} for r in polished[:12]],
        "win": best_win,
    }


def load_verified_disk_wins(data_dir: Path) -> dict[int, dict]:
    paths = sorted(data_dir.glob("win_n*.csv")) if data_dir.exists() else []
    if not paths:
        return {}
    ns = [int(p.stem.replace("win_n", "")) for p in paths]
    inc_map = u.load_incumbent_and_frontier(ns)
    out: dict[int, dict] = {}
    for p in paths:
        nw = int(p.stem.replace("win_n", ""))
        sub = pd.read_csv(p)
        sub["sid"] = sub["id"].astype(str).str.split("_").str[1].astype(int)
        sub = sub.sort_values("sid")
        x = sub["x"].to_numpy(np.float64)
        y = sub["y"].to_numpy(np.float64)
        th = np.radians(sub["deg"].to_numpy(np.float64))
        ver = u.verify_official(nw, x, y, th)
        if ver.get("ok") and float(ver["L_float"]) < inc_map[nw]["L_ours"] - 1e-11:
            out[nw] = {
                "n": nw, "label": p.stem, "rows": ver["rows"],
                "verify": {k: v for k, v in ver.items() if k != "rows"},
            }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", type=int, nargs="+", default=[269, 918, 545])
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--max-donors", type=int, default=6)
    args = ap.parse_args()
    n_threads = min(4, max(1, args.threads))

    for t in args.targets:
        if t not in OWNED_TARGETS:
            raise SystemExit(f"Target n={t} is outside OWNED_TARGETS!")

    t_all = time.time()
    df = pd.read_csv(MERGED_CSV)
    mats = {int(n): _norm(sub) for n, sub in df.groupby("n")}
    inc_front = u.load_incumbent_and_frontier(args.targets)

    per_n: dict = {}
    if OUT_JSON.exists():
        try:
            per_n = json.loads(OUT_JSON.read_text()).get("per_n", {})
        except Exception:
            per_n = {}

    all_wins = load_verified_disk_wins(DATA_DIR)
    for n in args.targets:
        res = solve_target(n, mats, inc_front, n_threads, args.quick, args.max_donors)
        w = res.pop("win")
        per_n[str(n)] = res
        if w is not None:
            prev_w = all_wins.get(n)
            if prev_w is None or float(w["verify"]["L_float"]) < float(prev_w["verify"]["L_float"]):
                all_wins[n] = w
                DATA_DIR.mkdir(parents=True, exist_ok=True)
                u.write_final_csv_from_wins({n: w}, DATA_DIR / f"win_n{n}.csv")

    out = {
        "targets": sorted(int(k) for k in per_n),
        "per_n": per_n,
        "elapsed_s": round(time.time() - t_all, 2),
        "wins": [
            {"n": n, "L_lit": w["verify"]["L_lit"], "L_pandas": w["verify"]["L_pandas"], "label": w["label"]}
            for n, w in sorted(all_wins.items())
        ],
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2, default=float))
    if all_wins:
        u.write_final_csv_from_wins(all_wins, FINAL_CSV)
        print(f"\nWROTE STRICT CERTIFIED WINS to {FINAL_CSV}: {sorted(all_wins)}")
    else:
        print("\nNO STRICT WINS FOUND (final.csv untouched/absent).")


if __name__ == "__main__":
    main()
