#!/usr/bin/env python3
"""Idea 20c: Wall-Unbalance & Void-Collapse Melts + max_oh Surplus Peels + Exact Margin Repair.

Combines:
1. Surplus Center-Peels (rem in [n, n+6]) across wall splits and interior cross-cuts, ranked by
   post-drop overhang (max_oh_rem) so low-overhang surplus=1..4 donors aren't starved.
2. Best (n-1) deficit void-insertion seeds with soft-core warm-up.
3. Incumbent pocket melts (modes 0..4): wall-pin void teleport, boundary-pin pocket melt,
   tilted-grid seam unlock, max-void collapse, and asymmetric wall-slack drift.
4. Polish with joint_xytheta_slp + multi-margin fixed_angle_lp_tighten + radial exact repair.
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pandas as pd

import utils as u
from idea17_run_slp import joint_xytheta_slp
from idea18_run import cpp_eval_sat_pen_bbox, write_final_csv_from_wins
from idea19_run import MERGED_CSV, cpp_expand_sym, full_to_free_sym, run_cpp_batch_ils

HERE = Path(__file__).resolve().parent
OUT_JSON_20 = HERE.parent / "docs" / "idea20_results.json"
FINAL_CSV = HERE.parent / "final.csv"
DATA_DIR = HERE.parent / "data"


def _norm(sub: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    x = sub["x"].to_numpy(np.float64).copy()
    y = sub["y"].to_numpy(np.float64).copy()
    th = sub["angle"].to_numpy(np.float64).copy()
    hx = 0.5 * (np.abs(np.cos(th)) + np.abs(np.sin(th)))
    x -= float((x - hx).min())
    y -= float((y - hx).min())
    L = float(max((x + hx).max(), (y + hx).max()))
    return x, y, th, L


def clearance_points(x, y, th, L, n_pts: int, step: float = 0.10) -> list[tuple[float, float, float]]:
    """Return top n_pts (px, py, clearance) maxima to all squares and walls."""
    g = np.arange(0.5, L - 0.5 + 1e-9, step)
    gx, gy = np.meshgrid(g, g, indexing="ij")
    px, py = gx.ravel(), gy.ravel()
    clr = np.minimum(np.minimum(px, L - px), np.minimum(py, L - py)) - 0.5
    c, s = np.cos(th), np.sin(th)
    for a in range(0, len(px), 4096):
        dx = px[a:a + 4096, None] - x[None, :]
        dy = py[a:a + 4096, None] - y[None, :]
        d = np.maximum(np.abs(c[None, :] * dx + s[None, :] * dy), np.abs(-s[None, :] * dx + c[None, :] * dy)) - 0.5
        clr[a:a + 4096] = np.minimum(clr[a:a + 4096], d.min(axis=1))
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
    return math.radians(float(np.median(t))) if len(t) else math.pi / 4


def _peel_mask_split(x, y, L, cL: int, cR: int, cB: int, cT: int) -> np.ndarray:
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
    """Collect center-peel donors (rem in [n, n+6]) across wall splits and interior slices."""
    k = math.isqrt(n)
    cands: list[dict] = []
    seen = set()

    for n2 in range(n + 1, min(1000, n + 5 * k + 18) + 1):
        if n2 not in mats:
            continue
        x, y, th, L = mats[n2]
        hx = 0.5 * (np.abs(np.cos(th)) + np.abs(np.sin(th)))
        for c in (1, 2, 3):
            L_box = L - c
            if not (L_inc - 0.16 <= L_box <= L_inc - 0.0005):
                continue
            # All split combinations (cL, cR=c-cL) x (cB, cT=c-cB)
            for cL in range(c + 1):
                cR = c - cL
                for cB in range(c + 1):
                    cT = c - cB
                    keep = _peel_mask_split(x, y, L, cL, cR, cB, cT)
                    nr = int(keep.sum())
                    if not (n <= nr <= n + 6):
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
                    if max_oh_rem >= 0.30:
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

        # Also check 5x5 grid of interior c=1 cross-slices for n2 in [n + k, n + 3*k]
        if n2 <= n + 3 * k:
            L_box = L - 1.0
            if L_inc - 0.08 <= L_box <= L_inc - 0.0005:
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
                        # Score seam stress around xc, yc
                        seam_d = np.minimum(np.abs(xs - (xc - 0.5)), np.abs(ys - (yc - 0.5)))
                        oh = np.maximum(0.0, 0.75 - seam_d) + np.maximum.reduce([
                            np.zeros(nr), h2 - xs, xs + h2 - L_box, h2 - ys, ys + h2 - L_box,
                        ])
                        surp = nr - n
                        max_oh = float(oh.max())
                        max_oh_rem = float(np.partition(oh, -(surp + 1))[-(surp + 1)]) if surp > 0 else max_oh
                        if max_oh_rem < 0.60:
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
    """Drop `surplus` worst overhang/seam squares, clamp into L0, and melt seam neighborhoods."""
    x0, y0, th0, oh0 = don["x"], don["y"], don["th"], don["oh"]
    L_box = don["L"]
    surplus = don["surplus"]
    hx0 = 0.5 * (np.abs(np.cos(th0)) + np.abs(np.sin(th0)))

    # Prefer squares with positive overhang or near peeled border
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
        assert len(x) == n

        # Clamp strictly inside [0, L0] and lightly jitter seam/overhang neighborhood
        x = np.clip(x, hx + 0.005, L0 - hx - 0.005)
        y = np.clip(y, hx + 0.005, L0 - hx - 0.005)
        hot = np.where(oh > 0.01)[0]
        if len(hot) > 0 and s > 0:
            cx, cy = float(x[hot[s % len(hot)]]), float(y[hot[s % len(hot)]])
            near = ((x - cx) ** 2 + (y - cy) ** 2) <= 3.5 ** 2
            nm = int(near.sum())
            x[near] = np.clip(x[near] + rng.uniform(-0.10, 0.10, nm), hx[near] + 0.005, L0 - hx[near] - 0.005)
            y[near] = np.clip(y[near] + rng.uniform(-0.10, 0.10, nm), hx[near] + 0.005, L0 - hx[near] - 0.005)
            th[near] += rng.uniform(-0.08, 0.08, nm)
        seeds.append(full_to_free_sym(x, y, th, L0))
    return seeds


def _incumbent_bottleneck_seeds(
    x_inc: np.ndarray, y_inc: np.ndarray, th_inc: np.ndarray, L_inc: float, L0: float,
    n_seeds: int, rng: np.random.Generator,
) -> list[np.ndarray]:
    """Generate 5-mode pocket melts: pin-teleport, wall-pin pocket, tilted seam, void collapse, wall-slack drift."""
    n = len(x_inc)
    sc = L0 / L_inc
    xb, yb, tb = x_inc * sc, y_inc * sc, th_inc.copy()
    hx = 0.5 * (np.abs(np.cos(tb)) + np.abs(np.sin(tb)))
    margin = np.minimum.reduce([xb - hx, L0 - xb - hx, yb - hx, L0 - yb - hx])
    tight_idx = np.argsort(margin)[:18]
    voids = clearance_points(xb, yb, tb, L0, n_pts=8, step=0.14)
    tilt = donor_tilt(tb)

    deg = np.degrees(tb) % 90.0
    tilted_mask = (deg > 2.0) & (deg < 88.0)
    tilted_idx = np.where(tilted_mask)[0]

    # Identify weakest contact walls (mimics [25,8,26,21] -> [22,25,21,25] balance on n=694)
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
        mode = s % 5
        if mode == 0 and len(voids) > 0:
            # Teleport 1-2 boundary-pinning squares into max-clearance voids & melt both spots
            k_move = 1 if (s // 5) % 2 == 0 else min(2, len(voids))
            picks = rng.choice(tight_idx[:12], size=k_move, replace=False)
            for j, pi in enumerate(picks):
                vx, vy, _ = voids[(s + j) % len(voids)]
                x[pi] = vx + float(rng.uniform(-0.08, 0.08))
                y[pi] = vy + float(rng.uniform(-0.08, 0.08))
                th[pi] = [0.0, math.pi / 4, tilt][(s + j) % 3] + float(rng.uniform(-0.08, 0.08))
                near = ((x - vx) ** 2 + (y - vy) ** 2) <= 4.2 ** 2
                nm = int(near.sum())
                x[near] += rng.uniform(-0.12, 0.12, nm)
                y[near] += rng.uniform(-0.12, 0.12, nm)
                th[near] += rng.uniform(-0.10, 0.10, nm)
        elif mode == 1:
            # Melt local R=4.2..6.5 pocket centered on a boundary-pinning square
            pi = int(tight_idx[(s // 5) % len(tight_idx)])
            cx, cy = float(x[pi]), float(y[pi])
            rad = float(rng.uniform(4.2, 6.5))
            near = ((x - cx) ** 2 + (y - cy) ** 2) <= rad * rad
            nm = int(near.sum())
            x[near] += rng.uniform(-0.14, 0.14, nm)
            y[near] += rng.uniform(-0.14, 0.14, nm)
            th[near] += rng.uniform(-0.16, 0.16, nm)
        elif mode == 2:
            # Tilted domain shear + local neighborhood angle unlock
            if len(tilted_idx) > 0:
                pi = int(tilted_idx[(s * 7) % len(tilted_idx)])
                cx, cy = float(x[pi]), float(y[pi])
                near = ((x - cx) ** 2 + (y - cy) ** 2) <= float(rng.uniform(5.0, 7.2)) ** 2
            else:
                near = np.ones(n, dtype=bool)
            nm = int(near.sum())
            x[near] += rng.uniform(-0.14, 0.14, nm)
            y[near] += rng.uniform(-0.14, 0.14, nm)
            th[near] += rng.uniform(-0.15, 0.15, nm)
        elif mode == 3 and len(voids) > 0:
            # Void-collapse pocket melt: pull neighborhood gently inward toward max-clearance void
            vx, vy, _ = voids[(s // 5) % len(voids)]
            rad = float(rng.uniform(4.5, 7.5))
            near = ((x - vx) ** 2 + (y - vy) ** 2) <= rad * rad
            nm = int(near.sum())
            pull = float(rng.uniform(0.03, 0.08))
            x[near] += pull * (vx - x[near]) + rng.uniform(-0.11, 0.11, nm)
            y[near] += pull * (vy - y[near]) + rng.uniform(-0.11, 0.11, nm)
            th[near] += rng.uniform(-0.16, 0.16, nm)
        else:
            # Asymmetric wall-slack drift: nudge loaded half toward weakest wall + melt bottleneck seam
            drift = float(rng.uniform(0.03, 0.09))
            cx, cy = (x[tight_idx[0]], y[tight_idx[0]]) if len(voids) == 0 else (
                0.5 * (float(x[tight_idx[s % 6]]) + voids[0][0]),
                0.5 * (float(y[tight_idx[s % 6]]) + voids[0][1]),
            )
            half_x = (x < L0 * 0.65) if dx_sign > 0 else (x > L0 * 0.35)
            half_y = (y < L0 * 0.65) if dy_sign > 0 else (y > L0 * 0.35)
            x[half_x] += dx_sign * drift
            y[half_y] += dy_sign * 0.6 * drift
            near = ((x - cx) ** 2 + (y - cy) ** 2) <= float(rng.uniform(5.2, 8.0)) ** 2
            nm = int(near.sum())
            x[near] += rng.uniform(-0.13, 0.13, nm)
            y[near] += rng.uniform(-0.13, 0.13, nm)
            th[near] += rng.uniform(-0.14, 0.14, nm)
        seeds.append(full_to_free_sym(x, y, th, L0))
    return seeds


def _deficit_n1_seeds(mats: dict, n: int, L_inc: float, L0: float, n_seeds: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Soft-core n-1 void insertion seeds (small clean subset)."""
    if (n - 1) not in mats:
        return []
    x, y, th, L = mats[n - 1]
    if L >= L_inc:
        return []
    sc = (L0 - 1.0) / max(1e-9, L - 1.0)
    xb = 0.5 + (x - 0.5) * sc
    yb = 0.5 + (y - 0.5) * sc
    pts = clearance_points(xb, yb, th, L0, n_pts=6, step=0.10)
    tilt = donor_tilt(th)
    angs = [0.0, math.pi / 4, tilt]
    seeds = []
    for s in range(n_seeds):
        px, py, _ = pts[s % len(pts)]
        xs = np.append(xb, px + float(rng.normal(0, 0.05)))
        ys = np.append(yb, py + float(rng.normal(0, 0.05)))
        ts = np.append(th, angs[(s // max(1, len(pts))) % len(angs)] + float(rng.normal(0, 0.04)))
        near = ((xs - px) ** 2 + (ys - py) ** 2) <= 3.5 ** 2
        nm = int(near.sum())
        xs[near] += rng.uniform(-0.10, 0.10, nm)
        ys[near] += rng.uniform(-0.10, 0.10, nm)
        seeds.append(full_to_free_sym(xs, ys, ts, L0))
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
        xs, ys, ts, Ls, ok_s = joint_xytheta_slp(x, y, th, rounds=16, tr_th0=0.025, tr_xy0=0.12, margin=1e-8)
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


def _select_polish_indices(labels: list[str], lf: np.ndarray, po: np.ndarray, k_keep: int) -> list[int]:
    """Select top diverse candidates across inc_melt, surplus_peel, and m1_ins."""
    score = lambda i: (0 if lf[i] < 1e9 else 1, float(lf[i]) if lf[i] < 1e9 else float(po[i]))
    order = sorted(range(len(labels)), key=score)
    chosen: list[int] = []
    used = set()
    # Reserve slots across families so peels aren't starved out by inc_melt
    for prefix, quota in (("inc_", max(4, k_keep // 2)), ("s", 3), ("x", 2), ("m", 1)):
        cnt = 0
        for i in order:
            if i not in used and labels[i].startswith(prefix):
                chosen.append(i)
                used.add(i)
                cnt += 1
                if cnt >= quota or len(chosen) >= k_keep:
                    break
    for i in order:
        if len(chosen) >= k_keep:
            break
        if i not in used:
            chosen.append(i)
            used.add(i)
    return chosen


def solve_target(n: int, mats: dict, inc_front: dict, n_threads: int, quick: bool, max_donors: int) -> dict:
    t0 = time.time()
    L_inc = inc_front[n]["L_ours"]
    L_front = inc_front[n]["L_front"]
    rng = np.random.default_rng(20300 + n)

    peels = collect_surplus_peels(n, L_inc, mats)[:max_donors]
    print(f"\n=== Idea 20c n={n} | L_inc={L_inc:.9f} L_front={L_front:.9f} | surplus_peels={len(peels)} ===", flush=True)
    for p in peels:
        print(f"   peel {p['src']}: surp=+{p['surplus']} L={p['L']:.6f} oh_rem={p['max_oh_rem']:.4f}", flush=True)

    L0 = L_inc + 0.04
    L_sched = [L0, L_inc + 0.012, L_inc - 2e-4, L_inc - 0.002, max(L_front + 2e-4, L_inc - 0.006)]
    ils_iters = [5, 8, 11, 13, 15] if quick else [7, 11, 15, 18, 22]

    q_all, labels, ls_all = [], [], []
    n_peel_seeds = 3 if quick else 6
    for p in peels:
        L0_p = max(L0, min(L_inc + 0.14, p["L"] + min(0.12, p["max_oh_rem"]) + 0.02))
        ls_p = [L0_p, L_inc + 0.015, L_inc - 2e-4, L_inc - 0.002, L_sched[-1]]
        for i, q in enumerate(_surplus_drop_seeds(p, n, L0_p, n_peel_seeds, rng)):
            q_all.append(q)
            labels.append(f"{p['src']}_s{i}")
            ls_all.append(ls_p)

    x_inc, y_inc, th_inc, _ = mats[n]
    for i, q in enumerate(_incumbent_bottleneck_seeds(x_inc, y_inc, th_inc, L_inc, L0, 25 if quick else 45, rng)):
        q_all.append(q)
        labels.append(f"inc_melt_{i}")
        ls_all.append(L_sched)

    for i, q in enumerate(_deficit_n1_seeds(mats, n, L_inc, L0, 4 if quick else 8, rng)):
        q_all.append(q)
        labels.append(f"m{n - 1}_d1_{i}")
        ls_all.append(L_sched)

    qb, qf, lf, eo, po, lb = run_cpp_batch_ils(
        n, 1, q_all, ls_all, ils_iters, a_start=0.78, n_melt_stages=4,
        n_threads=n_threads, base_seed=205500 + n,
    )
    n_feas = int(np.sum(lf < 1e9))
    best_feas = float(np.min(lf)) if n_feas else None
    print(f"  [ILS] N={len(labels)} feas={n_feas} best_Lf={best_feas} min_pen={float(np.min(po)):.3e} ({labels[int(np.argmin(po))]})", flush=True)

    keep = _select_polish_indices(labels, lf, po, 12 if quick else 20)
    tasks = []
    for i in keep:
        use_f = lf[i] < 1e9
        L_ref = float(lf[i]) if use_f else ls_all[i][-1]
        x, y, th = cpp_expand_sym(n, 1, L_ref, qf[i] if use_f else qb[i])
        fam = "surplus_peel" if not labels[i].startswith(("inc_", "m")) else ("inc_melt" if labels[i].startswith("inc_") else "m1_ins")
        tasks.append((i, labels[i], fam, n, x, y, th, float(eo[i]), 0.0 if use_f else float(po[i]),
                      float(lf[i]) if use_f else float(lb[i])))
    with mp.Pool(processes=min(4, max(1, n_threads), len(tasks))) as pool:
        polished = pool.map(_polish_and_repair_worker, tasks)
    polished.sort(key=lambda r: (0 if r["verify"].get("ok") else 1, float(r["verify"].get("L_float", 1e9)) if r["verify"].get("ok") else r["pen_fin"]))
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
        print(f"  BEST VALID n={n}: {best_valid['label']} L_lit={v['L_lit'][:16]} (inc={L_inc:.9f}, dL={float(v['L_float']) - L_inc:+.3e}, win={best_win is not None})", flush=True)
    else:
        print(f"  NO VALID POLISHED for n={n}", flush=True)
    return {
        "n": n, "L_inc": L_inc, "L_front": L_front, "dt_s": round(time.time() - t0, 2),
        "peels": [{"src": p["src"], "surplus": p["surplus"], "L": p["L"], "max_oh_rem": p["max_oh_rem"]} for p in peels],
        "n_starts": len(labels), "n_feas": n_feas, "best_cpp_Lf": best_feas,
        "top_polished": [{k: v for k, v in r.items() if k != "rows"} for r in polished[:10]],
        "win": best_win,
    }


def load_verified_disk_wins(data_dir: Path) -> dict[int, dict]:
    """Load and verify all saved data/win_n*.csv entries strictly beating merged.csv."""
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
    ap.add_argument("--targets", type=int, nargs="+", default=[340, 694, 790])
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--max-donors", type=int, default=6)
    args = ap.parse_args()
    n_threads = min(4, max(1, args.threads))

    t_all = time.time()
    df = pd.read_csv(MERGED_CSV)
    mats = {int(n): _norm(sub) for n, sub in df.groupby("n")}
    inc_front = u.load_incumbent_and_frontier(args.targets)

    per_n: dict = {}
    if OUT_JSON_20.exists():
        try:
            prev = json.loads(OUT_JSON_20.read_text())
            per_n = prev.get("per_n", {})
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
                write_final_csv_from_wins({n: w}, DATA_DIR / f"win_n{n}.csv")

    out = {
        "targets": sorted(int(k) for k in per_n),
        "per_n": per_n,
        "elapsed_s": round(time.time() - t_all, 2),
        "wins": [{"n": n, "L_lit": w["verify"]["L_lit"], "L_pandas": w["verify"]["L_pandas"], "label": w["label"]}
                 for n, w in sorted(all_wins.items())],
    }
    OUT_JSON_20.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON_20.write_text(json.dumps(out, indent=2, default=float))
    if all_wins:
        write_final_csv_from_wins(all_wins, FINAL_CSV)
        print(f"\nWROTE STRICT CERTIFIED WINS to {FINAL_CSV}: {sorted(all_wins)}")
    else:
        print("\nNO STRICT WINS FOUND (final.csv untouched/absent).")


if __name__ == "__main__":
    main()
