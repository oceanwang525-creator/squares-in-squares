#!/usr/bin/env python3
"""Hypothesis H9: Directed Wall-Contact Imbalance Drift + Masked-Pocket Soft-a Melt + joint_xytheta_slp."""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time
from decimal import Decimal
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import slp_utils as su  # noqa: E402

HERE = Path(__file__).resolve().parent
POC_DIR = HERE.parent
DOCS_DIR = POC_DIR / "docs"
DATA_DIR = POC_DIR / "data"
OUT_JSON = DOCS_DIR / "gate_results.json"
FINAL_CSV = POC_DIR / "final.csv"

GATE_TARGETS = [754, 810, 598]
SCALE_TARGETS = [
    916, 997, 979, 852, 909, 692, 695, 992, 977, 799,
    637, 993, 915, 925, 647, 973, 811, 978, 806, 861,
]


def clearance_points(
    x: np.ndarray, y: np.ndarray, th: np.ndarray, L: float, n_pts: int = 8, step: float = 0.14
) -> list[tuple[float, float, float]]:
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
    t = deg[(deg > 1.5) & (deg < 88.5)]
    return math.radians(float(np.median(t))) if len(t) else math.pi / 4.0


def analyze_wall_imbalance(x: np.ndarray, y: np.ndarray, th: np.ndarray, L: float) -> dict:
    """Identify weakest vs heaviest bounding-box walls and touching square indices."""
    hx = 0.5 * (np.abs(np.cos(th)) + np.abs(np.sin(th)))
    margins = {
        "L": x - hx,
        "R": L - (x + hx),
        "B": y - hx,
        "T": L - (y + hx),
    }
    inward_normals = {
        "L": (1.0, 0.0),
        "R": (-1.0, 0.0),
        "B": (0.0, 1.0),
        "T": (0.0, -1.0),
    }
    opp_wall = {"L": "R", "R": "L", "B": "T", "T": "B"}
    counts_1e8 = {w: int(np.sum(m < 1e-8)) for w, m in margins.items()}
    counts_5e4 = {w: int(np.sum(m < 5e-4)) for w, m in margins.items()}

    weak_wall = min(margins.keys(), key=lambda w: (counts_1e8[w], counts_5e4[w]))
    heavy_wall = opp_wall[weak_wall]
    max_wall = max(margins.keys(), key=lambda w: (counts_1e8[w], counts_5e4[w]))

    weak_idx = np.where(margins[weak_wall] < 5e-4)[0]
    if len(weak_idx) == 0:
        weak_idx = np.argsort(margins[weak_wall])[:4]
    heavy_idx = np.where(margins[heavy_wall] < 5e-4)[0]
    if len(heavy_idx) == 0:
        heavy_idx = np.argsort(margins[heavy_wall])[:8]

    return {
        "counts_1e8": counts_1e8,
        "counts_5e4": counts_5e4,
        "weak_wall": weak_wall,
        "heavy_wall": heavy_wall,
        "max_wall": max_wall,
        "weak_idx": weak_idx,
        "heavy_idx": heavy_idx,
        "inward_normal": inward_normals[weak_wall],
        "opp_normal": inward_normals[heavy_wall],
        "margins": margins,
    }


def _cap_pocket_mask(
    x: np.ndarray, y: np.ndarray, cx: float, cy: float, rad: float,
    must_include: np.ndarray | None = None, min_sq: int = 35, max_sq: int = 90,
) -> np.ndarray:
    """Build pocket boolean mask targeting [min_sq, max_sq] active squares around (cx, cy)."""
    d2 = (x - cx) ** 2 + (y - cy) ** 2
    mask = d2 <= rad * rad
    if must_include is not None and len(must_include) > 0:
        mask[must_include] = True
    cnt = int(mask.sum())
    if cnt > max_sq:
        order = np.argsort(d2)
        mask = np.zeros(len(x), dtype=bool)
        if must_include is not None and len(must_include) > 0:
            mask[must_include] = True
        for i in order:
            if int(mask.sum()) >= max_sq:
                break
            mask[i] = True
    elif cnt < min_sq:
        order = np.argsort(d2)
        mask[order[: min(len(x), min_sq)]] = True
    return mask


def build_masked_pocket_tasks(
    n: int,
    xb: np.ndarray,
    yb: np.ndarray,
    tb: np.ndarray,
    L_base: float,
    L0_pocket: float,
    rng: np.random.Generator,
) -> tuple[list[tuple], list[np.ndarray], list[np.ndarray], list[float], list[list[float]], list[str]]:
    """Build masked-pocket soft-a melt tasks with directed drift toward internal voids."""
    sc = L0_pocket / L_base
    x0, y0, t0 = xb * sc, yb * sc, tb.copy()
    wb = analyze_wall_imbalance(x0, y0, t0, L0_pocket)
    voids = clearance_points(x0, y0, t0, L0_pocket, n_pts=8, step=0.14)
    tilt = donor_tilt(t0)

    weak_idx = wb["weak_idx"]
    heavy_idx = wb["heavy_idx"]
    nx_in, ny_in = wb["inward_normal"]
    ox_in, oy_in = wb["opp_normal"]
    sched = [L0_pocket, L_base + 0.008, L_base + 0.001, L_base - 3e-4]

    coords, masks, drifts, d_strengths, ls_list, labels = [], [], [], [], [], []

    def add_task(x, y, th, mask, fx, fy, strength, label):
        coords.append((x, y, th))
        masks.append(mask)
        drifts.append(np.concatenate([fx, fy]).astype(np.float64))
        d_strengths.append(float(strength))
        ls_list.append(sched)
        labels.append(label)

    # Track A (10 seeds): Sparse-wall contacts + nearest high-clearance internal voids
    for s in range(10):
        x, y, th = x0.copy(), y0.copy(), t0.copy()
        wi = int(weak_idx[s % len(weak_idx)])
        wx, wy = float(x[wi]), float(y[wi])
        if voids:
            v_sorted = sorted(voids, key=lambda v: (v[0] - wx) ** 2 + (v[1] - wy) ** 2 - 25.0 * v[2])
            vx, vy, _ = v_sorted[(s // len(weak_idx)) % min(3, len(v_sorted))]
        else:
            vx, vy = 0.5 * L0_pocket, 0.5 * L0_pocket

        cx = 0.45 * wx + 0.55 * vx
        cy = 0.45 * wy + 0.55 * vy
        rad = float(rng.uniform(4.8, 7.2))
        near = _cap_pocket_mask(x, y, cx, cy, rad, must_include=weak_idx, min_sq=35, max_sq=90)

        fx = np.zeros(n, dtype=np.float64)
        fy = np.zeros(n, dtype=np.float64)
        # Push sparse-wall squares inward toward nearest internal void
        for j in weak_idx:
            dx, dy = vx - float(x[j]), vy - float(y[j])
            norm = max(1e-6, math.hypot(dx, dy))
            fx[j] = 0.7 * (dx / norm) + 0.3 * nx_in
            fy[j] = 0.7 * (dy / norm) + 0.3 * ny_in
        # Gently draw pocket squares toward void center
        p_idx = np.where(near)[0]
        for j in p_idx:
            if j not in weak_idx:
                dx, dy = vx - float(x[j]), vy - float(y[j])
                norm = max(1e-6, math.hypot(dx, dy))
                fx[j] = 0.35 * (dx / norm)
                fy[j] = 0.35 * (dy / norm)

        if s % 2 == 0:
            # Teleport 1 sparse-wall contact square into void + melt both neighborhoods
            x[wi] = vx + float(rng.uniform(-0.08, 0.08))
            y[wi] = vy + float(rng.uniform(-0.08, 0.08))
            th[wi] = [0.0, math.pi / 4.0, tilt][(s // 2) % 3] + float(rng.uniform(-0.10, 0.10))
            near |= ((x - wx) ** 2 + (y - wy) ** 2) <= 3.5 ** 2

        nm = int(near.sum())
        x[near] += rng.uniform(-0.13, 0.13, nm)
        y[near] += rng.uniform(-0.13, 0.13, nm)
        th[near] += rng.uniform(-0.26, 0.26, nm)
        strength = [0.05, 0.10, 0.16][s % 3]
        add_task(x, y, th, near, fx, fy, strength, f"wd_mpk_void_{s}")

    # Track B (8 seeds): Weak-wall focused pocket + pure inward drift away from sparse wall
    for s in range(8):
        x, y, th = x0.copy(), y0.copy(), t0.copy()
        wi = int(weak_idx[s % len(weak_idx)])
        cx, cy = float(x[wi]), float(y[wi])
        rad = float(rng.uniform(4.8, 7.2))
        near = _cap_pocket_mask(x, y, cx, cy, rad, must_include=weak_idx, min_sq=35, max_sq=88)

        fx = np.zeros(n, dtype=np.float64)
        fy = np.zeros(n, dtype=np.float64)
        fx[near] = nx_in * 0.5
        fy[near] = ny_in * 0.5
        fx[weak_idx] = nx_in * 1.0
        fy[weak_idx] = ny_in * 1.0

        nm = int(near.sum())
        x[near] += nx_in * float(rng.uniform(0.02, 0.06)) + rng.uniform(-0.12, 0.12, nm)
        y[near] += ny_in * float(rng.uniform(0.02, 0.06)) + rng.uniform(-0.12, 0.12, nm)
        th[near] += rng.uniform(-0.28, 0.28, nm)
        strength = [0.04, 0.08, 0.12, 0.18][s % 4]
        add_task(x, y, th, near, fx, fy, strength, f"wd_mpk_wall_{s}")

    # Track C (6 seeds): Overloaded opposite wall pocket + drift toward weak wall / void
    for s in range(6):
        x, y, th = x0.copy(), y0.copy(), t0.copy()
        hi = int(heavy_idx[(s * 3) % len(heavy_idx)])
        cx, cy = float(x[hi]), float(y[hi])
        rad = float(rng.uniform(5.0, 7.4))
        near = _cap_pocket_mask(x, y, cx, cy, rad, min_sq=40, max_sq=90)

        fx = np.zeros(n, dtype=np.float64)
        fy = np.zeros(n, dtype=np.float64)
        fx[near] = ox_in * 0.7
        fy[near] = oy_in * 0.7

        nm = int(near.sum())
        x[near] += ox_in * float(rng.uniform(0.02, 0.06)) + rng.uniform(-0.12, 0.12, nm)
        y[near] += oy_in * float(rng.uniform(0.02, 0.06)) + rng.uniform(-0.12, 0.12, nm)
        th[near] += rng.uniform(-0.26, 0.26, nm)
        add_task(x, y, th, near, fx, fy, 0.08, f"wd_mpk_opp_{s}")

    return coords, masks, drifts, d_strengths, ls_list, labels


def build_idea20_walldrift_tasks(
    n: int,
    xb: np.ndarray,
    yb: np.ndarray,
    tb: np.ndarray,
    L_base: float,
    L0: float,
    rng: np.random.Generator,
) -> tuple[list[tuple], list[np.ndarray], list[float], list[list[float]], list[str]]:
    """Build idea20-style wall-balanced soft-a melt tasks with directed drift force."""
    sc = L0 / L_base
    x0, y0, t0 = xb * sc, yb * sc, tb.copy()
    wb = analyze_wall_imbalance(x0, y0, t0, L0)
    voids = clearance_points(x0, y0, t0, L0, n_pts=8, step=0.14)
    tilt = donor_tilt(t0)

    weak_idx = wb["weak_idx"]
    nx_in, ny_in = wb["inward_normal"]
    margins = wb["margins"]
    min_margin = np.minimum.reduce([margins["L"], margins["R"], margins["B"], margins["T"]])
    tight_idx = np.argsort(min_margin)[:18]
    sched = [L0, L_base + 0.010, L_base + 0.001, L_base - 3e-4]

    coords, drifts, d_strengths, ls_list, labels = [], [], [], [], []

    for s in range(20):
        x, y, th = x0.copy(), y0.copy(), t0.copy()
        fx = np.zeros(n, dtype=np.float64)
        fy = np.zeros(n, dtype=np.float64)
        mode = s % 4

        if mode == 0 and voids:
            # Teleport 1-2 weak-wall/tightest squares into max-clearance voids
            k_move = 1 if (s // 4) % 2 == 0 else min(2, len(voids))
            picks = rng.choice(weak_idx if len(weak_idx) >= k_move else tight_idx[:10], size=k_move, replace=False)
            for j, pi in enumerate(picks):
                vx, vy, _ = voids[(s + j) % len(voids)]
                x[pi] = vx + float(rng.uniform(-0.08, 0.08))
                y[pi] = vy + float(rng.uniform(-0.08, 0.08))
                th[pi] = [0.0, math.pi / 4.0, tilt][(s + j) % 3] + float(rng.uniform(-0.08, 0.08))
                near = ((x - vx) ** 2 + (y - vy) ** 2) <= 4.2 ** 2
                nm = int(near.sum())
                x[near] += rng.uniform(-0.12, 0.12, nm)
                y[near] += rng.uniform(-0.12, 0.12, nm)
                th[near] += rng.uniform(-0.12, 0.12, nm)
            fx[weak_idx] = nx_in * 0.8
            fy[weak_idx] = ny_in * 0.8
            strength = 0.06
        elif mode == 1:
            # idea20 mode 1 (exact mechanism of n=694 win): pocket melt on weak-wall / boundary pin
            pi = int(weak_idx[s % len(weak_idx)]) if (s % 8 < 4) else int(tight_idx[(s // 4) % len(tight_idx)])
            cx, cy = float(x[pi]), float(y[pi])
            rad = float(rng.uniform(4.4, 6.8))
            near = ((x - cx) ** 2 + (y - cy) ** 2) <= rad * rad
            nm = int(near.sum())
            x[near] += rng.uniform(-0.14, 0.14, nm)
            y[near] += rng.uniform(-0.14, 0.14, nm)
            th[near] += rng.uniform(-0.18, 0.18, nm)
            fx[near] = nx_in * 0.6
            fy[near] = ny_in * 0.6
            strength = 0.05
        elif mode == 2 and voids:
            # Void-collapse inward pull
            vx, vy, _ = voids[(s // 4) % len(voids)]
            rad = float(rng.uniform(4.8, 7.5))
            near = ((x - vx) ** 2 + (y - vy) ** 2) <= rad * rad
            nm = int(near.sum())
            pull = float(rng.uniform(0.03, 0.08))
            x[near] += pull * (vx - x[near]) + rng.uniform(-0.11, 0.11, nm)
            y[near] += pull * (vy - y[near]) + rng.uniform(-0.11, 0.11, nm)
            th[near] += rng.uniform(-0.18, 0.18, nm)
            dx, dy = vx - x[near], vy - y[near]
            norms = np.maximum(1e-6, np.hypot(dx, dy))
            fx[near] = dx / norms
            fy[near] = dy / norms
            strength = 0.07
        else:
            # Asymmetric wall-slack drift: push entire loaded side toward sparse wall + melt bottleneck
            drift_step = float(rng.uniform(0.03, 0.08))
            if abs(nx_in) > 0.5:
                # Sparse wall is Left (nx_in > 0) or Right (nx_in < 0)
                # Shift bulk away from opposite heavy wall (in direction -nx_in? Wait: push sparse wall inward or heavy wall toward sparse wall!)
                # Let's test pushing sparse-wall half inward AND heavy-wall half toward sparse wall
                loaded_half = (x > L0 * 0.30) if nx_in > 0 else (x < L0 * 0.70)
                x[loaded_half] -= nx_in * drift_step
                fx[weak_idx] = nx_in
                fx[loaded_half] = -0.35 * nx_in
            else:
                loaded_half = (y > L0 * 0.30) if ny_in > 0 else (y < L0 * 0.70)
                y[loaded_half] -= ny_in * drift_step
                fy[weak_idx] = ny_in
                fy[loaded_half] = -0.35 * ny_in

            wi = int(weak_idx[s % len(weak_idx)])
            cx, cy = (float(x[wi]), float(y[wi])) if not voids else (
                0.5 * (float(x[wi]) + voids[0][0]),
                0.5 * (float(y[wi]) + voids[0][1]),
            )
            near = ((x - cx) ** 2 + (y - cy) ** 2) <= float(rng.uniform(5.2, 7.8)) ** 2
            nm = int(near.sum())
            x[near] += rng.uniform(-0.13, 0.13, nm)
            y[near] += rng.uniform(-0.13, 0.13, nm)
            th[near] += rng.uniform(-0.16, 0.16, nm)
            strength = 0.08

        coords.append((x, y, th))
        drifts.append(np.concatenate([fx, fy]).astype(np.float64))
        d_strengths.append(strength)
        ls_list.append(sched)
        labels.append(f"i20_drift_{s}")

    return coords, drifts, d_strengths, ls_list, labels


def _polish_worker(args: tuple) -> dict:
    """Finish a trajectory with 0-jitter joint_xytheta_slp + multi-margin LP tighten + official check."""
    idx, label, n, L_base, x, y, th, cpp_pen, cpp_L = args
    t0 = time.time()
    hx = 0.5 * (np.abs(np.cos(th)) + np.abs(np.sin(th)))
    x = x - float((x - hx).min())
    y = y - float((y - hx).min())

    if cpp_pen > 1e-7 and cpp_pen < 0.025:
        L_try = cpp_L + max(0.002, 1.8 * cpp_pen)
        xr, yr, thr, pen_r, L_r = su.cpp_lbfgs_relax(x, y, th, L_try, full_iters=160)
        if pen_r <= cpp_pen + 1e-6:
            x, y, th, cpp_pen, cpp_L = xr, yr, thr, pen_r, L_r

    pen_pre, L_pre = su.sat_max_overlap_and_bbox(x, y, th)
    ok_slp, L_slp = False, L_pre
    x_f, y_f, th_f = x.copy(), y.copy(), th.copy()

    if pen_pre < 0.015:
        xs, ys, ts, Ls, ok_s = su.joint_xytheta_slp(
            x, y, th, rounds=10, tr_th0=0.018, tr_xy0=0.08, margin=1e-8, tol=1e-8, patience=2, lbfgs_iters=80
        )
        if ok_s:
            x_f, y_f, th_f, L_slp, ok_slp = xs, ys, ts, Ls, True
            if L_slp < L_base + 5e-4:
                xs2, ys2, ts2, Ls2, ok_s2 = su.joint_xytheta_slp(
                    x_f, y_f, th_f, rounds=12, tr_th0=0.006, tr_xy0=0.025, margin=1e-8, tol=1e-11, patience=2, lbfgs_iters=80
                )
                if ok_s2 and Ls2 < L_slp:
                    x_f, y_f, th_f, L_slp = xs2, ys2, ts2, Ls2

    ver: dict = {"ok": False}
    best_ver_L = 1e9
    for m_lp in (1e-7, 2e-8, 5e-9, 1e-6):
        xf2, yf2, L_lp2, ok_lp2 = su.fixed_angle_lp_tighten(x_f, y_f, th_f, rounds=5, margin=m_lp)
        if ok_lp2 and L_lp2 <= L_slp + 5e-6:
            if L_lp2 < L_base - 1e-11 or not ver.get("ok"):
                cand_ver = su.verify_official(n, xf2, yf2, th_f)
                if cand_ver.get("ok"):
                    cL = float(cand_ver["L_float"])
                    if cL < best_ver_L:
                        best_ver_L = cL
                        x_f, y_f, L_slp, ok_slp, ver = xf2, yf2, L_lp2, True, cand_ver
                elif not ver.get("ok"):
                    ver = cand_ver

    pen_fin, L_fin = su.sat_max_overlap_and_bbox(x_f, y_f, th_f)
    if not ver.get("ok") and pen_fin < 5e-6:
        for eps in (2e-9, 1e-8, 5e-8, 2e-7, 1e-6):
            sc = 1.0 + eps
            cand_ver = su.verify_official(n, x_f * sc, y_f * sc, th_f)
            if cand_ver.get("ok"):
                x_f, y_f = x_f * sc, y_f * sc
                pen_fin, L_fin = su.sat_max_overlap_and_bbox(x_f, y_f, th_f)
                ver = cand_ver
                break

    return {
        "idx": idx,
        "label": label,
        "n": n,
        "cpp_pen": float(cpp_pen),
        "cpp_L": float(cpp_L),
        "ok_slp": bool(ok_slp),
        "pen_fin": float(pen_fin),
        "L_fin": float(L_fin),
        "verify": {k: v for k, v in ver.items() if k != "rows"},
        "rows": ver.get("rows"),
        "dt": round(time.time() - t0, 2),
    }


def solve_target(
    n: int, mats: dict, inc_info: dict, wins: dict, n_threads: int
) -> dict:
    t0 = time.time()
    L_inc = inc_info[n]["L_ours"]
    L_inc_dec = Decimal(inc_info[n]["L_inc_str"])
    L_front = inc_info[n]["L_front"]

    x_b, y_b, th_b, L_b = mats[n]
    wb = analyze_wall_imbalance(x_b, y_b, th_b, L_b)
    print(
        f"\n=== H9 WallDrift n={n} | L_inc={L_inc:.9f} L_front={L_front:.9f} | "
        f"walls_1e-8={wb['counts_1e8']} walls_5e-4={wb['counts_5e4']} weak={wb['weak_wall']} ===",
        flush=True,
    )

    # Stage 0: 0-jitter direct joint_xytheta_slp + multi-margin LP tighten
    dir_res = _polish_worker((0, "direct_0jit_slp", n, L_b, x_b, y_b, th_b, 0.0, L_b))
    polished = [dir_res]
    if dir_res["verify"].get("ok"):
        dL0 = float(Decimal(max(Decimal(dir_res["verify"]["L_lit"]), Decimal(dir_res["verify"]["L_pandas"]))) - L_inc_dec)
        print(f"  [Stage 0 direct SLP] dL={dL0:+.3e} ({dir_res['dt']}s)", flush=True)

    # Engine 1: Masked-pocket soft-a melt + directed drift + cavity_relocate_orbit (24 trajectories)
    rng = np.random.default_rng(90900 + n)
    L0_pocket = L_b + 0.032
    c_mpk, m_mpk, dr_mpk, ds_mpk, ls_mpk, lbl_mpk = build_masked_pocket_tasks(
        n, x_b, y_b, th_b, L_b, L0_pocket, rng
    )
    res_mpk = su.cpp_batch_masked_melt(
        c_mpk, m_mpk, ls_mpk, ils_iters_per_L=[4, 7, 10, 12],
        a_start=0.80, n_melt_stages=4, masked_iters=160, full_iters=220,
        drift_xy_list=dr_mpk, drift_strengths=ds_mpk,
        n_threads=n_threads, base_seed=91000 + n,
    )
    for r, lbl in zip(res_mpk, lbl_mpk):
        r["label"] = lbl

    # Engine 2: idea20-style wall-balanced soft-a melt + directed drift (20 trajectories)
    L0_i20 = L_b + 0.036
    c_i20, dr_i20, ds_i20, ls_i20, lbl_i20 = build_idea20_walldrift_tasks(
        n, x_b, y_b, th_b, L_b, L0_i20, rng
    )
    res_i20 = su.cpp_batch_melt(
        c_i20, ls_i20, a_start=0.84, n_melt_stages=4,
        seam_masks=None, seam_iters=0, full_iters=180, n_ruin_iters=6,
        drift_xy_list=dr_i20, drift_strengths=ds_i20,
        n_threads=n_threads, base_seed=92000 + n,
    )
    for r, lbl in zip(res_i20, lbl_i20):
        r["label"] = lbl

    all_cpp = res_mpk + res_i20
    n_feas = sum(1 for r in all_cpp if r["best_L_feas"] is not None)
    best_feas = min((r["best_L_feas"] for r in all_cpp if r["best_L_feas"] is not None), default=None)
    min_pen = min(r["cpp_pen"] for r in all_cpp)
    print(f"  [C++ Batch] N={len(all_cpp)} feas={n_feas} best_Lf={best_feas} min_pen={min_pen:.2e}", flush=True)

    # Pick diverse top 8 candidates across Engine 1 & Engine 2 for SLP polish
    score = lambda r: (0 if r["best_L_feas"] is not None else 1, r["best_L_feas"] if r["best_L_feas"] is not None else r["cpp_pen"])
    ordered = sorted(all_cpp, key=score)
    chosen, seen_lbl = [], set()
    for pref, qta in (("wd_mpk_void_", 3), ("wd_mpk_wall_", 2), ("wd_mpk_opp_", 1), ("i20_drift_", 3)):
        cnt = 0
        for r in ordered:
            if r["label"] not in seen_lbl and r["label"].startswith(pref):
                chosen.append(r); seen_lbl.add(r["label"]); cnt += 1
                if cnt >= qta or len(chosen) >= 8:
                    break
    for r in ordered:
        if len(chosen) >= 8:
            break
        if r["label"] not in seen_lbl:
            chosen.append(r); seen_lbl.add(r["label"])

    tasks = [
        (i + 1, r["label"], n, L_b, r["x"], r["y"], r["th"],
         0.0 if r["best_L_feas"] is not None else r["cpp_pen"],
         r["best_L_feas"] if r["best_L_feas"] is not None else r["L_bbox"])
        for i, r in enumerate(chosen)
    ]
    with mp.Pool(processes=min(4, max(1, n_threads), len(tasks))) as pool:
        polished.extend(pool.map(_polish_worker, tasks))

    polished.sort(
        key=lambda r: (
            0 if r["verify"].get("ok") else 1,
            float(r["verify"].get("L_float", 1e9)) if r["verify"].get("ok") else r["pen_fin"],
        )
    )

    best_valid, best_win = None, None
    best_dL = 1e9
    for r in polished:
        if r["verify"].get("ok"):
            max_L_dec = max(Decimal(r["verify"]["L_lit"]), Decimal(r["verify"]["L_pandas"]))
            dL = float(max_L_dec - L_inc_dec)
            if best_valid is None or dL < best_dL:
                best_valid = r
                best_dL = dL
            if max_L_dec < L_inc_dec:
                if best_win is None or max_L_dec < max(Decimal(best_win["verify"]["L_lit"]), Decimal(best_win["verify"]["L_pandas"])):
                    best_win = r

    status = "MISS"
    if best_valid is not None:
        v = best_valid["verify"]
        if best_dL <= -1e-9:
            status = "REAL_WIN"
        elif best_dL < 0.0:
            status = "TIE"
        print(
            f"  BEST VALID n={n}: {best_valid['label']} L_lit={v['L_lit'][:18]} "
            f"(dL={best_dL:+.3e}, {status}) [{time.time() - t0:.1f}s]",
            flush=True,
        )
    else:
        print(f"  NO VALID POLISHED n={n} [{time.time() - t0:.1f}s]", flush=True)

    return {
        "n": n,
        "L_inc": L_inc,
        "L_inc_str": str(L_inc_dec),
        "L_front": L_front,
        "walls_1e8": wb["counts_1e8"],
        "walls_5e4": wb["counts_5e4"],
        "weak_wall": wb["weak_wall"],
        "dt_s": round(time.time() - t0, 2),
        "n_starts": len(all_cpp) + 1,
        "n_feas": n_feas,
        "best_cpp_Lf": best_feas,
        "status": status,
        "best_dL": best_dL if best_valid is not None else None,
        "top_polished": [{k: v for k, v in r.items() if k != "rows"} for r in polished[:8]],
        "win": best_win,
    }


def load_disk_wins(data_dir: Path, mats: dict, inc_info: dict) -> dict[int, dict]:
    """Load and re-verify existing win_n*.csv checkpoints on disk via auto_union."""
    out = {}
    if not data_dir.exists():
        return out
    union_mod = su.get_union_module()
    for p in sorted(data_dir.glob("win_n*.csv")):
        nw = int(p.stem.replace("win_n", ""))
        try:
            lit_g, pd_g = union_mod.load(p)
        except Exception:
            continue
        if nw not in lit_g or nw not in pd_g:
            continue
        chk = union_mod.check((nw, lit_g[nw], pd_g[nw]))
        if chk is None:
            continue
        if nw not in inc_info:
            _, extra_info = su.load_incumbent_and_frontier([nw])
            inc_info.update(extra_info)
        inc_dec = Decimal(inc_info[nw]["L_inc_str"])
        if Decimal(chk[0]) < inc_dec and Decimal(chk[1]) < inc_dec:
            out[nw] = {
                "n": nw,
                "rows": lit_g[nw],
                "verify": {
                    "ok": True,
                    "L_lit": chk[0],
                    "L_pandas": chk[1],
                    "L_float": float(max(Decimal(chk[0]), Decimal(chk[1]))),
                },
            }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", type=int, nargs="+", default=GATE_TARGETS)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out-json", type=Path, default=OUT_JSON)
    args = ap.parse_args()
    n_threads = min(4, max(1, args.threads))

    t_all = time.time()
    mats, inc_info = su.load_incumbent_and_frontier(args.targets)
    wins = load_disk_wins(DATA_DIR, mats, inc_info)

    per_n = {}
    if args.out_json.exists():
        try:
            prev = json.loads(args.out_json.read_text())
            per_n = prev.get("per_n", {})
        except Exception:
            per_n = {}

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for n in args.targets:
        res = solve_target(n, mats, inc_info, wins, n_threads=n_threads)
        w = res.pop("win")
        per_n[str(n)] = res
        if w is not None:
            wins[n] = w
            su.write_final_csv_from_wins({n: w}, DATA_DIR / f"win_n{n}.csv")
            su.write_final_csv_from_wins(wins, FINAL_CSV)
            print(f"  [CHECKPOINT SAVED] win_n{n}.csv & final.csv (total={len(wins)})", flush=True)

        # Save incremental checkpoint JSON after every target
        args.out_json.write_text(
            json.dumps(
                {
                    "targets": sorted(int(k) for k in per_n),
                    "per_n": per_n,
                    "elapsed_s": round(time.time() - t_all, 2),
                },
                indent=2,
                default=float,
            )
        )

    su.write_final_csv_from_wins(wins, FINAL_CSV)
    union_res = su.verify_final_csv_with_union(FINAL_CSV)
    print(
        f"\nDONE in {time.time() - t_all:.1f}s | wins_or_ties={sorted(wins.keys())} | "
        f"union_verified={union_res['count']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
