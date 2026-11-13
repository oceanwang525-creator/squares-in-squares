"""Idea 2: Exact-capacity, wedge-filled, SAT-clipped from-scratch seed generators.

Strictly built from (n, L_target) geometry and multi-grain / catalog topologies;
NEVER reads incumbent coordinates from merged.csv.
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np


PYTHAGOREAN_ANGLES = [
    math.atan2(3.0, 4.0),    # 36.8699 deg (3-4-5)
    math.atan2(5.0, 12.0),   # 22.6199 deg (5-12-13)
    math.atan2(8.0, 15.0),   # 28.0725 deg (8-15-17)
    math.atan2(7.0, 24.0),   # 16.2602 deg (7-24-25)
    math.atan2(12.0, 35.0),  # 18.9246 deg (12-35-37)
    math.atan2(20.0, 21.0),  # 43.6028 deg (20-21-29)
]


def solve_shear_angle(w: int, W: float) -> float | None:
    """Solve w*cos(th) + sin(th) = W for th in [atan(1/w), pi/4]."""
    if w <= 0 or W <= 0.0:
        return None
    v_peak = math.sqrt(w * w + 1.0)
    v_45 = (w + 1.0) / math.sqrt(2.0)
    if W > v_peak or W < v_45:
        return None
    phi = math.atan2(1.0, float(w))
    ratio = min(1.0, max(-1.0, W / v_peak))
    th = phi + math.acos(ratio)
    if th < 0.0 or th > math.pi * 0.25 + 1e-9:
        return None
    return min(th, math.pi * 0.25)


def sat_filter_against(
    cands: np.ndarray,
    existing: np.ndarray,
    scale_a: float = 0.87,
    box_L: float | None = None,
) -> np.ndarray:
    """Keep candidate rows in `cands` (M,3) inside box_L avoiding SAT overlap vs `existing` at `scale_a`."""
    if len(cands) == 0:
        return cands
    pts = cands
    if box_L is not None:
        ext = 0.5 * scale_a * (np.abs(np.cos(pts[:, 2])) + np.abs(np.sin(pts[:, 2])))
        inside = (
            (pts[:, 0] - ext >= -1e-4)
            & (pts[:, 0] + ext <= box_L + 1e-4)
            & (pts[:, 1] - ext >= -1e-4)
            & (pts[:, 1] + ext <= box_L + 1e-4)
        )
        pts = pts[inside]
        if len(pts) == 0 or len(existing) == 0:
            return pts

    if len(existing) == 0:
        return pts

    half_a = 0.5 * scale_a
    cj = np.cos(existing[:, 2])
    sj = np.sin(existing[:, 2])
    keep_mask = np.ones(len(pts), dtype=bool)
    cut2 = (1.45 * scale_a) ** 2

    for i0 in range(0, len(pts), 256):
        ch = pts[i0 : i0 + 256]
        dx = existing[None, :, 0] - ch[:, 0:1]
        dy = existing[None, :, 1] - ch[:, 1:2]
        d2 = dx * dx + dy * dy
        near = d2 < cut2
        if not np.any(near):
            continue
        ci = np.cos(ch[:, 2])[:, None]
        si = np.sin(ch[:, 2])[:, None]
        cr = ci * cj[None, :] + si * sj[None, :]
        sr = si * cj[None, :] - ci * sj[None, :]
        rad = half_a * (1.0 + np.abs(cr) + np.abs(sr))
        p0 = np.abs(dx * ci + dy * si)
        p1 = np.abs(-dx * si + dy * ci)
        p2 = np.abs(dx * cj[None, :] + dy * sj[None, :])
        p3 = np.abs(-dx * sj[None, :] + dy * cj[None, :])
        vmax = np.maximum(np.maximum(p0, p1), np.maximum(p2, p3))
        hit = np.any(near & (vmax < rad - 1e-6), axis=1)
        keep_mask[i0 : i0 + 256] = ~hit

    return pts[keep_mask]


def make_sheared_strip_exact(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    w: int,
    th: float,
    sign: int,
    vertical_strip: bool = True,
    y_phase: float = 0.0,
    scale_margin: float = 0.90,
) -> np.ndarray:
    """Zero-sawtooth sheared rows inside [x_min, x_max] x [y_min, y_max] with partial end clipping."""
    if vertical_strip:
        W = x_max - x_min
        H_min, H_max = y_min, y_max
    else:
        W = y_max - y_min
        H_min, H_max = x_min, x_max

    if W <= 0.5 or H_max - H_min <= 0.5:
        return np.zeros((0, 3), dtype=np.float64)

    c, s = math.cos(th), math.sin(th)
    hx = 0.5 * (c + s)
    hx_m = hx * scale_margin
    dy_row = 1.0 / max(c, 1e-6)
    row_span = w * c + s
    x_pad = 0.5 * max(0.0, W - row_span)

    r_min = -int(math.ceil(w * s / dy_row)) - 2
    r_max = int(math.ceil((H_max - H_min + w * s) / dy_row)) + 2

    pts: list[tuple[float, float, float]] = []
    for r in range(r_min, r_max + 1):
        y0 = H_min + hx_m + (r + y_phase) * dy_row
        for k in range(w):
            uk = x_pad + hx + k * c
            vk = y0 + sign * k * s
            if (
                uk - hx_m >= -1e-4
                and uk + hx_m <= W + 1e-4
                and vk - hx_m >= H_min - 1e-4
                and vk + hx_m <= H_max + 1e-4
            ):
                ang = sign * th
                if vertical_strip:
                    pts.append((x_min + uk, vk, ang))
                else:
                    pts.append((vk, y_min + uk, -ang))
    return np.asarray(pts, dtype=np.float64) if pts else np.zeros((0, 3), dtype=np.float64)


def fill_wedges_and_complete(
    pts: np.ndarray,
    n: int,
    L: float,
    rng: np.random.Generator,
    wedge_scale: float = 0.87,
) -> np.ndarray:
    """Pack open corners/end-wedges cleanly with 0/45/local-angle candidates, then hit exact count n."""
    if len(pts) == 0:
        pts = np.zeros((0, 3), dtype=np.float64)
    else:
        ext = 0.5 * (np.abs(np.cos(pts[:, 2])) + np.abs(np.sin(pts[:, 2])))
        pts = pts.copy()
        pts[:, 0] = np.clip(pts[:, 0], ext * 0.90, L - ext * 0.90)
        pts[:, 1] = np.clip(pts[:, 1], ext * 0.90, L - ext * 0.90)

    if len(pts) > n:
        c = np.cos(pts[:, 2])
        s = np.sin(pts[:, 2])
        while len(pts) > n:
            dx = pts[None, :, 0] - pts[:, 0:1]
            dy = pts[None, :, 1] - pts[:, 1:2]
            d2 = dx * dx + dy * dy
            np.fill_diagonal(d2, 1e9)
            near = d2 < 2.0
            cr = c[:, None] * c[None, :] + s[:, None] * s[None, :]
            sr = s[:, None] * c[None, :] - c[:, None] * s[None, :]
            rad = 0.48 * (1.0 + np.abs(cr) + np.abs(sr))
            vmax = np.maximum(
                np.maximum(np.abs(dx * c[:, None] + dy * s[:, None]), np.abs(-dx * s[:, None] + dy * c[:, None])),
                np.maximum(np.abs(dx * c[None, :] + dy * s[None, :]), np.abs(-dx * s[None, :] + dy * c[None, :])),
            )
            pen = np.where(near, np.maximum(0.0, rad - vmax), 0.0)
            stress = (pen * pen).sum(axis=1)
            ext = 0.48 * (np.abs(c) + np.abs(s))
            wall = (
                np.maximum(0.0, ext - pts[:, 0]) ** 2
                + np.maximum(0.0, pts[:, 0] + ext - L) ** 2
                + np.maximum(0.0, ext - pts[:, 1]) ** 2
                + np.maximum(0.0, pts[:, 1] + ext - L) ** 2
            )
            score = stress + 5.0 * wall
            if score.max() > 1e-8:
                drop_idx = int(np.argmax(score))
            else:
                min_d2 = d2.min(axis=1)
                thresh = min_d2.min() + 0.015
                drop_idx = int(rng.choice(np.flatnonzero(min_d2 <= thresh)))
            pts = np.delete(pts, drop_idx, axis=0)
            c = np.delete(c, drop_idx)
            s = np.delete(s, drop_idx)
        return pts

    if len(pts) == n:
        return pts

    # Deficit > 0: dense probe grid filtered cleanly
    step = max(0.24, L / 92.0)
    g = np.arange(0.45, L - 0.44, step)
    gx, gy = np.meshgrid(g, g, indexing="ij")
    probe_xy = np.column_stack([gx.ravel(), gy.ravel()])
    if len(pts) > 0:
        min_d2 = np.full(len(probe_xy), 1e9, dtype=np.float64)
        near_ang = np.zeros(len(probe_xy), dtype=np.float64)
        for i0 in range(0, len(probe_xy), 1024):
            ch = probe_xy[i0 : i0 + 1024]
            d2 = np.sum((ch[:, None, :] - pts[None, :, :2]) ** 2, axis=-1)
            idx_m = np.argmin(d2, axis=1)
            min_d2[i0 : i0 + 1024] = d2[np.arange(len(ch)), idx_m]
            near_ang[i0 : i0 + 1024] = pts[idx_m, 2]
        open_mask = min_d2 >= 0.56 ** 2
        if np.sum(open_mask) >= (n - len(pts)):
            probe_xy = probe_xy[open_mask]
            near_ang = near_ang[open_mask]
            min_d2 = min_d2[open_mask]
    else:
        min_d2 = np.full(len(probe_xy), 1e9, dtype=np.float64)
        near_ang = np.zeros(len(probe_xy), dtype=np.float64)

    need = n - len(pts)
    pts_list = list(pts)
    half_a = 0.5 * wedge_scale
    for _ in range(need):
        if len(min_d2) == 0:
            break
        order = np.argsort(-min_d2)[: min(20, len(min_d2))]
        best_score = 1e9
        best_cand = None
        best_idx = int(order[0])
        arr_pts = np.asarray(pts_list, dtype=np.float64) if pts_list else np.zeros((0, 3))
        for oi in order[:8]:
            px, py = float(probe_xy[oi, 0]), float(probe_xy[oi, 1])
            for ang in (float(near_ang[oi]), 0.0, math.pi * 0.25, -float(near_ang[oi])):
                c0, s0 = math.cos(ang), math.sin(ang)
                ext = half_a * (abs(c0) + abs(s0))
                ew = (
                    max(0.0, ext - px) ** 2
                    + max(0.0, px + ext - L) ** 2
                    + max(0.0, ext - py) ** 2
                    + max(0.0, py + ext - L) ** 2
                )
                if len(arr_pts) > 0:
                    dx = arr_pts[:, 0] - px
                    dy = arr_pts[:, 1] - py
                    d2 = dx * dx + dy * dy
                    near = d2 < (1.55 * wedge_scale) ** 2
                    if np.any(near):
                        dxn, dyn = dx[near], dy[near]
                        cj, sj = np.cos(arr_pts[near, 2]), np.sin(arr_pts[near, 2])
                        rad = half_a * (1.0 + np.abs(c0 * cj + s0 * sj) + np.abs(s0 * cj - c0 * sj))
                        vmax = np.maximum(
                            np.maximum(np.abs(dxn * c0 + dyn * s0), np.abs(-dxn * s0 + dyn * c0)),
                            np.maximum(np.abs(dxn * cj + dyn * sj), np.abs(-dxn * sj + dyn * cj)),
                        )
                        pen = np.maximum(0.0, rad - vmax)
                        es = float(np.sum(pen * pen))
                    else:
                        es = 0.0
                else:
                    es = 0.0
                tot = es + 5.0 * ew
                if tot < best_score:
                    best_score = tot
                    best_cand = (px, py, ang)
                    best_idx = int(oi)
                if best_score == 0.0:
                    break
            if best_score == 0.0:
                break
        if best_cand is None:
            best_cand = (float(probe_xy[best_idx, 0]), float(probe_xy[best_idx, 1]), float(near_ang[best_idx]))
        pts_list.append(best_cand)
        new_d2 = (probe_xy[:, 0] - best_cand[0]) ** 2 + (probe_xy[:, 1] - best_cand[1]) ** 2
        closer = new_d2 < min_d2
        min_d2 = np.minimum(min_d2, new_d2)
        near_ang[closer] = best_cand[2]
        min_d2[best_idx] = -1.0

    return np.asarray(pts_list, dtype=np.float64)


def _pack_sw_sheared_or_upright(W1: float, W2: float, rng: np.random.Generator) -> np.ndarray:
    if W1 < 1.0 or W2 < 1.0:
        return np.zeros((0, 3), dtype=np.float64)
    best_pts = np.zeros((0, 3), dtype=np.float64)
    p1, p2 = int(math.floor(W1)), int(math.floor(W2))
    if p1 > 0 and p2 > 0:
        dx = (W1 - p1) / max(1, p1)
        dy = (W2 - p2) / max(1, p2)
        sx = (np.arange(p1) + 0.5) * (1.0 + min(0.04, dx * 0.5))
        sy = (np.arange(p2) + 0.5) * (1.0 + min(0.04, dy * 0.5))
        gx, gy = np.meshgrid(sx, sy, indexing="ij")
        best_pts = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(p1 * p2)])
    for vert, span_w in ((True, W1), (False, W2)):
        w_s = int(math.floor(span_w)) + 1
        th = solve_shear_angle(w_s, span_w - 1e-4)
        if th is not None and th <= 0.72:
            cand = make_sheared_strip_exact(0.0, W1, 0.0, W2, w_s, th, int(rng.choice([-1, 1])), vert, 0.0, 0.90)
            if len(cand) > len(best_pts):
                best_pts = cand
    return best_pts


def _build_l_pocket_layout(
    L: float, Kx: int, Ky: int, w1: int, w2: int, th1: float, th2: float, mode: int,
    sign1: int, sign2: int, ph1: float, ph2: float, pitch: float, rng: np.random.Generator,
) -> tuple[np.ndarray, str]:
    W1, W2 = L - Kx * pitch, L - Ky * pitch
    xs_c = L - Kx * pitch + (np.arange(Kx) + 0.5) * pitch
    ys_c = L - Ky * pitch + (np.arange(Ky) + 0.5) * pitch
    gx, gy = np.meshgrid(xs_c, ys_c, indexing="ij")
    core = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(Kx * Ky)])

    if mode == 0:
        west = make_sheared_strip_exact(0.0, W1, 0.0, L, w1, th1, sign1, True, ph1, 0.89)
        south = make_sheared_strip_exact(W1, L, 0.0, W2, w2, th2, sign2, False, ph2, 0.89)
        sw = np.zeros((0, 3), dtype=np.float64)
        mtag = "Wfull"
    elif mode == 1:
        south = make_sheared_strip_exact(0.0, L, 0.0, W2, w2, th2, sign2, False, ph2, 0.89)
        west = make_sheared_strip_exact(0.0, W1, W2, L, w1, th1, sign1, True, ph1, 0.89)
        sw = np.zeros((0, 3), dtype=np.float64)
        mtag = "Sfull"
    else:
        west = make_sheared_strip_exact(0.0, W1, W2, L, w1, th1, sign1, True, ph1, 0.89)
        south = make_sheared_strip_exact(W1, L, 0.0, W2, w2, th2, sign2, False, ph2, 0.89)
        sw = _pack_sw_sheared_or_upright(W1, W2, rng)
        mtag = "SWmix"

    pts = core
    for part in (west, south, sw):
        if len(part) > 0:
            part_ok = sat_filter_against(part, pts, scale_a=0.86, box_L=L)
            if len(part_ok) > 0:
                pts = np.vstack([pts, part_ok])
    return pts, f"tapered_L(K={Kx}x{Ky},w={w1}/{w2},{mtag})"


def seed_tapered_sheared_l_pocket(n: int, L: float, rng: np.random.Generator) -> tuple[np.ndarray, str]:
    """Exact-capacity screened sheared L-pocket (evaluates real clipped capacity across top candidates)."""
    k = int(math.floor(math.sqrt(n)))
    coarse: list[tuple[int, int, int, int, float, float, int, float]] = []

    for Kx in range(max(4, k - 13), max(5, k - 1)):
        W1 = L - Kx
        if W1 < 1.6:
            continue
        w1_opts = []
        for w1 in (int(math.floor(W1)) + 1, int(math.floor(W1)) + 2):
            th1 = solve_shear_angle(w1, W1 - 1e-4)
            if th1 is not None and th1 <= 0.72:
                w1_opts.append((w1, th1))
        if not w1_opts:
            continue
        for Ky in range(max(4, k - 13), max(5, k - 1)):
            W2 = L - Ky
            if W2 < 1.6:
                continue
            for w1, th1 in w1_opts:
                for w2 in (int(math.floor(W2)) + 1, int(math.floor(W2)) + 2):
                    th2 = solve_shear_angle(w2, W2 - 1e-4)
                    if th2 is None or th2 > 0.72:
                        continue
                    # Higher-K configurations give higher bulk density -> target close to L^2 * 0.96
                    rho_w = w1 * math.cos(th1) / W1
                    rho_s = w2 * math.cos(th2) / W2
                    for mode in (0, 1, 2):
                        if mode == 0:
                            cap = Kx * Ky + W1 * L * rho_w + Kx * W2 * rho_s - 0.8 * (w1 * math.sin(th1) + w2 * math.sin(th2))
                        elif mode == 1:
                            cap = Kx * Ky + W2 * L * rho_s + Ky * W1 * rho_w - 0.8 * (w1 * math.sin(th1) + w2 * math.sin(th2))
                        else:
                            cap = Kx * Ky + W1 * W2 * 0.95 + Ky * W1 * rho_w + Kx * W2 * rho_s - 0.8 * (w1 * math.sin(th1) + w2 * math.sin(th2))
                        # Prefer large core Kx*Ky among feasible capacities
                        score = abs(cap - n) - 0.015 * (Kx + Ky)
                        coarse.append((Kx, Ky, w1, w2, th1, th2, mode, score))

    if not coarse:
        return seed_pinwheel_4pocket(n, L, rng)

    coarse.sort(key=lambda c: c[-1])
    # Evaluate exact clipped count on 8 diverse top candidates and pick closest to [n-4, n+4]
    sample_idx = rng.choice(min(48, len(coarse)), size=min(8, len(coarse)), replace=False)
    best_pts = None
    best_desc = ""
    best_pen = 1e9
    pitch = float(rng.choice([0.985, 0.994, 1.0]))
    for idx in sample_idx:
        Kx, Ky, w1, w2, th1, th2, mode, _ = coarse[int(idx)]
        pts, desc = _build_l_pocket_layout(
            L, Kx, Ky, w1, w2, th1, th2, mode,
            int(rng.choice([-1, 1])), int(rng.choice([-1, 1])),
            float(rng.uniform(0.0, 0.9)), float(rng.uniform(0.0, 0.9)),
            pitch, rng,
        )
        diff = len(pts) - n
        # Slightly penalize deficit more than slight surplus
        pen = abs(diff) + (4.0 if diff < -8 else 0.0)
        if pen < best_pen:
            best_pen = pen
            best_pts = pts
            best_desc = desc

    assert best_pts is not None
    pts = fill_wedges_and_complete(best_pts, n, L, rng, wedge_scale=0.86)
    pts[:, :2] += rng.normal(0.0, 0.008, size=(n, 2))
    pts[:, 2] += rng.normal(0.0, 0.008, size=n)
    return pts, best_desc


def seed_pinwheel_4pocket(n: int, L: float, rng: np.random.Generator) -> tuple[np.ndarray, str]:
    """Zero-collision 4-edge asymmetric sheared pinwheel tuned for actual clipped count >= n - 5."""
    k = int(math.floor(math.sqrt(n)))
    handed = int(rng.choice([-1, 1]))
    best_pts = None
    best_desc = ""
    best_pen = 1e9

    for _ in range(18):
        Kx = int(rng.integers(max(4, k - 12), max(5, k - 2)))
        Ky = int(rng.integers(max(4, k - 12), max(5, k - 2)))
        rx, ry = L - Kx, L - Ky
        Px, Py = int(math.floor(rx)), int(math.floor(ry))
        if Px < 2 or Py < 2:
            continue
        fx, fy = rx - Px, ry - Py
        if Px == 2:
            pw, pe = 1, 0
        else:
            pw = int(rng.integers(1, Px - 1))
            pe = Px - 1 - pw
        if Py == 2:
            ps, pn = 1, 0
        else:
            ps = int(rng.integers(1, Py - 1))
            pn = Py - 1 - ps

        W_W = pw + 0.5 * (fx + 1.0) + float(rng.uniform(-0.06, 0.06))
        W_E = rx - W_W
        W_S = ps + 0.5 * (fy + 1.0) + float(rng.uniform(-0.06, 0.06))
        W_N = ry - W_S
        if min(W_W, W_E, W_S, W_N) < 1.1:
            continue
        ww, we = int(math.floor(W_W)) + 1, int(math.floor(W_E)) + 1
        ws, wn = int(math.floor(W_S)) + 1, int(math.floor(W_N)) + 1
        th_w = solve_shear_angle(ww, W_W - 1e-4)
        th_e = solve_shear_angle(we, W_E - 1e-4)
        th_s = solve_shear_angle(ws, W_S - 1e-4)
        th_n = solve_shear_angle(wn, W_N - 1e-4)
        if None in (th_w, th_e, th_s, th_n):
            continue
        assert th_w is not None and th_e is not None and th_s is not None and th_n is not None
        if max(th_w, th_e, th_s, th_n) > 0.68:
            continue

        xs_c = W_W + np.arange(Kx) + 0.5
        ys_c = W_S + np.arange(Ky) + 0.5
        gx, gy = np.meshgrid(xs_c, ys_c, indexing="ij")
        pts = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(Kx * Ky)])

        rects = [
            (0.0, W_W, 0.0, L - W_N, ww, th_w, True, handed),
            (0.0, L - W_E, L - W_N, L, wn, th_n, False, -handed),
            (L - W_E, L, W_S, L, we, th_e, True, handed),
            (W_W, L, 0.0, W_S, ws, th_s, False, -handed),
        ]
        for xmin, xmax, ymin, ymax, w, th, vert, sgn in rects:
            st = make_sheared_strip_exact(xmin, xmax, ymin, ymax, w, th, sgn, vert, float(rng.uniform(0.0, 0.9)), 0.89)
            st = sat_filter_against(st, pts, scale_a=0.86, box_L=L)
            if len(st) > 0:
                pts = np.vstack([pts, st])

        diff = len(pts) - n
        pen = abs(diff) + (4.0 if diff < -8 else 0.0)
        if pen < best_pen:
            best_pen = pen
            best_pts = pts
            best_desc = f"pinwheel4(K={Kx}x{Ky},w={ww}/{we}/{ws}/{wn})"
            if abs(diff) <= 3:
                break

    if best_pts is None:
        return seed_45_upright_hybrids(n, L, rng)

    pts = fill_wedges_and_complete(best_pts, n, L, rng, wedge_scale=0.86)
    pts[:, :2] += rng.normal(0.0, 0.008, size=(n, 2))
    pts[:, 2] += rng.normal(0.0, 0.008, size=n)
    return pts, best_desc


def seed_45_upright_hybrids(n: int, L: float, rng: np.random.Generator) -> tuple[np.ndarray, str]:
    """Capacity-screened 45-deg + upright composite topologies (diagonal band, 45-core, dual corners)."""
    inv_s2 = 1.0 / math.sqrt(2.0)
    variant = str(rng.choice(["diag_band", "core45", "dual_corner45", "alt45_band"]))

    best_pts = None
    best_pen = 1e9
    best_desc = ""

    for _ in range(10):
        if variant == "diag_band":
            w_b = int(rng.integers(2, max(4, int(L * 0.5))))
            u_mid = float(rng.uniform(0.35, 0.65)) * (L * math.sqrt(2.0))
            u0 = u_mid - 0.5 * (w_b - 1.0)
            m = int(math.ceil(L * 1.5))
            vs = np.arange(-m, m + 1) + float(rng.uniform(-0.4, 0.4))
            us = u0 + np.arange(w_b)
            uu, vv = np.meshgrid(us, vs, indexing="ij")
            px = (uu - vv) * inv_s2
            py = (uu + vv) * inv_s2
            band = np.column_stack([px.ravel(), py.ravel(), np.full(uu.size, math.pi * 0.25)])
            ext = inv_s2 * 0.90
            band = band[(band[:, 0] >= ext) & (band[:, 0] <= L - ext) & (band[:, 1] >= ext) & (band[:, 1] <= L - ext)]

            phx, phy = float(rng.uniform(0.0, 0.8)), float(rng.uniform(0.0, 0.8))
            gx = np.arange(0.47 + phx * 0.15, L - 0.46, 0.985)
            gy = np.arange(0.47 + phy * 0.15, L - 0.46, 0.985)
            ggx, ggy = np.meshgrid(gx, gy, indexing="ij")
            up = np.column_stack([ggx.ravel(), ggy.ravel(), np.zeros(ggx.size)])
            up_ok = sat_filter_against(up, band, scale_a=0.86, box_L=L)
            pts = np.vstack([band, up_ok]) if len(band) > 0 else up_ok
            desc = f"hybrid45_band(wb={w_b})"

        elif variant == "core45":
            scale_c = float(rng.uniform(0.30, 0.65))
            M = int(round(scale_c * L))
            m_half = M // 2
            ox = 0.5 * L + float(rng.uniform(-0.35, 0.35))
            oy = 0.5 * L + float(rng.uniform(-0.35, 0.35))
            g = np.arange(-m_half, M - m_half) * 0.99
            gu, gv = np.meshgrid(g, g, indexing="ij")
            px = ox + (gu - gv) * inv_s2
            py = oy + (gu + gv) * inv_s2
            core = np.column_stack([px.ravel(), py.ravel(), np.full(gu.size, math.pi * 0.25)])
            ext = inv_s2 * 0.90
            core = core[(core[:, 0] >= ext) & (core[:, 0] <= L - ext) & (core[:, 1] >= ext) & (core[:, 1] <= L - ext)]

            gx = np.arange(0.48, L - 0.47, 0.99)
            gy = np.arange(0.48, L - 0.47, 0.99)
            ggx, ggy = np.meshgrid(gx, gy, indexing="ij")
            up = np.column_stack([ggx.ravel(), ggy.ravel(), np.zeros(ggx.size)])
            up_ok = sat_filter_against(up, core, scale_a=0.86, box_L=L)
            pts = np.vstack([core, up_ok]) if len(core) > 0 else up_ok
            desc = f"hybrid45_core(M={M})"

        elif variant == "alt45_band":
            # Full-width diagonal corner triangle (below x+y=c_leg) filled at 45°, upright above
            c_leg = float(rng.uniform(0.65, 1.35)) * L
            m = int(math.ceil(L * 1.5))
            us = np.arange(0.0, L * math.sqrt(2.0), 0.99) + inv_s2
            vs = np.arange(-m, m + 1) * 0.99 + float(rng.uniform(-0.4, 0.4))
            uu, vv = np.meshgrid(us, vs, indexing="ij")
            px = (uu - vv) * inv_s2
            py = (uu + vv) * inv_s2
            tri = np.column_stack([px.ravel(), py.ravel(), np.full(uu.size, math.pi * 0.25)])
            ext = inv_s2 * 0.90
            tri = tri[(tri[:, 0] >= ext) & (tri[:, 0] <= L - ext) & (tri[:, 1] >= ext) & (tri[:, 1] <= L - ext) & (tri[:, 0] + tri[:, 1] <= c_leg)]

            kx = int(math.floor(L))
            gx = L - 0.49 - np.arange(kx)[::-1] * 0.995
            gy = L - 0.49 - np.arange(kx)[::-1] * 0.995
            ggx, ggy = np.meshgrid(gx, gy, indexing="ij")
            up = np.column_stack([ggx.ravel(), ggy.ravel(), np.zeros(ggx.size)])
            up_ok = sat_filter_against(up, tri, scale_a=0.86, box_L=L)
            pts = np.vstack([tri, up_ok]) if len(tri) > 0 else up_ok
            desc = f"hybrid45_split(leg={c_leg:.1f})"

        else:
            # Dual opposite 45-deg corners + upright central belt
            c_leg = float(rng.uniform(0.32, 0.65)) * L
            m = int(math.ceil(L * 1.4))
            us = np.arange(0.0, L * math.sqrt(2.0), 0.99) + inv_s2
            vs = np.arange(-m, m + 1) * 0.99
            uu, vv = np.meshgrid(us, vs, indexing="ij")
            px = (uu - vv) * inv_s2
            py = (uu + vv) * inv_s2
            all45 = np.column_stack([px.ravel(), py.ravel(), np.full(uu.size, math.pi * 0.25)])
            ext = inv_s2 * 0.90
            in_box = (all45[:, 0] >= ext) & (all45[:, 0] <= L - ext) & (all45[:, 1] >= ext) & (all45[:, 1] <= L - ext)
            in_sw = all45[:, 0] + all45[:, 1] <= c_leg
            in_ne = (L - all45[:, 0]) + (L - all45[:, 1]) <= c_leg
            corners = all45[in_box & (in_sw | in_ne)]

            gx = np.arange(0.49, L - 0.48, 0.995)
            gy = np.arange(0.49, L - 0.48, 0.995)
            ggx, ggy = np.meshgrid(gx, gy, indexing="ij")
            up = np.column_stack([ggx.ravel(), ggy.ravel(), np.zeros(ggx.size)])
            up_ok = sat_filter_against(up, corners, scale_a=0.86, box_L=L)
            pts = np.vstack([corners, up_ok]) if len(corners) > 0 else up_ok
            desc = f"hybrid45_2c(leg={c_leg:.1f})"

        diff = len(pts) - n
        pen = abs(diff) + (4.0 if diff < -8 else 0.0)
        if pen < best_pen:
            best_pen = pen
            best_pts = pts
            best_desc = desc

    assert best_pts is not None
    pts = fill_wedges_and_complete(best_pts, n, L, rng, wedge_scale=0.86)
    pts[:, :2] += rng.normal(0.0, 0.008, size=(n, 2))
    pts[:, 2] += rng.normal(0.0, 0.008, size=n)
    return pts, best_desc


def seed_clipped_chiral_pinwheel(n: int, L: float, rng: np.random.Generator) -> tuple[np.ndarray, str]:
    """Tilted Pythagorean/chiral interior core + SAT-clipped 4 pinwheel wings tuned to hit ~n."""
    angles_deg = [16.2602, 18.9246, 22.6199, 28.0725, 36.8699, 14.0, 20.0, 26.0, 32.0, 40.0]
    phi = math.radians(float(rng.choice(angles_deg))) * float(rng.choice([-1.0, 1.0]))
    c, s = math.cos(phi), math.sin(phi)

    best_pts = None
    best_pen = 1e9
    best_desc = ""
    wing_mode = str(rng.choice(["upright", "sheared"]))

    for rad_ratio in np.linspace(0.20, 0.42, 6):
        core_half = float(rad_ratio) * L
        m = int(math.ceil(core_half * 2.2)) + 2
        g = np.arange(-m, m + 1) * 0.985
        gx, gy = np.meshgrid(g, g, indexing="ij")
        rx = c * gx - s * gy + 0.5 * L
        ry = s * gx + c * gy + 0.5 * L
        in_core = (np.abs(rx - 0.5 * L) <= core_half) & (np.abs(ry - 0.5 * L) <= core_half)
        pts = np.column_stack([rx[in_core], ry[in_core], np.full(int(in_core.sum()), phi)])

        w_w = max(1.5, 0.5 * L - core_half - 0.1)
        for rot in range(4):
            if wing_mode == "sheared":
                w_cnt = max(2, int(math.floor(w_w)) + 1)
                th_s = solve_shear_angle(w_cnt, w_w) or abs(phi)
                st = make_sheared_strip_exact(0.0, w_w, 0.0, L - w_w * 0.4, w_cnt, min(th_s, 0.65), 1, True, 0.0, 0.89)
            else:
                nx = max(1, int(math.floor(w_w)))
                ny = max(1, int(math.floor(L - w_w * 0.4)))
                xs = np.arange(nx) + 0.5
                ys = np.arange(ny) + 0.5
                sgx, sgy = np.meshgrid(xs, ys, indexing="ij")
                st = np.column_stack([sgx.ravel(), sgy.ravel(), np.zeros(nx * ny)])
            if len(st) == 0:
                continue
            px, py, pa = st[:, 0].copy(), st[:, 1].copy(), st[:, 2].copy()
            if rot == 1:
                px, py, pa = L - st[:, 1], st[:, 0], st[:, 2] + math.pi * 0.5
            elif rot == 2:
                px, py, pa = L - st[:, 0], L - st[:, 1], st[:, 2]
            elif rot == 3:
                px, py, pa = st[:, 1], L - st[:, 0], st[:, 2] - math.pi * 0.5
            wing = np.column_stack([px, py, pa])
            wing_ok = sat_filter_against(wing, pts, scale_a=0.86, box_L=L)
            if len(wing_ok) > 0:
                pts = np.vstack([pts, wing_ok])

        diff = len(pts) - n
        pen = abs(diff) + (4.0 if diff < -8 else 0.0)
        if pen < best_pen:
            best_pen = pen
            best_pts = pts
            best_desc = f"chiral_clip(phi={math.degrees(phi):.1f},{wing_mode},r={rad_ratio:.2f})"

    assert best_pts is not None
    pts = fill_wedges_and_complete(best_pts, n, L, rng, wedge_scale=0.86)
    pts[:, :2] += rng.normal(0.0, 0.008, size=(n, 2))
    pts[:, 2] += rng.normal(0.0, 0.008, size=n)
    return pts, best_desc


def seed_2tier_stepped_pocket(n: int, L: float, rng: np.random.Generator) -> tuple[np.ndarray, str]:
    """Split West/South sheared boundary into 2 shallow-angle strips screened for capacity."""
    k = int(math.floor(math.sqrt(n)))
    best_pts = None
    best_pen = 1e9
    best_desc = ""

    for _ in range(10):
        Kx = int(rng.integers(max(4, k - 12), max(5, k - 3)))
        Ky = int(rng.integers(max(4, k - 12), max(5, k - 3)))
        W1, W2 = L - Kx, L - Ky

        xs_c = L - Kx + np.arange(Kx) + 0.5
        ys_c = L - Ky + np.arange(Ky) + 0.5
        gx, gy = np.meshgrid(xs_c, ys_c, indexing="ij")
        pts = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(Kx * Ky)])

        if W1 >= 3.5:
            P1 = int(math.floor(W1))
            f1 = W1 - P1
            p1a = max(1, P1 // 2)
            W1a = p1a + 0.5 * (f1 + 1.0)
            W1b = W1 - W1a
            w1a, w1b = int(math.floor(W1a)) + 1, int(math.floor(W1b)) + 1
            th1a = solve_shear_angle(w1a, W1a - 1e-4) or 0.34
            th1b = solve_shear_angle(w1b, W1b - 1e-4) or 0.34
            for x0, x1, ww, th, sgn in [(0.0, W1a, w1a, min(th1a, 0.65), 1), (W1a, W1, w1b, min(th1b, 0.65), -1)]:
                st = make_sheared_strip_exact(x0, x1, 0.0, L, ww, th, sgn, True, float(rng.uniform(0.0, 0.8)), 0.89)
                st = sat_filter_against(st, pts, scale_a=0.86, box_L=L)
                if len(st) > 0:
                    pts = np.vstack([pts, st])

        if W2 >= 3.5:
            P2 = int(math.floor(W2))
            f2 = W2 - P2
            p2a = max(1, P2 // 2)
            W2a = p2a + 0.5 * (f2 + 1.0)
            W2b = W2 - W2a
            for y0, y1 in [(0.0, W2a), (W2a, W2)]:
                span = y1 - y0
                ww = max(1, int(math.floor(span)) + 1)
                th = solve_shear_angle(ww, span - 1e-4) or 0.34
                st = make_sheared_strip_exact(W1, L, y0, y1, ww, min(th, 0.65), int(rng.choice([-1, 1])), False, 0.2, 0.89)
                st = sat_filter_against(st, pts, scale_a=0.86, box_L=L)
                if len(st) > 0:
                    pts = np.vstack([pts, st])
        elif W2 >= 1.1:
            ww = max(1, int(math.floor(W2)) + 1)
            th = solve_shear_angle(ww, W2 - 1e-4) or 0.38
            st = make_sheared_strip_exact(W1, L, 0.0, W2, ww, min(th, 0.68), 1, False, 0.1, 0.89)
            st = sat_filter_against(st, pts, scale_a=0.86, box_L=L)
            if len(st) > 0:
                pts = np.vstack([pts, st])

        diff = len(pts) - n
        pen = abs(diff) + (4.0 if diff < -8 else 0.0)
        if pen < best_pen:
            best_pen = pen
            best_pts = pts
            best_desc = f"step2tier(K={Kx}x{Ky})"

    assert best_pts is not None
    pts = fill_wedges_and_complete(best_pts, n, L, rng, wedge_scale=0.86)
    pts[:, :2] += rng.normal(0.0, 0.008, size=(n, 2))
    pts[:, 2] += rng.normal(0.0, 0.008, size=n)
    return pts, best_desc


def seed_pythagorean_bicrystal(n: int, L: float, rng: np.random.Generator) -> tuple[np.ndarray, str]:
    """Corner Pythagorean tilted sub-block + upright L-frame or symmetric tilt bicrystal."""
    best_pts = None
    best_pen = 1e9
    best_desc = ""

    use_pyth = rng.random() < 0.65
    for _ in range(8):
        if use_pyth:
            phi = float(rng.choice(PYTHAGOREAN_ANGLES[:5])) * float(rng.choice([-1.0, 1.0]))
            c, s = math.cos(phi), math.sin(phi)
            cw = float(rng.uniform(0.32, 0.68)) * L
            ch = float(rng.uniform(0.32, 0.68)) * L
            m = int(math.ceil(max(cw, ch) * 1.6))
            g = np.arange(-m, m + 1) * 0.99
            gx, gy = np.meshgrid(g, g, indexing="ij")
            rx = c * gx - s * gy + 0.5 * cw
            ry = s * gx + c * gy + 0.5 * ch
            ext = 0.45 * (abs(c) + abs(s))
            in_c = (rx >= ext) & (rx <= cw) & (ry >= ext) & (ry <= ch)
            pts = np.column_stack([rx[in_c], ry[in_c], np.full(int(in_c.sum()), phi)])

            k = int(math.floor(L))
            ux = L - 0.49 - np.arange(k)[::-1] * 0.995
            uy = L - 0.49 - np.arange(k)[::-1] * 0.995
            ugx, ugy = np.meshgrid(ux, uy, indexing="ij")
            up = np.column_stack([ugx.ravel(), ugy.ravel(), np.zeros(ugx.size)])
            up_ok = sat_filter_against(up, pts, scale_a=0.86, box_L=L)
            if len(up_ok) > 0:
                pts = np.vstack([pts, up_ok])
            desc = f"pyth_corner(phi={math.degrees(phi):.1f})"
        else:
            alpha = math.radians(float(rng.choice([12.0, 16.2602, 18.9246, 22.6199, 28.0725])))
            border = int(rng.choice([0, 1, 2]))
            if border > 0:
                gx = np.arange(0.49, L - 0.45, 0.99)
                gy = np.arange(0.49, L - 0.45, 0.99)
                ggx, ggy = np.meshgrid(gx, gy, indexing="ij")
                on_b = (ggx < border + 0.1) | (ggx > L - border - 0.1) | (ggy < border + 0.1) | (ggy > L - border - 0.1)
                pts = np.column_stack([ggx[on_b], ggy[on_b], np.zeros(int(on_b.sum()))])
            else:
                pts = np.zeros((0, 3), dtype=np.float64)
            m = int(math.ceil(L * 1.4))
            grid = np.arange(-m, m + 1) * 0.985
            ug, vg = np.meshgrid(grid, grid, indexing="ij")
            lo = border + 0.45
            hi = L - border - 0.45
            for sgn, x_lo, x_hi in [(-1.0, lo, 0.5 * L), (1.0, 0.5 * L, hi)]:
                ang = sgn * alpha
                ca, sa = math.cos(ang), math.sin(ang)
                rx = ca * ug - sa * vg + 0.5 * L
                ry = sa * ug + ca * vg + 0.5 * L
                keep = (rx >= x_lo) & (rx <= x_hi) & (ry >= lo) & (ry <= hi)
                cand = np.column_stack([rx[keep], ry[keep], np.full(int(keep.sum()), ang)])
                cand_ok = sat_filter_against(cand, pts, scale_a=0.86, box_L=L)
                if len(cand_ok) > 0:
                    pts = np.vstack([pts, cand_ok]) if len(pts) else cand_ok
            desc = f"sym_bicryst(a={math.degrees(alpha):.1f},b={border})"

        diff = len(pts) - n
        pen = abs(diff) + (4.0 if diff < -8 else 0.0)
        if pen < best_pen:
            best_pen = pen
            best_pts = pts
            best_desc = desc

    assert best_pts is not None
    pts = fill_wedges_and_complete(best_pts, n, L, rng, wedge_scale=0.86)
    pts[:, :2] += rng.normal(0.0, 0.008, size=(n, 2))
    pts[:, 2] += rng.normal(0.0, 0.008, size=n)
    return pts, best_desc


SEED_BUILDERS_V2: list[tuple[str, Callable[[int, float, np.random.Generator], tuple[np.ndarray, str]]]] = [
    ("tapered_sheared_l", seed_tapered_sheared_l_pocket),
    ("tapered_sheared_l_b", seed_tapered_sheared_l_pocket),
    ("pinwheel_4pocket", seed_pinwheel_4pocket),
    ("hybrid45_upright", seed_45_upright_hybrids),
    ("clipped_chiral", seed_clipped_chiral_pinwheel),
    ("step2tier_pocket", seed_2tier_stepped_pocket),
    ("pythagorean_bicryst", seed_pythagorean_bicrystal),
]


def build_seed_batch_v2(
    n: int,
    L_targets: list[float],
    batch_size: int,
    rng_seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Build exact-capacity SAT-clipped multi-grain batch of size `batch_size`."""
    rng = np.random.default_rng(rng_seed)
    pos = np.zeros((batch_size, n, 2), dtype=np.float64)
    th = np.zeros((batch_size, n), dtype=np.float64)
    box_L = np.zeros(batch_size, dtype=np.float64)
    meta: list[str] = []

    for b in range(batch_size):
        L_val = float(L_targets[b % len(L_targets)])
        _, builder = SEED_BUILDERS_V2[b % len(SEED_BUILDERS_V2)]
        pts, desc = builder(n, L_val, rng)
        pos[b] = pts[:, :2]
        th[b] = pts[:, 2]
        box_L[b] = L_val
        meta.append(f"{desc}@L={L_val:.5f}")

    return pos, th, box_L, meta
