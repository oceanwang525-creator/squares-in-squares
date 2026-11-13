"""From-scratch oriented multi-grain and catalog-inspired seed generators.

All seeds are built purely from (n, L_target) geometric primitives — NEVER from
incumbent packings in merged.csv. Designed for inflation from a_0 = 0.88 -> 1.00.
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
    # w*cos(t) + sin(t) = R * cos(t - phi), R = sqrt(w^2 + 1), phi = atan(1/w)
    phi = math.atan2(1.0, float(w))
    ratio = min(1.0, max(-1.0, W / v_peak))
    th = phi + math.acos(ratio)
    if th < 0.0 or th > math.pi * 0.25 + 1e-9:
        return None
    return min(th, math.pi * 0.25)


def make_sheared_strip(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    w: int,
    th: float,
    sign: int,
    vertical_strip: bool = True,
    y_phase: float = 0.0,
) -> np.ndarray:
    """Generate zero-sawtooth sheared rows inside [x_min, x_max] x [y_min, y_max]."""
    if vertical_strip:
        W = x_max - x_min
        H_min, H_max = y_min, y_max
    else:
        W = y_max - y_min
        H_min, H_max = x_min, x_max

    c, s = math.cos(th), math.sin(th)
    hx = 0.5 * (c + s)
    dy_row = 1.0 / max(c, 1e-6)
    row_span = w * c + s
    x_pad = 0.5 * max(0.0, W - row_span)

    r_min = -int(math.ceil(w * s / dy_row)) - 2
    r_max = int(math.ceil((H_max - H_min + w * s) / dy_row)) + 2

    pts: list[tuple[float, float, float]] = []
    for r in range(r_min, r_max + 1):
        y0 = H_min + hx + (r + y_phase) * dy_row
        for k in range(w):
            uk = x_pad + hx + k * c
            vk = y0 + sign * k * s
            if uk - hx >= -0.05 and uk + hx <= W + 0.05 and vk - hx >= H_min - 0.05 and vk + hx <= H_max + 0.05:
                ang = sign * th
                if vertical_strip:
                    pts.append((x_min + uk, vk, ang))
                else:
                    pts.append((vk, y_min + uk, -ang))
    return np.asarray(pts, dtype=np.float64) if pts else np.zeros((0, 3), dtype=np.float64)


def complete_to_n(
    pts: np.ndarray,
    n: int,
    L: float,
    rng: np.random.Generator,
    min_init_sep: float = 0.55,
) -> np.ndarray:
    """Dedup crowded overlaps, trim excess or void-fill deficit to exactly `n` squares."""
    if len(pts) == 0:
        pts = np.zeros((0, 3), dtype=np.float64)
    else:
        pts = pts.copy()
        pts[:, 0] = np.clip(pts[:, 0], 0.45, L - 0.45)
        pts[:, 1] = np.clip(pts[:, 1], 0.45, L - 0.45)

    # Greedy dedup if any seeds placed squares nearly on top of each other
    if len(pts) > 1:
        keep: list[int] = []
        d2_mat = np.sum((pts[:, None, :2] - pts[None, :, :2]) ** 2, axis=-1)
        np.fill_diagonal(d2_mat, 1e9)
        removed = np.zeros(len(pts), dtype=bool)
        for i in range(len(pts)):
            if removed[i]:
                continue
            keep.append(i)
            too_close = d2_mat[i] < (min_init_sep * min_init_sep)
            removed[too_close] = True
            removed[i] = False
        pts = pts[keep]

    # Trim excess by iteratively dropping the most crowded square
    while len(pts) > n:
        d2_mat = np.sum((pts[:, None, :2] - pts[None, :, :2]) ** 2, axis=-1)
        np.fill_diagonal(d2_mat, 1e9)
        min_d2 = d2_mat.min(axis=1)
        # Among near-ties, break randomly so we get diverse trims
        thresh = min_d2.min() + 0.01
        cands = np.flatnonzero(min_d2 <= thresh)
        drop_idx = int(rng.choice(cands))
        pts = np.delete(pts, drop_idx, axis=0)

    # Fill deficit using largest-void greedy placement
    if len(pts) < n:
        step = max(0.28, L / 85.0)
        g = np.arange(0.50, L - 0.49, step)
        gx, gy = np.meshgrid(g, g, indexing="ij")
        cands = np.column_stack([gx.ravel(), gy.ravel()])
        cands += rng.uniform(-0.08, 0.08, size=cands.shape)
        cands = np.clip(cands, 0.50, L - 0.50)

        if len(pts) > 0:
            # Chunked min distance to avoid large allocations
            min_d2 = np.full(len(cands), 1e9, dtype=np.float64)
            near_ang = np.zeros(len(cands), dtype=np.float64)
            for i0 in range(0, len(cands), 1024):
                c_chunk = cands[i0 : i0 + 1024]
                d2 = np.sum((c_chunk[:, None, :] - pts[None, :, :2]) ** 2, axis=-1)
                idx_m = np.argmin(d2, axis=1)
                min_d2[i0 : i0 + 1024] = d2[np.arange(len(c_chunk)), idx_m]
                near_ang[i0 : i0 + 1024] = pts[idx_m, 2]
        else:
            min_d2 = np.full(len(cands), 1e9, dtype=np.float64)
            near_ang = np.zeros(len(cands), dtype=np.float64)

        added: list[tuple[float, float, float]] = []
        need = n - len(pts)
        for _ in range(need):
            best_i = int(np.argmax(min_d2))
            px, py = float(cands[best_i, 0]), float(cands[best_i, 1])
            # Try inheriting nearest neighbor angle or small variation
            mode = rng.random()
            if mode < 0.65:
                ang = float(near_ang[best_i])
            elif mode < 0.85:
                ang = 0.0
            else:
                ang = float(rng.choice([math.pi * 0.25, -float(near_ang[best_i])]))
            added.append((px, py, ang))
            new_d2 = (cands[:, 0] - px) ** 2 + (cands[:, 1] - py) ** 2
            closer = new_d2 < min_d2
            min_d2 = np.minimum(min_d2, new_d2)
            near_ang[closer] = ang
            min_d2[best_i] = -1.0

        pts = np.vstack([pts, np.asarray(added, dtype=np.float64)]) if len(pts) else np.asarray(added, dtype=np.float64)

    return pts


def seed_sheared_l_pocket(n: int, L: float, rng: np.random.Generator) -> tuple[np.ndarray, str]:
    """NE upright core + West & South zero-sawtooth sheared pockets + SW block/miter."""
    k = int(math.floor(math.sqrt(n)))
    candidates: list[tuple[int, int, int, int, float, float]] = []
    for Kx in range(max(4, k - 12), max(5, k - 3)):
        W1 = L - Kx
        if W1 < 2.4:
            continue
        for w1 in (int(math.floor(W1)) + 1, int(math.floor(W1)) + 2):
            th1 = solve_shear_angle(w1, W1 - 1e-4)
            if th1 is None or th1 > 0.68:
                continue
            for Ky in range(max(4, k - 12), max(5, k - 3)):
                W2 = L - Ky
                if W2 < 2.4:
                    continue
                for w2 in (int(math.floor(W2)) + 1, int(math.floor(W2)) + 2):
                    th2 = solve_shear_angle(w2, W2 - 1e-4)
                    if th2 is None or th2 > 0.68:
                        continue
                    candidates.append((Kx, Ky, w1, w2, th1, th2))

    if not candidates:
        return seed_pinched_4edge_frame(n, L, rng)

    Kx, Ky, w1, w2, th1, th2 = candidates[int(rng.integers(len(candidates)))]
    W1 = L - Kx
    W2 = L - Ky
    pitch = float(rng.choice([0.985, 0.992, 1.0]))
    sign1 = int(rng.choice([-1, 1]))
    sign2 = int(rng.choice([-1, 1]))
    y_ph1 = float(rng.uniform(0.0, 0.9))
    y_ph2 = float(rng.uniform(0.0, 0.9))
    sw_mode = str(rng.choice(["upright", "miter45", "sheared_ext"]))

    # NE upright core
    xs_c = L - Kx * pitch + (np.arange(Kx) + 0.5) * pitch
    ys_c = L - Ky * pitch + (np.arange(Ky) + 0.5) * pitch
    gx, gy = np.meshgrid(xs_c, ys_c, indexing="ij")
    core = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(Kx * Ky)])

    if sw_mode == "upright":
        p1, p2 = int(math.floor(W1)), int(math.floor(W2))
        sx = np.arange(p1) + 0.5
        sy = np.arange(p2) + 0.5
        sgx, sgy = np.meshgrid(sx, sy, indexing="ij")
        sw = np.column_stack([sgx.ravel(), sgy.ravel(), np.zeros(p1 * p2)])
        west = make_sheared_strip(0.0, W1, float(p2), L, w1, th1, sign1, True, y_ph1)
        south = make_sheared_strip(float(p1), L, 0.0, W2, w2, th2, sign2, False, y_ph2)
    elif sw_mode == "miter45":
        # Miter along diagonal x = y in SW, with small 45-deg wedge
        west = make_sheared_strip(0.0, W1, 0.0, L, w1, th1, sign1, True, y_ph1)
        west = west[west[:, 1] >= west[:, 0] + 0.25]
        south = make_sheared_strip(0.0, L, 0.0, W2, w2, th2, sign2, False, y_ph2)
        south = south[south[:, 0] >= south[:, 1] + 0.25]
        m_diag = int(min(W1, W2) * 1.3)
        sw_pts = [(0.5 + 0.72 * i, 0.5 + 0.72 * i, math.pi * 0.25) for i in range(max(1, m_diag))]
        sw = np.asarray(sw_pts, dtype=np.float64)
    else:
        west = make_sheared_strip(0.0, W1, 0.0, L, w1, th1, sign1, True, y_ph1)
        south = make_sheared_strip(W1, L, 0.0, W2, w2, th2, sign2, False, y_ph2)
        sw = np.zeros((0, 3), dtype=np.float64)

    pts = np.vstack([core, sw, west, south])
    pts = complete_to_n(pts, n, L, rng)
    pts[:, :2] += rng.normal(0.0, 0.012, size=(n, 2))
    pts[:, 2] += rng.normal(0.0, 0.01, size=n)
    return pts, f"sheared_l_pocket(K={Kx}x{Ky},w={w1}/{w2},{sw_mode})"


def seed_pinched_4edge_frame(n: int, L: float, rng: np.random.Generator) -> tuple[np.ndarray, str]:
    """Interior upright core surrounded by 4 asymmetric sheared boundary strips."""
    k = int(math.floor(math.sqrt(n)))
    Kx = int(rng.integers(max(4, k - 14), max(5, k - 5)))
    Ky = int(rng.integers(max(4, k - 14), max(5, k - 5)))
    rem_x = L - Kx
    rem_y = L - Ky
    Px = int(math.floor(rem_x))
    Py = int(math.floor(rem_y))
    fx = rem_x - Px
    fy = rem_y - Py

    # Split rem_x = W_W + W_E so both W_W and W_E have large fractional part (fx+1)/2
    pw = max(1, int(rng.integers(1, max(2, Px))))
    pe = max(1, Px - 1 - pw) if Px >= 3 else 1
    if pw + pe + 1 != Px and Px >= 3:
        pe = Px - 1 - pw
    f_w = 0.5 * (fx + 1.0) + float(rng.uniform(-0.08, 0.08))
    W_W = max(1.5, pw + f_w)
    W_E = max(1.5, rem_x - W_W)

    ps = max(1, int(rng.integers(1, max(2, Py))))
    pn = max(1, Py - 1 - ps) if Py >= 3 else 1
    if ps + pn + 1 != Py and Py >= 3:
        pn = Py - 1 - ps
    f_s = 0.5 * (fy + 1.0) + float(rng.uniform(-0.08, 0.08))
    W_S = max(1.5, ps + f_s)
    W_N = max(1.5, rem_y - W_S)

    pitch = float(rng.choice([0.985, 0.995, 1.0]))
    xs_c = W_W + (np.arange(Kx) + 0.5) * pitch
    ys_c = W_S + (np.arange(Ky) + 0.5) * pitch
    gx, gy = np.meshgrid(xs_c, ys_c, indexing="ij")
    core = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(Kx * Ky)])

    strips = [core]
    handed = int(rng.choice([-1, 1]))
    for idx_s, (xmin, xmax, ymin, ymax, W, vert) in enumerate([
        (0.0, W_W, 0.0, L - W_N, W_W, True),
        (L - W_E, L, W_S, L, W_E, True),
        (W_W, L, 0.0, W_S, W_S, False),
        (0.0, L - W_W, L - W_N, L, W_N, False),
    ]):
        w = int(math.floor(W)) + 1
        th = solve_shear_angle(w, W - 1e-4)
        if th is None or th > 0.72:
            # Fallback to upright subgrid in that strip
            nx = max(1, int(math.floor(xmax - xmin)))
            ny = max(1, int(math.floor(ymax - ymin)))
            sx = xmin + np.arange(nx) + 0.5
            sy = ymin + np.arange(ny) + 0.5
            sgx, sgy = np.meshgrid(sx, sy, indexing="ij")
            strips.append(np.column_stack([sgx.ravel(), sgy.ravel(), np.zeros(nx * ny)]))
        else:
            sgn = handed * (1 if idx_s % 2 == 0 else -1)
            strips.append(make_sheared_strip(xmin, xmax, ymin, ymax, w, th, sgn, vert, float(rng.uniform(0, 0.8))))

    pts = np.vstack(strips)
    pts = complete_to_n(pts, n, L, rng)
    pts[:, :2] += rng.normal(0.0, 0.012, size=(n, 2))
    pts[:, 2] += rng.normal(0.0, 0.01, size=n)
    return pts, f"pinched_4edge(K={Kx}x{Ky},W={W_W:.2f}/{W_E:.2f})"


def seed_chiral_tilted_core(n: int, L: float, rng: np.random.Generator) -> tuple[np.ndarray, str]:
    """Tilted interior core at angle phi (incl. Pythagorean) + 4 symmetric wings."""
    angles_deg = [16.2602, 22.6199, 28.0725, 36.8699, 12.0, 19.5, 25.0, 32.0, 41.0]
    phi = math.radians(float(rng.choice(angles_deg))) * float(rng.choice([-1.0, 1.0]))
    rad_ratio = float(rng.uniform(0.24, 0.40))
    core_half = rad_ratio * L
    pitch = float(rng.choice([0.97, 0.99, 1.0]))

    m = int(math.ceil(core_half * 2.2)) + 2
    g = (np.arange(-m, m + 1)) * pitch
    gx, gy = np.meshgrid(g, g, indexing="ij")
    c, s = math.cos(phi), math.sin(phi)
    rx = c * gx - s * gy + 0.5 * L
    ry = s * gx + c * gy + 0.5 * L

    # Keep squares comfortably inside octagonal/square core region
    in_core = (np.abs(rx - 0.5 * L) <= core_half) & (np.abs(ry - 0.5 * L) <= core_half)
    core = np.column_stack([rx[in_core], ry[in_core], np.full(int(in_core.sum()), phi)])

    # Surrounding 4 wings along walls (upright or weakly sheared)
    wing_th = float(rng.choice([0.0, 0.0, math.radians(18.0), math.radians(24.0), -phi]))
    w_w = max(2.0, 0.5 * L - core_half - 0.35)
    wing_pts: list[np.ndarray] = [core]
    if abs(wing_th) < 0.05:
        nx = max(1, int(math.floor(w_w)))
        ny = max(1, int(math.floor(L - w_w)))
        # 4 pinwheel rectangles around the tilted core
        for rot in range(4):
            xs = np.arange(nx) + 0.5
            ys = np.arange(ny) + 0.5
            sgx, sgy = np.meshgrid(xs, ys, indexing="ij")
            px, py = sgx.ravel(), sgy.ravel()
            if rot == 1:
                px, py = L - py, px
            elif rot == 2:
                px, py = L - px, L - py
            elif rot == 3:
                px, py = py, L - px
            wing_pts.append(np.column_stack([px, py, np.zeros(len(px))]))
    else:
        w_cnt = max(2, int(math.floor(w_w)) + 1)
        th_s = solve_shear_angle(w_cnt, w_w) or abs(wing_th)
        for rot in range(4):
            st = make_sheared_strip(0.0, w_w, 0.0, L - w_w, w_cnt, min(th_s, 0.65), 1, True)
            px, py, pa = st[:, 0], st[:, 1], st[:, 2]
            if rot == 1:
                px, py, pa = L - py, px, pa + math.pi * 0.5
            elif rot == 2:
                px, py, pa = L - px, L - py, pa
            elif rot == 3:
                px, py, pa = py, L - px, pa - math.pi * 0.5
            wing_pts.append(np.column_stack([px, py, pa]))

    pts = np.vstack(wing_pts)
    pts = complete_to_n(pts, n, L, rng)
    pts[:, :2] += rng.normal(0.0, 0.015, size=(n, 2))
    pts[:, 2] += rng.normal(0.0, 0.015, size=n)
    return pts, f"chiral_core(phi={math.degrees(phi):.1f}deg)"


def seed_bicrystal_chevron(n: int, L: float, rng: np.random.Generator) -> tuple[np.ndarray, str]:
    """Symmetric tilt (+alpha, -alpha) bicrystal or V-chevron sheared pockets."""
    mode = str(rng.choice(["sym_tilt_2grain", "v_chevron_pocket"]))
    if mode == "sym_tilt_2grain":
        alpha = math.radians(float(rng.choice([8.0, 12.0, 16.2602, 20.0, 22.6199, 26.0])))
        border = int(rng.choice([1, 2, 3]))
        pitch = float(rng.choice([0.975, 0.99, 1.0]))
        pts_list: list[np.ndarray] = []

        # Thin upright wall border (optional)
        if border > 0:
            g_x = np.arange(0.5, L - 0.4, 1.0)
            g_y = np.arange(0.5, L - 0.4, 1.0)
            gx, gy = np.meshgrid(g_x, g_y, indexing="ij")
            on_edge = (gx < border + 0.1) | (gx > L - border - 0.1) | (gy < border + 0.1) | (gy > L - border - 0.1)
            pts_list.append(np.column_stack([gx[on_edge], gy[on_edge], np.zeros(int(on_edge.sum()))]))

        # Two symmetric tilted grains meeting along x = L/2 (or y = L/2)
        lo = border + 0.4
        hi = L - border - 0.4
        m = int(math.ceil(L * 1.4))
        grid = np.arange(-m, m + 1) * pitch
        ggx, ggy = np.meshgrid(grid, grid, indexing="ij")
        for grain_sgn, x_lo, x_hi in [(-1.0, lo, 0.5 * L - 0.1), (1.0, 0.5 * L + 0.1, hi)]:
            ang = grain_sgn * alpha
            c, s = math.cos(ang), math.sin(ang)
            rx = c * ggx - s * ggy + 0.5 * L
            ry = s * ggx + c * ggy + 0.5 * L
            keep = (rx >= x_lo) & (rx <= x_hi) & (ry >= lo) & (ry <= hi)
            pts_list.append(np.column_stack([rx[keep], ry[keep], np.full(int(keep.sum()), ang)]))

        pts = np.vstack(pts_list)
        pts = complete_to_n(pts, n, L, rng)
        return pts, f"bicrystal_2grain(alpha={math.degrees(alpha):.1f}deg,b={border})"

    # V-chevron pocket: NE upright core + W/S pockets split into +theta / -theta halves
    k = int(math.floor(math.sqrt(n)))
    Kx = int(rng.integers(max(4, k - 10), max(5, k - 4)))
    Ky = int(rng.integers(max(4, k - 10), max(5, k - 4)))
    W1, W2 = L - Kx, L - Ky
    w1, w2 = int(math.floor(W1)) + 1, int(math.floor(W2)) + 1
    th1 = solve_shear_angle(w1, W1 - 1e-4) or 0.35
    th2 = solve_shear_angle(w2, W2 - 1e-4) or 0.35
    th1 = min(th1, 0.65)
    th2 = min(th2, 0.65)

    xs_c = L - Kx + np.arange(Kx) + 0.5
    ys_c = L - Ky + np.arange(Ky) + 0.5
    gx, gy = np.meshgrid(xs_c, ys_c, indexing="ij")
    core = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(Kx * Ky)])

    ymid = 0.5 * (W2 + L)
    xmid = 0.5 * (W1 + L)
    w_bot = make_sheared_strip(0.0, W1, W2, ymid - 0.1, w1, th1, 1, True)
    w_top = make_sheared_strip(0.0, W1, ymid + 0.1, L, w1, th1, -1, True)
    s_l = make_sheared_strip(W1, xmid - 0.1, 0.0, W2, w2, th2, 1, False)
    s_r = make_sheared_strip(xmid + 0.1, L, 0.0, W2, w2, th2, -1, False)

    pts = np.vstack([core, w_bot, w_top, s_l, s_r])
    pts = complete_to_n(pts, n, L, rng)
    return pts, f"v_chevron(K={Kx}x{Ky})"


def seed_catalog_corner_lift(n: int, L: float, rng: np.random.Generator) -> tuple[np.ndarray, str]:
    """K x K frame with an m x m corner replaced by Pythagorean / 45-deg diagonal motif."""
    k = int(math.floor(math.sqrt(n)))
    m = int(rng.choice([5, 6, 8, 10, 12, 14]))
    m = min(m, k - 2)
    motif = str(rng.choice(["pythagorean", "diag45_gobel", "pinwheel_2ang"]))
    pitch = float(rng.choice([0.985, 0.995, 1.0]))

    # Upright K x K lattice in [0, L]^2, removing SW [0, m] x [0, m] corner
    g = L - k * pitch + (np.arange(k) + 0.5) * pitch
    gx, gy = np.meshgrid(g, g, indexing="ij")
    # Remove SW corner of size m x m (in lattice indices)
    mask_keep = ~((gx < L - (k - m) * pitch - 0.1) & (gy < L - (k - m) * pitch - 0.1))
    outer = np.column_stack([gx[mask_keep], gy[mask_keep], np.zeros(int(mask_keep.sum()))])

    cw = L - (k - m) * pitch
    if motif == "pythagorean":
        phi = float(rng.choice(PYTHAGOREAN_ANGLES[:4])) * float(rng.choice([-1.0, 1.0]))
        c, s = math.cos(phi), math.sin(phi)
        r_m = int(math.ceil(cw * 1.5))
        gg = np.arange(-r_m, r_m + 1) * 0.98
        ug, vg = np.meshgrid(gg, gg, indexing="ij")
        rx = c * ug - s * vg + 0.5 * cw
        ry = s * ug + c * vg + 0.5 * cw
        ok = (rx >= 0.5) & (rx <= cw - 0.35) & (ry >= 0.5) & (ry <= cw - 0.35)
        corner = np.column_stack([rx[ok], ry[ok], np.full(int(ok.sum()), phi)])
        tag = f"pyth_{math.degrees(abs(phi)):.1f}deg"
    elif motif == "diag45_gobel":
        # 45-deg diagonal band across the corner + small upright triangle
        w_band = max(1, int(rng.integers(1, 5)))
        inv_s2 = 1.0 / math.sqrt(2.0)
        pts_c: list[tuple[float, float, float]] = []
        a1 = max(1, int(cw // 2) - w_band)
        for i in range(a1):
            for j in range(a1 - i):
                pts_c.append((0.5 + i, 0.5 + j, 0.0))
        u0 = (a1 + 1.0) * inv_s2 + 0.5
        for kb in range(w_band):
            u = u0 + kb * 1.0
            v_lim = min(u - 0.8, math.sqrt(2.0) * cw - u - 0.8)
            if v_lim >= 0.0:
                cnt = int(math.floor(2.0 * v_lim)) + 1
                v_start = -0.5 * (cnt - 1)
                for q in range(cnt):
                    v = v_start + q
                    px, py = (u - v) * inv_s2, (u + v) * inv_s2
                    if 0.45 <= px <= cw - 0.35 and 0.45 <= py <= cw - 0.35:
                        pts_c.append((px, py, math.pi * 0.25))
        corner = np.asarray(pts_c, dtype=np.float64) if pts_c else np.zeros((0, 3))
        tag = f"diag45_w{w_band}"
    else:
        # 2-angle sheared corner block
        th = float(rng.uniform(0.28, 0.60))
        w = max(2, int(math.floor(cw)) + 1)
        th_s = solve_shear_angle(w, cw) or th
        corner = make_sheared_strip(0.0, cw, 0.0, cw, w, min(th_s, 0.68), 1, True)
        tag = f"shear_m{m}"

    pts = np.vstack([outer, corner]) if len(corner) > 0 else outer
    pts = complete_to_n(pts, n, L, rng)
    pts[:, :2] += rng.normal(0.0, 0.012, size=(n, 2))
    return pts, f"catalog_lift(m={m},{tag})"


SEED_BUILDERS: list[tuple[str, Callable[[int, float, np.random.Generator], tuple[np.ndarray, str]]]] = [
    ("sheared_l_pocket", seed_sheared_l_pocket),
    ("pinched_4edge_frame", seed_pinched_4edge_frame),
    ("chiral_tilted_core", seed_chiral_tilted_core),
    ("bicrystal_chevron", seed_bicrystal_chevron),
    ("catalog_corner_lift", seed_catalog_corner_lift),
]


def build_seed_batch(
    n: int,
    L_targets: list[float],
    batch_size: int,
    rng_seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Build a diverse multi-grain batch of size `batch_size` across `L_targets`."""
    rng = np.random.default_rng(rng_seed)
    pos = np.zeros((batch_size, n, 2), dtype=np.float64)
    th = np.zeros((batch_size, n), dtype=np.float64)
    box_L = np.zeros(batch_size, dtype=np.float64)
    meta: list[str] = []

    for b in range(batch_size):
        L_val = float(L_targets[b % len(L_targets)])
        # Cycle cleanly over all 5 distinct multi-grain seed families
        _, builder = SEED_BUILDERS[b % len(SEED_BUILDERS)]
        pts, desc = builder(n, L_val, rng)
        pos[b] = pts[:, :2]
        th[b] = pts[:, 2]
        box_L[b] = L_val
        meta.append(f"{desc}@L={L_val:.5f}")

    return pos, th, box_L, meta
