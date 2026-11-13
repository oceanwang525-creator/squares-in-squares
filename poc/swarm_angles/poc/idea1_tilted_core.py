"""Idea 1: Tilted central core (m1 x m2 or 2 symmetric blocks) + 4 corner upright staircases.

Fast scan on calibration (50, 626) and targets (401, 442, 530, 626).
"""
from __future__ import annotations

import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numba as nb
import numpy as np

import utils as u

OUT_DIR = Path(__file__).resolve().parents[1] / "docs"


@nb.njit(cache=True)
def count_upright_fill_around_tilted(
    L: float,
    tx: np.ndarray,
    ty: np.ndarray,
    th: np.ndarray,
    min_gap: float = -1e-4,
) -> tuple[int, np.ndarray, np.ndarray]:
    """Place maximal corner-anchored upright squares around tilted squares inside [0, L]^2.

    Each corner quadrant owns cells strictly inside its half-plane / quadrant (with
    clean integer partitioning along L/2), plus boundary cells resolved greedily.
    """
    k = int(math.floor(L))
    tc = np.cos(th)
    ts = np.sin(th)
    thx = 0.5 * (np.abs(tc) + np.abs(ts))

    # Candidate upright squares from 4 corners:
    # SW (i, j): center (0.5 + i, 0.5 + j)
    # SE (i, j): center (L - 0.5 - i, 0.5 + j)
    # NE (i, j): center (L - 0.5 - i, L - 0.5 - j)
    # NW (i, j): center (0.5 + i, L - 0.5 - j)
    # Use a partition threshold so SW/SE don't collide on x and SW/NW don't collide on y:
    # Actually, let each corner fill up to floor(L/2) or greedily add all valid cells
    # sorted by corner distance (i + j) so corner staircases are maximally packed!
    max_cand = 4 * k * k
    cx_list = np.empty(max_cand, dtype=np.float64)
    cy_list = np.empty(max_cand, dtype=np.float64)
    dist_list = np.empty(max_cand, dtype=np.float64)
    cnt = 0

    half_k = int(math.ceil(L / 2.0)) + 2
    for d in range(2 * k):
        for i in range(min(d + 1, k)):
            j = d - i
            if j >= k:
                continue
            # 4 corners at Manhattan distance d = i + j
            pts = (
                (0.5 + i, 0.5 + j),
                (L - 0.5 - i, 0.5 + j),
                (L - 0.5 - i, L - 0.5 - j),
                (0.5 + i, L - 0.5 - j),
            )
            for px, py in pts:
                if px < 0.5 - 1e-9 or px > L - 0.5 + 1e-9 or py < 0.5 - 1e-9 or py > L - 0.5 + 1e-9:
                    continue
                # Check SAT vs tilted squares
                ok = True
                for m in range(len(tx)):
                    dx = px - tx[m]
                    if abs(dx) >= 0.5 + thx[m] + min_gap:
                        continue
                    dy = py - ty[m]
                    if abs(dy) >= 0.5 + thx[m] + min_gap:
                        continue
                    # Axes: (1,0), (0,1), (tc,ts), (-ts,tc)
                    # Note upright square vs tilted square rad on (1,0) is 0.5 + thx[m] (already checked!)
                    cj, sj = tc[m], ts[m]
                    rad = 0.5 + thx[m] + min_gap
                    p3 = abs(dx * cj + dy * sj)
                    if p3 >= rad:
                        continue
                    p4 = abs(-dx * sj + dy * cj)
                    if p4 >= rad:
                        continue
                    ok = False
                    break
                if ok:
                    cx_list[cnt] = px
                    cy_list[cnt] = py
                    dist_list[cnt] = d
                    cnt += 1

    # Greedily keep upright squares in increasing corner-distance order (no overlap among uprights)
    kept_x = np.empty(cnt, dtype=np.float64)
    kept_y = np.empty(cnt, dtype=np.float64)
    n_kept = 0
    for idx in range(cnt):
        px = cx_list[idx]
        py = cy_list[idx]
        ok = True
        for q in range(n_kept):
            if abs(px - kept_x[q]) < 1.0 + min_gap and abs(py - kept_y[q]) < 1.0 + min_gap:
                ok = False
                break
        if ok:
            kept_x[n_kept] = px
            kept_y[n_kept] = py
            n_kept += 1

    return n_kept, kept_x[:n_kept], kept_y[:n_kept]


def build_rect_block(m1: int, m2: int, th: float, cx: float, cy: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    c, s = math.cos(th), math.sin(th)
    xs = np.empty(m1 * m2, dtype=np.float64)
    ys = np.empty(m1 * m2, dtype=np.float64)
    ths = np.full(m1 * m2, th, dtype=np.float64)
    idx = 0
    for i in range(m1):
        ui = i - 0.5 * (m1 - 1)
        for j in range(m2):
            vj = j - 0.5 * (m2 - 1)
            xs[idx] = cx + ui * c - vj * s
            ys[idx] = cy + ui * s + vj * c
            idx += 1
    return xs, ys, ths


def scan_single_block_worker(args: tuple) -> dict:
    n_target, L_probe, m1_list, thetas, shifts = args
    best_cap = 0
    best_cfg = None
    for m1 in m1_list:
        for m2 in range(max(1, m1 - 3), m1 + 1):
            for th in thetas:
                # Quick bounding check: rotated m1 x m2 fits inside [0, L_probe]^2
                c, s = abs(math.cos(th)), abs(math.sin(th))
                w_box = m1 * c + m2 * s
                h_box = m1 * s + m2 * c
                if w_box > L_probe + 0.02 or h_box > L_probe + 0.02:
                    continue
                for sx, sy in shifts:
                    cx = 0.5 * L_probe + sx
                    cy = 0.5 * L_probe + sy
                    tx, ty, tth = build_rect_block(m1, m2, th, cx, cy)
                    hx = 0.5 * (c + s)
                    if tx.min() - hx < -0.01 or tx.max() + hx > L_probe + 0.01:
                        continue
                    if ty.min() - hx < -0.01 or ty.max() + hx > L_probe + 0.01:
                        continue
                    n_up, ux, uy = count_upright_fill_around_tilted(L_probe, tx, ty, tth, min_gap=-5e-4)
                    cap = len(tx) + n_up
                    if cap > best_cap:
                        best_cap = cap
                        best_cfg = {
                            "type": "single_block",
                            "m1": m1,
                            "m2": m2,
                            "th_deg": round(math.degrees(th), 4),
                            "shift": (sx, sy),
                            "n_tilt": len(tx),
                            "n_up": int(n_up),
                            "cap": int(cap),
                        }
    return {"best_cap": best_cap, "best_cfg": best_cfg}


def scan_two_blocks_worker(args: tuple) -> dict:
    n_target, L_probe, m_list, thetas, d_offsets = args
    best_cap = 0
    best_cfg = None
    for m in m_list:
        for th in thetas:
            c, s = abs(math.cos(th)), abs(math.sin(th))
            span = m * (c + s)
            for dx in d_offsets:
                for dy in d_offsets:
                    cx1, cy1 = 0.5 * L_probe - dx, 0.5 * L_probe + dy
                    cx2, cy2 = 0.5 * L_probe + dx, 0.5 * L_probe - dy
                    if min(cx1, cy1, cx2, cy2) - 0.5 * span < -0.01:
                        continue
                    if max(cx1, cy1, cx2, cy2) + 0.5 * span > L_probe + 0.01:
                        continue
                    x1, y1, t1 = build_rect_block(m, m, th, cx1, cy1)
                    x2, y2, t2 = build_rect_block(m, m, math.pi / 2 - th, cx2, cy2)
                    # Check two blocks don't overlap each other
                    tx = np.concatenate([x1, x2])
                    ty = np.concatenate([y1, y2])
                    tth = np.concatenate([t1, t2])
                    pen, _ = u.sat_max_overlap_and_bbox(tx, ty, tth)
                    if pen > 5e-4:
                        continue
                    n_up, _, _ = count_upright_fill_around_tilted(L_probe, tx, ty, tth, min_gap=-5e-4)
                    cap = len(tx) + n_up
                    if cap > best_cap:
                        best_cap = cap
                        best_cfg = {
                            "type": "two_blocks",
                            "m": m,
                            "th_deg": round(math.degrees(th), 4),
                            "d_offset": (dx, dy),
                            "n_tilt": len(tx),
                            "n_up": int(n_up),
                            "cap": int(cap),
                        }
    return {"best_cap": best_cap, "best_cfg": best_cfg}


def run_fd_check() -> float:
    rng = np.random.default_rng(7)
    n = 12
    L = 3.8
    s0 = np.concatenate([rng.uniform(0.3, L - 0.3, n), rng.uniform(0.3, L - 0.3, n), rng.uniform(0.05, 0.75, n)])
    e0, g0 = u.sat_penalty_grad(s0, L, 1.0, 1.0, side=0.95)
    eps = 1e-7
    max_rel = 0.0
    for idx in range(len(s0)):
        sp = s0.copy(); sp[idx] += eps
        sm = s0.copy(); sm[idx] -= eps
        ep, _ = u.sat_penalty_grad(sp, L, 1.0, 1.0, side=0.95)
        em, _ = u.sat_penalty_grad(sm, L, 1.0, 1.0, side=0.95)
        fd = (ep - em) / (2 * eps)
        denom = max(1.0, abs(fd), abs(g0[idx]))
        rel = abs(fd - g0[idx]) / denom
        if rel > max_rel:
            max_rel = rel
    return max_rel


def main():
    fd_err = run_fd_check()
    print(f"SAT subgradient FD check (side=0.95): max_rel_err={fd_err:.3e}", flush=True)
    assert fd_err < 1e-6, f"FD error too high: {fd_err}"

    # Calibration check on n=626 at L=25.520815 (expected cap=626 via 17x17 @ 45deg)
    tx, ty, tth = build_rect_block(17, 17, math.pi / 4, 25.52081528 / 2, 25.52081528 / 2)
    n_up_626, _, _ = count_upright_fill_around_tilted(25.52081528, tx, ty, tth, min_gap=-1e-4)
    print(f"Calibration n=626 (17x17 @ 45deg, L=25.52081528): n_tilt=289, n_up={n_up_626}, total={289+n_up_626}", flush=True)

    # Also test shifted 17x17 by (+0.5, 0) or (0, 0.5) on n=626
    for sx, sy in [(0.0, 0.0), (0.5, 0.0), (0.25, 0.25), (0.5, 0.5)]:
        tx, ty, tth = build_rect_block(17, 17, math.pi / 4, 25.52081528 / 2 + sx, 25.52081528 / 2 + sy)
        nu, _, _ = count_upright_fill_around_tilted(25.52081528, tx, ty, tth, min_gap=-1e-4)
        print(f"  shift ({sx},{sy}): total={289+nu}", flush=True)

    targets = [401, 442, 530, 626]
    info = u.load_incumbent_and_frontier(targets)
    thetas = [math.radians(d) for d in np.linspace(5.0, 45.0, 81)]
    # Include exact Pythagorean & special angles
    thetas += [math.atan2(3, 4), math.atan2(5, 12), math.atan2(8, 15), math.atan2(7, 24), math.atan2(20, 21), math.atan2(1, 2)]
    shifts = [(0.0, 0.0), (0.25, 0.0), (0.5, 0.0), (0.25, 0.25), (0.5, 0.25), (0.5, 0.5)]

    results = []
    for n in targets:
        k = math.isqrt(n - 1)
        L_ours = info[n]["L_ours"]
        L_front = info[n]["L_front"]
        t0 = time.time()
        L_probe = L_ours - 0.002
        m_list = list(range(2, k + 1))
        chunks = [m_list[i::7] for i in range(7) if m_list[i::7]]
        with ProcessPoolExecutor(max_workers=7) as ex:
            res_single = list(ex.map(scan_single_block_worker, [(n, L_probe, c, thetas, shifts) for c in chunks]))

        best_s = max(res_single, key=lambda r: r["best_cap"])

        # Two-block scan (e.g. two 5x5 or up to k//2)
        m2_list = list(range(2, k // 2 + 3))
        d_offsets = list(np.linspace(2.0, k * 0.4, 9))
        chunks2 = [m2_list[i::7] for i in range(7) if m2_list[i::7]]
        with ProcessPoolExecutor(max_workers=7) as ex:
            res_two = list(ex.map(scan_two_blocks_worker, [(n, L_probe, c, thetas[::2], d_offsets) for c in chunks2]))
        best_t = max(res_two, key=lambda r: r["best_cap"])

        dt = time.time() - t0
        rec = {
            "n": n,
            "L_ours": L_ours,
            "L_front": L_front,
            "L_probe": L_probe,
            "best_single": best_s["best_cfg"],
            "best_two": best_t["best_cfg"],
            "seconds": round(dt, 2),
        }
        results.append(rec)
        print(f"n={n} (L_probe={L_probe:.6f}): single={best_s['best_cfg']} | two={best_t['best_cfg']} ({dt:.1f}s)", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "idea1_tilted_core.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
