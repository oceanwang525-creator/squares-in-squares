"""POC 1: Zero-Sawtooth Sheared-Row L-Pockets with Mitered Corner & Asymmetric Widths/Angles.

Falsification test for Ideas 1, 5, and 11 on:
  - Calibration: n=598 (K=24)
  - D=1 roots: n=401 (K=20), n=530 (K=23), n=842 (K=29)
  - High-gap targets: n=634 (K=25), n=908 (K=30), n=911 (K=30), n=971 (K=31)
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

import geom_utils as gu

OUT_DIR = Path(__file__).resolve().parents[1] / "docs"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def solve_shear_angle(w: int, W: float) -> float | None:
    """Solve w * cos(th) + sin(th) = W for th in [th_peak, pi/4]."""
    # Peak of w*cos(t) + sin(t) is at tan(t) = 1/w, value sqrt(w^2 + 1)
    # At t = pi/4, value is (w + 1) / sqrt(2)
    t_peak = math.atan2(1.0, float(w))
    v_peak = math.sqrt(w * w + 1.0)
    v_45 = (w + 1.0) / math.sqrt(2.0)
    if W > v_peak or W < v_45:
        return None
    if abs(W - v_peak) < 1e-12:
        return t_peak
    if abs(W - v_45) < 1e-12:
        return math.pi / 4.0
    return brentq(lambda t: w * math.cos(t) + math.sin(t) - W, t_peak, math.pi / 4.0, xtol=1e-12)


def make_west_pocket(W: float, y_min: float, y_max: float, w: int, th: float, sign: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate sheared rows in [0, W] x [y_min, y_max] with row angle sign*th."""
    c, s = math.cos(th), math.sin(th)
    hx = 0.5 * (c + s)
    dy_row = 1.0 / c
    # Extra margin in r to cover slanted ends
    r_min = -int(math.ceil(w * s / dy_row)) - 2
    r_max = int(math.ceil((y_max - y_min + w * s) / dy_row)) + 2
    xs, ys, ths = [], [], []
    # Distribute leftover horizontal slack evenly
    row_span_x = w * c + s
    x_pad = 0.5 * max(0.0, W - row_span_x)
    for r in range(r_min, r_max + 1):
        y0 = y_min + hx + r * dy_row
        for k in range(w):
            xk = x_pad + hx + k * c
            yk = y0 + sign * k * s
            if xk - hx >= -1e-9 and xk + hx <= W + 1e-9 and yk - hx >= y_min - 1e-9 and yk + hx <= y_max + 1e-9:
                xs.append(xk)
                ys.append(yk)
                ths.append(sign * th)
    return np.array(xs), np.array(ys), np.array(ths)


def make_south_pocket(H: float, x_min: float, x_max: float, w: int, th: float, sign: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transpose of west pocket into [x_min, x_max] x [0, H]."""
    px, py, pth = make_west_pocket(H, x_min, x_max, w, th, sign)
    return py, px, -pth


def greedy_fill_voids(
    x: np.ndarray,
    y: np.ndarray,
    th: np.ndarray,
    L: float,
    candidate_angles: list[float],
    grid_step: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Try adding non-overlapping squares in remaining empty space on a coarse grid."""
    pts_x = list(x)
    pts_y = list(y)
    pts_th = list(th)
    arr_x = np.array(pts_x)
    arr_y = np.array(pts_y)
    arr_c = np.cos(pts_th)
    arr_s = np.sin(pts_th)

    # Scan candidate centers where distance to nearest square center >= 0.95
    xs_grid = np.arange(0.5, L - 0.49, grid_step)
    ys_grid = np.arange(0.5, L - 0.49, grid_step)
    for gx in xs_grid:
        for gy in ys_grid:
            if len(arr_x) > 0:
                d2 = (arr_x - gx) ** 2 + (arr_y - gy) ** 2
                if d2.min() < 0.90:
                    continue
            for ang in candidate_angles:
                ca, sa = math.cos(ang), math.sin(ang)
                hxa = 0.5 * (abs(ca) + abs(sa))
                if gx - hxa < 1e-7 or gx + hxa > L - 1e-7 or gy - hxa < 1e-7 or gy + hxa > L - 1e-7:
                    continue
                # Check SAT against nearby existing squares
                ok = True
                if len(arr_x) > 0:
                    near = np.where((np.abs(arr_x - gx) < 1.55) & (np.abs(arr_y - gy) < 1.55))[0]
                    for j in near:
                        dx = gx - arr_x[j]
                        dy = gy - arr_y[j]
                        cj, sj = arr_c[j], arr_s[j]
                        rad = 0.5 + 0.5 * (abs(ca * cj + sa * sj) + abs(ca * sj - sa * cj))
                        p1 = abs(dx * ca + dy * sa)
                        p2 = abs(-dx * sa + dy * ca)
                        p3 = abs(dx * cj + dy * sj)
                        p4 = abs(-dx * sj + dy * cj)
                        if max(p1, p2, p3, p4) < rad + 1e-6:
                            ok = False
                            break
                if ok:
                    pts_x.append(gx)
                    pts_y.append(gy)
                    pts_th.append(ang)
                    arr_x = np.append(arr_x, gx)
                    arr_y = np.append(arr_y, gy)
                    arr_c = np.append(arr_c, ca)
                    arr_s = np.append(arr_s, sa)
                    break
    return np.array(pts_x), np.array(pts_y), np.array(pts_th)


def build_l_pocket_candidate(
    L: float,
    Kx: int,
    Ky: int,
    w1: int,
    w2: int,
    sign1: int,
    sign2: int,
    mode: str,
    miter_shift: float = 0.0,
    do_greedy: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Construct NE upright block (Kx x Ky) + West pocket + South pocket + SW corner."""
    W1 = L - Kx
    W2 = L - Ky
    th1 = solve_shear_angle(w1, W1 - 1e-5)
    th2 = solve_shear_angle(w2, W2 - 1e-5)
    if th1 is None or th2 is None:
        return None
    # Reject huge angles (> 40 deg) where sheared density loses to upright
    if th1 > 0.72 or th2 > 0.72:
        return None

    # NE upright block
    ux = np.array([L - Kx + 0.5 + i for i in range(Kx) for _ in range(Ky)], dtype=np.float64)
    uy = np.array([L - Ky + 0.5 + j for _ in range(Kx) for j in range(Ky)], dtype=np.float64)
    uth = np.zeros(len(ux), dtype=np.float64)

    if mode == "sw_upright":
        # SW corner upright block of floor(W1) x floor(W2)
        p1 = int(math.floor(W1))
        p2 = int(math.floor(W2))
        sw_x = np.array([0.5 + i for i in range(p1) for _ in range(p2)], dtype=np.float64)
        sw_y = np.array([0.5 + j for _ in range(p1) for j in range(p2)], dtype=np.float64)
        sw_th = np.zeros(len(sw_x), dtype=np.float64)
        wx, wy, wth = make_west_pocket(W1 - 1e-5, float(p2) + 1e-5, L, w1, th1, sign1)
        sx, sy, sth = make_south_pocket(W2 - 1e-5, float(p1) + 1e-5, L, w2, th2, sign2)
        x = np.concatenate([ux, sw_x, wx, sx])
        y = np.concatenate([uy, sw_y, wy, sy])
        th = np.concatenate([uth, sw_th, wth, sth])
    elif mode == "miter":
        # Extend both pockets into SW corner and clip by miter line y - x*W2/W1 >= miter_shift
        wx, wy, wth = make_west_pocket(W1 - 1e-5, 0.0, L, w1, th1, sign1)
        sx, sy, sth = make_south_pocket(W2 - 1e-5, 0.0, L, w2, th2, sign2)
        # Keep west squares whose bottom/right corner clears miter line with margin
        slope = W2 / max(W1, 1e-9)
        keep_w = (wy - slope * wx) >= (0.72 + miter_shift)
        keep_s = (sy - slope * sx) <= (-0.72 + miter_shift)
        wx, wy, wth = wx[keep_w], wy[keep_w], wth[keep_w]
        sx, sy, sth = sx[keep_s], sy[keep_s], sth[keep_s]
        x = np.concatenate([ux, wx, sx])
        y = np.concatenate([uy, wy, sy])
        th = np.concatenate([uth, wth, sth])
    else:
        return None

    # Check base overlaps before greedy fill
    pen, _ = gu.sat_max_overlap_and_bbox(x, y, th)
    if pen > 1e-6:
        return None

    # Optionally fill leftover corner/end pockets greedily
    if do_greedy:
        cands = [0.0, sign1 * th1, -sign2 * th2, math.pi / 4.0, -math.pi / 4.0, 0.5 * sign1 * th1]
        x, y, th = greedy_fill_voids(x, y, th, L, cands, grid_step=0.25)
    return x, y, th


def evaluate_target(n_target: int, L_ours: float, L_front: float) -> dict:
    """Sweep (Kx, Ky, w1, w2, signs, modes) at L_probe slightly below L_ours."""
    t0 = time.time()
    K = int(math.floor(L_ours))
    # Probe at L slightly below L_ours to test capacity
    L_probes = [L_ours - 0.002, L_front + 0.002] if L_ours - L_front > 0.005 else [L_ours - 0.0008]

    top_candidates = []  # (base_cap, est, cfg, L_probe, params)
    configs_tried = 0

    for L_probe in L_probes:
        Kp = int(math.floor(L_probe))
        fp = L_probe - Kp
        p_choices = list(range(4, min(20, Kp - 2)))
        for P1 in p_choices:
            Kx = Kp - P1
            W1 = P1 + fp
            w1_list = [w for w in (P1 + 1, P1 + 2) if solve_shear_angle(w, W1 - 1e-5) is not None]
            if not w1_list:
                continue
            for P2 in p_choices:
                Ky = Kp - P2
                W2 = P2 + fp
                w2_list = [w for w in (P2 + 1, P2 + 2) if solve_shear_angle(w, W2 - 1e-5) is not None]
                if not w2_list:
                    continue
                for w1 in w1_list:
                    for w2 in w2_list:
                        th1 = solve_shear_angle(w1, W1 - 1e-5)
                        th2 = solve_shear_angle(w2, W2 - 1e-5)
                        if th1 is None or th2 is None or th1 > 0.65 or th2 > 0.65:
                            continue
                        est = Kx * Ky + P1 * P2 + (L_probe - P2) * w1 * math.cos(th1) + (L_probe - P1) * w2 * math.cos(th2)
                        if est < n_target - 8:
                            continue
                        for mode in ("sw_upright", "miter"):
                            for s1 in (1, -1):
                                for s2 in (1, -1):
                                    if P1 == P2 and w1 == w2 and s1 == 1 and s2 == -1:
                                        continue
                                    configs_tried += 1
                                    res = build_l_pocket_candidate(
                                        L_probe, Kx, Ky, w1, w2, s1, s2, mode, do_greedy=False
                                    )
                                    if res is None:
                                        continue
                                    x, y, th = res
                                    cap = len(x)
                                    cfg = {
                                        "L_probe": round(L_probe, 6),
                                        "Kx": Kx,
                                        "Ky": Ky,
                                        "P1": P1,
                                        "P2": P2,
                                        "w1": w1,
                                        "w2": w2,
                                        "th1_deg": round(math.degrees(th1), 3),
                                        "th2_deg": round(math.degrees(th2), 3),
                                        "mode": mode,
                                        "signs": (s1, s2),
                                        "base_cap": cap,
                                    }
                                    top_candidates.append((cap, est, cfg, L_probe, (Kx, Ky, w1, w2, s1, s2, mode)))

    top_candidates.sort(key=lambda t: (t[0], t[1]), reverse=True)
    best_cap = 0
    best_cfg = None
    best_state = None
    for _, _, cfg, L_probe, params in top_candidates[:12]:
        Kx, Ky, w1, w2, s1, s2, mode = params
        res = build_l_pocket_candidate(L_probe, Kx, Ky, w1, w2, s1, s2, mode, do_greedy=True)
        if res is None:
            continue
        x, y, th = res
        cap = len(x)
        if cap > best_cap:
            best_cap = cap
            best_cfg = dict(cfg, cap=cap)
            best_state = (x, y, th, L_probe)

    certified_win = None
    relaxed_L = None
    if best_state is not None:
        bx, by, bth, bL = best_state
        if len(bx) >= n_target:
            x_t, y_t, th_t = bx[:n_target], by[:n_target], bth[:n_target]
            rx, ry, rth, rpen = gu.relax_lbfgs(x_t, y_t, th_t, bL, maxiter=250)
            if rpen < 1e-8:
                lx, ly, lL, lok = gu.fixed_angle_lp_tighten(rx, ry, rth, rounds=4)
                relaxed_L = lL
                if lok and lL < L_ours - 1e-9:
                    cert = gu.verify_official(n_target, lx, ly, rth)
                    if cert["ok"] and cert["L_float"] < L_ours:
                        certified_win = cert

    elapsed = time.time() - t0
    return {
        "n": n_target,
        "L_ours": L_ours,
        "L_front": L_front,
        "configs_tried": configs_tried,
        "best_cap": best_cap,
        "deficit": n_target - best_cap,
        "best_cfg": best_cfg,
        "relaxed_L": relaxed_L,
        "certified_win": {k: v for k, v in certified_win.items() if k != "rows"} if certified_win else None,
        "seconds": round(elapsed, 2),
    }


def main():
    import pandas as pd

    inc = gu.load_incumbent_Ls()
    gap_df = pd.read_csv(gu.REPO_ROOT / "poc" / "frontier_nov12" / "docs" / "per_n_gap.csv").set_index("n")

    targets = [598, 401, 530, 842, 634, 908, 911, 971]
    results = []
    for n in targets:
        L_ours = inc[n]
        L_front = float(gap_df.loc[n, "L_frontier"])
        print(f"Running POC 1 on n={n} (ours={L_ours:.6f}, front={L_front:.6f})...", flush=True)
        res = evaluate_target(n, L_ours, L_front)
        results.append(res)
        print(
            f"  -> n={n}: best_cap={res['best_cap']} (deficit={res['deficit']}), "
            f"configs={res['configs_tried']}, relaxed_L={res['relaxed_L']}, "
            f"win={res['certified_win'] is not None}, time={res['seconds']}s",
            flush=True,
        )
        if res["best_cfg"]:
            print(f"     cfg={res['best_cfg']}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "poc1_sheared_pockets.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
