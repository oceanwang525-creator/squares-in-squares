"""Idea 3 & 5: Embedded tapered sheared pockets (n=908 style), sub-rectangle reduction check, and 4-edge pinched frame."""
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
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


# ==============================================================================
# Part 1: Sub-rectangle reduction check: can (a + f) x (h + f) pack a*h + 1?
# ==============================================================================
def test_subrect_worker(args: tuple) -> dict:
    a, h, f, seeds = args
    W = a + f
    H = h + f
    n_target = a * h + 1
    best_pen = 1e9
    best_lp = None

    for s_idx in range(seeds):
        rng = np.random.default_rng(1000 * a + 37 * h + s_idx)
        mode = s_idx % 4
        xs = np.empty(n_target, dtype=np.float64)
        ys = np.empty(n_target, dtype=np.float64)
        ths = np.zeros(n_target, dtype=np.float64)

        if mode == 0:
            # Uniform tilted grid stretched slightly + 1 extra
            idx = 0
            for i in range(a):
                for j in range(h):
                    xs[idx] = (i + 0.5) * (W / a)
                    ys[idx] = (j + 0.5) * (H / h)
                    ths[idx] = rng.uniform(-0.6, 0.6)
                    idx += 1
            xs[-1] = rng.uniform(0.6, W - 0.6)
            ys[-1] = rng.uniform(0.6, H - 0.6)
            ths[-1] = rng.uniform(0.1, 0.7)
        elif mode == 1:
            # Staggered / herringbone columns along H
            th0 = math.radians(rng.uniform(10.0, 42.0))
            cols = a
            rows = (n_target + cols - 1) // cols
            idx = 0
            for i in range(cols):
                for j in range(rows):
                    if idx >= n_target:
                        break
                    xs[idx] = 0.55 + i * max(0.5, (W - 1.1) / max(1, cols - 1))
                    ys[idx] = 0.55 + j * max(0.5, (H - 1.1) / max(1, rows - 1))
                    ths[idx] = th0 if (i % 2 == 0) else -th0
                    idx += 1
        elif mode == 2:
            # Upright inner core + tilted corner/edge strip
            for idx in range(a * h):
                i, j = divmod(idx, h)
                xs[idx] = 0.5 + i * (W - 1.0) / max(1, a - 1) if a > 1 else 0.5 * W
                ys[idx] = 0.5 + j * (H - 1.0) / max(1, h - 1) if h > 1 else 0.5 * H
                if i < 2:
                    ths[idx] = math.radians(rng.uniform(18.0, 38.0))
            xs[-1] = rng.uniform(0.6, min(2.5, W - 0.6))
            ys[-1] = rng.uniform(0.6, H - 0.6)
            ths[-1] = math.radians(rng.uniform(18.0, 38.0))
        else:
            # Random oriented in W x H
            xs[:] = rng.uniform(0.6, W - 0.6, size=n_target)
            ys[:] = rng.uniform(0.6, H - 0.6, size=n_target)
            ths[:] = rng.uniform(0.0, 0.78, size=n_target)

        # Normalize square box vs rectangular box: embed in H x H by blocking x > W with wall penalty?
        # Simpler: scale coordinates? NO, squares aren't isotropic! Put W-wall check directly via dummy tall wall?
        # Even simpler: relax inside [0, H]^2 ONLY if W==H, else custom rect L-BFGS!
        from scipy.optimize import minimize

        def rect_obj(state: np.ndarray, cur_side: float) -> tuple[float, np.ndarray]:
            e, g = u.sat_penalty_grad(state, H, 1.0, 1.0, side=cur_side)
            # Add extra right wall at x = W - hx * cur_side
            nx = state[:n_target]
            nt = state[2 * n_target :]
            c = np.cos(nt)
            s = np.sin(nt)
            ac = np.abs(c)
            as_ = np.abs(s)
            hx = 0.5 * cur_side * (ac + as_)
            ov = (nx + hx) - W
            mask = ov > 0.0
            if np.any(mask):
                e += float(np.sum(ov[mask] ** 2))
                g[:n_target][mask] += 2.0 * ov[mask]
                sg_c = np.where(c >= 0.0, 1.0, -1.0)
                sg_s = np.where(s >= 0.0, 1.0, -1.0)
                dhx = 0.5 * cur_side * (-sg_c * s + sg_s * c)
                g[2 * n_target :][mask] += 2.0 * ov[mask] * dhx[mask]
            return e, g

        state = np.concatenate([xs, ys, ths])
        for side in (0.88, 0.94, 0.98, 1.0):
            res = minimize(lambda st, sd=side: rect_obj(st, sd), state, method="L-BFGS-B", jac=True, options={"maxiter": 140, "ftol": 1e-15, "gtol": 1e-12})
            state = res.x
        fx, fy, fth = state[:n_target], state[n_target : 2 * n_target], state[2 * n_target :]
        pen, _ = u.sat_max_overlap_and_bbox(fx, fy, fth, 1.0)
        hx = 0.5 * (np.abs(np.cos(fth)) + np.abs(np.sin(fth)))
        wpen = max(
            0.0,
            float((hx - fx).max()),
            float((fx + hx - W).max()),
            float((hx - fy).max()),
            float((fy + hx - H).max()),
        )
        tot_pen = max(pen, wpen)
        if tot_pen < best_pen:
            best_pen = tot_pen
        if tot_pen < 1e-6:
            best_lp = (W, H)
            break

    return {"a": a, "h": h, "n_sub": n_target, "W": round(W, 5), "H": round(H, 5), "best_pen": float(best_pen), "feasible": best_lp is not None}


# ==============================================================================
# Part 2: Embedded Tapered Sheared Pockets (n=908 style) in 4-corner L x L box
# ==============================================================================
@nb.njit(cache=True)
def fill_sheared_pocket_cols(
    L: float,
    ux: np.ndarray,
    uy: np.ndarray,
    uc: np.ndarray,
    us: np.ndarray,
    uhx: np.ndarray,
    th: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> tuple[int, np.ndarray, np.ndarray]:
    """Pack vertically-sheared columns (dy = sec th, col shift dx ~ cos th) inside a cavity clearing fixed squares."""
    c = math.cos(th)
    s = math.sin(th)
    hx = 0.5 * (abs(c) + abs(s))
    dy_step = 1.0 / c + 1e-4
    dx_step = c + 1e-4

    px = np.empty(400, dtype=np.float64)
    py = np.empty(400, dtype=np.float64)
    pc = np.full(400, c, dtype=np.float64)
    ps = np.full(400, s, dtype=np.float64)
    phx = np.full(400, hx, dtype=np.float64)
    n_tilt = 0

    x = max(hx, x_min)
    while x <= min(L - hx, x_max) and n_tilt < 395:
        best_y0 = -1.0
        best_cnt = 0
        y0_off = 0.0
        while y0_off < dy_step:
            cnt = 0
            y = max(hx, y_min) + y0_off
            while y <= min(L - hx, y_max):
                if u.sat_pair_clear_against_group(x, y, c, s, hx, ux, uy, uc, us, uhx):
                    if u.sat_pair_clear_against_group(x, y, c, s, hx, px[:n_tilt], py[:n_tilt], pc[:n_tilt], ps[:n_tilt], phx[:n_tilt]):
                        cnt += 1
                y += dy_step
            if cnt > best_cnt:
                best_cnt = cnt
                best_y0 = y0_off
            y0_off += 0.06
        if best_cnt > 0:
            y = max(hx, y_min) + best_y0
            while y <= min(L - hx, y_max) and n_tilt < 400:
                if u.sat_pair_clear_against_group(x, y, c, s, hx, ux, uy, uc, us, uhx):
                    if u.sat_pair_clear_against_group(x, y, c, s, hx, px[:n_tilt], py[:n_tilt], pc[:n_tilt], ps[:n_tilt], phx[:n_tilt]):
                        px[n_tilt] = x
                        py[n_tilt] = y
                        n_tilt += 1
                y += dy_step
            x += dx_step
        else:
            x += 0.16
    return n_tilt, px[:n_tilt], py[:n_tilt]


def build_quadrant_split_uprights(L: float, sw_w: int, sw_h: int, ne_w: int, ne_h: int, nw_cut: int, se_cut: int) -> tuple[np.ndarray, np.ndarray]:
    """Place SW=(0,0)-aligned and NE=(L,L)-aligned upright blocks + NW=(0,L) and SE=(L,0) corner fills."""
    pts = []
    # SW rectangle [0, sw_w] x [0, sw_h]
    for i in range(sw_w):
        for j in range(sw_h):
            pts.append((0.5 + i, 0.5 + j))
    # NE rectangle [L-ne_w, L] x [L-ne_h, L]
    for i in range(ne_w):
        for j in range(ne_h):
            pts.append((L - 0.5 - i, L - 0.5 - j))
    # NW corner staircase of leg nw_cut aligned to (0, L), clearing SW & NE
    for i in range(nw_cut):
        for j in range(nw_cut - i):
            cx = 0.5 + i
            cy = L - 0.5 - j
            if cx + 0.5 > L - ne_w and cy - 0.5 > L - ne_h:
                continue
            if cx - 0.5 < sw_w and cy - 0.5 < sw_h:
                continue
            pts.append((cx, cy))
    # SE corner staircase of leg se_cut aligned to (L, 0)
    for i in range(se_cut):
        for j in range(se_cut - i):
            cx = L - 0.5 - i
            cy = 0.5 + j
            if cx - 0.5 < sw_w and cy + 0.5 < sw_h:
                continue
            if cx + 0.5 > L - ne_w and cy + 0.5 > L - ne_h:
                continue
            pts.append((cx, cy))
    arr = np.array(pts, dtype=np.float64)
    return arr[:, 0], arr[:, 1]


def scan_embedded_pockets_worker(args: tuple) -> dict:
    n_target, L_probe, sw_vals, thetas = args
    k = math.isqrt(n_target - 1)
    best_cap = 0
    best_cfg = None
    best_coords = None

    for sw_w in sw_vals:
        for sw_h in range(max(4, sw_w - 2), min(k - 3, sw_w + 3)):
            ne_w = k - sw_w
            ne_h = k - sw_h
            for nw_cut in range(0, min(sw_w, ne_h) + 1, 2):
                for se_cut in (nw_cut,):
                    ux, uy = build_quadrant_split_uprights(L_probe, sw_w, sw_h, ne_w, ne_h, nw_cut, se_cut)
                    uc = np.ones(len(ux), dtype=np.float64)
                    us = np.zeros(len(ux), dtype=np.float64)
                    uhx = np.full(len(ux), 0.5, dtype=np.float64)
                    for th in thetas:
                        c = math.cos(th)
                        s = math.sin(th)
                        hx = 0.5 * (abs(c) + abs(s))
                        nt1, p1x, p1y = fill_sheared_pocket_cols(L_probe, ux, uy, uc, us, uhx, th, 0.0, L_probe - ne_w, float(sw_h), L_probe)
                        ux2 = np.concatenate([ux, p1x])
                        uy2 = np.concatenate([uy, p1y])
                        uc2 = np.concatenate([uc, np.full(nt1, c)])
                        us2 = np.concatenate([us, np.full(nt1, s)])
                        uhx2 = np.concatenate([uhx, np.full(nt1, hx)])
                        nt2, p2x, p2y = fill_sheared_pocket_cols(L_probe, ux2, uy2, uc2, us2, uhx2, th, float(sw_w), L_probe, 0.0, L_probe - ne_h)
                        cap = len(ux) + nt1 + nt2
                        if cap > best_cap:
                            best_cap = cap
                            best_cfg = {
                                "sw_w": sw_w,
                                "sw_h": sw_h,
                                "ne_w": ne_w,
                                "ne_h": ne_h,
                                "nw_cut": nw_cut,
                                "th_deg": round(math.degrees(th), 2),
                                "n_up": len(ux),
                                "nt1": int(nt1),
                                "nt2": int(nt2),
                                "cap": int(cap),
                            }
                            best_coords = (ux, uy, p1x, p1y, p2x, p2y, th)
    return {"best_cap": best_cap, "best_cfg": best_cfg, "coords": best_coords}


# ==============================================================================
# Part 3: 4-Edge Asymmetric Pinched Frame (f_N + f_S = 1 + f, tapered ends)
# ==============================================================================
def test_pinched_frame(n_target: int, L_probe: float) -> dict:
    """Place centered (k - w1 - w2) upright core and test tapered sheared strips on 4 edges."""
    k = math.isqrt(n_target - 1)
    best_cap = 0
    best_cfg = None
    # Suppose core is c x c upright centered in [0, L]^2, leaving margin M = (L - c)/2 = m_edge + (1 + f)/2?
    # Better: asymmetric split! Left/Right upright slabs, Top/Bottom tilted pinched pockets of width w + 0.5 + f/2!
    for core_w in range(max(6, k - 10), k - 1, 2):
        # core_w even/odd matching k parity so slack is w_tot + f:
        rem = k - core_w  # even
        w = rem // 2
        # If total width across left+right pockets is rem + f = 2*w + f,
        # splitting unevenly or pinching along middle gives w - 0.5 vs w + 0.5 + f: no, wait:
        # Take core shifted to left/bottom or centered at L/2:
        cx0 = 0.5 * (L_probe - core_w)
        pts = [(cx0 + 0.5 + i, cx0 + 0.5 + j) for i in range(core_w) for j in range(core_w)]
        ux = np.array([p[0] for p in pts], dtype=np.float64)
        uy = np.array([p[1] for p in pts], dtype=np.float64)
        uc = np.ones(len(ux), dtype=np.float64)
        us = np.zeros(len(ux), dtype=np.float64)
        uhx = np.full(len(ux), 0.5, dtype=np.float64)
        for th_deg in (15.0, 20.0, 24.5, 26.565, 30.0, 33.69, 36.87):
            th = math.radians(th_deg)
            nt_l, px_l, py_l = fill_sheared_pocket_cols(L_probe, ux, uy, uc, us, uhx, th, 0.0, cx0 + 0.1, 0.0, L_probe)
            nt_r, px_r, py_r = fill_sheared_pocket_cols(L_probe, ux, uy, uc, us, uhx, th, L_probe - cx0 - 0.1, L_probe, 0.0, L_probe)
            cap = len(ux) + nt_l + nt_r
            # Roughly check if left+right even beats upright 2 * w * core_w:
            if cap > best_cap:
                best_cap = cap
                best_cfg = {"core_w": core_w, "w": w, "th_deg": th_deg, "n_up": len(ux), "nt_l": int(nt_l), "nt_r": int(nt_r), "cap_lr": int(cap)}
    return best_cfg or {}


def main():
    t0 = time.time()
    print("=== Part 1: Sub-rectangle reduction check (a + f) x (h + f) >= a*h + 1 ===", flush=True)
    f_probe = 0.524  # below 20.526135's 0.526135!
    sub_jobs = []
    for a in range(2, 13):
        for h in (a, min(20, a + 4), 20):
            if h < a:
                continue
            if a * h + 1 <= 150:
                sub_jobs.append((a, h, f_probe, 16))
    with ProcessPoolExecutor(max_workers=7) as ex:
        sub_res = list(ex.map(test_subrect_worker, sub_jobs))
    sub_res.sort(key=lambda r: r["best_pen"])
    for r in sub_res[:8]:
        print(" ", r, flush=True)

    print("\n=== Part 2: Embedded 2-Pocket Sheared Cavities (n=401, 442, 530) ===", flush=True)
    targets = [401, 442, 530]
    info = u.load_incumbent_and_frontier(targets)
    thetas = [math.radians(d) for d in np.linspace(12.0, 44.0, 33)]
    pocket_res = []
    for n in targets:
        k = math.isqrt(n - 1)
        L_ours = info[n]["L_ours"]
        L_front = info[n]["L_front"]
        L_probe = L_ours - 0.002
        sw_vals = list(range(3, k - 2))
        chunks = [sw_vals[i::7] for i in range(7) if sw_vals[i::7]]
        with ProcessPoolExecutor(max_workers=7) as ex:
            res = list(ex.map(scan_embedded_pockets_worker, [(n, L_probe, c, thetas) for c in chunks]))
        best = max(res, key=lambda r: r["best_cap"])
        print(f"n={n} (L_probe={L_probe:.6f}): cap={best['best_cap']} (def={n - best['best_cap']}), cfg={best['best_cfg']}", flush=True)
        # Also test pinched frame quick check
        pf = test_pinched_frame(n, L_probe)
        print(f"  pinched_frame L+R sample: {pf}", flush=True)
        pocket_res.append({"n": n, "L_ours": L_ours, "L_front": L_front, "L_probe": L_probe, "best_cfg": best["best_cfg"], "deficit": n - best["best_cap"], "pinched": pf})

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "idea3_tapered_pockets.json").write_text(json.dumps({"subrect": sub_res, "pockets": pocket_res}, indent=2))
    print(f"\nDone in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
