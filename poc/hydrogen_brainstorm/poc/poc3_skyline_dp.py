"""POC 3: Idea 3 — Skyline Beam-Search DP in Fixed-Width Strips W = P + f.

Goal: Test whether non-lattice / alternating-angle / herringbone / end-capped
packing motifs inside narrow fixed-width strips [0, W] x [0, H] can beat BOTH:
  1. Upright columns: rho_upright(W) = floor(W)
  2. Uniform sheared row: rho_shear(W) = max_{w >= P+1} w*cos(theta_w)
     where w*cos(theta) + sin(theta) = W  (when sin(theta) < f, beats upright).

Uses an exact closed-form SAT vertical drop solver (Numba-accelerated) to place
squares at the lowest feasible y without overlapping any existing square, runs
beam search maximizing packed count per height (and continuous angle+position
compaction on top sequences), and checks the exact asymptotic slope rho(W) and
boundary offset c_0 = N - rho*H.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
import sys

import numpy as np
from numba import njit
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import geom_utils as gu

OUT_DIR = Path(__file__).resolve().parent.parent / "docs"


@njit(cache=True)
def sat_y_interval_overlap(dx: float, th1: float, th2: float) -> tuple[bool, float, float]:
    """Return (possible_in_x, dy_low, dy_high) for Minkowski sum S(th2) + (-S(th1)) at dx.

    Square 1 centered at (x1, y1, th1) overlaps square 2 at (x2, y2, th2)
    with dx = x1 - x2 iff possible_in_x and dy_low < (y1 - y2) < dy_high.
    """
    c1, s1 = math.cos(th1), math.sin(th1)
    c2, s2 = math.cos(th2), math.sin(th2)
    axes_x = (c1, -s1, c2, -s2)
    axes_y = (s1, c1, s2, c2)

    low = -1e9
    high = 1e9
    for k in range(4):
        ux = axes_x[k]
        uy = axes_y[k]
        r1 = 0.5 * (abs(ux * c1 + uy * s1) + abs(-ux * s1 + uy * c1))
        r2 = 0.5 * (abs(ux * c2 + uy * s2) + abs(-ux * s2 + uy * c2))
        R = r1 + r2
        if abs(uy) < 1e-12:
            if abs(dx * ux) >= R - 1e-12:
                return False, 0.0, 0.0
        else:
            a = (-R - dx * ux) / uy
            b = (R - dx * ux) / uy
            if a > b:
                a, b = b, a
            if a > low:
                low = a
            if b < high:
                high = b
            if low >= high - 1e-12:
                return False, 0.0, 0.0
    return True, low, high


@njit(cache=True)
def lowest_feasible_y(
    x_new: float,
    th_new: float,
    xs: np.ndarray,
    ys: np.ndarray,
    ths: np.ndarray,
    count: int,
    clearance: float = 1e-7,
) -> float:
    """Exact minimum y >= half_extent(th_new) with no overlap against xs[:count]."""
    hx = 0.5 * (abs(math.cos(th_new)) + abs(math.sin(th_new)))
    y = hx + clearance

    # Gather nearby intervals along y
    # Iterate pushes upward until no interval contains y
    for _ in range(count + 2):
        bumped = False
        next_y = y
        for j in range(count):
            dx = x_new - xs[j]
            if abs(dx) >= 1.414214:
                continue
            ok, dlo, dhi = sat_y_interval_overlap(dx, th_new, ths[j])
            if not ok:
                continue
            lo = ys[j] + dlo - clearance
            hi = ys[j] + dhi + clearance
            if lo < y < hi:
                if hi > next_y:
                    next_y = hi
                bumped = True
        if not bumped:
            return y
        y = next_y
    return y


@njit(cache=True)
def check_overlap_pair_count(xs: np.ndarray, ys: np.ndarray, ths: np.ndarray) -> float:
    n = len(xs)
    max_ov = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            if dx * dx + dy * dy >= 2.0001:
                continue
            ok, dlo, dhi = sat_y_interval_overlap(dx, ths[i], ths[j])
            if ok and dlo < dy < dhi:
                ov = min(dy - dlo, dhi - dy)
                if ov > max_ov:
                    max_ov = ov
    return max_ov


def solve_shear_angle(w: int, W: float) -> float | None:
    """Solve w*cos(th) + sin(th) = W on [atan(1/w), pi/4]."""
    t_peak = math.atan2(1.0, float(w))
    v_peak = math.sqrt(w * w + 1.0)
    v_45 = (w + 1.0) / math.sqrt(2.0)
    if W > v_peak + 1e-12 or W < v_45 - 1e-12:
        return None
    if W >= v_peak - 1e-12:
        return t_peak
    if W <= v_45 + 1e-12:
        return math.pi / 4.0
    th = t_peak + math.acos(max(-1.0, min(1.0, W / v_peak)))
    if t_peak - 1e-9 <= th <= math.pi / 4.0 + 1e-9:
        return min(math.pi / 4.0, max(t_peak, th))
    return None


def shear_baseline(W: float) -> tuple[int, float, float]:
    """Return (best_w, best_th_deg, rho_shear) for uniform sheared rows in width W."""
    P = int(math.floor(W))
    best_rho = float(P)
    best_w = P
    best_th = 0.0
    for w in range(P + 1, P + 4):
        th = solve_shear_angle(w, W - 1e-7)
        if th is None:
            continue
        rho = w * math.cos(th)
        if rho > best_rho:
            best_rho = rho
            best_w = w
            best_th = math.degrees(th)
    return best_w, best_th, best_rho


def strip_beam_search(
    W: float,
    target_n: int = 36,
    beam_width: int = 24,
    extra_angles_deg: tuple[float, ...] = (),
) -> dict:
    """Run skyline beam search in [0, W] up to target_n squares."""
    P = int(math.floor(W))
    best_w, best_th_deg, rho_shear = shear_baseline(W)

    # Focused angle palette: upright, shear optimal +/- offsets, herringbone, 45
    ang_set = {0.0, 45.0, -45.0}
    if best_th_deg > 0.1:
        for d in (-3.0, 0.0, 3.0):
            a = best_th_deg + d
            if 0.0 < a < 45.0:
                ang_set.add(round(a, 3))
                ang_set.add(round(-a, 3))
    for a in (10.0, 15.0, 20.0, 25.0, 30.0, 35.0, *extra_angles_deg):
        if -45.0 < a < 45.0:
            ang_set.add(a)
            ang_set.add(-a)
    angles = np.array([math.radians(a) for a in sorted(ang_set)], dtype=np.float64)
    hx_arr = 0.5 * (np.abs(np.cos(angles)) + np.abs(np.sin(angles)))

    # Candidate x grid for each angle: walls + dense uniform grid across [hx, W - hx]
    x_cands_by_ang = []
    for k in range(len(angles)):
        hx = hx_arr[k]
        if 2 * hx >= W:
            x_cands_by_ang.append(np.empty(0, dtype=np.float64))
            continue
        # Step ~0.08 gives good contact resolution while staying fast
        nx = max(5, int(round((W - 2 * hx) / 0.085)) + 1)
        xs_grid = np.linspace(hx + 1e-6, W - hx - 1e-6, nx)
        x_cands_by_ang.append(xs_grid)

    # Beam state: (score, xs_list, ys_list, ths_list, cnt, top_y, sum_top_y, sum_abs_th)
    beam = [(0.0, np.zeros(target_n), np.zeros(target_n), np.zeros(target_n), 0, 0.0, 0.0, 0.0)]

    best_heights_by_n = {}
    for step in range(target_n):
        next_states = []
        seen_sigs = set()
        for _, xs_b, ys_b, ths_b, cnt, cur_top, cur_sum_y, cur_sum_th in beam:
            local_cands = []
            for k in range(len(angles)):
                th_k = float(angles[k])
                hx = float(hx_arr[k])
                xc = x_cands_by_ang[k]
                for idx_x in range(len(xc)):
                    x_try = float(xc[idx_x])
                    y_try = lowest_feasible_y(x_try, th_k, xs_b, ys_b, ths_b, cnt, 1e-7)
                    top_new = max(cur_top, y_try + hx)
                    new_sum_y = cur_sum_y + (y_try + hx)
                    mean_top = new_sum_y / (cnt + 1)
                    # Score prioritizes packing low center-of-mass & filling holes while keeping top low
                    sc = mean_top + 0.35 * top_new + 0.001 * abs(x_try - 0.5 * W)
                    local_cands.append((sc, x_try, y_try, th_k, top_new, new_sum_y))
            local_cands.sort(key=lambda t: t[0])

            # Keep top placements across diverse angle buckets so tilted/mixed motifs survive row 1
            kept_per_bucket = {}
            for sc, x_try, y_try, th_k, top_new, new_sum_y in local_cands:
                bkt = 0 if abs(th_k) < 1e-4 else (1 if abs(th_k) < 0.45 else 2)
                if kept_per_bucket.get(bkt, 0) >= 4:
                    continue
                sig = (round(top_new, 2), round(x_try, 1), round(y_try, 1), round(math.degrees(th_k), 0))
                if sig in seen_sigs:
                    continue
                seen_sigs.add(sig)
                xs_new = xs_b.copy()
                ys_new = ys_b.copy()
                ths_new = ths_b.copy()
                xs_new[cnt] = x_try
                ys_new[cnt] = y_try
                ths_new[cnt] = th_k
                new_sum_th = cur_sum_th + abs(th_k)
                next_states.append((sc, xs_new, ys_new, ths_new, cnt + 1, top_new, new_sum_y, new_sum_th))
                kept_per_bucket[bkt] = kept_per_bucket.get(bkt, 0) + 1

        # Stratified beam selection by average tilt bucket so non-0 branches compete on row 2+
        buckets: dict[int, list] = {}
        for st in next_states:
            avg_deg = math.degrees(st[7] / st[4])
            sb = min(3, int(avg_deg // 10.0))
            buckets.setdefault(sb, []).append(st)
        new_beam = []
        per_b = max(3, beam_width // max(1, len(buckets)))
        leftovers = []
        for sb, lst in sorted(buckets.items()):
            lst.sort(key=lambda t: t[0])
            new_beam.extend(lst[:per_b])
            leftovers.extend(lst[per_b:])
        if len(new_beam) < beam_width:
            leftovers.sort(key=lambda t: t[0])
            new_beam.extend(leftovers[: beam_width - len(new_beam)])
        beam = new_beam[:beam_width]
        if (step + 1) in (12, 18, 24, 30, target_n):
            best_s = min(beam, key=lambda t: t[5])
            best_heights_by_n[step + 1] = float(best_s[5])

    best_final = min(beam, key=lambda t: t[5])
    xs = best_final[1][:target_n].copy()
    ys = best_final[2][:target_n].copy()
    ths = best_final[3][:target_n].copy()
    H_raw = best_final[5]

    # Continuous L-BFGS compaction in fixed width [0, W] minimizing H (bisection on H)
    lo_H, hi_H = max(0.5, target_n / W), H_raw
    best_x, best_y, best_th, best_H = xs.copy(), ys.copy(), ths.copy(), H_raw
    for _ in range(10):
        mid_H = 0.5 * (lo_H + hi_H)
        # Check feasibility by optimizing penalty inside [0, W] x [0, mid_H]
        rx, ry, rth, rpen = compact_strip_box(best_x, best_y, best_th, W, mid_H, maxiter=160)
        ov = check_overlap_pair_count(rx, ry, rth)
        hx_r = 0.5 * (np.abs(np.cos(rth)) + np.abs(np.sin(rth)))
        out_wall = max(
            0.0,
            float(np.max(hx_r - rx)),
            float(np.max(rx + hx_r - W)),
            float(np.max(hx_r - ry)),
            float(np.max(ry + hx_r - mid_H)),
        )
        if ov < 1e-8 and out_wall < 1e-8:
            hi_H = mid_H
            best_x, best_y, best_th, best_H = rx, ry, rth, mid_H
        else:
            lo_H = mid_H

    # Measure effective bulk density rho over middle rows (discarding top/bottom 1.5 layers) and total N/H
    rho_total = target_n / best_H
    # Count interior density using height spans between quantiles (row 20% to 80%)
    order_y = np.argsort(best_y)
    lo_idx = max(P, target_n // 5)
    hi_idx = min(target_n - P, (4 * target_n) // 5)
    span_h = float(best_y[order_y[hi_idx]] - best_y[order_y[lo_idx]])
    rho_interior = (hi_idx - lo_idx) / span_h if span_h > 0.5 else rho_total

    # Upright reference height for target_n in width P
    H_upright = math.ceil(target_n / max(P, 1))
    # Uniform sheared reference height: first row starts at sec(th)/2? No, each row takes sec(th) height;
    # actually a row of w squares needs height cos(th)+sin(th) for row 1, and delta_y = sec(th) per row.
    # For N = q*w + r: roughly ceil(N/best_w) rows? Tight height = (q-1)/cos(th) + (cos(th)+sin(th))
    q_shear = math.ceil(target_n / best_w)
    th_rad = math.radians(best_th_deg)
    H_shear_est = (q_shear - 1) / math.cos(th_rad) + (math.cos(th_rad) + math.sin(th_rad)) if q_shear >= 1 else 0.0

    # Angle histogram
    ang_deg = np.round(np.degrees(best_th), 1)
    uniq, cnts = np.unique(np.abs(ang_deg), return_counts=True)
    ang_summary = {float(u): int(c) for u, c in sorted(zip(uniq, cnts), key=lambda z: -z[1])[:5]}

    return {
        "W": W,
        "P": P,
        "f": round(W - P, 4),
        "target_n": target_n,
        "rho_upright": P,
        "shear_baseline_w": best_w,
        "shear_baseline_th_deg": round(best_th_deg, 3),
        "rho_shear": round(rho_shear, 5),
        "H_raw_dp": round(H_raw, 5),
        "H_compacted": round(best_H, 5),
        "H_upright": H_upright,
        "H_shear_est": round(H_shear_est, 5),
        "rho_total_dp": round(rho_total, 5),
        "rho_interior_dp": round(rho_interior, 5),
        "beats_upright": best_H < H_upright - 1e-4,
        "beats_shear": rho_total > rho_shear * (1.0 - 0.02) and best_H < H_shear_est - 1e-4,
        "top_abs_angles_deg": ang_summary,
        "checkpoints_H": {str(k): round(v, 4) for k, v in best_heights_by_n.items()},
    }


def compact_strip_box(
    x: np.ndarray, y: np.ndarray, th: np.ndarray, W: float, H: float, maxiter: int = 160
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """L-BFGS relax inside rectangular strip [0, W] x [0, H]."""
    n = len(x)

    def obj_grad(state):
        xs = state[0::3]
        ys = state[1::3]
        ths = state[2::3]
        g = np.zeros_like(state)
        tot = 0.0
        c = np.cos(ths)
        s = np.sin(ths)
        # Wall penalty in [0, W] x [0, H] via corners
        dx_c = np.array([-0.5, 0.5, 0.5, -0.5])
        dy_c = np.array([-0.5, -0.5, 0.5, 0.5])
        for k in range(4):
            cx = xs + dx_c[k] * c - dy_c[k] * s
            cy = ys + dx_c[k] * s + dy_c[k] * c
            dcx_dth = -dx_c[k] * s - dy_c[k] * c
            dcy_dth = dx_c[k] * c - dy_c[k] * s
            v0 = np.maximum(0.0, -cx)
            v1 = np.maximum(0.0, cx - W)
            v2 = np.maximum(0.0, -cy)
            v3 = np.maximum(0.0, cy - H)
            tot += float(np.sum(v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3))
            gx = 2.0 * (v1 - v0)
            gy = 2.0 * (v3 - v2)
            g[0::3] += gx
            g[1::3] += gy
            g[2::3] += gx * dcx_dth + gy * dcy_dth

        # Pair SAT penalty using gu Numba/analytical? Reuse gu.sat_penalty_grad logic on padded box?
        # Wait: gu.sat_penalty_grad checks walls at [0, L]. If W < H, shift nothing and just subtract wall?
        # Simpler: call gu.sat_penalty_grad with L = max(W, H) + 10.0 and w_wall=0.0!
        pair_val, pair_g = gu.sat_penalty_grad(state, max(W, H) + 10.0, w_wall=0.0, w_pair=1.0)
        return tot + pair_val, g + pair_g

    s0 = np.empty(3 * n, dtype=np.float64)
    s0[0::3] = x
    s0[1::3] = y
    s0[2::3] = th
    res = minimize(obj_grad, s0, jac=True, method="L-BFGS-B", options={"maxiter": maxiter, "ftol": 1e-15})
    return res.x[0::3], res.x[1::3], res.x[2::3], float(res.fun)


def bisect_strip_height(x0: np.ndarray, y0: np.ndarray, th0: np.ndarray, W: float, hi_init: float) -> float:
    """Find smallest H >= target_n / W achievable by continuous L-BFGS compaction in [0, W] x [0, H]."""
    n = len(x0)
    lo = max(0.5, n / W)
    hi = hi_init
    bx, by, bth = x0.copy(), y0.copy(), th0.copy()
    # First ensure initial state or slightly inflated hi is feasible
    ov0 = check_overlap_pair_count(bx, by, bth)
    if ov0 > 1e-8:
        rx, ry, rth, _ = compact_strip_box(bx, by, bth, W, hi, maxiter=220)
        ov1 = check_overlap_pair_count(rx, ry, rth)
        hx_r = 0.5 * (np.abs(np.cos(rth)) + np.abs(np.sin(rth)))
        out1 = max(0.0, float(np.max(hx_r - rx)), float(np.max(rx + hx_r - W)), float(np.max(hx_r - ry)), float(np.max(ry + hx_r - hi)))
        if ov1 < 1e-8 and out1 < 1e-8:
            bx, by, bth = rx, ry, rth
        else:
            return float("inf")

    for _ in range(12):
        mid = 0.5 * (lo + hi)
        rx, ry, rth, _ = compact_strip_box(bx, by, bth, W, mid, maxiter=180)
        ov = check_overlap_pair_count(rx, ry, rth)
        hx_r = 0.5 * (np.abs(np.cos(rth)) + np.abs(np.sin(rth)))
        out_w = max(0.0, float(np.max(hx_r - rx)), float(np.max(rx + hx_r - W)), float(np.max(hx_r - ry)), float(np.max(ry + hx_r - mid)))
        if ov < 1e-8 and out_w < 1e-8:
            hi = mid
            bx, by, bth = rx, ry, rth
        else:
            lo = mid
    return float(hi)


def eval_motif_strip(W: float, target_n: int) -> dict:
    """Test structured sheared+tapered-cap and alternating-angle strip motifs with L-BFGS compaction."""
    P = int(math.floor(W))
    best_H = float("inf")
    best_name = "none"
    motif_results = {}

    # Motif 1: Uniform sheared w = P+1 (or P+2) + compacted end caps
    for w in range(P + 1, P + 3):
        th = solve_shear_angle(w, W - 1e-5)
        if th is None:
            continue
        c, s = math.cos(th), math.sin(th)
        hx = 0.5 * (c + s)
        # Try (A) pure uniform sheared rows, (B) bottom upright P-row + sheared core
        for mode in ("uniform_shear", "upright_bottom_taper"):
            xs, ys, ths = [], [], []
            if mode == "upright_bottom_taper":
                for k in range(P):
                    xs.append(0.5 + k * (W - P) / max(P, 1) + k)
                    ys.append(0.5)
                    ths.append(0.0)
                y_start = 1.0 + hx + 0.05
            else:
                y_start = hx + 1e-5
            r = 0
            while len(xs) < target_n:
                y0 = y_start + r / c
                for k in range(w):
                    if len(xs) >= target_n:
                        break
                    xs.append(hx + k * c)
                    ys.append(y0 + k * s)
                    ths.append(th)
                r += 1
            xs_a, ys_a, ths_a = np.array(xs), np.array(ys), np.array(ths)
            hi = float(np.max(ys_a + 0.5 * (np.abs(np.cos(ths_a)) + np.abs(np.sin(ths_a))))) + 0.15
            H_comp = bisect_strip_height(xs_a, ys_a, ths_a, W, hi)
            key = f"{mode}_w{w}"
            motif_results[key] = round(H_comp, 5)
            if H_comp < best_H:
                best_H = H_comp
                best_name = key

    # Motif 2: Checkerboard alternating tilt (+a, -a) seeded rows compacted
    for a_deg in (12.0, 22.0):
        a = math.radians(a_deg)
        ca, sa = math.cos(a), math.sin(a)
        hxa = 0.5 * (ca + sa)
        if P * 2 * hxa > W + 0.3:
            continue
        xs, ys, ths = [], [], []
        cols = max(1, int(math.floor(W / (ca + sa))))
        r = 0
        while len(xs) < target_n:
            for k in range(cols):
                if len(xs) >= target_n:
                    break
                ang = a if (r + k) % 2 == 0 else -a
                xs.append(hxa + k * (W - 2 * hxa) / max(1, cols - 1) if cols > 1 else 0.5 * W)
                ys.append(hxa + r * (2 * hxa + 0.01))
                ths.append(ang)
            r += 1
        xs_a, ys_a, ths_a = np.array(xs), np.array(ys), np.array(ths)
        hi = float(np.max(ys_a + hxa)) + 0.1
        H_comp = bisect_strip_height(xs_a, ys_a, ths_a, W, hi)
        key = f"checker_{int(a_deg)}deg"
        motif_results[key] = round(H_comp, 5)
        if H_comp < best_H:
            best_H = H_comp
            best_name = key

    return {
        "best_motif": best_name,
        "best_motif_H": round(best_H, 5) if math.isfinite(best_H) else None,
        "best_motif_rho": round(target_n / best_H, 5) if math.isfinite(best_H) else None,
        "all_motifs_H": motif_results,
    }


def main():
    t0 = time.time()
    # Warm up Numba JIT on tiny strip
    _ = strip_beam_search(3.502, target_n=12, beam_width=6)

    strip_specs = [
        (2.502, 24),
        (3.502, 28),
        (4.502, 32),
        (5.502, 36),
        (6.502, 40),
        (7.502, 44),
        (7.661, 44),  # matches n=971 f=0.661 sheared pocket!
    ]
    results = []
    for W, n_target in strip_specs:
        ts = time.time()
        print(f"Running POC 3 beam search + motifs on W={W:.3f}, target_n={n_target}...", flush=True)
        res = strip_beam_search(W, target_n=n_target, beam_width=24)
        motif = eval_motif_strip(W, n_target)
        res["motif_compaction"] = motif
        res["seconds"] = round(time.time() - ts, 2)
        results.append(res)
        print(
            f"  -> W={W}: DP H={res['H_compacted']} | upright H={res['H_upright']} "
            f"| shear H_est={res['H_shear_est']} (rho={res['rho_shear']}) "
            f"| best_motif={motif['best_motif']} H={motif['best_motif_H']} (rho={motif['best_motif_rho']}) "
            f"({res['seconds']}s)",
            flush=True,
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "poc3_skyline_dp.json"
    out_path.write_text(json.dumps({"total_seconds": round(time.time() - t0, 2), "strips": results}, indent=2))
    print(f"Saved {out_path} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
