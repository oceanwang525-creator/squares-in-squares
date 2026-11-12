"""Core geometry, sheared L-pocket builders, exact end-void fillers, L-BFGS, SLP angle+center polish, and official certification."""
from __future__ import annotations

import os
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_var] = "1"

import math
import sys
from pathlib import Path
from typing import Iterable

import numba as nb
import numpy as np
from scipy.optimize import brentq, linprog, minimize

REPO_ROOT = Path(__file__).resolve().parents[3]
HB_POC = REPO_ROOT / "poc" / "hydrogen_brainstorm" / "poc"
if str(HB_POC) not in sys.path:
    sys.path.insert(0, str(HB_POC))

import geom_utils as gu  # noqa: E402


def solve_shear_angle(w: int, W: float) -> float | None:
    """Solve w * cos(th) + sin(th) = W for th in [atan(1/w), pi/4]."""
    if w <= 0 or W <= 0:
        return None
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


@nb.njit(cache=True)
def _can_place_sat(
    gx: float,
    gy: float,
    ang: float,
    arr_x: np.ndarray,
    arr_y: np.ndarray,
    arr_c: np.ndarray,
    arr_s: np.ndarray,
    L: float,
    margin: float = -1e-9,
) -> float:
    """Return min gap (>= 0 feasible; < 0 overlap/wall violation) for placing 1 square."""
    ca = math.cos(ang)
    sa = math.sin(ang)
    hxa = 0.5 * (abs(ca) + abs(sa))
    wall_gap = min(gx - hxa, L - gx - hxa, gy - hxa, L - gy - hxa)
    if wall_gap < -0.35:
        return wall_gap
    min_gap = wall_gap
    n = len(arr_x)
    for j in range(n):
        dx = gx - arr_x[j]
        dy = gy - arr_y[j]
        if abs(dx) >= 1.45 or abs(dy) >= 1.45:
            continue
        cj, sj = arr_c[j], arr_s[j]
        rad_a = 0.5 + 0.5 * (abs(ca * cj + sa * sj) + abs(ca * sj - sa * cj))
        p1 = abs(dx * ca + dy * sa) - rad_a
        p2 = abs(-dx * sa + dy * ca) - rad_a
        p3 = abs(dx * cj + dy * sj) - rad_a
        p4 = abs(-dx * sj + dy * cj) - rad_a
        sep = max(p1, p2, p3, p4)
        if sep < min_gap:
            min_gap = sep
            if min_gap < -0.35:
                return min_gap
    return min_gap - margin


def make_upright_grid(x0: float, x1: float, y0: float, y1: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate integer-spaced upright squares anchored at (x0, y0) fitting inside [x0, x1] x [y0, y1]."""
    nx = int(math.floor(x1 - x0 + 1e-9))
    ny = int(math.floor(y1 - y0 + 1e-9))
    if nx <= 0 or ny <= 0:
        return np.empty(0), np.empty(0), np.empty(0)
    xs = [x0 + 0.5 + i for i in range(nx) for _ in range(ny)]
    ys = [y0 + 0.5 + j for i in range(nx) for j in range(ny)]
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64), np.zeros(len(xs), dtype=np.float64)


def make_sheared_strip(
    W: float,
    y_min: float,
    y_max: float,
    w: int,
    th: float,
    sign: int,
    x_offset: float = 0.0,
    y_phase: float = 0.0,
    dy_scale: float = 1.0,
    x_pad_mode: str = "center",
    u_cols_right: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate sheared rows inside [x_offset, x_offset + W] x [y_min, y_max], optionally with upright cols on right."""
    W_shear = W - float(u_cols_right)
    if W_shear <= 0.5:
        return np.empty(0), np.empty(0), np.empty(0)

    c, s = math.cos(th), math.sin(th)
    hx = 0.5 * (abs(c) + abs(s))
    dy_row = (1.0 / max(c, 1e-9)) * dy_scale
    row_span_x = w * c + abs(s)
    if row_span_x > W_shear + 1e-9:
        return np.empty(0), np.empty(0), np.empty(0)

    slack = max(0.0, W_shear - row_span_x)
    if x_pad_mode == "left":
        x_pad = 0.0
    elif x_pad_mode == "right":
        x_pad = slack
    else:
        x_pad = 0.5 * slack

    r_min = -int(math.ceil(w * abs(s) / dy_row)) - 3
    r_max = int(math.ceil((y_max - y_min + w * abs(s)) / dy_row)) + 3

    xs, ys, ths = [], [], []
    for r in range(r_min, r_max + 1):
        y0 = y_min + hx + (r + y_phase) * dy_row
        for k in range(w):
            xk = x_offset + x_pad + hx + k * c
            yk = y0 + sign * k * s
            if (
                xk - hx >= x_offset - 1e-9
                and xk + hx <= x_offset + W_shear + 1e-9
                and yk - hx >= y_min - 1e-9
                and yk + hx <= y_max + 1e-9
            ):
                xs.append(xk)
                ys.append(yk)
                ths.append(sign * th)

    if u_cols_right > 0:
        ux0 = x_offset + W - float(u_cols_right)
        ny = int(math.floor(y_max - y_min + 1e-9))
        for ci in range(u_cols_right):
            for rj in range(ny):
                xs.append(ux0 + 0.5 + ci)
                ys.append(y_min + 0.5 + rj)
                ths.append(0.0)

    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64), np.array(ths, dtype=np.float64)


def fill_end_and_void_pockets(
    x: np.ndarray,
    y: np.ndarray,
    th: np.ndarray,
    L: float,
    pocket_boxes: list[tuple[float, float, float, float]],
    candidate_angles: Iterable[float],
    grid_step: float = 0.125,
    extra_soft_candidates: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[float, float, float, float]]]:
    """Fill remaining open space in specified pocket_boxes with flush wall anchors + fine SAT scan.

    Returns (x, y, th, best_soft_holes) where best_soft_holes holds top (gap, gx, gy, ang) with smallest negative gap.
    """
    pts_x = list(x)
    pts_y = list(y)
    pts_th = list(th)
    arr_x = np.array(pts_x, dtype=np.float64)
    arr_y = np.array(pts_y, dtype=np.float64)
    arr_c = np.cos(pts_th)
    arr_s = np.sin(pts_th)
    cand_angs = list(candidate_angles)

    # Generate wall-flush upright + corner-aligned anchor points inside each box
    probe_pts: list[tuple[float, float, bool]] = []  # (gx, gy, upright_only_priority)
    for x0, x1, y0, y1 in pocket_boxes:
        nx = max(1, int(math.floor(x1 - x0)) + 1)
        ny = max(1, int(math.floor(y1 - y0)) + 1)
        xs_flush = set()
        ys_flush = set()
        for i in range(nx):
            if x0 + 0.5 + i <= x1 - 0.5 + 1e-6:
                xs_flush.add(x0 + 0.5 + i)
            if x1 - 0.5 - i >= x0 + 0.5 - 1e-6:
                xs_flush.add(x1 - 0.5 - i)
        for j in range(ny):
            if y0 + 0.5 + j <= y1 - 0.5 + 1e-6:
                ys_flush.add(y0 + 0.5 + j)
            if y1 - 0.5 - j >= y0 + 0.5 - 1e-6:
                ys_flush.add(y1 - 0.5 - j)
        for gx in sorted(xs_flush):
            for gy in sorted(ys_flush):
                probe_pts.append((gx, gy, True))
        # Fine interior scan ONLY in pockets
        if x1 - x0 >= 0.95 and y1 - y0 >= 0.95:
            gxs = np.arange(max(0.5, x0 + 0.5), min(L - 0.5, x1 - 0.5) + 1e-6, grid_step)
            gys = np.arange(max(0.5, y0 + 0.5), min(L - 0.5, y1 - 0.5) + 1e-6, grid_step)
            for gx in gxs:
                for gy in gys:
                    probe_pts.append((float(gx), float(gy), False))

    soft_holes: list[tuple[float, float, float, float]] = []
    seen_cell = set()
    for gx, gy, _ in probe_pts:
        if len(arr_x) > 0:
            d2 = (arr_x - gx) ** 2 + (arr_y - gy) ** 2
            if float(d2.min()) < 0.72:
                continue
        best_neg_gap = -1e9
        best_neg_ang = 0.0
        placed = False
        for ang in cand_angs:
            gap = float(_can_place_sat(gx, gy, ang, arr_x, arr_y, arr_c, arr_s, L, -1e-9))
            if gap >= 0.0:
                pts_x.append(gx)
                pts_y.append(gy)
                pts_th.append(ang)
                arr_x = np.append(arr_x, gx)
                arr_y = np.append(arr_y, gy)
                arr_c = np.append(arr_c, math.cos(ang))
                arr_s = np.append(arr_s, math.sin(ang))
                placed = True
                break
            elif gap > best_neg_gap:
                best_neg_gap = gap
                best_neg_ang = ang
        if not placed and extra_soft_candidates > 0 and best_neg_gap > -0.25:
            ck = (round(gx, 1), round(gy, 1))
            if ck not in seen_cell:
                seen_cell.add(ck)
                soft_holes.append((best_neg_gap, gx, gy, best_neg_ang))

    soft_holes.sort(key=lambda t: t[0], reverse=True)
    # Deduplicate soft holes so seeded squares are >= 0.95 apart
    filtered_holes: list[tuple[float, float, float, float]] = []
    for gap, gx, gy, ang in soft_holes:
        if all((gx - hx_) ** 2 + (gy - hy_) ** 2 >= 0.85 for _, hx_, hy_, _ in filtered_holes):
            filtered_holes.append((gap, gx, gy, ang))
            if len(filtered_holes) >= extra_soft_candidates:
                break
    return np.array(pts_x, dtype=np.float64), np.array(pts_y, dtype=np.float64), np.array(pts_th, dtype=np.float64), filtered_holes


def slp_angle_center_polish(
    x: np.ndarray,
    y: np.ndarray,
    th: np.ndarray,
    rounds: int = 8,
    trust_xy: float = 0.06,
    trust_th0: float = 0.03,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, bool]:
    """Sequential LP over joint (dx, dy, dth, L) with finite-diff SAT gap + quadratic safety margin."""
    x = x.copy()
    y = y.copy()
    th = th.copy()
    n = len(x)
    _, cur_L = gu.sat_max_overlap_and_bbox(x, y, th)
    tau_th = trust_th0
    tau_xy = trust_xy
    eps = 1e-6

    for rd in range(rounds):
        # Build active & near pairs (< 1.55)
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d2 = dx * dx + dy * dy
        ii, jj = np.where((np.triu(np.ones((n, n), dtype=bool), k=1)) & (d2 < 2.40))

        # First shift state so min_x_corner = 0, min_y_corner = 0
        hx = 0.5 * (np.abs(np.cos(th)) + np.abs(np.sin(th)))
        x -= float(np.min(x - hx))
        y -= float(np.min(y - hx))
        cur_L = max(float(np.max(x + hx)), float(np.max(y + hx)))

        # Variables: [dx_0..n-1, dy_0..n-1, dth_0..n-1, L_new] -> 3n + 1 vars
        nv = 3 * n + 1
        c_obj = np.zeros(nv)
        c_obj[-1] = 1.0

        rows_r, cols_c, vals_v, rhs_b = [], [], [], []
        row_idx = 0
        # 2nd order angular curvature margin ~ 0.5 * tau_th^2
        margin = max(2e-9, 0.45 * tau_th * tau_th)

        # Box constraints for each square:
        # 0 <= (x + dx) - hx(th + dth), (x + dx) + hx(th + dth) <= L_new
        dhx = (0.5 * (np.abs(np.cos(th + eps)) + np.abs(np.sin(th + eps))) - hx) / eps
        for i in range(n):
            # -dx_i + dhx_i * dth_i <= x_i - hx_i - margin
            rows_r.extend([row_idx, row_idx])
            cols_c.extend([i, 2 * n + i])
            vals_v.extend([-1.0, float(dhx[i])])
            rhs_b.append(float(x[i] - hx[i] - margin))
            row_idx += 1

            # dx_i + dhx_i * dth_i - L_new <= -x_i - hx_i - margin
            rows_r.extend([row_idx, row_idx, row_idx])
            cols_c.extend([i, 2 * n + i, 3 * n])
            vals_v.extend([1.0, float(dhx[i]), -1.0])
            rhs_b.append(float(-x[i] - hx[i] - margin))
            row_idx += 1

            # -dy_i + dhx_i * dth_i <= y_i - hx_i - margin
            rows_r.extend([row_idx, row_idx])
            cols_c.extend([n + i, 2 * n + i])
            vals_v.extend([-1.0, float(dhx[i])])
            rhs_b.append(float(y[i] - hx[i] - margin))
            row_idx += 1

            # dy_i + dhx_i * dth_i - L_new <= -y_i - hx_i - margin
            rows_r.extend([row_idx, row_idx, row_idx])
            cols_c.extend([n + i, 2 * n + i, 3 * n])
            vals_v.extend([1.0, float(dhx[i]), -1.0])
            rhs_b.append(float(-y[i] - hx[i] - margin))
            row_idx += 1

        # Pairwise SAT separating axis constraints: pick largest-margin axis at current pose
        ci = np.cos(th[ii])
        si = np.sin(th[ii])
        cj = np.cos(th[jj])
        sj = np.sin(th[jj])
        ddx = x[ii] - x[jj]
        ddy = y[ii] - y[jj]

        # Compute exact gap along all 4 normals for each pair
        for idx_p in range(len(ii)):
            i = int(ii[idx_p])
            j = int(jj[idx_p])
            ax_list = (
                (float(ci[idx_p]), float(si[idx_p]), th[i], th[j], i, j),
                (float(-si[idx_p]), float(ci[idx_p]), th[i], th[j], i, j),
                (float(cj[idx_p]), float(sj[idx_p]), th[j], th[i], j, i),
                (float(-sj[idx_p]), float(cj[idx_p]), th[j], th[i], j, i),
            )
            best_g = -1e18
            best_k = 0
            best_sgn = 1.0
            for k in range(4):
                ux, uy, tha, thb, a_id, _ = ax_list[k]
                proj = ddx[idx_p] * ux + ddy[idx_p] * uy
                # rad = 0.5 + 0.5*(|cos(thb-tha)| + |sin(thb-tha)|)
                dth_ab = thb - tha
                rad = 0.5 + 0.5 * (abs(math.cos(dth_ab)) + abs(math.sin(dth_ab)))
                g_k = abs(proj) - rad
                if g_k > best_g:
                    best_g = g_k
                    best_k = k
                    best_sgn = 1.0 if proj >= 0.0 else -1.0

            # Linearize sgn * (dx_ij * u_x + dy_ij * u_y) - rad >= margin
            # i.e. -grad * dvar <= best_g - margin
            ux, uy, tha, thb, a_id, b_id = ax_list[best_k]

            def eval_gap(xi, yi, thi, xj, yj, thj):
                if a_id == i:
                    ca_l = math.cos(thi) if best_k == 0 else -math.sin(thi)
                    sa_l = math.sin(thi) if best_k == 0 else math.cos(thi)
                    d_th = thj - thi
                else:
                    ca_l = math.cos(thj) if best_k == 2 else -math.sin(thj)
                    sa_l = math.sin(thj) if best_k == 2 else math.cos(thj)
                    d_th = thi - thj
                proj_l = (xi - xj) * ca_l + (yi - yj) * sa_l
                rad_l = 0.5 + 0.5 * (abs(math.cos(d_th)) + abs(math.sin(d_th)))
                return best_sgn * proj_l - rad_l

            g0 = eval_gap(x[i], y[i], th[i], x[j], y[j], th[j])
            g_xi = (eval_gap(x[i] + eps, y[i], th[i], x[j], y[j], th[j]) - g0) / eps
            g_yi = (eval_gap(x[i], y[i] + eps, th[i], x[j], y[j], th[j]) - g0) / eps
            g_thi = (eval_gap(x[i], y[i], th[i] + eps, x[j], y[j], th[j]) - g0) / eps
            g_thj = (eval_gap(x[i], y[i], th[i], x[j], y[j], th[j] + eps) - g0) / eps

            # Constraint: -(g_xi*(dxi-dxj) + g_yi*(dyi-dyj) + g_thi*dthi + g_thj*dthj) <= g0 - margin
            rows_r.extend([row_idx] * 6)
            cols_c.extend([i, j, n + i, n + j, 2 * n + i, 2 * n + j])
            vals_v.extend([-g_xi, g_xi, -g_yi, g_yi, -g_thi, -g_thj])
            rhs_b.append(float(g0 - margin))
            row_idx += 1

        from scipy.sparse import csc_matrix
        A_ub = csc_matrix((vals_v, (rows_r, cols_c)), shape=(row_idx, nv))
        bounds = [(-tau_xy, tau_xy)] * (2 * n) + [(-tau_th, tau_th)] * n + [(0.5 * cur_L, cur_L + 0.05)]
        res = linprog(c_obj, A_ub=A_ub, b_ub=np.array(rhs_b), bounds=bounds, method="highs")
        if not res.success:
            tau_th *= 0.5
            tau_xy *= 0.5
            continue

        sol = res.x
        xn = x + sol[:n]
        yn = y + sol[n : 2 * n]
        thn = th + sol[2 * n : 3 * n]
        pen, L_meas = gu.sat_max_overlap_and_bbox(xn, yn, thn)
        if pen > 1e-10:
            # Repair tiny second-order chord overlap at L_meas
            xn, yn, thn, _ = relax_tight(xn, yn, thn, L_meas, maxiter=200)
            pen, L_meas = gu.sat_max_overlap_and_bbox(xn, yn, thn)
        if pen <= 1e-10 and L_meas < cur_L:
            x, y, th, cur_L = xn, yn, thn, L_meas
            if cur_L - L_meas < 1e-7:
                tau_th *= 0.6
        else:
            tau_th *= 0.5
            tau_xy *= 0.5

    # Final exact fixed-angle LP tighten
    lx, ly, lL, ok = strict_fixed_lp_tighten(x, y, th, rounds=6)
    return lx, ly, th, lL, ok


def relax_tight(
    x: np.ndarray,
    y: np.ndarray,
    th: np.ndarray,
    L: float,
    free_mask: np.ndarray | None = None,
    maxiter: int = 350,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Run high-weight SAT L-BFGS-B relaxation so quadratic convergence clears 1e-10 overlaps."""
    return gu.relax_lbfgs(
        x, y, th, L, free_mask=free_mask, maxiter=maxiter, w_wall=1e6, w_pair=1e6
    )


def strict_fixed_lp_tighten(
    x: np.ndarray,
    y: np.ndarray,
    th: np.ndarray,
    rounds: int = 6,
) -> tuple[np.ndarray, np.ndarray, float, bool]:
    """Fixed-angle LP tighten using margin above HiGHS primal tolerance and exact pen<=0 check."""
    for margin in (2.5e-7, 1e-6, 5e-6):
        lx, ly, lL, ok = gu.fixed_angle_lp_tighten(x, y, th, rounds=rounds, margin=margin)
        pen_f, _ = gu.sat_max_overlap_and_bbox(lx, ly, th)
        if pen_f <= 0.0:
            return lx, ly, lL, True
    # Fallback: if fixed-angle LP hit HiGHS slack, re-check after slight margin boost
    pen0, L0 = gu.sat_max_overlap_and_bbox(x, y, th)
    return x.copy(), y.copy(), L0, (pen0 <= 0.0)


def squeeze_L_by_relax_and_slp(
    x: np.ndarray,
    y: np.ndarray,
    th: np.ndarray,
    L_start: float,
    step: float = 0.004,
    max_bisect: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, bool]:
    """Compress feasible packing x,y,th downward in L via soft-box L-BFGS bisection + SLP polish."""
    bx, by, bth = x.copy(), y.copy(), th.copy()
    pen0, bL = gu.sat_max_overlap_and_bbox(bx, by, bth)
    if pen0 > 1e-10:
        rx, ry, rth, _ = relax_tight(bx, by, bth, L_start, maxiter=400)
        pen0, bL = gu.sat_max_overlap_and_bbox(rx, ry, rth)
        if pen0 > 1e-9:
            return bx, by, bth, bL, False
        bx, by, bth = rx, ry, rth

    # Step down until infeasible, then bisect
    cur_L = bL
    for _ in range(12):
        trial_L = cur_L - step
        sx = bx * (trial_L / cur_L)
        sy = by * (trial_L / cur_L)
        rx, ry, rth, _ = relax_tight(sx, sy, bth, trial_L, maxiter=300)
        pen, mL = gu.sat_max_overlap_and_bbox(rx, ry, rth)
        if pen <= 1e-10:
            bx, by, bth, cur_L = rx, ry, rth, mL
        else:
            lo, hi = trial_L, cur_L
            for _ in range(max_bisect):
                mid = 0.5 * (lo + hi)
                sx = bx * (mid / cur_L)
                sy = by * (mid / cur_L)
                rx, ry, rth, _ = relax_tight(sx, sy, bth, mid, maxiter=250)
                pen, mL = gu.sat_max_overlap_and_bbox(rx, ry, rth)
                if pen <= 1e-10:
                    bx, by, bth, cur_L = rx, ry, rth, mL
                    hi = mL
                else:
                    lo = mid
            break

    lx, ly, lth, lL, lok = slp_angle_center_polish(bx, by, bth, rounds=8)
    return lx, ly, lth, lL, lok
