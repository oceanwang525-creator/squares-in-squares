#!/usr/bin/env python3
"""H78 Constant-r Homology L-Border Cascade engine with seam cell surgery + lpslack + mechwalk."""
from __future__ import annotations

import math
import sys
import time
from decimal import Decimal
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import homocasc2_lib as hc  # noqa: E402
from cpp_wrap import cpp_eval_sat_pen_bbox, cpp_expand_sym, full_to_free_sym, run_cpp_masked_melt  # noqa: E402


def _corner_defs(L: float, k_cut: float) -> list[tuple[str, callable, float, float]]:
    return [
        ("BL", lambda x, y: (x <= L - k_cut) & (y <= L - k_cut), 0.0, 0.0),
        ("TL", lambda x, y: (x <= L - k_cut) & (y >= k_cut), 0.0, -1.0),
        ("BR", lambda x, y: (x >= k_cut) & (y <= L - k_cut), -1.0, 0.0),
        ("TR", lambda x, y: (x >= k_cut) & (y >= k_cut), -1.0, -1.0),
    ]


def _protrusion_amounts(x: np.ndarray, y: np.ndarray, h: np.ndarray, Lt: float) -> np.ndarray:
    return np.maximum.reduce([-(x - h), (x + h) - Lt, -(y - h), (y + h) - Lt])


def _seam_centers(x: np.ndarray, y: np.ndarray, th: np.ndarray, ref_pts: np.ndarray) -> np.ndarray:
    deg = np.degrees(th) % 90.0
    tilted = (deg > 1.0) & (deg < 89.0)
    if not np.any(tilted) or len(ref_pts) == 0:
        return ref_pts if len(ref_pts) else np.array([[0.5, 0.5]])
    tx, ty = x[tilted], y[tilted]
    d2 = ((tx[:, None] - ref_pts[None, :, 0]) ** 2 + (ty[:, None] - ref_pts[None, :, 1]) ** 2).min(axis=1)
    best = np.argsort(d2)[:2]
    return np.vstack([ref_pts, np.c_[tx[best], ty[best]]])


def extend_one_layer_all_corners(
    x: np.ndarray, y: np.ndarray, th: np.ndarray, L: float
) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray, float]]:
    """Extend normalized [0, L]^2 packing by one 1x1 upright L-border along each of the 4 corners."""
    m = int(math.floor(L))
    b = 0.5 + np.arange(m, dtype=np.float64) * (1.0 + 1e-9)
    gap = 1e-9
    off = L + 0.5 + gap
    out = []
    for tag, sx, sy, bx, by in [
        ("extTR", 0.0, 0.0, off, off),
        ("extTL", 1.0 + gap, 0.0, 0.5, off),
        ("extBR", 0.0, 1.0 + gap, off, 0.5),
        ("extBL", 1.0 + gap, 1.0 + gap, 0.5, 0.5),
    ]:
        hx_off = 1.0 + gap if bx == 0.5 else 0.0
        hy_off = 1.0 + gap if by == 0.5 else 0.0
        cx = np.concatenate([x + sx, np.full(m, bx), b + hx_off, [bx]])
        cy = np.concatenate([y + sy, b + hy_off, np.full(m, by), [by]])
        ct = np.concatenate([th, np.zeros(2 * m + 1)])
        nx, ny, nt, nL = hc.norm_pack(cx, cy, ct)
        out.append((tag, nx, ny, nt, nL))
    return out


def generate_peel_seeds(
    n_target: int,
    n_donor: int,
    xd: np.ndarray,
    yd: np.ndarray,
    td: np.ndarray,
    Ld: float,
    L_inc: float,
    rng: np.random.Generator,
    layers: int = 1,
    max_prot: int = 18,
    max_rem_tilt: int = 8,
) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]]:
    """
    Generate L-border peel seeds (layers=1 or 2) across BL/TL/BR/TR and multiple k_cut thresholds,
    including clamp, interior-void relocation, repaired relocation, and seam-cell surgery.
    """
    hd = hc.hx_of(td)
    tilt = hc.dominant_tilt(td)
    Lt = Ld - float(layers)
    if Lt >= L_inc + 0.010:
        return []

    Lt_use = min(L_inc - 3e-4, max(Lt, L_inc - 0.025))
    seeds: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]] = []
    seen_masks: set[tuple[str, bytes]] = set()

    k_cuts = (
        [1.01, 1.045, 1.075, 0.985] if layers == 1 else [float(layers) + d for d in (0.01, 0.045, 0.08, -0.015)]
    )
    shift_amt = float(layers)

    for k_cut in k_cuts:
        for cname, mask_fn, dx_u, dy_u in _corner_defs(Ld, k_cut=k_cut):
            dx, dy = dx_u * shift_amt, dy_u * shift_amt
            keep = mask_fn(xd, yd)
            sig = (cname, keep.tobytes())
            if sig in seen_masks:
                continue
            seen_masks.add(sig)

            rem = ~keep
            rem_deg = np.degrees(td[rem]) % 90.0
            rem_tilt = int(((rem_deg > 1.0) & (rem_deg < 89.0)).sum())
            if rem_tilt > max_rem_tilt:
                continue

            cnt = int(keep.sum())
            diff_n = cnt - n_target
            if diff_n < -1 or diff_n > 3:
                continue

            xk = xd[keep] + dx
            yk = yd[keep] + dy
            tk = td[keep].copy()
            hk = hd[keep]

            # If cnt == n_target - 1, insert 1 square in best interior clearance void first
            extra_ctrs: list[list[float]] = []
            if diff_n == -1:
                vp0 = hc.clearance_points(xk, yk, tk, Lt, 2, step=0.10, min_sep=0.80)
                if not vp0:
                    continue
                vx, vy, _ = vp0[0]
                xk = np.append(xk, vx)
                yk = np.append(yk, vy)
                tk = np.append(tk, 0.0)
                hk = hc.hx_of(tk)
                extra_ctrs.append([vx, vy])
                diff_n = 0

            prot_amt = _protrusion_amounts(xk, yk, hk, Lt)
            p_idx = np.where(prot_amt > 1e-5)[0]
            if len(p_idx) > max_prot + diff_n:
                continue

            tag_base = f"peel{layers}_{n_donor}_{cname}_k{k_cut:.3f}"

            if diff_n == 0 and len(p_idx) == 0:
                ctrs = np.array(extra_ctrs if extra_ctrs else [[0.5 * Lt, 0.5 * Lt]])
                seeds.append((f"{tag_base}_clean0", xk.copy(), yk.copy(), tk.copy(), Lt_use, ctrs))
                continue

            if diff_n == 0:
                orig_pts = np.c_[xk[p_idx], yk[p_idx]]
                if extra_ctrs:
                    orig_pts = np.vstack([orig_pts, np.array(extra_ctrs)])
                # Method (a): Direct boundary clamp + repair_overlaps
                for ang_tag, th_override in (("keepAng", None), ("rot0", 0.0), ("rot45", np.pi / 4.0)):
                    xc, yc, tc = xk.copy(), yk.copy(), tk.copy()
                    if th_override is not None:
                        tc[p_idx] = th_override
                    hc_arr = hc.hx_of(tc)
                    xc[p_idx] = np.clip(xc[p_idx], hc_arr[p_idx] + 1e-4, Lt - hc_arr[p_idx] - 1e-4)
                    yc[p_idx] = np.clip(yc[p_idx], hc_arr[p_idx] + 1e-4, Lt - hc_arr[p_idx] - 1e-4)
                    d2 = ((xc[:, None] - orig_pts[None, :, 0]) ** 2 + (yc[:, None] - orig_pts[None, :, 1]) ** 2).min(axis=1)
                    L_rep = min(L_inc + 0.002, max(Lt + 0.015, L_inc - 0.004))
                    xr, yr, tr, _ = hc.repair_overlaps(xc, yc, tc, L_rep, free_mask=(d2 <= 5.2 ** 2), maxiter=240)
                    ctrs = _seam_centers(xr, yr, tr, orig_pts)
                    seeds.append((f"{tag_base}_clamp_{ang_tag}", xr, yr, tr, Lt_use, ctrs))

                # Method (b): Relocate protruding squares into top interior voids (if <= 4 protruding)
                if len(p_idx) <= 4:
                    non_prot = np.ones(len(xk), dtype=bool)
                    non_prot[p_idx] = False
                    vp = hc.clearance_points(xk[non_prot], yk[non_prot], tk[non_prot], Lt, len(p_idx) + 6, step=0.10, min_sep=0.80)
                    if len(vp) >= len(p_idx):
                        for shift in range(min(3, len(vp) - len(p_idx) + 1)):
                            for atag, aval in (("u0", 0.0), ("tilt", tilt), ("d45", np.pi / 4.0)):
                                if shift > 0 and atag not in ("u0", "tilt"):
                                    continue
                                xv, yv, tv = xk.copy(), yk.copy(), tk.copy()
                                vpts = []
                                for k_i, sq_i in enumerate(p_idx):
                                    vx, vy, _ = vp[shift + k_i]
                                    xv[sq_i], yv[sq_i], tv[sq_i] = vx, vy, aval
                                    vpts.append([vx, vy])
                                ctrs = np.vstack([orig_pts, np.array(vpts)])
                                seeds.append((f"{tag_base}_reloc_v{shift}_{atag}", xv, yv, tv, Lt_use, ctrs))
                                if shift == 0 and atag in ("u0", "tilt"):
                                    d2 = ((xv[:, None] - ctrs[None, :, 0]) ** 2 + (yv[:, None] - ctrs[None, :, 1]) ** 2).min(axis=1)
                                    L_rep = min(L_inc + 0.003, max(Lt + 0.015, L_inc - 0.002))
                                    xr, yr, tr, _ = hc.repair_overlaps(xv, yv, tv, L_rep, free_mask=(d2 <= 4.5 ** 2), maxiter=220)
                                    seeds.append((f"{tag_base}_relocRep_v{shift}_{atag}", xr, yr, tr, Lt_use, ctrs))

                # Method (c): Seam Cell Surgery — swap protruding tips + adjacent seam squares into voids
                if len(p_idx) <= 3:
                    non_prot = np.ones(len(xk), dtype=bool)
                    non_prot[p_idx] = False
                    # Pick 1 boundary/seam square closest to each protruding square
                    seam_extra = []
                    for sq_i in p_idx:
                        dists = np.hypot(xk - xk[sq_i], yk - yk[sq_i])
                        dists[p_idx] = 1e9
                        for cand_j in np.argsort(dists)[:2]:
                            if cand_j not in seam_extra:
                                seam_extra.append(int(cand_j))
                                break
                    swap_idx = np.array(list(p_idx) + seam_extra[:2], dtype=int)
                    keep_sub = np.ones(len(xk), dtype=bool)
                    keep_sub[swap_idx] = False
                    vp2 = hc.clearance_points(xk[keep_sub], yk[keep_sub], tk[keep_sub], Lt, len(swap_idx) + 3, step=0.10, min_sep=0.80)
                    if len(vp2) >= len(swap_idx):
                        for atag, aval in (("u0", 0.0), ("tilt", tilt)):
                            xs, ys, ts = xk.copy(), yk.copy(), tk.copy()
                            vpts = []
                            for k_i, sq_i in enumerate(swap_idx):
                                vx, vy, _ = vp2[k_i]
                                xs[sq_i], ys[sq_i], ts[sq_i] = vx, vy, aval
                                vpts.append([vx, vy])
                            ctrs = np.vstack([orig_pts, np.array(vpts)])
                            seeds.append((f"{tag_base}_seamSurg_{atag}", xs, ys, ts, Lt_use, ctrs))
            else:
                if len(p_idx) >= diff_n:
                    order_desc = p_idx[np.argsort(-prot_amt[p_idx])]
                    order_asc = p_idx[np.argsort(prot_amt[p_idx])]
                    drop_modes = [("dropProt", order_desc[:diff_n])]
                    if len(p_idx) > diff_n:
                        drop_modes.append(("dropMinProt", order_asc[:diff_n]))
                else:
                    rem_need = diff_n - len(p_idx)
                    non_p = np.setdiff1d(np.arange(len(xk)), p_idx)
                    edge_dist = np.minimum.reduce([xk[non_p], Lt - xk[non_p], yk[non_p], Lt - yk[non_p]])
                    extra = non_p[np.argsort(edge_dist)[:rem_need]]
                    drop_modes = [("dropProtEdge", np.concatenate([p_idx, extra]))]

                for dtag, d_idx in drop_modes:
                    km = np.ones(len(xk), dtype=bool)
                    km[d_idx] = False
                    xr, yr, tr = xk[km].copy(), yk[km].copy(), tk[km].copy()
                    hr = hc.hx_of(tr)
                    p2 = np.where(_protrusion_amounts(xr, yr, hr, Lt) > 1e-5)[0]
                    if len(p2) == 0:
                        ctrs = np.c_[xk[d_idx], yk[d_idx]]
                        seeds.append((f"{tag_base}_{dtag}_clean", xr, yr, tr, Lt_use, ctrs))
                        continue

                    p2_pts = np.c_[xr[p2], yr[p2]]
                    xc, yc, tc = xr.copy(), yr.copy(), tr.copy()
                    xc[p2] = np.clip(xc[p2], hr[p2] + 1e-4, Lt - hr[p2] - 1e-4)
                    yc[p2] = np.clip(yc[p2], hr[p2] + 1e-4, Lt - hr[p2] - 1e-4)
                    d2 = ((xc[:, None] - p2_pts[None, :, 0]) ** 2 + (yc[:, None] - p2_pts[None, :, 1]) ** 2).min(axis=1)
                    L_rep = min(L_inc + 0.002, max(Lt + 0.015, L_inc - 0.004))
                    xr2, yr2, tr2, _ = hc.repair_overlaps(xc, yc, tc, L_rep, free_mask=(d2 <= 5.2 ** 2), maxiter=240)
                    seeds.append((f"{tag_base}_{dtag}_clamp", xr2, yr2, tr2, Lt_use, _seam_centers(xr2, yr2, tr2, p2_pts)))

                    if len(p2) <= 4:
                        non_p2 = np.ones(len(xr), dtype=bool)
                        non_p2[p2] = False
                        vp = hc.clearance_points(xr[non_p2], yr[non_p2], tr[non_p2], Lt, len(p2) + 5, step=0.10, min_sep=0.80)
                        if len(vp) >= len(p2):
                            for shift in range(min(2, len(vp) - len(p2) + 1)):
                                for atag, aval in (("u0", 0.0), ("tilt", tilt)):
                                    xv, yv, tv = xr.copy(), yr.copy(), tr.copy()
                                    vpts = []
                                    for k_i, sq_i in enumerate(p2):
                                        vx, vy, _ = vp[shift + k_i]
                                        xv[sq_i], yv[sq_i], tv[sq_i] = vx, vy, aval
                                        vpts.append([vx, vy])
                                    ctrs = np.vstack([np.c_[xk[d_idx], yk[d_idx]], p2_pts, np.array(vpts)])
                                    seeds.append((f"{tag_base}_{dtag}_reloc_v{shift}_{atag}", xv, yv, tv, Lt_use, ctrs))

    return seeds


def generate_interior_cross_seeds(
    n_target: int,
    n_donor: int,
    xd: np.ndarray,
    yd: np.ndarray,
    td: np.ndarray,
    Ld: float,
    L_inc: float,
    max_seeds: int = 8,
) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]]:
    """
    Scan interior vertical+horizontal 1-unit excision strips across the upright block
    (0.5-unit steps) and keep the cleanest cuts (minimum tilted excision and overlap).
    """
    deg = np.degrees(td) % 90.0
    upright = (deg <= 1.0) | (deg >= 89.0)
    if upright.sum() < 100:
        return []

    Lt = Ld - 1.0
    if Lt >= L_inc + 0.010:
        return []
    Lt_use = min(L_inc - 3e-4, max(Lt, L_inc - 0.025))

    candidates = []
    seen = set()
    cuts = np.arange(0.38 * Ld, 0.94 * Ld, 0.50)
    for xcut in cuts:
        in_x = np.abs(xd - xcut) <= 0.50
        if int((in_x & (~upright)).sum()) > 3:
            continue
        for ycut in cuts:
            rem = in_x | (np.abs(yd - ycut) <= 0.50)
            rem_t = int((rem & (~upright)).sum())
            if rem_t > 4:
                continue
            keep = ~rem
            cnt = int(keep.sum())
            diff_n = cnt - n_target
            if diff_n < -2 or diff_n > 3:
                continue
            sig = keep.tobytes()
            if sig in seen:
                continue
            seen.add(sig)
            candidates.append((rem_t, abs(diff_n), float(xcut), float(ycut), keep, diff_n))

    candidates.sort(key=lambda t: (t[0], t[1]))
    seeds: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]] = []
    for rem_t, _, xcut, ycut, keep, diff_n in candidates[:max_seeds]:
        xk = xd[keep].copy()
        yk = yd[keep].copy()
        tk = td[keep].copy()
        xk[xk > xcut] -= 1.0
        yk[yk > ycut] -= 1.0

        if diff_n > 0:
            ov = hc.overlap_pairs(xk, yk, tk)
            scores = np.zeros(len(xk))
            for pen, i, j in ov:
                scores[i] += pen
                scores[j] += pen
            scores[~((np.degrees(tk) % 90.0 <= 1.0) | (np.degrees(tk) % 90.0 >= 89.0))] -= 10.0
            drop = np.argsort(-scores)[:diff_n]
            km = np.ones(len(xk), dtype=bool)
            km[drop] = False
            xk, yk, tk = xk[km], yk[km], tk[km]
        elif diff_n < 0:
            vp = hc.clearance_points(xk, yk, tk, Lt, -diff_n + 2, step=0.10, min_sep=0.80)
            if len(vp) < -diff_n:
                continue
            add_x = np.array([vp[k][0] for k in range(-diff_n)])
            add_y = np.array([vp[k][1] for k in range(-diff_n)])
            xk = np.concatenate([xk, add_x])
            yk = np.concatenate([yk, add_y])
            tk = np.concatenate([tk, np.zeros(-diff_n)])

        ctrs = np.array([[xcut - 0.5, ycut - 0.5], [xcut - 0.5, 0.5 * Lt], [0.5 * Lt, ycut - 0.5]])
        d2 = ((xk[:, None] - ctrs[None, :, 0]) ** 2 + (yk[:, None] - ctrs[None, :, 1]) ** 2).min(axis=1)
        xr, yr, tr, _ = hc.repair_overlaps(xk, yk, tk, min(L_inc + 0.002, Lt + 0.015), free_mask=(d2 <= 4.5 ** 2), maxiter=220)
        seeds.append((f"xcross_{n_donor}_x{xcut:.1f}_y{ycut:.1f}_rt{rem_t}", xr, yr, tr, Lt_use, ctrs))

    return seeds


def build_melt_jobs_for_seed(
    label: str,
    X: np.ndarray,
    Y: np.ndarray,
    T: np.ndarray,
    Lt: float,
    centers: np.ndarray,
    L_inc: float,
    rng: np.random.Generator,
    n_seeds: int = 4,
) -> tuple[list, list, list, list]:
    n = len(X)
    _, bb = cpp_eval_sat_pen_bbox(X, Y, T)
    ov = hc.overlap_pairs(X, Y, T)
    hot = np.zeros(n, dtype=bool)
    for _, i, j in ov:
        hot[i] = hot[j] = True
    pts = [np.c_[X[hot], Y[hot]]] if hot.any() else []
    if len(centers):
        pts.append(centers)
    ctrs = np.vstack(pts) if pts else np.array([[0.5 * Lt, 0.5 * Lt]])

    L_ref = max(Lt, bb if bb < Lt + 0.012 else Lt)
    L0 = max(L_inc + 0.028, L_ref + 0.036)
    off = 0.5 * (L0 - L_ref)
    xb = X + off
    yb = Y + off
    h = hc.hx_of(T)
    xb = np.clip(xb, h + 1e-4, L0 - h - 1e-4)
    yb = np.clip(yb, h + 1e-4, L0 - h - 1e-4)

    L_goal_tight = min(Lt, L_inc - 2e-4)
    L_goal_soft = L_inc - 2e-4
    L_goal_mid = min(L_inc - 2e-4, max(Lt + 0.002, 0.5 * (Lt + L_inc))) if Lt < L_inc - 4e-4 else L_goal_soft
    sched_soft = [L0, L_inc + 0.008, L_inc + 0.0015, L_goal_soft]
    sched_mid = [L0, max(L_inc, Lt) + 0.006, max(L_goal_mid, Lt) + 0.001, L_goal_mid]
    sched_tight = [L0, max(L_inc, Lt) + 0.008, 0.5 * (L0 + L_goal_tight), L_goal_tight]

    q, m, ls, lab = [], [], [], []
    for s in range(n_seeds):
        if s == n_seeds - 1 and n_seeds >= 4:
            R = 100.0
        elif s == n_seeds - 2 and n_seeds >= 3:
            R = float(rng.uniform(9.5, 13.0))
        else:
            R = float(rng.uniform(5.8, 8.8))

        d2 = ((xb[:, None] - (ctrs[None, :, 0] + off)) ** 2 + (yb[:, None] - (ctrs[None, :, 1] + off)) ** 2).min(axis=1)
        near = np.ones(n, dtype=bool) if R > 50 else (d2 <= R * R)
        near |= (xb - h < off) | (xb + h > L0 - off) | (yb - h < off) | (yb + h > L0 - off)
        if near.sum() < 16:
            near[np.argsort(d2)[: min(24, n)]] = True

        x2, y2, t2 = xb.copy(), yb.copy(), T.copy()
        k = int(near.sum())
        jit = 0.0 if s == 0 else (0.045 if R > 50 else (0.06 if s % 2 == 1 else 0.11))
        if jit > 0:
            x2[near] += rng.uniform(-jit, jit, k)
            y2[near] += rng.uniform(-jit, jit, k)
            t2[near] += rng.uniform(-0.24, 0.24, k)
        q.append(full_to_free_sym(x2, y2, t2, L0))
        mm = near.astype(np.uint8)
        m.append(np.concatenate([mm, mm, mm]))
        sched = sched_soft if s % 2 == 0 else (sched_mid if s % 4 == 1 else sched_tight)
        ls.append(sched)
        lab.append(f"{label}_R{int(R) if R > 50 else f'{R:.1f}'}_s{s}")
    return q, m, ls, lab


def solve_target(
    n_target: int,
    seed_specs: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]],
    L_inc_dec: Decimal,
    threads: int = 4,
    rng: np.random.Generator | None = None,
    n_seeds_per_spec: int = 4,
    masked_iters: int = 200,
    full_iters: int = 230,
    top_polish: int = 8,
    run_mechwalk: bool = True,
) -> dict:
    if rng is None:
        rng = np.random.default_rng(20261140 + n_target)
    L_inc = float(L_inc_dec)
    t0 = time.time()

    best_cert: dict | None = None
    best_label: str | None = None
    best_seen_L: float = 1e18
    best_pack: tuple[np.ndarray, np.ndarray, np.ndarray, float] | None = None

    # 1. Fast direct SLP + lpslack on low-penetration seeds
    for sname, X, Y, T, Lt, _ in seed_specs:
        pen, _ = cpp_eval_sat_pen_bbox(X, Y, T)
        if pen < 1e-4:
            cert, pack = hc.polish_candidate(n_target, X, Y, T, L_inc_dec, use_lpslack=True)
            if cert is not None:
                if cert["L_float"] < best_seen_L:
                    best_seen_L = cert["L_float"]
                    best_pack = pack
                print(f"[{n_target}] direct-slp {sname}: L={cert['L_float']:.12f} dL={cert['dL']:+.6e}", flush=True)
                if cert["is_real_win"] and (best_cert is None or Decimal(cert["L_lit"]) < Decimal(best_cert["L_lit"])):
                    best_cert = cert
                    best_label = f"{sname}_direct"
                    hc.save_win_and_rebuild_final(n_target, best_cert, best_label)
                    print(
                        f"[{n_target}] *** REAL WIN (direct) *** {best_label} L={best_cert['L_lit'][:20]} dL={best_cert['dL']:+.6e}",
                        flush=True,
                    )

    # 2. C++ masked melt across all peel & surgery seeds
    allq, allm, alls, alll = [], [], [], []
    for sname, X, Y, T, Lt, ctrs in seed_specs:
        q, m, ls, lab = build_melt_jobs_for_seed(
            sname, X, Y, T, Lt, ctrs, L_inc, rng, n_seeds=n_seeds_per_spec
        )
        allq += q
        allm += m
        alls += ls
        alll += lab

    n_feas = 0
    best_cpp_Lf = 1e18
    polish_logs = []

    if allq:
        qb, qf, lf, eo, po, lb = run_cpp_masked_melt(
            n_target,
            1,
            allq,
            allm,
            alls,
            [6, 10, 14, 18],
            a_start=0.75,
            n_melt_stages=4,
            masked_iters=masked_iters,
            full_iters=full_iters,
            n_threads=min(4, threads),
            base_seed=int(rng.integers(10000, 999999)),
        )
        n_feas = int((lf < 1e9).sum())
        best_cpp_Lf = float(lf.min())
        print(
            f"[{n_target}] melt ({len(alll)} jobs, {time.time()-t0:.1f}s) feas={n_feas}/{len(lf)} "
            f"best_Lf={best_cpp_Lf:.9f} min_pen={float(po.min()):.2e}",
            flush=True,
        )

        ranked_all = sorted(range(len(alll)), key=lambda i: (0 if lf[i] < 1e9 else 1, lf[i] if lf[i] < 1e9 else po[i]))
        # Pick diverse best representatives across seed families so xcross/clamp don't crowd out reloc/BL/TR
        order: list[int] = []
        fam_counts: dict[str, int] = {}
        for i in ranked_all:
            lb_i = alll[i]
            fam = "xcross" if "xcross" in lb_i else (
                "ext" if "ext" in lb_i else (
                    "reloc" if "reloc" in lb_i else (
                        "seam" if "seamSurg" in lb_i else (
                            next((c for c in ("BL", "TL", "BR", "TR") if f"_{c}_" in lb_i), "other")
                        )
                    )
                )
            )
            if fam_counts.get(fam, 0) < 2:
                order.append(i)
                fam_counts[fam] = fam_counts.get(fam, 0) + 1
            if len(order) >= top_polish:
                break
        for i in ranked_all:
            if len(order) >= top_polish:
                break
            if i not in order:
                order.append(i)
        for idx_rank, i in enumerate(order):
            use_f = lf[i] < 1e9
            Lr = float(lf[i]) if use_f else alls[i][-1]
            x, y, th = cpp_expand_sym(n_target, 1, Lr, qf[i] if use_f else qb[i])
            cert, pack = hc.polish_candidate(n_target, x, y, th, L_inc_dec, use_lpslack=(idx_rank < 4))
            if cert is None:
                print(f"[{n_target}]   {alll[i]}: cppLf={lf[i]:.9f} pen={po[i]:.2e} -> polish=FAIL", flush=True)
                polish_logs.append({"label": alll[i], "cppLf": float(lf[i]), "pen": float(po[i]), "ok": False})
                continue
            if cert["L_float"] < best_seen_L:
                best_seen_L = cert["L_float"]
                best_pack = pack
            print(f"[{n_target}]   {alll[i]}: cppLf={lf[i]:.9f} -> L={cert['L_float']:.12f} dL={cert['dL']:+.6e}", flush=True)
            polish_logs.append({"label": alll[i], "cppLf": float(lf[i]), "L": cert["L_float"], "dL": cert["dL"], "ok": True})
            if cert["is_real_win"] and (best_cert is None or Decimal(cert["L_lit"]) < Decimal(best_cert["L_lit"])):
                best_cert = cert
                best_label = alll[i]
                hc.save_win_and_rebuild_final(n_target, best_cert, best_label)
                print(
                    f"[{n_target}] *** REAL WIN *** {best_label} L={best_cert['L_lit'][:20]} dL={best_cert['dL']:+.6e}",
                    flush=True,
                )

    # 3. Contact-Jacobian SVD mechanism walk + lpslack on best feasible basin
    if run_mechwalk and best_pack is not None and best_seen_L < L_inc + 0.045:
        bx, by, bt, bL = best_pack
        mw_rounds = 4 if best_seen_L < L_inc + 0.015 else 2
        print(f"[{n_target}] Running mechwalk_compress + lpslack (rounds={mw_rounds}) from best_seen_L={bL:.12f}...", flush=True)
        mw_cert, mw_pack = hc.mechwalk_compress(n_target, bx, by, bt, L_inc_dec, rng, rounds=mw_rounds, threads=min(4, threads))
        if mw_pack is not None and mw_pack[3] < best_seen_L:
            best_seen_L = mw_pack[3]
            best_pack = mw_pack
        if mw_cert is not None:
            print(f"[{n_target}]   mechwalk+lpslack: L={mw_cert['L_float']:.12f} dL={mw_cert['dL']:+.6e}", flush=True)
            if mw_cert["is_real_win"] and (best_cert is None or Decimal(mw_cert["L_lit"]) < Decimal(best_cert["L_lit"])):
                best_cert = mw_cert
                best_label = f"{best_label or 'basin'}_mechwalk_lpslack"
                hc.save_win_and_rebuild_final(n_target, best_cert, best_label)
                print(
                    f"[{n_target}] *** REAL WIN (mechwalk+lpslack) *** {best_label} L={best_cert['L_lit'][:20]} dL={best_cert['dL']:+.6e}",
                    flush=True,
                )

    return {
        "n": n_target,
        "L_inc": L_inc,
        "n_seeds": len(seed_specs),
        "n_jobs": len(alll),
        "n_feas": n_feas,
        "best_cpp_Lf": best_cpp_Lf if best_cpp_Lf < 1e17 else None,
        "best_seen_L": best_seen_L if best_seen_L < 1e17 else None,
        "win": best_cert is not None,
        "best_label": best_label,
        "L_win": best_cert["L_lit"] if best_cert else None,
        "dL": best_cert["dL"] if best_cert else (best_seen_L - L_inc if best_seen_L < 1e17 else None),
        "polish_logs": polish_logs,
        "best_pack": best_pack,
        "elapsed_s": round(time.time() - t0, 1),
    }
