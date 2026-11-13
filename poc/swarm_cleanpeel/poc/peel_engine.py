#!/usr/bin/env python3
"""H16 Clean-Peel seed generator and C++ masked-melt + joint SLP runner."""
from __future__ import annotations

import math
import time
from decimal import Decimal
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import cleanpeel_lib as cp  # noqa: E402
from cpp_wrap import cpp_eval_sat_pen_bbox, cpp_expand_sym, full_to_free_sym, run_cpp_masked_melt  # noqa: E402


def _corner_defs(L: float, k_cut: float = 1.02) -> list[tuple[str, callable]]:
    """Return 4 corner peel definitions: (name, mask_fn(x, y))."""
    return [
        ("BL", lambda x, y: (x <= L - k_cut) & (y <= L - k_cut), 0.0, 0.0),
        ("TL", lambda x, y: (x <= L - k_cut) & (y >= k_cut), 0.0, -1.0),
        ("BR", lambda x, y: (x >= k_cut) & (y <= L - k_cut), -1.0, 0.0),
        ("TR", lambda x, y: (x >= k_cut) & (y >= k_cut), -1.0, -1.0),
    ]


def _protrusion_amounts(x: np.ndarray, y: np.ndarray, h: np.ndarray, Lt: float) -> np.ndarray:
    return np.maximum.reduce([-(x - h), (x + h) - Lt, -(y - h), (y + h) - Lt])


def _seam_centers(x: np.ndarray, y: np.ndarray, th: np.ndarray, ref_pts: np.ndarray) -> np.ndarray:
    """Find tilted-upright seam squares nearest to ref_pts."""
    deg = np.degrees(th) % 90.0
    tilted = (deg > 1.0) & (deg < 89.0)
    if not np.any(tilted) or len(ref_pts) == 0:
        return ref_pts
    tx, ty = x[tilted], y[tilted]
    d2 = ((tx[:, None] - ref_pts[None, :, 0]) ** 2 + (ty[:, None] - ref_pts[None, :, 1]) ** 2).min(axis=1)
    best = np.argsort(d2)[:2]
    return np.vstack([ref_pts, np.c_[tx[best], ty[best]]])


def generate_cleanpeel_seeds(
    n_target: int,
    n_donor: int,
    xd: np.ndarray,
    yd: np.ndarray,
    td: np.ndarray,
    Ld: float,
    L_inc: float,
    rng: np.random.Generator,
    max_prot: int = 3,
    allow_tilted_border: bool = False,
) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]]:
    """
    Generate Method (a) Clamp+Repair and Method (b) Void-Relocation / DropMinProt_reloc
    seeds for n_donor -> n_target.
    """
    hd = cp.hx_of(td)
    tilt = cp.dominant_tilt(td)
    Lt = Ld - 1.0
    if Lt >= L_inc - 1e-6:
        return []

    seeds: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]] = []
    # Aim Lt_use slightly below L_inc so melt can stabilize without over-crushing
    Lt_use = min(L_inc - 4e-4, max(Lt, L_inc - 0.018))

    for cname, mask_fn, dx, dy in _corner_defs(Ld, k_cut=1.02):
        keep = mask_fn(xd, yd)
        rem = ~keep
        rem_deg = np.degrees(td[rem]) % 90.0
        rem_tilt = int(((rem_deg > 1.0) & (rem_deg < 89.0)).sum())
        if not allow_tilted_border and rem_tilt > 0:
            continue

        cnt = int(keep.sum())
        diff_n = cnt - n_target
        if diff_n < 0 or diff_n > 2:
            continue

        xk = xd[keep] + dx
        yk = yd[keep] + dy
        tk = td[keep].copy()
        hk = hd[keep]

        prot_amt = _protrusion_amounts(xk, yk, hk, Lt)
        p_idx = np.where(prot_amt > 1e-5)[0]
        if len(p_idx) > max_prot + diff_n:
            continue

        if diff_n == 0 and len(p_idx) == 0:
            ctrs = np.array([[0.5 * Lt, 0.5 * Lt]])
            seeds.append((f"peel_{n_donor}_{cname}_clean0", xk.copy(), yk.copy(), tk.copy(), Lt_use, ctrs))
            continue

        if diff_n == 0:
            orig_pts = np.c_[xk[p_idx], yk[p_idx]]

            # --- Method (a): Direct boundary clamp + repair_overlaps ---
            for ang_tag, th_override in (("keepAng", None), ("rot0", 0.0), ("rot45", np.pi / 4.0)):
                xc, yc, tc = xk.copy(), yk.copy(), tk.copy()
                if th_override is not None:
                    tc[p_idx] = th_override
                hc = cp.hx_of(tc)
                xc[p_idx] = np.clip(xc[p_idx], hc[p_idx] + 1e-4, Lt - hc[p_idx] - 1e-4)
                yc[p_idx] = np.clip(yc[p_idx], hc[p_idx] + 1e-4, Lt - hc[p_idx] - 1e-4)
                d2 = ((xc[:, None] - orig_pts[None, :, 0]) ** 2 + (yc[:, None] - orig_pts[None, :, 1]) ** 2).min(axis=1)
                local_free = d2 <= 4.8 ** 2
                L_rep = min(L_inc - 2e-4, max(Lt + 0.012, L_inc - 0.008))
                xr, yr, tr, pen_r = cp.repair_overlaps(xc, yc, tc, L_rep, free_mask=local_free, maxiter=240)
                ctrs = _seam_centers(xr, yr, tr, orig_pts)
                seeds.append((f"peel_{n_donor}_{cname}_clamp_{ang_tag}", xr, yr, tr, Lt_use, ctrs))

            # --- Method (b): Relocate protruding square(s) into top interior voids ---
            non_prot = np.ones(len(xk), dtype=bool)
            non_prot[p_idx] = False
            vp = cp.clearance_points(xk[non_prot], yk[non_prot], tk[non_prot], Lt, len(p_idx) + 6, step=0.10, min_sep=0.80)
            if len(vp) >= len(p_idx):
                ang_choices = [("u0", 0.0), ("tilt", tilt), ("d45", np.pi / 4.0)]
                for shift in range(min(3, len(vp) - len(p_idx) + 1)):
                    for atag, aval in ang_choices:
                        if shift > 0 and atag not in ("u0", "tilt"):
                            continue
                        xv, yv, tv = xk.copy(), yk.copy(), tk.copy()
                        vpts = []
                        for k_i, sq_i in enumerate(p_idx):
                            vx, vy, _ = vp[shift + k_i]
                            xv[sq_i], yv[sq_i], tv[sq_i] = vx, vy, aval
                            vpts.append([vx, vy])
                        ctrs = np.vstack([orig_pts, np.array(vpts)])
                        seeds.append((f"peel_{n_donor}_{cname}_reloc_v{shift}_{atag}", xv, yv, tv, Lt_use, ctrs))

        else:
            # diff_n > 0: dropProt and dropMinProt_reloc
            order_desc = p_idx[np.argsort(-prot_amt[p_idx])]
            order_asc = p_idx[np.argsort(prot_amt[p_idx])]
            drop_modes = [("dropProt", order_desc[:diff_n])]
            if len(p_idx) > diff_n:
                drop_modes.append(("dropMinProt", order_asc[:diff_n]))

            for dtag, d_idx in drop_modes:
                km = np.ones(len(xk), dtype=bool)
                km[d_idx] = False
                xr, yr, tr = xk[km].copy(), yk[km].copy(), tk[km].copy()
                hr = cp.hx_of(tr)
                p2 = np.where(_protrusion_amounts(xr, yr, hr, Lt) > 1e-5)[0]
                if len(p2) == 0:
                    ctrs = np.c_[xk[d_idx], yk[d_idx]]
                    seeds.append((f"peel_{n_donor}_{cname}_{dtag}_clean", xr, yr, tr, Lt_use, ctrs))
                    continue

                p2_pts = np.c_[xr[p2], yr[p2]]
                # Clamp remaining protruding squares
                xc, yc, tc = xr.copy(), yr.copy(), tr.copy()
                xc[p2] = np.clip(xc[p2], hr[p2] + 1e-4, Lt - hr[p2] - 1e-4)
                yc[p2] = np.clip(yc[p2], hr[p2] + 1e-4, Lt - hr[p2] - 1e-4)
                d2 = ((xc[:, None] - p2_pts[None, :, 0]) ** 2 + (yc[:, None] - p2_pts[None, :, 1]) ** 2).min(axis=1)
                xr2, yr2, tr2, _ = cp.repair_overlaps(xc, yc, tc, min(L_inc - 2e-4, Lt + 0.012), free_mask=(d2 <= 4.8 ** 2))
                seeds.append((f"peel_{n_donor}_{cname}_{dtag}_clamp", xr2, yr2, tr2, Lt_use, _seam_centers(xr2, yr2, tr2, p2_pts)))

                # Relocate remaining protruding squares into top voids (dropMinProt_reloc)
                non_p2 = np.ones(len(xr), dtype=bool)
                non_p2[p2] = False
                vp = cp.clearance_points(xr[non_p2], yr[non_p2], tr[non_p2], Lt, len(p2) + 5, step=0.10, min_sep=0.80)
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
                            seeds.append((f"peel_{n_donor}_{cname}_{dtag}_reloc_v{shift}_{atag}", xv, yv, tv, Lt_use, ctrs))

    # Deduplicate/prioritize seeds cleanly: clamp keepAng/rot0 first, then reloc_v0_u0/tilt, etc.
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
    n_seeds: int = 6,
    R_range: tuple[float, float] = (5.5, 8.2),
    include_wide: bool = True,
):
    n = len(X)
    pen, bb = cpp_eval_sat_pen_bbox(X, Y, T)
    ov = cp.overlap_pairs(X, Y, T)
    hot = np.zeros(n, dtype=bool)
    for _, i, j in ov:
        hot[i] = hot[j] = True
    pts = [np.c_[X[hot], Y[hot]]] if hot.any() else []
    if len(centers):
        pts.append(centers)
    ctrs = np.vstack(pts) if pts else np.array([[0.5 * Lt, 0.5 * Lt]])

    L_ref = max(Lt, bb if bb < Lt + 0.008 else Lt)
    L0 = max(L_inc + 0.025, L_ref + 0.035)
    off = 0.5 * (L0 - L_ref)
    xb = X + off
    yb = Y + off
    h = cp.hx_of(T)
    xb = np.clip(xb, h + 1e-4, L0 - h - 1e-4)
    yb = np.clip(yb, h + 1e-4, L0 - h - 1e-4)

    L_goal1 = min(Lt, L_inc - 2e-4)
    L_goal2 = min(L_inc - 2e-4, max(Lt + 0.002, 0.5 * (Lt + L_inc))) if Lt < L_inc - 4e-4 else L_inc - 2e-4
    L_mid = 0.5 * (L0 + L_goal1)
    sched1 = [L0, max(L_inc, Lt) + 0.008, L_mid, L_goal1]
    sched2 = [L0, max(L_inc, Lt) + 0.005, max(L_goal2, Lt) + 0.001, L_goal2]

    q, m, ls, lab = [], [], [], []
    for s in range(n_seeds):
        if include_wide and n_seeds >= 4 and s == n_seeds - 1:
            # Wide pocket R in [9.0, 11.5]
            R = float(rng.uniform(9.0, 11.5))
        else:
            R = float(rng.uniform(R_range[0], R_range[1]))
        d2 = ((xb[:, None] - (ctrs[None, :, 0] + off)) ** 2 + (yb[:, None] - (ctrs[None, :, 1] + off)) ** 2).min(axis=1)
        near = d2 <= R * R
        near |= (xb - h < off) | (xb + h > L0 - off) | (yb - h < off) | (yb + h > L0 - off)
        if near.sum() < 16:
            near[np.argsort(d2)[: min(24, n)]] = True
        x2, y2, t2 = xb.copy(), yb.copy(), T.copy()
        k = int(near.sum())
        jit = 0.0 if s == 0 else (0.05 if s % 2 == 1 else 0.11)
        if jit > 0:
            x2[near] += rng.uniform(-jit, jit, k)
            y2[near] += rng.uniform(-jit, jit, k)
            t2[near] += rng.uniform(-0.24, 0.24, k)
        q.append(full_to_free_sym(x2, y2, t2, L0))
        mm = near.astype(np.uint8)
        m.append(np.concatenate([mm, mm, mm]))
        ls.append(sched1 if s % 2 == 0 else sched2)
        lab.append(f"{label}_R{R:.1f}_s{s}")
    return q, m, ls, lab


def solve_cleanpeel_target(
    n_target: int,
    seed_specs: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]],
    L_inc_dec: Decimal,
    threads: int = 4,
    rng: np.random.Generator | None = None,
    n_seeds_per_spec: int = 6,
    masked_iters: int = 200,
    full_iters: int = 220,
    top_polish: int = 6,
) -> dict:
    if rng is None:
        rng = np.random.default_rng(20261113 + n_target)
    L_inc = float(L_inc_dec)
    t0 = time.time()

    best_cert: dict | None = None
    best_label: str | None = None
    best_seen_L: float = 1e18

    # Fast check: raw / repaired seeds via direct joint_xytheta_slp if pen is small
    for sname, X, Y, T, Lt, _ in seed_specs:
        pen, _ = cpp_eval_sat_pen_bbox(X, Y, T)
        if pen < 1e-4:
            cert = cp.polish_candidate(n_target, X, Y, T, L_inc_dec)
            if cert is not None:
                best_seen_L = min(best_seen_L, cert["L_float"])
                print(f"[{n_target}] direct-slp {sname}: L={cert['L_float']:.12f} dL={cert['dL']:+.6e}", flush=True)
                if cert["is_real_win"] and (best_cert is None or Decimal(cert["L_lit"]) < Decimal(best_cert["L_lit"])):
                    best_cert = cert
                    best_label = f"{sname}_direct"
                    cp.save_win_and_rebuild_final(n_target, best_cert, best_label)
                    print(f"[{n_target}] *** REAL WIN (direct-slp) *** {best_label} L={best_cert['L_lit'][:20]} dL={best_cert['dL']:+.6e}", flush=True)

    allq, allm, alls, alll = [], [], [], []
    for sname, X, Y, T, Lt, ctrs in seed_specs:
        q, m, ls, lab = build_melt_jobs_for_seed(
            sname, X, Y, T, Lt, ctrs, L_inc, rng, n_seeds=n_seeds_per_spec
        )
        allq += q
        allm += m
        alls += ls
        alll += lab

    if not allq:
        return {"n": n_target, "status": "no_seeds", "best_cert": best_cert, "best_label": best_label}

    qb, qf, lf, eo, po, lb = run_cpp_masked_melt(
        n_target,
        1,
        allq,
        allm,
        alls,
        [6, 10, 14, 18],
        a_start=0.80,
        n_melt_stages=4,
        masked_iters=masked_iters,
        full_iters=full_iters,
        n_threads=min(4, threads),
        base_seed=int(rng.integers(10000, 999999)),
    )
    n_feas = int((lf < 1e9).sum())
    print(
        f"[{n_target}] melt ({len(alll)} jobs, {time.time()-t0:.1f}s) feas={n_feas}/{len(lf)} best_Lf={float(lf.min()):.9f} min_pen={float(po.min()):.2e}",
        flush=True,
    )

    order = sorted(range(len(alll)), key=lambda i: (0 if lf[i] < 1e9 else 1, lf[i] if lf[i] < 1e9 else po[i]))[:top_polish]
    polish_logs = []
    for i in order:
        use_f = lf[i] < 1e9
        Lr = float(lf[i]) if use_f else alls[i][-1]
        x, y, th = cpp_expand_sym(n_target, 1, Lr, qf[i] if use_f else qb[i])
        cert = cp.polish_candidate(n_target, x, y, th, L_inc_dec)
        if cert is None:
            print(f"[{n_target}]   {alll[i]}: cppLf={lf[i]:.9f} pen={po[i]:.2e} -> polish=FAIL", flush=True)
            polish_logs.append({"label": alll[i], "cppLf": float(lf[i]), "pen": float(po[i]), "ok": False})
            continue
        best_seen_L = min(best_seen_L, cert["L_float"])
        print(f"[{n_target}]   {alll[i]}: cppLf={lf[i]:.9f} -> L={cert['L_float']:.12f} dL={cert['dL']:+.6e}", flush=True)
        polish_logs.append({"label": alll[i], "cppLf": float(lf[i]), "L": cert["L_float"], "dL": cert["dL"], "ok": True})
        if cert["is_real_win"] and (best_cert is None or Decimal(cert["L_lit"]) < Decimal(best_cert["L_lit"])):
            best_cert = cert
            best_label = alll[i]
            cp.save_win_and_rebuild_final(n_target, best_cert, best_label)
            print(f"[{n_target}] *** REAL WIN *** {best_label} L={best_cert['L_lit'][:20]} dL={best_cert['dL']:+.6e}", flush=True)

    return {
        "n": n_target,
        "L_inc": float(L_inc_dec),
        "n_seeds": len(seed_specs),
        "n_jobs": len(alll),
        "n_feas": n_feas,
        "best_cpp_Lf": float(lf.min()),
        "best_seen_L": best_seen_L if best_seen_L < 1e17 else None,
        "win": best_cert is not None,
        "best_label": best_label,
        "L_win": best_cert["L_lit"] if best_cert else None,
        "dL": best_cert["dL"] if best_cert else (best_seen_L - L_inc if best_seen_L < 1e17 else None),
        "polish_logs": polish_logs,
        "elapsed_s": round(time.time() - t0, 1),
    }
