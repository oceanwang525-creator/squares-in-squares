#!/usr/bin/env python3
"""H98 Upward Constant-r Cascade engine: Zero-overlap L-border + xcross_expand + global inflation + masked melt + mechwalk + lpslack."""
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

import upward_lib as ul  # noqa: E402
from cpp_wrap import (  # noqa: E402
    cpp_eval_sat_pen_bbox,
    cpp_expand_sym,
    full_to_free_sym,
    run_cpp_batch_ils,
    run_cpp_masked_melt,
)


def _line_positions(count: int, lo: float, hi: float) -> np.ndarray:
    if count <= 0:
        return np.empty(0, dtype=np.float64)
    if count == 1:
        return np.array([0.5 * (lo + hi)], dtype=np.float64)
    return np.linspace(lo + 0.5, hi - 0.5, count, dtype=np.float64)


def generate_lborder_seeds(
    n_target: int,
    n_donor: int,
    xd: np.ndarray,
    yd: np.ndarray,
    td: np.ndarray,
    Ld: float,
) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]]:
    """
    Generate all 4 zero-overlap outer L-border extensions (TR, TL, BR, BL) in both packed and spread layouts.
    Always has pen0 = 0.0 at L0 = Ld + 1.0 + 1e-6.
    """
    m = math.isqrt(n_target)
    assert m * m + (n_target - m * m) == n_target
    assert len(xd) == n_donor
    L0 = Ld + 1.0 + 1e-6
    seeds: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]] = []

    for mode, coords_m1 in [
        ("pack", np.arange(0.5, m - 1, 1.0, dtype=np.float64)),
        ("spread", np.linspace(0.5, Ld - 0.5, m - 1, dtype=np.float64) if m > 2 else np.array([0.5])),
    ]:
        for cname, sx, sy, wall_x, wall_y, off_x, off_y in [
            ("extTR", 0.0, 0.0, Ld + 0.5 + 1e-6, Ld + 0.5 + 1e-6, 0.0, 0.0),
            ("extTL", 1.0 + 1e-6, 0.0, 0.5, Ld + 0.5 + 1e-6, 1.0 + 1e-6, 0.0),
            ("extBR", 0.0, 1.0 + 1e-6, Ld + 0.5 + 1e-6, 0.5, 0.0, 1.0 + 1e-6),
            ("extBL", 1.0 + 1e-6, 1.0 + 1e-6, 0.5, 0.5, 1.0 + 1e-6, 1.0 + 1e-6),
        ]:
            cx = np.concatenate([xd + sx, np.full(m - 1, wall_x), coords_m1 + off_x, [wall_x]])
            cy = np.concatenate([yd + sy, coords_m1 + off_y, np.full(m - 1, wall_y), [wall_y]])
            ct = np.concatenate([td, np.zeros(2 * m - 1, dtype=np.float64)])
            nx, ny, nt, nL = ul.norm_pack(cx, cy, ct)
            ctrs = np.array([[wall_x, 0.5 * L0], [0.5 * L0, wall_y], [wall_x, wall_y]], dtype=np.float64)
            seeds.append((f"{cname}_{mode}_{n_donor}", nx, ny, nt, nL, ctrs))

    return seeds


def _build_cross_seed(
    xd: np.ndarray,
    yd: np.ndarray,
    td: np.ndarray,
    Ld: float,
    m: int,
    x0: float,
    y0: float,
    tag: str,
) -> tuple[str, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    """Split donor at (x0, y0), shift right/top by +1.0, insert 2m-1 upright squares into cross corridor."""
    eps = 1e-4
    right = xd >= x0
    top = yd >= y0
    xb = xd + np.where(right, 1.0 + eps, 0.0)
    yb = yd + np.where(top, 1.0 + eps, 0.0)

    # Number of vertical-strip squares below y0 and above y0 + 1
    if y0 <= 0.5:
        ky_lo = 0
    elif y0 >= Ld - 0.5:
        ky_lo = m - 1
    else:
        ky_lo = max(1, min(m - 2, int(round((y0 / Ld) * (m - 1)))))
    ky_hi = (m - 1) - ky_lo

    if x0 <= 0.5:
        kx_lo = 0
    elif x0 >= Ld - 0.5:
        kx_lo = m - 1
    else:
        kx_lo = max(1, min(m - 2, int(round((x0 / Ld) * (m - 1)))))
    kx_hi = (m - 1) - kx_lo

    vx_pos = x0 + 0.5
    hy_pos = y0 + 0.5

    vy_lo = _line_positions(ky_lo, 0.0, max(float(ky_lo), y0))
    vy_hi = _line_positions(ky_hi, min(y0 + 1.0 + eps, Ld + 1.0 - float(ky_hi)), Ld + 1.0 + eps)
    vx_all = np.full(m - 1, vx_pos, dtype=np.float64)
    vy_all = np.concatenate([vy_lo, vy_hi])

    hx_lo = _line_positions(kx_lo, 0.0, max(float(kx_lo), x0))
    hx_hi = _line_positions(kx_hi, min(x0 + 1.0 + eps, Ld + 1.0 - float(kx_hi)), Ld + 1.0 + eps)
    hx_all = np.concatenate([hx_lo, hx_hi])
    hy_all = np.full(m - 1, hy_pos, dtype=np.float64)

    cx = np.concatenate([xb, vx_all, hx_all, [vx_pos]])
    cy = np.concatenate([yb, vy_all, hy_all, [hy_pos]])
    ct = np.concatenate([td, np.zeros(2 * m - 1, dtype=np.float64)])
    nx, ny, nt, nL = ul.norm_pack(cx, cy, ct)

    # Local repair around cross corridor
    d_corr = np.minimum(np.abs(nx - vx_pos), np.abs(ny - hy_pos))
    xr, yr, tr, _ = ul.repair_overlaps(nx, ny, nt, max(Ld + 1.01, nL + 0.004), free_mask=(d_corr <= 3.0), maxiter=160)
    xr, yr, tr, nL = ul.norm_pack(xr, yr, tr)
    ctrs = np.array([[vx_pos, hy_pos], [vx_pos, 0.5 * nL], [0.5 * nL, hy_pos]], dtype=np.float64)
    return tag, xr, yr, tr, nL, ctrs


def generate_xcross_expand_seeds(
    n_target: int,
    n_donor: int,
    xd: np.ndarray,
    yd: np.ndarray,
    td: np.ndarray,
    Ld: float,
) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]]:
    """
    Generate interior xcross_expand and T-split seeds by finding cleanest grid cuts
    plus tilted-core bisector cuts across the (m-1, r) donor.
    """
    m = math.isqrt(n_target)
    h = ul.hx_of(td)
    deg = np.degrees(td) % 90.0
    tilted = (deg > 1.0) & (deg < 89.0)

    # Score candidate interior cut lines
    xs_grid = np.linspace(1.5, Ld - 1.5, max(5, int((Ld - 3.0) * 2) + 1))
    ys_grid = np.linspace(1.5, Ld - 1.5, max(5, int((Ld - 3.0) * 2) + 1))

    x_cands = []
    for xc in xs_grid:
        cut = (xd - h < xc) & (xd + h > xc)
        ct = int((cut & tilted).sum())
        ov = float(np.maximum(0.0, np.minimum(xc - (xd - h), (xd + h) - xc)).sum())
        x_cands.append((ct, ov, float(xc)))
    y_cands = []
    for yc in ys_grid:
        cut = (yd - h < yc) & (yd + h > yc)
        ct = int((cut & tilted).sum())
        ov = float(np.maximum(0.0, np.minimum(yc - (yd - h), (yd + h) - yc)).sum())
        y_cands.append((ct, ov, float(yc)))

    x_cands.sort()
    y_cands.sort()

    split_pairs: list[tuple[str, float, float]] = []
    # Top 2 cleanest interior crosses
    for i in range(min(2, len(x_cands))):
        for j in range(min(2, len(y_cands))):
            if i + j <= 1:
                xc, yc = x_cands[i][2], y_cands[j][2]
                split_pairs.append((f"xcross_clean_{i}{j}_x{xc:.1f}_y{yc:.1f}", xc, yc))

    # Tilted-core bisectors (split right through tilted brick so brick can flatten/expand)
    if tilted.any():
        tx0, tx1 = float((xd[tilted] - h[tilted]).min()), float((xd[tilted] + h[tilted]).max())
        ty0, ty1 = float((yd[tilted] - h[tilted]).min()), float((yd[tilted] + h[tilted]).max())
        x_mid = 0.5 * (tx0 + tx1)
        y_mid = 0.5 * (ty0 + ty1)
        split_pairs.append((f"xcross_tmid_x{x_mid:.1f}_y{y_mid:.1f}", x_mid, y_mid))
        # T-splits: 1 interior cut + 1 outer border
        split_pairs.append((f"tsplit_x{x_mid:.1f}_top", x_mid, Ld))
        split_pairs.append((f"tsplit_right_y{y_mid:.1f}", Ld, y_mid))
    if x_cands and y_cands:
        split_pairs.append((f"tsplit_clean_x{x_cands[0][2]:.1f}_top", x_cands[0][2], Ld))
        split_pairs.append((f"tsplit_clean_right_y{y_cands[0][2]:.1f}", Ld, y_cands[0][2]))

    seeds = []
    seen = set()
    for tag, xc, yc in split_pairs:
        key = (round(xc, 2), round(yc, 2))
        if key in seen:
            continue
        seen.add(key)
        seeds.append(_build_cross_seed(xd, yd, td, Ld, m, xc, yc, f"{tag}_{n_donor}"))
    return seeds


def _walldrift_perturb(
    x: np.ndarray,
    y: np.ndarray,
    th: np.ndarray,
    L0: float,
    ctrs: np.ndarray,
    rng: np.random.Generator,
    strength: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Push squares slightly away from heavy bounding walls toward interior seam/void centers + thermal jitter."""
    n = len(x)
    h = ul.hx_of(th)
    x2, y2, t2 = x.copy(), y.copy(), th.copy()
    # Bias toward centroid of seam centers or box center
    tx = float(np.mean(ctrs[:, 0])) if len(ctrs) else 0.5 * L0
    ty = float(np.mean(ctrs[:, 1])) if len(ctrs) else 0.5 * L0
    dx = tx - x2
    dy = ty - y2
    norm = np.maximum(1.0, np.hypot(dx, dy))
    x2 += strength * (dx / norm) + rng.uniform(-0.08, 0.08, n)
    y2 += strength * (dy / norm) + rng.uniform(-0.08, 0.08, n)
    t2 += rng.uniform(-0.22, 0.22, n)
    x2 = np.clip(x2, h + 1e-4, L0 - h - 1e-4)
    y2 = np.clip(y2, h + 1e-4, L0 - h - 1e-4)
    return x2, y2, t2


def solve_upward_target(
    n_target: int,
    seed_specs: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]],
    L_inc_dec: Decimal,
    threads: int = 2,
    rng: np.random.Generator | None = None,
    top_polish: int = 8,
    run_mechwalk: bool = True,
) -> dict:
    if rng is None:
        rng = np.random.default_rng(20269800 + n_target)
    n_threads = min(2, max(1, int(threads)))
    L_inc = float(L_inc_dec)
    t0 = time.time()

    best_cert: dict | None = None
    best_label: str | None = None
    best_seen_L: float = 1e18
    best_pack: tuple[np.ndarray, np.ndarray, np.ndarray, float] | None = None

    # Phase A: Direct SLP + lpslack on low-penetration L-border and xcross_expand seeds
    lpslack_budget = 2
    for sname, X, Y, T, nL, _ in seed_specs:
        pen, _ = cpp_eval_sat_pen_bbox(X, Y, T)
        if pen < 0.02:
            do_lp = (lpslack_budget > 0) and ("extTR_pack" in sname or "extBL_pack" in sname or "xcross" in sname or "tsplit" in sname)
            if do_lp:
                lpslack_budget -= 1
            cert, pack = ul.polish_candidate(n_target, X, Y, T, L_inc_dec, use_lpslack=do_lp)
            if cert is not None and pack is not None:
                if cert["L_float"] < best_seen_L:
                    best_seen_L = cert["L_float"]
                    best_pack = pack
                print(f"[{n_target}] direct {sname}: L={cert['L_float']:.9f} dL={cert['dL']:+.6e}", flush=True)
                if cert["is_real_win"] and (best_cert is None or Decimal(cert["L_lit"]) < Decimal(best_cert["L_lit"])):
                    best_cert = cert
                    best_label = f"{sname}_direct"
                    ul.save_win_and_rebuild_final(n_target, best_cert, best_label)

    # Phase B: Global inflation + walldrift shear across gentle (0.78) and deep (0.62) melts
    cand_pools: list[tuple[str, float, float, np.ndarray, np.ndarray, np.ndarray]] = []
    selected = [s for s in seed_specs if "pack" in s[0] or "xcross" in s[0] or "tsplit" in s[0]][:8]

    for a_batch, configs in [
        (0.78, [(0.02, -0.002), (0.06, -0.006)]),
        (0.62, [(0.04, -0.005), (0.10, -0.012)]),
    ]:
        q_glob, ls_glob, lab_glob = [], [], []
        for sname, X, Y, T, nL, ctrs in selected:
            L0 = max(L_inc + 0.030, nL + 0.028)
            off = 0.5 * (L0 - nL)
            h = ul.hx_of(T)
            xb = np.clip(X + off, h + 1e-4, L0 - h - 1e-4)
            yb = np.clip(Y + off, h + 1e-4, L0 - h - 1e-4)
            for idx_w, (w_bias, goal_dL) in enumerate(configs):
                xw, yw, tw = _walldrift_perturb(xb, yb, T, L0, ctrs, rng, strength=w_bias)
                q_glob.append(full_to_free_sym(xw, yw, tw, L0))
                ls_glob.append([L0, L_inc + 0.007, L_inc + 0.001, L_inc + goal_dL])
                lab_glob.append(f"glob_{sname}_a{a_batch:.2f}_w{idx_w}")
        if q_glob:
            qb, qf, lf, eo, po, lb = run_cpp_batch_ils(
                n_target,
                1,
                q_glob,
                ls_glob,
                [6, 10, 14, 18],
                a_start=a_batch,
                n_melt_stages=4,
                n_threads=n_threads,
                base_seed=int(rng.integers(10000, 999999)),
            )
            print(
                f"[{n_target}] global-inflation a={a_batch:.2f} ({len(q_glob)} jobs, {time.time()-t0:.1f}s) "
                f"feas={int((lf<1e9).sum())}/{len(lf)} best_Lf={float(lf.min()):.9f} min_pen={float(po.min()):.2e}",
                flush=True,
            )
            for i in range(len(lab_glob)):
                use_f = lf[i] < 1e9
                Lr = float(lf[i]) if use_f else ls_glob[i][-1]
                xp, yp, tp = cpp_expand_sym(n_target, 1, Lr, qf[i] if use_f else qb[i])
                cand_pools.append((lab_glob[i], float(lf[i]), float(po[i]), xp, yp, tp))

    # Phase C: C++ Masked Melt around cross corridor & L-border seam
    q_msk, m_msk, ls_msk, lab_msk = [], [], [], []
    for sname, X, Y, T, nL, ctrs in seed_specs[:10]:
        L0 = max(L_inc + 0.028, nL + 0.026)
        off = 0.5 * (L0 - nL)
        h = ul.hx_of(T)
        xb = np.clip(X + off, h + 1e-4, L0 - h - 1e-4)
        yb = np.clip(Y + off, h + 1e-4, L0 - h - 1e-4)
        d2 = ((xb[:, None] - (ctrs[None, :, 0] + off)) ** 2 + (yb[:, None] - (ctrs[None, :, 1] + off)) ** 2).min(axis=1)
        for R, jit in [(8.0, 0.08), (13.0, 0.12)]:
            near = d2 <= R * R
            if near.sum() < 24:
                near[np.argsort(d2)[: min(32, n_target)]] = True
            x2, y2, t2 = xb.copy(), yb.copy(), T.copy()
            k = int(near.sum())
            x2[near] += rng.uniform(-jit, jit, k)
            y2[near] += rng.uniform(-jit, jit, k)
            t2[near] += rng.uniform(-0.25, 0.25, k)
            x2 = np.clip(x2, h + 1e-4, L0 - h - 1e-4)
            y2 = np.clip(y2, h + 1e-4, L0 - h - 1e-4)
            q_msk.append(full_to_free_sym(x2, y2, t2, L0))
            mm = near.astype(np.uint8)
            m_msk.append(np.concatenate([mm, mm, mm]))
            ls_msk.append([L0, L_inc + 0.006, L_inc + 0.001, L_inc - 0.003])
            lab_msk.append(f"msk_{sname}_R{int(R)}")

    if q_msk:
        qb, qf, lf, eo, po, lb = run_cpp_masked_melt(
            n_target,
            1,
            q_msk,
            m_msk,
            ls_msk,
            [6, 10, 14, 18],
            a_start=0.72,
            n_melt_stages=4,
            masked_iters=180,
            full_iters=220,
            n_threads=n_threads,
            base_seed=int(rng.integers(10000, 999999)),
        )
        print(
            f"[{n_target}] masked-melt ({len(q_msk)} jobs, {time.time()-t0:.1f}s) "
            f"feas={int((lf<1e9).sum())}/{len(lf)} best_Lf={float(lf.min()):.9f} min_pen={float(po.min()):.2e}",
            flush=True,
        )
        for i in range(len(lab_msk)):
            use_f = lf[i] < 1e9
            Lr = float(lf[i]) if use_f else ls_msk[i][-1]
            xp, yp, tp = cpp_expand_sym(n_target, 1, Lr, qf[i] if use_f else qb[i])
            cand_pools.append((lab_msk[i], float(lf[i]), float(po[i]), xp, yp, tp))

    # Phase D: Polish top candidates via joint_xytheta_slp + fixed LP + lpslack
    cand_pools.sort(key=lambda c: (0 if c[1] < 1e9 else 1, c[1] if c[1] < 1e9 else c[2]))
    for idx_rank, (lbl, Lf_val, pen_val, xp, yp, tp) in enumerate(cand_pools[:top_polish]):
        cert, pack = ul.polish_candidate(n_target, xp, yp, tp, L_inc_dec, use_lpslack=(idx_rank < 3))
        if cert is not None and pack is not None:
            if cert["L_float"] < best_seen_L:
                best_seen_L = cert["L_float"]
                best_pack = pack
            print(
                f"[{n_target}] polish[{idx_rank}] {lbl}: L={cert['L_float']:.9f} dL={cert['dL']:+.6e}",
                flush=True,
            )
            if cert["is_real_win"] and (best_cert is None or Decimal(cert["L_lit"]) < Decimal(best_cert["L_lit"])):
                best_cert = cert
                best_label = lbl
                ul.save_win_and_rebuild_final(n_target, best_cert, best_label)
                print(
                    f"[{n_target}] *** REAL WIN *** {best_label} L={best_cert['L_lit'][:18]} dL={best_cert['dL']:+.6e}",
                    flush=True,
                )

    # Phase E: mechwalk + final lpslack on best basin
    if run_mechwalk and best_pack is not None:
        mw_cert, mw_pack = ul.mechwalk_compress(
            n_target, best_pack[0], best_pack[1], best_pack[2], L_inc_dec, rng, rounds=2, threads=n_threads
        )
        if mw_cert is not None and mw_pack is not None:
            if mw_cert["L_float"] < best_seen_L:
                best_seen_L = mw_cert["L_float"]
                best_pack = mw_pack
            print(
                f"[{n_target}] mechwalk+lpslack: L={mw_cert['L_float']:.9f} dL={mw_cert['dL']:+.6e}",
                flush=True,
            )
            if mw_cert["is_real_win"] and (best_cert is None or Decimal(mw_cert["L_lit"]) < Decimal(best_cert["L_lit"])):
                best_cert = mw_cert
                best_label = (best_label or "upward") + "_mechwalk"
                ul.save_win_and_rebuild_final(n_target, best_cert, best_label)

    return {
        "n": n_target,
        "L_inc": float(L_inc_dec),
        "L_inc_str": str(L_inc_dec)[:24],
        "win": best_cert is not None,
        "best_label": best_label,
        "best_seen_L": best_seen_L if best_seen_L < 1e17 else None,
        "dL": best_cert["dL"] if best_cert is not None else (best_seen_L - L_inc if best_seen_L < 1e17 else None),
        "elapsed_s": round(time.time() - t0, 1),
    }
