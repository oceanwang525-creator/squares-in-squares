#!/usr/bin/env python3
"""Evaluation, C++ seam melt, lpslack + mechwalk polish, and fail-closed certification for H101."""
from __future__ import annotations

import json
import math
import os
import time
from decimal import Decimal
from pathlib import Path

import numpy as np

import interlock2_lib as il
import brick_slp
import lpslack
import mechwalk_lib as ml
import union as auto_union
import utils as hu
from cpp_wrap import cpp_eval_sat_pen_bbox, cpp_expand_sym, full_to_free_sym, run_cpp_masked_melt


def save_win(n: int, cert: dict, label: str) -> bool:
    """Save strict win (dL <= -1e-9 in both literal and pandas) and rebuild final.csv."""
    if not cert.get("is_real_win"):
        return False
    il.DATA_DIR.mkdir(parents=True, exist_ok=True)
    wp = il.DATA_DIR / f"win_n{n}.json"
    new_max = max(Decimal(cert["L_lit"]), Decimal(cert["L_pd"]))
    if wp.exists():
        prev = json.loads(wp.read_text())
        if max(Decimal(prev["L_lit"]), Decimal(prev["L_pd"])) <= new_max:
            return False
    payload = {
        "n": int(n),
        "label": label,
        "L_lit": cert["L_lit"],
        "L_pd": cert["L_pd"],
        "L_inc": cert["L_inc"],
        "dL_lit": cert["dL_lit"],
        "dL_pd": cert["dL_pd"],
        "rows": cert["rows"],
    }
    auto_union.atomic(wp, json.dumps(payload, indent=2).encode())
    os.chmod(wp, 0o644)

    groups = {}
    for p in sorted(il.DATA_DIR.glob("win_n*.json")):
        d = json.loads(p.read_text())
        groups[int(d["n"])] = d["rows"]
    text = auto_union.final_text(groups)
    auto_union.atomic(il.FINAL_CSV, text.encode())
    os.chmod(il.FINAL_CSV, 0o644)
    return True


def mechwalk_polish_once(
    n: int,
    base_rows: list[dict[str, str]],
    L_inc_dec: Decimal,
    n_threads: int = 2,
) -> dict | None:
    """Run 1 soft-mode mechwalk + C++ rejam + SLP + lpslack pass on base_rows."""
    x0 = np.array([float(r["x"]) for r in base_rows], dtype=np.float64)
    y0 = np.array([float(r["y"]) for r in base_rows], dtype=np.float64)
    t0 = np.array([float(r["angle"]) for r in base_rows], dtype=np.float64)
    x0, y0, t0, L_cur = ml.normalize(x0, y0, t0)

    all_modes = []
    for tol in (2e-5, 2e-4, 1e-3):
        jac, _ = ml.contact_jacobian(x0, y0, t0, contact_tol=tol)
        sv, raw_modes = ml.soft_modes(jac, k=10)
        sel = ml.select_diverse_modes(raw_modes, sv, x0, y0, max_keep=3)
        for _, sval, mode_vec, n_eff in sel:
            if any(abs(float(np.dot(mode_vec, prev[1]))) > 0.90 for prev in all_modes):
                continue
            disp, _ = ml.mode_participation(mode_vec, n)
            all_modes.append((sval, mode_vec, disp))
    if not all_modes:
        return None
    all_modes = all_modes[:6]

    q_list, m_list, l_list = [], [], []
    ones = np.ones(3 * n, dtype=np.uint8)
    L_ref = min(L_cur, float(L_inc_dec))
    for _, mvec, disp in all_modes:
        md = float(np.max(disp))
        if md < 1e-8:
            continue
        for step in (0.015, 0.05):
            amp = min(step / md, 1.5)
            for sgn in (1.0, -1.0):
                wx, wy, wth, _ = ml.normalize(
                    x0 + sgn * amp * mvec[:n],
                    y0 + sgn * amp * mvec[n : 2 * n],
                    t0 + sgn * amp * mvec[2 * n :],
                )
                sched = [L_ref + 0.008, L_ref + 0.001, L_ref - 3e-4]
                q_list.append(full_to_free_sym(wx, wy, wth, sched[0]))
                m_list.append(ones)
                l_list.append(sched)
    if not q_list:
        return None

    qb, qf, lf, _, po, _ = run_cpp_masked_melt(
        n,
        1,
        q_list,
        m_list,
        l_list,
        [0, 0, 0],
        a_start=0.94,
        n_melt_stages=3,
        masked_iters=100,
        full_iters=220,
        n_threads=n_threads,
        base_seed=91000 + n,
    )
    order = sorted(
        range(len(q_list)),
        key=lambda i: (0 if lf[i] < 1e9 else 1, float(lf[i]) if lf[i] < 1e9 else float(po[i])),
    )[:3]

    best_cert = None
    best_dec = None
    for i in order:
        use_f = lf[i] < 1e9
        L_r = float(lf[i]) if use_f else l_list[i][-1]
        x, y, th = cpp_expand_sym(n, 1, L_r, qf[i] if use_f else qb[i])
        pen, bb = cpp_eval_sat_pen_bbox(x, y, th)
        if 1e-7 < pen < 0.03:
            x, y, th, _ = hu.relax_lbfgs(x, y, th, bb + max(0.003, 2.0 * pen), maxiter=200, w_wall=4.0, w_pair=1.0)
            pen, bb = cpp_eval_sat_pen_bbox(x, y, th)
        if pen > 0.02:
            continue
        xs, ys, ts, _, ok = brick_slp.joint_xytheta_slp_tight(x, y, th)
        if not ok:
            continue
        rows = lpslack.rows_from_arrays(
            np.column_stack([xs, ys]),
            ts,
        )
        res = il.polish_rows_lpslack_mechwalk(n, rows, L_inc_dec)
        if res.get("ok"):
            cd = max(Decimal(res["L_lit"]), Decimal(res["L_pd"]))
            if best_dec is None or cd < best_dec:
                best_dec = cd
                best_cert = res
    return best_cert


def eval_zero_gap_candidate(
    cand: dict,
    merged_rows: dict[int, list[dict[str, str]]],
    best_rows: dict[int, list[dict[str, str]]],
    L_inc_dec: Decimal,
    syms: tuple[tuple[int, int], ...] = ((0, 0), (3, 0)),
    try_mechwalk: bool = True,
    n_threads: int = 1,
) -> dict:
    """Evaluate a Regime 1 (zero_gap) 2-D overhang candidate across symmetries."""
    t0 = time.time()
    n, K = cand["n"], cand["K"]
    p1, p2 = cand["p1"], cand["p2"]
    ext1 = il.get_item_squares_50d(p1, merged_rows, best_rows)
    ext2 = il.get_item_squares_50d(p2, merged_rows, best_rows)
    if ext1 is None or ext2 is None:
        return {"n": n, "status": "EXTRACT_FAIL", "dt": round(time.time() - t0, 1)}
    sq1, W1, H1 = ext1
    sq2, W2, H2 = ext2

    best_res = None
    best_dec = None
    best_label = ""
    for s1, s2 in syms:
        q1 = il.sym_squares_50d(sq1, W1, H1, s1)
        q2 = il.sym_squares_50d(sq2, W2, H2, s2)
        rows0 = il.assemble_zero_gap_50d(
            K, n, q1, W1, H1, p1["w"], p1["h"], q2, W2, H2, p2["w"], p2["h"]
        )
        if rows0 is None:
            continue
        res = il.polish_rows_lpslack_mechwalk(n, rows0, L_inc_dec)
        lbl = f"zg_{p1['uid']}+{p2['uid']}_s{s1}{s2}"
        if res.get("ok"):
            cd = max(Decimal(res["L_lit"]), Decimal(res["L_pd"]))
            print(
                f"    [{lbl}] dL={res['dL_lit']:+.6e} L={res['L_lit'][:14]} ({time.time()-t0:.1f}s)",
                flush=True,
            )
            if best_dec is None or cd < best_dec:
                best_dec = cd
                best_res = res
                best_label = lbl
            if res["is_real_win"] and res["dL_lit"] <= -5e-4:
                save_win(n, res, lbl)

    if best_res is not None and try_mechwalk and best_res["dL_lit"] < 3e-3:
        mw = mechwalk_polish_once(n, best_res["rows"], L_inc_dec, n_threads=n_threads)
        if mw is not None and mw.get("ok"):
            cd = max(Decimal(mw["L_lit"]), Decimal(mw["L_pd"]))
            if cd < best_dec:
                best_dec = cd
                best_res = mw
                best_label += "_mw"
                print(
                    f"    [mechwalk {best_label}] dL={mw['dL_lit']:+.6e} L={mw['L_lit'][:14]}",
                    flush=True,
                )

    if best_res is None:
        return {"n": n, "status": "FAIL", "dt": round(time.time() - t0, 1)}
    if best_res["is_real_win"]:
        save_win(n, best_res, best_label)
    return {
        "n": n,
        "K": K,
        "D": cand["D"],
        "kx": cand["kx"],
        "ky": cand["ky"],
        "regime": "zero_gap",
        "status": "WIN" if best_res["is_real_win"] else "OK",
        "dL": best_res["dL_lit"],
        "L_lit": best_res["L_lit"],
        "L_pd": best_res["L_pd"],
        "L_inc": best_res["L_inc"],
        "label": best_label,
        "dt": round(time.time() - t0, 1),
    }


def eval_seam_touch_candidate(
    cand: dict,
    merged_rows: dict[int, list[dict[str, str]]],
    best_rows: dict[int, list[dict[str, str]]],
    L_inc_dec: Decimal,
    n_threads: int = 2,
) -> dict:
    """Evaluate a Regime 2 (seam_touch, D=g1+g2) candidate via C++ seam melt + SLP + lpslack."""
    t0 = time.time()
    n, K = cand["n"], cand["K"]
    p1, p2 = cand["p1"], cand["p2"]
    ext1 = il.get_item_squares_50d(p1, merged_rows, best_rows)
    ext2 = il.get_item_squares_50d(p2, merged_rows, best_rows)
    if ext1 is None or ext2 is None:
        return {"n": n, "status": "EXTRACT_FAIL", "dt": round(time.time() - t0, 1)}
    sq1, W1, H1 = ext1
    sq2, W2, H2 = ext2

    # Rank all 16 (s1, s2) symmetries by initial SAT penalty
    asm_cands = []
    for s1 in range(4):
        q1 = il.sym_squares_50d(sq1, W1, H1, s1)
        for s2 in range(4):
            q2 = il.sym_squares_50d(sq2, W2, H2, s2)
            asm = il.assemble_seam_touch_float(
                K, n, q1, W1, H1, p1["w"], p1["h"], q2, W2, H2, p2["w"], p2["h"]
            )
            if asm is None:
                continue
            ax, ay, at, aL, seam_box = asm
            pen0, _ = cpp_eval_sat_pen_bbox(ax, ay, at)
            asm_cands.append((pen0, s1, s2, ax, ay, at, aL, seam_box))

    if not asm_cands:
        return {"n": n, "status": "ASM_FAIL", "dt": round(time.time() - t0, 1)}
    asm_cands.sort(key=lambda r: r[0])

    rng = np.random.default_rng(31000 + n)
    best_res = None
    best_dec = None
    best_label = ""
    L_inc_f = float(L_inc_dec)

    for pen0, s1, s2, ax, ay, at, aL, seam_box in asm_cands[:2]:
        L0 = max(aL, L_inc_f) + 0.045
        sc = L0 / aL
        x0, y0, t0_arr = ax * sc, ay * sc, at.copy()
        sx0, sx1, sy0, sy1 = [v * sc for v in seam_box]
        scx, scy = 0.5 * (sx0 + sx1), 0.5 * (sy0 + sy1)

        L_t1 = max(aL + 0.002, min(aL + 0.015, L_inc_f - 0.002))
        scheds = [
            [L0, 0.5 * (L0 + L_t1), L_t1 + 0.004, L_t1],
            [L0, aL + 0.020, aL + 0.006, aL - 0.002],
            [L0, L_inc_f + 0.010, L_inc_f - 0.002, L_inc_f - 0.008],
        ]
        centers = [
            (scx, scy),
            (sx0, sy0),
            (sx1, sy1),
            (sx0, sy1),
            (sx1, sy0),
            (scx, sy0 - 1.2),
            (scx, sy1 + 1.2),
        ]

        q_list, m_list, ls_list = [], [], []
        for idx in range(12):
            cx, cy = centers[idx % len(centers)]
            rad = float(rng.uniform(4.2, 8.0))
            in_m = ((x0 - cx) ** 2 + (y0 - cy) ** 2) <= rad * rad
            # Always free squares inside/near seam_box
            in_seam = (x0 >= sx0 - 1.5) & (x0 <= sx1 + 1.5) & (y0 >= sy0 - 1.5) & (y0 <= sy1 + 1.5)
            m_sq = (in_m | in_seam).astype(np.uint8)
            mask = np.concatenate([m_sq, m_sq, m_sq])
            x, y, th = x0.copy(), y0.copy(), t0_arr.copy()
            x[in_seam] += rng.uniform(-0.08, 0.08, size=int(np.sum(in_seam)))
            y[in_seam] += rng.uniform(-0.08, 0.08, size=int(np.sum(in_seam)))
            th[in_seam] += rng.uniform(-0.18, 0.18, size=int(np.sum(in_seam)))
            q_list.append(full_to_free_sym(x, y, th, L0))
            m_list.append(mask)
            ls_list.append(scheds[idx % len(scheds)])

        qb, qf, lf, _, po, _ = run_cpp_masked_melt(
            n,
            1,
            q_list,
            m_list,
            ls_list,
            [6, 10, 14, 18],
            a_start=0.82,
            n_melt_stages=4,
            masked_iters=220,
            full_iters=220,
            n_threads=n_threads,
            base_seed=41000 + n * 17 + s1 * 4 + s2,
        )
        order = sorted(
            range(len(q_list)),
            key=lambda i: (0 if lf[i] < 1e9 else 1, float(lf[i]) if lf[i] < 1e9 else float(po[i])),
        )[:4]

        for i in order:
            use_f = lf[i] < 1e9
            L_ref = float(lf[i]) if use_f else ls_list[i][-1]
            x, y, th = cpp_expand_sym(n, 1, L_ref, qf[i] if use_f else qb[i])
            pen, bb = cpp_eval_sat_pen_bbox(x, y, th)
            if 1e-7 < pen < 0.12:
                x, y, th, _ = hu.relax_lbfgs(x, y, th, bb + max(0.004, 2.0 * pen), maxiter=350, w_wall=4.0, w_pair=1.0)
                pen, bb = cpp_eval_sat_pen_bbox(x, y, th)
            if pen > 0.025:
                continue
            xs, ys, ts, _, ok = brick_slp.joint_xytheta_slp_tight(x, y, th)
            if not ok:
                continue
            rows = lpslack.rows_from_arrays(np.column_stack([xs, ys]), ts)
            res = il.polish_rows_lpslack_mechwalk(n, rows, L_inc_dec)
            lbl = f"st_{p1['uid']}+{p2['uid']}_s{s1}{s2}_m{i}"
            if res.get("ok"):
                cd = max(Decimal(res["L_lit"]), Decimal(res["L_pd"]))
                print(
                    f"    [{lbl}] dL={res['dL_lit']:+.6e} L={res['L_lit'][:14]} ({time.time()-t0:.1f}s)",
                    flush=True,
                )
                if best_dec is None or cd < best_dec:
                    best_dec = cd
                    best_res = res
                    best_label = lbl
                if res["is_real_win"] and res["dL_lit"] <= -5e-4:
                    save_win(n, res, lbl)

    if best_res is None:
        return {
            "n": n,
            "K": K,
            "D": cand["D"],
            "kx": cand["kx"],
            "ky": cand["ky"],
            "regime": "seam_touch",
            "status": "MELT_FAIL",
            "dt": round(time.time() - t0, 1),
        }
    if best_res["is_real_win"]:
        save_win(n, best_res, best_label)
    return {
        "n": n,
        "K": K,
        "D": cand["D"],
        "kx": cand["kx"],
        "ky": cand["ky"],
        "regime": "seam_touch",
        "status": "WIN" if best_res["is_real_win"] else "OK",
        "dL": best_res["dL_lit"],
        "L_lit": best_res["L_lit"],
        "L_pd": best_res["L_pd"],
        "L_inc": best_res["L_inc"],
        "label": best_label,
        "dt": round(time.time() - t0, 1),
    }
