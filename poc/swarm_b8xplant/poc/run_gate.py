#!/usr/bin/env python3
"""15-Minute Falsification Gate for H25 + H28 on {858, 800, 745} + standalone B8_18x8 brick."""
from __future__ import annotations

import json
import math
import sys
import time
from decimal import Decimal
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import b8_lib as bl  # noqa: E402

REPO_ROOT = bl.REPO_ROOT
import brick_rect_lib as bslp  # noqa: E402


def build_n800_transplant_seeds(b8: dict) -> tuple[list[np.ndarray], list[np.ndarray], list[list[float]], list[str]]:
    """Build 5-region pinwheel seeds for n=800 (m=28, r=+16) using B8_10x18_NEW (188) + B8_18x10_NEW (188)."""
    m = 28
    L_target = 28.0 + b8["f"]
    q_list, m_list, ls_list, labels = [], [], [], []
    # Unit squares: BL 18x18 [0,18]x[0,18], TR 10x10 [18+f, 28+f]x[18+f, 28+f]
    gx, gy = [], []
    for i in range(18):
        for j in range(18):
            gx.append(i + 0.5)
            gy.append(j + 0.5)
    f = b8["f"]
    for i in range(10):
        for j in range(10):
            gx.append(18.0 + f + i + 0.5)
            gy.append(18.0 + f + j + 0.5)
    gx_arr = np.array(gx, dtype=np.float64)
    gy_arr = np.array(gy, dtype=np.float64)
    gt_arr = np.zeros(len(gx), dtype=np.float64)

    for br_mode in ("id", "flipx", "flipy", "rot180"):
        br = bl.transform_brick(b8, br_mode)
        tl = bl.transform_brick(br, "T")
        # TL in [0, 18+f] x [18, 28+f], BR in [18, 28+f] x [0, 18+f]
        ax = np.concatenate([tl["x"], br["x"] + 18.0, gx_arr])
        ay = np.concatenate([tl["y"] + 18.0, br["y"], gy_arr])
        at = np.concatenate([tl["th"], br["th"], gt_arr])
        assert len(ax) == 800
        for rad, tag_r in ((3.2, "r3"), (5.0, "r5"), (7.2, "r7")):
            L0 = L_target + 0.018
            sc = L0 / L_target
            near = ((ax - 18.4) ** 2 + (ay - 18.4) ** 2) <= rad * rad
            # Also include whole seam strip [17..20] x [17..20]
            near = near | ((ax >= 16.5) & (ax <= 20.5) & (ay >= 16.5) & (ay <= 20.5))
            q_list.append(bl.full_to_free_sym(ax * sc, ay * sc, at, L0))
            m_list.append(np.concatenate([near.astype(np.uint8)] * 3))
            ls_list.append([L0, L_target + 0.005, L_target + 0.0005, L_target - 0.0005])
            labels.append(f"xplant800_{br_mode}_{tag_r}")
    return q_list, m_list, ls_list, labels


def build_n858_transplant_seeds(
    b8: dict, x_inc: np.ndarray, y_inc: np.ndarray, th_inc: np.ndarray
) -> tuple[list[np.ndarray], list[np.ndarray], list[list[float]], list[str]]:
    """Build transplant seeds for n=858 (m=29, r=+17):
    1) Replace TL (18x8 + 2 rows = 188) in pw[858] with B8_18x10_NEW.
    2) Replace BR (8x18 + 2 cols = 188) in pw[858] with B8_10x18_NEW.
    3) 5-region pinwheel(29, B8_18x10_NEW, B8_10x18_NEW) = 857 + replace 2x2 by 2x2+1.
    """
    q_list, m_list, ls_list, labels = [], [], [], []
    f = b8["f"]
    L_target = 29.0 + f
    xi, yi, ti, Li = bl.normalize(x_inc, y_inc, th_inc)
    hi = bl.half_extent(ti)

    # Seed family 1: replace TL [0, 18.78] x [19.0, 29.78] (check exact count)
    # Find squares in [0, 18.8] x [19.0, 29.8] or adjust to exact 188
    in_tl = (xi <= 18.78) & (yi >= 19.0)
    # Keep ~in_tl, add B8_18x10_NEW; if count != 858, adjust by taking/dropping boundary squares
    tl_brick = bl.transform_brick(b8, "T")
    rem_mask = ~in_tl
    # Sort squares near boundary y=19.0, x<18.78 to match 858 - 188 = 670
    if int(rem_mask.sum()) != 670:
        # Select top 188 squares in x<18.78 sorted by (-y)
        cand_idx = np.flatnonzero(xi < 18.78)
        order = cand_idx[np.argsort(-yi[cand_idx])]
        drop_idx = set(order[:188].tolist())
        rem_mask = np.array([k not in drop_idx for k in range(len(xi))], dtype=bool)
    ax1 = np.concatenate([xi[rem_mask], tl_brick["x"]])
    ay1 = np.concatenate([yi[rem_mask], tl_brick["y"] + 19.0])
    at1 = np.concatenate([ti[rem_mask], tl_brick["th"]])
    assert len(ax1) == 858
    for rad, tag_r in ((3.8, "r4"), (6.0, "r6")):
        L0 = L_target + 0.015
        sc = L0 / max(Li, L_target)
        near = ((ax1 - 18.4) ** 2 + (ay1 - 19.8) ** 2) <= rad * rad
        near = near | ((ax1 <= 19.8) & (ay1 >= 18.0) & (ay1 <= 21.5))
        q_list.append(bl.full_to_free_sym(ax1 * sc, ay1 * sc, at1, L0))
        m_list.append(np.concatenate([near.astype(np.uint8)] * 3))
        ls_list.append([L0, L_target + 0.004, L_target - 1e-4, L_target - 0.001])
        labels.append(f"xplant858_TL_{tag_r}")

    # Seed family 2: 5-region pinwheel(29, B8_18x10_NEW, B8_10x18_NEW) = 857 + replace 2x2 with 2x2+1
    pw_res = bslp.assemble_pinwheel(29, tl_brick, b8, align_mode="flush_outer")
    if pw_res is not None:
        px, py, pt, Lp = pw_res
        # pw_res has 857 squares! Replace 2x2 in BL [0,2]x[0,2] or [1,3]x[1,3] or TR [27+f, 29+f]^2 with 2x2+1
        c5x = np.array([0.5, 0.5, 2.20710678, 2.20710678, 1.35355339], dtype=np.float64)
        c5y = np.array([0.5, 2.20710678, 0.5, 2.20710678, 1.35355339], dtype=np.float64)
        c5t = np.array([0.0, 0.0, 0.0, 0.0, 0.25 * math.pi], dtype=np.float64)
        for ox, oy, cname in ((0.0, 0.0, "BL0"), (26.29 + f, 26.29 + f, "TR")):
            if cname == "BL0":
                drop = (px < 2.1) & (py < 2.1)
                cx_add, cy_add = c5x, c5y
            else:
                drop = (px > 27.0 + f) & (py > 27.0 + f)
                cx_add, cy_add = c5x + (27.0 + f - 0.2), c5y + (27.0 + f - 0.2)
            if int(drop.sum()) == 4:
                ax2 = np.concatenate([px[~drop], cx_add])
                ay2 = np.concatenate([py[~drop], cy_add])
                at2 = np.concatenate([pt[~drop], c5t])
                assert len(ax2) == 858
                for rad, tag_r in ((5.5, "r55"), (8.0, "r80")):
                    L0 = Lp + 0.020
                    sc = L0 / Lp
                    cx0 = 1.4 if cname == "BL0" else 28.0
                    cy0 = 1.4 if cname == "BL0" else 28.0
                    near = (((ax2 - cx0) ** 2 + (ay2 - cy0) ** 2) <= rad * rad) | (
                        ((ax2 - 18.4) ** 2 + (ay2 - 18.4) ** 2) <= 3.5 * 3.5
                    )
                    q_list.append(bl.full_to_free_sym(ax2 * sc, ay2 * sc, at2, L0))
                    m_list.append(np.concatenate([near.astype(np.uint8)] * 3))
                    ls_list.append([L0, Lp + 0.005, Lp - 1e-4, Lp - 0.001])
                    labels.append(f"xplant858_pw5_{cname}_{tag_r}")

    return q_list, m_list, ls_list, labels


def build_n745_transplant_seeds(
    b8: dict, x_inc: np.ndarray, y_inc: np.ndarray, th_inc: np.ndarray
) -> tuple[list[np.ndarray], list[np.ndarray], list[list[float]], list[str]]:
    """Build cropped-B8_10x18_NEW (170 squares, 9x18) transplant seeds for n=745 (m=27, r=+16)."""
    q_list, m_list, ls_list, labels = [], [], [], []
    # Crop 18 rightmost squares from B8_10x18_NEW -> 170 squares (~9.78 x 18.7745)
    bx, by, bt = b8["x"].copy(), b8["y"].copy(), b8["th"].copy()
    order_r = np.argsort(-bx)
    keep170 = np.ones(len(bx), dtype=bool)
    keep170[order_r[:18]] = False
    cx, cy, Wc, Hc = bl.rect_bbox(bx[keep170], by[keep170], bt[keep170])
    ct = bt[keep170].copy()
    f = max(Wc - 9.0, Hc - 18.0)

    # Seed 1: symmetric pinwheel 27 = 18 + 9: 170 (TL 18x9) + 170 (BR 9x18) + 18x18 (324) + 9x9 (81) = 745
    tx, ty, W_t, H_t = bl.rect_bbox(cy, cx, 0.5 * math.pi - ct)
    tt = 0.5 * math.pi - ct
    gx, gy = [], []
    f_b = b8["f"]
    for i in range(18):
        for j in range(18):
            gx.append(i + 0.5)
            gy.append(j + 0.5)
    for i in range(9):
        for j in range(9):
            gx.append(18.0 + f_b + i + 0.5)
            gy.append(18.0 + f_b + j + 0.5)
    ax = np.concatenate([tx, cx + 18.0, np.array(gx)])
    ay = np.concatenate([ty + 18.0, cy, np.array(gy)])
    at = np.concatenate([tt, ct, np.zeros(len(gx))])
    assert len(ax) == 745
    L_t = 27.0 + f_b
    for rad, tag_r in ((3.5, "r35"), (5.5, "r55")):
        L0 = L_t + 0.025
        sc = L0 / L_t
        near = ((ax - 18.4) ** 2 + (ay - 18.4) ** 2) <= rad * rad
        near = near | (ax >= 26.5) | (ay >= 26.5)
        q_list.append(bl.full_to_free_sym(ax * sc, ay * sc, at, L0))
        m_list.append(np.concatenate([near.astype(np.uint8)] * 3))
        ls_list.append([L0, L_t + 0.008, L_t + 0.001, L_t - 0.0005])
        labels.append(f"xplant745_sym170_{tag_r}")

    # Seed 2: replace only BR [18, 27.78] x [0, 18.78] (170 squares) in inc[745]
    xi, yi, ti, Li = bl.normalize(x_inc, y_inc, th_inc)
    in_br = (xi >= 18.0) & (yi <= 18.78)
    if int(in_br.sum()) != 170:
        cand = np.flatnonzero(yi <= 18.78)
        order = cand[np.argsort(-xi[cand])]
        drop = set(order[:170].tolist())
        in_br = np.array([k in drop for k in range(len(xi))], dtype=bool)
    ax2 = np.concatenate([xi[~in_br], cx + 18.0])
    ay2 = np.concatenate([yi[~in_br], cy])
    at2 = np.concatenate([ti[~in_br], ct])
    assert len(ax2) == 745
    L0 = L_t + 0.020
    sc = L0 / max(Li, L_t)
    near2 = (ax2 >= 16.5) & (ay2 <= 20.0)
    q_list.append(bl.full_to_free_sym(ax2 * sc, ay2 * sc, at2, L0))
    m_list.append(np.concatenate([near2.astype(np.uint8)] * 3))
    ls_list.append([L0, L_t + 0.006, L_t + 0.0005, L_t - 0.0005])
    labels.append("xplant745_BR170")

    return q_list, m_list, ls_list, labels


def run_transplant_batch(
    n: int,
    q_list: list[np.ndarray],
    m_list: list[np.ndarray],
    ls_list: list[list[float]],
    labels: list[str],
    inc_dec: Decimal,
    top_k: int = 5,
    threads: int = 4,
) -> dict | None:
    if not q_list:
        return None
    seeds = []
    seam_masks = []
    for i, q in enumerate(q_list):
        x, y, th = bl.cpp_expand_sym(n, 1, ls_list[i][0], q)
        seeds.append((x, y, th))
        seam_masks.append(m_list[i][:n].astype(np.uint8))

    relaxed = bl.su.cpp_batch_melt(
        seeds,
        ls_list,
        sym_mode=1,
        a_start=0.95,
        n_melt_stages=4,
        seam_masks=seam_masks,
        seam_iters=220,
        full_iters=260,
        n_ruin_iters=0,
        n_threads=min(4, max(1, threads)),
        base_seed=71000 + n,
    )
    pool = []
    for bi, row in enumerate(relaxed):
        nx, ny, nth, Lm = bl.normalize(row["x"], row["y"], row["th"])
        if row["best_L_feas"] is not None:
            pool.append((0.0, Lm, nx, ny, nth, labels[bi]))
        elif row["pen"] < 0.02:
            pool.append((1.0 + float(row["pen"]), Lm, nx, ny, nth, labels[bi]))
    pool.sort(key=lambda t: (t[0], t[1]))
    best_cert = None
    for _, _, cx, cy, cth, lbl in pool[:top_k]:
        cert = bl.polish_and_certify(n, cx, cy, cth, inc_dec, slp_rounds=16)
        if cert is not None:
            cert["label"] = lbl
            if best_cert is None or cert["L_float"] < best_cert["L_float"]:
                best_cert = cert
    return best_cert


def run_standalone_b8_18x8_walk(x745: np.ndarray, y745: np.ndarray, th745: np.ndarray) -> dict:
    """Test H28 localized/standalone Contact-Jacobian SVD walk on B8_18x8 (152 squares) in [0,18+f]x[0,8+f]."""
    hx = bl.half_extent(th745)
    x745, y745, th745, _ = bl.normalize(x745, y745, th745)
    # Extract TL (18x8, +8) in [0, 18.78] x [19, 27.78]
    m = (x745 + hx <= 18.80) & (y745 - hx >= 18.99)
    bx, by, W0, H0 = bl.rect_bbox(x745[m], y745[m], th745[m])
    bt = th745[m].copy()
    f0 = max(W0 - 18.0, H0 - 8.0)
    print(f"[B8_18x8] extracted cnt={len(bx)} f0={f0:.9f}", flush=True)

    best_f = f0
    best_state = (bx.copy(), by.copy(), bt.copy())
    # Run SVD soft-mode walk on the 152-square brick + relax_rect_lbfgs + joint_xytheta_slp_rect
    free_all = np.ones(len(bx), dtype=bool)
    for rnd in range(2):
        improved_r = False
        for ctol in (1e-5, 5e-4, 2e-3):
            jac, _, _ = bl.localized_contact_jacobian(
                best_state[0], best_state[1], best_state[2], free_all, contact_tol=ctol
            )
            _, modes = bl.localized_soft_modes(jac, k=12)
            for mi in range(min(8, len(modes))):
                v = modes[mi]
                nf = len(bx)
                for amp in (0.008, 0.025, 0.07):
                    for sgn in (1.0, -1.0):
                        cx = best_state[0] + sgn * amp * v[:nf]
                        cy = best_state[1] + sgn * amp * v[nf : 2 * nf]
                        ct = best_state[2] + sgn * amp * v[2 * nf :]
                        for df_try in (-2e-3, -3e-4, 2e-4):
                            rx, ry, rt, pen = bslp.relax_rect_lbfgs(
                                cx, cy, ct, 18.0 + best_f + df_try, 8.0 + best_f + df_try, maxiter=350
                            )
                            if pen <= 1e-6:
                                break
                        if pen > 1e-6:
                            continue
                        sx, sy, st, sf, ok = bslp.joint_xytheta_slp_rect(rx, ry, rt, 18, 8, rounds=8)
                        if ok and sf < best_f - 1e-7:
                            best_f = sf
                            best_state = (sx, sy, st)
                            improved_r = True
                            print(f"  [B8_18x8] rnd={rnd} ctol={ctol} m{mi} amp={sgn*amp} -> f={best_f:.9f} (df={best_f-f0:+.6e})", flush=True)
        if not improved_r:
            break
    out_json = bl.DATA_DIR / "B8_18x8_walk.json"
    out_json.write_text(
        json.dumps(
            {"w": 18, "h": 8, "g": 8, "f0": f0, "f": best_f, "df": best_f - f0,
             "x": best_state[0].tolist(), "y": best_state[1].tolist(), "th": best_state[2].tolist()},
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"f0": f0, "f": best_f, "df": best_f - f0}


def main() -> None:
    t0 = time.time()
    gate_ns = [858, 800, 745]
    b8 = bl.ensure_b8_brick_json()
    print(f"Loaded B8_10x18_NEW: cnt={len(b8['x'])} fx={b8['fx']:.9f} fy={b8['fy']:.9f}", flush=True)

    frontier = bl.load_true_frontier_targets(gate_ns)
    for n in gate_ns:
        print(f"TRUE frontier n={n}: L={str(frontier[n]['L_dec'])[:20]} ({frontier[n]['source']})", flush=True)

    results = {}

    # 1. Part A: Direct B8_10x18_NEW transplants
    for n in gate_ns:
        t_n = time.time()
        inc_dec = frontier[n]["L_dec"]
        xi, yi, ti = frontier[n]["x"], frontier[n]["y"], frontier[n]["th"]
        if n == 800:
            ql, ml, lsl, lbs = build_n800_transplant_seeds(b8)
        elif n == 858:
            ql, ml, lsl, lbs = build_n858_transplant_seeds(b8, xi, yi, ti)
        else:
            ql, ml, lsl, lbs = build_n745_transplant_seeds(b8, xi, yi, ti)

        cert = run_transplant_batch(n, ql, ml, lsl, lbs, inc_dec, top_k=5, threads=4)
        saved = False
        if cert is not None and cert["is_real_win"]:
            saved = bl.save_win_locked(cert)
            print(f"[PART A WIN] n={n} {cert['label']} L={cert['L_dec_str'][:20]} dL={cert['dL']:+.6e} saved={saved} ({time.time()-t_n:.1f}s)", flush=True)
        else:
            msg = f"L={cert['L_dec_str'][:18]} dL={cert['dL']:+.3e}" if cert else "no_feas"
            print(f"[PART A MISS] n={n} {msg} ({time.time()-t_n:.1f}s)", flush=True)
        results[f"partA_{n}"] = {
            "win": bool(cert and cert["is_real_win"]),
            "L": cert["L_dec_str"] if cert else None,
            "dL": cert["dL"] if cert else None,
            "label": cert.get("label") if cert else None,
            "seconds": round(time.time() - t_n, 2),
        }

    # 2. Part B: Localized Sub-Brick Contact-Jacobian SVD Walk on {858, 800, 745}
    for n in gate_ns:
        t_n = time.time()
        inc_dec = frontier[n]["L_dec"]
        cur_x, cur_y, cur_th = frontier[n]["x"], frontier[n]["y"], frontier[n]["th"]
        best_cert_n = None
        for r_idx in range(2):
            masks = bl.find_tilted_brick_masks(cur_x, cur_y, cur_th, collar=2.15)
            improved_round = False
            for mtag, fmask in masks[:2]:
                cert, nx, ny, nt, nL = bl.run_localized_mechwalk(
                    n, cur_x, cur_y, cur_th, fmask, inc_dec,
                    tag=f"loc_r{r_idx}_{mtag}",
                    contact_tols=(1e-5, 1e-3),
                    amplitudes=(0.008, 0.03, 0.12),
                    n_modes=8,
                    top_k_slp=5, slp_rounds=14, threads=4, seed_base=61000 + r_idx * 1000,
                )
                if cert is not None and (best_cert_n is None or cert["L_float"] < best_cert_n["L_float"]):
                    best_cert_n = cert
                    if cert["L_float"] < float(inc_dec) - 1e-7:
                        cur_x, cur_y, cur_th = nx, ny, nt
                        improved_round = True
            if not improved_round:
                break

        if best_cert_n is not None and best_cert_n["is_real_win"]:
            saved = bl.save_win_locked(best_cert_n)
            print(f"[PART B WIN] n={n} {best_cert_n['label']} L={best_cert_n['L_dec_str'][:20]} dL={best_cert_n['dL']:+.6e} saved={saved} ({time.time()-t_n:.1f}s)", flush=True)
        else:
            msg = f"L={best_cert_n['L_dec_str'][:18]} dL={best_cert_n['dL']:+.3e}" if best_cert_n else "no_feas"
            print(f"[PART B MISS] n={n} {msg} ({time.time()-t_n:.1f}s)", flush=True)
        results[f"partB_{n}"] = {
            "win": bool(best_cert_n and best_cert_n["is_real_win"]),
            "L": best_cert_n["L_dec_str"] if best_cert_n else None,
            "dL": best_cert_n["dL"] if best_cert_n else None,
            "label": best_cert_n.get("label") if best_cert_n else None,
            "seconds": round(time.time() - t_n, 2),
        }

    # 3. Standalone B8_18x8 localized SVD walk
    b8_18x8_res = run_standalone_b8_18x8_walk(frontier[745]["x"], frontier[745]["y"], frontier[745]["th"])
    results["standalone_B8_18x8"] = b8_18x8_res

    bl.DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (bl.DOCS_DIR / "gate_report.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"GATE DONE in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
