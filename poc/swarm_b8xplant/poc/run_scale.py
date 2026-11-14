#!/usr/bin/env python3
"""Scale-up runner for H25 + H28:
- Track 0: Polish n=800 win further
- Track A: B8_10x18_NEW transplants on {918, 691, 858, 745} with shake-free cpp_batch_melt
- Track B: Localized Sub-Brick Contact-Jacobian SVD Walk across {918, 691, 745, 858, 916, 917, 638, 690, 744, 854, 913, 914}
- Track C: Standalone brick SVD walks (B8_18x8, B8_10x18_NEW, B7_17x8, B6_14x7, B10_22x11, B6_24x8, B5_22x7) + broadcast
- Track D: Upward 1x1 L-border extensions (m -> m+1, m+2) from every improved n
"""
from __future__ import annotations

import json
import math
import sys
import time
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import b8_lib as bl  # noqa: E402
import brick_rect_lib as bslp  # noqa: E402

SCALE_TARGETS = [918, 691, 745, 858, 916, 917, 638, 690, 744, 854, 913, 914]


def localized_walk_dual(
    n: int,
    x0: np.ndarray,
    y0: np.ndarray,
    th0: np.ndarray,
    free_mask: np.ndarray,
    inc_dec: Decimal,
    tag: str,
    threads: int = 4,
    seed_base: int = 81000,
) -> tuple[dict | None, np.ndarray, np.ndarray, np.ndarray, float]:
    """Run localized SVD walk testing BOTH seam_iters=0 (exact mechwalk recipe) and seam_iters=140."""
    x0, y0, th0, L0 = bl.normalize(x0, y0, th0)
    free_idx = np.flatnonzero(free_mask)
    nf = len(free_idx)
    if nf < 8:
        return None, x0, y0, th0, L0

    seeds: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    seam_masks: list[np.ndarray] = []
    schedules: list[list[float]] = []
    labels: list[str] = []
    mask_u8 = free_mask.astype(np.uint8)

    for ctol in (1e-5, 5e-4, 2e-3):
        jac, fidx, _ = bl.localized_contact_jacobian(x0, y0, th0, free_mask, contact_tol=ctol)
        if jac.shape[0] == 0:
            continue
        _, modes = bl.localized_soft_modes(jac, k=10)
        kept = []
        for m_i, mode in enumerate(modes):
            v = mode.astype(np.float64).copy()
            norm = float(np.linalg.norm(v))
            if norm < 1e-10:
                continue
            v /= norm
            if any(abs(float(np.dot(v, prev))) > 0.92 for prev in kept):
                continue
            kept.append(v)
            for amp in (0.004, 0.008, 0.025, 0.08):
                for sgn, slb in ((1.0, "p"), (-1.0, "m")):
                    cx, cy, cth = x0.copy(), y0.copy(), th0.copy()
                    cx[fidx] += sgn * amp * v[:nf]
                    cy[fidx] += sgn * amp * v[nf : 2 * nf]
                    cth[fidx] += sgn * amp * v[2 * nf :]
                    cx, cy, cth, _ = bl.normalize(cx, cy, cth)
                    seeds.append((cx, cy, cth))
                    seam_masks.append(mask_u8)
                    if (len(seeds) % 2) == 0:
                        schedules.append([L0 + 0.012, L0 + 0.003, L0 - 2e-4])
                    else:
                        schedules.append([L0 + 0.025, L0 + 0.006, L0 - 5e-4])
                    labels.append(f"{tag}_t{ctol:.0e}_m{m_i}_{slb}_{amp:.3g}")
            if len(kept) >= 8:
                break
        if ctol >= 1e-3 and len(kept) >= 2:
            inv_s2 = 1.0 / math.sqrt(2.0)
            kcap = min(4, len(kept))
            for i in range(kcap):
                for j in range(i + 1, kcap):
                    for d_sgn, dtag in ((1.0, "p"), (-1.0, "m")):
                        vd = inv_s2 * (kept[i] + d_sgn * kept[j])
                        for amp in (0.008, 0.025):
                            for sgn, slb in ((1.0, "p"), (-1.0, "m")):
                                cx, cy, cth = x0.copy(), y0.copy(), th0.copy()
                                cx[fidx] += sgn * amp * vd[:nf]
                                cy[fidx] += sgn * amp * vd[nf : 2 * nf]
                                cth[fidx] += sgn * amp * vd[2 * nf :]
                                cx, cy, cth, _ = bl.normalize(cx, cy, cth)
                                seeds.append((cx, cy, cth))
                                seam_masks.append(mask_u8)
                                schedules.append([L0 + 0.015, L0 + 0.004, L0 - 3e-4])
                                labels.append(f"{tag}_t{ctol:.0e}_d{i}{dtag}{j}_{slb}_{amp:.3g}")

    if not seeds:
        return None, x0, y0, th0, L0

    cand_pool: list[tuple[float, float, np.ndarray, np.ndarray, np.ndarray, str]] = []
    for siter in (0, 140):
        for start in range(0, len(seeds), 64):
            end = min(len(seeds), start + 64)
            relaxed = bl.su.cpp_batch_melt(
                seeds[start:end],
                schedules[start:end],
                sym_mode=1,
                a_start=0.965,
                n_melt_stages=3,
                seam_masks=seam_masks[start:end] if siter > 0 else None,
                seam_iters=siter,
                full_iters=260,
                n_ruin_iters=0,
                n_threads=min(4, max(1, threads)),
                base_seed=seed_base + n * 17 + siter * 7 + start,
            )
            for bi, row in enumerate(relaxed):
                nx, ny, nth, Lm = bl.normalize(row["x"], row["y"], row["th"])
                lbl = f"{labels[start + bi]}_si{siter}"
                if row["best_L_feas"] is not None:
                    cand_pool.append((0.0, Lm, nx, ny, nth, lbl))
                elif row["pen"] < 0.02:
                    cand_pool.append((1.0 + float(row["pen"]), Lm, nx, ny, nth, lbl))

    cand_pool.sort(key=lambda t: (t[0], t[1]))
    best_cert: dict | None = None
    best_state = (x0, y0, th0, L0)
    for _, _, cx, cy, cth, lbl in cand_pool[:6]:
        cert = bl.polish_and_certify(n, cx, cy, cth, inc_dec, slp_rounds=16)
        if cert is not None:
            cert["label"] = lbl
            if best_cert is None or cert["L_float"] < best_cert["L_float"]:
                best_cert = cert
            if cert["L_float"] < best_state[3]:
                best_state = (cx, cy, cth, cert["L_float"])
    return best_cert, best_state[0], best_state[1], best_state[2], best_state[3]


def build_masks_for_n(x: np.ndarray, y: np.ndarray, th: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Build localized masks (tilted clusters + 2-layer collar, plus quadrant fallback if >260 sq)."""
    masks = bl.find_tilted_brick_masks(x, y, th, collar=2.15)
    out: list[tuple[str, np.ndarray]] = []
    hx = bl.half_extent(th)
    L = float(max(np.max(x + hx), np.max(y + hx)))
    for tag, m in masks:
        if int(m.sum()) <= 260:
            out.append((tag, m))
        else:
            # Split oversized tilted cluster into 4 overlapping quadrant-collar masks
            for qname, qm in (
                ("TL", (x < 0.58 * L) & (y > 0.42 * L)),
                ("BR", (x > 0.42 * L) & (y < 0.58 * L)),
                ("BL", (x < 0.58 * L) & (y < 0.58 * L)),
                ("TR", (x > 0.42 * L) & (y > 0.42 * L)),
            ):
                sub = m & qm
                if 30 <= int(sub.sum()) <= 280:
                    out.append((f"{tag}_{qname}_{int(sub.sum())}f", sub))
    return out[:4]


def try_n918_transplant(b8: dict, inc_dec: Decimal, x_inc: np.ndarray, y_inc: np.ndarray, th_inc: np.ndarray) -> dict | None:
    """Transplant B8_10x18_NEW into n=918 TL [0, 10.7745] x [12.0, 30.7745] and melt sq 96,97 junction."""
    xi, yi, ti, Li = bl.normalize(x_inc, y_inc, th_inc)
    hx = bl.half_extent(ti)
    in_tl = (xi + hx <= 10.80) & (yi - hx >= 11.99)
    if int(in_tl.sum()) != 188:
        return None
    rem = ~in_tl
    L_tgt = 30.0 + b8["f"]
    seeds, masks, scheds, labels = [], [], [], []
    for mode in ("id", "flipx", "flipy", "rot180"):
        br = bl.transform_brick(b8, mode)
        ax = np.concatenate([xi[rem], br["x"]])
        ay = np.concatenate([yi[rem], br["y"] + 12.0])
        at = np.concatenate([ti[rem], br["th"]])
        for rad_tag, m in (
            ("junc4", (ax <= 12.5) & (ay >= 10.0) & (ay <= 15.5)),
            ("left_col", (ax <= 12.5) & (ay >= 9.5)),
            ("tl_bl", (ax <= 13.0) & (ay >= 8.0) & (ay <= 20.0)),
        ):
            L0 = L_tgt + 0.025
            sc = L0 / max(Li, L_tgt)
            seeds.append((ax * sc, ay * sc, at.copy()))
            masks.append(m.astype(np.uint8))
            scheds.append([L0, L_tgt + 0.008, L_tgt + 0.001, L_tgt - 0.0003])
            labels.append(f"xplant918_{mode}_{rad_tag}")

    best_cert = None
    for siter in (220, 0):
        relaxed = bl.su.cpp_batch_melt(
            seeds, scheds, sym_mode=1, a_start=0.95, n_melt_stages=4,
            seam_masks=masks if siter > 0 else None, seam_iters=siter,
            full_iters=260, n_ruin_iters=0, n_threads=4, base_seed=91800 + siter,
        )
        pool = []
        for bi, row in enumerate(relaxed):
            nx, ny, nth, Lm = bl.normalize(row["x"], row["y"], row["th"])
            if row["best_L_feas"] is not None:
                pool.append((0.0, Lm, nx, ny, nth, f"{labels[bi]}_si{siter}"))
            elif row["pen"] < 0.02:
                pool.append((1.0 + float(row["pen"]), Lm, nx, ny, nth, f"{labels[bi]}_si{siter}"))
        pool.sort(key=lambda t: (t[0], t[1]))
        for _, _, cx, cy, cth, lbl in pool[:4]:
            cert = bl.polish_and_certify(918, cx, cy, cth, inc_dec, slp_rounds=16)
            if cert is not None:
                cert["label"] = lbl
                if best_cert is None or cert["L_float"] < best_cert["L_float"]:
                    best_cert = cert
    return best_cert


def try_n691_transplant(b8: dict, inc_dec: Decimal, x_inc: np.ndarray, y_inc: np.ndarray, th_inc: np.ndarray) -> dict | None:
    """Crop 1 col from B8_10x18_NEW (170 sq, 9x18) and transplant into n=691 BR [17, 26.7745] x [0, 18.7745]."""
    xi, yi, ti, Li = bl.normalize(x_inc, y_inc, th_inc)
    # In n=691 (m=26), TL is 17x8,+7 [0, 17.7615] x [18, 26.7615]. BR has width 9 [17, 26.776] x [0, 18.776] (170 sq)
    cand = np.flatnonzero(yi <= 18.78)
    order = cand[np.argsort(-xi[cand])]
    drop = set(order[:170].tolist())
    rem = np.array([k not in drop for k in range(len(xi))], dtype=bool)
    L_tgt = 26.0 + b8["f"]
    seeds, masks, scheds, labels = [], [], [], []
    for mode in ("id", "flipx", "flipy", "rot180"):
        br = bl.transform_brick(b8, mode)
        ord_r = np.argsort(-br["x"])
        keep = np.ones(len(br["x"]), dtype=bool)
        keep[ord_r[:18]] = False
        cx, cy, _, _ = bl.rect_bbox(br["x"][keep], br["y"][keep], br["th"][keep])
        ct = br["th"][keep].copy()
        ax = np.concatenate([xi[rem], cx + 17.0])
        ay = np.concatenate([yi[rem], cy])
        at = np.concatenate([ti[rem], ct])
        assert len(ax) == 691
        L0 = L_tgt + 0.025
        sc = L0 / max(Li, L_tgt)
        m = (ax >= 15.5) & (ay <= 20.0)
        seeds.append((ax * sc, ay * sc, at.copy()))
        masks.append(m.astype(np.uint8))
        scheds.append([L0, L_tgt + 0.008, L_tgt + 0.001, L_tgt - 0.0003])
        labels.append(f"xplant691_{mode}")

    relaxed = bl.su.cpp_batch_melt(
        seeds, scheds, sym_mode=1, a_start=0.95, n_melt_stages=4,
        seam_masks=masks, seam_iters=220, full_iters=260, n_ruin_iters=0, n_threads=4, base_seed=69100,
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
    for _, _, cx, cy, cth, lbl in pool[:4]:
        cert = bl.polish_and_certify(691, cx, cy, cth, inc_dec, slp_rounds=16)
        if cert is not None:
            cert["label"] = lbl
            if best_cert is None or cert["L_float"] < best_cert["L_float"]:
                best_cert = cert
    return best_cert


def try_upward_l_border(
    n_base: int, x_base: np.ndarray, y_base: np.ndarray, th_base: np.ndarray, frontier_cache: dict[int, dict]
) -> list[dict]:
    """Extend a certified win by 1x1 L-borders (m -> m+1, m+2) and certify against TRUE frontier."""
    wins = []
    cx, cy, cth, cL = bl.normalize(x_base, y_base, th_base)
    m = int(math.floor(math.sqrt(n_base)))
    cur_n = n_base
    for step in range(2):
        # Add 2*m + 1 unit squares along outer border (top-right [L, L+1] or bottom-left [0,1])
        next_n = cur_n + 2 * m + 1
        if next_n > 1000:
            break
        if next_n not in frontier_cache:
            frontier_cache.update(bl.load_true_frontier_targets([next_n]))
        inc_dec = frontier_cache[next_n]["L_dec"]

        # Variant 1: top-right L-border
        bx = [cL + 0.5] * (m + 1) + [i + 0.5 for i in range(m)]
        by = [j + 0.5 for j in range(m)] + [cL + 0.5] + [cL + 0.5] * m
        # Check spacing along border: place m squares along x=cL+0.5 at y in [0.5 .. m-0.5] and m along y=cL+0.5 at x in [0.5 .. m-0.5] plus corner (cL+0.5, cL+0.5)
        ax1 = np.concatenate([cx, np.array(bx, dtype=np.float64)])
        ay1 = np.concatenate([cy, np.array(by, dtype=np.float64)])
        at1 = np.concatenate([cth, np.zeros(len(bx), dtype=np.float64)])

        # Variant 2: shift existing by (+1, +1), place bottom-left L-border along [0, 1]
        bx2 = [0.5] * (m + 1) + [i + 1.5 for i in range(m)]
        by2 = [0.5] + [j + 1.5 for j in range(m)] + [0.5] * m
        ax2 = np.concatenate([cx + 1.0, np.array(bx2, dtype=np.float64)])
        ay2 = np.concatenate([cy + 1.0, np.array(by2, dtype=np.float64)])
        at2 = np.concatenate([cth, np.zeros(len(bx2), dtype=np.float64)])

        best_step = None
        for vtag, ax, ay, at in (("TR_L", ax1, ay1, at1), ("BL_L", ax2, ay2, at2)):
            cert = bl.polish_and_certify(next_n, ax, ay, at, inc_dec, slp_rounds=16)
            if cert is not None:
                cert["label"] = f"lborder_{n_base}_to_{next_n}_{vtag}"
                if best_step is None or cert["L_float"] < best_step["L_float"]:
                    best_step = cert
        if best_step is not None and best_step["is_real_win"]:
            saved = bl.save_win_locked(best_step)
            print(f"[TRACK D WIN] n={next_n} {best_step['label']} L={best_step['L_dec_str'][:20]} dL={best_step['dL']:+.6e} saved={saved}", flush=True)
            wins.append(best_step)
            # Advance chain from the new win
            rows = sorted(best_step["rows"], key=lambda r: int(r["square_id"]))
            cx = np.array([float(r["x"]) for r in rows], dtype=np.float64)
            cy = np.array([float(r["y"]) for r in rows], dtype=np.float64)
            cth = np.array([float(r["angle"]) for r in rows], dtype=np.float64)
            cx, cy, cth, cL = bl.normalize(cx, cy, cth)
            cur_n = next_n
            m += 1
        else:
            msg = f"L={best_step['L_dec_str'][:18]} dL={best_step['dL']:+.3e}" if best_step else "no_feas"
            print(f"[TRACK D MISS] n={next_n} (from {cur_n}) {msg}", flush=True)
            break
    return wins


def _try_brick_direction(
    best_state: tuple[np.ndarray, np.ndarray, np.ndarray],
    v: np.ndarray,
    w: int,
    h: int,
    best_f: float,
    amps: tuple[float, ...],
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], float, float | None]:
    nf = len(best_state[0])
    for amp in amps:
        for sgn in (1.0, -1.0):
            cx = best_state[0] + sgn * amp * v[:nf]
            cy = best_state[1] + sgn * amp * v[nf : 2 * nf]
            ct = best_state[2] + sgn * amp * v[2 * nf :]
            rx, ry, rt = None, None, None
            for df_try in (-1.5e-3, -2e-4, 2e-4):
                tx, ty, tt, pen = bslp.relax_rect_lbfgs(
                    cx, cy, ct, w + best_f + df_try, h + best_f + df_try, maxiter=320
                )
                if pen <= 1e-6:
                    rx, ry, rt = tx, ty, tt
                    break
            if rx is None:
                continue
            sx, sy, st, sf, ok = bslp.joint_xytheta_slp_rect(rx, ry, rt, w, h, rounds=8)
            if ok and sf < best_f - 1e-7:
                return (sx, sy, st), sf, sgn * amp
    return best_state, best_f, None


def standalone_brick_walk(w: int, h: int, g: int, x: np.ndarray, y: np.ndarray, th: np.ndarray, name: str) -> dict:
    """Run standalone Contact-Jacobian SVD walk + H34 diagonal mode pairs on a (w, h, g) brick."""
    bx, by, W0, H0 = bl.rect_bbox(x, y, th)
    bt = th.copy()
    f0 = max(W0 - w, H0 - h)
    best_f = f0
    best_state = (bx.copy(), by.copy(), bt.copy())
    free_all = np.ones(len(bx), dtype=bool)
    inv_s2 = 1.0 / math.sqrt(2.0)

    for rnd in range(3):
        improved = False
        for ctol in (1e-5, 4e-4, 2e-3):
            jac, _, _ = bl.localized_contact_jacobian(
                best_state[0], best_state[1], best_state[2], free_all, contact_tol=ctol
            )
            if jac.shape[0] == 0:
                continue
            _, modes = bl.localized_soft_modes(jac, k=12)
            # Single singular modes (H28)
            n_single = min(10, len(modes))
            for mi in range(n_single):
                new_state, new_f, hit_amp = _try_brick_direction(
                    best_state, modes[mi], w, h, best_f, (0.004, 0.008, 0.025, 0.08)
                )
                if hit_amp is not None:
                    best_f = new_f
                    best_state = new_state
                    improved = True
                    print(f"  [BRICK {name}] rnd={rnd} ctol={ctol} m{mi} amp={hit_amp} -> f={best_f:.9f} (df={best_f-f0:+.6e})", flush=True)
            # H34 diagonal mode pairs (v_i ± v_j) / sqrt(2)
            kcap = min(6, len(modes))
            for i in range(kcap):
                for j in range(i + 1, kcap):
                    for d_sgn, dtag in ((1.0, "+"), (-1.0, "-")):
                        vd = inv_s2 * (modes[i] + d_sgn * modes[j])
                        new_state, new_f, hit_amp = _try_brick_direction(
                            best_state, vd, w, h, best_f, (0.008, 0.025, 0.06)
                        )
                        if hit_amp is not None:
                            best_f = new_f
                            best_state = new_state
                            improved = True
                            print(f"  [BRICK {name}] rnd={rnd} ctol={ctol} d({i}{dtag}{j}) amp={hit_amp} -> f={best_f:.9f} (df={best_f-f0:+.6e})", flush=True)
        if not improved:
            break
    out_p = bl.DATA_DIR / f"brick_{name}.json"
    out_p.write_text(
        json.dumps(
            {"w": w, "h": h, "g": g, "f0": f0, "f": best_f, "df": best_f - f0,
             "x": best_state[0].tolist(), "y": best_state[1].tolist(), "th": best_state[2].tolist()},
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "name": name, "w": w, "h": h, "g": g, "f0": f0, "f": best_f, "df": best_f - f0,
        "x": best_state[0], "y": best_state[1], "th": best_state[2],
    }


def extract_b5_22x8(packs: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, float]]) -> dict | None:
    """Extract B5_22x8 (181 sq in TL 22x8) from n=972 if present."""
    if 972 not in packs:
        return None
    x, y, th, L = packs[972]
    hx = bl.half_extent(th)
    m = (x + hx <= 22.99) & (y - hx >= L - 8.99)
    if int(m.sum()) != 181:
        return None
    bx, by, W, H = bl.rect_bbox(x[m], y[m], th[m])
    return {"w": 22, "h": 8, "g": 5, "x": bx, "y": by, "th": th[m].copy(), "fx": W - 22.0, "fy": H - 8.0, "f": max(W - 22.0, H - 8.0)}


def broadcast_improved_bricks(
    improved_list: list[dict], all_bricks: dict[tuple[int, int, int], dict], frontier: dict[int, dict]
) -> list[dict]:
    """Pinwheel-broadcast any brick with df <= -1e-6 across m=24..31 against TRUE frontier."""
    wins: list[dict] = []
    for br in improved_list:
        w, h, g = br["w"], br["h"], br["g"]
        bx, by, W, H = bl.rect_bbox(br["x"], br["y"], br["th"])
        b1 = {"w": w, "h": h, "g": g, "x": bx, "y": by, "th": br["th"].copy(), "fx": W - w, "fy": H - h, "f": max(W - w, H - h)}
        b1_T = bl.transform_brick(b1, "T")
        partners = [b1_T] + [
            v for v in all_bricks.values()
            if abs(v["f"] - b1["f"]) < 0.035 and (v["w"], v["h"], v["g"]) != (w, h, g)
        ]
        seen_n: set[int] = set()
        for m in range(24, 32):
            for b2 in partners:
                n = m * m + b1["g"] + b2["g"]
                if n > 1000 or n in seen_n:
                    continue
                asm = bslp.assemble_pinwheel(m, b1, b2, align_mode="center")
                if asm is None:
                    continue
                seen_n.add(n)
                if n not in frontier:
                    frontier.update(bl.load_true_frontier_targets([n]))
                inc_dec = frontier[n]["L_dec"]
                ax, ay, at, L0 = asm
                if L0 >= float(inc_dec) + 0.01:
                    continue
                cert = bl.polish_and_certify(n, ax, ay, at, inc_dec, slp_rounds=16)
                if cert is not None and cert["is_real_win"]:
                    cert["label"] = f"brick_{br['name']}_m{m}"
                    saved = bl.save_win_locked(cert)
                    print(f"[BROADCAST WIN] n={n} {cert['label']} L={cert['L_dec_str'][:20]} dL={cert['dL']:+.6e} saved={saved}", flush=True)
                    wins.append(cert)
    return wins


def main() -> None:
    t0 = time.time()
    b8 = bl.ensure_b8_brick_json()
    frontier = bl.load_true_frontier_targets(SCALE_TARGETS + [800, 639, 857, 743, 690, 979])
    report: dict = {"tracks": {}, "bricks": []}

    # Track 0: Re-certify/tighten existing n=800 win vs merged origin/main frontier
    win800_path = bl.DATA_DIR / "win_n800.json"
    if win800_path.exists():
        w800 = json.loads(win800_path.read_text(encoding="utf-8"))
        rows800 = sorted(w800["rows"], key=lambda r: int(r["square_id"]))
        x800 = np.array([float(r["x"]) for r in rows800], dtype=np.float64)
        y800 = np.array([float(r["y"]) for r in rows800], dtype=np.float64)
        t800 = np.array([float(r["angle"]) for r in rows800], dtype=np.float64)
        sx, sy, st, _, ok = bslp.joint_xytheta_slp_tight(x800, y800, t800)
        if ok:
            cert800 = bl.polish_and_certify(800, sx, sy, st, frontier[800]["L_dec"], slp_rounds=12)
            if cert800 and cert800["is_real_win"] and max(Decimal(cert800["L_lit"]), Decimal(cert800["L_pandas"])) < max(Decimal(w800["L_lit"]), Decimal(w800["L_pandas"])) - Decimal("1e-9"):
                cert800["label"] = "tight800_slp"
                bl.save_win_locked(cert800)
                print(f"[TRACK 0 WIN] n=800 tightened to L={cert800['L_dec_str'][:20]} dL={cert800['dL']:+.6e}", flush=True)
                x800, y800, t800 = sx, sy, st
        try_upward_l_border(800, x800, y800, t800, frontier)

    # Track C: Standalone brick SVD walks + H34 diagonal pairs + pinwheel broadcast
    packs = {}
    df = pd.read_csv(bl.MERGED_CSV)
    for nv, grp in df.groupby("n"):
        x, y, th, L = bl.normalize(
            grp["x"].to_numpy(float, copy=True),
            grp["y"].to_numpy(float, copy=True),
            grp["angle"].to_numpy(float, copy=True),
        )
        packs[int(nv)] = (x, y, th, L)
    bricks = bslp.extract_subbricks(packs)
    b5_22x8 = extract_b5_22x8(packs)
    if b5_22x8 is not None:
        bricks[(22, 8, 5)] = b5_22x8
    brick_targets = [
        ("B8_18x8", (18, 8, 8)),
        ("B7_17x8", (17, 8, 7)),
        ("B6_14x7", (14, 7, 6)),
        ("B10_22x11", (22, 11, 10)),
        ("B5_22x8", (22, 8, 5)),
        ("B6_24x8", (24, 8, 6)),
        ("B5_22x7", (22, 7, 5)),
    ]
    walked_bricks = []
    b8_res = standalone_brick_walk(10, 18, 8, b8["x"], b8["y"], b8["th"], "B8_10x18_NEW")
    walked_bricks.append(b8_res)
    print(f"[TRACK C BRICK] B8_10x18_NEW: f0={b8_res['f0']:.9f} -> f={b8_res['f']:.9f} df={b8_res['df']:+.6e}", flush=True)
    for bname, bk in brick_targets:
        if bk in bricks:
            b = bricks[bk]
            res = standalone_brick_walk(bk[0], bk[1], bk[2], b["x"], b["y"], b["th"], bname)
            walked_bricks.append(res)
            print(f"[TRACK C BRICK] {bname}: f0={res['f0']:.9f} -> f={res['f']:.9f} df={res['df']:+.6e}", flush=True)
        else:
            print(f"[TRACK C BRICK] {bname}: NOT FOUND", flush=True)
    report["bricks"] = [{k: v for k, v in r.items() if k not in ("x", "y", "th")} for r in walked_bricks]
    improved_bricks = [r for r in walked_bricks if r["df"] <= -1e-6]
    if improved_bricks:
        b_wins = broadcast_improved_bricks(improved_bricks, bricks, frontier)
        for bw in b_wins:
            rows = sorted(bw["rows"], key=lambda r: int(r["square_id"]))
            try_upward_l_border(
                bw["n"],
                np.array([float(r["x"]) for r in rows]),
                np.array([float(r["y"]) for r in rows]),
                np.array([float(r["angle"]) for r in rows]),
                frontier,
            )

    # Track B: Localized Sub-Brick SVD Walk + H34 across SCALE_TARGETS
    for n in SCALE_TARGETS:
        tn = time.time()
        inc_dec = frontier[n]["L_dec"]
        cur_x, cur_y, cur_th = frontier[n]["x"], frontier[n]["y"], frontier[n]["th"]
        wn = bl.DATA_DIR / f"win_n{n}.json"
        if wn.exists():
            wd = json.loads(wn.read_text(encoding="utf-8"))
            inc_dec = min(inc_dec, max(Decimal(wd["L_lit"]), Decimal(wd["L_pandas"])))
            rs = sorted(wd["rows"], key=lambda r: int(r["square_id"]))
            cur_x = np.array([float(r["x"]) for r in rs], dtype=np.float64)
            cur_y = np.array([float(r["y"]) for r in rs], dtype=np.float64)
            cur_th = np.array([float(r["angle"]) for r in rs], dtype=np.float64)

        best_cert_n = None
        for rnd in range(2):
            masks = build_masks_for_n(cur_x, cur_y, cur_th)
            improved = False
            for mtag, fmask in masks:
                cert, nx, ny, nt, _ = localized_walk_dual(
                    n, cur_x, cur_y, cur_th, fmask, inc_dec, tag=f"loc_r{rnd}_{mtag}", seed_base=82000 + rnd * 1000
                )
                if cert is not None and (best_cert_n is None or cert["L_float"] < best_cert_n["L_float"]):
                    best_cert_n = cert
                    if cert["is_real_win"]:
                        cur_x, cur_y, cur_th = nx, ny, nt
                        improved = True
            if not improved:
                break

        if best_cert_n is not None and best_cert_n["is_real_win"]:
            saved = bl.save_win_locked(best_cert_n)
            print(f"[TRACK B WIN] n={n} {best_cert_n['label']} L={best_cert_n['L_dec_str'][:20]} dL={best_cert_n['dL']:+.6e} saved={saved} ({time.time()-tn:.1f}s)", flush=True)
            rows = sorted(best_cert_n["rows"], key=lambda r: int(r["square_id"]))
            try_upward_l_border(
                n,
                np.array([float(r["x"]) for r in rows]),
                np.array([float(r["y"]) for r in rows]),
                np.array([float(r["angle"]) for r in rows]),
                frontier,
            )
        else:
            msg = f"L={best_cert_n['L_dec_str'][:18]} dL={best_cert_n['dL']:+.3e}" if best_cert_n else "no_feas"
            print(f"[TRACK B MISS] n={n} {msg} ({time.time()-tn:.1f}s)", flush=True)
        report["tracks"][f"B_{n}"] = {
            "win": bool(best_cert_n and best_cert_n["is_real_win"]),
            "L": best_cert_n["L_dec_str"] if best_cert_n else None,
            "dL": best_cert_n["dL"] if best_cert_n else None,
            "seconds": round(time.time() - tn, 2),
        }

    (bl.DOCS_DIR / "scale_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"SCALE COMPLETE in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
