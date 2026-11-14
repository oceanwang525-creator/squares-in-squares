#!/usr/bin/env python3
"""Fast pinwheel test using B7_10x22_vstrip (f=0.688507) + B6 bricks on n=974, 913, 973, 912, 854."""
from __future__ import annotations

import json
import sys
import time
from decimal import Decimal
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gstep_lib as gl  # noqa: E402

REPO_ROOT = gl.REPO_ROOT
sys.path.insert(0, str(REPO_ROOT / "poc" / "swarm_lpslack" / "poc"))
import lpslack  # noqa: E402
from cpp_wrap import cpp_eval_sat_pen_bbox  # noqa: E402


def build_pinwheel_zero_ov(
    m: int, b1: dict, b2: dict, sf: float = 0.0, f_override: float | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray] | None:
    w1, h1, w2, h2 = b1["w"], b1["h"], b2["w"], b2["h"]
    if w1 >= m or h1 >= m or w2 >= m or h2 >= m:
        return None
    if w1 + w2 > m or h1 + h2 > m:
        return None
    if w1 + w2 + h1 + h2 > 2 * m - 1:
        return None
    f = max(b1["f"], b2["f"]) if f_override is None else f_override
    L = m + f
    dx1 = sf * (f - b1["fx"])
    dy1 = (m - h1) + (1.0 - sf) * (f - b1["fy"])
    dx2 = (m - w2) + (1.0 - sf) * (f - b2["fx"])
    dy2 = sf * (f - b2["fy"])

    gx, gy = [], []
    for i in range(m):
        for j in range(m):
            if i < w1 and j >= m - h1:
                continue
            if i >= m - w2 and j < h2:
                continue
            if i < m - w2 and j < m - h1:
                gx.append(i + 0.5)
                gy.append(j + 0.5)
            else:
                gx.append(i + f + 0.5)
                gy.append(j + f + 0.5)

    ax = np.concatenate([b1["x"] + dx1, b2["x"] + dx2, np.array(gx, dtype=np.float64)])
    ay = np.concatenate([b1["y"] + dy1, b2["y"] + dy2, np.array(gy, dtype=np.float64)])
    at = np.concatenate([b1["th"], b2["th"], np.zeros(len(gx), dtype=np.float64)])
    if len(ax) != m * m + b1["g"] + b2["g"]:
        return None
    seam = (
        ((ax >= w1 - 1.5) & (ax <= w1 + f + 1.8))
        | ((ay >= m - h1 - 1.8) & (ay <= m - h1 + 1.8))
        | ((ax >= m - w2 - 1.8) & (ax <= m - w2 + 1.8))
        | ((ay >= h2 - 1.5) & (ay <= h2 + f + 1.8))
    ).astype(np.uint8)
    return ax, ay, at, L, seam


def solve_pair(n: int, m: int, b1_base: dict, b2_base: dict, inc_dec: Decimal, tag: str, base_seed: int) -> dict | None:
    best_cert: dict | None = None
    seeds, masks, scheds, labels = [], [], [], []
    for m1 in ("id", "flipx", "flipy", "rot180"):
        b1 = gl.bl.transform_brick(b1_base, m1)
        for m2 in ("id", "flipx", "flipy", "rot180"):
            b2 = gl.bl.transform_brick(b2_base, m2)
            f_hi = max(b1["fx"], b1["fy"], b2["fx"], b2["fy"])
            f_lo = min(b1["fx"], b1["fy"], b2["fx"], b2["fy"])
            f_avg = 0.5 * (max(b1["fx"], b1["fy"]) + max(b2["fx"], b2["fy"]))
            f_list = ((f_hi, "fhi"),) if abs(f_hi - f_lo) < 5e-4 else ((f_lo, "flo"), (f_avg, "favg"), (f_hi, "fhi"))
            sf_list = (0.0, 0.5) if abs(f_hi - f_lo) >= 5e-4 else (0.0,)
            for sf in sf_list:
                for f_use, ftag in f_list:
                    built = build_pinwheel_zero_ov(m, b1, b2, sf=sf, f_override=f_use)
                    if built is None:
                        continue
                    ax, ay, at, L0, sm = built
                    pen, _ = cpp_eval_sat_pen_bbox(ax, ay, at)
                    if pen <= 1e-10 and (m1, m2) in (("id", "id"), ("flipx", "flipy"), ("rot180", "rot180")) and ftag == "fhi":
                        sx, sy, st, _, ok = gl.bslp.joint_xytheta_slp_tight(ax, ay, at)
                        if ok:
                            cert = gl.bl.polish_and_certify(n, sx, sy, st, inc_dec, slp_rounds=22)
                            if cert is not None:
                                cert["label"] = f"{tag}_{m1}_{m2}_sf{sf}_direct"
                                if best_cert is None or cert["L_float"] < best_cert["L_float"]:
                                    best_cert = cert
                    if (m1, m2) in (("id", "id"), ("flipx", "id"), ("id", "flipx"), ("flipx", "flipy"), ("rot180", "rot180")):
                        seeds.append((ax, ay, at))
                        masks.append(sm)
                        L_tgt = m + f_use
                        scheds.append([L_tgt + 0.008, L_tgt + 0.001, L_tgt - 3e-4])
                        labels.append(f"{tag}_{m1}_{m2}_sf{sf}_{ftag}")

    if seeds:
        for siter in (160, 0):
            for start in range(0, len(seeds), 64):
                end = min(len(seeds), start + 64)
                relaxed = gl.bl.su.cpp_batch_melt(
                    seeds[start:end], scheds[start:end], sym_mode=1, a_start=0.965, n_melt_stages=3,
                    seam_masks=masks[start:end] if siter > 0 else None, seam_iters=siter,
                    full_iters=260, n_ruin_iters=0, n_threads=8, base_seed=base_seed + siter + start,
                )
                pool = []
                for bi, row in enumerate(relaxed):
                    nx, ny, nth, Lm = gl.bl.normalize(row["x"], row["y"], row["th"])
                    lbl = f"{labels[start + bi]}_si{siter}"
                    if row["best_L_feas"] is not None:
                        pool.append((0.0, Lm, nx, ny, nth, lbl))
                    elif row["pen"] < 0.015:
                        pool.append((1.0 + float(row["pen"]), Lm, nx, ny, nth, lbl))
                pool.sort(key=lambda t: (t[0], t[1]))
                for _, _, cx, cy, cth, lbl in pool[:4]:
                    sx, sy, st, _, ok = gl.bslp.joint_xytheta_slp_tight(cx, cy, cth)
                    use_x, use_y, use_t = (sx, sy, st) if ok else (cx, cy, cth)
                    cert = gl.bl.polish_and_certify(n, use_x, use_y, use_t, inc_dec, slp_rounds=22)
                    if cert is not None:
                        cert["label"] = lbl
                        if best_cert is None or cert["L_float"] < best_cert["L_float"]:
                            best_cert = cert

    # Also run lpslack polish on best_cert if feasible
    if best_cert is not None:
        rs = sorted(best_cert["rows"], key=lambda r: int(r["square_id"]))
        res_lp = lpslack.optimize_group(
            n, rs, str(inc_dec), outer_iters=6, fixed_rounds=10,
            slp_modes=("dual_non90", "non90"), tr_th_schedule=(4e-4, 1e-4), time_limit=15.0,
        )
        if res_lp.get("win") and res_lp.get("rows"):
            lit, pan = Decimal(res_lp["new_L_lit"]), Decimal(res_lp["new_L_pan"])
            worst = max(lit, pan)
            if worst < max(Decimal(best_cert["L_lit"]), Decimal(best_cert["L_pandas"])):
                best_cert["L_lit"] = str(lit)
                best_cert["L_pandas"] = str(pan)
                best_cert["L_dec_str"] = str(worst)
                best_cert["L_float"] = float(worst)
                best_cert["dL"] = float(worst - inc_dec)
                best_cert["is_real_win"] = bool(worst - inc_dec <= Decimal("-1e-9"))
                best_cert["rows"] = res_lp["rows"]
                best_cert["label"] = best_cert["label"] + "+lpslack"
    return best_cert


def main() -> None:
    t0 = time.time()
    frontier = gl.fast_load_true_frontier([974, 973, 913, 912, 853, 854], certify=True)
    for n, info in sorted(frontier.items()):
        print(f"Frontier n={n}: L={info['L_dec']} ({info['source']})", flush=True)

    bricks = gl.load_source_bricks()
    b7_path = gl.DATA_DIR / "brick_B7_10x22_vstrip.json"
    if not b7_path.exists():
        raise FileNotFoundError(b7_path)
    b7_10x22 = gl.load_brick_json(b7_path, "B7_10x22_vstrip")
    b7_22x10 = gl.bl.transform_brick(b7_10x22, "T")

    # Test n=974 (m=31, r=+13): B6_9x18 (9x18, f=0.705887) + B7_22x10 (22x10, f=0.688507)
    b6_9x18 = bricks["B6_9x18_n796_TL"]
    c974 = solve_pair(974, 31, b6_9x18, b7_22x10, frontier[974]["L_dec"], "gstep_974_9x18_22x10", 97400)
    if c974 is not None:
        print(f"[n=974] {c974['label']} L={c974['L_dec_str'][:20]} dL={c974['dL']:+.6e} win={c974['is_real_win']}", flush=True)
        if c974["is_real_win"]:
            gl.save_win_locked(c974)
    else:
        print("[n=974] NO FEASIBLE", flush=True)
    print(f"Elapsed {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
