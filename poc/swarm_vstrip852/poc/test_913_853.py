#!/usr/bin/env python3
"""Gate test on n=913 and n=853 with micro-expansion + seam melt + lpslack + pslp + mechwalk."""
from __future__ import annotations

import json
import os
import sys
from decimal import Decimal
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vstrip_lib as vl  # noqa: E402
import gstep_lib as gl  # noqa: E402
import build_and_polish as bp  # noqa: E402
import homocasc2_lib as hlib  # noqa: E402
import lpslack  # noqa: E402

sys.path.insert(0, str(vl.REPO_ROOT / "poc" / "row_ratchet_980" / "poc"))
import pslp  # noqa: E402


def build_pinwheel_any(m: int, b1: dict, b2: dict, sf: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray] | None:
    w1, h1, w2, h2 = b1["w"], b1["h"], b2["w"], b2["h"]
    if w1 >= m or h1 >= m or w2 >= m or h2 >= m:
        return None
    if w1 + w2 > m or h1 + h2 > m:
        return None
    if w1 + w2 + h1 + h2 > 2 * m:
        return None
    f = max(b1["f"], b2["f"])
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
        ((ax >= w1 - 2.2) & (ax <= w1 + f + 2.2))
        | ((ay >= m - h1 - 2.2) & (ay <= m - h1 + 2.2))
        | ((ax >= m - w2 - 2.2) & (ax <= m - w2 + 2.2))
        | ((ay >= h2 - 2.2) & (ay <= h2 + f + 2.2))
    ).astype(np.uint8)
    return ax, ay, at, L, seam


def safe_certify(n: int, x: np.ndarray, y: np.ndarray, th: np.ndarray, inc_dec: Decimal, label: str = "") -> dict | None:
    x, y, th, _ = hlib.norm_pack(x, y, th)
    for eps in (0.0, 1e-12, 1e-10, 1e-8, 1e-7):
        sc = 1.0 + eps
        c = hlib.certify_pack_against_frontier(n, x * sc, y * sc, th, inc_dec)
        if c is not None:
            c["label"] = label
            return c
    return None


def polish_pack(n: int, cert: dict, inc_dec: Decimal, rng: np.random.Generator, label: str) -> dict:
    best = cert
    best_L = Decimal(max(best["L_lit"], best["L_pd"]))
    bx, by, bt, _ = hlib._rows_to_arrays(best["rows"])

    # 1. lpslack
    c_lp = bp._run_lpslack_fast(
        n, bx, by, bt, inc_dec, cycles=2, outer_iters=10, fixed_rounds=8, slp_modes=("dual_non90", "non90"), time_limit=15.0
    )
    if c_lp is not None and Decimal(max(c_lp["L_lit"], c_lp["L_pd"])) < best_L:
        best = c_lp
        best_L = Decimal(max(best["L_lit"], best["L_pd"]))
        best["label"] = label + "+lpslack"
        bx, by, bt, _ = hlib._rows_to_arrays(best["rows"])
        print(f"    [lpslack] n={n} L={str(best_L)[:18]} dL={best['dL']:+.6e}", flush=True)

    # 2. pslp full-row kink-safe LP squeeze
    try:
        xs, ys, ts, _, _, _, ok = pslp.pslp(bx, by, bt, rounds=14, tr0=1e-3, margin=1e-9)
        if ok:
            c_ps = safe_certify(n, xs, ys, ts, inc_dec, label + "+pslp")
            if c_ps is not None and Decimal(max(c_ps["L_lit"], c_ps["L_pd"])) < best_L:
                best = c_ps
                best_L = Decimal(max(best["L_lit"], best["L_pd"]))
                bx, by, bt, _ = hlib._rows_to_arrays(best["rows"])
                print(f"    [pslp] n={n} L={str(best_L)[:18]} dL={best['dL']:+.6e}", flush=True)
    except Exception as exc:
        print(f"    [pslp warn] {exc}", flush=True)

    # 3. mechwalk + lpslack
    mw_cert, _ = hlib.mechwalk_compress(n, bx, by, bt, inc_dec, rng, rounds=2, threads=1)
    if mw_cert is not None and Decimal(max(mw_cert["L_lit"], mw_cert["L_pd"])) < best_L:
        best = mw_cert
        best_L = Decimal(max(best["L_lit"], best["L_pd"]))
        best["label"] = label + "+mechwalk"
        bx, by, bt, _ = hlib._rows_to_arrays(best["rows"])
        print(f"    [mechwalk] n={n} L={str(best_L)[:18]} dL={best['dL']:+.6e}", flush=True)
        c_lp2 = bp._run_lpslack_fast(
            n, bx, by, bt, inc_dec, cycles=2, outer_iters=10, fixed_rounds=8, slp_modes=("dual_non90", "non90"), time_limit=15.0
        )
        if c_lp2 is not None and Decimal(max(c_lp2["L_lit"], c_lp2["L_pd"])) < best_L:
            best = c_lp2
            best_L = Decimal(max(best["L_lit"], best["L_pd"]))
            best["label"] = label + "+mw+lpslack"
            print(f"    [mw+lp] n={n} L={str(best_L)[:18]} dL={best['dL']:+.6e}", flush=True)

    return best


def main() -> None:
    targets = [913, 853]
    frontier = gl.fast_load_true_frontier(targets, certify=True)
    for n in targets:
        print(f"Incumbent n={n}: L={frontier[n]['L_dec']} ({frontier[n]['source']})", flush=True)

    b_18x9 = gl.bl.transform_brick(vl.load_brick(vl.DATA_DIR / "brick_n852_9x18_vs_1.json", "n852_9x18_vs_1"), "T")
    b_18x9["tag"] = "n852_18x9_vs1"
    b_9x18 = vl.load_brick(vl.DATA_DIR / "brick_n852_9x18_vs_1.json", "n852_9x18_vs_1")
    b_11x21 = vl.load_brick(vl.DATA_DIR / "brick_n852_11x21_vs_0.json", "n852_11x21_vs_0")
    b_20x11 = gl.bl.transform_brick(vl.load_brick(vl.DATA_DIR / "brick_base_n852_11x20.json", "base_n852_11x20"), "T")
    b_20x11["tag"] = "n852_20x11"

    # 1. n=913 zero-overlap
    inc_913 = frontier[913]["L_dec"]
    rng = np.random.default_rng(91301)
    best_913 = None
    for m1 in ("id", "flipx", "flipy", "rot180"):
        b1 = gl.bl.transform_brick(b_18x9, m1)
        for m2 in ("id", "flipx", "flipy", "rot180"):
            b2 = gl.bl.transform_brick(b_11x21, m2)
            ax, ay, at, _, _ = build_pinwheel_any(30, b1, b2, sf=0.0)
            c = safe_certify(913, ax, ay, at, inc_913, f"n913_{m1}_{m2}")
            if c is not None:
                c = polish_pack(913, c, inc_913, rng, f"n913_{m1}_{m2}")
                if best_913 is None or Decimal(max(c["L_lit"], c["L_pd"])) < Decimal(max(best_913["L_lit"], best_913["L_pd"])):
                    best_913 = c
                break
        if best_913 is not None:
            break
    if best_913 is not None:
        print(f"FINAL n=913: L_lit={best_913['L_lit']} L_pd={best_913['L_pd']} dL={best_913['dL']:+.6e} win={best_913['is_real_win']}", flush=True)
        if best_913["is_real_win"]:
            (vl.DATA_DIR / "win_n913.json").write_text(json.dumps({
                "n": 913, "label": best_913["label"], "L_lit": best_913["L_lit"], "L_pd": best_913["L_pd"],
                "dL": best_913["dL"], "rows": best_913["rows"],
            }, indent=1))

    # 2. n=853 1-corner melted pinwheel
    inc_853 = frontier[853]["L_dec"]
    rng853 = np.random.default_rng(85301)
    seeds, masks, scheds, labels = [], [], [], []
    for m1 in ("id", "flipx", "flipy", "rot180"):
        b1 = gl.bl.transform_brick(b_9x18, m1)
        for m2 in ("id", "flipx", "flipy", "rot180"):
            b2 = gl.bl.transform_brick(b_20x11, m2)
            ax, ay, at, L_tgt, sm = build_pinwheel_any(29, b1, b2, sf=0.0)
            seeds.append((ax, ay, at))
            masks.append(sm)
            scheds.append([L_tgt + 0.020, L_tgt + 0.005, L_tgt + 0.0005])
            labels.append(f"n853_{m1}_{m2}")

    relaxed = gl.bl.su.cpp_batch_melt(
        seeds, scheds, sym_mode=1, a_start=0.955, n_melt_stages=3,
        seam_masks=masks, seam_iters=180, full_iters=240, n_ruin_iters=0, n_threads=2, base_seed=85301,
    )
    pool = []
    for bi, row in enumerate(relaxed):
        nx, ny, nth, Lm = gl.bl.normalize(row["x"], row["y"], row["th"])
        if row["best_L_feas"] is not None:
            pool.append((0.0, Lm, nx, ny, nth, labels[bi]))
        elif row["pen"] < 0.05:
            pool.append((1.0 + float(row["pen"]), Lm, nx, ny, nth, labels[bi]))
    pool.sort(key=lambda t: (t[0], t[1]))
    print(f"n=853 melt pool size={len(pool)}, top={[(round(t[0],4), round(t[1],6), t[5]) for t in pool[:4]]}", flush=True)
    best_853 = None
    for _, _, cx, cy, cth, lbl in pool[:3]:
        # Try SLP / lpslack
        xs, ys, ts, _, ok = bp.joint_xytheta_slp(cx, cy, cth, rounds=10, tr_th0=0.02, tr_xy0=0.08, margin=1e-8, tol=1e-9, patience=2)
        c = safe_certify(853, xs if ok else cx, ys if ok else cy, ts if ok else cth, inc_853, lbl)
        if c is not None:
            c = polish_pack(853, c, inc_853, rng853, lbl)
            if best_853 is None or Decimal(max(c["L_lit"], c["L_pd"])) < Decimal(max(best_853["L_lit"], best_853["L_pd"])):
                best_853 = c
    if best_853 is not None:
        print(f"FINAL n=853: L_lit={best_853['L_lit']} L_pd={best_853['L_pd']} dL={best_853['dL']:+.6e} win={best_853['is_real_win']}", flush=True)
        if best_853["is_real_win"]:
            (vl.DATA_DIR / "win_n853.json").write_text(json.dumps({
                "n": 853, "label": best_853["label"], "L_lit": best_853["L_lit"], "L_pd": best_853["L_pd"],
                "dL": best_853["dL"], "rows": best_853["rows"],
            }, indent=1))


if __name__ == "__main__":
    main()
