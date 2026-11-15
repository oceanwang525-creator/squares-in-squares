#!/usr/bin/env python3
"""Hypothesis H77 Focused Round 2:
1. Test f_hi-prioritized zero-overlap & 1-corner-touch pinwheels missed in Round 1:
   - n=976 (31,+15): B7_11x19_n975_cut (0.721540) + B8_20x11_n855 (0.733980)
   - n=915 (30,+15): B6_7x18_n915 (0.738152) + B9_22x12_n915 (0.740334)
   - n=916 (30,+16): B9_21x11_n916 (0.750957) + B7_9x19_n916 (0.750374) (corner melt)
                     B9_21x11_n916 (0.750957) + B7_8x19_n916 (0.752190)
   - n=855 (29,+14): B6_8x18_n855_cut (0.732065) + B8_20x11_n855 (0.733980)
2. Deep polish (lpslack time_limit=30s, outer_iters=14 + SVD mechwalk_compress) on all tight candidates & win_n800.
"""
from __future__ import annotations

import importlib.util as _ilu
import json
import math
import os
import sys
import time
from decimal import Decimal
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import io
import numpy as np
import pandas as pd

POC_DIR = Path(__file__).resolve().parent
RATCHET_POC = POC_DIR.parents[2] / "poc" / "row_ratchet_980" / "poc"
for _p in (POC_DIR, RATCHET_POC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import b7_lib as b7  # noqa: E402
import b8_lib as bl  # noqa: E402
import brick_rect_lib as bslp  # noqa: E402
import lpslack  # noqa: E402
import pslp  # noqa: E402
import run_h60_pingpong as h60  # noqa: E402
import union as auto_union  # noqa: E402


def load_brick(name: str) -> dict:
    d = json.loads((b7.DATA_DIR / f"brick_{name}.json").read_text(encoding="utf-8"))
    return {
        "name": name, "tag": name, "w": int(d["w"]), "h": int(d["h"]), "g": int(d["g"]),
        "src_n": int(d.get("src_n", 0)), "f0": float(d.get("f0", d["f"])),
        "fx": float(d["fx"]), "fy": float(d["fy"]), "f": float(d["f"]), "df": float(d.get("df", 0.0)),
        "x": np.array(d["x"], dtype=np.float64), "y": np.array(d["y"], dtype=np.float64),
        "th": np.array(d["th"], dtype=np.float64),
    }


def deep_polish_and_certify(n: int, x: np.ndarray, y: np.ndarray, th: np.ndarray, inc_dec: Decimal, label: str) -> dict | None:
    m = int(math.isqrt(n))
    x0, y0, t0, L0 = bl.normalize(x, y, th)
    fp, Xp, Yp, Tp, vp, _ = pslp.squeeze(x0, y0, t0, m, m, L0 - m + 1e-12, max_it=250)
    if vp <= 1e-12 and m + fp < L0:
        f_sh, Xs, Ys, Ts = pslp.shake_squeeze(Xp, Yp, Tp, m, m, fp, rounds=4, rng=np.random.default_rng(77 + n))
        if f_sh < fp:
            Xp, Yp, Tp = Xs, Ys, Ts
        x, y, th = Xp, Yp, Tp
    best = b7.polish_and_certify_b7(n, x, y, th, inc_dec, slp_rounds=24, run_lpslack=True)
    if best is None:
        return None
    best["label"] = label
    rows_in = sorted(best["rows"], key=lambda r: int(r["square_id"]))
    L_now = Decimal(best["L_dec_str"])
    if L_now < inc_dec + Decimal("5e-4"):
        lp_res = lpslack.optimize_group(
            n, rows_in, str(L_now),
            outer_iters=14, fixed_rounds=12,
            slp_modes=("dual_non90", "non90"),
            tr_th_schedule=(1e-3, 3e-4, 8e-5, 2e-5),
            time_limit=28.0,
        )
        if lp_res.get("rows"):
            rs_lp = sorted(lp_res["rows"], key=lambda r: int(r["square_id"]))
            text = auto_union.final_text({n: rs_lp})
            lit = auto_union.parse_text(text)
            pd_t = auto_union.parse_text(pd.read_csv(io.StringIO(text)).to_csv(index=False))
            res_chk = auto_union.check((n, lit[n], pd_t[n]))
            if res_chk is not None:
                L_lp = max(Decimal(res_chk[0]), Decimal(res_chk[1]))
                if L_lp < L_now:
                    dL = L_lp - inc_dec
                    best = {
                        "ok": True, "n": int(n), "L_lit": res_chk[0], "L_pandas": res_chk[1],
                        "L_dec_str": str(L_lp), "L_float": float(L_lp), "rows": rs_lp,
                        "dL": float(dL), "is_real_win": bool(dL <= Decimal("-1e-9")),
                        "label": f"{label}_deeplp",
                    }
    return best


def run_pair(n: int, m: int, b1_base: dict, b2_base: dict, inc_dec: Decimal, tag: str, seed: int) -> dict | None:
    best: dict | None = None
    seeds, masks, scheds, labels = [], [], [], []
    f_inc = float(inc_dec) - m

    for m1 in ("id", "flipx", "flipy", "rot180"):
        b1 = bl.transform_brick(b1_base, m1)
        for m2 in ("id", "flipx", "flipy", "rot180"):
            b2 = bl.transform_brick(b2_base, m2)
            f_hi = max(b1["fx"], b1["fy"], b2["fx"], b2["fy"])
            f_avg = 0.5 * (max(b1["fx"], b1["fy"]) + max(b2["fx"], b2["fy"]))
            for sf in (0.0, 0.5):
                for f_use, ftag in ((f_hi, "fhi"), (f_avg, "favg")):
                    ax, ay, at, L0, sm = h60.build_pinwheel_custom(m, b1, b2, sf=sf, f_override=f_use)
                    if len(ax) != n:
                        continue
                    pen, _ = bl.cpp_eval_sat_pen_bbox(ax, ay, at)
                    if pen <= 1e-10 and ftag == "fhi" and sf == 0.0 and (m1, m2) in (("id", "id"), ("flipx", "flipy"), ("flipx", "id"), ("id", "flipy"), ("rot180", "rot180")):
                        c = deep_polish_and_certify(n, ax, ay, at, inc_dec, f"{tag}_{m1}_{m2}_direct")
                        if c is not None:
                            print(f"    [{c['label']}] L={c['L_dec_str'][:22]} dL={c['dL']:+.6e} win={c['is_real_win']}", flush=True)
                            if best is None or Decimal(c["L_dec_str"]) < Decimal(best["L_dec_str"]):
                                best = c
                            if c["is_real_win"]:
                                b7.save_win_b7(c)
                                inc_dec = min(inc_dec, Decimal(c["L_dec_str"]))
                    seeds.append((ax, ay, at))
                    masks.append(sm)
                    L_tgt = m + min(f_use, f_inc)
                    scheds.append([L_tgt + 0.008, L_tgt + 0.0015, L_tgt - 2e-4])
                    labels.append(f"{tag}_{m1}_{m2}_sf{sf}_{ftag}")

    if seeds:
        for siter in (200, 0):
            relaxed = bl.su.cpp_batch_melt(
                seeds[:32], scheds[:32], sym_mode=1, a_start=0.955, n_melt_stages=3,
                seam_masks=masks[:32] if siter > 0 else None, seam_iters=siter,
                full_iters=320, n_ruin_iters=0, n_threads=2, base_seed=seed + siter,
            )
            pool = []
            for bi, row in enumerate(relaxed):
                nx, ny, nth, Lm = bl.normalize(row["x"], row["y"], row["th"])
                lbl = f"{labels[bi]}_si{siter}"
                if row["best_L_feas"] is not None:
                    pool.append((0.0, Lm, nx, ny, nth, lbl))
                elif row["pen"] < 0.02:
                    pool.append((1.0 + float(row["pen"]), Lm, nx, ny, nth, lbl))
            pool.sort(key=lambda t: (t[0], t[1]))
            for _, _, cx, cy, cth, lbl in pool[:3]:
                c = deep_polish_and_certify(n, cx, cy, cth, inc_dec, lbl)
                if c is not None:
                    print(f"    [{c['label']}] L={c['L_dec_str'][:22]} dL={c['dL']:+.6e} win={c['is_real_win']}", flush=True)
                    if best is None or Decimal(c["L_dec_str"]) < Decimal(best["L_dec_str"]):
                        best = c
                    if c["is_real_win"]:
                        b7.save_win_b7(c)
                        inc_dec = min(inc_dec, Decimal(c["L_dec_str"]))
    return best


def main() -> None:
    t0 = time.time()
    ns = [800, 855, 976, 915, 916, 914]
    port = b7.load_portfolio_targets(ns)

    # 1. Deep polish existing win_n800 and win_n914
    for wn in (800, 914):
        wp = b7.DATA_DIR / f"win_n{wn}.json"
        if wp.exists():
            d = json.loads(wp.read_text(encoding="utf-8"))
            rs = sorted(d["rows"], key=lambda r: int(r["square_id"]))
            x = np.array([float(r["x"]) for r in rs], dtype=np.float64)
            y = np.array([float(r["y"]) for r in rs], dtype=np.float64)
            th = np.array([float(r["angle"]) for r in rs], dtype=np.float64)
            inc_cur = min(port[wn]["L_dec"], Decimal(d["L_dec_str"]))
            print(f"[ROUND2 POLISH] win_n{wn} current={str(inc_cur)[:24]}", flush=True)
            c = deep_polish_and_certify(wn, x, y, th, inc_cur, f"repolish_n{wn}")
            if c is not None:
                print(f"  -> {c['label']}: L={c['L_dec_str'][:24]} dL={c['dL']:+.6e} win={c['is_real_win']}", flush=True)
                if c["is_real_win"]:
                    b7.save_win_b7(c)

    # 2. Focused missed pairs (remaining unrun from Round 2)
    jobs = [
        (915, 30, load_brick("B6_7x18_n915"), load_brick("B9_22x12_n915"), "B6_7x18+B9_22x12"),
        (915, 30, bl.transform_brick(load_brick("B6_7x18_n915"), "T"), bl.transform_brick(load_brick("B9_22x12_n915"), "T"), "B6_18x7+B9_12x22"),
        (855, 29, load_brick("B6_8x18_n855_cut"), load_brick("B8_20x11_n855"), "B6_8x18+B8_20x11"),
        (916, 30, load_brick("B7_8x19_n916"), load_brick("B9_21x11_n916"), "B7_8x19+B9_21x11"),
        (916, 30, load_brick("B7_9x19_n916"), load_brick("B9_21x11_n916"), "B7_9x19+B9_21x11_touch"),
        (976, 31, bl.transform_brick(load_brick("B7_11x19_n975_cut"), "T"), bl.transform_brick(load_brick("B8_20x11_n855"), "T"), "B7_19x11+B8_11x20"),
    ]
    out = {}
    for idx, (n, m, b1, b2, lbl) in enumerate(jobs):
        inc = port[n]["L_dec"]
        wp = b7.DATA_DIR / f"win_n{n}.json"
        if wp.exists():
            inc = min(inc, Decimal(json.loads(wp.read_text(encoding="utf-8"))["L_dec_str"]))
        print(f"\n[ROUND2 PAIR] n={n} {lbl} inc={str(inc)[:24]}", flush=True)
        res = run_pair(n, m, b1, b2, inc, f"n{n}_{lbl}", seed=99000 + n * 19 + idx * 41)
        out[f"{n}_{lbl}"] = {
            "L": res["L_dec_str"] if res else None,
            "dL": res["dL"] if res else None,
            "win": bool(res and res["is_real_win"]),
            "label": res.get("label") if res else None,
        }
    (b7.DOCS_DIR / "scale2_report.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n[H77 ROUND 2 COMPLETE] in {time.time()-t0:.1f}s -> {json.dumps(out, indent=2)}", flush=True)


if __name__ == "__main__":
    main()
