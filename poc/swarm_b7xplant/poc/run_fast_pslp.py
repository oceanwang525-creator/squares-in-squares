#!/usr/bin/env python3
"""Hypothesis H77 Fast pslp.squeeze + shake_squeeze + lpslack Pass (~3 min).

Applies kink-safe full-row pslp.squeeze + shake_squeeze + lpslack to:
- Existing wins: win_n800, win_n914, win_n915
- Direct zero-overlap pinwheel candidates: n=855 (8x18+20x11), n=976 (11x19+20x11)
"""
from __future__ import annotations

import io
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

import numpy as np
import pandas as pd

POC_DIR = Path(__file__).resolve().parent
RATCHET_POC = POC_DIR.parents[2] / "poc" / "row_ratchet_980" / "poc"
for _p in (POC_DIR, RATCHET_POC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import b7_lib as b7  # noqa: E402
import b8_lib as bl  # noqa: E402
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


def fast_pslp_certify(n: int, x: np.ndarray, y: np.ndarray, th: np.ndarray, inc_dec: Decimal, label: str) -> dict | None:
    m = int(math.isqrt(n))
    x0, y0, t0, L0 = bl.normalize(x, y, th)
    fp, Xp, Yp, Tp, vp, _ = pslp.squeeze(x0, y0, t0, m, m, L0 - m + 1e-12, max_it=350)
    if vp <= 1e-12 and m + fp <= L0 + 1e-12:
        f_sh, Xs, Ys, Ts = pslp.shake_squeeze(Xp, Yp, Tp, m, m, fp, rounds=6, rng=np.random.default_rng(770 + n))
        if f_sh < fp:
            Xp, Yp, Tp = Xs, Ys, Ts
        x, y, th = Xp, Yp, Tp
    best = b7.polish_and_certify_b7(n, x, y, th, inc_dec, slp_rounds=16, run_lpslack=True)
    if best is None:
        return None
    best["label"] = label
    rows_in = sorted(best["rows"], key=lambda r: int(r["square_id"]))
    L_now = Decimal(best["L_dec_str"])
    if L_now < inc_dec + Decimal("1e-4"):
        lp_res = lpslack.optimize_group(
            n, rows_in, str(L_now), outer_iters=12, fixed_rounds=10,
            slp_modes=("dual_non90", "non90"), tr_th_schedule=(6e-4, 1.5e-4, 4e-5), time_limit=20.0,
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
                        "label": f"{label}_pslp_lp",
                    }
    return best


def main() -> None:
    t0 = time.time()
    ns = [800, 914, 915, 855, 976]
    port = b7.load_portfolio_targets(ns)

    # 1. Existing wins
    for wn in (800, 914, 915):
        wp = b7.DATA_DIR / f"win_n{wn}.json"
        if not wp.exists():
            continue
        d = json.loads(wp.read_text(encoding="utf-8"))
        rs = sorted(d["rows"], key=lambda r: int(r["square_id"]))
        x = np.array([float(r["x"]) for r in rs], dtype=np.float64)
        y = np.array([float(r["y"]) for r in rs], dtype=np.float64)
        th = np.array([float(r["angle"]) for r in rs], dtype=np.float64)
        inc_cur = min(port[wn]["L_dec"], Decimal(d["L_dec_str"]))
        c = fast_pslp_certify(wn, x, y, th, inc_cur, f"fast_pslp_n{wn}")
        if c is not None:
            print(f"[WIN POLISH n={wn}] {c['label']}: L={c['L_dec_str'][:26]} dL={c['dL']:+.6e} win={c['is_real_win']}", flush=True)
            if c["is_real_win"]:
                b7.save_win_b7(c)

    # 2. Direct pinwheel fits on n=855 and n=976
    pair_specs = [
        (855, 29, load_brick("B6_8x18_n855_cut"), load_brick("B8_20x11_n855"), "855_8x18+20x11"),
        (855, 29, bl.transform_brick(load_brick("B6_8x18_n855_cut"), "T"), bl.transform_brick(load_brick("B8_20x11_n855"), "T"), "855_18x8+11x20"),
        (976, 31, load_brick("B7_11x19_n975_cut"), load_brick("B8_20x11_n855"), "976_11x19+20x11"),
        (976, 31, bl.transform_brick(load_brick("B7_11x19_n975_cut"), "T"), bl.transform_brick(load_brick("B8_20x11_n855"), "T"), "976_19x11+11x20"),
    ]
    for n, m, b1_base, b2_base, tag in pair_specs:
        inc = port[n]["L_dec"]
        for m1 in ("id", "flipx", "flipy", "rot180"):
            b1 = bl.transform_brick(b1_base, m1)
            for m2 in ("id", "flipx", "flipy", "rot180"):
                if (m1, m2) not in (("id", "id"), ("flipx", "flipy"), ("rot180", "rot180")):
                    continue
                b2 = bl.transform_brick(b2_base, m2)
                f_hi = max(b1["fx"], b1["fy"], b2["fx"], b2["fy"])
                ax, ay, at, _, _ = h60.build_pinwheel_custom(m, b1, b2, sf=0.0, f_override=f_hi)
                if len(ax) != n:
                    continue
                pen, _ = bl.cpp_eval_sat_pen_bbox(ax, ay, at)
                if pen <= 1e-10:
                    c = fast_pslp_certify(n, ax, ay, at, inc, f"{tag}_{m1}_{m2}")
                    if c is not None:
                        print(f"[DIRECT n={n}] {c['label']}: L={c['L_dec_str'][:26]} dL={c['dL']:+.6e} win={c['is_real_win']}", flush=True)
                        if c["is_real_win"]:
                            b7.save_win_b7(c)
                            inc = min(inc, Decimal(c["L_dec_str"]))

    print(f"[FAST PSLP COMPLETE] in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
