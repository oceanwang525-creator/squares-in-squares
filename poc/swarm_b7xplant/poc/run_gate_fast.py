#!/usr/bin/env python3
"""Fast H77 Gate Runner (CPU-light, <=2 threads): quick walk on B8_12x20_n914 + gate on {854, 914, 975}."""
from __future__ import annotations

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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import b7_lib as b7  # noqa: E402
import b8_lib as bl  # noqa: E402
import brick_rect_lib as bslp  # noqa: E402
import run_h60_pingpong as h60  # noqa: E402

import importlib.util as _ilu
_rg_spec = _ilu.spec_from_file_location("b7_run_gate", Path(__file__).resolve().parent / "run_gate.py")
rg = _ilu.module_from_spec(_rg_spec)
_rg_spec.loader.exec_module(rg)

GATE_NS = [854, 914, 975]


def quick_walk_b8(b: dict, name: str) -> dict:
    """Fast compress_brick_slp + 1-ctol top-5 SVD single & diagonal walk (~30s)."""
    w, h, g = b["w"], b["h"], b["g"]
    f0 = b["f"]
    cur = b7.compress_brick_slp(b, n_trials=6, seed=7799)
    best_f = cur["f"]
    best_state = (cur["x"].copy(), cur["y"].copy(), cur["th"].copy())
    print(f"  [BRICK {name}] pre-SLP: {f0:.9f} -> {best_f:.9f} (df={best_f-f0:+.6e})", flush=True)

    free_all = np.ones(len(best_state[0]), dtype=bool)
    jac, _, _ = bl.localized_contact_jacobian(best_state[0], best_state[1], best_state[2], free_all, contact_tol=1e-5)
    if jac.shape[0] > 0:
        _, raw_modes = bl.localized_soft_modes(jac, k=8)
        kept = [m / np.linalg.norm(m) for m in raw_modes if np.linalg.norm(m) > 1e-10][:5]
        inv_s2 = 1.0 / math.sqrt(2.0)
        dirs = list(kept) + [
            inv_s2 * (kept[i] + s * kept[j])
            for i in range(min(4, len(kept)))
            for j in range(i + 1, min(4, len(kept)))
            for s in (1.0, -1.0)
        ]
        nf = len(best_state[0])
        for di, v in enumerate(dirs):
            for amp in (0.004, 0.02):
                for sgn in (1.0, -1.0):
                    cx = best_state[0] + sgn * amp * v[:nf]
                    cy = best_state[1] + sgn * amp * v[nf : 2 * nf]
                    ct = best_state[2] + sgn * amp * v[2 * nf :]
                    tx, ty, tt, pen = bslp.relax_rect_lbfgs(cx, cy, ct, w + best_f - 2e-4, h + best_f - 2e-4, maxiter=200)
                    if pen <= 1e-6:
                        sx, sy, st, sf, ok = bslp.joint_xytheta_slp_rect(tx, ty, tt, w, h, rounds=8)
                        if ok and sf < best_f - 1e-8:
                            best_f = sf
                            best_state = (sx, sy, st)
                            print(f"  [BRICK {name}] dir={di} amp={sgn*amp} -> f={best_f:.9f} (df={best_f-f0:+.6e})", flush=True)

    bx, by, Wf, Hf = bl.rect_bbox(best_state[0], best_state[1], best_state[2])
    fx, fy = Wf - w, Hf - h
    best_f = max(fx, fy)
    out_p = b7.DATA_DIR / f"brick_{name}.json"
    out_p.write_text(
        json.dumps(
            {
                "name": name, "w": w, "h": h, "g": g, "src_n": int(b.get("src_n", 0)),
                "f0": float(f0), "fx": float(fx), "fy": float(fy), "f": float(best_f),
                "df": float(best_f - f0), "x": bx.tolist(), "y": by.tolist(), "th": best_state[2].tolist(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "name": name, "tag": name, "w": w, "h": h, "g": g, "src_n": int(b.get("src_n", 0)),
        "f0": float(f0), "fx": float(fx), "fy": float(fy), "f": float(best_f), "df": float(best_f - f0),
        "x": bx, "y": by, "th": best_state[2].copy(),
    }


def solve_pair_fast(
    n: int, m: int, b1_base: dict, b2_base: dict, inc_dec: Decimal, tag: str, base_seed: int
) -> dict | None:
    best_cert: dict | None = None
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
                    if f_use > f_inc + 0.015:
                        continue
                    ax, ay, at, L0, sm = h60.build_pinwheel_custom(m, b1, b2, sf=sf, f_override=f_use)
                    if len(ax) != n:
                        continue
                    pen, _ = bl.cpp_eval_sat_pen_bbox(ax, ay, at)
                    if pen <= 1e-10 and ftag == "fhi" and sf == 0.0 and (m1, m2) in (("id", "id"), ("flipx", "flipy"), ("rot180", "rot180")):
                        cert = b7.polish_and_certify_b7(n, ax, ay, at, inc_dec, slp_rounds=18, run_lpslack=True)
                        if cert is not None:
                            cert["label"] = f"{tag}_{m1}_{m2}_direct"
                            if best_cert is None or cert["L_float"] < best_cert["L_float"]:
                                best_cert = cert
                    if (m1, m2) in (("id", "id"), ("flipx", "id"), ("id", "flipx"), ("rot180", "rot180")):
                        seeds.append((ax, ay, at))
                        masks.append(sm)
                        L_tgt = m + min(f_use, f_inc)
                        scheds.append([L_tgt + 0.008, L_tgt + 0.0015, L_tgt - 2e-4])
                        labels.append(f"{tag}_{m1}_{m2}_sf{sf}_{ftag}")

    if not seeds:
        return best_cert

    for siter in (160, 0):
        relaxed = bl.su.cpp_batch_melt(
            seeds[:32], scheds[:32], sym_mode=1, a_start=0.965, n_melt_stages=3,
            seam_masks=masks[:32] if siter > 0 else None, seam_iters=siter,
            full_iters=240, n_ruin_iters=0, n_threads=2, base_seed=base_seed + siter,
        )
        pool = []
        for bi, row in enumerate(relaxed):
            nx, ny, nth, Lm = bl.normalize(row["x"], row["y"], row["th"])
            lbl = f"{labels[bi]}_si{siter}"
            if row["best_L_feas"] is not None:
                pool.append((0.0, Lm, nx, ny, nth, lbl))
            elif row["pen"] < 0.015:
                pool.append((1.0 + float(row["pen"]), Lm, nx, ny, nth, lbl))
        pool.sort(key=lambda t: (t[0], t[1]))
        for _, _, cx, cy, cth, lbl in pool[:3]:
            cert = b7.polish_and_certify_b7(n, cx, cy, cth, inc_dec, slp_rounds=18, run_lpslack=True)
            if cert is not None:
                cert["label"] = lbl
                if best_cert is None or cert["L_float"] < best_cert["L_float"]:
                    best_cert = cert
                    if best_cert["is_real_win"]:
                        inc_dec = min(inc_dec, Decimal(best_cert["L_dec_str"]))
    return best_cert


def main() -> None:
    t0 = time.time()
    b7.DATA_DIR.mkdir(parents=True, exist_ok=True)
    b7.DOCS_DIR.mkdir(parents=True, exist_ok=True)

    print("[H77] Loading origin/main portfolio...", flush=True)
    portfolio = b7.load_portfolio_targets(GATE_NS + [796, 972, 742, 974, 855, 976, 916, 977, 800])

    bricks: list[dict] = []
    seen_names: set[str] = set()
    for p in sorted(b7.DATA_DIR.glob("brick_*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        nm = str(d["name"])
        seen_names.add(nm)
        bricks.append({
            "name": nm, "tag": nm, "w": int(d["w"]), "h": int(d["h"]), "g": int(d["g"]),
            "src_n": int(d.get("src_n", 0)), "f0": float(d.get("f0", d["f"])),
            "fx": float(d["fx"]), "fy": float(d["fy"]), "f": float(d["f"]), "df": float(d.get("df", 0.0)),
            "x": np.array(d["x"], dtype=np.float64), "y": np.array(d["y"], dtype=np.float64), "th": np.array(d["th"], dtype=np.float64),
        })

    p914 = portfolio[914]
    b8_12x20 = b7.extract_corner_cut(p914["x"], p914["y"], p914["th"], p914["L"], "BR", 12, 20, 8, 914, "B8_12x20_n914_cut")
    if b8_12x20 is not None and "B8_12x20_n914_cut" not in seen_names:
        print("[H77] Quick-walking B8_12x20_n914_cut...", flush=True)
        bricks.append(quick_walk_b8(b8_12x20, "B8_12x20_n914_cut"))
        seen_names.add("B8_12x20_n914_cut")

    for name, bk in rg.harvest_all_bricks(portfolio):
        if name not in seen_names:
            bricks.append(dict(bk, name=name, f0=bk["f"], df=0.0))

    all_bricks = rg.expand_brick_orientations(bricks)
    results = {}
    for n in GATE_NS:
        m = int(math.isqrt(n))
        r = n - m * m
        inc_dec = portfolio[n]["L_dec"]
        f_inc = float(inc_dec) - m
        print(f"\n[H77 GATE] n={n} (m={m}, r=+{r}) inc={str(inc_dec)[:20]} f_inc={f_inc:.9f}", flush=True)

        pairs = []
        seen_dims = set()
        for b1 in all_bricks:
            if b1["w"] >= m or b1["h"] >= m:
                continue
            for b2 in all_bricks:
                if b1["g"] + b2["g"] != r or b2["w"] >= m or b2["h"] >= m:
                    continue
                w1, h1, w2, h2 = b1["w"], b1["h"], b2["w"], b2["h"]
                if not ((w1 + w2 <= m - 1 and h1 + h2 <= m) or (h1 + h2 <= m - 1 and w1 + w2 <= m)):
                    continue
                f_hi = max(b1["f"], b2["f"])
                f_lo = min(b1["f"], b2["f"])
                if f_hi > f_inc + 0.015 and f_lo > f_inc:
                    continue
                key = (b1["name"], b2["name"])
                if key in seen_dims:
                    continue
                seen_dims.add(key)
                sc = 0.65 * f_hi + 0.35 * f_lo - (0.002 if (b1.get("df", 0.0) < -1e-6 or b2.get("df", 0.0) < -1e-6) else 0.0)
                pairs.append((sc, b1, b2))

        pairs.sort(key=lambda t: t[0])
        filtered = []
        seen_geom = set()
        for sc, b1, b2 in pairs:
            gk = (b1["w"], b1["h"], b1["g"], b2["w"], b2["h"], b2["g"], round(b1["f"], 6), round(b2["f"], 6))
            if gk in seen_geom:
                continue
            seen_geom.add(gk)
            filtered.append((sc, b1, b2))
            if len(filtered) >= 5:
                break

        best = None
        cur_inc = inc_dec
        for idx, (_, b1, b2) in enumerate(filtered):
            print(f"  Pair {idx+1}: {b1['name']}({b1['w']}x{b1['h']},+{b1['g']},f={b1['f']:.6f}) + {b2['name']}({b2['w']}x{b2['h']},+{b2['g']},f={b2['f']:.6f})", flush=True)
            c = solve_pair_fast(n, m, b1, b2, cur_inc, f"n{n}_{b1['name']}+{b2['name']}", base_seed=77000 + n * 13 + idx * 50)
            if c is not None:
                print(f"    -> {c['label']}: L={c['L_dec_str'][:20]} dL={c['dL']:+.6e} win={c['is_real_win']}", flush=True)
                if best is None or Decimal(c["L_dec_str"]) < Decimal(best["L_dec_str"]):
                    best = c
                if c["is_real_win"]:
                    b7.save_win_b7(c)
                    cur_inc = min(cur_inc, Decimal(c["L_dec_str"]))

        results[str(n)] = {
            "inc": str(portfolio[n]["L_dec"]),
            "win": bool(best and best["is_real_win"]),
            "L": best["L_dec_str"] if best else None,
            "dL": best["dL"] if best else None,
            "label": best.get("label") if best else None,
        }

    (b7.DOCS_DIR / "gate_report.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    if not any(b7.DATA_DIR.glob("win_n*.json")):
        b7.write_header_only_final_csv()
    print(f"\n[H77 FAST GATE COMPLETE] in {time.time()-t0:.1f}s -> {results}", flush=True)


if __name__ == "__main__":
    main()
