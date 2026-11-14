#!/usr/bin/env python3
"""H94 (band3) scale-up sweep: zero-overlap (peel-by-1) + unblocked ct=1 / ov<=1 pinwheels."""
from __future__ import annotations

import json
import math
import os
import sys
import time
from decimal import Decimal
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import band3_lib as b3  # noqa: E402

for extra_dir in (
    b3.REPO_ROOT / "poc" / "swarm_cleanpeel2" / "data",
    b3.REPO_ROOT / "poc" / "swarm_gstep" / "data",
    b3.REPO_ROOT / "poc" / "swarm_stepdown" / "data",
):
    if extra_dir.exists() and extra_dir not in b3.EXTERNAL_BRICK_DIRS:
        b3.EXTERNAL_BRICK_DIRS.append(extra_dir)


def _lpslack_polish(n: int, cert: dict, inc_dec: Decimal, time_limit: float = 15.0) -> dict:
    rs = sorted(cert["rows"], key=lambda r: int(r["square_id"]))
    res_lp = b3.lpslack.optimize_group(
        n,
        rs,
        str(inc_dec),
        outer_iters=6,
        fixed_rounds=8,
        slp_modes=("dual_non90", "non90"),
        tr_th_schedule=(4e-4, 1e-4),
        time_limit=time_limit,
    )
    if res_lp.get("win") and res_lp.get("rows"):
        lit, pan = Decimal(res_lp["new_L_lit"]), Decimal(res_lp["new_L_pan"])
        worst = max(lit, pan)
        if worst < max(Decimal(cert["L_lit"]), Decimal(cert["L_pandas"])):
            cert["L_lit"] = str(lit)
            cert["L_pandas"] = str(pan)
            cert["L_dec_str"] = str(worst)
            cert["L_float"] = float(worst)
            cert["dL"] = float(worst - inc_dec)
            cert["is_real_win"] = bool(worst - inc_dec <= Decimal("-1e-9"))
            cert["rows"] = res_lp["rows"]
            cert["label"] = cert.get("label", "") + "+lpslack"
    return cert


def _update_best(best: dict | None, cand: dict | None) -> dict | None:
    if cand is None:
        return best
    if best is None or cand["L_float"] < best["L_float"]:
        return cand
    return best


def solve_pair_scale(
    n: int,
    m: int,
    b1_base: dict,
    b2_base: dict,
    inc_dec: Decimal,
    tag: str,
    base_seed: int,
    max_ov: int = 1,
) -> tuple[dict | None, list]:
    best_cert: dict | None = None
    inc_f = float(inc_dec)
    w1_0, h1_0, w2_0, h2_0 = b1_base["w"], b1_base["h"], b2_base["w"], b2_base["h"]
    is_zero_ov = (w1_0 + w2_0 <= m) and (h1_0 + h2_0 <= m) and (w1_0 + w2_0 + h1_0 + h2_0 <= 2 * m - 1)

    seeds, masks, scheds, labels = [], [], [], []
    for m1 in ("id", "flipx", "flipy", "rot180"):
        b1 = b3.bl.transform_brick(b1_base, m1)
        for m2 in ("id", "flipx", "flipy", "rot180"):
            b2 = b3.bl.transform_brick(b2_base, m2)
            f_hi = max(b1["fx"], b1["fy"], b2["fx"], b2["fy"])
            built = b3.build_pinwheel(m, b1, b2, sf=0.0, f_override=f_hi, max_ov=max_ov)
            if built is None:
                continue
            ax, ay, at, L_tgt, sm = built

            # 1) Direct / L-BFGS fast path for zero-overlap or nearly-feasible seeds
            pen0, L_box0 = b3.cpp_eval_sat_pen_bbox(ax, ay, at)
            if pen0 <= 1e-4 and L_box0 < inc_f + 1e-3:
                if pen0 > 1e-9:
                    L_try = min(inc_f - 1.2e-3, max(L_tgt, L_box0))
                    rx, ry, rt, rpen = b3.bslp.relax_rect_lbfgs(ax, ay, at, L_try, L_try, maxiter=250)
                else:
                    rx, ry, rt, rpen = ax, ay, at, pen0
                if rpen <= 1e-7:
                    c = b3.bl.polish_and_certify(n, rx, ry, rt, inc_dec, slp_rounds=14)
                    if c is not None:
                        c["label"] = f"{tag}_{m1}_{m2}_direct"
                        best_cert = _update_best(best_cert, c)

            # 2) Build widened corner + seam melt batch
            w1, h1, w2, h2 = b1["w"], b1["h"], b2["w"], b2["h"]
            corner_sm = (
                (sm > 0)
                | ((np.abs(ax - w1) <= 4.2) & (np.abs(ay - (m - h1)) <= 4.2))
                | ((np.abs(ax - (m - w2)) <= 4.2) & (np.abs(ay - h2) <= 4.2))
            ).astype(np.uint8)
            gap = inc_f - L_tgt
            L_hi = min(inc_f - 1.2e-3, L_tgt + max(0.0025, 0.58 * gap)) if gap > 1.5e-3 else L_tgt - 2e-4
            seeds.append((ax, ay, at))
            masks.append(corner_sm)
            scheds.append([L_tgt + 0.025, 0.5 * (L_tgt + 0.025 + L_hi), L_hi])
            labels.append(f"{tag}_{m1}_{m2}_hi")

            if gap > 0.004 and (m1, m2) in (("id", "id"), ("id", "rot180"), ("flipx", "flipy"), ("rot180", "id")):
                L_tight = min(inc_f - 8e-4, L_tgt + max(0.0012, 0.25 * gap))
                seeds.append((ax, ay, at))
                masks.append(corner_sm)
                scheds.append([L_tgt + 0.018, L_tgt + 0.005, L_tight])
                labels.append(f"{tag}_{m1}_{m2}_tight")

    if not seeds:
        return best_cert, []

    relaxed = b3.bl.su.cpp_batch_melt(
        seeds,
        scheds,
        sym_mode=1,
        a_start=0.965,
        n_melt_stages=3,
        seam_masks=masks,
        seam_iters=320,
        full_iters=450,
        n_ruin_iters=0,
        n_threads=2,
        base_seed=base_seed,
    )

    pool = []
    for bi, row in enumerate(relaxed):
        nx, ny, nth, Lm = b3.bl.normalize(row["x"], row["y"], row["th"])
        lbl = f"{labels[bi]}_melt"
        if row["best_L_feas"] is not None:
            pool.append((0.0, Lm, float(row["pen"]), nx, ny, nth, lbl))
        else:
            pool.append((1.0 + float(row["pen"]), Lm, float(row["pen"]), nx, ny, nth, lbl))
    pool.sort(key=lambda t: (t[0], t[1]))
    diag = [(t[0] == 0.0, round(t[1], 6), round(t[2], 6), t[6]) for t in pool[:4]]

    # Stage 2: L-BFGS & full-melt rescue on small-pen candidates before polish_and_certify
    rescue_seeds, rescue_scheds, rescue_labels = [], [], []
    for is_bad, Lm, pen, cx, cy, cth, lbl in pool[:5]:
        if pen <= 1e-8:
            c = b3.bl.polish_and_certify(n, cx, cy, cth, inc_dec, slp_rounds=16)
            if c is not None:
                c["label"] = lbl
                best_cert = _update_best(best_cert, c)
        elif pen <= 0.035:
            L_try = min(inc_f - 1.0e-3, max(Lm, float(inc_dec) - 0.5 * max(0.002, inc_f - Lm)))
            rx, ry, rt, rpen = b3.bslp.relax_rect_lbfgs(cx, cy, cth, L_try, L_try, maxiter=350)
            if rpen <= 1e-6:
                c = b3.bl.polish_and_certify(n, rx, ry, rt, inc_dec, slp_rounds=16)
                if c is not None:
                    c["label"] = f"{lbl}+lbfgs"
                    best_cert = _update_best(best_cert, c)
            elif rpen <= 0.025:
                rescue_seeds.append((rx, ry, rt))
                rescue_scheds.append([L_try + 0.004, L_try + 0.001, min(inc_f - 6e-4, L_try + 0.0015)])
                rescue_labels.append(f"{lbl}+fullmelt")

    if rescue_seeds:
        full_res = b3.bl.su.cpp_batch_melt(
            rescue_seeds,
            rescue_scheds,
            sym_mode=1,
            a_start=0.985,
            n_melt_stages=3,
            seam_masks=None,
            seam_iters=0,
            full_iters=500,
            n_ruin_iters=0,
            n_threads=2,
            base_seed=base_seed + 777,
        )
        for bi, row in enumerate(full_res):
            nx, ny, nth, Lm = b3.bl.normalize(row["x"], row["y"], row["th"])
            if float(row["pen"]) <= 0.02:
                if float(row["pen"]) > 1e-8:
                    L_try = min(inc_f - 7e-4, Lm + 0.002)
                    nx, ny, nth, _ = b3.bslp.relax_rect_lbfgs(nx, ny, nth, L_try, L_try, maxiter=250)
                c = b3.bl.polish_and_certify(n, nx, ny, nth, inc_dec, slp_rounds=16)
                if c is not None:
                    c["label"] = rescue_labels[bi]
                    best_cert = _update_best(best_cert, c)

    if best_cert is not None and best_cert["dL"] <= 0.0025:
        best_cert = _lpslack_polish(n, best_cert, inc_dec, time_limit=15.0)
    return best_cert, diag


def main() -> None:
    t0 = time.time()
    args = [int(a) for a in sys.argv[1:] if not a.startswith("--")]
    bricks = b3.load_all_bricks(include_external=True)
    all_targets = args or [m * m + r for m in range(20, 32) for r in range(13, 23)]
    frontier = b3.gl.fast_load_true_frontier(all_targets, certify=True)

    # Rank (n) by best theoretical gap among ov=0 and ct=1 pairs with need_g = r
    ranked: list[tuple[float, int, int, Decimal, list]] = []
    for n in all_targets:
        inc_dec = frontier[n]["L_dec"]
        wn = b3.DATA_DIR / f"win_n{n}.json"
        if wn.exists():
            wd = json.loads(wn.read_text(encoding="utf-8"))
            inc_dec = min(inc_dec, max(Decimal(wd["L_lit"]), Decimal(wd["L_pandas"])))
        inc_f = float(inc_dec)
        m = int(math.floor(inc_f))
        r = n - m * m
        f_inc = inc_f - m
        need_g = r

        best_by_shape: dict[tuple, tuple] = {}
        for i, b1 in enumerate(bricks):
            for j, b2 in enumerate(bricks):
                if i > j or b1["g"] + b2["g"] != need_g:
                    continue
                w1, h1, w2, h2 = b1["w"], b1["h"], b2["w"], b2["h"]
                if w1 >= m or h1 >= m or w2 >= m or h2 >= m:
                    continue
                if w1 + w2 <= m and h1 + h2 <= m:
                    fm = max(b1["f"], b2["f"])
                    if fm < f_inc - 8e-4:
                        is_z = (w1 + w2 + h1 + h2 <= 2 * m - 1)
                        k = (0 if is_z else 1, w1, h1, b1["g"], w2, h2, b2["g"])
                        if k not in best_by_shape or fm < best_by_shape[k][0]:
                            best_by_shape[k] = (fm, is_z, b1, b2)
        if not best_by_shape:
            continue
        cands = sorted(best_by_shape.values(), key=lambda t: (t[0] + (0.0 if t[1] else 0.001)))[:3]
        best_gap = cands[0][0] - f_inc
        ranked.append((best_gap, n, m, inc_dec, cands))

    ranked.sort(key=lambda t: t[0])
    if not args:
        ranked = ranked[:24]

    print(f"Running H94 scale-up on {len(ranked)} F3B targets (bricks={len(bricks)})...", flush=True)
    summary_rows = []
    wins = []
    for best_gap, n, m, inc_dec, cands in ranked:
        r = n - m * m
        print(
            f"\n--- Target n={n} (m={m}, r=+{r}) inc={inc_dec} best_est_dL={best_gap:+.5f} ({len(cands)} pairs) ---",
            flush=True,
        )
        best_n_cert: dict | None = None
        pair_diags = []
        for idx, (fm, is_z, b1, b2) in enumerate(cands):
            tag = f"n{n}_p{idx}_{b1['w']}x{b1['h']}+{b2['w']}x{b2['h']}_{'z' if is_z else 'ct1'}"
            cert, diag = solve_pair_scale(n, m, b1, b2, inc_dec, tag, base_seed=50000 + n * 10 + idx)
            pair_diags.append({"pair": tag, "fm": round(fm, 6), "diag": diag})
            if cert is not None:
                print(
                    f"  -> {tag}: L={cert['L_float']:.9f} dL={cert['dL']:+.6e} win={cert['is_real_win']} ({cert['label']})",
                    flush=True,
                )
                if best_n_cert is None or cert["L_float"] < best_n_cert["L_float"]:
                    best_n_cert = cert
            else:
                print(f"  -> {tag}: None | diag={diag[:2]}", flush=True)

        row_sum = {"n": n, "m": m, "r": r, "inc": str(inc_dec), "diags": pair_diags}
        if best_n_cert is not None and best_n_cert["is_real_win"]:
            b3.save_win_locked(best_n_cert)
            row_sum["win"] = {
                "L": best_n_cert["L_dec_str"],
                "dL": best_n_cert["dL"],
                "label": best_n_cert["label"],
            }
            wins.append((n, best_n_cert["dL"], best_n_cert["L_dec_str"], best_n_cert["label"]))
            print(f"*** CERTIFIED WIN SAVED n={n}: dL={best_n_cert['dL']:+.6e} L={best_n_cert['L_dec_str']} ***", flush=True)
        elif best_n_cert is not None:
            row_sum["near_miss"] = {"L": best_n_cert["L_dec_str"], "dL": best_n_cert["dL"], "label": best_n_cert["label"]}
        summary_rows.append(row_sum)

    b3.DOCS_DIR.mkdir(parents=True, exist_ok=True)
    out = {"elapsed_s": round(time.time() - t0, 1), "wins": wins, "targets": summary_rows}
    (b3.DOCS_DIR / "scale_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nDone in {time.time() - t0:.1f}s | {len(wins)} NEW WINS: {[(n, round(dl, 7)) for n, dl, _, _ in wins]}")


if __name__ == "__main__":
    main()
