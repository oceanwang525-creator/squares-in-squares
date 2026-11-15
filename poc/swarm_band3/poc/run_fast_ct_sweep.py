#!/usr/bin/env python3
"""Fast deduplicated K=2 corner-touch & ov<=1 sweep for H94 (band3)."""
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


def solve_pair_1batch(
    n: int,
    m: int,
    b1_base: dict,
    b2_base: dict,
    inc_dec: Decimal,
    tag: str,
    base_seed: int,
    max_ov: int = 1,
    do_lpslack: bool = True,
) -> tuple[dict | None, list]:
    best_cert: dict | None = None
    seeds, masks, scheds, labels = [], [], [], []
    inc_f = float(inc_dec)
    for m1 in ("id", "flipx", "flipy", "rot180"):
        b1 = b3.bl.transform_brick(b1_base, m1)
        for m2 in ("id", "flipx", "flipy", "rot180"):
            b2 = b3.bl.transform_brick(b2_base, m2)
            f_hi = max(b1["fx"], b1["fy"], b2["fx"], b2["fy"])
            built = b3.build_pinwheel(m, b1, b2, sf=0.0, f_override=f_hi, max_ov=max_ov)
            if built is None:
                continue
            ax, ay, at, L_tgt, sm = built
            # Widen seam mask around inner corner so corner-touching squares can slide/rotate
            w1, h1, w2, h2 = b1["w"], b1["h"], b2["w"], b2["h"]
            corner_sm = (
                (sm > 0)
                | ((np.abs(ax - w1) <= 3.2) & (np.abs(ay - (m - h1)) <= 3.2))
                | ((np.abs(ax - (m - w2)) <= 3.2) & (np.abs(ay - h2) <= 3.2))
            ).astype(np.uint8)
            L_final = min(inc_f - 3e-4, L_tgt + 0.0018) if L_tgt + 0.0018 < inc_f else 0.5 * (L_tgt + inc_f)
            seeds.append((ax, ay, at))
            masks.append(corner_sm)
            scheds.append([L_tgt + 0.018, L_tgt + 0.006, L_final])
            labels.append(f"{tag}_{m1}_{m2}")

    if not seeds:
        return None, []

    relaxed = b3.bl.su.cpp_batch_melt(
        seeds,
        scheds,
        sym_mode=1,
        a_start=0.965,
        n_melt_stages=3,
        seam_masks=masks,
        seam_iters=260,
        full_iters=280,
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
    diag = [(t[0] == 0.0, round(t[1], 6), round(t[2], 6), t[6]) for t in pool[:3]]

    for _, _, pen, cx, cy, cth, lbl in pool[:3]:
        if pen > 0.04:
            continue
        cert = b3.bl.polish_and_certify(n, cx, cy, cth, inc_dec, slp_rounds=14)
        if cert is not None:
            cert["label"] = lbl
            if best_cert is None or cert["L_float"] < best_cert["L_float"]:
                best_cert = cert

    if do_lpslack and best_cert is not None:
        rs = sorted(best_cert["rows"], key=lambda r: int(r["square_id"]))
        res_lp = b3.lpslack.optimize_group(
            n,
            rs,
            str(inc_dec),
            outer_iters=5,
            fixed_rounds=8,
            slp_modes=("dual_non90", "non90"),
            tr_th_schedule=(4e-4, 1e-4),
            time_limit=10.0,
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
    return best_cert, diag


def main() -> None:
    t0 = time.time()
    args = [int(a) for a in sys.argv[1:] if not a.startswith("--")]
    targets = args or [914, 976, 977, 545, 692, 639, 499, 858, 859, 919, 978, 800, 802, 799, 915, 916]
    bricks = b3.load_all_bricks(include_external=True)
    frontier = b3.gl.fast_load_true_frontier(targets, certify=True)
    results = {}

    for n in targets:
        inc_dec = frontier[n]["L_dec"]
        wn = b3.DATA_DIR / f"win_n{n}.json"
        if wn.exists():
            wd = json.loads(wn.read_text(encoding="utf-8"))
            inc_dec = min(inc_dec, max(Decimal(wd["L_lit"]), Decimal(wd["L_pandas"])))
        inc_f = float(inc_dec)
        m = int(math.floor(inc_f))
        r = n - m * m
        f_inc = inc_f - m

        # Deduplicate strictly by shape (w1, h1, g1, w2, h2, g2) keeping thinnest
        best_by_shape: dict[tuple, tuple] = {}
        for i, b1 in enumerate(bricks):
            for j, b2 in enumerate(bricks):
                if i > j:
                    continue
                if b1["g"] + b2["g"] != r:
                    continue
                if b1["w"] + b2["w"] > m + 1 or b1["h"] + b2["h"] > m + 1:
                    continue
                if b1["w"] + b2["w"] > m and b1["h"] + b2["h"] > m:
                    continue
                fm = max(b1["f"], b2["f"])
                if fm >= f_inc - 5e-4:
                    continue
                ov = max(0, b1["w"] + b2["w"] - m) + max(0, b1["h"] + b2["h"] - m)
                ct = 1 if (b1["w"] + b2["w"] + b1["h"] + b2["h"] >= 2 * m and ov == 0) else 0
                sk = tuple(sorted([(b1["w"], b1["h"], b1["g"]), (b2["w"], b2["h"], b2["g"])]))
                cand = (ov, ct, fm, b1, b2)
                if sk not in best_by_shape or (ov, ct, fm) < best_by_shape[sk][:3]:
                    best_by_shape[sk] = cand

        pairs = sorted(best_by_shape.values(), key=lambda t: (t[0], t[2], t[1]))
        print(f"\n[n={n} m={m} r=+{r} inc={inc_f:.6f}]: {len(pairs)} unique shape pairs", flush=True)
        best_n = None
        for pidx, (ov, ct, fm, b1, b2) in enumerate(pairs[:3]):
            if fm >= float(inc_dec) - m - 1e-6:
                continue
            tag = f"n{n}_{b1['tag']}_{b2['tag']}"
            print(f"  try[{pidx}] ov={ov} ct={ct} fm={fm:.5f}: {b1['w']}x{b1['h']}(g={b1['g']})+{b2['w']}x{b2['h']}(g={b2['g']})", flush=True)
            cert, diag = solve_pair_1batch(n, m, b1, b2, inc_dec, tag, base_seed=n * 100 + pidx, max_ov=1)
            print(f"    diag={diag}", flush=True)
            if cert is not None:
                print(
                    f"    -> {cert['label']}: L={cert['L_dec_str'][:18]} dL={cert['dL']:+.6e} win={cert['is_real_win']}",
                    flush=True,
                )
                if cert["is_real_win"]:
                    saved = b3.save_win_locked(cert)
                    print(f"    *** SAVED WIN n={n}: saved={saved} ***", flush=True)
                    inc_dec = min(inc_dec, max(Decimal(cert["L_lit"]), Decimal(cert["L_pandas"])))
                if best_n is None or cert["L_float"] < best_n["L_float"]:
                    best_n = cert
        results[str(n)] = {
            "inc": str(frontier[n]["L_dec"]),
            "win": bool(best_n and best_n["is_real_win"]),
            "best_L": best_n["L_dec_str"] if best_n else None,
            "dL": best_n["dL"] if best_n else None,
            "label": best_n["label"] if best_n else None,
        }

    b3.DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (b3.DOCS_DIR / "ct_sweep_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nCT SWEEP DONE in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
