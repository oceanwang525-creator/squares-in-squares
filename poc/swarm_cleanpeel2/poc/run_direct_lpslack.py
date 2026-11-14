#!/usr/bin/env python3
"""Fast zero-overlap direct SLP + lpslack gate check across {910, 972, 971, 911, 852, 795}."""
from __future__ import annotations

import json
import math
import sys
import time
from decimal import Decimal
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import cleanpeel2_lib as cp2  # noqa: E402
import b8_lib as bl  # noqa: E402
import run_gate_fast as rgf  # noqa: E402

LPSLACK_POC = cp2.REPO_ROOT / "poc" / "swarm_lpslack" / "poc"
if str(LPSLACK_POC) not in sys.path:
    sys.path.insert(0, str(LPSLACK_POC))
import lpslack  # noqa: E402

TARGETS = [972, 910, 971, 911, 852, 795]


def main() -> None:
    t0 = time.time()
    frontier = cp2.fast_load_true_frontier(TARGETS, certify=True)
    bricks = rgf.load_cp2_bricks()
    print(f"Loaded {len(bricks)} bricks in {time.time()-t0:.1f}s", flush=True)

    summary = {"targets": {}, "wins": []}
    for n in TARGETS:
        tn = time.time()
        m = int(math.isqrt(n))
        r = n - m * m
        inc_dec = frontier[n]["L_dec"]
        f_inc = float(inc_dec) - m
        pairs = rgf.pick_pairs_for_n(n, bricks, f_inc, top_k=8)
        best_cert = None
        pair_logs = []

        for idx, (_, b1_base, b2_base) in enumerate(pairs):
            kx = b1_base["w"] + b2_base["w"] - m
            ky = b1_base["h"] + b2_base["h"] - m
            if kx == 0 and ky == 0:
                continue  # f x f overlap cannot be resolved by direct SLP
            pair_best = None
            for m1 in ("id", "flipx", "rot180"):
                b1 = bl.transform_brick(b1_base, m1)
                for m2 in ("id", "flipy", "rot180"):
                    b2 = bl.transform_brick(b2_base, m2)
                    built = cp2.build_pinwheel_zero_ov(m, b1, b2, sf=0.0, f_override=None)
                    if built is None:
                        continue
                    ax, ay, at, _, _ = built
                    pen, _ = cp2.cpp_eval_sat_pen_bbox(ax, ay, at)
                    if pen > 1e-10:
                        continue
                    sx, sy, st, _, ok = cp2.bslp.joint_xytheta_slp_tight(ax, ay, at)
                    use_x, use_y, use_t = (sx, sy, st) if ok else (ax, ay, at)
                    cert = bl.polish_and_certify(n, use_x, use_y, use_t, inc_dec, slp_rounds=16)
                    if cert is not None:
                        cert["label"] = f"g{n}_{b1_base['tag']}+{b2_base['tag']}_{m1}_{m2}_direct"
                        if pair_best is None or cert["L_float"] < pair_best["L_float"]:
                            pair_best = cert
            if pair_best is not None:
                pair_logs.append({
                    "pair": f"{b1_base['tag']}({b1_base['w']}x{b1_base['h']}+{b1_base['g']},f={b1_base['f']:.6f})+{b2_base['tag']}({b2_base['w']}x{b2_base['h']}+{b2_base['g']},f={b2_base['f']:.6f})",
                    "kx": kx,
                    "ky": ky,
                    "L": pair_best["L_dec_str"][:18],
                    "dL": pair_best["dL"],
                    "label": pair_best["label"],
                })
                print(
                    f"  n={n} [{idx+1}] kx={kx} ky={ky} {b1_base['tag']}+{b2_base['tag']} -> L={pair_best['L_dec_str'][:18]} dL={pair_best['dL']:+.6e}",
                    flush=True,
                )
                if best_cert is None or pair_best["L_float"] < best_cert["L_float"]:
                    best_cert = pair_best

        if best_cert is not None and best_cert.get("rows") and best_cert["dL"] < 0.02:
            base_rows = [
                {
                    "square_id": str(row["square_id"]),
                    "x": str(row["x"]),
                    "y": str(row["y"]),
                    "angle": str(row["angle"]),
                }
                for row in best_cert["rows"]
            ]
            target_ref_dec = min(inc_dec, Decimal(best_cert["L_lit"]), Decimal(best_cert["L_pandas"]))
            lp_res = lpslack.optimize_group(n, base_rows, str(target_ref_dec))
            print(f"  n={n} lpslack status={lp_res.get('status')} new_L={str(lp_res.get('new_L'))[:18]}", flush=True)
            if lp_res.get("status") in ("WIN", "TINY_WIN") and lp_res.get("rows"):
                L_worst = max(Decimal(lp_res["literal_L"]), Decimal(lp_res["pandas_L"]))
                dL_true = L_worst - frontier[n]["L_dec"]
                lp_cert = {
                    "ok": True,
                    "n": int(n),
                    "L_lit": lp_res["literal_L"],
                    "L_pandas": lp_res["pandas_L"],
                    "L_dec_str": str(L_worst),
                    "L_float": float(L_worst),
                    "dL": float(dL_true),
                    "is_real_win": bool(dL_true <= Decimal("-1e-9")),
                    "label": f"{best_cert['label']}_lpslack",
                    "rows": [
                        {
                            "square_id": int(row["square_id"]),
                            "x": str(row["x"]),
                            "y": str(row["y"]),
                            "angle": str(row["angle"]),
                        }
                        for row in lp_res["rows"]
                    ],
                }
                if lp_cert["L_float"] < best_cert["L_float"]:
                    best_cert = lp_cert
                if lp_cert["is_real_win"]:
                    cp2.save_win_locked(lp_cert)
                    summary["wins"].append(n)
                    print(f"  *** REAL WIN n={n}: dL={lp_cert['dL']:+.6e} ***", flush=True)
        elif best_cert and best_cert["is_real_win"]:
            cp2.save_win_locked(best_cert)
            summary["wins"].append(n)
            print(f"  *** REAL WIN n={n}: dL={best_cert['dL']:+.6e} ***", flush=True)

        summary["targets"][str(n)] = {
            "n": n,
            "m": m,
            "r": r,
            "L_inc": str(frontier[n]["L_dec"]),
            "inc_src": frontier[n]["source"],
            "best_L": best_cert["L_dec_str"] if best_cert else None,
            "dL": best_cert["dL"] if best_cert else None,
            "win": bool(best_cert and best_cert["is_real_win"]),
            "label": best_cert["label"] if best_cert else None,
            "pairs_tested": pair_logs,
            "dt": round(time.time() - tn, 1),
        }

    summary["total_dt"] = round(time.time() - t0, 1)
    (cp2.DOCS_DIR / "gate_direct_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nDONE in {summary['total_dt']}s | wins={summary['wins']}", flush=True)


if __name__ == "__main__":
    main()
