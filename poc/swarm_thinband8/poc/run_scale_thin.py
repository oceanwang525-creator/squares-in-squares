#!/usr/bin/env python3
"""Scale-up sweep for H112 (thinband8) across all m = 22..29, r = +9..+15 targets."""
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import thinband8_lib as tb  # noqa: E402
import run_gate_thin as rgt  # noqa: E402

ALL_TARGETS = [
    # Wave 1 + Wave 2 prioritized: tightest kx<=0,ky<=0 negative-df pairs first
    635, 539, 636, 687, 587, 741, 637, 638,
    690, 691, 495, 588, 797, 798, 494, 541, 742, 855,
    856, 743, 686, 854, 852, 639, 640, 540, 689, 738,
    851, 688, 796, 853, 740, 685, 739, 744, 794, 850,
    795, 793, 586, 538,
] + [
    m * m + r
    for m in range(29, 21, -1)
    for r in range(9, 16)
    if (m * m + r) not in {
        635, 539, 636, 687, 587, 741, 637, 638,
        690, 691, 495, 588, 797, 798, 494, 541, 742, 855,
        856, 743, 686, 854, 852, 639, 640, 540, 689, 738,
        851, 688, 796, 853, 740, 685, 739, 744, 794, 850,
        795, 793, 586, 538,
    }
]


def main() -> None:
    t0 = time.time()
    tb.DATA_DIR.mkdir(parents=True, exist_ok=True)
    tb.DOCS_DIR.mkdir(parents=True, exist_ok=True)

    bricks = rgt.load_thinband_bricks()
    print(f"Loaded {len(bricks)} oriented bricks for scale-up", flush=True)
    frontier = tb.fast_load_true_frontier(ALL_TARGETS, certify=False)
    print(f"Loaded fast frontier for {len(frontier)} targets in {time.time()-t0:.1f}s", flush=True)

    summary_path = tb.DOCS_DIR / "scale_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        summary = {"targets": {}, "wins": []}
    for n in ALL_TARGETS:
        if str(n) in summary["targets"]:
            continue
        tn = time.time()
        m = int(math.isqrt(n))
        r = n - m * m
        f_inc_fast = float(frontier[n]["L_dec"]) - m
        pairs = rgt.pick_pairs_for_n(n, bricks, f_inc_fast, top_k=4)
        if not pairs:
            continue
        cert_n = tb.fast_load_true_frontier([n], certify=True)[n]
        wp = tb.DATA_DIR / f"win_n{n}.json"
        if wp.exists():
            wd = json.loads(wp.read_text(encoding="utf-8"))
            wdec = max(Decimal(wd["L_lit"]), Decimal(wd["L_pandas"]))
            if wdec < cert_n["L_dec"]:
                cert_n["L_dec"] = wdec
                cert_n["L"] = float(wdec)
                cert_n["source"] = "thinband8_saved"
        frontier[n] = cert_n
        inc_dec = frontier[n]["L_dec"]
        f_inc = float(inc_dec) - m
        if not pairs:
            continue

        print(
            f"\n=== SCALE n={n} (m={m}, r=+{r}) L_inc={str(inc_dec)[:18]} "
            f"(f_inc={f_inc:.9f}, src={frontier[n]['source']}) | pairs={len(pairs)} ===",
            flush=True,
        )
        best_cert = None
        pair_logs = []
        for idx, (sc, b1, b2) in enumerate(pairs):
            kx = b1["w"] + b2["w"] - m
            ky = b1["h"] + b2["h"] - m
            tag = f"tb8s_{n}_{b1['w']}x{b1['h']}g{b1['g']}+{b2['w']}x{b2['h']}g{b2['g']}"
            cert = tb.solve_pinwheel_pair(
                n, m, b1, b2, inc_dec, tag,
                base_seed=212000 + n * 10 + idx * 100,
                n_threads=2,
                do_lpslack=True,
            )
            if cert is not None:
                pair_logs.append({
                    "pair": tag, "kx": kx, "ky": ky,
                    "L": cert["L_dec_str"][:20], "dL": cert["dL"], "win": cert["is_real_win"],
                })
                print(
                    f"  [{idx+1}/{len(pairs)}] {tag} kx={kx} ky={ky} -> L={cert['L_dec_str'][:18]} dL={cert['dL']:+.6e} win={cert['is_real_win']}",
                    flush=True,
                )
                if best_cert is None or cert["L_float"] < best_cert["L_float"]:
                    best_cert = cert
                if cert["is_real_win"]:
                    saved = tb.save_win_locked(cert)
                    inc_dec = min(inc_dec, Decimal(cert["L_lit"]), Decimal(cert["L_pandas"]))
                    print(f"    *** CERTIFIED SCALE WIN n={n}: saved={saved} dL={cert['dL']:+.6e} ***", flush=True)

        is_win = bool(best_cert and best_cert["is_real_win"])
        if is_win:
            summary["wins"].append(n)
        summary["targets"][str(n)] = {
            "n": n, "m": m, "r": r,
            "L_inc": str(frontier[n]["L_dec"]),
            "inc_src": frontier[n]["source"],
            "win": is_win,
            "best_L": best_cert["L_dec_str"] if best_cert else None,
            "dL": best_cert["dL"] if best_cert else None,
            "label": best_cert["label"] if best_cert else None,
            "pairs_tested": pair_logs,
            "dt": round(time.time() - tn, 1),
        }
        (tb.DOCS_DIR / "scale_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    summary["total_dt"] = round(time.time() - t0, 1)
    (tb.DOCS_DIR / "scale_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n=== SCALE COMPLETE in {summary['total_dt']}s | wins={summary['wins']} ===", flush=True)


if __name__ == "__main__":
    main()
