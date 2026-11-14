#!/usr/bin/env python3
"""H109 (peelstep) 30-minute gate on {913, 854, 973, 853} + priority sweep cells."""
from __future__ import annotations

import argparse
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
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import peelstep_lib as pl  # noqa: E402

GATE_PRIMARY = [913, 854, 973, 853]
SWEEP_EXTRA = [797, 914, 855, 795, 796, 740, 741, 742, 976]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", default="", help="Comma-separated n values (default: gate + extra)")
    parser.add_argument("--top-k", type=int, default=6, help="Max pairs per target n")
    parser.add_argument("--max-excess", type=float, default=0.005, help="Max f_hi - f_inc threshold")
    parser.add_argument("--time-limit", type=float, default=1650.0, help="Soft wallclock cap in seconds")
    parser.add_argument("--out", default=str(pl.DOCS_DIR / "gate_summary.json"))
    args = parser.parse_args()

    t0 = time.time()
    pl.DATA_DIR.mkdir(parents=True, exist_ok=True)
    pl.DOCS_DIR.mkdir(parents=True, exist_ok=True)

    if args.targets.strip():
        targets = [int(x) for x in args.targets.split(",") if x.strip()]
    else:
        targets = GATE_PRIMARY + SWEEP_EXTRA

    bricks = pl.load_all_pooled_bricks(top_per_shape=2)
    print(f"Loaded {len(bricks)} pooled oriented bricks (top_per_shape=2)", flush=True)
    frontier = pl.cp2.fast_load_true_frontier(targets, certify=True)
    print(f"Loaded certified frontier for {len(targets)} targets in {time.time()-t0:.1f}s", flush=True)

    summary: dict = {"targets": {}, "wins": []}
    out_path = Path(args.out)

    for n in targets:
        if time.time() - t0 > args.time_limit:
            print(f"\n[STOP] Reached time limit {args.time_limit:.0f}s before n={n}", flush=True)
            break

        tn = time.time()
        m = int(math.isqrt(n))
        r = n - m * m
        inc_dec = frontier[n]["L_dec"]
        # Respect any already-found win in our own data dir
        wn = pl.DATA_DIR / f"win_n{n}.json"
        if wn.exists():
            wd = json.loads(wn.read_text(encoding="utf-8"))
            inc_dec = min(inc_dec, max(Decimal(wd["L_lit"]), Decimal(wd["L_pandas"])))

        f_inc = float(inc_dec) - m
        pairs = pl.pick_pairs_for_n(n, bricks, f_inc, max_excess=args.max_excess, top_k=args.top_k)
        print(
            f"\n=== n={n} (m={m}, r=+{r}) L_inc={str(inc_dec)[:18]} f_inc={f_inc:.6f} "
            f"[{frontier[n]['source']}] | pairs={len(pairs)} ===",
            flush=True,
        )

        best_cert = None
        pair_logs = []
        for idx, (score, b1, b2, dm) in enumerate(pairs):
            if time.time() - t0 > args.time_limit:
                break
            kx = b1["w"] + b2["w"] - m
            ky = b1["h"] + b2["h"] - m
            f_hi = max(b1["f"], b2["f"])
            print(
                f"  [{idx+1}/{len(pairs)}] kx={kx:+d} ky={ky:+d} dm={dm} "
                f"{b1['tag']}({b1['w']}x{b1['h']}+{b1['g']},f={b1['f']:.5f}) + "
                f"{b2['tag']}({b2['w']}x{b2['h']}+{b2['g']},f={b2['f']:.5f}) fhi={f_hi:.5f}",
                flush=True,
            )
            tag = f"ps{n}_{b1['tag']}+{b2['tag']}"
            cert, diag = pl.solve_pinwheel_pair(
                n, m, r, b1, b2, dm, inc_dec, tag,
                base_seed=109000 + n * 10 + idx * 100, n_threads=2,
            )
            if diag:
                print(f"    diag={diag}", flush=True)
            if cert is not None:
                print(
                    f"    -> L={cert['L_dec_str'][:18]} dL={cert['dL']:+.6e} "
                    f"win={cert['is_real_win']} ({cert['label']})",
                    flush=True,
                )
                pair_logs.append({
                    "pair": f"{b1['tag']}+{b2['tag']}",
                    "kx": kx,
                    "ky": ky,
                    "dm": dm,
                    "f_hi": round(f_hi, 6),
                    "L": cert["L_dec_str"][:20],
                    "dL": cert["dL"],
                    "win": cert["is_real_win"],
                })
                if best_cert is None or cert["L_float"] < best_cert["L_float"]:
                    best_cert = cert
                if cert["is_real_win"]:
                    saved = pl.save_win_locked(cert)
                    inc_dec = min(inc_dec, Decimal(cert["L_lit"]), Decimal(cert["L_pandas"]))
                    print(f"    *** CERTIFIED REAL WIN n={n}: saved={saved} dL={cert['dL']:+.6e} ***", flush=True)
            else:
                pair_logs.append({
                    "pair": f"{b1['tag']}+{b2['tag']}",
                    "kx": kx,
                    "ky": ky,
                    "dm": dm,
                    "f_hi": round(f_hi, 6),
                    "L": None,
                    "dL": None,
                    "win": False,
                })

        is_win = bool(best_cert and best_cert["is_real_win"])
        if is_win:
            summary["wins"].append(n)
        summary["targets"][str(n)] = {
            "n": n,
            "m": m,
            "r": r,
            "L_inc": str(frontier[n]["L_dec"]),
            "inc_src": frontier[n]["source"],
            "win": is_win,
            "best_L": best_cert["L_dec_str"] if best_cert else None,
            "dL": best_cert["dL"] if best_cert else None,
            "label": best_cert["label"] if best_cert else None,
            "pairs_tested": pair_logs,
            "dt": round(time.time() - tn, 1),
        }
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    summary["total_dt"] = round(time.time() - t0, 1)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n=== GATE COMPLETE in {summary['total_dt']}s | wins={summary['wins']} ===", flush=True)


if __name__ == "__main__":
    main()
