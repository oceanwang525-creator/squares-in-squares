#!/usr/bin/env python3
"""Scale-up runner across K=20..30, D=8..16 with pslp squeeze polish."""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import json
import math
import sys
import time
from decimal import Decimal
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import pocketpeel_lib as pp  # noqa: E402
sys.path.insert(0, str(HERE))
import run_gate as rg  # noqa: E402

RATCHET_POC = pp.REPO_ROOT / "poc" / "row_ratchet_980" / "poc"
if str(RATCHET_POC) not in sys.path:
    sys.path.insert(0, str(RATCHET_POC))
from pslp import squeeze as pslp_squeeze  # noqa: E402


def try_pslp_polish(n: int, cert: dict, true_inc_dec: Decimal) -> dict:
    """Apply full-row kink-safe LP squeeze (pslp) on a certified candidate."""
    if cert is None or not cert.get("rows"):
        return cert
    rows = sorted(cert["rows"], key=lambda r: int(r["square_id"]))
    a = np.array([(float(r["x"]), float(r["y"]), float(r["angle"])) for r in rows], dtype=np.float64)
    e = 0.5 * (np.abs(np.cos(a[:, 2])) + np.abs(np.sin(a[:, 2])))
    a[:, 0] -= (a[:, 0] - e).min()
    a[:, 1] -= (a[:, 1] - e).min()
    Lw = float(max((a[:, 0] + e).max(), (a[:, 1] + e).max()))
    K = int(math.isqrt(n))
    try:
        f, X, Y, T, viol, _ = pslp_squeeze(a[:, 0].copy(), a[:, 1].copy(), a[:, 2].copy(), K, K, Lw - K + 1e-12)
    except Exception:
        return cert
    if viol <= 1e-11 and K + f < Lw - 1e-10:
        cur_dec = Decimal(max(cert["L_lit"], cert["L_pd"]))
        for scale in (1.0, 1.0 + 2e-12, 1.0 + 1e-11):
            c2 = pp.hlib.certify_pack_against_frontier(n, X * scale, Y * scale, T, true_inc_dec)
            if c2 is not None and c2["L_lit"] == c2["L_pd"] and Decimal(c2["L_lit"]) < cur_dec:
                c2["label"] = f"{cert['label']}_pslp"
                return c2
    return cert


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k-min", type=int, default=20)
    ap.add_argument("--k-max", type=int, default=30)
    ap.add_argument("--d-min", type=int, default=8)
    ap.add_argument("--d-max", type=int, default=16)
    ap.add_argument("--slack-max", type=float, default=0.004)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--targets", default="")
    ap.add_argument("--out", default="scale_summary.json")
    args = ap.parse_args()

    t0 = time.time()
    pp.ensure_dirs()
    bricks = rg.load_all_bricks(include_external=True)

    if args.targets:
        cand_ns = [int(x) for x in args.targets.split(",") if x.strip()]
    else:
        cand_ns = [K * K + D for K in range(args.k_min, args.k_max + 1) for D in range(args.d_min, args.d_max + 1)]

    frontier = pp.load_merged_frontier(cand_ns, certify=False)

    # Rank targets by predicted df = f_hi - f_inc
    ranked = []
    for n in cand_ns:
        m = int(math.isqrt(n))
        f_inc = frontier[n]["L"] - m
        pairs = rg.pick_pairs_for_n(n, bricks, f_inc, top_k=args.top_k, allow_kx1=False, slack_max=args.slack_max)
        if pairs:
            df = max(pairs[0][3]["f"], pairs[0][4]["f"]) - f_inc
            ranked.append((df, n, pairs))
    ranked.sort(key=lambda t: t[0])
    print(f"Ranked {len(ranked)} targets in K={args.k_min}..{args.k_max}, D={args.d_min}..{args.d_max}", flush=True)

    summary_path = pp.DOCS_DIR / args.out
    report: dict = {"targets": {}, "wins": []}
    if summary_path.exists():
        try:
            report = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    for df_pred, n, pairs in ranked:
        if str(n) in report.get("targets", {}):
            continue
        tn = time.time()
        m = int(math.isqrt(n))
        r = n - m * m
        true_fr = pp.load_merged_frontier([n], certify=True)
        inc_dec = true_fr[n]["L_dec"]

        # Skip if already won with dL <= -1e-4
        wp = pp.WINS_DIR / f"win_n{n}.json"
        if wp.exists():
            prev = json.loads(wp.read_text(encoding="utf-8"))
            if prev.get("is_real_win") and prev.get("dL", 0) <= -1e-4:
                print(f"[SKIP] n={n} already won (dL={prev['dL']:+.6e})", flush=True)
                continue

        print(
            f"\n=== SCALE n={n} (m={m}, r=+{r}) df_pred={df_pred:+.6f} L_inc={str(inc_dec)[:18]} | pairs={len(pairs)} ===",
            flush=True,
        )
        best_cert: dict | None = None
        pair_logs = []
        for idx, (sc, kx, ky, b1, b2) in enumerate(pairs):
            tag = f"n{n}_{b1['tag']}+{b2['tag']}"
            print(
                f"  [{idx+1}/{len(pairs)}] kx={kx:+d} ky={ky:+d} "
                f"{b1['tag']}({b1['w']}x{b1['h']}+{b1['g']},f={b1['f']:.6f}) + "
                f"{b2['tag']}({b2['w']}x{b2['h']}+{b2['g']},f={b2['f']:.6f})",
                flush=True,
            )
            cert = rg.solve_pair(
                n, m, b1, b2, inc_dec, tag,
                base_seed=213000 + n * 13 + idx * 100,
                n_threads=2,
                run_mechwalk=True,
            )
            if cert is not None:
                if cert["dL"] < 1e-3:
                    cert = try_pslp_polish(n, cert, true_fr[n]["L_dec"])
                l_str = str(max(Decimal(cert["L_lit"]), Decimal(cert["L_pd"])))
                print(
                    f"    -> L={l_str[:18]} dL={cert['dL']:+.6e} win={cert['is_real_win']} ({cert['label']})",
                    flush=True,
                )
                pair_logs.append({
                    "pair": f"{b1['tag']}+{b2['tag']}",
                    "kx": kx,
                    "ky": ky,
                    "L": l_str[:18],
                    "dL": cert["dL"],
                    "win": cert["is_real_win"],
                    "label": cert["label"],
                })
                if best_cert is None or Decimal(max(cert["L_lit"], cert["L_pd"])) < Decimal(max(best_cert["L_lit"], best_cert["L_pd"])):
                    best_cert = cert
                if cert["is_real_win"] and cert["L_lit"] == cert["L_pd"]:
                    pp.save_win_json(cert)
                    pp.export_final_csv_from_wins()
                    inc_dec = min(inc_dec, Decimal(cert["L_lit"]))
                    print(f"    *** CERTIFIED REAL WIN n={n}: dL={cert['dL']:+.6e} ***", flush=True)
                    if cert["dL"] <= -1e-3:
                        break
            else:
                print("    -> no feasible", flush=True)
                pair_logs.append({"pair": f"{b1['tag']}+{b2['tag']}", "kx": kx, "ky": ky, "L": None, "dL": None, "win": False})

        is_win = bool(best_cert and best_cert["is_real_win"] and best_cert["L_lit"] == best_cert["L_pd"])
        if is_win and n not in report["wins"]:
            report["wins"].append(n)
        best_l_str = str(max(Decimal(best_cert["L_lit"]), Decimal(best_cert["L_pd"]))) if best_cert else None
        report["targets"][str(n)] = {
            "n": n,
            "m": m,
            "r": r,
            "df_pred": df_pred,
            "L_inc": str(true_fr[n]["L_dec"]),
            "win": is_win,
            "best_L": best_l_str,
            "dL": best_cert["dL"] if best_cert else None,
            "label": best_cert["label"] if best_cert else None,
            "pairs": pair_logs,
            "dt": round(time.time() - tn, 1),
        }
        summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    report["total_dt"] = round(time.time() - t0, 1)
    summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    pp.export_final_csv_from_wins()
    print(f"\n=== SCALE COMPLETE in {report['total_dt']}s | wins={report['wins']} ===", flush=True)


if __name__ == "__main__":
    main()
