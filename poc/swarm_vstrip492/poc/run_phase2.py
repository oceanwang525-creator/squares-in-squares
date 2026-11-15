#!/usr/bin/env python3
"""H106 (vstrip492) Phase 2 Scale-Up Sweep:
1) pslp squeeze on existing PR1 wins
2) remaining zero-overlap (ov=0, ct=0) targets across K=20..31, D=7..16
3) ct=1 and ov=1 corner-melted pinwheels + pslp + lpslack + mechwalk
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from decimal import Decimal
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import vstrip492_lib as vl  # noqa: E402

sys.path.insert(0, str(HERE))
from run_scale import _select_zero_ct0_pairs  # noqa: E402
from run_gate_probe import _select_candidate_pairs  # noqa: E402


def _squeeze_existing_wins(frontier: dict[int, dict]) -> None:
    print("=== Stage 1: pslp squeeze on saved PR1 wins ===", flush=True)
    for p in sorted(vl.DATA_DIR.glob("win_n*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        n = int(d["n"])
        if n not in frontier:
            continue
        inc_dec = frontier[n]["L_dec"]
        cur_L = max(Decimal(d["L_lit"]), Decimal(d["L_pandas"]))
        eff_inc = min(inc_dec, cur_L)
        cert = {
            "ok": True,
            "n": n,
            "L_lit": d["L_lit"],
            "L_pandas": d["L_pandas"],
            "L_dec_str": str(cur_L),
            "L_float": float(cur_L),
            "dL": float(cur_L - eff_inc),
            "is_real_win": False,
            "rows": d["rows"],
            "label": d.get("label", f"n{n}"),
        }
        out = vl.pslp_squeeze_cert(n, cert, eff_inc, shake_rounds=6, seed=n * 31 + 7)
        new_L = max(Decimal(out["L_lit"]), Decimal(out["L_pandas"]))
        if new_L <= cur_L - Decimal("1e-9"):
            out["dL"] = float(new_L - inc_dec)
            out["is_real_win"] = bool(new_L - inc_dec <= Decimal("-1e-9"))
            saved = vl.save_win_locked(out)
            print(
                f"  *** PSLP SQUEEZE WIN n={n}: {str(cur_L)[:18]} -> {str(new_L)[:18]} "
                f"(dL_extra={float(new_L-cur_L):+.6e}, saved={saved}) ***",
                flush=True,
            )


def _solve_phase2_target(args: tuple[int, str, list[dict], bool, int]) -> dict:
    n, inc_str, bricks, use_melt, max_pairs = args
    inc_dec = Decimal(inc_str)
    wn = vl.DATA_DIR / f"win_n{n}.json"
    if wn.exists():
        wd = json.loads(wn.read_text(encoding="utf-8"))
        inc_dec = min(inc_dec, max(Decimal(wd["L_lit"]), Decimal(wd["L_pandas"])))

    inc_f = float(inc_dec)
    m = int(math.floor(inc_f))
    r = n - m * m
    f_inc = inc_f - m

    if not use_melt:
        raw_pairs = _select_zero_ct0_pairs(m, r, f_inc, bricks, max_fm_margin=0.0018)
        pairs = [(0, fm, 0, b1, b2) for fm, b1, b2 in raw_pairs]
    else:
        pairs = _select_candidate_pairs(m, r, f_inc, bricks, max_fm_margin=0.0015)

    best_n: dict | None = None
    for pidx, (ov, fm, ct, b1, b2) in enumerate(pairs[:max_pairs]):
        cur_f_inc = float(inc_dec) - m
        if best_n is not None and best_n["is_real_win"] and (best_n["dL"] <= -1e-3 or fm >= cur_f_inc + 0.0008):
            break
        tag = f"n{n}_{b1['tag']}_{b2['tag']}"
        cert = vl.solve_pair(
            n,
            m,
            b1,
            b2,
            inc_dec,
            tag,
            base_seed=n * 100 + pidx + (50 if use_melt else 0),
            max_ov=1 if use_melt else 0,
            do_melt=use_melt,
            do_lpslack=True,
            do_mechwalk=False,
            do_pslp=True,
            n_threads=1,
        )
        if cert is not None:
            if cert["is_real_win"]:
                saved = vl.save_win_locked(cert)
                print(
                    f"  *** WIN n={n:3d} ({m},{r:2d}) {'MELT' if use_melt else 'ZERO'} p{pidx} ov={ov} ct={ct}: "
                    f"{b1['w']}x{b1['h']}+{b2['w']}x{b2['h']} L={cert['L_dec_str'][:18]} "
                    f"dL={cert['dL']:+.6e} saved={saved} ***",
                    flush=True,
                )
                inc_dec = min(inc_dec, max(Decimal(cert["L_lit"]), Decimal(cert["L_pandas"])))
            if best_n is None or cert["L_float"] < best_n["L_float"]:
                best_n = cert

    return {
        "n": n,
        "m": m,
        "r": r,
        "mode": "melt" if use_melt else "zero",
        "inc": inc_str,
        "n_cands": len(pairs),
        "win": bool(best_n and best_n["is_real_win"]),
        "best_L": best_n["L_dec_str"] if best_n else None,
        "best_dL": best_n["dL"] if best_n else None,
        "label": best_n["label"] if best_n else None,
    }


def main() -> None:
    t0 = time.time()
    bricks = vl.load_all_bricks(include_external=True, max_per_shape=2)
    targets = [m * m + r for m in range(20, 32) for r in range(7, 17) if m * m + r <= 1000]
    frontier = vl.fast_load_true_frontier(targets)
    print(f"Loaded {len(bricks)} brick shapes across K=20..31, D=7..16 ({len(targets)} targets).", flush=True)

    if "--squeeze" in sys.argv:
        _squeeze_existing_wins(frontier)

    ckpt_p1 = vl.DOCS_DIR / "scale_results.json"
    done_p1 = json.loads(ckpt_p1.read_text(encoding="utf-8")) if ckpt_p1.exists() else {}

    ckpt_p2 = vl.DOCS_DIR / "phase2_results.json"
    vl.DOCS_DIR.mkdir(parents=True, exist_ok=True)
    results = json.loads(ckpt_p2.read_text(encoding="utf-8")) if ckpt_p2.exists() else {}

    # Stage 2A/2B: Big-Move ONLY sweep (freeze zero_ct0 micro-polish >= -1e-4 per #717/#725)
    zero_jobs: list[tuple[float, int]] = []
    melt_jobs: list[tuple[float, int]] = []
    for n in targets:
        d = frontier[n]
        m = int(math.floor(d["L"]))
        r = n - m * m
        f_inc = d["L"] - m
        z_pairs = _select_zero_ct0_pairs(m, r, f_inc, bricks, max_fm_margin=-0.0001)
        key_z = f"zero_{n}"
        if z_pairs and str(n) not in done_p1 and key_z not in results:
            zero_jobs.append((z_pairs[0][0] - f_inc, n))

        m_pairs = _select_candidate_pairs(m, r, f_inc, bricks, max_fm_margin=0.0002)
        has_non_zero = [p for p in m_pairs if (p[0] > 0 or p[2] > 0) and (p[1] - f_inc) <= -0.0001]
        key_m = f"melt_{n}"
        if has_non_zero and key_m not in results:
            melt_jobs.append((has_non_zero[0][1] - f_inc, n))

    zero_jobs.sort()
    melt_jobs.sort()
    print(f"Queued {len(zero_jobs)} zero_ct0 (df<=-1e-4) + {len(melt_jobs)} ct=1/ov=1 melt (df<=-1e-4) targets.", flush=True)

    all_job_args = [(n, str(frontier[n]["L_dec"]), bricks, True, 2) for _, n in melt_jobs] + [
        (n, str(frontier[n]["L_dec"]), bricks, False, 3) for _, n in zero_jobs
    ]

    with ProcessPoolExecutor(max_workers=2) as pool:
        futs = {pool.submit(_solve_phase2_target, j): (j[3], j[0]) for j in all_job_args}
        for fut in as_completed(futs):
            res = fut.result()
            k = f"{res['mode']}_{res['n']}"
            results[k] = res
            ckpt_p2.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
            print(
                f"[done {len(results)}/{len(all_job_args)}] {k} win={res['win']} "
                f"dL={res['best_dL']} ({res['n_cands']} cands)",
                flush=True,
            )

    wins = [r for r in results.values() if r["win"]]
    print(f"\nPHASE 2 COMPLETE ({len(wins)}/{len(results)} wins) in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
