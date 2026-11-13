#!/usr/bin/env python3
"""Detached checkpointed H12 scaling sweep across un-slid end-cap and asymmetric-slack packings."""
from __future__ import annotations

import argparse
from decimal import Decimal
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
HIGHD_POC = REPO_ROOT / "poc" / "swarm_highd" / "poc"
for p in (HERE, HIGHD_POC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import run_gate as rg  # noqa: E402
import strip_slide as ss  # noqa: E402
from cpp_wrap import cpp_expand_sym, run_cpp_masked_melt  # noqa: E402

CKPT_DIR = HERE.parent / "docs" / "scale_ckpt"
SUMMARY_JSON = HERE.parent / "docs" / "scale_summary.json"
DEFAULT_TARGETS = [
    # Exact n=915 twins (un-slid + asymmetric slack)
    686, 791, 911,
    # High un-slid corner-void targets
    985, 751, 859, 639, 800, 690, 696, 498, 796, 592, 739, 802, 909, 975,
    # Asymmetric-slack & remaining brief targets
    683, 740, 857, 854, 918, 919, 848, 980, 986, 923, 922, 803, 976, 458, 697,
]


def load_existing_wins() -> dict[int, dict]:
    """Load already-certified win_n*.csv files from poc/swarm_armslack/data/."""
    union_mod = rg._load_union_mod()
    wins: dict[int, dict] = {}
    if not rg.DATA_DIR.exists():
        return wins
    for p in sorted(rg.DATA_DIR.glob("win_n*.csv")):
        lit_g, pd_g = union_mod.load(p)
        for n, rows in lit_g.items():
            res = union_mod.check((n, rows, pd_g[n]))
            if res is not None:
                wins[int(n)] = {
                    "rows": [
                        {
                            "square_id": str(r["square_id"]),
                            "x": r["x"],
                            "y": r["y"],
                            "angle": r["angle"],
                        }
                        for r in sorted(rows, key=lambda z: int(z["square_id"]))
                    ],
                    "verify": {"ok": True, "L_lit": res[0], "L_pandas": res[1]},
                }
    return wins


def solve_one(
    n: int,
    x_b: np.ndarray,
    y_b: np.ndarray,
    th_b: np.ndarray,
    L_b: float,
    L_base_lit: Decimal,
    L_base_pd: Decimal,
    n_threads: int,
) -> dict:
    t0 = time.time()
    rng = np.random.default_rng(22000 + n)
    info = ss.detect_endcap_strips(x_b, y_b, th_b, L_b)

    # Combine strip-junction seeds + outer-endcap+BL seeds (always include both when unslid)
    q1, m1, l1, lb1 = ss.build_armslack_seeds(n, x_b, y_b, th_b, L_b, rng, mode="strip")
    q2, m2, l2, lb2 = ss.build_armslack_seeds(n, x_b, y_b, th_b, L_b, rng, mode="asym")
    q_list = q1[:16] + q2[:16]
    m_list = m1[:16] + m2[:16]
    ls_list = l1[:16] + l2[:16]
    labels = lb1[:16] + lb2[:16]

    ils_iters = [5, 8, 12, 15]
    qb, qf, lf, eo, po, lb = run_cpp_masked_melt(
        n, 1, q_list, m_list, ls_list, ils_iters,
        a_start=0.78, n_melt_stages=4, masked_iters=180, full_iters=180,
        n_threads=n_threads, base_seed=99000 + n,
    )
    n_feas = int(np.sum(lf < 1e9))
    best_feas = float(np.min(lf)) if n_feas else None
    print(
        f"\n=== Scale n={n} | L_b={L_b:.9f} dA={info['dA']:+.5f} dB={info['dB']:+.5f} "
        f"r_y0={info['r_y0']:.3f} t_x0={info['t_x0']:.3f} | N={len(labels)} feas={n_feas} min_pen={float(np.min(po)):.2e} ===",
        flush=True,
    )

    score_fn = lambda i: (0 if lf[i] < 1e9 else 1, float(lf[i]) if lf[i] < 1e9 else float(po[i]))
    order = sorted(range(len(labels)), key=score_fn)
    keep, used = [], set()
    for pref in ("asym_ec_bl_", "asym_corridor_", "asym_junc_bl_", "r_slide_", "t_slide_", "rt_slide_"):
        for i in order:
            if i not in used and labels[i].startswith(pref):
                keep.append(i)
                used.add(i)
                break
    for i in order:
        if len(keep) >= 6:
            break
        if i not in used:
            keep.append(i)
            used.add(i)

    tasks = []
    for i in keep:
        use_f = lf[i] < 1e9
        L_ref = float(lf[i]) if use_f else ls_list[i][-1]
        x, y, th = cpp_expand_sym(n, 1, L_ref, qf[i] if use_f else qb[i])
        tasks.append((i, labels[i], n, L_b, x, y, th, 0.0 if use_f else float(po[i]), float(lf[i]) if use_f else float(lb[i])))

    with mp.Pool(processes=min(4, max(1, n_threads), len(tasks))) as pool:
        polished = pool.map(rg._polish_worker, tasks)
    polished.sort(key=lambda r: (0 if r["verify"].get("ok") else 1, float(r["verify"].get("L_float", 1e9)) if r["verify"].get("ok") else r["pen_fin"]))

    best_valid, best_improve, best_dL = None, None, None
    for r in polished:
        v = r["verify"]
        if v.get("ok"):
            dl_lit = Decimal(v["L_lit"]) - L_base_lit
            dl_pd = Decimal(v["L_pandas"]) - L_base_pd
            worst_dl = max(dl_lit, dl_pd)
            if best_valid is None or worst_dl < best_dL:
                best_valid = r
                best_dL = worst_dl
            if dl_lit < Decimal("0") and dl_pd < Decimal("0"):
                if best_improve is None or worst_dl < max(Decimal(best_improve["verify"]["L_lit"]) - L_base_lit, Decimal(best_improve["verify"]["L_pandas"]) - L_base_pd):
                    best_improve = r

    status = "MISS"
    dL_val = None
    if best_improve is not None:
        v = best_improve["verify"]
        dL_val = float(max(Decimal(v["L_lit"]) - L_base_lit, Decimal(v["L_pandas"]) - L_base_pd))
        status = "REAL_WIN" if dL_val <= -1e-9 else "TIE"
        print(f"  [{status}] n={n}: {best_improve['label']} dL={dL_val:+.6e} L_lit={v['L_lit'][:18]} [{time.time()-t0:.1f}s]", flush=True)
    elif best_valid is not None:
        dL_val = float(best_dL)
        print(f"  [MISS] n={n}: best={best_valid['label']} dL={dL_val:+.6e} [{time.time()-t0:.1f}s]", flush=True)
    else:
        print(f"  [NO VALID] n={n} [{time.time()-t0:.1f}s]", flush=True)

    return {
        "n": n,
        "L_base": L_b,
        "L_base_lit": str(L_base_lit),
        "L_base_pd": str(L_base_pd),
        "dt_s": round(time.time() - t0, 2),
        "n_starts": len(labels),
        "n_feas": n_feas,
        "best_cpp_Lf": best_feas,
        "min_cpp_pen": float(np.min(po)),
        "status": status,
        "best_dL": dL_val,
        "top_polished": [{k: v for k, v in r.items() if k != "rows"} for r in polished],
        "win": best_improve,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", type=int, nargs="+", default=DEFAULT_TARGETS)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--stop-consec-nonreal", type=int, default=12)
    ap.add_argument("--summary-json", type=Path, default=SUMMARY_JSON)
    args = ap.parse_args()
    n_threads = min(4, max(1, args.threads))

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    t_all = time.time()
    merged_df = pd.read_csv(rg.MERGED_CSV)
    base_puzzles = rg.read_submission(rg.MERGED_CSV)
    wins = load_existing_wins()
    print(f"Loaded existing wins: {sorted(wins.keys())}", flush=True)

    results = []
    consec_nonreal = 0
    for n in args.targets:
        ckpt_path = CKPT_DIR / f"n_{n}.json"
        if ckpt_path.exists():
            res = json.loads(ckpt_path.read_text())
            results.append(res)
            if res["status"] == "REAL_WIN":
                consec_nonreal = 0
            else:
                consec_nonreal += 1
            if consec_nonreal >= args.stop_consec_nonreal:
                print(f"Stopping at {consec_nonreal} consecutive non-real-wins.", flush=True)
                break
            continue

        x_b, y_b, th_b, L_b, L_lit, L_pd = rg.load_base_puzzle(n, merged_df, base_puzzles)
        res = solve_one(n, x_b, y_b, th_b, L_b, L_lit, L_pd, n_threads=n_threads)
        w = res.pop("win")
        results.append(res)
        ckpt_path.write_text(json.dumps(res, indent=2))

        if w is not None and res["status"] == "REAL_WIN":
            wins[n] = w
            rg.DATA_DIR.mkdir(parents=True, exist_ok=True)
            rg.write_final_csv({n: w}, rg.DATA_DIR / f"win_n{n}.csv")
            rg.write_final_csv(wins, rg.FINAL_CSV)
            print(f"  [SAVED] n={n} ({res['status']}) -> updated final.csv (total={sorted(wins.keys())})", flush=True)

        if res["status"] == "REAL_WIN":
            consec_nonreal = 0
        else:
            consec_nonreal += 1

        args.summary_json.write_text(
            json.dumps(
                {
                    "results": results,
                    "wins": sorted(wins.keys()),
                    "consec_nonreal": consec_nonreal,
                    "dt_total_s": round(time.time() - t_all, 2),
                },
                indent=2,
            )
        )
        if consec_nonreal >= args.stop_consec_nonreal:
            print(f"Stopping early after {consec_nonreal}/{args.stop_consec_nonreal} consecutive non-real-wins.", flush=True)
            break

    print(f"\nScale sweep finished in {time.time()-t_all:.1f}s | wins={sorted(wins.keys())}", flush=True)


if __name__ == "__main__":
    main()
