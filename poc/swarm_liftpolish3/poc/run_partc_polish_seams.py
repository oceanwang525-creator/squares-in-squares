#!/usr/bin/env python3
"""Part C seam builder + lpslack + pslp.squeeze + mechwalk polish on origin/main 1f337f9."""
from __future__ import annotations

import os

for _v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_v] = "1"

import argparse
import concurrent.futures
import json
import math
import sys
import time
from decimal import Decimal
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "poc" / "pocket_lift_scan3" / "poc"))
sys.path.insert(0, str(REPO / "poc" / "swarm_lpslack" / "poc"))
sys.path.insert(0, str(REPO / "poc" / "swarm_homocasc2" / "poc"))
sys.path.insert(0, str(REPO / "poc" / "row_ratchet_980" / "poc"))

import build_cert as bc  # noqa: E402
import lpslack  # noqa: E402
import homocasc2_lib as hc  # noqa: E402
import pslp  # noqa: E402

hc.save_win_and_rebuild_final = lambda *a, **kw: None
hc.rebuild_final_csv = lambda *a, **kw: None

DATA_DIR = REPO / "poc" / "swarm_liftpolish3" / "data"
WINS_DIR = DATA_DIR / "wins_r2"
REPORT_OUT = REPO / "poc" / "swarm_liftpolish3" / "docs" / "partc_seams_results.json"


def save_win(n_target: int, L_inc_dec: Decimal, rows: list[dict], label: str) -> dict | None:
    cert = hc.certify_rows_against_frontier(n_target, rows, L_inc_dec)
    if not cert or not cert.get("is_real_win"):
        return None
    WINS_DIR.mkdir(parents=True, exist_ok=True)
    out_p = WINS_DIR / f"win_n{n_target}.json"
    if out_p.exists():
        try:
            old = json.loads(out_p.read_text(encoding="utf-8"))
            if Decimal(old["L_lit"]) <= Decimal(cert["L_lit"]):
                return cert
        except Exception:
            pass
    payload = {
        "n": n_target,
        "label": label,
        "L_lit": cert["L_lit"],
        "L_pd": cert["L_pd"],
        "L_kr": cert["L_kr"],
        "L_float": cert["L_float"],
        "L_inc": str(L_inc_dec),
        "dL": cert["dL"],
        "rows": rows,
    }
    out_p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return cert


def polish_rows(n_target: int, rows: list[dict], L_inc_dec: Decimal, label_prefix: str) -> dict:
    t0 = time.time()
    init_res = lpslack.certify_pair(n_target, rows)
    if init_res is None:
        return {"n": n_target, "status": "INIT_INFEAS", "label": label_prefix, "elapsed_s": round(time.time() - t0, 1)}
    cur_rows = rows
    cur_L = max(init_res[0], init_res[1])
    best_cert = save_win(n_target, L_inc_dec, cur_rows, f"{label_prefix}_build")

    # 1. lpslack
    lp_res = lpslack.optimize_group(
        n_target,
        cur_rows,
        str(cur_L),
        outer_iters=10,
        fixed_rounds=10,
        tr_th_schedule=(1e-3, 5e-4, 1e-4, 5e-5),
    )
    if lp_res.get("status") in ("WIN", "TINY_WIN") and "rows" in lp_res:
        cand_rows = lp_res["rows"]
        c_lp = save_win(n_target, L_inc_dec, cand_rows, f"{label_prefix}_lpslack")
        res_lp = lpslack.certify_pair(n_target, cand_rows)
        if res_lp is not None and max(res_lp[0], res_lp[1]) < cur_L:
            cur_rows = cand_rows
            cur_L = max(res_lp[0], res_lp[1])
        if c_lp is not None and (best_cert is None or Decimal(c_lp["L_lit"]) < Decimal(best_cert["L_lit"])):
            best_cert = c_lp

    # 2. If promising (within +2e-3 of L_inc_dec or already winning), run pslp.squeeze + mechwalk + lpslack
    if cur_L < L_inc_dec + Decimal("2e-3"):
        x, y, th, _ = hc._rows_to_arrays(cur_rows)
        xs, ys, ts, L_norm = hc.norm_pack(x, y, th)
        K = int(math.floor(L_norm))
        f0 = L_norm - K
        try:
            f_sq, Xn, Yn, Tn, vn, _ = pslp.squeeze(xs, ys, ts, K, K, f0, max_it=300)
            if vn <= 1e-11:
                c_sq, _ = hc.polish_candidate(n_target, Xn, Yn, Tn, cur_L, use_lpslack=True)
                if c_sq is not None and Decimal(c_sq["L_lit"]) < cur_L:
                    cur_rows = c_sq["rows"]
                    cur_L = Decimal(max(c_sq["L_lit"], c_sq["L_pd"]))
                    saved = save_win(n_target, L_inc_dec, cur_rows, f"{label_prefix}_pslp")
                    if saved is not None and (best_cert is None or Decimal(saved["L_lit"]) < Decimal(best_cert["L_lit"])):
                        best_cert = saved
        except Exception as exc:
            print(f"[{n_target}] pslp err: {exc}", flush=True)

        x, y, th, _ = hc._rows_to_arrays(cur_rows)
        rng = np.random.default_rng(20261114 + n_target)
        mw_cert, _ = hc.mechwalk_compress(n_target, x, y, th, cur_L, rng, rounds=3, threads=1)
        if mw_cert is not None and Decimal(mw_cert["L_lit"]) < cur_L - Decimal("1e-12"):
            cur_rows = mw_cert["rows"]
            cur_L = Decimal(max(mw_cert["L_lit"], mw_cert["L_pd"]))
            saved = save_win(n_target, L_inc_dec, cur_rows, f"{label_prefix}_mw")
            if saved is not None and (best_cert is None or Decimal(saved["L_lit"]) < Decimal(best_cert["L_lit"])):
                best_cert = saved

    dL_best = float(Decimal(best_cert["L_lit"]) - L_inc_dec) if best_cert else float(cur_L - L_inc_dec)
    return {
        "n": n_target,
        "status": "WIN" if best_cert else "NO_WIN",
        "L_inc": str(L_inc_dec),
        "L_best": best_cert["L_lit"] if best_cert else str(cur_L),
        "dL": dL_best,
        "label": label_prefix,
        "elapsed_s": round(time.time() - t0, 1),
    }


def _worker_built(arg: tuple[int, str, str, str]) -> dict:
    n, json_path, L_inc_str, label = arg
    rows = json.loads(Path(json_path).read_text(encoding="utf-8"))
    return polish_rows(n, rows, Decimal(L_inc_str), label)


def _worker_seam(arg: tuple[int, int, list[dict], str, str]) -> dict:
    n, K, pk_specs, L_inc_str, label = arg
    donors = _DONORS
    built_pks = []
    for spec in pk_specs:
        pk = spec["p"]
        tr = spec["T"]
        sq, W, H = bc.pocket_setup(donors[pk["src"]], pk, transpose=tr)
        built_pks.append({
            "src": pk["src"],
            "w": spec["w"],
            "h": spec["h"],
            "g": spec["g"],
            "f": spec["f"],
            "boxh": (W, H),
            "squares": sq,
        })
    out, _ = bc.build(K, built_pks, donors)
    if out is None:
        return {"n": n, "status": "BUILD_NONE", "label": label, "elapsed_s": 0.0}
    rows = bc.to_rows(out, n)
    if rows is None:
        return {"n": n, "status": "COUNT_MISMATCH", "label": label, "elapsed_s": 0.0}
    return polish_rows(n, rows, Decimal(L_inc_str), label)


_DONORS = None


def _init_donors(csv_path: str) -> None:
    global _DONORS
    _DONORS = bc.load_rows(csv_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-dL-rig", type=float, default=2.5e-3)
    ap.add_argument("--max-seams", type=int, default=45)
    args = ap.parse_args()

    rep = json.loads((DATA_DIR / "partc_report.json").read_text(encoding="utf-8"))
    inc_str = {int(r["n"]): r["L"] for r in rep["results"]}
    inc_f = {n: float(L) for n, L in inc_str.items()}
    scan_ns = set(int(x) for x in (DATA_DIR / "partc_scan_rows.txt").read_text().split(",") if x)
    pockets = json.loads((DATA_DIR / "pockets_partc_all.json").read_text(encoding="utf-8"))

    # Stage 1: polish built_partc and built_partc_prop
    built_jobs = []
    for folder, tag in (("built_partc", "partc_dp"), ("built_partc_prop", "partc_prop")):
        d = DATA_DIR / folder
        if d.exists():
            for p in sorted(d.glob("n*.json")):
                n = int(p.stem[1:])
                built_jobs.append((n, str(p), inc_str[n], tag))

    results = []
    if built_jobs:
        print(f"Polishing {len(built_jobs)} built Part C packings...", flush=True)
        with concurrent.futures.ProcessPoolExecutor(max_workers=min(2, args.workers)) as ex:
            for res in ex.map(_worker_built, built_jobs):
                results.append(res)
                print(f"BUILT n={res['n']} {res['status']} dL={res.get('dL'):+.3e} ({res['elapsed_s']}s)", flush=True)

    # Stage 2: enumerate slack-tolerant 2-pocket (and 1-pocket) seams from Part C donors
    best = {}
    for p in pockets:
        for tr in (False, True):
            w, h = (p["h"], p["w"]) if tr else (p["w"], p["h"])
            g, f = p["g"], p["f"]
            k = (w, h, g)
            if k not in best or f < best[k]["f"]:
                best[k] = {"w": w, "h": h, "g": g, "f": f, "src": p["src"], "T": tr, "p": p}
    items = sorted(best.values(), key=lambda x: x["f"])
    pareto = []
    for a in items:
        if not any(b["w"] <= a["w"] and b["h"] <= a["h"] and b["g"] >= a["g"] for b in pareto):
            pareto.append(a)

    cands = {}
    for K in range(18, 32):
        for a in pareto:
            if a["w"] <= K and a["h"] <= K:
                n = K * K + a["g"]
                if n in inc_f and a["src"] != n and (a["src"] in scan_ns or a["src"] >= 2000):
                    dL = (K + a["f"]) - inc_f[n]
                    if dL <= args.max_dL_rig:
                        prev = cands.get(n)
                        if prev is None or dL < prev[0]:
                            cands[n] = (dL, K, [a])
        for i, a in enumerate(pareto):
            if a["w"] > K or a["h"] > K:
                continue
            for b in pareto[i:]:
                if a["w"] + b["w"] <= K and a["h"] + b["h"] <= K and (a["w"] + a["h"] + b["w"] + b["h"] + 2) <= 2 * K + 1:
                    n = K * K + a["g"] + b["g"]
                    if n not in inc_f or a["src"] == n or b["src"] == n:
                        continue
                    if not (a["src"] in scan_ns or a["src"] >= 2000 or b["src"] in scan_ns or b["src"] >= 2000):
                        continue
                    dL = (K + max(a["f"], b["f"])) - inc_f[n]
                    max_thr = 4.5e-3 if K >= 27 else args.max_dL_rig
                    if dL <= max_thr:
                        prev = cands.get(n)
                        if prev is None or dL < prev[0]:
                            cands[n] = (dL, K, [a, b])

    ranked = sorted(cands.items(), key=lambda kv: kv[1][0])[: args.max_seams]
    seam_jobs = []
    for n, (dL_rig, K, ps) in ranked:
        lbl = "+".join(f"{p['w']}x{p['h']}g{p['g']}s{p['src']}" for p in ps)
        seam_jobs.append((n, K, ps, inc_str[n], lbl))

    print(f"Running lpslack+pslp+mechwalk on {len(seam_jobs)} Part C seam candidates...", flush=True)
    donors_csv = str(DATA_DIR / "partc_donors.csv")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=min(2, args.workers), initializer=_init_donors, initargs=(donors_csv,)
    ) as ex:
        futs = [ex.submit(_worker_seam, j) for j in seam_jobs]
        for fut in concurrent.futures.as_completed(futs):
            res = fut.result()
            results.append(res)
            print(
                f"SEAM n={res['n']} {res['status']} dL={res.get('dL')} [{res['label']}] ({res['elapsed_s']}s)",
                flush=True,
            )
            REPORT_OUT.write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")

    wins = [r for r in results if r["status"] == "WIN"]
    print(f"Finished Part C polish+seams: {len(wins)} WIN(s) out of {len(results)}", flush=True)


if __name__ == "__main__":
    main()
