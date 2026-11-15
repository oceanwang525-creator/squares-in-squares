#!/usr/bin/env python3
"""Build predicted staircase seams in F3B_high_r (D=15..28, K=18..31), melt overlaps, polish via lpslack + mechwalk + pslp."""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import collections
import itertools
import json
import multiprocessing as mp_proc
import sys
import time
from decimal import Decimal
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
for _p in (
    str(REPO / "poc" / "baseline" / "poc" / "src"),
    str(REPO / "poc" / "auto_union" / "poc"),
    str(REPO / "poc" / "pocket_lift_scan3" / "poc"),
    str(REPO / "poc" / "swarm_homology" / "poc"),
    str(REPO / "poc" / "swarm_mechwalk" / "poc"),
    str(REPO / "poc" / "swarm_lpslack" / "poc"),
    str(REPO / "poc" / "swarm_homocasc2" / "poc"),
    str(REPO / "poc" / "swarm_softseamdp" / "poc"),
    str(REPO / "poc" / "row_ratchet_980" / "poc"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import build_cert as bc  # noqa: E402
import build_and_polish as bp  # noqa: E402
import homocasc2_lib as hlib  # noqa: E402
import lpslack  # noqa: E402
import pslp  # noqa: E402
import union as auto_union  # noqa: E402
from squares.core import certify, kaggle_roundtrip  # noqa: E402

HIGH_DIR = REPO / "poc" / "swarm_highdpocket"
DOCS_DIR = HIGH_DIR / "docs"
CKPT_DIR = DOCS_DIR / "ckpt"

_G_BY_KEY = None
_G_ROWS_ALL = None


def _is_self_split(n: int, seam: dict) -> bool:
    return all(int(c["src"]) in (n, 10000 + n) for c in seam["comp"])


def _pslp_polish(n: int, K: int, cert: dict, L_inc_dec: Decimal) -> dict | None:
    """Run kink-safe full-row pslp.squeeze followed by lpslack shrink+certify."""
    bx, by, bt, L_cur = hlib._rows_to_arrays(cert["rows"])
    f0 = L_cur - K + 1e-12
    try:
        fn, Xn, Yn, Tn, vn, _ = pslp.squeeze(bx, by, bt, K, K, f0, center=1e-4)
    except Exception:
        return None
    if vn > 1e-12:
        return None
    if K + fn >= L_cur - 1e-10:
        return None
    centers = np.column_stack([Xn, Yn])
    shrunk = lpslack.try_shrink_and_certify(n, centers, Tn, Decimal("999.0"))
    if shrunk is None:
        return None
    lit, pan, srows = shrunk
    cand = hlib.certify_rows_against_frontier(n, srows, L_inc_dec)
    if cand is not None and Decimal(max(cand["L_lit"], cand["L_pd"])) < Decimal(max(cert["L_lit"], cert["L_pd"])):
        return cand
    return None


def _save_ckpt(n: int, K: int, D: int, L_inc_str: str, cert: dict, label: str, meta: dict | None = None) -> None:
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CKPT_DIR / f"n_{n}.json"
    if ckpt_path.exists():
        prev = json.loads(ckpt_path.read_text())
        if Decimal(max(prev["L_lit"], prev["L_pd"])) <= Decimal(max(cert["L_lit"], cert["L_pd"])):
            return
    ckpt = {
        "n": n,
        "K": K,
        "D": D,
        "label": label,
        "L_inc": L_inc_str,
        "L_lit": cert["L_lit"],
        "L_pd": cert["L_pd"],
        "dL": cert["dL"],
        "meta": meta,
        "rows": cert["rows"],
    }
    auto_union.atomic(ckpt_path, json.dumps(ckpt, indent=1).encode())


def process_cell(args_tuple) -> dict:
    cell, by_key, rows_all, L_inc_str, run_mechwalk, max_seams, keep_self = args_tuple
    n = int(cell["n"])
    K = int(cell["K"])
    D = int(cell["D"])
    L_inc_dec = Decimal(L_inc_str)
    rng = np.random.default_rng(100003 + n * 17)
    t0 = time.time()

    best_cert: dict | None = None
    best_seam = None
    seam_logs = []

    seams_cand = cell["seams"] if keep_self else [s for s in cell["seams"] if not _is_self_split(n, s)]
    for s_idx, seam in enumerate(seams_cand[:max_seams]):
        pockets = bp.resolve_pockets(seam["comp"], by_key, rows_all)
        if pockets is None:
            seam_logs.append({"seam_idx": s_idx, "status": "missing_donor"})
            continue

        m = len(pockets)
        candidates_geom = []
        for order in itertools.permutations(range(m)):
            pt = [
                (pockets[i]["w"], pockets[i]["h"], float(pockets[i]["boxh"][0]) - pockets[i]["w"], float(pockets[i]["boxh"][1]) - pockets[i]["h"])
                for i in order
            ]
            for X, Y in bc.layouts_all(K, pt):
                xs, ys, ths, S_f, n_pk, raw_tot, exact_rows = bp.build_layout_arrays(K, n, pockets, order, X, Y)
                if xs.size == 0:
                    continue
                candidates_geom.append((abs(raw_tot - n), -(raw_tot - n), xs, ys, ths, S_f, n_pk, raw_tot, exact_rows))

        if not candidates_geom:
            seam_logs.append({"seam_idx": s_idx, "status": "no_layout"})
            continue

        candidates_geom.sort(key=lambda t: (0 if t[7] >= n else 1, t[0]))
        tried_feas = False
        best_seam_dL = None
        notes_all = []
        for l_idx, (_, _, xs, ys, ths, S_f, n_pk, raw_tot, exact_rows) in enumerate(candidates_geom[:2]):
            cert_init = None
            if raw_tot >= n and exact_rows is not None:
                try:
                    Llit = certify(n, exact_rows)
                    Lpan = certify(n, kaggle_roundtrip({n: exact_rows})[n])
                    cert_init = (Llit, Lpan)
                except Exception:
                    cert_init = None

            if cert_init is not None:
                tried_feas = True
                c_res, notes = bp.screen_layout(n, xs, ys, ths, L_inc_dec)
                notes_all.extend(notes)
            else:
                melted = bp.melt_overlapping_seam(n, xs, ys, ths, n_pk, L_inc_dec, rng)
                if melted is None:
                    notes_all.append(f"L{l_idx}:melt_fail(raw={raw_tot})")
                    continue
                tried_feas = True
                mx, my, mth, _ = melted
                c_res, notes = bp.screen_layout(n, mx, my, mth, L_inc_dec)
                notes_all.extend([f"L{l_idx}:melted"] + notes)

            if c_res is not None:
                if best_seam_dL is None or c_res["dL"] < best_seam_dL:
                    best_seam_dL = c_res["dL"]
                if best_cert is None or Decimal(max(c_res["L_lit"], c_res["L_pd"])) < Decimal(max(best_cert["L_lit"], best_cert["L_pd"])):
                    best_cert = c_res
                    best_seam = seam
                    if c_res["is_real_win"]:
                        _save_ckpt(n, K, D, L_inc_str, best_cert, "seam_screen", best_seam)
            if c_res is not None and c_res["dL"] <= -5e-4:
                break

        seam_logs.append({
            "seam_idx": s_idx,
            "m": seam["m"],
            "dL_rigid": seam["dL_rigid"],
            "feasible": tried_feas,
            "best_dL": best_seam_dL,
            "notes": notes_all,
        })
        if best_cert is not None and best_cert["dL"] <= -5e-4:
            break

    if best_cert is not None and best_cert["dL"] < 2.0e-3:
        bx, by, bt, _ = hlib._rows_to_arrays(best_cert["rows"])
        c_deep = bp._run_lpslack_fast(
            n, bx, by, bt, L_inc_dec, cycles=2, outer_iters=10, fixed_rounds=8, slp_modes=("dual_non90", "non90"), time_limit=8.0
        )
        if c_deep is not None and Decimal(max(c_deep["L_lit"], c_deep["L_pd"])) < Decimal(max(best_cert["L_lit"], best_cert["L_pd"])):
            best_cert = c_deep
            bx, by, bt, _ = hlib._rows_to_arrays(best_cert["rows"])

        c_pslp = _pslp_polish(n, K, best_cert, L_inc_dec)
        if c_pslp is not None:
            best_cert = c_pslp
            bx, by, bt, _ = hlib._rows_to_arrays(best_cert["rows"])

        if run_mechwalk and best_cert["dL"] < 1.5e-3:
            mw_cert, _ = hlib.mechwalk_compress(n, bx, by, bt, L_inc_dec, rng, rounds=2, threads=1)
            if mw_cert is not None and Decimal(max(mw_cert["L_lit"], mw_cert["L_pd"])) < Decimal(max(best_cert["L_lit"], best_cert["L_pd"])):
                best_cert = mw_cert
                c_pslp2 = _pslp_polish(n, K, best_cert, L_inc_dec)
                if c_pslp2 is not None:
                    best_cert = c_pslp2

        if best_cert["is_real_win"]:
            _save_ckpt(n, K, D, L_inc_str, best_cert, "seam_deep", best_seam)

    elapsed = round(time.time() - t0, 2)
    out = {
        "n": n,
        "K": K,
        "D": D,
        "L_inc": L_inc_str,
        "best_dL_rigid": cell["best_dL_rigid"],
        "elapsed_s": elapsed,
        "seam_logs": seam_logs,
        "win": bool(best_cert is not None and best_cert["is_real_win"]),
    }
    if best_cert is not None:
        out["L_lit"] = best_cert["L_lit"]
        out["L_pd"] = best_cert["L_pd"]
        out["dL"] = best_cert["dL"]
    return out


def _run_cell_fork(args_tuple) -> dict:
    cell, L_inc_str, run_mechwalk, max_seams, keep_self = args_tuple
    return process_cell((cell, _G_BY_KEY, _G_ROWS_ALL, L_inc_str, run_mechwalk, max_seams, keep_self))


def export_final_csv() -> Path:
    groups = {}
    if CKPT_DIR.exists():
        for p in sorted(CKPT_DIR.glob("n_*.json")):
            d = json.loads(p.read_text())
            groups[int(d["n"])] = d["rows"]
    text = auto_union.final_text(groups)
    out_csv = HIGH_DIR / "final.csv"
    auto_union.atomic(out_csv, text.encode())
    os.chmod(out_csv, 0o644)
    return out_csv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--pockets", nargs="+", required=True)
    ap.add_argument("--limit", type=int, default=6)
    ap.add_argument("--n-list")
    ap.add_argument("--d-min", type=int, default=15)
    ap.add_argument("--d-max", type=int, default=28)
    ap.add_argument("--max-seams", type=int, default=3)
    ap.add_argument("--keep-self", action="store_true")
    ap.add_argument("--no-mechwalk", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pred = json.loads(Path(args.pred).read_text())["cells"]
    rep = {int(r["n"]): r["L"] for r in json.loads(Path(args.report).read_text())["results"]}
    by_key = bp.load_pockets_index(args.pockets)
    rows_all = bc.load_rows(args.csv)

    cells = [c for c in pred if args.d_min <= int(c["D"]) <= args.d_max and 18 <= int(c["K"]) <= 31]
    if args.n_list:
        want = {int(x) for x in args.n_list.split(",") if x.strip()}
        cells = [c for c in cells if c["n"] in want]
    else:
        if not args.keep_self:
            cells = [c for c in cells if any(not _is_self_split(c["n"], s) for s in c["seams"])]
        if args.limit > 0:
            cells = cells[: args.limit]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    global _G_BY_KEY, _G_ROWS_ALL
    _G_BY_KEY = by_key
    _G_ROWS_ALL = rows_all

    print(f"Seam build/polish: {len(cells)} cells (workers={min(2, args.workers)}, max_seams={args.max_seams})", flush=True)
    tasks = [(c, rep[c["n"]], not args.no_mechwalk, args.max_seams, args.keep_self) for c in cells]
    results = []
    if min(2, args.workers) <= 1 or len(tasks) <= 1:
        for t in tasks:
            r = _run_cell_fork(t)
            results.append(r)
            print(f"  n={r['n']} (K={r['K']},D={r['D']}) dL_rigid={r['best_dL_rigid']:+.6f} -> win={r['win']} best_dL={r.get('dL')} ({r['elapsed_s']}s)", flush=True)
            out_path.write_text(json.dumps(results, indent=1))
    else:
        with mp_proc.get_context("fork").Pool(2) as pool:
            for r in pool.imap_unordered(_run_cell_fork, tasks, chunksize=1):
                results.append(r)
                print(f"  n={r['n']} (K={r['K']},D={r['D']}) dL_rigid={r['best_dL_rigid']:+.6f} -> win={r['win']} best_dL={r.get('dL')} ({r['elapsed_s']}s)", flush=True)
                out_path.write_text(json.dumps(results, indent=1))

    out_path.write_text(json.dumps(results, indent=1))
    export_final_csv()
    wins = [r for r in results if r["win"]]
    print(f"Done: {len(wins)}/{len(results)} certified wins", flush=True)


if __name__ == "__main__":
    main()
