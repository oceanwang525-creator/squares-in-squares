#!/usr/bin/env python3
"""Build soft-seam DP candidates + run kink-safe pslp.squeeze <-> mechwalk <-> lpslack ratchet."""
from __future__ import annotations

import os

for _var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_var] = "1"

import argparse
import concurrent.futures
import importlib.util
import itertools
import json
import math
import multiprocessing as mp
import sys
import time
from decimal import Decimal
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
SOFT_DIR = REPO / "poc" / "swarm_softseamdp"
DATA_DIR = SOFT_DIR / "data"
WORK_DIR = DATA_DIR / "ratchet_work"
CKPT_DIR = SOFT_DIR / "docs" / "ckpt"

for p in (
    REPO / "poc" / "swarm_softseamdp" / "poc",
    REPO / "poc" / "row_ratchet_980" / "poc",
    REPO / "poc" / "swarm_mechwalk" / "poc",
    REPO / "poc" / "swarm_highd" / "poc",
    REPO / "poc" / "pocket_lift_scan3" / "poc",
    REPO / "poc" / "swarm_lpslack" / "poc",
    REPO / "poc" / "auto_union" / "poc",
    REPO / "poc" / "baseline" / "poc" / "src",
):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

import build_and_polish as bp  # noqa: E402
import build_cert as bc  # noqa: E402
import mechwalk_lib as ml  # noqa: E402
import pslp  # noqa: E402
from cpp_wrap import get_cpp_lib  # noqa: E402

_RG_SPEC = importlib.util.spec_from_file_location("mechwalk_run_gate", REPO / "poc" / "swarm_mechwalk" / "poc" / "run_gate.py")
assert _RG_SPEC and _RG_SPEC.loader
run_gate = importlib.util.module_from_spec(_RG_SPEC)
_RG_SPEC.loader.exec_module(run_gate)

_G_BY_KEY = None
_G_ROWS_ALL = None


def _pd_str(cert: dict) -> str:
    return cert["L_pandas"] if "L_pandas" in cert else cert["L_pd"]


def _cert_dec(cert: dict) -> Decimal:
    return max(Decimal(cert["L_lit"]), Decimal(_pd_str(cert)))


def _norm_cert(cert: dict, L_inc_dec: Decimal | None = None) -> dict:
    dL = cert.get("dL")
    if dL is None and L_inc_dec is not None:
        dL = float(_cert_dec(cert) - L_inc_dec)
    return {"L_lit": cert["L_lit"], "L_pd": _pd_str(cert), "dL": dL, "rows": cert["rows"]}


def _certify_with_micro_expand(n: int, x: np.ndarray, y: np.ndarray, th: np.ndarray) -> tuple[dict | None, np.ndarray, np.ndarray]:
    """Normalize and micro-expand if needed to certify cleanly in both literal and pandas."""
    xn, yn, thn, _ = ml.normalize(x.copy(), y.copy(), th.copy())
    best_cert = None
    best_dec = None
    best_xy = (xn, yn)
    for eps in (0.0, 1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8):
        sx = xn * (1.0 + eps)
        sy = yn * (1.0 + eps)
        cert = ml.certify_canonical(n, sx, sy, thn)
        if cert.get("ok"):
            dec = max(Decimal(cert["L_lit"]), Decimal(cert["L_pandas"]))
            if best_dec is None or dec < best_dec:
                best_dec = dec
                best_cert = cert
                best_xy = (sx, sy)
            break
    return best_cert, best_xy[0], best_xy[1]


def pslp_squeeze_cert(
    n: int, K: int, x: np.ndarray, y: np.ndarray, th: np.ndarray, cur_dec: Decimal, try_center: bool = True
) -> tuple[dict | None, np.ndarray, np.ndarray, np.ndarray, Decimal]:
    """Run kink-safe pslp.squeeze (optionally with center > 0 kick) and dual-certify."""
    best_cert = None
    best_dec = cur_dec
    best_state = (x.copy(), y.copy(), th.copy())

    def _one(xc: np.ndarray, yc: np.ndarray, tc: np.ndarray, center: float, max_it: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        xn, yn, tn, L0 = ml.normalize(xc.copy(), yc.copy(), tc.copy())
        f, Xf, Yf, Tf, v, _ = pslp.squeeze(
            xn, yn, tn, K, K, L0 - K + 1e-12, trust0=0.02, max_it=max_it, center=center
        )
        return Xf, Yf, Tf, v

    X1, Y1, T1, v1 = _one(x, y, th, 0.0, 400)
    if v1 <= 1e-11:
        cert, sx, sy = _certify_with_micro_expand(n, X1, Y1, T1)
        if cert is not None:
            dec = max(Decimal(cert["L_lit"]), Decimal(cert["L_pandas"]))
            if dec < best_dec:
                best_dec = dec
                best_cert = cert
                best_state = (sx.copy(), sy.copy(), T1.copy())

    if try_center:
        Xc, Yc, Tc, vc = _one(best_state[0], best_state[1], best_state[2], 0.02, 160)
        if vc <= 1e-11:
            X2, Y2, T2, v2 = _one(Xc, Yc, Tc, 0.0, 400)
            if v2 <= 1e-11:
                cert2, sx2, sy2 = _certify_with_micro_expand(n, X2, Y2, T2)
                if cert2 is not None:
                    dec2 = max(Decimal(cert2["L_lit"]), Decimal(cert2["L_pandas"]))
                    if dec2 < best_dec:
                        best_dec = dec2
                        best_cert = cert2
                        best_state = (sx2.copy(), sy2.copy(), T2.copy())

    return best_cert, best_state[0], best_state[1], best_state[2], best_dec


def _rows_to_arrays(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sr = sorted(rows, key=lambda r: int(r["square_id"]))
    x = np.array([float(r["x"]) for r in sr], dtype=np.float64)
    y = np.array([float(r["y"]) for r in sr], dtype=np.float64)
    th = np.array([float(r["angle"]) for r in sr], dtype=np.float64)
    return x, y, th


def build_best_for_cell(args_tuple) -> dict:
    """Build all qualifying seams/layouts for one (K, D) cell, run lpslack + pslp, return best feasible."""
    cell, L_inc_str, max_seams, keep_self, max_keep_dL = args_tuple
    t0 = time.time()
    n = int(cell["n"])
    K = int(cell["K"])
    D = int(cell["D"])
    L_inc_dec = Decimal(L_inc_str)
    rng = np.random.default_rng(31000 + n)

    best_dec: Decimal | None = None
    best_cert: dict | None = None
    best_seam: dict | None = None
    notes: list[str] = []

    seams_tried = 0
    for si, seam in enumerate(cell["seams"]):
        if seams_tried >= max_seams:
            break
        if not keep_self and bp._is_self_split(n, seam):
            continue
        seams_tried += 1
        pockets = bp.resolve_pockets(seam["comp"], _G_BY_KEY, _G_ROWS_ALL)
        if pockets is None:
            continue
        m = len(pockets)
        lay_cands = []
        seen_xy = set()
        for order in itertools.permutations(range(m)):
            pt = [
                (pockets[i]["w"], pockets[i]["h"], float(pockets[i]["boxh"][0]) - pockets[i]["w"], float(pockets[i]["boxh"][1]) - pockets[i]["h"])
                for i in order
            ]
            for X, Y in bc.layouts_all(K, pt):
                key = (tuple(X), tuple(Y), tuple((pockets[i]["w"], pockets[i]["h"], pockets[i]["g"], pockets[i]["src"]) for i in order))
                if key in seen_xy:
                    continue
                seen_xy.add(key)
                xs, ys, ths, S_f, n_pocket, n_full, rows_init = bp.build_layout_arrays(K, n, pockets, order, X, Y)
                if rows_init is None:
                    continue
                pen, L_raw = bp.cpp_eval_sat_pen_bbox(xs, ys, ths, side=1.0)
                lay_cands.append((pen > 1e-6, abs(n_full - n), pen, L_raw, xs, ys, ths, n_pocket))
        lay_cands.sort(key=lambda t: (t[0], t[1], t[2], t[3]))
        for _, _, pen, _, xs, ys, ths, n_pocket in lay_cands[:2]:
            # Certify rigid build if already valid
            cert_raw = ml.certify_canonical(n, xs, ys, ths)
            if cert_raw.get("ok"):
                dec = _cert_dec(cert_raw)
                if best_dec is None or dec < best_dec:
                    best_dec, best_cert, best_seam = dec, _norm_cert(cert_raw, L_inc_dec), seam
            elif pen > 1e-9:
                # Melt seam overlap
                melted = bp.melt_overlapping_seam(n, xs, ys, ths, n_pocket, L_inc_dec + Decimal("0.01"), rng)
                if melted is None:
                    continue
                xs, ys, ths, _ = melted
                cert_m = ml.certify_canonical(n, xs, ys, ths)
                if cert_m.get("ok"):
                    dec = _cert_dec(cert_m)
                    if best_dec is None or dec < best_dec:
                        best_dec, best_cert, best_seam = dec, _norm_cert(cert_m, L_inc_dec), seam

            # Fast lpslack polish on this layout
            lp_res = bp._run_lpslack_fast(
                n,
                xs,
                ys,
                ths,
                L_inc_dec,
                cycles=1,
                outer_iters=7,
                fixed_rounds=7,
                slp_modes=("dual_non90",),
                time_limit=6.0,
            )
            if lp_res is not None and lp_res.get("ok"):
                dec = _cert_dec(lp_res)
                if best_dec is None or dec < best_dec:
                    best_dec, best_cert, best_seam = dec, _norm_cert(lp_res, L_inc_dec), seam

            # Quick pslp.squeeze check on current layout's best feasible state
            cur_cert = lp_res if (lp_res is not None and lp_res.get("ok")) else cert_raw
            if cur_cert is not None and cur_cert.get("ok"):
                cx, cy, cth = _rows_to_arrays(cur_cert["rows"])
                cur_d = _cert_dec(cur_cert)
                p_cert, _, _, _, p_dec = pslp_squeeze_cert(n, K, cx, cy, cth, cur_d, try_center=False)
                if p_cert is not None and (best_dec is None or p_dec < best_dec):
                    best_dec, best_cert, best_seam = p_dec, _norm_cert(p_cert, L_inc_dec), seam
                    notes.append(f"s{si}_pslp={float(p_dec - L_inc_dec):+.3e}")

    out = {
        "n": n,
        "K": K,
        "D": D,
        "L_inc": L_inc_str,
        "best_dL_rigid": cell["best_dL_rigid"],
        "elapsed_s": round(time.time() - t0, 2),
        "feasible": best_cert is not None,
        "win": False,
        "dL": None,
        "notes": notes,
    }
    if best_cert is not None and best_dec is not None:
        dL = float(best_dec - L_inc_dec)
        out["dL"] = dL
        out["L_lit"] = best_cert["L_lit"]
        out["L_pd"] = _pd_str(best_cert)
        out["win"] = best_dec < L_inc_dec
        if out["win"]:
            bp._save_ckpt(
                n,
                K,
                D,
                L_inc_str,
                _norm_cert(best_cert),
                best_seam,
            )
        if dL <= max_keep_dL:
            WORK_DIR.mkdir(parents=True, exist_ok=True)
            (WORK_DIR / f"cand_{n}.json").write_text(
                json.dumps(
                    {
                        "n": n,
                        "K": K,
                        "D": D,
                        "L_inc": L_inc_str,
                        "L_lit": best_cert["L_lit"],
                        "L_pd": _pd_str(best_cert),
                        "dL": dL,
                        "seam": best_seam,
                        "rows": best_cert["rows"],
                    },
                    indent=1,
                )
            )
    print(
        f"  [build] n={n} ({K},{D}) feas={out['feasible']} win={out['win']} "
        f"dL={out['dL']} notes={notes[:4]} [{out['elapsed_s']:.1f}s]",
        flush=True,
    )
    return out


def ratchet_one_candidate(cand_path: Path, cycles: int = 3, mech_rounds: int = 2) -> dict:
    """Alternate mechwalk (solve_target) -> pslp.squeeze -> lpslack on one feasible candidate."""
    t0 = time.time()
    d = json.loads(cand_path.read_text())
    n, K, D = int(d["n"]), int(d["K"]), int(d["D"])
    L_inc_dec = Decimal(d["L_inc"])
    best_cert = {"L_lit": d["L_lit"], "L_pd": d["L_pd"], "rows": d["rows"]}
    best_dec = max(Decimal(d["L_lit"]), Decimal(d["L_pd"]))
    start_dL = float(best_dec - L_inc_dec)
    x, y, th = _rows_to_arrays(d["rows"])
    x, y, th, Lf = ml.normalize(x, y, th)

    trace = [f"start={start_dL:+.3e}"]
    for cyc in range(cycles):
        prev_dec = best_dec
        # 1. Mechwalk mode-walk + mode-melt from current packing
        inc_state = {
            "x": x.copy(),
            "y": y.copy(),
            "th": th.copy(),
            "L": Lf,
            "L_dec": best_dec,
            "source": "softseam_ratchet",
        }
        try:
            _, mw_win = run_gate.solve_target(
                n,
                inc_state,
                n_rounds=mech_rounds,
                beam_width=2,
                top_k_polish=6,
                n_threads=2,
            )
        except Exception as exc:
            trace.append(f"c{cyc}_mw_err={exc}")
            mw_win = None

        if mw_win is not None:
            dec_mw = _cert_dec(mw_win)
            if dec_mw < best_dec:
                best_dec = dec_mw
                best_cert = _norm_cert(mw_win)
                x, y, th = _rows_to_arrays(mw_win["rows"])
                x, y, th, Lf = ml.normalize(x, y, th)
                trace.append(f"c{cyc}_mw={float(best_dec - L_inc_dec):+.3e}")

        # 2. Kink-safe pslp.squeeze (+ center kick)
        p_cert, px, py, pth, p_dec = pslp_squeeze_cert(n, K, x, y, th, best_dec, try_center=True)
        if p_cert is not None and p_dec < best_dec:
            best_dec = p_dec
            best_cert = _norm_cert(p_cert)
            x, y, th, Lf = ml.normalize(px, py, pth)
            trace.append(f"c{cyc}_pslp={float(best_dec - L_inc_dec):+.3e}")

        # 3. Deep lpslack + final pslp tighten
        lp_res = bp._run_lpslack_fast(
            n,
            x,
            y,
            th,
            best_dec,
            cycles=2,
            outer_iters=14,
            fixed_rounds=12,
            slp_modes=("dual_non90",),
            time_limit=15.0,
        )
        if lp_res is not None and lp_res.get("ok"):
            dec_lp = _cert_dec(lp_res)
            if dec_lp < best_dec:
                best_dec = dec_lp
                best_cert = _norm_cert(lp_res)
                x, y, th = _rows_to_arrays(lp_res["rows"])
                x, y, th, Lf = ml.normalize(x, y, th)
                trace.append(f"c{cyc}_lp={float(best_dec - L_inc_dec):+.3e}")
                p2_cert, px2, py2, pth2, p2_dec = pslp_squeeze_cert(n, K, x, y, th, best_dec, try_center=False)
                if p2_cert is not None and p2_dec < best_dec:
                    best_dec = p2_dec
                    best_cert = _norm_cert(p2_cert)
                    x, y, th, Lf = ml.normalize(px2, py2, pth2)
                    trace.append(f"c{cyc}_pslp2={float(best_dec - L_inc_dec):+.3e}")

        if prev_dec - best_dec < Decimal("1e-11"):
            break

    win = best_dec < L_inc_dec
    if win:
        best_cert["dL"] = float(best_dec - L_inc_dec)
        bp._save_ckpt(n, K, D, d["L_inc"], best_cert, d.get("seam"))

    # Update candidate cache
    d.update(
        L_lit=best_cert["L_lit"],
        L_pd=best_cert["L_pd"],
        dL=float(best_dec - L_inc_dec),
        rows=best_cert["rows"],
    )
    cand_path.write_text(json.dumps(d, indent=1))
    res = {
        "n": n,
        "K": K,
        "D": D,
        "start_dL": start_dL,
        "end_dL": float(best_dec - L_inc_dec),
        "win": win,
        "trace": trace,
        "elapsed_s": round(time.time() - t0, 2),
    }
    print(f"  [ratchet] n={n} ({K},{D}) win={win} {start_dL:+.3e} -> {res['end_dL']:+.3e} {trace} [{res['elapsed_s']:.1f}s]", flush=True)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=str(DATA_DIR / "pred_all.json"))
    ap.add_argument("--report", default=str(DATA_DIR / "union_report.json"))
    ap.add_argument("--csv", default=str(DATA_DIR / "donors_all.csv"))
    ap.add_argument(
        "--pockets",
        nargs="+",
        default=[str(DATA_DIR / "pockets_all_sane.json"), str(DATA_DIR / "gadlib_pockets.json")],
    )
    ap.add_argument("--n-list", type=int, nargs="+")
    ap.add_argument("--max-seams", type=int, default=5)
    ap.add_argument("--keep-self", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--cycles", type=int, default=3)
    ap.add_argument("--mech-rounds", type=int, default=2)
    ap.add_argument("--max-keep-dL", type=float, default=3.0e-3)
    ap.add_argument("--ratchet-thresh", type=float, default=1.5e-3)
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--out", default=str(SOFT_DIR / "docs" / "ratchet_results.json"))
    args = ap.parse_args()

    get_cpp_lib()
    rep = {int(r["n"]): r["L"] for r in json.loads(Path(args.report).read_text())["results"]}
    pred = json.loads(Path(args.pred).read_text())["cells"]
    if args.n_list:
        nset = set(args.n_list)
        pred_map = {int(c["n"]): c for c in pred}
        cells = [pred_map[n] for n in args.n_list if n in pred_map]
    else:
        cells = pred

    build_results = []
    if not args.skip_build and cells:
        global _G_BY_KEY, _G_ROWS_ALL
        _G_BY_KEY = bp.load_pockets_index(args.pockets)
        _G_ROWS_ALL = bc.load_rows(args.csv)
        tasks = [(c, rep[int(c["n"])], args.max_seams, args.keep_self, args.max_keep_dL) for c in cells]
        print(f"Building {len(tasks)} DP cells with workers={min(2, args.workers)} ...", flush=True)
        w = max(1, min(2, args.workers))
        if w == 1:
            build_results = [build_best_for_cell(t) for t in tasks]
        else:
            ctx = mp.get_context("fork")
            with concurrent.futures.ProcessPoolExecutor(max_workers=w, mp_context=ctx) as ex:
                build_results = list(ex.map(build_best_for_cell, tasks))

    # Ratchet all saved candidates within ratchet_thresh (plus any requested n_list)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    ratchet_results = []
    cand_files = sorted(WORK_DIR.glob("cand_*.json"))
    if args.n_list:
        want = {int(n) for n in args.n_list}
        cand_files = [p for p in cand_files if int(p.stem.split("_")[1]) in want]

    # Sort closest dL first so best near-misses run immediately
    ranked = []
    for p in cand_files:
        d = json.loads(p.read_text())
        n = int(d["n"])
        # Always refresh L_inc against current union_report.json
        if n in rep:
            d["L_inc"] = rep[n]
            p.write_text(json.dumps(d, indent=1))
        cur_dec = max(Decimal(d["L_lit"]), Decimal(d["L_pd"]))
        dL_now = float(cur_dec - Decimal(d["L_inc"]))
        if dL_now <= args.ratchet_thresh:
            ranked.append((dL_now, p))
    ranked.sort()

    print(f"Running mechwalk <-> pslp <-> lpslack ratchet on {len(ranked)} candidates ...", flush=True)
    for _, p in ranked:
        ratchet_results.append(ratchet_one_candidate(p, cycles=args.cycles, mech_rounds=args.mech_rounds))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps({"build": build_results, "ratchet": ratchet_results}, indent=1)
    )
    wins = [r for r in ratchet_results if r["win"]]
    print(f"Done: {len(wins)} ratchet wins / {len(ratchet_results)} candidates -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
