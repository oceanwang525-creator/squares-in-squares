#!/usr/bin/env python3
"""Build predicted 2-pocket / 3-pocket seams, melt minor seam overlaps via C++ + SLP, and polish via lpslack + mechwalk."""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import collections
import itertools
import json
import math
import multiprocessing as mp_proc
import sys
import time
from decimal import Decimal
from pathlib import Path

import mpmath as mp
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
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import build_cert as bc  # noqa: E402
import homocasc2_lib as hlib  # noqa: E402
import lpslack  # noqa: E402
import union as auto_union  # noqa: E402
import utils as hu  # noqa: E402
from cpp_wrap import cpp_eval_sat_pen_bbox, cpp_expand_sym, full_to_free_sym, run_cpp_masked_melt  # noqa: E402
from joint_slp import joint_xytheta_slp  # noqa: E402
from squares.core import certify, kaggle_roundtrip  # noqa: E402

SOFT_DIR = REPO / "poc" / "swarm_softseamdp"
DATA_DIR = SOFT_DIR / "data"
DOCS_DIR = SOFT_DIR / "docs"
CKPT_DIR = DOCS_DIR / "ckpt"

_G_BY_KEY = None
_G_ROWS_ALL = None


def _run_cell_fork(args_tuple) -> dict:
    cell, L_inc_str, run_mechwalk, max_seams, keep_self = args_tuple
    return process_cell((cell, _G_BY_KEY, _G_ROWS_ALL, L_inc_str, run_mechwalk, max_seams, keep_self))


def load_pockets_index(paths: list[str]) -> dict[tuple[int, int, int, int], list[dict]]:
    by_key: dict[tuple[int, int, int, int], list[dict]] = collections.defaultdict(list)
    for path in paths:
        p = Path(path)
        if not p.exists():
            continue
        for g in json.loads(p.read_text()):
            by_key[(int(g["src"]), int(g["w"]), int(g["h"]), int(g["g"]))].append(g)
    return by_key


def resolve_pockets(comp: list[dict], by_key: dict, rows_all: dict) -> list[dict] | None:
    pockets = []
    for c in comp:
        src, cw, ch, cg = int(c["src"]), int(c["w"]), int(c["h"]), int(c["g"])
        cands = by_key.get((src, cw, ch, cg), [])
        if not cands:
            cands = by_key.get((src, ch, cw, cg), [])
        if not cands or src not in rows_all:
            return None
        pk = min(cands, key=lambda g: float(g["f"]))
        tr = (cw, ch) != (int(pk["w"]), int(pk["h"]))
        sq, W, H = bc.pocket_setup(rows_all[src], pk, tr)
        pockets.append(
            dict(
                src=src,
                w=cw,
                h=ch,
                g=cg,
                f=float(c["f"]),
                box=pk,
                boxh=(W, H),
                squares=sq,
                tr=tr,
            )
        )
    return pockets


def build_layout_arrays(
    K: int, n: int, pockets: list[dict], order: tuple[int, ...], X: list[int], Y: list[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int, int, list[dict] | None]:
    """Assemble one (order, X, Y) layout into float arrays (adding least-overlapping grid B cells if needed) and exact rows."""
    p_ord = [pockets[i] for i in order]
    parts = [(p["w"], p["h"], float(p["boxh"][0]) - p["w"], float(p["boxh"][1]) - p["h"]) for p in p_ord]
    with mp.workdps(50):
        fmax = max(max(p["boxh"][0] - p["w"], p["boxh"][1] - p["h"]) for p in p_ord)
        S = mp.mpf(K) + fmax
        cells = bc.grid_cells(K, S, parts, X, Y)

        out = []
        for p, x0, y0 in zip(p_ord, X, Y):
            dx, dy = mp.mpf(x0) + bc.EPS, mp.mpf(y0) + bc.EPS
            for s in p["squares"]:
                out.append(dict(x=s["x"] + dx, y=s["y"] + dy, angle=s["angle"]))
        n_pocket = len(out)
        for cx, cy in cells:
            out.append(dict(x=cx, y=cy, angle="0"))

    if len(out) > n:
        grid_idx = sorted(range(n_pocket, len(out)), key=lambda i: -(float(out[i]["x"]) + float(out[i]["y"])))
        drop = set(grid_idx[: len(out) - n])
        out = [p for i, p in enumerate(out) if i not in drop]
    elif len(out) < n:
        need = n - len(out)
        boxes = [
            (float(X[i]), float(Y[i]), float(X[i]) + parts[i][0] + parts[i][2], float(Y[i]) + parts[i][1] + parts[i][3])
            for i in range(len(parts))
        ]
        S_f = float(S)
        existing = {(round(float(p["x"]), 6), round(float(p["y"]), 6)) for p in out[n_pocket:]}
        skipped = []
        for a in range(K):
            for b in range(K):
                x0 = S_f - 1.0 - a - a * 1e-11
                y0 = S_f - 1.0 - b - b * 1e-11
                if x0 < -1e-9 or y0 < -1e-9:
                    continue
                if not any(x0 >= X[t] - 1e-12 and y0 >= Y[t] - 1e-12 for t in range(len(parts))):
                    continue
                cx, cy = x0 + 0.5, y0 + 0.5
                if (round(cx, 6), round(cy, 6)) in existing:
                    continue
                ov = 0.0
                for bx0, by0, bx1, by1 in boxes:
                    ix = max(0.0, min(x0 + 1.0, bx1) - max(x0, bx0))
                    iy = max(0.0, min(y0 + 1.0, by1) - max(y0, by0))
                    ov += ix * iy
                skipped.append((ov, cx + cy, cx, cy))
        skipped.sort(key=lambda t: (t[0], t[1]))
        if len(skipped) < need:
            return np.empty(0), np.empty(0), np.empty(0), math.inf, n_pocket, len(out), None
        for _, _, cx, cy in skipped[:need]:
            out.append(dict(x=mp.mpf(cx), y=mp.mpf(cy), angle="0"))

    xs = np.array([float(s["x"]) for s in out], dtype=np.float64)
    ys = np.array([float(s["y"]) for s in out], dtype=np.float64)
    ths = np.array([float(s["angle"]) for s in out], dtype=np.float64)
    rows = bc.to_rows(out, n)
    return xs, ys, ths, float(S), n_pocket, len(cells) + n_pocket, rows


def melt_overlapping_seam(
    n: int,
    x: np.ndarray,
    y: np.ndarray,
    th: np.ndarray,
    n_pocket: int,
    L_inc_dec: Decimal,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float] | None:
    """Melt minor seam overlaps using uniform micro-expansion (if tiny) or seam-focused C++ masked melt + LBFGS + joint SLP."""
    x, y, th, L_cur = hlib.norm_pack(x, y, th)
    pen, bb = cpp_eval_sat_pen_bbox(x, y, th)
    if pen <= 1e-12:
        return x, y, th, bb

    if pen <= 1e-4:
        for mult in (3.0, 10.0, 30.0):
            sc = 1.0 + max(1e-7, mult * pen)
            xe, ye = x * sc, y * sc
            pe, be = cpp_eval_sat_pen_bbox(xe, ye, th)
            if pe <= 1e-12:
                return xe, ye, th, be

    ov_pairs = hlib.overlap_pairs(x, y, th, tol=1e-10)
    active = np.zeros(n, dtype=bool)
    for _, i, j in ov_pairs:
        active[i] = True
        active[j] = True
    if not np.any(active):
        active[: min(n, max(n_pocket, 32))] = True
    else:
        ov_idx = np.nonzero(active)[0]
        d2 = (x[:, None] - x[None, ov_idx]) ** 2 + (y[:, None] - y[None, ov_idx]) ** 2
        active = np.min(d2, axis=1) <= 3.5 ** 2
    if active.sum() < 16:
        active[: min(n, n_pocket)] = True

    if pen < 0.08:
        xr, yr, tr, pr = hu.relax_lbfgs(x, y, th, bb + max(0.002, 2.0 * pen), free_mask=active, maxiter=200, w_wall=4.0)
        if pr > 1e-7:
            xr, yr, tr, pr = hu.relax_lbfgs(xr, yr, tr, bb + max(0.003, 2.0 * pr), free_mask=None, maxiter=200, w_wall=4.0)
        if pr < 0.02:
            xs, ys, ts, Ls, ok = joint_xytheta_slp(xr, yr, tr, rounds=8, tr_th0=0.02, tr_xy0=0.08, margin=1e-8, tol=1e-8, patience=2)
            if ok:
                p_slp, b_slp = cpp_eval_sat_pen_bbox(xs, ys, ts)
                if p_slp <= 1e-11:
                    return xs, ys, ts, b_slp

    allq, allm, alls = [], [], []
    mm = active.astype(np.uint8)
    mask_vec = np.concatenate([mm, mm, mm])
    L_inc_f = float(L_inc_dec)
    for L_start, L_end in [
        (bb + max(0.006, 1.5 * pen), min(bb + 0.002, L_inc_f + 0.002)),
        (bb + max(0.015, 2.5 * pen), bb + 0.004),
        (bb + max(0.025, 3.0 * pen), L_inc_f),
    ]:
        off = 0.5 * (L_start - bb)
        xm = np.clip(x + off, hlib.hx_of(th) + 1e-4, L_start - hlib.hx_of(th) - 1e-4)
        ym = np.clip(y + off, hlib.hx_of(th) + 1e-4, L_start - hlib.hx_of(th) - 1e-4)
        sched = [L_start, 0.5 * (L_start + L_end), L_end, L_end - 0.001]
        allq.append(full_to_free_sym(xm, ym, th, L_start))
        allm.append(mask_vec)
        alls.append(sched)

    qb, qf, lf, eo, po, lb = run_cpp_masked_melt(
        n,
        1,
        allq,
        allm,
        alls,
        [8, 12, 16, 20],
        a_start=0.72,
        n_melt_stages=4,
        masked_iters=160,
        full_iters=200,
        n_threads=1,
        base_seed=int(rng.integers(10000, 999999)),
    )
    order = sorted(range(len(allq)), key=lambda i: (0 if lf[i] < 1e9 else 1, lf[i] if lf[i] < 1e9 else po[i]))
    for i in order:
        use_f = lf[i] < 1e9
        Lr = float(lf[i]) if use_f else alls[i][-1]
        xp, yp, tp = cpp_expand_sym(n, 1, Lr, qf[i] if use_f else qb[i])
        pr, bbr = cpp_eval_sat_pen_bbox(xp, yp, tp)
        if 1e-8 < pr < 0.15:
            xp, yp, tp, pr = hu.relax_lbfgs(xp, yp, tp, bbr + max(0.002, 2.0 * pr), free_mask=None, maxiter=200, w_wall=4.0)
        if pr < 0.02:
            xs, ys, ts, Ls, ok = joint_xytheta_slp(xp, yp, tp, rounds=8, tr_th0=0.02, tr_xy0=0.08, margin=1e-8, tol=1e-8, patience=2)
            if ok:
                ps, bbs = cpp_eval_sat_pen_bbox(xs, ys, ts)
                if ps <= 1e-11:
                    return xs, ys, ts, bbs
    return None


def _run_lpslack_fast(
    n: int,
    x: np.ndarray,
    y: np.ndarray,
    th: np.ndarray,
    L_inc_dec: Decimal,
    cycles: int = 1,
    outer_iters: int = 6,
    fixed_rounds: int = 6,
    slp_modes: tuple[str, ...] = ("dual_non90",),
    time_limit: float = 6.0,
) -> dict | None:
    xn, yn, tn, Ln = hlib.norm_pack(x, y, th)
    rows_in = [
        {"square_id": str(i), "x": repr(float(xn[i])), "y": repr(float(yn[i])), "angle": repr(float(tn[i]))}
        for i in range(n)
    ]
    ceiling = str(Decimal(repr(Ln + 0.5)))
    res = lpslack.optimize_group(
        n,
        rows_in,
        ceiling,
        outer_iters=outer_iters,
        fixed_rounds=fixed_rounds,
        slp_modes=slp_modes,
        tr_th_schedule=(2e-3, 5e-4, 1e-4, 5e-5),
        cycles=cycles,
        time_limit=time_limit,
    )
    if res.get("status") in ("WIN", "TINY_WIN") and "rows" in res:
        return hlib.certify_rows_against_frontier(n, res["rows"], L_inc_dec)
    return None


def screen_layout(
    n: int,
    x: np.ndarray,
    y: np.ndarray,
    th: np.ndarray,
    L_inc_dec: Decimal,
) -> tuple[dict | None, list[str]]:
    """Fast screen: raw cert + fast 1-mode lpslack (+ joint_xytheta_slp check)."""
    notes = []
    best_cert = hlib.certify_pack_against_frontier(n, x, y, th, L_inc_dec)
    if best_cert is not None:
        notes.append(f"init={best_cert['L_float']:.7f}({best_cert['dL']:+.2e})")

    c_lp = _run_lpslack_fast(
        n, x, y, th, L_inc_dec, cycles=1, outer_iters=6, fixed_rounds=6, slp_modes=("dual_non90",), time_limit=6.0
    )
    if c_lp is not None:
        notes.append(f"lpslack={c_lp['L_float']:.7f}({c_lp['dL']:+.2e})")
        if best_cert is None or Decimal(max(c_lp["L_lit"], c_lp["L_pd"])) < Decimal(max(best_cert["L_lit"], best_cert["L_pd"])):
            best_cert = c_lp

    # Quick joint_xytheta_slp (~1-2s); follow with lpslack only if slp improved or no feasible cert yet
    if best_cert is None or best_cert["dL"] > -1e-4:
        xs, ys, ts, Ls, ok = joint_xytheta_slp(x, y, th, rounds=8, tr_th0=0.02, tr_xy0=0.08, margin=1e-8, tol=1e-9, patience=2)
        if ok:
            c_slp = hlib.certify_pack_against_frontier(n, xs, ys, ts, L_inc_dec)
            improved = c_slp is not None and (
                best_cert is None or Decimal(max(c_slp["L_lit"], c_slp["L_pd"])) < Decimal(max(best_cert["L_lit"], best_cert["L_pd"])) - Decimal("1e-5")
            )
            if c_slp is not None and (best_cert is None or Decimal(max(c_slp["L_lit"], c_slp["L_pd"])) < Decimal(max(best_cert["L_lit"], best_cert["L_pd"]))):
                best_cert = c_slp
                notes.append(f"slp={c_slp['L_float']:.7f}({c_slp['dL']:+.2e})")
            if improved or best_cert is None:
                c_lp2 = _run_lpslack_fast(
                    n, xs, ys, ts, L_inc_dec, cycles=1, outer_iters=6, fixed_rounds=6, slp_modes=("dual_non90",), time_limit=6.0
                )
                if c_lp2 is not None:
                    notes.append(f"slp+lp={c_lp2['L_float']:.7f}({c_lp2['dL']:+.2e})")
                    if best_cert is None or Decimal(max(c_lp2["L_lit"], c_lp2["L_pd"])) < Decimal(max(best_cert["L_lit"], best_cert["L_pd"])):
                        best_cert = c_lp2

    return best_cert, notes


def _is_self_split(n: int, seam: dict) -> bool:
    return all(int(c["src"]) in (n, 4000 + n, 5000 + n) for c in seam["comp"])


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
        pockets = resolve_pockets(seam["comp"], by_key, rows_all)
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
                xs, ys, ths, S_f, n_pk, raw_tot, exact_rows = build_layout_arrays(K, n, pockets, order, X, Y)
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
                c_res, notes = screen_layout(n, xs, ys, ths, L_inc_dec)
                notes_all.extend(notes)
            else:
                melted = melt_overlapping_seam(n, xs, ys, ths, n_pk, L_inc_dec, rng)
                if melted is None:
                    notes_all.append(f"L{l_idx}:melt_fail(raw={raw_tot})")
                    print(f"  [n={n} s{s_idx}/L{l_idx}] melt_fail (raw_tot={raw_tot}/{n})", flush=True)
                    continue
                tried_feas = True
                mx, my, mth, _ = melted
                c_res, notes = screen_layout(n, mx, my, mth, L_inc_dec)
                notes_all.extend([f"L{l_idx}:melted"] + notes)

            if c_res is not None:
                print(f"  [n={n} s{s_idx}/L{l_idx}] dL={c_res['dL']:+.6f} ({' '.join(notes[-2:])})", flush=True)
                if best_seam_dL is None or c_res["dL"] < best_seam_dL:
                    best_seam_dL = c_res["dL"]
                if best_cert is None or Decimal(max(c_res["L_lit"], c_res["L_pd"])) < Decimal(max(best_cert["L_lit"], best_cert["L_pd"])):
                    best_cert = c_res
                    best_seam = seam
                    if c_res["is_real_win"]:
                        _save_ckpt(n, K, D, L_inc_str, best_cert, best_seam)
            # If we already got a solid win (<= -5e-4) on layout 0, no need to burn time on layout 1
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

    # Phase 2: Deep polish (lpslack cycles=2 dual_non90+non90, plus mechwalk if enabled) on near-wins/wins
    if best_cert is not None and best_cert["dL"] < 1.5e-3:
        bx, by, bt, _ = hlib._rows_to_arrays(best_cert["rows"])
        c_deep = _run_lpslack_fast(
            n, bx, by, bt, L_inc_dec, cycles=2, outer_iters=10, fixed_rounds=8, slp_modes=("dual_non90", "non90"), time_limit=8.0
        )
        if c_deep is not None and Decimal(max(c_deep["L_lit"], c_deep["L_pd"])) < Decimal(max(best_cert["L_lit"], best_cert["L_pd"])):
            best_cert = c_deep
            bx, by, bt, _ = hlib._rows_to_arrays(best_cert["rows"])
            print(f"  [n={n} deep_lpslack] dL={best_cert['dL']:+.6f}", flush=True)
        if run_mechwalk and best_cert["dL"] < 1.0e-3:
            mw_cert, _ = hlib.mechwalk_compress(n, bx, by, bt, L_inc_dec, rng, rounds=2, threads=1)
            if mw_cert is not None:
                print(f"  [n={n} mechwalk] dL={mw_cert['dL']:+.6f}", flush=True)
                if Decimal(max(mw_cert["L_lit"], mw_cert["L_pd"])) < Decimal(max(best_cert["L_lit"], best_cert["L_pd"])):
                    best_cert = mw_cert
        if best_cert["is_real_win"]:
            _save_ckpt(n, K, D, L_inc_str, best_cert, best_seam)

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


def _save_ckpt(n: int, K: int, D: int, L_inc_str: str, cert: dict, seam: dict | None) -> None:
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
        "L_inc": L_inc_str,
        "L_lit": cert["L_lit"],
        "L_pd": cert["L_pd"],
        "dL": cert["dL"],
        "seam": seam,
        "rows": cert["rows"],
    }
    ckpt_path.write_text(json.dumps(ckpt, indent=1))


def export_final_csv() -> Path:
    groups = {}
    if CKPT_DIR.exists():
        for p in sorted(CKPT_DIR.glob("n_*.json")):
            d = json.loads(p.read_text())
            groups[int(d["n"])] = d["rows"]
    text = auto_union.final_text(groups)
    out_csv = SOFT_DIR / "final.csv"
    auto_union.atomic(out_csv, text.encode())
    os.chmod(out_csv, 0o644)
    if groups:
        lit, pd_t = auto_union.load(out_csv)
        for n in groups:
            if auto_union.check((n, lit[n], pd_t[n])) is None:
                raise ValueError(f"Exported final.csv failed dual certification for n={n}")
    return out_csv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--pockets", nargs="+", required=True)
    ap.add_argument("--limit", type=int, default=8, help="Max top cells to process (0 = all)")
    ap.add_argument("--n-list", help="Comma-separated specific ns to run")
    ap.add_argument("--max-seams", type=int, default=3)
    ap.add_argument("--max-dL-rigid", type=float, default=None)
    ap.add_argument("--keep-self", action="store_true", help="Also try pure self-split seams")
    ap.add_argument("--resume", action="store_true", help="Skip cells already in --out or docs/ckpt")
    ap.add_argument("--no-mechwalk", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pred = json.loads(Path(args.pred).read_text())["cells"]
    rep = {int(r["n"]): r["L"] for r in json.loads(Path(args.report).read_text())["results"]}
    by_key = load_pockets_index(args.pockets)
    rows_all = bc.load_rows(args.csv)

    if args.n_list:
        want = {int(x) for x in args.n_list.split(",") if x.strip()}
        cells = [c for c in pred if c["n"] in want]
    else:
        cells = pred
        if args.max_dL_rigid is not None:
            cells = [c for c in cells if c["best_dL_rigid"] <= args.max_dL_rigid]
        if not args.keep_self:
            cells = [c for c in cells if any(not _is_self_split(c["n"], s) for s in c["seams"])]
        if args.limit > 0:
            cells = cells[: args.limit]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results_by_n: dict[int, dict] = {}
    if args.resume:
        if out_path.exists():
            for r in json.loads(out_path.read_text()):
                results_by_n[int(r["n"])] = r
        done_ns = set(results_by_n.keys())
        if CKPT_DIR.exists():
            for p in CKPT_DIR.glob("n_*.json"):
                done_ns.add(int(p.stem.split("_")[1]))
        cells = [c for c in cells if int(c["n"]) not in done_ns]

    global _G_BY_KEY, _G_ROWS_ALL
    _G_BY_KEY = by_key
    _G_ROWS_ALL = rows_all

    print(
        f"Processing {len(cells)} cells (workers={min(2, args.workers)}, max_seams={args.max_seams}, mechwalk={not args.no_mechwalk}, keep_self={args.keep_self})",
        flush=True,
    )

    tasks = [(c, rep[c["n"]], not args.no_mechwalk, args.max_seams, args.keep_self) for c in cells]

    def _record(r: dict) -> None:
        results_by_n[int(r["n"])] = r
        print(f"n={r['n']} (K={r['K']},D={r['D']}) dL_rigid={r['best_dL_rigid']:+.6f} -> win={r['win']} best_dL={r.get('dL')} ({r['elapsed_s']}s)", flush=True)
        out_path.write_text(json.dumps(list(results_by_n.values()), indent=1))

    if min(2, args.workers) <= 1 or len(tasks) == 1:
        for t in tasks:
            _record(_run_cell_fork(t))
    elif tasks:
        with mp_proc.get_context("fork").Pool(2) as pool:
            for r in pool.imap_unordered(_run_cell_fork, tasks, chunksize=1):
                _record(r)

    results = list(results_by_n.values())
    out_path.write_text(json.dumps(results, indent=1))
    out_csv = export_final_csv()
    wins = [r for r in results if r["win"]]
    print(f"Done: {len(wins)}/{len(results)} certified wins; exported {out_csv}", flush=True)


if __name__ == "__main__":
    main()
