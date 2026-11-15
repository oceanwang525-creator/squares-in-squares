#!/usr/bin/env python3
"""Round 3 (H116 + H121): zero_gap + kx=1,ky=1 overhang2d + wx8 + watch89 + pslp squeeze."""
from __future__ import annotations

import argparse
import glob
import json
import math
import multiprocessing as mp
import os
import sys
import time
from decimal import Decimal
from pathlib import Path

for _env in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_env] = "1"

import numpy as np

POC_DIR = Path(__file__).resolve().parent
SWARM_DIR = POC_DIR.parent
REPO = SWARM_DIR.parents[1]
sys.path.insert(0, str(POC_DIR))
sys.path.insert(0, str(REPO / "poc" / "swarm_highd" / "poc"))
sys.path.insert(0, str(REPO / "poc" / "swarm_lpslack" / "poc"))
sys.path.insert(0, str(REPO / "poc" / "swarm_mechwalk" / "poc"))
sys.path.insert(0, str(REPO / "poc" / "row_ratchet_980" / "poc"))
sys.path.insert(0, str(REPO / "poc" / "auto_union" / "poc"))

import interlock2_lib as il
import interlock2_eval as ie
import lpslack
import pslp
import union as auto_union
import utils as hu
from cpp_wrap import cpp_eval_sat_pen_bbox, cpp_expand_sym, full_to_free_sym, run_cpp_masked_melt

PARTB_SANE = Path(
    "/home/lucas/.gemini/antigravity-cli/worktrees/a4fb0b93-b63b-426d-aa12-67454e99bc08"
    "/swarm-liftpolish3-18693d3f/poc/swarm_liftpolish3/data/pockets_partb_sane.json"
)


def load_round3_pockets(
    best_rows: dict[int, list[dict[str, str]]],
    merged_rows: dict[int, list[dict[str, str]]],
) -> dict[tuple[int, int, int], list[dict]]:
    """Load e3 + scanned_5rows + scanned_watch89 + bricks_wx8 + swarm bricks + partb."""
    by_whg = il.load_all_pockets(best_rows, extra_scan_path=il.DATA_DIR / "scanned_5rows.json")
    expanded: dict[tuple[int, int, int], list[dict]] = {k: list(v) for k, v in by_whg.items()}

    def _add_brick(b: dict, name: str) -> None:
        w, h, g = int(b["w"]), int(b["h"]), int(b["g"])
        if g < 1 or w < 1 or h < 1:
            return
        x = np.asarray(b["x"], dtype=np.float64)
        y = np.asarray(b["y"], dtype=np.float64)
        th = np.asarray(b["th"], dtype=np.float64)
        if len(x) != w * h + g:
            return
        hx = 0.5 * (np.abs(np.cos(th)) + np.abs(np.sin(th)))
        W = float((x + hx).max() - (x - hx).min())
        H = float((y + hx).max() - (y - hx).min())
        fx, fy = W - w, H - h
        if max(fx, fy) >= 0.999:
            return
        pen, _ = cpp_eval_sat_pen_bbox(x - float((x - hx).min()), y - float((y - hx).min()), th)
        if pen > 1e-6:
            return
        for tr in ((False, True) if w != h else (False,)):
            rw, rh = (h, w) if tr else (w, h)
            rfx, rfy = (fy, fx) if tr else (fx, fy)
            expanded.setdefault((rw, rh, g), []).append(
                {
                    "uid": f"{name}_{rw}x{rh}_g{g}",
                    "kind": "brick",
                    "brick": b,
                    "transpose": tr,
                    "w": rw,
                    "h": rh,
                    "g": g,
                    "fx": rfx,
                    "fy": rfy,
                    "f": max(rfx, rfy),
                }
            )

    wx8_path = il.DATA_DIR / "bricks_wx8.json"
    if wx8_path.exists():
        for idx, b in enumerate(json.loads(wx8_path.read_text())):
            _add_brick(b, f"wx8_{idx}_{b.get('name', '')[:28]}")

    for pat in (
        "poc/swarm_b8xplant/data/brick_*.json",
        "poc/swarm_cleanpeel2/data/brick_*.json",
        "poc/swarm_stepdown/data/brick_*.json",
        "poc/swarm_band3/data/brick_*.json",
    ):
        for jp in sorted(glob.glob(str(REPO / pat))):
            _add_brick(json.loads(Path(jp).read_text()), Path(jp).stem)

    def _add_pockets_file(path: Path, rows_map: dict[int, list[dict[str, str]]], rows_tag: str, prefix: str) -> None:
        if not path.exists():
            return
        prep_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        for pk in json.loads(path.read_text()):
            src = int(pk["src"])
            if src not in rows_map:
                continue
            if src not in prep_cache:
                prep_cache[src] = il._prep_float(rows_map[src])
            lx, hx, ly, hy = prep_cache[src]
            m = (
                (lx >= pk["x0"] - 1e-8)
                & (hx <= pk["x1"] + 1e-8)
                & (ly >= pk["y0"] - 1e-8)
                & (hy <= pk["y1"] + 1e-8)
            )
            if int(m.sum()) != int(pk["cnt"]):
                continue
            w, h, g, f = int(pk["w"]), int(pk["h"]), int(pk["g"]), float(pk["f"])
            if f >= 0.999 or g < 1 or w < 1 or h < 1:
                continue
            fx = float(hx[m].max() - lx[m].min()) - w
            fy = float(hy[m].max() - ly[m].min()) - h
            for tr in ((False, True) if w != h else (False,)):
                rw, rh = (h, w) if tr else (w, h)
                rfx, rfy = (fy, fx) if tr else (fx, fy)
                expanded.setdefault((rw, rh, g), []).append(
                    {
                        "uid": f"{prefix}_n{src}_{rw}x{rh}_g{g}",
                        "kind": "donor_pocket",
                        "src": src,
                        "rows_map": rows_tag,
                        "pk": pk,
                        "transpose": tr,
                        "w": rw,
                        "h": rh,
                        "g": g,
                        "fx": rfx,
                        "fy": rfy,
                        "f": max(rfx, rfy),
                    }
                )

    _add_pockets_file(il.DATA_DIR / "scanned_watch89.json", best_rows, "best", "w89")
    _add_pockets_file(PARTB_SANE, merged_rows, "merged", "pb")

    pruned: dict[tuple[int, int, int], list[dict]] = {}
    for k, lst in expanded.items():
        lst.sort(key=lambda r: r["f"])
        kept, seen = [], set()
        for it in lst:
            if it["uid"] in seen:
                continue
            seen.add(it["uid"])
            kept.append(it)
            if len(kept) >= 2:
                break
        pruned[k] = kept
    return pruned


def find_round3_candidates(
    by_whg: dict[tuple[int, int, int], list[dict]],
    inc_L: dict[int, float],
    zg_margin: float = 0.007,
    o2_margin: float = 0.002,
) -> tuple[list[dict], list[dict]]:
    """Find zero_gap (|I|=0) and balanced overhang2d (kx=1, ky=1) candidates."""
    best_items = [lst[0] for lst in by_whg.values()]
    zg_hits: dict[int, list[dict]] = {}
    o2_hits: dict[int, list[dict]] = {}

    for K in range(16, 32):
        cand = [it for it in best_items if it["w"] < K and it["h"] < K]
        for i in range(len(cand)):
            p1 = cand[i]
            for j in range(i, len(cand)):
                p2 = cand[j]
                kx = max(0, p1["w"] + p2["w"] - K)
                ky = max(0, p1["h"] + p2["h"] - K)
                L_pred = K + max(p1["f"], p2["f"])

                # 1) zero_gap: kx+ky in {1,2}, kx*ky == 0
                if (kx + ky) in (1, 2) and kx * ky == 0:
                    ok = (kx > 0 and p1["h"] + p2["h"] <= K - 1) or (ky > 0 and p1["w"] + p2["w"] <= K - 1)
                    if ok:
                        D = p1["g"] + p2["g"] - (kx + ky)
                        n = K * K + D
                        if D >= 1 and n <= 1000 and n in inc_L:
                            dL = L_pred - inc_L[n]
                            if dL <= zg_margin:
                                zg_hits.setdefault(n, []).append({
                                    "n": n, "K": K, "D": D, "kx": kx, "ky": ky,
                                    "regime": "zero_gap", "p1": p1, "p2": p2,
                                    "L_pred": L_pred, "L_inc_est": inc_L[n], "dL_pred": dL,
                                })

                # 2) overhang2d (H116): kx == 1 and ky == 1
                if kx == 1 and ky == 1:
                    for del_cnt in (1, 2, 3):
                        D = p1["g"] + p2["g"] + 1 - del_cnt
                        n = K * K + D
                        if D >= 1 and n <= 1000 and n in inc_L:
                            dL = L_pred - inc_L[n]
                            if dL <= o2_margin:
                                o2_hits.setdefault(n, []).append({
                                    "n": n, "K": K, "D": D, "kx": 1, "ky": 1,
                                    "del_cnt": del_cnt,
                                    "regime": "overhang2d", "p1": p1, "p2": p2,
                                    "L_pred": L_pred, "L_inc_est": inc_L[n], "dL_pred": dL,
                                })

    def _dedup(hits_dict: dict[int, list[dict]], per_n: int) -> list[dict]:
        out = []
        for n, lst in hits_dict.items():
            lst.sort(key=lambda r: r["dL_pred"])
            seen = set()
            kept_n = 0
            for r in lst:
                sa = tuple(sorted([(r["p1"]["w"], r["p1"]["h"], r["p1"]["g"]), (r["p2"]["w"], r["p2"]["h"], r["p2"]["g"])]))
                sb = tuple(sorted([(r["p1"]["h"], r["p1"]["w"], r["p1"]["g"]), (r["p2"]["h"], r["p2"]["w"], r["p2"]["g"])]))
                sig = (min(sa, sb), r.get("del_cnt", 0))
                if sig in seen:
                    continue
                seen.add(sig)
                out.append(r)
                kept_n += 1
                if kept_n >= per_n:
                    break
        out.sort(key=lambda r: r["dL_pred"])
        return out

    return _dedup(zg_hits, 2), _dedup(o2_hits, 2)


def pslp_polish_rows(n: int, rows: list[dict[str, str]], L_inc_dec: Decimal) -> dict | None:
    """Apply full-row pslp.squeeze + lpslack 50-digit certification."""
    x = np.array([float(r["x"]) for r in rows], dtype=np.float64)
    y = np.array([float(r["y"]) for r in rows], dtype=np.float64)
    th = np.array([float(r["angle"]) for r in rows], dtype=np.float64)
    x0, x1, y0, y1 = pslp.box_of(x, y, th)
    x -= x0
    y -= y0
    L0 = max(x1 - x0, y1 - y0)
    f1, xs, ys, ts, v1, _ = pslp.squeeze(x, y, th, 0, 0, L0, trust0=0.015, trust_max=0.04, max_it=350)
    if v1 > 1e-10:
        return None
    x0, x1, y0, y1 = pslp.box_of(xs, ys, ts)
    if max(x1 - x0, y1 - y0) >= float(L_inc_dec) + 2e-5:
        return None
    xs -= x0
    ys -= y0
    rows_s = lpslack.rows_from_arrays(np.column_stack([xs, ys]), ts)
    cert = il.polish_rows_lpslack_mechwalk(n, rows_s, L_inc_dec)
    return cert if cert.get("ok") else None


def assemble_overhang2d_float(
    K: int, n: int, del_cnt: int,
    sq1: list[dict], W1, H1, w1: int, h1: int,
    sq2: list[dict], W2, H2, w2: int, h2: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, tuple[float, float]] | None:
    """Assemble BL P1 + TR P2 with kx=1, ky=1, deleting del_cnt squares nearest the 1x1 intersection."""
    fmax = max(float(W1) - w1, float(H1) - h1, float(W2) - w2, float(H2) - h2)
    S = float(K) + fmax + 0.002
    xs, ys, ts = [], [], []
    for s in sq1:
        xs.append(float(s["x"]) + 1e-4)
        ys.append(float(s["y"]) + 1e-4)
        ts.append(float(s["angle"]))
    dx2, dy2 = S - float(W2) - 1e-4, S - float(H2) - 1e-4
    for s in sq2:
        xs.append(float(s["x"]) + dx2)
        ys.append(float(s["y"]) + dy2)
        ts.append(float(s["angle"]))

    # Fill upright BR (K - w1) x (K - h2) and TL (K - w2) x (K - h1)
    # BR starts at x >= W1, y <= K - h2
    ux0 = S - (K - w1) + 0.5
    uy0 = 0.5
    for ix in range(K - w1):
        for iy in range(K - h2):
            xs.append(ux0 + ix)
            ys.append(uy0 + iy)
            ts.append(0.0)
    ux1 = 0.5
    uy1 = S - (K - h1) + 0.5
    for ix in range(K - w2):
        for iy in range(K - h1):
            xs.append(ux1 + ix)
            ys.append(uy1 + iy)
            ts.append(0.0)

    ax = np.array(xs, dtype=np.float64)
    ay = np.array(ys, dtype=np.float64)
    at = np.array(ts, dtype=np.float64)

    # Overlap center of I = [S - W2, W1] x [S - H2, H1]
    cx = 0.5 * ((S - float(W2)) + float(W1))
    cy = 0.5 * ((S - float(H2)) + float(H1))
    n_p12 = len(sq1) + len(sq2)
    d2 = (ax[:n_p12] - cx) ** 2 + (ay[:n_p12] - cy) ** 2
    drop_idx = set(np.argsort(d2)[:del_cnt].tolist())
    keep = [i for i in range(len(ax)) if i not in drop_idx]
    if len(keep) != n:
        return None
    ax, ay, at = ax[keep], ay[keep], at[keep]
    x0, x1, y0, y1 = pslp.box_of(ax, ay, at)
    ax -= x0
    ay -= y0
    return ax, ay, at, float(max(x1 - x0, y1 - y0)), (cx - x0, cy - y0)


def eval_overhang2d_candidate(
    cand: dict,
    merged_rows: dict[int, list[dict[str, str]]],
    best_rows: dict[int, list[dict[str, str]]],
    L_inc_dec: Decimal,
    n_threads: int = 1,
) -> dict:
    t0 = time.time()
    n, K = cand["n"], cand["K"]
    p1, p2 = cand["p1"], cand["p2"]
    ext1 = il.get_item_squares_50d(p1, merged_rows, best_rows)
    ext2 = il.get_item_squares_50d(p2, merged_rows, best_rows)
    if ext1 is None or ext2 is None:
        return {"n": n, "status": "EXTRACT_FAIL", "dt": round(time.time() - t0, 1)}
    sq1, W1, H1 = ext1
    sq2, W2, H2 = ext2

    cands = []
    for s1 in range(4):
        q1 = il.sym_squares_50d(sq1, W1, H1, s1)
        for s2 in range(4):
            q2 = il.sym_squares_50d(sq2, W2, H2, s2)
            asm = assemble_overhang2d_float(
                K, n, cand["del_cnt"], q1, W1, H1, p1["w"], p1["h"], q2, W2, H2, p2["w"], p2["h"]
            )
            if asm is None:
                continue
            ax, ay, at, aL, center = asm
            pen0, _ = cpp_eval_sat_pen_bbox(ax, ay, at)
            cands.append((pen0, s1, s2, ax, ay, at, aL, center))
    if not cands:
        return {"n": n, "status": "ASM_FAIL", "dt": round(time.time() - t0, 1)}
    cands.sort(key=lambda r: r[0])

    L_inc_f = float(L_inc_dec)
    rng = np.random.default_rng(42000 + n)
    best_res, best_dec, best_label = None, None, ""

    for pen0, s1, s2, ax, ay, at, aL, (cx, cy) in cands[:2]:
        L0 = max(aL, L_inc_f) + 0.045
        sc = L0 / aL
        x0, y0, t0_arr = ax * sc, ay * sc, at.copy()
        scx, scy = cx * sc, cy * sc

        q_list, m_list, ls_list = [], [], []
        scheds = [
            [L0, 0.5 * (L0 + L_inc_f), L_inc_f + 0.005, L_inc_f - 0.001],
            [L0, L_inc_f + 0.015, L_inc_f + 0.003, L_inc_f - 0.003],
        ]
        for idx in range(6):
            rad = float(rng.uniform(4.8, 7.5))
            in_m = ((x0 - scx) ** 2 + (y0 - scy) ** 2) <= rad * rad
            in_core = ((x0 - scx) ** 2 + (y0 - scy) ** 2) <= 2.2 * 2.2
            m_sq = in_m.astype(np.uint8)
            mask = np.concatenate([m_sq, m_sq, m_sq])
            x, y, th = x0.copy(), y0.copy(), t0_arr.copy()
            x[in_core] += rng.uniform(-0.08, 0.08, size=int(np.sum(in_core)))
            y[in_core] += rng.uniform(-0.08, 0.08, size=int(np.sum(in_core)))
            th[in_core] += rng.uniform(-0.20, 0.20, size=int(np.sum(in_core)))
            q_list.append(full_to_free_sym(x, y, th, L0))
            m_list.append(mask)
            ls_list.append(scheds[idx % len(scheds)])

        qb, qf, lf, _, po, _ = run_cpp_masked_melt(
            n, 1, q_list, m_list, ls_list, [6, 10, 14, 18],
            a_start=0.82, n_melt_stages=4, masked_iters=220, full_iters=220,
            n_threads=n_threads, base_seed=int(rng.integers(1, 1_000_000)),
        )
        order = sorted(
            range(len(q_list)),
            key=lambda i: (0 if lf[i] < 1e9 else 1, float(lf[i]) if lf[i] < 1e9 else float(po[i])),
        )[:2]

        for m_i in order:
            use_f = lf[m_i] < 1e9
            L_ref = float(lf[m_i]) if use_f else ls_list[m_i][-1]
            cand_x, cand_y, cand_t = cpp_expand_sym(n, 1, L_ref, qf[m_i] if use_f else qb[m_i])
            pen, bb = cpp_eval_sat_pen_bbox(cand_x, cand_y, cand_t)
            if 1e-7 < pen < 0.15:
                cand_x, cand_y, cand_t, _ = hu.relax_lbfgs(
                    cand_x, cand_y, cand_t, bb + max(0.005, 2.0 * pen), maxiter=350, w_wall=4.0, w_pair=1.0
                )
                pen, bb = cpp_eval_sat_pen_bbox(cand_x, cand_y, cand_t)
            bx0, bx1, by0, by1 = pslp.box_of(cand_x, cand_y, cand_t)
            cand_x -= bx0
            cand_y -= by0
            bL = float(max(bx1 - bx0, by1 - by0))
            if pen > 1e-6:
                s_exp = 1.0 + min(0.03, max(0.003, 1.8 * pen))
                cand_x *= s_exp
                cand_y *= s_exp
                bL *= s_exp

            fn, xs, ys, ts, vn, _ = pslp.squeeze(cand_x, cand_y, cand_t, 0, 0, bL, trust0=0.02, trust_max=0.05, max_it=350)
            if vn > 1e-9:
                print(f"    [o2_infeas n={n} d{cand['del_cnt']} s{s1}{s2}_m{m_i}] pen={pen:.3e} vn={vn:.3e} fn-Linc={fn-L_inc_f:+.4f}", flush=True)
                continue
            bx0, bx1, by0, by1 = pslp.box_of(xs, ys, ts)
            sqL = float(max(bx1 - bx0, by1 - by0))
            if sqL >= L_inc_f + 2e-3:
                print(f"    [o2_gap n={n} d{cand['del_cnt']} s{s1}{s2}_m{m_i}] sqL-Linc={sqL-L_inc_f:+.4f}", flush=True)
                continue
            xs -= bx0
            ys -= by0
            rows_s = lpslack.rows_from_arrays(np.column_stack([xs, ys]), ts)
            cert = il.polish_rows_lpslack_mechwalk(n, rows_s, L_inc_dec)
            if not cert.get("ok"):
                continue
            lbl = f"o2_{p1['uid']}+{p2['uid']}_d{cand['del_cnt']}_s{s1}{s2}_m{m_i}"
            print(f"    [{lbl}] dL={cert['dL_lit']:+.6e} L={cert['L_lit'][:14]} ({time.time()-t0:.1f}s)", flush=True)
            cd = max(Decimal(cert["L_lit"]), Decimal(cert["L_pd"]))
            if best_dec is None or cd < best_dec:
                best_dec, best_res, best_label = cd, cert, lbl
            if cert["dL_lit"] < 2e-3:
                mw = ie.mechwalk_polish_once(n, cert["rows"], L_inc_dec, n_threads=n_threads)
                if mw is not None and mw.get("ok"):
                    mcd = max(Decimal(mw["L_lit"]), Decimal(mw["L_pd"]))
                    if mcd < best_dec:
                        best_dec, best_res, best_label = mcd, mw, lbl + "_mw"
                        print(f"    [mechwalk {best_label}] dL={mw['dL_lit']:+.6e}", flush=True)

    if best_res is None:
        return {"n": n, "regime": "overhang2d", "status": "FAIL", "dt": round(time.time() - t0, 1)}
    if best_res["is_real_win"]:
        ie.save_win(n, best_res, best_label)
    return {
        "n": n, "K": K, "D": cand["D"], "kx": 1, "ky": 1, "del_cnt": cand["del_cnt"],
        "regime": "overhang2d", "status": "WIN" if best_res["is_real_win"] else "OK",
        "dL": best_res["dL_lit"], "L_lit": best_res["L_lit"], "L_pd": best_res["L_pd"],
        "L_inc": best_res["L_inc"], "label": best_label, "dt": round(time.time() - t0, 1),
    }


def exact_incumbents_single_pass(target_ns: set[int]) -> dict[int, Decimal]:
    best_by_n: dict[int, tuple[float, list[dict[str, str]]]] = {}
    for p in sorted((REPO / "poc").glob("*/final.csv")):
        txt = p.read_text()
        if len(txt.splitlines()) <= 1:
            continue
        grps = auto_union.parse_text(txt)
        for n in target_ns:
            if n not in grps or len(grps[n]) != n:
                continue
            rows = grps[n]
            xs = [float(r["x"]) for r in rows]
            ys = [float(r["y"]) for r in rows]
            angs = [float(r["angle"]) for r in rows]
            hx = [0.5 * (abs(math.cos(t)) + abs(math.sin(t))) for t in angs]
            L_est = max(
                max(x + h for x, h in zip(xs, hx)) - min(x - h for x, h in zip(xs, hx)),
                max(y + h for y, h in zip(ys, hx)) - min(y - h for y, h in zip(ys, hx)),
            )
            if n not in best_by_n or L_est < best_by_n[n][0] - 1e-12:
                best_by_n[n] = (L_est, rows)
    return {n: Decimal(auto_union.certify(n, best_by_n[n][1])) for n in sorted(target_ns)}


_G: dict = {}


def _init_w(merged_rows, best_rows, exact_inc):
    _G["merged_rows"] = merged_rows
    _G["best_rows"] = best_rows
    _G["exact_inc"] = exact_inc


def _run_task(cand: dict) -> dict:
    n = cand["n"]
    L_inc = _G["exact_inc"][n]
    if cand["regime"] == "overhang2d":
        res = eval_overhang2d_candidate(cand, _G["merged_rows"], _G["best_rows"], L_inc, n_threads=1)
    else:
        res = ie.eval_zero_gap_candidate(
            cand, _G["merged_rows"], _G["best_rows"], L_inc,
            syms=((0, 0), (3, 0)), try_mechwalk=False, n_threads=1, return_rows=True,
        )
        cur_rows = res.pop("rows", None)
        # Apply mechwalk + pslp.squeeze polish on close/winning zero_gap candidates
        if cur_rows is not None and res.get("status") in ("WIN", "OK") and float(res.get("dL", 1.0)) < 1.2e-3:
            mw = ie.mechwalk_polish_once(n, cur_rows, L_inc, n_threads=1)
            if mw is not None and mw.get("ok") and mw["dL_lit"] < res["dL"]:
                cur_rows = mw["rows"]
                res["dL"] = mw["dL_lit"]
                res["L_lit"] = mw["L_lit"]
                res["L_pd"] = mw["L_pd"]
                res["label"] += "_mw"
                if mw["is_real_win"]:
                    res["status"] = "WIN"
                    ie.save_win(n, mw, res["label"])
            ps = pslp_polish_rows(n, cur_rows, L_inc)
            if ps is not None and ps["is_real_win"] and (
                res.get("status") != "WIN" or ps["dL_lit"] < res["dL"]
            ):
                res["status"] = "WIN"
                res["dL"] = ps["dL_lit"]
                res["L_lit"] = ps["L_lit"]
                res["L_pd"] = ps["L_pd"]
                res["label"] += "_pslp"
                ie.save_win(n, ps, res["label"])
    res.setdefault("regime", cand["regime"])
    res["pair"] = f"{cand['p1']['uid']}+{cand['p2']['uid']}"
    res["dL_pred"] = cand["dL_pred"]
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--zg-margin", type=float, default=0.006)
    ap.add_argument("--o2-margin", type=float, default=0.004)
    ap.add_argument("--max-zg", type=int, default=18)
    ap.add_argument("--max-o2", type=int, default=14)
    args = ap.parse_args()

    t0 = time.time()
    print("=== H101 Round 3 (H116 kx=1,ky=1 + H121 wx8 + watch89 + pslp) ===", flush=True)
    best_rows, inc_L = il.load_portfolio_best()
    merged_rows = il.load_merged_rows()
    by_whg = load_round3_pockets(best_rows, merged_rows)
    print(f"Loaded {len(by_whg)} unique (w,h,g) keys ({sum(len(v) for v in by_whg.values())} items)", flush=True)

    zg_cands, o2_cands = find_round3_candidates(by_whg, inc_L, zg_margin=args.zg_margin, o2_margin=args.o2_margin)
    # Prioritize new sources (wx8_ or w89_) and untested ns for zero_gap
    already_won = {748, 863, 910, 924}
    zg_todo = []
    for c in zg_cands:
        has_new = c["p1"]["uid"].startswith(("wx8_", "w89_")) or c["p2"]["uid"].startswith(("wx8_", "w89_"))
        if c["n"] in already_won and not has_new:
            continue
        if abs(c["dL_pred"]) < 1e-8 and c["n"] in (591, 857) and not has_new:
            continue
        zg_todo.append(c)
    # Sort new-source or negative-dL_pred first
    zg_todo.sort(key=lambda c: (0 if (c["p1"]["uid"].startswith(("wx8_", "w89_")) or c["p2"]["uid"].startswith(("wx8_", "w89_"))) else 1, c["dL_pred"]))
    zg_todo = zg_todo[: args.max_zg]
    o2_d2 = [c for c in o2_cands if c["del_cnt"] == 2][:8]
    o2_d1 = [c for c in o2_cands if c["del_cnt"] == 1][:4]
    o2_d3 = [c for c in o2_cands if c["del_cnt"] == 3][:4]
    o2_todo = (o2_d2 + o2_d1 + o2_d3)[: args.max_o2]

    todo = o2_todo + zg_todo
    print(f"Selected {len(zg_todo)} zero_gap + {len(o2_todo)} overhang2d = {len(todo)} targets:", flush=True)
    for c in todo:
        print(
            f"  [{c['regime']:10s}] n={c['n']:4d} K={c['K']} D={c['D']:2d} k=({c['kx']},{c['ky']}) "
            f"dL_pred={c['dL_pred']:+.6f} | {c['p1']['uid']} + {c['p2']['uid']}",
            flush=True,
        )
    if args.dry_run:
        return

    res_path = il.DOCS_DIR / "round3_results.json"
    results = []
    done_keys = set()
    if args.resume and res_path.exists():
        results = json.loads(res_path.read_text()).get("results", [])
        done_keys = {(r.get("regime"), r.get("n"), r.get("pair")) for r in results}
        todo = [
            c for c in todo
            if (c["regime"], c["n"], f"{c['p1']['uid']}+{c['p2']['uid']}") not in done_keys
        ]
        print(f"Resuming: {len(done_keys)} done, {len(todo)} remaining", flush=True)

    target_ns = {c["n"] for c in todo}
    exact_inc = exact_incumbents_single_pass(target_ns) if target_ns else {}
    print(f"Certified exact incumbents for {len(exact_inc)} ns in {time.time()-t0:.1f}s", flush=True)

    if todo:
        with mp.get_context("fork").Pool(processes=2, initializer=_init_w, initargs=(merged_rows, best_rows, exact_inc)) as pool:
            for res in pool.imap_unordered(_run_task, todo):
                results.append(res)
                print(
                    f"  [R3 DONE] [{res.get('regime')}] n={res['n']} status={res['status']} "
                    f"dL={res.get('dL')} pair={res.get('pair')} dt={res.get('dt')}s",
                    flush=True,
                )
                res_path.write_text(
                    json.dumps(
                        {"elapsed_s": round(time.time() - t0, 1), "results": results},
                        indent=2,
                    )
                )

    results.sort(key=lambda r: float(r["dL"]) if r.get("dL") is not None else 999.0)
    res_path.write_text(
        json.dumps({"elapsed_s": round(time.time() - t0, 1), "results": results}, indent=2)
    )
    wins = [r for r in results if r.get("status") == "WIN"]
    print(f"\n=== ROUND 3 SUMMARY: {len(wins)}/{len(results)} WINS in {time.time()-t0:.1f}s ===", flush=True)
    for w in wins:
        print(f"  WIN n={w['n']} [{w['regime']}]: dL={w['dL']:+.6e} L={w['L_lit']} ({w['pair']})", flush=True)


if __name__ == "__main__":
    main()
