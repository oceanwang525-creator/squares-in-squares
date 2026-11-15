#!/usr/bin/env python3
"""1-Axis Overhang Pinwheel DP (kx + ky = 1, |I| = 0, g1 + g2 - 1 = D) + Build & Polish for F3B_high_r."""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import heapq
import json
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
    str(REPO / "poc" / "swarm_lpslack" / "poc"),
    str(REPO / "poc" / "swarm_mechwalk" / "poc"),
    str(REPO / "poc" / "swarm_homocasc2" / "poc"),
    str(REPO / "poc" / "swarm_interlock2" / "poc"),
    str(REPO / "poc" / "row_ratchet_980" / "poc"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import build_cert as bc  # noqa: E402
import homocasc2_lib as hlib  # noqa: E402
import interlock2_eval as ie  # noqa: E402
import interlock2_lib as il  # noqa: E402
import lpslack  # noqa: E402
import pslp  # noqa: E402
import union as auto_union  # noqa: E402

HIGH_DIR = REPO / "poc" / "swarm_highdpocket"
CKPT_DIR = HIGH_DIR / "docs" / "ckpt"

_G_ROWS = None


def load_oriented_items(pocket_paths: list[str]) -> list[dict]:
    """Load oriented (w, h, g) items keeping best f per (w, h, g) from sanitized pocket JSONs."""
    best: dict[tuple[int, int, int], dict] = {}
    for p in pocket_paths:
        pp = Path(p)
        if not pp.exists():
            continue
        for g in json.loads(pp.read_text()):
            gw, gh, gg, gf, gsrc = int(g["w"]), int(g["h"]), int(g["g"]), float(g["f"]), int(g["src"])
            for w, h in ((gw, gh), (gh, gw)):
                k = (w, h, gg)
                tr = (w, h) != (gw, gh)
                if k not in best or gf < best[k]["f"] - 1e-12:
                    best[k] = {
                        "w": w,
                        "h": h,
                        "g": gg,
                        "f": gf,
                        "src": gsrc,
                        "tr": tr,
                        "pk": g,
                    }
    return list(best.values())


def search_overhang(
    items: list[dict],
    L_inc: dict[int, float],
    k_min: int = 18,
    k_max: int = 31,
    d_min: int = 15,
    d_max: int = 28,
    slack_max: float = 6.0e-3,
    top_k: int = 3,
    exact_g: bool = True,
) -> list[dict]:
    """Find top-k 1-axis overhang (kx+ky=1, |I|=0) pairs for every (K, D) cell."""
    cells: list[dict] = []
    for K in range(k_min, k_max + 1):
        d_list = [D for D in range(d_min, d_max + 1) if K * K + D in L_inc and K * K + D <= 1000]
        if not d_list:
            continue
        f_lim = {D: L_inc[K * K + D] - K + slack_max for D in d_list}
        f_max = max(f_lim.values())
        cand = [it for it in items if it["f"] <= f_max + 1e-12 and 1 <= it["w"] < K and 1 <= it["h"] < K]
        cand.sort(key=lambda r: (r["f"], -r["g"]))
        M = len(cand)
        if M == 0:
            continue

        ws = np.array([it["w"] for it in cand], dtype=np.int32)
        hs = np.array([it["h"] for it in cand], dtype=np.int32)
        gs = np.array([it["g"] for it in cand], dtype=np.int32)
        fs = np.array([it["f"] for it in cand], dtype=np.float64)

        heaps: dict[int, list[tuple[float, int, int, int, int]]] = {D: [] for D in d_list}
        seen: dict[int, set[tuple]] = {D: set() for D in d_list}

        for i in range(M):
            wi, hi, gi, fi = int(ws[i]), int(hs[i]), int(gs[i]), float(fs[i])
            wsum = wi + ws[i:]
            hsum = hi + hs[i:]
            kx1 = (wsum == K + 1) & (hsum <= K - 1)
            ky1 = (hsum == K + 1) & (wsum <= K - 1)
            ok = kx1 | ky1
            js = np.nonzero(ok)[0] + i
            for j in js:
                j = int(j)
                gsum = gi + int(gs[j]) - 1
                f2 = max(fi, float(fs[j]))
                kx = max(0, wi + int(ws[j]) - K)
                ky = max(0, hi + int(hs[j]) - K)
                for D in d_list:
                    if (gsum == D if exact_g else gsum >= D) and f2 <= f_lim[D] + 1e-12:
                        sk = tuple(sorted([(wi, hi, gi, cand[i]["src"]), (int(ws[j]), int(hs[j]), int(gs[j]), cand[j]["src"])]))
                        if sk in seen[D]:
                            continue
                        hp = heaps[D]
                        if len(hp) < top_k:
                            seen[D].add(sk)
                            heapq.heappush(hp, (-f2, i, j, kx, ky))
                        elif f2 < -hp[0][0] - 1e-12:
                            seen[D].add(sk)
                            _, oi, oj, _, _ = heapq.heapreplace(hp, (-f2, i, j, kx, ky))
                            osk = tuple(sorted([
                                (cand[oi]["w"], cand[oi]["h"], cand[oi]["g"], cand[oi]["src"]),
                                (cand[oj]["w"], cand[oj]["h"], cand[oj]["g"], cand[oj]["src"]),
                            ]))
                            seen[D].discard(osk)

        for D in d_list:
            hp = heaps[D]
            if not hp:
                continue
            n = K * K + D
            inc = L_inc[n]
            cands = sorted(hp, key=lambda t: -t[0])
            pairs = []
            for neg_f, i, j, kx, ky in cands:
                f2 = -neg_f
                p1, p2 = cand[i], cand[j]
                pairs.append({
                    "f": f2,
                    "L_rigid": K + f2,
                    "dL_rigid": K + f2 - inc,
                    "kx": kx,
                    "ky": ky,
                    "p1": p1,
                    "p2": p2,
                })
            cells.append({
                "n": n,
                "K": K,
                "D": D,
                "L_inc": inc,
                "best_dL_rigid": pairs[0]["dL_rigid"],
                "pairs": pairs,
            })

    cells.sort(key=lambda c: (c["best_dL_rigid"], -c["n"]))
    return cells


def _extract_50d(rows_map: dict[int, list[tuple[str, str, str, str]]], pk: dict, tr: bool) -> tuple[list[dict], mp.mpf, mp.mpf] | None:
    src = int(pk["src"])
    if src not in rows_map:
        return None
    sq, W, H = bc.pocket_setup(rows_map[src], pk, tr)
    # bc.pocket_setup returns angle as string; sym_squares_50d handles it
    return sq, W, H


def _pslp_polish(n: int, K: int, cert: dict, L_inc_dec: Decimal) -> dict | None:
    bx, by, bt, L_cur = hlib._rows_to_arrays(cert["rows"])
    try:
        fn, Xn, Yn, Tn, vn, _ = pslp.squeeze(bx, by, bt, K, K, L_cur - K + 1e-12, center=1e-4)
    except Exception:
        return None
    if vn > 1e-12 or K + fn >= L_cur - 1e-10:
        return None
    shrunk = lpslack.try_shrink_and_certify(n, np.column_stack([Xn, Yn]), Tn, Decimal("999.0"))
    if shrunk is None:
        return None
    lit, pan, srows = shrunk
    dL_lit = float(lit - L_inc_dec)
    dL_pan = float(pan - L_inc_dec)
    is_win = (lit <= L_inc_dec - Decimal("1e-9")) and (pan <= L_inc_dec - Decimal("1e-9"))
    cand = {
        "ok": True,
        "n": n,
        "L_lit": str(lit),
        "L_pd": str(pan),
        "L_inc": str(L_inc_dec),
        "dL_lit": dL_lit,
        "dL_pd": dL_pan,
        "is_real_win": bool(is_win),
        "rows": srows,
    }
    if max(lit, pan) < max(Decimal(cert["L_lit"]), Decimal(cert["L_pd"])):
        return cand
    return None


def _save_ckpt(n: int, K: int, D: int, L_inc_str: str, cert: dict, label: str) -> None:
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
        "dL": max(cert["dL_lit"], cert["dL_pd"]),
        "rows": cert["rows"],
    }
    auto_union.atomic(ckpt_path, json.dumps(ckpt, indent=1).encode())


def eval_cell(args_tuple) -> dict:
    cell, L_inc_str, max_pairs, try_mechwalk = args_tuple
    t0 = time.time()
    n, K, D = int(cell["n"]), int(cell["K"]), int(cell["D"])
    L_inc_dec = Decimal(L_inc_str)
    rows_map = _G_ROWS

    best_res = None
    best_dec = None
    best_lbl = ""

    for p_idx, pair in enumerate(cell["pairs"][:max_pairs]):
        p1, p2 = pair["p1"], pair["p2"]
        ext1 = _extract_50d(rows_map, p1["pk"], p1["tr"])
        ext2 = _extract_50d(rows_map, p2["pk"], p2["tr"])
        if ext1 is None or ext2 is None:
            continue
        sq1, W1, H1 = ext1
        sq2, W2, H2 = ext2
        for s1, s2 in ((0, 0), (3, 0), (1, 0), (2, 0)):
            q1 = il.sym_squares_50d(sq1, W1, H1, s1)
            q2 = il.sym_squares_50d(sq2, W2, H2, s2)
            rows0 = il.assemble_zero_gap_50d(K, n, q1, W1, H1, p1["w"], p1["h"], q2, W2, H2, p2["w"], p2["h"])
            if rows0 is None:
                continue
            res = il.polish_rows_lpslack_mechwalk(n, rows0, L_inc_dec)
            if not res.get("ok"):
                continue
            ps = _pslp_polish(n, K, res, L_inc_dec)
            if ps is not None:
                res = ps
            cd = max(Decimal(res["L_lit"]), Decimal(res["L_pd"]))
            lbl = f"oh_p{p_idx}_s{s1}{s2}_{p1['src']}+{p2['src']}"
            if best_dec is None or cd < best_dec:
                best_dec = cd
                best_res = res
                best_lbl = lbl
            if res["is_real_win"]:
                _save_ckpt(n, K, D, L_inc_str, res, lbl)
            if res["dL_lit"] <= -5e-4:
                break
        if best_res is not None and best_res["dL_lit"] <= -5e-4:
            break

    if best_res is not None and try_mechwalk and best_res["dL_lit"] < 2.0e-3:
        mw = ie.mechwalk_polish_once(n, best_res["rows"], L_inc_dec, n_threads=1)
        if mw is not None and mw.get("ok"):
            ps = _pslp_polish(n, K, mw, L_inc_dec)
            if ps is not None:
                mw = ps
            cd = max(Decimal(mw["L_lit"]), Decimal(mw["L_pd"]))
            if cd < best_dec:
                best_dec = cd
                best_res = mw
                best_lbl += "+mw"
            if mw["is_real_win"]:
                _save_ckpt(n, K, D, L_inc_str, mw, best_lbl)

    out = {
        "n": n,
        "K": K,
        "D": D,
        "L_inc": L_inc_str,
        "best_dL_rigid": cell["best_dL_rigid"],
        "elapsed_s": round(time.time() - t0, 2),
        "win": bool(best_res is not None and best_res["is_real_win"]),
        "label": best_lbl,
    }
    if best_res is not None:
        out["L_lit"] = best_res["L_lit"]
        out["L_pd"] = best_res["L_pd"]
        out["dL"] = max(best_res["dL_lit"], best_res["dL_pd"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--pockets", nargs="+", required=True)
    ap.add_argument("--out-pred", required=True)
    ap.add_argument("--out-eval")
    ap.add_argument("--slack-max", type=float, default=6.0e-3)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--limit", type=int, default=6)
    ap.add_argument("--n-list")
    ap.add_argument("--max-pairs", type=int, default=2)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--no-mechwalk", action="store_true")
    args = ap.parse_args()

    rep_rows = json.loads(Path(args.report).read_text())["results"]
    L_inc = {int(r["n"]): float(r["L"]) for r in rep_rows}
    L_inc_str = {int(r["n"]): r["L"] for r in rep_rows}

    items = load_oriented_items(args.pockets)
    cells = search_overhang(items, L_inc, slack_max=args.slack_max, top_k=args.top_k)
    Path(args.out_pred).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_pred).write_text(json.dumps({"cells": cells}, indent=1))
    print(f"Overhang DP: {len(items)} oriented items -> {len(cells)} qualifying (K,D) cells (D=15..28, K=18..31)", flush=True)
    for c in cells[:20]:
        p = c["pairs"][0]
        print(
            f"  n={c['n']} (K={c['K']},D={c['D']}) dL_rigid={c['best_dL_rigid']:+.6f} "
            f"({p['p1']['w']},{p['p1']['h']},+{p['p1']['g']})<{p['p1']['src']}> + "
            f"({p['p2']['w']},{p['p2']['h']},+{p['p2']['g']})<{p['p2']['src']}>",
            flush=True,
        )

    if not args.out_eval:
        return

    if args.n_list:
        want = {int(x) for x in args.n_list.split(",") if x.strip()}
        eval_cells = [c for c in cells if c["n"] in want]
    else:
        eval_cells = cells[: args.limit] if args.limit > 0 else cells

    global _G_ROWS
    _G_ROWS = bc.load_rows(args.csv)
    tasks = [(c, L_inc_str[c["n"]], args.max_pairs, not args.no_mechwalk) for c in eval_cells]
    results = []
    out_eval = Path(args.out_eval)
    out_eval.parent.mkdir(parents=True, exist_ok=True)
    if min(2, args.workers) <= 1 or len(tasks) <= 1:
        for t in tasks:
            r = eval_cell(t)
            results.append(r)
            print(f"  [OH] n={r['n']} (K={r['K']},D={r['D']}) dL_rigid={r['best_dL_rigid']:+.6f} -> win={r['win']} dL={r.get('dL')} ({r['elapsed_s']}s)", flush=True)
            out_eval.write_text(json.dumps(results, indent=1))
    else:
        with mp_proc.get_context("fork").Pool(2) as pool:
            for r in pool.imap_unordered(eval_cell, tasks, chunksize=1):
                results.append(r)
                print(f"  [OH] n={r['n']} (K={r['K']},D={r['D']}) dL_rigid={r['best_dL_rigid']:+.6f} -> win={r['win']} dL={r.get('dL')} ({r['elapsed_s']}s)", flush=True)
                out_eval.write_text(json.dumps(results, indent=1))
    out_eval.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
