#!/usr/bin/env python3
"""Scale-up runner for Hypothesis H101 zero_gap (kx+ky in {1,2}, |I|=0) pinwheels."""
from __future__ import annotations

import argparse
import glob
import json
import multiprocessing as mp
import os
import sys
import time
from decimal import Decimal
from pathlib import Path

for _env in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_env] = "1"

import numpy as np

POC_DIR = Path(__file__).resolve().parent
SWARM_DIR = POC_DIR.parent
REPO = SWARM_DIR.parents[1]
sys.path.insert(0, str(POC_DIR))
sys.path.insert(0, str(REPO / "poc" / "swarm_highd" / "poc"))
sys.path.insert(0, str(REPO / "poc" / "swarm_lpslack" / "poc"))
sys.path.insert(0, str(REPO / "poc" / "swarm_mechwalk" / "poc"))
sys.path.insert(0, str(REPO / "poc" / "auto_union" / "poc"))

import interlock2_lib as il
import interlock2_eval as ie
import union as auto_union
from cpp_wrap import cpp_eval_sat_pen_bbox

PARTB_SANE = Path(
    "/home/lucas/.gemini/antigravity-cli/worktrees/a4fb0b93-b63b-426d-aa12-67454e99bc08"
    "/swarm-liftpolish3-18693d3f/poc/swarm_liftpolish3/data/pockets_partb_sane.json"
)


def enrich_pockets(
    by_whg: dict[tuple[int, int, int], list[dict]],
    merged_rows: dict[int, list[dict[str, str]]],
) -> dict[tuple[int, int, int], list[dict]]:
    """Add b8xplant/cleanpeel2 bricks and partb sanitized pockets, pruning top-2 per (w,h,g)."""
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

    for pat in (
        "poc/swarm_b8xplant/data/brick_*.json",
        "poc/swarm_cleanpeel2/data/brick_*.json",
    ):
        for jp in sorted(glob.glob(str(REPO / pat))):
            _add_brick(json.loads(Path(jp).read_text()), Path(jp).stem)

    if PARTB_SANE.exists():
        prep_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        for pk in json.loads(PARTB_SANE.read_text()):
            src = int(pk["src"])
            if src not in merged_rows:
                continue
            if src not in prep_cache:
                prep_cache[src] = il._prep_float(merged_rows[src])
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
                        "uid": f"pb_n{src}_{rw}x{rh}_g{g}",
                        "kind": "donor_pocket",
                        "src": src,
                        "rows_map": "merged",
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


def find_zero_gap_multi(
    by_whg: dict[tuple[int, int, int], list[dict]],
    inc_L: dict[int, float],
    dL_margin: float = 0.008,
    per_n: int = 2,
) -> list[dict]:
    """Find top `per_n` distinct zero_gap pairs per target n up to dL_margin."""
    best_items = [lst[0] for lst in by_whg.values()]
    hits: dict[int, list[dict]] = {}
    for K in range(16, 32):
        cand = [it for it in best_items if it["w"] < K and it["h"] < K]
        for i in range(len(cand)):
            p1 = cand[i]
            for j in range(i, len(cand)):
                p2 = cand[j]
                kx = max(0, p1["w"] + p2["w"] - K)
                ky = max(0, p1["h"] + p2["h"] - K)
                k = kx + ky
                if k not in (1, 2) or kx * ky != 0:
                    continue
                if kx > 0 and p1["h"] + p2["h"] > K - 1:
                    continue
                if ky > 0 and p1["w"] + p2["w"] > K - 1:
                    continue
                D = p1["g"] + p2["g"] - k
                if D < 1:
                    continue
                n = K * K + D
                if n > 1000 or n not in inc_L:
                    continue
                L_pred = K + max(p1["f"], p2["f"])
                dL = L_pred - inc_L[n]
                if dL <= dL_margin:
                    hits.setdefault(n, []).append(
                        {
                            "n": n,
                            "K": K,
                            "D": D,
                            "kx": kx,
                            "ky": ky,
                            "regime": "zero_gap",
                            "p1": p1,
                            "p2": p2,
                            "L_pred": L_pred,
                            "L_inc_est": inc_L[n],
                            "dL_pred": dL,
                        }
                    )
    out = []
    for n, lst in hits.items():
        lst.sort(key=lambda r: r["dL_pred"])
        seen_shapes = set()
        kept_n = 0
        for r in lst:
            s_a = tuple(sorted([(r["p1"]["w"], r["p1"]["h"], r["p1"]["g"]), (r["p2"]["w"], r["p2"]["h"], r["p2"]["g"])]))
            s_b = tuple(sorted([(r["p1"]["h"], r["p1"]["w"], r["p1"]["g"]), (r["p2"]["h"], r["p2"]["w"], r["p2"]["g"])]))
            sig = min(s_a, s_b)
            if sig in seen_shapes:
                continue
            seen_shapes.add(sig)
            out.append(r)
            kept_n += 1
            if kept_n >= per_n:
                break
    out.sort(key=lambda r: r["dL_pred"])
    return out


def exact_incumbents_single_pass(target_ns: set[int]) -> dict[int, Decimal]:
    """Scan all poc/*/final.csv once, float-prefilter, and exact-certify min L per target n."""
    best_by_n: dict[int, tuple[float, list[dict[str, str]]]] = {}
    import math
    for p in sorted((REPO / "poc").glob("*/final.csv")):
        if p.parent.name == "swarm_interlock2":
            continue
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
    out: dict[int, Decimal] = {}
    for n in sorted(target_ns):
        out[n] = Decimal(auto_union.certify(n, best_by_n[n][1]))
    return out


_G: dict = {}


def _init_worker(merged_rows, best_rows, exact_inc):
    _G["merged_rows"] = merged_rows
    _G["best_rows"] = best_rows
    _G["exact_inc"] = exact_inc


def _run_one(cand: dict) -> dict:
    n = cand["n"]
    L_inc_dec = _G["exact_inc"][n]
    syms = ((0, 0), (3, 0), (0, 3), (3, 3)) if cand["dL_pred"] <= 0.0045 else ((0, 0), (3, 0))
    res = ie.eval_zero_gap_candidate(
        cand,
        _G["merged_rows"],
        _G["best_rows"],
        L_inc_dec,
        syms=syms,
        try_mechwalk=True,
        n_threads=1,
    )
    res["pair"] = f"{cand['p1']['uid']}+{cand['p2']['uid']}"
    res["dL_pred"] = cand["dL_pred"]
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--margin", type=float, default=0.0065)
    ap.add_argument("--max-tasks", type=int, default=30)
    args = ap.parse_args()

    t0 = time.time()
    print("=== H101 Scale-Up: zero_gap (kx+ky in {1,2}, |I|=0) ===", flush=True)
    best_rows, inc_L = il.load_portfolio_best()
    merged_rows = il.load_merged_rows()
    by_whg = il.load_all_pockets(best_rows, extra_scan_path=il.DATA_DIR / "scanned_5rows.json")
    by_whg = enrich_pockets(by_whg, merged_rows)
    print(f"Loaded {len(by_whg)} unique (w,h,g) keys ({sum(len(v) for v in by_whg.values())} items)", flush=True)

    # Load already-tested exact labels from gate_results.json
    tested_pairs: set[tuple[int, str]] = set()
    gr_path = il.DOCS_DIR / "gate_results.json"
    if gr_path.exists():
        gr = json.loads(gr_path.read_text())
        for r in gr.get("results", []):
            if r.get("regime") == "zero_gap":
                lbl = r.get("label", "")
                if lbl.startswith("zg_"):
                    core = lbl[3:].split("_s")[0]
                    tested_pairs.add((int(r["n"]), core))

    gate_done_ns = {454, 640, 791, 795, 910, 912}
    cands = find_zero_gap_multi(by_whg, inc_L, dL_margin=args.margin, per_n=2)
    todo = []
    for c in cands:
        sig = f"{c['p1']['uid']}+{c['p2']['uid']}"
        if (c["n"], sig) in tested_pairs and c["n"] != 849:
            continue
        if c["n"] in gate_done_ns and c["dL_pred"] > 0.001:
            continue
        # Skip 0.000000 exact ties with existing staircase incumbents unless new source
        if abs(c["dL_pred"]) < 1e-8 and c["n"] in (591, 857):
            continue
        todo.append(c)
    todo = todo[: args.max_tasks]
    print(f"Selected {len(todo)} scale targets (dL_pred <= {args.margin:+.4f}):", flush=True)
    for c in todo:
        print(
            f"  n={c['n']:4d} K={c['K']} D={c['D']:2d} k=({c['kx']},{c['ky']}) dL_pred={c['dL_pred']:+.6f} | {c['p1']['uid']} + {c['p2']['uid']}",
            flush=True,
        )
    if args.dry_run:
        return

    target_ns = {c["n"] for c in todo}
    exact_inc = exact_incumbents_single_pass(target_ns)
    print(f"Certified exact incumbents for {len(exact_inc)} ns in {time.time()-t0:.1f}s", flush=True)

    ctx = mp.get_context("fork")
    results = []
    with ctx.Pool(processes=2, initializer=_init_worker, initargs=(merged_rows, best_rows, exact_inc)) as pool:
        for res in pool.imap_unordered(_run_one, todo):
            results.append(res)
            print(
                f"  [SCALE DONE] n={res['n']} status={res['status']} dL={res.get('dL')} pair={res.get('pair')} dt={res.get('dt')}s",
                flush=True,
            )

    results.sort(key=lambda r: float(r.get("dL", 999.0)))
    out_path = il.DOCS_DIR / "scale_results.json"
    out_path.write_text(json.dumps({"elapsed_s": round(time.time() - t0, 1), "results": results}, indent=2))
    wins = [r for r in results if r.get("status") == "WIN"]
    print(f"\n=== SCALE SUMMARY: {len(wins)}/{len(results)} WINS in {time.time()-t0:.1f}s ===", flush=True)
    for w in wins:
        print(f"  WIN n={w['n']}: dL={w['dL']:+.6e} L={w['L_lit']} ({w['pair']})", flush=True)


if __name__ == "__main__":
    main()
