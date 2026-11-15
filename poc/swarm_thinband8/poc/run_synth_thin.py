#!/usr/bin/env python3
"""Step 1+2 for H112 (thinband8): Harvest & synthesize w in {7, 8, 9}, h in {15..23}, g in {4..8} bricks."""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import thinband8_lib as tb  # noqa: E402
import gstep_lib as gl  # noqa: E402


def _compress_job(spec: tuple[str, dict, int]) -> dict | None:
    name, cand, seed = spec
    out_p = tb.DATA_DIR / "synth" / f"brick_{name}.json"
    if out_p.exists():
        b = tb.load_brick_json(out_p, tag=name)
        return {"name": name, "w": b["w"], "h": b["h"], "g": b["g"], "f": b["f"], "cached": True}
    comp = tb.compress_brick_slp(cand, n_trials=6, seed=seed)
    if comp is None:
        return None
    comp = tb.canon_brick_long_y(comp)
    comp["tag"] = name
    tb.save_brick_json(comp, name, subdir="synth")
    return {"name": name, "w": comp["w"], "h": comp["h"], "g": comp["g"], "f": comp["f"], "cached": False}


def _mechwalk_job(spec: tuple[str, str, int]) -> dict:
    src_path, out_name, rounds = spec
    out_p = tb.DATA_DIR / "synth" / f"brick_{out_name}.json"
    b = tb.load_brick_json(Path(src_path), tag=out_name)
    f0 = b["f"]
    if out_p.exists():
        bn = tb.load_brick_json(out_p, tag=out_name)
        return {"name": out_name, "w": bn["w"], "h": bn["h"], "g": bn["g"], "f0": f0, "f": bn["f"], "cached": True}
    wb = tb.brick_mechwalk(b, out_name, max_rounds=rounds, save=True)
    return {"name": out_name, "w": wb["w"], "h": wb["h"], "g": wb["g"], "f0": f0, "f": wb["f"], "cached": False}


def _collect_candidates(harvested: list[dict]) -> list[tuple[float, str, dict]]:
    cands: list[tuple[float, str, dict]] = []
    # Keep best donor per (w, h, g) for w in {7, 8, 9, 10}
    best_donor: dict[tuple[int, int, int], dict] = {}
    for b in harvested:
        k = (b["w"], b["h"], b["g"])
        if k not in best_donor or b["f"] < best_donor[k]["f"]:
            best_donor[k] = b

    def _add_cand(donor_f: float, name: str, b: dict | None) -> None:
        if b is None:
            return
        cb = tb.canon_brick_long_y(b)
        if cb["w"] in (7, 8, 9) and 15 <= cb["h"] <= 23 and 4 <= cb["g"] <= 8:
            cb["tag"] = name
            cands.append((donor_f, name, cb))

    # 1. Peel short axis (10->9->8->7, 9->8->7) and long axis (24->...->15) from top donors
    donors = [b for b in best_donor.values() if b["f"] <= 0.765 and 15 <= b["h"] <= 24]
    for d in donors:
        df = d["f"]
        dtag = f"{d['w']}x{d['h']}g{d['g']}"
        # Short-axis peels (left/right)
        for s1 in ("left", "right"):
            p1 = tb.peel_outer_unit_layer(d, s1, f"p_{dtag}_{s1[0]}")
            if p1 is not None:
                _add_cand(df, f"P1_{dtag}_{s1[0]}_{p1['w']}x{p1['h']}g{p1['g']}", p1)
                for s2 in ("left", "right"):
                    p2 = tb.peel_outer_unit_layer(p1, s2, f"p_{dtag}_{s1[0]}{s2[0]}")
                    if p2 is not None:
                        _add_cand(df, f"P2_{dtag}_{s1[0]}{s2[0]}_{p2['w']}x{p2['h']}g{p2['g']}", p2)
                for l1 in ("top", "bottom"):
                    pl = tb.peel_outer_unit_layer(p1, l1, f"p_{dtag}_{s1[0]}{l1[0]}")
                    if pl is not None:
                        _add_cand(df, f"PL_{dtag}_{s1[0]}{l1[0]}_{pl['w']}x{pl['h']}g{pl['g']}", pl)

        # Long-axis peels (top/bottom, 1 to 4 steps)
        for l1 in ("top", "bottom"):
            cur = tb.peel_outer_unit_layer(d, l1, f"l1_{dtag}_{l1[0]}")
            step = 1
            while cur is not None and step <= 4:
                _add_cand(df, f"L{step}_{dtag}_{l1[0]}_{cur['w']}x{cur['h']}g{cur['g']}", cur)
                if cur["w"] in (8, 9, 10):
                    for s1 in ("left", "right"):
                        ps = tb.peel_outer_unit_layer(cur, s1)
                        if ps is not None:
                            _add_cand(df, f"LS{step}_{dtag}_{l1[0]}{s1[0]}_{ps['w']}x{ps['h']}g{ps['g']}", ps)
                cur = tb.peel_outer_unit_layer(cur, l1, f"l{step+1}_{dtag}_{l1[0]}")
                step += 1

    # 2. vstrip addition (+1 h, +1 g) and removal (-1 h, -1 g) on w in {7, 8, 9} bricks with f <= 0.74
    for d in donors:
        if d["w"] not in (7, 8, 9) or d["f"] > 0.740:
            continue
        df = d["f"]
        dtag = f"{d['w']}x{d['h']}g{d['g']}"
        for tag, cand in gl.splice_vertical_strip(d)[:2]:
            ks = tag.split("_")[-1]
            _add_cand(df, f"VSadd_{dtag}_{ks}_{cand['w']}x{cand['h']}g{cand['g']}", cand)
            for s1 in ("left", "right"):
                ps = tb.peel_outer_unit_layer(cand, s1)
                if ps is not None:
                    _add_cand(df, f"VSaddP_{dtag}_{ks}{s1[0]}_{ps['w']}x{ps['h']}g{ps['g']}", ps)
        for tag, cand in tb.remove_vertical_strip(d)[:2]:
            ks = tag.split("_")[-1]
            _add_cand(df, f"VSrm_{dtag}_{ks}_{cand['w']}x{cand['h']}g{cand['g']}", cand)

    return cands


def main() -> None:
    t0 = time.time()
    tb.DATA_DIR.mkdir(parents=True, exist_ok=True)
    tb.DOCS_DIR.mkdir(parents=True, exist_ok=True)

    harvested = tb.harvest_all_thin_bricks(save_harvest=True)
    best_harv: dict[tuple[int, int, int], float] = {}
    for b in harvested:
        k = (b["w"], b["h"], b["g"])
        if k not in best_harv or b["f"] < best_harv[k]:
            best_harv[k] = b["f"]
    print(f"Harvested {len(harvested)} bricks across {len(best_harv)} (w,h,g) keys", flush=True)

    raw_cands = _collect_candidates(harvested)
    by_target: dict[tuple[int, int, int], list[tuple[float, float, str, dict]]] = {}
    seen_name: set[str] = set()
    for df, nm, b in raw_cands:
        if nm in seen_name:
            continue
        seen_name.add(nm)
        k = (b["w"], b["h"], b["g"])
        f_cur = best_harv.get(k)
        if f_cur is not None and f_cur <= 0.685:
            continue
        if b["f"] <= 0.88:
            by_target.setdefault(k, []).append((df, b["f"], nm, b))

    jobs: list[tuple[str, dict, int]] = []
    for k in sorted(by_target):
        lst = by_target[k]
        by_df = sorted(lst, key=lambda t: (t[0], t[1]))
        by_rf = sorted(lst, key=lambda t: (t[1], t[0]))
        picked = []
        for cand_list in (by_df, by_rf):
            df0, rf0, nm0, b0 = cand_list[0]
            if not any(nm0 == pnm for pnm in picked):
                picked.append(nm0)
                jobs.append((nm0, b0, 11200 + len(jobs)))

    print(f"Dispatching {len(jobs)} SLP brick compression jobs (max_workers=2)...", flush=True)
    comp_results: list[dict] = []
    with ProcessPoolExecutor(max_workers=2) as pool:
        futs = [pool.submit(_compress_job, j) for j in jobs]
        for fut in as_completed(futs):
            res = fut.result()
            if res is not None:
                comp_results.append(res)
                k = (res["w"], res["h"], res["g"])
                f_old = best_harv.get(k)
                tag_imp = "NEW" if f_old is None else f"df={res['f'] - f_old:+.6f}"
                print(f"  [COMP] {res['name']:36s} {res['w']}x{res['h']}+{res['g']} f={res['f']:.6f} ({tag_imp})", flush=True)

    # Step 2b: Mechwalk the best brick per (w, h, g) where 0.675 < f <= 0.740
    best_per_key: dict[tuple[int, int, int], tuple[float, Path]] = {}
    for p in sorted((tb.DATA_DIR / "synth").glob("brick_*.json")):
        if "_mw" in p.stem:
            continue
        b = tb.load_brick_json(p)
        k = (b["w"], b["h"], b["g"])
        if k not in best_per_key or b["f"] < best_per_key[k][0]:
            best_per_key[k] = (b["f"], p)
    for p in sorted((tb.DATA_DIR / "harvest").glob("brick_*.json")):
        b = tb.load_brick_json(p)
        if b["w"] in (7, 8, 9) and 15 <= b["h"] <= 23 and 4 <= b["g"] <= 8:
            k = (b["w"], b["h"], b["g"])
            if k not in best_per_key or b["f"] < best_per_key[k][0]:
                best_per_key[k] = (b["f"], p)

    mw_jobs: list[tuple[str, str, int]] = []
    for k, (fval, p) in sorted(best_per_key.items()):
        if 0.675 < fval <= 0.740:
            out_nm = f"MW_{k[0]}x{k[1]}g{k[2]}"
            mw_jobs.append((str(p), out_nm, 2))

    print(f"\nDispatching {len(mw_jobs)} brick_mechwalk polish jobs (max_workers=2)...", flush=True)
    mw_results: list[dict] = []
    with ProcessPoolExecutor(max_workers=2) as pool:
        futs = [pool.submit(_mechwalk_job, j) for j in mw_jobs]
        for fut in as_completed(futs):
            res = fut.result()
            mw_results.append(res)
            print(f"  [MW] {res['name']:18s} f={res['f0']:.6f} -> {res['f']:.6f} (df={res['f']-res['f0']:+.6f})", flush=True)

    # Final catalog summary across harvest + synth
    catalog: dict[tuple[int, int, int], dict] = {}
    for sub in ("harvest", "synth"):
        for p in sorted((tb.DATA_DIR / sub).glob("brick_*.json")):
            b = tb.load_brick_json(p)
            if b["w"] in (7, 8, 9) and 15 <= b["h"] <= 23 and 4 <= b["g"] <= 8:
                k = (b["w"], b["h"], b["g"])
                if k not in catalog or b["f"] < catalog[k]["f"]:
                    catalog[k] = {"w": b["w"], "h": b["h"], "g": b["g"], "f": b["f"], "path": str(p.relative_to(tb.REPO_ROOT))}

    summary = {
        "dt_s": round(time.time() - t0, 1),
        "n_compressed": len(comp_results),
        "n_mechwalked": len(mw_results),
        "catalog": [catalog[k] for k in sorted(catalog)],
    }
    (tb.DOCS_DIR / "synth_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n=== SYNTH COMPLETE in {summary['dt_s']}s | {len(catalog)} (w,h,g) slots filled ===", flush=True)
    for row in summary["catalog"]:
        print(f"  {row['w']:2d}x{row['h']:2d}+{row['g']}: f={row['f']:.6f} ({Path(row['path']).name})", flush=True)


if __name__ == "__main__":
    main()
