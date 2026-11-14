#!/usr/bin/env python3
"""Scale-up sweep for H74 (cleanpeel2) across {910, 851, 795, 740, 687, 973, 974, 909, 850}."""
from __future__ import annotations

import json
import math
import sys
import time
from decimal import Decimal
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import cleanpeel2_lib as cp2  # noqa: E402
import b8_lib as bl  # noqa: E402
import run_gate_fast as rgf  # noqa: E402

LPSLACK_POC = cp2.REPO_ROOT / "poc" / "swarm_lpslack" / "poc"
if str(LPSLACK_POC) not in sys.path:
    sys.path.insert(0, str(LPSLACK_POC))
import lpslack  # noqa: E402

SCALE_TARGETS = [973, 910, 851, 795, 740, 687, 974, 909, 850]


def prepare_extra_bricks() -> list[dict]:
    """Peel/walk B6_24x8 -> 21x8+6, 20x8+6 and walk donor_h63_brick_B5_20x10_n972."""
    out: list[dict] = []
    b24 = cp2.load_brick_json(cp2.REPO_ROOT / "poc" / "swarm_b5xplant" / "data" / "donor_h63_brick_B6_24x8.json")
    for k, tag in ((3, "B6_21x8_peelR"), (4, "B6_20x8_peelR")):
        p = cp2.DATA_DIR / f"brick_{tag}.json"
        if p.exists():
            out.append(cp2.load_brick_json(p))
            continue
        cur = b24
        ok = True
        for _ in range(k):
            cur = cp2.peel_outer_unit_layer(cur, "right", tag=tag)
            if cur is None:
                ok = False
                break
        if ok:
            slp_b = cp2.compress_brick_slp(cur, n_trials=12, seed=74900 + k)
            walk_b = cp2.brick_mechwalk(slp_b, name=tag, max_rounds=2, save=True)
            out.append(walk_b)

    p20 = cp2.DATA_DIR / "brick_B5_20x10_h63_walk.json"
    if p20.exists():
        out.append(cp2.load_brick_json(p20))
    else:
        b20 = cp2.load_brick_json(cp2.REPO_ROOT / "poc" / "swarm_b5xplant" / "data" / "donor_h63_brick_B5_20x10_n972.json")
        w20 = cp2.brick_mechwalk(b20, name="B5_20x10_h63_walk", max_rounds=2, save=True)
        out.append(w20)
    return out


def load_all_bricks() -> list[dict]:
    bricks: list[dict] = []
    seen: set[tuple[str, int, int, int]] = set()

    def _add(b: dict) -> None:
        for cand in (b, bl.transform_brick(b, "T")):
            t = cand.get("tag") or b.get("tag") or f"b_{cand['w']}x{cand['h']}"
            if cand["w"] != b["w"] or cand["h"] != b["h"]:
                t = t + "_T"
            cand = dict(cand, tag=t)
            key = (t, cand["w"], cand["h"], cand["g"])
            if key not in seen:
                seen.add(key)
                bricks.append(cand)

    for b in prepare_extra_bricks():
        _add(b)
    for folder in (
        cp2.DATA_DIR,
        cp2.REPO_ROOT / "poc" / "swarm_gstep" / "data",
    ):
        for p in sorted(folder.glob("brick_*.json")):
            _add(cp2.load_brick_json(p))
    for p in sorted((cp2.REPO_ROOT / "poc" / "swarm_b5xplant" / "data").glob("donor_h63_brick_*.json")):
        b = cp2.load_brick_json(p)
        b["tag"] = p.stem.replace("donor_h63_brick_", "")
        _add(b)
    for cache_p in (
        cp2.REPO_ROOT / "poc" / "swarm_b5xplant" / "data" / "compressed_b5_b4_cache.json",
        cp2.REPO_ROOT / "poc" / "swarm_thinb4" / "data" / "compressed_bricks_cache.json",
    ):
        if cache_p.exists():
            raw_c = json.loads(cache_p.read_text(encoding="utf-8"))
            items = raw_c.values() if isinstance(raw_c, dict) else raw_c
            for item in items:
                b = cp2.make_brick(
                    item["w"], item["h"], item["g"],
                    np.array(item["x"], dtype=np.float64),
                    np.array(item["y"], dtype=np.float64),
                    np.array(item["th"], dtype=np.float64),
                    tag=item["tag"],
                )
                _add(b)
    return bricks


def main() -> None:
    t0 = time.time()
    bricks = load_all_bricks()
    print(f"Loaded {len(bricks)} oriented bricks for scale-up", flush=True)
    frontier = cp2.fast_load_true_frontier(SCALE_TARGETS, certify=True)
    print(f"Loaded certified frontier in {time.time()-t0:.1f}s", flush=True)

    summary = {"targets": {}, "wins": []}
    for n in SCALE_TARGETS:
        tn = time.time()
        m = int(math.isqrt(n))
        r = n - m * m
        inc_dec = frontier[n]["L_dec"]
        f_inc = float(inc_dec) - m
        pairs = rgf.pick_pairs_for_n(n, bricks, f_inc, top_k=10)
        best_cert = None
        pair_logs = []

        print(f"\n=== SCALE n={n} (m={m}, r=+{r}) L_inc={str(inc_dec)[:18]} pairs={len(pairs)} ===", flush=True)
        for idx, (_, b1_base, b2_base) in enumerate(pairs):
            kx = b1_base["w"] + b2_base["w"] - m
            ky = b1_base["h"] + b2_base["h"] - m
            if kx == 0 and ky == 0:
                if max(b1_base["f"], b2_base["f"]) < f_inc - 1e-4:
                    cert = rgf.fast_solve_pair(
                        n, m, b1_base, b2_base, inc_dec,
                        f"s{n}_{b1_base['tag']}+{b2_base['tag']}",
                        base_seed=94000 + n * 10 + idx * 100, n_threads=2,
                    )
                    if cert is not None:
                        pair_logs.append({"pair": f"{b1_base['tag']}+{b2_base['tag']}", "kx": kx, "ky": ky, "L": cert["L_dec_str"][:18], "dL": cert["dL"]})
                        print(f"  [{idx+1}] melt kx=0 ky=0 -> L={cert['L_dec_str'][:18]} dL={cert['dL']:+.6e}", flush=True)
                        if best_cert is None or cert["L_float"] < best_cert["L_float"]:
                            best_cert = cert
                continue

            pair_best = None
            for m1 in ("id", "flipx", "rot180"):
                b1 = bl.transform_brick(b1_base, m1)
                for m2 in ("id", "flipy", "rot180"):
                    b2 = bl.transform_brick(b2_base, m2)
                    built = cp2.build_pinwheel_zero_ov(m, b1, b2, sf=0.0, f_override=None)
                    if built is None:
                        continue
                    ax, ay, at, _, _ = built
                    pen, _ = cp2.cpp_eval_sat_pen_bbox(ax, ay, at)
                    if pen > 1e-10:
                        continue
                    sx, sy, st, _, ok = cp2.bslp.joint_xytheta_slp_tight(ax, ay, at)
                    use_x, use_y, use_t = (sx, sy, st) if ok else (ax, ay, at)
                    cert = bl.polish_and_certify(n, use_x, use_y, use_t, inc_dec, slp_rounds=16)
                    if cert is not None:
                        cert["label"] = f"s{n}_{b1_base['tag']}+{b2_base['tag']}_{m1}_{m2}_direct"
                        if pair_best is None or cert["L_float"] < pair_best["L_float"]:
                            pair_best = cert
            if pair_best is not None:
                pair_logs.append({"pair": f"{b1_base['tag']}+{b2_base['tag']}", "kx": kx, "ky": ky, "L": pair_best["L_dec_str"][:18], "dL": pair_best["dL"]})
                print(f"  [{idx+1}] kx={kx} ky={ky} {b1_base['tag']}+{b2_base['tag']} -> L={pair_best['L_dec_str'][:18]} dL={pair_best['dL']:+.6e}", flush=True)
                if best_cert is None or pair_best["L_float"] < best_cert["L_float"]:
                    best_cert = pair_best

        if best_cert is not None and best_cert.get("rows") and best_cert["dL"] < 0.015:
            base_rows = [
                {"square_id": str(row["square_id"]), "x": str(row["x"]), "y": str(row["y"]), "angle": str(row["angle"])}
                for row in best_cert["rows"]
            ]
            target_ref = min(inc_dec, Decimal(best_cert["L_lit"]), Decimal(best_cert["L_pandas"]))
            lp_res = lpslack.optimize_group(n, base_rows, str(target_ref))
            if lp_res.get("status") in ("WIN", "TINY_WIN") and lp_res.get("rows"):
                L_worst = max(Decimal(lp_res["literal_L"]), Decimal(lp_res["pandas_L"]))
                dL_true = L_worst - frontier[n]["L_dec"]
                lp_cert = {
                    "ok": True,
                    "n": int(n),
                    "L_lit": lp_res["literal_L"],
                    "L_pandas": lp_res["pandas_L"],
                    "L_dec_str": str(L_worst),
                    "L_float": float(L_worst),
                    "dL": float(dL_true),
                    "is_real_win": bool(dL_true <= Decimal("-1e-9")),
                    "label": f"{best_cert['label']}_lpslack",
                    "rows": [
                        {"square_id": int(row["square_id"]), "x": str(row["x"]), "y": str(row["y"]), "angle": str(row["angle"])}
                        for row in lp_res["rows"]
                    ],
                }
                if lp_cert["L_float"] < best_cert["L_float"]:
                    best_cert = lp_cert
        if best_cert and best_cert["is_real_win"]:
            cp2.save_win_locked(best_cert)
            summary["wins"].append(n)
            print(f"  *** CERTIFIED SCALE WIN n={n}: dL={best_cert['dL']:+.6e} ***", flush=True)

        summary["targets"][str(n)] = {
            "n": n, "m": m, "r": r,
            "L_inc": str(frontier[n]["L_dec"]),
            "inc_src": frontier[n]["source"],
            "best_L": best_cert["L_dec_str"] if best_cert else None,
            "dL": best_cert["dL"] if best_cert else None,
            "win": bool(best_cert and best_cert["is_real_win"]),
            "label": best_cert["label"] if best_cert else None,
            "pairs_tested": pair_logs,
            "dt": round(time.time() - tn, 1),
        }
        (cp2.DOCS_DIR / "scale_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    summary["total_dt"] = round(time.time() - t0, 1)
    (cp2.DOCS_DIR / "scale_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSCALE COMPLETE in {summary['total_dt']}s | wins={summary['wins']}", flush=True)


if __name__ == "__main__":
    main()
