#!/usr/bin/env python3
"""30-Minute Self-Judging Gate for H74 (cleanpeel2) on {971, 910, 911, 852, 795}."""
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

LPSLACK_POC = cp2.REPO_ROOT / "poc" / "swarm_lpslack" / "poc"
if str(LPSLACK_POC) not in sys.path:
    sys.path.insert(0, str(LPSLACK_POC))
import lpslack  # noqa: E402

GATE_TARGETS = [971, 910, 911, 852, 795, 972]


def load_cp2_bricks() -> list[dict]:
    """Load all walked/source bricks from data/brick_*.json plus transposes, plus donor B5_10x19_n850."""
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

    for p in sorted(cp2.DATA_DIR.glob("brick_*.json")):
        b = cp2.load_brick_json(p)
        _add(b)

    # Also include B5_10x19 from thinb4/b5xplant caches if available
    for cache_p in (
        cp2.REPO_ROOT / "poc" / "swarm_b5xplant" / "data" / "compressed_b5_b4_cache.json",
        cp2.REPO_ROOT / "poc" / "swarm_thinb4" / "data" / "compressed_bricks_cache.json",
    ):
        if cache_p.exists():
            raw_c = json.loads(cache_p.read_text(encoding="utf-8"))
            items = raw_c.values() if isinstance(raw_c, dict) else raw_c
            for item in items:
                if item["g"] == 5 and (item["w"], item["h"]) in ((10, 19), (19, 10)):
                    b = cp2.make_brick(
                        item["w"], item["h"], item["g"],
                        np.array(item["x"], dtype=np.float64),
                        np.array(item["y"], dtype=np.float64),
                        np.array(item["th"], dtype=np.float64),
                        tag=item["tag"],
                    )
                    _add(b)
    return bricks


def pick_pairs_for_n(n: int, bricks: list[dict], f_inc: float, top_k: int = 6) -> list[tuple[float, dict, dict]]:
    m = int(math.isqrt(n))
    r = n - m * m
    cands: list[tuple[float, dict, dict]] = []
    seen_sym: set[tuple] = set()
    for b1 in bricks:
        if b1["w"] >= m or b1["h"] >= m:
            continue
        for b2 in bricks:
            if b1["g"] + b2["g"] != r:
                continue
            if b2["w"] >= m or b2["h"] >= m:
                continue
            kx = b1["w"] + b2["w"] - m
            ky = b1["h"] + b2["h"] - m
            if kx > 0 or ky > 0:
                continue
            f_hi = max(b1["f"], b2["f"])
            f_lo = min(b1["f"], b2["f"])
            if f_lo > f_inc + 0.035:
                continue
            # Canonicalize up to swap and transpose
            t1 = b1["tag"][:-2] if b1["tag"].endswith("_T") else b1["tag"]
            t2 = b2["tag"][:-2] if b2["tag"].endswith("_T") else b2["tag"]
            is_t1 = b1["tag"].endswith("_T")
            is_t2 = b2["tag"].endswith("_T")
            k_norm = tuple(sorted([(t1, is_t1), (t2, is_t2)]))
            k_trans = tuple(sorted([(t1, not is_t1), (t2, not is_t2)]))
            canon = min(k_norm, k_trans)
            if canon in seen_sym:
                continue
            seen_sym.add(canon)
            ov_bonus = -0.003 if (kx + ky < 0) else 0.0
            score = 0.7 * f_hi + 0.3 * f_lo + ov_bonus
            cands.append((score, b1, b2))
    cands.sort(key=lambda z: z[0])
    out: list[tuple[float, dict, dict]] = []
    geom_cnt: dict[tuple[int, int, int, int], int] = {}
    for sc, b1, b2 in cands:
        gk = tuple(sorted([(b1["w"], b1["h"]), (b2["w"], b2["h"])]))
        if geom_cnt.get(gk, 0) >= 2:
            continue
        geom_cnt[gk] = geom_cnt.get(gk, 0) + 1
        out.append((sc, b1, b2))
        if len(out) >= top_k:
            break
    return out


def main() -> None:
    t0 = time.time()
    cp2.DOCS_DIR.mkdir(parents=True, exist_ok=True)
    frontier = cp2.fast_load_true_frontier(GATE_TARGETS, certify=True)
    bricks = load_cp2_bricks()
    print(f"Loaded {len(bricks)} oriented bricks from {cp2.DATA_DIR}", flush=True)

    report: dict = {"targets": {}, "wins": []}
    for n in GATE_TARGETS:
        tn = time.time()
        m = int(math.isqrt(n))
        r = n - m * m
        inc_dec = frontier[n]["L_dec"]
        f_inc = float(inc_dec) - m
        pairs = pick_pairs_for_n(n, bricks, f_inc, top_k=6)
        print(f"\n=== GATE n={n} (m={m}, r=+{r}) L_inc={str(inc_dec)[:18]} (f_inc={f_inc:.9f}, src={frontier[n]['source']}) | pairs={len(pairs)} ===", flush=True)

        best_cert = None
        for idx, (sc, b1, b2) in enumerate(pairs):
            tag = f"g{n}_{b1['tag']}+{b2['tag']}"
            print(f"  [{idx+1}/{len(pairs)}] {b1['tag']}({b1['w']}x{b1['h']}+{b1['g']},f={b1['f']:.6f}) + {b2['tag']}({b2['w']}x{b2['h']}+{b2['g']},f={b2['f']:.6f}) kx={b1['w']+b2['w']-m} ky={b1['h']+b2['h']-m}", flush=True)
            cert = cp2.solve_pinwheel_pair(
                n, m, b1, b2, inc_dec, tag, base_seed=74000 + n * 10 + idx * 100, n_threads=8
            )
            if cert is not None:
                print(f"    -> L={cert['L_dec_str'][:18]} dL={cert['dL']:+.6e} win={cert['is_real_win']} ({cert['label']})", flush=True)
                if best_cert is None or cert["L_float"] < best_cert["L_float"]:
                    best_cert = cert
                if cert["is_real_win"]:
                    saved = cp2.save_win_locked(cert)
                    inc_dec = min(inc_dec, Decimal(cert["L_lit"]), Decimal(cert["L_pandas"]))
                    print(f"    *** CERTIFIED REAL WIN n={n}: saved={saved} dL={cert['dL']:+.6e} ***", flush=True)
            else:
                print(f"    -> no feasible", flush=True)

        if best_cert is not None and best_cert.get("rows"):
            base_rows = [
                {
                    "square_id": str(r["square_id"]),
                    "x": str(r["x"]),
                    "y": str(r["y"]),
                    "angle": str(r["angle"]),
                }
                for r in best_cert["rows"]
            ]
            target_ref_dec = min(inc_dec, Decimal(best_cert["L_lit"]), Decimal(best_cert["L_pandas"]))
            lp_res = lpslack.optimize_group(n, base_rows, str(target_ref_dec))
            if lp_res.get("status") in ("WIN", "TINY_WIN") and lp_res.get("rows"):
                print(f"  [lpslack polish n={n}] {best_cert['L_dec_str'][:18]} -> {lp_res['new_L'][:18]} (dL_extra=-{lp_res['dL']})", flush=True)
                lit_l = lp_res["literal_L"]
                pan_l = lp_res["pandas_L"]
                L_worst = max(Decimal(lit_l), Decimal(pan_l))
                dL_true = L_worst - frontier[n]["L_dec"]
                lp_cert = {
                    "ok": True,
                    "n": int(n),
                    "L_lit": lit_l,
                    "L_pandas": pan_l,
                    "L_dec_str": str(L_worst),
                    "L_float": float(L_worst),
                    "dL": float(dL_true),
                    "is_real_win": bool(dL_true <= Decimal("-1e-9")),
                    "label": f"{best_cert['label']}_lpslack",
                    "rows": [
                        {
                            "square_id": int(r["square_id"]),
                            "x": str(r["x"]),
                            "y": str(r["y"]),
                            "angle": str(r["angle"]),
                        }
                        for r in lp_res["rows"]
                    ],
                }
                if lp_cert["L_float"] < best_cert["L_float"]:
                    best_cert = lp_cert
                if lp_cert["is_real_win"]:
                    saved = cp2.save_win_locked(lp_cert)
                    print(f"    *** LPSLACK IMPROVED WIN n={n}: saved={saved} dL={lp_cert['dL']:+.6e} ***", flush=True)

        is_win = bool(best_cert and best_cert["is_real_win"])
        if is_win:
            report["wins"].append(n)
        report["targets"][str(n)] = {
            "n": n, "m": m, "r": r,
            "L_inc": str(frontier[n]["L_dec"]),
            "inc_src": frontier[n]["source"],
            "win": is_win,
            "best_L": best_cert["L_dec_str"] if best_cert else None,
            "dL": best_cert["dL"] if best_cert else None,
            "label": best_cert["label"] if best_cert else None,
            "dt": round(time.time() - tn, 1),
        }

    (cp2.DOCS_DIR / "gate_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n=== GATE COMPLETE in {time.time()-t0:.1f}s | WINS={report['wins']} ===", flush=True)


if __name__ == "__main__":
    main()
