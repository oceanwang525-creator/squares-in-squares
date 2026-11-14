#!/usr/bin/env python3
"""Step 3 Gate Runner for H112 (thinband8): K=2 kx,ky in {-1,0,+1} pinwheels on 10 gate targets."""
from __future__ import annotations

import json
import math
import os
import sys
import time
from decimal import Decimal
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import thinband8_lib as tb  # noqa: E402
import b8_lib as bl  # noqa: E402

GATE_TARGETS = [795, 741, 687, 854, 796, 853, 740, 852, 688, 797]


def load_thinband_bricks(extra_comp_dirs: bool = True) -> list[dict]:
    """Load harvested + synthesized bricks (top-2 per (w,h,g)), plus transposes."""
    raw: list[dict] = []
    for sub in ("synth", "harvest"):
        d = tb.DATA_DIR / sub
        if not d.exists():
            continue
        for p in sorted(d.glob("brick_*.json")):
            raw.append(tb.load_brick_json(p))

    if extra_comp_dirs:
        # Include top-1 h=10 bricks from cleanpeel2/gstep/thinb4 as companions
        for folder in (
            tb.REPO_ROOT / "poc" / "swarm_cleanpeel2" / "data",
            tb.REPO_ROOT / "poc" / "swarm_gstep" / "data",
        ):
            if not folder.exists():
                continue
            for p in sorted(folder.glob("brick_*.json")):
                b = tb.load_brick_json(p)
                if min(b["w"], b["h"]) <= 11:
                    raw.append(b)

    by_shape: dict[tuple[int, int, int], list[dict]] = {}
    seen_sig: set[tuple] = set()
    for b in raw:
        for cand in (b, bl.transform_brick(b, "T")):
            if cand is not b:
                cand["tag"] = b["tag"] + "_T"
            k = (cand["w"], cand["h"], cand["g"])
            sig = (*k, round(cand["f"], 6), round(float(cand["x"].mean()), 4))
            if sig in seen_sig:
                continue
            seen_sig.add(sig)
            by_shape.setdefault(k, []).append(cand)

    kept: list[dict] = []
    for k in sorted(by_shape):
        lst = sorted(by_shape[k], key=lambda c: c["f"])
        kept.extend(lst[:2])
    return kept


def pick_pairs_for_n(
    n: int,
    bricks: list[dict],
    f_inc: float,
    top_k: int = 8,
) -> list[tuple[float, dict, dict]]:
    m = int(math.isqrt(n))
    r = n - m * m
    cands: list[tuple[float, dict, dict]] = []
    seen_sym: set[tuple] = set()

    for b1 in bricks:
        if b1["w"] >= m or b1["h"] >= m:
            continue
        if b1["f"] > f_inc + 0.018:
            continue
        is_thin1 = min(b1["w"], b1["h"]) in (7, 8, 9)
        for b2 in bricks:
            if b1["g"] + b2["g"] != r:
                continue
            if not (is_thin1 or min(b2["w"], b2["h"]) in (7, 8, 9)):
                continue
            if b2["w"] >= m or b2["h"] >= m:
                continue
            kx = b1["w"] + b2["w"] - m
            ky = b1["h"] + b2["h"] - m
            if kx > 1 or ky > 1 or (kx > 0 and ky > 0):
                continue
            if kx < -2 or ky < -2:
                continue
            fmax = max(b1["f"], b2["f"])
            if fmax > f_inc + 0.015:
                continue

            # Canonical symmetry deduplication: (b1, b2) vs (b2_T, b1_T)
            t1 = b1["tag"][:-2] if b1["tag"].endswith("_T") else b1["tag"]
            t2 = b2["tag"][:-2] if b2["tag"].endswith("_T") else b2["tag"]
            i1, i2 = b1["tag"].endswith("_T"), b2["tag"].endswith("_T")
            canon = min(
                tuple(sorted([(t1, i1), (t2, i2)])),
                tuple(sorted([(t1, not i1), (t2, not i2)])),
            )
            if canon in seen_sym:
                continue
            seen_sym.add(canon)

            ov_pen = 0.0
            if kx > 0 or ky > 0:
                ov_pen = 0.004
            elif kx == 0 and ky == 0:
                ov_pen = 0.0015
            score = fmax + ov_pen
            cands.append((score, b1, b2))

    cands.sort(key=lambda z: z[0])
    out: list[tuple[float, dict, dict]] = []
    geom_cnt: dict[tuple, int] = {}
    for sc, b1, b2 in cands:
        gk = tuple(sorted([(b1["w"], b1["h"], b1["g"]), (b2["w"], b2["h"], b2["g"])]))
        if geom_cnt.get(gk, 0) >= 1:
            continue
        geom_cnt[gk] = 1
        out.append((sc, b1, b2))
        if len(out) >= top_k:
            break
    return out


def main() -> None:
    t0 = time.time()
    tb.DATA_DIR.mkdir(parents=True, exist_ok=True)
    tb.DOCS_DIR.mkdir(parents=True, exist_ok=True)

    targets = [int(a) for a in sys.argv[1:]] if len(sys.argv) > 1 else GATE_TARGETS
    bricks = load_thinband_bricks()
    print(f"Loaded {len(bricks)} oriented bricks for gate", flush=True)
    frontier = tb.fast_load_true_frontier(targets, certify=True)
    print(f"Loaded certified frontier for {len(frontier)} targets in {time.time()-t0:.1f}s", flush=True)

    summary = {"targets": {}, "wins": []}
    for n in targets:
        tn = time.time()
        m = int(math.isqrt(n))
        r = n - m * m
        inc_dec = frontier[n]["L_dec"]
        f_inc = float(inc_dec) - m
        pairs = pick_pairs_for_n(n, bricks, f_inc, top_k=8)
        print(
            f"\n=== GATE n={n} (m={m}, r=+{r}) L_inc={str(inc_dec)[:18]} "
            f"(f_inc={f_inc:.9f}, src={frontier[n]['source']}) | pairs={len(pairs)} ===",
            flush=True,
        )

        best_cert = None
        pair_logs = []
        for idx, (sc, b1, b2) in enumerate(pairs):
            kx = b1["w"] + b2["w"] - m
            ky = b1["h"] + b2["h"] - m
            tag = f"tb8_{n}_{b1['w']}x{b1['h']}g{b1['g']}+{b2['w']}x{b2['h']}g{b2['g']}"
            print(
                f"  [{idx+1}/{len(pairs)}] {b1['tag'][:20]}({b1['w']}x{b1['h']}+{b1['g']},f={b1['f']:.5f}) + "
                f"{b2['tag'][:20]}({b2['w']}x{b2['h']}+{b2['g']},f={b2['f']:.5f}) kx={kx} ky={ky}",
                flush=True,
            )
            cert = tb.solve_pinwheel_pair(
                n, m, b1, b2, inc_dec, tag,
                base_seed=112000 + n * 10 + idx * 100,
                n_threads=2,
                do_lpslack=True,
            )
            if cert is not None:
                pair_logs.append({
                    "pair": tag, "kx": kx, "ky": ky,
                    "L": cert["L_dec_str"][:20], "dL": cert["dL"], "win": cert["is_real_win"],
                })
                print(
                    f"    -> L={cert['L_dec_str'][:18]} dL={cert['dL']:+.6e} win={cert['is_real_win']} ({cert['label']})",
                    flush=True,
                )
                if best_cert is None or cert["L_float"] < best_cert["L_float"]:
                    best_cert = cert
                if cert["is_real_win"]:
                    saved = tb.save_win_locked(cert)
                    inc_dec = min(inc_dec, Decimal(cert["L_lit"]), Decimal(cert["L_pandas"]))
                    print(f"    *** CERTIFIED REAL WIN n={n}: saved={saved} dL={cert['dL']:+.6e} ***", flush=True)
            else:
                print("    -> no feasible", flush=True)

        is_win = bool(best_cert and best_cert["is_real_win"])
        if is_win:
            summary["wins"].append(n)
        summary["targets"][str(n)] = {
            "n": n, "m": m, "r": r,
            "L_inc": str(frontier[n]["L_dec"]),
            "inc_src": frontier[n]["source"],
            "win": is_win,
            "best_L": best_cert["L_dec_str"] if best_cert else None,
            "dL": best_cert["dL"] if best_cert else None,
            "label": best_cert["label"] if best_cert else None,
            "pairs_tested": pair_logs,
            "dt": round(time.time() - tn, 1),
        }
        (tb.DOCS_DIR / "gate_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    summary["total_dt"] = round(time.time() - t0, 1)
    (tb.DOCS_DIR / "gate_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n=== GATE COMPLETE in {summary['total_dt']}s | wins={summary['wins']} ===", flush=True)


if __name__ == "__main__":
    main()
