#!/usr/bin/env python3
"""H52 Step 2: Checkpointed Scale-Up Across r in {+7, +8, +9, +10}."""
from __future__ import annotations

import json
import math
import os
import sys
import time
from decimal import Decimal
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import thinb4_lib as tl  # noqa: E402
from cpp_wrap import cpp_eval_sat_pen_bbox  # noqa: E402


def ensure_missing_bricks_compressed(packs: dict, bank: dict[str, dict]) -> dict[str, dict]:
    """Compress any newly extracted base bricks (like B5_10x20_b4) not yet in cache."""
    if not tl.CACHE_PATH.exists():
        return bank
    cdata = json.loads(tl.CACHE_PATH.read_text())
    updated = False
    for key, b in list(bank.items()):
        if key.endswith("_T") or key in cdata:
            continue
        cb = tl.compress_brick_standalone(b, n_trials=16, n_workers=4)
        print(
            f"  [brick+] {key:16s} ({b['w']}x{b['h']},+{b['g']}) f0={b['f']:.6f} -> f={cb['f']:.6f}",
            flush=True,
        )
        bank[key] = cb
        bank[key + "_T"] = tl.bl.transpose_brick(cb)
        cdata[key] = {
            "w": cb["w"], "h": cb["h"], "g": cb["g"],
            "src_n": cb["src_n"], "tag": cb["tag"],
            "x": cb["x"].tolist(), "y": cb["y"].tolist(), "th": cb["th"].tolist(),
        }
        updated = True
    if updated:
        tl.auto_union.atomic(tl.CACHE_PATH, json.dumps(cdata).encode())
    return bank


def build_scale_candidates(n: int, bank: dict[str, dict], L_inc: float, max_cands: int = 5) -> list[dict]:
    """Find best H52 brick pair assemblies for n across zero-overlap, 1D-overlap, and corner-touch."""
    m = int(math.isqrt(n))
    r = n - m * m
    f_inc = L_inc - m
    cands: list[dict] = []
    seen = set()

    bricks = list(bank.values())
    pair_list = []
    for b1 in bricks:
        if b1["g"] >= r or b1["w"] >= m or b1["h"] >= m:
            continue
        for b2 in bricks:
            if b1["g"] + b2["g"] != r or b2["w"] >= m or b2["h"] >= m:
                continue
            kx = b1["w"] + b2["w"] - m
            ky = b1["h"] + b2["h"] - m
            if kx > 2 or ky > 2 or (kx > 0 and ky > 0):
                continue
            # Prefer strict zero-overlap (kx<=0, ky<=0, kx+ky<0), then 1D slack overlap, then kx=ky=0
            is_true_zw = (kx <= 0 and ky <= 0 and (kx + ky < 0))
            f_max = max(b1["f"], b2["f"])
            f_tight = max(b1["fx"], b2["fy"])
            f_min = min(b1["f"], b2["f"])
            f_est = 0.5 * (b1["f"] + b2["f"]) if is_true_zw else min(f_max, f_tight)
            if f_min >= f_inc - 1e-5 and f_max >= f_inc - 1e-5:
                continue
            if min(f_max, f_tight) > f_inc + 0.025 and not is_true_zw:
                continue
            if is_true_zw and f_max > f_inc + 0.030:
                continue

            k1 = (b1["tag"].replace("_T", ""), b1["w"], b1["h"], b1["g"])
            k2 = (b2["tag"].replace("_T", ""), b2["w"], b2["h"], b2["g"])
            k1t = (b1["tag"].replace("_T", ""), b1["h"], b1["w"], b1["g"])
            k2t = (b2["tag"].replace("_T", ""), b2["h"], b2["w"], b2["g"])
            sig = tuple(sorted((k1, k2)))
            sigt = tuple(sorted((k1t, k2t)))
            if sig in seen or sigt in seen:
                continue
            seen.add(sig)
            seen.add(sigt)

            tier = 0 if is_true_zw else (1 if (kx <= 0 or ky <= 0) else 2)
            # Prioritize pairs where max(f1,f2) < f_inc first, then slack-sharing pairs
            strict_below = 0 if f_max < f_inc - 1e-4 else 1
            pair_list.append(((tier, strict_below, f_est, f_max), b1, b2))

    pair_list.sort(key=lambda z: z[0])
    for _, b1, b2 in pair_list[:6]:
        asms = tl.assemble_pair_best(m, r, b1, b2, shift_fracs=(0.0, 0.5, 0.8))
        cands.extend(asms)

    cands.sort(key=lambda c: (0 if c["pen0"] < 1e-5 else 1, c["L"] + 0.05 * c["pen0"]))
    return cands[:max_cands]


def main() -> None:
    t0 = time.time()
    r8 = [684, 849, 737, 792, 908, 969, 633, 584]
    r9 = [850, 909, 793, 738, 685, 634]
    r10 = [910, 851, 794, 739, 686]
    r7 = [683, 736, 791, 848, 907, 632, 583]
    base_targets = r9 + r10 + r7 + r8

    ckpt_path = tl.DOCS_DIR / "scale_ckpt.json"
    done = json.loads(ckpt_path.read_text()) if ckpt_path.exists() else {}
    for wp in sorted(tl.DATA_DIR.glob("win_n*.json")):
        d = json.loads(wp.read_text())
        wn = str(d["n"])
        done[wn] = {
            "n": int(d["n"]),
            "status": "WIN",
            "dL": d["dL"],
            "L_lit": d["L_lit"],
            "label": d["label"],
        }

    print("=== Hypothesis H52 Step 2: Scale-Up Across r = +7, +8, +9, +10 ===", flush=True)
    packs, best_L = tl.fast_load_frontier(target_ns=base_targets)
    bank = tl.build_h52_bank(packs, compress=True)
    bank = ensure_missing_bricks_compressed(packs, bank)

    # Discover any additional r in {7,8,9,10} targets (m=18..31) with strong candidate advantage
    extra = []
    for n in range(330, 1000):
        if n in base_targets:
            continue
        m = int(math.isqrt(n))
        r = n - m * m
        if r not in (7, 8, 9, 10):
            continue
        cands = build_scale_candidates(n, bank, packs[n][3], max_cands=1)
        if cands and cands[0]["pen0"] < 1e-4 and cands[0]["L"] < packs[n][3] + 0.004:
            extra.append((cands[0]["L"] - packs[n][3], n))
    extra.sort()
    extra_ns = [n for _, n in extra[:10]]
    if extra_ns:
        _, ex_L = tl.fast_load_frontier(target_ns=extra_ns)
        best_L.update(ex_L)
    targets = base_targets + extra_ns
    print(f"Loaded {len(bank)} bricks in {time.time()-t0:.1f}s; targets ({len(targets)}): {targets}", flush=True)

    wins = [v for v in done.values() if v.get("status") == "WIN"]
    for idx, n in enumerate(targets, 1):
        if str(n) in done and done[str(n)].get("status") == "WIN":
            print(f"[Scale {idx}/{len(targets)}] n={n} already WIN ({done[str(n)]['L_lit'][:14]}), skipping", flush=True)
            continue
        m = int(math.isqrt(n))
        r = n - m * m
        print(f"\n[Scale {idx}/{len(targets)}] n={n} (m={m}, r=+{r}), L_inc={best_L[n]}", flush=True)
        cands = build_scale_candidates(n, bank, float(best_L[n]), max_cands=4)
        if not cands:
            print(f"  No candidates for n={n}", flush=True)
            done[str(n)] = {"n": n, "status": "NO_CANDS"}
        else:
            res = tl.eval_target(n, cands, best_L[n], n_threads=4, n_seeds=20, do_mechwalk_on_win=False)
            done[str(n)] = res
            if res.get("status") == "WIN":
                wins.append(res)
            print(f"  [Result n={n}] -> {res}", flush=True)
        tl.DOCS_DIR.mkdir(parents=True, exist_ok=True)
        tl.auto_union.atomic(ckpt_path, json.dumps(done, indent=2).encode())

    print(f"\n=== SCALE COMPLETE in {time.time()-t0:.1f}s | Wins ({len(wins)}): {[w['n'] for w in wins]} ===", flush=True)


if __name__ == "__main__":
    main()
