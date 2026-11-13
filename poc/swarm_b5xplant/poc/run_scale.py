#!/usr/bin/env python3
"""H40 Scale-Up Sweep across r in {+5, +8, +9, +10, +11} + auto-discovered B5/B4 gaps."""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import b5_lib as bl  # noqa: E402
from cpp_wrap import cpp_eval_sat_pen_bbox  # noqa: E402


def build_single_brick_candidates(n: int, bank: list[dict], L_inc: float) -> list[dict]:
    """Assemble single n850 brick (r=+4 or r=+5) + complementary unit grid in [0, m+f]^2."""
    m = int(math.isqrt(n))
    r = n - m * m
    f_inc = L_inc - m
    cands = []
    seen = set()
    for b in bank:
        if "n850" not in b["tag"] or b["g"] != r:
            continue
        if b["w"] > m or b["h"] > m or b["f_min"] >= f_inc + 0.03:
            continue
        sig = (b["tag"].replace("_T", ""), b["w"], b["h"], b["g"])
        if sig in seen:
            continue
        seen.add(sig)
        w1, h1 = b["w"], b["h"]
        f_full = b["f"]
        f_tight = b["fx"] if b["fx"] <= b["fy"] else b["fy"]
        f_modes = ["tight", "full"] if (f_full - f_tight > 0.01 and f_full <= f_inc + 0.020) else ["tight"]
        for fm in f_modes:
            f = max(f_tight if fm == "tight" else f_full, 0.642572)
            L = float(m) + f
            best_s = None
            for s in range(4):
                sb = bl.sym_brick(b, s)
                x1, y1, t1 = sb["x"], sb["y"] + (L - (h1 + sb["fy"])), sb["th"]
                gx, gy = [], []
                for i in range(m):
                    for j in range(m):
                        if i < w1 and j >= m - h1:
                            continue
                        gx.append(i + 0.5 + (f if i >= w1 else 0.0))
                        gy.append(j + 0.5 + (f if j >= h1 and i >= w1 else 0.0))
                ax = np.concatenate([x1, np.array(gx, dtype=np.float64)])
                ay = np.concatenate([y1, np.array(gy, dtype=np.float64)])
                at = np.concatenate([t1, np.zeros(len(gx), dtype=np.float64)])
                if len(ax) != n:
                    continue
                pen, _ = cpp_eval_sat_pen_bbox(ax, ay, at)
                if best_s is None or pen < best_s[0]:
                    sbox = (max(0.0, float(w1) - 1.5), min(L, float(w1) + f + 2.0), max(0.0, float(m - h1) - 2.0), min(L, float(m - h1) + 2.0))
                    best_s = (pen, ax, ay, at, L, sbox, f"{fm}_s{s}")
            if best_s is not None:
                pen, ax, ay, at, aL, sbox, cfg = best_s
                cands.append({
                    "n": n, "m": m, "r": r, "b1": b, "b2": b,
                    "pen0": float(pen), "x": ax, "y": ay, "th": at, "L": aL, "sbox": sbox,
                    "tag": f"single_{b['tag']}_{cfg}",
                })
    cands.sort(key=lambda c: (0 if (c["pen0"] < 1e-5 and c["L"] <= L_inc + 0.005) else 1, c["L"] + 0.025 * c["pen0"]))
    return cands[:3]


def main():
    t0 = time.time()
    base_targets = [
        793, 910, 911, 909,
        851, 852, 794, 738, 685,
        795, 739, 740, 686, 687,
        634, 635, 970,
        366, 405, 446, 684, 737, 908,
        971, 972,
    ]
    ckpt_path = bl.DOCS_DIR / "scale_ckpt.json"
    done = json.loads(ckpt_path.read_text()) if ckpt_path.exists() else {}
    # Pre-seed already certified win_n*.json
    for wp in sorted(bl.DATA_DIR.glob("win_n*.json")):
        d = json.loads(wp.read_text())
        wn = str(d["n"])
        if wn not in done:
            done[wn] = {"n": int(d["n"]), "status": "WIN", "dL": d["dL"], "L_lit": d["L_lit"], "label": d["label"]}

    print("=== H40 Scale-Up Sweep ===", flush=True)
    packs, best_L = bl.fast_load_frontier(target_ns=base_targets)
    bank = bl.load_b5_bank(packs, compress_b5_b4=True)

    extra = []
    for n in range(360, 1000):
        if n in base_targets:
            continue
        m = int(math.isqrt(n))
        r = n - m * m
        if r not in (8, 9, 10, 11):
            continue
        f_inc = packs[n][3] - m
        if f_inc > 0.648:
            cands = bl.find_b5_candidates(n, bank, packs[n][3], max_cands=2)
            if cands and cands[0]["L"] - packs[n][3] < -0.003:
                extra.append((cands[0]["L"] - packs[n][3], n))
    extra.sort()
    extra_ns = [n for _, n in extra[:10]]
    if extra_ns:
        _, extra_L = bl.fast_load_frontier(target_ns=extra_ns)
        best_L.update(extra_L)
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
        if r in (4, 5):
            cands = build_single_brick_candidates(n, bank, float(best_L[n]))
        else:
            cands = bl.find_b5_candidates(n, bank, float(best_L[n]), max_cands=3)

        if not cands:
            print(f"  No candidates for n={n}", flush=True)
            done[str(n)] = {"n": n, "status": "NO_CANDS"}
        else:
            res = bl.eval_target(n, cands, best_L[n], n_threads=4)
            done[str(n)] = res
            if res.get("status") == "WIN":
                wins.append(res)
            print(f"  [Result n={n}] -> {res}", flush=True)

        bl.DOCS_DIR.mkdir(parents=True, exist_ok=True)
        ckpt_path.write_text(json.dumps(done, indent=2))

    print(f"\n=== Scale Sweep Complete in {time.time()-t0:.1f}s | Wins ({len(wins)}): {[w['n'] for w in wins]} ===", flush=True)


if __name__ == "__main__":
    main()
