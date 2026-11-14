#!/usr/bin/env python3
"""H79 (gstep) Gate & Scale-Up Runner: K=2 Zero-Overlap Pinwheels + Seam Melt + lpslack."""
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

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gstep_lib as gl  # noqa: E402

REPO_ROOT = gl.REPO_ROOT
sys.path.insert(0, str(REPO_ROOT / "poc" / "swarm_lpslack" / "poc"))
import lpslack  # noqa: E402
from cpp_wrap import cpp_eval_sat_pen_bbox  # noqa: E402


def load_all_bricks() -> list[dict]:
    """Load all source bricks and synthesized/peeled bricks in data/."""
    seen: dict[tuple[int, int, int, str], dict] = {}

    def _add(b: dict, name: str):
        b = dict(b)
        b["tag"] = name
        seen[(b["w"], b["h"], b["g"], name)] = b

    for nm, b in gl.load_source_bricks().items():
        _add(b, nm)

    for p in sorted(gl.DATA_DIR.glob("brick_*.json")):
        b = gl.load_brick_json(p)
        _add(b, p.stem.replace("brick_", ""))

    # Deduplicate exact (w, h, g) keeping top-2 thinnest per (w, h, g), and include transpose
    by_shape: dict[tuple[int, int, int], list[dict]] = {}
    for b in seen.values():
        by_shape.setdefault((b["w"], b["h"], b["g"]), []).append(b)
        bt = gl.bl.transform_brick(b, "T")
        bt["tag"] = b["tag"] + "_T"
        by_shape.setdefault((bt["w"], bt["h"], bt["g"]), []).append(bt)

    kept: list[dict] = []
    for key, lst in sorted(by_shape.items()):
        lst.sort(key=lambda x: x["f"])
        kept.extend(lst[:2])
    return kept


def build_pinwheel_zero_ov(
    m: int, b1: dict, b2: dict, sf: float = 0.0, f_override: float | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray] | None:
    w1, h1, w2, h2 = b1["w"], b1["h"], b2["w"], b2["h"]
    if w1 >= m or h1 >= m or w2 >= m or h2 >= m:
        return None
    if w1 + w2 > m or h1 + h2 > m:
        return None
    if w1 + w2 + h1 + h2 > 2 * m - 1:
        return None
    f = max(b1["f"], b2["f"]) if f_override is None else f_override
    L = m + f
    dx1 = sf * (f - b1["fx"])
    dy1 = (m - h1) + (1.0 - sf) * (f - b1["fy"])
    dx2 = (m - w2) + (1.0 - sf) * (f - b2["fx"])
    dy2 = sf * (f - b2["fy"])

    gx, gy = [], []
    for i in range(m):
        for j in range(m):
            if i < w1 and j >= m - h1:
                continue
            if i >= m - w2 and j < h2:
                continue
            if i < m - w2 and j < m - h1:
                gx.append(i + 0.5)
                gy.append(j + 0.5)
            else:
                gx.append(i + f + 0.5)
                gy.append(j + f + 0.5)

    ax = np.concatenate([b1["x"] + dx1, b2["x"] + dx2, np.array(gx, dtype=np.float64)])
    ay = np.concatenate([b1["y"] + dy1, b2["y"] + dy2, np.array(gy, dtype=np.float64)])
    at = np.concatenate([b1["th"], b2["th"], np.zeros(len(gx), dtype=np.float64)])
    if len(ax) != m * m + b1["g"] + b2["g"]:
        return None
    seam = (
        ((ax >= w1 - 1.5) & (ax <= w1 + f + 1.8))
        | ((ay >= m - h1 - 1.8) & (ay <= m - h1 + 1.8))
        | ((ax >= m - w2 - 1.8) & (ax <= m - w2 + 1.8))
        | ((ay >= h2 - 1.5) & (ay <= h2 + f + 1.8))
    ).astype(np.uint8)
    return ax, ay, at, L, seam


def solve_pair_fast(
    n: int, m: int, b1_base: dict, b2_base: dict, inc_dec: Decimal, tag: str, base_seed: int, do_melt: bool = True
) -> dict | None:
    best_cert: dict | None = None
    seeds, masks, scheds, labels = [], [], [], []

    for m1 in ("id", "flipx", "flipy", "rot180"):
        b1 = gl.bl.transform_brick(b1_base, m1)
        for m2 in ("id", "flipx", "flipy", "rot180"):
            b2 = gl.bl.transform_brick(b2_base, m2)
            f_hi = max(b1["fx"], b1["fy"], b2["fx"], b2["fy"])
            f_lo = min(b1["fx"], b1["fy"], b2["fx"], b2["fy"])
            f_avg = 0.5 * (max(b1["fx"], b1["fy"]) + max(b2["fx"], b2["fy"]))
            f_list = ((f_hi, "fhi"),) if abs(f_hi - f_lo) < 5e-4 else ((f_lo, "flo"), (f_avg, "favg"), (f_hi, "fhi"))
            sf_list = (0.0, 0.5) if abs(f_hi - f_lo) >= 5e-4 else (0.0,)
            for sf in sf_list:
                for f_use, ftag in f_list:
                    built = build_pinwheel_zero_ov(m, b1, b2, sf=sf, f_override=f_use)
                    if built is None:
                        continue
                    ax, ay, at, _, sm = built
                    pen, _ = cpp_eval_sat_pen_bbox(ax, ay, at)
                    if pen <= 1e-10 and (m1, m2) in (("id", "id"), ("flipx", "flipy")) and ftag == "fhi" and sf == 0.0:
                        cert = gl.bl.polish_and_certify(n, ax, ay, at, inc_dec, slp_rounds=8)
                        if cert is not None:
                            cert["label"] = f"{tag}_{m1}_{m2}_direct"
                            if best_cert is None or cert["L_float"] < best_cert["L_float"]:
                                best_cert = cert
                    if do_melt and (m1, m2) in (("id", "id"), ("flipx", "flipy"), ("id", "flipx"), ("rot180", "rot180")) and sf == 0.0 and ftag in ("favg", "fhi"):
                        seeds.append((ax, ay, at))
                        masks.append(sm)
                        L_tgt = m + f_use
                        scheds.append([L_tgt + 0.008, L_tgt + 0.001, L_tgt - 3e-4])
                        labels.append(f"{tag}_{m1}_{m2}_sf{sf}_{ftag}")

    if do_melt and seeds:
        for siter in (160,):
            for start in range(0, len(seeds), 32):
                end = min(len(seeds), start + 32)
                relaxed = gl.bl.su.cpp_batch_melt(
                    seeds[start:end], scheds[start:end], sym_mode=1, a_start=0.965, n_melt_stages=3,
                    seam_masks=masks[start:end], seam_iters=siter,
                    full_iters=220, n_ruin_iters=0, n_threads=2, base_seed=base_seed + siter + start,
                )
                pool = []
                for bi, row in enumerate(relaxed):
                    nx, ny, nth, Lm = gl.bl.normalize(row["x"], row["y"], row["th"])
                    lbl = f"{labels[start + bi]}_si{siter}"
                    if row["best_L_feas"] is not None:
                        pool.append((0.0, Lm, nx, ny, nth, lbl))
                    elif row["pen"] < 0.015:
                        pool.append((1.0 + float(row["pen"]), Lm, nx, ny, nth, lbl))
                pool.sort(key=lambda t: (t[0], t[1]))
                for _, _, cx, cy, cth, lbl in pool[:2]:
                    cert = gl.bl.polish_and_certify(n, cx, cy, cth, inc_dec, slp_rounds=12)
                    if cert is not None:
                        cert["label"] = lbl
                        if best_cert is None or cert["L_float"] < best_cert["L_float"]:
                            best_cert = cert

    if best_cert is not None:
        rs = sorted(best_cert["rows"], key=lambda r: int(r["square_id"]))
        res_lp = lpslack.optimize_group(
            n, rs, str(inc_dec), outer_iters=5, fixed_rounds=8,
            slp_modes=("dual_non90", "non90"), tr_th_schedule=(4e-4, 1e-4), time_limit=10.0,
        )
        if res_lp.get("win") and res_lp.get("rows"):
            lit, pan = Decimal(res_lp["new_L_lit"]), Decimal(res_lp["new_L_pan"])
            worst = max(lit, pan)
            if worst < max(Decimal(best_cert["L_lit"]), Decimal(best_cert["L_pandas"])):
                best_cert["L_lit"] = str(lit)
                best_cert["L_pandas"] = str(pan)
                best_cert["L_dec_str"] = str(worst)
                best_cert["L_float"] = float(worst)
                best_cert["dL"] = float(worst - inc_dec)
                best_cert["is_real_win"] = bool(worst - inc_dec <= Decimal("-1e-9"))
                best_cert["rows"] = res_lp["rows"]
                best_cert["label"] = best_cert["label"] + "+lpslack"
    return best_cert


def main() -> None:
    t0 = time.time()
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    direct_only = "--direct-only" in sys.argv
    targets = [int(a) for a in args] if args else [974, 973, 913, 912, 853, 854, 797, 975, 914]
    frontier = gl.fast_load_true_frontier(targets, certify=True)
    bricks = load_all_bricks()
    print(f"Loaded {len(bricks)} brick shapes/orientations; targets={targets}", flush=True)

    results = {}
    for n in targets:
        inc_dec = frontier[n]["L_dec"]
        inc_f = float(inc_dec)
        m = int(math.floor(inc_f))
        r = n - m * m
        f_inc = inc_f - m
        # Also include local win if already exists
        wn = gl.DATA_DIR / f"win_n{n}.json"
        if wn.exists():
            wd = json.loads(wn.read_text(encoding="utf-8"))
            inc_dec = min(inc_dec, max(Decimal(wd["L_lit"]), Decimal(wd["L_pandas"])))
            inc_f = float(inc_dec)
            f_inc = inc_f - m

        # Enumerate valid zero-overlap pairs
        pairs = []
        seen_pair_tags = set()
        for i, b1 in enumerate(bricks):
            for j, b2 in enumerate(bricks):
                if i > j:
                    continue
                if b1["g"] + b2["g"] != r:
                    continue
                if b1["w"] + b2["w"] > m or b1["h"] + b2["h"] > m:
                    continue
                if b1["w"] + b2["w"] + b1["h"] + b2["h"] > 2 * m - 1:
                    continue
                fmax = max(b1["f"], b2["f"])
                if fmax >= f_inc + 0.005:
                    continue
                ptag = tuple(sorted([(b1["tag"], b1["w"], b1["h"], b1["g"]), (b2["tag"], b2["w"], b2["h"], b2["g"])]))
                if ptag in seen_pair_tags:
                    continue
                seen_pair_tags.add(ptag)
                pairs.append((fmax, b1, b2))
        pairs.sort(key=lambda t: t[0])

        print(f"\n[n={n} m={m} r=+{r} inc={float(inc_dec):.9f} (f_inc={f_inc:.6f})]: {len(pairs)} candidate pairs", flush=True)
        for fmax, b1, b2 in pairs[:6]:
            print(f"  cand: {b1['tag']}({b1['w']}x{b1['h']},g={b1['g']},f={b1['f']:.5f}) + {b2['tag']}({b2['w']}x{b2['h']},g={b2['g']},f={b2['f']:.5f}) -> fmax={fmax:.6f}", flush=True)

        best_n = None
        # Pass 1: Fast direct pinwheels across top 8 pairs
        for pidx, (fmax, b1, b2) in enumerate(pairs[:8]):
            if fmax >= f_inc - 1e-7:
                continue
            tag = f"n{n}_{b1['tag']}_{b2['tag']}"
            cert = solve_pair_fast(n, m, b1, b2, inc_dec, tag, base_seed=n * 100 + pidx, do_melt=False)
            if cert is not None:
                print(f"  -> direct {cert['label']}: L={cert['L_dec_str'][:20]} dL={cert['dL']:+.6e} win={cert['is_real_win']}", flush=True)
                if cert["is_real_win"]:
                    saved = gl.save_win_locked(cert)
                    print(f"  *** SAVED WIN n={n}: saved={saved} ***", flush=True)
                    inc_dec = min(inc_dec, max(Decimal(cert["L_lit"]), Decimal(cert["L_pandas"])))
                    f_inc = float(inc_dec) - m
                if best_n is None or cert["L_float"] < best_n["L_float"]:
                    best_n = cert

        # Pass 2: Focused seam melt on top 3 pairs that still can improve inc_dec
        if not direct_only:
            for pidx, (fmax, b1, b2) in enumerate(pairs[:3]):
                if fmax >= f_inc + 0.001:
                    continue
                if best_n is not None and best_n["L_float"] <= m + fmax + 2e-6:
                    continue
                tag = f"n{n}_{b1['tag']}_{b2['tag']}"
                cert = solve_pair_fast(n, m, b1, b2, inc_dec, tag, base_seed=n * 100 + pidx, do_melt=True)
                if cert is not None:
                    print(f"  -> melt {cert['label']}: L={cert['L_dec_str'][:20]} dL={cert['dL']:+.6e} win={cert['is_real_win']}", flush=True)
                    if cert["is_real_win"]:
                        saved = gl.save_win_locked(cert)
                        print(f"  *** SAVED WIN n={n}: saved={saved} ***", flush=True)
                        inc_dec = min(inc_dec, max(Decimal(cert["L_lit"]), Decimal(cert["L_pandas"])))
                        f_inc = float(inc_dec) - m
                    if best_n is None or cert["L_float"] < best_n["L_float"]:
                        best_n = cert
        results[str(n)] = {
            "inc": str(frontier[n]["L_dec"]),
            "win": bool(best_n and best_n["is_real_win"]),
            "best_L": best_n["L_dec_str"] if best_n else None,
            "dL": best_n["dL"] if best_n else None,
            "label": best_n["label"] if best_n else None,
        }

    gl.DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (gl.DOCS_DIR / "gate_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nGATE COMPLETE in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
