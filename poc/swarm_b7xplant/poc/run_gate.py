#!/usr/bin/env python3
"""Hypothesis H77 Gate Runner: Diagonal-mode brick_mechwalk on g=+6,+7,+8 bricks + zero-overlap pinwheels on {854, 914, 975}."""
from __future__ import annotations

import json
import math
import multiprocessing as mp
import sys
import time
from decimal import Decimal
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import b7_lib as b7  # noqa: E402
import b8_lib as bl  # noqa: E402
import brick_rect_lib as bslp  # noqa: E402
import run_h60_pingpong as h60  # noqa: E402


GATE_NS = [854, 914, 975]
DONOR_NS = [741, 742, 743, 796, 797, 798, 800, 854, 855, 856, 914, 915, 916, 972, 974, 975, 976, 977]


def _load_brick_file(path: Path, name: str, src_n: int = 0) -> dict | None:
    if not path.exists():
        return None
    d = json.loads(path.read_text(encoding="utf-8"))
    return b7.make_brick(
        int(d["w"]),
        int(d["h"]),
        int(d["g"]),
        np.array(d["x"], dtype=np.float64),
        np.array(d["y"], dtype=np.float64),
        np.array(d["th"], dtype=np.float64),
        src_n=int(d.get("src_n", src_n)),
        tag=name,
    )


def harvest_all_bricks(portfolio: dict[int, dict]) -> list[tuple[str, dict]]:
    """Harvest g=+5..+8 bricks from caches, b8xplant data, and origin/main packings."""
    raw: dict[str, dict] = {}

    def _add(name: str, bk: dict | None) -> None:
        if bk is None:
            return
        if name not in raw or bk["f"] < raw[name]["f"] - 1e-10:
            bk = dict(bk)
            bk["tag"] = name
            raw[name] = bk

    # 1. Existing b8xplant data bricks
    b8_data = b7.REPO_ROOT / "poc" / "swarm_b8xplant" / "data"
    for fname, name in (
        ("brick_B6_10x21_n972.json", "B6_10x21_n972_b8"),
        ("brick_B6_24x8.json", "B6_24x8_b8"),
        ("brick_B7_17x8.json", "B7_17x8_b8"),
        ("brick_B8_10x18_NEW.json", "B8_10x18_NEW"),
        ("brick_B8_18x8.json", "B8_18x8_b8"),
        ("brick_H60_B5_20x11_n970s6.json", "B5_20x11_n970s6"),
        ("brick_B5_20x10_n972.json", "B5_20x10_n972"),
        ("brick_B5_22x8.json", "B5_22x8_b8"),
    ):
        _add(name, _load_brick_file(b8_data / fname, name))

    # 2. b5xplant h63 & h53 caches
    b5_data = b7.REPO_ROOT / "poc" / "swarm_b5xplant" / "data"
    for cname in ("h63_compressed_cache.json", "h53_compressed_cache.json"):
        cp = b5_data / cname
        if not cp.exists():
            continue
        for item in json.loads(cp.read_text(encoding="utf-8")):
            if int(item["g"]) >= 6:
                bk = b7.make_brick(
                    int(item["w"]),
                    int(item["h"]),
                    int(item["g"]),
                    np.array(item["x"], dtype=np.float64),
                    np.array(item["y"], dtype=np.float64),
                    np.array(item["th"], dtype=np.float64),
                    src_n=int(item.get("src_n", 0)),
                    tag=str(item["tag"]),
                )
                _add(str(item["tag"]), bk)

    # 3. Direct corner cuts from origin/main portfolio packings
    cuts = [
        # n=854
        (854, "TL", 18, 11, 6, "B6_18x11_n854_cut"),
        (854, "TL", 19, 11, 7, "B7_19x11_n854_cut"),
        (854, "BR", 9, 18, 6, "B6_9x18_n854_cut"),
        # n=914
        (914, "TL", 18, 9, 6, "B6_18x9_n914_cut"),
        (914, "BR", 12, 20, 8, "B8_12x20_n914_cut"),
        (914, "BR", 12, 19, 7, "B7_12x19_n914_cut"),
        (914, "BR", 12, 18, 6, "B6_12x18_n914_cut"),
        # n=975
        (975, "TL", 11, 19, 7, "B7_11x19_n975_cut"),
        (975, "TL", 11, 18, 6, "B6_11x18_n975_cut"),
        (975, "BR", 20, 11, 7, "B7_20x11_n975_cut"),
        (975, "BR", 19, 11, 6, "B6_19x11_n975_cut"),
        # n=796
        (796, "TL", 9, 18, 6, "B6_9x18_n796_cut"),
        # n=855
        (855, "TL", 8, 18, 6, "B6_8x18_n855_cut"),
        (855, "BR", 20, 11, 8, "B8_20x11_n855_cut"),
        (855, "BR", 19, 11, 7, "B7_19x11_n855_cut"),
        (855, "BR", 18, 11, 6, "B6_18x11_n855_cut"),
        # n=976
        (976, "TL", 11, 20, 8, "B8_11x20_n976_cut"),
        (976, "TL", 11, 19, 7, "B7_11x19_n976_cut"),
        (976, "BR", 19, 11, 7, "B7_19x11_n976_cut"),
        # n=916
        (916, "TL", 8, 19, 7, "B7_8x19_n916_cut"),
        (916, "BR", 20, 11, 8, "B8_20x11_n916_cut"),
        (916, "BR", 19, 11, 7, "B7_19x11_n916_cut"),
        # n=977
        (977, "BR", 20, 9, 7, "B7_20x9_n977_cut"),
        (977, "BR", 19, 9, 6, "B6_19x9_n977_cut"),
        (977, "TL", 11, 20, 8, "B8_11x20_n977_cut"),
        # n=800
        (800, "TR", 21, 21, 8, "B8_21x21_n800_cut"),
        (800, "TL", 18, 8, 8, "B8_18x8_n800_cut"),
        (800, "BR", 9, 18, 8, "B8_9x18_n800_cut"),
        # n=743 & n=798 & n=915
        (743, "BR", 20, 11, 8, "B8_20x11_n743_cut"),
        (743, "BR", 19, 11, 7, "B7_19x11_n743_cut"),
        (798, "BR", 20, 12, 8, "B8_20x12_n798_cut"),
        (798, "BR", 19, 12, 7, "B7_19x12_n798_cut"),
        (915, "BR", 21, 12, 8, "B8_21x12_n915_cut"),
        (915, "BR", 20, 12, 7, "B7_20x12_n915_cut"),
    ]
    for sn, corner, w, h, g, tag in cuts:
        if sn not in portfolio:
            continue
        p = portfolio[sn]
        bk = b7.extract_corner_cut(p["x"], p["y"], p["th"], p["L"], corner, w, h, g, sn, tag)
        _add(tag, bk)

    # Deduplicate bricks that only differ by name if identical geometry & keep best per (w, h, g, rounded_f)
    return sorted(raw.items(), key=lambda kv: (kv[1]["g"], kv[1]["f"], kv[0]))


def _walk_worker(args: tuple[str, dict]) -> dict:
    name, bk = args
    p = b7.DATA_DIR / f"brick_{name}.json"
    if p.exists():
        d = json.loads(p.read_text(encoding="utf-8"))
        return {
            "name": name,
            "tag": name,
            "w": int(d["w"]),
            "h": int(d["h"]),
            "g": int(d["g"]),
            "src_n": int(d.get("src_n", 0)),
            "f0": float(d.get("f0", d["f"])),
            "fx": float(d["fx"]),
            "fy": float(d["fy"]),
            "f": float(d["f"]),
            "df": float(d.get("df", 0.0)),
            "x": np.array(d["x"], dtype=np.float64),
            "y": np.array(d["y"], dtype=np.float64),
            "th": np.array(d["th"], dtype=np.float64),
        }
    return b7.standalone_brick_walk_diag(bk, name, rounds=3, do_slp_pre=True)


def run_brick_walks(harvested: list[tuple[str, dict]], workers: int = 16) -> list[dict]:
    """Run standalone diagonal-mode brick_mechwalk in parallel on all harvested g>=6 bricks (keep g=5 too)."""
    b7.DATA_DIR.mkdir(parents=True, exist_ok=True)
    to_walk = [(name, bk) for name, bk in harvested if bk["g"] >= 6]
    keep_g5 = [dict(bk, name=name, f0=bk["f"], df=0.0) for name, bk in harvested if bk["g"] == 5]
    print(f"[H77] Walking {len(to_walk)} g>=6 bricks across {workers} workers...", flush=True)
    with mp.get_context("fork").Pool(min(workers, len(to_walk))) as pool:
        walked = pool.map(_walk_worker, to_walk)
    print("\n=== Brick Walk Summary ===", flush=True)
    for r in sorted(walked, key=lambda d: (d["g"], d["f"])):
        print(
            f"  {r['name']:26s} ({r['w']:2d}x{r['h']:2d}, +{r['g']}) "
            f"f0={r['f0']:.9f} -> f={r['f']:.9f} (df={r['df']:+.6e})",
            flush=True,
        )
    return walked + keep_g5


def expand_brick_orientations(bricks: list[dict]) -> list[dict]:
    """Include both (w,h) and transposed (h,w) forms, keeping Pareto-best per (w,h,g,tag)."""
    out = []
    for bk in bricks:
        out.append(bk)
        bt = bl.transform_brick(bk, "T")
        bt["name"] = bk["name"] + "_T"
        bt["tag"] = bt["name"]
        bt["f0"] = bk.get("f0", bk["f"])
        bt["df"] = bk.get("df", 0.0)
        out.append(bt)
    return out


def solve_zero_overlap_pair(
    n: int,
    m: int,
    b1_base: dict,
    b2_base: dict,
    inc_dec: Decimal,
    tag: str,
    base_seed: int,
) -> dict | None:
    """Assemble zero-overlap pinwheel, run direct tight SLP + batch melt + lpslack + dual cert."""
    best_cert: dict | None = None
    seeds, masks, scheds, labels = [], [], [], []
    f_inc = float(inc_dec) - m

    for m1 in ("id", "flipx", "flipy", "rot180"):
        b1 = bl.transform_brick(b1_base, m1)
        for m2 in ("id", "flipx", "flipy", "rot180"):
            b2 = bl.transform_brick(b2_base, m2)
            f_hi = max(b1["fx"], b1["fy"], b2["fx"], b2["fy"])
            f_avg = 0.5 * (max(b1["fx"], b1["fy"]) + max(b2["fx"], b2["fy"]))
            f_lo = min(max(b1["fx"], b1["fy"]), max(b2["fx"], b2["fy"]))
            for sf in (0.0, 0.5, 0.8):
                for f_use, ftag in ((f_hi, "fhi"), (f_avg, "favg"), (f_lo, "flo")):
                    if f_use > f_inc + 0.015:
                        continue
                    ax, ay, at, L0, sm = h60.build_pinwheel_custom(m, b1, b2, sf=sf, f_override=f_use)
                    if len(ax) != n:
                        continue
                    pen, _ = bl.cpp_eval_sat_pen_bbox(ax, ay, at)
                    if pen <= 1e-10 and ftag == "fhi" and sf in (0.0, 0.5):
                        if (m1, m2) in (("id", "id"), ("flipx", "flipy"), ("rot180", "rot180"), ("flipx", "id"), ("id", "flipx")):
                            cert = b7.polish_and_certify_b7(n, ax, ay, at, inc_dec, slp_rounds=22, run_lpslack=True)
                            if cert is not None:
                                cert["label"] = f"{tag}_{m1}_{m2}_sf{sf}_direct"
                                if best_cert is None or cert["L_float"] < best_cert["L_float"]:
                                    best_cert = cert
                    if (m1, m2) in (("id", "id"), ("flipx", "id"), ("id", "flipx"), ("rot180", "rot180"), ("flipy", "flipy")):
                        seeds.append((ax, ay, at))
                        masks.append(sm)
                        L_tgt = m + min(f_use, f_inc)
                        scheds.append([L_tgt + 0.008, L_tgt + 0.0015, L_tgt - 2e-4])
                        labels.append(f"{tag}_{m1}_{m2}_sf{sf}_{ftag}")

    if not seeds:
        return best_cert

    for siter in (160, 0):
        for start in range(0, len(seeds), 64):
            end = min(len(seeds), start + 64)
            relaxed = bl.su.cpp_batch_melt(
                seeds[start:end],
                scheds[start:end],
                sym_mode=1,
                a_start=0.965,
                n_melt_stages=3,
                seam_masks=masks[start:end] if siter > 0 else None,
                seam_iters=siter,
                full_iters=260,
                n_ruin_iters=0,
                n_threads=4,
                base_seed=base_seed + siter + start,
            )
            pool = []
            for bi, row in enumerate(relaxed):
                nx, ny, nth, Lm = bl.normalize(row["x"], row["y"], row["th"])
                lbl = f"{labels[start + bi]}_si{siter}"
                if row["best_L_feas"] is not None:
                    pool.append((0.0, Lm, nx, ny, nth, lbl))
                elif row["pen"] < 0.015:
                    pool.append((1.0 + float(row["pen"]), Lm, nx, ny, nth, lbl))
            pool.sort(key=lambda t: (t[0], t[1]))
            for _, _, cx, cy, cth, lbl in pool[:4]:
                cert = b7.polish_and_certify_b7(n, cx, cy, cth, inc_dec, slp_rounds=22, run_lpslack=True)
                if cert is not None:
                    cert["label"] = lbl
                    if best_cert is None or cert["L_float"] < best_cert["L_float"]:
                        best_cert = cert
                        if best_cert["is_real_win"]:
                            inc_dec = min(inc_dec, Decimal(best_cert["L_dec_str"]))
    return best_cert


def solve_target_n(n: int, all_bricks: list[dict], inc_dec: Decimal, max_pairs: int = 10) -> dict | None:
    """Find top zero-overlap brick pairs for (m, r) and solve them."""
    m = int(math.isqrt(n))
    r = n - m * m
    f_inc = float(inc_dec) - m
    print(f"\n[H77 TARGET] n={n} (m={m}, r=+{r}) inc={str(inc_dec)[:20]} f_inc={f_inc:.9f}", flush=True)

    pairs: list[tuple[float, dict, dict]] = []
    seen_dims: set[tuple] = set()
    for b1 in all_bricks:
        if b1["w"] >= m or b1["h"] >= m:
            continue
        for b2 in all_bricks:
            if b1["g"] + b2["g"] != r:
                continue
            if b2["w"] >= m or b2["h"] >= m:
                continue
            w1, h1, w2, h2 = b1["w"], b1["h"], b2["w"], b2["h"]
            if not ((w1 + w2 <= m - 1 and h1 + h2 <= m) or (h1 + h2 <= m - 1 and w1 + w2 <= m)):
                continue
            f_hi = max(b1["f"], b2["f"])
            f_lo = min(b1["f"], b2["f"])
            if f_hi > f_inc + 0.015 and f_lo > f_inc:
                continue
            key = (b1["name"], b2["name"])
            if key in seen_dims:
                continue
            seen_dims.add(key)
            score = 0.65 * f_hi + 0.35 * f_lo - (0.002 if (b1.get("df", 0.0) < -1e-6 or b2.get("df", 0.0) < -1e-6) else 0.0)
            pairs.append((score, b1, b2))

    pairs.sort(key=lambda t: t[0])
    # Deduplicate symmetric/duplicate (w1,h1,g1,w2,h2,g2,round(f1,6),round(f2,6))
    filtered: list[tuple[float, dict, dict]] = []
    seen_geom: set[tuple] = set()
    for sc, b1, b2 in pairs:
        gk = (b1["w"], b1["h"], b1["g"], b2["w"], b2["h"], b2["g"], round(b1["f"], 6), round(b2["f"], 6))
        if gk in seen_geom:
            continue
        seen_geom.add(gk)
        filtered.append((sc, b1, b2))
        if len(filtered) >= max_pairs:
            break

    print(f"  Selected {len(filtered)} zero-overlap pairs:", flush=True)
    for _, b1, b2 in filtered:
        print(
            f"    {b1['name']}({b1['w']}x{b1['h']},+{b1['g']},f={b1['f']:.6f}) + "
            f"{b2['name']}({b2['w']}x{b2['h']},+{b2['g']},f={b2['f']:.6f})",
            flush=True,
        )

    best: dict | None = None
    cur_inc = inc_dec
    for idx, (_, b1, b2) in enumerate(filtered):
        tag = f"n{n}_{b1['name']}+{b2['name']}"
        c = solve_zero_overlap_pair(n, m, b1, b2, cur_inc, tag, base_seed=77000 + n * 13 + idx * 50)
        if c is not None:
            print(f"  [CAND {idx+1}] {c['label']}: L={c['L_dec_str'][:20]} dL={c['dL']:+.6e} win={c['is_real_win']}", flush=True)
            if best is None or Decimal(c["L_dec_str"]) < Decimal(best["L_dec_str"]):
                best = c
            if c["is_real_win"]:
                saved = b7.save_win_b7(c)
                print(f"  *** REAL WIN n={n}: L={c['L_dec_str'][:22]} dL={c['dL']:+.6e} saved={saved} ***", flush=True)
                cur_inc = min(cur_inc, Decimal(c["L_dec_str"]))
    return best


def main() -> None:
    t0 = time.time()
    b7.DATA_DIR.mkdir(parents=True, exist_ok=True)
    b7.DOCS_DIR.mkdir(parents=True, exist_ok=True)
    print("[H77] Loading origin/main portfolio...", flush=True)
    portfolio = b7.load_portfolio_targets(DONOR_NS)
    harvested = harvest_all_bricks(portfolio)
    print(f"[H77] Harvested {len(harvested)} bricks.", flush=True)
    walked = run_brick_walks(harvested, workers=16)
    all_bricks = expand_brick_orientations(walked)

    results = {}
    for n in GATE_NS:
        best = solve_target_n(n, all_bricks, portfolio[n]["L_dec"], max_pairs=10)
        results[str(n)] = {
            "inc": str(portfolio[n]["L_dec"]),
            "win": bool(best and best["is_real_win"]),
            "L": best["L_dec_str"] if best else None,
            "dL": best["dL"] if best else None,
            "label": best.get("label") if best else None,
        }

    (b7.DOCS_DIR / "gate_report.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    if not any(b7.DATA_DIR.glob("win_n*.json")):
        b7.write_header_only_final_csv()
    print(f"\n[H77 GATE COMPLETE] in {time.time()-t0:.1f}s -> {results}", flush=True)


if __name__ == "__main__":
    main()
