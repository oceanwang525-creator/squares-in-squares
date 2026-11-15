#!/usr/bin/env python3
"""Hypothesis H77 Scale-Up Runner across {743, 798, 855, 800, 856, 915, 976, 916, 977}.

Phase 1: Harvest g=+6..+9 bricks from origin/main packings, win_n914, gstep/b8xplant caches,
         clean peels, and quick-walk priority unwalked bricks with diagonal-mode (d(i+-j)) SVD + SLP
         using <=2 worker processes (OMP_NUM_THREADS=1).
Phase 2: Assemble zero-overlap pinwheels + direct SLP/lpslack + cpp_batch_melt + dual certification,
         checkpointing every win immediately via b7_lib.save_win_b7.
"""
from __future__ import annotations

import concurrent.futures as cf
import importlib.util as _ilu
import json
import math
import os
import sys
import time
from decimal import Decimal
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np

POC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(POC_DIR))
import b7_lib as b7  # noqa: E402
import b8_lib as bl  # noqa: E402
import brick_rect_lib as bslp  # noqa: E402
import run_h60_pingpong as h60  # noqa: E402

_rg_spec = _ilu.spec_from_file_location("b7_run_gate", POC_DIR / "run_gate.py")
rg = _ilu.module_from_spec(_rg_spec)
_rg_spec.loader.exec_module(rg)

_rgf_spec = _ilu.spec_from_file_location("b7_run_gate_fast", POC_DIR / "run_gate_fast.py")
rgf = _ilu.module_from_spec(_rgf_spec)
_rgf_spec.loader.exec_module(rgf)

# Order targets by best immediate geometric fit & headroom first
SCALE_NS = [855, 976, 800, 915, 916, 977, 856, 743, 798]
DONOR_NS = sorted(set(SCALE_NS) | {741, 742, 796, 797, 854, 914, 972, 974, 975})


def _peel_one_side(bk: dict, side: str, tag: str) -> dict | None:
    w, h, g = bk["w"], bk["h"], bk["g"]
    x, y, th = bk["x"], bk["y"], bk["th"]
    W, H = w + bk["fx"], h + bk["fy"]
    if side == "L":
        m = x >= 1.0 - 1e-3
        nw, nh = w - 1, h
    elif side == "R":
        m = x <= W - 1.0 + 1e-3
        nw, nh = w - 1, h
    elif side == "B":
        m = y >= 1.0 - 1e-3
        nw, nh = w, h - 1
    else:
        m = y <= H - 1.0 + 1e-3
        nw, nh = w, h - 1
    if nw < 7 or nh < 7 or int(m.sum()) - nw * nh != g:
        return None
    return b7.make_brick(nw, nh, g, x[m], y[m], th[m], src_n=bk.get("src_n", 0), tag=tag)


def _walk_task(args: tuple[str, dict]) -> dict:
    name, bk = args
    p = b7.DATA_DIR / f"brick_{name}.json"
    if p.exists():
        d = json.loads(p.read_text(encoding="utf-8"))
        return {
            "name": name, "tag": name, "w": int(d["w"]), "h": int(d["h"]), "g": int(d["g"]),
            "src_n": int(d.get("src_n", 0)), "f0": float(d.get("f0", d["f"])),
            "fx": float(d["fx"]), "fy": float(d["fy"]), "f": float(d["f"]), "df": float(d.get("df", 0.0)),
            "x": np.array(d["x"], dtype=np.float64), "y": np.array(d["y"], dtype=np.float64),
            "th": np.array(d["th"], dtype=np.float64),
        }
    return rgf.quick_walk_b8(bk, name)


def build_priority_bricks(portfolio: dict[int, dict]) -> list[dict]:
    bricks: dict[str, dict] = {}

    def _put(name: str, bk: dict | None) -> None:
        if bk is None or bk["f"] >= 0.785:
            return
        if name not in bricks or bk["f"] < bricks[name]["f"] - 1e-10:
            bricks[name] = dict(bk, name=name, tag=name, f0=bk.get("f0", bk["f"]), df=bk.get("df", 0.0))

    # 1. Load already-walked bricks in poc/swarm_b7xplant/data/brick_*.json
    for p in sorted(b7.DATA_DIR.glob("brick_*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        nm = str(d["name"])
        _put(nm, {
            "name": nm, "tag": nm, "w": int(d["w"]), "h": int(d["h"]), "g": int(d["g"]),
            "src_n": int(d.get("src_n", 0)), "f0": float(d.get("f0", d["f"])),
            "fx": float(d["fx"]), "fy": float(d["fy"]), "f": float(d["f"]), "df": float(d.get("df", 0.0)),
            "x": np.array(d["x"], dtype=np.float64), "y": np.array(d["y"], dtype=np.float64),
            "th": np.array(d["th"], dtype=np.float64),
        })

    # 2. Load g=+6..+9 bricks from swarm_gstep & swarm_b8xplant
    for folder in (b7.REPO_ROOT / "poc" / "swarm_gstep" / "data", b7.REPO_ROOT / "poc" / "swarm_b8xplant" / "data"):
        if not folder.exists():
            continue
        for p in sorted(folder.glob("brick_*.json")):
            d = json.loads(p.read_text(encoding="utf-8"))
            if "w" in d and "g" in d and 6 <= int(d["g"]) <= 9:
                nm = f"ext_{p.stem}"
                bk = b7.make_brick(
                    int(d["w"]), int(d["h"]), int(d["g"]),
                    np.array(d["x"], dtype=np.float64), np.array(d["y"], dtype=np.float64),
                    np.array(d["th"], dtype=np.float64), src_n=int(d.get("src_n", 0)), tag=nm,
                )
                _put(nm, bk)

    # 3. Extract win_n914 bricks
    w914_p = b7.DATA_DIR / "win_n914.json"
    if w914_p.exists():
        w914 = json.loads(w914_p.read_text(encoding="utf-8"))
        rs = sorted(w914["rows"], key=lambda r: int(r["square_id"]))
        x9, y9, t9, L9 = bl.normalize([float(r["x"]) for r in rs], [float(r["y"]) for r in rs], [float(r["angle"]) for r in rs])
        _put("B8_12x20_win914", b7.extract_corner_cut(x9, y9, t9, L9, "TL", 12, 20, 8, 914, "B8_12x20_win914"))
        _put("B6_18x9_win914", b7.extract_corner_cut(x9, y9, t9, L9, "BR", 18, 9, 6, 914, "B6_18x9_win914"))

    # 4. Corner cuts for all priority bricks + peels
    cuts = [
        (855, "BR", 20, 11, 8, "B8_20x11_n855"),
        (800, "TL", 18, 9, 8, "B8_18x9_n800"),
        (800, "BR", 20, 20, 8, "B8_20x20_n800"),
        (916, "TL", 8, 19, 7, "B7_8x19_n916"),
        (916, "TL", 9, 19, 7, "B7_9x19_n916"),
        (916, "BR", 21, 11, 9, "B9_21x11_n916"),
        (915, "BR", 22, 12, 9, "B9_22x12_n915"),
        (915, "TL", 7, 18, 6, "B6_7x18_n915"),
        (977, "TL", 12, 21, 9, "B9_12x21_n977"),
        (977, "BR", 20, 9, 7, "B7_20x9_n977"),
        (798, "TL", 13, 21, 9, "B9_13x21_n798"),
        (798, "TL", 19, 20, 8, "B8_19x20_n798"),
        (743, "TL", 18, 20, 8, "B8_18x20_n743"),
        (976, "TL", 11, 20, 8, "B8_11x20_n976"),
        (976, "BR", 18, 21, 6, "B6_18x21_n976"),
        (976, "BR", 21, 21, 7, "B7_21x21_n976"),
    ]
    for sn, corner, w, h, g, name in cuts:
        if sn in portfolio:
            p = portfolio[sn]
            bk = b7.extract_corner_cut(p["x"], p["y"], p["th"], p["L"], corner, w, h, g, sn, name)
            _put(name, bk)
            if bk is not None and name in ("B8_19x20_n798", "B8_18x20_n743"):
                p1 = _peel_one_side(bk, "R", f"{name}_pR")
                _put(f"{name}_pR", p1)
                if p1 is not None:
                    p2 = _peel_one_side(p1, "R", f"{name}_pRR")
                    _put(f"{name}_pRR", p2)

    # Also harvest everything else via run_gate.harvest_all_bricks
    for name, bk in rg.harvest_all_bricks(portfolio):
        _put(name, bk)

    # Priority list of bricks to ensure walked with d(i+-j) SVD + SLP
    walk_names = [
        "B8_20x11_n855",
        "B8_18x9_n800",
        "B8_10x18_NEW",
        "B7_17x8_b8",
        "B7_8x19_n916",
        "B7_9x19_n916",
        "B9_21x11_n916",
        "B9_22x12_n915",
        "B6_7x18_n915",
        "B9_12x21_n977",
        "B7_20x9_n977",
        "B9_13x21_n798",
        "ext_brick_B8_10x23_k4",
        "ext_brick_B8_10x22_from23_top",
        "B8_18x8_b8",
        "B8_19x20_n798_pRR",
        "B8_11x20_n976",
        "B8_12x20_win914",
        "ext_brick_B7_10x21_vs_k5",
        "ext_brick_B7_9x22_vs_k3",
    ]
    todo = [(nm, bricks[nm]) for nm in walk_names if nm in bricks and not (b7.DATA_DIR / f"brick_{nm}.json").exists()]
    if todo:
        print(f"[H77 SCALE] Quick-walking {len(todo)} priority bricks with max_workers=2...", flush=True)
        with cf.ProcessPoolExecutor(max_workers=2) as pool:
            for res in pool.map(_walk_task, todo):
                bricks[res["name"]] = res
                print(
                    f"  [WALKED {res['name']}] {res['w']}x{res['h']} (+{res['g']}): {res['f0']:.6f} -> {res['f']:.6f} (df={res['df']:+.3e})",
                    flush=True,
                )
    return list(bricks.values())


def solve_pair_scale(
    n: int, m: int, b1_base: dict, b2_base: dict, inc_dec: Decimal, tag: str, base_seed: int
) -> dict | None:
    best_cert: dict | None = None
    seeds, masks, scheds, labels = [], [], [], []
    f_inc = float(inc_dec) - m

    for m1 in ("id", "flipx", "flipy", "rot180"):
        b1 = bl.transform_brick(b1_base, m1)
        for m2 in ("id", "flipx", "flipy", "rot180"):
            b2 = bl.transform_brick(b2_base, m2)
            f_hi = max(b1["fx"], b1["fy"], b2["fx"], b2["fy"])
            f_avg = 0.5 * (max(b1["fx"], b1["fy"]) + max(b2["fx"], b2["fy"]))
            for sf in (0.0, 0.5):
                for f_use, ftag in ((f_hi, "fhi"), (f_avg, "favg")):
                    if f_use > f_inc + 0.015:
                        continue
                    ax, ay, at, L0, sm = h60.build_pinwheel_custom(m, b1, b2, sf=sf, f_override=f_use)
                    if len(ax) != n:
                        continue
                    pen, _ = bl.cpp_eval_sat_pen_bbox(ax, ay, at)
                    if pen <= 1e-10 and ftag == "fhi" and sf == 0.0 and (m1, m2) in (
                        ("id", "id"), ("flipx", "flipy"), ("flipx", "id"), ("id", "flipy"), ("rot180", "rot180")
                    ):
                        cert = b7.polish_and_certify_b7(n, ax, ay, at, inc_dec, slp_rounds=20, run_lpslack=True)
                        if cert is not None:
                            cert["label"] = f"{tag}_{m1}_{m2}_direct"
                            if best_cert is None or Decimal(cert["L_dec_str"]) < Decimal(best_cert["L_dec_str"]):
                                best_cert = cert
                            if cert["is_real_win"]:
                                inc_dec = min(inc_dec, Decimal(cert["L_dec_str"]))
                                f_inc = float(inc_dec) - m
                    if (m1, m2) in (("id", "id"), ("flipx", "id"), ("id", "flipx"), ("rot180", "rot180")):
                        seeds.append((ax, ay, at))
                        masks.append(sm)
                        L_tgt = m + min(f_use, f_inc)
                        scheds.append([L_tgt + 0.008, L_tgt + 0.0015, L_tgt - 2e-4])
                        labels.append(f"{tag}_{m1}_{m2}_sf{sf}_{ftag}")

    if not seeds:
        return best_cert

    for siter in (180, 0):
        relaxed = bl.su.cpp_batch_melt(
            seeds[:32], scheds[:32], sym_mode=1, a_start=0.965, n_melt_stages=3,
            seam_masks=masks[:32] if siter > 0 else None, seam_iters=siter,
            full_iters=280, n_ruin_iters=0, n_threads=2, base_seed=base_seed + siter,
        )
        pool = []
        for bi, row in enumerate(relaxed):
            nx, ny, nth, Lm = bl.normalize(row["x"], row["y"], row["th"])
            lbl = f"{labels[bi]}_si{siter}"
            if row["best_L_feas"] is not None:
                pool.append((0.0, Lm, nx, ny, nth, lbl))
            elif row["pen"] < 0.015:
                pool.append((1.0 + float(row["pen"]), Lm, nx, ny, nth, lbl))
        pool.sort(key=lambda t: (t[0], t[1]))
        for _, _, cx, cy, cth, lbl in pool[:3]:
            cert = b7.polish_and_certify_b7(n, cx, cy, cth, inc_dec, slp_rounds=20, run_lpslack=True)
            if cert is not None:
                cert["label"] = lbl
                if best_cert is None or Decimal(cert["L_dec_str"]) < Decimal(best_cert["L_dec_str"]):
                    best_cert = cert
                if cert["is_real_win"]:
                    inc_dec = min(inc_dec, Decimal(cert["L_dec_str"]))
                    f_inc = float(inc_dec) - m
    return best_cert


def main() -> None:
    t0 = time.time()
    print("[H77 SCALE] Loading origin/main portfolio...", flush=True)
    portfolio = b7.load_portfolio_targets(DONOR_NS)
    base_bricks = build_priority_bricks(portfolio)
    all_bricks = rg.expand_brick_orientations(base_bricks)
    print(f"[H77 SCALE] {len(base_bricks)} base bricks -> {len(all_bricks)} oriented bricks ready ({time.time()-t0:.1f}s)", flush=True)

    report = {}
    for n in SCALE_NS:
        m = int(math.isqrt(n))
        r = n - m * m
        inc_dec = portfolio[n]["L_dec"]
        f_inc = float(inc_dec) - m
        print(f"\n[H77 SCALE] n={n} (m={m}, r=+{r}) inc={str(inc_dec)[:20]} f_inc={f_inc:.9f}", flush=True)

        pairs = []
        seen_dims = set()
        for b1 in all_bricks:
            if b1["w"] >= m or b1["h"] >= m:
                continue
            for b2 in all_bricks:
                if b1["g"] + b2["g"] != r or b2["w"] >= m or b2["h"] >= m:
                    continue
                w1, h1, w2, h2 = b1["w"], b1["h"], b2["w"], b2["h"]
                if not ((w1 + w2 <= m - 1 and h1 + h2 <= m) or (h1 + h2 <= m - 1 and w1 + w2 <= m)):
                    continue
                f_hi = max(b1["f"], b2["f"])
                f_lo = min(b1["f"], b2["f"])
                if f_hi > f_inc + 0.014 and f_lo > f_inc:
                    continue
                key = (b1["name"], b2["name"])
                if key in seen_dims:
                    continue
                seen_dims.add(key)
                sc = 0.65 * f_hi + 0.35 * f_lo - (0.002 if (b1.get("df", 0.0) < -1e-6 or b2.get("df", 0.0) < -1e-6) else 0.0)
                pairs.append((sc, b1, b2))

        pairs.sort(key=lambda t: t[0])
        filtered = []
        seen_geom = set()
        for sc, b1, b2 in pairs:
            gk = (b1["w"], b1["h"], b1["g"], b2["w"], b2["h"], b2["g"], round(b1["f"], 6), round(b2["f"], 6))
            if gk in seen_geom:
                continue
            seen_geom.add(gk)
            filtered.append((sc, b1, b2))
            if len(filtered) >= 6:
                break

        best = None
        cur_inc = inc_dec
        for idx, (_, b1, b2) in enumerate(filtered):
            print(
                f"  Pair {idx+1}/{len(filtered)}: {b1['name']}({b1['w']}x{b1['h']},+{b1['g']},f={b1['f']:.6f}) + {b2['name']}({b2['w']}x{b2['h']},+{b2['g']},f={b2['f']:.6f})",
                flush=True,
            )
            c = solve_pair_scale(n, m, b1, b2, cur_inc, f"n{n}_{b1['name']}+{b2['name']}", base_seed=88000 + n * 17 + idx * 53)
            if c is not None:
                print(f"    -> {c['label']}: L={c['L_dec_str'][:20]} dL={c['dL']:+.6e} win={c['is_real_win']}", flush=True)
                if best is None or Decimal(c["L_dec_str"]) < Decimal(best["L_dec_str"]):
                    best = c
                if c["is_real_win"]:
                    b7.save_win_b7(c)
                    cur_inc = min(cur_inc, Decimal(c["L_dec_str"]))

        report[str(n)] = {
            "inc": str(portfolio[n]["L_dec"]),
            "win": bool(best and best["is_real_win"]),
            "L": best["L_dec_str"] if best else None,
            "dL": best["dL"] if best else None,
            "label": best.get("label") if best else None,
        }
        (b7.DOCS_DIR / "scale_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n[H77 SCALE COMPLETE] in {time.time()-t0:.1f}s -> {json.dumps(report, indent=2)}", flush=True)


if __name__ == "__main__":
    main()
