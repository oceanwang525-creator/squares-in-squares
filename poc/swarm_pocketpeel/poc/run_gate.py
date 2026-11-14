#!/usr/bin/env python3
"""Step 3: 2-Pocket Staircase/Pinwheel DP + Zero-Overlap & Seam-Melt + lpslack + mechwalk."""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import argparse
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

import pocketpeel_lib as pp  # noqa: E402
import b8_lib as bl  # noqa: E402
import brick_rect_lib as bslp  # noqa: E402
from cpp_wrap import cpp_eval_sat_pen_bbox  # noqa: E402

GATE_TARGETS = [795, 740, 853, 913, 854, 914]


def load_all_bricks(include_external: bool = True) -> list[dict]:
    """Load all pocketpeel bricks + optional gstep/cleanpeel2 bricks, deduplicated (top-2 per oriented w,h,g)."""
    raw_list: list[dict] = []

    def _add(b: dict, src_group: str) -> None:
        for is_t, cand in ((False, b), (True, bl.transform_brick(b, "T"))):
            t = cand.get("tag") or b.get("tag") or f"b_{cand['w']}x{cand['h']}"
            if is_t:
                t = t + "_T"
            cand = dict(cand, tag=t, src_group=src_group)
            raw_list.append(cand)

    for p in sorted(pp.BRICKS_DIR.glob("brick_*.json")):
        _add(pp.load_brick(p), "pocketpeel")

    if include_external:
        for ext_dir, sg in (
            (pp.REPO_ROOT / "poc" / "swarm_gstep" / "data", "gstep"),
            (pp.REPO_ROOT / "poc" / "swarm_cleanpeel2" / "data", "cleanpeel2"),
        ):
            if ext_dir.exists():
                for p in sorted(ext_dir.glob("brick_*.json")):
                    _add(pp.load_brick(p), sg)

    # Keep top-2 distinct bricks per oriented (w, h, g) sorted by f
    by_geom: dict[tuple[int, int, int], list[dict]] = {}
    for b in raw_list:
        k = (b["w"], b["h"], b["g"])
        by_geom.setdefault(k, []).append(b)

    out: list[dict] = []
    for k, lst in sorted(by_geom.items()):
        lst.sort(key=lambda b: (b["f"], 0 if b["src_group"] == "pocketpeel" else 1))
        seen_f: list[float] = []
        for b in lst:
            if any(abs(b["f"] - sf) < 1e-6 for sf in seen_f):
                continue
            seen_f.append(b["f"])
            out.append(b)
            if len(seen_f) >= 2:
                break
    return out


def pick_pairs_for_n(
    n: int,
    bricks: list[dict],
    f_inc: float,
    top_k: int = 8,
    allow_kx1: bool = True,
    slack_max: float = 0.025,
) -> list[tuple[float, int, int, dict, dict]]:
    """Select top 2-pocket pairs for n=(m, D) across zero-overlap, kx=ky=0, and 1-unit seam overlap."""
    m = int(math.isqrt(n))
    r = n - m * m
    cands: list[tuple[float, int, int, dict, dict]] = []
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
            if kx > 1 or ky > 1 or kx + ky > 1:
                continue
            if not allow_kx1 and (kx > 0 or ky > 0):
                continue
            f_hi = max(b1["f"], b2["f"])
            f_lo = min(b1["f"], b2["f"])
            if f_hi > f_inc + slack_max and f_lo > f_inc + 0.005:
                continue

            t1 = b1["tag"][:-2] if b1["tag"].endswith("_T") else b1["tag"]
            t2 = b2["tag"][:-2] if b2["tag"].endswith("_T") else b2["tag"]
            is_t1 = b1["tag"].endswith("_T")
            is_t2 = b2["tag"].endswith("_T")
            canon = min(
                tuple(sorted([(t1, is_t1), (t2, is_t2)])),
                tuple(sorted([(t1, not is_t1), (t2, not is_t2)])),
            )
            if canon in seen_sym:
                continue
            seen_sym.add(canon)

            # Penalty for overlap geometry vs zero-overlap
            geom_pen = 0.0
            if kx == 0 and ky == 0:
                geom_pen = 0.0005
            elif kx > 0 or ky > 0:
                geom_pen = 0.0120
            elif kx + ky < 0:
                geom_pen = -0.0010
            sc = f_hi + 0.05 * f_lo + geom_pen
            cands.append((sc, kx, ky, b1, b2))

    cands.sort(key=lambda z: z[0])
    out: list[tuple[float, int, int, dict, dict]] = []
    geom_cnt: dict[tuple[int, int, int, int, int, int], int] = {}
    for sc, kx, ky, b1, b2 in cands:
        gk = tuple(sorted([(b1["w"], b1["h"], b1["g"]), (b2["w"], b2["h"], b2["g"])]))
        if geom_cnt.get(gk, 0) >= 1:
            continue
        geom_cnt[gk] = geom_cnt.get(gk, 0) + 1
        out.append((sc, kx, ky, b1, b2))
        if len(out) >= top_k:
            break
    return out


def solve_pair(
    n: int,
    m: int,
    b1_base: dict,
    b2_base: dict,
    inc_dec: Decimal,
    tag: str,
    base_seed: int,
    n_threads: int = 2,
    run_mechwalk: bool = True,
) -> dict | None:
    """Solve one 2-pocket pair via direct zero-ov SLP + masked C++ seam melt + lpslack + mechwalk."""
    best_cert: dict | None = None
    seeds, masks, scheds, labels = [], [], [], []

    def _update(cand: dict | None) -> None:
        nonlocal best_cert
        if cand is None:
            return
        if best_cert is None or Decimal(max(cand["L_lit"], cand["L_pd"])) < Decimal(max(best_cert["L_lit"], best_cert["L_pd"])):
            best_cert = cand

    kx = b1_base["w"] + b2_base["w"] - m
    ky = b1_base["h"] + b2_base["h"] - m

    # 1) Direct zero-overlap SLP when kx <= 0, ky <= 0, kx + ky < 0
    if kx <= 0 and ky <= 0 and kx + ky < 0:
        for m1 in ("id", "flipx", "rot180"):
            b1 = bl.transform_brick(b1_base, m1)
            for m2 in ("id", "flipy", "rot180"):
                b2 = bl.transform_brick(b2_base, m2)
                built = pp.build_pinwheel_any(m, b1, b2, sf=0.0, f_override=None)
                if built is None:
                    continue
                ax, ay, at, _, _, _, _ = built
                pen, _ = cpp_eval_sat_pen_bbox(ax, ay, at)
                if pen > 1e-10:
                    continue
                sx, sy, st, _, ok = bslp.joint_xytheta_slp_tight(ax, ay, at)
                use_x, use_y, use_t = (sx, sy, st) if ok else (ax, ay, at)
                c = pp.polish_lpslack_and_mechwalk(
                    n, use_x, use_y, use_t, inc_dec, f"{tag}_{m1}_{m2}_direct",
                    run_mechwalk=False, seed=base_seed,
                )
                _update(c)
                if best_cert is not None and best_cert["dL"] < -1e-4:
                    break
            if best_cert is not None and best_cert["dL"] < -1e-4:
                break

    # 2) Build C++ masked seam-melt batch (handles kx=ky=0, kx+ky=1, and asymmetric sf relaxation)
    for m1 in ("id", "flipx", "flipy", "rot180"):
        b1 = bl.transform_brick(b1_base, m1)
        for m2 in ("id", "flipx", "flipy", "rot180"):
            if (m1, m2) not in (
                ("id", "id"),
                ("flipx", "id"),
                ("id", "flipy"),
                ("flipx", "flipy"),
                ("rot180", "rot180"),
                ("rot180", "id"),
                ("id", "rot180"),
                ("flipy", "flipx"),
            ):
                continue
            b2 = bl.transform_brick(b2_base, m2)
            f_hi = max(b1["fx"], b1["fy"], b2["fx"], b2["fy"])
            f_lo = min(b1["fx"], b1["fy"], b2["fx"], b2["fy"])
            f_avg = 0.5 * (max(b1["fx"], b1["fy"]) + max(b2["fx"], b2["fy"]))
            if abs(f_hi - f_lo) < 5e-4:
                sf_list = (0.0,)
                f_list = ((f_hi, "fhi"),)
            else:
                sf_list = (0.0, 0.5)
                f_list = ((f_avg, "favg"), (f_hi, "fhi"))
            for sf in sf_list:
                for f_use, ftag in f_list:
                    built = pp.build_pinwheel_any(m, b1, b2, sf=sf, f_override=f_use)
                    if built is None:
                        continue
                    ax, ay, at, _, sm, _, _ = built
                    seeds.append((ax, ay, at))
                    masks.append(sm)
                    L_tgt = m + f_use
                    extra = 0.015 if (kx > 0 or ky > 0) else 0.008
                    scheds.append([L_tgt + extra, L_tgt + 0.002, L_tgt - 2e-4])
                    labels.append(f"{tag}_{m1}_{m2}_sf{sf}_{ftag}")

    if seeds and not (kx + ky < 0 and best_cert is not None and abs(b1_base["fx"] - b1_base["fy"]) < 5e-4 and abs(b2_base["fx"] - b2_base["fy"]) < 5e-4):
        cap = 4 if (kx + ky < 0) else 6
        seeds = seeds[:cap]
        masks = masks[:cap]
        scheds = scheds[:cap]
        labels = labels[:cap]
        relaxed = bl.su.cpp_batch_melt(
            seeds,
            scheds,
            sym_mode=1,
            a_start=0.96,
            n_melt_stages=3,
            seam_masks=masks,
            seam_iters=80,
            full_iters=120,
            n_ruin_iters=0,
            n_threads=n_threads,
            base_seed=base_seed,
        )
        pool = []
        for bi, row in enumerate(relaxed):
            nx, ny, nth, Lm = bl.normalize(row["x"], row["y"], row["th"])
            lbl = f"{labels[bi]}_melt"
            if row["best_L_feas"] is not None:
                pool.append((0.0, Lm, nx, ny, nth, lbl))
            elif row["pen"] < 0.02:
                pool.append((1.0 + float(row["pen"]), Lm, nx, ny, nth, lbl))
        pool.sort(key=lambda t: (t[0], t[1]))
        for _, _, cx, cy, cth, lbl in pool[:2]:
            sx, sy, st, _, ok = bslp.joint_xytheta_slp_tight(cx, cy, cth)
            use_x, use_y, use_t = (sx, sy, st) if ok else (cx, cy, cth)
            c = pp.polish_lpslack_and_mechwalk(
                n, use_x, use_y, use_t, inc_dec, lbl, run_mechwalk=False, seed=base_seed + 17
            )
            _update(c)

    if run_mechwalk and best_cert is not None and best_cert["dL"] < 1.5e-3:
        bx, by, bt, _ = pp.hlib._rows_to_arrays(best_cert["rows"])
        c_mw = pp.polish_lpslack_and_mechwalk(
            n, bx, by, bt, inc_dec, best_cert["label"], run_mechwalk=True, seed=base_seed + 99
        )
        _update(c_mw)

    return best_cert


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default=",".join(str(x) for x in GATE_TARGETS))
    ap.add_argument("--top-k", type=int, default=6)
    ap.add_argument("--slack-max", type=float, default=0.025)
    ap.add_argument("--allow-kx1", action="store_true")
    ap.add_argument("--no-mechwalk", action="store_true")
    ap.add_argument("--out", default="gate_summary.json")
    args = ap.parse_args()

    t0 = time.time()
    pp.ensure_dirs()
    targets = [int(x) for x in args.targets.split(",") if x.strip()]
    frontier = pp.load_merged_frontier(targets, certify=True)
    bricks = load_all_bricks(include_external=True)
    print(f"Loaded {len(bricks)} oriented bricks; solving targets={targets}", flush=True)

    report: dict = {"targets": {}, "wins": []}
    for n in targets:
        tn = time.time()
        m = int(math.isqrt(n))
        r = n - m * m
        inc_dec = frontier[n]["L_dec"]
        f_inc = float(inc_dec) - m
        pairs = pick_pairs_for_n(n, bricks, f_inc, top_k=args.top_k, allow_kx1=args.allow_kx1, slack_max=args.slack_max)
        print(
            f"\n=== TARGET n={n} (m={m}, r=+{r}) L_inc={str(inc_dec)[:18]} (f_inc={f_inc:.9f}) | pairs={len(pairs)} ===",
            flush=True,
        )

        best_cert: dict | None = None
        pair_logs = []
        for idx, (sc, kx, ky, b1, b2) in enumerate(pairs):
            tag = f"n{n}_{b1['tag']}+{b2['tag']}"
            print(
                f"  [{idx+1}/{len(pairs)}] kx={kx:+d} ky={ky:+d} "
                f"{b1['tag']}({b1['w']}x{b1['h']}+{b1['g']},f={b1['f']:.6f}) + "
                f"{b2['tag']}({b2['w']}x{b2['h']}+{b2['g']},f={b2['f']:.6f})",
                flush=True,
            )
            cert = solve_pair(
                n, m, b1, b2, inc_dec, tag,
                base_seed=113000 + n * 13 + idx * 100,
                n_threads=2,
                run_mechwalk=not args.no_mechwalk,
            )
            if cert is not None:
                l_str = str(max(Decimal(cert["L_lit"]), Decimal(cert["L_pd"])))
                print(
                    f"    -> L={l_str[:18]} dL={cert['dL']:+.6e} win={cert['is_real_win']} ({cert['label']})",
                    flush=True,
                )
                pair_logs.append({
                    "pair": f"{b1['tag']}+{b2['tag']}",
                    "kx": kx,
                    "ky": ky,
                    "L": l_str[:18],
                    "dL": cert["dL"],
                    "win": cert["is_real_win"],
                    "label": cert["label"],
                })
                if best_cert is None or Decimal(max(cert["L_lit"], cert["L_pd"])) < Decimal(max(best_cert["L_lit"], best_cert["L_pd"])):
                    best_cert = cert
                if cert["is_real_win"] and cert["L_lit"] == cert["L_pd"]:
                    pp.save_win_json(cert)
                    pp.export_final_csv_from_wins()
                    inc_dec = min(inc_dec, Decimal(cert["L_lit"]))
                    print(f"    *** CERTIFIED REAL WIN n={n}: dL={cert['dL']:+.6e} ***", flush=True)
                    if cert["dL"] <= -1e-3:
                        break
            else:
                print("    -> no feasible", flush=True)
                pair_logs.append({"pair": f"{b1['tag']}+{b2['tag']}", "kx": kx, "ky": ky, "L": None, "dL": None, "win": False})

        is_win = bool(best_cert and best_cert["is_real_win"] and best_cert["L_lit"] == best_cert["L_pd"])
        if is_win:
            report["wins"].append(n)
        best_l_str = str(max(Decimal(best_cert["L_lit"]), Decimal(best_cert["L_pd"]))) if best_cert else None
        report["targets"][str(n)] = {
            "n": n,
            "m": m,
            "r": r,
            "L_inc": str(frontier[n]["L_dec"]),
            "win": is_win,
            "best_L": best_l_str,
            "dL": best_cert["dL"] if best_cert else None,
            "label": best_cert["label"] if best_cert else None,
            "pairs": pair_logs,
            "dt": round(time.time() - tn, 1),
        }
        (pp.DOCS_DIR / args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")

    report["total_dt"] = round(time.time() - t0, 1)
    (pp.DOCS_DIR / args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    exported = pp.export_final_csv_from_wins()
    print(f"\n=== COMPLETE in {report['total_dt']}s | wins={report['wins']} | exported={exported} ===", flush=True)


if __name__ == "__main__":
    main()
