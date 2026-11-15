#!/usr/bin/env python3
"""Fast H113 scale-up across K=20..30, D=8..16 using all 0e74cfc peeled + external bricks."""
from __future__ import annotations

import os

for _v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
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
from cpp_wrap import cpp_eval_sat_pen_bbox  # noqa: E402
from joint_slp import joint_xytheta_slp  # noqa: E402

sys.path.insert(0, str(HERE))
import run_gate as rg  # noqa: E402
import run_scale as rs  # noqa: E402


FLIPS_8 = (
    ("id", "id"),
    ("flipx", "id"),
    ("id", "flipy"),
    ("flipx", "flipy"),
    ("rot180", "rot180"),
    ("rot180", "id"),
    ("id", "rot180"),
    ("flipy", "flipx"),
)


def load_extended_bricks() -> list[dict]:
    """Load pocketpeel bricks + gstep/cleanpeel2/peelstep/b7xplant/b8xplant/stepdown."""
    raw_list: list[dict] = []

    def _add(b: dict, sg: str) -> None:
        for is_t, cand in ((False, b), (True, bl.transform_brick(b, "T"))):
            t = cand.get("tag") or b.get("tag") or f"b_{cand['w']}x{cand['h']}"
            if is_t:
                t = t + "_T"
            raw_list.append(dict(cand, tag=t, src_group=sg))

    for p in sorted(pp.BRICKS_DIR.glob("brick_*.json")):
        _add(pp.load_brick(p), "pocketpeel")

    ext_dirs = (
        ("gstep", pp.REPO_ROOT / "poc" / "swarm_gstep" / "data"),
        ("cleanpeel2", pp.REPO_ROOT / "poc" / "swarm_cleanpeel2" / "data"),
        ("peelstep", pp.REPO_ROOT / "poc" / "swarm_peelstep" / "data"),
        ("b7xplant", pp.REPO_ROOT / "poc" / "swarm_b7xplant" / "data"),
        ("b8xplant", pp.REPO_ROOT / "poc" / "swarm_b8xplant" / "data"),
        ("stepdown", pp.REPO_ROOT / "poc" / "swarm_stepdown" / "data"),
    )
    for sg, d in ext_dirs:
        if d.exists():
            for p in sorted(d.rglob("brick_*.json")):
                try:
                    _add(pp.load_brick(p), sg)
                except Exception:
                    continue

    by_geom: dict[tuple[int, int, int], list[dict]] = {}
    for b in raw_list:
        by_geom.setdefault((b["w"], b["h"], b["g"]), []).append(b)

    out: list[dict] = []
    for _, lst in sorted(by_geom.items()):
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


def fast_screen_and_polish(
    n: int,
    m: int,
    pairs: list[tuple[float, int, int, dict, dict]],
    inc_dec: Decimal,
    base_seed: int,
    screen_slack: float = 0.0025,
) -> tuple[dict | None, list[dict]]:
    """Fast 8-flip direct+C++ batch screen across pairs, polishing only promising candidates."""
    L_inc = float(inc_dec)
    direct_cands: list[tuple[float, np.ndarray, np.ndarray, np.ndarray, str]] = []
    seeds, masks, scheds, labels = [], [], [], []

    for idx, (_, kx, ky, b1b, b2b) in enumerate(pairs):
        f_hi = max(b1b["fx"], b1b["fy"], b2b["fx"], b2b["fy"])
        L_tgt = m + f_hi
        extra = 0.015 if (kx > 0 or ky > 0) else 0.007
        for m1, m2 in FLIPS_8:
            b1 = bl.transform_brick(b1b, m1)
            b2 = bl.transform_brick(b2b, m2)
            built = pp.build_pinwheel_any(m, b1, b2, sf=0.0, f_override=f_hi)
            if built is None:
                continue
            ax, ay, at, _, sm, _, _ = built
            lbl = f"n{n}_{b1b['tag']}+{b2b['tag']}_{m1}_{m2}"
            if kx <= 0 and ky <= 0 and kx + ky < 0:
                pen, L0 = cpp_eval_sat_pen_bbox(ax, ay, at)
                if pen <= 1e-10 and L0 < L_inc + screen_slack:
                    direct_cands.append((L0, ax, ay, at, f"{lbl}_direct"))
                    continue
            seeds.append((ax, ay, at))
            masks.append(sm)
            scheds.append([L_tgt + extra, L_tgt + 0.002, L_tgt - 2e-4])
            labels.append(f"{lbl}_melt")

    melt_cands: list[tuple[float, np.ndarray, np.ndarray, np.ndarray, str]] = []
    if seeds:
        rel = bl.su.cpp_batch_melt(
            seeds,
            scheds,
            sym_mode=1,
            a_start=0.955,
            n_melt_stages=3,
            seam_masks=masks,
            seam_iters=110,
            full_iters=160,
            n_ruin_iters=0,
            n_threads=2,
            base_seed=base_seed,
        )
        for i, r in enumerate(rel):
            nx, ny, nth, Lm = bl.normalize(r["x"], r["y"], r["th"])
            Lf = r["best_L_feas"]
            if Lf is not None and float(Lf) < L_inc + screen_slack:
                melt_cands.append((float(Lf), nx, ny, nth, labels[i]))
            elif r["pen"] < 3e-4 and Lm < L_inc + screen_slack:
                melt_cands.append((Lm + 2.0 * float(r["pen"]), nx, ny, nth, labels[i]))

    direct_cands.sort(key=lambda t: t[0])
    melt_cands.sort(key=lambda t: t[0])
    shortlist = direct_cands[:2] + melt_cands[:3]

    best_cert: dict | None = None
    logs: list[dict] = []
    for rank, (L_est, cx, cy, cth, lbl) in enumerate(shortlist):
        pen0, _ = cpp_eval_sat_pen_bbox(cx, cy, cth)
        use_x, use_y, use_t = cx, cy, cth
        if pen0 > 1e-9:
            sx, sy, st, Ls, ok = joint_xytheta_slp(
                cx, cy, cth, rounds=6, tr_th0=0.008, tr_xy0=0.03, margin=1e-8, tol=1e-10, patience=2
            )
            if ok and Ls < L_inc + screen_slack:
                use_x, use_y, use_t = sx, sy, st
        c = pp.polish_lpslack_and_mechwalk(
            n, use_x, use_y, use_t, inc_dec, lbl, run_mechwalk=False, seed=base_seed + 10 + rank
        )
        if c is not None and c["dL"] < 8e-4:
            bx, by, bt, _ = pp.hlib._rows_to_arrays(c["rows"])
            c_mw = pp.polish_lpslack_and_mechwalk(
                n, bx, by, bt, inc_dec, lbl, run_mechwalk=True, seed=base_seed + 50 + rank
            )
            if c_mw is not None and Decimal(max(c_mw["L_lit"], c_mw["L_pd"])) < Decimal(max(c["L_lit"], c["L_pd"])):
                c = c_mw
            c = rs.try_pslp_polish(n, c, inc_dec)
        if c is not None:
            logs.append({"label": c["label"], "dL": c["dL"], "win": c["is_real_win"]})
            if best_cert is None or Decimal(max(c["L_lit"], c["L_pd"])) < Decimal(max(best_cert["L_lit"], best_cert["L_pd"])):
                best_cert = c
            if c["is_real_win"] and c["dL"] <= -1e-3:
                break
    return best_cert, logs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k-min", type=int, default=20)
    ap.add_argument("--k-max", type=int, default=30)
    ap.add_argument("--d-min", type=int, default=8)
    ap.add_argument("--d-max", type=int, default=16)
    ap.add_argument("--slack-max", type=float, default=0.008)
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--allow-kx1", action="store_true")
    ap.add_argument("--targets", default="")
    ap.add_argument("--out", default="fast_scale_summary.json")
    args = ap.parse_args()

    t0 = time.time()
    pp.ensure_dirs()
    bricks = load_extended_bricks()
    peel_cnt = sum(1 for b in bricks if b["src_group"] == "pocketpeel")
    print(f"Loaded {len(bricks)} bricks ({peel_cnt} pocketpeel)", flush=True)

    if args.targets:
        cand_ns = [int(x) for x in args.targets.split(",") if x.strip()]
    else:
        cand_ns = [K * K + D for K in range(args.k_min, args.k_max + 1) for D in range(args.d_min, args.d_max + 1)]

    frontier = pp.load_merged_frontier(cand_ns, certify=False)
    ranked = []
    for n in cand_ns:
        m = int(math.isqrt(n))
        f_inc = frontier[n]["L"] - m
        pairs = rg.pick_pairs_for_n(
            n, bricks, f_inc, top_k=args.top_k, allow_kx1=args.allow_kx1, slack_max=args.slack_max
        )
        if pairs:
            df = max(pairs[0][3]["f"], pairs[0][4]["f"]) - f_inc
            ranked.append((df, n, pairs))
    ranked.sort(key=lambda t: t[0])
    print(f"Ranked {len(ranked)} targets in K={args.k_min}..{args.k_max}, D={args.d_min}..{args.d_max}", flush=True)

    summary_path = pp.DOCS_DIR / args.out
    report: dict = {"targets": {}, "wins": []}
    if summary_path.exists():
        try:
            report = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    for df_pred, n, pairs in ranked:
        if str(n) in report.get("targets", {}):
            continue
        tn = time.time()
        m = int(math.isqrt(n))
        r = n - m * m
        true_fr = pp.load_merged_frontier([n], certify=True)
        inc_dec = true_fr[n]["L_dec"]

        wp = pp.WINS_DIR / f"win_n{n}.json"
        if wp.exists():
            prev = json.loads(wp.read_text(encoding="utf-8"))
            if Decimal(max(prev["L_lit"], prev["L_pd"])) <= inc_dec:
                print(f"[SKIP] n={n} already beats origin/main", flush=True)
                continue

        print(f"=== n={n} ({m},+{r}) df_pred={df_pred:+.6f} L_inc={str(inc_dec)[:18]} pairs={len(pairs)} ===", flush=True)
        best_cert, logs = fast_screen_and_polish(n, m, pairs, inc_dec, base_seed=313000 + n * 17)
        is_win = bool(best_cert and best_cert["is_real_win"] and best_cert["L_lit"] == best_cert["L_pd"])
        if is_win:
            pp.save_win_json(best_cert)
            if n not in report["wins"]:
                report["wins"].append(n)
            print(f"  *** REAL WIN n={n}: dL={best_cert['dL']:+.6e} ({best_cert['label']}) [{time.time()-tn:.1f}s] ***", flush=True)
        else:
            dl_s = f"{best_cert['dL']:+.6e}" if best_cert else "None"
            print(f"  -> no win (best dL={dl_s}) [{time.time()-tn:.1f}s]", flush=True)

        report["targets"][str(n)] = {
            "n": n,
            "m": m,
            "r": r,
            "df_pred": df_pred,
            "L_inc": str(inc_dec),
            "win": is_win,
            "dL": best_cert["dL"] if best_cert else None,
            "label": best_cert["label"] if best_cert else None,
            "shortlist": logs,
            "dt": round(time.time() - tn, 1),
        }
        summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    report["total_dt"] = round(time.time() - t0, 1)
    summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n=== FAST SCALE COMPLETE in {report['total_dt']}s | wins={report['wins']} ===", flush=True)


if __name__ == "__main__":
    main()
