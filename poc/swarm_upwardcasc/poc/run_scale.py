#!/usr/bin/env python3
"""Fast high-yield upward (m-1, r) -> (m, r) scale sweep for Hypothesis H98."""
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

import numpy as np

HERE = Path(__file__).resolve().parent
SWARM_DIR = HERE.parent
REPO = SWARM_DIR.parents[1]
DOCS_DIR = SWARM_DIR / "docs"
DATA_DIR = SWARM_DIR / "data"
sys.path.insert(0, str(HERE))

import scan_pairs as sp  # noqa: E402
import upward_engine as ue  # noqa: E402
import upward_lib as ul  # noqa: E402


def _win_to_target(w: dict, label_prefix: str = "win") -> dict:
    """Convert a certified win/cert dict into a donor target entry matching load_portfolio."""
    Lw = Decimal(w["L_lit"])
    rows = sorted(w["rows"], key=lambda r: int(r["square_id"]))
    xs = np.array([float(r["x"]) for r in rows], dtype=np.float64)
    ys = np.array([float(r["y"]) for r in rows], dtype=np.float64)
    ths = np.array([float(r["angle"]) for r in rows], dtype=np.float64)
    return {
        "L": float(Lw),
        "L_dec": Lw,
        "x": xs,
        "y": ys,
        "th": ths,
        "source": f"{label_prefix}_n{w['n']}",
    }


def load_existing_wins() -> dict[int, dict]:
    wins: dict[int, dict] = {}
    for p in sorted(DATA_DIR.glob("win_n*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            n = int(d["n"])
            wins[n] = d
        except Exception:
            continue
    return wins


def solve_fast_upward(
    n_target: int,
    n_donor: int,
    donor: dict,
    L_inc_dec: Decimal,
    rng: np.random.Generator,
    threads: int = 2,
    mech_rounds: int = 3,
    lpslack_iters: int = 8,
) -> dict:
    """Fast upward solver: L-border + xcross/tsplit SLP -> 3-round mechwalk -> lpslack."""
    t0 = time.time()
    xd = donor["x"]
    yd = donor["y"]
    td = donor["th"]
    Ld = float(donor["L"])
    L_inc = float(L_inc_dec)

    seeds = []
    seeds.extend(ue.generate_lborder_seeds(n_target, n_donor, xd, yd, td, Ld))
    seeds.extend(ue.generate_xcross_expand_seeds(n_target, n_donor, xd, yd, td, Ld))

    best_cert: dict | None = None
    best_label = "none"
    polished_lborder: list[tuple[float, str, np.ndarray, np.ndarray, np.ndarray, float]] = []
    polished_other: list[tuple[float, str, np.ndarray, np.ndarray, np.ndarray, float]] = []

    # Sort seeds: zero-overlap L-border first, then lowest penetration tsplit/xcross
    lborder_seeds = [s for s in seeds if s[0].startswith("ext")]
    other_seeds = [s for s in seeds if not s[0].startswith("ext")]

    other_ranked = []
    for name, xs, ys, ts, Ls, fmask in other_seeds:
        ovs = ul.overlap_pairs(xs, ys, ts)
        pen = sum(p[0] for p in ovs)
        other_ranked.append((pen, name, xs, ys, ts, Ls, fmask))
    other_ranked.sort(key=lambda t: t[0])

    # Polish all L-border seeds + top 6 lowest-penetration interior-cut seeds
    to_polish = [(name, xs, ys, ts, Ls, fmask, True) for name, xs, ys, ts, Ls, fmask in lborder_seeds]
    for pen, name, xs, ys, ts, Ls, fmask in other_ranked[:6]:
        if pen < 0.05:
            to_polish.append((name, xs, ys, ts, Ls, fmask, False))

    for name, xs, ys, ts, Ls, fmask, is_lb in to_polish:
        cert, pack = ul.polish_candidate(
            n_target, xs, ys, ts, L_inc_dec, free_mask=fmask, use_lpslack=False
        )
        if cert is not None:
            if best_cert is None or Decimal(cert["L_lit"]) < Decimal(best_cert["L_lit"]):
                best_cert = cert
                best_label = f"slp_{name}"
        if pack is not None:
            xp, yp, tp, Lp = pack
            bucket = polished_lborder if is_lb else polished_other
            bucket.append((Lp, name, xp, yp, tp, Lp))

    polished_lborder.sort(key=lambda t: t[0])
    polished_other.sort(key=lambda t: t[0])

    # Run mechwalk + lpslack on the best distinct basins (up to 2 L-border + 1 interior-cut)
    basins: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, float]] = []
    if polished_lborder:
        basins.append((polished_lborder[0][1], polished_lborder[0][2], polished_lborder[0][3], polished_lborder[0][4], polished_lborder[0][5]))
        # Pick a second distinct L-border corner orientation if available
        top_prefix = polished_lborder[0][1][:5]
        for Lp, name, xp, yp, tp, _ in polished_lborder[1:]:
            if name[:5] != top_prefix:
                basins.append((name, xp, yp, tp, Lp))
                break
    if polished_other:
        basins.append((polished_other[0][1], polished_other[0][2], polished_other[0][3], polished_other[0][4], polished_other[0][5]))

    best_L_after = float(best_cert["L_float"]) if best_cert is not None else min([b[4] for b in basins], default=1e9)
    for b_idx, (bname, xb, yb, tb, Lb) in enumerate(basins):
        print(f"    [basin {b_idx+1}/{len(basins)} n={n_target}] {bname} start_L={Lb:.9f} (inc={L_inc:.9f})", flush=True)
        mc_cert, (xm, ym, tm, Lm) = ul.mechwalk_compress(
            n_target, xb, yb, tb, L_inc_dec, rng, rounds=mech_rounds, threads=threads
        )
        if mc_cert is not None:
            if best_cert is None or Decimal(mc_cert["L_lit"]) < Decimal(best_cert["L_lit"]):
                best_cert = mc_cert
                best_label = f"mech_{bname}"
        if Lm < best_L_after:
            best_L_after = Lm

        # Follow up with lpslack if close to or below incumbent
        if Lm <= L_inc + 5e-4:
            lp_cert, lp_pack = ul.run_lpslack_polish(n_target, xm, ym, tm, L_inc_dec, outer_iters=lpslack_iters)
            if lp_cert is not None:
                if best_cert is None or Decimal(lp_cert["L_lit"]) < Decimal(best_cert["L_lit"]):
                    best_cert = lp_cert
                    best_label = f"lpslack_{bname}"
            if lp_pack is not None and lp_pack[3] < best_L_after:
                best_L_after = lp_pack[3]

    dt = time.time() - t0
    out = {
        "n": n_target,
        "donor": n_donor,
        "L_inc": str(L_inc_dec),
        "best_L_seen": round(best_L_after, 12),
        "elapsed_s": round(dt, 1),
        "win": False,
        "label": best_label,
    }
    if best_cert is not None:
        out["win"] = True
        out["L_new"] = best_cert["L_lit"]
        out["dL"] = best_cert["dL"]
        saved = ul.save_win_and_rebuild_final(n_target, best_cert, best_label)
        out["saved"] = saved
    return out


def build_candidate_queue() -> list[dict]:
    """Select all high-yield (m-1, r) -> (m, r) pairs from merged.csv + poc/*/final.csv."""
    best = sp.load_merged(REPO / "poc/pell_corrugation/data/border-scan/merged.csv")
    for p in sorted((REPO / "poc").glob("*/final.csv")):
        if p.stat().st_size <= 20:
            continue
        if p.parent.name in {"swarm_upwardcasc", "main_union_nov12", "ring910", "swarm_coordinator"}:
            continue
        parsed = sp.parse_final_csv_text(p.read_text(encoding="utf-8", errors="ignore"), str(p.relative_to(REPO)))
        for n, entry in parsed.items():
            if n not in best or entry[0] < best[n][0] - 1e-13:
                best[n] = entry

    by_mr: dict[tuple[int, int], int] = {}
    for n in range(1, 1001):
        m = math.isqrt(n)
        by_mr[(m, n - m * m)] = n

    non45: list[dict] = []
    deg45: list[dict] = []
    for (m, r), n_tgt in sorted(by_mr.items()):
        if m < 12 or (m - 1, r) not in by_mr:
            continue
        n_don = by_mr[(m - 1, r)]
        if n_tgt not in best or n_don not in best:
            continue
        L_tgt = best[n_tgt][0]
        L_don, _, _, td, _ = best[n_don]
        f_tgt = L_tgt - m
        f_don = L_don - (m - 1)
        df_up = f_tgt - f_don
        higher_fs = [best[by_mr[(m2, r)]][0] - m2 for m2 in range(m + 1, 32) if (m2, r) in by_mr and by_mr[(m2, r)] in best]
        gap_hi = f_tgt - min(higher_fs) if higher_fs else 0.0
        deg = np.degrees(td) % 90.0
        tilt_mask = (deg > 0.5) & (deg < 89.5)
        if not tilt_mask.any():
            continue
        med_tilt = float(np.median(np.minimum(deg[tilt_mask], 90.0 - deg[tilt_mask])))
        is_45 = med_tilt >= 43.0

        item = {
            "n": n_tgt,
            "donor": n_don,
            "m": m,
            "r": r,
            "df_up": round(df_up, 9),
            "gap_hi": round(gap_hi, 9),
            "med_deg": round(med_tilt, 2),
            "is_45": is_45,
        }
        if not is_45 and ((df_up >= -5e-4 and (gap_hi >= 5e-4 or df_up > 1e-7)) or (m >= 16 and df_up >= -1.5e-3 and gap_hi >= 0.01)):
            non45.append(item)
        elif is_45 and m >= 16 and df_up >= -2e-4 and gap_hi >= 0.012:
            deg45.append(item)

    # Sort non45 by m ascending so improved donors cascade up chains, with higher gap_hi first per m
    non45.sort(key=lambda d: (d["m"], -d["gap_hi"], -d["df_up"]))
    # Sort 45-deg by gap_hi descending, limit to top 14
    deg45.sort(key=lambda d: (-d["gap_hi"], d["m"]))
    return non45 + deg45[:14]


def main() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    queue = build_candidate_queue()
    need_ns = sorted(({q["n"] for q in queue} | {q["donor"] for q in queue} | {582}) - {579})
    portfolio = ul.load_portfolio(need_ns, include_prs=False)
    targets = portfolio["targets"]
    incumbents_L = {n: targets[n]["L_dec"] for n in need_ns}
    if any(q["n"] == 579 for q in queue):
        best_tmp = sp.load_merged(REPO / "poc/pell_corrugation/data/border-scan/merged.csv")
        if 579 in best_tmp:
            # Guard against merged.csv scientific notation formatting on n=579 by using strict float margin
            incumbents_L[579] = Decimal(f"{best_tmp[579][0] - 1e-8:.15f}")
    existing_wins = load_existing_wins()

    # Load any existing win coordinates into targets as updated donors if strictly better
    for n_w, w in existing_wins.items():
        Lw = Decimal(w["L_lit"])
        if n_w in targets and Lw < targets[n_w]["L_dec"]:
            targets[n_w] = _win_to_target(w)

    print(f"Queue size: {len(queue)} pairs ({sum(not q['is_45'] for q in queue)} non-45deg, {sum(q['is_45'] for q in queue)} 45deg)", flush=True)
    for q in queue:
        print(f"  {q['n']}<-{q['donor']} (m={q['m']},r={q['r']:+d}) df={q['df_up']:+.6f} gap_hi={q['gap_hi']:.5f} med={q['med_deg']:.1f}", flush=True)

    rng = np.random.default_rng(20261114)
    results: list[dict] = []
    ckpt_path = DOCS_DIR / "scale_results.json"

    # First: deeper re-polish of existing n=582 win if present
    if 582 in existing_wins:
        w582 = existing_wins[582]
        L_inc_582 = incumbents_L[582]
        t582 = targets[582]
        print("\n=== Re-polish n=582 existing win ===", flush=True)
        mc_cert, (xm, ym, tm, Lm) = ul.mechwalk_compress(
            582, t582["x"], t582["y"], t582["th"], Decimal(w582["L_lit"]), rng, rounds=4, threads=2
        )
        lp_cert, _ = ul.run_lpslack_polish(582, xm, ym, tm, L_inc_582, outer_iters=10)
        best_582 = lp_cert if (lp_cert and Decimal(lp_cert["L_lit"]) < Decimal(w582["L_lit"])) else mc_cert
        if best_582 and Decimal(best_582["L_lit"]) < Decimal(w582["L_lit"]):
            print(f"  NEW 582 WIN: {best_582['L_lit']} (dL={best_582['dL']:+.3e})", flush=True)
            ul.save_win_and_rebuild_final(582, best_582, "repolish_mech_lpslack")
            best_582_with_n = {"n": 582, **best_582}
            targets[582] = _win_to_target(best_582_with_n)
        else:
            print(f"  582 stayed at {w582['L_lit']}", flush=True)

    for idx, item in enumerate(queue, 1):
        n_tgt = item["n"]
        n_don = item["donor"]
        print(
            f"\n=== [{idx}/{len(queue)}] n={n_tgt} <- donor={n_don} (m={item['m']}, r={item['r']:+d}, df={item['df_up']:+.6f}, gap_hi={item['gap_hi']:.4f}, med={item['med_deg']}) ===",
            flush=True,
        )
        L_inc_main = incumbents_L[n_tgt]
        if n_tgt in existing_wins and Decimal(existing_wins[n_tgt]["L_lit"]) < L_inc_main:
            print(f"  Already won n={n_tgt} at {existing_wins[n_tgt]['L_lit']}, skipping base run.", flush=True)
            continue

        res = solve_fast_upward(
            n_tgt, n_don, targets[n_don], L_inc_main, rng, threads=2, mech_rounds=3, lpslack_iters=8
        )
        res.update({"m": item["m"], "r": item["r"], "is_45": item["is_45"]})
        results.append(res)
        print(f"  RESULT n={n_tgt}: {res}", flush=True)

        # If we won n_tgt, update targets[n_tgt] so higher (m+1, r) chains inherit the improved donor!
        if res.get("win"):
            w_path = DATA_DIR / f"win_n{n_tgt}.json"
            if w_path.exists():
                w = json.loads(w_path.read_text(encoding="utf-8"))
                targets[n_tgt] = _win_to_target(w)
                existing_wins[n_tgt] = w

        ckpt_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    wins_count = sum(1 for r in results if r.get("win"))
    print(f"\nScale sweep complete: {wins_count} new wins out of {len(results)} targets tested.", flush=True)


if __name__ == "__main__":
    main()
