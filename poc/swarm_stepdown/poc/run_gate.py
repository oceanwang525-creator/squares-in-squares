#!/usr/bin/env python3
"""H97 (stepdown) 30-minute gate across {367, 492, 585, 591, 740, 795}.

Tests:
1. Direct (m, r+1) -> (m, r) stress-chain deletion + walldrift shear + global C++ thermal melt
   + angle escape + joint_xytheta_slp + soft-mode mechwalk + lpslack.
2. Rectangular brick g -> g-1 thermal-inflation step-down + 2-brick pinwheel re-assembly.
"""
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

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stepdown_lib as sl  # noqa: E402


def _wall_analysis(x: np.ndarray, y: np.ndarray, th: np.ndarray, L: float) -> dict:
    hx = sl.hx_of(th)
    margins = {"L": x - hx, "R": L - (x + hx), "B": y - hx, "T": L - (y + hx)}
    counts = {w: int(np.sum(m < 5e-4)) for w, m in margins.items()}
    opp = {"L": "R", "R": "L", "B": "T", "T": "B"}
    weak = min(margins, key=lambda w: counts[w])
    heavy = opp[weak]
    inward = {"L": (1.0, 0.0), "R": (-1.0, 0.0), "B": (0.0, 1.0), "T": (0.0, -1.0)}
    return {"weak": weak, "heavy": heavy, "margins": margins, "inward_weak": inward[weak], "inward_heavy": inward[heavy]}


def _pick_deletion_indices(x: np.ndarray, y: np.ndarray, th: np.ndarray, L: float, k_max: int = 8) -> list[tuple[int, str]]:
    """Select diverse high-stress, brick-tip, brick-core, and heavy-wall square indices to delete."""
    n = len(x)
    hx = sl.hx_of(th)
    deg = np.degrees(th) % 90.0
    tilted = (deg > 1.5) & (deg < 88.5)
    t_idx = np.where(tilted)[0]

    # Contact degree proxy (center dist < 1.12)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d2 = dx * dx + dy * dy
    contact_deg = np.sum((d2 > 0) & (d2 < 1.12 * 1.12), axis=1)

    wa = _wall_analysis(x, y, th, L)
    edge_dist = np.minimum.reduce([x - hx, L - (x + hx), y - hx, L - (y + hx)])
    ctr_dist = (x - 0.5 * L) ** 2 + (y - 0.5 * L) ** 2

    picks: list[tuple[int, str]] = []
    seen: set[int] = set()

    def _add(idx: int, tag: str) -> None:
        idx = int(idx)
        if idx not in seen and len(picks) < k_max:
            seen.add(idx)
            picks.append((idx, tag))

    if len(t_idx) > 0:
        # 1. Highest contact-degree tilted square
        for idx in t_idx[np.argsort(-contact_deg[t_idx])[:2]]:
            _add(idx, "tilt_highdeg")
        # 2. Tilted square nearest center (core of tilted arch)
        for idx in t_idx[np.argsort(ctr_dist[t_idx])[:2]]:
            _add(idx, "tilt_core")
        # 3. Tilted square nearest outer wall (tip/boundary)
        for idx in t_idx[np.argsort(edge_dist[t_idx])[:2]]:
            _add(idx, "tilt_tip")

    # 4. Heavy-wall touching square
    heavy_idx = np.argsort(wa["margins"][wa["heavy"]])
    for idx in heavy_idx[:2]:
        _add(idx, f"wall_{wa['heavy']}")

    # Fallback if still < k_max
    for idx in np.argsort(-contact_deg):
        _add(idx, "highdeg")

    return picks


def build_direct_stepdown_batch(
    n_tgt: int,
    dnr: dict,
    tgt: dict,
    rng: np.random.Generator,
) -> tuple[list[tuple[np.ndarray, np.ndarray, np.ndarray]], list[list[float]], list[np.ndarray | None], list[np.ndarray | None], list[float], list[str]]:
    """Build 24 diverse trajectories for (n_dnr -> n_tgt) + direct n_tgt thermal walldrift."""
    xd, yd, thd, Ld = dnr["x"], dnr["y"], dnr["th"], float(dnr["L"])
    xt, yt, tht, Lt = tgt["x"], tgt["y"], tgt["th"], float(tgt["L"])
    L_inc = Lt

    picks = _pick_deletion_indices(xd, yd, thd, Ld, k_max=7)
    wa = _wall_analysis(xd, yd, thd, Ld)
    deg_d = np.degrees(thd) % 90.0
    is_45_dominant = (np.sum((deg_d > 40.0) & (deg_d < 50.0)) > 0.5 * max(1, int(np.sum((deg_d > 1.5) & (deg_d < 88.5))))) or (
        np.sum((deg_d > 1.5) & (deg_d < 88.5)) < 40
    )

    coords_list, L_sched_list, seam_list, drift_list, dstr_list, tags = [], [], [], [], [], []
    sched_deep = [L_inc + 0.015, L_inc + 0.002, L_inc - 0.008, L_inc - 0.020, L_inc - 0.032]
    sched_mid = [L_inc + 0.008, L_inc + 0.001, L_inc - 0.005, L_inc - 0.012, L_inc - 0.022]

    for p_idx, p_tag in picks:
        keep = np.ones(len(xd), dtype=bool)
        keep[p_idx] = False
        xk, yk, tk = xd[keep].copy(), yd[keep].copy(), thd[keep].copy()
        hx_pos, hy_pos = float(xd[p_idx]), float(yd[p_idx])

        # Drift 1: Radial inward pull toward hole + heavy-wall inward push
        dxh = hx_pos - xk
        dyh = hy_pos - yk
        dist_h = np.maximum(0.5, np.hypot(dxh, dyh))
        w_h = np.exp(-dist_h / 4.5)
        drx = w_h * (dxh / dist_h) + 0.35 * wa["inward_heavy"][0]
        dry = w_h * (dyh / dist_h) + 0.35 * wa["inward_heavy"][1]
        drift_hole = np.concatenate([drx, dry]).astype(np.float64)

        # Tilted + ring mask
        deg_k = np.degrees(tk) % 90.0
        tilt_k = (deg_k > 1.5) & (deg_k < 88.5)
        hole_ring = dist_h <= 4.5
        smask = (tilt_k | hole_ring).astype(bool)
        if int(smask.sum()) < 25:
            smask = (dist_h <= 7.0).astype(bool)

        # Trajectory A: Global melt + radial hole collapse drift
        xa = xk.copy() + rng.uniform(-0.04, 0.04, n_tgt)
        ya = yk.copy() + rng.uniform(-0.04, 0.04, n_tgt)
        ta = tk.copy() + rng.uniform(-0.08, 0.08, n_tgt)
        coords_list.append((xa, ya, ta))
        L_sched_list.append(sched_mid)
        seam_list.append(None)
        drift_list.append(drift_hole)
        dstr_list.append(0.06)
        tags.append(f"del_{p_tag}_global_drift")

        # Trajectory B: Tilted-brick + hole-ring masked melt
        xb = xk.copy()
        yb = yk.copy()
        tb = tk.copy()
        xb[smask] += rng.uniform(-0.08, 0.08, int(smask.sum()))
        yb[smask] += rng.uniform(-0.08, 0.08, int(smask.sum()))
        tb[smask] += rng.uniform(-0.15, 0.15, int(smask.sum()))
        coords_list.append((xb, yb, tb))
        L_sched_list.append(sched_mid)
        seam_list.append(smask)
        drift_list.append(drift_hole)
        dstr_list.append(0.04)
        tags.append(f"del_{p_tag}_masked")

        # Trajectory C: Deep global re-angle / 45-deg escape
        xc = xk.copy() + rng.uniform(-0.07, 0.07, n_tgt)
        yc = yk.copy() + rng.uniform(-0.07, 0.07, n_tgt)
        tc = tk.copy()
        if is_45_dominant:
            # Perturb tilted / central region toward ~24 deg or random non-45
            target_ang = math.radians(24.0 if len(tags) % 2 == 0 else 66.0)
            mod_mask = tilt_k | (dist_h <= 6.5)
            tc[mod_mask] = target_ang + rng.uniform(-0.22, 0.22, int(mod_mask.sum()))
        else:
            tc += rng.uniform(-0.22, 0.22, n_tgt)
        coords_list.append((xc, yc, tc))
        L_sched_list.append(sched_deep)
        seam_list.append(None)
        drift_list.append(drift_hole)
        dstr_list.append(0.05)
        tags.append(f"del_{p_tag}_reangle")

    # Add 3 direct target thermal+walldrift trajectories from tgt itself
    wat = _wall_analysis(xt, yt, tht, Lt)
    dr_t = np.concatenate([
        np.full(n_tgt, wat["inward_heavy"][0]),
        np.full(n_tgt, wat["inward_heavy"][1]),
    ]).astype(np.float64)
    for k in range(3):
        xk = xt.copy() + rng.uniform(-0.06, 0.06, n_tgt)
        yk = yt.copy() + rng.uniform(-0.06, 0.06, n_tgt)
        tk = tht.copy() + rng.uniform(-0.18, 0.18, n_tgt)
        if is_45_dominant and k >= 1:
            # Re-angle a 12x12 quadrant to ~25 deg
            qm = (xt < 0.6 * Lt) & (yt < 0.6 * Lt)
            tk[qm] = math.radians(25.0) + rng.uniform(-0.15, 0.15, int(qm.sum()))
        coords_list.append((xk, yk, tk))
        L_sched_list.append(sched_deep)
        seam_list.append(None)
        drift_list.append(dr_t)
        dstr_list.append(0.05)
        tags.append(f"tgt_global_{k}")

    return coords_list, L_sched_list, seam_list, drift_list, dstr_list, tags


def run_direct_gate_target(n_tgt: int, n_dnr: int, portfolio: dict) -> dict:
    t0 = time.time()
    tgt = portfolio["targets"][n_tgt]
    dnr = portfolio["donors"][n_dnr]
    L_inc_dec = tgt["L_dec"]
    L_inc = float(L_inc_dec)
    rng = np.random.default_rng(7000 + n_tgt)

    coords_list, L_sched_list, seam_list, drift_list, dstr_list, tags = build_direct_stepdown_batch(
        n_tgt, dnr, tgt, rng
    )

    # Split into masked vs unmasked batches for su.cpp_batch_melt
    unmasked_idx = [i for i, s in enumerate(seam_list) if s is None]
    masked_idx = [i for i, s in enumerate(seam_list) if s is not None]

    results_all = []
    if unmasked_idx:
        res_u = sl.su.cpp_batch_melt(
            [coords_list[i] for i in unmasked_idx],
            [L_sched_list[i] for i in unmasked_idx],
            a_start=0.83,
            n_melt_stages=4,
            seam_masks=None,
            seam_iters=0,
            full_iters=220,
            n_ruin_iters=6,
            drift_xy_list=[drift_list[i] for i in unmasked_idx],
            drift_strengths=[dstr_list[i] for i in unmasked_idx],
            n_threads=2,
            base_seed=1000 + n_tgt,
        )
        for i, r in zip(unmasked_idx, res_u):
            results_all.append((i, r))

    if masked_idx:
        res_m = sl.su.cpp_batch_melt(
            [coords_list[i] for i in masked_idx],
            [L_sched_list[i] for i in masked_idx],
            a_start=0.86,
            n_melt_stages=4,
            seam_masks=[seam_list[i] for i in masked_idx],
            seam_iters=200,
            full_iters=240,
            n_ruin_iters=6,
            drift_xy_list=[drift_list[i] for i in masked_idx],
            drift_strengths=[dstr_list[i] for i in masked_idx],
            n_threads=2,
            base_seed=2000 + n_tgt,
        )
        for i, r in zip(masked_idx, res_m):
            results_all.append((i, r))

    # Rank candidates by (pen <= 1e-6, L_bbox)
    cands = []
    for idx, r in results_all:
        pen = float(r["pen"])
        Lb = float(r["L_bbox"])
        cands.append((0 if pen <= 1e-6 else 1, Lb + 10.0 * pen, pen, Lb, idx, r))
    cands.sort(key=lambda z: (z[0], z[1]))

    n_feas = sum(1 for c in cands if c[2] <= 1e-7)
    min_pen = min(c[2] for c in cands)
    print(f"  [n={n_tgt}<-{n_dnr}] melt done in {time.time()-t0:.1f}s: feas={n_feas}/{len(cands)} min_pen={min_pen:.3e} top3_L={[round(c[3],6) for c in cands[:3]]}", flush=True)

    best_L = 1e9
    best_cert = None
    best_tag = ""
    # Polish top 5 candidates (feasible or lowest pen*10+L)
    for rank, (_, _, pen, Lb, idx, r) in enumerate(cands[:5]):
        if pen > 0.02:
            continue
        px, py, pt = r["x"], r["y"], r["th"]
        if pen > 1e-8:
            # Extra relax step before SLP
            px, py, pt, pen, Lb = sl.su.cpp_lbfgs_relax(px, py, pt, Lb + 0.008, full_iters=260, seed=300 + rank)
        sx, sy, st, sL, ok = sl.slp_and_polish(
            n_tgt, px, py, pt, run_mechwalk=(rank < 2), run_lpslack=(rank < 2)
        )
        if not ok:
            continue
        if sL < best_L:
            best_L = sL
            best_tag = tags[idx]
        if sL < L_inc - 1e-9:
            cert = sl.certify_pack_against_frontier(n_tgt, sx, sy, st, L_inc_dec)
            if cert is not None and cert["is_real_win"]:
                saved = sl.save_win(n_tgt, cert, f"direct_{tags[idx]}")
                print(f"    *** REAL WIN n={n_tgt}: {L_inc:.9f} -> {cert['L_float']:.9f} (dL={cert['dL']:+.3e}) tag={tags[idx]} saved={saved}", flush=True)
                if best_cert is None or Decimal(cert["L_lit"]) < Decimal(best_cert["L_lit"]):
                    best_cert = cert

    dL_best = (best_cert["dL"] if best_cert else (best_L - L_inc))
    return {
        "n_tgt": n_tgt,
        "n_dnr": n_dnr,
        "L_inc": L_inc,
        "feas": n_feas,
        "total": len(cands),
        "best_L": float(best_cert["L_float"] if best_cert else best_L),
        "dL": float(dL_best),
        "win": bool(best_cert is not None),
        "tag": best_tag,
        "elapsed": round(time.time() - t0, 1),
    }


def extract_and_test_bricks_track1(portfolio: dict) -> list[dict]:
    """Extract tilted bricks from {592, 741, 796, 585}, thermally step-down g->g-1, and test pinwheels."""
    print("\n=== Track 1: Thermal Sub-Brick Step-Down + Pinwheel Re-Assembly ===", flush=True)
    # Build brick bank: existing compressed bricks + stepped-down bricks
    bank: list[dict] = []
    for name in [
        "brick_B4_9x16_peel.json",
        "brick_H60_B4_10x16_mw632.json",
        "brick_H60_B4_11x16_n849.json",
        "brick_B4_18x9_n850.json",
        "brick_H60_B5_20x11_n970s6.json",
        "brick_B5_20x10_n972.json",
        "brick_B5_10x20_n850.json",
        "brick_B5_22x7.json",
        "brick_B5_22x8.json",
        "brick_B6_10x21_n972.json",
        "brick_B6_24x8.json",
        "brick_B7_17x8.json",
        "brick_B8_10x18_NEW.json",
        "brick_B8_18x8.json",
    ]:
        p = sl.REPO / "poc" / "swarm_b8xplant" / "data" / name
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        x = np.array(d["x"], dtype=np.float64)
        y = np.array(d["y"], dtype=np.float64)
        th = np.array(d["th"], dtype=np.float64)
        x, y, W, H = sl.brl.rect_bbox(x, y, th)
        b = {
            "w": int(d["w"]), "h": int(d["h"]), "g": int(d["g"]),
            "fx": W - d["w"], "fy": H - d["h"], "f": max(W - d["w"], H - d["h"]),
            "x": x, "y": y, "th": th, "src_n": 0, "tag": name.replace("brick_", "").replace(".json", ""),
        }
        bank.append(b)
        if b["w"] != b["h"]:
            bank.append(sl.make_sym_brick(b, "T"))

    # Also add B3 bricks from thinb4 cache
    tb4_cache = sl.REPO / "poc" / "swarm_thinb4" / "data" / "compressed_bricks_cache.json"
    if tb4_cache.exists():
        cb = json.loads(tb4_cache.read_text())
        for k in ["B3_9x15_n848", "B3_14x9_n632", "B3_10x14_n736", "B5_10x19_n793"]:
            if k in cb:
                v = cb[k]
                x = np.array(v["x"], dtype=np.float64)
                y = np.array(v["y"], dtype=np.float64)
                th = np.array(v["th"], dtype=np.float64)
                x, y, W, H = sl.brl.rect_bbox(x, y, th)
                b = {
                    "w": int(v["w"]), "h": int(v["h"]), "g": int(v["g"]),
                    "fx": W - v["w"], "fy": H - v["h"], "f": max(W - v["w"], H - v["h"]),
                    "x": x, "y": y, "th": th, "src_n": v.get("src_n", 0), "tag": k,
                }
                bank.append(b)
                if b["w"] != b["h"]:
                    bank.append(sl.make_sym_brick(b, "T"))

    # Add stepped-down bricks from data/*_g*.json + freshly stepped-down B7_17x8 -> g6, B8_10x18 -> g7, B5_22x8 -> g4
    for jp in sorted(sl.DATA_DIR.glob("brick_*_g*.json")):
        d = json.loads(jp.read_text())
        x = np.array(d["x"], dtype=np.float64)
        y = np.array(d["y"], dtype=np.float64)
        th = np.array(d["th"], dtype=np.float64)
        x, y, W, H = sl.brl.rect_bbox(x, y, th)
        b = {
            "w": int(d["w"]), "h": int(d["h"]), "g": int(d["g"]),
            "fx": W - d["w"], "fy": H - d["h"], "f": max(W - d["w"], H - d["h"]),
            "x": x, "y": y, "th": th, "src_n": 0, "tag": jp.stem,
        }
        bank.append(b)
        if b["w"] != b["h"]:
            bank.append(sl.make_sym_brick(b, "T"))

    # Check all zero-overlap pinwheel pairs against gate targets
    track1_results = []
    for n_tgt, _ in sl.GATE_PAIRS:
        m = int(math.floor(math.sqrt(n_tgt)))
        r = n_tgt - m * m
        tgt = portfolio["targets"][n_tgt]
        L_inc_dec = tgt["L_dec"]
        L_inc = float(L_inc_dec)
        f_inc = L_inc - m

        cands = []
        for i in range(len(bank)):
            b1 = bank[i]
            if b1["f"] >= f_inc - 5e-4 or b1["w"] >= m or b1["h"] >= m:
                continue
            for j in range(i, len(bank)):
                b2 = bank[j]
                if b2["f"] >= f_inc - 5e-4 or b2["w"] >= m or b2["h"] >= m:
                    continue
                if b1["g"] + b2["g"] == r and b1["w"] + b2["w"] <= m and b1["h"] + b2["h"] <= m:
                    cands.append((max(b1["f"], b2["f"]), b1, b2))
        cands.sort(key=lambda z: z[0])
        print(f"  [pinwheel n={n_tgt}] {len(cands)} zero-overlap candidate pairs below f_inc={f_inc:.6f}", flush=True)
        for f0, b1, b2 in cands[:4]:
            asms = sl.assemble_zero_overlap_pinwheel(m, r, b1, b2)
            for ax, ay, at, aL, atag in asms[:2]:
                sx, sy, st, sL, ok = sl.slp_and_polish(n_tgt, ax, ay, at, run_mechwalk=True, run_lpslack=True)
                if ok and sL < L_inc - 1e-9:
                    cert = sl.certify_pack_against_frontier(n_tgt, sx, sy, st, L_inc_dec)
                    if cert and cert["is_real_win"]:
                        saved = sl.save_win(n_tgt, cert, f"pinwheel_{atag}")
                        print(f"    *** PINWHEEL WIN n={n_tgt}: {L_inc:.9f} -> {cert['L_float']:.9f} (dL={cert['dL']:+.3e}) saved={saved}", flush=True)
                        track1_results.append({"n": n_tgt, "dL": cert["dL"], "tag": atag})
    return track1_results


def main() -> None:
    t_start = time.time()
    need = {x for pair in sl.GATE_PAIRS for x in pair}
    print(f"Loading portfolio for gate pairs {sl.GATE_PAIRS}...", flush=True)
    portfolio = sl.load_portfolio(need, include_extra_donors=True)
    for n_tgt, n_dnr in sl.GATE_PAIRS:
        print(f"  pair n={n_tgt} (L={float(portfolio['targets'][n_tgt]['L_dec']):.9f}) <- dnr={n_dnr} (L={float(portfolio['donors'][n_dnr]['L_dec']):.9f} [{portfolio['donors'][n_dnr]['source']}])", flush=True)

    t1 = extract_and_test_bricks_track1(portfolio)
    print("\n=== Track 2: Direct (m, r+1)->(m, r) Stress-Chain Deletion + Walldrift + Global C++ Melt ===", flush=True)
    t2 = []
    for n_tgt, n_dnr in sl.GATE_PAIRS:
        res = run_direct_gate_target(n_tgt, n_dnr, portfolio)
        t2.append(res)

    wins = sl.rebuild_final_csv()
    summary = {"elapsed_s": round(time.time() - t_start, 1), "track1_wins": t1, "track2": t2, "certified_wins": wins}
    sl.DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (sl.DOCS_DIR / "gate_results.json").write_text(json.dumps(summary, indent=2))
    print(f"\nGate complete in {summary['elapsed_s']}s. Certified wins: {wins}", flush=True)


if __name__ == "__main__":
    main()
