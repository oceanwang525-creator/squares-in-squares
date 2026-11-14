#!/usr/bin/env python3
"""Multiscale whole-block ruin/rebuild for squares-in-squares packings.

The incumbent is partitioned into irregular Voronoi blocks at several cell
scales.  A release mask contains a complete block (or two adjacent blocks),
plus the seam squares that touch it.  Released blocks receive rigid
translation/rotation before C++ L-BFGS/ILS rebuild.  The best probes are then
polished with the proven joint SLP/fixed-angle LP and exported only after the
unchanged official metric accepts both literal and pandas transports.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import multiprocessing as mp
import os
import sys
import time
from decimal import Decimal
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASELINE_SRC = ROOT / "poc/baseline/poc/src"
HIGHD_POC = ROOT / "poc/swarm_highd/poc"
AUTO_UNION = ROOT / "poc/auto_union/poc/union.py"
BASELINE_CSV = ROOT / "poc/pell_corrugation/data/border-scan/merged.csv"
GAP_CSV = ROOT / "poc/frontier_nov12/docs/per_n_gap.csv"
POC_DIR = HERE.parents[1]
DATA_DIR = POC_DIR / "data"
DOCS_DIR = POC_DIR / "docs"
LOG_DIR = DOCS_DIR / "logs"
CKPT_DIR = DOCS_DIR / "ckpt"

for _p in (HERE, HIGHD_POC, BASELINE_SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import cpp_wrap  # noqa: E402
import utils as highd  # noqa: E402
from joint_slp import joint_xytheta_slp  # noqa: E402
import squares.core as core  # noqa: E402

_spec = importlib.util.spec_from_file_location("multiscale_union_mod", AUTO_UNION)
union_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(union_mod)


def load_base_group(n: int):
    groups = core.read_submission(BASELINE_CSV)
    rows = groups[n]
    x = np.array([float(r["x"]) for r in rows], dtype=np.float64)
    y = np.array([float(r["y"]) for r in rows], dtype=np.float64)
    th = np.array([float(r["angle"]) for r in rows], dtype=np.float64)
    hx = 0.5 * (np.abs(np.cos(th)) + np.abs(np.sin(th)))
    x -= float((x - hx).min())
    y -= float((y - hx).min())
    pen, base_l = cpp_wrap.cpp_eval_sat_pen_bbox(x, y, th)
    if pen > 1e-12:
        raise ValueError(f"incumbent n={n} is not float-feasible: pen={pen}")
    return x, y, th, base_l


def baseline_limits(n: int) -> tuple[Decimal, Decimal]:
    rows = core.read_submission(BASELINE_CSV)[n]
    lit = Decimal(core.certify(n, rows))
    transported = core.kaggle_roundtrip({n: rows})[n]
    pan = Decimal(core.certify(n, transported))
    return lit, pan


def block_centers(n: int, base_l: float, cells: int, rng: np.random.Generator):
    """Regular grid centers, with a small irregular pull toward square centroids."""
    xs = np.arange(cells) + 0.5
    centers = []
    for ix, fx in enumerate(xs):
        for iy, fy in enumerate(xs):
            cx = base_l * fx / cells
            cy = base_l * fy / cells
            centers.append((cx, cy))
    centers = np.asarray(centers, dtype=np.float64)
    jitter = np.minimum(0.18 * base_l / cells, 0.18)
    centers += rng.uniform(-jitter, jitter, centers.shape)
    return centers


def seam_mask(block: np.ndarray, coords: np.ndarray, width: float):
    d2 = np.sum((coords - coords[block].mean(axis=0)) ** 2, axis=1)
    keep = np.zeros(len(coords), dtype=bool)
    # Seam is irregular because it follows only nearby squares, not an AABB.
    tree = cKDTree(coords[block])
    for idx in tree.query_ball_point(coords, width, workers=1):
        keep[np.asarray(idx, dtype=np.int64)] = True
    # The unweighted centroid operation above is only a seed; the tree query
    # gives the actual square-level seam.  Keep every center within width of it.
    keep |= d2 < (width * 3.0) ** 2
    return np.sort(keep)


def transform_block(x: np.ndarray, y: np.ndarray, th: np.ndarray,
                    block: np.ndarray, dx: float, dy: float,
                    dth: float, rng: np.random.Generator):
    cx = float(x[block].mean())
    cy = float(y[block].mean())
    c, s = math.cos(dth), math.sin(dth)
    ox, oy = x.copy(), y.copy()
    px, py = ox[block] - cx, oy[block] - cy
    x[block] = cx + dx + c * px - s * py
    y[block] = cy + dy + s * px + c * py
    th[block] += dth
    # Rigid blocks are deliberately not sheared.  The C++ melt can rebalance
    # the interface, but it starts from whole-block motion rather than deletion.
    return x, y, th


def build_seeds(n: int, x: np.ndarray, y: np.ndarray, th: np.ndarray,
                base_l: float, n_seeds: int, cells_list: list[int],
                l_expand: float, seed_offset: int):
    rng = np.random.default_rng(seed_offset * 7919 + n)
    coords = np.column_stack([x, y])
    scale_l = base_l + l_expand
    scale = scale_l / base_l
    xs0, ys0, ts0 = x * scale, y * scale, th.copy()
    q0 = cpp_wrap.full_to_free_sym(xs0, ys0, ts0, scale_l)
    q_list, mask_list, labels = [], [], []
    cells_cycle = [cells_list[i % len(cells_list)] for i in range(n_seeds)]

    for seed_i, cells in enumerate(cells_cycle):
        centers = block_centers(n, base_l, cells, rng)
        center_idx = seed_i % len(centers)
        cx, cy = centers[center_idx]
        block = np.argsort((coords[:, 0] - cx) ** 2 +
                           (coords[:, 1] - cy) ** 2)[:0]
        # Voronoi ownership is the nearest seed.  This avoids rectangular
        # donor crops and keeps seams dictated by actual square positions.
        owner = cKDTree(centers).query(coords, workers=1)[1]
        block = np.flatnonzero(owner == center_idx)
        if len(block) < 4:
            continue
        pair = seed_i % 4 == 3
        if pair:
            distances = np.linalg.norm(centers - np.array([cx, cy]), axis=1)
            neighbour = int(np.argsort(distances)[1])
            block = np.flatnonzero((owner == center_idx) | (owner == neighbour))
        width = float(rng.uniform(0.25, 0.85)) + (0.22 if pair else 0.0)
        release = seam_mask(block, coords, width)
        release_idx = np.flatnonzero(release)
        if len(release) < len(block):
            continue
        x0, y0, t0 = xs0.copy(), ys0.copy(), ts0.copy()
        dth = float(rng.uniform(-0.16, 0.16))
        dx = float(rng.uniform(-0.075, 0.075))
        dy = float(rng.uniform(-0.075, 0.075))
        x0, y0, t0 = transform_block(x0, y0, t0, block, dx, dy, dth, rng)
        loose = np.setdiff1d(release_idx, block, assume_unique=False)
        if len(loose):
            mag = 0.016 if not pair else 0.022
            x0[loose] += rng.uniform(-mag, mag, len(loose))
            y0[loose] += rng.uniform(-mag, mag, len(loose))
            t0[loose] += rng.uniform(-0.24, 0.24, len(loose))
        mask = np.zeros(3 * n, dtype=np.uint8)
        mask[release_idx] = 1
        mask[n + release_idx] = 1
        mask[2 * n + release_idx] = 1
        q = cpp_wrap.full_to_free_sym(x0, y0, t0, scale_l)
        tag = "pair" if pair else "block"
        q_list.append(q)
        mask_list.append(mask)
        labels.append(f"mbr_s{cells}_{tag}_{seed_i:03d}")
    return q_list, mask_list, labels, scale_l


def certify_candidate(n: int, rows: list[dict], baseline: tuple[Decimal, Decimal]) -> dict:
    """Certify exported final text, then its actual pandas transport."""
    try:
        lit = Decimal(core.certify(n, rows))
        rt = core.kaggle_roundtrip({n: rows})
        pan = Decimal(core.certify(n, rt[n]))
        text = union_mod.final_text({n: rows})
        groups = union_mod.parse_text(text)
        if n not in groups or len(groups[n]) != n:
            raise ValueError("final text did not retain complete group")
        final_lit = Decimal(core.certify(n, groups[n]))
        final_groups = union_mod.parse_text(pd_roundtrip_csv(text))
        final_pan = Decimal(core.certify(n, final_groups[n]))
        exported_lit = max(final_lit, final_pan)
        base_max = max(baseline)
        d_l = exported_lit - base_max
        return {
            "ok": exported_lit <= Decimal("-1e-9") + base_max,
            "strict": d_l <= Decimal("-1e-9"),
            "solid": d_l <= Decimal("-1e-5"),
            "dL": str(d_l),
            "L_exported_literal": str(final_lit),
            "L_exported_pandas": str(final_pan),
            "L_core_literal": str(lit),
            "L_core_pandas": str(pan),
            "rows": groups[n] if d_l <= Decimal("-1e-9") else None,
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)[:200], "rows": None}


def pd_roundtrip_csv(text: str) -> str:
    import io
    import pandas as pd
    return pd.read_csv(io.StringIO(text)).to_csv(index=False)


def polish_one(task) -> dict:
    idx, label, n, base_l, x, y, th, baseline, pen, l_bbox = task
    t0 = time.time()
    feasible = l_bbox < 1e17
    if feasible:
        x0, y0, th0 = x.copy(), y.copy(), th.copy()
    else:
        reference = max(base_l + 0.0012, l_bbox + max(0.002, 1.7 * pen))
        x0, y0, th0, _ = highd.relax_lbfgs(
            x, y, th, reference, maxiter=220, w_wall=3.0, w_pair=1.0)

    pen0, l0 = cpp_wrap.cpp_eval_sat_pen_bbox(x0, y0, th0)
    best = {
        "label": label, "pen": float(pen0), "lp": float(l0),
        "result": {"ok": False, "rows": None},
    }
    candidates = []
    if pen0 <= 0.012:
        try:
            xs, ys, ts, ls, ok = joint_xytheta_slp(
                x0, y0, th0, rounds=10, tr_th0=0.018, tr_xy0=0.08,
                margin=1e-8, tol=1e-9, patience=2)
            if ok and ls < base_l + 0.004:
                candidates.append((xs, ys, ts, ls))
        except Exception:
            pass
    candidates.append((x0, y0, th0, min(l0, base_l + 0.004)))

    for cand_x, cand_y, cand_th, _ in candidates:
        for margin in (1e-7, 2e-8, 5e-9):
            try:
                xf, yf, lp_l, ok = highd.fixed_angle_lp_tighten(
                    cand_x, cand_y, cand_th, rounds=6, margin=margin)
            except Exception:
                continue
            if not ok:
                continue
            result = certify_candidate(n, [
                {"square_id": str(i), "x": repr(float(xf[i])),
                 "y": repr(float(yf[i])), "angle": repr(float(cand_th[i]))}
                for i in range(n)
            ], baseline)
            if result.get("ok"):
                key = Decimal(result["dL"])
                old = Decimal(best["result"]["dL"]) if best["result"].get("ok") else Decimal(1)
                if key < old:
                    best.update({"lp": lp_l, "result": result})
    return {
        "idx": idx, "label": label, "cpp_pen": float(pen),
        "cpp_L_bbox": float(l_bbox), "feasible": bool(feasible),
        "best": {k: v for k, v in best.items() if k != "result"},
        "cert": {k: v for k, v in best["result"].items() if k != "rows"},
        "rows": best["result"].get("rows"),
        "dt": round(time.time() - t0, 3),
    }


def append_json_line(path: Path, value: dict):
    with path.open("a") as log:
        log.write(json.dumps(value, sort_keys=True) + "\n")


def solve_target(args) -> dict:
    n, n_seeds, threads, l_expand, drop, ils_scale, top_k, seed_offset, overwrite = args
    t_start = time.time()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = CKPT_DIR / f"target_{n}.json"
    if checkpoint.exists() and not overwrite:
        return json.loads(checkpoint.read_text())

    x, y, th, base_l = load_base_group(n)
    baseline = baseline_limits(n)
    q_list, mask_list, labels, scale_l = build_seeds(
        n, x, y, th, base_l, n_seeds, [2, 3, 4, 5, 6], l_expand, seed_offset)
    schedule = [scale_l]
    for d in drop:
        schedule.append(base_l - d)
    ils_iters = [max(2, int(ils_scale * (2 + 3 * i))) for i in range(len(schedule))]

    log_path = LOG_DIR / f"run_n{n}.log"
    with log_path.open("a") as log:
        log.write(json.dumps({
            "time": time.strftime("%Y-%m-%d %H:%M:%S"), "n": n,
            "base_l": base_l, "baseline_L_lit": str(baseline[0]),
            "baseline_L_pandas": str(baseline[1]), "seeds": len(q_list),
            "schedule": schedule, "ils_iters": ils_iters,
        }) + "\n")
        schedule_per_seed = [schedule for _ in q_list]
        q_out, q_feas, out_Lf, out_E, out_pen, out_Lb = cpp_wrap.run_cpp_masked_melt(
            n, 1, q_list, mask_list, schedule_per_seed, ils_iters,
            a_start=0.84, n_melt_stages=4, masked_iters=90, full_iters=150,
            n_threads=threads, base_seed=seed_offset * 1000003 + n)

    dim = len(q_list[0])
    tasks = []
    for i in range(len(q_list)):
        if out_Lf[i] < 1e17:
            xx, yy, tt = cpp_wrap.cpp_expand_sym(n, 1, float(out_Lf[i]), q_feas[i])
        else:
            side = float(out_Lb[i]) if out_Lb[i] > 1e-12 else schedule[-1]
            xx, yy, tt = cpp_wrap.cpp_expand_sym(n, 1, side, q_out[i])
        tasks.append((i, labels[i], n, base_l, xx, yy, tt, baseline))
    evals = []
    for t in tasks:
        pen, lb = cpp_wrap.cpp_eval_sat_pen_bbox(t[4], t[5], t[6])
        evals.append((pen, lb))
    order = sorted(range(len(tasks)), key=lambda i: (0 if evals[i][1] < 1e17 else 1, evals[i][1], evals[i][0]))
    polished = []
    polish_tasks = []
    for i in order[:top_k]:
        eval_pen, eval_bbox = evals[i]
        polish_tasks.append(tasks[i] + (float(eval_pen), float(eval_bbox)))
    with mp.get_context("spawn").Pool(processes=max(1, threads)) as pool:
        for result in pool.imap_unordered(polish_one, polish_tasks):
            polished.append(result)
            append_json_line(log_path, result)

    real = [r for r in polished if r["cert"].get("strict")]
    real.sort(key=lambda r: Decimal(r["cert"]["dL"]))
    result = {
        "n": n, "runner": "multiscale_block_rebuild",
        "baseline_L_lit": str(baseline[0]), "baseline_L_pandas": str(baseline[1]),
        "n_seeds": len(q_list), "schedule": schedule,
        "status": "REAL_WIN" if real else ("VALID" if any(r["cert"].get("ok") for r in polished) else "MISS"),
        "best": real[0] if real else (min(polished, key=lambda r: (0 if r["cert"].get("ok") else 1,
                                                                   Decimal(r["cert"].get("dL", 1)) if r["cert"].get("ok") else 1,
                                                                   r["cpp_L_bbox"])) if polished else None),
        "polished_count": len(polished), "dt": round(time.time() - t_start, 2),
    }
    if real:
        best_rows = real[0].get("rows")
        if best_rows:
            win_path = DATA_DIR / f"win_n{n}.csv"
            win_path.write_text(union_mod.final_text({n: best_rows}), encoding="utf-8")
            result["win_csv"] = str(win_path.relative_to(ROOT))
    checkpoint.write_text(json.dumps(result, indent=2, sort_keys=True))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", type=int, nargs="+", default=[969, 791, 908])
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=96)
    ap.add_argument("--l-expand", type=float, default=0.034)
    ap.add_argument("--drop", type=float, nargs="+", default=[0.0015, 0.0035, 0.006])
    ap.add_argument("--ils-scale", type=int, default=6)
    ap.add_argument("--top-k", type=int, default=14)
    ap.add_argument("--seed-offset", type=int, default=1000)
    ap.add_argument("--rerun", action="store_true")
    args = ap.parse_args()
    tasks = []
    for n in args.targets:
        tasks.append((n, args.seeds, args.threads, args.l_expand,
                      tuple(args.drop), args.ils_scale, args.top_k,
                      args.seed_offset, args.rerun))
    if len(tasks) == 1:
        results = [solve_target(tasks[0])]
    else:
        with mp.get_context("spawn").Pool(processes=min(len(tasks), 3)) as pool:
            results = pool.map(solve_target, tasks)
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    mp.freeze_support()
    main()
