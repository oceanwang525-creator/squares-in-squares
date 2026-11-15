#!/usr/bin/env python3
"""H114 (vstrip852): Periodic-strip splicing & compression on n=852/912/974 sub-pockets."""
from __future__ import annotations

import json
import math
import os
import sys
from decimal import Decimal
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
VSTRIP_DIR = HERE.parent
DATA_DIR = VSTRIP_DIR / "data"
DOCS_DIR = VSTRIP_DIR / "docs"
FINAL_CSV = VSTRIP_DIR / "final.csv"
REPO_ROOT = VSTRIP_DIR.parents[1]

for _p in (
    REPO_ROOT / "poc" / "swarm_gstep" / "poc",
    REPO_ROOT / "poc" / "swarm_b8xplant" / "poc",
    REPO_ROOT / "poc" / "auto_union" / "poc",
    REPO_ROOT / "poc" / "swarm_lpslack" / "poc",
    REPO_ROOT / "poc" / "swarm_homocasc2" / "poc",
    REPO_ROOT / "poc" / "swarm_softseamdp" / "poc",
):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import b8_lib as bl  # noqa: E402
import brick_rect_lib as bslp  # noqa: E402
import gstep_lib as gl  # noqa: E402
import homocasc2_lib as hlib  # noqa: E402
import lpslack  # noqa: E402
import union as auto_union  # noqa: E402
from cpp_wrap import cpp_eval_sat_pen_bbox  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "poc" / "row_ratchet_980" / "poc"))
import pslp  # noqa: E402


def save_brick(b: dict, name: str) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_p = DATA_DIR / f"brick_{name}.json"
    payload = {
        "name": name,
        "w": int(b["w"]),
        "h": int(b["h"]),
        "g": int(b["g"]),
        "fx": float(b["fx"]),
        "fy": float(b["fy"]),
        "f": float(b["f"]),
        "x": b["x"].tolist(),
        "y": b["y"].tolist(),
        "th": b["th"].tolist(),
    }
    out_p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_p


def load_brick(path: Path, tag: str = "") -> dict:
    d = json.loads(path.read_text(encoding="utf-8"))
    return gl.make_brick(
        int(d["w"]),
        int(d["h"]),
        int(d["g"]),
        np.array(d["x"], dtype=np.float64),
        np.array(d["y"], dtype=np.float64),
        np.array(d["th"], dtype=np.float64),
        tag=tag or d.get("name", path.stem),
    )


def orient_for_vstrip(b: dict) -> dict:
    """Ensure w <= h and median tilt angle ~23 deg (<= 45 deg) so strips run along x."""
    cur = dict(b)
    if cur["w"] > cur["h"]:
        cur = bl.transform_brick(cur, "T")
    deg = np.degrees(np.mod(cur["th"], 0.5 * math.pi))
    tilt = np.minimum(deg, 90.0 - deg) > 2.0
    if np.any(tilt) and float(np.median(deg[tilt])) > 45.0:
        cur = bl.transform_brick(cur, "flipx")
    cur["tag"] = b.get("tag", "brick")
    return cur


def extract_champion_bricks() -> dict[str, dict]:
    """Extract sub-pockets from n=852, n=912, n=974 (merged.csv / softseamdp), and gstep."""
    out: dict[str, dict] = {}
    df_m = pd.read_csv(bl.MERGED_CSV, usecols=["n", "square_id", "x", "y", "angle"])

    # 1. n=852 from poc/swarm_softseamdp/final.csv if present, else merged.csv
    soft_csv = REPO_ROOT / "poc" / "swarm_softseamdp" / "final.csv"
    g852 = pd.DataFrame()
    if soft_csv.exists():
        df_soft = pd.read_csv(soft_csv)
        if not df_soft.empty:
            for col in ("x", "y", "deg"):
                df_soft[col] = df_soft[col].astype(str).str.lstrip("s").astype(float)
            df_soft["n"] = df_soft["id"].str.split("_").str[0].astype(int)
            g852 = df_soft[df_soft["n"] == 852]
    if not g852.empty:
        x = g852["x"].to_numpy()
        y = g852["y"].to_numpy()
        th = np.deg2rad(g852["deg"].to_numpy())
    else:
        g = df_m[df_m["n"] == 852]
        x, y, th, _ = bl.normalize(
            g["x"].to_numpy(float), g["y"].to_numpy(float), g["angle"].to_numpy(float)
        )
    e = 0.5 * (np.abs(np.cos(th)) + np.abs(np.sin(th)))

    m_9x17 = (x - e >= -0.1) & (x + e <= 9.75) & (y - e >= 11.9) & (y + e <= 30.0)
    b_9x17 = gl.make_brick(9, 17, 5, x[m_9x17], y[m_9x17], th[m_9x17], "n852_9x17")
    out["n852_9x17"] = orient_for_vstrip(b_9x17)

    m_20x11 = (x - e >= 8.9) & (x + e <= 30.0) & (y - e >= -0.1) & (y + e <= 11.75)
    b_20x11 = gl.make_brick(20, 11, 6, x[m_20x11], y[m_20x11], th[m_20x11], "n852_11x20")
    out["n852_11x20"] = orient_for_vstrip(b_20x11)

    # 2. n=912 and n=974 from merged.csv
    for n, specs in [
        (912, [("n912_9x21", 9, 21, 6, -0.1, 9.75, 9.0, 31.0)]),
        (
            974,
            [
                ("n974_9x21", 9, 21, 6, -0.1, 9.75, 10.0, 32.0),
                ("n974_10x22", 22, 10, 7, 9.0, 32.0, -0.1, 10.75),
            ],
        ),
    ]:
        g = df_m[df_m["n"] == n]
        xn, yn, thn, _ = bl.normalize(
            g["x"].to_numpy(float), g["y"].to_numpy(float), g["angle"].to_numpy(float)
        )
        en = 0.5 * (np.abs(np.cos(thn)) + np.abs(np.sin(thn)))
        for nm, w, h, gv, x0, x1, y0, y1 in specs:
            mk = (xn - en >= x0) & (xn + en <= x1) & (yn - en >= y0) & (yn + en <= y1)
            if int(mk.sum()) == w * h + gv:
                b = gl.make_brick(w, h, gv, xn[mk], yn[mk], thn[mk], nm)
                out[nm] = orient_for_vstrip(b)

    # 3. Pre-existing high-value gstep / b8xplant bricks
    extra_paths = {
        "gstep_11x21_k5": REPO_ROOT / "poc" / "swarm_gstep" / "data" / "brick_B6_11x21_k5.json",
        "gstep_11x20_top": REPO_ROOT / "poc" / "swarm_gstep" / "data" / "brick_B6_11x20_from11x21_top.json",
        "gstep_9x21_left": REPO_ROOT / "poc" / "swarm_gstep" / "data" / "brick_B6_9x21_peel_left.json",
        "gstep_10x22_vs": REPO_ROOT / "poc" / "swarm_gstep" / "data" / "brick_B7_10x22_vstrip.json",
        "b8_11x20_g5": REPO_ROOT / "poc" / "swarm_b8xplant" / "data" / "brick_H60_B5_20x11_n970s6.json",
    }
    for nm, p in extra_paths.items():
        if p.exists():
            b = load_brick(p, nm)
            b["tag"] = nm
            out[nm] = orient_for_vstrip(b)

    return out


def splice_horizontal_column_gapfill(b: dict) -> list[tuple[str, dict]]:
    """Splice 1 tilted column + AA squares: (w, h, g) -> (w+1, h, g+1), filling cap AA gaps if needed."""
    b = orient_for_vstrip(b)
    w, h, g = b["w"], b["h"], b["g"]
    x, y, th = b["x"], b["y"], b["th"]
    deg = np.degrees(np.mod(th, 0.5 * math.pi))
    tilt = np.minimum(deg, 90.0 - deg) > 2.0
    strips, ang_med = gl._cluster_strips(x, y, th, tilt)
    dx_step = math.cos(ang_med)
    dy_step = math.sin(ang_med)

    aa_idx = np.flatnonzero(~tilt)
    aa_rows: list[tuple[float, list[int]]] = []
    for i in aa_idx:
        yi = float(y[i])
        hit = None
        for ri, (ry, lst) in enumerate(aa_rows):
            if abs(ry - yi) < 0.35:
                lst.append(int(i))
                aa_rows[ri] = (float(np.mean([y[k] for k in lst])), lst)
                hit = ri
                break
        if hit is None:
            aa_rows.append((yi, [int(i)]))
    aa_rows.sort(key=lambda t: t[0])

    out: list[tuple[str, dict]] = []
    for x_cut in (0.5 * w - 1.2, 0.5 * w - 0.3, 0.5 * w + 0.6):
        nx_l, ny_l, nt_l = [], [], []
        for st in strips:
            left = [i for i in st if x[i] <= x_cut]
            right = [i for i in st if x[i] > x_cut]
            for i in left:
                nx_l.append(float(x[i]))
                ny_l.append(float(y[i]))
                nt_l.append(float(th[i]))
            if left and right:
                ref = left[-1]
                nx_l.append(float(x[ref] + dx_step))
                ny_l.append(float(y[ref] + dy_step))
                nt_l.append(float(th[ref]))
                shift_x, shift_y = dx_step, dy_step
            elif not left:
                shift_x, shift_y = dx_step, dy_step
            else:
                shift_x, shift_y = 0.0, 0.0
            for i in right:
                nx_l.append(float(x[i] + shift_x))
                ny_l.append(float(y[i] + shift_y))
                nt_l.append(float(th[i]))

        aa_x, aa_y, aa_t = [], [], []
        gap_slots: list[tuple[float, float, float]] = []
        for ry, row in aa_rows:
            row_s = sorted(row, key=lambda i: x[i])
            left_aa = [i for i in row_s if x[i] <= x_cut]
            right_aa = [i for i in row_s if x[i] > x_cut]
            for i in left_aa:
                aa_x.append(float(x[i]))
                aa_y.append(float(y[i]))
                aa_t.append(float(th[i]))
            for i in right_aa:
                aa_x.append(float(x[i] + 1.0))
                aa_y.append(float(y[i]))
                aa_t.append(float(th[i]))
            if left_aa and right_aa:
                if (x[right_aa[0]] - x[left_aa[-1]]) < 1.35:
                    aa_x.append(float(x[left_aa[-1]] + 1.0))
                    aa_y.append(float(0.5 * (y[left_aa[-1]] + y[right_aa[0]])))
                    aa_t.append(0.0)
                else:
                    # Widened cap gap: candidate AA slots adjacent to left_aa[-1] or right_aa[0]
                    dist_edge = min(abs(ry - 0.5), abs(ry - (h + 0.2)))
                    gap_slots.append((dist_edge, float(x[left_aa[-1]] + 1.0), float(y[left_aa[-1]])))
                    gap_slots.append((dist_edge + 0.05, float(x[right_aa[0]]), float(y[right_aa[0]])))

        tot_n = len(nx_l) + len(aa_x)
        target_n = (w + 1) * h + (g + 1)
        diff = target_n - tot_n
        if diff < 0 or diff > len(gap_slots):
            continue
        if diff > 0:
            gap_slots.sort(key=lambda t: t[0])
            for _, gx, gy in gap_slots[:diff]:
                aa_x.append(gx)
                aa_y.append(gy)
                aa_t.append(0.0)

        ax = np.array(nx_l + aa_x, dtype=np.float64)
        ay = np.array(ny_l + aa_y, dtype=np.float64)
        at = np.array(nt_l + aa_t, dtype=np.float64)
        cand = gl.make_brick(w + 1, h, g + 1, ax, ay, at, tag=f"{b['tag']}_hcol_x{x_cut:.1f}")
        out.append((cand["tag"], cand))
    return out


def pslp_squeeze_brick(b: dict, shake_rounds: int = 5, seed: int = 42) -> dict:
    """Fast kink-safe LP squeeze + shake_squeeze on a rectangular brick (w, h, g)."""
    w, h, g = b["w"], b["h"], b["g"]
    X, Y, T = b["x"].copy(), b["y"].copy(), b["th"].copy()
    x0, x1, y0, y1 = pslp.box_of(X, Y, T)
    X -= x0
    Y -= y0
    f0 = max(x1 - x0 - w, y1 - y0 - h)
    fn, Xn, Yn, Tn, vn, _ = pslp.squeeze(X, Y, T, w, h, f0)
    if vn <= 1e-12 and fn < b["f"] - 1e-10:
        b = gl.make_brick(w, h, g, Xn, Yn, Tn, tag=b["tag"])
    if shake_rounds > 0:
        rng = np.random.default_rng(seed)
        fs, Xs, Ys, Ts = pslp.shake_squeeze(b["x"], b["y"], b["th"], w, h, b["f"], shake_rounds, rng)
        pen_s, _ = cpp_eval_sat_pen_bbox(Xs, Ys, Ts)
        if pen_s <= 1e-8 and fs < b["f"] - 1e-10:
            b = gl.make_brick(w, h, g, Xs, Ys, Ts, tag=b["tag"])
    return b


def compress_and_walk_brick(
    b: dict,
    name: str,
    f_targets: tuple[float, ...] = (0.76, 0.72, 0.695, 0.680, 0.668),
    n_trials: int = 5,
    walk_rounds: int = 5,
    seed: int = 42,
) -> dict | None:
    """Relax spliced/peeled brick, SLP-compress, and run fast pslp + shake_squeeze."""
    comp = gl.relax_and_compress_brick(b, f_targets=f_targets, n_trials=n_trials, seed=seed)
    if comp is None:
        return None
    comp["tag"] = name
    if walk_rounds > 0:
        comp = pslp_squeeze_brick(comp, shake_rounds=walk_rounds, seed=seed + 1000)
    save_brick(comp, name)
    return comp


def load_all_synthesized_bricks() -> list[dict]:
    """Load all saved bricks in DATA_DIR plus champion base bricks (and their transposes)."""
    raw: list[dict] = []
    for p in sorted(DATA_DIR.glob("brick_*.json")):
        b = load_brick(p, p.stem.replace("brick_", ""))
        raw.append(b)
    best: dict[tuple[int, int, int, str], dict] = {}
    for b in raw:
        for op, suf in (("id", ""), ("T", "_T")):
            bo = b if op == "id" else bl.transform_brick(b, "T")
            bo["tag"] = b["tag"] + suf
            best[(bo["w"], bo["h"], bo["g"], bo["tag"])] = bo
    # Keep top-2 thinnest per (w, h, g)
    by_shape: dict[tuple[int, int, int], list[dict]] = {}
    for bo in best.values():
        by_shape.setdefault((bo["w"], bo["h"], bo["g"]), []).append(bo)
    out: list[dict] = []
    for k, lst in sorted(by_shape.items()):
        lst.sort(key=lambda r: r["f"])
        out.extend(lst[:2])
    return out
