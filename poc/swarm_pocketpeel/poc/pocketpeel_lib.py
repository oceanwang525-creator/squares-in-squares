#!/usr/bin/env python3
"""H113 (pocketpeel): Sub-pocket 1-row/1-col peeling, brick compression, and 2-pocket seam assembly."""
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

import csv
import fcntl
import io
import json
import math
import sys
import time
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PP_DIR = HERE.parent
DATA_DIR = PP_DIR / "data"
DOCS_DIR = PP_DIR / "docs"
BRICKS_DIR = DATA_DIR / "bricks"
WINS_DIR = DATA_DIR / "wins"
FINAL_CSV = PP_DIR / "final.csv"

REPO_ROOT = PP_DIR.parents[1]
B8_POC = REPO_ROOT / "poc" / "swarm_b8xplant" / "poc"
CP2_POC = REPO_ROOT / "poc" / "swarm_cleanpeel2" / "poc"
AUTO_UNION_POC = REPO_ROOT / "poc" / "auto_union" / "poc"
LPSLACK_POC = REPO_ROOT / "poc" / "swarm_lpslack" / "poc"
HOMOCASC2_POC = REPO_ROOT / "poc" / "swarm_homocasc2" / "poc"
MECHWALK_POC = REPO_ROOT / "poc" / "swarm_mechwalk" / "poc"

for _p in (B8_POC, CP2_POC, AUTO_UNION_POC, LPSLACK_POC, HOMOCASC2_POC, MECHWALK_POC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import b8_lib as bl  # noqa: E402
import brick_rect_lib as bslp  # noqa: E402
import cleanpeel2_lib as cp2  # noqa: E402
import homocasc2_lib as hlib  # noqa: E402
import lpslack  # noqa: E402
import union as auto_union  # noqa: E402
from cpp_wrap import cpp_eval_sat_pen_bbox  # noqa: E402

MERGED_CSV = bl.MERGED_CSV


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    BRICKS_DIR.mkdir(parents=True, exist_ok=True)
    WINS_DIR.mkdir(parents=True, exist_ok=True)


def make_brick(w: int, h: int, g: int, x: np.ndarray, y: np.ndarray, th: np.ndarray, tag: str = "") -> dict:
    return cp2.make_brick(w, h, g, x, y, th, tag=tag)


def save_brick(b: dict, name: str) -> Path:
    ensure_dirs()
    out_p = BRICKS_DIR / f"brick_{name}.json"
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
    return cp2.load_brick_json(path, tag=tag)


def load_merged_frontier(target_ns: list[int] | set[int] | None = None, certify: bool = False) -> dict[int, dict]:
    """Load packings from origin/main merged.csv (already unioned)."""
    df = pd.read_csv(MERGED_CSV, usecols=["n", "square_id", "x", "y", "angle"])
    if target_ns is not None:
        need = {int(n) for n in target_ns}
        df = df[df["n"].isin(need)]
    best: dict[int, dict] = {}
    for n_val, grp in df.groupby("n"):
        n = int(n_val)
        grp_s = grp.sort_values("square_id")
        x, y, th, L_f = bl.normalize(
            grp_s["x"].to_numpy(float, copy=True),
            grp_s["y"].to_numpy(float, copy=True),
            grp_s["angle"].to_numpy(float, copy=True),
        )
        if certify:
            rows = [
                {
                    "square_id": int(r.square_id),
                    "x": repr(float(r.x)),
                    "y": repr(float(r.y)),
                    "angle": repr(float(r.angle)),
                }
                for r in grp_s.itertuples(index=False)
            ]
            L_dec = Decimal(auto_union.certify(n, rows))
        else:
            L_dec = Decimal(f"{L_f:.17g}")
        best[n] = {"x": x, "y": y, "th": th, "L": L_f, "L_dec": L_dec, "source": "merged.csv"}
    return best


def extract_pocket_bbox(
    pack: dict, pk: dict, tag: str, tol: float = 1e-6
) -> dict | None:
    """Extract pocket from pack dict using scanned bbox (x0, x1, y0, y1)."""
    x, y, th = pack["x"], pack["y"], pack["th"]
    hx = bl.half_extent(th)
    lx, rx, ly, ry = x - hx, x + hx, y - hx, y + hx
    m = (
        (lx >= float(pk["x0"]) - tol)
        & (rx <= float(pk["x1"]) + tol)
        & (ly >= float(pk["y0"]) - tol)
        & (ry <= float(pk["y1"]) + tol)
    )
    w, h, g = int(pk["w"]), int(pk["h"]), int(pk["g"])
    if int(m.sum()) != w * h + g:
        return None
    b = make_brick(w, h, g, x[m], y[m], th[m], tag=tag)
    pen, _ = cpp_eval_sat_pen_bbox(b["x"], b["y"], b["th"])
    if pen > 1e-7:
        return None
    return b


def peel_outer_unit_layer(b: dict, side: str, tag: str = "") -> dict | None:
    """Peel 1 outer unit row ('top'/'bottom') or column ('left'/'right') from a brick."""
    return cp2.peel_outer_unit_layer(b, side, tag=tag)


def compress_brick_slp(b: dict, n_trials: int = 12, seed: int = 42) -> dict:
    """Standalone rectangular compression via downward L-BFGS-B + joint_xytheta_slp_rect."""
    w, h, g = b["w"], b["h"], b["g"]
    x0, y0, t0 = b["x"], b["y"], b["th"]
    W0, H0 = w + b["fx"], h + b["fy"]
    f0 = b["f"]
    fmin = min(b["fx"], b["fy"])
    rng = np.random.default_rng(seed)
    best = dict(b, tag=f"{b['tag']}_slp") if cpp_eval_sat_pen_bbox(x0, y0, t0)[0] <= 1e-8 else None
    deg = np.degrees(np.mod(t0, 0.5 * math.pi))
    tilt = np.minimum(deg, 90.0 - deg) > 2.0

    # First try direct SLP without perturbation if already tight
    sx, sy, st, _, ok = bslp.joint_xytheta_slp_rect(
        x0, y0, t0, w, h, rounds=14, tr_th0=0.022, tr_xy0=0.09, tol=1e-11
    )
    if ok and cpp_eval_sat_pen_bbox(sx, sy, st)[0] <= 1e-8:
        cb = make_brick(w, h, g, sx, sy, st, tag=f"{b['tag']}_slp")
        if best is None or cb["f"] < best["f"] - 1e-10:
            best = cb

    ft_sched = (
        max(0.64, 0.5 * (f0 + fmin)),
        max(0.64, fmin + 0.015),
        max(0.638, fmin + 0.002),
        max(0.635, fmin - 0.003),
    )

    for trial in range(n_trials):
        cx, cy, ct = x0.copy(), y0.copy(), t0.copy()
        if trial > 0:
            if trial % 3 == 1:
                mask = ~tilt
            elif trial % 3 == 2:
                mask = (cy < 2.2) | (cy > H0 - 2.2) | (cx < 2.2) | (cx > W0 - 2.2)
            else:
                mask = np.ones(len(cx), dtype=bool)
            nm = int(mask.sum())
            if nm > 0:
                amp = 0.05 if trial % 3 != 0 else 0.025
                cx[mask] += rng.uniform(-amp, amp, nm)
                cy[mask] += rng.uniform(-amp, amp, nm)
                ct[mask] += rng.uniform(-1.5 * amp, 1.5 * amp, nm)
        for ft in ft_sched:
            cx, cy, ct, _ = bslp.relax_rect_lbfgs(cx, cy, ct, w + ft, h + ft, maxiter=280)
        sx, sy, st, _, ok = bslp.joint_xytheta_slp_rect(
            cx, cy, ct, w, h, rounds=14, tr_th0=0.022, tr_xy0=0.09, tol=1e-11
        )
        if ok:
            pen_s, _ = cpp_eval_sat_pen_bbox(sx, sy, st)
            if pen_s <= 1e-8:
                cb = make_brick(w, h, g, sx, sy, st, tag=f"{b['tag']}_slp")
                if best is None or cb["f"] < best["f"] - 1e-10:
                    best = cb
    return best if best is not None else b


def brick_mechwalk(b: dict, name: str, max_rounds: int = 2, save: bool = True) -> dict:
    """Standalone Contact-Jacobian SVD walk + d(i+-j) diagonal mode pairs on a (w, h, g) brick."""
    w, h, g = b["w"], b["h"], b["g"]
    bx, by, W0, H0 = bl.rect_bbox(b["x"], b["y"], b["th"])
    bt = b["th"].copy()
    f0 = max(W0 - w, H0 - h)
    best_f = f0
    best_state = (bx.copy(), by.copy(), bt.copy())
    free_all = np.ones(len(bx), dtype=bool)
    inv_s2 = 1.0 / math.sqrt(2.0)

    for rnd in range(max_rounds):
        improved = False
        for ctol in (1e-5, 4e-4, 2e-3):
            jac, _, _ = bl.localized_contact_jacobian(
                best_state[0], best_state[1], best_state[2], free_all, contact_tol=ctol
            )
            if jac.shape[0] == 0:
                continue
            _, modes = bl.localized_soft_modes(jac, k=10)
            n_single = min(8, len(modes))
            for mi in range(n_single):
                new_state, new_f, hit_amp = cp2._try_brick_direction(
                    best_state, modes[mi], w, h, best_f, (0.004, 0.012, 0.035, 0.08)
                )
                if hit_amp is not None:
                    best_f = new_f
                    best_state = new_state
                    improved = True
            kcap = min(5, len(modes))
            for i in range(kcap):
                for j in range(i + 1, kcap):
                    for d_sgn in (1.0, -1.0):
                        vd = inv_s2 * (modes[i] + d_sgn * modes[j])
                        new_state, new_f, hit_amp = cp2._try_brick_direction(
                            best_state, vd, w, h, best_f, (0.01, 0.03, 0.07)
                        )
                        if hit_amp is not None:
                            best_f = new_f
                            best_state = new_state
                            improved = True
        if not improved:
            break

    out = make_brick(w, h, g, best_state[0], best_state[1], best_state[2], tag=name)
    if save:
        save_brick(out, name)
    return out


def build_pinwheel_any(
    m: int, b1: dict, b2: dict, sf: float = 0.0, f_override: float | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, int, int] | None:
    """Assemble 5-region pinwheel for w1+w2 <= m+1 and h1+h2 <= m+1 (kx <= 1, ky <= 1)."""
    w1, h1, w2, h2 = b1["w"], b1["h"], b2["w"], b2["h"]
    if w1 >= m or h1 >= m or w2 >= m or h2 >= m:
        return None
    kx = w1 + w2 - m
    ky = h1 + h2 - m
    if kx > 1 or ky > 1 or kx + ky > 1:
        return None

    f = max(b1["f"], b2["f"]) if f_override is None else f_override
    L = m + f
    dx1 = sf * (f - b1["fx"])
    dy1 = (m - h1) + (1.0 - sf) * (f - b1["fy"])
    dx2 = (m - w2) + (1.0 - sf) * (f - b2["fx"])
    dy2 = sf * (f - b2["fy"])

    # Anchor grid cells outside pockets; if kx>0 or ky>0, avoid double-counting overlapping region
    gx, gy = [], []
    skipped_cells = []
    for i in range(m):
        for j in range(m):
            in1 = (i < w1) and (j >= m - h1)
            in2 = (i >= m - w2) and (j < h2)
            if in1 or in2:
                continue
            if i < m - w2 and j < m - h1:
                gx.append(i + 0.5)
                gy.append(j + 0.5)
            elif i >= w1 and j >= h2:
                gx.append(i + f + 0.5)
                gy.append(j + f + 0.5)
            else:
                # Central strip when kx > 0 and ky < 0, or ky > 0 and kx < 0
                skipped_cells.append((i + 0.5 * f + 0.5, j + 0.5 * f + 0.5))

    for cx, cy in skipped_cells:
        gx.append(cx)
        gy.append(cy)

    ax = np.concatenate([b1["x"] + dx1, b2["x"] + dx2, np.array(gx, dtype=np.float64)])
    ay = np.concatenate([b1["y"] + dy1, b2["y"] + dy2, np.array(gy, dtype=np.float64)])
    at = np.concatenate([b1["th"], b2["th"], np.zeros(len(gx), dtype=np.float64)])
    if len(ax) != m * m + b1["g"] + b2["g"]:
        return None
    seam = (
        ((ax >= w1 - 1.8) & (ax <= w1 + f + 2.0))
        | ((ay >= m - h1 - 2.0) & (ay <= m - h1 + 2.0))
        | ((ax >= m - w2 - 2.0) & (ax <= m - w2 + 2.0))
        | ((ay >= h2 - 1.8) & (ay <= h2 + f + 2.0))
    ).astype(np.uint8)
    return ax, ay, at, L, seam, kx, ky


def polish_lpslack_and_mechwalk(
    n: int,
    x: np.ndarray,
    y: np.ndarray,
    th: np.ndarray,
    inc_dec: Decimal,
    label: str,
    run_mechwalk: bool = True,
    seed: int = 11301,
) -> dict | None:
    """Certify, run lpslack (dual_non90 + non90), and optionally mechwalk_compress."""
    best = hlib.certify_pack_against_frontier(n, x, y, th, inc_dec)
    if best is not None:
        best["label"] = label

    cur_x, cur_y, cur_th = x, y, th
    xn, yn, tn, Ln = hlib.norm_pack(cur_x, cur_y, cur_th)
    rows_in = [
        {"square_id": str(i), "x": repr(float(xn[i])), "y": repr(float(yn[i])), "angle": repr(float(tn[i]))}
        for i in range(n)
    ]
    target_ce = str(min(inc_dec, Decimal(repr(Ln + 0.5))))
    if best is not None:
        target_ce = str(Decimal(max(best["L_lit"], best["L_pd"])))

    res = lpslack.optimize_group(
        n,
        rows_in,
        target_ce,
        outer_iters=10,
        fixed_rounds=8,
        slp_modes=("dual_non90", "non90"),
        tr_th_schedule=(2e-3, 5e-4, 1e-4, 5e-5),
        cycles=2,
        time_limit=12.0,
    )
    if res.get("status") in ("WIN", "TINY_WIN") and "rows" in res:
        c_lp = hlib.certify_rows_against_frontier(n, res["rows"], inc_dec)
        if c_lp is not None:
            c_lp["label"] = f"{label}_lpslack"
            if best is None or Decimal(max(c_lp["L_lit"], c_lp["L_pd"])) < Decimal(max(best["L_lit"], best["L_pd"])):
                best = c_lp
                cur_x, cur_y, cur_th, _ = hlib._rows_to_arrays(c_lp["rows"])

    if run_mechwalk and best is not None and best["dL"] < 2.0e-3:
        rng = np.random.default_rng(seed)
        L_ref = Decimal(max(best["L_lit"], best["L_pd"]))
        mw_cert, _ = hlib.mechwalk_compress(n, cur_x, cur_y, cur_th, inc_dec, rng, rounds=2, threads=1)
        if mw_cert is not None and Decimal(max(mw_cert["L_lit"], mw_cert["L_pd"])) < L_ref:
            mw_cert["label"] = f"{best['label']}_mw"
            best = mw_cert
            # Final lpslack pass after mechwalk
            bx, by, bt, _ = hlib._rows_to_arrays(best["rows"])
            rows_in2 = [
                {"square_id": str(i), "x": repr(float(bx[i])), "y": repr(float(by[i])), "angle": repr(float(bt[i]))}
                for i in range(n)
            ]
            res2 = lpslack.optimize_group(
                n,
                rows_in2,
                str(Decimal(max(best["L_lit"], best["L_pd"]))),
                outer_iters=8,
                fixed_rounds=8,
                slp_modes=("dual_non90", "non90"),
                tr_th_schedule=(1e-3, 3e-4, 1e-4, 3e-5),
                cycles=1,
                time_limit=10.0,
            )
            if res2.get("status") in ("WIN", "TINY_WIN") and "rows" in res2:
                c_lp2 = hlib.certify_rows_against_frontier(n, res2["rows"], inc_dec)
                if c_lp2 is not None and Decimal(max(c_lp2["L_lit"], c_lp2["L_pd"])) < Decimal(max(best["L_lit"], best["L_pd"])):
                    c_lp2["label"] = f"{best['label']}_lp2"
                    best = c_lp2
    return best


def save_win_json(cert: dict) -> Path:
    ensure_dirs()
    n = int(cert["n"])
    out_p = WINS_DIR / f"win_n{n}.json"
    out_p.write_text(json.dumps(cert, indent=2), encoding="utf-8")
    return out_p


def export_final_csv_from_wins() -> list[int]:
    """Write certified literal==pandas wins from data/wins/win_n*.json to final.csv."""
    ensure_dirs()
    groups: dict[int, list[dict]] = {}
    for wp in sorted(WINS_DIR.glob("win_n*.json")):
        d = json.loads(wp.read_text(encoding="utf-8"))
        if d.get("is_real_win") and d.get("L_lit") == d.get("L_pd") and d.get("rows"):
            groups[int(d["n"])] = d["rows"]
    if not groups:
        FINAL_CSV.write_text("id,x,y,deg\n", encoding="utf-8")
        return []
    FINAL_CSV.write_text(auto_union.final_text(groups), encoding="utf-8")
    return sorted(groups.keys())
