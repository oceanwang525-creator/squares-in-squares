#!/usr/bin/env python3
"""H126 (band3ext): next-gen band3 2-brick corner-melted pinwheel over the combined sub-brick pool.

Reuses poc/swarm_band3/poc/{band3_lib,run_scale,pslp_polish_and_sweep}.py and adds:
  * a generalized corner-brick harvester that runs on arbitrary certified packings
    (all recent win_n*.json files of vstrip492 / vstrip852 / peelstep / pocketpeel / thinband8 /
    b7xplant / band3 + the current origin/main merged.csv),
  * a combined brick loader (band3 external dirs + the new libraries),
  * del_k = g1 + g2 - r in {0, 1, 2}: build the pinwheel with n + del_k squares and delete del_k
    axis-aligned grid squares nearest the two inner corners before the masked corner melt,
  * an extended frontier (origin/main merged.csv + poc/*/final.csv + open-PR finals in ext/).
"""
from __future__ import annotations

import io
import json
import math
import os
import sys
from decimal import Decimal
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXT_DIR_ROOT = HERE.parent
DATA_DIR = EXT_DIR_ROOT / "data"
DOCS_DIR = EXT_DIR_ROOT / "docs"
EXT_DIR = EXT_DIR_ROOT / "ext"
FINAL_CSV = EXT_DIR_ROOT / "final.csv"
REPO_ROOT = EXT_DIR_ROOT.parents[1]
BAND3_POC = REPO_ROOT / "poc" / "swarm_band3" / "poc"
PSLP_POC = REPO_ROOT / "poc" / "row_ratchet_980" / "poc"

for _p in (BAND3_POC, PSLP_POC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import band3_lib as b3  # noqa: E402
import union as auto_union  # noqa: E402
import importlib.util as _ilu  # noqa: E402

_pspec = _ilu.spec_from_file_location("rr980_pslp", PSLP_POC / "pslp.py")
pslp = _ilu.module_from_spec(_pspec)
_pspec.loader.exec_module(pslp)

_spec = _ilu.spec_from_file_location("band3_run_scale", BAND3_POC / "run_scale.py")
_b3rs = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_b3rs)
_lpslack_polish = _b3rs._lpslack_polish
_update_best = _b3rs._update_best

bl = b3.bl
WT = Path("/home/lucas/.gemini/antigravity-cli/worktrees/a4fb0b93-b63b-426d-aa12-67454e99bc08")

NEW_BRICK_DIRS = [
    REPO_ROOT / "poc" / "swarm_cleanpeel2" / "data",
    REPO_ROOT / "poc" / "swarm_gstep" / "data",
    REPO_ROOT / "poc" / "swarm_stepdown" / "data",
    REPO_ROOT / "poc" / "swarm_peelstep" / "data",
    REPO_ROOT / "poc" / "swarm_pocketpeel" / "data" / "bricks",
    REPO_ROOT / "poc" / "swarm_b7xplant" / "data",
    REPO_ROOT / "poc" / "swarm_thinband8" / "data",
    EXT_DIR / "poc" / "swarm_vstrip492" / "data",
    EXT_DIR / "watchxplant_pr1" / "data" / "bricks",
    DATA_DIR / "harvest",
]

WIN_GLOBS = [
    REPO_ROOT / "poc" / "swarm_band3" / "data",
    REPO_ROOT / "poc" / "swarm_peelstep" / "data",
    REPO_ROOT / "poc" / "swarm_pocketpeel" / "data" / "wins",
    REPO_ROOT / "poc" / "swarm_thinband8" / "data",
    REPO_ROOT / "poc" / "swarm_b7xplant" / "data",
    REPO_ROOT / "poc" / "swarm_gstep" / "data",
    REPO_ROOT / "poc" / "swarm_liftpolish3" / "data" / "wins_r2",
    EXT_DIR / "poc" / "swarm_vstrip492" / "data",
    EXT_DIR / "poc" / "swarm_vstrip852" / "data" / "wins",
    EXT_DIR / "thinband8_pr2" / "data",
    EXT_DIR / "watchxplant_pr1" / "data",
]

EXTRA_FINALS = [
    EXT_DIR / "poc" / "swarm_vstrip492" / "final.csv",
    EXT_DIR / "poc" / "swarm_vstrip852" / "final.csv",
    EXT_DIR / "thinband8_pr2" / "final.csv",
    EXT_DIR / "watchxplant_pr1" / "final.csv",
]


# ----------------------------------------------------------------------------------------------
# Harvest
# ----------------------------------------------------------------------------------------------
def harvest_from_packing(
    n: int,
    x: np.ndarray,
    y: np.ndarray,
    th: np.ndarray,
    min_g: int = 3,
    max_f: float = 0.86,
    min_tilt: int = 12,
    src: str = "",
) -> list[dict]:
    x, y, th, L = bl.normalize(x, y, th)
    m = int(math.floor(L))
    f = L - m
    if f > max_f + 0.02 or m < 18:
        return []
    deg = np.degrees(np.mod(th, 0.5 * math.pi))
    tilt = np.minimum(deg, 90.0 - deg) > 2.0
    hx = bl.half_extent(th)
    out: list[dict] = []
    for corner in ("TL", "BR", "BL", "TR"):
        by_g: dict[int, tuple] = {}
        for w in range(5, min(m, 25)):
            for h in range(5, min(m, 25)):
                if w + h > 38:
                    continue
                if corner == "TL":
                    mask = (x + hx <= w + f + 0.02) & (y - hx >= m - h - 0.02)
                elif corner == "BR":
                    mask = (x - hx >= m - w - 0.02) & (y + hx <= h + f + 0.02)
                elif corner == "BL":
                    mask = (x + hx <= w + f + 0.02) & (y + hx <= h + f + 0.02)
                else:
                    mask = (x - hx >= m - w - 0.02) & (y - hx >= m - h - 0.02)
                g = int(mask.sum()) - w * h
                if g < min_g or g > 14 or int((mask & tilt).sum()) < min_tilt:
                    continue
                _, _, W, H = bl.rect_bbox(x[mask], y[mask], th[mask])
                fc = max(W - w, H - h)
                if fc <= min(max_f, f + 0.002):
                    item = (g, w + h, w, h, fc, mask)
                    if g not in by_g or (item[1], item[4]) < (by_g[g][1], by_g[g][4]):
                        by_g[g] = item
        for g, (_, _, w, h, fc, mask) in by_g.items():
            b = b3.make_brick(w, h, g, x[mask], y[mask], th[mask], tag=f"H{g}_{w}x{h}_n{n}{src}_{corner}")
            if b["w"] > b["h"]:
                b = bl.transform_brick(b, "T")
                b["tag"] = f"H{g}_{b['w']}x{b['h']}_n{n}{src}_{corner}T"
            out.append(b)
    return out


def _rows_to_arrays(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rs = sorted(rows, key=lambda r: int(r["square_id"]))
    return (
        np.array([float(r["x"]) for r in rs], dtype=np.float64),
        np.array([float(r["y"]) for r in rs], dtype=np.float64),
        np.array([float(r["angle"]) for r in rs], dtype=np.float64),
    )


def harvest_all(merged_n_range: tuple[int, int] = (400, 1000), save: bool = True) -> list[dict]:
    """Harvest corner bricks from all recent win files + merged.csv; keep best f per (w,h,g)."""
    best: dict[tuple[int, int, int], dict] = {}

    def _add(bs: list[dict]) -> None:
        for b in bs:
            k = (b["w"], b["h"], b["g"])
            if k not in best or b["f"] < best[k]["f"] - 1e-9:
                best[k] = b

    for d in WIN_GLOBS:
        if not d.exists():
            continue
        for p in sorted(d.glob("win_n*.json")):
            try:
                wd = json.loads(p.read_text(encoding="utf-8"))
                n = int(wd.get("n") or int(p.stem.split("n")[-1]))
                x, y, th = _rows_to_arrays(wd["rows"])
                _add(harvest_from_packing(n, x, y, th, src="w"))
            except Exception as exc:  # noqa: BLE001
                print(f"[harvest] skip {p}: {exc}", flush=True)
    df = pd.read_csv(bl.MERGED_CSV, usecols=["n", "square_id", "x", "y", "angle"])
    lo, hi = merged_n_range
    for n_val, grp in df[(df["n"] >= lo) & (df["n"] <= hi)].groupby("n"):
        n = int(n_val)
        m = int(math.isqrt(n))
        if n - m * m < 4:
            continue
        gs = grp.sort_values("square_id")
        _add(
            harvest_from_packing(
                n,
                gs["x"].to_numpy(float, copy=True),
                gs["y"].to_numpy(float, copy=True),
                gs["angle"].to_numpy(float, copy=True),
            )
        )
    out = list(best.values())
    if save:
        hd = DATA_DIR / "harvest"
        hd.mkdir(parents=True, exist_ok=True)
        for b in out:
            payload = {
                "name": b["tag"], "w": int(b["w"]), "h": int(b["h"]), "g": int(b["g"]),
                "fx": float(b["fx"]), "fy": float(b["fy"]), "f": float(b["f"]),
                "x": b["x"].tolist(), "y": b["y"].tolist(), "th": b["th"].tolist(),
            }
            (hd / f"brick_{b['tag']}.json").write_text(json.dumps(payload), encoding="utf-8")
    return out


def load_pool() -> list[dict]:
    for d in NEW_BRICK_DIRS:
        if d not in b3.EXTERNAL_BRICK_DIRS:
            b3.EXTERNAL_BRICK_DIRS.append(d)
    return b3.load_all_bricks(include_external=True)


# ----------------------------------------------------------------------------------------------
# Frontier
# ----------------------------------------------------------------------------------------------
def load_frontier(ns: list[int]) -> dict[int, dict]:
    fr = b3.gl.fast_load_true_frontier(ns, certify=True)
    need = set(int(n) for n in ns)
    prefixes = tuple(f"{n:03}_" for n in sorted(need))
    for p in EXTRA_FINALS:
        if not p.exists():
            continue
        lines = p.read_text(encoding="utf-8").splitlines()
        sub = [lines[0]] + [ln for ln in lines[1:] if ln.startswith(prefixes)]
        if len(sub) <= 1:
            continue
        txt = "\n".join(sub) + "\n"
        lit = auto_union.parse_text(txt)
        pdt = auto_union.parse_text(pd.read_csv(io.StringIO(txt)).to_csv(index=False))
        for n in need:
            if n in lit and n in pdt and len(lit[n]) == n:
                res = auto_union.check((n, lit[n], pdt[n]))
                if res is None:
                    continue
                L_dec = max(Decimal(res[0]), Decimal(res[1]))
                if n not in fr or L_dec < fr[n]["L_dec"]:
                    fr[n] = {"L_dec": L_dec, "L": float(L_dec), "source": str(p.relative_to(REPO_ROOT))}
    return fr


# ----------------------------------------------------------------------------------------------
# Pair enumeration
# ----------------------------------------------------------------------------------------------
def enumerate_pairs(bricks: list[dict], m: int, r: int, f_inc: float, slack: float = 0.008,
                    del_ks: tuple[int, ...] = (0, 1, 2)) -> list[tuple]:
    """Enumerate zero-overlap (ov=0: ct=0 or ct=1) 2-brick pinwheel pairs with g1+g2-r in del_ks."""
    best: dict[tuple, tuple] = {}
    for i, b1 in enumerate(bricks):
        for j in range(i, len(bricks)):
            b2 = bricks[j]
            d = b1["g"] + b2["g"] - r
            if d not in del_ks:
                continue
            w1, h1, w2, h2 = b1["w"], b1["h"], b2["w"], b2["h"]
            if max(w1, h1, w2, h2) >= m:
                continue
            if w1 + w2 > m or h1 + h2 > m:
                continue
            fm = max(b1["f"], b2["f"])
            if fm > f_inc + slack:
                continue
            ct = int(w1 + w2 == m and h1 + h2 == m)
            key = (d, w1, h1, b1["g"], w2, h2, b2["g"])
            score = fm - f_inc - 0.0020 * d + 0.0008 * ct
            if key not in best or score < best[key][0]:
                best[key] = (score, fm, d, ct, b1, b2)
    return sorted(best.values(), key=lambda t: t[0])


# ----------------------------------------------------------------------------------------------
# Solver
# ----------------------------------------------------------------------------------------------
def _delete_corner_squares(ax, ay, at, sm, d, corners):
    if d <= 0:
        return ax, ay, at, sm
    is_grid = np.abs(at) < 1e-12
    keep = np.ones(len(ax), dtype=bool)
    for k in range(d):
        cx, cy = corners[k % len(corners)]
        dist = np.hypot(ax - cx, ay - cy)
        dist[~(is_grid & keep)] = np.inf
        keep[int(np.argmin(dist))] = False
    return ax[keep], ay[keep], at[keep], sm[keep]


def solve_pair_ext(n: int, m: int, b1_base: dict, b2_base: dict, d: int, inc_dec: Decimal, tag: str,
                   base_seed: int, n_threads: int = 1, orient_cap: int = 16) -> tuple[dict | None, list]:
    best_cert: dict | None = None
    inc_f = float(inc_dec)
    seeds, masks, scheds, labels = [], [], [], []
    for m1 in ("id", "flipx", "flipy", "rot180"):
        b1 = bl.transform_brick(b1_base, m1)
        for m2 in ("id", "flipx", "flipy", "rot180"):
            if len(seeds) >= orient_cap * 2:
                break
            b2 = bl.transform_brick(b2_base, m2)
            f_hi = max(b1["fx"], b1["fy"], b2["fx"], b2["fy"])
            built = b3.build_pinwheel(m, b1, b2, sf=0.0, f_override=f_hi, max_ov=1)
            if built is None:
                continue
            ax, ay, at, L_tgt, sm = built
            w1, h1, w2, h2 = b1["w"], b1["h"], b2["w"], b2["h"]
            corners = [(w1 + 0.5 * f_hi, m - h1 + 0.5 * f_hi), (m - w2 + 0.5 * f_hi, h2 + 0.5 * f_hi)]
            ax, ay, at, sm = _delete_corner_squares(ax, ay, at, sm, d, corners)
            if len(ax) != n:
                continue
            pen0, L_box0 = b3.cpp_eval_sat_pen_bbox(ax, ay, at)
            if pen0 <= 1e-4 and L_box0 < inc_f + 1e-3:
                if pen0 > 1e-9:
                    L_try = min(inc_f - 1.2e-3, max(L_tgt, L_box0))
                    rx, ry, rt, rpen = b3.bslp.relax_rect_lbfgs(ax, ay, at, L_try, L_try, maxiter=250)
                else:
                    rx, ry, rt, rpen = ax, ay, at, pen0
                if rpen <= 1e-7:
                    c = bl.polish_and_certify(n, rx, ry, rt, inc_dec, slp_rounds=14)
                    if c is not None:
                        c["label"] = f"{tag}_{m1}_{m2}_direct"
                        best_cert = _update_best(best_cert, c)
            rad = 4.2 + 0.8 * d
            corner_sm = (
                (sm > 0)
                | ((np.abs(ax - w1) <= rad) & (np.abs(ay - (m - h1)) <= rad))
                | ((np.abs(ax - (m - w2)) <= rad) & (np.abs(ay - h2) <= rad))
            ).astype(np.uint8)
            gap = inc_f - L_tgt
            L_hi = min(inc_f - 1.2e-3, L_tgt + max(0.0025, 0.58 * gap)) if gap > 1.5e-3 else min(inc_f - 1.2e-3, L_tgt - 2e-4 - 0.002 * d)
            seeds.append((ax, ay, at))
            masks.append(corner_sm)
            scheds.append([L_tgt + 0.025, 0.5 * (L_tgt + 0.025 + L_hi), L_hi])
            labels.append(f"{tag}_{m1}_{m2}_hi")
            if (m1, m2) in (("id", "id"), ("id", "rot180"), ("flipx", "flipy"), ("rot180", "id")):
                L_tight = min(inc_f - 8e-4, L_tgt + max(0.0012, 0.25 * gap)) if gap > 0.004 else inc_f - 2.5e-3
                seeds.append((ax, ay, at))
                masks.append(corner_sm)
                scheds.append([L_tgt + 0.018, L_tgt + 0.005, L_tight])
                labels.append(f"{tag}_{m1}_{m2}_tight")
    if not seeds:
        return best_cert, []

    relaxed = bl.su.cpp_batch_melt(
        seeds, scheds, sym_mode=1, a_start=0.965, n_melt_stages=3, seam_masks=masks,
        seam_iters=320, full_iters=450, n_ruin_iters=0, n_threads=n_threads, base_seed=base_seed,
    )
    pool = []
    for bi, row in enumerate(relaxed):
        nx, ny, nth, Lm = bl.normalize(row["x"], row["y"], row["th"])
        lbl = f"{labels[bi]}_melt"
        feas = row["best_L_feas"] is not None
        pool.append((0.0 if feas else 1.0 + float(row["pen"]), Lm, float(row["pen"]), nx, ny, nth, lbl))
    pool.sort(key=lambda t: (t[0], t[1]))
    diag = [(t[0] == 0.0, round(t[1], 6), round(t[2], 6), t[6]) for t in pool[:4]]

    rescue_seeds, rescue_scheds, rescue_labels = [], [], []
    for _, Lm, pen, cx, cy, cth, lbl in pool[:5]:
        if pen <= 1e-8:
            c = bl.polish_and_certify(n, cx, cy, cth, inc_dec, slp_rounds=16)
            if c is not None:
                c["label"] = lbl
                best_cert = _update_best(best_cert, c)
        elif pen <= 0.035:
            L_try = min(inc_f - 2.0e-4, max(Lm + 0.0015, inc_f - 0.35 * max(0.001, inc_f - Lm)))
            rx, ry, rt, rpen = b3.bslp.relax_rect_lbfgs(cx, cy, cth, L_try, L_try, maxiter=350)
            if rpen > 1e-6:
                L_try2 = inc_f + 0.0015
                rx2, ry2, rt2, rpen2 = b3.bslp.relax_rect_lbfgs(cx, cy, cth, L_try2, L_try2, maxiter=300)
                if rpen2 < rpen:
                    rx, ry, rt, rpen, L_try = rx2, ry2, rt2, rpen2, L_try2
            if rpen <= 1e-6:
                c = bl.polish_and_certify(n, rx, ry, rt, inc_dec, slp_rounds=16)
                if c is not None:
                    c["label"] = f"{lbl}+lbfgs"
                    best_cert = _update_best(best_cert, c)
            elif rpen <= 0.025:
                rescue_seeds.append((rx, ry, rt))
                rescue_scheds.append([L_try + 0.004, L_try + 0.001, min(inc_f - 4e-4, L_try + 0.0015)])
                rescue_labels.append(f"{lbl}+fullmelt")
    if rescue_seeds:
        full_res = bl.su.cpp_batch_melt(
            rescue_seeds, rescue_scheds, sym_mode=1, a_start=0.980, n_melt_stages=3, seam_masks=None,
            seam_iters=0, full_iters=400, n_ruin_iters=0, n_threads=n_threads, base_seed=base_seed + 777,
        )
        for bi, row in enumerate(full_res):
            nx, ny, nth, Lm = bl.normalize(row["x"], row["y"], row["th"])
            if float(row["pen"]) <= 0.02:
                if float(row["pen"]) > 1e-8:
                    L_try = min(inc_f + 0.001, Lm + 0.002)
                    nx, ny, nth, _ = b3.bslp.relax_rect_lbfgs(nx, ny, nth, L_try, L_try, maxiter=250)
                c = bl.polish_and_certify(n, nx, ny, nth, inc_dec, slp_rounds=16)
                if c is not None:
                    c["label"] = rescue_labels[bi]
                    best_cert = _update_best(best_cert, c)

    if best_cert is not None and best_cert["dL"] <= 0.006:
        best_cert = pslp_refine(n, best_cert, inc_dec)
    if best_cert is not None and best_cert["dL"] <= 0.003:
        best_cert = _lpslack_polish(n, best_cert, inc_dec, time_limit=15.0)
    return best_cert, diag


def _corner_block_masks(X: np.ndarray, Y: np.ndarray, T: np.ndarray, reach: float = 2.6) -> tuple[np.ndarray, np.ndarray]:
    deg = np.degrees(np.mod(T, 0.5 * math.pi))
    til = np.minimum(deg, 90.0 - deg) > 2.0
    if not np.any(til):
        z = np.zeros(len(X), dtype=bool)
        return z, z
    P = np.column_stack([X[til], Y[til]])
    d = np.hypot(X[:, None] - P[None, :, 0], Y[:, None] - P[None, :, 1]).min(axis=1)
    rigid = d >= reach
    upright = np.any((P[None, :, 0] > X[:, None]) & (P[None, :, 1] > Y[:, None]), axis=1)
    downleft = np.any((P[None, :, 0] < X[:, None]) & (P[None, :, 1] < Y[:, None]), axis=1)
    moving = rigid & (~upright)
    frozen = rigid & (~downleft) & (~moving)
    return frozen, moving


def pslp_refine(n: int, cert: dict, inc_dec: Decimal, shake_rounds: int = 3, seed: int = 123) -> dict:
    m = int(math.floor(float(inc_dec)))
    x, y, th = _rows_to_arrays(cert["rows"])
    nx, ny, nth, L0 = bl.normalize(x, y, th)
    f0 = L0 - m
    bx, by, bt, bf = nx, ny, nth, f0
    frozen, moving = _corner_block_masks(bx, by, bt)
    for ctr, fr, mv in ((0.0, None, None), (0.15, frozen if np.any(frozen) else None, moving if np.any(moving) else None)):
        try:
            f1, X1, Y1, T1, v1, _ = pslp.squeeze(bx, by, bt, m, m, bf, max_it=250, center=ctr, frozen=fr, moving=mv)
            if v1 <= 1e-12 and f1 < bf - 1e-10 and pslp.viol(X1, Y1, T1, m, m, f1) <= 1e-12:
                bx, by, bt, bf = X1, Y1, T1, f1
        except Exception:  # noqa: BLE001
            pass
    if shake_rounds > 0:
        rng = np.random.default_rng(seed + n)
        try:
            f2, X2, Y2, T2 = pslp.shake_squeeze(bx, by, bt, m, m, bf, rounds=shake_rounds, rng=rng)
            if f2 < bf - 1e-10 and pslp.viol(X2, Y2, T2, m, m, f2) <= 1e-12:
                bx, by, bt, bf = X2, Y2, T2, f2
        except Exception:  # noqa: BLE001
            pass
    if bf < f0 - 1e-10:
        c2 = bl.polish_and_certify(n, bx, by, bt, inc_dec, slp_rounds=16)
        if c2 is not None and c2["L_float"] < cert["L_float"]:
            c2["label"] = cert.get("label", "") + "+pslp"
            return c2
    return cert


def save_win_locked(win_entry: dict) -> bool:
    return bl.save_win_locked(win_entry, data_dir=DATA_DIR, final_path=FINAL_CSV)
