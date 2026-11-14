#!/usr/bin/env python3
"""H94 (band3): Brick harvesting, synthesis, and asymmetric/corner-melted pinwheel builder."""
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
BAND3_DIR = HERE.parent
DATA_DIR = BAND3_DIR / "data"
DOCS_DIR = BAND3_DIR / "docs"
FINAL_CSV = BAND3_DIR / "final.csv"
REPO_ROOT = BAND3_DIR.parents[1]

B8_POC = REPO_ROOT / "poc" / "swarm_b8xplant" / "poc"
AUTO_UNION_POC = REPO_ROOT / "poc" / "auto_union" / "poc"
LPSLACK_POC = REPO_ROOT / "poc" / "swarm_lpslack" / "poc"
GSTEP_WORKTREE_POC = Path(
    "/home/lucas/.gemini/antigravity-cli/worktrees/a4fb0b93-b63b-426d-aa12-67454e99bc08/swarm-gstep-6d3db1b1/poc/swarm_gstep/poc"
)

for _p in (B8_POC, AUTO_UNION_POC, LPSLACK_POC, GSTEP_WORKTREE_POC):
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import b8_lib as bl  # noqa: E402
import brick_rect_lib as bslp  # noqa: E402
import gstep_lib as gl  # noqa: E402
import lpslack  # noqa: E402
from cpp_wrap import cpp_eval_sat_pen_bbox  # noqa: E402


EXTERNAL_BRICK_DIRS = [
    Path("/home/lucas/.gemini/antigravity-cli/worktrees/a4fb0b93-b63b-426d-aa12-67454e99bc08/swarm-gstep-6d3db1b1/poc/swarm_gstep/data"),
    Path("/home/lucas/.gemini/antigravity-cli/worktrees/a4fb0b93-b63b-426d-aa12-67454e99bc08/swarm-b7xplant-a586260b/poc/swarm_b7xplant/data"),
    Path("/home/lucas/.gemini/antigravity-cli/worktrees/a4fb0b93-b63b-426d-aa12-67454e99bc08/swarm-cleanpeel2-8a77ac94/poc/swarm_cleanpeel2/data"),
    REPO_ROOT / "poc" / "swarm_b8xplant" / "data",
]


def make_brick(w: int, h: int, g: int, x: np.ndarray, y: np.ndarray, th: np.ndarray, tag: str = "") -> dict:
    bx, by, W, H = bl.rect_bbox(x, y, th)
    fx, fy = float(W - w), float(H - h)
    return {
        "w": int(w),
        "h": int(h),
        "g": int(g),
        "x": bx.copy(),
        "y": by.copy(),
        "th": np.asarray(th, dtype=np.float64).copy(),
        "fx": fx,
        "fy": fy,
        "f": max(fx, fy),
        "tag": tag,
    }


def save_brick_json(b: dict, name: str) -> Path:
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


def load_brick_json(path: Path, tag: str = "") -> dict:
    d = json.loads(path.read_text(encoding="utf-8"))
    return make_brick(
        int(d["w"]),
        int(d["h"]),
        int(d["g"]),
        np.array(d["x"], dtype=np.float64),
        np.array(d["y"], dtype=np.float64),
        np.array(d["th"], dtype=np.float64),
        tag=tag or d.get("name", path.stem.replace("brick_", "")),
    )


def harvest_incumbent_bricks(min_g: int = 6, max_f: float = 0.80, save: bool = True) -> list[dict]:
    """Harvest minimal-perimeter high-g corner bricks from merged.csv (n >= 400)."""
    df = pd.read_csv(bl.MERGED_CSV, usecols=["n", "square_id", "x", "y", "angle"])
    best_by_key: dict[tuple[int, int, int], dict] = {}

    for n_val, grp in df[df["n"] >= 400].groupby("n"):
        n = int(n_val)
        gs = grp.sort_values("square_id")
        x, y, th, L = bl.normalize(
            gs["x"].to_numpy(float, copy=True),
            gs["y"].to_numpy(float, copy=True),
            gs["angle"].to_numpy(float, copy=True),
        )
        m = int(math.floor(L))
        r = n - m * m
        f = L - m
        if r < 8 or f > max_f + 0.02:
            continue
        deg = np.degrees(np.mod(th, 0.5 * math.pi))
        tilt = np.minimum(deg, 90.0 - deg) > 2.0
        hx = bl.half_extent(th)

        for corner in ("TL", "BR", "BL", "TR"):
            cand_for_corner: list[tuple[int, int, int, int, float, np.ndarray]] = []
            for w in range(7, min(m, 25)):
                for h in range(7, min(m, 25)):
                    if w + h > 36:
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
                    if g < min_g or int((mask & tilt).sum()) < 30:
                        continue
                    _, _, W, H = bl.rect_bbox(x[mask], y[mask], th[mask])
                    fc = max(W - w, H - h)
                    if fc <= min(max_f, f + 0.002):
                        cand_for_corner.append((g, w + h, w, h, fc, mask))
            if not cand_for_corner:
                continue
            # Keep minimal (w+h) achieving each g >= min_g in this corner
            by_g: dict[int, tuple] = {}
            for item in cand_for_corner:
                g = item[0]
                if g not in by_g or (item[1], item[4]) < (by_g[g][1], by_g[g][4]):
                    by_g[g] = item
            for g, (_, _, w, h, fc, mask) in by_g.items():
                b = make_brick(w, h, g, x[mask], y[mask], th[mask], tag=f"B{g}_{w}x{h}_n{n}_{corner}")
                if w > h:
                    b = bl.transform_brick(b, "T")
                    b["tag"] = f"B{g}_{b['w']}x{b['h']}_n{n}_{corner}T"
                k = (b["w"], b["h"], b["g"])
                if k not in best_by_key or b["f"] < best_by_key[k]["f"] - 1e-9:
                    best_by_key[k] = b

    out = list(best_by_key.values())
    if save:
        for b in out:
            save_brick_json(b, b["tag"])
    return out


def load_all_bricks(include_external: bool = True) -> list[dict]:
    """Load all local + external bricks, deduplicate by (w, h, g), and include transposes."""
    raw: list[dict] = []
    for p in sorted(DATA_DIR.glob("brick_*.json")):
        raw.append(load_brick_json(p))
    if include_external:
        for d in EXTERNAL_BRICK_DIRS:
            if not d.exists():
                continue
            for p in sorted(d.glob("*.json")):
                try:
                    jd = json.loads(p.read_text(encoding="utf-8"))
                    if all(k in jd for k in ("w", "h", "g", "x", "y", "th")):
                        raw.append(load_brick_json(p, tag=jd.get("name", p.stem.replace("brick_", ""))))
                except Exception:
                    continue

    by_shape: dict[tuple[int, int, int], list[dict]] = {}
    seen_tags: set[tuple[int, int, int, str]] = set()
    for b in raw:
        for cand in (b, bl.transform_brick(b, "T")):
            if cand is not b:
                cand["tag"] = b["tag"] + "_T"
            key = (cand["w"], cand["h"], cand["g"])
            tk = (*key, cand["tag"])
            if tk in seen_tags:
                continue
            seen_tags.add(tk)
            by_shape.setdefault(key, []).append(cand)

    kept: list[dict] = []
    for key, lst in sorted(by_shape.items()):
        lst.sort(key=lambda c: c["f"])
        kept.extend(lst[:2])
    return kept


def build_pinwheel(
    m: int,
    b1: dict,
    b2: dict,
    sf: float = 0.0,
    f_override: float | None = None,
    max_ov: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray] | None:
    """Build TL(b1) + BR(b2) pinwheel allowing zero-overlap or corner/1-row seam overlap."""
    w1, h1, w2, h2 = b1["w"], b1["h"], b2["w"], b2["h"]
    if w1 >= m or h1 >= m or w2 >= m or h2 >= m:
        return None
    if w1 + w2 > m + max_ov or h1 + h2 > m + max_ov:
        return None
    # Cannot overlap both in x and y simultaneously by >0 grid cells
    if w1 + w2 > m and h1 + h2 > m:
        return None
    f = max(b1["f"], b2["f"]) if f_override is None else f_override
    L = m + f
    dx1 = sf * (f - b1["fx"])
    dy1 = (m - h1) + (1.0 - sf) * (f - b1["fy"])
    dx2 = (m - w2) + (1.0 - sf) * (f - b2["fx"])
    dy2 = sf * (f - b2["fy"])

    gx, gy = [], []
    for i in range(m):
        for j in range(m):
            if i < w1 and j >= m - h1:
                continue
            if i >= m - w2 and j < h2:
                continue
            if i < m - w2 and j < m - h1:
                gx.append(i + 0.5)
                gy.append(j + 0.5)
            else:
                gx.append(i + f + 0.5)
                gy.append(j + f + 0.5)

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
    return ax, ay, at, L, seam


def solve_pair(
    n: int,
    m: int,
    b1_base: dict,
    b2_base: dict,
    inc_dec: Decimal,
    tag: str,
    base_seed: int,
    max_ov: int = 1,
    do_melt: bool = True,
    do_lpslack: bool = True,
) -> dict | None:
    best_cert: dict | None = None
    seeds, masks, scheds, labels = [], [], [], []
    w1, h1, w2, h2 = b1_base["w"], b1_base["h"], b2_base["w"], b2_base["h"]
    is_zero_ov = (w1 + w2 <= m) and (h1 + h2 <= m) and (w1 + w2 + h1 + h2 <= 2 * m - 1)

    for m1 in ("id", "flipx", "flipy", "rot180"):
        b1 = bl.transform_brick(b1_base, m1)
        for m2 in ("id", "flipx", "flipy", "rot180"):
            b2 = bl.transform_brick(b2_base, m2)
            f_hi = max(b1["fx"], b1["fy"], b2["fx"], b2["fy"])
            f_lo = min(b1["fx"], b1["fy"], b2["fx"], b2["fy"])
            f_avg = 0.5 * (max(b1["fx"], b1["fy"]) + max(b2["fx"], b2["fy"]))
            f_list = ((f_hi, "fhi"),) if abs(f_hi - f_lo) < 5e-4 else ((f_lo, "flo"), (f_avg, "favg"), (f_hi, "fhi"))
            for f_use, ftag in f_list:
                built = build_pinwheel(m, b1, b2, sf=0.0, f_override=f_use, max_ov=max_ov)
                if built is None:
                    continue
                ax, ay, at, L_tgt, sm = built
                if is_zero_ov and ftag == "fhi" and (m1, m2) in (("id", "id"), ("flipx", "flipy")):
                    pen, _ = cpp_eval_sat_pen_bbox(ax, ay, at)
                    if pen <= 1e-10:
                        cert = bl.polish_and_certify(n, ax, ay, at, inc_dec, slp_rounds=8)
                        if cert is not None:
                            cert["label"] = f"{tag}_{m1}_{m2}_direct"
                            if best_cert is None or cert["L_float"] < best_cert["L_float"]:
                                best_cert = cert
                if do_melt and ftag in ("favg", "fhi"):
                    seeds.append((ax, ay, at))
                    masks.append(sm)
                    scheds.append([L_tgt + 0.010, L_tgt + 0.002, L_tgt - 4e-4])
                    labels.append(f"{tag}_{m1}_{m2}_{ftag}")

    if do_melt and seeds:
        for start in range(0, len(seeds), 32):
            end = min(len(seeds), start + 32)
            relaxed = bl.su.cpp_batch_melt(
                seeds[start:end],
                scheds[start:end],
                sym_mode=1,
                a_start=0.965,
                n_melt_stages=3,
                seam_masks=masks[start:end],
                seam_iters=220,
                full_iters=260,
                n_ruin_iters=0,
                n_threads=2,
                base_seed=base_seed + start,
            )
            pool = []
            for bi, row in enumerate(relaxed):
                nx, ny, nth, Lm = bl.normalize(row["x"], row["y"], row["th"])
                lbl = f"{labels[start + bi]}_melt"
                if row["best_L_feas"] is not None:
                    pool.append((0.0, Lm, nx, ny, nth, lbl))
                elif row["pen"] < 0.02:
                    pool.append((1.0 + float(row["pen"]), Lm, nx, ny, nth, lbl))
            pool.sort(key=lambda t: (t[0], t[1]))
            for _, _, cx, cy, cth, lbl in pool[:2]:
                cert = bl.polish_and_certify(n, cx, cy, cth, inc_dec, slp_rounds=12)
                if cert is not None:
                    cert["label"] = lbl
                    if best_cert is None or cert["L_float"] < best_cert["L_float"]:
                        best_cert = cert

    if do_lpslack and best_cert is not None:
        rs = sorted(best_cert["rows"], key=lambda r: int(r["square_id"]))
        res_lp = lpslack.optimize_group(
            n,
            rs,
            str(inc_dec),
            outer_iters=5,
            fixed_rounds=8,
            slp_modes=("dual_non90", "non90"),
            tr_th_schedule=(4e-4, 1e-4),
            time_limit=10.0,
        )
        if res_lp.get("win") and res_lp.get("rows"):
            lit, pan = Decimal(res_lp["new_L_lit"]), Decimal(res_lp["new_L_pan"])
            worst = max(lit, pan)
            if worst < max(Decimal(best_cert["L_lit"]), Decimal(best_cert["L_pandas"])):
                best_cert["L_lit"] = str(lit)
                best_cert["L_pandas"] = str(pan)
                best_cert["L_dec_str"] = str(worst)
                best_cert["L_float"] = float(worst)
                best_cert["dL"] = float(worst - inc_dec)
                best_cert["is_real_win"] = bool(worst - inc_dec <= Decimal("-1e-9"))
                best_cert["rows"] = res_lp["rows"]
                best_cert["label"] = best_cert["label"] + "+lpslack"
    return best_cert


def save_win_locked(win_entry: dict) -> bool:
    return bl.save_win_locked(win_entry, data_dir=DATA_DIR, final_path=FINAL_CSV)
