#!/usr/bin/env python3
"""H109 (peelstep): Unified Cross-Donor Brick Pool + kx,ky in {-1,0,+1} Pinwheel & Masked C++ Melt."""
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

HERE = Path(__file__).resolve().parent
PEELSTEP_DIR = HERE.parent
DATA_DIR = PEELSTEP_DIR / "data"
DOCS_DIR = PEELSTEP_DIR / "docs"
FINAL_CSV = PEELSTEP_DIR / "final.csv"
REPO_ROOT = PEELSTEP_DIR.parents[1]

B8_POC = REPO_ROOT / "poc" / "swarm_b8xplant" / "poc"
AUTO_UNION_POC = REPO_ROOT / "poc" / "auto_union" / "poc"
LPSLACK_POC = REPO_ROOT / "poc" / "swarm_lpslack" / "poc"
CLEANPEEL2_POC = REPO_ROOT / "poc" / "swarm_cleanpeel2" / "poc"

for _p in (B8_POC, AUTO_UNION_POC, LPSLACK_POC, CLEANPEEL2_POC):
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import b8_lib as bl  # noqa: E402
import brick_rect_lib as bslp  # noqa: E402
import cleanpeel2_lib as cp2  # noqa: E402
import lpslack  # noqa: E402
from cpp_wrap import cpp_eval_sat_pen_bbox  # noqa: E402

WORKTREES_ROOT = Path("/home/lucas/.gemini/antigravity-cli/worktrees/a4fb0b93-b63b-426d-aa12-67454e99bc08")
DONOR_BRICK_DIRS = [
    WORKTREES_ROOT / "swarm-cleanpeel2-8a77ac94" / "poc" / "swarm_cleanpeel2" / "data",
    WORKTREES_ROOT / "swarm-stepdown-9d2222ea" / "poc" / "swarm_stepdown" / "data",
    WORKTREES_ROOT / "swarm-gstep-6d3db1b1" / "poc" / "swarm_gstep" / "data",
    WORKTREES_ROOT / "swarm-band3-74558442" / "poc" / "swarm_band3" / "data",
]


def make_brick(w: int, h: int, g: int, x: np.ndarray, y: np.ndarray, th: np.ndarray, tag: str = "") -> dict:
    return cp2.make_brick(w, h, g, x, y, th, tag=tag)


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


def load_all_pooled_bricks(top_per_shape: int = 2) -> list[dict]:
    """Pool all bricks from local data + cleanpeel2 + stepdown + gstep + band3, plus transposes."""
    raw: list[dict] = []
    dirs = [DATA_DIR] + [d for d in DONOR_BRICK_DIRS if d.exists()]
    for d in dirs:
        if not d.exists():
            continue
        for p in sorted(d.glob("brick_*.json")):
            try:
                raw.append(load_brick_json(p))
            except Exception:
                continue

    by_shape: dict[tuple[int, int, int], list[dict]] = {}
    seen_sig: set[tuple[int, int, int, float]] = set()
    for b in raw:
        for cand in (b, bl.transform_brick(b, "T")):
            if cand is not b:
                cand["tag"] = b["tag"] + "_T"
            key = (cand["w"], cand["h"], cand["g"])
            sig = (*key, round(cand["f"], 6))
            if sig in seen_sig:
                continue
            seen_sig.add(sig)
            by_shape.setdefault(key, []).append(cand)

    kept: list[dict] = []
    for key in sorted(by_shape):
        lst = sorted(by_shape[key], key=lambda c: c["f"])
        kept.extend(lst[:top_per_shape])
    return kept


def _drop_inner_corner_square(b: dict, corner: str) -> dict:
    """Drop 1 square nearest the inner corner ('BR' for TL brick, 'TL' for BR brick)."""
    x, y, th = b["x"], b["y"], b["th"]
    W = b["w"] + b["fx"]
    H = b["h"] + b["fy"]
    if corner == "BR":
        d2 = (x - W) ** 2 + y ** 2
    elif corner == "TL":
        d2 = x ** 2 + (y - H) ** 2
    else:
        raise ValueError(corner)
    drop = int(np.argmin(d2))
    keep = np.ones(len(x), dtype=bool)
    keep[drop] = False
    return make_brick(b["w"], b["h"], b["g"] - 1, x[keep], y[keep], th[keep], tag=b["tag"] + f"_drop{corner}")


def build_pinwheel_kxky(
    m: int,
    r: int,
    b1: dict,
    b2: dict,
    sf: float = 0.0,
    f_override: float | None = None,
    drop_mode: str = "none",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray] | None:
    """Assemble TL(b1) + BR(b2) 5-region pinwheel for kx, ky in {-1, 0, +1}.

    - When not (kx == 1 and ky == 1): requires g1 + g2 == r.
    - When kx == 1 and ky == 1:
        - drop_mode == 'none': requires g1 + g2 == r - 1 (all squares kept; 1x1 grid overlap).
        - drop_mode in ('b1_BR', 'b2_TL'): requires g1 + g2 == r (drops 1 inner-corner square).
    """
    w1, h1, w2, h2 = b1["w"], b1["h"], b2["w"], b2["h"]
    kx = w1 + w2 - m
    ky = h1 + h2 - m
    if kx not in (-1, 0, 1) or ky not in (-1, 0, 1):
        return None
    if w1 >= m or h1 >= m or w2 >= m or h2 >= m:
        return None

    b1_use, b2_use = b1, b2
    if kx == 1 and ky == 1:
        if drop_mode == "b1_BR":
            b1_use = _drop_inner_corner_square(b1, "BR")
        elif drop_mode == "b2_TL":
            b2_use = _drop_inner_corner_square(b2, "TL")

    f = max(b1_use["f"], b2_use["f"]) if f_override is None else f_override
    L = m + f
    dx1 = sf * (f - b1_use["fx"])
    dy1 = (m - h1) + (1.0 - sf) * (f - b1_use["fy"])
    dx2 = (m - w2) + (1.0 - sf) * (f - b2_use["fx"])
    dy2 = sf * (f - b2_use["fy"])

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

    ax = np.concatenate([b1_use["x"] + dx1, b2_use["x"] + dx2, np.array(gx, dtype=np.float64)])
    ay = np.concatenate([b1_use["y"] + dy1, b2_use["y"] + dy2, np.array(gy, dtype=np.float64)])
    at = np.concatenate([b1_use["th"], b2_use["th"], np.zeros(len(gx), dtype=np.float64)])
    if len(ax) != m * m + r:
        return None

    seam = (
        ((ax >= w1 - 1.8) & (ax <= w1 + f + 2.0))
        | ((ay >= m - h1 - 2.0) & (ay <= m - h1 + 2.0))
        | ((ax >= m - w2 - 2.0) & (ax <= m - w2 + 2.0))
        | ((ay >= h2 - 1.8) & (ay <= h2 + f + 2.0))
        | ((np.abs(ax - w1) <= 3.4) & (np.abs(ay - (m - h1)) <= 3.4))
        | ((np.abs(ax - (m - w2)) <= 3.4) & (np.abs(ay - h2) <= 3.4))
    ).astype(np.uint8)
    return ax, ay, at, L, seam


def pick_pairs_for_n(
    n: int,
    bricks: list[dict],
    f_inc: float,
    max_excess: float = 0.005,
    top_k: int = 8,
) -> list[tuple[float, dict, dict, str]]:
    """Select diverse top pinwheel pairs with kx,ky in {-1,0,+1} and max(f1,f2) <= f_inc + max_excess."""
    m = int(math.isqrt(n))
    r = n - m * m
    best_by_shape: dict[tuple, tuple] = {}

    for i, b1 in enumerate(bricks):
        if b1["w"] >= m or b1["h"] >= m or b1["f"] > f_inc + max_excess:
            continue
        for j, b2 in enumerate(bricks):
            if i > j:
                continue
            if b2["w"] >= m or b2["h"] >= m or b2["f"] > f_inc + max_excess:
                continue
            kx = b1["w"] + b2["w"] - m
            ky = b1["h"] + b2["h"] - m
            if kx not in (-1, 0, 1) or ky not in (-1, 0, 1):
                continue

            g_sum = b1["g"] + b2["g"]
            if kx == 1 and ky == 1:
                if g_sum == r - 1:
                    drop_modes = ("none",)
                elif g_sum == r:
                    drop_modes = ("b1_BR",)
                else:
                    continue
            else:
                if g_sum != r:
                    continue
                drop_modes = ("none",)

            f_hi = max(b1["f"], b2["f"])
            f_lo = min(b1["f"], b2["f"])
            ov_pen = 0.0
            if kx == 1 and ky == 1:
                ov_pen = 0.030 if drop_modes[0] == "none" else 0.025
            elif kx == 1 or ky == 1:
                ov_pen = 0.002
            elif kx + ky < 0:
                ov_pen = -0.0005  # zero-ov gap slightly easier than exact touch
            score = 0.75 * f_hi + 0.25 * f_lo + ov_pen

            k1 = (b1["w"], b1["h"], b1["g"])
            k2 = (b2["w"], b2["h"], b2["g"])
            t1 = (b1["h"], b1["w"], b1["g"])
            t2 = (b2["h"], b2["w"], b2["g"])
            canon_shape = min((k1, k2), (k2, k1), (t1, t2), (t2, t1))

            for dm in drop_modes:
                sk = (canon_shape, tuple(sorted((kx, ky))), dm)
                cand = (score, f_hi, b1, b2, dm)
                if sk not in best_by_shape or (score, f_hi) < best_by_shape[sk][:2]:
                    best_by_shape[sk] = cand

    ranked = sorted(best_by_shape.values(), key=lambda t: (t[0], t[1]))
    has_non11 = any(
        not (b1["w"] + b2["w"] - m == 1 and b1["h"] + b2["h"] - m == 1)
        for _, _, b1, b2, _ in ranked
    )
    # Keep up to top_k, capping identical (kx, ky) geometry class slightly so we get variety
    out: list[tuple[float, dict, dict, str]] = []
    kxky_cnt: dict[tuple[int, int], int] = {}
    for score, _, b1, b2, dm in ranked:
        kx = b1["w"] + b2["w"] - m
        ky = b1["h"] + b2["h"] - m
        if has_non11 and kx == 1 and ky == 1:
            continue
        if kxky_cnt.get((kx, ky), 0) >= max(3, top_k // 2):
            continue
        kxky_cnt[(kx, ky)] = kxky_cnt.get((kx, ky), 0) + 1
        out.append((score, b1, b2, dm))
        if len(out) >= top_k:
            break
    return out


def run_lpslack_polish(n: int, cert: dict, inc_dec: Decimal) -> dict:
    """Apply lpslack optimization on a certified pack and return improved cert if better."""
    if not cert or not cert.get("rows"):
        return cert
    base_rows = [
        {
            "square_id": str(row["square_id"]),
            "x": str(row["x"]),
            "y": str(row["y"]),
            "angle": str(row["angle"]),
        }
        for row in cert["rows"]
    ]
    target_ref = min(inc_dec, Decimal(cert["L_lit"]), Decimal(cert["L_pandas"]))
    lp_res = lpslack.optimize_group(
        n,
        base_rows,
        str(target_ref),
        outer_iters=6,
        fixed_rounds=8,
        slp_modes=("dual_non90", "non90"),
        tr_th_schedule=(4e-4, 1e-4),
        time_limit=12.0,
    )
    if lp_res.get("status") in ("WIN", "TINY_WIN") and lp_res.get("rows"):
        lit_l = lp_res["literal_L"]
        pan_l = lp_res["pandas_L"]
        L_worst = max(Decimal(lit_l), Decimal(pan_l))
        if L_worst < max(Decimal(cert["L_lit"]), Decimal(cert["L_pandas"])):
            dL_true = L_worst - inc_dec
            return {
                "ok": True,
                "n": int(n),
                "L_lit": lit_l,
                "L_pandas": pan_l,
                "L_dec_str": str(L_worst),
                "L_float": float(L_worst),
                "dL": float(dL_true),
                "is_real_win": bool(dL_true <= Decimal("-1e-9")),
                "label": f"{cert['label']}+lpslack",
                "rows": [
                    {
                        "square_id": int(row["square_id"]),
                        "x": str(row["x"]),
                        "y": str(row["y"]),
                        "angle": str(row["angle"]),
                    }
                    for row in lp_res["rows"]
                ],
            }
    return cert


def solve_pinwheel_pair(
    n: int,
    m: int,
    r: int,
    b1_base: dict,
    b2_base: dict,
    drop_mode: str,
    inc_dec: Decimal,
    tag: str,
    base_seed: int,
    n_threads: int = 2,
) -> tuple[dict | None, list]:
    """Direct SLP (if zero-ov) + masked C++ corner/seam melt + compress_slp + lpslack."""
    best_cert: dict | None = None
    seeds, masks, scheds, labels = [], [], [], []
    inc_f = float(inc_dec)
    kx = b1_base["w"] + b2_base["w"] - m
    ky = b1_base["h"] + b2_base["h"] - m
    is_zero_ov = (kx <= 0 and ky <= 0)

    drop_cands = ("b1_BR", "b2_TL") if drop_mode == "b1_BR" else (drop_mode,)

    for dm in drop_cands:
        for m1 in ("id", "flipx", "flipy", "rot180"):
            b1 = bl.transform_brick(b1_base, m1)
            for m2 in ("id", "flipx", "flipy", "rot180"):
                b2 = bl.transform_brick(b2_base, m2)
                f_hi = max(b1["fx"], b1["fy"], b2["fx"], b2["fy"])
                f_avg = 0.5 * (max(b1["fx"], b1["fy"]) + max(b2["fx"], b2["fy"]))
                f_opts = ((f_hi, "fhi"),) if abs(b1["f"] - b2["f"]) < 5e-4 else ((f_hi, "fhi"), (f_avg, "favg"))
                for f_use, ftag in f_opts:
                    built = build_pinwheel_kxky(m, r, b1, b2, sf=0.0, f_override=f_use, drop_mode=dm)
                    if built is None:
                        continue
                    ax, ay, at, L_tgt, sm = built
                    if is_zero_ov and ftag == "fhi" and (m1, m2) in (("id", "id"), ("flipx", "flipy"), ("rot180", "rot180")):
                        pen, _ = cpp_eval_sat_pen_bbox(ax, ay, at)
                        if pen <= 1e-10:
                            sx, sy, st, _, ok = bslp.joint_xytheta_slp_tight(ax, ay, at)
                            use_x, use_y, use_t = (sx, sy, st) if ok else (ax, ay, at)
                            cert = bl.polish_and_certify(n, use_x, use_y, use_t, inc_dec, slp_rounds=16)
                            if cert is not None:
                                cert["label"] = f"{tag}_{m1}_{m2}_direct"
                                if best_cert is None or cert["L_float"] < best_cert["L_float"]:
                                    best_cert = cert

                    if ftag == "fhi" or len(seeds) < 24:
                        if inc_f - L_tgt > 0.005:
                            L_end = min(inc_f - 0.001, max(L_tgt + 0.001, inc_f - 0.004))
                            L_mid = max(L_tgt + 0.008, L_end + 0.006)
                            L_start = max(L_tgt + 0.022, L_end + 0.016)
                        else:
                            L_end = min(inc_f - 3e-4, L_tgt + 0.001) if L_tgt + 0.001 < inc_f else 0.5 * (L_tgt + inc_f)
                            L_mid = L_tgt + 0.005
                            L_start = L_tgt + 0.018
                        seeds.append((ax, ay, at))
                        masks.append(sm)
                        scheds.append([L_start, L_mid, L_end])
                        labels.append(f"{tag}_{dm}_{m1}_{m2}_{ftag}")

    diag = []
    if seeds:
        # Cap at 24 seeds per pair for fast 30-min throughput
        if len(seeds) > 24:
            seeds, masks, scheds, labels = seeds[:24], masks[:24], scheds[:24], labels[:24]
        relaxed = bl.su.cpp_batch_melt(
            seeds,
            scheds,
            sym_mode=1,
            a_start=0.962,
            n_melt_stages=3,
            seam_masks=masks,
            seam_iters=220,
            full_iters=260,
            n_ruin_iters=0,
            n_threads=n_threads,
            base_seed=base_seed,
        )
        pool = []
        for bi, row in enumerate(relaxed):
            nx, ny, nth, Lm = bl.normalize(row["x"], row["y"], row["th"])
            lbl = f"{labels[bi]}_melt"
            if row["best_L_feas"] is not None:
                pool.append((0.0, Lm, float(row["pen"]), nx, ny, nth, lbl))
            else:
                pool.append((1.0 + float(row["pen"]), Lm, float(row["pen"]), nx, ny, nth, lbl))
        pool.sort(key=lambda t: (t[0], t[1]))
        diag = [(t[0] == 0.0, round(t[1], 6), round(t[2], 6), t[6]) for t in pool[:3]]

        for _, _, pen, cx, cy, cth, lbl in pool[:3]:
            if pen > 0.03:
                continue
            sx, sy, st, _, ok = bslp.joint_xytheta_slp_tight(cx, cy, cth)
            use_x, use_y, use_t = (sx, sy, st) if ok else (cx, cy, cth)
            cert = bl.polish_and_certify(n, use_x, use_y, use_t, inc_dec, slp_rounds=18)
            if cert is not None:
                cert["label"] = lbl
                if best_cert is None or cert["L_float"] < best_cert["L_float"]:
                    best_cert = cert

    if best_cert is not None and best_cert["L_float"] < inc_f + 0.008:
        best_cert = run_lpslack_polish(n, best_cert, inc_dec)

    return best_cert, diag


def save_win_locked(win_entry: dict) -> bool:
    return bl.save_win_locked(win_entry, data_dir=DATA_DIR, final_path=FINAL_CSV)
