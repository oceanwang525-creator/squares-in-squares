#!/usr/bin/env python3
"""H40: Harvest & Broadcast B5_10x19 (f=0.642571, g=+5) & B4_17x9 (f=0.642571, g=+4) Sub-Bricks."""
from __future__ import annotations

import csv
import fcntl
import json
import math
import multiprocessing as mp
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
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[3]
BRICKSLP_POC = REPO / "poc" / "swarm_brickslp" / "poc"
HIGHD_POC = REPO / "poc" / "swarm_highd" / "poc"
HOMO_POC = REPO / "poc" / "swarm_homology" / "poc"
NEW159_POC = REPO / "poc" / "swarm_new159" / "poc"
GAD_POC = REPO / "poc" / "rectangle_gadgets" / "poc"
UNION_POC = REPO / "poc" / "auto_union" / "poc"
MECH_POC = REPO / "poc" / "swarm_mechwalk" / "poc"

for _p in (UNION_POC, HOMO_POC, HIGHD_POC, NEW159_POC, GAD_POC, BRICKSLP_POC, MECH_POC):
    if str(_p) not in sys.path:
        sys.path.append(str(_p))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import brick_slp  # noqa: E402
import frontier_io  # noqa: E402
import mechwalk_lib as ml  # noqa: E402
import union as auto_union  # noqa: E402
import utils as u  # noqa: E402
from cpp_wrap import cpp_eval_sat_pen_bbox, cpp_expand_sym, full_to_free_sym, run_cpp_masked_melt  # noqa: E402

B5_DIR = REPO / "poc" / "swarm_b5xplant"
DATA_DIR = B5_DIR / "data"
DOCS_DIR = B5_DIR / "docs"
FINAL_CSV = B5_DIR / "final.csv"
MERGED_CSV = REPO / "poc" / "pell_corrugation" / "data" / "border-scan" / "merged.csv"

WIN_N850_PATH = DATA_DIR / "donor_win_n850.json"
H32_WIN_N910_PATH = DATA_DIR / "donor_h32_win_n910.json"
BRICKSLP_DATA = REPO / "poc" / "swarm_brickslp" / "data"


def fast_load_frontier(
    target_ns: list[int] | None = None,
) -> tuple[dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, float]], dict[int, Decimal]]:
    """Fast load origin/main frontier from merged.csv + poc/*/final.csv (<2s)."""
    packs: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, float]] = {}
    best_L_dec: dict[int, Decimal] = {}
    need = set(int(n) for n in target_ns) if target_ns else set()

    df = pd.read_csv(MERGED_CSV, usecols=["n", "square_id", "x", "y", "angle"])
    for n_val, sub in df.groupby("n"):
        n = int(n_val)
        sub = sub.sort_values("square_id")
        x, y, th, L = frontier_io.norm_pack(
            sub["x"].to_numpy(np.float64),
            sub["y"].to_numpy(np.float64),
            sub["angle"].to_numpy(np.float64),
        )
        packs[n] = (x, y, th, L)
        best_L_dec[n] = Decimal(f"{L:.17g}")

    if need:
        raw_merged = auto_union.parse_text(MERGED_CSV.read_text(encoding="utf-8"))
        for n in need:
            if n in raw_merged:
                best_L_dec[n] = Decimal(auto_union.certify(n, raw_merged[n]))

    for fp in sorted((REPO / "poc").glob("*/final.csv")):
        if fp.parent.name == "swarm_b5xplant" or fp.stat().st_size <= 20:
            continue
        groups: dict[int, list[tuple[int, float, float, float]]] = {}
        try:
            with fp.open(encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    n_s, sid_s = row["id"].split("_")
                    n = int(n_s)
                    groups.setdefault(n, []).append(
                        (int(sid_s), float(row["x"].lstrip("s")), float(row["y"].lstrip("s")), math.radians(float(row["deg"].lstrip("s"))))
                    )
        except Exception:
            continue

        # Only call slow mpmath parse_text if at least one n beats float64 L
        need_cert: list[int] = []
        for n, rows in groups.items():
            if len(rows) != n or n not in packs:
                continue
            rows.sort(key=lambda z: z[0])
            nx, ny, nt, L = frontier_io.norm_pack(
                np.array([z[1] for z in rows], dtype=np.float64),
                np.array([z[2] for z in rows], dtype=np.float64),
                np.array([z[3] for z in rows], dtype=np.float64),
            )
            if n in need and L < packs[n][3] + 1e-9:
                need_cert.append(n)
            elif L < packs[n][3] - 1e-12:
                packs[n] = (nx, ny, nt, L)
                best_L_dec[n] = Decimal(f"{L:.17g}")

        if need_cert:
            try:
                lit, pd_t = auto_union.load(fp)
                for n in need_cert:
                    if n in lit and n in pd_t:
                        res = auto_union.check((n, lit[n], pd_t[n]))
                        if res is not None:
                            Ld = max(Decimal(res[0]), Decimal(res[1]))
                            if Ld < best_L_dec[n]:
                                best_L_dec[n] = Ld
                                r_s = sorted(lit[n], key=lambda r: int(r["square_id"]))
                                packs[n] = frontier_io.norm_pack(
                                    np.array([float(r["x"]) for r in r_s]),
                                    np.array([float(r["y"]) for r in r_s]),
                                    np.array([float(r["angle"]) for r in r_s]),
                                )
            except Exception:
                pass

    return packs, best_L_dec


def make_brick(w: int, h: int, g: int, x: np.ndarray, y: np.ndarray, th: np.ndarray, src_n: int, tag: str) -> dict | None:
    if g <= 0 or w <= 0 or h <= 0 or len(x) != w * h + g:
        return None
    nx, ny, W, H = brick_slp.rect_bbox(x, y, th)
    pen, _ = cpp_eval_sat_pen_bbox(nx, ny, th)
    if pen > 1e-7:
        return None
    fx, fy = W - w, H - h
    f = max(fx, fy)
    if f >= 1.0 - 1e-7:
        return None
    return {
        "w": int(w), "h": int(h), "g": int(g),
        "f": float(f), "fx": float(fx), "fy": float(fy),
        "f_min": float(min(fx, fy)),
        "src_n": int(src_n), "tag": tag,
        "x": nx.copy(), "y": ny.copy(), "th": th.copy(),
    }


def transpose_brick(b: dict) -> dict:
    """Transpose (x, y, th) -> (y, x, pi/2 - th), swapping (w, h) and (fx, fy)."""
    nx = b["y"].copy()
    ny = b["x"].copy()
    nt = (0.5 * math.pi - b["th"]).copy()
    nx, ny, W, H = brick_slp.rect_bbox(nx, ny, nt)
    w, h = b["h"], b["w"]
    fx, fy = W - w, H - h
    return {
        "w": w, "h": h, "g": b["g"],
        "f": float(max(fx, fy)), "fx": float(fx), "fy": float(fy),
        "f_min": float(min(fx, fy)),
        "src_n": b["src_n"], "tag": b["tag"] + "_T",
        "x": nx, "y": ny, "th": nt,
    }


def sym_brick(b: dict, s: int) -> dict:
    """Apply D2 reflection/180-rotation s in {0,1,2,3} preserving (w, h)."""
    x, y, th = b["x"].copy(), b["y"].copy(), b["th"].copy()
    w, h = b["w"], b["h"]
    W, H = w + b["fx"], h + b["fy"]
    if s & 1:
        x = W - x
        th = -th
    if s & 2:
        y = H - y
        th = -th
    nx, ny, W2, H2 = brick_slp.rect_bbox(x, y, th)
    fx, fy = W2 - w, H2 - h
    return dict(b, x=nx, y=ny, th=th, fx=float(fx), fy=float(fy), f=float(max(fx, fy)), f_min=float(min(fx, fy)))


def compress_brick_rect(b: dict, n_trials: int = 12, seed: int = 42) -> dict | None:
    """Compress a brick in rectangular box using boundary-melt + joint_xytheta_slp_rect."""
    w, h, g = b["w"], b["h"], b["g"]
    x0, y0, t0 = b["x"], b["y"], b["th"]
    W0, H0 = w + b["fx"], h + b["fy"]
    f0 = b["f"]
    rng = np.random.default_rng(seed)
    best = None
    tilt = np.minimum(np.mod(t0, 0.5 * math.pi), 0.5 * math.pi - np.mod(t0, 0.5 * math.pi)) > math.radians(2.0)

    for trial in range(n_trials):
        cx, cy, ct = x0.copy(), y0.copy(), t0.copy()
        if trial > 0:
            if trial % 3 == 1:
                mask = ~tilt
            elif trial % 3 == 2:
                mask = (cy < 2.8) | (cy > H0 - 2.8) | (cx < 2.8) | (cx > W0 - 2.8)
            else:
                mask = np.ones(len(cx), dtype=bool)
            nm = int(mask.sum())
            if nm > 0:
                amp = 0.12 if trial % 3 != 0 else 0.05
                cx[mask] += rng.uniform(-amp, amp, nm)
                cy[mask] += rng.uniform(-amp, amp, nm)
                ct[mask] += rng.uniform(-2.0 * amp, 2.0 * amp, nm)
        for ft in (max(0.68, f0 - 0.15), max(0.655, f0 - 0.22), 0.645):
            cx, cy, ct, _ = brick_slp.relax_rect_lbfgs(cx, cy, ct, w + ft, h + ft, maxiter=350)
        sx, sy, st, sf, ok = brick_slp.joint_xytheta_slp_rect(cx, cy, ct, w, h, rounds=12, tr_th0=0.02, tr_xy0=0.09, tol=1e-11)
        if ok:
            cb = make_brick(w, h, g, sx, sy, st, b["src_n"], b["tag"] + "_slp")
            if cb is not None and (best is None or cb["f"] < best["f"]):
                best = cb
    return best


def load_b5_bank(packs: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, float]], compress_b5_b4: bool = True) -> list[dict]:
    """Load B5_10x19, B4_17x9/16x9/15x9, B6_10x21, B4_19x9, and frontier complementary bricks."""
    bank: list[dict] = []

    def _add_pair(b: dict | None) -> None:
        if b is None:
            return
        bank.append(b)
        bank.append(transpose_brick(b))

    # 1. Harvest B5_10x20 (f=0.642572), B4_18x9 (f=0.642572), and 1-row/col stripped variants from win_n850.json
    if WIN_N850_PATH.exists():
        rec = json.loads(WIN_N850_PATH.read_text())
        rows = rec["rows"]
        x = np.array([float(r["x"]) for r in rows], dtype=np.float64)
        y = np.array([float(r["y"]) for r in rows], dtype=np.float64)
        th = np.array([float(r["angle"]) for r in rows], dtype=np.float64)
        x, y, th, _ = frontier_io.norm_pack(x, y, th)
        h = frontier_io.hx_of(th)

        m_b5_20 = (x + h <= 10.65) & (y - h >= 8.999)
        b5_20 = make_brick(10, 20, 5, x[m_b5_20], y[m_b5_20], th[m_b5_20], 850, "B5_10x20_n850")
        _add_pair(b5_20)

        m_b4_18 = (x - h >= 10.999) & (y + h <= 9.65)
        b4_18 = make_brick(18, 9, 4, x[m_b4_18], y[m_b4_18], th[m_b4_18], 850, "B4_18x9_n850")
        _add_pair(b4_18)

        m_b5 = (x + h <= 10.6426) & (y - h >= 9.6425)
        b5 = make_brick(10, 19, 5, x[m_b5], y[m_b5], th[m_b5], 850, "B5_10x19_n850")
        _add_pair(b5)

        m_b4 = (x - h >= 11.6425) & (y + h <= 9.6426)
        b4_17 = make_brick(17, 9, 4, x[m_b4], y[m_b4], th[m_b4], 850, "B4_17x9_n850")
        _add_pair(b4_17)

        if b4_17 is not None:
            h2 = frontier_io.hx_of(b4_17["th"])
            for w_sub, cut, tag in [(16, 16.92, "B4_16x9_n850"), (15, 16.0, "B4_15x9_n850")]:
                sub = (b4_17["x"] + h2) <= cut
                b_sub = make_brick(w_sub, 9, 4, b4_17["x"][sub], b4_17["y"][sub], b4_17["th"][sub], 850, tag)
                _add_pair(b_sub)

        if compress_b5_b4:
            cache_path = DATA_DIR / "compressed_b5_b4_cache.json"
            if cache_path.exists():
                cdata = json.loads(cache_path.read_text())
                for item in cdata:
                    cb = make_brick(
                        item["w"], item["h"], item["g"],
                        np.array(item["x"], dtype=np.float64),
                        np.array(item["y"], dtype=np.float64),
                        np.array(item["th"], dtype=np.float64),
                        item["src_n"], item["tag"],
                    )
                    _add_pair(cb)
            else:
                DATA_DIR.mkdir(parents=True, exist_ok=True)
                cdata = []
                for idx_c, raw_b in enumerate([b5, b4_17]):
                    if raw_b is None:
                        continue
                    cb = compress_brick_rect(raw_b, n_trials=12, seed=4200 + idx_c * 31)
                    if cb is not None:
                        _add_pair(cb)
                        cdata.append({
                            "w": cb["w"], "h": cb["h"], "g": cb["g"],
                            "src_n": cb["src_n"], "tag": cb["tag"],
                            "x": cb["x"].tolist(), "y": cb["y"].tolist(), "th": cb["th"].tolist(),
                        })
                cache_path.write_text(json.dumps(cdata))

    # 2. Harvest B6_10x21 & B4_19x9 from brickslp win_n910.json / h32_win_n910.json
    for fp, specs in [
        (BRICKSLP_DATA / "win_n910.json", [(0, 10.7, 9.0, 30.7, 10, 21, "B6_10x21_w910"), (11.0, 30.7, 0.0, 9.7, 19, 9, "B4_19x9_w910")]),
        (H32_WIN_N910_PATH, [(11.0, 30.7, 0.0, 9.8, 19, 9, "B4_19x9_h32")]),
        (BRICKSLP_DATA / "win_n911.json", [(0, 10.7, 9.0, 30.7, 10, 21, "B6_10x21_w911"), (13.0, 30.7, 0.0, 9.7, 17, 9, "B5_17x9_w911")]),
        (BRICKSLP_DATA / "win_n971.json", [(0, 10.7, 10.0, 31.7, 10, 21, "B6_10x21_w971"), (15.0, 31.7, 0.0, 10.7, 16, 10, "B4_16x10_w971")]),
    ]:
        if not fp.exists():
            continue
        rec = json.loads(fp.read_text())
        rows = rec["rows"]
        x = np.array([float(r["x"]) for r in rows], dtype=np.float64)
        y = np.array([float(r["y"]) for r in rows], dtype=np.float64)
        th = np.array([float(r["angle"]) for r in rows], dtype=np.float64)
        x, y, th, _ = frontier_io.norm_pack(x, y, th)
        h = frontier_io.hx_of(th)
        for X0, X1, Y0, Y1, w, hb, tag in specs:
            idx = np.where((x - h >= X0 - 0.05) & (x + h <= X1 + 0.05) & (y - h >= Y0 - 0.05) & (y + h <= Y1 + 0.05))[0]
            g = len(idx) - w * hb
            if g > 0:
                _add_pair(make_brick(w, hb, g, x[idx], y[idx], th[idx], int(rec["n"]), tag))

    # 3. Also extract standard frontier bricks (clean cuts across merged.csv)
    std_bricks = brick_slp.extract_subbricks(packs)
    for (w, h, g), b in std_bricks.items():
        if b["f"] < 0.72 and g >= 1:
            bank.append(dict(b, f_min=min(b["fx"], b["fy"])))

    return bank


def tilted_mask(th: np.ndarray, thresh_deg: float = 2.0) -> np.ndarray:
    t = np.degrees(np.mod(th, 0.5 * math.pi))
    return np.minimum(t, 90.0 - t) > thresh_deg


def assemble_b5_pair(
    m: int,
    r: int,
    b1: dict,
    b2: dict,
    s1: int,
    s2: int,
    f_mode: str = "tight",
    drop_from: str = "b1",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, tuple[float, float, float, float]] | None:
    """Assemble TL brick b1 and BR brick b2 inside [0, m+f]^2 with exact square count m^2 + r.

    Handles:
      - Disjoint / corner-touching (kx <= 0, ky <= 0)
      - 1D overlap (kx > 0, ky <= -1 or kx <= -1, ky > 0) where b1 and b2 remain disjoint
      - 2D interlocking (kx >= 0, ky >= 0, kx + ky > 0)
    Returns (x, y, th, L, seam_box) or None.
    """
    if b1["g"] + b2["g"] != r:
        return None
    sb1 = sym_brick(b1, s1)
    sb2 = sym_brick(b2, s2)
    w1, h1, g1 = sb1["w"], sb1["h"], sb1["g"]
    w2, h2, g2 = sb2["w"], sb2["h"], sb2["g"]
    if w1 >= m or h1 >= m or w2 >= m or h2 >= m:
        return None

    kx = w1 + w2 - m
    ky = h1 + h2 - m
    if kx > 4 or ky > 4:
        return None

    # Determine target f for assembly
    if f_mode == "tight":
        # Tight against short axes of b1 (width) and b2 (height), or max(fx1, fy2)
        f = max(sb1["fx"], sb2["fy"])
    elif f_mode == "mid":
        f = 0.5 * (max(sb1["fx"], sb2["fy"]) + max(sb1["f"], sb2["f"]))
    else:
        f = max(sb1["f"], sb2["f"])
    L = float(m) + f

    # Place b1 at TL [0, w1+fx1] x [L-(h1+fy1), L]
    dx1, dy1 = 0.0, L - (h1 + sb1["fy"])
    # Place b2 at BR [L-(w2+fx2), L] x [0, h2+fy2]
    dx2, dy2 = L - (w2 + sb2["fx"]), 0.0

    x1, y1, t1 = sb1["x"] + dx1, sb1["y"] + dy1, sb1["th"].copy()
    x2, y2, t2 = sb2["x"] + dx2, sb2["y"] + dy2, sb2["th"].copy()

    tilt1 = tilted_mask(t1)
    tilt2 = tilted_mask(t2)
    # Check tilted cores of b1 and b2 do not collide
    if np.any(tilt1) and np.any(tilt2):
        tpen, _ = cpp_eval_sat_pen_bbox(
            np.concatenate([x1[tilt1], x2[tilt2]]),
            np.concatenate([y1[tilt1], y2[tilt2]]),
            np.concatenate([t1[tilt1], t2[tilt2]]),
        )
        if tpen > 1e-5:
            return None

    # Grid dimensions:
    # TR: [m - w1 cols] x [m - h2 rows] anchored at (L - (m - w1), L - (m - h2)) .. (L, L)
    # BL: [m - w2 cols] x [m - h1 rows] anchored at (0, 0) .. (m - w2, m - h1)
    n_tr_w, n_tr_h = m - w1, m - h2
    n_bl_w, n_bl_h = m - w2, m - h1

    # Overlap / deficit between TR and BL:
    # (w1*h1 + w2*h2 + n_tr_w*n_tr_h + n_bl_w*n_bl_h) = m^2 + kx * ky
    diff = kx * ky
    tr_cells = {(i, j) for i in range(n_tr_w) for j in range(n_tr_h)}
    bl_cells = {(i, j) for i in range(n_bl_w) for j in range(n_bl_h)}

    extra_x, extra_y = [], []
    if kx <= 0 and ky <= 0:
        # Exact overlap BL ∩ TR is {(i, j) : w1 <= i < m - w2 and h2 <= j < m - h1} of size kx * ky
        if diff > 0:
            bl_cells = {ij for ij in bl_cells if not (ij[0] >= w1 and ij[1] >= h2)}
    elif kx > 0 and ky < 0:
        # diff < 0: need |diff| extra unit squares in stepped central seam
        # Place them in the pocket between BL right edge (x ~ m-w2) and b2 top / b1 bottom
        need = -diff
        col_x = float(m - w2) + 0.5
        y_lo = min(float(m - h1), h2 + sb2["fy"])
        y_hi = max(float(m - h1), L - (h1 + sb1["fy"]))
        for k_idx in range(need):
            ix = k_idx // max(1, -ky)
            iy = k_idx % max(1, -ky)
            extra_x.append(min(L - 0.5, max(0.5, col_x + ix)))
            extra_y.append(min(L - 0.5, max(0.5, y_lo + iy + 0.5 if y_lo + iy + 0.5 < L else y_hi - iy - 0.5)))
    elif ky > 0 and kx < 0:
        need = -diff
        row_y = float(m - h1) + 0.5
        x_lo = min(float(m - w2), w1 + sb1["fx"])
        for k_idx in range(need):
            iy = k_idx // max(1, -kx)
            ix = k_idx % max(1, -kx)
            extra_x.append(min(L - 0.5, max(0.5, x_lo + ix + 0.5)))
            extra_y.append(min(L - 0.5, max(0.5, row_y + iy)))
    elif kx >= 0 and ky >= 0 and (kx > 0 or ky > 0):
        # 2D interlocking: drop kx*ky from b1 or b2 (prefer non-tilted in overlap I)
        k_drop = kx * ky
        if k_drop > 0:
            if drop_from == "b1":
                cand = np.where((~tilt1) & (x1 > m - w2 - 0.4) & (y1 < h2 + sb2["fy"] + 0.4))[0]
                if len(cand) < k_drop:
                    return None
                drop_idx = cand[np.argsort(-(x1[cand] - y1[cand]))[:k_drop]]
                keep = np.ones(len(x1), dtype=bool)
                keep[drop_idx] = False
                x1, y1, t1 = x1[keep], y1[keep], t1[keep]
            elif drop_from == "b2":
                cand = np.where((~tilt2) & (x2 < w1 + sb1["fx"] + 0.4) & (y2 > m - h1 - 0.4))[0]
                if len(cand) < k_drop:
                    return None
                drop_idx = cand[np.argsort(-(-x2[cand] + y2[cand]))[:k_drop]]
                keep = np.ones(len(x2), dtype=bool)
                keep[drop_idx] = False
                x2, y2, t2 = x2[keep], y2[keep], t2[keep]
            else:
                return None

    gx = [i + 0.5 for (i, _) in sorted(bl_cells)]
    gy = [j + 0.5 for (_, j) in sorted(bl_cells)]
    gx.extend((L - n_tr_w) + i + 0.5 for (i, _) in sorted(tr_cells))
    gy.extend((L - n_tr_h) + j + 0.5 for (_, j) in sorted(tr_cells))
    gx.extend(extra_x)
    gy.extend(extra_y)

    ax = np.concatenate([x1, x2, np.array(gx, dtype=np.float64)])
    ay = np.concatenate([y1, y2, np.array(gy, dtype=np.float64)])
    at = np.concatenate([t1, t2, np.zeros(len(gx), dtype=np.float64)])
    if len(ax) != m * m + r:
        return None

    # Seam box spans the BL-to-bricks interface and central stepped seam
    sx0 = max(0.0, float(m - w2) - 2.5)
    sx1 = min(L, float(w1) + f + 2.0)
    sy0 = max(0.0, float(m - h1) - 2.5)
    sy1 = min(L, float(h2) + f + 2.0)
    return ax, ay, at, L, (sx0, sx1, sy0, sy1)


def build_b5_melt_seeds(
    n: int,
    xb: np.ndarray,
    yb: np.ndarray,
    tb: np.ndarray,
    Lb: float,
    seam_box: tuple[float, float, float, float],
    rng: np.random.Generator,
    L_inc: float,
    L_expand: float = 0.050,
    n_seeds: int = 24,
) -> tuple[list[np.ndarray], list[np.ndarray], list[list[float]], list[str]]:
    """Build C++ masked-melt seeds centered on the BL / brick-edge / stepped-seam interface."""
    L0 = max(Lb, L_inc) + L_expand
    sc = L0 / Lb
    x0, y0, t0 = xb * sc, yb * sc, tb.copy()
    sx0, sx1, sy0, sy1 = [v * sc for v in seam_box]
    scx, scy = 0.5 * (sx0 + sx1), 0.5 * (sy0 + sy1)

    m = math.isqrt(n)
    f_inc = L_inc - m
    L_t_mid = min(Lb + 0.008, L_inc - 0.001)
    L_t_lo = max(float(m) + 0.642, min(Lb, L_inc - 0.015))
    scheds = [
        [L0, 0.5 * (L0 + L_inc), L_inc + 0.004, L_t_mid],
        [L0, L_inc + 0.014, L_inc + 0.001, max(Lb - 0.001, L_inc - 0.008)],
        [L0, 0.5 * (L0 + L_t_lo), L_t_lo + 0.006, L_t_lo],
    ]

    # Key melt centers: central seam, BL corner, b1 bottom edge, b2 left edge
    centers = [
        (scx, scy, "seam_ctr"),
        (sx1, sy0, "b2_top_left"),
        (sx0, sy1, "b1_bot_right"),
        (0.5 * sx1, sy1, "b1_bot_edge"),
        (sx0, 0.5 * sy1, "b2_left_edge"),
        (0.5 * sx0, 0.5 * sy0, "bl_quad"),
        (sx0, sy0, "bl_corner"),
        (sx1, sy1, "tr_corner"),
    ]

    t_deg = np.degrees(np.mod(t0, 0.5 * math.pi))
    is_unit = np.minimum(t_deg, 90.0 - t_deg) < 2.0

    q_list, m_list, ls_list, labels = [], [], [], []
    for s_idx in range(n_seeds):
        x, y, th = x0.copy(), y0.copy(), t0.copy()
        cx, cy, cname = centers[s_idx % len(centers)]
        rad = float(rng.uniform(4.8, 9.5))
        near = ((x - cx) ** 2 + (y - cy) ** 2) <= rad * rad
        in_seam = (x >= sx0 - 1.0) & (x <= sx1 + 1.0) & (y >= sy0 - 1.0) & (y <= sy1 + 1.0)
        if s_idx % 3 == 0:
            # All squares near center/seam
            mask = near | in_seam
        elif s_idx % 3 == 1:
            # Unit-preserving outside core + tight seam
            mask = (near & is_unit) | in_seam
        else:
            # Whole BL + brick boundary edges (leaves deep tilted cores frozen)
            in_bl = (x <= sx1 + 0.5) & (y <= sy1 + 0.5)
            mask = (in_bl & is_unit) | in_seam | near

        nm = int(mask.sum())
        if nm > 0:
            x[mask] += rng.uniform(-0.16, 0.16, nm)
            y[mask] += rng.uniform(-0.16, 0.16, nm)
            th[mask] += rng.uniform(-0.26, 0.26, nm)

        q_list.append(full_to_free_sym(x, y, th, L0))
        m_list.append(np.concatenate([mask.astype(np.uint8)] * 3))
        ls_list.append(scheds[s_idx % len(scheds)])
        labels.append(f"b5m_{cname}_{s_idx}")

    return q_list, m_list, ls_list, labels


def polish_and_certify(n: int, x: np.ndarray, y: np.ndarray, th: np.ndarray, L_inc_dec: Decimal, label: str, do_mechwalk: bool = False) -> dict:
    """Relax if needed, run tight multi-stage joint SLP (+ optional mechwalk), fixed-angle LP, and literal+pandas certify."""
    hx = frontier_io.hx_of(th)
    x = x - float((x - hx).min())
    y = y - float((y - hx).min())
    pen, bb = cpp_eval_sat_pen_bbox(x, y, th)
    if 1e-7 < pen < 0.15:
        x, y, th, _ = u.relax_lbfgs(x, y, th, bb + max(0.004, 2.0 * pen), maxiter=400, w_wall=4.0, w_pair=1.0)
        pen, bb = cpp_eval_sat_pen_bbox(x, y, th)
        if 1e-6 < pen < 0.06:
            x, y, th, _ = u.relax_lbfgs(x, y, th, bb + max(0.008, 2.5 * pen), maxiter=350, w_wall=4.0, w_pair=1.0)
            pen, bb = cpp_eval_sat_pen_bbox(x, y, th)
    if pen > 0.025:
        return {"ok": False, "n": n, "label": label, "pen": float(pen), "bb": float(bb)}

    xs, ys, ts, Ls, ok = brick_slp.joint_xytheta_slp_tight(x, y, th)
    if not ok:
        return {"ok": False, "n": n, "label": label, "pen": float(pen), "bb": float(bb)}

    # Optional quick soft-mode mechanism walk kick + re-SLP if requested
    if do_mechwalk:
        try:
            jac, _ = ml.contact_jacobian(xs, ys, ts, contact_tol=4e-4)
            sv, raw_m = ml.soft_modes(jac, k=8)
            sel = ml.select_diverse_modes(raw_m, sv, xs, ys, max_keep=3)
            for _, _, mv, _ in sel:
                for amp in (0.008, -0.008, 0.015):
                    kx = xs + amp * mv[:n]
                    ky = ys + amp * mv[n : 2 * n]
                    kt = ts + amp * mv[2 * n :]
                    kpen, kbb = cpp_eval_sat_pen_bbox(kx, ky, kt)
                    if kpen > 1e-7:
                        kx, ky, kt, _ = u.relax_lbfgs(kx, ky, kt, Ls + 0.001, maxiter=150, w_wall=4.0, w_pair=1.0)
                    kxs, kys, kts, kLs, kok = brick_slp.joint_xytheta_slp_tight(kx, ky, kt)
                    if kok and kLs < Ls - 1e-7:
                        xs, ys, ts, Ls = kxs, kys, kts, kLs
        except Exception:
            pass

    best_cert = None
    best_dec = None
    for mlp in (1e-7, 2e-8, 5e-9):
        xf, yf, _, okf = u.fixed_angle_lp_tighten(xs, ys, ts, rounds=5, margin=mlp)
        cx, cy = (xf, yf) if okf else (xs, ys)
        cert = frontier_io.certify_against_frontier(n, cx, cy, ts, L_inc_dec)
        if cert is not None:
            c_dec = max(Decimal(cert["L_lit"]), Decimal(cert["L_pd"]))
            if best_dec is None or c_dec < best_dec:
                best_dec = c_dec
                best_cert = cert
            if cert["is_real_win"] and mlp <= 1e-7:
                break

    if best_cert is None:
        return {"ok": False, "n": n, "label": label, "pen": float(pen), "bb": float(bb)}
    best_cert["label"] = label
    return best_cert


def save_win_b5(n: int, cert: dict, label: str) -> bool:
    """Save certified REAL WIN (dL <= -1e-9) atomically and rebuild poc/swarm_b5xplant/final.csv."""
    if not cert.get("is_real_win"):
        return False
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = DATA_DIR / ".final.lock"
    with open(lock_path, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        wp = DATA_DIR / f"win_n{n}.json"
        if wp.exists():
            prev = json.loads(wp.read_text())
            if max(Decimal(prev["L_lit"]), Decimal(prev["L_pd"])) <= max(Decimal(cert["L_lit"]), Decimal(cert["L_pd"])):
                return False
        payload = {
            "n": int(n),
            "label": label,
            "L_lit": cert["L_lit"],
            "L_pd": cert["L_pd"],
            "L_float": cert["L_float"],
            "dL": cert["dL"],
            "rows": cert["rows"],
        }
        auto_union.atomic(wp, json.dumps(payload, indent=2).encode())
        os.chmod(wp, 0o644)

        groups = {}
        for p in sorted(DATA_DIR.glob("win_n*.json")):
            d = json.loads(p.read_text())
            groups[int(d["n"])] = d["rows"]
        text = auto_union.final_text(groups)
        auto_union.atomic(FINAL_CSV, text.encode())
        os.chmod(FINAL_CSV, 0o644)
    return True


def _polish_worker_mp(args: tuple) -> dict:
    n, x, y, th, L_inc_str, label, do_mw = args
    return polish_and_certify(n, x, y, th, Decimal(L_inc_str), label, do_mechwalk=do_mw)


def find_b5_candidates(n: int, bank: list[dict], L_inc: float, max_cands: int = 4) -> list[dict]:
    """Find best (b1, b2) assemblies for n, prioritizing pairs using n850 B5/B4 bricks."""
    m = int(math.isqrt(n))
    r = n - m * m
    f_inc = L_inc - m
    if r <= 1:
        return []

    good = [b for b in bank if min(b["fx"], b["fy"]) < f_inc + 0.030 and b["w"] < m and b["h"] < m]

    pair_scores = []
    seen_pairs = set()
    for b1 in good:
        for b2 in good:
            if b1["g"] + b2["g"] != r:
                continue
            kx = b1["w"] + b2["w"] - m
            ky = b1["h"] + b2["h"] - m
            if kx > 4 or ky > 4 or (kx > 2 and ky >= 0) or (ky > 2 and kx >= 0):
                continue
            uses_n850 = ("n850" in b1["tag"]) or ("n850" in b2["tag"])
            if not uses_n850:
                continue
            f_pair_tight = max(b1["fx"], b2["fy"])
            if f_pair_tight >= f_inc + 0.025:
                continue
            s1_tag = b1["tag"].replace("_T", "")
            s2_tag = b2["tag"].replace("_T", "")
            canon = ((s1_tag, b1["w"], b1["h"], b1["g"]), (s2_tag, b2["w"], b2["h"], b2["g"]))
            canon_t = ((s2_tag, b2["h"], b2["w"], b2["g"]), (s1_tag, b1["h"], b1["w"], b1["g"]))
            if canon in seen_pairs or canon_t in seen_pairs:
                continue
            seen_pairs.add(canon)
            seen_pairs.add(canon_t)

            overlap_pen = max(0, kx) * 0.006 + max(0, ky) * 0.006 + max(0, kx) * max(0, ky) * 0.012
            iso_pen = 0.04 * max(0.0, max(b1["f"], b2["f"]) - f_pair_tight)
            score = f_pair_tight + iso_pen + overlap_pen
            pair_scores.append((score, b1, b2))

    pair_scores.sort(key=lambda z: z[0])

    cands = []
    for _, b1, b2 in pair_scores[: max_cands * 3]:
        f_full = max(b1["f"], b2["f"])
        f_tight = max(b1["fx"], b2["fy"])
        f_modes = ["tight", "full"] if (f_full - f_tight >= 0.01 and f_full <= f_inc + 0.020) else ["tight"]
        for fm in f_modes:
            best_for_fm = None
            for s1 in range(4):
                for s2 in range(4):
                    dfs = ["b1", "b2"] if (b1["w"] + b2["w"] > m and b1["h"] + b2["h"] > m) else ["b1"]
                    for df in dfs:
                        asm = assemble_b5_pair(m, r, b1, b2, s1, s2, f_mode=fm, drop_from=df)
                        if asm is None:
                            continue
                        ax, ay, at, aL, sbox = asm
                        pen, _ = cpp_eval_sat_pen_bbox(ax, ay, at)
                        if best_for_fm is None or pen < best_for_fm[0]:
                            best_for_fm = (pen, ax, ay, at, aL, sbox, f"{fm}_s{s1}{s2}_{df}")
            if best_for_fm is not None:
                pen, ax, ay, at, aL, sbox, cfg = best_for_fm
                cands.append({
                    "n": n, "m": m, "r": r,
                    "b1": b1, "b2": b2,
                    "pen0": float(pen), "x": ax, "y": ay, "th": at, "L": aL, "sbox": sbox,
                    "tag": f"{b1['tag']}+{b2['tag']}_{cfg}",
                })
                if len(cands) >= max_cands * 2:
                    break
        if len(cands) >= max_cands * 2:
            break

    cands.sort(key=lambda c: (0 if (c["pen0"] < 1e-5 and c["L"] <= L_inc + 0.005) else 1, c["L"] + 0.025 * c["pen0"]))
    return cands[:max_cands]


def eval_target(n: int, cands: list[dict], L_inc_dec: Decimal, n_threads: int = 4) -> dict:
    """Evaluate assembled candidates for target n via direct SLP + C++ masked melt + SLP."""
    t0 = time.time()
    best_res = None
    for c_idx, cand in enumerate(cands):
        # Skip candidates whose baseline L0 is already far above current L_inc_dec
        if best_res is not None and cand["L"] > float(L_inc_dec) + 0.0025:
            print(f"  [cand {c_idx+1}/{len(cands)}] SKIP {cand['tag']} (L0={cand['L']:.6f} > cur={float(L_inc_dec):.6f})", flush=True)
            continue

        tc = time.time()
        ax, ay, at, aL, sbox, tag = cand["x"], cand["y"], cand["th"], cand["L"], cand["sbox"], cand["tag"]
        print(
            f"  [cand {c_idx+1}/{len(cands)}] {tag} (L0={aL:.6f}, inc={float(L_inc_dec):.6f}, pen0={cand['pen0']:.4f})",
            flush=True,
        )

        if cand["pen0"] <= 1e-4:
            direct = polish_and_certify(n, ax, ay, at, L_inc_dec, f"{tag}_direct", do_mechwalk=False)
            if direct.get("ok"):
                print(
                    f"    -> direct ok dL={direct['dL']:+.6e} L={direct['L_lit'][:14]} ({time.time()-tc:.1f}s)",
                    flush=True,
                )
                if direct["is_real_win"]:
                    save_win_b5(n, direct, direct["label"])
                    L_inc_dec = min(L_inc_dec, Decimal(direct["L_lit"]))
                if best_res is None or direct["dL"] < best_res["dL"]:
                    best_res = direct

        rng = np.random.default_rng(40000 + n * 17 + c_idx)
        q_list, m_list, ls_list, labels = build_b5_melt_seeds(
            n, ax, ay, at, aL, sbox, rng, L_inc=float(L_inc_dec), L_expand=0.055, n_seeds=20
        )
        qb, qf, lf, _, po, _ = run_cpp_masked_melt(
            n, 1, q_list, m_list, ls_list, [5, 8, 12, 16],
            a_start=0.82, n_melt_stages=4, masked_iters=180, full_iters=180,
            n_threads=n_threads, base_seed=50000 + n * 23 + c_idx * 100,
        )
        n_feas = int(np.sum(lf < 1e9))
        min_po = float(np.min(po))
        order = sorted(
            range(len(labels)),
            key=lambda i: (0 if lf[i] < 1e9 else 1, float(lf[i]) if lf[i] < 1e9 else float(po[i])),
        )[:4]

        tasks = []
        for i in order:
            use_f = lf[i] < 1e9
            L_ref = float(lf[i]) if use_f else ls_list[i][-1]
            x, y, th = cpp_expand_sym(n, 1, L_ref, qf[i] if use_f else qb[i])
            tasks.append((n, x, y, th, str(L_inc_dec), f"{tag}_{labels[i]}", False))

        with mp.Pool(processes=min(4, len(tasks))) as pool:
            pol = pool.map(_polish_worker_mp, tasks)

        ok_list = [r for r in pol if r.get("ok")]
        fail_pens = [round(r.get("pen", 99.0), 4) for r in pol if not r.get("ok")]
        if ok_list:
            b = min(ok_list, key=lambda r: r["dL"])
            print(
                f"    -> feas={n_feas}/{len(labels)} min_po={min_po:.4e} best_dL={b['dL']:+.6e} L={b['L_lit'][:14]} ({time.time()-tc:.1f}s)",
                flush=True,
            )
            if b["is_real_win"]:
                save_win_b5(n, b, b["label"])
                L_inc_dec = min(L_inc_dec, Decimal(b["L_lit"]))
            if best_res is None or b["dL"] < best_res["dL"]:
                best_res = b
        else:
            print(
                f"    -> feas={n_feas}/{len(labels)} min_po={min_po:.4e} ALL FAILED SLP (pens={fail_pens}, {time.time()-tc:.1f}s)",
                flush=True,
            )

    dt = time.time() - t0
    if best_res is None:
        return {"n": n, "status": "FAIL", "dt": round(dt, 1)}
    return {
        "n": n,
        "status": "WIN" if best_res["is_real_win"] else "OK",
        "dL": best_res["dL"],
        "L_lit": best_res["L_lit"],
        "label": best_res["label"],
        "dt": round(dt, 1),
    }

