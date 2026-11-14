#!/usr/bin/env python3
"""Hypothesis H52: Compact g=+4 & g=+3 Corner Sub-Brick Extraction, Compression, and Pinwheel Broadcast."""
from __future__ import annotations

import csv
import fcntl
import json
import math
import multiprocessing as mp
import os
import sys
import tempfile
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
THIN_DIR = HERE.parent
DATA_DIR = THIN_DIR / "data"
DOCS_DIR = THIN_DIR / "docs"
FINAL_CSV = THIN_DIR / "final.csv"

REPO = Path(__file__).resolve().parents[3]
B5_POC = REPO / "poc" / "swarm_b5xplant" / "poc"
BRICKSLP_POC = REPO / "poc" / "swarm_brickslp" / "poc"
MECH_POC = REPO / "poc" / "swarm_mechwalk" / "poc"
HIGHD_POC = REPO / "poc" / "swarm_highd" / "poc"
HOMO_POC = REPO / "poc" / "swarm_homology" / "poc"
NEW159_POC = REPO / "poc" / "swarm_new159" / "poc"
GAD_POC = REPO / "poc" / "rectangle_gadgets" / "poc"
UNION_POC = REPO / "poc" / "auto_union" / "poc"

for _p in (UNION_POC, HOMO_POC, HIGHD_POC, NEW159_POC, GAD_POC, BRICKSLP_POC, MECH_POC, B5_POC):
    if str(_p) not in sys.path:
        sys.path.append(str(_p))

import b5_lib as bl  # noqa: E402
import brick_slp as bs  # noqa: E402
import frontier_io as fio  # noqa: E402
import union as auto_union  # noqa: E402
import utils as u  # noqa: E402
from cpp_wrap import cpp_eval_sat_pen_bbox, cpp_expand_sym, full_to_free_sym, run_cpp_masked_melt  # noqa: E402

MERGED_CSV = REPO / "poc" / "pell_corrugation" / "data" / "border-scan" / "merged.csv"
ACTIVE_FINALS = [
    REPO / "poc" / "swarm_b5xplant" / "final.csv",
    REPO / "poc" / "swarm_mechwalk" / "final.csv",
    REPO / "poc" / "swarm_b8xplant" / "final.csv",
]
CACHE_PATH = DATA_DIR / "compressed_bricks_cache.json"


def fast_load_frontier(
    target_ns: list[int] | None = None,
) -> tuple[dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, float]], dict[int, Decimal]]:
    """Fast load frontier from merged.csv + active swarm_{b5xplant,mechwalk,b8xplant}/final.csv (<2s)."""
    packs: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, float]] = {}
    best_L_dec: dict[int, Decimal] = {}
    need = {int(n) for n in target_ns} if target_ns else set()

    df = pd.read_csv(MERGED_CSV, usecols=["n", "square_id", "x", "y", "angle"])
    for n_val, sub in df.groupby("n"):
        n = int(n_val)
        sub = sub.sort_values("square_id")
        x, y, th, L = fio.norm_pack(
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

    for fp in ACTIVE_FINALS:
        if not fp.exists() or fp.stat().st_size <= 20:
            continue
        groups: dict[int, list[tuple[int, float, float, float]]] = {}
        with fp.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                n_s, sid_s = row["id"].split("_")
                n = int(n_s)
                groups.setdefault(n, []).append(
                    (
                        int(sid_s),
                        float(row["x"].lstrip("s")),
                        float(row["y"].lstrip("s")),
                        math.radians(float(row["deg"].lstrip("s"))),
                    )
                )

        need_cert: list[int] = []
        for n, rows in groups.items():
            if len(rows) != n or n not in packs:
                continue
            rows.sort(key=lambda z: z[0])
            nx, ny, nt, L = fio.norm_pack(
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
            lit, pd_t = auto_union.load(fp)
            for n in need_cert:
                if n in lit and n in pd_t:
                    res = auto_union.check((n, lit[n], pd_t[n]))
                    if res is not None:
                        Ld = max(Decimal(res[0]), Decimal(res[1]))
                        if Ld < best_L_dec[n]:
                            best_L_dec[n] = Ld
                            r_s = sorted(lit[n], key=lambda r: int(r["square_id"]))
                            packs[n] = fio.norm_pack(
                                np.array([float(r["x"]) for r in r_s]),
                                np.array([float(r["y"]) for r in r_s]),
                                np.array([float(r["angle"]) for r in r_s]),
                            )

    return packs, best_L_dec


def extract_corner_rect(
    x: np.ndarray,
    y: np.ndarray,
    th: np.ndarray,
    L: float,
    rx0: float,
    rx1: float,
    ry0: float,
    ry1: float,
    w: int,
    h: int,
    src_n: int,
    tag: str,
    tol: float = 0.05,
) -> dict | None:
    """Extract an exact corner sub-brick, verify zero crossing squares & pen==0, and normalize to [0, w+fx]x[0, h+fy]."""
    hx = fio.hx_of(th)
    x0, x1, y0, y1 = x - hx, x + hx, y - hx, y + hx
    in_m = (x0 >= rx0 - tol) & (x1 <= rx1 + tol) & (y0 >= ry0 - tol) & (y1 <= ry1 + tol)
    ov_m = (x1 > rx0 + tol) & (x0 < rx1 - tol) & (y1 > ry0 + tol) & (y0 < ry1 - tol)
    if int(np.sum(ov_m & (~in_m))) != 0:
        return None
    cnt = int(np.sum(in_m))
    g = cnt - w * h
    if g <= 0:
        return None
    b = bl.make_brick(w, h, g, x[in_m], y[in_m], th[in_m], src_n, tag)
    return b


def strip_brick_border(b: dict, new_w: int, new_h: int, tag: str) -> dict | None:
    """Try stripping outer unit-square rows/cols from any of the 4 D2 reflections to reach (new_w, new_h)."""
    w, h, g = b["w"], b["h"], b["g"]
    if new_w > w or new_h > h or (new_w == w and new_h == h):
        return None
    best = None
    for s in range(4):
        sb = bl.sym_brick(b, s)
        bx, by, bt = sb["x"], sb["y"], sb["th"]
        bh = fio.hx_of(bt)
        # Keep squares inside [0, new_w + fx + 0.08] x [0, new_h + fy + 0.08]
        rx1 = new_w + sb["fx"] + 0.08
        ry1 = new_h + sb["fy"] + 0.08
        in_m = (bx + bh <= rx1) & (by + bh <= ry1)
        ov_m = (bx - bh < rx1 - 0.16) & (by - bh < ry1 - 0.16)
        if int(np.sum(ov_m & (~in_m))) != 0:
            continue
        if int(np.sum(in_m)) == new_w * new_h + g:
            cand = bl.make_brick(new_w, new_h, g, bx[in_m], by[in_m], bt[in_m], b["src_n"], tag)
            if cand is not None and (best is None or cand["f"] < best["f"]):
                best = cand
    return best


def _compress_trial(args: tuple) -> tuple | None:
    idx, w, h, g, x0, y0, th0, f0, mode, seed = args
    rng = np.random.default_rng(seed)
    x, y, th = x0.copy(), y0.copy(), th0.copy()
    if mode == "direct":
        xs, ys, ts, fs, ok = bs.joint_xytheta_slp_rect(x, y, th, w, h, rounds=14, tr_th0=0.020, tr_xy0=0.09, tol=1e-11)
        if ok:
            xs2, ys2, ts2, fs2, ok2 = bs.joint_xytheta_slp_rect(xs, ys, ts, w, h, rounds=12, tr_th0=0.004, tr_xy0=0.015, tol=1e-12)
            if ok2 and fs2 < fs:
                xs, ys, ts, fs = xs2, ys2, ts2, fs2
        return (fs - f0, fs, xs, ys, ts, f"rect_direct_{idx}") if ok else None

    f_exp = f0 + float(rng.uniform(0.016, 0.036))
    nx, ny, W0, H0 = bs.rect_bbox(x, y, th)
    scx = (w + f_exp) / W0
    scy = (h + f_exp) / H0
    x = nx * scx
    y = ny * scy
    cx = float(rng.uniform(0.8, w + f_exp - 0.8))
    cy = float(rng.uniform(0.8, h + f_exp - 0.8))
    rad = float(rng.uniform(2.8, 6.2))
    near = ((x - cx) ** 2 + (y - cy) ** 2) <= rad * rad
    if mode == "wall" and len(x) > 0:
        hx = fio.hx_of(th)
        wm = np.minimum.reduce([x - hx, (w + f_exp) - x - hx, y - hx, (h + f_exp) - y - hx])
        ti = int(np.argsort(wm)[idx % min(6, len(wm))])
        near = ((x - x[ti]) ** 2 + (y - y[ti]) ** 2) <= rad * rad
    elif mode == "seam":
        t_deg = np.degrees(np.mod(th, 0.5 * math.pi))
        is_u = np.minimum(t_deg, 90.0 - t_deg) < 2.0
        near = is_u | near
    nm = int(np.sum(near))
    if nm > 0:
        x[near] += rng.uniform(-0.14, 0.14, nm)
        y[near] += rng.uniform(-0.14, 0.14, nm)
        th[near] += rng.uniform(-0.28, 0.28, nm)

    for ft in (f0 + 0.014, f0 + 0.002, max(0.58, f0 - 0.002)):
        x, y, th, E = bs.relax_rect_lbfgs(x, y, th, w + ft, h + ft, maxiter=450)
        if E > 1e-4:
            break
    pen, _ = cpp_eval_sat_pen_bbox(x, y, th)
    if pen > 0.02:
        return None
    xs, ys, ts, fs, ok = bs.joint_xytheta_slp_rect(x, y, th, w, h, rounds=12, tr_th0=0.020, tr_xy0=0.09, tol=1e-10)
    if not ok:
        return None
    xs2, ys2, ts2, fs2, ok2 = bs.joint_xytheta_slp_rect(xs, ys, ts, w, h, rounds=10, tr_th0=0.004, tr_xy0=0.015, tol=1e-12)
    if ok2 and fs2 < fs:
        xs, ys, ts, fs = xs2, ys2, ts2, fs2
    return (fs - f0, fs, xs, ys, ts, f"rect_{mode}_{idx}")


def compress_brick_standalone(b: dict, n_trials: int = 16, n_workers: int = 4) -> dict:
    """Run direct joint_xytheta_slp_rect + 1-round standalone rejam/mmelt compression on a brick."""
    w, h, g, bf = b["w"], b["h"], b["g"], b["f"]
    trials = [(0, w, h, g, b["x"], b["y"], b["th"], bf, "direct", 9100)]
    for i in range(1, n_trials):
        mode = ("wall", "melt", "seam")[i % 3]
        trials.append((i, w, h, g, b["x"], b["y"], b["th"], bf, mode, 9100 + i * 37))

    with mp.Pool(processes=n_workers) as pool:
        res = [r for r in pool.map(_compress_trial, trials) if r is not None]
    res.sort(key=lambda r: r[0])
    if res and res[0][0] < -1e-9:
        _, nf, nx, ny, nt, rtag = res[0]
        cb = bl.make_brick(w, h, g, nx, ny, nt, b["src_n"], b["tag"] + "_cmp")
        if cb is not None and cb["f"] < b["f"] - 1e-9:
            return cb
    return b


def build_h52_bank(
    packs: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, float]],
    compress: bool = True,
    force_recompress: bool = False,
) -> dict[str, dict]:
    """Extract and optionally compress all H52 compact g=+4, g=+3, g=+5, g=+6 bricks (plus transposes)."""
    raw_bricks: dict[str, dict] = {}

    def _reg(b: dict | None, name: str | None = None) -> dict | None:
        if b is None:
            return None
        key = name or b["tag"]
        b = dict(b, tag=key)
        raw_bricks[key] = b
        return b

    # 1. Extract from frontier packings
    # n=684 (m=26): TL 9x16 (+4), BR 17x9 (+4)
    if 684 in packs:
        x, y, th, L = packs[684]
        f = L - 26.0
        _reg(extract_corner_rect(x, y, th, L, 0.0, 9.0 + f, L - (16.0 + f), L, 9, 16, 684, "B4_9x16_n684"))
        _reg(extract_corner_rect(x, y, th, L, L - (17.0 + f), L, 0.0, 9.0 + f, 17, 9, 684, "B4_17x9_n684"))

    # n=737 (m=27): BR 10x16 (+4)
    if 737 in packs:
        x, y, th, L = packs[737]
        f = L - 27.0
        _reg(extract_corner_rect(x, y, th, L, 27.0 - 10.0, L, 0.0, 16.0 + f, 10, 16, 737, "B4_10x16_n737"))

    # n=849 (m=29): TL 11x16 (+4), BR 18x12 (+4)
    if 849 in packs:
        x, y, th, L = packs[849]
        f = L - 29.0
        _reg(extract_corner_rect(x, y, th, L, 0.0, 11.0 + f, L - (16.0 + f), L, 11, 16, 849, "B4_11x16_n849"))
        _reg(extract_corner_rect(x, y, th, L, L - (18.0 + f), L, 0.0, 12.0 + f, 18, 12, 849, "B4_18x12_n849"))

    # n=908 (m=30): TL 13x16 (+4), BR 17x13 (+4), TL 12x16 (+4)
    if 908 in packs:
        x, y, th, L = packs[908]
        f = L - 30.0
        _reg(extract_corner_rect(x, y, th, L, 0.0, 13.0 + f, L - (16.0 + f), L, 13, 16, 908, "B4_13x16_n908"))
        _reg(extract_corner_rect(x, y, th, L, L - (17.0 + f), L, 0.0, 13.0 + f, 17, 13, 908, "B4_17x13_n908"))
        _reg(extract_corner_rect(x, y, th, L, 0.0, 12.0 + f, L - (16.0 + f), L, 12, 16, 908, "B4_12x16_n908"))

    # n=969 (m=31): TL 16x14 (+4)
    if 969 in packs:
        x, y, th, L = packs[969]
        f = L - 31.0
        _reg(extract_corner_rect(x, y, th, L, 0.0, 16.0 + f, L - (14.0 + f), L, 16, 14, 969, "B4_16x14_n969"))

    # n=848 (m=29): BR 9x15 (+3), TL 10x6 (+2)
    if 848 in packs:
        x, y, th, L = packs[848]
        f = L - 29.0
        b915 = _reg(extract_corner_rect(x, y, th, L, L - (9.0 + f), L, 0.0, 15.0 + f, 9, 15, 848, "B3_9x15_n848"))
        if b915 is not None:
            for nw, nh, stg in [(9, 14, "B3_9x14_n848"), (9, 13, "B3_9x13_n848"), (8, 15, "B3_8x15_n848")]:
                _reg(strip_brick_border(b915, nw, nh, stg))
        _reg(extract_corner_rect(x, y, th, L, 0.0, 10.0 + f, L - (6.0 + f), L, 10, 6, 848, "B2_10x6_n848"))

    # n=632, n=736, n=791 (g=+3 bricks)
    if 632 in packs:
        x, y, th, L = packs[632]
        f = L - 25.0
        b149 = _reg(extract_corner_rect(x, y, th, L, 0.0, 14.0 + f, L - (9.0 + f), L, 14, 9, 632, "B3_14x9_n632"))
        if b149 is not None:
            _reg(strip_brick_border(b149, 13, 9, "B3_13x9_n632"))
            _reg(strip_brick_border(b149, 13, 8, "B3_13x8_n632"))
    if 736 in packs:
        x, y, th, L = packs[736]
        f = L - 27.0
        _reg(extract_corner_rect(x, y, th, L, 0.0, 10.0 + f, L - (14.0 + f), L, 10, 14, 736, "B3_10x14_n736"))

    # n=793 (m=28): TL 10x19 (+5)
    if 793 in packs:
        x, y, th, L = packs[793]
        f = L - 28.0
        _reg(extract_corner_rect(x, y, th, L, 0.0, 10.0 + f, L - (19.0 + f), L, 10, 19, 793, "B5_10x19_n793"))

    # n=909 (from mechwalk win_n909.json / packs[909]): TL 10x20 (+5) & TL 11x20 (+5)
    mw_909 = REPO / "poc" / "swarm_mechwalk" / "data" / "win_n909.json"
    if mw_909.exists():
        rec = json.loads(mw_909.read_text())
        rows = rec["rows"]
        x, y, th, L = fio.norm_pack(
            [float(r["x"]) for r in rows], [float(r["y"]) for r in rows], [float(r["angle"]) for r in rows]
        )
        f = L - 30.0
        _reg(extract_corner_rect(x, y, th, L, 0.0, 10.0 + f, L - (20.0 + f), L, 10, 20, 909, "B5_10x20_b4"))
        _reg(extract_corner_rect(x, y, th, L, 0.0, 11.0 + f, L - (20.0 + f), L, 11, 20, 909, "B5_11x20_b4"))

    # Also B5_10x20_n850 (tight fx=fy=0.642572) from donor_win_n850.json
    if bl.WIN_N850_PATH.exists():
        rec = json.loads(bl.WIN_N850_PATH.read_text())
        rows = rec["rows"]
        x, y, th, L = fio.norm_pack(
            [float(r["x"]) for r in rows], [float(r["y"]) for r in rows], [float(r["angle"]) for r in rows]
        )
        h = fio.hx_of(th)
        m_b5_20 = (x + h <= 10.65) & (y - h >= 8.999)
        _reg(bl.make_brick(10, 20, 5, x[m_b5_20], y[m_b5_20], th[m_b5_20], 850, "B5_10x20_n850"))
        m_b5_19 = (x + h <= 10.6426) & (y - h >= 9.6425)
        _reg(bl.make_brick(10, 19, 5, x[m_b5_19], y[m_b5_19], th[m_b5_19], 850, "B5_10x19_n850"))

    # n=972 (from mechwalk win_n972.json): BR 10x21 (+6)
    mw_972 = REPO / "poc" / "swarm_mechwalk" / "data" / "win_n972.json"
    if mw_972.exists():
        rec = json.loads(mw_972.read_text())
        rows = rec["rows"]
        x, y, th, L = fio.norm_pack(
            [float(r["x"]) for r in rows], [float(r["y"]) for r in rows], [float(r["angle"]) for r in rows]
        )
        f = L - 31.0
        _reg(extract_corner_rect(x, y, th, L, L - (10.0 + f), L, 0.0, 21.0 + f, 10, 21, 972, "B6_10x21_b4"))

    # 2. Compress raw bricks (with disk cache)
    compressed_base: dict[str, dict] = {}
    if compress and CACHE_PATH.exists() and not force_recompress:
        cdata = json.loads(CACHE_PATH.read_text())
        for key, item in cdata.items():
            cb = bl.make_brick(
                item["w"], item["h"], item["g"],
                np.array(item["x"], dtype=np.float64),
                np.array(item["y"], dtype=np.float64),
                np.array(item["th"], dtype=np.float64),
                item["src_n"], item["tag"],
            )
            if cb is not None:
                compressed_base[key] = cb
    elif compress:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        for key, b in sorted(raw_bricks.items()):
            cb = compress_brick_standalone(b, n_trials=16, n_workers=4)
            df = cb["f"] - b["f"]
            print(f"  [brick] {key:16s} ({b['w']}x{b['h']},+{b['g']}) f0={b['f']:.6f} -> f={cb['f']:.6f} (fx={cb['fx']:.6f},fy={cb['fy']:.6f}) df={df:+.3e}", flush=True)
            compressed_base[key] = cb
        cdata = {
            key: {
                "w": cb["w"], "h": cb["h"], "g": cb["g"],
                "src_n": cb["src_n"], "tag": cb["tag"],
                "x": cb["x"].tolist(), "y": cb["y"].tolist(), "th": cb["th"].tolist(),
            }
            for key, cb in compressed_base.items()
        }
        auto_union.atomic(CACHE_PATH, json.dumps(cdata).encode())
    else:
        compressed_base = raw_bricks

    # Ensure both base and _T transpose are registered
    bank: dict[str, dict] = {}
    for key, b in compressed_base.items():
        bank[key] = b
        tb = bl.transpose_brick(b)
        bank[key + "_T"] = tb
    # Also keep raw bricks if any weren't in cache
    for key, b in raw_bricks.items():
        if key not in bank:
            bank[key] = b
            bank[key + "_T"] = bl.transpose_brick(b)

    return bank


def assemble_pinwheel_shifted_sym(
    m: int,
    b1: dict,
    b2: dict,
    s1: int,
    s2: int,
    shift_frac: float = 0.5,
    mode: str = "tr_left_bl_up",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, tuple[float, float, float, float]] | None:
    """Zero-overlap pinwheel assembly (w1+w2<=m, h1+h2<=m) with H32 pre-shifted seam slack-sharing."""
    sb1 = bl.sym_brick(b1, s1)
    sb2 = bl.sym_brick(b2, s2)
    w1, h1, g1 = sb1["w"], sb1["h"], sb1["g"]
    w2, h2, g2 = sb2["w"], sb2["h"], sb2["g"]
    if w1 + w2 > m or h1 + h2 > m:
        return None
    f = max(sb1["f"], sb2["f"])
    L = float(m) + f

    # Put b1 at TL [0, w1+fx1] x [L-(h1+fy1), L] and b2 at BR [L-(w2+fx2), L] x [0, h2+fy2]
    dx1, dy1 = 0.0, L - (h1 + sb1["fy"])
    dx2, dy2 = L - (w2 + sb2["fx"]), 0.0
    x1, y1, t1 = sb1["x"] + dx1, sb1["y"] + dy1, sb1["th"].copy()
    x2, y2, t2 = sb2["x"] + dx2, sb2["y"] + dy2, sb2["th"].copy()

    # Slack available at TL bottom (f - fy1) and TL right (f - fx1), or BR top/left
    gap_y = max(0.0, f - sb1["fy"])
    gap_x = max(0.0, f - sb1["fx"])
    # Also if b2 has more slack than b1, shift b2? Actually shifting BL/TR uses b1's slack when b1 is smaller-f,
    # and if b2 is smaller-f we can shift BL down / TR right or shift b2 inward!
    gap_b2_y = max(0.0, f - sb2["fy"])
    gap_b2_x = max(0.0, f - sb2["fx"])

    if gap_x + gap_y >= gap_b2_x + gap_b2_y:
        delta_y = shift_frac * gap_y
        delta_x = shift_frac * gap_x
        shift_b2_x, shift_b2_y = 0.0, 0.0
    else:
        # b2 has extra slack: shift b2 inward toward center by shift_frac * gap_b2,
        # OR equivalently shift BL/TR away from b1 toward b2!
        delta_y = -shift_frac * gap_b2_y
        delta_x = -shift_frac * gap_b2_x
        shift_b2_x = -shift_frac * gap_b2_x * 0.5
        shift_b2_y = -shift_frac * gap_b2_y * 0.5
        x2 += shift_b2_x
        y2 += shift_b2_y

    gx, gy = [], []
    for i in range(m):
        for j in range(m):
            if i < w1 and j >= m - h1:
                continue
            if i >= m - w2 and j < h2:
                continue
            if i < m - w2 and j < m - h1:
                px = i + 0.5
                py = j + 0.5 + (delta_y if (delta_y > 0 and mode in ("tr_left_bl_up", "bl_up")) else 0.0)
                gx.append(px)
                gy.append(py)
            else:
                sx = -delta_x if (delta_x > 0 and mode in ("tr_left_bl_up", "tr_left") and i >= w1) else 0.0
                sy = 0.0
                gx.append(i + f + 0.5 + sx)
                gy.append(j + f + 0.5 + sy)

    ax = np.concatenate([x1, x2, np.array(gx, dtype=np.float64)])
    ay = np.concatenate([y1, y2, np.array(gy, dtype=np.float64)])
    at = np.concatenate([t1, t2, np.zeros(len(gx), dtype=np.float64)])
    if len(ax) != m * m + g1 + g2:
        return None
    sx0 = max(0.0, float(m - w2) - 2.5)
    sx1 = min(L, float(w1) + f + 2.0)
    sy0 = max(0.0, float(m - h1) - 2.5)
    sy1 = min(L, float(h2) + f + 2.0)
    return ax, ay, at, L, (sx0, sx1, sy0, sy1)


def assemble_pair_best(
    m: int,
    r: int,
    b1: dict,
    b2: dict,
    shift_fracs: tuple[float, ...] = (0.0, 0.5, 0.8),
) -> list[dict]:
    """Assemble (b1, b2) across D2 symmetries and H32 shift_fracs; return lowest-pen0 configs."""
    w1, h1, g1 = b1["w"], b1["h"], b1["g"]
    w2, h2, g2 = b2["w"], b2["h"], b2["g"]
    if g1 + g2 != r or w1 >= m or h1 >= m or w2 >= m or h2 >= m:
        return []
    kx = w1 + w2 - m
    ky = h1 + h2 - m
    out = []

    if kx <= 0 and ky <= 0:
        # Zero-overlap pinwheel!
        gap_f = abs(b1["f"] - b2["f"])
        fracs = shift_fracs if gap_f > 2e-4 else (0.0,)
        for sf in fracs:
            best_s = None
            for s1 in range(4):
                for s2 in range(4):
                    asm = assemble_pinwheel_shifted_sym(m, b1, b2, s1, s2, shift_frac=sf)
                    if asm is None:
                        continue
                    ax, ay, at, aL, sbox = asm
                    pen, bb = cpp_eval_sat_pen_bbox(ax, ay, at)
                    key = (pen, bb, ax, ay, at, aL, sbox, f"zw_sf{sf:.1f}_s{s1}{s2}")
                    if best_s is None or (pen < best_s[0] - 1e-8) or (abs(pen - best_s[0]) <= 1e-8 and bb < best_s[1]):
                        best_s = key
            if best_s is not None:
                pen, bb, ax, ay, at, aL, sbox, cfg = best_s
                out.append({
                    "pen0": float(pen), "x": ax, "y": ay, "th": at, "L": float(bb), "sbox": sbox,
                    "tag": f"{b1['tag']}+{b2['tag']}_{cfg}",
                })
    else:
        # 1D or 2D overlap via b5_lib.assemble_b5_pair
        f_full = max(b1["f"], b2["f"])
        f_tight = max(b1["fx"], b2["fy"])
        f_modes = ["tight", "full"] if f_full - f_tight >= 0.008 else ["tight"]
        for fm in f_modes:
            best_s = None
            for s1 in range(4):
                for s2 in range(4):
                    dfs = ["b1", "b2"] if (kx > 0 and ky > 0) else ["b1"]
                    for df in dfs:
                        asm = bl.assemble_b5_pair(m, r, b1, b2, s1, s2, f_mode=fm, drop_from=df)
                        if asm is None:
                            continue
                        ax, ay, at, aL, sbox = asm
                        pen, bb = cpp_eval_sat_pen_bbox(ax, ay, at)
                        key = (pen, bb, ax, ay, at, aL, sbox, f"{fm}_s{s1}{s2}_{df}")
                        if best_s is None or pen < best_s[0]:
                            best_s = key
            if best_s is not None:
                pen, bb, ax, ay, at, aL, sbox, cfg = best_s
                out.append({
                    "pen0": float(pen), "x": ax, "y": ay, "th": at, "L": float(aL), "sbox": sbox,
                    "tag": f"{b1['tag']}+{b2['tag']}_{cfg}",
                })
    return out


def polish_and_certify(
    n: int, x: np.ndarray, y: np.ndarray, th: np.ndarray, L_inc_dec: Decimal, label: str, do_mechwalk: bool = False
) -> dict:
    """Relax if needed, run tight multi-stage joint SLP, fixed-angle LP, and literal+pandas certify."""
    hx = fio.hx_of(th)
    x = x - float((x - hx).min())
    y = y - float((y - hx).min())
    pen, bb = cpp_eval_sat_pen_bbox(x, y, th)
    if 1e-7 < pen < 0.20:
        x, y, th, _ = u.relax_lbfgs(x, y, th, bb + max(0.004, 2.0 * pen), maxiter=400, w_wall=4.0, w_pair=1.0)
        pen, bb = cpp_eval_sat_pen_bbox(x, y, th)
        if 1e-6 < pen < 0.08:
            x, y, th, _ = u.relax_lbfgs(x, y, th, bb + max(0.008, 2.5 * pen), maxiter=350, w_wall=4.0, w_pair=1.0)
            pen, bb = cpp_eval_sat_pen_bbox(x, y, th)
    if pen > 0.025:
        return {"ok": False, "n": n, "label": label, "pen": float(pen), "bb": float(bb)}

    xs, ys, ts, Ls, ok = bs.joint_xytheta_slp_tight(x, y, th)
    if not ok:
        return {"ok": False, "n": n, "label": label, "pen": float(pen), "bb": float(bb)}

    if do_mechwalk:
        try:
            import mechwalk_lib as ml
            jac, _ = ml.contact_jacobian(xs, ys, ts, contact_tol=4e-4)
            sv, raw_m = ml.soft_modes(jac, k=8)
            sel = ml.select_diverse_modes(raw_m, sv, xs, ys, max_keep=3)
            for _, _, mv, _ in sel:
                for amp in (0.008, -0.008, 0.015):
                    kx = xs + amp * mv[:n]
                    ky = ys + amp * mv[n : 2 * n]
                    kt = ts + amp * mv[2 * n :]
                    kpen, _ = cpp_eval_sat_pen_bbox(kx, ky, kt)
                    if kpen > 1e-7:
                        kx, ky, kt, _ = u.relax_lbfgs(kx, ky, kt, Ls + 0.001, maxiter=150, w_wall=4.0, w_pair=1.0)
                    kxs, kys, kts, kLs, kok = bs.joint_xytheta_slp_tight(kx, ky, kt)
                    if kok and kLs < Ls - 1e-7:
                        xs, ys, ts, Ls = kxs, kys, kts, kLs
        except Exception:
            pass

    best_cert = None
    best_dec = None
    for mlp in (1e-7, 2e-8, 5e-9):
        xf, yf, _, okf = u.fixed_angle_lp_tighten(xs, ys, ts, rounds=5, margin=mlp)
        cx, cy = (xf, yf) if okf else (xs, ys)
        cert = fio.certify_against_frontier(n, cx, cy, ts, L_inc_dec)
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


def _polish_worker(args: tuple) -> dict:
    n, x, y, th, L_inc_str, label, do_mw = args
    return polish_and_certify(n, x, y, th, Decimal(L_inc_str), label, do_mechwalk=do_mw)


def save_win_thinb4(n: int, cert: dict, label: str) -> bool:
    """Save certified REAL WIN (dL <= -1e-9) atomically and rebuild poc/swarm_thinb4/final.csv."""
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


def build_thinb4_melt_seeds(
    n: int,
    xb: np.ndarray,
    yb: np.ndarray,
    tb: np.ndarray,
    Lb: float,
    seam_box: tuple[float, float, float, float],
    rng: np.random.Generator,
    L_inc: float,
    L_expand: float = 0.045,
    n_seeds: int = 20,
) -> tuple[list[np.ndarray], list[np.ndarray], list[list[float]], list[str]]:
    """Build C++ masked-melt seeds with realistic L targets relative to min(Lb, L_inc)."""
    L_ref = min(Lb, L_inc)
    L0 = max(Lb, L_inc) + L_expand
    sc = L0 / Lb
    x0, y0, t0 = xb * sc, yb * sc, tb.copy()
    sx0, sx1, sy0, sy1 = [v * sc for v in seam_box]
    scx, scy = 0.5 * (sx0 + sx1), 0.5 * (sy0 + sy1)

    scheds = [
        [L0, 0.5 * (L0 + L_ref), L_ref + 0.002, L_ref - 0.0002],
        [L0, L_ref + 0.008, L_ref + 0.0005, L_ref - 0.0010],
        [L0, L_ref + 0.012, L_ref + 0.001, L_ref - 0.0030],
    ]
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
        rad = float(rng.uniform(4.8, 9.8))
        near = ((x - cx) ** 2 + (y - cy) ** 2) <= rad * rad
        in_seam = (x >= sx0 - 1.2) & (x <= sx1 + 1.2) & (y >= sy0 - 1.2) & (y <= sy1 + 1.2)
        if s_idx % 3 == 0:
            mask = near | in_seam
        elif s_idx % 3 == 1:
            mask = (near & is_unit) | in_seam
        else:
            in_bl = (x <= sx1 + 0.5) & (y <= sy1 + 0.5)
            mask = (in_bl & is_unit) | in_seam | near
        nm = int(mask.sum())
        if nm > 0:
            x[mask] += rng.uniform(-0.14, 0.14, nm)
            y[mask] += rng.uniform(-0.14, 0.14, nm)
            th[mask] += rng.uniform(-0.24, 0.24, nm)
        q_list.append(full_to_free_sym(x, y, th, L0))
        m_list.append(np.concatenate([mask.astype(np.uint8)] * 3))
        ls_list.append(scheds[s_idx % len(scheds)])
        labels.append(f"tb4m_{cname}_{s_idx}")
    return q_list, m_list, ls_list, labels


def eval_target(
    n: int, cands: list[dict], L_inc_dec: Decimal, n_threads: int = 4, n_seeds: int = 20, do_mechwalk_on_win: bool = False
) -> dict:
    """Evaluate assembled candidates via direct SLP + C++ masked melt + SLP."""
    t0 = time.time()
    best_res = None

    for c_idx, cand in enumerate(cands):
        if best_res is not None and cand["L"] > float(L_inc_dec) + 0.004 and cand["pen0"] > 1e-4:
            print(f"  [cand {c_idx+1}/{len(cands)}] SKIP {cand['tag']} (L0={cand['L']:.6f}, pen0={cand['pen0']:.4f})", flush=True)
            continue

        tc = time.time()
        ax, ay, at, aL, sbox, tag = cand["x"], cand["y"], cand["th"], cand["L"], cand["sbox"], cand["tag"]
        print(
            f"  [cand {c_idx+1}/{len(cands)}] {tag} (L0={aL:.6f}, inc={float(L_inc_dec):.6f}, pen0={cand['pen0']:.4e})",
            flush=True,
        )

        if cand["pen0"] <= 1e-4:
            direct = polish_and_certify(n, ax, ay, at, L_inc_dec, f"{tag}_direct", do_mechwalk=do_mechwalk_on_win)
            if direct.get("ok"):
                print(
                    f"    -> direct ok dL={direct['dL']:+.6e} L={direct['L_lit'][:14]} ({time.time()-tc:.1f}s)",
                    flush=True,
                )
                if direct["is_real_win"]:
                    save_win_thinb4(n, direct, direct["label"])
                    L_inc_dec = min(L_inc_dec, max(Decimal(direct["L_lit"]), Decimal(direct["L_pd"])))
                if best_res is None or direct["dL"] < best_res["dL"]:
                    best_res = direct

        rng = np.random.default_rng(52000 + n * 19 + c_idx)
        q_list, m_list, ls_list, labels = build_thinb4_melt_seeds(
            n, ax, ay, at, aL, sbox, rng, L_inc=float(L_inc_dec), L_expand=0.045, n_seeds=n_seeds
        )
        qb, qf, lf, _, po, _ = run_cpp_masked_melt(
            n, 1, q_list, m_list, ls_list, [5, 8, 12, 16],
            a_start=0.82, n_melt_stages=4, masked_iters=180, full_iters=180,
            n_threads=n_threads, base_seed=62000 + n * 23 + c_idx * 100,
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
            pol = pool.map(_polish_worker, tasks)

        ok_list = [r for r in pol if r.get("ok")]
        fail_pens = [round(r.get("pen", 99.0), 4) for r in pol if not r.get("ok")]
        if ok_list:
            b = min(ok_list, key=lambda r: r["dL"])
            print(
                f"    -> feas={n_feas}/{len(labels)} min_po={min_po:.4e} best_dL={b['dL']:+.6e} L={b['L_lit'][:14]} ({time.time()-tc:.1f}s)",
                flush=True,
            )
            if b["is_real_win"]:
                save_win_thinb4(n, b, b["label"])
                L_inc_dec = min(L_inc_dec, max(Decimal(b["L_lit"]), Decimal(b["L_pd"])))
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
