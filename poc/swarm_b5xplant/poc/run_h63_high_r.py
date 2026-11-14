"""Hypothesis H63: Newly ratcheted donor sub-bricks + compact g=+6/+7/+8 high-r broadcast."""
from __future__ import annotations

import json
import math
import multiprocessing as mp
import pathlib
import sys
import time
from decimal import Decimal

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
for p in (HERE, HERE.parent.parent / "swarm_interlock" / "poc"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import b5_lib as bl
import frontier_io
import run_h50_h51 as h50
import run_h53_h55 as h53
from cpp_wrap import cpp_eval_sat_pen_bbox, cpp_expand_sym, run_cpp_masked_melt

H63_CACHE = bl.DATA_DIR / "h63_compressed_cache.json"
H63_CKPT = bl.DOCS_DIR / "h63_ckpt.json"


def _load_any_json_pack(path: pathlib.Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    d = json.loads(path.read_text(encoding="utf-8"))
    rows = sorted(d["rows"], key=lambda r: int(r["square_id"]))
    x = np.array([float(r["x"]) for r in rows], dtype=np.float64)
    y = np.array([float(r["y"]) for r in rows], dtype=np.float64)
    ang = np.array([float(r["angle"]) for r in rows], dtype=np.float64)
    th = np.radians(ang) if float(np.max(np.abs(ang))) > 6.3 else ang
    return frontier_io.norm_pack(x, y, th)


def _load_brick_json(path: pathlib.Path, tag: str, src_n: int = 0) -> dict | None:
    if not path.exists():
        return None
    d = json.loads(path.read_text(encoding="utf-8"))
    return bl.make_brick(
        int(d["w"]), int(d["h"]), int(d["g"]),
        np.array(d["x"], dtype=np.float64),
        np.array(d["y"], dtype=np.float64),
        np.array(d["th"], dtype=np.float64),
        src_n, tag,
    )


def build_h63_bank(packs: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, float]]) -> list[dict]:
    """Harvest H63 ratcheted bricks + compact g=+6/+7/+8 bricks + SLP compressions."""
    bank: list[dict] = []
    seen: set[tuple] = set()

    def _add(b: dict | None) -> None:
        if b is None:
            return
        for cand in (b, bl.transpose_brick(b)):
            key = (cand["tag"], cand["w"], cand["h"], cand["g"])
            if key not in seen:
                seen.add(key)
                bank.append(cand)

    cache_by_tag: dict[str, dict] = {}
    if H63_CACHE.exists():
        for item in json.loads(H63_CACHE.read_text(encoding="utf-8")):
            cb = bl.make_brick(
                item["w"], item["h"], item["g"],
                np.array(item["x"], dtype=np.float64),
                np.array(item["y"], dtype=np.float64),
                np.array(item["th"], dtype=np.float64),
                item["src_n"], item["tag"],
            )
            if cb is not None:
                cache_by_tag[cb["tag"]] = cb
                _add(cb)

    to_compress: list[tuple[dict, int, int]] = []

    def _add_with_slp(b: dict | None, n_trials: int = 10, seed: int = 42) -> None:
        if b is None:
            return
        _add(b)
        slp_tag = b["tag"] + "_slp"
        if slp_tag not in cache_by_tag:
            to_compress.append((b, n_trials, seed))

    # 1. Newly ratcheted mechwalk n=970 & b8xplant n=909, n=737 donors
    p_mw970 = bl.DATA_DIR / "donor_h63_mw970.json"
    if p_mw970.exists():
        x, y, th, L = _load_any_json_pack(p_mw970)
        _add(h53._extract_corner_cut(x, y, th, L, "TL", 11, 16, 4, 970, "B4_11x16_mw970", margin=0.76))
        _add(h53._extract_corner_cut(x, y, th, L, "BR", 19, 11, 4, 970, "B4_19x11_mw970", margin=0.76))

    p_b8_909 = bl.DATA_DIR / "donor_h63_b8_909.json"
    if p_b8_909.exists():
        x, y, th, L = _load_any_json_pack(p_b8_909)
        _add(h53._extract_corner_cut(x, y, th, L, "BR", 20, 11, 5, 909, "B5_20x11_b8_909", margin=0.76))
        _add(h53._extract_corner_cut(x, y, th, L, "BR", 19, 11, 4, 909, "B4_19x11_b8_909", margin=0.76))
        _add(h53._extract_corner_cut(x, y, th, L, "TL", 16, 16, 4, 909, "B4_16x16_b8_909", margin=0.76))

    p_b8_737 = bl.DATA_DIR / "donor_h63_b8_737.json"
    if p_b8_737.exists():
        x, y, th, L = _load_any_json_pack(p_b8_737)
        _add(h53._extract_corner_cut(x, y, th, L, "TL", 11, 16, 4, 737, "B4_11x16_b8_737", margin=0.76))
        _add(h53._extract_corner_cut(x, y, th, L, "BR", 16, 11, 4, 737, "B4_16x11_b8_737", margin=0.76))

    # 2. Standalone b8xplant bricks (compact high-r + compressed B5/B6)
    standalone_specs = [
        ("donor_h63_brick_B5_20x11_n970.json", "B5_20x11_n970_h60", 970, False),
        ("donor_h63_brick_B6_10x21_n972.json", "B6_10x21_n972_b8", 972, True),
        ("donor_h63_brick_B5_20x10_n972.json", "B5_20x10_n972_b8", 972, False),
        ("donor_h63_brick_B6_24x8.json", "B6_24x8_b8", 0, True),
        ("donor_h63_brick_B5_22x8.json", "B5_22x8_b8", 0, True),
        ("donor_h63_brick_B5_22x7.json", "B5_22x7_b8", 0, True),
        ("donor_h63_brick_B7_17x8.json", "B7_17x8_b8", 0, True),
        ("donor_h63_brick_B8_10x18.json", "B8_10x18_b8", 0, True),
    ]
    for idx_s, (fname, tag, sn, do_slp) in enumerate(standalone_specs):
        bk = _load_brick_json(bl.DATA_DIR / fname, tag, sn)
        if do_slp:
            _add_with_slp(bk, n_trials=10, seed=6300 + idx_s)
        else:
            _add(bk)

    # 3. Compact bricks harvested from origin/main (d12df26+)
    extractions = [
        (971, "TL", 11, 19, 4, "B4_11x19_n971", 0.72, False),
        (971, "TL", 11, 20, 5, "B5_11x20_n971", 0.72, True),
        (741, "TL", 15, 9, 4, "B4_15x9_n741", 0.76, True),
        (741, "TL", 21, 9, 6, "B6_21x9_n741", 0.76, True),
        (741, "BR", 20, 18, 6, "B6_20x18_n741", 0.76, True),
        (796, "TL", 17, 11, 5, "B5_17x11_n796", 0.76, True),
        (796, "TL", 21, 13, 6, "B6_21x13_n796", 0.76, True),
        (797, "TL", 8, 15, 5, "B5_8x15_n797", 0.78, True),
        (797, "BR", 18, 12, 6, "B6_18x12_n797", 0.78, True),
        (797, "BR", 19, 12, 7, "B7_19x12_n797", 0.78, True),
        (854, "TL", 18, 11, 6, "B6_18x11_n854", 0.78, True),
        (854, "TL", 19, 11, 7, "B7_19x11_n854", 0.78, True),
        (854, "BR", 21, 18, 6, "B6_21x18_n854", 0.76, True),
        (742, "BR", 19, 21, 7, "B7_19x21_n742", 0.78, True),
        (974, "TR", 21, 16, 6, "B6_21x16_n974", 0.76, True),
        (974, "TR", 21, 18, 7, "B7_21x18_n974", 0.76, True),
        (974, "TR", 21, 21, 8, "B8_21x21_n974", 0.76, False),
    ]
    for idx_e, (sn, corner, w, h, g, tag, mg, do_slp) in enumerate(extractions):
        if sn not in packs:
            continue
        x, y, th, L = packs[sn]
        bk = h53._extract_corner_cut(x, y, th, L, corner, w, h, g, sn, tag, margin=mg)
        if do_slp:
            _add_with_slp(bk, n_trials=10, seed=6400 + idx_e)
        else:
            _add(bk)

    if to_compress:
        print(f"Compressing {len(to_compress)} H63 bricks via SLP...", flush=True)
        for b, nt, sd in to_compress:
            cb = h53.compress_brick_slp(b, n_trials=nt, seed=sd)
            if cb is not None:
                print(f"  {b['tag']} (f={b['f']:.6f}) -> {cb['tag']} (fx={cb['fx']:.6f}, fy={cb['fy']:.6f})", flush=True)
                _add(cb)
                cache_by_tag[cb["tag"]] = cb
        cdata = [
            {
                "w": cb["w"], "h": cb["h"], "g": cb["g"],
                "src_n": cb["src_n"], "tag": cb["tag"],
                "x": cb["x"].tolist(), "y": cb["y"].tolist(), "th": cb["th"].tolist(),
            }
            for cb in cache_by_tag.values()
        ]
        H63_CACHE.write_text(json.dumps(cdata), encoding="utf-8")

    # Merge H53-H55 + H50-H51 bank (reuses cached compressed bricks)
    for b in h53.build_h53_bank(packs):
        _add(b)
    return bank


def assemble_preshifted_flex(
    m: int,
    r: int,
    b1: dict,
    b2: dict,
    s1: int,
    s2: int,
    sf: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, tuple[float, float, float, float]] | None:
    """Zero-overlap (kx<=0, ky<=0 including corner touch) assembly with pre-shift fraction sf."""
    if b1["g"] + b2["g"] != r:
        return None
    sb1 = bl.sym_brick(b1, s1)
    sb2 = bl.sym_brick(b2, s2)
    w1, h1 = sb1["w"], sb1["h"]
    w2, h2 = sb2["w"], sb2["h"]
    if w1 >= m or h1 >= m or w2 >= m or h2 >= m:
        return None
    kx = w1 + w2 - m
    ky = h1 + h2 - m
    if kx > 0 or ky > 0:
        return None

    f1 = max(sb1["fx"], sb1["fy"])
    f2 = max(sb2["fx"], sb2["fy"])
    f_hi = max(f1, f2)
    f_lo = min(f1, f2)
    f = f_hi - sf * (f_hi - f_lo)
    L = float(m) + f

    x1, y1, t1 = sb1["x"].copy(), sb1["y"] + (L - (h1 + sb1["fy"])), sb1["th"].copy()
    x2, y2, t2 = sb2["x"] + (L - (w2 + sb2["fx"])), sb2["y"].copy(), sb2["th"].copy()

    tilt1 = bl.tilted_mask(t1)
    tilt2 = bl.tilted_mask(t2)
    if np.any(tilt1) and np.any(tilt2):
        tpen, _ = cpp_eval_sat_pen_bbox(
            np.concatenate([x1[tilt1], x2[tilt2]]),
            np.concatenate([y1[tilt1], y2[tilt2]]),
            np.concatenate([t1[tilt1], t2[tilt2]]),
        )
        if tpen > 1e-4:
            return None

    n_tr_w, n_tr_h = m - w1, m - h2
    n_bl_w, n_bl_h = m - w2, m - h1
    tr_cells = {(i, j) for i in range(n_tr_w) for j in range(n_tr_h)}
    bl_cells = {(i, j) for i in range(n_bl_w) for j in range(n_bl_h)}
    if kx < 0 and ky < 0:
        bl_cells = {ij for ij in bl_cells if not (ij[0] >= w1 and ij[1] >= h2)}

    tr_xs = np.array([L - n_tr_w + i + 0.5 for (i, _) in sorted(tr_cells)], dtype=np.float64)
    tr_ys = np.array([L - n_tr_h + j + 0.5 for (_, j) in sorted(tr_cells)], dtype=np.float64)
    bl_xs = np.array([i + 0.5 for (i, _) in sorted(bl_cells)], dtype=np.float64)
    bl_ys = np.array([j + 0.5 for (_, j) in sorted(bl_cells)], dtype=np.float64)

    ax = np.concatenate([x1, x2, tr_xs, bl_xs])
    ay = np.concatenate([y1, y2, tr_ys, bl_ys])
    at = np.concatenate([t1, t2, np.zeros(len(tr_xs) + len(bl_xs), dtype=np.float64)])
    if len(ax) != m * m + r:
        return None
    seam_box = (
        max(0.0, float(min(w1, m - w2)) - 1.2),
        min(L, float(max(w1 + f, m - w2)) + 1.2),
        max(0.0, float(min(h2, m - h1)) - 1.2),
        min(L, float(max(h2 + f, m - h1)) + 1.2),
    )
    return ax, ay, at, L, seam_box


PRIORITY_PAIRS_H63: dict[int, list[tuple[str, str]]] = {
    910: [
        ("B5_20x11_n970_h60", "B5_20x10_n909_slp_T"),
        ("B5_20x11_n970_h60", "B5_20x10_n972_b8_T"),
        ("B5_11x20_n971_slp", "B5_20x10_n909_slp"),
        ("B4_10x16_n632_slp", "B6_10x21_n972_b8_slp"),
        ("B4_9x16_n684_slp_T", "B6_10x21_n972_b8_slp"),
    ],
    850: [
        ("B4_9x16_n684_slp", "B5_20x11_n970_h60"),
        ("B4_11x16_n849", "B5_20x11_n970_h60"),
        ("B4_10x16_n632_slp", "B5_20x11_n970_h60"),
        ("B4_11x16_mw970", "B5_20x11_n970_h60"),
    ],
    851: [
        ("B4_10x16_n632_slp_T", "B6_10x21_n972_b8_slp"),
        ("B4_18x9_n850", "B6_10x21_n972_b8_slp"),
        ("B4_9x16_n684_slp_T", "B6_10x21_n972_b8_slp"),
        ("B5_20x11_n970_h60", "B5_20x10_n909_slp_T"),
    ],
    972: [
        ("B5_20x11_n970_h60", "B6_10x21_n972_b8_slp"),
        ("B5_20x10_n909_slp", "B6_10x21_n972_b8_slp"),
        ("B5_11x20_n971_slp_T", "B6_10x21_n972_b8_slp"),
    ],
    911: [
        ("B5_20x11_n970_h60", "B6_10x21_n972_b8_slp"),
        ("B5_20x10_n909_slp", "B6_10x21_n972_b8_slp"),
        ("B5_22x8_b8_slp_T", "B6_10x21_n972_b8_slp_T"),
        ("B5_22x7_b8_slp_T", "B6_10x21_n972_b8_slp_T"),
    ],
    795: [
        ("B5_10x19_n850_T", "B6_9x18_n796_slp"),
        ("B5_22x8_b8_slp_T", "B6_21x9_n741_slp"),
        ("B5_22x7_b8_slp_T", "B6_21x9_n741_slp"),
        ("B5_17x11_n796_slp", "B6_9x18_n796_slp"),
        ("B5_8x15_n797_slp", "B6_21x13_n796_slp"),
    ],
    852: [
        ("B5_20x10_n909_slp", "B6_9x18_n796_slp"),
        ("B5_20x11_n970_h60", "B6_9x18_n796_slp"),
        ("B5_22x8_b8_slp", "B6_9x18_n796_slp"),
        ("B5_10x19_n850", "B6_9x18_n796_slp_T"),
    ],
    912: [
        ("B6_10x21_n972_b8_slp", "B6_9x18_n796_slp_T"),
        ("B6_10x21_n972_b8_slp", "B6_21x9_n741_slp"),
        ("B6_10x21_n972_b8_slp", "B6_18x11_n854_slp"),
        ("B5_22x8_b8_slp", "B7_17x8_b8_slp_T"),
    ],
    742: [
        ("B6_9x18_n796_slp", "B7_17x8_b8_slp"),
        ("B5_8x15_n797_slp", "B8_10x18_b8_slp_T"),
        ("B6_21x9_n741_slp_T", "B7_17x8_b8_slp"),
        ("B6_18x11_n854_slp_T", "B7_19x12_n797_slp_T"),
    ],
    797: [
        ("B6_10x21_n972_b8_slp", "B7_17x8_b8_slp"),
        ("B6_9x18_n796_slp", "B7_17x8_b8_slp"),
        ("B5_10x19_n850", "B8_10x18_b8_slp"),
        ("B5_8x15_n797_slp", "B8_10x18_b8_slp_T"),
    ],
    854: [
        ("B6_10x21_n972_b8_slp", "B7_17x8_b8_slp"),
        ("B6_10x21_n972_b8_slp", "B7_19x11_n854_slp"),
        ("B5_20x10_n909_slp", "B8_10x18_b8_slp"),
        ("B5_22x8_b8_slp", "B8_10x18_b8_slp"),
    ],
    913: [
        ("B6_10x21_n972_b8_slp", "B7_17x8_b8_slp"),
        ("B6_10x21_n972_b8_slp", "B7_19x11_n854_slp"),
        ("B5_20x10_n909_slp", "B8_10x18_b8_slp"),
        ("B5_20x11_n970_h60", "B8_10x18_b8_slp"),
    ],
    974: [
        ("B5_20x11_n970_h60", "B8_10x18_b8_slp"),
        ("B6_10x21_n972_b8_slp", "B7_19x11_n854_slp"),
        ("B6_10x21_n972_b8_slp", "B7_17x8_b8_slp"),
        ("B6_10x21_n972_b8_slp", "B7_21x18_n974_slp_T"),
    ],
}


def find_h63_candidates(n: int, bank: list[dict], L_inc: float, max_cands: int = 8) -> list[dict]:
    m = int(math.isqrt(n))
    r = n - m * m
    f_inc = L_inc - m
    by_tag = {b["tag"]: b for b in bank}

    raw_pairs: list[tuple[float, dict, dict, bool]] = []
    seen_canon: set[tuple] = set()

    max_f_allow = f_inc + (0.065 if r >= 11 else 0.030)
    max_ov_sum = 3 if r >= 10 else 2

    def _consider(b1: dict, b2: dict, is_prio: bool) -> None:
        if b1["g"] + b2["g"] != r or b1["w"] >= m or b1["h"] >= m or b2["w"] >= m or b2["h"] >= m:
            return
        kx, ky = b1["w"] + b2["w"] - m, b1["h"] + b2["h"] - m
        zero_ov = (kx <= 0 and ky <= 0)
        if not zero_ov and not (max(0, kx) + max(0, ky) <= max_ov_sum and kx <= 2 and ky <= 2):
            return
        if not is_prio and max(b1["fx"], b1["fy"], b2["fx"], b2["fy"]) > max_f_allow:
            return
        canon = (b1["tag"], b2["tag"])
        if canon in seen_canon:
            return
        seen_canon.add(canon)
        ov_pen = 0.0 if zero_ov else 0.015 * (max(0, kx) + max(0, ky))
        f_eff = 0.5 * (b1["f"] + b2["f"]) + 0.5 * max(b1["f"], b2["f"])
        score = ov_pen + f_eff - (0.08 if is_prio else 0.0)
        raw_pairs.append((score, b1, b2, zero_ov))

    for t1, t2 in PRIORITY_PAIRS_H63.get(n, []):
        if t1 in by_tag and t2 in by_tag:
            _consider(by_tag[t1], by_tag[t2], True)

    good = [b for b in bank if b["f"] <= max_f_allow and b["w"] < m and b["h"] < m]
    for b1 in good:
        for b2 in good:
            _consider(b1, b2, False)

    raw_pairs.sort(key=lambda z: z[0])
    cands: list[dict] = []
    used_pair_keys: set[tuple] = set()

    for _, b1, b2, zero_ov in raw_pairs:
        pk = (b1["tag"].replace("_slp", ""), b2["tag"].replace("_slp", ""), zero_ov)
        if pk in used_pair_keys and len(cands) >= max_cands:
            continue
        if zero_ov:
            sfs = (0.0, 0.5, 0.8) if abs(b1["f"] - b2["f"]) > 5e-4 else (0.0,)
            for sf in sfs:
                best = None
                for s1 in range(4):
                    for s2 in range(4):
                        asm = assemble_preshifted_flex(m, r, b1, b2, s1, s2, sf=sf)
                        if asm is None:
                            continue
                        ax, ay, at, aL, sbox = asm
                        pen, _ = cpp_eval_sat_pen_bbox(ax, ay, at)
                        if best is None or pen < best[0]:
                            best = (pen, ax, ay, at, aL, sbox, f"z_sf{int(sf*10)}_s{s1}{s2}")
                if best is not None:
                    pen, ax, ay, at, aL, sbox, cfg = best
                    cands.append({
                        "x": ax, "y": ay, "th": at, "L": aL, "seam_box": sbox,
                        "b1": b1, "b2": b2, "pen0": pen, "sf": sf,
                        "name": f"{b1['tag']}+{b2['tag']}_{cfg}",
                    })
                    used_pair_keys.add(pk)
        else:
            fmodes = ("tight", "mid") if abs(b1["f"] - b2["f"]) > 3e-3 else ("tight",)
            for fm in fmodes:
                best = None
                for s1 in range(4):
                    for s2 in range(4):
                        asm = bl.assemble_b5_pair(m, r, b1, b2, s1, s2, f_mode=fm, drop_from="b1")
                        if asm is None:
                            continue
                        ax, ay, at, aL, sbox = asm
                        pen, _ = cpp_eval_sat_pen_bbox(ax, ay, at)
                        if best is None or pen < best[0]:
                            best = (pen, ax, ay, at, aL, sbox, f"ov_{fm[0]}_s{s1}{s2}")
                if best is not None:
                    pen, ax, ay, at, aL, sbox, cfg = best
                    cands.append({
                        "x": ax, "y": ay, "th": at, "L": aL, "seam_box": sbox,
                        "b1": b1, "b2": b2, "pen0": pen, "sf": 0.0,
                        "name": f"{b1['tag']}+{b2['tag']}_{cfg}",
                    })
                    used_pair_keys.add(pk)
        if len(cands) >= max_cands * 2:
            break

    return cands[:max_cands]


def save_h63_win(n: int, res: dict, label: str) -> None:
    """Save H63 win strictly to h63_win_n{n}.json without touching win_n*.json or final.csv."""
    bl.DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = dict(res, n=n, label=label)
    (bl.DATA_DIR / f"h63_win_n{n}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


def eval_target_h63(n: int, bank: list[dict], L_inc_dec: Decimal, max_cands: int = 8) -> dict:
    t0 = time.time()
    m = int(math.isqrt(n))
    r = n - m * m
    cands = find_h63_candidates(n, bank, float(L_inc_dec), max_cands=max_cands)
    print(f"\n[H63] n={n} (m={m}, r=+{r}), L_inc={L_inc_dec}, cands={len(cands)}", flush=True)
    if not cands:
        return {"n": n, "status": "NOCAND", "dt": round(time.time() - t0, 1)}

    best_res = None
    any_win = False
    for idx, c in enumerate(cands, 1):
        tc = time.time()
        L_cur = float(L_inc_dec)
        print(f"  [cand {idx}/{len(cands)}] {c['name']} (L0={c['L']:.6f}, inc={L_cur:.6f}, pen0={c['pen0']:.4f})", flush=True)

        if c["pen0"] <= 0.06:
            direct = bl.polish_and_certify(n, c["x"], c["y"], c["th"], L_inc_dec, f"{c['name']}_direct", do_mechwalk=False)
            if direct.get("ok"):
                print(f"    -> direct ok dL={direct['dL']:+.6e} L={direct['L_lit'][:14]} ({time.time()-tc:.1f}s)", flush=True)
                if direct["is_real_win"]:
                    save_h63_win(n, direct, direct["label"])
                    L_inc_dec = min(L_inc_dec, Decimal(direct["L_lit"]))
                    any_win = True
                if best_res is None or Decimal(direct["L_lit"]) < Decimal(best_res["L_lit"]):
                    best_res = direct

        rng = np.random.default_rng(63000 + n * 17 + idx)
        q_list, m_list, ls_list, labels = h50.build_h50_melt_seeds(
            n, c["x"], c["y"], c["th"], c["L"], c["seam_box"], rng, L_inc=float(L_inc_dec), pen0=c["pen0"], n_seeds=24
        )
        qb, qf, lf, _, po, _ = run_cpp_masked_melt(
            n, 1, q_list, m_list, ls_list, [5, 8, 12, 16],
            a_start=0.82, n_melt_stages=4, masked_iters=220, full_iters=220,
            n_threads=4, base_seed=67000 + n * 29 + idx * 100,
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
            tasks.append((n, x, y, th, str(L_inc_dec), f"{c['name']}_{labels[i]}", False))

        with mp.Pool(processes=min(4, len(tasks))) as pool:
            pol = pool.map(bl._polish_worker_mp, tasks)

        ok_list = [res_i for res_i in pol if res_i.get("ok")]
        fail_pens = [round(res_i.get("pen", 99.0), 4) for res_i in pol if not res_i.get("ok")]
        if ok_list:
            b = min(ok_list, key=lambda res_i: Decimal(res_i["L_lit"]))
            print(f"    -> feas={n_feas}/{len(labels)} min_po={min_po:.4e} best_dL={b['dL']:+.6e} L={b['L_lit'][:14]} ({time.time()-tc:.1f}s)", flush=True)
            if b["is_real_win"]:
                save_h63_win(n, b, b["label"])
                L_inc_dec = min(L_inc_dec, Decimal(b["L_lit"]))
                any_win = True
                print(f"    *** NEW H63 WIN n={n}: L_lit={b['L_lit'][:26]} dL={b['dL']:.6e} ({b['label']}) ***", flush=True)
            if best_res is None or Decimal(b["L_lit"]) < Decimal(best_res["L_lit"]):
                best_res = b
        else:
            print(f"    -> feas={n_feas}/{len(labels)} min_po={min_po:.4e} ALL FAILED SLP (pens={fail_pens}, {time.time()-tc:.1f}s)", flush=True)

    if best_res is not None and best_res.get("rows") and Decimal(best_res["L_lit"]) <= L_inc_dec + Decimal("0.002"):
        rows = sorted(best_res["rows"], key=lambda r_i: int(r_i["square_id"]))
        mx = np.array([float(r_i["x"]) for r_i in rows], dtype=np.float64)
        my = np.array([float(r_i["y"]) for r_i in rows], dtype=np.float64)
        mt = np.radians(np.array([float(r_i["angle"]) for r_i in rows], dtype=np.float64))
        mw = bl.polish_and_certify(n, mx, my, mt, L_inc_dec, f"{best_res['label']}_mw", do_mechwalk=True)
        if mw.get("ok") and Decimal(mw["L_lit"]) < Decimal(best_res["L_lit"]):
            if mw["is_real_win"]:
                save_h63_win(n, mw, mw["label"])
                L_inc_dec = min(L_inc_dec, Decimal(mw["L_lit"]))
                any_win = True
                print(f"    *** MECHWALK IMPROVED H63 WIN n={n}: L_lit={mw['L_lit'][:26]} dL={mw['dL']:.6e} ***", flush=True)
            best_res = mw

    dt = round(time.time() - t0, 1)
    if best_res is None:
        return {"n": n, "status": "FAIL", "dt": dt}
    return {
        "n": n,
        "status": "WIN" if any_win else "OK",
        "dL": best_res["dL"],
        "L_lit": best_res["L_lit"],
        "label": best_res["label"],
        "dt": dt,
    }


def main() -> None:
    t0 = time.time()
    bl.DOCS_DIR.mkdir(parents=True, exist_ok=True)
    targets = [910, 850, 851, 972, 911, 795, 852, 912, 742, 797, 854, 913, 974]
    donor_ns = [971, 741, 796, 585, 797, 854, 742, 974, 849, 684, 739, 970, 793]
    packs, L_map = bl.fast_load_frontier(target_ns=sorted(set(targets + donor_ns)))
    bank = build_h63_bank(packs)
    print(f"H63 bank loaded: {len(bank)} oriented bricks", flush=True)
    for tag in [
        "B4_11x16_mw970", "B4_19x11_mw970", "B5_20x11_b8_909", "B4_19x11_b8_909",
        "B4_11x16_b8_737", "B5_20x11_n970_h60", "B6_10x21_n972_b8_slp",
        "B5_22x8_b8_slp", "B6_24x8_b8_slp", "B7_17x8_b8_slp", "B8_10x18_b8_slp",
        "B6_21x9_n741_slp", "B5_8x15_n797_slp", "B7_19x11_n854_slp",
    ]:
        matches = [b for b in bank if b["tag"] == tag]
        if matches:
            b = matches[0]
            print(f"  brick {b['tag']}: {b['w']}x{b['h']} g={b['g']} fx={b['fx']:.6f} fy={b['fy']:.6f}", flush=True)

    ckpt: dict[str, dict] = {}
    if H63_CKPT.exists():
        ckpt = json.loads(H63_CKPT.read_text(encoding="utf-8"))

    wins: list[int] = []
    for idx, n in enumerate(targets, 1):
        if str(n) in ckpt:
            if ckpt[str(n)].get("status") == "WIN":
                wins.append(n)
            print(f"\n[H63 {idx}/{len(targets)}] n={n} already in ckpt ({ckpt[str(n)].get('status')}), skipping", flush=True)
            continue
        res = eval_target_h63(n, bank, L_map[n], max_cands=8)
        print(f"  [Result n={n}] -> {res}", flush=True)
        ckpt[str(n)] = res
        H63_CKPT.write_text(json.dumps(ckpt, indent=2), encoding="utf-8")
        if res.get("status") == "WIN":
            wins.append(n)

    print(f"\n=== H63 Complete in {time.time() - t0:.1f}s | Wins ({len(wins)}): {wins} ===", flush=True)


if __name__ == "__main__":
    main()
