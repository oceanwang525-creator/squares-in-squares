"""Hypotheses H53, H54 & H55: Ultra-thin sub-brick harvest, SLP compression, and zero-overlap/pre-shifted pinwheels."""
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
import brick_slp
import frontier_io
import run_h50_h51 as h50
from cpp_wrap import cpp_eval_sat_pen_bbox, cpp_expand_sym, run_cpp_masked_melt

H53_CACHE = bl.DATA_DIR / "h53_compressed_cache.json"
H53_CKPT = bl.DOCS_DIR / "h53_ckpt.json"


def compress_brick_slp(b: dict, n_trials: int = 16, seed: int = 42) -> dict | None:
    """Compress a brick in its rectangular bounding box using gentle downward LBFGS + joint_xytheta_slp_rect."""
    w, h, g = b["w"], b["h"], b["g"]
    x0, y0, t0 = b["x"], b["y"], b["th"]
    W0, H0 = w + b["fx"], h + b["fy"]
    f0 = b["f"]
    fmin = min(b["fx"], b["fy"])
    rng = np.random.default_rng(seed)
    best = None
    tilt = np.minimum(np.mod(t0, 0.5 * math.pi), 0.5 * math.pi - np.mod(t0, 0.5 * math.pi)) > math.radians(2.0)

    if f0 > 0.67:
        ft_sched = (f0 - 0.005, max(0.665, f0 - 0.025), max(0.650, f0 - 0.045))
    else:
        ft_sched = (f0, max(0.635, fmin + 0.002), max(0.634, fmin - 0.004))

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
                amp = 0.06 if trial % 3 != 0 else 0.03
                cx[mask] += rng.uniform(-amp, amp, nm)
                cy[mask] += rng.uniform(-amp, amp, nm)
                ct[mask] += rng.uniform(-1.5 * amp, 1.5 * amp, nm)
        if trial > 0:
            for ft in ft_sched:
                cx, cy, ct, _ = brick_slp.relax_rect_lbfgs(cx, cy, ct, w + ft, h + ft, maxiter=250)
        sx, sy, st, _, ok = brick_slp.joint_xytheta_slp_rect(
            cx, cy, ct, w, h, rounds=14, tr_th0=0.022, tr_xy0=0.09, tol=1e-11
        )
        if ok:
            cb = bl.make_brick(w, h, g, sx, sy, st, b["src_n"], b["tag"] + "_slp")
            if cb is not None and (best is None or cb["f"] < best["f"]):
                best = cb
    return best


def _extract_corner_cut(
    x: np.ndarray,
    y: np.ndarray,
    th: np.ndarray,
    L: float,
    corner: str,
    w: int,
    h: int,
    g_exp: int,
    src_n: int,
    tag: str,
    margin: float = 0.75,
) -> dict | None:
    hx = frontier_io.hx_of(th)
    if corner == "TL":
        m = (x + hx <= w + margin) & (y - hx >= L - (h + margin))
    elif corner == "TR":
        m = (x - hx >= L - (w + margin)) & (y - hx >= L - (h + margin))
    elif corner == "BL":
        m = (x + hx <= w + margin) & (y + hx <= h + margin)
    else:  # BR
        m = (x - hx >= L - (w + margin)) & (y + hx <= h + margin)
    if int(m.sum()) != w * h + g_exp:
        return None
    return bl.make_brick(w, h, g_exp, x[m], y[m], th[m], src_n, tag)


def build_h53_bank(packs: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, float]]) -> list[dict]:
    """Harvest H53-H55 ultra-thin and g=+6/+5/+4 bricks plus cached SLP compressions."""
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
    if H53_CACHE.exists():
        for item in json.loads(H53_CACHE.read_text(encoding="utf-8")):
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

    def _add_with_slp(b: dict | None, n_trials: int = 16, seed: int = 42) -> None:
        if b is None:
            return
        _add(b)
        slp_tag = b["tag"] + "_slp"
        if slp_tag not in cache_by_tag:
            to_compress.append((b, n_trials, seed))

    # 1. thinb4 win_n909 (L = 30.638903279)
    p909 = bl.DATA_DIR / "donor_thin_win_n909.json"
    if p909.exists():
        x, y, th, L = h50._load_mw_rad_json(p909)
        _add(_extract_corner_cut(x, y, th, L, "BR", 20, 11, 5, 909, "B5_20x11_n909", margin=0.69))
        _add_with_slp(_extract_corner_cut(x, y, th, L, "BR", 20, 10, 5, 909, "B5_20x10_n909", margin=0.72), 20, 9091)
        _add(_extract_corner_cut(x, y, th, L, "TL", 10, 16, 4, 909, "B4_10x16_n909", margin=0.69))

    # 2. thinb4 win_n632 (L = 25.638364706)
    p632 = bl.DATA_DIR / "donor_thin_win_n632.json"
    if p632.exists():
        x, y, th, L = h50._load_mw_rad_json(p632)
        _add(_extract_corner_cut(x, y, th, L, "BR", 11, 16, 4, 632, "B4_11x16_n632", margin=0.69))
        b4_10x16 = _extract_corner_cut(x, y, th, L, "BR", 10, 16, 4, 632, "B4_10x16_n632", margin=0.70)
        _add_with_slp(b4_10x16, 20, 6321)

    # 3. thinb4 win_n849 (L = 29.636883385) + main n=849 (L = 29.637029)
    p849 = bl.DATA_DIR / "donor_thin_win_n849.json"
    if p849.exists():
        x, y, th, L = h50._load_mw_rad_json(p849)
        _add(_extract_corner_cut(x, y, th, L, "TL", 12, 16, 4, 849, "B4_12x16_n849", margin=0.69))
    if 849 in packs:
        x, y, th, L = packs[849]
        _add(_extract_corner_cut(x, y, th, L, "TL", 11, 16, 4, 849, "B4_11x16_n849", margin=0.69))

    # 4. main n=684 (B4_9x16_n684), n=739 (B5_9x17_n739), n=796 (B6_9x18_n796), n=585 (B6_11x18_n585)
    if 684 in packs:
        x, y, th, L = packs[684]
        _add_with_slp(_extract_corner_cut(x, y, th, L, "TL", 9, 16, 4, 684, "B4_9x16_n684", margin=0.69), 20, 6841)
    if 739 in packs:
        x, y, th, L = packs[739]
        _add_with_slp(_extract_corner_cut(x, y, th, L, "TL", 9, 17, 5, 739, "B5_9x17_n739", margin=0.73), 16, 7391)
    if 796 in packs:
        x, y, th, L = packs[796]
        _add_with_slp(_extract_corner_cut(x, y, th, L, "BR", 9, 18, 6, 796, "B6_9x18_n796", margin=0.75), 16, 7961)
    if 585 in packs:
        x, y, th, L = packs[585]
        _add_with_slp(_extract_corner_cut(x, y, th, L, "BR", 11, 18, 6, 585, "B6_11x18_n585", margin=0.75), 16, 5851)

    if to_compress:
        print(f"Compressing {len(to_compress)} H53 bricks via SLP...", flush=True)
        for b, nt, sd in to_compress:
            cb = compress_brick_slp(b, n_trials=nt, seed=sd)
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
        H53_CACHE.write_text(json.dumps(cdata), encoding="utf-8")

    # Merge H50/H51 bank (reusing existing harvested compressed bricks)
    for b in h50.build_h50_bank(packs):
        _add(b)
    return bank


def assemble_preshifted(
    m: int,
    r: int,
    b1: dict,
    b2: dict,
    s1: int,
    s2: int,
    sf: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, tuple[float, float, float, float]] | None:
    """Assemble zero-overlap (kx<=0, ky<=0) pinwheel with pre-shift fraction sf in [0, 1)."""
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
    if kx > 0 or ky > 0 or (w1 + w2 + h1 + h2 > 2 * m - 1):
        return None

    f1 = max(sb1["fx"], sb1["fy"])
    f2 = max(sb2["fx"], sb2["fy"])
    f_hi = max(f1, f2)
    f_lo = min(f1, f2)
    f = f_hi - sf * (f_hi - f_lo)
    L = float(m) + f

    # Place b1 at TL and b2 at BR
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


PRIORITY_PAIRS: dict[int, list[tuple[str, str]]] = {
    970: [
        ("B4_11x16_n849", "B5_20x11_n909"),
        ("B4_12x16_n849", "B5_10x19_n850_T"),
        ("B4_12x16_n849", "B5_10x19_n793_T"),
        ("B4_11x16_n632", "B5_20x11_n909"),
    ],
    737: [
        ("B4_11x16_n632", "B4_11x16_n849_T"),
        ("B4_10x16_n632_slp", "B4_11x16_n849_T"),
        ("B4_10x16_n909", "B4_11x16_n849_T"),
        ("B4_10x16_n632_slp", "B4_10x16_n632_slp_T"),
    ],
    909: [
        ("B4_10x16_n909", "B5_20x11_n909"),
        ("B4_10x16_n632_slp", "B5_20x11_n909"),
        ("B4_10x16_n970", "B5_20x11_n909"),
        ("B4_10x16_n632", "B5_20x11_n909"),
    ],
    971: [
        ("B5_20x10_n909_slp_T", "B5_20x11_n909"),
        ("B4_10x16_n909", "B6_10x21_b4_T"),
        ("B4_10x16_n632_slp", "B6_10x21_b4_T"),
        ("B5_10x20_b4_slp", "B5_20x11_n909"),
    ],
    684: [
        ("B4_9x16_n684_slp", "B4_10x16_n632_slp_T"),
        ("B4_9x16_n684_slp", "B4_10x16_n909_T"),
        ("B4_9x16_n684", "B4_16x10_b4"),
        ("B4_9x16_n684_slp", "B4_16x10_n970"),
    ],
    850: [
        ("B4_9x16_n684_slp", "B5_20x11_n909"),
        ("B4_9x16_n684", "B5_20x11_n909"),
        ("B4_10x16_n909", "B5_10x19_n850_T"),
    ],
    741: [
        ("B6_9x18_n796_slp", "B6_9x18_n796_slp_T"),
        ("B6_9x18_n796", "B6_9x18_n796_T"),
        ("B6_9x18_n796_slp", "B6_9x18_n796_T"),
    ],
    853: [
        ("B6_11x18_n585_slp", "B6_11x18_n585_slp_T"),
        ("B6_11x18_n585", "B6_11x18_n585_T"),
        ("B6_9x18_n796_slp", "B6_11x18_n585_slp_T"),
    ],
    687: [
        ("B5_9x17_n739_slp", "B6_9x18_n796_slp_T"),
        ("B5_9x17_n739", "B6_9x18_n796_T"),
        ("B5_17x9_w911_T", "B6_9x18_n796_slp"),
    ],
    688: [
        ("B6_9x18_n796_slp", "B6_9x18_n796_slp_T"),
        ("B6_9x18_n796", "B6_9x18_n796_T"),
    ],
    796: [
        ("B6_9x18_n796_slp", "B6_9x18_n796_slp_T"),
        ("B6_11x18_n585_slp", "B6_9x18_n796_slp_T"),
        ("B6_9x18_n796", "B6_9x18_n796_T"),
    ],
    912: [
        ("B6_11x18_n585_slp", "B6_11x18_n585_slp_T"),
        ("B6_10x21_b4", "B6_9x18_n796_slp_T"),
        ("B6_10x21_b4", "B6_9x18_n796_T"),
    ],
    852: [
        ("B6_9x18_n796_slp", "B5_20x11_n909"),
        ("B5_9x17_n739_slp", "B6_11x18_n585_slp_T"),
        ("B6_9x18_n796", "B5_20x11_n909"),
    ],
    795: [
        ("B5_9x17_n739_slp", "B6_11x18_n585_slp_T"),
        ("B5_10x19_n850", "B6_9x18_n796_slp_T"),
        ("B5_9x17_n739", "B6_11x18_n585_T"),
    ],
}


def find_h53_candidates(n: int, bank: list[dict], L_inc: float, max_cands: int = 6) -> list[dict]:
    m = int(math.isqrt(n))
    r = n - m * m
    f_inc = L_inc - m
    by_tag = {b["tag"]: b for b in bank}

    raw_pairs: list[tuple[float, dict, dict, bool]] = []
    seen_canon: set[tuple] = set()

    def _consider(b1: dict, b2: dict, is_prio: bool) -> None:
        if b1["g"] + b2["g"] != r or b1["w"] >= m or b1["h"] >= m or b2["w"] >= m or b2["h"] >= m:
            return
        kx, ky = b1["w"] + b2["w"] - m, b1["h"] + b2["h"] - m
        zero_ov = (kx <= 0 and ky <= 0 and (b1["w"] + b2["w"] + b1["h"] + b2["h"] <= 2 * m - 1))
        if not zero_ov and not (kx + ky <= 2 and kx <= 2 and ky <= 2):
            return
        if not is_prio and max(b1["fx"], b1["fy"], b2["fx"], b2["fy"]) > f_inc + 0.03:
            return
        canon = (b1["tag"], b2["tag"])
        if canon in seen_canon:
            return
        seen_canon.add(canon)
        score = (0.0 if zero_ov else 0.02 * (max(0, kx) + max(0, ky))) + max(b1["f"], b2["f"]) - (0.05 if is_prio else 0.0)
        raw_pairs.append((score, b1, b2, zero_ov))

    for t1, t2 in PRIORITY_PAIRS.get(n, []):
        if t1 in by_tag and t2 in by_tag:
            _consider(by_tag[t1], by_tag[t2], True)

    good = [b for b in bank if b["f"] < f_inc + 0.025 and b["w"] < m and b["h"] < m]
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
        sfs = (0.0, 0.5, 0.8) if (zero_ov and abs(b1["f"] - b2["f"]) > 2e-4) else (0.0,)
        if zero_ov:
            for sf in sfs:
                best = None
                for s1 in range(4):
                    for s2 in range(4):
                        asm = assemble_preshifted(m, r, b1, b2, s1, s2, sf=sf)
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
            best = None
            for s1 in range(4):
                for s2 in range(4):
                    asm = bl.assemble_b5_pair(m, r, b1, b2, s1, s2, f_mode="tight", drop_from="b1")
                    if asm is None:
                        continue
                    ax, ay, at, aL, sbox = asm
                    pen, _ = cpp_eval_sat_pen_bbox(ax, ay, at)
                    if best is None or pen < best[0]:
                        best = (pen, ax, ay, at, aL, sbox, f"ov_s{s1}{s2}")
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


def save_h53_win(n: int, res: dict, label: str) -> None:
    """Save H53 win strictly to h53_win_n{n}.json without touching win_n*.json or final.csv."""
    bl.DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = dict(res, n=n, label=label)
    (bl.DATA_DIR / f"h53_win_n{n}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


def eval_target_h53(n: int, bank: list[dict], L_inc_dec: Decimal, max_cands: int = 6) -> dict:
    t0 = time.time()
    m = int(math.isqrt(n))
    r = n - m * m
    cands = find_h53_candidates(n, bank, float(L_inc_dec), max_cands=max_cands)
    print(f"\n[H53] n={n} (m={m}, r=+{r}), L_inc={L_inc_dec}, cands={len(cands)}", flush=True)
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
                    save_h53_win(n, direct, direct["label"])
                    L_inc_dec = min(L_inc_dec, Decimal(direct["L_lit"]))
                    any_win = True
                if best_res is None or Decimal(direct["L_lit"]) < Decimal(best_res["L_lit"]):
                    best_res = direct

        rng = np.random.default_rng(31000 + n * 17 + idx)
        q_list, m_list, ls_list, labels = h50.build_h50_melt_seeds(
            n, c["x"], c["y"], c["th"], c["L"], c["seam_box"], rng, L_inc=float(L_inc_dec), pen0=c["pen0"], n_seeds=24
        )
        qb, qf, lf, _, po, _ = run_cpp_masked_melt(
            n, 1, q_list, m_list, ls_list, [5, 8, 12, 16],
            a_start=0.82, n_melt_stages=4, masked_iters=220, full_iters=220,
            n_threads=4, base_seed=35000 + n * 29 + idx * 100,
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
                save_h53_win(n, b, b["label"])
                L_inc_dec = min(L_inc_dec, Decimal(b["L_lit"]))
                any_win = True
                print(f"    *** NEW H53 WIN n={n}: L_lit={b['L_lit'][:26]} dL={b['dL']:.6e} ({b['label']}) ***", flush=True)
            if best_res is None or Decimal(b["L_lit"]) < Decimal(best_res["L_lit"]):
                best_res = b
        else:
            print(f"    -> feas={n_feas}/{len(labels)} min_po={min_po:.4e} ALL FAILED SLP (pens={fail_pens}, {time.time()-tc:.1f}s)", flush=True)

    # One final mechwalk kick on the single best packing for n if competitive
    if best_res is not None and best_res.get("rows") and Decimal(best_res["L_lit"]) <= L_inc_dec + Decimal("0.002"):
        rows = sorted(best_res["rows"], key=lambda r_i: int(r_i["square_id"]))
        mx = np.array([float(r_i["x"]) for r_i in rows], dtype=np.float64)
        my = np.array([float(r_i["y"]) for r_i in rows], dtype=np.float64)
        mt = np.radians(np.array([float(r_i["angle"]) for r_i in rows], dtype=np.float64))
        mw = bl.polish_and_certify(n, mx, my, mt, L_inc_dec, f"{best_res['label']}_mw", do_mechwalk=True)
        if mw.get("ok") and Decimal(mw["L_lit"]) < Decimal(best_res["L_lit"]):
            if mw["is_real_win"]:
                save_h53_win(n, mw, mw["label"])
                L_inc_dec = min(L_inc_dec, Decimal(mw["L_lit"]))
                any_win = True
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
    targets = [970, 737, 909, 741, 853, 971, 684, 850, 687, 688, 796, 912, 852, 795]
    donor_ns = [849, 684, 739, 796, 585, 970, 793]
    packs, L_map = bl.fast_load_frontier(target_ns=sorted(set(targets + donor_ns)))
    bank = build_h53_bank(packs)
    print(f"H53-H55 bank loaded: {len(bank)} oriented bricks", flush=True)
    for tag in [
        "B5_20x11_n909", "B5_20x10_n909", "B5_20x10_n909_slp", "B4_10x16_n909",
        "B4_11x16_n632", "B4_10x16_n632", "B4_10x16_n632_slp",
        "B4_12x16_n849", "B4_11x16_n849", "B4_9x16_n684", "B4_9x16_n684_slp",
        "B6_9x18_n796", "B6_9x18_n796_slp", "B6_11x18_n585", "B6_11x18_n585_slp",
        "B5_9x17_n739", "B5_9x17_n739_slp",
    ]:
        matches = [b for b in bank if b["tag"] == tag]
        if matches:
            b = matches[0]
            print(f"  brick {b['tag']}: {b['w']}x{b['h']} g={b['g']} fx={b['fx']:.6f} fy={b['fy']:.6f}", flush=True)

    ckpt: dict[str, dict] = {}
    if H53_CKPT.exists():
        ckpt = json.loads(H53_CKPT.read_text(encoding="utf-8"))

    wins: list[int] = []
    for idx, n in enumerate(targets, 1):
        if str(n) in ckpt:
            if ckpt[str(n)].get("status") == "WIN":
                wins.append(n)
            print(f"\n[H53 {idx}/{len(targets)}] n={n} already in ckpt ({ckpt[str(n)].get('status')}), skipping", flush=True)
            continue
        res = eval_target_h53(n, bank, L_map[n], max_cands=6)
        print(f"  [Result n={n}] -> {res}", flush=True)
        ckpt[str(n)] = res
        H53_CKPT.write_text(json.dumps(ckpt, indent=2), encoding="utf-8")
        if res.get("status") == "WIN":
            wins.append(n)

    print(f"\n=== H53-H55 Complete in {time.time() - t0:.1f}s | Wins ({len(wins)}): {wins} ===", flush=True)


if __name__ == "__main__":
    main()
