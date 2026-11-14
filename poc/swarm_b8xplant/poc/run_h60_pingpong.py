#!/usr/bin/env python3
"""Hypothesis H60: Ping-pong ratcheting with B4_10x16_mw632 (f=0.637920), B5_20x11_n970 (f=0.638712),
B4_11x16_n849 (f=0.637029), B4_12x16_n849 (f=0.636883), and peeled B4_9x16 on n=737, 909, 970, 684, 850."""
from __future__ import annotations

import json
import sys
import time
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import b8_lib as bl  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
import brick_rect_lib as bslp  # noqa: E402
import run_scale as rs  # noqa: E402
import run_track_e_h49 as rte  # noqa: E402

MW_632_PATH = Path(
    "/home/lucas/.gemini/antigravity-cli/worktrees/a4fb0b93-b63b-426d-aa12-67454e99bc08/swarm-mechwalk-8abd0e01/poc/swarm_mechwalk/data/b6s2_win_n632.json"
)
B5_970_PATH = Path(
    "/home/lucas/.gemini/antigravity-cli/worktrees/a4fb0b93-b63b-426d-aa12-67454e99bc08/swarm-b5xplant-d7159aed/poc/swarm_b5xplant/data/h53_win_n970.json"
)
THIN_849_PATH = Path(
    "/home/lucas/.gemini/antigravity-cli/worktrees/a4fb0b93-b63b-426d-aa12-67454e99bc08/swarm-thinb4-f5748c47/poc/swarm_thinb4/data/win_n849.json"
)
B5_737_PATH = Path(
    "/home/lucas/.gemini/antigravity-cli/worktrees/a4fb0b93-b63b-426d-aa12-67454e99bc08/swarm-b5xplant-d7159aed/poc/swarm_b5xplant/data/h53_win_n737.json"
)
B5_909_PATH = Path(
    "/home/lucas/.gemini/antigravity-cli/worktrees/a4fb0b93-b63b-426d-aa12-67454e99bc08/swarm-b5xplant-d7159aed/poc/swarm_b5xplant/data/h53_win_n909.json"
)


def load_json_pack(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    d = json.loads(path.read_text(encoding="utf-8"))
    rs_w = sorted(d["rows"], key=lambda r: int(r["square_id"]))
    return bl.normalize(
        [float(r["x"]) for r in rs_w],
        [float(r["y"]) for r in rs_w],
        [float(r["angle"]) for r in rs_w],
    )


def make_brick(w: int, h: int, g: int, x: np.ndarray, y: np.ndarray, th: np.ndarray) -> dict:
    bx, by, W, H = bl.rect_bbox(x, y, th)
    return {
        "w": w, "h": h, "g": g, "x": bx, "y": by, "th": th.copy(),
        "fx": W - w, "fy": H - h, "f": max(W - w, H - h),
    }


def extract_h60_bricks(frontier: dict[int, dict]) -> dict[str, dict]:
    bricks: dict[str, dict] = {}

    # 1. B4_10x16_mw632 from b6s2_win_n632.json (BR 10x16, N=164)
    x632, y632, t632, _ = load_json_pack(MW_632_PATH)
    hx = bl.half_extent(t632)
    m10x16 = (x632 - hx >= 14.8) & (y632 + hx <= 16.8)
    assert int(m10x16.sum()) == 164, f"mw632 BR10x16 count={int(m10x16.sum())}"
    b10x16_raw = make_brick(10, 16, 4, x632[m10x16], y632[m10x16], t632[m10x16])
    bricks["B4_10x16_mw632_raw"] = b10x16_raw

    # Also square-aspect SLP + walk on B4_10x16_mw632
    sx, sy, st, _, ok = bslp.joint_xytheta_slp_rect(b10x16_raw["x"], b10x16_raw["y"], b10x16_raw["th"], 10, 16, rounds=10)
    if ok:
        res_10 = rs.standalone_brick_walk(10, 16, 4, sx, sy, st, "H60_B4_10x16_mw632")
        bricks["B4_10x16_mw632_walk"] = make_brick(10, 16, 4, res_10["x"], res_10["y"], res_10["th"])
        print(f"[H60 BRICK] B4_10x16_mw632_raw fx={b10x16_raw['fx']:.9f} fy={b10x16_raw['fy']:.9f} | walk f={res_10['f']:.9f}", flush=True)

    # Peel 1 outer column from B4_10x16_mw632_raw (10x16 -> 9x16, 148 sq)
    keep9 = np.ones(len(b10x16_raw["x"]), dtype=bool)
    # Check leftmost vs rightmost 16 unit-like squares; drop the cleaner outer column
    ord_l = np.argsort(b10x16_raw["x"])
    ord_r = np.argsort(-b10x16_raw["x"])
    best_peel = None
    for tag_p, drop_idx in (("dropL", ord_l[:16]), ("dropR", ord_r[:16])):
        k = keep9.copy()
        k[drop_idx] = False
        px, py, Wp, Hp = bl.rect_bbox(b10x16_raw["x"][k], b10x16_raw["y"][k], b10x16_raw["th"][k])
        pt = b10x16_raw["th"][k].copy()
        rx, ry, rt, pen = bslp.relax_rect_lbfgs(px, py, pt, 9.0 + 0.642, 16.0 + 0.642, maxiter=400)
        if pen <= 1e-6:
            spx, spy, spt, spf, okp = bslp.joint_xytheta_slp_rect(rx, ry, rt, 9, 16, rounds=12)
            if okp and (best_peel is None or spf < best_peel[0]):
                best_peel = (spf, spx, spy, spt, tag_p)
    # Also try 9x16 from n=684 (merged.csv)
    if 684 in frontier:
        f684 = frontier[684]
        hx684 = bl.half_extent(f684["th"])
        for t6, m6 in (("684_TL", (f684["x"] + hx684 <= 9.8) & (f684["y"] - hx684 >= 10.0 - 0.1)),
                       ("684_BR", (f684["x"] - hx684 >= 26.0 - 9.1) & (f684["y"] + hx684 <= 16.8))):
            if int(m6.sum()) == 148:
                b9 = make_brick(9, 16, 4, f684["x"][m6], f684["y"][m6], f684["th"][m6])
                if best_peel is None or b9["f"] < best_peel[0]:
                    best_peel = (b9["f"], b9["x"], b9["y"], b9["th"], t6)
    if best_peel is not None:
        res_9 = rs.standalone_brick_walk(9, 16, 4, best_peel[1], best_peel[2], best_peel[3], "H60_B4_9x16_peel")
        bricks["B4_9x16_peel"] = make_brick(9, 16, 4, res_9["x"], res_9["y"], res_9["th"])
        print(f"[H60 BRICK] B4_9x16_peel ({best_peel[4]}): f0={res_9['f0']:.9f} -> f={res_9['f']:.9f} df={res_9['df']:+.6e}", flush=True)

    # 2. B5_20x11_n970 from h53_win_n970.json (BR 20x11, N=225)
    x970, y970, t970, _ = load_json_pack(B5_970_PATH)
    hx970 = bl.half_extent(t970)
    m20x11 = (x970 - hx970 >= 31.0 - 20.1) & (y970 + hx970 <= 11.8)
    assert int(m20x11.sum()) == 225, f"b5_970 BR20x11 count={int(m20x11.sum())}"
    b20x11 = make_brick(20, 11, 5, x970[m20x11], y970[m20x11], t970[m20x11])
    res_20 = rs.standalone_brick_walk(20, 11, 5, b20x11["x"], b20x11["y"], b20x11["th"], "H60_B5_20x11_n970")
    bricks["B5_20x11_n970"] = make_brick(20, 11, 5, res_20["x"], res_20["y"], res_20["th"])
    print(f"[H60 BRICK] B5_20x11_n970: f0={res_20['f0']:.9f} -> f={res_20['f']:.9f} df={res_20['df']:+.6e}", flush=True)

    # 3. B4_11x16_n849 from merged.csv (TL 11x16, N=180)
    f849 = frontier[849]
    hx849 = bl.half_extent(f849["th"])
    m11x16 = (f849["x"] + hx849 <= 11.8) & (f849["y"] - hx849 >= 29.0 - 16.1)
    assert int(m11x16.sum()) == 180, f"merged 849 TL11x16 count={int(m11x16.sum())}"
    b11x16 = make_brick(11, 16, 4, f849["x"][m11x16], f849["y"][m11x16], f849["th"][m11x16])
    res_11 = rs.standalone_brick_walk(11, 16, 4, b11x16["x"], b11x16["y"], b11x16["th"], "H60_B4_11x16_n849")
    bricks["B4_11x16_n849"] = make_brick(11, 16, 4, res_11["x"], res_11["y"], res_11["th"])
    print(f"[H60 BRICK] B4_11x16_n849: f0={res_11['f0']:.9f} -> f={res_11['f']:.9f} df={res_11['df']:+.6e}", flush=True)

    # 4. B4_12x16_n849 from thinb4 win_n849.json (TL 12x16, N=196)
    xt849, yt849, tt849, _ = load_json_pack(THIN_849_PATH)
    hxt849 = bl.half_extent(tt849)
    m12x16 = (xt849 + hxt849 <= 12.8) & (yt849 - hxt849 >= 29.0 - 16.1)
    assert int(m12x16.sum()) == 196, f"thin849 TL12x16 count={int(m12x16.sum())}"
    b12x16 = make_brick(12, 16, 4, xt849[m12x16], yt849[m12x16], tt849[m12x16])
    res_12 = rs.standalone_brick_walk(12, 16, 4, b12x16["x"], b12x16["y"], b12x16["th"], "H60_B4_12x16_n849")
    bricks["B4_12x16_n849"] = make_brick(12, 16, 4, res_12["x"], res_12["y"], res_12["th"])
    print(f"[H60 BRICK] B4_12x16_n849: f0={res_12['f0']:.9f} -> f={res_12['f']:.9f} df={res_12['df']:+.6e}", flush=True)

    return bricks


def build_pinwheel_custom(
    m: int, b1: dict, b2: dict, sf: float = 0.0, f_override: float | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    """Assemble 5-region pinwheel with slack parameter sf in [0, 1] and return (ax, ay, at, L, seam_mask)."""
    w1, h1, w2, h2 = b1["w"], b1["h"], b2["w"], b2["h"]
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
    seam = (
        ((ax >= w1 - 1.5) & (ax <= w1 + f + 1.8))
        | ((ay >= m - h1 - 1.8) & (ay <= m - h1 + 1.8))
        | ((ax >= m - w2 - 1.8) & (ax <= m - w2 + 1.8))
        | ((ay >= h2 - 1.5) & (ay <= h2 + f + 1.8))
    ).astype(np.uint8)
    return ax, ay, at, L, seam


def solve_pinwheel_pair(
    n: int, m: int, b1_base: dict, b2_base: dict, inc_dec: Decimal, tag: str, base_seed: int
) -> tuple[dict | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Test zero-overlap direct SLP + multi-sf/transform seam-melted pinwheels."""
    best_cert: dict | None = None
    best_state: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    # 1. Direct SLP on zero-overlap pinwheel orientations (super clean & fast)
    seeds, masks, scheds, labels = [], [], [], []
    for m1 in ("id", "flipx", "flipy", "rot180"):
        b1 = bl.transform_brick(b1_base, m1)
        for m2 in ("id", "flipx", "flipy", "rot180"):
            b2 = bl.transform_brick(b2_base, m2)
            for sf in (0.0, 0.5, 0.8):
                # Also pre-shifted target L using min(fx/fy) when one brick is thinner
                f_avg = 0.5 * (max(b1["fx"], b1["fy"]) + max(b2["fx"], b2["fy"]))
                f_hi = max(b1["fx"], b1["fy"], b2["fx"], b2["fy"])
                f_lo = min(b1["fy"], b2["fx"], b2["fy"])
                for f_use, ftag in ((f_lo, "flo"), (f_avg, "favg"), (f_hi, "fhi")):
                    ax, ay, at, L0, sm = build_pinwheel_custom(m, b1, b2, sf=sf, f_override=f_use)
                    assert len(ax) == n, f"len={len(ax)} != {n}"
                    pen, _ = bl.cpp_eval_sat_pen_bbox(ax, ay, at)
                    if pen <= 1e-10 and (m1, m2) in (("id", "id"), ("flipx", "flipy"), ("rot180", "rot180")) and ftag == "fhi":
                        sx, sy, st, _, ok = bslp.joint_xytheta_slp_tight(ax, ay, at)
                        if ok:
                            cert = bl.polish_and_certify(n, sx, sy, st, inc_dec, slp_rounds=22)
                            if cert is not None:
                                cert["label"] = f"{tag}_{m1}_{m2}_sf{sf}_direct"
                                if best_cert is None or cert["L_float"] < best_cert["L_float"]:
                                    best_cert = cert
                                    best_state = (sx, sy, st)
                    if sf in (0.0, 0.5, 0.8) and (m1, m2) in (("id", "id"), ("flipx", "id"), ("id", "flipx"), ("rot180", "rot180"), ("flipy", "flipy")):
                        seeds.append((ax, ay, at))
                        masks.append(sm)
                        L_tgt = m + f_use
                        scheds.append([L_tgt + 0.006, L_tgt + 0.001, L_tgt - 2e-4])
                        labels.append(f"{tag}_{m1}_{m2}_sf{sf}_{ftag}")

    # 2. Batch seam-melt + tight SLP on collected seeds
    for siter in (160, 0):
        for start in range(0, len(seeds), 64):
            end = min(len(seeds), start + 64)
            relaxed = bl.su.cpp_batch_melt(
                seeds[start:end], scheds[start:end], sym_mode=1, a_start=0.965, n_melt_stages=3,
                seam_masks=masks[start:end] if siter > 0 else None, seam_iters=siter,
                full_iters=260, n_ruin_iters=0, n_threads=4, base_seed=base_seed + siter + start,
            )
            pool = []
            for bi, row in enumerate(relaxed):
                nx, ny, nth, Lm = bl.normalize(row["x"], row["y"], row["th"])
                lbl = f"{labels[start + bi]}_si{siter}"
                if row["best_L_feas"] is not None:
                    pool.append((0.0, Lm, nx, ny, nth, lbl))
                elif row["pen"] < 0.015:
                    pool.append((1.0 + float(row["pen"]), Lm, nx, ny, nth, lbl))
            pool.sort(key=lambda t: (t[0], t[1]))
            for _, _, cx, cy, cth, lbl in pool[:4]:
                sx, sy, st, _, ok = bslp.joint_xytheta_slp_tight(cx, cy, cth)
                use_x, use_y, use_t = (sx, sy, st) if ok else (cx, cy, cth)
                cert = bl.polish_and_certify(n, use_x, use_y, use_t, inc_dec, slp_rounds=22)
                if cert is not None:
                    cert["label"] = lbl
                    if best_cert is None or cert["L_float"] < best_cert["L_float"]:
                        best_cert = cert
                        rs_c = sorted(cert["rows"], key=lambda r: int(r["square_id"]))
                        best_state = (
                            np.array([float(r["x"]) for r in rs_c]),
                            np.array([float(r["y"]) for r in rs_c]),
                            np.array([float(r["angle"]) for r in rs_c]),
                        )
    return best_cert, *(best_state if best_state is not None else (None, None, None))


def main() -> None:
    t0 = time.time()
    frontier = bl.load_true_frontier_targets([632, 684, 737, 849, 850, 909, 970])
    rte.purge_stale_wins(frontier)
    report: dict = {"bricks": {}, "results": {}}

    # Track pending external baselines for reference logging
    ref_pending = {}
    for rn, rp in ((737, B5_737_PATH), (909, B5_909_PATH), (970, B5_970_PATH)):
        if rp.exists():
            _, _, _, rL = load_json_pack(rp)
            ref_pending[rn] = rL

    bricks = extract_h60_bricks(frontier)
    for k, b in bricks.items():
        report["bricks"][k] = {"w": b["w"], "h": b["h"], "g": b["g"], "fx": b["fx"], "fy": b["fy"], "f": b["f"]}

    # Step 1: n=737 (m=27, r=+8) -> B4_10x16_mw632(10,16) + B4_11x16_n849_T(16,11) and + B4_12x16_n849_T(16,12)? (10+16=26<=26, 16+11=27<=27)
    b11_T = bl.transform_brick(bricks["B4_11x16_n849"], "T")
    best_737 = None
    for b10_key in ("B4_10x16_mw632_raw", "B4_10x16_mw632_walk"):
        if b10_key not in bricks:
            continue
        c, _, _, _ = solve_pinwheel_pair(737, 27, bricks[b10_key], b11_T, frontier[737]["L_dec"], f"h60_737_{b10_key}", 73700)
        if c and (best_737 is None or c["L_float"] < best_737["L_float"]):
            best_737 = c
    if best_737 and best_737["is_real_win"]:
        bl.save_win_locked(best_737)
        print(f"[H60 WIN 737] {best_737['label']} L={best_737['L_dec_str'][:22]} dL={best_737['dL']:+.6e} (vs pending={best_737['L_float']-ref_pending.get(737,99):+.6e})", flush=True)
    report["results"]["737"] = {"win": bool(best_737 and best_737["is_real_win"]), "L": best_737["L_dec_str"] if best_737 else None, "dL": best_737["dL"] if best_737 else None}

    # Step 2: n=909 (m=30, r=+9) -> B4_10x16_mw632(10,16) + B5_20x11_n970(20,11)
    best_909 = None
    best_909_state = None
    for b10_key in ("B4_10x16_mw632_raw", "B4_10x16_mw632_walk"):
        if b10_key not in bricks:
            continue
        c, sx, sy, st = solve_pinwheel_pair(909, 30, bricks[b10_key], bricks["B5_20x11_n970"], frontier[909]["L_dec"], f"h60_909_{b10_key}", 90900)
        if c and (best_909 is None or c["L_float"] < best_909["L_float"]):
            best_909 = c
            best_909_state = (sx, sy, st)
    if best_909 and best_909["is_real_win"]:
        bl.save_win_locked(best_909)
        print(f"[H60 WIN 909] {best_909['label']} L={best_909['L_dec_str'][:22]} dL={best_909['dL']:+.6e} (vs pending={best_909['L_float']-ref_pending.get(909,99):+.6e})", flush=True)
    report["results"]["909"] = {"win": bool(best_909 and best_909["is_real_win"]), "L": best_909["L_dec_str"] if best_909 else None, "dL": best_909["dL"] if best_909 else None}

    # Extract ratcheted B5_20x11 from best_909_state if available, and walk it!
    b20_970_list = [bricks["B5_20x11_n970"]]
    if best_909_state[0] is not None:
        x9, y9, t9 = best_909_state
        hx9 = bl.half_extent(t9)
        m20r = (x9 - hx9 >= 30.0 - 20.1) & (y9 + hx9 <= 11.8)
        if int(m20r.sum()) == 225:
            b20r = make_brick(20, 11, 5, x9[m20r], y9[m20r], t9[m20r])
            res_20r = rs.standalone_brick_walk(20, 11, 5, b20r["x"], b20r["y"], b20r["th"], "H60_B5_20x11_ratchet909")
            b20_rat = make_brick(20, 11, 5, res_20r["x"], res_20r["y"], res_20r["th"])
            print(f"[H60 RATCHET 20x11] f={b20_rat['f']:.9f} (vs n970 {bricks['B5_20x11_n970']['f']:.9f})", flush=True)
            b20_970_list.insert(0, b20_rat)

    # Step 3: n=970 (m=31, r=+9) -> B4_11x16_n849(11,16) + ratcheted B5_20x11(20,11)
    best_970 = None
    for b20_cand in b20_970_list:
        c, _, _, _ = solve_pinwheel_pair(970, 31, bricks["B4_11x16_n849"], b20_cand, frontier[970]["L_dec"], "h60_970", 97000)
        if c and (best_970 is None or c["L_float"] < best_970["L_float"]):
            best_970 = c
    if best_970 and best_970["is_real_win"]:
        bl.save_win_locked(best_970)
        print(f"[H60 WIN 970] {best_970['label']} L={best_970['L_dec_str'][:22]} dL={best_970['dL']:+.6e} (vs pending={best_970['L_float']-ref_pending.get(970,99):+.6e})", flush=True)
    report["results"]["970"] = {"win": bool(best_970 and best_970["is_real_win"]), "L": best_970["L_dec_str"] if best_970 else None, "dL": best_970["dL"] if best_970 else None}

    # Step 4: n=684 (m=26, r=+8) & n=850 (m=29, r=+9) using B4_9x16_peel
    if "B4_9x16_peel" in bricks:
        b9 = bricks["B4_9x16_peel"]
        b10_T = bl.transform_brick(bricks.get("B4_10x16_mw632_walk", bricks["B4_10x16_mw632_raw"]), "T")
        c684, _, _, _ = solve_pinwheel_pair(684, 26, b9, b10_T, frontier[684]["L_dec"], "h60_684", 68400)
        if c684 and c684["is_real_win"]:
            bl.save_win_locked(c684)
            print(f"[H60 WIN 684] {c684['label']} L={c684['L_dec_str'][:22]} dL={c684['dL']:+.6e}", flush=True)
        report["results"]["684"] = {"win": bool(c684 and c684["is_real_win"]), "L": c684["L_dec_str"] if c684 else None, "dL": c684["dL"] if c684 else None}

        c850, _, _, _ = solve_pinwheel_pair(850, 29, b9, b20_970_list[0], frontier[850]["L_dec"], "h60_850", 85000)
        if c850 and c850["is_real_win"]:
            bl.save_win_locked(c850)
            print(f"[H60 WIN 850] {c850['label']} L={c850['L_dec_str'][:22]} dL={c850['dL']:+.6e}", flush=True)
        report["results"]["850"] = {"win": bool(c850 and c850["is_real_win"]), "L": c850["L_dec_str"] if c850 else None, "dL": c850["dL"] if c850 else None}

    rte.rebuild_final_csv()
    (bl.DOCS_DIR / "h60_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"H60 COMPLETE in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
