#!/usr/bin/env python3
"""Track E (Broadcast compressed bricks B5_22x8, B6_24x8, B10_22x11, B8_10x18_NEW) + H49 (walk B5_10x20, B4_18x9 from win_n850)."""
from __future__ import annotations

import json
import math
import sys
import time
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

import b8_lib as bl  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
import brick_rect_lib as bslp  # noqa: E402
import run_scale as rs  # noqa: E402

ALL_TARGET_NS = sorted({
    638, 639, 684, 690, 691, 737, 740, 744, 745,
    792, 793, 794, 795, 796, 798, 799, 800,
    849, 850, 851, 852, 854, 856, 857, 858,
    908, 909, 910, 911, 913, 914, 916, 917, 918,
    969, 970, 971, 972, 973, 977,
})


def load_saved_brick(path: Path) -> dict:
    d = json.loads(path.read_text(encoding="utf-8"))
    x = np.array(d["x"], dtype=np.float64)
    y = np.array(d["y"], dtype=np.float64)
    th = np.array(d["th"], dtype=np.float64)
    x, y, W, H = bl.rect_bbox(x, y, th)
    w, h = int(d["w"]), int(d["h"])
    return {
        "w": w, "h": h, "g": int(d["g"]),
        "x": x, "y": y, "th": th,
        "fx": W - w, "fy": H - h, "f": max(W - w, H - h),
        "name": path.stem.replace("brick_", ""),
    }


def extract_h49_bricks() -> list[tuple[str, dict]]:
    """Extract B5_10x20 & B4_18x9 (win_n850), clean_632 17x11 (win_n909), B6_10x21 & B5_20x10 (win_n972)."""
    out: list[tuple[str, dict]] = []

    def _load_norm(p: Path):
        d = json.loads(p.read_text(encoding="utf-8"))
        rows = sorted(d["rows"], key=lambda r: int(r["square_id"]))
        return bl.normalize(
            [float(r["x"]) for r in rows],
            [float(r["y"]) for r in rows],
            [float(r["angle"]) for r in rows],
        )

    def _mk(w: int, h: int, g: int, x: np.ndarray, y: np.ndarray, th: np.ndarray, mask: np.ndarray, exp: int) -> dict:
        assert int(mask.sum()) == exp, f"Expected {exp} in {w}x{h}+{g}, got {int(mask.sum())}"
        bx, by, W, H = bl.rect_bbox(x[mask], y[mask], th[mask])
        return {"w": w, "h": h, "g": g, "x": bx, "y": by, "th": th[mask].copy(), "fx": W - w, "fy": H - h, "f": max(W - w, H - h)}

    x850, y850, t850, L850 = _load_norm(bl.REPO_ROOT / "poc" / "swarm_homocasc" / "data" / "win_n850.json")
    hx850 = bl.half_extent(t850)
    out.append(("B5_10x20_n850", _mk(10, 20, 5, x850, y850, t850, (x850 + hx850 <= 10.8) & (y850 - hx850 >= L850 - 20.8), 205)))
    out.append(("B4_18x9_n850", _mk(18, 9, 4, x850, y850, t850, (x850 - hx850 >= L850 - 18.8) & (y850 + hx850 <= 9.8), 166)))

    x909, y909, t909, _ = _load_norm(bl.REPO_ROOT / "poc" / "swarm_b5xplant" / "data" / "win_n909.json")
    hx909 = bl.half_extent(t909)
    out.append(("clean_632_17x11_n909", _mk(17, 11, 4, x909, y909, t909, (x909 - hx909 >= 30.0 - 17.1) & (y909 + hx909 <= 11.8), 191)))

    x972, y972, t972, _ = _load_norm(bl.REPO_ROOT / "poc" / "swarm_b5xplant" / "data" / "win_n972.json")
    hx972 = bl.half_extent(t972)
    out.append(("B6_10x21_n972", _mk(10, 21, 6, x972, y972, t972, (x972 - hx972 >= 31.0 - 10.1) & (y972 + hx972 <= 21.8), 216)))
    out.append(("B5_20x10_n972", _mk(20, 10, 5, x972, y972, t972, (x972 + hx972 <= 20.8) & (y972 - hx972 >= 31.0 - 10.1), 205)))
    return out


def rebuild_final_csv() -> None:
    """Rebuild poc/swarm_b8xplant/final.csv deterministically from saved data/win_n*.json files."""
    groups: dict[int, list[dict]] = {}
    for wp in sorted(bl.DATA_DIR.glob("win_n*.json")):
        d = json.loads(wp.read_text(encoding="utf-8"))
        groups[int(d["n"])] = sorted(d["rows"], key=lambda r: int(r["square_id"]))
    bl.FINAL_CSV.write_text(bl.auto_union.final_text(groups), encoding="utf-8")


def purge_stale_wins(frontier: dict[int, dict]) -> None:
    """Drop any win_n*.json that no longer strictly beats the merged origin/main frontier, then rebuild final.csv."""
    for wp in sorted(bl.DATA_DIR.glob("win_n*.json")):
        d = json.loads(wp.read_text(encoding="utf-8"))
        n = int(d["n"])
        if n not in frontier:
            frontier.update(bl.load_true_frontier_targets([n]))
        cur_dec = max(Decimal(d["L_lit"]), Decimal(d["L_pandas"]))
        if cur_dec >= frontier[n]["L_dec"] - Decimal("1e-9"):
            print(f"[PURGE STALE] n={n} saved={str(cur_dec)[:20]} vs frontier={str(frontier[n]['L_dec'])[:20]}", flush=True)
            wp.unlink()
    rebuild_final_csv()


def try_n972_transplant(b5_22x8: dict, b6_24x8: dict, entry_972: dict) -> dict | None:
    """Replace TL 22x8 (181 sq) and BR 8x24 (198 sq) in n=972 directly with compressed bricks."""
    x0, y0, t0, L0 = bl.normalize(entry_972["x"], entry_972["y"], entry_972["th"])
    hx = bl.half_extent(t0)
    m = 31
    m_tl = (x0 + hx <= 22.8) & (y0 - hx >= m - 8.1)
    m_br = (x0 - hx >= m - 8.1) & (y0 + hx <= 24.8)
    if int(m_tl.sum()) != 181 or int(m_br.sum()) != 198 or np.any(m_tl & m_br):
        return None
    rem = ~(m_tl | m_br)

    # Shift remaining interior/corner squares smoothly towards target L
    f_tgt = max(b5_22x8["f"], b6_24x8["f"])
    L_tgt = m + f_tgt
    b6_T = bl.transform_brick(b6_24x8, "T")

    seeds, masks, scheds, labels = [], [], [], []
    rx, ry, rt = x0[rem], y0[rem], t0[rem]
    # Note interior grid splits at x~23, y~23: squares above/right move with +f_tgt instead of +f0
    f0 = L0 - m
    rx_shift = rx.copy()
    ry_shift = ry.copy()
    rx_shift[rx > 15.0] += (f_tgt - f0)
    ry_shift[ry > 15.0] += (f_tgt - f0)

    for mode1 in ("id", "flipx", "flipy", "rot180"):
        br1 = bl.transform_brick(b5_22x8, mode1)
        for mode2 in ("id", "flipx", "flipy", "rot180"):
            br2 = bl.transform_brick(b6_T, mode2)
            ax = np.concatenate([rx_shift, br1["x"], br2["x"] + m - (br2["w"] + br2["fx"] - br2["w"])])
            # Place br1 in TL [0, 22+f_tgt] x [23+f_tgt-br1_h?, 31+f_tgt] flush outer and inner variants
            # Flush outer: br1 at x=0, y=L_tgt - (8 + br1['fy']); br2 at x=L_tgt - (8 + br2['fx']), y=0
            ax1 = np.concatenate([rx_shift, br1["x"], br2["x"] + L_tgt - (br2["w"] + br2["fx"])])
            ay1 = np.concatenate([ry_shift, br1["y"] + L_tgt - (br1["h"] + br1["fy"]), br2["y"]])
            at1 = np.concatenate([rt, br1["th"], br2["th"]])
            seam_m = np.zeros(972, dtype=np.uint8)
            seam_m[: len(rx_shift)] = ((rx > 21.0) | (ry > 21.0)).astype(np.uint8)
            # Only melt boundary collar around seams
            c_m = ((ax1 >= 21.0) & (ax1 <= 24.5)) | ((ay1 >= 21.0) & (ay1 <= 25.5))
            seeds.append((ax1, ay1, at1))
            masks.append(c_m.astype(np.uint8))
            scheds.append([L_tgt + 0.012, L_tgt + 0.002, L_tgt - 1e-4])
            labels.append(f"x972_{mode1}_{mode2}")

    # Test direct SLP on best flush outer + melt all 16
    relaxed = bl.su.cpp_batch_melt(
        seeds, scheds, sym_mode=1, a_start=0.96, n_melt_stages=3,
        seam_masks=masks, seam_iters=180, full_iters=260, n_ruin_iters=0, n_threads=4, base_seed=97200,
    )
    pool = []
    for bi, row in enumerate(relaxed):
        nx, ny, nth, Lm = bl.normalize(row["x"], row["y"], row["th"])
        if row["best_L_feas"] is not None:
            pool.append((0.0, Lm, nx, ny, nth, labels[bi]))
        elif row["pen"] < 0.02:
            pool.append((1.0 + float(row["pen"]), Lm, nx, ny, nth, labels[bi]))
    pool.sort(key=lambda t: (t[0], t[1]))
    best_cert = None
    for _, _, cx, cy, cth, lbl in pool[:4]:
        cert = bl.polish_and_certify(972, cx, cy, cth, entry_972["L_dec"], slp_rounds=18)
        if cert is not None:
            cert["label"] = lbl
            if best_cert is None or cert["L_float"] < best_cert["L_float"]:
                best_cert = cert
    return best_cert


def try_n639_transplant(b8_new: dict, entry_639: dict) -> dict | None:
    """Replace BR 10x18 (188 sq) in n=639 with compressed brick_B8_10x18_NEW."""
    x0, y0, t0, L0 = bl.normalize(entry_639["x"], entry_639["y"], entry_639["th"])
    hx = bl.half_extent(t0)
    m_br = (x0 >= 14.99) & (y0 <= 18.78)
    assert int(m_br.sum()) == 188
    rem = ~m_br
    L_tgt = 25.0 + b8_new["f"]
    seeds, masks, scheds, labels = [], [], [], []
    for mode in ("id", "flipx", "flipy", "rot180"):
        br = bl.transform_brick(b8_new, mode)
        ax = np.concatenate([x0[rem], br["x"] + L_tgt - (br["w"] + br["fx"])])
        ay = np.concatenate([y0[rem], br["y"]])
        at = np.concatenate([t0[rem], br["th"]])
        cm = ((ax >= 13.8) & (ax <= 16.5)) | ((ay >= 17.2) & (ay <= 20.0))
        seeds.append((ax, ay, at))
        masks.append(cm.astype(np.uint8))
        scheds.append([L_tgt + 0.008, L_tgt + 0.001, L_tgt - 1e-4])
        labels.append(f"x639_{mode}")
    relaxed = bl.su.cpp_batch_melt(
        seeds, scheds, sym_mode=1, a_start=0.965, n_melt_stages=3,
        seam_masks=masks, seam_iters=140, full_iters=260, n_ruin_iters=0, n_threads=4, base_seed=63900,
    )
    pool = []
    for bi, row in enumerate(relaxed):
        nx, ny, nth, Lm = bl.normalize(row["x"], row["y"], row["th"])
        if row["best_L_feas"] is not None:
            pool.append((0.0, Lm, nx, ny, nth, labels[bi]))
        elif row["pen"] < 0.02:
            pool.append((1.0 + float(row["pen"]), Lm, nx, ny, nth, labels[bi]))
    pool.sort(key=lambda t: (t[0], t[1]))
    best_cert = None
    for _, _, cx, cy, cth, lbl in pool[:3]:
        cert = bl.polish_and_certify(639, cx, cy, cth, entry_639["L_dec"], slp_rounds=16)
        if cert is not None:
            cert["label"] = lbl
            if best_cert is None or cert["L_float"] < best_cert["L_float"]:
                best_cert = cert
    return best_cert


def try_n800_compressed(b8_new: dict, entry_800: dict) -> dict | None:
    """Re-run n=800 5-region pinwheel transplant with compressed brick_B8_10x18_NEW (f=0.774495438)."""
    f = b8_new["f"]
    L_target = 28.0 + f
    gx, gy = [], []
    for i in range(18):
        for j in range(18):
            gx.append(i + 0.5)
            gy.append(j + 0.5)
    for i in range(10):
        for j in range(10):
            gx.append(18.0 + f + i + 0.5)
            gy.append(18.0 + f + j + 0.5)
    gx_arr = np.array(gx, dtype=np.float64)
    gy_arr = np.array(gy, dtype=np.float64)
    gt_arr = np.zeros(len(gx), dtype=np.float64)

    seeds, masks, scheds, labels = [], [], [], []
    for br_mode in ("id", "flipx", "flipy", "rot180"):
        br = bl.transform_brick(b8_new, br_mode)
        tl = bl.transform_brick(br, "T")
        ax = np.concatenate([tl["x"], br["x"] + 18.0, gx_arr])
        ay = np.concatenate([tl["y"] + 18.0, br["y"], gy_arr])
        at = np.concatenate([tl["th"], br["th"], gt_arr])
        for rad, tag_r in ((3.2, "r3"), (5.0, "r5"), (7.2, "r7")):
            L0 = L_target + 0.018
            sc = L0 / L_target
            near = ((ax - 18.4) ** 2 + (ay - 18.4) ** 2) <= rad * rad
            near = near | ((ax >= 16.5) & (ax <= 20.5) & (ay >= 16.5) & (ay <= 20.5))
            seeds.append((ax * sc, ay * sc, at.copy()))
            masks.append(near.astype(np.uint8))
            scheds.append([L0, L_target + 0.005, L_target + 0.0005, L_target - 0.0005])
            labels.append(f"x800_{br_mode}_{tag_r}")

    relaxed = bl.su.cpp_batch_melt(
        seeds, scheds, sym_mode=1, a_start=0.95, n_melt_stages=4,
        seam_masks=masks, seam_iters=220, full_iters=280, n_ruin_iters=0, n_threads=4, base_seed=80011,
    )
    pool = []
    for bi, row in enumerate(relaxed):
        nx, ny, nth, Lm = bl.normalize(row["x"], row["y"], row["th"])
        if row["best_L_feas"] is not None:
            pool.append((0.0, Lm, nx, ny, nth, labels[bi]))
        elif row["pen"] < 0.02:
            pool.append((1.0 + float(row["pen"]), Lm, nx, ny, nth, labels[bi]))
    pool.sort(key=lambda t: (t[0], t[1]))
    best_cert = None
    for _, _, cx, cy, cth, lbl in pool[:4]:
        cert = bl.polish_and_certify(800, cx, cy, cth, entry_800["L_dec"], slp_rounds=18)
        if cert is not None:
            cert["label"] = lbl
            if best_cert is None or cert["L_float"] < best_cert["L_float"]:
                best_cert = cert
    return best_cert


def assemble_pinwheel_any(m: int, b1: dict, b2: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, float] | None:
    """Assemble 5-region pinwheel for disjoint OR 1-corner (w1+w2==m, h1+h2==m) layouts."""
    asm = bslp.assemble_pinwheel(m, b1, b2, align_mode="center")
    if asm is not None:
        return asm
    w1, h1, w2, h2 = b1["w"], b1["h"], b2["w"], b2["h"]
    # Case 2: w1+w2 <= m and h1+h2 <= m-1 (still 100% disjoint when f < 1)
    # Case 3: w1+w2 == m and h1+h2 == m (only single f x f corner overlap, resolved by junction melt)
    if not ((w1 + w2 <= m and h1 + h2 <= m - 1) or (h1 + h2 <= m and w1 + w2 <= m - 1) or (w1 + w2 == m and h1 + h2 == m)):
        return None
    f = max(b1["f"], b2["f"])
    L = m + f
    dx1 = 0.5 * (f - b1["fx"])
    dy1 = (m - h1) + 0.5 * (f - b1["fy"])
    dx2 = (m - w2) + 0.5 * (f - b2["fx"])
    dy2 = 0.5 * (f - b2["fy"])
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
    if len(gx) + len(b1["x"]) + len(b2["x"]) != m * m + b1["g"] + b2["g"]:
        return None
    ax = np.concatenate([b1["x"] + dx1, b2["x"] + dx2, np.array(gx, dtype=np.float64)])
    ay = np.concatenate([b1["y"] + dy1, b2["y"] + dy2, np.array(gy, dtype=np.float64)])
    at = np.concatenate([b1["th"], b2["th"], np.zeros(len(gx), dtype=np.float64)])
    return ax, ay, at, L


def try_n909_clean632(clean632: dict, entry_909: dict) -> dict | None:
    """Replace BR 17x11 (191 sq) in n=909 with compressed brick_clean_632_17x11_n909 (f=0.638861557)."""
    x0, y0, t0, L0 = bl.normalize(entry_909["x"], entry_909["y"], entry_909["th"])
    hx = bl.half_extent(t0)
    m_br = (x0 - hx >= 30.0 - 17.1) & (y0 + hx <= 11.8)
    if int(m_br.sum()) != 191:
        return None
    rem = ~m_br
    L_tgt = 30.0 + clean632["f"]
    seeds, masks, scheds, labels = [], [], [], []
    for mode in ("id", "flipx", "flipy", "rot180"):
        br = bl.transform_brick(clean632, mode)
        ax = np.concatenate([x0[rem], br["x"] + L_tgt - (br["w"] + br["fx"])])
        ay = np.concatenate([y0[rem], br["y"]])
        at = np.concatenate([t0[rem], br["th"]])
        cm = ((ax >= 11.8) & (ax <= 14.8)) | ((ay >= 10.2) & (ay <= 13.0))
        seeds.append((ax, ay, at))
        masks.append(cm.astype(np.uint8))
        scheds.append([L_tgt + 0.008, L_tgt + 0.001, L_tgt - 2e-4])
        labels.append(f"x909_c632_{mode}")
    relaxed = bl.su.cpp_batch_melt(
        seeds, scheds, sym_mode=1, a_start=0.965, n_melt_stages=3,
        seam_masks=masks, seam_iters=160, full_iters=260, n_ruin_iters=0, n_threads=4, base_seed=90900,
    )
    pool = []
    for bi, row in enumerate(relaxed):
        nx, ny, nth, Lm = bl.normalize(row["x"], row["y"], row["th"])
        if row["best_L_feas"] is not None:
            pool.append((0.0, Lm, nx, ny, nth, labels[bi]))
        elif row["pen"] < 0.02:
            pool.append((1.0 + float(row["pen"]), Lm, nx, ny, nth, labels[bi]))
    pool.sort(key=lambda t: (t[0], t[1]))
    best_cert = None
    for _, _, cx, cy, cth, lbl in pool[:3]:
        cert = bl.polish_and_certify(909, cx, cy, cth, entry_909["L_dec"], slp_rounds=16)
        if cert is not None:
            cert["label"] = lbl
            if best_cert is None or cert["L_float"] < best_cert["L_float"]:
                best_cert = cert
    return best_cert


def try_pinwheel_broadcast(bricks_to_cast: list[dict], all_bricks: dict[tuple[int, int, int], dict], frontier: dict[int, dict]) -> list[dict]:
    """Broadcast compressed bricks across m=23..31 via assemble_pinwheel_any + seam melt + SLP."""
    wins: list[dict] = []
    pool_b = list(all_bricks.values()) + bricks_to_cast
    seen_combos: set[tuple[int, int, int, int, int, int, int]] = set()
    for b1 in bricks_to_cast:
        for b1_ori in (b1, bl.transform_brick(b1, "T")):
            for b2 in pool_b:
                for b2_ori in (b2, bl.transform_brick(b2, "T")):
                    for m in range(23, 32):
                        n = m * m + b1_ori["g"] + b2_ori["g"]
                        if n > 1000 or n not in frontier:
                            continue
                        key = (n, b1_ori["w"], b1_ori["h"], b1_ori["g"], b2_ori["w"], b2_ori["h"], b2_ori["g"])
                        if key in seen_combos:
                            continue
                        asm = assemble_pinwheel_any(m, b1_ori, b2_ori)
                        if asm is None:
                            continue
                        seen_combos.add(key)
                        inc_dec = frontier[n]["L_dec"]
                        ax, ay, at, L0 = asm
                        if L0 >= float(inc_dec) + 0.0015:
                            continue
                        w1, h1 = b1_ori["w"], b1_ori["h"]
                        w2, h2 = b2_ori["w"], b2_ori["h"]
                        corner_ov = (w1 + w2 == m and h1 + h2 == m)
                        if not corner_ov:
                            cert = bl.polish_and_certify(n, ax, ay, at, inc_dec, slp_rounds=16)
                            if cert is not None and cert["is_real_win"]:
                                cert["label"] = f"pw_{w1}x{h1}+{w2}x{h2}_m{m}"
                                saved = bl.save_win_locked(cert)
                                print(f"[TRACK E PW WIN] n={n} {cert['label']} L={cert['L_dec_str'][:20]} dL={cert['dL']:+.6e} saved={saved}", flush=True)
                                wins.append(cert)
                                frontier[n]["L_dec"] = max(Decimal(cert["L_lit"]), Decimal(cert["L_pandas"]))
                                continue
                        f_val = L0 - m
                        sm = (
                            ((ax >= w1 - 1.2) & (ax <= w1 + f_val + 1.5))
                            | ((ay >= m - h1 - 1.5) & (ay <= m - h1 + 1.5))
                            | ((ax >= m - w2 - 1.5) & (ax <= m - w2 + 1.5))
                            | ((ay >= h2 - 1.2) & (ay <= h2 + f_val + 1.5))
                        ).astype(np.uint8)
                        relaxed = bl.su.cpp_batch_melt(
                            [(ax, ay, at)], [[L0 + 0.015, L0 + 0.002, L0 - 2e-4]],
                            sym_mode=1, a_start=0.955, n_melt_stages=3,
                            seam_masks=[sm], seam_iters=180, full_iters=240, n_ruin_iters=0, n_threads=4, base_seed=n * 13,
                        )
                        r0 = relaxed[0]
                        nx, ny, nth, _ = bl.normalize(r0["x"], r0["y"], r0["th"])
                        cert2 = bl.polish_and_certify(n, nx, ny, nth, inc_dec, slp_rounds=16)
                        if cert2 is not None and cert2["is_real_win"]:
                            cert2["label"] = f"pw_melt_{w1}x{h1}+{w2}x{h2}_m{m}"
                            saved = bl.save_win_locked(cert2)
                            print(f"[TRACK E PW-MELT WIN] n={n} {cert2['label']} L={cert2['L_dec_str'][:20]} dL={cert2['dL']:+.6e} saved={saved}", flush=True)
                            wins.append(cert2)
                            frontier[n]["L_dec"] = max(Decimal(cert2["L_lit"]), Decimal(cert2["L_pandas"]))
    return wins


def main() -> None:
    t0 = time.time()
    frontier = bl.load_true_frontier_targets(list(range(520, 1001)))
    purge_stale_wins(frontier)
    report: dict = {"h49_bricks": [], "track_e": [], "wins": []}

    # Load existing compressed bricks from Track C
    compressed_bricks: list[dict] = []
    for fname in ("brick_B5_22x8.json", "brick_B6_24x8.json", "brick_B10_22x11.json", "brick_B8_10x18_NEW.json"):
        p = bl.DATA_DIR / fname
        if p.exists():
            b = load_saved_brick(p)
            compressed_bricks.append(b)
            print(f"[LOADED] {fname}: {b['w']}x{b['h']} (+{b['g']}) f={b['f']:.9f}", flush=True)

    # H49: Reuse already-walked brick JSONs if present, else run standalone_brick_walk
    h49_extracted = extract_h49_bricks()
    f744 = frontier[744]
    hx744 = bl.half_extent(f744["th"])
    m5x15 = (f744["x"] + hx744 <= 5.85) & (f744["y"] - hx744 >= 27.0 - 15.85)
    if int(m5x15.sum()) == 80:
        x5, y5, W5, H5 = bl.rect_bbox(f744["x"][m5x15], f744["y"][m5x15], f744["th"][m5x15])
        h49_extracted.append(
            ("B5_5x15_n744", {"w": 5, "h": 15, "g": 5, "x": x5, "y": y5, "th": f744["th"][m5x15].copy(), "fx": W5 - 5.0, "fy": H5 - 15.0, "f": max(W5 - 5.0, H5 - 15.0)})
        )

    h49_walked_dicts: list[dict] = []
    for bname, binit in h49_extracted:
        saved_p = bl.DATA_DIR / f"brick_{bname}.json"
        if saved_p.exists():
            raw = json.loads(saved_p.read_text(encoding="utf-8"))
            r = {"name": bname, "w": raw["w"], "h": raw["h"], "g": raw["g"], "f0": raw["f0"], "f": raw["f"], "df": raw["df"]}
            print(f"[H49 CACHED] {bname}: f0={r['f0']:.9f} -> f={r['f']:.9f} df={r['df']:+.6e}", flush=True)
        else:
            r = rs.standalone_brick_walk(binit["w"], binit["h"], binit["g"], binit["x"], binit["y"], binit["th"], bname)
            print(f"[H49 BRICK] {bname}: f0={r['f0']:.9f} -> f={r['f']:.9f} df={r['df']:+.6e}", flush=True)
        report["h49_bricks"].append({k: v for k, v in r.items() if k not in ("x", "y", "th")})
        bd = load_saved_brick(saved_p)
        h49_walked_dicts.append(bd)
        if r["df"] <= -1e-6 or bname in ("B5_10x20_n850", "B4_18x9_n850", "clean_632_17x11_n909", "B6_10x21_n972", "B5_20x10_n972"):
            compressed_bricks.append(bd)

    # Track E1/E2/E3: Direct transplants
    b_map = {b["name"]: b for b in compressed_bricks}
    if "B5_22x8" in b_map and "B6_24x8" in b_map:
        c972 = try_n972_transplant(b_map["B5_22x8"], b_map["B6_24x8"], frontier[972])
        if c972 is not None:
            msg972 = f"L={c972['L_dec_str'][:20]} dL={c972['dL']:+.6e} win={c972['is_real_win']}"
            print(f"[TRACK E 972] {msg972}", flush=True)
            report["track_e"].append({"n": 972, "msg": msg972})
            if c972["is_real_win"]:
                bl.save_win_locked(c972)
                report["wins"].append(972)

    if "clean_632_17x11_n909" in b_map:
        c909 = try_n909_clean632(b_map["clean_632_17x11_n909"], frontier[909])
        msg909 = f"L={c909['L_dec_str'][:20]} dL={c909['dL']:+.6e} win={c909['is_real_win']}" if c909 else "no_feas"
        print(f"[TRACK E 909] {msg909}", flush=True)
        report["track_e"].append({"n": 909, "msg": msg909})
        if c909 and c909["is_real_win"]:
            bl.save_win_locked(c909)
            report["wins"].append(909)

    if "B8_10x18_NEW" in b_map:
        c800 = try_n800_compressed(b_map["B8_10x18_NEW"], frontier[800])
        msg800 = f"L={c800['L_dec_str'][:20]} dL={c800['dL']:+.6e} win={c800['is_real_win']}" if c800 else "no_feas"
        print(f"[TRACK E 800] {msg800}", flush=True)
        report["track_e"].append({"n": 800, "msg": msg800})
        if c800 and c800["is_real_win"]:
            bl.save_win_locked(c800)
            report["wins"].append(800)

        c639 = try_n639_transplant(b_map["B8_10x18_NEW"], frontier[639])
        msg639 = f"L={c639['L_dec_str'][:20]} dL={c639['dL']:+.6e} win={c639['is_real_win']}" if c639 else "no_feas"
        print(f"[TRACK E 639] {msg639}", flush=True)
        report["track_e"].append({"n": 639, "msg": msg639})
        if c639 and c639["is_real_win"]:
            bl.save_win_locked(c639)
            report["wins"].append(639)

    # Track E4: Full pinwheel broadcast across all bricks from frontier
    df = pd.read_csv(bl.MERGED_CSV)
    packs = {}
    for nv, grp in df.groupby("n"):
        x, y, th, L = bl.normalize(
            grp["x"].to_numpy(float, copy=True),
            grp["y"].to_numpy(float, copy=True),
            grp["angle"].to_numpy(float, copy=True),
        )
        packs[int(nv)] = (x, y, th, L)
    # Overlay frontier entries onto packs so extract_subbricks uses latest origin/main bricks
    for n, ent in frontier.items():
        packs[n] = (ent["x"], ent["y"], ent["th"], ent["L"])
    all_bricks = bslp.extract_subbricks(packs)
    # Inject all walked bricks into all_bricks cleanly
    for b in compressed_bricks + h49_walked_dicts:
        bk = (b["w"], b["h"], b["g"])
        if bk not in all_bricks or b["f"] < all_bricks[bk]["f"]:
            all_bricks[bk] = b

    pw_wins = try_pinwheel_broadcast(compressed_bricks, all_bricks, frontier)
    for w in pw_wins:
        report["wins"].append(w["n"])

    # Upward L-border extensions from every new win
    for wn in list(dict.fromkeys(report["wins"])):
        wp = bl.DATA_DIR / f"win_n{wn}.json"
        if not wp.exists():
            continue
        wd = json.loads(wp.read_text(encoding="utf-8"))
        rs_w = sorted(wd["rows"], key=lambda r: int(r["square_id"]))
        lwins = rs.try_upward_l_border(
            wn,
            np.array([float(r["x"]) for r in rs_w]),
            np.array([float(r["y"]) for r in rs_w]),
            np.array([float(r["angle"]) for r in rs_w]),
            frontier,
        )
        for lw in lwins:
            report["wins"].append(lw["n"])

    rebuild_final_csv()
    (bl.DOCS_DIR / "track_e_h49_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"TRACK E + H49 COMPLETE in {time.time()-t0:.1f}s wins={report['wins']}", flush=True)


if __name__ == "__main__":
    main()
