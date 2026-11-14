#!/usr/bin/env python3
"""Targeted follow-up: tighten win_n639 and test direct brick swaps into d8b3699 n=800 & n=909."""
from __future__ import annotations

import json
import sys
import time
from decimal import Decimal
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import b8_lib as bl  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
import brick_rect_lib as bslp  # noqa: E402
import run_track_e_h49 as rte  # noqa: E402


def try_n800_direct_swap(b8_new: dict, entry_800: dict) -> dict | None:
    """Replace BR 10x18 and/or TL 18x10 inside d8b3699's already-melted n=800 (28.774579277)."""
    x0, y0, t0, L0 = bl.normalize(entry_800["x"], entry_800["y"], entry_800["th"])
    hx = bl.half_extent(t0)
    m = 28
    # Inspect BR 10x18 and TL 18x10 regions in n=800
    m_br = (x0 - hx >= 17.99) & (y0 + hx <= 18.79)
    m_tl = (x0 + hx <= 18.79) & (y0 - hx >= 17.99)
    print(f"[n=800 inspect] m_br={int(m_br.sum())} m_tl={int(m_tl.sum())} overlap={int((m_br & m_tl).sum())}", flush=True)

    L_tgt = 28.0 + b8_new["f"]
    seeds, masks, scheds, labels = [], [], [], []

    # Try replacing BR 10x18 if count == 188 and disjoint from TL
    if int(m_br.sum()) == 188:
        rem = ~m_br
        for mode in ("id", "flipx", "flipy", "rot180"):
            br = bl.transform_brick(b8_new, mode)
            ax = np.concatenate([x0[rem], br["x"] + L_tgt - (br["w"] + br["fx"])])
            ay = np.concatenate([y0[rem], br["y"]])
            at = np.concatenate([t0[rem], br["th"]])
            cm = ((ax >= 16.5) & (ax <= 20.5)) | ((ay >= 16.5) & (ay <= 20.5))
            seeds.append((ax, ay, at))
            masks.append(cm.astype(np.uint8))
            scheds.append([L_tgt + 0.008, L_tgt + 0.001, L_tgt - 1e-4])
            labels.append(f"dswap800_br_{mode}")

    # Try replacing BOTH BR 10x18 and TL 18x10 if counts are 188 & 188 disjoint
    if int(m_br.sum()) == 188 and int(m_tl.sum()) == 188 and not np.any(m_br & m_tl):
        rem2 = ~(m_br | m_tl)
        for mode in ("id", "flipx", "flipy", "rot180"):
            br = bl.transform_brick(b8_new, mode)
            tl = bl.transform_brick(br, "T")
            ax = np.concatenate([x0[rem2], br["x"] + L_tgt - (br["w"] + br["fx"]), tl["x"]])
            ay = np.concatenate([y0[rem2], br["y"], tl["y"] + L_tgt - (tl["h"] + tl["fy"])])
            at = np.concatenate([t0[rem2], br["th"], tl["th"]])
            cm = ((ax >= 16.5) & (ax <= 20.5)) | ((ay >= 16.5) & (ay <= 20.5))
            seeds.append((ax, ay, at))
            masks.append(cm.astype(np.uint8))
            scheds.append([L_tgt + 0.012, L_tgt + 0.002, L_tgt - 1e-4])
            labels.append(f"dswap800_both_{mode}")

    if not seeds:
        return None
    relaxed = bl.su.cpp_batch_melt(
        seeds, scheds, sym_mode=1, a_start=0.96, n_melt_stages=3,
        seam_masks=masks, seam_iters=180, full_iters=260, n_ruin_iters=0, n_threads=4, base_seed=80088,
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
        cert = bl.polish_and_certify(800, cx, cy, cth, entry_800["L_dec"], slp_rounds=20)
        if cert is not None:
            cert["label"] = lbl
            if best_cert is None or cert["L_float"] < best_cert["L_float"]:
                best_cert = cert
    return best_cert


def main() -> None:
    t0 = time.time()
    frontier = bl.load_true_frontier_targets([639, 800, 909])
    rte.purge_stale_wins(frontier)

    # 1. Tighten win_n639 further
    wp639 = bl.DATA_DIR / "win_n639.json"
    if wp639.exists():
        w = json.loads(wp639.read_text(encoding="utf-8"))
        rs_w = sorted(w["rows"], key=lambda r: int(r["square_id"]))
        x = np.array([float(r["x"]) for r in rs_w], dtype=np.float64)
        y = np.array([float(r["y"]) for r in rs_w], dtype=np.float64)
        th = np.array([float(r["angle"]) for r in rs_w], dtype=np.float64)
        sx, sy, st, _, ok = bslp.joint_xytheta_slp_tight(x, y, th)
        if ok:
            cert = bl.polish_and_certify(639, sx, sy, st, frontier[639]["L_dec"], slp_rounds=24)
            if cert and cert["is_real_win"] and max(Decimal(cert["L_lit"]), Decimal(cert["L_pandas"])) < max(Decimal(w["L_lit"]), Decimal(w["L_pandas"])) - Decimal("1e-9"):
                cert["label"] = "x639_tight24"
                bl.save_win_locked(cert)
                print(f"[TIGHT 639 WIN] L={cert['L_dec_str'][:22]} dL={cert['dL']:+.6e}", flush=True)

    # 2. Try direct n=800 swap on d8b3699's n=800
    b8 = rte.load_saved_brick(bl.DATA_DIR / "brick_B8_10x18_NEW.json")
    c800 = try_n800_direct_swap(b8, frontier[800])
    if c800:
        print(f"[DIRECT 800] L={c800['L_dec_str'][:22]} dL={c800['dL']:+.6e} win={c800['is_real_win']}", flush=True)
        if c800["is_real_win"]:
            bl.save_win_locked(c800)

    # 3. Try direct n=909 clean_632 swap on d8b3699's n=909
    c632 = rte.load_saved_brick(bl.DATA_DIR / "brick_clean_632_17x11_n909.json")
    c909 = rte.try_n909_clean632(c632, frontier[909])
    if c909:
        print(f"[DIRECT 909] L={c909['L_dec_str'][:22]} dL={c909['dL']:+.6e} win={c909['is_real_win']}", flush=True)
        if c909["is_real_win"]:
            bl.save_win_locked(c909)

    rte.rebuild_final_csv()
    print(f"DONE in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
