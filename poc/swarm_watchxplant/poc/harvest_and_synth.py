#!/usr/bin/env python3
"""Extract tilted sub-bricks (g=+3..+12) from public_watch_nov14 + today's new champions,
synthesize 1-row/1-col peels & vstrip/hcol splices, compress via SLP+pslp+mechwalk,
and scan m=21..31, r=+6..+18 targets for zero-gap / kx+ky=1 pinwheel opportunities."""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import watchxplant_lib as wx  # noqa: E402

NEW_CHAMPION_NS = [973, 795, 793, 974, 743, 914, 912, 975, 854, 863, 924, 585, 634]


def load_donor_packs() -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, float, str]]:
    """Load public_watch_nov14 (n>=232) + today's new champions."""
    packs: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, float, str]] = {}

    pw_path = wx.REPO_ROOT / "poc" / "public_watch_nov14" / "final.csv"
    if pw_path.exists():
        pw_lit = wx.auto_union.parse_text(pw_path.read_text(encoding="utf-8"))
        for n, rows in pw_lit.items():
            if n >= 232 and len(rows) == n:
                x0, y0, th0 = wx._rows_to_arrays(rows)
                x, y, th, L = wx.bl.normalize(x0, y0, th0)
                packs[int(n)] = (x, y, th, L, f"pw_{n}")

    champ_fr = wx.fast_load_true_frontier(NEW_CHAMPION_NS, certify=False)
    for n, entry in champ_fr.items():
        if n not in packs or entry["L"] < packs[n][3] - 1e-9:
            packs[int(n)] = (entry["x"], entry["y"], entry["th"], entry["L"], f"ch_{n}")
    return packs


def extract_bricks_from_pack(
    n: int,
    x: np.ndarray,
    y: np.ndarray,
    th: np.ndarray,
    L: float,
    tag_prefix: str,
    g_min: int = 3,
    g_max: int = 12,
    max_f: float = 0.85,
) -> list[dict]:
    """Scan all 4 corners for minimal-perimeter tilted sub-bricks with g in [g_min, g_max]."""
    m = int(math.floor(L))
    f = L - m
    if f > max_f + 0.02:
        return []
    deg = np.degrees(np.mod(th, 0.5 * math.pi))
    tilt = np.minimum(deg, 90.0 - deg) > 2.0
    hx = wx.bl.half_extent(th)

    found: list[dict] = []
    for corner in ("TL", "BR", "BL", "TR"):
        by_g: dict[int, tuple[int, float, int, int, np.ndarray]] = {}
        by_shape: dict[tuple[int, int, int], tuple[float, np.ndarray]] = {}
        for w in range(6, min(m, 25)):
            for h in range(6, min(m, 25)):
                if w + h < 18 or w + h > 36 or max(w, h) < 10:
                    continue
                if corner == "TL":
                    mask = (x + hx <= w + f + 0.03) & (y - hx >= m - h - 0.03)
                elif corner == "BR":
                    mask = (x - hx >= m - w - 0.03) & (y + hx <= h + f + 0.03)
                elif corner == "BL":
                    mask = (x + hx <= w + f + 0.03) & (y + hx <= h + f + 0.03)
                else:
                    mask = (x - hx >= m - w - 0.03) & (y - hx >= m - h - 0.03)

                g = int(mask.sum()) - w * h
                if g < g_min or g > g_max:
                    continue
                if int((mask & tilt).sum()) < max(12, (w * h) // 4):
                    continue
                _, _, W, H = wx.bl.rect_bbox(x[mask], y[mask], th[mask])
                fc = max(W - w, H - h)
                if fc > min(max_f, f + 0.002):
                    continue
                if g not in by_g or (w + h, fc) < (by_g[g][0], by_g[g][1]):
                    by_g[g] = (w + h, fc, w, h, mask)
                if fc <= 0.78:
                    sk = (min(w, h), max(w, h), g)
                    if sk not in by_shape or fc < by_shape[sk][0]:
                        by_shape[sk] = (fc, w, h, g, mask)

        kept_masks: dict[tuple[int, int, int], np.ndarray] = {}
        for g, (_, _, w, h, mask) in by_g.items():
            kept_masks[(w, h, g)] = mask
        for _, (_, w, h, g, mask) in by_shape.items():
            kept_masks.setdefault((w, h, g), mask)

        for (w, h, g), mask in kept_masks.items():
            b = wx.make_brick(w, h, g, x[mask], y[mask], th[mask], tag=f"{tag_prefix}_{corner}_{w}x{h}_g{g}")
            pen, _ = wx.cpp_eval_sat_pen_bbox(b["x"], b["y"], b["th"])
            if pen > 1e-7:
                continue
            if b["w"] > b["h"]:
                b = wx.bl.transform_brick(b, "T")
                b["tag"] = f"{tag_prefix}_{corner}_{b['w']}x{b['h']}_g{g}_T"
            found.append(b)
    return found


def _canonical_shape(b: dict) -> tuple[int, int, int]:
    return (min(b["w"], b["h"]), max(b["w"], b["h"]), b["g"])


def main() -> None:
    t0 = time.time()
    wx.ensure_dirs()

    ext_bricks = wx.load_all_pooled_bricks(top_per_shape=2, include_external=True)
    ext_best: dict[tuple[int, int, int], float] = {}
    for b in ext_bricks:
        sk = _canonical_shape(b)
        ext_best[sk] = min(ext_best.get(sk, 1e9), b["f"])
    print(f"Loaded {len(ext_bricks)} external oriented bricks ({len(ext_best)} unique shapes)", flush=True)

    packs = load_donor_packs()
    print(f"Loaded {len(packs)} donor packings (public_watch_nov14 n>=232 + new champions)", flush=True)

    raw_extracted: list[dict] = []
    for n in sorted(packs):
        x, y, th, L, prefix = packs[n]
        raw_extracted.extend(extract_bricks_from_pack(n, x, y, th, L, prefix))
    print(f"Extracted {len(raw_extracted)} raw corner bricks (g=+3..+12)", flush=True)

    # Keep best per canonical shape that either is brand-new or improves/ties external catalog
    cand_by_shape: dict[tuple[int, int, int], list[dict]] = {}
    for b in raw_extracted:
        sk = _canonical_shape(b)
        cand_by_shape.setdefault(sk, []).append(b)

    best_extracted: list[dict] = []
    for sk, lst in sorted(cand_by_shape.items()):
        lst.sort(key=lambda c: c["f"])
        top = lst[0]
        prev = ext_best.get(sk, 1e9)
        if (prev > 1e8 and top["f"] <= 0.785) or top["f"] <= prev + 0.0015:
            best_extracted.append(top)

    # Sort tightest / highest-leverage first; use fast pslp.squeeze (no multi-min brick_mechwalk)
    best_extracted.sort(key=lambda b: (-b["g"] if b["f"] <= 0.76 else 0, b["f"], b["w"] + b["h"]))
    print(f"Selected {len(best_extracted)} novel/competitive extracted bricks (pslp-compressing)...", flush=True)
    saved_extracted: list[dict] = []
    for idx, b in enumerate(best_extracted):
        sk = _canonical_shape(b)
        prev = ext_best.get(sk, 1e9)
        do_slp = (idx < 30 and b["g"] >= 4 and b["f"] <= 0.77)
        cb = wx.compress_brick_pslp(
            b,
            n_trials=(2 if do_slp else 0),
            seed=12200 + idx,
            do_shake=(idx < 60),
        )
        if prev > 1e8 or cb["f"] < prev - 1e-4:
            wx.save_brick_json(cb, cb["tag"])
            saved_extracted.append(cb)
            ext_best[sk] = min(prev, cb["f"])
            tag_new = "NEW_SHAPE" if prev > 1e8 else f"IMPROVED(prev={prev:.6f})"
            print(f"  [EXT+] {cb['tag']}: {cb['w']}x{cb['h']}+g{cb['g']} f={cb['f']:.6f} ({tag_new})", flush=True)

    # Synthesize 1-row/1-col peels + vstrip/hcol splices on top extracted bricks & tightest parents
    synth_parents = sorted(
        [b for b in saved_extracted if b["f"] <= 0.77 and b["g"] in range(4, 11)],
        key=lambda b: (b["f"], b["w"] + b["h"], -b["g"]),
    )[:20]
    print(f"\nSynthesizing peels & vstrip/hcol splices on {len(synth_parents)} extracted parents...", flush=True)

    saved_synth: list[dict] = []
    seed_base = 123000
    for idx, p in enumerate(synth_parents):
        cands: list[tuple[str, dict]] = []
        for side in ("left", "right", "top", "bottom"):
            pb = wx.cp2.peel_outer_unit_layer(p, side, tag=f"{p['tag']}_p{side[0]}")
            if pb is not None:
                cands.append(("PEEL", pb))
        if p["f"] <= 0.755 and p["w"] <= 13 and p["h"] <= 23:
            for sname, sp in wx.gl.splice_vertical_strip(p)[:2]:
                cands.append(("VSTRIP", wx.make_brick(sp["w"], sp["h"], sp["g"], sp["x"], sp["y"], sp["th"], tag=f"{p['tag']}_{sname}")))
            for sname, sp in wx.gl.splice_horizontal_column(p)[:2]:
                cands.append(("HCOL", wx.make_brick(sp["w"], sp["h"], sp["g"], sp["x"], sp["y"], sp["th"], tag=f"{p['tag']}_{sname}")))

        for kind, raw_b in cands:
            seed_base += 1
            if raw_b["w"] > raw_b["h"]:
                raw_b = wx.bl.transform_brick(raw_b, "T")
                raw_b["tag"] = raw_b.get("tag", "") + "_T"
            sk = _canonical_shape(raw_b)
            prev = ext_best.get(sk, 1e9)
            if prev < 1e8 and min(raw_b["fx"], raw_b["fy"]) > prev + 0.015:
                continue
            if raw_b["f"] > 0.88:
                continue
            cb = wx.compress_brick_pslp(raw_b, n_trials=2, seed=seed_base, do_shake=True)
            if cb["f"] < min(prev - 1.5e-4, 0.795):
                wx.save_brick_json(cb, cb["tag"])
                saved_synth.append(cb)
                ext_best[sk] = cb["f"]
                print(
                    f"  [{kind}+] {cb['tag']}: {cb['w']}x{cb['h']}+g{cb['g']} f={cb['f']:.6f} (prev={prev:.6f})",
                    flush=True,
                )

    # Scan m=21..31, r=+6..+18 with the unified catalog
    all_bricks = wx.load_all_pooled_bricks(top_per_shape=2, include_external=True)
    scan_ns = [m * m + r for m in range(21, 32) for r in range(6, 19)]
    fr = wx.fast_load_true_frontier(scan_ns, certify=False)

    hits: list[tuple[float, int, int, int, float, str, str, int, int]] = []
    for n in scan_ns:
        m = int(math.isqrt(n))
        r = n - m * m
        f_inc = fr[n]["L"] - m
        pairs = wx.pl.pick_pairs_for_n(n, all_bricks, f_inc, max_excess=0.002, top_k=3)
        if not pairs:
            continue
        best_p = min(pairs, key=lambda t: max(t[1]["f"], t[2]["f"]))
        _, b1, b2, dm = best_p
        f_hi = max(b1["f"], b2["f"])
        kx = b1["w"] + b2["w"] - m
        ky = b1["h"] + b2["h"] - m
        hits.append((f_hi - f_inc, n, m, r, f_inc, b1["tag"], b2["tag"], kx, ky))

    hits.sort()
    print(f"\n=== SCAN m=21..31, r=+6..+18 ({len(hits)} targets with pairs, {len(all_bricks)} bricks) ===", flush=True)
    for est_dl, n, m, r, f_inc, t1, t2, kx, ky in hits[:30]:
        print(
            f"  n={n:3d} ({m},+{r:2d}): est_dL={est_dl:+.5f} f_inc={f_inc:.5f} kx={kx:+d} ky={ky:+d} | {t1} + {t2}",
            flush=True,
        )

    summary = {
        "elapsed_s": round(time.time() - t0, 1),
        "n_donors": len(packs),
        "n_raw_extracted": len(raw_extracted),
        "n_saved_extracted": len(saved_extracted),
        "n_saved_synth": len(saved_synth),
        "n_total_bricks": len(all_bricks),
        "saved_bricks": [
            {"tag": b["tag"], "w": b["w"], "h": b["h"], "g": b["g"], "f": round(b["f"], 6)}
            for b in (saved_extracted + saved_synth)
        ],
        "top_hits": [
            {
                "n": n,
                "m": m,
                "r": r,
                "est_dL": round(est_dl, 6),
                "f_inc": round(f_inc, 6),
                "kx": kx,
                "ky": ky,
                "pair": f"{t1}+{t2}",
            }
            for est_dl, n, m, r, f_inc, t1, t2, kx, ky in hits[:40]
        ],
    }
    (wx.DOCS_DIR / "harvest_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nHarvest & synth complete in {summary['elapsed_s']}s", flush=True)


if __name__ == "__main__":
    main()
