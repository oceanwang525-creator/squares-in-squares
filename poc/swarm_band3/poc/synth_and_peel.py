#!/usr/bin/env python3
"""Scan F3B (need_g = r) targets and synthesize peeled / vstrip bricks for H94 (band3)."""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import band3_lib as b3  # noqa: E402

# Include cleanpeel2 and local gstep data if available in current repo
for extra_dir in (
    b3.REPO_ROOT / "poc" / "swarm_cleanpeel2" / "data",
    b3.REPO_ROOT / "poc" / "swarm_gstep" / "data",
):
    if extra_dir.exists() and extra_dir not in b3.EXTERNAL_BRICK_DIRS:
        b3.EXTERNAL_BRICK_DIRS.append(extra_dir)


def scan_targets(bricks: list[dict], frontier: dict, targets: list[int]) -> tuple[list, list]:
    hits_z, hits_c = [], []
    for n in targets:
        inc_dec = frontier[n]["L_dec"]
        inc_f = float(inc_dec)
        m = int(math.floor(inc_f))
        r = n - m * m
        f_inc = inc_f - m
        need_g = r  # build_pinwheel creates m^2 + g1 + g2 squares when ov <= 1 with disjoint bricks
        bz, bc = None, None
        for i, b1 in enumerate(bricks):
            for j, b2 in enumerate(bricks):
                if i > j or b1["g"] + b2["g"] != need_g:
                    continue
                w1, h1, w2, h2 = b1["w"], b1["h"], b2["w"], b2["h"]
                if w1 >= m or h1 >= m or w2 >= m or h2 >= m:
                    continue
                if w1 + w2 <= m and h1 + h2 <= m:
                    fm = max(b1["f"], b2["f"])
                    if fm < f_inc - 5e-4:
                        if w1 + w2 + h1 + h2 <= 2 * m - 1:
                            if bz is None or fm < bz[0]:
                                bz = (fm, b1, b2)
                        else:
                            if bc is None or fm < bc[0]:
                                bc = (fm, b1, b2)
        if bz is not None:
            hits_z.append((bz[0] - f_inc, n, m, r, f_inc, *bz))
        if bc is not None:
            hits_c.append((bc[0] - f_inc, n, m, r, f_inc, *bc))
    hits_z.sort()
    hits_c.sort()
    return hits_z, hits_c


def print_scan(title: str, hits_z: list, hits_c: list, limit: int = 25) -> None:
    print(f"\n=== {title} ===", flush=True)
    print(f"ZERO-OV (ct=0) hits: {len(hits_z)}")
    for dl, n, m, r, fi, fm, b1, b2 in hits_z[:limit]:
        s = f"{b1['w']}x{b1['h']}(g{b1['g']})+{b2['w']}x{b2['h']}(g{b2['g']})"
        print(f"  n={n} (m={m},r=+{r}): est_dL={dl:+.5f} inc={fi:.5f} fm={fm:.5f} | {s} [{b1['tag']}+{b2['tag']}]")
    print(f"CORNER-TOUCH (ct=1) hits: {len(hits_c)}")
    for dl, n, m, r, fi, fm, b1, b2 in hits_c[:limit]:
        s = f"{b1['w']}x{b1['h']}(g{b1['g']})+{b2['w']}x{b2['h']}(g{b2['g']})"
        print(f"  n={n} (m={m},r=+{r}): est_dL={dl:+.5f} inc={fi:.5f} fm={fm:.5f} | {s} [{b1['tag']}+{b2['tag']}]")


def synthesize_peels_and_splices(bricks: list[dict]) -> list[dict]:
    best_by_key: dict[tuple[int, int, int], float] = {}
    for b in bricks:
        k = (b["w"], b["h"], b["g"])
        best_by_key[k] = min(best_by_key.get(k, 1e9), b["f"])

    # Prioritize thin g in (7, 8, 9) bricks used by top F3B r=+14..+20 targets
    parents = [
        b
        for b in bricks
        if b["g"] in (7, 8, 9) and b["w"] <= b["h"] and (b["w"] + b["h"] <= 33) and b["f"] <= 0.805
    ]
    parents.sort(key=lambda b: (b["f"], b["w"] + b["h"], -b["g"]))
    # Keep top 1 per (w, h, g)
    seen_p: set[tuple[int, int, int]] = set()
    uniq_parents: list[dict] = []
    for p in parents:
        k = (p["w"], p["h"], p["g"])
        if k not in seen_p:
            seen_p.add(k)
            uniq_parents.append(p)

    print(f"Peeling/splicing {len(uniq_parents[:24])} unique parent bricks (w<=h, g in 7..9, w+h<=33)...", flush=True)
    new_bricks: list[dict] = []
    for idx, p in enumerate(uniq_parents[:24]):
        for side in ("left", "right", "bottom", "top"):
            peeled = b3.gl.peel_outer_unit_layer(p, side, tag=f"{p['tag']}_p{side[0]}")
            if peeled is None:
                continue
            nw, nh, ng = peeled["w"], peeled["h"], peeled["g"]
            key = (nw, nh, ng)
            tkey = (nh, nw, ng)
            cur_best = min(best_by_key.get(key, 1e9), best_by_key.get(tkey, 1e9))
            # Relax peeled brick
            f_start = max(peeled["f"], 0.72)
            ft_list = tuple(sorted({round(min(0.83, f_start + 0.03), 3), 0.79, 0.76, 0.735, 0.715, 0.695}, reverse=True))
            comp = b3.gl.relax_and_compress_brick(peeled, f_targets=ft_list, n_trials=4, seed=1000 + idx * 10)
            if comp is None:
                continue
            if comp["f"] < min(cur_best - 3e-4, 0.815):
                comp = b3.make_brick(comp["w"], comp["h"], comp["g"], comp["x"], comp["y"], comp["th"], tag=peeled["tag"])
                b3.save_brick_json(comp, comp["tag"])
                best_by_key[key] = comp["f"]
                best_by_key[tkey] = comp["f"]
                new_bricks.append(comp)
                print(
                    f"  [PEEL+] {comp['tag']}: {nw}x{nh} g={ng} f={comp['f']:.6f} (prev={cur_best:.6f})",
                    flush=True,
                )

    # Also try vstrip / hcol splice on top vstrip / peel_left parents
    splice_parents = [
        b
        for b in bricks
        if ("vstrip" in b["tag"] or "peel" in b["tag"] or "mw" in b["tag"] or b["g"] >= 7)
        and b["w"] <= 12
        and b["h"] <= 23
        and b["f"] <= 0.75
    ]
    seen_sp: set[tuple[int, int, int]] = set()
    for idx, p in enumerate(splice_parents[:10]):
        k = (p["w"], p["h"], p["g"])
        if k in seen_sp:
            continue
        seen_sp.add(k)
        for fn_name, fn in (("vstrip", b3.gl.splice_vertical_strip), ("hcol", b3.gl.splice_horizontal_column)):
            sp_list = fn(p)
            for sname, sp in sp_list[:2]:
                key = (sp["w"], sp["h"], sp["g"])
                tkey = (sp["h"], sp["w"], sp["g"])
                cur_best = min(best_by_key.get(key, 1e9), best_by_key.get(tkey, 1e9))
                comp = b3.gl.relax_and_compress_brick(
                    sp, f_targets=(0.80, 0.77, 0.745, 0.725, 0.705, 0.69), n_trials=6, seed=2000 + idx * 10
                )
                if comp is not None and comp["f"] < min(cur_best - 3e-4, 0.815):
                    sp_tag = f"{p['tag']}_{sname}"
                    comp = b3.make_brick(comp["w"], comp["h"], comp["g"], comp["x"], comp["y"], comp["th"], tag=sp_tag)
                    b3.save_brick_json(comp, comp["tag"])
                    best_by_key[key] = comp["f"]
                    best_by_key[tkey] = comp["f"]
                    new_bricks.append(comp)
                    print(
                        f"  [SPLICE+] {comp['tag']}: {comp['w']}x{comp['h']} g={comp['g']} f={comp['f']:.6f} (prev={cur_best:.6f})",
                        flush=True,
                    )
    return new_bricks


def main() -> None:
    t0 = time.time()
    scan_only = "--scan-only" in sys.argv
    if "--harvest" in sys.argv:
        b3.harvest_incumbent_bricks(min_g=6, max_f=0.83, save=True)
    bricks = b3.load_all_bricks(include_external=True)
    targets = [m * m + r for m in range(20, 32) for r in range(13, 23)]
    frontier = b3.gl.fast_load_true_frontier(targets, certify=False)

    hz, hc = scan_targets(bricks, frontier, targets)
    print_scan(f"Initial catalog ({len(bricks)} bricks)", hz, hc)

    if not scan_only:
        added = synthesize_peels_and_splices(bricks)
        bricks2 = b3.load_all_bricks(include_external=True)
        hz2, hc2 = scan_targets(bricks2, frontier, targets)
        print_scan(f"Post-peel/splice catalog ({len(bricks2)} bricks, +{len(added)} new)", hz2, hc2)
        b3.DOCS_DIR.mkdir(parents=True, exist_ok=True)
        summary = {
            "n_bricks": len(bricks2),
            "n_new": len(added),
            "new_tags": [(b["tag"], b["w"], b["h"], b["g"], round(b["f"], 6)) for b in added],
            "zero_ov_hits": [
                {
                    "n": n,
                    "m": m,
                    "r": r,
                    "est_dL": round(dl, 6),
                    "inc": round(fi, 6),
                    "fm": round(fm, 6),
                    "pair": f"{b1['tag']}+{b2['tag']}",
                }
                for dl, n, m, r, fi, fm, b1, b2 in hz2
            ],
            "ct1_hits": [
                {
                    "n": n,
                    "m": m,
                    "r": r,
                    "est_dL": round(dl, 6),
                    "inc": round(fi, 6),
                    "fm": round(fm, 6),
                    "pair": f"{b1['tag']}+{b2['tag']}",
                }
                for dl, n, m, r, fi, fm, b1, b2 in hc2
            ],
        }
        (b3.DOCS_DIR / "synth_scan.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
