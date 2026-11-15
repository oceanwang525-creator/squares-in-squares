#!/usr/bin/env python3
"""H121: Synthesize peeled and vstrip-spliced w x 8 (19x8..24x8, g=10..14) bricks via pslp."""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

for _env in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_env] = "1"

import numpy as np

POC_DIR = Path(__file__).resolve().parent
SWARM_DIR = POC_DIR.parent
REPO = SWARM_DIR.parents[1]

sys.path.insert(0, str(POC_DIR))
sys.path.insert(0, str(REPO / "poc" / "swarm_gstep" / "poc"))
sys.path.insert(0, str(REPO / "poc" / "swarm_b8xplant" / "poc"))
sys.path.insert(0, str(REPO / "poc" / "row_ratchet_980" / "poc"))
sys.path.insert(0, str(REPO / "poc" / "swarm_highd" / "poc"))

import interlock2_lib as il
import gstep_lib as gl
import b8_lib as bl
import pslp
from cpp_wrap import cpp_eval_sat_pen_bbox

OUT_PATH = il.DATA_DIR / "bricks_wx8.json"


def normalize_brick_boxes(b: dict) -> dict:
    x, y, th = b["x"], b["y"], b["th"]
    x0, x1, y0, y1 = pslp.box_of(x, y, th)
    return gl.make_brick(b["w"], b["h"], b["g"], x - x0, y - y0, th, tag=b.get("tag", ""))


def peel_vertical_strip(b: dict) -> list[tuple[str, dict]]:
    """Remove 1 interior tilted strip + 1 AA square per crossing column: (w, h, g) -> (w, h-1, g-1)."""
    w, h, g = b["w"], b["h"], b["g"]
    if w > h:
        bt = bl.transform_brick(b, "T")
        bt["tag"] = b.get("tag", "")
        res = []
        for tag, sb in peel_vertical_strip(bt):
            sbt = bl.transform_brick(sb, "T")
            sbt["tag"] = tag + "_T"
            res.append((sbt["tag"], sbt))
        return res

    x, y, th = b["x"], b["y"], b["th"]
    deg = np.degrees(np.mod(th, 0.5 * math.pi))
    tilt = np.minimum(deg, 90.0 - deg) > 2.0
    strips, _ = gl._cluster_strips(x, y, th, tilt)
    aa_idx = np.flatnonzero(~tilt)
    aa_cols = gl._cluster_aa_columns(x, aa_idx)
    if len(strips) < 7 or g < 2 or h < 8:
        return []
    max_len = max(len(s) for s in strips)
    out = []
    for k in range(3, len(strips) - 3):
        sk = strips[k]
        if len(sk) < max_len - 1:
            continue
        sk1 = strips[k + 1]
        dx_l, dy_l = [], []
        for i in sk:
            j = min(sk1, key=lambda j: abs(x[j] - x[i]))
            if abs(x[j] - x[i]) < 0.45:
                dx_l.append(float(x[j] - x[i]))
                dy_l.append(float(y[j] - y[i]))
        if not dy_l:
            continue
        dx_s = float(np.median(dx_l))
        dy_s = float(np.median(dy_l))

        nx_l, ny_l, nt_l = [], [], []
        for si, st in enumerate(strips):
            if si == k:
                continue
            for i in st:
                if si < k:
                    nx_l.append(float(x[i]))
                    ny_l.append(float(y[i]))
                else:
                    nx_l.append(float(x[i] - dx_s))
                    ny_l.append(float(y[i] - dy_s))
                nt_l.append(float(th[i]))

        y_left, y_right = float(y[sk[0]]), float(y[sk[-1]])
        aa_nx, aa_ny, aa_nt = [], [], []
        for col in aa_cols:
            col_s = sorted(col, key=lambda i: y[i])
            cx_m = float(np.mean([x[i] for i in col_s]))
            yc = y_left if cx_m < 0.5 * w else y_right
            # Drop the single AA square closest to yc if within 0.65
            best_drop = min(col_s, key=lambda i: abs(y[i] - yc))
            drop_y = float(y[best_drop]) if abs(y[best_drop] - yc) < 0.65 else None
            for i in col_s:
                if drop_y is not None and i == best_drop:
                    continue
                shift = -1.0 if (drop_y is not None and y[i] > drop_y) or (drop_y is None and y[i] > yc) else 0.0
                aa_nx.append(float(x[i]))
                aa_ny.append(float(y[i] + shift))
                aa_nt.append(float(th[i]))

        tot_n = len(nx_l) + len(aa_nx)
        if tot_n != w * (h - 1) + (g - 1):
            continue
        cand = gl.make_brick(
            w, h - 1, g - 1,
            np.array(nx_l + aa_nx, dtype=np.float64),
            np.array(ny_l + aa_ny, dtype=np.float64),
            np.array(nt_l + aa_nt, dtype=np.float64),
            tag=f"{b.get('tag', '')}_peel_k{k}",
        )
        out.append((cand["tag"], cand))
    return out


def splice_vertical_safe(b: dict) -> list[tuple[str, dict]]:
    w, h = b["w"], b["h"]
    if w > h:
        bt = bl.transform_brick(b, "T")
        bt["tag"] = b.get("tag", "")
        res = []
        for tag, sb in gl.splice_vertical_strip(bt):
            sbt = bl.transform_brick(sb, "T")
            sbt["tag"] = tag + "_T"
            res.append((sbt["tag"], sbt))
        return res
    return gl.splice_vertical_strip(b)


def optimize_brick(cand: dict, rng: np.random.Generator) -> dict | None:
    b = normalize_brick_boxes(cand)
    w, h, g = b["w"], b["h"], b["g"]
    X, Y, T = b["x"], b["y"], b["th"]
    # Expand slightly to make initial state feasible if overlap exists, then pslp.squeeze
    f0 = max(b["fx"], b["fy"])
    mp, mw = pslp.feas(X, Y, T, w + f0, h + f0)
    if min(mp, mw) < 0:
        scale = 1.03
        X = X * scale
        Y = Y * scale
        x0, x1, y0, y1 = pslp.box_of(X, Y, T)
        X -= x0
        Y -= y0
        f0 = max(x1 - x0 - w, y1 - y0 - h) + 1e-4

    f1, X1, Y1, T1, v1, _ = pslp.squeeze(X, Y, T, w, h, f0, trust0=0.03, trust_max=0.06, max_it=450)
    if v1 > 1e-11:
        return None
    f2, X2, Y2, T2 = pslp.shake_squeeze(X1, Y1, T1, w, h, f1, rounds=5, rng=rng)
    x0, x1, y0, y1 = pslp.box_of(X2, Y2, T2)
    X2 = X2 - x0
    Y2 = Y2 - y0
    pen, _ = cpp_eval_sat_pen_bbox(X2, Y2, T2)
    if pen > 1e-9:
        return None
    return gl.make_brick(w, h, g, X2, Y2, T2, tag=b["tag"])


def main() -> None:
    t0 = time.time()
    best_rows, _ = il.load_portfolio_best()
    merged_rows = il.load_merged_rows()
    by_whg = il.load_all_pockets(best_rows)

    seed_keys = [
        (8, 22, 12), (8, 23, 13),
        (20, 8, 11), (21, 8, 12), (21, 8, 11),
        (7, 18, 9), (7, 19, 10), (8, 21, 11),
    ]
    # Also scan all e3 pockets with min(w,h) in {7,8} and g >= 8
    for (w, h, g) in by_whg:
        if min(w, h) in (7, 8) and g >= 8 and max(w, h) >= 17 and (w, h, g) not in seed_keys:
            seed_keys.append((w, h, g))

    seeds = []
    seen_uids = set()
    for k in seed_keys:
        for it in by_whg.get(k, []):
            if it["uid"] in seen_uids or it["f"] >= 0.92:
                continue
            seen_uids.add(it["uid"])
            ext = il.get_item_squares_50d(it, merged_rows, best_rows)
            if ext is None:
                continue
            sq, W, H = ext
            x = np.array([float(s["x"]) for s in sq], dtype=np.float64)
            y = np.array([float(s["y"]) for s in sq], dtype=np.float64)
            th = np.array([float(s["angle"]) for s in sq], dtype=np.float64)
            seeds.append(gl.make_brick(k[0], k[1], k[2], x, y, th, tag=it["uid"]))

    print(f"Loaded {len(seeds)} thin seed bricks (min(w,h) in {{7,8}}, g>=8)", flush=True)
    rng = np.random.default_rng(121)

    best_by_whg: dict[tuple[int, int, int], dict] = {}
    for sb in seeds:
        k = (sb["w"], sb["h"], sb["g"])
        # Baseline squeezed version of seed too!
        opt_seed = optimize_brick(sb, rng)
        if opt_seed is not None:
            if k not in best_by_whg or opt_seed["f"] < best_by_whg[k]["f"]:
                best_by_whg[k] = opt_seed

        cands = []
        try:
            cands.extend(splice_vertical_safe(sb)[:4])
        except Exception:
            pass
        try:
            cands.extend(peel_vertical_strip(sb)[:4])
        except Exception:
            pass
        for tag, cb in cands:
            if max(cb["w"], cb["h"]) > 25 or min(cb["w"], cb["h"]) < 7:
                continue
            opt = optimize_brick(cb, rng)
            if opt is None:
                continue
            ck = (opt["w"], opt["h"], opt["g"])
            if ck not in best_by_whg or opt["f"] < best_by_whg[ck]["f"] - 1e-9:
                best_by_whg[ck] = opt
                print(f"  [SYNTH] {ck} f={opt['f']:.6f} <- {tag}", flush=True)

    # Second-order splice from synthesized (8, 23, 13) -> (8, 24, 14) etc.
    for ck in list(best_by_whg.keys()):
        sb = best_by_whg[ck]
        if max(sb["w"], sb["h"]) <= 23:
            try:
                for tag, cb in splice_vertical_safe(sb)[:3]:
                    if max(cb["w"], cb["h"]) > 25:
                        continue
                    opt = optimize_brick(cb, rng)
                    if opt is None:
                        continue
                    nk = (opt["w"], opt["h"], opt["g"])
                    if nk not in best_by_whg or opt["f"] < best_by_whg[nk]["f"] - 1e-9:
                        best_by_whg[nk] = opt
                        print(f"  [SYNTH2] {nk} f={opt['f']:.6f} <- {tag}", flush=True)
            except Exception:
                pass

    out_list = []
    for k in sorted(best_by_whg.keys()):
        b = best_by_whg[k]
        out_list.append({
            "name": f"wx8_{b['tag']}",
            "w": int(b["w"]),
            "h": int(b["h"]),
            "g": int(b["g"]),
            "fx": float(b["fx"]),
            "fy": float(b["fy"]),
            "f": float(b["f"]),
            "x": b["x"].tolist(),
            "y": b["y"].tolist(),
            "th": b["th"].tolist(),
        })
    OUT_PATH.write_text(json.dumps(out_list))
    print(f"Saved {len(out_list)} optimized w x 8 bricks to {OUT_PATH} in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
