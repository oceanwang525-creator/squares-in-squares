#!/usr/bin/env python3
"""H112 (thinband8): Elongated h in {7, 8, 9} Sub-Brick Synthesis & Asymmetric Pinwheel Sweep."""
from __future__ import annotations

import csv
import fcntl
import io
import json
import math
import os
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
THIN_DIR = HERE.parent
DATA_DIR = THIN_DIR / "data"
DOCS_DIR = THIN_DIR / "docs"
FINAL_CSV = THIN_DIR / "final.csv"
REPO_ROOT = THIN_DIR.parents[1]

B8_POC = REPO_ROOT / "poc" / "swarm_b8xplant" / "poc"
AUTO_UNION_POC = REPO_ROOT / "poc" / "auto_union" / "poc"
GSTEP_POC = REPO_ROOT / "poc" / "swarm_gstep" / "poc"
LPSLACK_POC = REPO_ROOT / "poc" / "swarm_lpslack" / "poc"

for _p in (B8_POC, AUTO_UNION_POC, GSTEP_POC, LPSLACK_POC):
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import b8_lib as bl  # noqa: E402
import brick_rect_lib as bslp  # noqa: E402
import gstep_lib as gl  # noqa: E402
import lpslack  # noqa: E402
import union as auto_union  # noqa: E402
from cpp_wrap import cpp_eval_sat_pen_bbox  # noqa: E402

MERGED_CSV = bl.MERGED_CSV


def make_brick(w: int, h: int, g: int, x: np.ndarray, y: np.ndarray, th: np.ndarray, tag: str = "") -> dict:
    bx, by, W, H = bl.rect_bbox(x, y, th)
    fx, fy = float(W - w), float(H - h)
    return {
        "w": int(w),
        "h": int(h),
        "g": int(g),
        "x": bx.copy(),
        "y": by.copy(),
        "th": np.asarray(th, dtype=np.float64).copy(),
        "fx": fx,
        "fy": fy,
        "f": max(fx, fy),
        "tag": tag,
    }


def canon_brick_long_y(b: dict) -> dict:
    """Return brick oriented so w <= h (short axis is x, long axis is y)."""
    if b["w"] <= b["h"]:
        return b
    bt = bl.transform_brick(b, "T")
    bt["tag"] = b["tag"] + "_T" if not b["tag"].endswith("_T") else b["tag"][:-2]
    return bt


def save_brick_json(b: dict, name: str, subdir: str = "bricks") -> Path:
    out_dir = DATA_DIR / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_p = out_dir / f"brick_{name}.json"
    payload = {
        "name": name,
        "w": int(b["w"]),
        "h": int(b["h"]),
        "g": int(b["g"]),
        "fx": float(b["fx"]),
        "fy": float(b["fy"]),
        "f": float(b["f"]),
        "x": b["x"].tolist(),
        "y": b["y"].tolist(),
        "th": b["th"].tolist(),
    }
    out_p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_p


def load_brick_json(path: Path, tag: str = "") -> dict:
    d = json.loads(path.read_text(encoding="utf-8"))
    return make_brick(
        int(d["w"]),
        int(d["h"]),
        int(d["g"]),
        np.array(d["x"], dtype=np.float64),
        np.array(d["y"], dtype=np.float64),
        np.array(d["th"], dtype=np.float64),
        tag=tag or d.get("name", path.stem.replace("brick_", "")),
    )


def _load_brick_from_git(branch: str, rel_path: str, tag: str = "") -> dict | None:
    try:
        txt = subprocess.check_output(["git", "show", f"{branch}:{rel_path}"], text=True, stderr=subprocess.DEVNULL)
        d = json.loads(txt)
        if not all(k in d for k in ("w", "h", "g", "x", "y", "th")):
            return None
        return make_brick(
            int(d["w"]),
            int(d["h"]),
            int(d["g"]),
            np.array(d["x"], dtype=np.float64),
            np.array(d["y"], dtype=np.float64),
            np.array(d["th"], dtype=np.float64),
            tag=tag or d.get("name", Path(rel_path).stem.replace("brick_", "")),
        )
    except Exception:
        return None


def _extract_bricks_from_pack_dict(d: dict, label: str, h_set: set[int] = (7, 8, 9, 10)) -> list[dict]:
    rs = sorted(d["rows"], key=lambda r: int(r["square_id"]))
    ang = np.array([float(r["angle"]) for r in rs], dtype=np.float64)
    if float(np.max(np.abs(ang))) > 6.3:
        ang = np.radians(ang)
    x, y, th, L = bl.normalize(
        np.array([float(r["x"]) for r in rs], dtype=np.float64),
        np.array([float(r["y"]) for r in rs], dtype=np.float64),
        ang,
    )
    m = int(math.floor(L))
    f = L - m
    deg = np.degrees(np.mod(th, 0.5 * math.pi))
    tilt = np.minimum(deg, 90.0 - deg) > 2.0
    hx = bl.half_extent(th)
    out: list[dict] = []
    for corner in ("TL", "BR", "BL", "TR"):
        for short in sorted(h_set):
            for long in range(15, min(m, 25)):
                for w, h, ori in ((short, long, "V"), (long, short, "H")):
                    if corner == "TL":
                        mask = (x + hx <= w + f + 0.02) & (y - hx >= m - h - 0.02)
                    elif corner == "BR":
                        mask = (x - hx >= m - w - 0.02) & (y + hx <= h + f + 0.02)
                    elif corner == "BL":
                        mask = (x + hx <= w + f + 0.02) & (y + hx <= h + f + 0.02)
                    else:
                        mask = (x - hx >= m - w - 0.02) & (y - hx >= m - h - 0.02)
                    g = int(mask.sum()) - w * h
                    if not (4 <= g <= 8) or int((mask & tilt).sum()) < 25:
                        continue
                    b = make_brick(w, h, g, x[mask], y[mask], th[mask], tag=f"B{g}_{w}x{h}_{label}_{corner}{ori}")
                    if b["f"] <= f + 0.003:
                        pen, _ = cpp_eval_sat_pen_bbox(b["x"], b["y"], b["th"])
                        if pen <= 1e-7:
                            out.append(b)
    return out


def harvest_all_thin_bricks(save_harvest: bool = True) -> list[dict]:
    """Harvest all h in {7, 8, 9} (plus h=10 donors) bricks from local dirs, PR branches, and wins."""
    raw: list[dict] = []

    local_dirs = [
        REPO_ROOT / "poc" / "swarm_cleanpeel2" / "data",
        REPO_ROOT / "poc" / "swarm_gstep" / "data",
        REPO_ROOT / "poc" / "swarm_b7xplant" / "data",
        REPO_ROOT / "poc" / "swarm_b8xplant" / "data",
        REPO_ROOT / "poc" / "swarm_b5xplant" / "data",
    ]
    for folder in local_dirs:
        if not folder.exists():
            continue
        for p in sorted(folder.glob("*.json")):
            if "win_" in p.name or "cache" in p.name:
                continue
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                if all(k in d for k in ("w", "h", "g", "x", "y", "th")):
                    b = load_brick_json(p)
                    if min(b["w"], b["h"]) in (7, 8, 9, 10):
                        raw.append(b)
            except Exception:
                continue

    for cache_p in (
        REPO_ROOT / "poc" / "swarm_b5xplant" / "data" / "compressed_b5_b4_cache.json",
        REPO_ROOT / "poc" / "swarm_thinb4" / "data" / "compressed_bricks_cache.json",
    ):
        if cache_p.exists():
            raw_c = json.loads(cache_p.read_text(encoding="utf-8"))
            items = raw_c.values() if isinstance(raw_c, dict) else raw_c
            for item in items:
                if min(int(item["w"]), int(item["h"])) in (7, 8, 9, 10):
                    raw.append(
                        make_brick(
                            int(item["w"]), int(item["h"]), int(item["g"]),
                            np.array(item["x"], dtype=np.float64),
                            np.array(item["y"], dtype=np.float64),
                            np.array(item["th"], dtype=np.float64),
                            tag=str(item["tag"]),
                        )
                    )

    # Also pull bricks from origin/swarm-gstep-pr2 and origin/swarm-stepdown-pr1
    for br, prefix in (
        ("origin/swarm-gstep-pr2", "poc/swarm_gstep/data"),
        ("origin/swarm-stepdown-pr1", "poc/swarm_stepdown/data"),
    ):
        try:
            files = subprocess.check_output(
                ["git", "ls-tree", "-r", "--name-only", br, prefix],
                text=True, stderr=subprocess.DEVNULL,
            ).splitlines()
        except Exception:
            files = []
        for gp in files:
            if not Path(gp).name.startswith("brick_") or not gp.endswith(".json"):
                continue
            b = _load_brick_from_git(br, gp)
            if b is not None and min(b["w"], b["h"]) in (7, 8, 9, 10):
                raw.append(b)

    # Harvest corner bricks from breakthrough win JSONs
    win_sources: list[tuple[str, dict]] = []
    for wp in (
        REPO_ROOT / "poc" / "swarm_band3" / "data" / "win_n975.json",
        REPO_ROOT / "poc" / "swarm_cleanpeel2" / "data" / "win_n974.json",
        REPO_ROOT / "poc" / "swarm_gstep" / "data" / "win_n974.json",
        REPO_ROOT / "poc" / "swarm_gstep" / "data" / "win_n975.json",
    ):
        if wp.exists():
            win_sources.append((wp.stem, json.loads(wp.read_text(encoding="utf-8"))))
    for br, gp, lbl in (
        ("origin/swarm-gstep-pr2", "poc/swarm_gstep/data/win_n912.json", "gstep912"),
        ("origin/swarm-gstep-pr2", "poc/swarm_gstep/data/win_n853.json", "gstep853"),
        ("origin/swarm-stepdown-pr1", "poc/swarm_stepdown/data/win_n912.json", "stepdown912"),
    ):
        try:
            txt = subprocess.check_output(["git", "show", f"{br}:{gp}"], text=True, stderr=subprocess.DEVNULL)
            win_sources.append((lbl, json.loads(txt)))
        except Exception:
            continue

    for lbl, wd in win_sources:
        raw.extend(_extract_bricks_from_pack_dict(wd, lbl, h_set={7, 8, 9}))

    # Harvest thin bricks from merged.csv (n=539, 587, 634, 684, 739, 795, 796, 797, 975)
    df = pd.read_csv(MERGED_CSV, usecols=["n", "square_id", "x", "y", "angle"])
    for n_val, grp in df[df["n"].isin([539, 587, 634, 684, 739, 795, 796, 797, 975])].groupby("n"):
        n = int(n_val)
        gs = grp.sort_values("square_id")
        rows = [
            {"square_id": int(r.square_id), "x": float(r.x), "y": float(r.y), "angle": float(r.angle)}
            for r in gs.itertuples(index=False)
        ]
        raw.extend(_extract_bricks_from_pack_dict({"rows": rows}, f"m{n}", h_set={7, 8, 9}))

    # Canonicalize to w <= h, deduplicate keeping top-2 per (w, h, g) by f
    by_shape: dict[tuple[int, int, int], list[dict]] = {}
    seen_sigs: set[tuple] = set()
    for b in raw:
        cb = canon_brick_long_y(b)
        pen, _ = cpp_eval_sat_pen_bbox(cb["x"], cb["y"], cb["th"])
        if pen > 1e-7:
            continue
        sig = (cb["w"], cb["h"], cb["g"], round(cb["f"], 6), round(float(cb["x"].mean()), 4))
        if sig in seen_sigs:
            continue
        seen_sigs.add(sig)
        by_shape.setdefault((cb["w"], cb["h"], cb["g"]), []).append(cb)

    kept: list[dict] = []
    for key, lst in sorted(by_shape.items()):
        lst.sort(key=lambda c: c["f"])
        for rank, b in enumerate(lst[:2]):
            if save_harvest:
                save_brick_json(b, f"H_{b['w']}x{b['h']}_g{b['g']}_r{rank}_{b['tag'][:24]}", subdir="harvest")
            kept.append(b)
    return kept


def peel_outer_unit_layer(b: dict, side: str, tag: str = "") -> dict | None:
    return gl.peel_outer_unit_layer(b, side, tag=tag)


def remove_vertical_strip(b: dict, strip_idx: int | None = None) -> list[tuple[str, dict]]:
    """Remove 1 interior tilted strip + matching AA squares: (w, h, g) -> (w, h-1, g-1) for w <= h."""
    w, h, g = b["w"], b["h"], b["g"]
    if g <= 3 or h <= 14:
        return []
    if w > h:
        bt = bl.transform_brick(b, "T")
        res = []
        for tag, sb in remove_vertical_strip(bt, strip_idx=strip_idx):
            sbt = bl.transform_brick(sb, "T")
            sbt["tag"] = tag + "_T"
            res.append((sbt["tag"], sbt))
        return res

    x, y, th = b["x"], b["y"], b["th"]
    deg = np.degrees(np.mod(th, 0.5 * math.pi))
    tilt = np.minimum(deg, 90.0 - deg) > 2.0
    strips, _ = gl._cluster_strips(x, y, th, tilt)
    if len(strips) < 8:
        return []
    aa_idx = np.flatnonzero(~tilt)
    aa_cols = gl._cluster_aa_columns(x, aa_idx)

    max_len = max(len(s) for s in strips)
    cand_indices = (
        [strip_idx]
        if strip_idx is not None
        else [k for k in range(3, len(strips) - 3) if len(strips[k]) >= max_len - 1]
    )

    out: list[tuple[str, dict]] = []
    for k in cand_indices:
        sk = strips[k]
        sk1 = strips[k + 1]
        dx_list, dy_list = [], []
        for i in sk:
            best_j = min(sk1, key=lambda j: abs(x[j] - x[i]))
            if abs(x[best_j] - x[i]) < 0.45:
                dx_list.append(float(x[best_j] - x[i]))
                dy_list.append(float(y[best_j] - y[i]))
        if not dy_list:
            continue
        dx_s = float(np.median(dx_list))
        dy_s = float(np.median(dy_list))

        nx_l, ny_l, nt_l = [], [], []
        for si, st in enumerate(strips):
            if si == k:
                continue
            for i in st:
                if si < k:
                    nx_l.append(float(x[i]))
                    ny_l.append(float(y[i]))
                    nt_l.append(float(th[i]))
                else:
                    nx_l.append(float(x[i] - dx_s))
                    ny_l.append(float(y[i] - dy_s))
                    nt_l.append(float(th[i]))

        y_left = float(y[sk[0]])
        y_right = float(y[sk[-1]])
        aa_new_x, aa_new_y, aa_new_t = [], [], []
        for col in aa_cols:
            col_s = sorted(col, key=lambda i: y[i])
            cx_mean = float(np.mean([x[i] for i in col_s]))
            y_cut = y_left if cx_mean < 0.5 * w else y_right
            below = [i for i in col_s if y[i] <= y_cut + 0.15]
            above = [i for i in col_s if y[i] > y_cut + 0.15]
            # Drop one AA square near y_cut if a continuous column crosses y_cut
            if below and above and (y[above[0]] - y[below[-1]]) < 1.35:
                above = above[1:]
            for i in below:
                aa_new_x.append(float(x[i]))
                aa_new_y.append(float(y[i]))
                aa_new_t.append(float(th[i]))
            for i in above:
                aa_new_x.append(float(x[i]))
                aa_new_y.append(float(y[i] - 1.0))
                aa_new_t.append(float(th[i]))

        tot_n = len(nx_l) + len(aa_new_x)
        target_n = w * (h - 1) + (g - 1)
        if tot_n != target_n:
            continue
        ax = np.array(nx_l + aa_new_x, dtype=np.float64)
        ay = np.array(ny_l + aa_new_y, dtype=np.float64)
        at = np.array(nt_l + aa_new_t, dtype=np.float64)
        cand = make_brick(w, h - 1, g - 1, ax, ay, at, tag=f"{b['tag']}_rmvs_k{k}")
        out.append((cand["tag"], cand))
    return out


def compress_brick_slp(b: dict, n_trials: int = 12, seed: int = 42) -> dict | None:
    """Relax and compress a (possibly slightly overlapping) brick into (w+f)x(h+f) via L-BFGS-B + SLP."""
    w, h, g = b["w"], b["h"], b["g"]
    x0, y0, t0 = b["x"], b["y"], b["th"]
    pen0, _ = cpp_eval_sat_pen_bbox(x0, y0, t0)
    best: dict | None = None
    if pen0 <= 1e-8:
        sx, sy, st, _, ok = bslp.joint_xytheta_slp_rect(x0, y0, t0, w, h, rounds=14, tr_th0=0.022, tr_xy0=0.09, tol=1e-11)
        if ok and cpp_eval_sat_pen_bbox(sx, sy, st)[0] <= 1e-8:
            best = make_brick(w, h, g, sx, sy, st, tag=b["tag"])
        else:
            best = make_brick(w, h, g, x0, y0, t0, tag=b["tag"])

    f0 = best["f"] if best is not None else max(0.75, b["f"])
    if f0 > 0.72:
        ft_sched = (min(0.78, f0 + 0.01), 0.735, 0.705, 0.680, 0.655)
    else:
        ft_sched = (f0 + 0.002, max(0.640, f0 - 0.015), max(0.635, f0 - 0.035))

    rng = np.random.default_rng(seed)
    deg = np.degrees(np.mod(t0, 0.5 * math.pi))
    tilt = np.minimum(deg, 90.0 - deg) > 2.0
    W0, H0 = w + b["fx"], h + b["fy"]

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
                amp = 0.05 if trial % 3 != 0 else 0.025
                cx[mask] += rng.uniform(-amp, amp, nm)
                cy[mask] += rng.uniform(-amp, amp, nm)
                ct[mask] += rng.uniform(-1.3 * amp, 1.3 * amp, nm)
        for ft in ft_sched:
            cx, cy, ct, pen = bslp.relax_rect_lbfgs(cx, cy, ct, w + ft, h + ft, maxiter=360)
            if pen <= 1e-7:
                sx, sy, st, _, ok = bslp.joint_xytheta_slp_rect(
                    cx, cy, ct, w, h, rounds=14, tr_th0=0.022, tr_xy0=0.09, tol=1e-11
                )
                if ok and cpp_eval_sat_pen_bbox(sx, sy, st)[0] <= 1e-8:
                    cb = make_brick(w, h, g, sx, sy, st, tag=b["tag"])
                    if best is None or cb["f"] < best["f"] - 1e-10:
                        best = cb
    return best


def brick_mechwalk(b: dict, name: str, max_rounds: int = 2, save: bool = True) -> dict:
    """Standalone Contact-Jacobian SVD walk + d(i+-j) diagonal mode pairs on a (w, h, g) brick."""
    w, h, g = b["w"], b["h"], b["g"]
    bx, by, W0, H0 = bl.rect_bbox(b["x"], b["y"], b["th"])
    bt = b["th"].copy()
    best_f = max(W0 - w, H0 - h)
    best_state = (bx.copy(), by.copy(), bt.copy())
    free_all = np.ones(len(bx), dtype=bool)
    inv_s2 = 1.0 / math.sqrt(2.0)

    for rnd in range(max_rounds):
        improved = False
        for ctol in (1e-5, 4e-4, 2e-3):
            jac, _, _ = bl.localized_contact_jacobian(
                best_state[0], best_state[1], best_state[2], free_all, contact_tol=ctol
            )
            if jac.shape[0] == 0:
                continue
            _, modes = bl.localized_soft_modes(jac, k=10)
            for mi in range(min(8, len(modes))):
                new_state, new_f, hit_amp = gl._try_brick_direction(
                    best_state, modes[mi], w, h, best_f, (0.005, 0.015, 0.045)
                )
                if hit_amp is not None:
                    best_f = new_f
                    best_state = new_state
                    improved = True
            kcap = min(5, len(modes))
            for i in range(kcap):
                for j in range(i + 1, kcap):
                    for d_sgn in (1.0, -1.0):
                        vd = inv_s2 * (modes[i] + d_sgn * modes[j])
                        new_state, new_f, hit_amp = gl._try_brick_direction(
                            best_state, vd, w, h, best_f, (0.010, 0.035)
                        )
                        if hit_amp is not None:
                            best_f = new_f
                            best_state = new_state
                            improved = True
        if not improved:
            break

    out = make_brick(w, h, g, best_state[0], best_state[1], best_state[2], tag=name)
    if save:
        save_brick_json(out, name, subdir="synth")
    return out


def fast_load_true_frontier(target_ns: list[int] | set[int], certify: bool = True) -> dict[int, dict]:
    """Load TRUE frontier (merged.csv + all poc/*/final.csv on origin/main, excluding self)."""
    need = {int(n) for n in target_ns}
    best: dict[int, dict] = {}
    df = pd.read_csv(MERGED_CSV, usecols=["n", "square_id", "x", "y", "angle"])
    for n_val, grp in df[df["n"].isin(need)].groupby("n"):
        n = int(n_val)
        grp_s = grp.sort_values("square_id")
        x, y, th, L_f = bl.normalize(
            grp_s["x"].to_numpy(float, copy=True),
            grp_s["y"].to_numpy(float, copy=True),
            grp_s["angle"].to_numpy(float, copy=True),
        )
        if certify:
            rows = [
                {"square_id": int(r.square_id), "x": repr(float(r.x)), "y": repr(float(r.y)), "angle": repr(float(r.angle))}
                for r in grp_s.itertuples(index=False)
            ]
            L_dec = Decimal(auto_union.certify(n, rows))
        else:
            L_dec = Decimal(f"{L_f:.17g}")
        best[n] = {"x": x, "y": y, "th": th, "L": L_f, "L_dec": L_dec, "source": "merged.csv"}

    need_prefixes = tuple(f"{n:03d}_" for n in sorted(need))
    for p in sorted((REPO_ROOT / "poc").glob("*/final.csv")):
        if p == FINAL_CSV or not p.exists() or p.stat().st_size <= 20:
            continue
        txt = p.read_text(encoding="utf-8")
        lines = txt.splitlines()
        if len(lines) <= 1:
            continue
        sub = [lines[0]] + [ln for ln in lines[1:] if ln.startswith(need_prefixes)]
        if len(sub) <= 1:
            continue
        sub_txt = "\n".join(sub) + "\n"
        if not certify:
            groups_raw: dict[int, list[tuple[int, float, float, float]]] = {}
            for row in csv.DictReader(io.StringIO(sub_txt)):
                ni = int(row["id"].split("_")[0])
                sid = int(row["id"].split("_")[1])
                groups_raw.setdefault(ni, []).append(
                    (sid, float(row["x"].lstrip("s")), float(row["y"].lstrip("s")), math.radians(float(row["deg"].lstrip("s"))))
                )
            for ni, rlist in groups_raw.items():
                if ni in need and len(rlist) == ni:
                    rlist.sort()
                    nx, ny, nt, L_f = bl.normalize([r[1] for r in rlist], [r[2] for r in rlist], [r[3] for r in rlist])
                    if ni not in best or L_f < best[ni]["L"] - 1e-12:
                        best[ni] = {"x": nx, "y": ny, "th": nt, "L": L_f, "L_dec": Decimal(f"{L_f:.17g}"), "source": str(p.relative_to(REPO_ROOT))}
            continue
        try:
            lit = auto_union.parse_text(sub_txt)
            pd_t = auto_union.parse_text(pd.read_csv(io.StringIO(sub_txt)).to_csv(index=False))
        except Exception:
            continue
        for n in need:
            if n in lit and n in pd_t and len(lit[n]) == n:
                res = auto_union.check((n, lit[n], pd_t[n]))
                if res is None:
                    continue
                L_dec = max(Decimal(res[0]), Decimal(res[1]))
                if n not in best or L_dec < best[n]["L_dec"]:
                    rows_lit = sorted(lit[n], key=lambda r: int(r["square_id"]))
                    x, y, th, L_f = bl.normalize(
                        np.array([float(r["x"]) for r in rows_lit], dtype=np.float64),
                        np.array([float(r["y"]) for r in rows_lit], dtype=np.float64),
                        np.array([float(r["angle"]) for r in rows_lit], dtype=np.float64),
                    )
                    best[n] = {"x": x, "y": y, "th": th, "L": L_f, "L_dec": L_dec, "source": str(p.relative_to(REPO_ROOT))}
    return best


def build_pinwheel_kxky(
    m: int,
    b1: dict,
    b2: dict,
    sf: float = 0.0,
    f_override: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray] | None:
    """Build TL(b1) + BR(b2) pinwheel allowing kx, ky in {-1, 0, +1} (with kx+ky <= 1, not both +1)."""
    w1, h1, w2, h2 = b1["w"], b1["h"], b2["w"], b2["h"]
    if w1 >= m or h1 >= m or w2 >= m or h2 >= m:
        return None
    kx = w1 + w2 - m
    ky = h1 + h2 - m
    if kx > 1 or ky > 1 or (kx > 0 and ky > 0):
        return None
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
    if len(ax) != m * m + b1["g"] + b2["g"]:
        return None
    pad = 2.4 if (kx > 0 or ky > 0 or (kx == 0 and ky == 0)) else 1.8
    seam = (
        ((ax >= w1 - pad) & (ax <= w1 + f + pad))
        | ((ay >= m - h1 - pad) & (ay <= m - h1 + pad))
        | ((ax >= m - w2 - pad) & (ax <= m - w2 + pad))
        | ((ay >= h2 - pad) & (ay <= h2 + f + pad))
    ).astype(np.uint8)
    return ax, ay, at, L, seam


def run_lpslack_polish(n: int, cert: dict, inc_dec: Decimal, time_limit: float = 12.0) -> dict:
    """Apply poc/swarm_lpslack/poc/lpslack.py to a certified candidate, returning best cert."""
    if not cert.get("rows"):
        return cert
    base_rows = [
        {"square_id": str(row["square_id"]), "x": str(row["x"]), "y": str(row["y"]), "angle": str(row["angle"])}
        for row in cert["rows"]
    ]
    target_ref = min(inc_dec, Decimal(cert["L_lit"]), Decimal(cert["L_pandas"]))
    lp_res = lpslack.optimize_group(
        n, base_rows, str(target_ref),
        outer_iters=6, fixed_rounds=10,
        slp_modes=("dual_non90", "non90", "all"),
        tr_th_schedule=(5e-4, 1e-4, 3e-5),
        time_limit=time_limit,
    )
    if lp_res.get("status") in ("WIN", "TINY_WIN") and lp_res.get("rows"):
        lit = Decimal(lp_res["literal_L"])
        pan = Decimal(lp_res["pandas_L"])
        L_worst = max(lit, pan)
        if L_worst < max(Decimal(cert["L_lit"]), Decimal(cert["L_pandas"])):
            dL_true = L_worst - inc_dec
            return {
                "ok": True,
                "n": int(n),
                "L_lit": str(lit),
                "L_pandas": str(pan),
                "L_dec_str": str(L_worst),
                "L_float": float(L_worst),
                "dL": float(dL_true),
                "is_real_win": bool(dL_true <= Decimal("-1e-9")),
                "label": f"{cert['label']}+lpslack",
                "rows": [
                    {"square_id": int(r["square_id"]), "x": str(r["x"]), "y": str(r["y"]), "angle": str(r["angle"])}
                    for r in lp_res["rows"]
                ],
            }
    return cert


def solve_pinwheel_pair(
    n: int,
    m: int,
    b1_base: dict,
    b2_base: dict,
    inc_dec: Decimal,
    tag: str,
    base_seed: int,
    n_threads: int = 2,
    do_lpslack: bool = True,
) -> dict | None:
    """Solve K=2 pinwheel (kx, ky in {-1,0,+1}) via direct SLP (if feasible) + 1-corner C++ melt + SLP + lpslack."""
    best_cert: dict | None = None
    seeds, masks, scheds, labels = [], [], [], []
    w1, h1, w2, h2 = b1_base["w"], b1_base["h"], b2_base["w"], b2_base["h"]
    kx, ky = w1 + w2 - m, h1 + h2 - m
    is_zero_ov = (kx <= 0 and ky <= 0 and kx + ky < 0)

    for m1 in ("id", "flipx", "flipy", "rot180"):
        b1 = bl.transform_brick(b1_base, m1)
        for m2 in ("id", "flipx", "flipy", "rot180"):
            b2 = bl.transform_brick(b2_base, m2)
            f_hi = max(b1["fx"], b1["fy"], b2["fx"], b2["fy"])
            f_lo = min(b1["fx"], b1["fy"], b2["fx"], b2["fy"])
            f_avg = 0.5 * (max(b1["fx"], b1["fy"]) + max(b2["fx"], b2["fy"]))
            f_list = ((f_hi, "fhi"),) if abs(f_hi - f_lo) < 5e-4 else ((f_avg, "favg"), (f_hi, "fhi"))

            for f_use, ftag in f_list:
                built = build_pinwheel_kxky(m, b1, b2, sf=0.0, f_override=f_use)
                if built is None:
                    continue
                ax, ay, at, _, sm = built
                if is_zero_ov and ftag == "fhi" and (m1, m2) in (("id", "id"), ("flipx", "flipy"), ("rot180", "rot180")):
                    pen, _ = cpp_eval_sat_pen_bbox(ax, ay, at)
                    if pen <= 1e-10:
                        sx, sy, st, _, ok = bslp.joint_xytheta_slp_tight(ax, ay, at)
                        use_x, use_y, use_t = (sx, sy, st) if ok else (ax, ay, at)
                        cert = bl.polish_and_certify(n, use_x, use_y, use_t, inc_dec, slp_rounds=16)
                        if cert is not None:
                            cert["label"] = f"{tag}_{m1}_{m2}_direct"
                            if best_cert is None or cert["L_float"] < best_cert["L_float"]:
                                best_cert = cert

                # Queue corner/seam melts for promising orientations
                if ftag == "fhi" or (m1, m2) in (("id", "id"), ("flipx", "flipy"), ("id", "flipx"), ("rot180", "rot180")):
                    L_tgt = m + f_use
                    off = (0.018, 0.005, 0.0005) if (kx > 0 or ky > 0 or (kx == 0 and ky == 0)) else (0.010, 0.002, -3e-4)
                    seeds.append((ax, ay, at))
                    masks.append(sm)
                    scheds.append([L_tgt + off[0], L_tgt + off[1], L_tgt + off[2]])
                    labels.append(f"{tag}_{m1}_{m2}_{ftag}")

    if seeds:
        # Cap at 20 diverse seeds to keep runtime lean
        if len(seeds) > 20:
            seeds, masks, scheds, labels = seeds[:20], masks[:20], scheds[:20], labels[:20]
        relaxed = bl.su.cpp_batch_melt(
            seeds,
            scheds,
            sym_mode=1,
            a_start=0.960,
            n_melt_stages=3,
            seam_masks=masks,
            seam_iters=160,
            full_iters=220,
            n_ruin_iters=0,
            n_threads=n_threads,
            base_seed=base_seed,
        )
        pool = []
        for bi, row in enumerate(relaxed):
            nx, ny, nth, Lm = bl.normalize(row["x"], row["y"], row["th"])
            lbl = f"{labels[bi]}_melt"
            if row["best_L_feas"] is not None:
                pool.append((0.0, Lm, nx, ny, nth, lbl))
            elif row["pen"] < 0.02:
                pool.append((1.0 + float(row["pen"]), Lm, nx, ny, nth, lbl))
        pool.sort(key=lambda t: (t[0], t[1]))
        for _, _, cx, cy, cth, lbl in pool[:3]:
            sx, sy, st, _, ok = bslp.joint_xytheta_slp_tight(cx, cy, cth)
            use_x, use_y, use_t = (sx, sy, st) if ok else (cx, cy, cth)
            cert = bl.polish_and_certify(n, use_x, use_y, use_t, inc_dec, slp_rounds=16)
            if cert is not None:
                cert["label"] = lbl
                if best_cert is None or cert["L_float"] < best_cert["L_float"]:
                    best_cert = cert

    if do_lpslack and best_cert is not None and best_cert["L_float"] < float(inc_dec) + 0.005:
        best_cert = run_lpslack_polish(n, best_cert, inc_dec, time_limit=12.0)

    return best_cert


def save_win_locked(win_entry: dict) -> bool:
    return bl.save_win_locked(win_entry, data_dir=DATA_DIR, final_path=FINAL_CSV)
