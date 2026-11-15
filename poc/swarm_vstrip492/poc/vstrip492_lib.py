#!/usr/bin/env python3
"""H106 (vstrip492): vstrip splicing (g->g+1, g->g-1), 1-row/1-col peeling, and corner-melted pinwheels."""
from __future__ import annotations

import csv
import io
import json
import math
import os
import sys
from decimal import Decimal
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
VSTRIP_DIR = HERE.parent
DATA_DIR = VSTRIP_DIR / "data"
DOCS_DIR = VSTRIP_DIR / "docs"
FINAL_CSV = VSTRIP_DIR / "final.csv"
REPO_ROOT = VSTRIP_DIR.parents[1]

B8_POC = REPO_ROOT / "poc" / "swarm_b8xplant" / "poc"
CP2_POC = REPO_ROOT / "poc" / "swarm_cleanpeel2" / "poc"
GSTEP_POC = REPO_ROOT / "poc" / "swarm_gstep" / "poc"
LPSLACK_POC = REPO_ROOT / "poc" / "swarm_lpslack" / "poc"
HOMOCASC2_POC = REPO_ROOT / "poc" / "swarm_homocasc2" / "poc"
PSLP_POC = REPO_ROOT / "poc" / "row_ratchet_980" / "poc"
AUTO_UNION_POC = REPO_ROOT / "poc" / "auto_union" / "poc"

for _p in (B8_POC, CP2_POC, GSTEP_POC, LPSLACK_POC, HOMOCASC2_POC, PSLP_POC, AUTO_UNION_POC):
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import b8_lib as bl  # noqa: E402
import brick_rect_lib as bslp  # noqa: E402
import cleanpeel2_lib as cp2  # noqa: E402
import gstep_lib as gl  # noqa: E402
import homocasc2_lib as hc2  # noqa: E402
import lpslack  # noqa: E402
import pslp  # noqa: E402
import union as auto_union  # noqa: E402
from cpp_wrap import cpp_eval_sat_pen_bbox  # noqa: E402

EXTERNAL_BRICK_DIRS = [
    REPO_ROOT / "poc" / "swarm_gstep" / "data",
    REPO_ROOT / "poc" / "swarm_cleanpeel2" / "data",
    REPO_ROOT / "poc" / "swarm_peelstep" / "data",
    REPO_ROOT / "poc" / "swarm_pocketpeel" / "data" / "bricks",
    REPO_ROOT / "poc" / "swarm_watchxplant" / "data" / "bricks",
    REPO_ROOT / "poc" / "swarm_stepdown" / "data",
    REPO_ROOT / "poc" / "swarm_b7xplant" / "data",
    REPO_ROOT / "poc" / "swarm_b8xplant" / "data",
]


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


def save_brick_json(b: dict, name: str) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_p = DATA_DIR / f"brick_{name}.json"
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


def _side_from_arr(x: np.ndarray, y: np.ndarray, th: np.ndarray) -> float:
    h = 0.5 * (np.abs(np.cos(th)) + np.abs(np.sin(th)))
    return float(max((x + h).max() - (x - h).min(), (y + h).max() - (y - h).min()))


def fast_load_true_frontier(target_ns: list[int] | set[int]) -> dict[int, dict]:
    """Fast portfolio loader: float pre-sorts merged.csv + all poc/*/final.csv, then dual-certifies top candidates."""
    need = {int(n) for n in target_ns}
    cands: dict[int, list[tuple[float, str, list[dict[str, str]]]]] = {n: [] for n in need}

    need_s = {str(n) for n in need}
    df = pd.read_csv(bl.MERGED_CSV, usecols=["n", "square_id", "x", "y", "angle"], dtype=str)
    for n_val, grp in df[df["n"].isin(need_s)].groupby("n"):
        n = int(n_val)
        grp_s = grp.assign(_sid=grp["square_id"].astype(int)).sort_values("_sid")
        xv = grp_s["x"].astype(float).to_numpy()
        yv = grp_s["y"].astype(float).to_numpy()
        av = grp_s["angle"].astype(float).to_numpy()
        L_f = _side_from_arr(xv, yv, av)
        lit_rows = [
            {"square_id": r.square_id, "x": r.x, "y": r.y, "angle": r.angle}
            for r in grp_s.itertuples(index=False)
        ]
        pd_rows = [
            {
                "square_id": str(int(r.square_id)),
                "x": repr(float(r.x)),
                "y": repr(float(r.y)),
                "angle": repr(float(r.angle)),
            }
            for r in grp_s.itertuples(index=False)
        ]
        cands[n].append((L_f, "merged.csv", lit_rows, pd_rows))

    prefixes = tuple(f"{n:03d}_" if n < 1000 else "1000_" for n in sorted(need))
    for p in sorted((REPO_ROOT / "poc").glob("*/final.csv")):
        if p == FINAL_CSV or not p.exists() or p.stat().st_size <= 25:
            continue
        txt = p.read_text(encoding="utf-8", errors="ignore")
        lines = txt.splitlines()
        if len(lines) <= 1 or not lines[0].startswith("id,x,y,deg"):
            continue
        sub = [ln for ln in lines[1:] if ln.startswith(prefixes)]
        if not sub:
            continue
        by_n_pts: dict[int, list[tuple[int, str, str, str, float, float, float]]] = {}
        for ln in sub:
            parts = ln.split(",")
            if len(parts) != 4:
                continue
            p_id = parts[0].split("_")
            if len(p_id) != 2 or not p_id[0].isdigit() or not p_id[1].isdigit():
                continue
            n_i, sid = int(p_id[0]), int(p_id[1])
            if n_i not in need:
                continue
            xs = parts[1][1:] if parts[1].startswith("s") else parts[1]
            ys = parts[2][1:] if parts[2].startswith("s") else parts[2]
            ds = parts[3][1:] if parts[3].startswith("s") else parts[3]
            by_n_pts.setdefault(n_i, []).append((sid, xs, ys, ds, float(xs), float(ys), math.radians(float(ds))))

        rel = str(p.relative_to(REPO_ROOT))
        for n_i, pts in by_n_pts.items():
            if len(pts) != n_i:
                continue
            pts.sort(key=lambda t: t[0])
            arr = np.array([(t[4], t[5], t[6]) for t in pts], dtype=np.float64)
            L_f = _side_from_arr(arr[:, 0], arr[:, 1], arr[:, 2])
            rows_raw = [
                {
                    "square_id": str(t[0]),
                    "x": t[1],
                    "y": t[2],
                    "angle": repr(t[6]),
                    "_deg_raw": t[3],
                }
                for t in pts
            ]
            cands[n_i].append((L_f, rel, rows_raw, sub))

    best: dict[int, dict] = {}
    for n in sorted(need):
        ranked = sorted(cands[n], key=lambda t: t[0])
        best_L: Decimal | None = None
        best_entry: dict | None = None
        for item in ranked:
            if best_L is not None and item[0] > float(best_L) + 1e-8:
                break
            src = item[1]
            try:
                if src == "merged.csv":
                    lit_rows = item[2]
                    pd_rows = item[3]
                else:
                    # Re-certify exact CSV slice from that final.csv
                    sub_lines = [ln for ln in item[3] if ln.startswith(f"{n:03d}_" if n < 1000 else "1000_")]
                    text = "id,x,y,deg\n" + "\n".join(sub_lines) + "\n"
                    lit_d = auto_union.parse_text(text)
                    pd_d = auto_union.parse_text(pd.read_csv(io.StringIO(text)).to_csv(index=False))
                    lit_rows, pd_rows = lit_d[n], pd_d[n]
                res = auto_union.check((n, lit_rows, pd_rows))
            except Exception:
                continue
            if res is None:
                continue
            L_dec = max(Decimal(res[0]), Decimal(res[1]))
            if best_L is None or L_dec < best_L:
                best_L = L_dec
                rows_s = sorted(lit_rows, key=lambda r: int(r["square_id"]))
                x, y, th, L_f = bl.normalize(
                    np.array([float(r["x"]) for r in rows_s], dtype=np.float64),
                    np.array([float(r["y"]) for r in rows_s], dtype=np.float64),
                    np.array([float(r["angle"]) for r in rows_s], dtype=np.float64),
                )
                best_entry = {"x": x, "y": y, "th": th, "L": L_f, "L_dec": L_dec, "source": src}
        if best_entry is None:
            raise RuntimeError(f"No certified frontier found for n={n}")
        best[n] = best_entry
    return best


def remove_vertical_strip(b: dict, strip_idx: int | None = None) -> list[tuple[str, dict]]:
    """Remove 1 interior tilted strip + matching AA border squares: (w, h, g) -> (w, h-1, g-1)."""
    w, h, g = b["w"], b["h"], b["g"]
    if h <= 4 or g <= 1:
        return []
    x, y, th = b["x"], b["y"], b["th"]
    if w > h:
        bt = bl.transform_brick(b, "T")
        res = []
        for tag, sb in remove_vertical_strip(bt, strip_idx=strip_idx):
            sbt = bl.transform_brick(sb, "T")
            sbt["tag"] = tag + "_T"
            res.append((sbt["tag"], sbt))
        return res

    deg = np.degrees(np.mod(th, 0.5 * math.pi))
    tilt = np.minimum(deg, 90.0 - deg) > 2.0
    strips, _ = gl._cluster_strips(x, y, th, tilt)
    if len(strips) < 7:
        return []
    aa_idx = np.flatnonzero(~tilt)
    aa_cols = gl._cluster_aa_columns(x, aa_idx)

    max_len = max(len(s) for s in strips)
    cand_indices = (
        [strip_idx]
        if strip_idx is not None
        else [k for k in range(2, len(strips) - 2) if len(strips[k]) >= max_len - 1]
    )

    out: list[tuple[str, dict]] = []
    target_n = w * (h - 1) + (g - 1)
    for k in cand_indices:
        if k <= 0 or k >= len(strips) - 1:
            continue
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

        # Keep strips < k, drop strip k, shift strips > k by (-dx_s, -dy_s)
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

        y_left = float(y[sk[0]])
        y_right = float(y[sk[-1]])
        aa_new_x, aa_new_y, aa_new_t = [], [], []
        for col in aa_cols:
            col_s = sorted(col, key=lambda i: y[i])
            cx_mean = float(np.mean([x[i] for i in col_s]))
            y_cut = y_left if cx_mean < 0.5 * w else y_right
            below = [i for i in col_s if y[i] <= y_cut + 0.15]
            above = [i for i in col_s if y[i] > y_cut + 0.15]
            for i in below:
                aa_new_x.append(float(x[i]))
                aa_new_y.append(float(y[i]))
                aa_new_t.append(float(th[i]))
            # Drop 1 AA square if continuous crossing y_cut, shift remaining above by -1.0
            drop_first = bool(below and above and (y[above[0]] - y[below[-1]]) < 1.35)
            start_idx = 1 if drop_first else 0
            for i in above[start_idx:]:
                aa_new_x.append(float(x[i]))
                aa_new_y.append(float(y[i] - 1.0))
                aa_new_t.append(float(th[i]))

        if len(nx_l) + len(aa_new_x) != target_n:
            continue

        ax = np.array(nx_l + aa_new_x, dtype=np.float64)
        ay = np.array(ny_l + aa_new_y, dtype=np.float64)
        at = np.array(nt_l + aa_new_t, dtype=np.float64)
        cand = make_brick(w, h - 1, g - 1, ax, ay, at, tag=f"{b['tag']}_delvs_k{k}")
        out.append((cand["tag"], cand))
    return out


def compress_and_walk_brick(
    b: dict,
    name: str,
    f_targets: tuple[float, ...] = (0.76, 0.72, 0.695, 0.675, 0.655),
    n_trials: int = 10,
    walk_rounds: int = 2,
    seed: int = 42,
    save: bool = True,
) -> dict | None:
    """Relax spliced/peeled brick, compress via SLP, and polish via brick_mechwalk."""
    pen0, _ = cpp_eval_sat_pen_bbox(b["x"], b["y"], b["th"])
    if pen0 <= 1e-8:
        comp = cp2.compress_brick_slp(b, n_trials=n_trials, seed=seed)
    else:
        comp = gl.relax_and_compress_brick(b, f_targets=f_targets, n_trials=n_trials, seed=seed)
    if comp is None:
        return None
    pen_c, _ = cpp_eval_sat_pen_bbox(comp["x"], comp["y"], comp["th"])
    if pen_c > 1e-8:
        return None
    walked = gl.brick_mechwalk(comp, name, max_rounds=walk_rounds, save=False)
    walked["tag"] = name
    if save:
        save_brick_json(walked, name)
    return walked


def extract_donor_subbricks(donor_ns: list[int] = (492, 852, 911, 975)) -> dict[str, dict]:
    """Extract corner tilted sub-bricks from donor frontier packings."""
    frontier = fast_load_true_frontier(list(donor_ns))
    extracted: dict[str, dict] = {}
    for n in donor_ns:
        d = frontier[n]
        x, y, th, L = d["x"], d["y"], d["th"], d["L"]
        m = int(math.floor(L))
        f = L - m
        hx = bl.half_extent(th)
        deg = np.degrees(np.mod(th, 0.5 * math.pi))
        tilt = np.minimum(deg, 90.0 - deg) > 2.0

        for corner in ("TL", "BR", "BL", "TR"):
            cand_by_g: dict[int, tuple] = {}
            for w in range(4, m):
                for h in range(4, m):
                    if corner == "TL":
                        mask = (x + hx <= w + f + 0.05) & (y - hx >= m - h - 0.05)
                    elif corner == "BR":
                        mask = (x - hx >= m - w - 0.05) & (y + hx <= h + f + 0.05)
                    elif corner == "BL":
                        mask = (x + hx <= w + f + 0.05) & (y + hx <= h + f + 0.05)
                    else:
                        mask = (x - hx >= m - w - 0.05) & (y - hx >= m - h - 0.05)
                    g = int(mask.sum()) - w * h
                    if g < 3 or g > 8 or int((mask & tilt).sum()) < 20:
                        continue
                    # Avoid redundant huge squares that wrap around opposite tilted cluster
                    if w >= m - 2 or h >= m - 2:
                        continue
                    _, _, W, H = bl.rect_bbox(x[mask], y[mask], th[mask])
                    fc = max(W - w, H - h)
                    if fc <= f + 0.005:
                        # Keep minimal perimeter for each g, and also keep all distinct minimal-w or minimal-h
                        key = g
                        if key not in cand_by_g or (w + h, fc) < (cand_by_g[key][0], cand_by_g[key][1]):
                            cand_by_g[key] = (w + h, fc, w, h, mask)
            for g, (_, fc, w, h, mask) in sorted(cand_by_g.items()):
                tag = f"B{g}_{w}x{h}_n{n}_{corner}"
                b = make_brick(w, h, g, x[mask], y[mask], th[mask], tag=tag)
                pen, _ = cpp_eval_sat_pen_bbox(b["x"], b["y"], b["th"])
                if pen <= 1e-8:
                    extracted[tag] = b
    return extracted


def load_all_bricks(include_external: bool = True, max_per_shape: int = 2) -> list[dict]:
    """Load all local + external bricks, deduplicate by (w, h, g), and include transposes."""
    raw: list[dict] = []
    if DATA_DIR.exists():
        for p in sorted(DATA_DIR.glob("brick_*.json")):
            raw.append(load_brick_json(p))
    if include_external:
        for d in EXTERNAL_BRICK_DIRS:
            if not d.exists():
                continue
            for p in sorted(d.glob("*.json")):
                if p.name.startswith("win_"):
                    continue
                try:
                    jd = json.loads(p.read_text(encoding="utf-8"))
                    if all(k in jd for k in ("w", "h", "g", "x", "y", "th")):
                        raw.append(load_brick_json(p, tag=jd.get("name", p.stem.replace("brick_", ""))))
                except Exception:
                    continue

    by_shape: dict[tuple[int, int, int], list[dict]] = {}
    seen_tags: set[tuple[int, int, int, str]] = set()
    for b in raw:
        for cand in (b, bl.transform_brick(b, "T")):
            if cand is not b:
                cand["tag"] = b["tag"] + "_T"
            key = (cand["w"], cand["h"], cand["g"])
            tk = (*key, cand["tag"])
            if tk in seen_tags:
                continue
            seen_tags.add(tk)
            by_shape.setdefault(key, []).append(cand)

    kept: list[dict] = []
    for _, lst in sorted(by_shape.items()):
        lst.sort(key=lambda c: c["f"])
        kept.extend(lst[:max_per_shape])
    return kept


def build_pinwheel(
    m: int,
    b1: dict,
    b2: dict,
    sf: float = 0.0,
    f_override: float | None = None,
    max_ov: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray] | None:
    """Build TL(b1) + BR(b2) pinwheel allowing zero-overlap or corner/1-row seam overlap."""
    w1, h1, w2, h2 = b1["w"], b1["h"], b2["w"], b2["h"]
    if w1 >= m or h1 >= m or w2 >= m or h2 >= m:
        return None
    if w1 + w2 > m + max_ov or h1 + h2 > m + max_ov:
        return None
    if w1 + w2 > m and h1 + h2 > m:
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
    seam = (
        ((ax >= w1 - 1.8) & (ax <= w1 + f + 2.0))
        | ((ay >= m - h1 - 2.0) & (ay <= m - h1 + 2.0))
        | ((ax >= m - w2 - 2.0) & (ax <= m - w2 + 2.0))
        | ((ay >= h2 - 1.8) & (ay <= h2 + f + 2.0))
        | ((np.abs(ax - w1) <= 3.4) & (np.abs(ay - (m - h1)) <= 3.4))
        | ((np.abs(ax - (m - w2)) <= 3.4) & (np.abs(ay - h2) <= 3.4))
    ).astype(np.uint8)
    return ax, ay, at, L, seam


def pslp_squeeze_cert(
    n: int,
    cert: dict,
    inc_dec: Decimal,
    shake_rounds: int = 4,
    seed: int = 123,
) -> dict:
    """Run kink-safe free-angle pslp squeeze + shake_squeeze + polish + lpslack on a cert."""
    m = int(math.floor(float(inc_dec)))
    rs = sorted(cert["rows"], key=lambda r: int(r["square_id"]))
    x0 = np.array([float(r["x"]) for r in rs], dtype=np.float64)
    y0 = np.array([float(r["y"]) for r in rs], dtype=np.float64)
    t0 = np.array([float(r["angle"]) for r in rs], dtype=np.float64)
    nx, ny, nth, L0 = bl.normalize(x0, y0, t0)
    f0 = L0 - m
    f1, X1, Y1, T1, v1, _ = pslp.squeeze(nx, ny, nth, m, m, f0, max_it=200)
    bx, by, bt, bf = (X1, Y1, T1, f1) if v1 <= 1e-12 else (nx, ny, nth, f0)
    if shake_rounds > 0:
        rng = np.random.default_rng(seed)
        f2, X2, Y2, T2 = pslp.shake_squeeze(bx, by, bt, m, m, bf, rounds=shake_rounds, rng=rng)
        if f2 < bf - 1e-10 and pslp.viol(X2, Y2, T2, m, m, f2) <= 1e-12:
            bx, by, bt, bf = X2, Y2, T2, f2
    pcert = bl.polish_and_certify(n, bx, by, bt, inc_dec, slp_rounds=14)
    if pcert is not None:
        pcert["label"] = cert["label"] + "+pslp"
        pcert = _apply_lpslack(n, pcert, inc_dec)
        if max(Decimal(pcert["L_lit"]), Decimal(pcert["L_pandas"])) < max(Decimal(cert["L_lit"]), Decimal(cert["L_pandas"])):
            return pcert
    return cert


def _apply_lpslack(n: int, cert: dict, inc_dec: Decimal) -> dict:
    rs = sorted(cert["rows"], key=lambda r: int(r["square_id"]))
    ceiling = str(max(inc_dec, Decimal(repr(cert["L_float"] + 0.1))))
    res_lp = lpslack.optimize_group(
        n,
        rs,
        ceiling,
        outer_iters=6,
        fixed_rounds=8,
        slp_modes=("dual_non90", "non90"),
        tr_th_schedule=(4e-4, 1e-4),
        time_limit=12.0,
    )
    if res_lp.get("status") in ("WIN", "TINY_WIN") and res_lp.get("rows"):
        lit_s = res_lp.get("literal_L", res_lp.get("new_L_lit"))
        pan_s = res_lp.get("pandas_L", res_lp.get("new_L_pan"))
        lit, pan = Decimal(lit_s), Decimal(pan_s)
        worst = max(lit, pan)
        if worst < max(Decimal(cert["L_lit"]), Decimal(cert["L_pandas"])):
            cert["L_lit"] = str(lit)
            cert["L_pandas"] = str(pan)
            cert["L_dec_str"] = str(worst)
            cert["L_float"] = float(worst)
            cert["dL"] = float(worst - inc_dec)
            cert["is_real_win"] = bool(worst - inc_dec <= Decimal("-1e-9"))
            cert["rows"] = res_lp["rows"]
            cert["label"] = cert["label"] + "+lpslack"
    return cert


def solve_pair(
    n: int,
    m: int,
    b1_base: dict,
    b2_base: dict,
    inc_dec: Decimal,
    tag: str,
    base_seed: int,
    max_ov: int = 1,
    do_melt: bool = True,
    do_lpslack: bool = True,
    do_mechwalk: bool = True,
    do_pslp: bool = True,
    n_threads: int = 2,
) -> dict | None:
    """Assemble pinwheel (zero-ov or corner-touch/ov<=1), C++ seam melt, SLP, lpslack, and mechwalk_compress."""
    best_cert: dict | None = None
    seeds, masks, scheds, labels = [], [], [], []
    w1, h1, w2, h2 = b1_base["w"], b1_base["h"], b2_base["w"], b2_base["h"]
    is_zero_ov = (w1 + w2 <= m) and (h1 + h2 <= m) and (w1 + w2 + h1 + h2 <= 2 * m - 1)

    for m1 in ("id", "flipx", "flipy", "rot180"):
        b1 = bl.transform_brick(b1_base, m1)
        for m2 in ("id", "flipx", "flipy", "rot180"):
            b2 = bl.transform_brick(b2_base, m2)
            f_hi = max(b1["fx"], b1["fy"], b2["fx"], b2["fy"])
            f_lo = min(b1["fx"], b1["fy"], b2["fx"], b2["fy"])
            f_list = ((f_hi, "fhi"),) if abs(f_hi - f_lo) < 0.004 else ((f_lo, "flo"), (f_hi, "fhi"))
            for f_use, ftag in f_list:
                built = build_pinwheel(m, b1, b2, sf=0.0, f_override=f_use, max_ov=max_ov)
                if built is None:
                    continue
                ax, ay, at, L_tgt, sm = built
                if is_zero_ov and ftag == "fhi" and (m1, m2) in (("id", "id"), ("flipx", "flipy"), ("flipy", "flipx"), ("rot180", "rot180")):
                    pen, _ = cpp_eval_sat_pen_bbox(ax, ay, at)
                    if pen <= 1e-10:
                        cert = bl.polish_and_certify(n, ax, ay, at, inc_dec, slp_rounds=10)
                        if cert is not None:
                            cert["label"] = f"{tag}_{m1}_{m2}_direct"
                            if best_cert is None or cert["L_float"] < best_cert["L_float"]:
                                best_cert = cert
                if do_melt and ftag == "fhi":
                    inc_f = float(inc_dec)
                    if inc_f - L_tgt > 0.005:
                        L_end = min(inc_f - 0.001, max(L_tgt + 0.001, inc_f - 0.004))
                        L_mid = max(L_tgt + 0.008, L_end + 0.006)
                        L_start = max(L_tgt + 0.022, L_end + 0.016)
                    else:
                        L_end = min(inc_f - 3e-4, L_tgt + 0.001) if L_tgt + 0.001 < inc_f else 0.5 * (L_tgt + inc_f)
                        L_mid = L_tgt + 0.005
                        L_start = L_tgt + 0.018
                    seeds.append((ax, ay, at))
                    masks.append(sm)
                    scheds.append([L_start, L_mid, L_end])
                    labels.append(f"{tag}_{m1}_{m2}_{ftag}")

    if do_melt and seeds and (best_cert is None or best_cert["dL"] > -5e-4):
        for start in range(0, len(seeds), 16):
            end = min(len(seeds), start + 16)
            relaxed = bl.su.cpp_batch_melt(
                seeds[start:end],
                scheds[start:end],
                sym_mode=1,
                a_start=0.962,
                n_melt_stages=3,
                seam_masks=masks[start:end],
                seam_iters=220,
                full_iters=260,
                n_ruin_iters=0,
                n_threads=min(2, max(1, n_threads)),
                base_seed=base_seed + start,
            )
            pool = []
            for bi, row in enumerate(relaxed):
                nx, ny, nth, Lm = bl.normalize(row["x"], row["y"], row["th"])
                lbl = f"{labels[start + bi]}_melt"
                if row["best_L_feas"] is not None:
                    pool.append((0.0, Lm, nx, ny, nth, lbl))
                elif row["pen"] < 0.025:
                    pool.append((1.0 + float(row["pen"]), Lm, nx, ny, nth, lbl))
            pool.sort(key=lambda t: (t[0], t[1]))
            for _, _, cx, cy, cth, lbl in pool[:2]:
                sx, sy, st, _, ok = bslp.joint_xytheta_slp_tight(cx, cy, cth)
                ux, uy, ut = (sx, sy, st) if ok else (cx, cy, cth)
                cert = bl.polish_and_certify(n, ux, uy, ut, inc_dec, slp_rounds=14)
                if cert is not None:
                    cert["label"] = lbl
                    if best_cert is None or cert["L_float"] < best_cert["L_float"]:
                        best_cert = cert

    if do_lpslack and best_cert is not None:
        best_cert = _apply_lpslack(n, best_cert, inc_dec)

    if do_pslp and best_cert is not None and best_cert["L_float"] < float(inc_dec) + 0.002:
        best_cert = pslp_squeeze_cert(n, best_cert, inc_dec, shake_rounds=3, seed=base_seed + 55)

    if do_mechwalk and best_cert is not None and best_cert["L_float"] < float(inc_dec) + 0.0005:
        rs = sorted(best_cert["rows"], key=lambda r: int(r["square_id"]))
        cx = np.array([float(r["x"]) for r in rs], dtype=np.float64)
        cy = np.array([float(r["y"]) for r in rs], dtype=np.float64)
        ct = np.array([float(r["angle"]) for r in rs], dtype=np.float64)
        rng = np.random.default_rng(base_seed + 77)
        mw_cert, _ = hc2.mechwalk_compress(n, cx, cy, ct, inc_dec, rng, rounds=1, threads=min(2, max(1, n_threads)))
        if mw_cert is not None:
            mw_L = max(Decimal(mw_cert["L_lit"]), Decimal(mw_cert["L_pd"]))
            cur_L = max(Decimal(best_cert["L_lit"]), Decimal(best_cert["L_pandas"]))
            if mw_L < cur_L:
                best_cert = {
                    "ok": True,
                    "n": int(n),
                    "L_lit": mw_cert["L_lit"],
                    "L_pandas": mw_cert["L_pd"],
                    "L_dec_str": str(mw_L),
                    "L_float": float(mw_L),
                    "dL": float(mw_L - inc_dec),
                    "is_real_win": bool(mw_L - inc_dec <= Decimal("-1e-9")),
                    "rows": mw_cert["rows"],
                    "label": best_cert["label"] + "+mechwalk",
                }
    return best_cert


def save_win_locked(win_entry: dict) -> bool:
    return bl.save_win_locked(win_entry, data_dir=DATA_DIR, final_path=FINAL_CSV)
