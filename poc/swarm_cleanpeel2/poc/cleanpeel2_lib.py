#!/usr/bin/env python3
"""H74 (cleanpeel2): Sub-Brick Row/Column Clean-Peel + Standalone Brick Mechwalk + Zero-Overlap Pinwheels."""
from __future__ import annotations

import csv
import fcntl
import io
import json
import math
import os
import sys
from decimal import Decimal
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CP2_DIR = HERE.parent
DATA_DIR = CP2_DIR / "data"
DOCS_DIR = CP2_DIR / "docs"
FINAL_CSV = CP2_DIR / "final.csv"

REPO_ROOT = CP2_DIR.parents[1]
B8_POC = REPO_ROOT / "poc" / "swarm_b8xplant" / "poc"
AUTO_UNION_POC = REPO_ROOT / "poc" / "auto_union" / "poc"

for _p in (B8_POC, AUTO_UNION_POC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import b8_lib as bl  # noqa: E402
import brick_rect_lib as bslp  # noqa: E402
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
        tag=tag or d.get("name", path.stem),
    )


def load_json_pack(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    d = json.loads(path.read_text(encoding="utf-8"))
    rs = sorted(d["rows"], key=lambda r: int(r["square_id"]))
    ang = np.array([float(r["angle"]) for r in rs], dtype=np.float64)
    if float(np.max(np.abs(ang))) > 6.3:
        ang = np.radians(ang)
    return bl.normalize(
        np.array([float(r["x"]) for r in rs], dtype=np.float64),
        np.array([float(r["y"]) for r in rs], dtype=np.float64),
        ang,
    )


def extract_corner_brick(
    x: np.ndarray, y: np.ndarray, th: np.ndarray, L: float,
    corner: str, w: int, h: int, g: int, tag: str, margin: float = 0.78,
) -> dict | None:
    hx = bl.half_extent(th)
    if corner == "TL":
        m = (x + hx <= w + margin) & (y - hx >= L - (h + margin))
    elif corner == "TR":
        m = (x - hx >= L - (w + margin)) & (y - hx >= L - (h + margin))
    elif corner == "BL":
        m = (x + hx <= w + margin) & (y + hx <= h + margin)
    elif corner == "BR":
        m = (x - hx >= L - (w + margin)) & (y + hx <= h + margin)
    else:
        raise ValueError(f"Unknown corner {corner}")
    if int(m.sum()) != w * h + g:
        return None
    b = make_brick(w, h, g, x[m], y[m], th[m], tag=tag)
    pen, _ = cpp_eval_sat_pen_bbox(b["x"], b["y"], b["th"])
    if pen > 1e-7:
        return None
    return b


def peel_outer_unit_layer(b: dict, side: str, tag: str = "") -> dict | None:
    """Peel 1 outer unit row ('top'/'bottom') or column ('left'/'right') from a brick."""
    w, h, g = b["w"], b["h"], b["g"]
    x, y, th = b["x"], b["y"], b["th"]
    hx = bl.half_extent(th)
    if side == "top":
        if h <= 2:
            return None
        idx_drop = np.argsort(-(y + hx))[:w]
        nw, nh = w, h - 1
    elif side == "bottom":
        if h <= 2:
            return None
        idx_drop = np.argsort(y - hx)[:w]
        nw, nh = w, h - 1
    elif side == "left":
        if w <= 2:
            return None
        idx_drop = np.argsort(x - hx)[:h]
        nw, nh = w - 1, h
    elif side == "right":
        if w <= 2:
            return None
        idx_drop = np.argsort(-(x + hx))[:h]
        nw, nh = w - 1, h
    else:
        raise ValueError(f"Unknown side {side}")

    keep = np.ones(len(x), dtype=bool)
    keep[idx_drop] = False
    if int(keep.sum()) != nw * nh + g:
        return None
    return make_brick(nw, nh, g, x[keep], y[keep], th[keep], tag=tag or f"{b['tag']}_peel_{side}")


def compress_brick_slp(b: dict, n_trials: int = 16, seed: int = 42) -> dict:
    """Standalone rectangular compression via downward L-BFGS-B + joint_xytheta_slp_rect."""
    w, h, g = b["w"], b["h"], b["g"]
    x0, y0, t0 = b["x"], b["y"], b["th"]
    W0, H0 = w + b["fx"], h + b["fy"]
    f0 = b["f"]
    fmin = min(b["fx"], b["fy"])
    rng = np.random.default_rng(seed)
    best = dict(b, tag=f"{b['tag']}_slp") if cpp_eval_sat_pen_bbox(x0, y0, t0)[0] <= 1e-8 else None
    deg = np.degrees(np.mod(t0, 0.5 * math.pi))
    tilt = np.minimum(deg, 90.0 - deg) > 2.0

    if f0 > 0.68:
        ft_sched = (f0 - 0.01, max(0.665, f0 - 0.06), max(0.648, min(fmin + 0.006, f0 - 0.12)), max(0.642, fmin))
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
            for ft in ft_sched:
                cx, cy, ct, _ = bslp.relax_rect_lbfgs(cx, cy, ct, w + ft, h + ft, maxiter=280)
        sx, sy, st, _, ok = bslp.joint_xytheta_slp_rect(
            cx, cy, ct, w, h, rounds=14, tr_th0=0.022, tr_xy0=0.09, tol=1e-11
        )
        if ok:
            pen_s, _ = cpp_eval_sat_pen_bbox(sx, sy, st)
            if pen_s <= 1e-8:
                cb = make_brick(w, h, g, sx, sy, st, tag=f"{b['tag']}_slp")
                if best is None or cb["f"] < best["f"] - 1e-10:
                    best = cb
    return best if best is not None else b


def _try_brick_direction(
    best_state: tuple[np.ndarray, np.ndarray, np.ndarray],
    v: np.ndarray,
    w: int,
    h: int,
    best_f: float,
    amps: tuple[float, ...],
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], float, float | None]:
    nf = len(best_state[0])
    for amp in amps:
        for sgn in (1.0, -1.0):
            cx = best_state[0] + sgn * amp * v[:nf]
            cy = best_state[1] + sgn * amp * v[nf : 2 * nf]
            ct = best_state[2] + sgn * amp * v[2 * nf :]
            rx, ry, rt = None, None, None
            for df_try in (-1.5e-3, -2e-4, 2e-4):
                tx, ty, tt, pen = bslp.relax_rect_lbfgs(
                    cx, cy, ct, w + best_f + df_try, h + best_f + df_try, maxiter=320
                )
                if pen <= 1e-6:
                    rx, ry, rt = tx, ty, tt
                    break
            if rx is None:
                continue
            sx, sy, st, sf, ok = bslp.joint_xytheta_slp_rect(rx, ry, rt, w, h, rounds=10)
            if ok:
                pen_s, _ = cpp_eval_sat_pen_bbox(sx, sy, st)
                if pen_s <= 1e-8 and sf < best_f - 1e-7:
                    return (sx, sy, st), sf, sgn * amp
    return best_state, best_f, None


def brick_mechwalk(b: dict, name: str, max_rounds: int = 3, save: bool = True) -> dict:
    """Standalone Contact-Jacobian SVD walk + d(i+-j) diagonal mode pairs on a (w, h, g) brick."""
    w, h, g = b["w"], b["h"], b["g"]
    bx, by, W0, H0 = bl.rect_bbox(b["x"], b["y"], b["th"])
    bt = b["th"].copy()
    f0 = max(W0 - w, H0 - h)
    best_f = f0
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
            _, modes = bl.localized_soft_modes(jac, k=12)
            n_single = min(10, len(modes))
            for mi in range(n_single):
                new_state, new_f, hit_amp = _try_brick_direction(
                    best_state, modes[mi], w, h, best_f, (0.004, 0.008, 0.025, 0.08)
                )
                if hit_amp is not None:
                    best_f = new_f
                    best_state = new_state
                    improved = True
                    print(f"  [BRICK {name}] rnd={rnd} ctol={ctol} m{mi} amp={hit_amp} -> f={best_f:.9f} (df={best_f-f0:+.6e})", flush=True)
            kcap = min(6, len(modes))
            for i in range(kcap):
                for j in range(i + 1, kcap):
                    for d_sgn, dtag in ((1.0, "+"), (-1.0, "-")):
                        vd = inv_s2 * (modes[i] + d_sgn * modes[j])
                        new_state, new_f, hit_amp = _try_brick_direction(
                            best_state, vd, w, h, best_f, (0.008, 0.025, 0.06)
                        )
                        if hit_amp is not None:
                            best_f = new_f
                            best_state = new_state
                            improved = True
                            print(f"  [BRICK {name}] rnd={rnd} ctol={ctol} d({i}{dtag}{j}) amp={hit_amp} -> f={best_f:.9f} (df={best_f-f0:+.6e})", flush=True)
        if not improved:
            break

    out = make_brick(w, h, g, best_state[0], best_state[1], best_state[2], tag=name)
    if save:
        save_brick_json(out, name)
    return out


def fast_load_true_frontier(
    target_ns: list[int] | set[int], certify: bool = True
) -> dict[int, dict]:
    """Load TRUE frontier (merged.csv + all poc/*/final.csv on origin/main) for target_ns."""
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
                {
                    "square_id": int(r.square_id),
                    "x": repr(float(r.x)),
                    "y": repr(float(r.y)),
                    "angle": repr(float(r.angle)),
                }
                for r in grp_s.itertuples(index=False)
            ]
            L_dec = Decimal(auto_union.certify(n, rows))
        else:
            L_dec = Decimal(f"{L_f:.17g}")
        best[n] = {"x": x, "y": y, "th": th, "L": L_f, "L_dec": L_dec, "source": "merged.csv"}

    need_prefixes = tuple(f"{n:03}_" for n in sorted(need))
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
        try:
            groups_raw: dict[int, list[tuple[int, float, float, float]]] = {}
            for row in csv.DictReader(io.StringIO(sub_txt)):
                ni = int(row["id"].split("_")[0])
                sid = int(row["id"].split("_")[1])
                groups_raw.setdefault(ni, []).append(
                    (sid, float(row["x"].lstrip("s")), float(row["y"].lstrip("s")), math.radians(float(row["deg"].lstrip("s"))))
                )
        except Exception:
            continue
        cand_ns = [
            ni for ni, rlist in groups_raw.items()
            if ni in need and len(rlist) == ni
        ]
        if not cand_ns:
            continue
        if not certify:
            for ni in cand_ns:
                rlist = sorted(groups_raw[ni])
                nx, ny, nt, L_f = bl.normalize([r[1] for r in rlist], [r[2] for r in rlist], [r[3] for r in rlist])
                if ni not in best or L_f < best[ni]["L"] - 1e-12:
                    best[ni] = {"x": nx, "y": ny, "th": nt, "L": L_f, "L_dec": Decimal(f"{L_f:.17g}"), "source": str(p.relative_to(REPO_ROOT))}
            continue
        try:
            lit = auto_union.parse_text(sub_txt)
            pd_t = auto_union.parse_text(pd.read_csv(io.StringIO(sub_txt)).to_csv(index=False))
        except Exception:
            continue
        for n in cand_ns:
            if n in lit and n in pd_t:
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


def build_pinwheel_zero_ov(
    m: int, b1: dict, b2: dict, sf: float = 0.0, f_override: float | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray] | None:
    """Assemble 5-region pinwheel when w1+w2 <= m and h1+h2 <= m (kx<=0, ky<=0)."""
    w1, h1, w2, h2 = b1["w"], b1["h"], b2["w"], b2["h"]
    if w1 >= m or h1 >= m or w2 >= m or h2 >= m:
        return None
    if w1 + w2 > m or h1 + h2 > m:
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
        ((ax >= w1 - 1.5) & (ax <= w1 + f + 1.8))
        | ((ay >= m - h1 - 1.8) & (ay <= m - h1 + 1.8))
        | ((ax >= m - w2 - 1.8) & (ax <= m - w2 + 1.8))
        | ((ay >= h2 - 1.5) & (ay <= h2 + f + 1.8))
    ).astype(np.uint8)
    return ax, ay, at, L, seam


def solve_pinwheel_pair(
    n: int, m: int, b1_base: dict, b2_base: dict, inc_dec: Decimal, tag: str, base_seed: int,
    n_threads: int = 4,
) -> dict | None:
    """Test zero-overlap direct SLP + multi-sf/transform seam-melted pinwheels."""
    best_cert: dict | None = None

    seeds, masks, scheds, labels = [], [], [], []
    for m1 in ("id", "flipx", "flipy", "rot180"):
        b1 = bl.transform_brick(b1_base, m1)
        for m2 in ("id", "flipx", "flipy", "rot180"):
            b2 = bl.transform_brick(b2_base, m2)
            f_hi = max(b1["fx"], b1["fy"], b2["fx"], b2["fy"])
            f_lo = min(b1["fx"], b1["fy"], b2["fx"], b2["fy"])
            f_avg = 0.5 * (max(b1["fx"], b1["fy"]) + max(b2["fx"], b2["fy"]))
            if abs(f_hi - f_lo) < 5e-4:
                sf_list = (0.0,)
                f_list = ((f_hi, "fhi"),)
            else:
                sf_list = (0.0, 0.5, 0.8)
                f_list = ((f_lo, "flo"), (f_avg, "favg"), (f_hi, "fhi"))
            for sf in sf_list:
                for f_use, ftag in f_list:
                    built = build_pinwheel_zero_ov(m, b1, b2, sf=sf, f_override=f_use)
                    if built is None:
                        continue
                    ax, ay, at, L0, sm = built
                    assert len(ax) == n, f"len={len(ax)} != {n}"
                    pen, _ = cpp_eval_sat_pen_bbox(ax, ay, at)
                    if pen <= 1e-10 and (m1, m2) in (("id", "id"), ("flipx", "flipy"), ("rot180", "rot180")) and ftag == "fhi":
                        sx, sy, st, _, ok = bslp.joint_xytheta_slp_tight(ax, ay, at)
                        if ok:
                            cert = bl.polish_and_certify(n, sx, sy, st, inc_dec, slp_rounds=22)
                            if cert is not None:
                                cert["label"] = f"{tag}_{m1}_{m2}_sf{sf}_direct"
                                if best_cert is None or cert["L_float"] < best_cert["L_float"]:
                                    best_cert = cert
                    if (m1, m2) in (("id", "id"), ("flipx", "id"), ("id", "flipx"), ("flipx", "flipy"), ("rot180", "rot180"), ("flipy", "flipy")):
                        seeds.append((ax, ay, at))
                        masks.append(sm)
                        L_tgt = m + f_use
                        scheds.append([L_tgt + 0.008, L_tgt + 0.001, L_tgt - 3e-4])
                        labels.append(f"{tag}_{m1}_{m2}_sf{sf}_{ftag}")

    if not seeds:
        return best_cert

    for siter in (160, 0):
        for start in range(0, len(seeds), 64):
            end = min(len(seeds), start + 64)
            relaxed = bl.su.cpp_batch_melt(
                seeds[start:end], scheds[start:end], sym_mode=1, a_start=0.965, n_melt_stages=3,
                seam_masks=masks[start:end] if siter > 0 else None, seam_iters=siter,
                full_iters=260, n_ruin_iters=0, n_threads=n_threads, base_seed=base_seed + siter + start,
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
    return best_cert


def save_win_locked(win_entry: dict) -> bool:
    return bl.save_win_locked(win_entry, data_dir=DATA_DIR, final_path=FINAL_CSV)
