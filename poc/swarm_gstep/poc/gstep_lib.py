#!/usr/bin/env python3
"""H79 (gstep): Cross-Remainder g-Step Staircase Surgery + Brick Mechwalk + K=2 Pinwheels."""
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
GSTEP_DIR = HERE.parent
DATA_DIR = GSTEP_DIR / "data"
DOCS_DIR = GSTEP_DIR / "docs"
FINAL_CSV = GSTEP_DIR / "final.csv"

REPO_ROOT = GSTEP_DIR.parents[1]
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

CLEANPEEL2_DATA = Path(
    "/home/lucas/.gemini/antigravity-cli/worktrees/a4fb0b93-b63b-426d-aa12-67454e99bc08/swarm-cleanpeel2-8a77ac94/poc/swarm_cleanpeel2/data"
)


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


def load_source_bricks() -> dict[str, dict]:
    """Load ultra-thin g=+5 and g=+6 source bricks."""
    bricks: dict[str, dict] = {}
    b5_p = REPO_ROOT / "poc" / "swarm_b8xplant" / "data" / "brick_H60_B5_20x11_n970s6.json"
    if b5_p.exists():
        bricks["B5_20x11"] = load_brick_json(b5_p, "H60_B5_20x11_n970s6")
        bricks["B5_11x20"] = bl.transform_brick(bricks["B5_20x11"], "T")
        bricks["B5_11x20"]["tag"] = "B5_11x20"

    for nm in (
        "B6_10x21_mw972_walk",
        "B6_10x20_mw972_top",
        "B6_9x18_n796_TL",
        "B5_20x10_peel_top",
        "B5_19x11_peel_right",
        "B5_19x10_n850_peelR",
    ):
        p = DATA_DIR / f"brick_{nm}.json"
        if not p.exists() and (CLEANPEEL2_DATA / f"brick_{nm}.json").exists():
            p = CLEANPEEL2_DATA / f"brick_{nm}.json"
        if p.exists():
            bricks[nm] = load_brick_json(p, nm)
    return bricks


def _cluster_strips(x: np.ndarray, y: np.ndarray, th: np.ndarray, tilt_mask: np.ndarray):
    """Cluster tilted squares into parallel tilted strips sorted by u = y - x*tan(theta_med)."""
    ti = np.flatnonzero(tilt_mask)
    deg = np.degrees(np.mod(th[ti], 0.5 * math.pi))
    ang_med = math.radians(float(np.median(deg)))
    slope = math.tan(ang_med)
    u = y[ti] - x[ti] * slope
    order = np.argsort(u)
    strips: list[list[int]] = []
    cur: list[int] = []
    cur_u = None
    for idx in order:
        sq = int(ti[idx])
        uv = float(u[idx])
        if cur_u is None or abs(uv - cur_u) > 0.45:
            if cur:
                strips.append(sorted(cur, key=lambda i: x[i]))
            cur = [sq]
            cur_u = uv
        else:
            cur.append(sq)
            cur_u = float(np.mean([y[i] - x[i] * slope for i in cur]))
    if cur:
        strips.append(sorted(cur, key=lambda i: x[i]))
    return strips, ang_med


def _cluster_aa_columns(x: np.ndarray, aa_idx: np.ndarray) -> list[list[int]]:
    cols: list[tuple[float, list[int]]] = []
    for i in aa_idx:
        xi = float(x[i])
        hit = None
        for ci, (cx, lst) in enumerate(cols):
            if abs(cx - xi) < 0.35:
                lst.append(int(i))
                cols[ci] = (float(np.mean([x[k] for k in lst])), lst)
                hit = ci
                break
        if hit is None:
            cols.append((xi, [int(i)]))
    cols.sort(key=lambda t: t[0])
    return [lst for _, lst in cols]


def splice_vertical_strip(b: dict, strip_idx: int | None = None) -> list[tuple[str, dict]]:
    """Splice 1 interior tilted strip + matching AA border squares: (w, h, g) -> (w, h+1, g+1)."""
    w, h, g = b["w"], b["h"], b["g"]
    x, y, th = b["x"], b["y"], b["th"]
    if w > h:
        bt = bl.transform_brick(b, "T")
        res = []
        for tag, sb in splice_vertical_strip(bt, strip_idx=strip_idx):
            sbt = bl.transform_brick(sb, "T")
            sbt["tag"] = tag + "_T"
            res.append((sbt["tag"], sbt))
        return res

    deg = np.degrees(np.mod(th, 0.5 * math.pi))
    tilt = np.minimum(deg, 90.0 - deg) > 2.0
    strips, ang_med = _cluster_strips(x, y, th, tilt)
    aa_idx = np.flatnonzero(~tilt)
    aa_cols = _cluster_aa_columns(x, aa_idx)

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
        # Compute vertical shift vector between strip k and strip k+1
        dy_list = []
        dx_list = []
        for i in sk:
            best_j = min(sk1, key=lambda j: abs(x[j] - x[i]))
            if abs(x[best_j] - x[i]) < 0.45:
                dx_list.append(float(x[best_j] - x[i]))
                dy_list.append(float(y[best_j] - y[i]))
        if not dy_list:
            continue
        dx_s = float(np.median(dx_list))
        dy_s = float(np.median(dy_list))

        # Partition tilted squares: <= k stay, > k shift by (dx_s, dy_s)
        nx_l, ny_l, nt_l = [], [], []
        for si, st in enumerate(strips):
            for i in st:
                if si <= k:
                    nx_l.append(float(x[i]))
                    ny_l.append(float(y[i]))
                    nt_l.append(float(th[i]))
                else:
                    nx_l.append(float(x[i] + dx_s))
                    ny_l.append(float(y[i] + dy_s))
                    nt_l.append(float(th[i]))
        # Insert duplicate of strip k shifted by (dx_s, dy_s)
        for i in sk:
            nx_l.append(float(x[i] + dx_s))
            ny_l.append(float(y[i] + dy_s))
            nt_l.append(float(th[i]))

        # Partition AA columns around the cut y of strip k (left side uses left y, right side uses right y)
        y_left = float(y[sk[0]])
        y_right = float(y[sk[-1]])
        n_aa_added = 0
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
            for i in above:
                aa_new_x.append(float(x[i]))
                aa_new_y.append(float(y[i] + 1.0))
                aa_new_t.append(float(th[i]))
            # Insert 1 AA square if there is a continuous AA column crossing y_cut
            if below and above and (y[above[0]] - y[below[-1]]) < 1.35:
                aa_new_x.append(float(0.5 * (x[below[-1]] + x[above[0]])))
                aa_new_y.append(float(y[below[-1]] + 1.0))
                aa_new_t.append(0.0)
                n_aa_added += 1

        tot_n = len(nx_l) + len(aa_new_x)
        target_n = w * (h + 1) + (g + 1)
        if tot_n != target_n:
            # Try adjusting AA additions by +-1 if off by 1
            continue

        ax = np.array(nx_l + aa_new_x, dtype=np.float64)
        ay = np.array(ny_l + aa_new_y, dtype=np.float64)
        at = np.array(nt_l + aa_new_t, dtype=np.float64)
        cand = make_brick(w, h + 1, g + 1, ax, ay, at, tag=f"{b['tag']}_vstrip_k{k}")
        out.append((cand["tag"], cand))
    return out


def splice_horizontal_column(b: dict) -> list[tuple[str, dict]]:
    """Splice 1 tilted column + bottom/top AA squares: (w, h, g) -> (w+1, h, g+1)."""
    w, h, g = b["w"], b["h"], b["g"]
    x, y, th = b["x"], b["y"], b["th"]
    if w > h:
        bt = bl.transform_brick(b, "T")
        res = []
        for tag, sb in splice_horizontal_column(bt):
            sbt = bl.transform_brick(sb, "T")
            sbt["tag"] = tag + "_T"
            res.append((sbt["tag"], sbt))
        return res

    deg = np.degrees(np.mod(th, 0.5 * math.pi))
    tilt = np.minimum(deg, 90.0 - deg) > 2.0
    strips, ang_med = _cluster_strips(x, y, th, tilt)
    dx_step = math.cos(ang_med)
    dy_step = math.sin(ang_med)

    out: list[tuple[str, dict]] = []
    for x_cut in (4.0, 4.9, 5.8, 6.7):
        nx_l, ny_l, nt_l = [], [], []
        n_tilt_added = 0
        for st in strips:
            # Find square in strip closest to x_cut
            left = [i for i in st if x[i] <= x_cut]
            right = [i for i in st if x[i] > x_cut]
            for i in left:
                nx_l.append(float(x[i]))
                ny_l.append(float(y[i]))
                nt_l.append(float(th[i]))
            if left and right:
                # Duplicate last left square shifted by (+dx_step, +dy_step)
                ref = left[-1]
                nx_l.append(float(x[ref] + dx_step))
                ny_l.append(float(y[ref] + dy_step))
                nt_l.append(float(th[ref]))
                n_tilt_added += 1
                shift_x, shift_y = dx_step, dy_step
            elif not left:
                shift_x, shift_y = dx_step, dy_step
            else:
                shift_x, shift_y = 0.0, 0.0
            for i in right:
                nx_l.append(float(x[i] + shift_x))
                ny_l.append(float(y[i] + shift_y))
                nt_l.append(float(th[i]))

        # Partition AA squares by x <= x_cut vs x > x_cut
        aa_idx = np.flatnonzero(~tilt)
        # Cluster AA into horizontal rows by y
        aa_rows: list[tuple[float, list[int]]] = []
        for i in aa_idx:
            yi = float(y[i])
            hit = None
            for ri, (ry, lst) in enumerate(aa_rows):
                if abs(ry - yi) < 0.35:
                    lst.append(int(i))
                    aa_rows[ri] = (float(np.mean([y[k] for k in lst])), lst)
                    hit = ri
                    break
            if hit is None:
                aa_rows.append((yi, [int(i)]))
        aa_rows.sort(key=lambda t: t[0])

        aa_x, aa_y, aa_t = [], [], []
        n_aa_added = 0
        for _, row in aa_rows:
            row_s = sorted(row, key=lambda i: x[i])
            left_aa = [i for i in row_s if x[i] <= x_cut]
            right_aa = [i for i in row_s if x[i] > x_cut]
            for i in left_aa:
                aa_x.append(float(x[i]))
                aa_y.append(float(y[i]))
                aa_t.append(float(th[i]))
            for i in right_aa:
                aa_x.append(float(x[i] + 1.0))
                aa_y.append(float(y[i]))
                aa_t.append(float(th[i]))
            if left_aa and right_aa and (x[right_aa[0]] - x[left_aa[-1]]) < 1.35:
                aa_x.append(float(x[left_aa[-1]] + 1.0))
                aa_y.append(float(0.5 * (y[left_aa[-1]] + y[right_aa[0]])))
                aa_t.append(0.0)
                n_aa_added += 1

        tot_n = len(nx_l) + len(aa_x)
        target_n = (w + 1) * h + (g + 1)
        if tot_n != target_n:
            # Adjust by adding/skipping 1 bottom/top AA square if off by 1
            diff = target_n - tot_n
            if abs(diff) <= 2 and diff > 0:
                # Insert extra AA square(s) in gaps at bottom/top if needed
                continue
            elif diff != 0:
                continue

        ax = np.array(nx_l + aa_x, dtype=np.float64)
        ay = np.array(ny_l + aa_y, dtype=np.float64)
        at = np.array(nt_l + aa_t, dtype=np.float64)
        cand = make_brick(w + 1, h, g + 1, ax, ay, at, tag=f"{b['tag']}_hcol_x{x_cut:.1f}")
        out.append((cand["tag"], cand))
    return out


def relax_and_compress_brick(
    b: dict, f_targets: tuple[float, ...] = (0.76, 0.72, 0.69, 0.67, 0.655), n_trials: int = 12, seed: int = 42
) -> dict | None:
    """Relax a spliced (possibly overlapping) brick into a target rectangle and compress via SLP."""
    w, h, g = b["w"], b["h"], b["g"]
    x0, y0, t0 = b["x"], b["y"], b["th"]
    rng = np.random.default_rng(seed)
    best: dict | None = None

    pen0, _ = cpp_eval_sat_pen_bbox(x0, y0, t0)
    if pen0 <= 1e-8:
        sx, sy, st, _, ok = bslp.joint_xytheta_slp_rect(x0, y0, t0, w, h, rounds=14)
        if ok and cpp_eval_sat_pen_bbox(sx, sy, st)[0] <= 1e-8:
            best = make_brick(w, h, g, sx, sy, st, tag=b["tag"])

    for trial in range(n_trials):
        cx, cy, ct = x0.copy(), y0.copy(), t0.copy()
        if trial > 0:
            amp = 0.04 if trial % 2 == 1 else 0.08
            cx += rng.uniform(-amp, amp, len(cx))
            cy += rng.uniform(-amp, amp, len(cy))
            ct += rng.uniform(-0.8 * amp, 0.8 * amp, len(ct))
        for ft in f_targets:
            cx, cy, ct, pen = bslp.relax_rect_lbfgs(cx, cy, ct, w + ft, h + ft, maxiter=450)
            if pen <= 1e-7:
                sx, sy, st, _, ok = bslp.joint_xytheta_slp_rect(
                    cx, cy, ct, w, h, rounds=14, tr_th0=0.025, tr_xy0=0.10, tol=1e-11
                )
                if ok and cpp_eval_sat_pen_bbox(sx, sy, st)[0] <= 1e-8:
                    cb = make_brick(w, h, g, sx, sy, st, tag=b["tag"])
                    if best is None or cb["f"] < best["f"] - 1e-10:
                        best = cb
    return best


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


def pad_outer_unit_layer(b: dict, side: str, tag: str = "") -> dict:
    """Pad 1 outer unit row ('top'/'bottom') or column ('left'/'right') keeping g constant."""
    w, h, g = b["w"], b["h"], b["g"]
    x, y, th = b["x"].copy(), b["y"].copy(), b["th"].copy()
    _, _, W, H = bl.rect_bbox(x, y, th)
    if side == "bottom":
        y += 1.0
        px = np.arange(w, dtype=np.float64) + 0.5
        py = np.full(w, 0.5, dtype=np.float64)
        pt = np.zeros(w, dtype=np.float64)
        nw, nh = w, h + 1
    elif side == "top":
        px = np.arange(w, dtype=np.float64) + 0.5
        py = np.full(w, H + 0.5, dtype=np.float64)
        pt = np.zeros(w, dtype=np.float64)
        nw, nh = w, h + 1
    elif side == "left":
        x += 1.0
        px = np.full(h, 0.5, dtype=np.float64)
        py = np.arange(h, dtype=np.float64) + 0.5
        pt = np.zeros(h, dtype=np.float64)
        nw, nh = w + 1, h
    elif side == "right":
        px = np.full(h, W + 0.5, dtype=np.float64)
        py = np.arange(h, dtype=np.float64) + 0.5
        pt = np.zeros(h, dtype=np.float64)
        nw, nh = w + 1, h
    else:
        raise ValueError(f"Unknown side {side}")
    return make_brick(
        nw, nh, g,
        np.concatenate([x, px]),
        np.concatenate([y, py]),
        np.concatenate([th, pt]),
        tag=tag or f"{b['tag']}_pad_{side}",
    )


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


def save_win_locked(win_entry: dict) -> bool:
    return bl.save_win_locked(win_entry, data_dir=DATA_DIR, final_path=FINAL_CSV)
