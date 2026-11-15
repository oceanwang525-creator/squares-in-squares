#!/usr/bin/env python3
"""Hypothesis H97 (stepdown): Same-m equal-f plateau step-down utilities.

Combines:
1. Portfolio loader (origin/main baseline + optional PR donors) & dual certification.
2. Direct (m, r+1) -> (m, r) stress-chain deletion + walldrift shear + global C++ thermal melt
   + joint_xytheta_slp + soft-mode mechwalk + lpslack polish.
3. Sub-brick extraction, g -> g-1 step-down re-packing in (w+f)x(h+f), and 2-brick pinwheel assembly.
"""
from __future__ import annotations

import csv
import fcntl
import importlib.util
import io
import json
import math
import os
import subprocess
import sys
import warnings
from decimal import Decimal
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
HOMO_POC = REPO / "poc" / "swarm_homology" / "poc"
WALL_POC = REPO / "poc" / "swarm_walldrift" / "poc"
UNION_POC = REPO / "poc" / "auto_union" / "poc"
MECH_POC = REPO / "poc" / "swarm_mechwalk" / "poc"
LPSLACK_POC = REPO / "poc" / "swarm_lpslack" / "poc"
B5_POC = REPO / "poc" / "swarm_b5xplant" / "poc"
B8_POC = REPO / "poc" / "swarm_b8xplant" / "poc"
BASELINE_SRC = REPO / "poc" / "baseline" / "poc" / "src"

for _p in (
    str(HOMO_POC),
    str(UNION_POC),
    str(MECH_POC),
    str(LPSLACK_POC),
    str(B5_POC),
    str(B8_POC),
    str(BASELINE_SRC),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import brick_rect_lib as brl  # noqa: E402
import lpslack  # noqa: E402
import mechwalk_lib as mwl  # noqa: E402
import union as auto_union  # noqa: E402
from joint_slp import joint_xytheta_slp  # noqa: E402


def _load_walldrift_slp_utils():
    spec = importlib.util.spec_from_file_location("walldrift_slp_utils", WALL_POC / "slp_utils.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


su = _load_walldrift_slp_utils()

STEPDOWN_DIR = REPO / "poc" / "swarm_stepdown"
DATA_DIR = STEPDOWN_DIR / "data"
DOCS_DIR = STEPDOWN_DIR / "docs"
FINAL_CSV = STEPDOWN_DIR / "final.csv"
MERGED_CSV = REPO / "poc" / "pell_corrugation" / "data" / "border-scan" / "merged.csv"

PAIRS_ALL: list[tuple[int, int]] = [
    (367, 368),
    (492, 493),
    (585, 586),
    (591, 592),
    (295, 296),
    (495, 496),
    (264, 265),
    (740, 741),
    (743, 744),
    (795, 796),
    (912, 913),
    (916, 917),
]

GATE_PAIRS: list[tuple[int, int]] = [
    (367, 368),
    (492, 493),
    (585, 586),
    (591, 592),
    (740, 741),
    (795, 796),
]


def hx_of(th: np.ndarray) -> np.ndarray:
    return 0.5 * (np.abs(np.cos(th)) + np.abs(np.sin(th)))


def norm_pack(
    x: np.ndarray, y: np.ndarray, th: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    x = np.asarray(x, dtype=np.float64).copy()
    y = np.asarray(y, dtype=np.float64).copy()
    th = np.asarray(th, dtype=np.float64).copy()
    h = hx_of(th)
    x -= float((x - h).min())
    y -= float((y - h).min())
    L = float(max((x + h).max(), (y + h).max()))
    return x, y, th, L


def _strip_s(val: str) -> str:
    return val[1:] if val.startswith("s") else val


def _fast_side_from_rows(rows: list[dict[str, str]]) -> float:
    x = np.array([float(_strip_s(str(r["x"]))) for r in rows], dtype=np.float64)
    y = np.array([float(_strip_s(str(r["y"]))) for r in rows], dtype=np.float64)
    ang = np.array([float(_strip_s(str(r["angle"]))) for r in rows], dtype=np.float64)
    h = hx_of(ang)
    return float(max((x + h).max() - (x - h).min(), (y + h).max() - (y - h).min()))


def _extract_n_lines(text: str, need: set[int]) -> str | None:
    lines = text.splitlines()
    if len(lines) <= 1 or not lines[0].startswith("id,x,y,deg"):
        return None
    prefixes = tuple(f"{n:03d}_" if n < 1000 else "1000_" for n in need)
    sub = [lines[0]] + [ln for ln in lines[1:] if ln.startswith(prefixes)]
    if len(sub) <= 1:
        return None
    return "\n".join(sub) + "\n"


def _rows_to_arrays(
    rows: list[dict[str, str]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    srows = sorted(rows, key=lambda r: int(r["square_id"]))
    x = np.array([float(_strip_s(str(r["x"]))) for r in srows], dtype=np.float64)
    y = np.array([float(_strip_s(str(r["y"]))) for r in srows], dtype=np.float64)
    th = np.array([float(_strip_s(str(r["angle"]))) for r in srows], dtype=np.float64)
    return norm_pack(x, y, th)


def load_portfolio(
    target_ns: set[int] | list[int], include_extra_donors: bool = True
) -> dict[str, dict[int, dict]]:
    """Load exact origin/main incumbent targets and (optionally) fresher PR donors."""
    need = {int(n) for n in target_ns}
    cands: dict[int, list[tuple[float, str, list[dict[str, str]]]]] = {n: [] for n in need}

    with MERGED_CSV.open("r", newline="", encoding="utf-8") as f:
        tmp: dict[int, list[dict[str, str]]] = {}
        for r in csv.DictReader(f):
            n = int(r["n"])
            if n in need:
                tmp.setdefault(n, []).append(
                    {
                        "square_id": r["square_id"],
                        "x": _strip_s(r["x"]),
                        "y": _strip_s(r["y"]),
                        "angle": _strip_s(r["angle"]),
                    }
                )
        for n, rows in tmp.items():
            cands[n].append((_fast_side_from_rows(rows), "merged.csv", rows))

    for p in sorted((REPO / "poc").glob("*/final.csv")):
        if p == FINAL_CSV:
            continue
        try:
            sub_txt = _extract_n_lines(p.read_text(encoding="utf-8", errors="ignore"), need)
        except Exception:
            continue
        if sub_txt is None:
            continue
        try:
            lit = auto_union.parse_text(sub_txt)
        except Exception:
            continue
        rel = str(p.relative_to(REPO))
        for n, rows in lit.items():
            if n in need and len(rows) == n:
                cands[n].append((_fast_side_from_rows(rows), rel, rows))

    targets: dict[int, dict] = {}
    for n in sorted(need):
        ranked = sorted(cands[n], key=lambda t: t[0])
        best_L: Decimal | None = None
        best_entry: dict | None = None
        for _, src, rows in ranked[:4]:
            text = auto_union.final_text({n: rows})
            lit = auto_union.parse_text(text)
            pd_t = auto_union.parse_text(pd.read_csv(io.StringIO(text)).to_csv(index=False))
            res = auto_union.check((n, lit[n], pd_t[n]))
            if res is None:
                continue
            L_dec = max(Decimal(res[0]), Decimal(res[1]))
            if best_L is None or L_dec < best_L:
                best_L = L_dec
                x, y, th, L_f = _rows_to_arrays(lit[n])
                best_entry = {
                    "x": x,
                    "y": y,
                    "th": th,
                    "L": L_f,
                    "L_dec": L_dec,
                    "source": src,
                }
        if best_entry is None:
            raise RuntimeError(f"No certified incumbent found for n={n}")
        targets[n] = best_entry

    donors: dict[int, dict] = {n: dict(targets[n]) for n in targets}
    if include_extra_donors:
        extra_refs = [
            "origin/swarm-lpslack-pr3:poc/swarm_lpslack/final.csv",
            "origin/swarm-homocasc2-pr1:poc/swarm_homocasc2/final.csv",
        ]
        for ref in extra_refs:
            try:
                raw = subprocess.check_output(["git", "-C", str(REPO), "show", ref], text=True)
                sub_txt = _extract_n_lines(raw, need)
                if sub_txt is None:
                    continue
                lit = auto_union.parse_text(sub_txt)
                pd_t = auto_union.parse_text(pd.read_csv(io.StringIO(sub_txt)).to_csv(index=False))
                for n, rows in lit.items():
                    if n in need and len(rows) == n and n in pd_t and len(pd_t[n]) == n:
                        res = auto_union.check((n, rows, pd_t[n]))
                        if res is None:
                            continue
                        L_dec = max(Decimal(res[0]), Decimal(res[1]))
                        if n not in donors or L_dec < donors[n]["L_dec"]:
                            x, y, th, L_f = _rows_to_arrays(rows)
                            donors[n] = {
                                "x": x,
                                "y": y,
                                "th": th,
                                "L": L_f,
                                "L_dec": L_dec,
                                "source": ref.split(":")[0],
                            }
            except Exception as exc:
                warnings.warn(f"Could not load donor ref {ref}: {exc}")

    return {"targets": targets, "donors": donors}


def certify_rows_against_frontier(
    n: int, rows: list[dict[str, str]], L_inc_dec: Decimal
) -> dict | None:
    """Certify rows in literal, pandas final.csv transport, and core.kaggle_roundtrip."""
    text = auto_union.final_text({n: rows})
    lit = auto_union.parse_text(text)
    pd_t = auto_union.parse_text(pd.read_csv(io.StringIO(text)).to_csv(index=False))
    res = auto_union.check((n, lit[n], pd_t[n]))
    if res is None:
        return None
    kr = lpslack.certify_pair(n, rows)
    if kr is None:
        return None
    L_lit = Decimal(res[0])
    L_pd = Decimal(res[1])
    L_kr = max(kr[0], kr[1])
    L_worst = max(L_lit, L_pd, L_kr)
    dL = L_worst - L_inc_dec
    return {
        "ok": True,
        "n": n,
        "L_lit": str(L_lit),
        "L_pd": str(L_pd),
        "L_kr": str(L_kr),
        "L_float": float(L_worst),
        "dL": float(dL),
        "is_real_win": bool(dL <= Decimal("-1e-9")),
        "rows": rows,
    }


def certify_pack_against_frontier(
    n: int, x: np.ndarray, y: np.ndarray, th: np.ndarray, L_inc_dec: Decimal
) -> dict | None:
    """Normalize arrays, try separating-axis shrink, and dual-certify."""
    hx = hx_of(th)
    sx = x - float((x - hx).min()) + 1e-12
    sy = y - float((y - hx).min()) + 1e-12
    centers = np.column_stack([sx, sy])

    best_cert: dict | None = None
    shrunk = lpslack.try_shrink_and_certify(n, centers, th, Decimal("999.0"))
    if shrunk is not None:
        _, _, srows = shrunk
        c_shrunk = certify_rows_against_frontier(n, srows, L_inc_dec)
        if c_shrunk is not None:
            best_cert = c_shrunk

    raw_rows = [
        {
            "square_id": str(i),
            "x": repr(float(sx[i])),
            "y": repr(float(sy[i])),
            "angle": repr(float(th[i])),
        }
        for i in range(n)
    ]
    c_raw = certify_rows_against_frontier(n, raw_rows, L_inc_dec)
    if c_raw is not None:
        if best_cert is None or Decimal(c_raw["L_lit"]) < Decimal(best_cert["L_lit"]):
            best_cert = c_raw
    return best_cert


def save_win(n: int, cert: dict, tag: str) -> bool:
    """Save certified win atomically to data/win_n{n}.json (only if strictly better)."""
    if not cert or not cert.get("is_real_win"):
        return False
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_p = DATA_DIR / f"win_n{n}.json"
    lock_p = DATA_DIR / ".win.lock"
    with lock_p.open("w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        if out_p.exists():
            old = json.loads(out_p.read_text())
            if Decimal(cert["L_lit"]) >= Decimal(old["L_lit"]):
                return False
        payload = {
            "n": n,
            "tag": tag,
            "L_lit": cert["L_lit"],
            "L_pd": cert["L_pd"],
            "L_kr": cert["L_kr"],
            "L_float": cert["L_float"],
            "dL": cert["dL"],
            "rows": cert["rows"],
        }
        auto_union.atomic(out_p, json.dumps(payload).encode())
        return True


def rebuild_final_csv() -> list[int]:
    """Rebuild poc/swarm_stepdown/final.csv from all win_n*.json (or header-only if none)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    groups: dict[int, list[dict[str, str]]] = {}
    for p in sorted(DATA_DIR.glob("win_n*.json")):
        d = json.loads(p.read_text())
        groups[int(d["n"])] = d["rows"]
    text = auto_union.final_text(groups)
    auto_union.atomic(FINAL_CSV, text.encode())
    os.chmod(FINAL_CSV, 0o644)
    if groups:
        lit, pd_t = auto_union.load(FINAL_CSV)
        for n in groups:
            if auto_union.check((n, lit[n], pd_t[n])) is None:
                raise ValueError(f"Exported final.csv failed dual certification for n={n}")
    return sorted(groups)


def slp_and_polish(
    n: int,
    x: np.ndarray,
    y: np.ndarray,
    th: np.ndarray,
    run_mechwalk: bool = True,
    run_lpslack: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, bool]:
    """Run multi-stage joint_xytheta_slp + fixed-angle LP + optional soft-mode mechwalk + lpslack."""
    cx, cy, cth = norm_pack(x, y, th)[:3]
    pen0, L0 = su.sat_max_overlap_and_bbox(cx, cy, cth)
    best_x, best_y, best_th, best_L = cx.copy(), cy.copy(), cth.copy(), L0
    best_ok = pen0 <= 1e-9

    # Stage 1: multi-stage joint_xytheta_slp
    for rounds, tr_th, tr_xy, tol in [
        (12, 0.025, 0.10, 1e-8),
        (10, 0.006, 0.028, 1e-10),
        (8, 0.0012, 0.006, 1e-11),
        (6, 0.00025, 0.0015, 1e-12),
    ]:
        xs, ys, ts, Ls, ok = joint_xytheta_slp(
            cx, cy, cth, rounds=rounds, tr_th0=tr_th, tr_xy0=tr_xy, margin=1e-8, tol=tol, patience=2
        )
        if ok:
            cx, cy, cth = xs, ys, ts
            if not best_ok or Ls < best_L:
                best_x, best_y, best_th, best_L, best_ok = xs.copy(), ys.copy(), ts.copy(), Ls, True

    # Stage 2: fixed-angle LP tighten
    xf, yf, Lf, okf = su.fixed_angle_lp_tighten(best_x, best_y, best_th, rounds=8, margin=1e-8)
    if okf and (not best_ok or Lf < best_L):
        best_x, best_y, best_L, best_ok = xf, yf, Lf, True

    if not best_ok:
        return best_x, best_y, best_th, best_L, False

    # Stage 3: light soft-mode mechwalk (k=10 modes, 2 step scales)
    if run_mechwalk:
        try:
            jac, _ = mwl.contact_jacobian(best_x, best_y, best_th, best_L)
            if jac.shape[0] > 0:
                sigs, vecs = mwl.soft_modes(jac, k=min(12, 3 * n - 2))
                diverse = mwl.select_diverse_modes(
                    vecs, sigs, best_x, best_y, max_keep=6, cos_thresh=0.88
                )
                for _, _, v in diverse[:4]:
                    vmax = float(np.max(np.abs(v)))
                    if vmax < 1e-12:
                        continue
                    v_norm = v / vmax
                    for step in (-0.04, 0.04, -0.10, 0.10):
                        px = best_x + step * v_norm[:n]
                        py = best_y + step * v_norm[n : 2 * n]
                        pt = best_th + step * v_norm[2 * n : 3 * n]
                        # Quick re-jam in C++ then SLP
                        pen_p, L_p = su.sat_max_overlap_and_bbox(px, py, pt)
                        if pen_p > 1e-6:
                            px, py, pt, pen_p, L_p = su.cpp_lbfgs_relax(
                                px, py, pt, min(L_p, best_L + 0.002), full_iters=140, seed=17
                            )
                        xs, ys, ts, Ls, ok = joint_xytheta_slp(
                            px, py, pt, rounds=8, tr_th0=0.008, tr_xy0=0.03, margin=1e-8, tol=1e-11
                        )
                        if ok and Ls < best_L - 1e-9:
                            best_x, best_y, best_th, best_L = xs, ys, ts, Ls
        except Exception:
            pass

    # Stage 4: lpslack selective angle release + exact shrink
    if run_lpslack:
        try:
            rows_in = lpslack.rows_from_arrays(np.column_stack([best_x, best_y]), best_th)
            res = lpslack.optimize_group(n, rows_in, Decimal(f"{best_L + 1e-4:.17g}"))
            if res is not None and res.get("ok"):
                rx, ry, rt, rL = _rows_to_arrays(res["rows"])
                pen_r, _ = su.sat_max_overlap_and_bbox(rx, ry, rt)
                if pen_r <= 1e-9 and rL < best_L:
                    best_x, best_y, best_th, best_L = rx, ry, rt, rL
        except Exception:
            pass

    return best_x, best_y, best_th, best_L, True


def stepdown_brick(
    b: dict,
    n_del: int = 1,
    n_trials: int = 16,
    target_f_floor: float | None = None,
    rng_seed: int = 2026,
) -> dict | None:
    """Delete n_del tilted/boundary square(s) from sub-brick b (g -> g - n_del) and re-pack in (w+f)x(h+f)."""
    w, h, g0 = int(b["w"]), int(b["h"]), int(b["g"])
    g_new = g0 - n_del
    if g_new < 1:
        return None
    x0 = np.asarray(b["x"], dtype=np.float64)
    y0 = np.asarray(b["y"], dtype=np.float64)
    th0 = np.asarray(b["th"], dtype=np.float64)
    nx0, ny0, W0, H0 = brl.rect_bbox(x0, y0, th0)
    f0 = max(W0 - w, H0 - h)

    deg = np.degrees(th0) % 90.0
    tilted_idx = np.where((deg > 1.5) & (deg < 88.5))[0]
    if len(tilted_idx) < n_del:
        tilted_idx = np.arange(len(x0))

    # Candidate deletions: tilted squares near brick corners, center, or edges
    rng = np.random.default_rng(rng_seed)
    hx0 = hx_of(th0)
    edge_dist = np.minimum.reduce([nx0 - hx0, W0 - (nx0 + hx0), ny0 - hx0, H0 - (ny0 + hx0)])
    # Sort tilted squares by proximity to corner / center / edge
    c_dist = (nx0 - 0.5 * W0) ** 2 + (ny0 - 0.5 * H0) ** 2
    del_cands = []
    if n_del == 1:
        order_edge = tilted_idx[np.argsort(edge_dist[tilted_idx])]
        order_ctr = tilted_idx[np.argsort(c_dist[tilted_idx])]
        picks = list(order_edge[:4]) + list(order_ctr[:4])
        while len(picks) < n_trials:
            picks.append(int(rng.choice(tilted_idx)))
        del_cands = [[int(p)] for p in picks[:n_trials]]
    else:
        for t in range(n_trials):
            picks = rng.choice(tilted_idx, size=n_del, replace=False)
            del_cands.append([int(p) for p in picks])

    f_floor = target_f_floor if target_f_floor is not None else max(0.55, f0 - 0.08)
    best_brick: dict | None = None

    for t_idx, drops in enumerate(del_cands):
        keep = np.ones(len(x0), dtype=bool)
        keep[drops] = False
        xk, yk, tk = nx0[keep].copy(), ny0[keep].copy(), th0[keep].copy()

        # Global shake around the hole + expand slightly to melt
        hole_x = float(np.mean(nx0[drops]))
        hole_y = float(np.mean(ny0[drops]))
        f_exp = f0 + float(rng.uniform(0.01, 0.04))
        xk *= (w + f_exp) / W0
        yk *= (h + f_exp) / H0
        dist_hole = np.hypot(xk - hole_x, yk - hole_y)
        rad = float(rng.uniform(3.5, max(5.5, 0.65 * max(W0, H0))))
        near = dist_hole <= rad
        # Directed pull toward hole inside pocket
        if np.any(near):
            pull = float(rng.uniform(0.15, 0.40))
            dxh = hole_x - xk[near]
            dyh = hole_y - yk[near]
            nrm = np.maximum(1.0, np.hypot(dxh, dyh))
            xk[near] += pull * dxh / nrm + rng.uniform(-0.12, 0.12, int(near.sum()))
            yk[near] += pull * dyh / nrm + rng.uniform(-0.12, 0.12, int(near.sum()))
            tk[near] += rng.uniform(-0.22, 0.22, int(near.sum()))

        # Descending L-BFGS-B compression in (w+ft) x (h+ft)
        for ft in [f0 + 0.01, max(f_floor + 0.02, f0 - 0.015), max(f_floor, f0 - 0.040)]:
            xk, yk, tk, E = brl.relax_rect_lbfgs(xk, yk, tk, w + ft, h + ft, maxiter=350)
            if E > 5e-4:
                break

        pen, _ = su.sat_max_overlap_and_bbox(xk, yk, tk)
        if pen > 0.03:
            continue

        xs, ys, ts, fs, ok = brl.joint_xytheta_slp_rect(
            xk, yk, tk, w, h, rounds=12, tr_th0=0.022, tr_xy0=0.09, tol=1e-10
        )
        if not ok:
            continue
        xs2, ys2, ts2, fs2, ok2 = brl.joint_xytheta_slp_rect(
            xs, ys, ts, w, h, rounds=10, tr_th0=0.004, tr_xy0=0.018, tol=1e-12
        )
        if ok2 and fs2 < fs:
            xs, ys, ts, fs = xs2, ys2, ts2, fs2

        xs, ys, W_b, H_b = brl.rect_bbox(xs, ys, ts)
        fb = max(W_b - w, H_b - h)
        if best_brick is None or fb < best_brick["f"]:
            best_brick = {
                "w": w,
                "h": h,
                "g": g_new,
                "fx": W_b - w,
                "fy": H_b - h,
                "f": fb,
                "x": xs,
                "y": ys,
                "th": ts,
                "src_n": b.get("src_n", 0),
                "tag": f"{b.get('tag', 'brick')}_stepg{g_new}",
            }

    return best_brick


def make_sym_brick(b: dict, mode: str) -> dict:
    """Apply D4 symmetry transform {'id','T','flipx','flipy','rot90','rot180','rot270'} to a brick dict."""
    x = np.asarray(b["x"], dtype=np.float64).copy()
    y = np.asarray(b["y"], dtype=np.float64).copy()
    th = np.asarray(b["th"], dtype=np.float64).copy()
    w, h = int(b["w"]), int(b["h"])
    x, y, W, H = brl.rect_bbox(x, y, th)
    if mode == "id":
        nx, ny, nt, nw, nh = x, y, th, w, h
    elif mode == "T":
        nx, ny, nt, nw, nh = y, x, 0.5 * math.pi - th, h, w
    elif mode == "flipx":
        nx, ny, nt, nw, nh = W - x, y, -th, w, h
    elif mode == "flipy":
        nx, ny, nt, nw, nh = x, H - y, -th, w, h
    elif mode == "rot180":
        nx, ny, nt, nw, nh = W - x, H - y, th, w, h
    elif mode == "rot90":
        nx, ny, nt, nw, nh = H - y, x, th + 0.5 * math.pi, h, w
    elif mode == "rot270":
        nx, ny, nt, nw, nh = y, W - x, th - 0.5 * math.pi, h, w
    else:
        raise ValueError(mode)
    nx, ny, W2, H2 = brl.rect_bbox(nx, ny, nt)
    return {
        "w": nw,
        "h": nh,
        "g": int(b["g"]),
        "fx": W2 - nw,
        "fy": H2 - nh,
        "f": max(W2 - nw, H2 - nh),
        "x": nx,
        "y": ny,
        "th": nt,
        "src_n": b.get("src_n", 0),
        "tag": b.get("tag", "brick"),
    }


def assemble_zero_overlap_pinwheel(
    m: int, r: int, b1: dict, b2: dict, shift_fracs: tuple[float, ...] = (0.0, 0.5)
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, float, str]]:
    """Assemble b1 (TL) and b2 (BR) with zero-overlap (w1+w2<=m and h1+h2<=m)."""
    if b1["g"] + b2["g"] != r:
        return []
    if b1["w"] + b2["w"] > m or b1["h"] + b2["h"] > m:
        return []
    out = []
    modes = ("id", "flipx", "flipy", "rot180")
    for m1 in modes:
        sb1 = make_sym_brick(b1, m1)
        for m2 in modes:
            sb2 = make_sym_brick(b2, m2)
            w1, h1 = sb1["w"], sb1["h"]
            w2, h2 = sb2["w"], sb2["h"]
            f = max(sb1["f"], sb2["f"])
            L = float(m) + f
            for sf in shift_fracs:
                dx1, dy1 = 0.0, L - (h1 + sb1["fy"])
                dx2, dy2 = L - (w2 + sb2["fx"]), 0.0
                x1, y1, t1 = sb1["x"] + dx1, sb1["y"] + dy1, sb1["th"].copy()
                x2, y2, t2 = sb2["x"] + dx2, sb2["y"] + dy2, sb2["th"].copy()

                gap_y = max(0.0, f - sb1["fy"])
                gap_x = max(0.0, f - sb1["fx"])
                delta_y = sf * gap_y
                delta_x = sf * gap_x

                gx, gy = [], []
                for i in range(m):
                    for j in range(m):
                        if i < w1 and j >= m - h1:
                            continue
                        if i >= m - w2 and j < h2:
                            continue
                        if i < m - w2 and j < m - h1:
                            gx.append(i + 0.5)
                            gy.append(j + 0.5 + delta_y)
                        else:
                            sx = -delta_x if i >= w1 else 0.0
                            gx.append(i + f + 0.5 + sx)
                            gy.append(j + f + 0.5)

                ax = np.concatenate([x1, x2, np.array(gx, dtype=np.float64)])
                ay = np.concatenate([y1, y2, np.array(gy, dtype=np.float64)])
                at = np.concatenate([t1, t2, np.zeros(len(gx), dtype=np.float64)])
                if len(ax) != m * m + r:
                    continue
                pen, bb = su.sat_max_overlap_and_bbox(ax, ay, at)
                out.append((pen, bb, ax, ay, at, f"{sb1['tag']}+{sb2['tag']}_{m1}_{m2}_sf{sf:.1f}"))
    out.sort(key=lambda z: (z[0], z[1]))
    # Keep top 4 distinct
    return [(z[2], z[3], z[4], z[1], z[5]) for z in out[:4]]
