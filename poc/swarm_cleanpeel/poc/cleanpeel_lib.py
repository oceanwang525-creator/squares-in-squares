#!/usr/bin/env python3
"""Shared utilities, true-frontier loader, official certifier, and peel/melt helpers for H16 (swarm_cleanpeel)."""
from __future__ import annotations

import csv
import fcntl
import glob
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

REPO = Path(__file__).resolve().parents[3]
HOMO_POC = REPO / "poc" / "swarm_homology" / "poc"
UNION_POC = REPO / "poc" / "auto_union" / "poc"
for _p in (str(HOMO_POC), str(UNION_POC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import union as auto_union  # noqa: E402
import utils as u  # noqa: E402
from cpp_wrap import cpp_eval_sat_pen_bbox, cpp_expand_sym, full_to_free_sym, run_cpp_masked_melt  # noqa: E402
from joint_slp import joint_xytheta_slp  # noqa: E402

CLEANPEEL_DIR = REPO / "poc" / "swarm_cleanpeel"
DATA_DIR = CLEANPEEL_DIR / "data"
DOCS_DIR = CLEANPEEL_DIR / "docs"
FINAL_CSV = CLEANPEEL_DIR / "final.csv"

MERGED_CSV = REPO / "poc" / "pell_corrugation" / "data" / "border-scan" / "merged.csv"
GUESTDATA_CSV = REPO / "poc" / "guestdata_atlas" / "final.csv"


def hx_of(th: np.ndarray) -> np.ndarray:
    return 0.5 * (np.abs(np.cos(th)) + np.abs(np.sin(th)))


def norm_pack(x: np.ndarray, y: np.ndarray, th: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    x = np.asarray(x, dtype=np.float64).copy()
    y = np.asarray(y, dtype=np.float64).copy()
    th = np.asarray(th, dtype=np.float64).copy()
    h = hx_of(th)
    x -= (x - h).min()
    y -= (y - h).min()
    L = float(max((x + h).max(), (y + h).max()))
    return x, y, th, L


def _strip_s(val: str) -> str:
    return val[1:] if val.startswith("s") else val


def _rows_from_merged_csv(path: Path, filter_ns: set[int] | None = None) -> dict[int, list[dict]]:
    groups: dict[int, list[dict]] = {}
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n = int(row["n"])
            if filter_ns is not None and n not in filter_ns:
                continue
            groups.setdefault(n, []).append(
                {
                    "square_id": int(row["square_id"]),
                    "x": _strip_s(row["x"]),
                    "y": _strip_s(row["y"]),
                    "angle": _strip_s(row["angle"]),  # radians
                }
            )
    return groups


def _rows_from_final_csv(path: Path, filter_ns: set[int] | None = None) -> dict[int, list[dict]]:
    """Parse id,x,y,deg CSV converting deg -> angle (radians) via mpmath."""
    import mpmath as mp

    raw: dict[int, list[tuple[int, str, str, str]]] = {}
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["id", "x", "y", "deg"]:
            return {}
        for row in reader:
            parts = row["id"].split("_")
            if len(parts) != 2:
                continue
            n = int(parts[0])
            if filter_ns is not None and n not in filter_ns:
                continue
            sid = int(parts[1])
            raw.setdefault(n, []).append((sid, _strip_s(row["x"]), _strip_s(row["y"]), _strip_s(row["deg"])))

    groups: dict[int, list[dict]] = {}
    with mp.workdps(95):
        for n, items in raw.items():
            if len(items) != n:
                continue
            rows = []
            for sid, xs, ys, degs in items:
                ang = mp.nstr(mp.mpf(degs) * mp.pi / 180, 75)
                rows.append({"square_id": sid, "x": xs, "y": ys, "angle": ang})
            groups[n] = rows
    return groups


def load_frontier(
    target_ns: set[int] | None = None,
    include_self: bool = False,
) -> tuple[dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, float]], dict[int, Decimal], dict[int, str]]:
    """
    Load best-per-n across merged.csv (angle in rad) + poc/guestdata_atlas/final.csv (deg)
    + poc/swarm_*/final.csv (deg).
    If target_ns is given, only certify and return those n's (fast!).
    """
    sources: list[Path] = []
    if MERGED_CSV.exists():
        sources.append(MERGED_CSV)
    if GUESTDATA_CSV.exists():
        sources.append(GUESTDATA_CSV)
    for p_str in sorted(glob.glob(str(REPO / "poc" / "swarm_*" / "final.csv"))):
        p = Path(p_str)
        if not include_self and p.parent.name == "swarm_cleanpeel":
            continue
        sources.append(p)

    # First pass in float64 to collect candidate (n, src, L_float) cheaply
    best_float: dict[int, tuple[float, Path, list[dict]]] = {}
    all_cands: dict[int, list[tuple[Path, list[dict]]]] = {}

    for src in sources:
        if src == MERGED_CSV:
            groups = _rows_from_merged_csv(src, filter_ns=target_ns)
        else:
            groups = _rows_from_final_csv(src, filter_ns=target_ns)
        for n, rows in groups.items():
            if len(rows) != n:
                continue
            x = np.array([float(r["x"]) for r in rows], dtype=np.float64)
            y = np.array([float(r["y"]) for r in rows], dtype=np.float64)
            th = np.array([float(r["angle"]) for r in rows], dtype=np.float64)
            _, _, _, Lf = norm_pack(x, y, th)
            all_cands.setdefault(n, []).append((src, rows))
            if n not in best_float or Lf < best_float[n][0]:
                best_float[n] = (Lf, src, rows)

    packs: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, float]] = {}
    best_L_dec: dict[int, Decimal] = {}
    best_src: dict[int, str] = {}

    for n, cands in all_cands.items():
        min_Lf = best_float[n][0]
        # Certify all candidates within 1e-6 of min_Lf so float ties pick exact best Decimal
        for src, rows in cands:
            x = np.array([float(r["x"]) for r in rows], dtype=np.float64)
            y = np.array([float(r["y"]) for r in rows], dtype=np.float64)
            th = np.array([float(r["angle"]) for r in rows], dtype=np.float64)
            xn, yn, tn, Lf = norm_pack(x, y, th)
            if Lf > min_Lf + 1e-6:
                continue
            try:
                Ld = Decimal(auto_union.certify(n, rows))
            except Exception:
                continue
            if n not in best_L_dec or Ld < best_L_dec[n]:
                best_L_dec[n] = Ld
                packs[n] = (xn, yn, tn, Lf)
                best_src[n] = str(src.relative_to(REPO))

    return packs, best_L_dec, best_src


def exact_official_L_for_pack(
    n: int, x: np.ndarray, y: np.ndarray, th: np.ndarray
) -> tuple[Decimal | None, Decimal | None, list[dict] | None]:
    """Certify using auto_union.check in BOTH literal and pandas transport modes."""
    hx = hx_of(th)
    min_x = float((x - hx).min())
    min_y = float((y - hx).min())
    sx = x if min_x >= 0.0 else (x - min_x + 1e-10)
    sy = y if min_y >= 0.0 else (y - min_y + 1e-10)
    rows = [
        {
            "square_id": i,
            "x": repr(float(sx[i])),
            "y": repr(float(sy[i])),
            "angle": repr(float(th[i])),
        }
        for i in range(n)
    ]
    text = auto_union.final_text({n: rows})
    lit = auto_union.parse_text(text)
    pd_t = auto_union.parse_text(pd.read_csv(io.StringIO(text)).to_csv(index=False))
    res = auto_union.check((n, lit[n], pd_t[n]))
    if res is None:
        return None, None, None
    return Decimal(res[0]), Decimal(res[1]), rows


def certify_pack_against_frontier(
    n: int, x: np.ndarray, y: np.ndarray, th: np.ndarray, L_inc_dec: Decimal
) -> dict | None:
    """Certify candidate against exact incumbent L_inc_dec using literal + pandas union.check."""
    L_lit, L_pd, rows = exact_official_L_for_pack(n, x, y, th)
    if L_lit is None or L_pd is None:
        return None
    L_worst = max(L_lit, L_pd)
    dL = L_worst - L_inc_dec
    return {
        "ok": True,
        "n": n,
        "L_lit": str(L_lit),
        "L_pd": str(L_pd),
        "L_float": float(L_worst),
        "dL": float(dL),
        "is_real_win": bool(dL <= Decimal("-1e-9")),
        "rows": rows,
    }


def save_win_and_rebuild_final(n: int, cert: dict, label: str) -> bool:
    """Atomically save per-n win JSON (with file lock) and rebuild poc/swarm_cleanpeel/final.csv."""
    if not cert.get("is_real_win"):
        return False
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = DATA_DIR / ".win_lock"
    with open(lock_path, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        win_path = DATA_DIR / f"win_n{n}.json"
        if win_path.exists():
            prev = json.loads(win_path.read_text())
            if Decimal(max(prev["L_lit"], prev["L_pd"])) <= Decimal(max(cert["L_lit"], cert["L_pd"])):
                return False
        payload = {
            "n": n,
            "label": label,
            "L_lit": cert["L_lit"],
            "L_pd": cert["L_pd"],
            "L_float": cert["L_float"],
            "dL": cert["dL"],
            "rows": cert["rows"],
        }
        auto_union.atomic(win_path, json.dumps(payload, indent=2).encode())
        os.chmod(win_path, 0o644)
        rebuild_final_csv()
        return True


def rebuild_final_csv() -> None:
    """Rebuild poc/swarm_cleanpeel/final.csv from all certified win_n*.json files (or header-only if none)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    groups = {}
    for p in sorted(DATA_DIR.glob("win_n*.json")):
        d = json.loads(p.read_text())
        groups[int(d["n"])] = d["rows"]
    text = auto_union.final_text(groups)
    auto_union.atomic(FINAL_CSV, text.encode())
    os.chmod(FINAL_CSV, 0o644)


def overlap_pairs(X: np.ndarray, Y: np.ndarray, T: np.ndarray, tol: float = 1e-9) -> list[tuple[float, int, int]]:
    n = len(X)
    c, s = np.cos(T), np.sin(T)
    out = []
    order = np.argsort(X)
    Xs = X[order]
    for a in range(n):
        i = int(order[a])
        b = a + 1
        while b < n and Xs[b] - X[i] < 1.5:
            j = int(order[b])
            b += 1
            dx, dy = X[j] - X[i], Y[j] - Y[i]
            if abs(dy) > 1.5:
                continue
            rad = 0.5 * (1.0 + abs(c[i] * c[j] + s[i] * s[j]) + abs(c[i] * s[j] - s[i] * c[j]))
            pen = min(
                rad - abs(dx * c[i] + dy * s[i]),
                rad - abs(-dx * s[i] + dy * c[i]),
                rad - abs(dx * c[j] + dy * s[j]),
                rad - abs(-dx * s[j] + dy * c[j]),
            )
            if pen > tol:
                out.append((float(pen), i, j))
    return out


def repair_overlaps(
    x: np.ndarray,
    y: np.ndarray,
    th: np.ndarray,
    L_box: float,
    free_mask: np.ndarray | None = None,
    maxiter: int = 240,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Clamp to [hx, L_box - hx] and relax SAT overlaps via L-BFGS-B."""
    h = hx_of(th)
    xr = np.clip(x, h + 1e-6, L_box - h - 1e-6)
    yr = np.clip(y, h + 1e-6, L_box - h - 1e-6)
    tr = th.copy()
    pen, _ = cpp_eval_sat_pen_bbox(xr, yr, tr)
    if pen <= 1e-9:
        return xr, yr, tr, pen
    xr, yr, tr, pen = u.relax_lbfgs(xr, yr, tr, L_box, free_mask=free_mask, maxiter=maxiter, w_wall=4.0)
    if pen > 1e-6 and free_mask is not None:
        xr, yr, tr, pen = u.relax_lbfgs(xr, yr, tr, L_box, free_mask=None, maxiter=maxiter, w_wall=4.0)
    return xr, yr, tr, pen


def clearance_points(
    x: np.ndarray, y: np.ndarray, th: np.ndarray, L: float, n_pts: int, step: float = 0.10, min_sep: float = 0.85
) -> list[tuple[float, float, float]]:
    g = np.arange(0.5, L - 0.5 + 1e-9, step)
    gx, gy = np.meshgrid(g, g, indexing="ij")
    px, py = gx.ravel(), gy.ravel()
    clr = np.minimum(np.minimum(px, L - px), np.minimum(py, L - py)) - 0.5
    c, s = np.cos(th), np.sin(th)
    for a in range(0, len(px), 4096):
        dx = px[a : a + 4096, None] - x[None, :]
        dy = py[a : a + 4096, None] - y[None, :]
        d = np.maximum(np.abs(c * dx + s * dy), np.abs(-s * dx + c * dy)) - 0.5
        clr[a : a + 4096] = np.minimum(clr[a : a + 4096], d.min(axis=1))
    order = np.argsort(-clr)
    out = []
    for i in order:
        if len(out) >= n_pts:
            break
        if all((px[i] - q[0]) ** 2 + (py[i] - q[1]) ** 2 > min_sep ** 2 for q in out):
            out.append((float(px[i]), float(py[i]), float(clr[i])))
    return out


def dominant_tilt(th: np.ndarray) -> float:
    deg = np.degrees(th) % 90.0
    tilted = deg[(deg > 1.0) & (deg < 89.0)]
    if len(tilted) == 0:
        return 0.0
    hist, edges = np.histogram(tilted, bins=45, range=(0.0, 90.0))
    b = int(np.argmax(hist))
    return float(np.radians(0.5 * (edges[b] + edges[b + 1])))


def polish_candidate(
    n: int,
    x: np.ndarray,
    y: np.ndarray,
    th: np.ndarray,
    L_inc_dec: Decimal,
    free_mask: np.ndarray | None = None,
) -> dict | None:
    """Relax if slight overlap, run multi-stage joint_xytheta_slp + fixed-angle LP tighten + official check."""
    L_inc = float(L_inc_dec)
    pen, bb = cpp_eval_sat_pen_bbox(x, y, th)
    if pen > 1e-7 and pen < 0.15:
        x, y, th, pen = u.relax_lbfgs(x, y, th, bb + max(0.002, 2.0 * pen), free_mask=free_mask, maxiter=260, w_wall=4.0)
        pen, bb = cpp_eval_sat_pen_bbox(x, y, th)
        if pen > 1e-6 and free_mask is not None:
            x, y, th, pen = u.relax_lbfgs(x, y, th, bb + max(0.002, 2.0 * pen), free_mask=None, maxiter=220, w_wall=4.0)
            pen, bb = cpp_eval_sat_pen_bbox(x, y, th)
    if pen > 0.02:
        return None

    xs, ys, ts, Ls, ok = joint_xytheta_slp(x, y, th, rounds=10, tr_th0=0.022, tr_xy0=0.09, margin=1e-8, tol=1e-8, patience=2)
    if not ok:
        return None
    if Ls < L_inc + 3e-3:
        xs2, ys2, ts2, Ls2, ok2 = joint_xytheta_slp(xs, ys, ts, rounds=12, tr_th0=0.007, tr_xy0=0.025, margin=1e-8, tol=1e-11, patience=3)
        if ok2 and Ls2 < Ls:
            xs, ys, ts, Ls = xs2, ys2, ts2, Ls2

    best = None
    for mlp in (1e-6, 2e-8):
        xf, yf, Lf, okf = u.fixed_angle_lp_tighten(xs, ys, ts, rounds=3, margin=mlp)
        if not okf:
            continue
        cert = certify_pack_against_frontier(n, xf, yf, ts, L_inc_dec)
        if cert is not None and (best is None or Decimal(cert["L_lit"]) < Decimal(best["L_lit"])):
            best = cert
            if mlp <= 1e-7 and cert["is_real_win"]:
                break
    if best is None:
        cert = certify_pack_against_frontier(n, xs, ys, ts, L_inc_dec)
        if cert is not None:
            best = cert
    if best is None and pen <= 1e-12:
        cert = certify_pack_against_frontier(n, x, y, th, L_inc_dec)
        if cert is not None:
            best = cert
    return best
