#!/usr/bin/env python3
"""Shared utilities, frontier loader, official literal+pandas certifier, and C++ melt+SLP polish for swarm_sametilt."""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile
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
if str(HOMO_POC) not in sys.path:
    sys.path.insert(0, str(HOMO_POC))
if str(UNION_POC) not in sys.path:
    sys.path.insert(0, str(UNION_POC))

import union as auto_union  # noqa: E402
import utils as u  # noqa: E402
from cpp_wrap import cpp_eval_sat_pen_bbox, cpp_expand_sym, full_to_free_sym, run_cpp_masked_melt  # noqa: E402
from joint_slp import joint_xytheta_slp  # noqa: E402

SAMETILT_DIR = REPO / "poc" / "swarm_sametilt"
DATA_DIR = SAMETILT_DIR / "data"
DOCS_DIR = SAMETILT_DIR / "docs"
FINAL_CSV = SAMETILT_DIR / "final.csv"

AUDIT_UNION = Path("/mnt/e/squares-pr21-audit-nov13/proof/union.csv")
MERGED_CSV = REPO / "poc" / "pell_corrugation" / "data" / "border-scan" / "merged.csv"
GUESTDATA_CSV = REPO / "poc" / "guestdata_atlas" / "final.csv"


def hx_of(th: np.ndarray) -> np.ndarray:
    return 0.5 * (np.abs(np.cos(th)) + np.abs(np.sin(th)))


def norm_pack(x: np.ndarray, y: np.ndarray, th: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    x = np.asarray(x, float).copy()
    y = np.asarray(y, float).copy()
    th = np.asarray(th, float).copy()
    h = hx_of(th)
    x -= (x - h).min()
    y -= (y - h).min()
    L = float(max((x + h).max(), (y + h).max()))
    return x, y, th, L


def load_frontier() -> tuple[dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, float]], dict[int, Decimal]]:
    """Load best-known packings and exact Decimal L across audit union (PR21), guestdata_atlas, and PR22."""
    packs: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, float]] = {}
    best_L_dec: dict[int, Decimal] = {}

    lit_json = Path("/mnt/e/squares-pr21-audit-nov13/proof/union-literal.json")
    exact_map: dict[int, Decimal] = {}
    if lit_json.exists():
        rep = json.loads(lit_json.read_text())
        for row in rep.get("results", []):
            if row.get("L") is not None:
                exact_map[int(row["n"])] = Decimal(str(row["L"]))

    src_csv = AUDIT_UNION if AUDIT_UNION.exists() else MERGED_CSV
    df = pd.read_csv(src_csv)
    for n_val, sub in df.groupby("n"):
        n = int(n_val)
        x, y, th, L = norm_pack(sub["x"].to_numpy(), sub["y"].to_numpy(), sub["angle"].to_numpy())
        packs[n] = (x, y, th, L)
        best_L_dec[n] = exact_map.get(n, Decimal(f"{L:.17g}"))

    # Also overlay PR #22 if available in git remote
    for p in ("poc/swarm_homology/final.csv", "poc/swarm_new159/final.csv"):
        try:
            txt = subprocess.check_output(
                ["git", "-C", str(REPO), "show", f"origin/swarm-homology-new159:{p}"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            a = auto_union.parse_text(txt)
            for n, rows in a.items():
                x = np.array([float(r["x"]) for r in rows], dtype=float)
                y = np.array([float(r["y"]) for r in rows], dtype=float)
                th = np.array([float(r["angle"]) for r in rows], dtype=float)
                x, y, th, L = norm_pack(x, y, th)
                Ld = Decimal(auto_union.certify(n, rows))
                if n not in best_L_dec or Ld < best_L_dec[n]:
                    packs[n] = (x, y, th, L)
                    best_L_dec[n] = Ld
        except Exception:
            pass

    return packs, best_L_dec


def exact_official_L_for_pack(n: int, x: np.ndarray, y: np.ndarray, th: np.ndarray) -> tuple[Decimal | None, Decimal | None, list[dict] | None]:
    """Certify using union.check in BOTH literal and pandas transport modes via s-prefixed final_text."""
    hx = hx_of(th)
    min_x = float((x - hx).min())
    min_y = float((y - hx).min())
    sx = x if min_x >= 0.0 else (x - min_x + 1e-10)
    sy = y if min_y >= 0.0 else (y - min_y + 1e-10)
    rows = []
    for i in range(n):
        rows.append(
            {
                "square_id": i,
                "x": repr(float(sx[i])),
                "y": repr(float(sy[i])),
                "angle": repr(float(th[i])),
            }
        )
    text = auto_union.final_text({n: rows})
    lit, pd_t = auto_union.parse_text(text), auto_union.parse_text(pd.read_csv(io.StringIO(text)).to_csv(index=False))
    res = auto_union.check((n, lit[n], pd_t[n]))
    if res is None:
        return None, None, None
    return Decimal(res[0]), Decimal(res[1]), rows


def certify_pack_against_frontier(n: int, x: np.ndarray, y: np.ndarray, th: np.ndarray, L_inc_dec: Decimal) -> dict | None:
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


def save_win_and_rebuild_final(n: int, cert: dict, label: str) -> None:
    """Atomically save per-n win JSON (with file lock) and rebuild poc/swarm_sametilt/final.csv."""
    import fcntl

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = DATA_DIR / ".win_lock"
    with open(lock_path, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        win_path = DATA_DIR / f"win_n{n}.json"
        if win_path.exists():
            prev = json.loads(win_path.read_text())
            if Decimal(max(prev["L_lit"], prev["L_pd"])) <= Decimal(max(cert["L_lit"], cert["L_pd"])):
                return
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

        groups = {}
        for p in sorted(DATA_DIR.glob("win_n*.json")):
            d = json.loads(p.read_text())
            groups[int(d["n"])] = d["rows"]
        text = auto_union.final_text(groups)
        auto_union.atomic(FINAL_CSV, text.encode())
        os.chmod(FINAL_CSV, 0o644)


def overlap_pairs(X: np.ndarray, Y: np.ndarray, T: np.ndarray, tol: float = 1e-9):
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


def clearance_points(x: np.ndarray, y: np.ndarray, th: np.ndarray, L: float, n_pts: int, step: float = 0.10, min_sep: float = 0.85):
    g = np.arange(0.5, L - 0.5 + 1e-9, step)
    gx, gy = np.meshgrid(g, g, indexing="ij")
    px, py = gx.ravel(), gy.ravel()
    clr = np.minimum(np.minimum(px, L - px), np.minimum(py, L - py)) - 0.5
    c, s = np.cos(th), np.sin(th)
    for a in range(0, len(px), 4096):
        dx = px[a:a + 4096, None] - x[None, :]
        dy = py[a:a + 4096, None] - y[None, :]
        d = np.maximum(np.abs(c * dx + s * dy), np.abs(-s * dx + c * dy)) - 0.5
        clr[a:a + 4096] = np.minimum(clr[a:a + 4096], d.min(axis=1))
    order = np.argsort(-clr)
    out = []
    for i in order:
        if len(out) >= n_pts:
            break
        if all((px[i] - q[0]) ** 2 + (py[i] - q[1]) ** 2 > min_sep ** 2 for q in out):
            out.append((float(px[i]), float(py[i]), float(clr[i])))
    return out


def polish_candidate(n: int, x: np.ndarray, y: np.ndarray, th: np.ndarray, L_inc_dec: Decimal, free_mask: np.ndarray | None = None) -> dict | None:
    """Relax if slight overlap, run multi-stage joint_xytheta_slp + fixed-angle LP tighten + official check."""
    L_inc = float(L_inc_dec)
    pen, bb = cpp_eval_sat_pen_bbox(x, y, th)
    if pen > 1e-7 and pen < 0.12:
        x, y, th, pen = u.relax_lbfgs(x, y, th, bb + max(0.002, 2.0 * pen), free_mask=free_mask, maxiter=260, w_wall=4.0)
        pen, bb = cpp_eval_sat_pen_bbox(x, y, th)
        if pen > 1e-6 and free_mask is not None:
            x, y, th, pen = u.relax_lbfgs(x, y, th, bb + max(0.002, 2.0 * pen), free_mask=None, maxiter=220, w_wall=4.0)
            pen, bb = cpp_eval_sat_pen_bbox(x, y, th)
    if pen > 0.02:
        return None

    xs, ys, ts, Ls, ok = joint_xytheta_slp(x, y, th, rounds=8, tr_th0=0.022, tr_xy0=0.09, margin=1e-8, tol=1e-8, patience=2)
    if not ok:
        return None
    if Ls < L_inc + 2e-3:
        xs2, ys2, ts2, Ls2, ok2 = joint_xytheta_slp(xs, ys, ts, rounds=10, tr_th0=0.007, tr_xy0=0.025, margin=1e-8, tol=1e-11, patience=2)
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
