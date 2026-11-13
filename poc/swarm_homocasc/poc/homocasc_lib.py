#!/usr/bin/env python3
"""Shared utilities, fast origin/main portfolio loader, dual certification, and SLP+mechwalk polish for H30."""
from __future__ import annotations

import csv
import fcntl
import io
import json
import os
import sys
import tempfile
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
UNION_POC = REPO / "poc" / "auto_union" / "poc"
MECH_POC = REPO / "poc" / "swarm_mechwalk" / "poc"
for _p in (str(HOMO_POC), str(UNION_POC), str(MECH_POC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import union as auto_union  # noqa: E402
import utils as u  # noqa: E402
from cpp_wrap import cpp_eval_sat_pen_bbox, cpp_expand_sym, full_to_free_sym, run_cpp_masked_melt  # noqa: E402
from joint_slp import joint_xytheta_slp  # noqa: E402

HOMOCASC_DIR = REPO / "poc" / "swarm_homocasc"
DATA_DIR = HOMOCASC_DIR / "data"
DOCS_DIR = HOMOCASC_DIR / "docs"
FINAL_CSV = HOMOCASC_DIR / "final.csv"

MERGED_CSV = REPO / "poc" / "pell_corrugation" / "data" / "border-scan" / "merged.csv"
BRICKSLP_DATA = Path(
    "/home/lucas/.gemini/antigravity-cli/worktrees/a4fb0b93-b63b-426d-aa12-67454e99bc08"
    "/swarm-brickslp-1e87a2dd/poc/swarm_brickslp/data"
)


def hx_of(th: np.ndarray) -> np.ndarray:
    return 0.5 * (np.abs(np.cos(th)) + np.abs(np.sin(th)))


def norm_pack(x: np.ndarray, y: np.ndarray, th: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
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


def _csv_has_any_n(path: Path, need: set[int]) -> bool:
    """Fast pre-scan of id/n column in final.csv before full mpmath parsing."""
    if path.stat().st_size <= 20:
        return False
    prefixes = {f"{n:03d}_" for n in need}
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        header = f.readline()
        if not header.startswith("id,x,y,deg"):
            return False
        for line in f:
            if line[:4] in prefixes or (1000 in need and line.startswith("1000_")):
                return True
    return False


def load_portfolio(target_ns: set[int] | list[int], include_brickslp: bool = False) -> dict[int, dict]:
    """
    Fast exact portfolio loader for target_ns across merged.csv + poc/*/final.csv (excluding swarm_homocasc).
    Optionally loads brickslp win_n*.json as donor entries under key 'brickslp_donors' (never lowers target L_dec!).
    """
    need = {int(n) for n in target_ns}
    best: dict[int, dict] = {}

    df = pd.read_csv(MERGED_CSV, usecols=["n", "square_id", "x", "y", "angle"])
    sub_df = df[df["n"].isin(need)]
    # Also parse exact raw strings for need from merged.csv in one pass
    raw_rows: dict[int, list[dict]] = {n: [] for n in need}
    with MERGED_CSV.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            n = int(r["n"])
            if n in raw_rows:
                raw_rows[n].append(
                    {
                        "square_id": int(r["square_id"]),
                        "x": _strip_s(r["x"]),
                        "y": _strip_s(r["y"]),
                        "angle": _strip_s(r["angle"]),
                    }
                )

    for n_val, grp in sub_df.groupby("n"):
        n = int(n_val)
        grp_s = grp.sort_values("square_id")
        rows = sorted(raw_rows[n], key=lambda r: int(r["square_id"]))
        L_dec = Decimal(auto_union.certify(n, rows))
        x = grp_s["x"].to_numpy(dtype=np.float64, copy=True)
        y = grp_s["y"].to_numpy(dtype=np.float64, copy=True)
        th = grp_s["angle"].to_numpy(dtype=np.float64, copy=True)
        x, y, th, L_f = norm_pack(x, y, th)
        best[n] = {"x": x, "y": y, "th": th, "L": L_f, "L_dec": L_dec, "source": "merged.csv"}

    for p in sorted((REPO / "poc").glob("*/final.csv")):
        if p == FINAL_CSV or not _csv_has_any_n(p, need):
            continue
        try:
            lit, pd_t = auto_union.load(p)
        except Exception:
            continue
        for n in need:
            if n not in lit or n not in pd_t or len(lit[n]) != n:
                continue
            res = auto_union.check((n, lit[n], pd_t[n]))
            if res is None:
                continue
            L_dec = max(Decimal(res[0]), Decimal(res[1]))
            if n not in best or L_dec < best[n]["L_dec"]:
                rows = sorted(lit[n], key=lambda r: int(r["square_id"]))
                x = np.array([float(r["x"]) for r in rows], dtype=np.float64)
                y = np.array([float(r["y"]) for r in rows], dtype=np.float64)
                th = np.array([float(r["angle"]) for r in rows], dtype=np.float64)
                x, y, th, L_f = norm_pack(x, y, th)
                best[n] = {
                    "x": x,
                    "y": y,
                    "th": th,
                    "L": L_f,
                    "L_dec": L_dec,
                    "source": str(p.relative_to(REPO)),
                }

    if include_brickslp:
        donors: dict[int, dict] = {}
        for n in need:
            if n in best:
                donors[n] = dict(best[n])
        if BRICKSLP_DATA.exists():
            for jp in sorted(BRICKSLP_DATA.glob("win_n*.json")):
                d = json.loads(jp.read_text())
                n = int(d["n"])
                if n not in need:
                    continue
                L_b = max(Decimal(d["L_lit"]), Decimal(d.get("L_pd", d.get("L_pandas", d["L_lit"]))))
                if n not in donors or L_b < donors[n]["L_dec"]:
                    rows = sorted(d["rows"], key=lambda r: int(r["square_id"]))
                    x = np.array([float(r["x"]) for r in rows], dtype=np.float64)
                    y = np.array([float(r["y"]) for r in rows], dtype=np.float64)
                    th = np.array([float(r["angle"]) for r in rows], dtype=np.float64)
                    x, y, th, L_f = norm_pack(x, y, th)
                    donors[n] = {
                        "x": x,
                        "y": y,
                        "th": th,
                        "L": L_f,
                        "L_dec": L_b,
                        "source": f"brickslp:{jp.name}",
                    }
        return {"targets": best, "donors": donors}

    return best


def certify_pack_against_frontier(
    n: int, x: np.ndarray, y: np.ndarray, th: np.ndarray, L_inc_dec: Decimal
) -> dict | None:
    """Certify candidate against exact incumbent L_inc_dec using literal + pandas union.check."""
    hx = hx_of(th)
    sx = x - float((x - hx).min()) + 1e-12
    sy = y - float((y - hx).min()) + 1e-12
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
        return None
    L_lit, L_pd = Decimal(res[0]), Decimal(res[1])
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


def rebuild_final_csv() -> None:
    """Rebuild poc/swarm_homocasc/final.csv from all certified win_n*.json files (or header-only if none)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    groups = {}
    for p in sorted(DATA_DIR.glob("win_n*.json")):
        d = json.loads(p.read_text())
        groups[int(d["n"])] = d["rows"]
    text = auto_union.final_text(groups)
    auto_union.atomic(FINAL_CSV, text.encode())
    os.chmod(FINAL_CSV, 0o644)
    lit, pd_t = auto_union.load(FINAL_CSV)
    for n in groups:
        if auto_union.check((n, lit[n], pd_t[n])) is None:
            raise ValueError(f"Exported final.csv failed dual certification for n={n}")


def save_win_and_rebuild_final(n: int, cert: dict, label: str) -> bool:
    """Atomically save per-n win JSON (with file lock) and rebuild poc/swarm_homocasc/final.csv."""
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
    x: np.ndarray, y: np.ndarray, th: np.ndarray, L: float, n_pts: int, step: float = 0.10, min_sep: float = 0.80
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
) -> tuple[dict | None, tuple[np.ndarray, np.ndarray, np.ndarray, float] | None]:
    """Relax if slight overlap, run 2-stage joint_xytheta_slp + fixed-angle LP tighten + official check."""
    L_inc = float(L_inc_dec)
    x, y, th, bb = norm_pack(x, y, th)
    pen, bb = cpp_eval_sat_pen_bbox(x, y, th)
    if 1e-7 < pen < 0.15:
        x, y, th, pen = u.relax_lbfgs(x, y, th, bb + max(0.002, 2.0 * pen), free_mask=free_mask, maxiter=240, w_wall=4.0)
        pen, bb = cpp_eval_sat_pen_bbox(x, y, th)
        if pen > 1e-6 and free_mask is not None:
            x, y, th, pen = u.relax_lbfgs(x, y, th, bb + max(0.002, 2.0 * pen), free_mask=None, maxiter=200, w_wall=4.0)
            pen, bb = cpp_eval_sat_pen_bbox(x, y, th)
    if pen > 0.02:
        return None, None

    xs, ys, ts, Ls, ok = joint_xytheta_slp(x, y, th, rounds=10, tr_th0=0.022, tr_xy0=0.09, margin=1e-8, tol=1e-8, patience=2)
    if not ok:
        return None, None
    if Ls < L_inc + 0.015:
        xs2, ys2, ts2, Ls2, ok2 = joint_xytheta_slp(
            xs, ys, ts, rounds=12, tr_th0=0.007, tr_xy0=0.025, margin=1e-8, tol=1e-11, patience=3
        )
        if ok2 and Ls2 < Ls:
            xs, ys, ts, Ls = xs2, ys2, ts2, Ls2

    best = None
    best_pack = (xs, ys, ts, Ls)
    for mlp in (1e-7, 2e-8, 5e-9, 1e-6):
        xf, yf, Lf, okf = u.fixed_angle_lp_tighten(xs, ys, ts, rounds=4, margin=mlp)
        if not okf:
            continue
        cert = certify_pack_against_frontier(n, xf, yf, ts, L_inc_dec)
        if cert is not None and (best is None or Decimal(cert["L_lit"]) < Decimal(best["L_lit"])):
            best = cert
            best_pack = (xf, yf, ts, cert["L_float"])
            if mlp <= 1e-7 and cert["is_real_win"]:
                break
    if best is None:
        cert = certify_pack_against_frontier(n, xs, ys, ts, L_inc_dec)
        if cert is not None:
            best = cert
            best_pack = (xs, ys, ts, cert["L_float"])
    if best is None and pen <= 1e-12:
        cert = certify_pack_against_frontier(n, x, y, th, L_inc_dec)
        if cert is not None:
            best = cert
            best_pack = (x, y, th, cert["L_float"])
    return best, best_pack


def mechwalk_compress(
    n: int,
    x: np.ndarray,
    y: np.ndarray,
    th: np.ndarray,
    L_inc_dec: Decimal,
    rng: np.random.Generator,
    rounds: int = 2,
    threads: int = 4,
) -> tuple[dict | None, tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
    """Run sparse contact-Jacobian SVD soft-mode mechanism walk + C++ masked melt + SLP on a basin."""
    import mechwalk_lib as ml

    cur_x, cur_y, cur_t, cur_L = norm_pack(x, y, th)
    best_cert: dict | None = None
    best_pack = (cur_x, cur_y, cur_t, cur_L)

    for rnd in range(rounds):
        all_modes = []
        for tol in (2e-5, 5e-4, 3e-3):
            jac, _ = ml.contact_jacobian(cur_x, cur_y, cur_t, contact_tol=tol)
            sv, raw_modes = ml.soft_modes(jac, k=16)
            sel = ml.select_diverse_modes(raw_modes, sv, cur_x, cur_y, max_keep=6)
            for _, sval, mode_vec, n_eff in sel:
                if any(abs(float(np.dot(mode_vec, prev[2]))) > 0.90 for prev in all_modes):
                    continue
                disp, _ = ml.mode_participation(mode_vec, n)
                all_modes.append((sval, n_eff, mode_vec, disp))
        if not all_modes:
            break
        all_modes = all_modes[:12]

        allq, allm, alls, alll = [], [], [], []
        L0 = cur_L + 0.012
        off = 0.5 * (L0 - cur_L)
        sched = [L0, cur_L + 0.002, cur_L - 2e-4, cur_L - 1.2e-3]
        for m_i, (_, _, mvec, disp) in enumerate(all_modes):
            max_disp = float(np.max(disp))
            if max_disp < 1e-12:
                continue
            for step_sz in (0.025, 0.075, 0.16):
                for sign in (1.0, -1.0):
                    sc = sign * (step_sz / max_disp)
                    xm = cur_x + sc * mvec[:n] + off
                    ym = cur_y + sc * mvec[n : 2 * n] + off
                    tm = cur_t + sc * mvec[2 * n :]
                    hm = hx_of(tm)
                    xm = np.clip(xm, hm + 1e-4, L0 - hm - 1e-4)
                    ym = np.clip(ym, hm + 1e-4, L0 - hm - 1e-4)
                    active = (disp * (step_sz / max_disp)) >= 0.004
                    if active.sum() < 16:
                        active[np.argsort(-disp)[: min(28, n)]] = True
                    allq.append(full_to_free_sym(xm, ym, tm, L0))
                    mm = active.astype(np.uint8)
                    allm.append(np.concatenate([mm, mm, mm]))
                    alls.append(sched)
                    alll.append(f"mw_r{rnd}_m{m_i}_s{step_sz:.3f}_{'+' if sign>0 else '-'}")

        if not allq:
            break
        qb, qf, lf, eo, po, lb = run_cpp_masked_melt(
            n,
            1,
            allq,
            allm,
            alls,
            [6, 10, 14, 18],
            a_start=0.72,
            n_melt_stages=4,
            masked_iters=160,
            full_iters=200,
            n_threads=min(4, threads),
            base_seed=int(rng.integers(10000, 999999)),
        )
        order = sorted(range(len(alll)), key=lambda i: (0 if lf[i] < 1e9 else 1, lf[i] if lf[i] < 1e9 else po[i]))[:4]
        improved = False
        for i in order:
            use_f = lf[i] < 1e9
            Lr = float(lf[i]) if use_f else alls[i][-1]
            xp, yp, tp = cpp_expand_sym(n, 1, Lr, qf[i] if use_f else qb[i])
            cert, pack = polish_candidate(n, xp, yp, tp, L_inc_dec)
            if cert is not None and pack is not None:
                if best_cert is None or Decimal(cert["L_lit"]) < Decimal(best_cert["L_lit"]):
                    best_cert = cert
                if pack[3] < cur_L - 1e-8:
                    cur_x, cur_y, cur_t, cur_L = pack
                    best_pack = pack
                    improved = True
        if not improved:
            break
    return best_cert, best_pack
