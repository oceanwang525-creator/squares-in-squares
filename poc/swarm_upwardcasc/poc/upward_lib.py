#!/usr/bin/env python3
"""Shared utilities for H98 (swarm_upwardcasc): portfolio loader, dual certifier, SLP + mechwalk + lpslack."""
from __future__ import annotations

import csv
import fcntl
import io
import json
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
UNION_POC = REPO / "poc" / "auto_union" / "poc"
MECH_POC = REPO / "poc" / "swarm_mechwalk" / "poc"
LPSLACK_POC = REPO / "poc" / "swarm_lpslack" / "poc"
BASELINE_SRC = REPO / "poc" / "baseline" / "poc" / "src"

for _p in (str(HOMO_POC), str(UNION_POC), str(MECH_POC), str(LPSLACK_POC), str(BASELINE_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import lpslack  # noqa: E402
import union as auto_union  # noqa: E402
import utils as u  # noqa: E402
from cpp_wrap import (  # noqa: E402
    cpp_eval_sat_pen_bbox,
    cpp_expand_sym,
    full_to_free_sym,
    run_cpp_batch_ils,
    run_cpp_masked_melt,
)
from joint_slp import joint_xytheta_slp  # noqa: E402

UPWARD_DIR = REPO / "poc" / "swarm_upwardcasc"
DATA_DIR = UPWARD_DIR / "data"
DOCS_DIR = UPWARD_DIR / "docs"
FINAL_CSV = UPWARD_DIR / "final.csv"
MERGED_CSV = REPO / "poc" / "pell_corrugation" / "data" / "border-scan" / "merged.csv"


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


def _rows_to_arrays(rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    srows = sorted(rows, key=lambda r: int(r["square_id"]))
    x = np.array([float(_strip_s(str(r["x"]))) for r in srows], dtype=np.float64)
    y = np.array([float(_strip_s(str(r["y"]))) for r in srows], dtype=np.float64)
    th = np.array([float(_strip_s(str(r["angle"]))) for r in srows], dtype=np.float64)
    return norm_pack(x, y, th)


def load_portfolio(target_ns: set[int] | list[int], include_prs: bool = True) -> dict[str, dict[int, dict]]:
    """
    Load certified origin/main incumbents into `targets` and sharpest PR-branch donors into `donors`.
    Also populates `pr_best` with exact Decimal L across pending PRs for comparison.
    """
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
        if p == FINAL_CSV or p.stat().st_size <= 20:
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
                best_entry = {"x": x, "y": y, "th": th, "L": L_f, "L_dec": L_dec, "source": src}
        if best_entry is None:
            raise RuntimeError(f"No certified incumbent found for n={n}")
        targets[n] = best_entry

    donors: dict[int, dict] = {n: dict(targets[n]) for n in targets}
    pr_best: dict[int, Decimal] = {n: targets[n]["L_dec"] for n in targets}

    if include_prs:
        for spec, label in [
            ("origin/swarm-lpslack-pr3:poc/swarm_lpslack/final.csv", "origin/swarm-lpslack-pr3"),
            ("origin/swarm-homocasc2-pr1:poc/swarm_homocasc2/final.csv", "origin/swarm-homocasc2-pr1"),
        ]:
            try:
                raw = subprocess.check_output(
                    ["git", "-C", str(REPO), "show", spec],
                    text=True,
                    stderr=subprocess.DEVNULL,
                )
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
                        if L_dec < pr_best[n]:
                            pr_best[n] = L_dec
                        if n not in donors or L_dec < donors[n]["L_dec"]:
                            x, y, th, L_f = _rows_to_arrays(rows)
                            donors[n] = {"x": x, "y": y, "th": th, "L": L_f, "L_dec": L_dec, "source": label}
            except Exception as exc:
                warnings.warn(f"Could not load PR donors from {spec}: {exc}")

    if DATA_DIR.exists():
        for wp in sorted(DATA_DIR.glob("win_n*.json")):
            try:
                d = json.loads(wp.read_text())
                n = int(d["n"])
                if n not in need:
                    continue
                L_w = max(Decimal(d["L_lit"]), Decimal(d["L_pd"]))
                if L_w < donors[n]["L_dec"]:
                    x, y, th, L_f = _rows_to_arrays(d["rows"])
                    donors[n] = {"x": x, "y": y, "th": th, "L": L_f, "L_dec": L_w, "source": f"upward:{wp.name}"}
            except Exception:
                continue

    return {"targets": targets, "donors": donors, "pr_best": pr_best}


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
    """Normalize arrays, optionally try uniform separating-axis shrink, and dual-certify."""
    hx = hx_of(th)
    sx = x - float((x - hx).min()) + 1e-12
    sy = y - float((y - hx).min()) + 1e-12
    centers = np.column_stack([sx, sy])

    shrunk = lpslack.try_shrink_and_certify(n, centers, th, Decimal("999.0"))
    best_cert: dict | None = None
    if shrunk is not None:
        _, _, srows = shrunk
        c_shrunk = certify_rows_against_frontier(n, srows, L_inc_dec)
        if c_shrunk is not None:
            best_cert = c_shrunk

    raw_rows = [
        {"square_id": str(i), "x": repr(float(sx[i])), "y": repr(float(sy[i])), "angle": repr(float(th[i]))}
        for i in range(n)
    ]
    c_raw = certify_rows_against_frontier(n, raw_rows, L_inc_dec)
    if c_raw is not None:
        if best_cert is None or Decimal(c_raw["L_lit"]) < Decimal(best_cert["L_lit"]):
            best_cert = c_raw
    return best_cert


def rebuild_final_csv() -> None:
    """Rebuild poc/swarm_upwardcasc/final.csv from all certified win_n*.json files (or header-only)."""
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
    """Atomically save per-n win JSON (with file lock) and rebuild poc/swarm_upwardcasc/final.csv."""
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
            "L_kr": cert.get("L_kr", cert["L_pd"]),
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


def run_lpslack_polish(
    n: int, x: np.ndarray, y: np.ndarray, th: np.ndarray, L_inc_dec: Decimal, outer_iters: int = 6
) -> tuple[dict | None, tuple[np.ndarray, np.ndarray, np.ndarray, float] | None]:
    """Run H75 lpslack alternating soft-angle SLP + unbounded fixed-angle LP + exact shrink."""
    xn, yn, tn, Ln = norm_pack(x, y, th)
    rows = [
        {"square_id": str(i), "x": repr(float(xn[i])), "y": repr(float(yn[i])), "angle": repr(float(tn[i]))}
        for i in range(n)
    ]
    ceiling = str(Decimal(repr(Ln + 0.5)))
    res = lpslack.optimize_group(
        n,
        rows,
        ceiling,
        outer_iters=outer_iters,
        fixed_rounds=10,
        slp_modes=("dual_non90", "non90"),
        tr_th_schedule=(5e-4, 1.5e-4, 4e-5),
        time_limit=10.0,
    )
    if res.get("status") not in ("WIN", "TINY_WIN") or "rows" not in res:
        return None, None
    out_rows = res["rows"]
    cert = certify_rows_against_frontier(n, out_rows, L_inc_dec)
    if cert is None:
        return None, None
    xo, yo, to, _ = _rows_to_arrays(out_rows)
    return cert, (xo, yo, to, cert["L_float"])


def polish_candidate(
    n: int,
    x: np.ndarray,
    y: np.ndarray,
    th: np.ndarray,
    L_inc_dec: Decimal,
    free_mask: np.ndarray | None = None,
    use_lpslack: bool = True,
) -> tuple[dict | None, tuple[np.ndarray, np.ndarray, np.ndarray, float] | None]:
    """Relax if slight overlap, run 2-stage joint_xytheta_slp + fixed-angle LP + lpslack + official check."""
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

    best: dict | None = None
    best_pack: tuple[np.ndarray, np.ndarray, np.ndarray, float] = (xs, ys, ts, Ls)
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

    if use_lpslack and best is not None and best["L_float"] < L_inc + 0.015:
        lp_cert, lp_pack = run_lpslack_polish(n, best_pack[0], best_pack[1], best_pack[2], L_inc_dec, outer_iters=6)
        if lp_cert is not None and lp_pack is not None:
            if Decimal(lp_cert["L_lit"]) < Decimal(best["L_lit"]):
                best = lp_cert
                best_pack = lp_pack

    return best, best_pack


def mechwalk_compress(
    n: int,
    x: np.ndarray,
    y: np.ndarray,
    th: np.ndarray,
    L_inc_dec: Decimal,
    rng: np.random.Generator,
    rounds: int = 2,
    threads: int = 2,
) -> tuple[dict | None, tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
    """Run sparse contact-Jacobian SVD soft-mode mechanism walk + C++ masked melt + SLP + lpslack."""
    import mechwalk_lib as ml

    n_threads = min(2, max(1, int(threads)))
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
        all_modes = all_modes[:10]

        allq, allm, alls, alll = [], [], [], []
        L0 = cur_L + 0.012
        off = 0.5 * (L0 - cur_L)
        sched = [L0, cur_L + 0.002, cur_L - 2e-4, cur_L - 1.5e-3]
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
            n_threads=n_threads,
            base_seed=int(rng.integers(10000, 999999)),
        )
        order = sorted(range(len(alll)), key=lambda i: (0 if lf[i] < 1e9 else 1, lf[i] if lf[i] < 1e9 else po[i]))[:4]
        improved = False
        for i in order:
            use_f = lf[i] < 1e9
            Lr = float(lf[i]) if use_f else alls[i][-1]
            xp, yp, tp = cpp_expand_sym(n, 1, Lr, qf[i] if use_f else qb[i])
            cert, pack = polish_candidate(n, xp, yp, tp, L_inc_dec, use_lpslack=False)
            if cert is not None and pack is not None:
                if best_cert is None or Decimal(cert["L_lit"]) < Decimal(best_cert["L_lit"]):
                    best_cert = cert
                if pack[3] < cur_L - 1e-8:
                    cur_x, cur_y, cur_t, cur_L = pack
                    best_pack = pack
                    improved = True
        if not improved:
            break

    lp_cert, lp_pack = run_lpslack_polish(n, best_pack[0], best_pack[1], best_pack[2], L_inc_dec, outer_iters=8)
    if lp_cert is not None and lp_pack is not None:
        if best_cert is None or Decimal(lp_cert["L_lit"]) < Decimal(best_cert["L_lit"]):
            best_cert = lp_cert
            best_pack = lp_pack
    return best_cert, best_pack
