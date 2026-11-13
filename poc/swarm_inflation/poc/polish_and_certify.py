"""Fixed-angle LP + Lipschitz-safe free-angle SLP polish and dual certification.

Runs in /mnt/e/squares-in-squares/.venv/bin/python (SciPy HiGHS + squares.core).
Takes candidate packings produced by GPU inflation, repairs tiny residual
overlaps via minimal uniform dilation if needed, runs fixed-angle LP + Lip-SLP
polish, and strictly gates with BOTH literal certify() and pandas roundtrip.
"""
from __future__ import annotations

import os

for _var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_var] = "1"

import argparse
import csv
import io
import json
import sys
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[3]
BASELINE_SRC = REPO / "poc" / "baseline" / "poc" / "src"
if str(BASELINE_SRC) not in sys.path:
    sys.path.insert(0, str(BASELINE_SRC))

from squares import _official_metric as metric  # noqa: E402
from squares import core  # noqa: E402

PAIR_RADIUS = 1.65
MARGIN = 1e-9
LIP = 0.7071067811865476


def extents(theta: np.ndarray) -> np.ndarray:
    return 0.5 * (np.abs(np.cos(theta)) + np.abs(np.sin(theta)))


def exact_bbox_side(centers: np.ndarray, theta: np.ndarray) -> float:
    e = extents(theta)
    return float(
        max(
            (centers[:, 0] + e).max() - (centers[:, 0] - e).min(),
            (centers[:, 1] + e).max() - (centers[:, 1] - e).min(),
        )
    )


def shift_positive(centers: np.ndarray, theta: np.ndarray) -> np.ndarray:
    e = extents(theta)
    out = centers.copy()
    out[:, 0] -= (out[:, 0] - e).min() - 1e-11
    out[:, 1] -= (out[:, 1] - e).min() - 1e-11
    return out


def sat_max_penetration(centers: np.ndarray, theta: np.ndarray) -> float:
    """Exact float64 SAT max penetration using cKDTree broad phase."""
    n = len(centers)
    if n <= 1:
        return 0.0
    pairs = list(cKDTree(centers).query_pairs(1.45))
    if not pairs:
        return 0.0
    c = np.cos(theta)
    s = np.sin(theta)
    max_pen = 0.0
    for i, j in pairs:
        dx = centers[j, 0] - centers[i, 0]
        dy = centers[j, 1] - centers[i, 1]
        ci, si = c[i], s[i]
        cj, sj = c[j], s[j]
        rad = 0.5 * (1.0 + abs(ci * cj + si * sj) + abs(si * cj - ci * sj))
        p1 = abs(dx * ci + dy * si)
        p2 = abs(-dx * si + dy * ci)
        p3 = abs(dx * cj + dy * sj)
        p4 = abs(-dx * sj + dy * cj)
        pen = rad - max(p1, p2, p3, p4)
        if pen > max_pen:
            max_pen = pen
    return max_pen


def minimal_feasible_dilation(
    centers: np.ndarray,
    theta: np.ndarray,
    tau: float = 1e-10,
) -> tuple[np.ndarray, float]:
    """Scale centers uniformly around origin just enough to guarantee tau clearance."""
    c_pos = shift_positive(centers, theta)
    pairs = list(cKDTree(c_pos).query_pairs(1.50))
    if not pairs:
        return c_pos, 1.0
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    lam = 1.0
    for i, j in pairs:
        dx = c_pos[i, 0] - c_pos[j, 0]
        dy = c_pos[i, 1] - c_pos[j, 1]
        if dx * dx + dy * dy >= 2.05:
            continue
        ci, si = cos_t[i], sin_t[i]
        cj, sj = cos_t[j], sin_t[j]
        rad = 0.5 * (1.0 + abs(ci * cj + si * sj) + abs(-si * cj + ci * sj))
        vmax = max(
            abs(dx * ci + dy * si),
            abs(-dx * si + dy * ci),
            abs(dx * cj + dy * sj),
            abs(-dx * sj + dy * cj),
        )
        if vmax <= 1e-12:
            return c_pos, 1e9
        req = (rad + tau) / vmax
        if req > lam:
            lam = req
    if lam > 1.0:
        c_pos = c_pos * lam
    return shift_positive(c_pos, theta), lam


def fixed_angle_lp_polish(
    centers: np.ndarray,
    theta: np.ndarray,
    rounds: int = 8,
    time_limit: float = 30.0,
) -> tuple[np.ndarray, float]:
    """Unbounded fixed-angle SLP compression (no trust region on centers)."""
    n = len(theta)
    cur = shift_positive(centers, theta)
    best_c = cur.copy()
    best_L = exact_bbox_side(cur, theta)

    u = np.column_stack([np.cos(theta), np.sin(theta)])
    v = np.column_stack([-np.sin(theta), np.cos(theta)])
    ext = extents(theta)

    for _ in range(rounds):
        pairs = list(cKDTree(cur).query_pairs(PAIR_RADIUS))
        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []
        rhs: list[float] = []

        def add(columns: list[int], values: list[float], b: float) -> None:
            row = len(rhs)
            rows.extend([row] * len(columns))
            cols.extend(columns)
            vals.extend(values)
            rhs.append(b)

        for i, j in pairs:
            axes = np.array([u[i], v[i], u[j], v[j]])
            delta = cur[j] - cur[i]
            proj = axes @ delta
            radii = 0.5 * (
                np.abs(axes @ u[i])
                + np.abs(axes @ v[i])
                + np.abs(axes @ u[j])
                + np.abs(axes @ v[j])
            )
            alpha = np.abs(proj) - radii
            k = int(np.argmax(alpha))
            a = axes[k] * (1.0 if proj[k] >= 0.0 else -1.0)
            add(
                [2 * i, 2 * i + 1, 2 * j, 2 * j + 1],
                [a[0], a[1], -a[0], -a[1]],
                -radii[k] - MARGIN,
            )

        for i in range(n):
            for k in range(2):
                add([2 * i + k], [-1.0], -ext[i] - MARGIN)
                add([2 * i + k, 2 * n], [1.0, -1.0], -ext[i] - MARGIN)

        A = coo_matrix((vals, (rows, cols)), shape=(len(rhs), 2 * n + 1)).tocsr()
        c_obj = np.zeros(2 * n + 1)
        c_obj[-1] = 1.0
        res = linprog(
            c_obj,
            A_ub=A,
            b_ub=np.array(rhs),
            bounds=[(None, None)] * (2 * n) + [(0.0, None)],
            method="highs",
            options={
                "dual_feasibility_tolerance": 1e-9,
                "primal_feasibility_tolerance": 1e-9,
                "time_limit": time_limit,
            },
        )
        if not res.success:
            break
        proposed = np.asarray(res.x[:-1]).reshape(n, 2)
        if sat_max_penetration(proposed, theta) > 1e-11:
            break
        side = exact_bbox_side(proposed, theta)
        shift = float(np.abs(proposed - cur).max())
        cur = proposed
        if side < best_L:
            best_L = side
            best_c = proposed.copy()
        if shift < 1e-9:
            break

    return shift_positive(best_c, theta), best_L


def lip_slp_polish(
    centers: np.ndarray,
    theta: np.ndarray,
    rounds: int = 8,
    time_limit: float = 45.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Lipschitz-safe free-angle SLP (dx, dy, dtheta, t, dL)."""
    n = len(theta)
    cur_c = shift_positive(centers, theta)
    cur_t = theta.copy()
    best_L = exact_bbox_side(cur_c, cur_t)
    ta, tm = 0.004, 0.012

    for _ in range(rounds):
        u = np.column_stack([np.cos(cur_t), np.sin(cur_t)])
        v = np.column_stack([-np.sin(cur_t), np.cos(cur_t)])
        pairs = list(cKDTree(cur_c).query_pairs(PAIR_RADIUS))
        rows, cols, vals, rhs = [], [], [], []

        def add(columns: list[int], values: list[float], b: float) -> None:
            row = len(rhs)
            rows.extend([row] * len(columns))
            cols.extend(columns)
            vals.extend(values)
            rhs.append(b)

        iL = 4 * n
        for i, j in pairs:
            axes = np.array([u[i], v[i], u[j], v[j]])
            delta = cur_c[j] - cur_c[i]
            proj = axes @ delta
            rad = 0.5 * (
                np.abs(axes @ u[i])
                + np.abs(axes @ v[i])
                + np.abs(axes @ u[j])
                + np.abs(axes @ v[j])
            )
            alpha = np.abs(proj) - rad
            k = int(np.argmax(alpha))
            direction = 1.0 if proj[k] >= 0.0 else -1.0
            a = axes[k] * direction
            slack = direction * proj[k] - rad[k] - MARGIN
            add(
                [2 * i, 2 * i + 1, 2 * j, 2 * j + 1, 3 * n + i, 3 * n + j],
                [a[0], a[1], -a[0], -a[1], LIP, LIP],
                slack,
            )

        e = extents(cur_t)
        for i in range(n):
            add([3 * n + i, 2 * n + i], [1.0, -1.0], 0.0)
            add([3 * n + i, 2 * n + i], [1.0, 1.0], 0.0)
            add([2 * i, 3 * n + i, iL], [1.0, LIP, -1.0], best_L - cur_c[i, 0] - e[i] - MARGIN)
            add([2 * i, 3 * n + i], [-1.0, LIP], cur_c[i, 0] - e[i] - MARGIN)
            add([2 * i + 1, 3 * n + i, iL], [1.0, LIP, -1.0], best_L - cur_c[i, 1] - e[i] - MARGIN)
            add([2 * i + 1, 3 * n + i], [-1.0, LIP], cur_c[i, 1] - e[i] - MARGIN)

        A = coo_matrix((vals, (rows, cols)), shape=(len(rhs), 4 * n + 1)).tocsr()
        bounds = (
            [(-tm, tm)] * (2 * n)
            + [(-ta, ta)] * n
            + [(0.0, ta)] * n
            + [(-0.08, 0.08)]
        )
        c_obj = np.zeros(4 * n + 1)
        c_obj[-1] = 1.0
        res = linprog(
            c_obj,
            A_ub=A,
            b_ub=np.array(rhs),
            bounds=bounds,
            method="highs",
            options={
                "dual_feasibility_tolerance": 1e-9,
                "primal_feasibility_tolerance": 1e-9,
                "time_limit": time_limit,
            },
        )
        if not res.success:
            ta *= 0.5
            tm *= 0.5
            if ta < 1e-7:
                break
            continue

        x = res.x
        nc = cur_c + np.column_stack([x[0 : 2 * n : 2], x[1 : 2 * n : 2]])
        nt = cur_t + x[2 * n : 3 * n]
        if sat_max_penetration(nc, nt) > 1e-11:
            ta *= 0.5
            tm *= 0.5
            if ta < 1e-7:
                break
            continue

        new_L = exact_bbox_side(nc, nt)
        if new_L <= best_L + 1e-11:
            cur_c, cur_t = shift_positive(nc, nt), nt
            best_L = exact_bbox_side(cur_c, cur_t)
            if np.abs(x).max() < 1e-9:
                break
            ta = min(0.004, max(ta * 1.25, 1e-6))
            tm = min(0.012, max(tm * 1.25, 1e-6))
        else:
            ta *= 0.5
            tm *= 0.5
            if ta < 1e-7:
                break

    return cur_c, cur_t, best_L


def joint_xytheta_slp_polish(
    centers: np.ndarray,
    theta: np.ndarray,
    rounds: int = 12,
    time_limit: float = 45.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Exact first-order signed-Jacobian joint (dx, dy, dtheta, dL) HiGHS SLP."""
    n = len(theta)
    cur_c, _ = minimal_feasible_dilation(shift_positive(centers, theta), theta)
    cur_c = shift_positive(cur_c, theta)
    cur_t = theta.copy()
    best_L = exact_bbox_side(cur_c, cur_t)
    ta, tm = 0.008, 0.020
    kink_eps = 1e-4

    for _ in range(rounds):
        u = np.column_stack([np.cos(cur_t), np.sin(cur_t)])
        v = np.column_stack([-np.sin(cur_t), np.cos(cur_t)])
        pairs = list(cKDTree(cur_c).query_pairs(PAIR_RADIUS))
        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []
        rhs: list[float] = []
        n_aux = 0

        def add(columns: list[int], values: list[float], b: float) -> None:
            row = len(rhs)
            rows.extend([row] * len(columns))
            cols.extend(columns)
            vals.extend(values)
            rhs.append(b)

        def new_aux() -> int:
            nonlocal n_aux
            col = 3 * n + 1 + n_aux
            n_aux += 1
            return col

        iL = 3 * n
        curv_buf = (0.4 * ta * ta) if ta > 2e-4 else 0.0

        for i, j in pairs:
            axes = (u[i], v[i], u[j], v[j])
            delta = cur_c[j] - cur_c[i]
            proj = np.array([float(w @ delta) for w in axes])
            rad = np.array(
                [
                    0.5
                    * (
                        abs(float(w @ u[i]))
                        + abs(float(w @ v[i]))
                        + abs(float(w @ u[j]))
                        + abs(float(w @ v[j]))
                    )
                    for w in axes
                ]
            )
            alpha = np.abs(proj) - rad
            k_best = int(np.argmax(alpha))
            alpha_max = float(alpha[k_best])
            active_ks = [
                k_cand
                for k_cand in range(4)
                if k_cand == k_best or alpha[k_cand] >= alpha_max - (2.0 * ta + 0.015)
            ]
            for k in active_ks:
                direction = 1.0 if proj[k] >= 0.0 else -1.0
                s = axes[k] * direction
                slack = direction * proj[k] - rad[k] - MARGIN - curv_buf

                # ds / dt_i and ds / dt_j
                if k == 0:
                    ds_di = v[i] * direction
                    ds_dj = np.zeros(2)
                elif k == 1:
                    ds_di = -u[i] * direction
                    ds_dj = np.zeros(2)
                elif k == 2:
                    ds_di = np.zeros(2)
                    ds_dj = v[j] * direction
                else:
                    ds_di = np.zeros(2)
                    ds_dj = -u[j] * direction

                c_ti = float(ds_di @ delta)
                c_tj = float(ds_dj @ delta)
                q_cols: list[int] = []

                other_axes = ((u[j], v[j], v[j], -u[j]) if k < 2 else (u[i], v[i], v[i], -u[i]))
                other_is_j = k < 2
                for idx_e in range(2):
                    e = other_axes[idx_e]
                    de = other_axes[idx_e + 2]
                    p = float(s @ e)
                    dp_di = float(ds_di @ e) + (0.0 if other_is_j else float(s @ de))
                    dp_dj = float(ds_dj @ e) + (float(s @ de) if other_is_j else 0.0)
                    if abs(p) >= kink_eps:
                        sgn = 1.0 if p > 0.0 else -1.0
                        c_ti -= 0.5 * sgn * dp_di
                        c_tj -= 0.5 * sgn * dp_dj
                    else:
                        q = new_aux()
                        add([2 * n + i, 2 * n + j, q], [dp_di, dp_dj, -1.0], 0.0)
                        add([2 * n + i, 2 * n + j, q], [-dp_di, -dp_dj, -1.0], 0.0)
                        q_cols.append(q)

                r_cols = [2 * i, 2 * i + 1, 2 * j, 2 * j + 1, 2 * n + i, 2 * n + j]
                r_vals = [float(s[0]), float(s[1]), float(-s[0]), float(-s[1]), -c_ti, -c_tj]
                for q in q_cols:
                    r_cols.append(q)
                    r_vals.append(0.5)
                add(r_cols, r_vals, slack)

        # Boundary constraints with exact d(ext_i)/dtheta_i (plus kink aux when cos or sin near 0)
        e = extents(cur_t)
        w_curv = MARGIN + ((0.2 * ta * ta) if ta > 2e-4 else 0.0)
        for i in range(n):
            cx, cy = float(cur_c[i, 0]), float(cur_c[i, 1])
            ct, st = float(u[i, 0]), float(u[i, 1])
            d_ext = 0.0
            q_ext = -1
            if abs(ct) >= kink_eps:
                d_ext += 0.5 * (1.0 if ct > 0.0 else -1.0) * (-st)
            else:
                q_ext = new_aux()
                add([2 * n + i, q_ext], [1.0, -1.0], 0.0)
                add([2 * n + i, q_ext], [-1.0, -1.0], 0.0)
            if abs(st) >= kink_eps:
                d_ext += 0.5 * (1.0 if st > 0.0 else -1.0) * ct
            elif q_ext < 0:
                q_ext = new_aux()
                add([2 * n + i, q_ext], [1.0, -1.0], 0.0)
                add([2 * n + i, q_ext], [-1.0, -1.0], 0.0)

            bcols_hi_x = [2 * i, 2 * n + i, iL] + ([q_ext] if q_ext >= 0 else [])
            bvals_hi_x = [1.0, d_ext, -1.0] + ([0.5] if q_ext >= 0 else [])
            add(bcols_hi_x, bvals_hi_x, best_L - cx - e[i] - w_curv)

            bcols_lo_x = [2 * i, 2 * n + i] + ([q_ext] if q_ext >= 0 else [])
            bvals_lo_x = [-1.0, d_ext] + ([0.5] if q_ext >= 0 else [])
            add(bcols_lo_x, bvals_lo_x, cx - e[i] - w_curv)

            bcols_hi_y = [2 * i + 1, 2 * n + i, iL] + ([q_ext] if q_ext >= 0 else [])
            bvals_hi_y = [1.0, d_ext, -1.0] + ([0.5] if q_ext >= 0 else [])
            add(bcols_hi_y, bvals_hi_y, best_L - cy - e[i] - w_curv)

            bcols_lo_y = [2 * i + 1, 2 * n + i] + ([q_ext] if q_ext >= 0 else [])
            bvals_lo_y = [-1.0, d_ext] + ([0.5] if q_ext >= 0 else [])
            add(bcols_lo_y, bvals_lo_y, cy - e[i] - w_curv)

        total_vars = 3 * n + 1 + n_aux
        A = coo_matrix((vals, (rows, cols)), shape=(len(rhs), total_vars)).tocsr()
        bounds = (
            [(-tm, tm)] * (2 * n)
            + [(-ta, ta)] * n
            + [(-0.08, 0.08)]
            + [(0.0, None)] * n_aux
        )
        c_obj = np.zeros(total_vars)
        c_obj[iL] = 1.0
        res = linprog(
            c_obj,
            A_ub=A,
            b_ub=np.array(rhs),
            bounds=bounds,
            method="highs",
            options={
                "dual_feasibility_tolerance": 1e-9,
                "primal_feasibility_tolerance": 1e-9,
                "time_limit": time_limit,
            },
        )
        if not res.success:
            ta *= 0.5
            tm *= 0.5
            if ta < 1e-7:
                break
            continue

        x = res.x
        dc = np.column_stack([x[0 : 2 * n : 2], x[1 : 2 * n : 2]])
        dt = x[2 * n : 3 * n]
        nc = cur_c + dc
        nt = cur_t + dt
        if sat_max_penetration(nc, nt) > 1e-11:
            # Try exact minimal feasible dilation or scaled step along LP direction
            accepted = False
            for alpha_step in (1.0, 0.5, 0.25):
                nc_s = cur_c + alpha_step * dc
                nt_s = cur_t + alpha_step * dt
                if sat_max_penetration(nc_s, nt_s) > 1e-11:
                    nc_s, sc_s = minimal_feasible_dilation(shift_positive(nc_s, nt_s), nt_s)
                    if sc_s > 1.002 or sat_max_penetration(nc_s, nt_s) > 1e-11:
                        continue
                new_L_s = exact_bbox_side(nc_s, nt_s)
                if new_L_s < best_L - 1e-11:
                    cur_c, cur_t = shift_positive(nc_s, nt_s), nt_s
                    best_L = exact_bbox_side(cur_c, cur_t)
                    accepted = True
                    break
            if not accepted:
                ta *= 0.5
                tm *= 0.5
                if ta < 1e-7:
                    break
            continue

        new_L = exact_bbox_side(nc, nt)
        if new_L <= best_L + 1e-11:
            cur_c, cur_t = shift_positive(nc, nt), nt
            best_L = exact_bbox_side(cur_c, cur_t)
            if max(float(np.abs(dc).max()), float(np.abs(dt).max())) < 1e-9:
                break
            ta = min(0.010, max(ta * 1.25, 1e-6))
            tm = min(0.025, max(tm * 1.25, 1e-6))
        else:
            ta *= 0.5
            tm *= 0.5
            if ta < 1e-7:
                break

    return cur_c, cur_t, best_L


def to_rows(centers: np.ndarray, theta: np.ndarray) -> list[dict[str, str]]:
    return [
        {
            "square_id": str(i),
            "x": repr(float(centers[i, 0])),
            "y": repr(float(centers[i, 1])),
            "angle": repr(float(theta[i])),
        }
        for i in range(len(centers))
    ]


def dual_certify(n: int, centers: np.ndarray, theta: np.ndarray) -> dict:
    """Check official literal certify AND actual pandas read_csv/to_csv roundtrip."""
    c_pos = shift_positive(centers, theta)
    for eps in (0.0, 1e-13, 1e-12, 1e-11, 1e-10, 1e-9):
        if eps > 0.0:
            with localcontext() as ctx:
                ctx.prec = 70
                base_rows = to_rows(c_pos, theta)
                rows = [
                    {
                        "square_id": r["square_id"],
                        "x": format(Decimal(r["x"]) * (Decimal(1) + Decimal(str(eps))), ".60g"),
                        "y": format(Decimal(r["y"]) * (Decimal(1) + Decimal(str(eps))), ".60g"),
                        "angle": r["angle"],
                    }
                    for r in base_rows
                ]
        else:
            rows = to_rows(c_pos, theta)

        _, _, L_lit, reason_lit = core.evaluate(n, rows)
        if reason_lit != "ok" or L_lit is None:
            continue

        # Actual pandas roundtrip matching Kaggle ingestion
        records = [dict(r, id=i, n=n) for i, r in enumerate(rows)]
        buf = io.StringIO()
        pd.DataFrame(records, columns=core.COLUMNS).to_csv(buf, index=False)
        frame = pd.read_csv(io.StringIO(buf.getvalue()))
        parsed = metric._parse_submission_csv(frame.to_csv(index=False))
        _, _, L_pd, reason_pd = core.evaluate(n, parsed[n])
        if reason_pd != "ok" or L_pd is None:
            continue

        return {
            "ok": True,
            "L_lit": L_lit,
            "L_pd": L_pd,
            "L_max": str(max(Decimal(L_lit), Decimal(L_pd))),
            "eps_used": eps,
            "rows": rows,
        }

    return {"ok": False, "L_lit": None, "L_pd": None, "rows": None}


def load_incumbent_L(n: int, merged_csv: Path | None = None) -> Decimal:
    """Read exact incumbent L for puzzle n from merged.csv via official metric."""
    if merged_csv is None:
        merged_csv = REPO / "poc" / "pell_corrugation" / "data" / "border-scan" / "merged.csv"
    rows: list[dict[str, str]] = []
    with open(merged_csv, newline="", encoding="utf-8") as handle:
        for r in csv.DictReader(handle):
            if int(r["n"]) == n:
                rows.append(
                    {
                        "square_id": r["square_id"],
                        "x": r["x"],
                        "y": r["y"],
                        "angle": r["angle"],
                    }
                )
    if len(rows) != n:
        raise ValueError(f"Expected {n} incumbent rows in {merged_csv}, found {len(rows)}")
    return Decimal(core.certify(n, rows))


def polish_candidates(
    n: int,
    pos_batch: np.ndarray,
    th_batch: np.ndarray,
    meta: list[str],
    out_win_csv: Path | None = None,
    lp_rounds: int = 8,
    slp_rounds: int = 8,
    slp_top_k: int = 4,
    out_best_csv: Path | None = None,
) -> dict:
    """Polish candidate batch for puzzle `n` and certify best against merged.csv."""
    inc_L = load_incumbent_L(n)
    best_cert_L: Decimal | None = None
    best_cert_info: dict | None = None
    candidate_logs: list[dict] = []

    # Stage 1: fast fixed-angle LP on all candidates with reasonable lam <= 1.05
    stage1: list[tuple[float, int, np.ndarray, np.ndarray, dict]] = []
    for idx in range(len(pos_batch)):
        p0 = pos_batch[idx]
        t0 = th_batch[idx]
        p_feas, lam = minimal_feasible_dilation(p0, t0, tau=1e-10)
        L_pre = exact_bbox_side(p_feas, t0)
        entry = {
            "idx": idx,
            "seed_meta": meta[idx] if idx < len(meta) else "",
            "lam": float(lam),
            "L_pre_lp": float(L_pre),
        }

        if lam > 1.05:
            entry["status"] = "high_lam_skip"
            candidate_logs.append(entry)
            continue

        p_lp1, L_lp1 = fixed_angle_lp_polish(p_feas, t0, rounds=lp_rounds)
        entry["L_lp1"] = float(L_lp1)
        stage1.append((float(L_lp1), idx, p_lp1, t0, entry))

    stage1.sort(key=lambda item: item[0])

    # Stage 2: run joint (dx, dy, dtheta, dL) SLP + final LP + dual certify on best slp_top_k
    for rank, (L_lp1, idx, p_lp1, t0, entry) in enumerate(stage1):
        if rank < slp_top_k:
            p_slp, t_slp, _ = joint_xytheta_slp_polish(p_lp1, t0, rounds=max(slp_rounds, 12))
            p_lp2, L_lp2 = fixed_angle_lp_polish(p_slp, t_slp, rounds=lp_rounds)
        else:
            p_lp2, t_slp, L_lp2 = p_lp1, t0, L_lp1
        entry["L_post_lp"] = float(L_lp2)

        # Only run full mpmath certification on top slp_top_k or candidates beating inc_L
        if rank >= slp_top_k and Decimal(str(L_lp2)) >= inc_L:
            entry["status"] = "screened_out"
            candidate_logs.append(entry)
            continue

        cert = dual_certify(n, p_lp2, t_slp)
        if not cert["ok"]:
            entry["status"] = "cert_fail"
            candidate_logs.append(entry)
            continue

        L_c = Decimal(cert["L_max"])
        entry["L_lit"] = cert["L_lit"]
        entry["L_pd"] = cert["L_pd"]
        entry["eps_used"] = cert["eps_used"]
        entry["status"] = "WIN" if L_c < inc_L else "certified_not_win"
        candidate_logs.append(entry)

        if best_cert_L is None or L_c < best_cert_L:
            best_cert_L = L_c
            best_cert_info = dict(cert, meta=entry["seed_meta"], L_post_lp=L_lp2)

    candidate_logs.sort(key=lambda e: e["idx"])

    if out_best_csv is not None and best_cert_info is not None:
        core.write_submission(out_best_csv, {n: best_cert_info["rows"]}, validate=True)

    is_win = best_cert_L is not None and best_cert_L < inc_L
    if is_win and out_win_csv is not None and best_cert_info is not None:
        core.write_submission(out_win_csv, {n: best_cert_info["rows"]}, validate=True)
        # Verify written CSV file via both literal read_submission and pandas roundtrip
        loaded = core.read_submission(out_win_csv)
        lit_check = core.certify(n, loaded[n])
        df_pd = pd.read_csv(out_win_csv)
        pd_parsed = metric._parse_submission_csv(df_pd.to_csv(index=False))
        pd_check = core.certify(n, pd_parsed[n])
        best_cert_info["verified_file_L_lit"] = lit_check
        best_cert_info["verified_file_L_pd"] = pd_check

    return {
        "n": n,
        "inc_L": str(inc_L),
        "best_cert_L": str(best_cert_L) if best_cert_L is not None else None,
        "best_L_lit": best_cert_info["L_lit"] if best_cert_info else None,
        "best_L_pd": best_cert_info["L_pd"] if best_cert_info else None,
        "win": bool(is_win),
        "candidates": candidate_logs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-npz", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--win-csv", default="")
    parser.add_argument("--lp-rounds", type=int, default=8)
    parser.add_argument("--slp-rounds", type=int, default=8)
    args = parser.parse_args()

    data = np.load(args.input_npz, allow_pickle=True)
    n = int(data["n"])
    pos = np.asarray(data["pos"], dtype=np.float64)
    th = np.asarray(data["th"], dtype=np.float64)
    meta = [str(x) for x in data["meta"]]
    out_csv = Path(args.win_csv) if args.win_csv else None

    res = polish_candidates(
        n=n,
        pos_batch=pos,
        th_batch=th,
        meta=meta,
        out_win_csv=out_csv,
        lp_rounds=args.lp_rounds,
        slp_rounds=args.slp_rounds,
    )
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(res, handle, indent=2)
    print(
        f"n={n} inc={res['inc_L'][:14]} best={str(res['best_cert_L'])[:14]} win={res['win']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
