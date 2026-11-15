"""Auto-partitioning reduced active-arm model for general 4-corner two-arm / pinwheel packings.

Generalizes poc/row_ratchet_980/poc/row980.py from fixed BL/TR blocks to arbitrary
combinations of BL_static, TR_rigid, TL_rigid, and BR_rigid 0-deg grid blocks.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.sparse.csgraph import connected_components

from pslp_4c import feas, squeeze


def normalize_arr(a: np.ndarray) -> tuple[np.ndarray, float]:
    """Shift a=(N,3) [x, y, theta] so min(x-e)=0, min(y-e)=0, return (a_norm, L)."""
    out = np.asarray(a, dtype=np.float64).copy()
    e = 0.5 * (np.abs(np.cos(out[:, 2])) + np.abs(np.sin(out[:, 2])))
    out[:, 0] -= float((out[:, 0] - e).min())
    out[:, 1] -= float((out[:, 1] - e).min())
    L = float(max((out[:, 0] + e).max(), (out[:, 1] + e).max()))
    return out, L


def full_viol(full: np.ndarray, L: float) -> float:
    mp, mw = feas(full[:, 0].copy(), full[:, 1].copy(), full[:, 2].copy(), L, L)
    return float(max(0.0, -min(mp, mw)))


def build_model(
    a: np.ndarray,
    L: float,
    act_reach: float = 2.6,
    layer: float = 1.6,
    wall_tol: float = 0.25,
    min_comp: int = 4,
) -> dict:
    """Automatically partition squares into active arms + 4 corner rigid 0-deg grid blocks."""
    n = len(a)
    x, y, th = a[:, 0], a[:, 1], a[:, 2]
    e = 0.5 * (np.abs(np.cos(th)) + np.abs(np.sin(th)))
    deg_mod = np.degrees(np.mod(th, math.pi / 2.0))
    til = np.minimum(deg_mod, 90.0 - deg_mod) > 2.0

    if not np.any(til):
        idx = np.arange(n, dtype=np.int64)
        z = np.zeros(n, dtype=bool)
        return dict(
            idx=idx,
            reduced=False,
            fx=z,
            fy=z,
            fth=z,
            mx=z,
            my=z,
            mx_full=z,
            my_full=z,
            block_counts={"active": n},
        )

    P = a[til, :2]
    d_til = np.full(n, 1e9)
    for i in range(0, n, 256):
        d_til[i:i + 256] = np.min(
            np.hypot(x[i:i + 256, None] - P[None, :, 0], y[i:i + 256, None] - P[None, :, 1]),
            axis=1,
        )
    active = d_til < act_reach

    # Also treat any slightly-tilted (> 0.2 deg) square outside act_reach as active
    near_zero = np.minimum(deg_mod, 90.0 - deg_mod) <= 0.2
    active |= ~near_zero

    rigid_cand = np.flatnonzero(~active)
    block_id = np.zeros(n, dtype=np.int32)  # 0=free, 1=BL, 2=TR, 3=TL, 4=BR
    if len(rigid_cand) > 0:
        rx, ry = x[rigid_cand], y[rigid_cand]
        adj = ((rx[:, None] - rx[None, :]) ** 2 + (ry[:, None] - ry[None, :]) ** 2) < (1.15 ** 2)
        n_comp, labels = connected_components(adj, directed=False)
        for c in range(n_comp):
            m = rigid_cand[labels == c]
            if len(m) < min_comp:
                active[m] = True
                continue
            wl = float((x[m] - e[m]).min()) < wall_tol
            wr = float(L - (x[m] + e[m]).max()) < wall_tol
            wb = float((y[m] - e[m]).min()) < wall_tol
            wt = float(L - (y[m] + e[m]).max()) < wall_tol

            # Reject components spanning opposing walls or missing corner anchor
            if (wl and wr) or (wb and wt):
                active[m] = True
            elif wl and wb:
                block_id[m] = 1  # BL_static: (0, 0)
            elif wr and wt:
                block_id[m] = 2  # TR_rigid: (dL, dL)
            elif wl and wt:
                block_id[m] = 3  # TL_rigid: (0, dL)
            elif wr and wb:
                block_id[m] = 4  # BR_rigid: (dL, 0)
            else:
                active[m] = True

    fx_full = (block_id == 1) | (block_id == 3)
    fy_full = (block_id == 1) | (block_id == 4)
    mx_full = (block_id == 2) | (block_id == 4)
    my_full = (block_id == 2) | (block_id == 3)
    fth_full = block_id != 0

    A = a[active, :2]
    dl = np.full(n, 1e9)
    for i in range(0, n, 256):
        dl[i:i + 256] = np.min(
            np.hypot(x[i:i + 256, None] - A[None, :, 0], y[i:i + 256, None] - A[None, :, 1]),
            axis=1,
        )
    keep = active | (dl < layer)

    # Ensure no two ~keep squares from different rigid blocks are within interaction range (< 1.8)
    rest_idx = np.flatnonzero(~keep)
    if len(rest_idx) > 1:
        r_blk = block_id[rest_idx]
        if len(np.unique(r_blk)) > 1:
            rrx, rry = x[rest_idx], y[rest_idx]
            diff_blk = r_blk[:, None] != r_blk[None, :]
            close = ((rrx[:, None] - rrx[None, :]) ** 2 + (rry[:, None] - rry[None, :]) ** 2) < (1.8 ** 2)
            touch_diff = np.any(diff_blk & close, axis=1)
            if np.any(touch_diff):
                keep[rest_idx[touch_diff]] = True

    if int(keep.sum()) > 0.85 * n or int((~keep).sum()) < 20:
        idx = np.arange(n, dtype=np.int64)
        z = np.zeros(n, dtype=bool)
        return dict(
            idx=idx,
            reduced=False,
            fx=z,
            fy=z,
            fth=z,
            mx=z,
            my=z,
            mx_full=z,
            my_full=z,
            block_counts={
                "active": int(active.sum()),
                "BL": int((block_id == 1).sum()),
                "TR": int((block_id == 2).sum()),
                "TL": int((block_id == 3).sum()),
                "BR": int((block_id == 4).sum()),
                "kept": n,
            },
        )

    idx = np.flatnonzero(keep)
    return dict(
        idx=idx,
        reduced=True,
        fx=fx_full[idx],
        fy=fy_full[idx],
        fth=fth_full[idx],
        mx=mx_full[idx],
        my=my_full[idx],
        mx_full=mx_full,
        my_full=my_full,
        block_counts={
            "active": int(active.sum()),
            "BL": int((block_id == 1).sum()),
            "TR": int((block_id == 2).sum()),
            "TL": int((block_id == 3).sum()),
            "BR": int((block_id == 4).sum()),
            "kept": len(idx),
        },
    )


def run_squeeze_reduced(a: np.ndarray, L: float, K: int, model: dict | None = None, **kw) -> tuple[np.ndarray, float, float, int]:
    """Squeeze using the auto-detected reduced model; falls back to full if no rigid blocks."""
    if model is None:
        model = build_model(a, L)
    idx = model["idx"]
    sub = a[idx]
    f0 = L - K
    if model["reduced"]:
        f, Xn, Yn, Tn, v, it = squeeze(
            sub[:, 0].copy(),
            sub[:, 1].copy(),
            sub[:, 2].copy(),
            K,
            K,
            f0,
            frozen_x=model["fx"],
            frozen_y=model["fy"],
            frozen_th=model["fth"],
            move_x=model["mx"],
            move_y=model["my"],
            **kw,
        )
        full = a.copy()
        full[idx, 0] = Xn
        full[idx, 1] = Yn
        full[idx, 2] = Tn
        rest = np.ones(len(a), dtype=bool)
        rest[idx] = False
        shift = f - f0
        full[rest & model["mx_full"], 0] += shift
        full[rest & model["my_full"], 1] += shift
    else:
        f, Xn, Yn, Tn, v, it = squeeze(
            a[:, 0].copy(), a[:, 1].copy(), a[:, 2].copy(), K, K, f0, **kw
        )
        full = np.column_stack([Xn, Yn, Tn])
    return full, K + f, v, it


def run_squeeze_full(a: np.ndarray, L: float, K: int, **kw) -> tuple[np.ndarray, float, float, int]:
    """Full-row pslp squeeze where every square is free."""
    f0 = L - K
    f, Xn, Yn, Tn, v, it = squeeze(
        a[:, 0].copy(), a[:, 1].copy(), a[:, 2].copy(), K, K, f0, **kw
    )
    return np.column_stack([Xn, Yn, Tn]), K + f, v, it
