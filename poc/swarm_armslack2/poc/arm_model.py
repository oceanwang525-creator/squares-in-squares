"""4-corner auto-partitioning and arm-adjacent 0-deg layer liberation for H125 (armslack2)."""
from __future__ import annotations

import math
import numpy as np
from scipy.sparse.csgraph import connected_components

from pslp import feas, squeeze


def normalize_arr(a: np.ndarray) -> tuple[np.ndarray, float]:
    """Shift a=(N,3) [x, y, theta] so min(x-e)=0, min(y-e)=0, return (a_norm, L)."""
    out = np.asarray(a, dtype=np.float64).copy()
    e = 0.5 * (np.abs(np.cos(out[:, 2])) + np.abs(np.sin(out[:, 2])))
    out[:, 0] -= float((out[:, 0] - e).min())
    out[:, 1] -= float((out[:, 1] - e).min())
    L = float(max((out[:, 0] + e).max(), (out[:, 1] + e).max()))
    return out, L


def full_viol(full: np.ndarray, L: float) -> float:
    """Return max pair/wall violation (>= 0) for full state a=(N,3) in [0, L]^2."""
    mp, mw = feas(full[:, 0].copy(), full[:, 1].copy(), full[:, 2].copy(), L, L)
    return float(max(0.0, -min(mp, mw)))


def _min_dist_to_pts(x: np.ndarray, y: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Chunked Euclidean distance from each (x[i], y[i]) to nearest point in pts."""
    n = len(x)
    if len(pts) == 0:
        return np.full(n, 1e9, dtype=np.float64)
    out = np.full(n, 1e9, dtype=np.float64)
    for i in range(0, n, 256):
        out[i : i + 256] = np.min(
            np.hypot(x[i : i + 256, None] - pts[None, :, 0], y[i : i + 256, None] - pts[None, :, 1]),
            axis=1,
        )
    return out


def cluster_tilted_arms(
    a: np.ndarray,
    tilt_deg: float = 2.0,
    link_radius: float = 1.9,
) -> list[np.ndarray]:
    """Cluster strongly tilted squares into spatially connected arms/bricks (sorted largest-first)."""
    th = a[:, 2]
    deg_mod = np.degrees(np.mod(th, math.pi / 2.0))
    til_idx = np.flatnonzero(np.minimum(deg_mod, 90.0 - deg_mod) > tilt_deg)
    if len(til_idx) == 0:
        return []
    px, py = a[til_idx, 0], a[til_idx, 1]
    adj = ((px[:, None] - px[None, :]) ** 2 + (py[:, None] - py[None, :]) ** 2) < (link_radius ** 2)
    n_comp, labels = connected_components(adj, directed=False)
    clusters = [til_idx[labels == c] for c in range(n_comp)]
    clusters.sort(key=lambda idx: -len(idx))
    return clusters


def identify_interface_layers(
    a: np.ndarray,
    layer1_r: float = 1.65,
    layer2_r: float = 2.75,
    tilt_deg: float = 2.0,
) -> dict[str, np.ndarray]:
    """Identify tilted squares and 1-2 layers of arm-adjacent 0-deg boundary squares."""
    th = a[:, 2]
    deg_mod = np.degrees(np.mod(th, math.pi / 2.0))
    ang_err = np.minimum(deg_mod, 90.0 - deg_mod)
    tilted = ang_err > tilt_deg
    slight = (ang_err > 0.2) & (~tilted)
    zero = ~tilted & (~slight)
    d_til = _min_dist_to_pts(a[:, 0], a[:, 1], a[tilted, :2])
    layer1 = zero & (d_til <= layer1_r)
    layer2 = zero & (d_til <= layer2_r)
    return {
        "tilted": tilted,
        "slight": slight,
        "zero": zero,
        "layer1": layer1,
        "layer2": layer2,
        "d_til": d_til,
    }


def build_model(
    a: np.ndarray,
    L: float,
    act_reach: float = 3.2,
    layer: float = 1.6,
    wall_tol: float = 0.25,
    min_comp: int = 4,
    max_keep_frac: float = 0.88,
) -> dict:
    """Partition squares into free arm+boundary squares and 4 corner rigid 0-deg grid blocks."""
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

    d_til = _min_dist_to_pts(x, y, a[til, :2])
    near_zero = np.minimum(deg_mod, 90.0 - deg_mod) <= 0.2
    active = (d_til < act_reach) | (~near_zero)

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

            if (wl and wr) or (wb and wt):
                active[m] = True
            elif wl and wb:
                block_id[m] = 1
            elif wr and wt:
                block_id[m] = 2
            elif wl and wt:
                block_id[m] = 3
            elif wr and wb:
                block_id[m] = 4
            else:
                active[m] = True

    fx_full = (block_id == 1) | (block_id == 3)
    fy_full = (block_id == 1) | (block_id == 4)
    mx_full = (block_id == 2) | (block_id == 4)
    my_full = (block_id == 2) | (block_id == 3)
    fth_full = block_id != 0

    dl = _min_dist_to_pts(x, y, a[active, :2])
    keep = active | (dl < layer)

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

    if int(keep.sum()) > max_keep_frac * n or int(np.sum(block_id != 0)) < 8:
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

    idx = np.flatnonzero(keep)
    counts = {
        "active": int(np.sum(block_id == 0)),
        "BL": int(np.sum(block_id == 1)),
        "TR": int(np.sum(block_id == 2)),
        "TL": int(np.sum(block_id == 3)),
        "BR": int(np.sum(block_id == 4)),
        "lp_n": int(len(idx)),
    }
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
        block_counts=counts,
    )


def run_squeeze_reduced(
    a: np.ndarray,
    L: float,
    K: int,
    model: dict | None = None,
    act_reach: float = 3.2,
    **kw,
) -> tuple[np.ndarray, float, float, int]:
    """Run pslp.squeeze on reduced 4-corner model (or full model if not reducible)."""
    if model is None:
        model = build_model(a, L, act_reach=act_reach)
    f0 = L - K
    if model["reduced"]:
        idx = model["idx"]
        f, Xn, Yn, Tn, v, it = squeeze(
            a[idx, 0].copy(),
            a[idx, 1].copy(),
            a[idx, 2].copy(),
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
