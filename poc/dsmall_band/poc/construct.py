"""dsmall_band constructor: upright grid + one free-angle tilted block/band.

Family (exact lattice geometry, float64 search, Decimal certification later):

  * a rotated lattice block of ``m x w`` unit squares at angle ``theta``
    (centres ``c0 + k*e1 + r*e2``, pitch 1 in both rotated directions, so the
    block is internally overlap free by construction and exactly tiles a
    rotated ``m x w`` rectangle),
  * an upright unit grid anchored at the box's bottom-left corner: every
    integer cell ``[i,i+1] x [j,j+1]`` that does not overlap the block,
  * corner/ring caps: extra squares at ``theta`` on the lattice ring around the
    block and single caps at 45 deg in the four corner voids, inserted greedily.

Count = block + upright cells + caps, all mutual overlaps checked exactly with
SAT.  The only free parameters are (theta, m, w, block bbox origin bx, by);
the upright grid is not allowed to move (an upright grid is only dense when it
is flush with the box corner).
"""
from __future__ import annotations

import math

import numpy as np

from geom import conflict_mask, half_extent, inside, side


def _block(theta: float, m: int, w: int, bx: float, by: float, L: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    e1 = np.array([c, s])
    e2 = np.array([-s, c])
    h = 0.5 * (c + s)
    c0 = np.array([bx + h + (w - 1) * s, by + h])
    ks = np.arange(m)
    rs = np.arange(w)
    K, R = np.meshgrid(ks, rs, indexing="ij")
    K = K.ravel()
    R = R.ravel()
    P = np.empty((len(K), 3))
    P[:, 0] = c0[0] + K * e1[0] + R * e2[0]
    P[:, 1] = c0[1] + K * e1[1] + R * e2[1]
    P[:, 2] = theta
    return P


def _axis_grid(ncol: int) -> np.ndarray:
    ii, jj = np.meshgrid(np.arange(ncol), np.arange(ncol), indexing="ij")
    P = np.empty((ncol * ncol, 3))
    P[:, 0] = ii.ravel() + 0.5
    P[:, 1] = jj.ravel() + 0.5
    P[:, 2] = 0.0
    return P


def _cells_blocked(block: np.ndarray, ncol: int) -> np.ndarray:
    """Boolean mask over the ncol*ncol integer cells of conflicts with `block`."""
    mask = np.zeros(ncol * ncol, dtype=bool)
    if len(block) == 0:
        return mask
    th = 0.5 + half_extent(block[0, 2])          # 0.5 * (1 + g)
    cx, cy = block[:, 0], block[:, 1]
    lo_i = np.floor(cx - th).astype(int)
    lo_j = np.floor(cy - th).astype(int)
    off = np.arange(5)
    I = lo_i[:, None, None] + off[None, :, None]
    J = lo_j[:, None, None] + off[None, None, :]
    ok = (I >= 0) & (I < ncol) & (J >= 0) & (J < ncol)
    px = I + 0.5
    py = J + 0.5
    dx = px - cx[:, None, None] + np.zeros_like(py)
    dy = py - cy[:, None, None] + np.zeros_like(px)
    c, s = math.cos(block[0, 2]), math.sin(block[0, 2])
    hit = np.abs(dx) < th
    hit &= np.abs(dy) < th
    hit &= np.abs(dx * c + dy * s) < th
    hit &= np.abs(-dx * s + dy * c) < th
    hit &= ok
    if hit.any():
        ids = (I * ncol + J)[hit]          # _axis_grid id convention: i*ncol + j
        mask[ids] = True
    return mask


def build(theta: float, m: int, w: int, bx: float, by: float, L: float,
          ncol: int | None = None, caps: bool = True) -> tuple[int, np.ndarray]:
    """Return (count, packing) for one parameter set."""
    if ncol is None:
        ncol = int(math.floor(L + 1e-9))
    blk = _block(theta, m, w, bx, by, L)
    blk = blk[inside(blk, L)]
    cells = _axis_grid(ncol)
    blocked = _cells_blocked(blk, ncol)
    cells = cells[~blocked]
    cur = np.vstack([blk, cells])
    if not caps:
        return len(cur), cur

    # ---- ring caps at theta (grow the block outward where the box allows) ----
    c, s = math.cos(theta), math.sin(theta)
    e1 = np.array([c, s])
    e2 = np.array([-s, c])
    h = 0.5 * (c + s)
    c0 = np.array([bx + h + (w - 1) * s, by + h])
    for k in range(-2, m + 2):
        for r in range(-2, w + 2):
            if 0 <= k < m and 0 <= r < w:
                continue
            p = np.array([[c0[0] + k * e1[0] + r * e2[0],
                           c0[1] + k * e1[1] + r * e2[1], theta]])
            if not inside(p, L)[0]:
                continue
            if conflict_mask(p, cur).any():
                continue
            cur = np.vstack([cur, p])

    # ---- 45 deg single corner caps ----
    for cap in ([0.0, 0.0, math.pi / 4], [L, 0.0, math.pi / 4],
                [0.0, L, math.pi / 4], [L, L, math.pi / 4],
                [0.0, 0.0, -math.pi / 4], [L, 0.0, -math.pi / 4],
                [0.0, L, -math.pi / 4], [L, L, -math.pi / 4]):
        p = np.array([cap])
        if not inside(p, L)[0]:
            continue
        if conflict_mask(p, cur).any():
            continue
        cur = np.vstack([cur, p])
    return len(cur), cur
