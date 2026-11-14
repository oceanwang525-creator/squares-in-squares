"""dsmall_band family v3: quartered upright grid (each quadrant anchored to its
own box corner) + one free-angle tilted block.

This is the layout the tracked D=1 rows actually use: the upright cells are a
ncol x ncol grid whose columns/rows are split at (ic, jc); the left/top
quadrants sit on the integer lattice, the right/bottom quadrants are shifted by
``d = L - ncol`` so they stay flush with the far box edges.  The shift leaves a
cross-shaped free channel of width ``d`` through the box, and the tilted block
is free to poke into it -- which is how the layout gains squares *above*
``ncol**2`` at all (a single-lattice grid can never be beaten: every tilted
square that touches a cell costs that whole cell).

Parameters: theta, m, w, bx, by (block), ic, jc (split column/row).
"""
from __future__ import annotations

import math

import numpy as np

from construct import _block, _axis_grid
from geom import conflict_mask, half_extent, inside


def _cells_blocked_off(blk: np.ndarray, ncol: int, ox: float, oy: float) -> np.ndarray:
    mask = np.zeros(ncol * ncol, dtype=bool)
    if len(blk) == 0:
        return mask
    th = 0.5 + half_extent(blk[0, 2])
    cx = blk[:, 0] - ox
    cy = blk[:, 1] - oy
    lo_i = np.floor(cx - th).astype(int)
    lo_j = np.floor(cy - th).astype(int)
    off = np.arange(5)
    I = lo_i[:, None, None] + off[None, :, None]
    J = lo_j[:, None, None] + off[None, None, :]
    ok = (I >= 0) & (I < ncol) & (J >= 0) & (J < ncol)
    dx = (I + 0.5) - cx[:, None, None] + np.zeros_like(J, dtype=float)
    dy = (J + 0.5) - cy[:, None, None] + np.zeros_like(I, dtype=float)
    c, s = math.cos(blk[0, 2]), math.sin(blk[0, 2])
    hit = np.abs(dx) < th
    hit &= np.abs(dy) < th
    hit &= np.abs(dx * c + dy * s) < th
    hit &= np.abs(-dx * s + dy * c) < th
    hit &= ok
    if hit.any():
        mask[(I * ncol + J)[hit]] = True   # id convention: i*ncol + j
    return mask


def quadrant_masks(ncol: int, ic: int, jc: int):
    ii, jj = np.meshgrid(np.arange(ncol), np.arange(ncol), indexing="ij")
    left = (ii < ic).ravel()
    bottom = (jj < jc).ravel()
    return dict(q00=left & bottom, q10=(~left) & bottom,
                q01=left & (~bottom), q11=(~left) & (~bottom))


def build3(theta: float, m: int, w: int, bx: float, by: float, L: float,
           ic: int, jc: int, caps: bool = False) -> tuple[int, np.ndarray]:
    ncol = int(math.floor(L + 1e-9))
    d = L - ncol
    blk = _block(theta, m, w, bx, by, L)
    blk = blk[inside(blk, L)]
    qm = quadrant_masks(ncol, ic, jc)
    ii, jj = np.meshgrid(np.arange(ncol), np.arange(ncol), indexing="ij")
    ii = ii.ravel()
    jj = jj.ravel()
    x = ii + 0.5
    y = jj + 0.5
    x = np.where(qm["q10"] | qm["q11"], x + d, x)
    y = np.where(qm["q01"] | qm["q11"], y + d, y)
    cells = np.stack([x, y, np.zeros_like(x)], axis=1)
    # drop cells a shifted quadrant pushes outside the box (only when d>... )
    cells = cells[inside(cells, L)]

    blocked = np.zeros(ncol * ncol, dtype=bool)
    for (ox, oy), key in ((((0.0, 0.0), "q00")), ((d, 0.0), "q10"),
                          ((0.0, d), "q01"), ((d, d), "q11")):
        mk = qm[key]
        if not mk.any():
            continue
        blocked |= _cells_blocked_off(blk, ncol, ox, oy) & mk
    keep = ~blocked
    cur = np.vstack([blk, cells[keep]])
    if not caps:
        return len(cur), cur

    c, s = math.cos(theta), math.sin(theta)
    e1 = np.array([c, s])
    e2 = np.array([-s, c])
    h = 0.5 * (c + s)
    c0 = np.array([bx + h + (w - 1) * s, by + h])
    for k in range(-3, m + 3):
        for r in range(-3, w + 3):
            if 0 <= k < m and 0 <= r < w:
                continue
            p = np.array([[c0[0] + k * e1[0] + r * e2[0],
                           c0[1] + k * e1[1] + r * e2[1], theta]])
            if not inside(p, L)[0]:
                continue
            if conflict_mask(p, cur).any():
                continue
            cur = np.vstack([cur, p])
    for ang in (math.pi / 4, -math.pi / 4):
        for cx in (0.0, L):
            for cy in (0.0, L):
                p = np.array([[cx, cy, ang]])
                if not inside(p, L)[0]:
                    continue
                if conflict_mask(p, cur).any():
                    continue
                cur = np.vstack([cur, p])
    return len(cur), cur
