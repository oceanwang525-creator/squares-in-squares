"""dsmall_band family v2: two corner-anchored upright staircase slabs + one
free-angle tilted block in the channel between them (+ corner caps).

Why v2: v1 (a single bottom-left-anchored upright grid) can never beat the plain
ncol x ncol grid, because a tilted square that overlaps a grid cell costs that
whole cell.  The tracked D=1 rows reach c = L-K = 0.5208 with the *split*
layout: cells in the bottom-left triangle are anchored at the box's bottom-left
corner, cells in the top-right triangle are anchored at the box's top-right
corner (so they are shifted by L-ncol), and the 45 deg block lives in the
diagonal channel that the two slabs leave free.

Parameters
----------
theta      tilt of the block (free, 4..86 deg)
m, w       rotated m x w lattice block (pitch 1 in both rotated directions)
bx, by     block bbox lower-left corner inside the box
t, tp      staircase legs of the BL slab (cells i+j <= t) and the mirrored TR
           slab (cells (ncol-1-i)+(ncol-1-j) <= tp); t+tp <= 2*ncol-3 keeps the
           two slabs mutually disjoint.
"""
from __future__ import annotations

import math

import numpy as np

from construct import _block, _axis_grid, _cells_blocked
from geom import conflict_mask, half_extent, inside


def slab_present(ncol: int, t: int, tp: int) -> np.ndarray:
    """Boolean (ncol*ncol,) over integer cells (i + j*ncol)."""
    ii, jj = np.meshgrid(np.arange(ncol), np.arange(ncol), indexing="ij")
    bl = (ii + jj) <= t
    tr = ((ncol - 1 - ii) + (ncol - 1 - jj)) <= tp
    present = bl | tr
    # slabs must be disjoint; assert-free guard
    if (bl & tr).any():
        raise ValueError("slabs overlap: t+tp too large")
    return present.ravel()


def build2(theta: float, m: int, w: int, bx: float, by: float, L: float,
           t: int | None = None, tp: int | None = None,
           caps: bool = False) -> tuple[int, np.ndarray]:
    ncol = int(math.floor(L + 1e-9))
    if t is None:
        t = ncol - 2
    if tp is None:
        tp = ncol - 2
    present = slab_present(ncol, t, tp)
    blk = _block(theta, m, w, bx, by, L)
    blk = blk[inside(blk, L)]
    blocked = _cells_blocked(blk, ncol)
    cells = _axis_grid(ncol)
    keep = present & ~blocked
    cur = np.vstack([blk, cells[keep]])
    if not caps:
        return len(cur), cur

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
