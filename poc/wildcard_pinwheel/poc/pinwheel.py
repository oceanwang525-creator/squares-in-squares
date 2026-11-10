"""Rational-slope four-block pinwheel construction (wildcard head, Nova).

Idea
----
Take four congruent ``a`` by ``b`` blocks of unit squares (``a`` squares long,
``b`` squares wide, ``a >= b``) and place their centres at the four diagonal
positions of a 4-fold symmetric pinwheel:

    centre_k = R(k*90 deg) @ (dx, dy),      angle_k = theta + k*90 deg

with ``theta = atan2(q, p)`` for a coprime integer slope ``(p, q)``.  The four
blocks are rigid: their inner unit squares keep the same rotation, so the union
is a true tilted pinwheel rather than a sheared grid.

Any space left inside the resulting axis-aligned bounding square is filled with
upright unit squares on the integer grid.  Total capacity is

    capacity = 4*a*b + (#upright cells that clear all four blocks).

The measured side is the translation-invariant bounding side

    L = max(max_x - min_x, max_y - min_y)

computed directly from the placed vertices (never from area).  This module only
builds float placements and does the *combinatorial* capacity bookkeeping; the
official metric is the only authority on L and feasibility (see ``run.py``).
"""
from __future__ import annotations

import math

import numpy as np

CLEARANCE = 1e-3  # required float gap for any pair that must not touch


def axes(theta: float) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.array([math.cos(theta), math.sin(theta)]),
        np.array([-math.sin(theta), math.cos(theta)]),
    )


def rect_rect_gap(c1, t1, a1, b1, c2, t2, a2, b2) -> float:
    """Separating-axis gap: >0 means a strictly positive hole between the rects."""
    u1, v1 = axes(t1)
    u2, v2 = axes(t2)
    d = np.asarray(c2) - np.asarray(c1)
    best = -1e18
    for ax in (u1, v1, u2, v2):
        r1 = 0.5 * (abs(ax @ u1) * a1 + abs(ax @ v1) * b1)
        r2 = 0.5 * (abs(ax @ u2) * a2 + abs(ax @ v2) * b2)
        best = max(best, abs(d @ ax) - r1 - r2)
    return float(best)


def make_blocks(a: int, b: int, theta: float, dx: float, dy: float):
    blocks = []
    for k in range(4):
        c, s = math.cos(k * math.pi / 2), math.sin(k * math.pi / 2)
        R = np.array([[c, -s], [s, c]])
        centre = R @ np.array([dx, dy])
        blocks.append((centre, theta + k * math.pi / 2, a, b))
    return blocks


def rect_corners(centre, theta, a, b):
    u, v = axes(theta)
    return [centre + su * (a / 2) * u + sv * (b / 2) * v
            for su in (-1.0, 1.0) for sv in (-1.0, 1.0)]


def bbox(blocks):
    pts = np.array([p for (c, t, a, b) in blocks for p in rect_corners(c, t, a, b)])
    return pts.min(0), pts.max(0)


def blocks_pairwise_ok(blocks) -> bool:
    n = len(blocks)
    for i in range(n):
        for j in range(i + 1, n):
            if rect_rect_gap(*blocks[i], *blocks[j]) < CLEARANCE:
                return False
    return True


def fill_cells(blocks, L: float):
    """Upright unit cells clearing every block by CLEARANCE, with min gap kept."""
    nc = int(math.floor(L + 1e-9))
    if nc <= 0:
        return []
    xs = np.arange(nc) + 0.5
    ys = np.arange(nc) + 0.5
    X, Y = np.meshgrid(xs, ys)
    pts = np.stack([X.ravel(), Y.ravel()], 1)
    gap = np.full(len(pts), 1e18)
    for (c, th, a, b) in blocks:
        u, v = axes(th)
        local = np.full(len(pts), -1e18)
        for ax in (u, v, np.array([1.0, 0.0]), np.array([0.0, 1.0])):
            r = 0.5 * (abs(ax @ u) * a + abs(ax @ v) * b)
            hc = 0.5 * (abs(ax[0]) + abs(ax[1]))
            local = np.maximum(local, np.abs((pts - c) @ ax) - r - hc)
        gap = np.minimum(gap, local)
    keep = gap > CLEARANCE
    return [(float(p[0]), float(p[1]), float(g)) for p, g in zip(pts[keep], gap[keep])]


def block_unit_cells(a: int, b: int, theta: float, centre):
    """The a*b unit squares of one rigid block, as (x, y, angle)."""
    u, v = axes(theta)
    out = []
    for i in range(a):
        for j in range(b):
            off = ((i + 0.5) - a / 2) * u + ((j + 0.5) - b / 2) * v
            c = centre + off
            out.append((float(c[0]), float(c[1]), float(theta)))
    return out


def build(a: int, b: int, p: int, q: int, d: float, family: str, n: int):
    """Build one pinwheel, returning (rows, L, info) or None if capacity < n.

    Rows are (x, y, angle) float centres; ``d`` places the four block centres at
    ``(+/-d, +/-d)`` (family ``corner``) or ``(+/-d, 0),(0, +/-d)`` (``side``).
    """
    theta = math.atan2(q, p)
    if family == "corner":
        dx, dy = d, d
    elif family == "side":
        dx, dy = d, 0.0
    else:
        raise ValueError(f"unknown family {family!r}")

    blocks = make_blocks(a, b, theta, dx, dy)
    if not blocks_pairwise_ok(blocks):
        return None

    lo, hi = bbox(blocks)
    L = float(max(hi - lo))
    blocks = [(c - lo, t, aa, bb) for (c, t, aa, bb) in blocks]

    cells = fill_cells(blocks, L)
    total = 4 * a * b + len(cells)
    if total < n or 4 * a * b > n:
        return None

    rows = []
    for (c, t, aa, bb) in blocks:
        rows.extend(block_unit_cells(aa, bb, t, c))

    need = n - len(rows)
    cells.sort(key=lambda z: z[2], reverse=True)  # loosest first
    for (x, y, _g) in cells[:need]:
        rows.append((x, y, 0.0))

    info = dict(L=L, capacity=total, blocks=4 * a * b, fill=len(cells),
                used_fill=need, a=a, b=b, p=p, q=q, d=d, family=family)
    return rows, L, info
