"""Bent-band constructions for n = k^2 + D, D near k.

Two self-contained builders are provided:

* ``monolithic_45`` -- the exact four-staircase / 45-degree band family used as
  the "monolithic tilted band" baseline.  It is a copy of the placement rule
  used by the Pell-17 work, with the exact capacity arithmetic made explicit.
* ``bent_two_block`` -- the bent variant under test: two lattice blocks at
  independent angles t1, t2 plus corner-anchored upright fills.  The two blocks
  are anchored in opposite corners and their overlap is removed exactly, so the
  emitted point set is always a valid (non-overlapping) set of unit squares.

Nothing here reads incumbent coordinates.  All geometry is synthetic.
"""
from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "poc/baseline/poc/src"))
from squares.core import certify  # noqa: E402

S2 = math.sqrt(2.0)
INV_S2 = 1.0 / S2


# --------------------------------------------------------------------------- #
# exact contact helpers
# --------------------------------------------------------------------------- #
def _paired_overlap(a: np.ndarray, b: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """Separating-axis test for pairs (a[i], b[i]) of unit squares."""
    if not len(a):
        return np.zeros(0, dtype=bool)
    ua = np.column_stack((np.cos(a[:, 2]), np.sin(a[:, 2])))
    va = np.column_stack((-ua[:, 1], ua[:, 0]))
    ub = np.column_stack((np.cos(b[:, 2]), np.sin(b[:, 2])))
    vb = np.column_stack((-ub[:, 1], ub[:, 0]))
    d = a[:, :2] - b[:, :2]
    bad = np.ones(len(a), dtype=bool)
    for axis in (ua, va, ub, vb):
        r = sum(np.abs((axis * v).sum(1)) for v in (ua, va, ub, vb)) / 2.0
        bad &= np.abs((d * axis).sum(1)) < r - tol
    return bad


def _conflicts(q: np.ndarray) -> np.ndarray:
    if len(q) < 2:
        return np.zeros((0, 2), dtype=int)
    pairs = np.array(list(cKDTree(q[:, :2]).query_pairs(1.415)), dtype=int).reshape(-1, 2)
    if not len(pairs):
        return pairs
    return pairs[_paired_overlap(q[pairs[:, 0]], q[pairs[:, 1]])]


def _inside(q: np.ndarray, side: float, tol: float = 1e-12) -> np.ndarray:
    h = 0.5 * (np.abs(np.cos(q[:, 2])) + np.abs(np.sin(q[:, 2])))
    return (
        (q[:, 0] >= h - tol)
        & (q[:, 1] >= h - tol)
        & (q[:, 0] <= side - h + tol)
        & (q[:, 1] <= side - h + tol)
    )


def extent(q: np.ndarray) -> float:
    h = 0.5 * (np.abs(np.cos(q[:, 2])) + np.abs(np.sin(q[:, 2])))
    lo = np.min(q[:, :2] - h[:, None], axis=0)
    hi = np.max(q[:, :2] + h[:, None], axis=0)
    return float(np.max(hi - lo))


def _normalize(q: np.ndarray, margin: float = 0.0) -> np.ndarray:
    q = q.copy()
    h = 0.5 * (np.abs(np.cos(q[:, 2])) + np.abs(np.sin(q[:, 2])))
    lo = np.min(q[:, :2] - h[:, None], axis=0)
    q[:, :2] -= lo - margin
    return q


# --------------------------------------------------------------------------- #
# monolithic four-staircase / 45-degree band (exact baseline family)
# --------------------------------------------------------------------------- #
def _triangle(a: int, x0: float, y0: float, sx: int, sy: int) -> list[list[float]]:
    """Staircase triangle of ``a`` rows anchored at (x0, y0)."""
    out = []
    for i in range(a):
        for j in range(a - i):
            out.append([x0 + sx * i, y0 + sy * j, 0.0])
    return out


def monolithic_45(w: int, l: int, a1: int, a2: int, b1: int, b2: int):
    """Four upright corner staircases + one central 45-degree band.

    Side length is ``L = (a1 + a2) / 2 + 1 + w / sqrt(2)``.  Capacity is
    ``tri(a1)+tri(a2)+tri(b1)+tri(b2)+w*l`` with ``tri(n)=n(n+1)/2``; two corner
    cells are dropped when a staircase crosses a wall (see ``drop`` below).
    Returns ``(points, L)`` with points an (n,3) array of ``x, y, angle``.
    """
    L = 0.5 * (a1 + a2) + 1.0 + w * INV_S2
    q: list[list[float]] = []
    drop = 0
    # SE (a1), NW (a2): the two flanks of the band.
    for i in range(a1):
        for j in range(a1 - i):
            if i == a1 - 1 and j == 0 and a1 + b1 > L + 1e-9:
                drop += 1
                continue
            if i == 0 and j == a1 - 1 and a1 + b2 > L + 1e-9:
                drop += 1
                continue
            q.append([L - 0.5 - i, 0.5 + j, 0.0])
    for i in range(a2):
        for j in range(a2 - i):
            if i == 0 and j == a2 - 1 and a2 + b1 > L + 1e-9:
                drop += 1
                continue
            if i == a2 - 1 and j == 0 and a2 + b2 > L + 1e-9:
                drop += 1
                continue
            q.append([0.5 + i, L - 0.5 - j, 0.0])
    q.extend(_triangle(b1, 0.5, 0.5, +1, +1))
    q.extend(_triangle(b2, L - 0.5, L - 0.5, -1, -1))

    v_start = -0.5 * (w - 1.0) - 0.5 * (a1 - a2) * INV_S2
    u_wall = 0.5 * (w - 1.0) + 0.5 * abs(a1 - a2) * INV_S2 + INV_S2
    u_lo = max((b1 + 1.0) * INV_S2 + 0.5, u_wall)
    u_hi = S2 * L - max((b2 + 1.0) * INV_S2 + 0.5, u_wall)
    u_cols = np.linspace(u_lo, u_hi, l) if l > 1 else np.array([0.5 * (u_lo + u_hi)])
    for uc in u_cols:
        for s in range(w):
            v = v_start + s
            q.append([(uc + v) * INV_S2, (uc - v) * INV_S2, math.pi / 4.0])
    return np.array(q, dtype=float), L, drop


def monolithic_capacity(a1: int, a2: int, b1: int, b2: int, w: int, l: int) -> int:
    tri = lambda n: n * (n + 1) // 2
    return tri(a1) + tri(a2) + tri(b1) + tri(b2) + w * l


# --------------------------------------------------------------------------- #
# bent two-block variant
# --------------------------------------------------------------------------- #
def _block(l: int, w: int, theta: float, anchor: np.ndarray, sign: int) -> np.ndarray:
    u = np.array([math.cos(theta), math.sin(theta)])
    v = np.array([-math.sin(theta), math.cos(theta)])
    ii, jj = np.meshgrid(np.arange(l), np.arange(w), indexing="ij")
    c = (
        anchor
        + sign * (ii.ravel()[:, None] + 0.5) * u
        + (jj.ravel()[:, None] - (w - 1) / 2.0) * v
    )
    return np.column_stack((c, np.full(len(c), theta)))


def _corner_fill(q: np.ndarray, side: float) -> np.ndarray:
    """Add upright unit squares from the four corner-anchored half-grids."""
    k = int(math.floor(side + 1e-9))
    ii, jj = np.meshgrid(np.arange(k), np.arange(k))
    base = np.column_stack(((ii + 0.5).ravel(), (jj + 0.5).ravel(), np.zeros(k * k)))
    cur = q.copy()
    added = []
    for fx, fy in itertools.product((0, 1), repeat=2):
        sites = base.copy()
        if fx:
            sites[:, 0] = side - sites[:, 0]
        if fy:
            sites[:, 1] = side - sites[:, 1]
        sites = sites[_inside(sites, side)]
        if not len(sites):
            continue
        if len(cur):
            near = cKDTree(cur[:, :2]).query_ball_point(sites[:, :2], 1.415)
            ai, bi = [], []
            for s, ids in enumerate(near):
                for i in ids:
                    ai.append(s)
                    bi.append(i)
            bad = np.zeros(len(sites), dtype=bool)
            if ai:
                ai = np.array(ai)
                bi = np.array(bi)
                bad[ai[_paired_overlap(sites[ai], cur[bi])]] = True
            sites = sites[~bad]
        if len(sites):
            added.append(sites)
            cur = np.vstack([cur, sites])
    return np.vstack(added) if added else np.zeros((0, 3))


def bent_two_block(L: float, t1: float, t2: float, w1: int, w2: int, l1: int, l2: int):
    """Two tilted blocks from opposite corners + corner-anchored upright fill.

    Block A grows from (0, 0) along ``t1``; block B grows from (L, L) backwards
    along ``t2``.  Overlap between the blocks is removed from B deterministically.
    Returns ``(points, stats)``; capacity is ``len(points)``.
    """
    a = _block(l1, w1, t1, np.array([0.0, 0.0]), +1)
    b = _block(l2, w2, t2, np.array([L, L]), -1)
    a = a[_inside(a, L)]
    b = b[_inside(b, L)]
    dropped = 0
    if len(a) and len(b):
        near = cKDTree(a[:, :2]).query_ball_point(b[:, :2], 1.415)
        ai, bi = [], []
        for s, ids in enumerate(near):
            for i in ids:
                ai.append(i)
                bi.append(s)
        bad = np.zeros(len(b), dtype=bool)
        if ai:
            ai = np.array(ai)
            bi = np.array(bi)
            bad[bi[_paired_overlap(a[ai], b[bi])]] = True
        dropped = int(bad.sum())
        b = b[~bad]
    band = np.vstack([a, b]) if len(b) else a
    fill = _corner_fill(band, L)
    pts = np.vstack([band, fill]) if len(fill) else band
    return pts, dict(block_a=len(a), block_b=len(b), dropped=dropped, fill=len(fill))


# --------------------------------------------------------------------------- #
# output / certification
# --------------------------------------------------------------------------- #
def to_rows(q: np.ndarray, dilate: float = 1.00000002, decimals: int = 17) -> list[dict]:
    """Center-dilate by ``dilate`` (adds positive clearance) and format strings."""
    q = _normalize(q)
    h = 0.5 * (np.abs(np.cos(q[:, 2])) + np.abs(np.sin(q[:, 2])))
    lo = np.min(q[:, :2] - h[:, None], axis=0)
    hi = np.max(q[:, :2] + h[:, None], axis=0)
    centre = 0.5 * (lo + hi)
    q = q.copy()
    q[:, :2] = (q[:, :2] - centre) * dilate + centre
    fmt = "%." + str(decimals) + "f"
    return [
        dict(square_id=str(i), x=fmt % q[i, 0], y=fmt % q[i, 1], angle=fmt % q[i, 2])
        for i in range(len(q))
    ]


def csv_text(n: int, rows: list[dict]) -> str:
    lines = ["id,n,square_id,x,y,angle"]
    for i, row in enumerate(rows):
        lines.append(
            f"{i},{n},{row['square_id']},{row['x']},{row['y']},{row['angle']}"
        )
    return "\n".join(lines) + "\n"


def certify_literal(n: int, rows: list[dict]) -> str:
    return certify(n, rows)


def certify_pandas(n: int, rows: list[dict]) -> str:
    import io

    import pandas as pd

    text = csv_text(n, rows)
    frame = pd.read_csv(io.StringIO(text))
    back = frame.to_csv(index=False)
    transported = pd.read_csv(io.StringIO(back))
    return certify(
        n,
        [
            dict(
                square_id=str(r.square_id),
                x=str(r.x),
                y=str(r.y),
                angle=str(r.angle),
            )
            for r in transported.itertuples(index=False)
        ],
    )
