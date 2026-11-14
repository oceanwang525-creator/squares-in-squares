"""Exact-ish float64 geometry helpers for the dsmall_band constructor.

Conventions
-----------
A square is (x, y, ang) with unit side, centre (x, y), rotated by ``ang`` radians.
The half-extent of a unit square along the x and y axes is
``h(ang) = (|cos ang| + |sin ang|) / 2``.

SAT for two *unit* squares only needs the four axis directions of the two
squares, and for every one of them the projection radii sum to
``0.5 * (1 + g)`` with ``g = |cos(a-b)| + |sin(a-b)|`` -- so the overlap test is
``|d.n| < 0.5*(1+g)`` on all four axes.
"""
from __future__ import annotations

import math

import numpy as np

HALF = 0.5


def half_extent(ang: float) -> float:
    return 0.5 * (abs(math.cos(ang)) + abs(math.sin(ang)))


def bbox(P: np.ndarray) -> tuple[float, float, float, float]:
    """P is (N,3) x, y, ang.  Returns xmin, ymin, xmax, ymax."""
    h = 0.5 * (np.abs(np.cos(P[:, 2])) + np.abs(np.sin(P[:, 2])))
    return (P[:, 0] - h).min(), (P[:, 1] - h).min(), (P[:, 0] + h).max(), (P[:, 1] + h).max()


def side(P: np.ndarray) -> float:
    x0, y0, x1, y1 = bbox(P)
    return max(x1 - x0, y1 - y0)


def inside(P: np.ndarray, L: float, tol: float = 1e-9) -> np.ndarray:
    h = 0.5 * (np.abs(np.cos(P[:, 2])) + np.abs(np.sin(P[:, 2])))
    return ((P[:, 0] - h) >= -tol) & ((P[:, 1] - h) >= -tol) & \
           ((P[:, 0] + h) <= L + tol) & ((P[:, 1] + h) <= L + tol)


def conflict_mask(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """A (Na,3), B (Nb,3).  Returns (Na,Nb) bool: True where the squares overlap."""
    if len(A) == 0 or len(B) == 0:
        return np.zeros((len(A), len(B)), dtype=bool)
    d = A[:, None, :2] - B[None, :, :2]
    da = A[:, 2][:, None]
    db = B[:, 2][None, :]
    g = 0.5 * (1.0 + np.abs(np.cos(da - db)) + np.abs(np.sin(da - db)))
    ca, sa = np.cos(da), np.sin(da)
    cb, sb = np.cos(db), np.sin(db)
    axes = [(ca, sa), (-sa, ca), (cb, sb), (-sb, cb)]
    out = np.ones(d.shape[:2], dtype=bool)
    for ax, ay in axes:
        proj = np.abs(d[:, :, 0] * ax + d[:, :, 1] * ay)
        out &= proj < g - 1e-12
    return out


def conflicts_with(A: np.ndarray, B: np.ndarray) -> bool:
    return bool(conflict_mask(A, B).any())


def count_valid(P: np.ndarray) -> int:
    """Number of squares in P keeping non-overlap (greedy, first-come)."""
    keep = []
    for i in range(len(P)):
        if not keep:
            keep.append(i)
            continue
        k = keep
        if not conflict_mask(P[i:i + 1], P[k]).any():
            keep.append(i)
    return len(keep)
