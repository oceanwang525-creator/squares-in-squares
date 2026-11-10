"""Bounded discrete capacity search for the monolithic 45-degree band family.

The search is deliberately small (a few thousand candidate parameter tuples).
It answers two questions for each target n:

1. What is the smallest side length L for which an exact-capacity
   ``monolithic_45`` configuration exists?
2. What is the largest capacity the same family can reach at a fixed side L
   (used to measure the arithmetic shortfall at the incumbent side)?
"""
from __future__ import annotations

import math

from bent_band import INV_S2, S2, monolithic_45, monolithic_capacity


def _tri(n: int) -> int:
    return n * (n + 1) // 2


def _tinv(r: int):
    if r < 0:
        return None
    b = int((math.isqrt(8 * r + 1) - 1) // 2)
    return b if _tri(b) == r else None


def feasible(w: int, l: int, a1: int, a2: int, b1: int, b2: int) -> float | None:
    """Return L when the placement is wall-consistent, else None."""
    L = 0.5 * (a1 + a2) + 1.0 + w * INV_S2
    if max(a1, a2) + max(b1, b2) > L + 1e-9:
        return None
    u_wall = 0.5 * (w - 1.0) + 0.5 * abs(a1 - a2) * INV_S2 + INV_S2
    u_lo = max((b1 + 1.0) * INV_S2 + 0.5, u_wall)
    u_hi = S2 * L - max((b2 + 1.0) * INV_S2 + 0.5, u_wall)
    if l - 1 > u_hi - u_lo + 1e-9:
        return None
    v_start = -0.5 * (w - 1.0) - 0.5 * (a1 - a2) * INV_S2
    if u_lo + v_start < 1 - 1e-9:
        return None
    if u_hi + v_start + (w - 1) > S2 * L - 1 + 1e-9:
        return None
    if u_lo - v_start - (w - 1) < 1 - 1e-9:
        return None
    if u_hi - v_start > S2 * L - 1 + 1e-9:
        return None
    return L


def best_exact(
    n: int,
    wmax: int = 22,
    amax: int = 42,
    bmax: int = 14,
    lspan: int = 18,
    limit: int | None = None,
):
    """Minimal-L parameter tuple with exact capacity n, verified by the builder."""
    best = None
    tried = 0
    for w in range(1, wmax + 1):
        for a1 in range(0, amax):
            for a2 in range(0, amax):
                ta = _tri(a1) + _tri(a2)
                if ta >= n:
                    break
                for b1 in range(0, bmax):
                    for b2 in range(0, bmax):
                        R = n - ta - _tri(b1) - _tri(b2)
                        if R <= 0:
                            break
                        if R % w:
                            continue
                        l = R // w
                        if l < 1 or l > w + lspan:
                            continue
                        L = feasible(w, l, a1, a2, b1, b2)
                        if L is None:
                            continue
                        tried += 1
                        if limit is not None and tried > limit:
                            return best, tried
                        if best is None or L < best[0] - 1e-12:
                            q, Lb, drop = monolithic_45(w, l, a1, a2, b1, b2)
                            if len(q) != n or drop:
                                continue
                            best = (Lb, w, l, a1, a2, b1, b2)
    return best, tried


def max_capacity_at(Linc: float, wmax: int = 24, amax: int = 56, bmax: int = 22,
                    lmax: int = 58):
    """Largest capacity of the family with side length <= ``Linc``."""
    best_cap = 0
    best = None
    for w in range(1, wmax + 1):
        for A in range(0, amax):
            L = 0.5 * A + 1.0 + w * INV_S2
            if L > Linc + 1e-12:
                continue
            for a1 in range(0, A + 1):
                a2 = A - a1
                m = max(a1, a2)
                bb = int(L - m + 1e-9)
                if bb < 0:
                    continue
                for b1 in range(0, min(bmax, bb) + 1):
                    for b2 in range(0, min(bmax, bb) + 1):
                        ta = _tri(a1) + _tri(a2) + _tri(b1) + _tri(b2)
                        for l in range(1, lmax + 1):
                            if feasible(w, l, a1, a2, b1, b2) is None:
                                break
                            cap = ta + w * l
                            if cap > best_cap:
                                best_cap = cap
                                best = (w, l, a1, a2, b1, b2, round(L, 6))
    return best_cap, best
