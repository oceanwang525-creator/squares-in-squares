"""Bounded CPU seed portfolio, never a packing certificate.

Coordinates are unit-square centers in [0,L]^2, angles in radians. Lane zero
is the untouched input baseline, not a re-certified baseline. Every other
lane is an uncertified repair proposal and can overlap or cross a wall.
count is the requested number of LANES (1..24), not the square count.
No file IO or optimization occurs in build_portfolio. Pass the certified
public merge's n-row slice as q; keep the source CSV unchanged.
"""
from dataclasses import asdict
import math
import numpy as np

try:
    from .band_geometry import BandSpec, generate, rectangle
except ImportError:
    from band_geometry import BandSpec, generate, rectangle


def _distinct_exact(points, fallback, n):
    """Restore exact n using distinct centers; overlaps deliberately allowed."""
    rows, seen = [], set()
    for source in (points, fallback):
        for row in source:
            key = tuple(np.round(row[:2], 10))
            if key not in seen:
                rows.append(row.copy())
                seen.add(key)
                if len(rows) == n:
                    return np.asarray(rows)
    raise ValueError("not enough distinct centers to restore n")


def _spec(L, width, theta, offset, phase, cap):
    c, s = abs(math.cos(theta)), abs(math.sin(theta))
    u = np.array([math.cos(theta), math.sin(theta)])
    v = np.array([-u[1], u[0]])
    shift = np.asarray(offset) + phase[0]*u + phase[1]*v
    length = max(1, math.floor(min(
        (L-2*abs(shift[0])-width*c)/max(s, 1e-9),
        (L-2*abs(shift[1])-width*s)/max(c, 1e-9))))
    trims = tuple((i % 2, (i+1) % 2) for i in range(width)) if cap else ()
    # Short bands must retain at least one cell per row.
    if length < 3:
        trims = ()
    return BandSpec(L, width, length, theta, tuple(offset), tuple(phase),
                    end_trims=trims)


def _strip(q, L, rng, lane):
    """Rip tilted cells plus a nearby randomly offset diagonal stripe."""
    theta = math.pi/4 + (lane % 3 - 1)*.055
    # The long diagonal is v, consistent with band_geometry.rectangle.
    u = np.array([math.cos(theta), math.sin(theta)])
    projection = (q[:, :2]-L/2) @ u
    offset = float(rng.uniform(-.7, .7))
    width = 1 + lane % 4
    tilted = np.abs(np.sin(2*q[:, 2])) > .08
    mask = tilted | (np.abs(projection-offset) < width/2)
    if not mask.any():
        mask[np.argmin(np.abs(projection-offset))] = True
    idx = np.flatnonzero(mask)
    phase = (float(rng.uniform(-.45, .45)), float(rng.uniform(-.45, .45)))
    # Capacity is bounded by box geometry, NEVER by removed count. Surplus
    # removed cells are reinserted at their original locations below.
    spec = _spec(L, width, theta, tuple(offset*u), phase, False)
    length = spec.length
    sites = rectangle(spec)
    h = (np.abs(np.cos(sites[:, 2])) + np.abs(np.sin(sites[:, 2])))/2
    fits = ((sites[:, :2] >= h[:, None]-1e-10) &
            (sites[:, :2] <= L-h[:, None]+1e-10)).all(axis=1)
    sites = sites[fits]
    # A tiny box can have zero capacity for this orientation.

    # Endpoint assignments vary which transverse rows lose surplus end cells.
    order = np.lexsort((np.arange(len(sites)) % length,
                        np.arange(len(sites)) // length))
    if lane % 2:
        order = order[::-1]
    if lane % 3 == 0:
        order = rng.permutation(order)
    sites = sites[order[:len(idx)]]
    candidate = q.copy()
    # Retain actual original coordinates for any excess beyond capacity.
    # Select assignment endpoints/random order without growing the rectangle.
    assigned = idx.copy()
    if lane % 2:
        assigned = assigned[::-1]
    if lane % 3 == 0:
        assigned = rng.permutation(assigned)
    assigned = assigned[:len(sites)]
    candidate[assigned] = sites
    # Distinct centers avoid a zero collision gradient from exact coincidences.
    # Only replaced rows are nudged; all other baseline rows remain identical.
    seen = {tuple(np.round(row[:2], 10)) for row in q[~mask]}
    for i in idx:
        while tuple(np.round(candidate[i, :2], 10)) in seen:
            candidate[i, :2] += rng.uniform(.0001, .0003, 2)
        seen.add(tuple(np.round(candidate[i, :2], 10)))
    return candidate, dict(kind="atlas_strip_reinsert", removed_count=len(idx),
        inserted_count=len(idx), removed_indices=idx.tolist(), width=width,
        reconstructed_count=len(sites), restored_original_count=len(idx)-len(sites),
        rectangle_capacity=int(fits.sum()), length=length,
        stripe_offset=offset, theta=theta, phase=list(phase),
        endpoint_assignment=["forward", "reverse"][lane % 2],
        randomized_assignment=lane % 3 == 0)


def build_portfolio(q, L, count=24, rng=None):
    """Return (B,N,3) float64 array and one JSON-safe metadata dict per lane.

    At most 24 lanes. Four lane types interleave after a retained baseline:
    tiny global shakes, Atlas strip rip/reinsert, and analytic finite bands
    with varied widths, offsets, lattice phases and staggered endpoint caps.
    Analytic deficits use distinct original centers as repair insertions;
    analytic surplus is deterministically cropped. Neither operation proves
    feasibility. No positive validity assertion is made, even for lane zero.
    """
    q = np.asarray(q, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 3 or not len(q) or not np.isfinite(q).all():
        raise ValueError("q must be nonempty finite (N,3)")
    if not np.isfinite(L) or L < 1:
        raise ValueError("L must be finite and >=1")
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or not 1 <= count <= 24:
        raise ValueError("count must be an integer in 1..24")
    if len(np.unique(np.round(q[:, :2], 10), axis=0)) != len(q):
        raise ValueError("baseline must have distinct centers")
    rng = np.random.default_rng(rng)
    lanes = [q.copy()]
    metadata = [dict(kind="baseline", unchanged=True)]
    for lane in range(1, count):
        mode = lane % 4
        if mode == 1:
            scale = (1e-4, 1e-3, 6e-3)[(lane//4) % 3]
            p = q.copy()
            p[:, :2] += rng.normal(0, scale, (len(q), 2))
            p[:, 2] += rng.normal(0, scale, len(q))
            meta = dict(kind="tiny_global_perturb", xy_sigma=scale,
                        angle_sigma=scale)
        elif mode in (0, 2):
            p, meta = _strip(q, L, rng, lane)
        else:
            width = min(max(1, int(L)-1), 1+(lane//4) % 5)
            offset = ((lane//4 % 3-1)*.23, (lane//8 % 3-1)*.17)
            phase = ((lane//4 % 2)*.21, -(lane//4 % 3)*.13)
            spec = _spec(L, width, math.pi/4, offset, phase, lane//4 % 2)
            try:
                original, explicit = generate(spec)
                meta = dict(kind="analytic_band", spec=asdict(explicit),
                            generated_count=len(original), generation_failed=False)
                # Prioritize band cells, then corner cells, then distinct
                # baseline insertions. Exact n always, never silent n-1.
                p = _distinct_exact(original, q, len(q))
            except ValueError as exc:
                p, meta = _strip(q, L, rng, lane)
                meta.update(requested_kind="analytic_band",
                            generation_failed=True, reason=str(exc))
        lanes.append(p)
        metadata.append(meta)
    for i, meta in enumerate(metadata):
        meta.update(lane=i, n=len(q), certified=False,
                    requires_repair=i != 0,
                    note="Uncertified seed; overlaps and wall violations possible.")
    return np.stack(lanes), metadata
