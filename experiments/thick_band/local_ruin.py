"""Bounded Santa-inspired local ruin/recreate reference, CPU only.

Finite candidates plus sparse binary MILP are a heuristic, not a continuous
packing solver or official certificate. Coordinates are centers in [0,L]^2,
angles radians. Upright rows and all rows outside the ruin are immutable.
"""
import time
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.optimize import milp, Bounds, LinearConstraint
try:
    from .band_polish import sat_check
except ImportError:
    from band_polish import sat_check


def _clear(a, others):
    """Strict 4-axis SAT, no positive overlap tolerance."""
    if not len(others):
        return True
    a = np.asarray(a, dtype=np.longdouble)
    b = np.asarray(others, dtype=np.longdouble)
    d = b[:, :2] - a[:2]
    ua = np.array([np.cos(a[2]), np.sin(a[2])])
    va = np.array([-ua[1], ua[0]])
    ub = np.column_stack((np.cos(b[:, 2]), np.sin(b[:, 2])))
    vb = np.column_stack((-ub[:, 1], ub[:, 0]))
    best = np.full(len(b), -np.inf, dtype=np.longdouble)
    for axes in (np.broadcast_to(ua, d.shape), np.broadcast_to(va, d.shape), ub, vb):
        radii = .5 * (np.abs(axes @ ua) + np.abs(axes @ va) +
                      np.abs(np.sum(axes * ub, axis=1)) +
                      np.abs(np.sum(axes * vb, axis=1)))
        best = np.maximum(best, np.abs(np.sum(d * axes, axis=1)) - radii)
    return bool(np.all(best >= 0))


def _inside(q, L):
    z = np.asarray(q, dtype=np.longdouble)
    h = (np.abs(np.cos(z[:, 2])) + np.abs(np.sin(z[:, 2]))) / 2
    return bool(np.all(z[:, :2] - h[:, None] >= 0) and
                np.all(z[:, :2] + h[:, None] <= L))


def local_ruin(source, L, target, *, seed=0, remove_count=10,
               window_grid=9, candidate_cap=600, attempt_cap=12000,
               time_limit=3., upright_tol=1e-12):
    """Return {'x': exact-target contiguous fp64 array or None, 'meta': dict}.

    Select the least center-dense L/4 window, tie-breaking by clipped AABB
    overlap area (only a density heuristic). Ruin up to remove_count nearest
    *nonupright* centers. All upright source rows remain byte identical.
    Candidate centers stay within that window or the removed-center bounding
    rectangle. Angles include nearby originals, upright, and seeded angles;
    centers include original hints, a half-unit grid and seeded samples.
    Cooperative wall-time budget covers geometry/MILP; final full-pair SAT
    runs even after budget expiration, never skipped. No optimality claim.
    Input is copied and never mutated. Source count and target are explicit,
    so an n-1 source can request exactly n, never an implicit extra square.
    """
    start = time.monotonic()
    q = np.array(source, dtype=np.float64, order='C', copy=True)
    if q.size == 0:
        q = q.reshape(0, 3)
    if q.ndim != 2 or q.shape[1] != 3 or not np.isfinite(q).all():
        raise ValueError('source must be finite (n,3)')
    if (not np.isfinite([L, time_limit, upright_tol]).all() or L <= 0 or
            time_limit <= 0 or upright_tol < 0 or
            any(int(v) != v for v in (target, remove_count, window_grid, candidate_cap, attempt_cap)) or
            target < 0 or not 0 <= remove_count <= 32 or
            not 2 <= window_grid <= 64 or not 1 <= candidate_cap <= 10000 or attempt_cap < 1):
        raise ValueError('invalid dimensions, target or bounds')
    target = int(target)
    deadline = start + time_limit
    meta = dict(source_n=len(q), target=target, candidate_cap_hit=False,
                attempt_cap_hit=False, timed_out=False, candidates=0,
                attempts=0, edges=0, removed_indices=[], outside_indices=[],
                milp_status=None, graph_optimal=False)
    def finish(x=None, reason='ok'):
        meta.update(reason=reason, elapsed=time.monotonic()-start)
        return dict(x=x, meta=meta)
    if not _inside(q, L) or (len(q) and not sat_check(q)[0]):
        return finish(reason='invalid_source')
    upright = np.abs(np.remainder(q[:, 2] + np.pi/4, np.pi/2) - np.pi/4) <= upright_tol
    half = L/8
    centers = np.linspace(half, L-half, window_grid)
    h = .5*(np.abs(np.cos(q[:, 2])) + np.abs(np.sin(q[:, 2])))
    best = None
    for x in centers:
        for y in centers:
            if time.monotonic() >= deadline:
                meta['timed_out'] = True
                return finish(reason='time_limit')
            c = np.array([x, y])
            count = np.sum(np.all(np.abs(q[:, :2]-c) <= half, axis=1))
            overlap = np.maximum(0, np.minimum(q[:, :2]+h[:, None], c+half) -
                                 np.maximum(q[:, :2]-h[:, None], c-half))
            key = (int(count), float(np.prod(overlap, axis=1).sum()), x, y)
            if best is None or key < best[0]:
                best = (key, c)
    c = best[1]
    eligible = np.flatnonzero(~upright)
    order = np.argsort(np.linalg.norm(q[eligible, :2]-c, axis=1), kind='stable')
    # Never ruin distant ribbons just because this patch contains only uprights.
    near = eligible[order]
    near = near[np.max(np.abs(q[near, :2]-c), axis=1) <= half + 1]
    removed = near[:remove_count]
    outside_ids = np.setdiff1d(np.arange(len(q)), removed)
    outside = q[outside_ids].copy()
    needed = target - len(outside)
    meta.update(window_center=c.tolist(), window_side=L/4,
                removed_indices=removed.tolist(), outside_indices=outside_ids.tolist(), refill_target=needed)
    if needed < 0 or needed > 33:
        return finish(reason='unsupported_refill_count')
    if needed == 0:
        return finish(np.ascontiguousarray(outside))
    low, high = c-half, c+half
    if len(removed):
        low = np.minimum(low, q[removed, :2].min(axis=0))
        high = np.maximum(high, q[removed, :2].max(axis=0))
    rng = np.random.default_rng(seed)
    angles = np.unique(np.r_[0., q[removed, 2], rng.uniform(-np.pi/4, np.pi/4, 3)])
    obstacle_tree = cKDTree(outside[:, :2]) if len(outside) else None
    candidates = []; seen = set()
    def add(row):
        if time.monotonic() >= deadline:
            meta['timed_out'] = True
            return False
        if meta['attempts'] >= attempt_cap:
            meta['attempt_cap_hit'] = True
            return False
        if len(candidates) >= candidate_cap:
            meta['candidate_cap_hit'] = True
            return False
        meta['attempts'] += 1
        key = tuple(float(v) for v in row)
        if key in seen:
            return True
        seen.add(key)
        row = np.asarray(row, dtype=np.float64)
        if not _inside(row[None], L):
            return True
        ids = [] if obstacle_tree is None else obstacle_tree.query_ball_point(row[:2], np.nextafter(np.sqrt(2.), np.inf))
        if _clear(row, outside[ids]):
            candidates.append(row)
        return True
    for row in q[removed]:
        if not add(row):
            break
    # Include exact global lattice phase to preserve contact-compatible candidates.
    stop = False
    for x in np.arange(.5, L, .5):
        if x < low[0] or x > high[0]:
            continue
        for y in np.arange(.5, L, .5):
            if y < low[1] or y > high[1]:
                continue
            for angle in angles:
                if not add([x, y, angle]):
                    stop = True
                    break
            if stop: break
        if stop: break
    while not stop:
        row = np.r_[rng.uniform(low, high), rng.choice(angles)]
        stop = not add(row)
    cand = np.ascontiguousarray(candidates, dtype=np.float64).reshape(-1, 3)
    meta['candidates'] = len(cand)
    if meta['timed_out'] or len(cand) < needed:
        return finish(reason='time_limit' if meta['timed_out'] else 'insufficient_candidates')
    edges = []
    tree = cKDTree(cand[:, :2])
    for i in range(len(cand)):
        for j in sorted(tree.query_ball_point(cand[i, :2], np.nextafter(np.sqrt(2.), np.inf))):
            if time.monotonic() >= deadline:
                meta['timed_out'] = True
                return finish(reason='time_limit')
            if j > i and not _clear(cand[i], cand[j:j+1]):
                edges.append((i, j))
    meta['edges'] = len(edges)
    rows = []; cols = []
    for k, (i, j) in enumerate(edges):
        rows.extend((k, k)); cols.extend((i, j))
    rows.extend([len(edges)]*len(cand)); cols.extend(range(len(cand)))
    A = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(edges)+1, len(cand))).tocsc()
    lower = np.r_[np.full(len(edges), -np.inf), needed]
    upper = np.r_[np.ones(len(edges)), needed]
    remaining = deadline-time.monotonic()
    if remaining <= 0:
        meta['timed_out'] = True
        return finish(reason='time_limit')
    res = milp(np.zeros(len(cand)), integrality=np.ones(len(cand)),
               bounds=Bounds(0, 1), constraints=LinearConstraint(A, lower, upper),
               options={'time_limit': remaining, 'node_limit': 10000})
    meta['milp_status'] = int(res.status)
    meta['timed_out'] = time.monotonic() >= deadline
    # This is feasibility only, never global continuous packing optimality.
    if res.x is None:
        return finish(reason='milp_no_solution')
    selected = cand[res.x > .5]
    x = np.ascontiguousarray(np.concatenate((outside, selected)), dtype=np.float64)
    if len(x) != target or not _inside(x, L) or not sat_check(x)[0]:
        return finish(reason='strict_final_reject')
    if x[:len(outside)].tobytes() != outside.tobytes():
        return finish(reason='outside_changed')
    return finish(x)
