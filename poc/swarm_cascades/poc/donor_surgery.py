"""Generic interior cross-cut, slab-expansion, and seam-mask synthesis helpers for swarm_cascades."""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def normalize_coords(x: np.ndarray, y: np.ndarray, th: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Shift coordinates so min(x - hx) == 0, min(y - hy) == 0; return (x, y, th, L_bbox)."""
    c = np.abs(np.cos(th))
    s = np.abs(np.sin(th))
    h = 0.5 * (c + s)
    xmin = float(np.min(x - h))
    ymin = float(np.min(y - h))
    nx = x - xmin
    ny = y - ymin
    L = max(float(np.max(nx + h)), float(np.max(ny + h)))
    return nx, ny, th.copy(), L


def find_interior_cross_cuts(
    x: np.ndarray,
    y: np.ndarray,
    th: np.ndarray,
    n_target: int,
    max_candidates: int = 8,
    seam_radius: float = 1.6,
) -> list[dict]:
    """Search interior cross-strip deletions (dk > 0) yielding exact n_target squares.

    Removes squares with x in [cx - wx/2, cx + wx/2] OR y in [cy - wy/2, cy + wy/2],
    then shifts x > cx + wx/2 by -sx and y > cy + wy/2 by -sy where sx, sy are near dk.
    Ranks candidates by low initial overlap/bbox proxy and upright phase preservation.
    """
    n_d = len(x)
    x, y, th, L_d = normalize_coords(x, y, th)
    k_d = math.isqrt(n_d)
    k_t = math.isqrt(n_target)
    dk = k_d - k_t
    if dk <= 0:
        return []

    cx_grid = np.linspace(0.28 * L_d, 0.72 * L_d, 23)
    cy_grid = np.linspace(0.28 * L_d, 0.72 * L_d, 23)
    w_offsets = np.linspace(-0.35, 0.35, 29)

    seen_signatures = set()
    candidates = []

    for cx in cx_grid:
        for wx_off in w_offsets:
            wx = dk + wx_off
            if wx <= 0.2:
                continue
            keep_x = (x < cx - 0.5 * wx) | (x > cx + 0.5 * wx)
            cnt_after_x = int(keep_x.sum())
            # Early prune: after cutting y-strip of ~dk rows we need n_target
            if cnt_after_x < n_target:
                continue
            for cy in cy_grid:
                for wy_off in (-0.30, -0.15, 0.0, 0.15, 0.30, wx_off):
                    wy = dk + wy_off
                    if wy <= 0.2:
                        continue
                    keep = keep_x & ((y < cy - 0.5 * wy) | (y > cy + 0.5 * wy))
                    if int(keep.sum()) != n_target:
                        continue

                    sig = keep.tobytes()
                    if sig in seen_signatures:
                        continue
                    seen_signatures.add(sig)

                    # Shift by integer dk first (preserves upright grid phase!) or wx/wy
                    for shift_mode, sx, sy in (
                        ("int_dk", float(dk), float(dk)),
                        ("cut_w", float(wx), float(wy)),
                    ):
                        nx = x[keep].copy()
                        ny = y[keep].copy()
                        nth = th[keep].copy()
                        nx[nx > cx] -= sx
                        ny[ny > cy] -= sy
                        nx, ny, nth, L_new = normalize_coords(nx, ny, nth)
                        seam_x = cx - 0.5 * sx
                        seam_y = cy - 0.5 * sy
                        seam_mask = (
                            (np.abs(nx - seam_x) < seam_radius)
                            | (np.abs(ny - seam_y) < seam_radius)
                        ).astype(np.uint8)

                        # Quick score: L_new + overlap estimate via near pairs near seams
                        in_seam = np.where(seam_mask > 0)[0]
                        min_d2 = 999.0
                        if len(in_seam) > 1:
                            dx = nx[in_seam, None] - nx[None, in_seam]
                            dy = ny[in_seam, None] - ny[None, in_seam]
                            d2 = dx * dx + dy * dy
                            np.fill_diagonal(d2, 999.0)
                            # Penalize close centers (< 0.95^2)
                            close = np.maximum(0.0, 0.92 - np.sqrt(np.maximum(d2, 1e-12))).sum()
                        else:
                            close = 0.0
                        score = (L_new - k_t) + 0.015 * float(close) + 0.002 * abs(wx - dk) + 0.002 * abs(wy - dk)
                        candidates.append({
                            "score": score,
                            "L_raw": L_new,
                            "close_pen": float(close),
                            "x": nx,
                            "y": ny,
                            "th": nth,
                            "seam_mask": seam_mask,
                            "meta": f"cut_{shift_mode}_cx{cx:.1f}_cy{cy:.1f}_wx{wx:.2f}_wy{wy:.2f}",
                        })

    # Deduplicate by rounded score + L_raw and return diverse top candidates
    candidates.sort(key=lambda d: d["score"])
    out = []
    used_cxcy = []
    for c in candidates:
        # Keep diversity across cut locations
        if len(out) >= max_candidates:
            break
        out.append(c)
    return out


def find_corner_band_cuts(
    x: np.ndarray,
    y: np.ndarray,
    th: np.ndarray,
    n_target: int,
    max_candidates: int = 8,
    seam_radius: float = 1.6,
) -> list[dict]:
    """Delete an integer dk-wide upright L-strip from each of the 4 outer edges (or interior slices
    along just the upright side away from tilted pockets).
    Automatically detects which quadrant(s) hold non-45° tilted squares and places the cut away from them.
    """
    n_d = len(x)
    x, y, th, L_d = normalize_coords(x, y, th)
    k_d = math.isqrt(n_d)
    k_t = math.isqrt(n_target)
    dk = k_d - k_t
    if dk <= 0:
        return []

    ang_mod = np.abs(np.sin(2.0 * th))
    tilted = ang_mod > 0.05
    if tilted.any():
        tx_mean = float(x[tilted].mean()) / L_d
        ty_mean = float(y[tilted].mean()) / L_d
    else:
        tx_mean, ty_mean = 0.5, 0.5

    # Prefer cuts near the side OPPOSITE the tilted pocket center
    cx_prefs = np.linspace(0.15 * L_d, 0.85 * L_d, 29)
    cy_prefs = np.linspace(0.15 * L_d, 0.85 * L_d, 29)
    # Sort cx, cy away from tilted cluster first
    cx_prefs = sorted(cx_prefs, key=lambda v: -abs(v / L_d - tx_mean))[:12]
    cy_prefs = sorted(cy_prefs, key=lambda v: -abs(v / L_d - ty_mean))[:12]

    candidates = []
    seen = set()
    for cx in cx_prefs:
        for wx in np.linspace(dk - 0.35, dk + 0.35, 15):
            keep_x = (x < cx - 0.5 * wx) | (x > cx + 0.5 * wx)
            if int(keep_x.sum()) < n_target:
                continue
            for cy in cy_prefs:
                for wy in np.linspace(dk - 0.35, dk + 0.35, 15):
                    keep = keep_x & ((y < cy - 0.5 * wy) | (y > cy + 0.5 * wy))
                    if int(keep.sum()) != n_target:
                        continue
                    sig = keep.tobytes()
                    if sig in seen:
                        continue
                    seen.add(sig)
                    nx = x[keep].copy()
                    ny = y[keep].copy()
                    nth = th[keep].copy()
                    nx[nx > cx] -= dk
                    ny[ny > cy] -= dk
                    nx, ny, nth, L_new = normalize_coords(nx, ny, nth)
                    seam_mask = (
                        (np.abs(nx - (cx - 0.5 * dk)) < seam_radius)
                        | (np.abs(ny - (cy - 0.5 * dk)) < seam_radius)
                    ).astype(np.uint8)
                    # Count tilted squares clipped/near seam
                    seam_tilted = int(((np.abs(np.sin(2.0 * nth)) > 0.05) & (seam_mask > 0)).sum())
                    score = (L_new - k_t) + 0.005 * seam_tilted + 0.001 * (abs(wx - dk) + abs(wy - dk))
                    candidates.append({
                        "score": score,
                        "L_raw": L_new,
                        "seam_tilted": seam_tilted,
                        "x": nx,
                        "y": ny,
                        "th": nth,
                        "seam_mask": seam_mask,
                        "meta": f"away_cx{cx:.1f}_cy{cy:.1f}_wx{wx:.2f}_wy{wy:.2f}",
                    })
    candidates.sort(key=lambda d: d["score"])
    return candidates[:max_candidates]


def local_pocket_perturb_seeds(
    x: np.ndarray,
    y: np.ndarray,
    th: np.ndarray,
    n_seeds: int = 8,
    seed0: int = 100,
) -> list[dict]:
    """Generate targeted seam/pocket perturbation seeds around non-axis-aligned junctions."""
    rng = np.random.default_rng(seed0)
    x, y, th, L = normalize_coords(x, y, th)
    n = len(x)
    ang = np.abs(np.sin(2.0 * th))
    tilted_idx = np.where(ang > 0.05)[0]
    if len(tilted_idx) == 0:
        # Just perturb highest-x/y corner
        dist_corner = (L - x) + (L - y)
        seam_mask = (dist_corner < np.percentile(dist_corner, 15)).astype(np.uint8)
    else:
        # Distance to nearest upright square AND nearest tilted square -> junction!
        up_idx = np.where(ang <= 0.05)[0]
        if len(up_idx) > 0:
            dx = x[:, None] - x[None, up_idx]
            dy = y[:, None] - y[None, up_idx]
            d_up = np.sqrt(np.min(dx * dx + dy * dy, axis=1))
            dx2 = x[:, None] - x[None, tilted_idx]
            dy2 = y[:, None] - y[None, tilted_idx]
            d_til = np.sqrt(np.min(dx2 * dx2 + dy2 * dy2, axis=1))
            seam_mask = ((d_up < 2.3) & (d_til < 2.3)).astype(np.uint8)
        else:
            seam_mask = np.ones(n, dtype=np.uint8)
        # Also include tilted pocket boundary squares
        seam_mask[(ang > 0.05) & ((x < 2.5) | (x > L - 2.5) | (y < 2.5) | (y > L - 2.5))] = 1

    if int(seam_mask.sum()) < 10:
        seam_mask[:] = 1

    seeds = []
    idx_seam = np.where(seam_mask > 0)[0]
    for s in range(n_seeds):
        nx, ny, nth = x.copy(), y.copy(), th.copy()
        if s == 0:
            pass  # exact baseline with seam mask
        elif s % 3 == 1:
            # Small positional + angular kick on junction
            nx[idx_seam] += rng.normal(0.0, 0.03, len(idx_seam))
            ny[idx_seam] += rng.normal(0.0, 0.03, len(idx_seam))
            nth[idx_seam] += rng.normal(0.0, 0.025, len(idx_seam))
        elif s % 3 == 2:
            # Coherent tilt shift of tilted pocket by +-0.5 deg..1.5 deg
            dth = math.radians(rng.choice([-1.2, -0.6, 0.6, 1.2]))
            nth[ang > 0.05] += dth
            nx[idx_seam] += rng.normal(0.0, 0.015, len(idx_seam))
            ny[idx_seam] += rng.normal(0.0, 0.015, len(idx_seam))
        else:
            # Uniform slight radial squeeze + junction kick
            scale = 0.9992
            nx *= scale
            ny *= scale
            nx[idx_seam] += rng.normal(0.0, 0.02, len(idx_seam))
            ny[idx_seam] += rng.normal(0.0, 0.02, len(idx_seam))
            nth[idx_seam] += rng.normal(0.0, 0.015, len(idx_seam))
        seeds.append({
            "x": nx,
            "y": ny,
            "th": nth,
            "seam_mask": seam_mask.copy(),
            "meta": f"perturb_{s}",
        })
    return seeds
