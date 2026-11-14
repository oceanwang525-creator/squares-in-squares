"""H131 seed builder: third-generation masked starts around L0 = L_base + 0.042."""
from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
HIGHD_POC = REPO_ROOT / "poc" / "swarm_highd" / "poc"
if str(HIGHD_POC) not in sys.path:
    sys.path.insert(0, str(HIGHD_POC))

from cpp_wrap import full_to_free_sym  # noqa: E402
import idea1_highd_pilot as i1  # noqa: E402

RADIUS_MIN = 5.0
RADIUS_MAX = 8.7


def _mask3(active: np.ndarray) -> np.ndarray:
    m = active.astype(np.uint8)
    return np.concatenate([m, m, m])


def tightest_contact_clusters(
    x: np.ndarray,
    y: np.ndarray,
    th: np.ndarray,
    L: float,
    n_pts: int = 8,
    min_sep: float = 2.6,
) -> list[tuple[float, float, float]]:
    """Rank squares by local SAT contact-stress density + wall tightness and pick separated centers."""
    n = len(x)
    c, s = np.cos(th), np.sin(th)
    hx = 0.5 * (np.abs(c) + np.abs(s))
    wm = np.minimum.reduce([x - hx, L - x - hx, y - hx, L - y - hx])

    # Local contact stress score using SAT separating-axis clearance on close pairs
    stress = np.maximum(0.0, 0.05 - wm) * 2.0
    for i in range(n):
        xi, yi, ci, si, hxi = float(x[i]), float(y[i]), float(c[i]), float(s[i]), float(hx[i])
        dx = x[i + 1 :] - xi
        dy = y[i + 1 :] - yi
        bnd = hxi + hx[i + 1 :] + 0.12
        cand = np.where((np.abs(dx) < bnd) & (np.abs(dy) < bnd))[0]
        for off in cand:
            j = i + 1 + int(off)
            dxj, dyj = float(dx[off]), float(dy[off])
            cj, sj = float(c[j]), float(s[j])
            rad = 0.5 * (1.0 + abs(ci * cj + si * sj) + abs(ci * sj - si * cj))
            gap = max(
                abs(dxj * ci + dyj * si) - rad,
                abs(-dxj * si + dyj * ci) - rad,
                abs(dxj * cj + dyj * sj) - rad,
                abs(-dxj * sj + dyj * cj) - rad,
            )
            if gap < 0.08:
                w = (0.08 - gap) / 0.08
                stress[i] += w
                stress[j] += w

    order = np.argsort(-stress)
    out: list[tuple[float, float, float]] = []
    for idx in order:
        if len(out) >= n_pts:
            break
        px, py = float(x[idx]), float(y[idx])
        if all((px - qx) ** 2 + (py - qy) ** 2 > min_sep * min_sep for qx, qy, _ in out):
            out.append((px, py, float(stress[idx])))
    return out


def build_h131_seeds(
    n: int,
    x_base: np.ndarray,
    y_base: np.ndarray,
    th_base: np.ndarray,
    L_base: float,
    rng: np.random.Generator,
    L_expand: float = 0.042,
) -> tuple[list[np.ndarray], list[np.ndarray], list[list[float]], list[str]]:
    """Build 30 masked starts: 12 void + 10 contact + 8 seam with R in [5.0, 8.7]."""
    L0 = L_base + L_expand
    sc = L0 / L_base
    xb, yb, tb = x_base * sc, y_base * sc, th_base.copy()
    hx = 0.5 * (np.abs(np.cos(tb)) + np.abs(np.sin(tb)))

    margin = np.minimum.reduce([xb - hx, L0 - xb - hx, yb - hx, L0 - yb - hx])
    tight_idx = np.argsort(margin)[:18]
    voids = i1.clearance_points(xb, yb, tb, L0, n_pts=8, step=0.14)
    hot_clusters = tightest_contact_clusters(x_base, y_base, th_base, L_base, n_pts=8)
    hot_clusters_scaled = [(hx_c * sc, hy_c * sc, hs) for hx_c, hy_c, hs in hot_clusters]
    tilt = i1.donor_tilt(tb)

    deg = np.degrees(tb) % 90.0
    deg_m = np.minimum(deg, 90.0 - deg)
    up_mask = deg_m < 1.0
    tilted_idx = np.where(~up_mask)[0]
    seam_idx = tilted_idx
    if np.any(up_mask) and len(tilted_idx) > 0:
        d_ut = np.hypot(xb[tilted_idx, None] - xb[None, up_mask], yb[tilted_idx, None] - yb[None, up_mask])
        close_t = tilted_idx[np.where(d_ut.min(axis=1) < 1.35)[0]]
        if len(close_t) > 0:
            seam_idx = close_t

    mL = xb - hx - float((xb - hx).min())
    mR = float((xb + hx).max()) - (xb + hx)
    mB = yb - hx - float((yb - hx).min())
    mT = float((yb + hx).max()) - (yb + hx)
    cL, cR = int(np.sum(mL < 5e-4)), int(np.sum(mR < 5e-4))
    cB, cT = int(np.sum(mB < 5e-4)), int(np.sum(mT < 5e-4))
    dx_sign = 1.0 if cR <= cL else -1.0
    dy_sign = 1.0 if cT <= cB else -1.0

    # A five-stage retreat gives the wider starts one extra relaxation window
    # before the final tighter stages, while keeping every stage feasible.
    sched_p = [
        L0,
        L_base + 0.011,
        L_base + 0.001,
        L_base - 4e-4,
        L_base - 0.0028,
    ]
    q_list, m_list, ls_list, labels = [], [], [], []

    # 1. mpk_void (12 seeds): top clearance voids (half cavity-relocate + half non-teleport)
    for s in range(12):
        x, y, th = xb.copy(), yb.copy(), tb.copy()
        vx, vy, _ = voids[s % len(voids)] if voids else (0.5 * L0, 0.5 * L0, 0.1)
        rad = float(rng.uniform(RADIUS_MIN, RADIUS_MAX))
        near = ((x - vx) ** 2 + (y - vy) ** 2) <= rad * rad
        if len(tight_idx) > 0 and s % 2 == 0:
            pi = int(tight_idx[(s // 2) % len(tight_idx)])
            px_old, py_old = float(x[pi]), float(y[pi])
            x[pi] = vx + float(rng.uniform(-0.08, 0.08))
            y[pi] = vy + float(rng.uniform(-0.08, 0.08))
            th[pi] = [0.0, np.pi / 4.0, tilt][(s // 2) % 3] + float(rng.uniform(-0.10, 0.10))
            near |= ((x - px_old) ** 2 + (y - py_old) ** 2) <= 4.0 ** 2
            near[pi] = True
        nm = int(near.sum())
        if nm > 0:
            x[near] += rng.uniform(-0.18, 0.18, nm)
            y[near] += rng.uniform(-0.18, 0.18, nm)
            th[near] += rng.uniform(-0.36, 0.36, nm)
        q_list.append(full_to_free_sym(x, y, th, L0))
        m_list.append(_mask3(near))
        ls_list.append(sched_p)
        labels.append(f"mpk_void_{s}")

    # 2. mpk_hot (10 seeds): tightest contact-stress clusters (half with weak-wall slack drift)
    for s in range(10):
        x, y, th = xb.copy(), yb.copy(), tb.copy()
        if hot_clusters_scaled:
            cx, cy, _ = hot_clusters_scaled[s % len(hot_clusters_scaled)]
        else:
            pi = int(tight_idx[s % len(tight_idx)])
            cx, cy = float(x[pi]), float(y[pi])
        rad = float(rng.uniform(RADIUS_MIN, RADIUS_MAX))
        near = ((x - cx) ** 2 + (y - cy) ** 2) <= rad * rad
        nm = int(near.sum())
        if nm > 0:
            drift = float(rng.uniform(0.035, 0.07)) if (s % 2 == 1) else 0.0
            x[near] += dx_sign * drift * 0.5 + rng.uniform(-0.17, 0.17, nm)
            y[near] += dy_sign * drift * 0.5 + rng.uniform(-0.17, 0.17, nm)
            th[near] += rng.uniform(-0.38, 0.38, nm)
        q_list.append(full_to_free_sym(x, y, th, L0))
        m_list.append(_mask3(near))
        ls_list.append(sched_p)
        labels.append(f"mpk_hot_{s}")

    # 3. mpk_seam (8 seeds): around (k-0.5, k-0.5), its symmetric corner, and upright/tilted interface
    k = math.isqrt(n)
    c_seam = min(max(k - 0.5, 2.0), L0 - 2.0)
    c_sym = min(max(L0 - (k - 0.5), 2.0), L0 - 2.0)
    for s in range(8):
        x, y, th = xb.copy(), yb.copy(), tb.copy()
        if s == 0:
            cx, cy = c_seam, c_seam
        elif s == 1:
            cx, cy = c_sym, c_sym
        elif len(seam_idx) > 0:
            pi = int(seam_idx[((s - 2) * 23 + 7) % len(seam_idx)])
            cx, cy = float(x[pi]), float(y[pi])
        else:
            cx, cy = 0.5 * L0, 0.5 * L0
        rad = float(rng.uniform(RADIUS_MIN, RADIUS_MAX))
        near = ((x - cx) ** 2 + (y - cy) ** 2) <= rad * rad
        nm = int(near.sum())
        if nm > 0:
            x[near] += rng.uniform(-0.17, 0.17, nm)
            y[near] += rng.uniform(-0.17, 0.17, nm)
            th[near] += rng.uniform(-0.36, 0.36, nm)
        q_list.append(full_to_free_sym(x, y, th, L0))
        m_list.append(_mask3(near))
        ls_list.append(sched_p)
        labels.append(f"mpk_seam_{s}")

    return q_list, m_list, ls_list, labels
