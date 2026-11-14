"""H132 breadth-first masked-pocket starts with target-specific perturbations."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
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

@dataclass(frozen=True)
class SeedPreset:
    """Fresh deterministic parameters for one incumbent topology."""

    seed: int
    cpp_seed: int
    l_expand: float
    radius_min: float
    radius_max: float
    coord_jitter: float
    angle_jitter: float
    a_start: float


PRESETS = {
    # Low-n incumbents retain more internal slack, so the pocket can breathe wider.
    367: SeedPreset(1320367, 1321367, 0.044, 4.0, 7.4, 0.18, 0.36, 0.76),
    # n=902 is dense; keep the perturbed pockets tighter and the melt more gentle.
    902: SeedPreset(1320902, 1321902, 0.030, 5.2, 8.7, 0.13, 0.28, 0.80),
    # The n=530 seam winner defines this mid-density family.
    528: SeedPreset(1320528, 1321528, 0.0450, 4.4, 7.9, 0.17, 0.35, 0.75),
    529: SeedPreset(1320529, 1321529, 0.0460, 4.4, 7.9, 0.17, 0.35, 0.75),
    530: SeedPreset(1320530, 1321530, 0.0470, 4.4, 7.9, 0.17, 0.35, 0.75),
    531: SeedPreset(1320531, 1321531, 0.0460, 4.4, 7.9, 0.17, 0.35, 0.75),
    532: SeedPreset(1320532, 1321532, 0.0450, 4.4, 7.9, 0.17, 0.35, 0.75),
    533: SeedPreset(1320533, 1321533, 0.0480, 4.4, 7.9, 0.17, 0.35, 0.75),
    576: SeedPreset(1320576, 1321576, 0.0460, 4.4, 7.9, 0.17, 0.35, 0.75),
    577: SeedPreset(1320577, 1321577, 0.0470, 4.4, 7.9, 0.17, 0.35, 0.75),
    578: SeedPreset(1320578, 1321578, 0.0450, 4.4, 7.9, 0.17, 0.35, 0.75),
    # Remaining D=1 high-gap rows reuse the winning mid-density pocket scale.
    442: SeedPreset(1320442, 1321442, 0.0470, 4.4, 7.9, 0.17, 0.35, 0.75),
    485: SeedPreset(1320485, 1321485, 0.0470, 4.4, 7.9, 0.17, 0.35, 0.75),
    785: SeedPreset(1320785, 1321785, 0.0470, 4.4, 7.9, 0.17, 0.35, 0.75),
    786: SeedPreset(1320786, 1321786, 0.0470, 4.4, 7.9, 0.17, 0.35, 0.75),
    842: SeedPreset(1320842, 1321842, 0.0470, 4.4, 7.9, 0.17, 0.35, 0.75),
    901: SeedPreset(1320901, 1321901, 0.0470, 4.4, 7.9, 0.17, 0.35, 0.75),
    962: SeedPreset(1320962, 1321962, 0.0470, 4.4, 7.9, 0.17, 0.35, 0.75),
    # Final rigid/lift-produced D=1/D=2 transfer batch.
    441: SeedPreset(1320441, 1321441, 0.0460, 4.4, 7.9, 0.17, 0.35, 0.75),
    443: SeedPreset(1320443, 1321443, 0.0480, 4.4, 7.9, 0.17, 0.35, 0.75),
    447: SeedPreset(1320447, 1321447, 0.0470, 4.4, 7.9, 0.17, 0.35, 0.75),
    484: SeedPreset(1320484, 1321484, 0.0460, 4.4, 7.9, 0.17, 0.35, 0.75),
    486: SeedPreset(1320486, 1321486, 0.0480, 4.4, 7.9, 0.17, 0.35, 0.75),
    491: SeedPreset(1320491, 1321491, 0.0470, 4.4, 7.9, 0.17, 0.35, 0.75),
    740: SeedPreset(1320740, 1321740, 0.0470, 4.4, 7.9, 0.17, 0.35, 0.75),
    743: SeedPreset(1320743, 1321743, 0.0470, 4.4, 7.9, 0.17, 0.35, 0.75),
    798: SeedPreset(1320798, 1321798, 0.0470, 4.4, 7.9, 0.17, 0.35, 0.75),
    # Newly merged high-value neighbors, with independent streams.
    538: SeedPreset(1320538, 1321538, 0.0470, 4.4, 7.9, 0.17, 0.35, 0.75),
    588: SeedPreset(1320588, 1321588, 0.0470, 4.4, 7.9, 0.17, 0.35, 0.75),
    799: SeedPreset(1320799, 1321799, 0.0470, 4.4, 7.9, 0.17, 0.35, 0.75),
    803: SeedPreset(1320803, 1321803, 0.0470, 4.4, 7.9, 0.17, 0.35, 0.75),
    # Slightly looser members of the dense 900-902 family.
    900: SeedPreset(1320900, 1321900, 0.0310, 5.2, 8.7, 0.13, 0.28, 0.80),
}

DEFAULT_PRESET = SeedPreset(1329999, 1339999, 0.039, 4.6, 8.1, 0.16, 0.33, 0.77)


def preset_for(n: int) -> SeedPreset:
    """Return the fixed seed/perturbation preset for a target."""
    if n in PRESETS:
        return PRESETS[n]
    return SeedPreset(
        DEFAULT_PRESET.seed + n,
        DEFAULT_PRESET.cpp_seed + n,
        DEFAULT_PRESET.l_expand + 1e-5 * (n % 7),
        DEFAULT_PRESET.radius_min,
        DEFAULT_PRESET.radius_max,
        DEFAULT_PRESET.coord_jitter,
        DEFAULT_PRESET.angle_jitter,
        DEFAULT_PRESET.a_start,
    )


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


def build_h13_seeds(
    n: int,
    x_base: np.ndarray,
    y_base: np.ndarray,
    th_base: np.ndarray,
    L_base: float,
    rng: np.random.Generator,
    L_expand: float | None = None,
    waves: int = 1,
    l_expand_values: Sequence[float] | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray], list[list[float]], list[str]]:
    """Build 24*waves masked starts with optional per-wave expansions."""
    preset = preset_for(n)
    waves = max(1, int(waves))
    if l_expand_values is None:
        base_expand = preset.l_expand if L_expand is None else L_expand
        expansions = [base_expand] * waves
    else:
        expansions = list(l_expand_values)
        if len(expansions) < waves:
            raise ValueError("l_expand_values must provide one expansion per wave")

    q_list, m_list, ls_list, labels = [], [], [], []
    k = math.isqrt(n)
    for wave in range(waves):
        L0 = L_base + expansions[wave]
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
        sched_p = [L0, L_base + 0.006, L_base - 3e-4, L_base - 0.0022]

        # 1. Clearance void starts, with half relocating a tight square.
        for s in range(10):
            x, y, th = xb.copy(), yb.copy(), tb.copy()
            vi = (3 * wave + s) % len(voids) if voids else -1
            vx, vy, _ = voids[vi] if vi >= 0 else (0.5 * L0, 0.5 * L0, 0.1)
            rad = float(rng.uniform(preset.radius_min, preset.radius_max))
            near = ((x - vx) ** 2 + (y - vy) ** 2) <= rad * rad
            if len(tight_idx) > 0 and s % 2 == 0:
                pi = int(tight_idx[(5 * wave + s // 2) % len(tight_idx)])
                px_old, py_old = float(x[pi]), float(y[pi])
                x[pi] = vx + float(rng.uniform(-0.08, 0.08))
                y[pi] = vy + float(rng.uniform(-0.08, 0.08))
                th[pi] = [0.0, np.pi / 4.0, tilt][(s // 2) % 3] + float(rng.uniform(-0.10, 0.10))
                near |= ((x - px_old) ** 2 + (y - py_old) ** 2) <= 3.5 ** 2
                near[pi] = True
            nm = int(near.sum())
            if nm > 0:
                x[near] += rng.uniform(-preset.coord_jitter, preset.coord_jitter, nm)
                y[near] += rng.uniform(-preset.coord_jitter, preset.coord_jitter, nm)
                th[near] += rng.uniform(-preset.angle_jitter, preset.angle_jitter, nm)
            q_list.append(full_to_free_sym(x, y, th, L0))
            m_list.append(_mask3(near))
            ls_list.append(sched_p)
            labels.append(f"mpk_void_w{wave}_{s}")

        # 2. Tight-contact stress clusters.
        for s in range(8):
            x, y, th = xb.copy(), yb.copy(), tb.copy()
            if hot_clusters_scaled:
                cx, cy, _ = hot_clusters_scaled[(3 * wave + s) % len(hot_clusters_scaled)]
            else:
                pi = int(tight_idx[s % len(tight_idx)])
                cx, cy = float(x[pi]), float(y[pi])
            rad = float(rng.uniform(preset.radius_min, preset.radius_max))
            near = ((x - cx) ** 2 + (y - cy) ** 2) <= rad * rad
            nm = int(near.sum())
            if nm > 0:
                drift = float(rng.uniform(0.03, 0.06)) if (s % 2 == 1) else 0.0
                x[near] += dx_sign * drift * 0.5 + rng.uniform(-preset.coord_jitter, preset.coord_jitter, nm)
                y[near] += dy_sign * drift * 0.5 + rng.uniform(-preset.coord_jitter, preset.coord_jitter, nm)
                th[near] += rng.uniform(-preset.angle_jitter, preset.angle_jitter, nm)
            q_list.append(full_to_free_sym(x, y, th, L0))
            m_list.append(_mask3(near))
            ls_list.append(sched_p)
            labels.append(f"mpk_hot_w{wave}_{s}")

        # 3. Upright/tilted seams and the two square-grid seam corners.
        c_seam = min(max(k - 0.5, 2.0), L0 - 2.0)
        c_sym = min(max(L0 - (k - 0.5), 2.0), L0 - 2.0)
        for s in range(6):
            x, y, th = xb.copy(), yb.copy(), tb.copy()
            if s == 0:
                cx, cy = c_seam, c_seam
            elif s == 1:
                cx, cy = c_sym, c_sym
            elif len(seam_idx) > 0:
                pi = int(seam_idx[(17 * wave + (s - 2) * 19) % len(seam_idx)])
                cx, cy = float(x[pi]), float(y[pi])
            else:
                cx, cy = 0.5 * L0, 0.5 * L0
            rad = float(rng.uniform(preset.radius_min, preset.radius_max))
            near = ((x - cx) ** 2 + (y - cy) ** 2) <= rad * rad
            nm = int(near.sum())
            if nm > 0:
                x[near] += rng.uniform(-preset.coord_jitter, preset.coord_jitter, nm)
                y[near] += rng.uniform(-preset.coord_jitter, preset.coord_jitter, nm)
                th[near] += rng.uniform(-preset.angle_jitter, preset.angle_jitter, nm)
            q_list.append(full_to_free_sym(x, y, th, L0))
            m_list.append(_mask3(near))
            ls_list.append(sched_p)
            labels.append(f"mpk_seam_w{wave}_{s}")

    return q_list, m_list, ls_list, labels
