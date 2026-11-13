"""End-cap strip detection, explicit gap-release sliding, and asymmetric-slack pocket seed builders."""
from __future__ import annotations

import numpy as np


def _half_extents(th: np.ndarray) -> np.ndarray:
    return 0.5 * (np.abs(np.cos(th)) + np.abs(np.sin(th)))


def detect_endcap_strips(
    x: np.ndarray,
    y: np.ndarray,
    th: np.ndarray,
    L: float,
) -> dict:
    """Detect TR junction (cut_x, cut_y), right/top end-cap upright strips, and arm slacks."""
    hx = _half_extents(th)
    deg = np.degrees(th) % 90.0
    up = np.minimum(deg, 90.0 - deg) < 1.5
    tilted = ~up

    if np.any(tilted):
        armA_t = tilted & (x < y)
        armB_t = tilted & (y < x)
        cut_x = float((x[armA_t] + hx[armA_t]).max()) if np.any(armA_t) else 0.4 * L
        cut_y = float((y[armB_t] + hx[armB_t]).max()) if np.any(armB_t) else 0.4 * L
        bl_x = float((x[armB_t] - hx[armB_t]).min()) if np.any(armB_t) else 0.3 * L
        bl_y = float((y[armA_t] - hx[armA_t]).min()) if np.any(armA_t) else 0.3 * L
    else:
        cut_x = cut_y = 0.4 * L
        bl_x = bl_y = 0.3 * L

    # Right end-cap upright strip: near x = L - 0.5, stopping before full TR rows (row_upright_cnt >= 10)
    r_cand = np.where(up & (x > L - 1.15) & (y - 0.5 < cut_y + 0.5))[0]
    r_cand = r_cand[np.argsort(y[r_cand])]
    r_keep = []
    for i in r_cand:
        row_cnt = int(np.sum(up & (np.abs(y - y[i]) < 0.25) & (x > cut_x - 0.5)))
        if row_cnt >= 10 and y[i] > cut_y - 0.6:
            break
        r_keep.append(i)
    r_idx = np.array(r_keep, dtype=np.int64)

    # Top end-cap upright strip: near y = L - 0.5, stopping before full TR cols (col_upright_cnt >= 10)
    t_cand = np.where(up & (y > L - 1.15) & (x - 0.5 < cut_x + 0.5))[0]
    t_cand = t_cand[np.argsort(x[t_cand])]
    t_keep = []
    for i in t_cand:
        col_cnt = int(np.sum(up & (np.abs(x - x[i]) < 0.25) & (y > cut_y - 0.5)))
        if col_cnt >= 10 and x[i] > cut_x - 0.6:
            break
        t_keep.append(i)
    t_idx = np.array(t_keep, dtype=np.int64)

    armA_mask = x < cut_x + 0.1
    armB_mask = y < cut_y + 0.1
    armA_max_y = float((y[armA_mask] + hx[armA_mask]).max()) if np.any(armA_mask) else L
    armB_max_x = float((x[armB_mask] + hx[armB_mask]).max()) if np.any(armB_mask) else L

    return {
        "cut_x": cut_x,
        "cut_y": cut_y,
        "bl_x": bl_x,
        "bl_y": bl_y,
        "r_idx": r_idx,
        "t_idx": t_idx,
        "dA": armA_max_y - L,
        "dB": armB_max_x - L,
        "r_y0": float(y[r_idx[0]]) if len(r_idx) else 0.5,
        "t_x0": float(x[t_idx[0]]) if len(t_idx) else 0.5,
    }


def slide_endcap_strip(
    x: np.ndarray,
    y: np.ndarray,
    th: np.ndarray,
    L: float,
    strip_idx: np.ndarray,
    axis: str,
    inward_dx: float = 0.15,
    tilt_last_rad: float | None = None,
    drop_last: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Slide first m-1 squares of an end-cap strip against the wall, moving gap to TR junction.

    For axis='right': slides y to 0.5, 1.5, ..., m-1.5; shifts last square inward to
    (L - 0.5 - inward_dx, m - 0.5).
    For axis='top': slides x to 0.5, 1.5, ..., m-1.5; shifts last square inward to
    (m - 0.5, L - 0.5 - inward_dx).
    Returns (nx, ny, nth, moved_idx).
    """
    nx, ny, nth = x.copy(), y.copy(), th.copy()
    idx = strip_idx[:-1] if (drop_last and len(strip_idx) >= 4) else strip_idx
    m = len(idx)
    if m < 3:
        return nx, ny, nth, np.array([], dtype=np.int64)

    wall_pos = L - 0.5005
    if axis == "right":
        for k in range(m - 1):
            nx[idx[k]] = wall_pos
            ny[idx[k]] = 0.5 + float(k)
            nth[idx[k]] = 0.0
        last = idx[-1]
        nx[last] = L - 0.5 - inward_dx
        ny[last] = float(m - 1) + 0.5
        nth[last] = 0.0 if tilt_last_rad is None else float(tilt_last_rad)
    elif axis == "top":
        for k in range(m - 1):
            ny[idx[k]] = wall_pos
            nx[idx[k]] = 0.5 + float(k)
            nth[idx[k]] = 0.0
        last = idx[-1]
        ny[last] = L - 0.5 - inward_dx
        nx[last] = float(m - 1) + 0.5
        nth[last] = 0.0 if tilt_last_rad is None else float(tilt_last_rad)
    else:
        raise ValueError(f"Unknown axis={axis!r}")

    return nx, ny, nth, idx


def _mask3(active: np.ndarray) -> np.ndarray:
    m = active.astype(np.uint8)
    return np.concatenate([m, m, m])


def _full_to_free_sym(x: np.ndarray, y: np.ndarray, th: np.ndarray, L: float) -> np.ndarray:
    hc = 0.5 * L
    return np.concatenate([x - hc, y - hc, th]).astype(np.float64)


def _donor_tilt(th: np.ndarray) -> float:
    deg = np.degrees(th) % 90.0
    m = np.minimum(deg, 90.0 - deg) > 3.0
    if not np.any(m):
        return float(np.radians(26.565))
    vals = th[m] % (0.5 * np.pi)
    return float(np.median(vals))


def build_armslack_seeds(
    n: int,
    x_base: np.ndarray,
    y_base: np.ndarray,
    th_base: np.ndarray,
    L_base: float,
    rng: np.random.Generator,
    mode: str = "auto",
) -> tuple[list[np.ndarray], list[np.ndarray], list[list[float]], list[str]]:
    """Build explicit strip-slide + masked-pocket melt seeds for H12 gate & scaling."""
    L0 = L_base + 0.032
    sc = L0 / L_base
    xb, yb, tb = x_base * sc, y_base * sc, th_base.copy()
    info = detect_endcap_strips(xb, yb, tb, L0)
    cut_x, cut_y = info["cut_x"], info["cut_y"]
    r_idx, t_idx = info["r_idx"], info["t_idx"]
    tilt = _donor_tilt(tb)

    has_asym_b = info["dB"] < -2e-3
    has_asym_a = info["dA"] < -2e-3
    if mode == "auto":
        mode = "asym" if (has_asym_b or has_asym_a) else "strip"

    sched_std = [L0, L_base + 0.008, L_base + 0.0005, L_base - 0.0010]
    sched_deep = [L0, L_base + 0.006, L_base - 0.0002, L_base - 0.0025]

    q_list, m_list, ls_list, labels = [], [], [], []

    def add_seed(x: np.ndarray, y: np.ndarray, th: np.ndarray, near: np.ndarray, sched: list[float], label: str, jitter: float = 0.08):
        nm = int(near.sum())
        if nm == 0:
            return
        xs, ys, ts = x.copy(), y.copy(), th.copy()
        xs[near] += rng.uniform(-jitter, jitter, nm)
        ys[near] += rng.uniform(-jitter, jitter, nm)
        ts[near] += rng.uniform(-0.18, 0.18, nm)
        q_list.append(_full_to_free_sym(xs, ys, ts, L0))
        m_list.append(_mask3(near))
        ls_list.append(sched)
        labels.append(label)

    if mode == "strip":
        # 1. Right-strip slide + TR-B junction pocket (8 seeds)
        for s in range(8):
            drop_last = (s % 2 == 1) and len(r_idx) >= 5
            inward = float(rng.uniform(0.13, 0.22))
            tilt_last = None if s < 6 else (tilt if s == 6 else np.pi / 4.0)
            x, y, th, moved = slide_endcap_strip(
                xb, yb, tb, L0, r_idx, "right", inward_dx=inward, tilt_last_rad=tilt_last, drop_last=drop_last
            )
            cx, cy = L0 - 1.5, max(3.0, cut_y - 1.0)
            rad = float(rng.uniform(4.8, 7.0))
            near = ((x - cx) ** 2 + (y - cy) ** 2) <= rad * rad
            near[moved] = True
            # Nudge adjacent TR junction squares slightly inward-left
            adj = near & (x > L0 - 3.2) & (y > cut_y - 2.0) & (y < cut_y + 1.5)
            x[adj] -= float(rng.uniform(0.02, 0.06))
            add_seed(x, y, th, near, sched_std if s < 6 else sched_deep, f"r_slide_{s}")

        # 2. Top-strip slide + TR-A junction pocket (8 seeds)
        for s in range(8):
            drop_last = (s % 2 == 1) and len(t_idx) >= 5
            inward = float(rng.uniform(0.13, 0.22))
            tilt_last = None if s < 6 else (0.5 * np.pi - tilt if s == 6 else np.pi / 4.0)
            x, y, th, moved = slide_endcap_strip(
                xb, yb, tb, L0, t_idx, "top", inward_dx=inward, tilt_last_rad=tilt_last, drop_last=drop_last
            )
            cx, cy = max(3.0, cut_x - 1.0), L0 - 1.5
            rad = float(rng.uniform(4.8, 7.0))
            near = ((x - cx) ** 2 + (y - cy) ** 2) <= rad * rad
            near[moved] = True
            adj = near & (y > L0 - 3.2) & (x > cut_x - 2.0) & (x < cut_x + 1.5)
            y[adj] -= float(rng.uniform(0.02, 0.06))
            add_seed(x, y, th, near, sched_std if s < 6 else sched_deep, f"t_slide_{s}")

        # 3. Dual Right+Top strip slide + both junction pockets (8 seeds)
        for s in range(8):
            drop_last = (s % 2 == 1) and min(len(r_idx), len(t_idx)) >= 5
            inw_r = float(rng.uniform(0.13, 0.20))
            inw_t = float(rng.uniform(0.13, 0.20))
            x, y, th, mr = slide_endcap_strip(xb, yb, tb, L0, r_idx, "right", inward_dx=inw_r, drop_last=drop_last)
            x, y, th, mt = slide_endcap_strip(x, y, th, L0, t_idx, "top", inward_dx=inw_t, drop_last=drop_last)
            rad = float(rng.uniform(4.8, 6.8))
            near_r = ((x - (L0 - 1.5)) ** 2 + (y - max(3.0, cut_y - 1.0)) ** 2) <= rad * rad
            near_t = ((x - max(3.0, cut_x - 1.0)) ** 2 + (y - (L0 - 1.5)) ** 2) <= rad * rad
            near = near_r | near_t
            near[mr] = True
            near[mt] = True
            add_seed(x, y, th, near, sched_std if s < 5 else sched_deep, f"rt_slide_{s}")

    else:
        # Asymmetric-slack corridor equalization (e.g. n=803, 986, 923)
        # Tight arm is Arm A if dB < dA (Arm B has slack), else Arm B.
        tight_is_a = info["dB"] <= info["dA"]
        tight_idx = t_idx if tight_is_a else r_idx
        tight_axis = "top" if tight_is_a else "right"
        slack_idx = r_idx if tight_is_a else t_idx
        slack_axis = "right" if tight_is_a else "top"
        bl_cx = min(cut_x, info["bl_x"]) - 0.3
        bl_cy = min(cut_y, info["bl_y"]) - 0.3
        bl_cx = max(4.5, min(bl_cx, 10.0))
        bl_cy = max(4.5, min(bl_cy, 10.0))

        # 1. Tight-arm end-cap slide + dual pocket (outer end-cap + BL corner) (10 seeds)
        for s in range(10):
            drop_last = (s % 2 == 1) and len(tight_idx) >= 5
            inw = float(rng.uniform(0.14, 0.22))
            x, y, th, mt = slide_endcap_strip(xb, yb, tb, L0, tight_idx, tight_axis, inward_dx=inw, drop_last=drop_last)
            rad1 = float(rng.uniform(5.0, 7.0))
            rad2 = float(rng.uniform(5.0, 7.0))
            if tight_is_a:
                ec_x, ec_y = max(3.0, 0.5 * cut_x), L0 - 1.5
            else:
                ec_x, ec_y = L0 - 1.5, max(3.0, 0.5 * cut_y)
            near = (((x - ec_x) ** 2 + (y - ec_y) ** 2) <= rad1 * rad1) | (
                ((x - bl_cx) ** 2 + (y - bl_cy) ** 2) <= rad2 * rad2
            )
            near[mt] = True
            # Gentle push toward BL corner and slack arm
            if tight_is_a:
                y[near & (y > cut_y)] -= float(rng.uniform(0.01, 0.04))
            else:
                x[near & (x > cut_x)] -= float(rng.uniform(0.01, 0.04))
            add_seed(x, y, th, near, sched_deep if s % 2 == 0 else sched_std, f"asym_ec_bl_{s}")

        # 2. Full tight corridor + BL corner + slack-arm inner end (8 seeds)
        for s in range(8):
            inw = float(rng.uniform(0.14, 0.20))
            x, y, th, mt = slide_endcap_strip(xb, yb, tb, L0, tight_idx, tight_axis, inward_dx=inw, drop_last=(s % 2 == 1))
            if s % 3 == 0 and len(slack_idx) >= 4:
                x, y, th, ms = slide_endcap_strip(x, y, th, L0, slack_idx, slack_axis, inward_dx=inw, drop_last=False)
            else:
                ms = np.array([], dtype=np.int64)
            if tight_is_a:
                near = ((x < cut_x + 2.0) & (y > bl_cy - 2.0)) | (((x - bl_cx) ** 2 + (y - bl_cy) ** 2) <= 6.2 ** 2)
                y[near & (y > cut_y)] -= float(rng.uniform(0.015, 0.04))
            else:
                near = ((y < cut_y + 2.0) & (x > bl_cx - 2.0)) | (((x - bl_cx) ** 2 + (y - bl_cy) ** 2) <= 6.2 ** 2)
                x[near & (x > cut_x)] -= float(rng.uniform(0.015, 0.04))
            near[mt] = True
            if len(ms) > 0:
                near[ms] = True
            add_seed(x, y, th, near, sched_deep, f"asym_corridor_{s}", jitter=0.07)

        # 3. Tight-arm junction pocket (around cut) + BL corner (6 seeds)
        for s in range(6):
            inw = float(rng.uniform(0.14, 0.22))
            x, y, th, mt = slide_endcap_strip(xb, yb, tb, L0, tight_idx, tight_axis, inward_dx=inw, drop_last=(s % 2 == 1))
            rad = float(rng.uniform(5.0, 7.0))
            jx, jy = (max(3.0, cut_x - 1.0), L0 - 1.5) if tight_is_a else (L0 - 1.5, max(3.0, cut_y - 1.0))
            near = (((x - jx) ** 2 + (y - jy) ** 2) <= rad * rad) | (
                ((x - bl_cx) ** 2 + (y - bl_cy) ** 2) <= 5.5 ** 2
            )
            near[mt] = True
            add_seed(x, y, th, near, sched_std if s < 3 else sched_deep, f"asym_junc_bl_{s}")

    return q_list, m_list, ls_list, labels
