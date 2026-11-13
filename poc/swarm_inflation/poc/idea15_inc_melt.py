#!/usr/bin/env python3
"""Idea 15: Argon Pivot — GPU Soft-a Incumbent Melt + Surplus Peels + Joint SLP.

Generates 1024-seed batches per target n from merged.csv incumbents:
  - Modes 0..4 (inc_melt): boundary-pin teleport, boundary-pin pocket melt,
    tilted-seam unlock, void-collapse melt, asymmetric wall-slack drift.
  - Surplus peels (n+1..n+5k+18 -> n, incl. n+1 -> n tip deletions & 972->971,
    792->791, 970->969): c in {1,2,3} border splits + interior c=1 cross-cuts.
  - Deficit (n-1 -> n) void-insertion fallback.
Writes binary surg_n<n>.bin in 8-slot striped family order for CUDA cpp_pipeline
(--idea 15, fast_mode=6 soft-a 0.85->1.00 and fast_mode=5 0.94->1.00) and finishes
exported GPU basins with exact-Jacobian joint_xytheta_slp_polish + dual_certify.
"""
from __future__ import annotations

import argparse
import math
import os
import struct
import subprocess
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from poc.swarm_inflation.poc import polish_and_certify as pc

REPO_ROOT = Path(__file__).resolve().parents[3]
MERGED_CSV = REPO_ROOT / "poc" / "pell_corrugation" / "data" / "border-scan" / "merged.csv"
ALL_TEN = [367, 687, 971, 796, 910, 972, 791, 969, 851, 333]


def load_all_incumbents(csv_path: Path = MERGED_CSV) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
    """Load and bottom-left normalize all 1..1000 packings from merged.csv."""
    df = pd.read_csv(csv_path)
    mats: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, float]] = {}
    for n, sub in df.groupby("n", sort=True):
        x = sub["x"].to_numpy(np.float64).copy()
        y = sub["y"].to_numpy(np.float64).copy()
        th = sub["angle"].to_numpy(np.float64).copy()
        hx = 0.5 * (np.abs(np.cos(th)) + np.abs(np.sin(th)))
        x -= float((x - hx).min())
        y -= float((y - hx).min())
        L = float(max((x + hx).max(), (y + hx).max()))
        mats[int(n)] = (x, y, th, L)
    return mats


def clearance_points(
    x: np.ndarray, y: np.ndarray, th: np.ndarray, L: float, n_pts: int = 8, step: float = 0.14
) -> list[tuple[float, float, float]]:
    """Return top n_pts (px, py, clearance) local maxima inside [0.5, L-0.5]^2."""
    g = np.arange(0.5, L - 0.5 + 1e-9, step)
    gx, gy = np.meshgrid(g, g, indexing="ij")
    px, py = gx.ravel(), gy.ravel()
    clr = np.minimum(np.minimum(px, L - px), np.minimum(py, L - py)) - 0.5
    c, s = np.cos(th), np.sin(th)
    for a in range(0, len(px), 4096):
        dx = px[a : a + 4096, None] - x[None, :]
        dy = py[a : a + 4096, None] - y[None, :]
        d = np.maximum(
            np.abs(c[None, :] * dx + s[None, :] * dy),
            np.abs(-s[None, :] * dx + c[None, :] * dy),
        ) - 0.5
        clr[a : a + 4096] = np.minimum(clr[a : a + 4096], d.min(axis=1))
    order = np.argsort(-clr)
    out: list[tuple[float, float, float]] = []
    for i in order:
        if len(out) >= n_pts:
            break
        if all((px[i] - q[0]) ** 2 + (py[i] - q[1]) ** 2 > 0.85 ** 2 for q in out):
            out.append((float(px[i]), float(py[i]), float(clr[i])))
    return out


def donor_tilt(th: np.ndarray) -> float:
    deg = np.degrees(th) % 90.0
    t = deg[(deg > 1.0) & (deg < 89.0)]
    return math.radians(float(np.median(t))) if len(t) else math.pi / 4


def _clamp_box(
    x: np.ndarray, y: np.ndarray, th: np.ndarray, L: float, pad: float = 0.002
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    th_w = ((th + 0.25 * math.pi) % (0.5 * math.pi)) - 0.25 * math.pi
    hx = 0.5 * (np.abs(np.cos(th_w)) + np.abs(np.sin(th_w)))
    return (
        np.clip(x, hx + pad, L - hx - pad),
        np.clip(y, hx + pad, L - hx - pad),
        th_w,
    )


def _inc_melt_seed(
    x_inc: np.ndarray,
    y_inc: np.ndarray,
    th_inc: np.ndarray,
    L_inc: float,
    mode: int,
    slot_idx: int,
    voids: list[tuple[float, float, float]],
    tight_idx: np.ndarray,
    tilted_idx: np.ndarray,
    dx_sign: float,
    dy_sign: float,
    tilt: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate one incumbent-melt seed (mode 0..4) pre-scaled strictly below L_inc."""
    deltas = (0.00015, 0.0004, 0.0010, 0.0022, 0.0045, 0.0090, 0.0140)
    delta = deltas[slot_idx % len(deltas)]
    L0 = L_inc - delta
    sc = L0 / L_inc
    x = x_inc * sc
    y = y_inc * sc
    th = th_inc.copy()
    n = len(x)

    if mode == 0 and voids:
        k_mv = 1 if (slot_idx % 2 == 0) else min(2, len(voids))
        picks = rng.choice(tight_idx[:14], size=k_mv, replace=False)
        for j, pi in enumerate(picks):
            vx, vy, _ = voids[(slot_idx + j) % len(voids)]
            x[pi] = vx + float(rng.uniform(-0.08, 0.08)) * (L0 / L_inc)
            y[pi] = vy + float(rng.uniform(-0.08, 0.08)) * (L0 / L_inc)
            th[pi] = [0.0, 0.25 * math.pi, tilt][(slot_idx + j) % 3] + float(rng.uniform(-0.08, 0.08))
            near = ((x - vx) ** 2 + (y - vy) ** 2) <= 4.4 ** 2
            nm = int(near.sum())
            x[near] += rng.uniform(-0.12, 0.12, nm)
            y[near] += rng.uniform(-0.12, 0.12, nm)
            th[near] += rng.uniform(-0.10, 0.10, nm)
    elif mode == 1:
        pi = int(tight_idx[slot_idx % len(tight_idx)])
        cx, cy = float(x[pi]), float(y[pi])
        rad = float(rng.uniform(4.2, 6.8))
        near = ((x - cx) ** 2 + (y - cy) ** 2) <= rad * rad
        nm = int(near.sum())
        x[near] += rng.uniform(-0.14, 0.14, nm)
        y[near] += rng.uniform(-0.14, 0.14, nm)
        th[near] += rng.uniform(-0.16, 0.16, nm)
    elif mode == 2:
        if len(tilted_idx) > 0:
            pi = int(tilted_idx[(slot_idx * 7) % len(tilted_idx)])
            cx, cy = float(x[pi]), float(y[pi])
            near = ((x - cx) ** 2 + (y - cy) ** 2) <= float(rng.uniform(5.0, 7.5)) ** 2
        else:
            near = np.ones(n, dtype=bool)
        nm = int(near.sum())
        x[near] += rng.uniform(-0.14, 0.14, nm)
        y[near] += rng.uniform(-0.14, 0.14, nm)
        th[near] += rng.uniform(-0.15, 0.15, nm)
    elif mode == 3 and voids:
        vx, vy, _ = voids[slot_idx % len(voids)]
        rad = float(rng.uniform(4.5, 7.8))
        near = ((x - vx) ** 2 + (y - vy) ** 2) <= rad * rad
        nm = int(near.sum())
        pull = float(rng.uniform(0.03, 0.08))
        x[near] += pull * (vx - x[near]) + rng.uniform(-0.11, 0.11, nm)
        y[near] += pull * (vy - y[near]) + rng.uniform(-0.11, 0.11, nm)
        th[near] += rng.uniform(-0.16, 0.16, nm)
    else:
        drift = float(rng.uniform(0.03, 0.09))
        if voids:
            cx = 0.5 * (float(x[tight_idx[slot_idx % 6]]) + voids[0][0])
            cy = 0.5 * (float(y[tight_idx[slot_idx % 6]]) + voids[0][1])
        else:
            cx, cy = float(x[tight_idx[0]]), float(y[tight_idx[0]])
        half_x = (x < L0 * 0.65) if dx_sign > 0 else (x > L0 * 0.35)
        half_y = (y < L0 * 0.65) if dy_sign > 0 else (y > L0 * 0.35)
        x[half_x] += dx_sign * drift
        y[half_y] += dy_sign * 0.6 * drift
        near = ((x - cx) ** 2 + (y - cy) ** 2) <= float(rng.uniform(5.2, 8.0)) ** 2
        nm = int(near.sum())
        x[near] += rng.uniform(-0.13, 0.13, nm)
        y[near] += rng.uniform(-0.13, 0.13, nm)
        th[near] += rng.uniform(-0.14, 0.14, nm)

    return _clamp_box(x, y, th, L0, pad=0.002)


def collect_surplus_peels(
    n: int, L_inc: float, mats: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, float]]
) -> tuple[list[dict], list[dict]]:
    """Collect border-peel/tip-deletion donors and interior cross-cut donors for target n."""
    k = math.isqrt(n)
    border_cands: list[dict] = []
    cross_cands: list[dict] = []
    seen = set()

    # 1) Direct n+1 -> n pocket-tip / corner-pin deletions (incl. 972->971, 792->791, 970->969)
    if (n + 1) in mats:
        x1, y1, th1, L1 = mats[n + 1]
        if L1 <= L_inc + 0.08:
            hx1 = 0.5 * (np.abs(np.cos(th1)) + np.abs(np.sin(th1)))
            oh1 = 0.05 + np.maximum.reduce([
                np.zeros(n + 1), hx1 - x1, x1 + hx1 - L1, hx1 - y1, y1 + hx1 - L1
            ])
            border_cands.append({
                "src": f"tip_{n+1}",
                "x": x1, "y": y1, "th": th1, "oh": oh1,
                "L": L1, "surplus": 1, "max_oh_rem": 0.02,
            })

    for n2 in range(n + 1, min(1000, n + 5 * k + 18) + 1):
        if n2 not in mats:
            continue
        x, y, th, L = mats[n2]
        hx = 0.5 * (np.abs(np.cos(th)) + np.abs(np.sin(th)))
        for c in (1, 2, 3):
            L_box = L - c
            if not (L_inc - 0.16 <= L_box <= L_inc - 0.0003):
                continue
            for cL in range(c + 1):
                cR = c - cL
                for cB in range(c + 1):
                    cT = c - cB
                    keep = np.ones(len(x), dtype=bool)
                    if cL > 0:
                        keep &= x >= cL + 0.05
                    if cR > 0:
                        keep &= x <= L - cR - 0.05
                    if cB > 0:
                        keep &= y >= cB + 0.05
                    if cT > 0:
                        keep &= y <= L - cT - 0.05
                    nr = int(keep.sum())
                    if not (n <= nr <= n + 6):
                        continue
                    xs = x[keep] - cL
                    ys = y[keep] - cB
                    ts = th[keep]
                    h2 = hx[keep]
                    oh = np.maximum.reduce([
                        np.zeros(nr), h2 - xs, xs + h2 - L_box, h2 - ys, ys + h2 - L_box,
                    ])
                    surp = nr - n
                    max_oh_rem = float(np.partition(oh, -(surp + 1))[-(surp + 1)]) if surp > 0 else float(oh.max())
                    if max_oh_rem >= 0.30:
                        continue
                    key = ("b", n2, nr, round(max_oh_rem, 4), round(L_box, 6))
                    if key in seen:
                        continue
                    seen.add(key)
                    border_cands.append({
                        "src": f"s{n2}_{cL}{cR}_{cB}{cT}",
                        "x": xs, "y": ys, "th": ts, "oh": oh,
                        "L": L_box, "surplus": surp, "max_oh_rem": max_oh_rem,
                    })

        if n2 <= n + 3 * k:
            L_box = L - 1.0
            if L_inc - 0.08 <= L_box <= L_inc - 0.0003:
                for xc in np.linspace(2.5, L - 2.5, 5):
                    kx = (x < xc - 0.5) | (x > xc + 0.5)
                    for yc in np.linspace(2.5, L - 2.5, 5):
                        keep = kx & ((y < yc - 0.5) | (y > yc + 0.5))
                        nr = int(keep.sum())
                        if not (n <= nr <= n + 4):
                            continue
                        xs = np.where(x[keep] > xc + 0.5, x[keep] - 1.0, x[keep])
                        ys = np.where(y[keep] > yc + 0.5, y[keep] - 1.0, y[keep])
                        ts = th[keep]
                        h2 = hx[keep]
                        seam_d = np.minimum(np.abs(xs - (xc - 0.5)), np.abs(ys - (yc - 0.5)))
                        oh = np.maximum(0.0, 0.75 - seam_d) + np.maximum.reduce([
                            np.zeros(nr), h2 - xs, xs + h2 - L_box, h2 - ys, ys + h2 - L_box,
                        ])
                        surp = nr - n
                        max_oh_rem = float(np.partition(oh, -(surp + 1))[-(surp + 1)]) if surp > 0 else float(oh.max())
                        if max_oh_rem < 0.60:
                            key = ("x", n2, nr, round(L_box, 6), round(max_oh_rem, 3))
                            if key not in seen:
                                seen.add(key)
                                cross_cands.append({
                                    "src": f"x{n2}_{int(xc)}_{int(yc)}",
                                    "x": xs, "y": ys, "th": ts, "oh": oh,
                                    "L": L_box, "surplus": surp, "max_oh_rem": max_oh_rem,
                                })

    border_cands.sort(key=lambda r: (r["max_oh_rem"], r["surplus"], r["L"]))
    cross_cands.sort(key=lambda r: (r["max_oh_rem"], r["surplus"], r["L"]))
    return border_cands, cross_cands


def _peel_seed(
    don: dict, n: int, L_inc: float, slot_idx: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Drop `surplus` highest-overhang/seam squares, clamp strictly below L_inc, and melt."""
    x0, y0, th0, oh0 = don["x"], don["y"], don["th"], don["oh"]
    surplus = don["surplus"]
    L_box = don["L"]
    L0 = min(L_box, L_inc - (0.0004 + 0.0012 * (slot_idx % 4)))
    sc = L0 / L_box
    rank = np.argsort(-oh0)
    pool_idx = rank[: max(surplus + 6, min(len(rank), 16))]

    if surplus == 0:
        keep = np.ones(len(x0), dtype=bool)
    elif slot_idx % 5 == 0:
        keep = np.ones(len(x0), dtype=bool)
        keep[rank[:surplus]] = False
    else:
        drop = rng.choice(pool_idx, size=surplus, replace=False)
        keep = np.ones(len(x0), dtype=bool)
        keep[drop] = False

    x = x0[keep] * sc
    y = y0[keep] * sc
    th = th0[keep].copy()
    oh = oh0[keep]
    x, y, th = _clamp_box(x, y, th, L0, pad=0.004)
    hot = np.where(oh > 0.01)[0]
    if len(hot) > 0 and (slot_idx % 7 != 0):
        cx, cy = float(x[hot[slot_idx % len(hot)]]), float(y[hot[slot_idx % len(hot)]])
        near = ((x - cx) ** 2 + (y - cy) ** 2) <= 3.8 ** 2
        nm = int(near.sum())
        x[near] += rng.uniform(-0.11, 0.11, nm)
        y[near] += rng.uniform(-0.11, 0.11, nm)
        th[near] += rng.uniform(-0.09, 0.09, nm)
        x, y, th = _clamp_box(x, y, th, L0, pad=0.003)
    return x, y, th


def build_seeds_for_n(
    n: int,
    mats: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, float]],
    count: int = 1024,
    rng_seed: int = 1500,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Build `count` striped seeds across 8 mechanisms matching cpp_pipeline --idea 15."""
    rng = np.random.default_rng(rng_seed + n * 31)
    x_inc, y_inc, th_inc, L_inc = mats[n]
    hx = 0.5 * (np.abs(np.cos(th_inc)) + np.abs(np.sin(th_inc)))
    margin = np.minimum.reduce([x_inc - hx, L_inc - x_inc - hx, y_inc - hx, L_inc - y_inc - hx])
    tight_idx = np.argsort(margin)[:20]
    voids = clearance_points(x_inc, y_inc, th_inc, L_inc, n_pts=8, step=0.14)
    tilt = donor_tilt(th_inc)
    deg = np.degrees(th_inc) % 90.0
    tilted_idx = np.where((deg > 2.0) & (deg < 88.0))[0]

    mL = x_inc - hx - float((x_inc - hx).min())
    mR = float((x_inc + hx).max()) - (x_inc + hx)
    mB = y_inc - hx - float((y_inc - hx).min())
    mT = float((y_inc + hx).max()) - (y_inc + hx)
    dx_sign = 1.0 if int(np.sum(mR < 5e-4)) <= int(np.sum(mL < 5e-4)) else -1.0
    dy_sign = 1.0 if int(np.sum(mT < 5e-4)) <= int(np.sum(mB < 5e-4)) else -1.0

    border_peels, cross_peels = collect_surplus_peels(n, L_inc, mats)

    # Optional n-1 void-insertion base
    has_def = (n - 1) in mats and mats[n - 1][3] < L_inc
    def_pts: list[tuple[float, float, float]] = []
    if has_def:
        xm, ym, thm, Lm = mats[n - 1]
        L_def = L_inc - 0.0006
        sc_d = (L_def - 1.0) / max(1e-9, Lm - 1.0)
        xb_d = 0.5 + (xm - 0.5) * sc_d
        yb_d = 0.5 + (ym - 0.5) * sc_d
        def_pts = clearance_points(xb_d, yb_d, thm, L_def, n_pts=6, step=0.12)
        def_tilt = donor_tilt(thm)

    out_x = np.zeros((count, n), dtype=np.float64)
    out_y = np.zeros((count, n), dtype=np.float64)
    out_th = np.zeros((count, n), dtype=np.float64)
    labels: list[str] = []

    for b in range(count):
        slot = b % 8
        grp = b // 8
        if slot < 5:
            xs, ys, ts = _inc_melt_seed(
                x_inc, y_inc, th_inc, L_inc, slot, grp, voids, tight_idx, tilted_idx,
                dx_sign, dy_sign, tilt, rng,
            )
            labels.append(f"melt_m{slot}")
        elif slot == 5 and border_peels:
            don = border_peels[grp % min(len(border_peels), 16)]
            xs, ys, ts = _peel_seed(don, n, L_inc, grp, rng)
            labels.append(don["src"])
        elif slot == 6 and (cross_peels or border_peels):
            pool = cross_peels if cross_peels else border_peels
            don = pool[grp % min(len(pool), 16)]
            xs, ys, ts = _peel_seed(don, n, L_inc, grp, rng)
            labels.append(don["src"])
        elif slot == 7 and has_def and def_pts:
            px, py, _ = def_pts[grp % len(def_pts)]
            L0 = L_inc - 0.0006
            ang_choices = [0.0, 0.25 * math.pi, def_tilt]
            xs = np.append(xb_d, px + float(rng.normal(0.0, 0.05)))
            ys = np.append(yb_d, py + float(rng.normal(0.0, 0.05)))
            ts = np.append(thm, ang_choices[(grp // len(def_pts)) % 3] + float(rng.normal(0.0, 0.04)))
            near = ((xs - px) ** 2 + (ys - py) ** 2) <= 3.6 ** 2
            nm = int(near.sum())
            xs[near] += rng.uniform(-0.10, 0.10, nm)
            ys[near] += rng.uniform(-0.10, 0.10, nm)
            xs, ys, ts = _clamp_box(xs, ys, ts, L0, pad=0.003)
            labels.append(f"def_{n-1}")
        else:
            xs, ys, ts = _inc_melt_seed(
                x_inc, y_inc, th_inc, L_inc, grp % 5, grp, voids, tight_idx, tilted_idx,
                dx_sign, dy_sign, tilt, rng,
            )
            labels.append(f"melt_fb{grp%5}")

        out_x[b] = xs
        out_y[b] = ys
        out_th[b] = ts

    return out_x, out_y, out_th, labels


def write_surg_bin(path: Path, x: np.ndarray, y: np.ndarray, th: np.ndarray) -> None:
    M, n = x.shape
    with open(path, "wb") as f:
        f.write(struct.pack("ii", int(M), int(n)))
        for b in range(M):
            f.write(np.ascontiguousarray(x[b], dtype=np.float64).tobytes())
            f.write(np.ascontiguousarray(y[b], dtype=np.float64).tobytes())
            f.write(np.ascontiguousarray(th[b], dtype=np.float64).tobytes())


def read_top_bin(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with open(path, "rb") as f:
        M, n = struct.unpack("ii", f.read(8))
        pos = np.zeros((M, n, 2), dtype=np.float64)
        th = np.zeros((M, n), dtype=np.float64)
        for k in range(M):
            pos[k, :, 0] = np.frombuffer(f.read(8 * n), dtype=np.float64)
            pos[k, :, 1] = np.frombuffer(f.read(8 * n), dtype=np.float64)
            th[k, :] = np.frombuffer(f.read(8 * n), dtype=np.float64)
    return pos, th


def run_idea15(n_list: list[int], batch: int, fast_modes: list[int], seed: int, run_gpu: bool = True) -> None:
    mats = load_all_incumbents()
    surg_dir = Path("/tmp/v15_surg")
    surg_dir.mkdir(parents=True, exist_ok=True)
    export_dir = Path("/tmp/v15_top")
    export_dir.mkdir(parents=True, exist_ok=True)

    for n in n_list:
        sx, sy, st, labels = build_seeds_for_n(n, mats, count=batch, rng_seed=seed)
        write_surg_bin(surg_dir / f"surg_n{n}.bin", sx, sy, st)
        print(
            f"[v15_seed] n={n} inc_L={mats[n][3]:.6f}: wrote {batch} seeds "
            f"(peels={sum(1 for l in labels if not l.startswith('melt'))}) -> {surg_dir / f'surg_n{n}.bin'}",
            flush=True,
        )

    if run_gpu:
        n_arg = ",".join(str(x) for x in n_list)
        env = dict(os.environ, OMP_NUM_THREADS="4")
        for fm in fast_modes:
            cmd = [
                "/tmp/cpp_pipeline",
                "--idea", "15",
                "--n-list", n_arg,
                "--batch", str(batch),
                "--fast-mode", str(fm),
                "--elastic-L", "1",
                "--seed", str(seed + fm * 10),
                "--surg-dir", str(surg_dir),
                "--export-top", str(export_dir),
            ]
            subprocess.run(cmd, check=True, env=env)

    # Finish top exported GPU basins with joint_xytheta_slp_polish + dual_certify
    data_dir = REPO_ROOT / "poc" / "swarm_inflation" / "data"
    for n in n_list:
        all_p, all_t, all_m = [], [], []
        for fm in fast_modes:
            fp = export_dir / f"top_n{n}_fm{fm}.bin"
            if not fp.exists():
                continue
            pos, th = read_top_bin(fp)
            for k in range(min(len(pos), 16)):
                all_p.append(pos[k])
                all_t.append(th[k])
                all_m.append(f"v15_fm{fm}#{k}")
        if not all_p:
            continue
        res = pc.polish_candidates(
            n,
            np.asarray(all_p, dtype=np.float64),
            np.asarray(all_t, dtype=np.float64),
            all_m,
            out_win_csv=data_dir / f"win_n{n}.csv",
            out_best_csv=data_dir / f"cand_v15_n{n}.csv",
            lp_rounds=10,
            slp_rounds=16,
            slp_top_k=10,
        )
        print(
            f"[v15_slp] n={n} best_L_lit={res['best_L_lit']} inc={res['inc_L']} win={res['win']}",
            flush=True,
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-list", type=str, default="367,687,971")
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--fast-modes", type=str, default="6,5")
    ap.add_argument("--seed", type=int, default=1500)
    ap.add_argument("--no-gpu", action="store_true")
    args = ap.parse_args()
    n_list = [int(s) for s in args.n_list.split(",") if s.strip()]
    fms = [int(s) for s in args.fast_modes.split(",") if s.strip()]
    run_idea15(n_list, args.batch, fms, args.seed, run_gpu=not args.no_gpu)


if __name__ == "__main__":
    main()
