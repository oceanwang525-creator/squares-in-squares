#!/usr/bin/env python3
"""Analyze GuestData's n=910 same-tilt dual-crystal construction and PR #21 novel packings."""
from __future__ import annotations

import collections
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "poc" / "swarm_homology" / "poc"))
import utils as u  # noqa: E402

AUDIT_UNION = Path("/mnt/e/squares-pr21-audit-nov13/proof/union.csv")
AUDIT_REPO = Path("/mnt/e/squares-pr21-audit-nov13/repo")
AUDIT_N910 = Path("/mnt/e/squares-pr21-audit-nov13/proof/n910-analysis.json")
OUT_DIR = REPO / "poc" / "swarm_sametilt" / "docs"


def hx_of(th: np.ndarray) -> np.ndarray:
    return 0.5 * (np.abs(np.cos(th)) + np.abs(np.sin(th)))


def norm(x: np.ndarray, y: np.ndarray, th: np.ndarray):
    x = np.asarray(x, float).copy()
    y = np.asarray(y, float).copy()
    th = np.asarray(th, float).copy()
    h = hx_of(th)
    x -= (x - h).min()
    y -= (y - h).min()
    L = float(max((x + h).max(), (y + h).max()))
    return x, y, th, L


def mod90_deg(th: np.ndarray) -> np.ndarray:
    d = np.degrees(th) % 90.0
    d = np.where(d > 89.999, 0.0, d)
    return d


def angle_summary(th: np.ndarray) -> dict:
    d = mod90_deg(th)
    near0 = int(np.sum((d < 5.0) | (d > 85.0)))
    near25 = int(np.sum((d >= 15.0) & (d <= 35.0)))
    near45 = int(np.sum((d > 35.0) & (d < 55.0)))
    near65 = int(np.sum((d >= 55.0) & (d <= 75.0)))
    other = len(d) - near0 - near25 - near45 - near65
    return {
        "n": len(d),
        "near0": near0,
        "near25": near25,
        "near45": near45,
        "near65": near65,
        "other": other,
    }


def analyze_lattice(name: str, x: np.ndarray, y: np.ndarray, th: np.ndarray):
    deg = mod90_deg(th)
    print(f"\n--- Region {name} (count={len(x)}) ---")
    print(f"  x in [{x.min():.4f}, {x.max():.4f}], y in [{y.min():.4f}, {y.max():.4f}]")
    print(f"  deg_mod90: mean={deg.mean():.4f}, std={deg.std():.4f}, min={deg.min():.4f}, max={deg.max():.4f}")
    # Find nearest-neighbor displacement vectors among tilted squares
    pts = np.c_[x, y]
    diffs = pts[None, :, :] - pts[:, None, :]
    dists = np.linalg.norm(diffs, axis=2)
    np.fill_diagonal(dists, 1e9)
    nn_d = dists.min(axis=1)
    print(f"  NN dist: min={nn_d.min():.5f}, median={np.median(nn_d):.5f}, max={nn_d.max():.5f}")
    # Cluster short vectors (< 1.25) with dx >= 0 or dy > 0
    vecs = []
    for i in range(len(pts)):
        for j in range(len(pts)):
            if i == j:
                continue
            if dists[i, j] < 1.18:
                dx, dy = diffs[i, j]
                if dx > 1e-3 or (abs(dx) <= 1e-3 and dy > 0):
                    vecs.append((dx, dy, dists[i, j]))
    vecs = np.array(vecs)
    if len(vecs):
        # Histogram by angle of displacement vector
        ang = np.degrees(np.arctan2(vecs[:, 1], vecs[:, 0]))
        bins = collections.Counter(np.round(ang, 0))
        print("  Short (<1.18) disp vector angles (top 6):")
        for a, cnt in bins.most_common(6):
            mask = np.abs(ang - a) < 2.5
            sub = vecs[mask]
            print(f"    ang~{a:6.1f} deg: count={cnt:4d}, mean_dx={sub[:,0].mean():.5f}, mean_dy={sub[:,1].mean():.5f}, mean_d={sub[:,2].mean():.5f}")


def main():
    df = pd.read_csv(AUDIT_UNION)
    print(f"Loaded {AUDIT_UNION} with {df['n'].nunique()} groups.")

    # Extract n=910
    sub910 = df[df["n"] == 910].copy()
    x, y, th, L = norm(sub910["x"].to_numpy(), sub910["y"].to_numpy(), sub910["angle"].to_numpy())
    deg = mod90_deg(th)
    print(f"\n=== GuestData n=910 (L = {L:.15f}) ===")
    print("Angle summary:", angle_summary(th))

    # Segment regions
    tilted = (deg > 50.0) & (deg < 80.0)
    left_tilted = tilted & (x < 11.2) & (y > 9.5)
    bot_tilted = tilted & (x > 11.2) & (y < 10.2)
    other_tilted = tilted & ~(left_tilted | bot_tilted)
    print(f"Tilted (~65deg) breakdown: left={left_tilted.sum()}, bot={bot_tilted.sum()}, other={other_tilted.sum()}")

    analyze_lattice("LEFT_65", x[left_tilted], y[left_tilted], th[left_tilted])
    analyze_lattice("BOT_65", x[bot_tilted], y[bot_tilted], th[bot_tilted])

    # Transition squares (between 5 and 85 mod 90, excluding main 65 clusters or checking 5..55 / 75..85)
    trans = ((deg >= 2.0) & (deg <= 55.0)) | ((deg >= 75.0) & (deg <= 88.0))
    print(f"\nTransition squares (2..55 or 75..88 deg): count={trans.sum()}")
    for i in np.where(trans)[0]:
        print(f"  idx={i:3d}: x={x[i]:8.4f}, y={y[i]:8.4f}, deg_mod90={deg[i]:7.3f}")

    # Axis squares (<2 or >88)
    axis = ~tilted & ~trans
    bl = axis & (x < 11.5) & (y < 10.5)
    tr = axis & (x >= 10.2) & (y >= 9.4)
    br = axis & (x >= 26.5) & (y < 9.5)
    tl_strip = axis & (x < 10.5) & (y >= 26.0)
    rem = axis & ~(bl | tr | br | tl_strip)
    print(f"\nAxis squares breakdown (total={axis.sum()}):")
    print(f"  BL (x<11.5, y<10.5): {bl.sum()}, x=[{x[bl].min():.3f},{x[bl].max():.3f}], y=[{y[bl].min():.3f},{y[bl].max():.3f}]")
    print(f"  TR (x>=10.2, y>=9.4): {tr.sum()}, x=[{x[tr].min():.3f},{x[tr].max():.3f}], y=[{y[tr].min():.3f},{y[tr].max():.3f}]")
    print(f"  BR (x>=26.5, y<9.5): {br.sum()}, x=[{x[br].min():.3f},{x[br].max():.3f}], y=[{y[br].min():.3f},{y[br].max():.3f}]")
    print(f"  TL_strip (x<10.5, y>=26): {tl_strip.sum()}")
    print(f"  Remaining axis: {rem.sum()}")
    for i in np.where(rem)[0]:
        print(f"    rem idx={i:3d}: x={x[i]:8.4f}, y={y[i]:8.4f}, deg={deg[i]:6.3f}")

    # Check TR grid structure (row/col spacing)
    tx, ty = x[tr], y[tr]
    print(f"\nTR grid: unique rounded x (~0.1): {len(np.unique(np.round(tx, 1)))}, unique rounded y (~0.1): {len(np.unique(np.round(ty, 1)))}")
    print("  TR x sorted sample:", np.round(np.sort(np.unique(np.round(tx, 2))), 2))
    print("  TR y sorted sample:", np.round(np.sort(np.unique(np.round(ty, 2))), 2))

    # Check largest clearance voids in n=910
    g = np.arange(0.5, L - 0.5 + 1e-9, 0.10)
    gx, gy = np.meshgrid(g, g, indexing="ij")
    px, py = gx.ravel(), gy.ravel()
    clr = np.minimum(np.minimum(px, L - px), np.minimum(py, L - py)) - 0.5
    c, s = np.cos(th), np.sin(th)
    for a in range(0, len(px), 4096):
        dx = px[a:a + 4096, None] - x[None, :]
        dy = py[a:a + 4096, None] - y[None, :]
        d = np.maximum(np.abs(c * dx + s * dy), np.abs(-s * dx + c * dy)) - 0.5
        clr[a:a + 4096] = np.minimum(clr[a:a + 4096], d.min(axis=1))
    order = np.argsort(-clr)
    voids = []
    for i in order:
        if len(voids) >= 15:
            break
        if all((px[i] - q[0]) ** 2 + (py[i] - q[1]) ** 2 > 0.85 ** 2 for q in voids):
            voids.append((float(px[i]), float(py[i]), float(clr[i])))
    print("\nTop 15 voids in GuestData n=910 (clearance to nearest square / wall):")
    for vx, vy, vc in voids:
        print(f"  (x={vx:7.3f}, y={vy:7.3f}) clr={vc:+.4f}")

    # Check per-square minimum distance to neighbors/walls (slack)
    h = hx_of(th)
    wall_clr = np.minimum(np.minimum(x - h, L - (x + h)), np.minimum(y - h, L - (y + h)))
    pts = np.c_[x, y]
    min_nb = np.full(len(x), 1e9)
    for i in range(len(x)):
        for j in range(len(x)):
            if i == j:
                continue
            dx, dy = x[j] - x[i], y[j] - y[i]
            if abs(dx) > 1.8 or abs(dy) > 1.8:
                continue
            rad = 0.5 * (1.0 + abs(c[i]*c[j] + s[i]*s[j]) + abs(c[i]*s[j] - s[i]*c[j]))
            gap = max(
                abs(dx*c[i] + dy*s[i]) - rad,
                abs(-dx*s[i] + dy*c[i]) - rad,
                abs(dx*c[j] + dy*s[j]) - rad,
                abs(-dx*s[j] + dy*c[j]) - rad,
            )
            if gap < min_nb[i]:
                min_nb[i] = gap
    sq_clr = np.minimum(wall_clr, min_nb)
    print("\nTop 15 highest-clearance (loosest) squares in n=910:")
    for i in np.argsort(-sq_clr)[:15]:
        print(f"  idx={i:3d}: x={x[i]:7.3f}, y={y[i]:7.3f}, deg={deg[i]:6.2f}, clr={sq_clr[i]:.5f}")

    # Inspect other novel PR #21 wins and target neighborhoods
    check_ns = sorted(set(
        [463, 639, 691, 692, 701, 745, 793, 858, 861, 918, 973]
        + [686, 739, 794, 851, 971]
        + list(range(905, 916))
        + list(range(848, 857))
        + list(range(969, 977))
    ))
    print("\n=== Target & Novel PR#21 Packings in audit union.csv ===")
    for n in check_ns:
        sub = df[df["n"] == n]
        if len(sub) == 0:
            continue
        xn, yn, thn, Ln = norm(sub["x"].to_numpy(), sub["y"].to_numpy(), sub["angle"].to_numpy())
        sm = angle_summary(thn)
        m = int(math.floor(Ln))
        print(f"  n={n:3d} (m={m}, k={n-m*m:+3d}): L={Ln:.12f} | axis={sm['near0']:3d}, 25d={sm['near25']:3d}, 45d={sm['near45']:3d}, 65d={sm['near65']:3d}, oth={sm['other']:2d}")


if __name__ == "__main__":
    main()
