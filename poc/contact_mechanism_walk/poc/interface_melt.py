#!/usr/bin/env python3
"""Target the tilted/upright contact interface of a certified candidate."""
from __future__ import annotations

import argparse
import json
import os
import sys
from decimal import Decimal
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import mechanism_walk as mw  # noqa: E402


def load_candidate(path: Path) -> tuple[int, Decimal, np.ndarray, np.ndarray, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    n = int(payload["n"])
    rows = payload["rows"]
    if len(rows) != n:
        raise ValueError(f"expected {n} rows, found {len(rows)}")
    x = np.array([float(row["x"].lstrip("s")) for row in rows], dtype=np.float64)
    y = np.array([float(row["y"].lstrip("s")) for row in rows], dtype=np.float64)
    theta = np.array(
        [float(row["angle"].lstrip("s")) for row in rows], dtype=np.float64
    )
    return n, Decimal(str(payload["incumbent_L"])), x, y, theta


def orientation_deviation(theta: np.ndarray) -> np.ndarray:
    degrees = np.mod(np.degrees(theta), 90.0)
    return np.minimum(degrees, 90.0 - degrees)


def separating_axis_gap(
    x: np.ndarray,
    y: np.ndarray,
    theta: np.ndarray,
    i: int,
    j: int,
) -> float:
    c, s = np.cos(theta), np.sin(theta)
    dx, dy = x[j] - x[i], y[j] - y[i]
    axes = (
        (c[i], s[i]),
        (-s[i], c[i]),
        (c[j], s[j]),
        (-s[j], c[j]),
    )
    cth = c[i] * c[j] + s[i] * s[j]
    sth = c[i] * s[j] - s[i] * c[j]
    radius = 0.5 * (1.0 + abs(cth) + abs(sth))
    return max(abs(dx * ax + dy * ay) - radius for ax, ay in axes)


def cross_orientation_interface(
    x: np.ndarray,
    y: np.ndarray,
    theta: np.ndarray,
    upright_tol_deg: float,
    gap_tol: float,
    search_radius: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    deviation = orientation_deviation(theta)
    upright = np.flatnonzero(deviation < upright_tol_deg)
    tilted = np.flatnonzero(deviation >= upright_tol_deg)
    pairs: list[tuple[int, int, float]] = []
    for i in tilted:
        for j in upright:
            if abs(x[i] - x[j]) > search_radius or abs(y[i] - y[j]) > search_radius:
                continue
            gap = separating_axis_gap(x, y, theta, int(i), int(j))
            if gap <= gap_tol:
                pairs.append((int(i), int(j), float(gap)))
    if not pairs:
        raise ValueError("candidate has no tilted/upright interface contacts")

    tilted_iface = np.array(sorted({i for i, _, _ in pairs}), dtype=np.int64)
    upright_iface = np.array(sorted({j for _, j, _ in pairs}), dtype=np.int64)
    both_iface = np.array(
        sorted(set(tilted_iface.tolist()) | set(upright_iface.tolist())),
        dtype=np.int64,
    )
    return tilted_iface, upright_iface, both_iface


def mask3(indices: np.ndarray, n: int) -> np.ndarray:
    active = np.zeros(n, dtype=np.uint8)
    active[indices] = 1
    # cpp_batch_melt expands the per-square mask over x, y, and theta.
    return active


def centered_for_box(
    x: np.ndarray,
    y: np.ndarray,
    source_side: float,
    target_side: float,
) -> tuple[np.ndarray, np.ndarray]:
    return (
        x + 0.5 * (target_side - source_side),
        y + 0.5 * (target_side - source_side),
    )


def build_seeds(
    x: np.ndarray,
    y: np.ndarray,
    theta: np.ndarray,
    side: float,
    tilted_iface: np.ndarray,
    upright_iface: np.ndarray,
    both_iface: np.ndarray,
) -> tuple[
    list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    list[np.ndarray],
    list[list[float]],
    list[str],
]:
    n = len(x)
    rng = np.random.default_rng(794_2026)
    seeds: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    masks: list[np.ndarray] = []
    schedules: list[list[float]] = []
    labels: list[str] = []

    def add(
        label: str,
        active: np.ndarray,
        *,
        slack: float,
        jitter: float,
        angle_jitter: float,
        move: str = "none",
    ) -> None:
        l0 = side + slack
        sx, sy = centered_for_box(x.copy(), y.copy(), side, l0)
        st = theta.copy()
        if move == "apart":
            for i, j in zip(tilted_iface, upright_iface[: len(tilted_iface)]):
                dx, dy = sx[j] - sx[i], sy[j] - sy[i]
                norm = float(np.hypot(dx, dy))
                if norm > 1e-12:
                    sx[i] -= 0.012 * dx / norm
                    sy[i] -= 0.012 * dy / norm
                    sx[j] += 0.012 * dx / norm
                    sy[j] += 0.012 * dy / norm
        elif move == "together":
            for i, j in zip(tilted_iface, upright_iface[: len(tilted_iface)]):
                dx, dy = sx[j] - sx[i], sy[j] - sy[i]
                norm = float(np.hypot(dx, dy))
                if norm > 1e-12:
                    sx[i] += 0.006 * dx / norm
                    sy[i] += 0.006 * dy / norm
                    sx[j] -= 0.006 * dx / norm
                    sy[j] -= 0.006 * dy / norm
        if jitter > 0.0:
            sx[active] += rng.uniform(-jitter, jitter, len(active))
            sy[active] += rng.uniform(-jitter, jitter, len(active))
        if angle_jitter > 0.0:
            st[active] += rng.uniform(-angle_jitter, angle_jitter, len(active))
        seeds.append((sx, sy, st))
        masks.append(mask3(active, n))
        end = side + min(-2e-4, -0.08 * slack)
        schedules.append([l0, side + 0.0015, side - 3e-5, end])
        labels.append(label)

    # Deterministic starts check whether a masked re-jam alone changes topology.
    for idx, (active, mask_name) in enumerate(
        (
            (both_iface, "both"),
            (tilted_iface, "tilted"),
            (upright_iface, "upright"),
        )
    ):
        for slack in (0.003, 0.010, 0.024):
            add(
                f"{mask_name}_plain_s{slack:g}",
                active,
                slack=slack,
                jitter=0.0,
                angle_jitter=0.0,
            )

    # Mixed-interface jitter repairs both sides of the seam at once.
    for seed_idx in range(18):
        jitter = (0.025, 0.05, 0.09)[seed_idx % 3]
        angle_jitter = (0.02, 0.05, 0.09)[(seed_idx // 3) % 3]
        slack = (0.006, 0.014, 0.028)[seed_idx % 3]
        add(
            f"both_random_{seed_idx}_j{jitter:g}_a{angle_jitter:g}",
            both_iface,
            slack=slack,
            jitter=jitter,
            angle_jitter=angle_jitter,
        )

    # One-sided masks expose whether only one orientation family should move.
    for seed_idx in range(8):
        for active, mask_name in (
            (tilted_iface, "tilted"),
            (upright_iface, "upright"),
        ):
            add(
                f"{mask_name}_random_{seed_idx}",
                active,
                slack=0.008 + 0.004 * (seed_idx % 3),
                jitter=0.035 + 0.012 * (seed_idx % 3),
                angle_jitter=0.03 + 0.015 * (seed_idx % 3),
            )

    # Directed pair moves create opening/closing room without global noise.
    for move in ("apart", "together"):
        for seed_idx in range(6):
            add(
                f"both_{move}_{seed_idx}",
                both_iface,
                slack=0.006 + 0.006 * seed_idx,
                jitter=0.012 + 0.008 * seed_idx,
                angle_jitter=0.02 + 0.01 * seed_idx,
                move=move,
            )
    return seeds, masks, schedules, labels


def polish(n: int, incumbent: Decimal, row: dict) -> tuple[Decimal, dict] | None:
    x, y, _ = mw.normalize(row["x"], row["y"], row["th"])
    theta = np.asarray(row["th"], dtype=np.float64).copy()
    best: tuple[Decimal, dict] | None = None
    for rounds, tr_theta, tr_xy in (
        (12, 0.004, 0.006),
        (16, 0.0015, 0.0025),
        (20, 0.0006, 0.001),
    ):
        sx, sy, st, _, ok = mw.su.joint_xytheta_slp(
            x,
            y,
            theta,
            rounds=rounds,
            tr_th0=tr_theta,
            tr_xy0=tr_xy,
            margin=1e-9,
        )
        if not ok:
            continue
        for margin in (1e-8, 5e-9, 2e-9, 1e-9, 5e-10):
            fx, fy, _, fixed_ok = mw.su.fixed_angle_lp_tighten(
                sx, sy, st, rounds=20, margin=margin
            )
            if not fixed_ok:
                continue
            verdict = mw.su.verify_official(n, fx, fy, st)
            if not verdict.get("ok"):
                continue
            candidate = max(
                Decimal(verdict["L_lit"]),
                Decimal(verdict["L_pandas"]),
            )
            if best is None or candidate < best[0]:
                best = (
                    candidate,
                    {
                        "L": str(candidate),
                        "dL": float(candidate - incumbent),
                        "rows": verdict["rows"],
                        "settings": (rounds, tr_theta, tr_xy, margin),
                    },
                )
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gap-tol", type=float, default=1e-4)
    parser.add_argument("--upright-tol-deg", type=float, default=0.5)
    parser.add_argument("--search-radius", type=float, default=1.8)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    n, incumbent, x, y, theta = load_candidate(args.candidate)
    tilted, upright, both = cross_orientation_interface(
        x,
        y,
        theta,
        args.upright_tol_deg,
        args.gap_tol,
        args.search_radius,
    )
    source_side = float(
        max(
            np.max(x + mw.half_extent(theta)),
            np.max(y + mw.half_extent(theta)),
        )
    )
    seeds, masks, schedules, labels = build_seeds(
        x, y, theta, source_side, tilted, upright, both
    )
    print(
        f"n={n} interface tilted={len(tilted)} upright={len(upright)} "
        f"both={len(both)} seeds={len(seeds)}",
        flush=True,
    )
    rows = mw.su.cpp_batch_melt(
        seeds,
        schedules,
        sym_mode=1,
        a_start=0.86,
        n_melt_stages=4,
        seam_masks=masks,
        seam_iters=180,
        full_iters=280,
        n_ruin_iters=4,
        n_threads=min(4, max(1, args.threads)),
        base_seed=794_2201,
    )
    ranked = sorted(
        range(len(rows)),
        key=lambda i: (
            0 if rows[i]["best_L_feas"] is not None else 1,
            rows[i]["best_L_feas"]
            if rows[i]["best_L_feas"] is not None
            else rows[i]["pen"],
        ),
    )[: args.top_k]

    best: tuple[Decimal, dict, str] | None = None
    reports: list[dict] = []
    for idx in ranked:
        polished = polish(n, incumbent, rows[idx])
        report = {
            "label": labels[idx],
            "cpp_L": rows[idx]["best_L_feas"],
            "cpp_pen": rows[idx]["pen"],
            "polished": None if polished is None else polished[1],
        }
        reports.append(report)
        print(
            f"{labels[idx]} cpp_L={rows[idx]['best_L_feas']} "
            f"polished={None if polished is None else polished[0]}",
            flush=True,
        )
        if polished is not None and (best is None or polished[0] < best[0]):
            best = (polished[0], polished[1], labels[idx])

    payload = {
        "n": n,
        "incumbent_L": str(incumbent),
        "source_L": source_side,
        "interface": {
            "tilted": tilted.tolist(),
            "upright": upright.tolist(),
            "both": both.tolist(),
        },
        "best_label": None if best is None else best[2],
        "best": None if best is None else best[1],
        "reports": reports,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if best is None:
        print("MISS: no certified masked-interface candidate", flush=True)
    else:
        print(
            f"BEST {best[2]} L={best[0]} dL={float(best[0] - incumbent):+.9e}",
            flush=True,
        )


if __name__ == "__main__":
    main()
