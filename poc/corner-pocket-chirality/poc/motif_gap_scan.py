#!/usr/bin/env python3
"""Insert a compact tilted-pair motif into the open region of a packing."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "poc" / "highn-mondrian" / "poc"))
import chirality_scan as cs  # noqa: E402
import scan  # noqa: E402


def read_squares(path: Path) -> np.ndarray:
    with path.open(newline="", encoding="utf-8") as handle:
        return np.asarray(
            [
                (float(row["x"]), float(row["y"]), float(row["angle"]))
                for row in csv.DictReader(handle)
            ],
            dtype=np.float64,
        )


def pair_motifs(
    packings: dict[int, list[dict[str, str]]],
    sources: tuple[int, ...],
    max_box_area: float,
) -> list[np.ndarray]:
    pockets = cs.extract_pockets(packings, sources, 80, 0.1, 1.5)
    motifs: dict[tuple[tuple[float, float, float], ...], np.ndarray] = {}
    for pocket in pockets:
        squares = pocket.squares
        for first in range(len(squares)):
            for second in range(first + 1, len(squares)):
                pair = squares[[first, second]]
                xlo, xhi, ylo, yhi = cs.polygon_bounds(pair)
                width = float(xhi.max() - xlo.min())
                height = float(yhi.max() - ylo.min())
                if width * height > max_box_area:
                    continue
                normalized, _ = cs.normalize_bundle(
                    pair, np.empty((0, 3), dtype=np.float64)
                )
                for symmetry in cs.SYMMETRIES:
                    oriented, _ = cs.transform_bundle(
                        normalized,
                        np.empty((0, 3), dtype=np.float64),
                        symmetry,
                    )
                    key = tuple(
                        map(
                            tuple,
                            np.round(
                                oriented[
                                    np.lexsort(
                                        (
                                            oriented[:, 2],
                                            oriented[:, 1],
                                            oriented[:, 0],
                                        )
                                    )
                                ],
                                7,
                            ),
                        )
                    )
                    motifs.setdefault(key, oriented)
    return list(motifs.values())


def forbidden_translation(
    fixed: np.ndarray,
    motif: np.ndarray,
    tx: np.ndarray,
    ty: np.ndarray,
    step: float,
) -> np.ndarray:
    mot_xlo, mot_xhi, mot_ylo, mot_yhi = cs.polygon_bounds(motif)
    half_extent = 0.5 * (
        np.abs(np.cos(motif[:, 2])) + np.abs(np.sin(motif[:, 2]))
    )
    diff = np.zeros((len(ty) + 1, len(tx) + 1), dtype=np.int32)
    fixed_xlo, fixed_xhi, fixed_ylo, fixed_yhi = cs.polygon_bounds(fixed)
    x0, y0 = float(tx[0]), float(ty[0])
    for index in range(len(motif)):
        low_x = fixed_xlo - motif[index, 0] - half_extent[index]
        high_x = fixed_xhi - motif[index, 0] + half_extent[index]
        low_y = fixed_ylo - motif[index, 1] - half_extent[index]
        high_y = fixed_yhi - motif[index, 1] + half_extent[index]
        ix0 = np.ceil((low_x - x0) / step - 1e-12).astype(np.int64)
        ix1 = np.floor((high_x - x0) / step + 1e-12).astype(np.int64)
        iy0 = np.ceil((low_y - y0) / step - 1e-12).astype(np.int64)
        iy1 = np.floor((high_y - y0) / step + 1e-12).astype(np.int64)
        np.clip(ix0, 0, len(tx) - 1, out=ix0)
        np.clip(ix1, -1, len(tx) - 1, out=ix1)
        np.clip(iy0, 0, len(ty) - 1, out=iy0)
        np.clip(iy1, -1, len(ty) - 1, out=iy1)
        valid = (ix1 >= ix0) & (iy1 >= iy0)
        for xa, xb, ya, yb in zip(
            ix0[valid],
            ix1[valid],
            iy0[valid],
            iy1[valid],
        ):
            diff[ya, xa] += 1
            diff[ya, xb + 1] -= 1
            diff[yb + 1, xa] -= 1
            diff[yb + 1, xb + 1] += 1
    occupied = diff.cumsum(axis=0).cumsum(axis=1)[: len(ty), : len(tx)] > 0
    return occupied


def scan_motif(
    fixed: np.ndarray,
    motif: np.ndarray,
    side: float,
    step: float,
    fixed_tree: cKDTree,
) -> dict[str, object] | None:
    xlo, xhi, ylo, yhi = cs.polygon_bounds(motif)
    width = float(xhi.max() - xlo.min())
    height = float(yhi.max() - ylo.min())
    tx = np.arange(0.0, side - width + 1e-12, step)
    ty = np.arange(0.0, side - height + 1e-12, step)
    occupied = forbidden_translation(fixed, motif, tx, ty, step)
    free_y, free_x = np.nonzero(~occupied)
    if not len(free_x):
        return None
    px = tx[free_x] + 0.5 * width
    py = ty[free_y] + 0.5 * height
    if len(px) > 200_000:
        sample = np.linspace(0, len(px) - 1, 200_000, dtype=np.int64)
        px, py = px[sample], py[sample]
    distance, _ = fixed_tree.query(np.column_stack([px, py]))
    for choice in np.argsort(distance)[-64:][::-1]:
        placed = motif.copy()
        placed[:, 0] += tx[free_x[choice]]
        placed[:, 1] += ty[free_y[choice]]
        if cs.separated(placed, fixed):
            return {
                "placement": placed,
                "clearance": float(distance[choice]),
                "translation": [
                    float(tx[free_x[choice]]),
                    float(ty[free_y[choice]]),
                ],
            }
    return None


def write_candidate(path: Path, target: int, squares: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["n", "square_id", "x", "y", "angle"])
        for square_id, (x, y, angle) in enumerate(squares[:target]):
            writer.writerow(
                [
                    target,
                    square_id,
                    repr(float(x)),
                    repr(float(y)),
                    repr(float(angle)),
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--seed-csv", required=True, type=Path)
    parser.add_argument("--side", required=True, type=float)
    parser.add_argument("--target", required=True, type=int)
    parser.add_argument("--sources", default="908,969,970,971,972")
    parser.add_argument("--max-box-area", type=float, default=3.0)
    parser.add_argument("--grid-step", type=float, default=0.05)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--candidate-out", type=Path)
    args = parser.parse_args()

    started = time.time()
    packings = scan.read_groups(args.baseline)
    sources = tuple(int(value) for value in args.sources.split(","))
    motifs = pair_motifs(packings, sources, args.max_box_area)
    fixed = read_squares(args.seed_csv)
    fixed_tree = cKDTree(fixed[:, :2])
    best = None
    attempts = []
    for index, motif in enumerate(motifs):
        result = scan_motif(
            fixed,
            motif,
            args.side,
            args.grid_step,
            fixed_tree,
        )
        attempts.append(
            {
                "motif": index,
                "found": result is not None,
                "clearance": None if result is None else result["clearance"],
            }
        )
        if result is None:
            continue
        clearance = float(result["clearance"])
        if best is None or clearance > best[0]:
            best = (clearance, index, result)
        print(
            f"motif={index} clearance={clearance:.6f}",
            flush=True,
        )
    report: dict[str, object] = {
        "baseline": str(args.baseline.resolve()),
        "seed_csv": str(args.seed_csv.resolve()),
        "side": args.side,
        "target": args.target,
        "sources": sources,
        "max_box_area": args.max_box_area,
        "grid_step": args.grid_step,
        "motif_count": len(motifs),
        "elapsed_seconds": time.time() - started,
        "best": None,
        "attempts": attempts,
    }
    if best is not None:
        clearance, index, result = best
        placement = np.asarray(result["placement"], dtype=np.float64)
        candidate = np.vstack([fixed, placement])
        if len(candidate) < args.target:
            raise SystemExit(
                f"motif reached only {len(candidate)}/{args.target}"
            )
        if not cs.separated(placement[:1], placement[1:]) or not cs.separated(
            placement, fixed
        ):
            raise SystemExit("candidate contains an overlap")
        report["best"] = {
            "motif": index,
            "clearance": clearance,
            "translation": result["translation"],
            "squares": placement.tolist(),
            "count": len(candidate),
        }
        if args.candidate_out:
            write_candidate(args.candidate_out, args.target, candidate)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    best_text = "none" if best is None else f"{best[0]:.6f}"
    print(
        f"motifs={len(motifs)} found={sum(item['found'] for item in attempts)} "
        f"best={best_text} elapsed={time.time() - started:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
