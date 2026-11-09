"""Mid-register structured border search across n in [200, 700].

Systematically searches structured perimeter wraps (symmetric and asymmetric
East/North/West/South allocations) around cached Pell-17 cores (w=5, 6, 8, 9, 10),
with corner-pocket placement for residual notch-filling squares and partial
border prefix orderings. Candidates with bounding box extent <= incumbent L + 0.002
are polished with the 4-sign-piece epigraph LP (polish_band) and evaluated
with squares.core.certify. All candidates with strictly negative delta L are
exported to poc/pell_corrugation/data/candidates_mid.json.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import sys
import time
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
SYS_SRC = REPO_ROOT / "poc/baseline/poc/src"
if str(SYS_SRC) not in sys.path:
    sys.path.insert(0, str(SYS_SRC))
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import pell_corrugation as p
from squares.core import certify, read_submission

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
BORDER_SCAN_DIR = DATA_DIR / "border-scan"
MERGED_CSV = BORDER_SCAN_DIR / "merged.csv"
MERGED_OFFICIAL_JSON = BORDER_SCAN_DIR / "merged-literal-official.json"
OUTPUT_CANDIDATES_JSON = DATA_DIR / "candidates_mid.json"

CORE_FILES = {
    5: BORDER_SCAN_DIR / "core-w5.npz",
    6: BORDER_SCAN_DIR / "core-w6.npz",
    8: BORDER_SCAN_DIR / "core-w8.npz",
    9: BORDER_SCAN_DIR / "core-w9.npz",
    10: BORDER_SCAN_DIR / "core-w10.npz",
}


def load_incumbents(
    csv_path: Path = MERGED_CSV,
    json_path: Path = MERGED_OFFICIAL_JSON,
) -> tuple[dict[int, float], dict[int, str]]:
    """Load incumbent packings and certified side lengths for n in [200, 700]."""
    incumbents_float: dict[int, float] = {}
    incumbents_str: dict[int, str] = {}

    if json_path.exists():
        data = json.loads(json_path.read_text(encoding="utf-8"))
        for entry in data.get("results", []):
            n_val = int(entry["n"])
            if 200 <= n_val <= 700:
                l_str = str(entry["L"])
                incumbents_str[n_val] = l_str
                incumbents_float[n_val] = float(l_str)

    # Fill any missing values directly from merged.csv
    missing = [n for n in range(200, 701) if n not in incumbents_float]
    if missing and csv_path.exists():
        puzzles = read_submission(csv_path)
        for n_val in missing:
            if n_val in puzzles:
                l_cert = certify(n_val, puzzles[n_val])
                incumbents_str[n_val] = l_cert
                incumbents_float[n_val] = float(l_cert)

    return incumbents_float, incumbents_str


def load_cached_cores() -> dict[int, tuple[np.ndarray, float, dict[str, Any]]]:
    """Load cached Pell-17 cores w=5, 6, 8, 9, 10 from border-scan directory."""
    cores: dict[int, tuple[np.ndarray, float, dict[str, Any]]] = {}
    for w, path in CORE_FILES.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing core file: {path}")
        z = np.load(path)
        q = z["q"]
        meta: dict[str, Any] = {}
        if "meta" in z:
            try:
                meta = json.loads(str(z["meta"]))
            except Exception:
                meta = {}
        meta["w"] = w
        meta["n0"] = len(q)
        q_norm = p.normalize_box(q, margin=0.0)
        l_core = p.extent(q_norm)
        cores[w] = (q_norm, l_core, meta)
    return cores


def build_perimeter_wrap(
    q_core: np.ndarray,
    t_E: int,
    t_N: int,
    t_W: int,
    t_S: int,
    margin: float = 1e-7,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Wrap core with structured upright borders along East, North, West, South.

    Returns:
        (q_combined, border_only, L_box)
    """
    q_core = p.normalize_box(q_core, margin=0.0)
    L_core = p.extent(q_core)

    L_x = t_W + L_core + t_E + 2.0 * margin
    L_y = t_S + L_core + t_N + 2.0 * margin
    L_box = max(L_x, L_y)
    K = int(math.floor(L_box))

    # Shift core so it clears West and South border channels
    q_placed_core = q_core.copy()
    q_placed_core[:, 0] += t_W + margin
    q_placed_core[:, 1] += t_S + margin

    border_squares: list[list[float]] = []

    # 1. West boundary channels: full columns x = 0.5 + i
    for i in range(t_W):
        x = 0.5 + i * (1.0 + 1e-8)
        for j in range(K):
            border_squares.append([x, 0.5 + j * (1.0 + 1e-8), 0.0])

    # 2. East boundary channels: full columns x = L_box - 0.5 - i
    for i in range(t_E):
        x = L_box - 0.5 - i * (1.0 + 1e-8)
        for j in range(K):
            border_squares.append([x, 0.5 + j * (1.0 + 1e-8), 0.0])

    # 3. South boundary channels: rows y = 0.5 + j, x spanning inner span
    for j in range(t_S):
        y = 0.5 + j * (1.0 + 1e-8)
        for i in range(K - t_W - t_E):
            x = 0.5 + (t_W + i) * (1.0 + 1e-8)
            border_squares.append([x, y, 0.0])

    # 4. North boundary channels: rows y = L_box - 0.5 - j, x spanning inner span
    for j in range(t_N):
        y = L_box - 0.5 - j * (1.0 + 1e-8)
        for i in range(K - t_W - t_E):
            x = 0.5 + (t_W + i) * (1.0 + 1e-8)
            border_squares.append([x, y, 0.0])

    border_arr = np.array(border_squares, dtype=float) if border_squares else np.zeros((0, 3), dtype=float)
    if len(border_arr) > 0:
        q_combined = np.vstack([q_placed_core, border_arr])
    else:
        q_combined = q_placed_core

    return q_combined, border_arr, L_box


def find_corner_pocket_sites(
    q_base: np.ndarray,
    L_box: float,
    max_sites: int = 40,
) -> list[list[float]]:
    """Identify feasible corner-pocket and notch-filling integer grid sites.

    Searches empty half-integer grid positions near the 4 corners of the container
    and tests Separating Axis Theorem clearance against existing squares.
    """
    K = int(math.floor(L_box))
    occupied_centers = q_base[:, :2]

    # Generate candidate grid points near corners (within distance 4 from any edge)
    candidate_pts: list[tuple[float, float, float]] = []
    seen = set()

    corner_ranges = [
        (range(0, min(5, K)), range(0, min(5, K))),  # SW
        (range(max(0, K - 5), K), range(0, min(5, K))),  # SE
        (range(0, min(5, K)), range(max(0, K - 5), K)),  # NW
        (range(max(0, K - 5), K), range(max(0, K - 5), K)),  # NE
    ]

    for ix_range, iy_range in corner_ranges:
        for ix in ix_range:
            for iy in iy_range:
                x = 0.5 + ix
                y = 0.5 + iy
                if not (0.5 <= x <= L_box - 0.5 and 0.5 <= y <= L_box - 0.5):
                    continue
                key = (round(x, 4), round(y, 4))
                if key in seen:
                    continue
                seen.add(key)

                # Fast bounding box check against existing centers
                dists = np.max(np.abs(occupied_centers - np.array([x, y])), axis=1)
                if np.min(dists) < 0.999:
                    continue

                # Corner distance priority (prefer snug corner slots)
                corner_dist = min(
                    x**2 + y**2,
                    (L_box - x)**2 + y**2,
                    x**2 + (L_box - y)**2,
                    (L_box - x)**2 + (L_box - y)**2,
                )
                candidate_pts.append((corner_dist, x, y))

    candidate_pts.sort(key=lambda t: t[0])

    valid_pocket_squares: list[list[float]] = []
    current_q = q_base.copy()

    for _, x, y in candidate_pts:
        test_sq = np.array([[x, y, 0.0]])
        test_combined = np.vstack([current_q, test_sq])
        ok, gap = p.sat_check(test_combined, clearance=1e-8)
        if ok and gap >= 0.0:
            valid_pocket_squares.append([x, y, 0.0])
            current_q = test_combined
            if len(valid_pocket_squares) >= max_sites:
                break

    return valid_pocket_squares


def generate_allocations() -> list[tuple[int, int, int, int]]:
    """Enumerate symmetric and asymmetric (t_E, t_N, t_W, t_S) allocations."""
    allocs: list[tuple[int, int, int, int]] = []
    seen = set()

    # 1. Symmetric L-borders: (t, t, 0, 0), (0, t, t, 0), (t, 0, 0, t), (0, 0, t, t)
    for t in range(1, 6):
        for te, tn, tw, ts in (
            (t, t, 0, 0),
            (0, t, t, 0),
            (t, 0, 0, t),
            (0, 0, t, t),
        ):
            if (te, tn, tw, ts) not in seen:
                seen.add((te, tn, tw, ts))
                allocs.append((te, tn, tw, ts))

    # 2. Symmetric uniform rings: (t, t, t, t)
    for t in (1, 2, 3):
        alloc = (t, t, t, t)
        if alloc not in seen:
            seen.add(alloc)
            allocs.append(alloc)

    # 3. Asymmetric allocations: t_E, t_N, t_W, t_S in 0..4 with |dx - dy| <= 1
    for te in range(5):
        for tn in range(5):
            for tw in range(5):
                for ts in range(5):
                    if te + tn + tw + ts == 0:
                        continue
                    dx = te + tw
                    dy = tn + ts
                    if abs(dx - dy) <= 1 and max(dx, dy) <= 6:
                        alloc = (te, tn, tw, ts)
                        if alloc not in seen:
                            seen.add(alloc)
                            allocs.append(alloc)

    return allocs


def run_search(
    target_min: int = 200,
    target_max: int = 700,
    verbose: bool = True,
) -> dict[str, Any]:
    """Execute mid-register structured border search across n in [target_min, target_max]."""
    t_start = time.time()
    incumbents_float, incumbents_str = load_incumbents()
    cores = load_cached_cores()
    allocations = generate_allocations()

    candidates_by_n: dict[int, list[tuple[float, np.ndarray, dict[str, Any]]]] = {}
    screened_count = 0

    def offer_candidate(q: np.ndarray, meta: dict[str, Any]) -> None:
        nonlocal screened_count
        n_val = len(q)
        if not (target_min <= n_val <= target_max):
            return
        if n_val not in incumbents_float:
            return
        screened_count += 1
        l_ext = p.extent(q)
        # Filter: bounding box extent <= incumbent L + 0.002
        if l_ext <= incumbents_float[n_val] + 0.002:
            if n_val not in candidates_by_n:
                candidates_by_n[n_val] = []
            candidates_by_n[n_val].append((l_ext, q.copy(), meta))

    if verbose:
        print(f"Loaded {len(cores)} cores and {len(incumbents_float)} incumbents.")
        print(f"Generating structured wraps across {len(allocations)} allocations...")

    # Systematically explore structured perimeter wraps around each core
    for w, (q_core, l_core, c_meta) in cores.items():
        n0 = len(q_core)

        # Baseline unadorned core if within target range
        offer_candidate(q_core, dict(c_meta, kind="core", w=w, alloc=(0, 0, 0, 0)))

        for te, tn, tw, ts in allocations:
            q_wrap, border, l_box = build_perimeter_wrap(q_core, te, tn, tw, ts)
            n_wrap = len(q_wrap)
            alloc_meta = dict(
                c_meta,
                w=w,
                alloc=(te, tn, tw, ts),
                n0=n0,
                n_wrap=n_wrap,
                L_box=l_box,
            )

            # Full structured wrap
            offer_candidate(q_wrap, dict(alloc_meta, kind="structured_full_wrap"))

            # Residual notch-filling via corner-pocket squares (n > n_wrap)
            if n_wrap < target_max:
                pockets = find_corner_pocket_sites(q_wrap, l_box, max_sites=min(20, target_max - n_wrap))
                if pockets:
                    pockets_arr = np.array(pockets, dtype=float)
                    for r in range(1, len(pockets_arr) + 1):
                        q_with_pocket = np.vstack([q_wrap, pockets_arr[:r]])
                        offer_candidate(
                            q_with_pocket,
                            dict(alloc_meta, kind="structured_corner_pocket", added_pockets=r),
                        )

            # Partial border prefix cuts (n0 < n < n_wrap)
            if len(border) > 0:
                for axis in (0, 1):
                    for rev in (False, True):
                        order = np.argsort(border[:, axis], kind="stable")
                        if rev:
                            order = order[::-1]
                        sorted_border = border[order]
                        for count in range(1, len(sorted_border) + 1):
                            n_cand = n0 + count
                            if not (target_min <= n_cand <= target_max):
                                continue
                            if n_cand in incumbents_float and l_box <= incumbents_float[n_cand] + 0.002:
                                q_partial = np.vstack([q_wrap[:n0], sorted_border[:count]])
                                offer_candidate(
                                    q_partial,
                                    dict(alloc_meta, kind="structured_partial_border", axis=axis, reverse=rev),
                                )

    if verbose:
        shortlisted_targets = sorted(candidates_by_n.keys())
        print(f"Screened {screened_count} candidates; shortlisted {len(shortlisted_targets)} targets.")

    # Apply 4-sign piece LP polish and certify candidates
    winning_candidates: dict[str, Any] = {}
    polished_count = 0

    for n_val in sorted(candidates_by_n.keys()):
        cand_list = candidates_by_n[n_val]
        cand_list.sort(key=lambda t: t[0])
        inc_L_flt = incumbents_float[n_val]
        inc_L_str = incumbents_str[n_val]
        inc_L_dec = Decimal(inc_L_str)

        # Polish the top candidate configurations for this n
        for l_init, q_raw, meta in cand_list[:3]:
            polished_count += 1
            res = p.polish_band(
                q_raw,
                L_init=l_init,
                iterations=20,
                position_trust=0.05,
                angle_trust=0.005,
                clearance=5e-9,
            )

            test_coords = [q_raw]
            if res is not None and res.feasible and res.L < l_init:
                test_coords.insert(0, res.q)

            for cand_q in test_coords:
                if p.extent(cand_q) >= inc_L_flt - 1e-9:
                    continue

                formatted_rows = p.rows(cand_q, dilate=1.0)
                try:
                    cert_L_str = certify(n_val, formatted_rows)
                    cert_L_dec = Decimal(cert_L_str)
                    delta_dec = cert_L_dec - inc_L_dec

                    if delta_dec < Decimal("0"):
                        # Strictly negative delta L
                        n_str = str(n_val)
                        if n_str not in winning_candidates or Decimal(winning_candidates[n_str]["candidate_L"]) > cert_L_dec:
                            winning_candidates[n_str] = {
                                "n": n_val,
                                "candidate_L": cert_L_str,
                                "incumbent_L": inc_L_str,
                                "delta_L": str(delta_dec),
                                "beats_incumbent": True,
                                "construction": meta,
                                "coordinates": formatted_rows,
                            }
                            if verbose:
                                print(f"WIN: n={n_val} L={cert_L_str} delta={delta_dec}")
                            break
                except Exception:
                    continue

            if str(n_val) in winning_candidates:
                break

    # Persist all winning candidates with strictly negative delta L
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_CANDIDATES_JSON.write_text(json.dumps(winning_candidates, indent=2), encoding="utf-8")

    elapsed = time.time() - t_start
    if verbose:
        print(f"Completed in {elapsed:.2f}s. Polished {polished_count} candidates. Wins: {len(winning_candidates)}")
        print(f"Exported to {OUTPUT_CANDIDATES_JSON}")

    return {
        "elapsed_seconds": elapsed,
        "screened": screened_count,
        "polished": polished_count,
        "wins": len(winning_candidates),
        "winning_targets": sorted([int(k) for k in winning_candidates.keys()]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Mid-register structured border search for n in [200, 700]")
    parser.add_argument("--min-n", type=int, default=200, help="Minimum n target (default 200)")
    parser.add_argument("--max-n", type=int, default=700, help="Maximum n target (default 700)")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
    args = parser.parse_args()

    run_search(target_min=args.min_n, target_max=args.max_n, verbose=not args.quiet)


if __name__ == "__main__":
    main()
