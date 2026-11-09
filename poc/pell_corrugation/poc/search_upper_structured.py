"""Upper-register structured border search across n in [700, 1000].

Constructs exact structured multi-sided grid borders with thickness t in 1..4
and valid splits between East, North, West, and South sides on cached Pell cores
(w=12, 13, 15, 16). Leftover squares for intermediate counts are placed in corner
pockets with SAT clearance >= 5e-9. Candidates with unpolished extent <= incumbent L + 0.002
are polished with 4-sign-piece LP (polish_band) and verified with official certify.
Strictly winning packings are written to poc/pell_corrugation/data/candidates_upper.json.
"""
from __future__ import annotations

import argparse
from decimal import Decimal
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

# Ensure baseline and pell_corrugation modules are importable
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
BASELINE_SRC = REPO_ROOT / "poc" / "baseline" / "poc" / "src"
PELL_POC = HERE

if str(BASELINE_SRC) not in sys.path:
    sys.path.insert(0, str(BASELINE_SRC))
if str(PELL_POC) not in sys.path:
    sys.path.insert(0, str(PELL_POC))

import pell_corrugation as p
from squares.core import certify, read_submission

DATA_DIR = HERE.parent / "data"
BORDER_SCAN_DIR = DATA_DIR / "border-scan"
MERGED_CSV = BORDER_SCAN_DIR / "merged.csv"
CANDIDATES_UPPER_JSON = DATA_DIR / "candidates_upper.json"

DEFAULT_CORES = [12, 13, 15, 16]
PITCH_DELTA = 1e-8
MARGIN = 1e-7


def generate_structured_splits(t_max: int = 4) -> list[tuple[int, int, int, int, int]]:
    """Generate all valid splits (t, tE, tN, tW, tS) with thickness t in 1..t_max.

    t is the bounding thickness max(tE + tW, tN + tS).
    """
    splits = []
    for t in range(1, t_max + 1):
        for tx in range(0, t + 1):
            for ty in range(0, t + 1):
                if max(tx, ty) != t:
                    continue
                for tE in range(tx + 1):
                    tW = tx - tE
                    for tN in range(ty + 1):
                        tS = ty - tN
                        splits.append((t, tE, tN, tW, tS))
    return splits


def build_candidate_packing(
    q_core: np.ndarray,
    tE: int,
    tN: int,
    tW: int,
    tS: int,
    target_count: int,
    margin: float = MARGIN,
    pitch_delta: float = PITCH_DELTA,
) -> tuple[np.ndarray, float, dict] | None:
    """Wrap normalized core packing with multi-sided grid border and corner pockets to hit target_count."""
    n0 = len(q_core)
    if target_count < n0:
        return None

    m = target_count - n0
    q_c = p.normalize_box(q_core, margin=0.0)
    L_core = p.extent(q_c)
    k = int(math.floor(L_core))

    tx = tE + tW
    ty = tN + tS
    T = max(tx, ty)
    if T == 0 and m > 0:
        return None

    pitch = 1.0 + pitch_delta
    L_total = L_core + T + 2.0 * margin

    x_off = tW * pitch + margin
    y_off = tS * pitch + margin

    q_shifted = q_c.copy()
    q_shifted[:, 0] += x_off
    q_shifted[:, 1] += y_off

    # 1. Generate side strip positions (along the core edges)
    # Side order: East, North, West, South
    strip_squares: list[list[float]] = []

    # East strip: tE columns of k squares
    for i in range(tE):
        x = L_total - 0.5 - i * pitch
        for j in range(k):
            y = y_off + 0.5 + j * pitch
            strip_squares.append([x, y, 0.0])

    # North strip: tN rows of k squares
    for j in range(tN):
        y = L_total - 0.5 - j * pitch
        for i in range(k):
            x = x_off + 0.5 + i * pitch
            strip_squares.append([x, y, 0.0])

    # West strip: tW columns of k squares
    for i in range(tW):
        x = 0.5 + i * pitch
        for j in range(k):
            y = y_off + 0.5 + j * pitch
            strip_squares.append([x, y, 0.0])

    # South strip: tS rows of k squares
    for j in range(tS):
        y = 0.5 + j * pitch
        for i in range(k):
            x = x_off + 0.5 + i * pitch
            strip_squares.append([x, y, 0.0])

    # 2. Generate corner pocket positions
    corner_squares: list[list[float]] = []

    # NE corner pocket: tE x tN
    for i in range(tE):
        for j in range(tN):
            corner_squares.append([L_total - 0.5 - i * pitch, L_total - 0.5 - j * pitch, 0.0])

    # NW corner pocket: tW x tN
    for i in range(tW):
        for j in range(tN):
            corner_squares.append([0.5 + i * pitch, L_total - 0.5 - j * pitch, 0.0])

    # SE corner pocket: tE x tS
    for i in range(tE):
        for j in range(tS):
            corner_squares.append([L_total - 0.5 - i * pitch, 0.5 + j * pitch, 0.0])

    # SW corner pocket: tW x tS
    for i in range(tW):
        for j in range(tS):
            corner_squares.append([0.5 + i * pitch, 0.5 + j * pitch, 0.0])

    total_capacity = len(strip_squares) + len(corner_squares)
    if m > total_capacity:
        return None

    border: list[list[float]] = []
    num_strips_total = len(strip_squares)

    if m <= num_strips_total:
        # Check if leftover squares can be placed in corner pockets first
        # to preserve full contiguous side strips
        border = strip_squares[:m]
        residual_in_corners = 0
    else:
        # All full side strips are used; leftover residual squares go into corner pockets
        border = list(strip_squares)
        residual = m - num_strips_total
        border.extend(corner_squares[:residual])
        residual_in_corners = residual

    packing = np.vstack([q_shifted, np.array(border)]) if border else q_shifted
    meta = {
        "t": T,
        "tE": tE,
        "tN": tN,
        "tW": tW,
        "tS": tS,
        "strip_squares": min(m, num_strips_total),
        "corner_residual_squares": residual_in_corners,
        "total_border_capacity": total_capacity,
        "L_unpolished": float(p.extent(packing)),
    }
    return packing, L_total, meta


def run_search(
    target_min: int = 700,
    target_max: int = 1000,
    core_ws: list[int] | None = None,
    t_max: int = 4,
    polish_slack: float = 0.002,
    output_path: Path = CANDIDATES_UPPER_JSON,
    verbose: bool = True,
) -> dict[str, dict]:
    """Execute upper register structured border search and save strictly winning candidates."""
    if core_ws is None:
        core_ws = DEFAULT_CORES

    if verbose:
        print(f"Reading incumbent submissions from {MERGED_CSV}...", flush=True)
    baseline_sub = read_submission(MERGED_CSV)

    # Cache incumbent float extents and certified 50-digit strings
    incumbent_L_float: dict[int, float] = {}
    incumbent_L_str: dict[int, str] = {}

    # Fast load from merged-literal-official.json if available
    official_json = BORDER_SCAN_DIR / "merged-literal-official.json"
    if official_json.exists():
        try:
            with open(official_json, "r", encoding="utf-8") as f:
                official_data = json.load(f)
            for r in official_data.get("results", []):
                nv = int(r["n"])
                incumbent_L_float[nv] = float(r["L"])
                incumbent_L_str[nv] = str(r["L"])
        except Exception:
            pass

    # Ensure all target n are covered
    for n in range(target_min, target_max + 1):
        if n not in incumbent_L_float:
            rows_n = baseline_sub[n]
            # Fast float extent
            q_tmp = np.array([[float(r["x"]), float(r["y"]), float(r["angle"])] for r in rows_n])
            incumbent_L_float[n] = p.extent(q_tmp)
            incumbent_L_str[n] = certify(n, rows_n)

    if verbose:
        print(f"Loaded incumbents for n in [{target_min}, {target_max}].", flush=True)

    # Load cached cores
    cores: dict[int, tuple[np.ndarray, dict]] = {}
    for w in core_ws:
        core_file = BORDER_SCAN_DIR / f"core-w{w}.npz"
        if not core_file.exists():
            if verbose:
                print(f"Warning: core file {core_file} does not exist, skipping w={w}.")
            continue
        z = np.load(core_file)
        q = z["q"]
        meta = json.loads(str(z["meta"])) if "meta" in z else {}
        cores[w] = (q, meta)
        if verbose:
            print(f"Loaded core w={w}: n0={len(q)}, L={p.extent(q):.8f}")

    splits = generate_structured_splits(t_max=t_max)
    if verbose:
        print(f"Generated {len(splits)} structured border splits (t in 1..{t_max}).", flush=True)

    winning_candidates: dict[str, dict] = {}

    for target_n in range(target_min, target_max + 1):
        inc_L_float = incumbent_L_float[target_n]
        inc_L_str = incumbent_L_str.get(target_n)

        best_cand_for_n = None
        best_cand_meta = None
        best_cand_extent = float("inf")

        for w, (q_core, core_meta) in cores.items():
            n0 = len(q_core)
            if target_n < n0:
                continue

            # Core itself if target_n == n0
            if target_n == n0:
                cand_L = p.extent(q_core)
                if cand_L < best_cand_extent:
                    best_cand_for_n = q_core.copy()
                    best_cand_extent = cand_L
                    best_cand_meta = dict(core_meta, w=w, n0=n0, kind="core-unwrapped")
                continue

            for (t, tE, tN, tW, tS) in splits:
                res = build_candidate_packing(q_core, tE, tN, tW, tS, target_n)
                if res is None:
                    continue
                cand_q, L_tot, split_meta = res
                cand_L = split_meta["L_unpolished"]

                # Screen candidate
                if cand_L <= inc_L_float + polish_slack:
                    if cand_L < best_cand_extent:
                        best_cand_extent = cand_L
                        best_cand_for_n = cand_q
                        best_cand_meta = dict(split_meta, w=w, n0=n0, kind="structured-multisided-border")

        if best_cand_for_n is None:
            continue

        # Evaluate candidate for target_n
        options = [best_cand_for_n]
        if best_cand_extent <= inc_L_float + polish_slack:
            # Run polish_band
            if verbose:
                print(f"n={target_n}: candidate unpolished L={best_cand_extent:.8f} <= incumbent {inc_L_float:.8f} + {polish_slack}, polishing...", flush=True)
            try:
                pol = p.polish_band(best_cand_for_n, best_cand_extent, iterations=25, position_trust=0.05, angle_trust=0.005)
                if pol is not None and pol.L < best_cand_extent:
                    ok_sat, gap = p.sat_check(pol.q)
                    if ok_sat and gap >= 4.9e-9:
                        options.insert(0, pol.q)
                        if verbose:
                            print(f"n={target_n}: polish improved L to {pol.L:.8f} (gap={gap:.2e})", flush=True)
            except Exception as e:
                if verbose:
                    print(f"n={target_n}: polish exception: {e}", flush=True)

        for cand in options:
            cur_L = p.extent(cand)
            if cur_L >= inc_L_float - 1e-9:
                continue

            # Verify SAT clearance >= 5e-9
            ok_sat, gap = p.sat_check(cand)
            if not ok_sat or gap < 4.99e-9:
                continue

            # Format rows and certify
            row_list = p.rows(cand)
            try:
                cert_L = certify(target_n, row_list)
            except Exception as e:
                if verbose:
                    print(f"n={target_n}: certification failed: {e}", flush=True)
                continue

            if inc_L_str is None:
                inc_L_str = certify(target_n, baseline_sub[target_n])
                incumbent_L_str[target_n] = inc_L_str

            dec_cert = Decimal(cert_L)
            dec_inc = Decimal(inc_L_str)
            delta_L = dec_cert - dec_inc

            if delta_L < Decimal("-1e-12"):
                # Strict win!
                win_record = {
                    "n": target_n,
                    "certified_L": cert_L,
                    "incumbent_L": inc_L_str,
                    "delta_L": str(delta_L),
                    "delta_L_float": float(delta_L),
                    "sat_min_gap": float(gap),
                    "metadata": best_cand_meta,
                    "packing": row_list,
                }
                winning_candidates[str(target_n)] = win_record
                if verbose:
                    print(f"*** STRICT WIN: n={target_n}, certified L={cert_L}, incumbent L={inc_L_str}, delta={delta_L} ***", flush=True)
                break

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(winning_candidates, f, indent=2)

    if verbose:
        print(f"Saved {len(winning_candidates)} winning candidates to {output_path}.", flush=True)

    return winning_candidates


def main() -> None:
    parser = argparse.ArgumentParser(description="Upper register structured border search [700, 1000]")
    parser.add_argument("--target-min", type=int, default=700, help="Minimum target n (default: 700)")
    parser.add_argument("--target-max", type=int, default=1000, help="Maximum target n (default: 1000)")
    parser.add_argument("--max-t", type=int, default=4, help="Maximum border thickness (default: 4)")
    parser.add_argument("--polish-slack", type=float, default=0.002, help="Screening slack for polish (default: 0.002)")
    parser.add_argument("--output", type=Path, default=CANDIDATES_UPPER_JSON, help="Output JSON path")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose logging")
    args = parser.parse_args()

    run_search(
        target_min=args.target_min,
        target_max=args.target_max,
        t_max=args.max_t,
        polish_slack=args.polish_slack,
        output_path=args.output,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
