"""Lower-register structured border search for n in [50, 200].

Constructs structured 1-layer, 2-layer, and 3-layer grid borders around
cached Pell-17 cores (w=1, 2, 3) with corner notches filled, targeting
packings that improve upon pure square-grid incumbents or truncated wrappers.
Verifies winning arrangements with squares.core.certify and certify_transport,
saving all strict improvements to poc/pell_corrugation/data/candidates_lower.json.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

POC_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
BORDER_SCAN_DIR = DATA_DIR / "border-scan"
DEFAULT_OUTPUT = DATA_DIR / "candidates_lower.json"

sys.path.insert(0, str(POC_DIR))
import pell_corrugation as p

sys.path.insert(0, str(REPO_ROOT / "poc/baseline/poc/src"))
from squares.core import certify, grid, read_submission


def load_cached_cores(
    border_scan_dir: Path = BORDER_SCAN_DIR,
    widths: Tuple[int, ...] = (1, 2, 3),
) -> Dict[int, Dict[str, Any]]:
    """Load cached Pell-17 cores from core-w*.npz files.

    For w=1, 2, 3, loads core coordinates, extent, and metadata.
    """
    cores: Dict[int, Dict[str, Any]] = {}
    for w in widths:
        path = border_scan_dir / f"core-w{w}.npz"
        if not path.exists():
            print(f"[warning] Cached core not found: {path}", file=sys.stderr)
            continue
        z = np.load(path)
        q = np.asarray(z["q"], dtype=float)
        meta = json.loads(str(z["meta"])) if "meta" in z else {}
        L = p.extent(q)
        cores[w] = {
            "w": w,
            "q": q,
            "L": L,
            "n0": len(q),
            "meta": meta,
        }
    return cores


def load_grid_envelopes(min_n: int = 50, max_n: int = 200) -> Dict[int, Dict[str, Any]]:
    """Load canonical integer square grid packings and their side lengths."""
    envelopes: Dict[int, Dict[str, Any]] = {}
    for n in range(min_n, max_n + 1):
        rows = grid(n)
        q = np.array(
            [[float(r["x"]), float(r["y"]), float(r["angle"])] for r in rows],
            dtype=float,
        )
        L = float(math.isqrt(n - 1) + 1)
        envelopes[n] = {
            "n": n,
            "L": L,
            "q": q,
            "rows": rows,
        }
    return envelopes


def load_incumbents(
    csv_path: Path = BORDER_SCAN_DIR / "merged.csv",
    min_n: int = 50,
    max_n: int = 200,
) -> Dict[int, Dict[str, Any]]:
    """Read incumbent packings and their certified side lengths from merged.csv."""
    puzzles = read_submission(csv_path)

    # Try pre-evaluated official evaluation reports if available
    ref_L: Dict[int, float] = {}
    for report_name in ("merged-pandas-official.json", "baseline-official.json"):
        report_path = csv_path.parent / report_name
        if report_path.exists():
            try:
                rep_data = json.loads(report_path.read_text())
                if "results" in rep_data:
                    for r in rep_data["results"]:
                        ref_L[int(r["n"])] = float(r["L"])
                    break
            except Exception:
                pass

    incumbents: Dict[int, Dict[str, Any]] = {}
    for n in range(min_n, max_n + 1):
        if n not in puzzles:
            continue
        rows = puzzles[n]
        if n in ref_L:
            L_val = ref_L[n]
        else:
            try:
                L_val = float(certify(n, rows))
            except Exception:
                L_val = float(math.isqrt(n - 1) + 1)
        incumbents[n] = {
            "n": n,
            "L": L_val,
            "rows": rows,
        }
    return incumbents


def build_structured_border_envelope(
    q_core: np.ndarray,
    t: int,
    orientation: str = "NE",
    margin: float = 1e-7,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Construct a structured t-layer border decomposed into two arms and a corner notch block.

    Args:
        q_core: Base core packing (normalized box [0, L_core]^2).
        t: Border thickness (layers 1, 2, or 3).
        orientation: One of 'NE', 'NW', 'SE', 'SW'.
        margin: Small safety offset to maintain strict SAT positive separation.

    Returns:
        (q_core_placed, arm1_squares, arm2_squares, corner_squares, L_total)
    """
    q_norm = p.normalize_box(q_core, margin=0.0)
    L_core = p.extent(q_norm)
    L_total = L_core + t + 2.0 * margin
    k_core = int(math.floor(L_core))
    pitch = 1.0 + 1e-8

    # Determine core placement shift inside container [0, L_total]^2
    if orientation == "NE":
        shift_x, shift_y = 0.0, 0.0
    elif orientation == "SW":
        shift_x, shift_y = float(t), float(t)
    elif orientation == "NW":
        shift_x, shift_y = float(t), 0.0
    elif orientation == "SE":
        shift_x, shift_y = 0.0, float(t)
    else:
        raise ValueError(f"Unknown orientation: {orientation}")

    q_placed = q_norm.copy()
    q_placed[:, 0] += shift_x
    q_placed[:, 1] += shift_y

    arm1_list: List[List[float]] = []
    arm2_list: List[List[float]] = []
    corner_list: List[List[float]] = []

    if orientation == "NE":
        # East arm: t columns x k_core rows along East edge of core
        for i in range(t):
            x = L_total - 0.5 - i * pitch
            for j in range(k_core):
                y = 0.5 + j * pitch
                arm1_list.append([x, y, 0.0])
        # North arm: k_core columns x t rows along North edge of core
        for j in range(t):
            y = L_total - 0.5 - j * pitch
            for i in range(k_core):
                x = 0.5 + i * pitch
                arm2_list.append([x, y, 0.0])
        # Corner notch junction: t x t block at top-right
        for i in range(t):
            x = L_total - 0.5 - i * pitch
            for j in range(t):
                y = L_total - 0.5 - j * pitch
                corner_list.append([x, y, 0.0])

    elif orientation == "SW":
        # West arm: t columns x k_core rows along West edge
        for i in range(t):
            x = 0.5 + i * pitch
            for j in range(k_core):
                y = t + 0.5 + j * pitch
                arm1_list.append([x, y, 0.0])
        # South arm: k_core columns x t rows along South edge
        for j in range(t):
            y = 0.5 + j * pitch
            for i in range(k_core):
                x = t + 0.5 + i * pitch
                arm2_list.append([x, y, 0.0])
        # Corner notch junction: t x t block at bottom-left
        for i in range(t):
            x = 0.5 + i * pitch
            for j in range(t):
                y = 0.5 + j * pitch
                corner_list.append([x, y, 0.0])

    elif orientation == "NW":
        # West arm: t columns x k_core rows
        for i in range(t):
            x = 0.5 + i * pitch
            for j in range(k_core):
                y = 0.5 + j * pitch
                arm1_list.append([x, y, 0.0])
        # North arm: k_core columns x t rows
        for j in range(t):
            y = L_total - 0.5 - j * pitch
            for i in range(k_core):
                x = t + 0.5 + i * pitch
                arm2_list.append([x, y, 0.0])
        # Corner notch junction: t x t block at top-left
        for i in range(t):
            x = 0.5 + i * pitch
            for j in range(t):
                y = L_total - 0.5 - j * pitch
                corner_list.append([x, y, 0.0])

    elif orientation == "SE":
        # East arm: t columns x k_core rows
        for i in range(t):
            x = L_total - 0.5 - i * pitch
            for j in range(k_core):
                y = t + 0.5 + j * pitch
                arm1_list.append([x, y, 0.0])
        # South arm: k_core columns x t rows
        for j in range(t):
            y = 0.5 + j * pitch
            for i in range(k_core):
                x = 0.5 + i * pitch
                arm2_list.append([x, y, 0.0])
        # Corner notch junction: t x t block at bottom-right
        for i in range(t):
            x = L_total - 0.5 - i * pitch
            for j in range(t):
                y = 0.5 + j * pitch
                corner_list.append([x, y, 0.0])

    arm1 = np.array(arm1_list, dtype=float) if arm1_list else np.empty((0, 3))
    arm2 = np.array(arm2_list, dtype=float) if arm2_list else np.empty((0, 3))
    corner = np.array(corner_list, dtype=float) if corner_list else np.empty((0, 3))

    return q_placed, arm1, arm2, corner, L_total


def generate_candidate_packings(
    cores: Dict[int, Dict[str, Any]],
    target_ns: Tuple[int, ...] = tuple(range(50, 201)),
    layers: Tuple[int, ...] = (1, 2, 3),
    orientations: Tuple[str, ...] = ("NE", "NW", "SE", "SW"),
    ref_L: Optional[Dict[int, float]] = None,
) -> Dict[int, List[Dict[str, Any]]]:
    """Generate structured border candidate packings across cores and orientations."""
    target_set = set(target_ns)
    min_target = min(target_ns)
    max_target = max(target_ns)
    candidates_by_n: Dict[int, List[Dict[str, Any]]] = {n: [] for n in target_ns}

    for w, core_info in cores.items():
        q_raw = core_info["q"]
        n0 = core_info["n0"]

        # Strategy 1: Core coordinate prefix deletions for n <= n0
        q_norm = p.normalize_box(q_raw, margin=0.0)
        for axis in (0, 1):
            for rev in (False, True):
                order = np.argsort(q_norm[:, axis], kind="stable")
                if rev:
                    order = order[::-1]
                for n_sub in range(min_target, min(n0, max_target) + 1):
                    if n_sub in target_set:
                        sub_q = q_norm[order[:n_sub]]
                        L_sub = p.extent(sub_q)
                        if ref_L is not None and L_sub >= ref_L[n_sub] + 0.01:
                            continue
                        candidates_by_n[n_sub].append({
                            "q": sub_q,
                            "L": L_sub,
                            "w": w,
                            "t": 0,
                            "orientation": "none",
                            "notches_filled": 0,
                            "kind": "core-prefix-deletion",
                            "axis": axis,
                            "rev": rev,
                        })

        # Strategy 2: Structured 1, 2, 3 layer borders with corner notches filled
        for t in layers:
            for ori in orientations:
                q_placed, arm1, arm2, corner, L_tot = build_structured_border_envelope(
                    q_raw, t=t, orientation=ori
                )
                full_arms = np.vstack([arm1, arm2]) if len(arm1) and len(arm2) else (arm1 if len(arm1) else arm2)
                n_arms = len(full_arms)
                n_corner = len(corner)

                # Order corner notches by distance from container center (inward first)
                if n_corner > 0:
                    center = np.array([L_tot / 2.0, L_tot / 2.0])
                    corner_dists = np.sum((corner[:, :2] - center) ** 2, axis=1)
                    corner_order = np.argsort(corner_dists)
                else:
                    corner_order = np.array([], dtype=int)

                # Pattern A: Full arms + c corner notch squares filled (0 <= c <= t^2)
                for c in range(n_corner + 1):
                    n_cand = n0 + n_arms + c
                    if n_cand in target_set:
                        selected_corner = corner[corner_order[:c]] if c > 0 else np.empty((0, 3))
                        cand_q = np.vstack([q_placed, full_arms, selected_corner])
                        L_cand = p.extent(cand_q)
                        if ref_L is not None and L_cand >= ref_L[n_cand] + 0.01:
                            continue
                        candidates_by_n[n_cand].append({
                            "q": cand_q,
                            "L": L_cand,
                            "w": w,
                            "t": t,
                            "orientation": ori,
                            "notches_filled": c,
                            "kind": "structured-border-full-arms-notched",
                        })

                # Pattern B: Balanced arm distribution for intermediate n
                if len(arm1) > 0 and len(arm2) > 0:
                    for needed in range(1, len(full_arms) + 1):
                        n_cand = n0 + needed
                        if n_cand in target_set:
                            n1 = needed // 2
                            n2 = needed - n1
                            if n1 <= len(arm1) and n2 <= len(arm2):
                                cand_q = np.vstack([q_placed, arm1[:n1], arm2[:n2]])
                                L_cand = p.extent(cand_q)
                                if ref_L is None or L_cand < ref_L[n_cand] + 0.01:
                                    candidates_by_n[n_cand].append({
                                        "q": cand_q,
                                        "L": L_cand,
                                        "w": w,
                                        "t": t,
                                        "orientation": ori,
                                        "notches_filled": 0,
                                        "kind": "structured-border-symmetric-arms",
                                    })

                # Pattern C: Coordinate-ordered border subsets
                all_border = np.vstack([full_arms, corner]) if n_corner > 0 else full_arms
                for axis in (0, 1):
                    for rev in (False, True):
                        b_order = np.argsort(all_border[:, axis], kind="stable")
                        if rev:
                            b_order = b_order[::-1]
                        for n_cand in target_ns:
                            needed = n_cand - n0
                            if 1 <= needed <= len(all_border):
                                cand_q = np.vstack([q_placed, all_border[b_order[:needed]]])
                                L_cand = p.extent(cand_q)
                                if ref_L is not None and L_cand >= ref_L[n_cand] + 0.01:
                                    continue
                                candidates_by_n[n_cand].append({
                                    "q": cand_q,
                                    "L": L_cand,
                                    "w": w,
                                    "t": t,
                                    "orientation": ori,
                                    "notches_filled": min(needed, n_corner),
                                    "kind": "structured-border-coordinate-ordered",
                                    "axis": axis,
                                    "rev": rev,
                                })

    return candidates_by_n


def run_lower_register_search(
    min_n: int = 50,
    max_n: int = 200,
    widths: Tuple[int, ...] = (1, 2, 3),
    layers: Tuple[int, ...] = (1, 2, 3),
    do_polish: bool = True,
    output_path: Path = DEFAULT_OUTPUT,
    border_scan_dir: Path = BORDER_SCAN_DIR,
) -> Dict[str, Any]:
    """Execute lower-register structured border search and save strictly winning candidates."""
    start_time = time.time()
    target_ns = tuple(range(min_n, max_n + 1))

    print(f"Loading cached cores for w={widths}...", flush=True)
    cores = load_cached_cores(border_scan_dir, widths=widths)
    for w, c in cores.items():
        print(f"  Core w={w}: n0={c['n0']}, L={c['L']:.6f}", flush=True)

    print("Loading integer grid envelopes...", flush=True)
    grid_envelopes = load_grid_envelopes(min_n, max_n)

    print("Reading incumbent packings from merged.csv...", flush=True)
    incumbents = load_incumbents(border_scan_dir / "merged.csv", min_n, max_n)
    ref_L = {n: inc["L"] for n, inc in incumbents.items()}

    pure_grid_count = sum(
        1 for n in target_ns if abs(ref_L.get(n, 0.0) - grid_envelopes[n]["L"]) < 1e-7
    )
    print(f"Incumbents in [{min_n}, {max_n}]: {len(incumbents)} loaded ({pure_grid_count} pure grid)", flush=True)

    print("Generating structured border candidates...", flush=True)
    candidates_by_n = generate_candidate_packings(
        cores=cores,
        target_ns=target_ns,
        layers=layers,
        ref_L=ref_L,
    )

    wins: Dict[str, Any] = {}
    win_count = 0
    screened_count = 0

    for n in sorted(target_ns):
        cands = candidates_by_n.get(n, [])
        if not cands:
            continue

        cands.sort(key=lambda c: c["L"])
        inc_L = ref_L[n]

        # Screen top promising candidates per n
        for cand_info in cands[:5]:
            best_q = cand_info["q"]
            best_L = cand_info["L"]
            screened_count += 1

            if best_L >= inc_L + (0.005 if do_polish else -1e-9):
                continue

            sat_ok, gap = p.sat_check(best_q)
            if not sat_ok:
                continue

            polished = False
            if do_polish and (best_L >= inc_L - 1e-9 or best_L < inc_L):
                try:
                    res = p.polish_band(
                        best_q,
                        best_L,
                        iterations=20,
                        position_trust=0.06,
                        angle_trust=0.008,
                        clearance=5e-9,
                    )
                    if res is not None and res.L < best_L:
                        best_q = res.q
                        best_L = res.L
                        polished = True
                except Exception as e:
                    print(f"  [n={n}] Polish notice: {e}", flush=True)

            if best_L >= inc_L - 1e-9:
                continue

            cand_rows = p.rows(best_q)
            try:
                cert_L_str = certify(n, cand_rows)
                cert_L = float(cert_L_str)
            except Exception as e:
                print(f"  [n={n}] Certification rejected: {e}", flush=True)
                continue

            if cert_L >= inc_L - 1e-9:
                continue

            csv_str = p.csv_text({n: cand_rows})
            chk = p.certify_transport(csv_str).get(str(n), {})
            literal_ok = chk.get("literal", {}).get("ok", False)
            pandas_ok = chk.get("actual_pandas", {}).get("ok", False)

            if not (literal_ok and pandas_ok):
                print(f"  [n={n}] Transport verification failed: literal={literal_ok}, pandas={pandas_ok}", flush=True)
                continue

            delta_L = cert_L - inc_L
            win_count += 1
            print(f"  WIN n={n}: candidate_L={cert_L_str} < inc_L={inc_L:.10f} (delta={delta_L:.4e})", flush=True)

            construction_meta = {
                "w": cand_info["w"],
                "t": cand_info["t"],
                "orientation": cand_info["orientation"],
                "notches_filled": cand_info["notches_filled"],
                "kind": cand_info["kind"],
                "polished": polished,
            }

            wins[str(n)] = {
                "n": n,
                "candidate_L": cert_L_str,
                "incumbent_L": f"{inc_L:.50f}" if isinstance(inc_L, float) else str(inc_L),
                "delta_L": delta_L,
                "sat_min_gap": float(gap),
                "construction": construction_meta,
                "certificates": chk,
                "rows": cand_rows,
            }
            break

    elapsed = time.time() - start_time
    summary = {
        "target_range": [min_n, max_n],
        "cores_searched": list(widths),
        "layers_searched": list(layers),
        "promising_screened": screened_count,
        "wins_certified": win_count,
        "elapsed_seconds": elapsed,
        "pandas_version": pd.__version__,
    }

    output_data = {
        "summary": summary,
        "candidates": wins,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output_data, indent=2))
    print(f"Results saved to {output_path}: {win_count} strictly winning candidates verified.", flush=True)

    return output_data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lower-register structured border search for n in [50, 200]"
    )
    parser.add_argument("--min-n", type=int, default=50, help="Minimum n (default: 50)")
    parser.add_argument("--max-n", type=int, default=200, help="Maximum n (default: 200)")
    parser.add_argument(
        "--widths",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help="Core widths to search (default: 1 2 3)",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help="Border layer counts (default: 1 2 3)",
    )
    parser.add_argument(
        "--no-polish",
        action="store_true",
        help="Skip polish_band step",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output JSON path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--border-scan-dir",
        type=Path,
        default=BORDER_SCAN_DIR,
        help=f"Directory containing core-w*.npz and merged.csv (default: {BORDER_SCAN_DIR})",
    )
    args = parser.parse_args()

    run_lower_register_search(
        min_n=args.min_n,
        max_n=args.max_n,
        widths=tuple(args.widths),
        layers=tuple(args.layers),
        do_polish=not args.no_polish,
        output_path=args.output,
        border_scan_dir=args.border_scan_dir,
    )


if __name__ == "__main__":
    main()
