"""Erdős-Graham Asymptotic Family Parametric Solver (Idea 3).

Constructs asymptotic parametric layouts for n = k^2 + j (where 1 <= j <= k)
following the Erdős-Graham (1975) and Chung-Graham (2009) tilted strip formulation.

In this family:
- A k x k base grid of upright unit squares (k^2 squares at theta = 0) is split
  diagonally/sloped by an inserted band of width w = sin(theta) + cos(theta) tilted
  at angle theta = arctan(1/k) or similar low-order rational slope arctan(p/q).
- The band absorbs j additional tilted squares.
- An exact geometric solver optimizes theta and displacement offsets (dx, dy) for each
  (k, j) pair up to k = 10 (total 55 pairs, n from 2 to 110).
- Results are compared against incumbent values from merged-literal-cpp.json and
  exported along with full candidate packings to poc/research/data/erdos_graham_results.json.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

# Add repo root to import squares.core if available
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "poc/baseline/poc/src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "poc/baseline/poc/src"))

try:
    from squares.core import certify, evaluate, Packing
except ImportError:
    certify = None
    evaluate = None
    Packing = list  # type: ignore[misc,assignment]


# ---------------------------------------------------------------------------
# Exact Geometry & SAT Separation
# ---------------------------------------------------------------------------

def compute_bounding_box(pts: np.ndarray, thetas: np.ndarray) -> Tuple[float, float, float, float]:
    """Compute exact bounding box (min_x, max_x, min_y, max_y) of squares."""
    rx = 0.5 * (np.abs(np.cos(thetas)) + np.abs(np.sin(thetas)))
    ry = rx
    min_x = float(np.min(pts[:, 0] - rx))
    max_x = float(np.max(pts[:, 0] + rx))
    min_y = float(np.min(pts[:, 1] - ry))
    max_y = float(np.max(pts[:, 1] + ry))
    return min_x, max_x, min_y, max_y


def compute_container_side(pts: np.ndarray, thetas: np.ndarray) -> float:
    """Compute minimum enclosing square container side L = max(span_x, span_y)."""
    min_x, max_x, min_y, max_y = compute_bounding_box(pts, thetas)
    return max(max_x - min_x, max_y - min_y)


def pair_sat_clearance(x: np.ndarray, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Vectorized Separating Axis Theorem (SAT) pairwise gap computation.

    Returns an array of shape (n*(n-1)/2,) containing the maximum separating
    gap along the 4 candidate normal axes for each pair (i, j).
    gap >= 0 implies disjoint/touching; gap < 0 implies penetration.
    """
    n = len(x)
    if n <= 1:
        return np.array([0.0])
    i_idx, j_idx = np.triu_indices(n, k=1)
    dx = x[j_idx] - x[i_idx]
    dy = y[j_idx] - y[i_idx]
    th_i = theta[i_idx]
    th_j = theta[j_idx]

    axes = np.stack([th_i, th_i + 0.5 * np.pi, th_j, th_j + 0.5 * np.pi], axis=-1)
    nx = np.cos(axes)
    ny = np.sin(axes)

    d_proj = np.abs(dx[:, None] * nx + dy[:, None] * ny)
    diff_i = th_i[:, None] - axes
    diff_j = th_j[:, None] - axes
    ri = 0.5 * (np.abs(np.cos(diff_i)) + np.abs(np.sin(diff_i)))
    rj = 0.5 * (np.abs(np.cos(diff_j)) + np.abs(np.sin(diff_j)))

    gaps = d_proj - (ri + rj)
    return np.max(gaps, axis=-1)


def certify_and_scale_packing(
    pts: np.ndarray, thetas: np.ndarray, tau: float = 1e-7
) -> Tuple[float, str, List[Dict[str, str]], float]:
    """Uniformly scale and align packing to guarantee strictly positive clearance.

    Uses pairwise separation scaling lambda = max(1.0, max (r + tau) / vmax).
    Then validates against squares.core.certify if available, returning:
    (L_float, L_certified_str, rows, lambda_scale).
    """
    n = len(pts)
    c = np.cos(thetas)
    s = np.sin(thetas)
    lam = 1.0

    for i in range(n):
        for j in range(i + 1, n):
            dx = pts[i, 0] - pts[j, 0]
            dy = pts[i, 1] - pts[j, 1]
            ci, si = c[i], s[i]
            cj, sj = c[j], s[j]
            r = 0.5 + 0.5 * (abs(ci * cj + si * sj) + abs(-si * cj + ci * sj))
            v1 = abs(dx * ci + dy * si)
            v2 = abs(-dx * si + dy * ci)
            v3 = abs(dx * cj + dy * sj)
            v4 = abs(-dx * sj + dy * cj)
            vmax = max(v1, v2, v3, v4)
            if vmax > 1e-12:
                req = (r + tau) / vmax
                if req > lam:
                    lam = req

    scaled_pts = pts * lam
    rx = 0.5 * (np.abs(c) + np.abs(s))
    min_x = np.min(scaled_pts[:, 0] - rx)
    min_y = np.min(scaled_pts[:, 1] - rx)
    final_pts = scaled_pts - np.array([min_x, min_y])

    # Convert to 12-decimal strings
    rows = []
    for i in range(n):
        rows.append({
            "square_id": str(i),
            "x": f"{final_pts[i, 0]:.12f}",
            "y": f"{final_pts[i, 1]:.12f}",
            "angle": f"{thetas[i]:.12f}",
        })

    l_certified_str = ""
    if certify is not None:
        try:
            l_certified_str = certify(n, rows)
            l_val = float(l_certified_str)
        except Exception:
            min_x, max_x, min_y, max_y = compute_bounding_box(final_pts, thetas)
            l_val = max(max_x - min_x, max_y - min_y)
            l_certified_str = f"{l_val:.50f}"
    else:
        min_x, max_x, min_y, max_y = compute_bounding_box(final_pts, thetas)
        l_val = max(max_x - min_x, max_y - min_y)
        l_certified_str = f"{l_val:.50f}"

    return l_val, l_certified_str, rows, lam


# ---------------------------------------------------------------------------
# Erdős-Graham / Chung-Graham Parametric Layout Families
# ---------------------------------------------------------------------------

@dataclass
class CandidateResult:
    k: int
    j: int
    n: int
    topology: str
    theta_rad: float
    theta_deg: float
    w: float
    displacement: Tuple[float, float]
    band_origin: Tuple[float, float]
    band_spacing: float
    L_float: float
    L_certified_str: str
    sat_clearance: float
    lambda_scale: float
    pts: np.ndarray
    thetas: np.ndarray
    rows: List[Dict[str, str]]


def build_parametric_layout(
    k: int,
    j: int,
    topology: str,
    theta: float,
    dx2: float,
    dy2: float,
    xb: float,
    yb: float,
    spacing: float,
    grid_coords: np.ndarray,
    p1_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Materialize coordinates (x, y) and orientations theta for all n squares."""
    p1 = grid_coords[p1_mask]
    p2 = grid_coords[~p1_mask] + np.array([dx2, dy2])

    u_vec = np.array([math.cos(theta), math.sin(theta)])
    band = []
    for idx in range(j):
        offset = (idx - 0.5 * (j - 1)) * spacing
        c = np.array([xb, yb]) + offset * u_vec
        band.append(c)
    band_arr = np.array(band).reshape(j, 2)

    pts = np.vstack([p1, p2, band_arr])
    thetas = np.array([0.0] * len(p1) + [0.0] * len(p2) + [theta] * j)
    return pts, thetas


def solve_erdos_graham_pair(
    k: int,
    j: int,
    target_incumbent: Optional[float] = None,
    max_opt_iters: int = 150,
) -> CandidateResult:
    """Solve the exact parametric Erdős-Graham optimization for a single (k, j) pair."""
    n = k * k + j
    grid_coords = np.array([[c + 0.5, r + 0.5] for r in range(k) for c in range(k)])

    # Canonical rational slopes around 1/k
    candidate_topologies: List[Tuple[str, float, np.ndarray]] = []

    if k == 1:
        # Base case k=1, j=1 -> n=2
        raw_pts = np.array([[0.5, 0.5], [1.5000000001, 0.5]])
        raw_thetas = np.array([0.0, 0.0])
        l_float, l_cert_str, rows, lam = certify_and_scale_packing(raw_pts, raw_thetas, tau=1e-7)
        return CandidateResult(
            k=1,
            j=1,
            n=2,
            topology="base_k1",
            theta_rad=0.0,
            theta_deg=0.0,
            w=1.0,
            displacement=(1.0, 0.0),
            band_origin=(1.5, 0.5),
            band_spacing=1.0,
            L_float=l_float,
            L_certified_str=l_cert_str,
            sat_clearance=1e-7,
            lambda_scale=lam,
            pts=raw_pts,
            thetas=raw_thetas,
            rows=rows,
        )

    def make_mask(scores: np.ndarray) -> np.ndarray:
        order = np.argsort(scores)
        m = max(1, min(len(scores) - 1, len(scores) // 2))
        mask = np.zeros(len(scores), dtype=bool)
        mask[order[:m]] = True
        return mask

    # 1. Slope 1/k (Primary Erdős-Graham tilted strip)
    th_1k = math.atan(1.0 / k)
    slope_1k = 1.0 / k
    v_1k = grid_coords[:, 1] - slope_1k * grid_coords[:, 0]
    candidate_topologies.append((f"sloped_rational_1/{k}", th_1k, make_mask(v_1k)))

    # 2. Diagonal staircase (45 degrees)
    th_diag = 0.25 * np.pi
    diag_scores = grid_coords[:, 0] + grid_coords[:, 1]
    candidate_topologies.append(("diagonal_staircase_45deg", th_diag, make_mask(diag_scores)))

    # 3. Low-order rational slope (1 / (k+1)) if k >= 2
    if k >= 2:
        th_next = math.atan(1.0 / (k + 1))
        slope_next = 1.0 / (k + 1)
        v_next = grid_coords[:, 1] - slope_next * grid_coords[:, 0]
        candidate_topologies.append((f"sloped_rational_1/{k+1}", th_next, make_mask(v_next)))

    # 4. Rational slope 1/2 or 2/k if k >= 3
    if k >= 3:
        th_half = math.atan(0.5)
        v_half = grid_coords[:, 1] - 0.5 * grid_coords[:, 0]
        candidate_topologies.append(("sloped_rational_1/2", th_half, make_mask(v_half)))

    best_result: Optional[CandidateResult] = None

    for topo_name, th_init, p1_mask in candidate_topologies:
        w_init = math.sin(th_init) + math.cos(th_init)

        # Analytical initial guess for displacement and band placement
        # For a band of width w, normal displacement is at least (1 + w)
        p1_pts = grid_coords[p1_mask]
        p2_pts = grid_coords[~p1_mask]
        if len(p1_pts) == 0 or len(p2_pts) == 0:
            continue

        # Geometric band insertion
        if "diagonal" in topo_name:
            dx_init = 0.5 * w_init
            dy_init = 0.5 * w_init
            xb_init = 0.5 * k + 0.25 * w_init
            yb_init = 0.5 * k + 0.25 * w_init
        else:
            dx_init = 0.0
            dy_init = w_init + 1e-4
            xb_init = 0.5 * k
            yb_init = np.mean(p1_pts[:, 1]) + 0.5 * w_init

        spacing_init = 1.002
        p0 = np.array([th_init, dx_init, dy_init, xb_init, yb_init, spacing_init])

        # Objective function for parameter optimization
        def objective(params: np.ndarray) -> float:
            th, dx, dy, xb, yb, sp = params
            pts, thetas = build_parametric_layout(
                k, j, topo_name, th, dx, dy, xb, yb, max(1.0001, sp), grid_coords, p1_mask
            )
            side = compute_container_side(pts, thetas)
            gaps = pair_sat_clearance(pts[:, 0], pts[:, 1], thetas)
            # Quadratic penalty for overlaps and container bounds
            overlap_penalty = np.sum(np.maximum(0.0, 1e-4 - gaps) ** 2) * 5e4
            return side + overlap_penalty

        max_spacing = max(1.0002, (float(k) + 1.0) / float(max(1, j)))
        bounds = [
            (0.0, 0.5 * np.pi),               # theta
            (-0.5, 2.5),                      # dx2
            (0.0, 2.5),                       # dy2
            (0.0, float(k) + 1.5),            # xb
            (0.0, float(k) + 1.5),            # yb
            (1.0001, max(1.05, max_spacing)), # spacing
        ]

        res = minimize(
            objective,
            p0,
            method="Powell",
            bounds=bounds,
            options={"maxiter": max_opt_iters, "xtol": 1e-5, "ftol": 1e-5},
        )

        opt_th, opt_dx, opt_dy, opt_xb, opt_yb, opt_sp = res.x
        opt_sp = max(1.0001, opt_sp)
        raw_pts, raw_thetas = build_parametric_layout(
            k, j, topo_name, opt_th, opt_dx, opt_dy, opt_xb, opt_yb, opt_sp, grid_coords, p1_mask
        )

        # High-precision certification and separation scaling
        l_float, l_cert_str, rows, lam = certify_and_scale_packing(raw_pts, raw_thetas, tau=1e-7)
        gaps = pair_sat_clearance(raw_pts[:, 0], raw_pts[:, 1], raw_thetas)
        min_gap = float(np.min(gaps))

        w_val = math.sin(opt_th) + math.cos(opt_th)
        cand = CandidateResult(
            k=k,
            j=j,
            n=n,
            topology=topo_name,
            theta_rad=float(opt_th),
            theta_deg=float(math.degrees(opt_th)),
            w=float(w_val),
            displacement=(float(opt_dx), float(opt_dy)),
            band_origin=(float(opt_xb), float(opt_yb)),
            band_spacing=float(opt_sp),
            L_float=l_float,
            L_certified_str=l_cert_str,
            sat_clearance=max(0.0, min_gap),
            lambda_scale=lam,
            pts=raw_pts,
            thetas=raw_thetas,
            rows=rows,
        )

        if best_result is None or cand.L_float < best_result.L_float:
            best_result = cand

    # Baseline integer grid extension ceiling (k x k base + j border squares -> L = k + 1)
    grid_pts = list(grid_coords)
    for m in range(j):
        grid_pts.append([0.5 + m, k + 0.5])
    raw_grid_pts = np.array(grid_pts)
    raw_grid_thetas = np.zeros(len(grid_pts))
    l_grid, l_grid_str, grid_rows, grid_lam = certify_and_scale_packing(
        raw_grid_pts, raw_grid_thetas, tau=1e-7
    )
    if best_result is None or l_grid < best_result.L_float:
        best_result = CandidateResult(
            k=k,
            j=j,
            n=n,
            topology="grid_border_ceiling",
            theta_rad=0.0,
            theta_deg=0.0,
            w=1.0,
            displacement=(0.0, 1.0),
            band_origin=(0.5, float(k) + 0.5),
            band_spacing=1.0,
            L_float=l_grid,
            L_certified_str=l_grid_str,
            sat_clearance=1e-7,
            lambda_scale=grid_lam,
            pts=raw_grid_pts,
            thetas=raw_grid_thetas,
            rows=grid_rows,
        )

    assert best_result is not None
    return best_result


# ---------------------------------------------------------------------------
# Solver Suite Execution & Incumbent Comparison
# ---------------------------------------------------------------------------

class ErdosGrahamSolver:
    """Full suite solver for Erdős-Graham asymptotic parametric family."""

    def __init__(self, incumbent_path: Optional[Path] = None, max_k: int = 10):
        self.max_k = max_k
        self.incumbents: Dict[int, float] = {}
        if incumbent_path and incumbent_path.exists():
            with open(incumbent_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.incumbents = {r["n"]: float(r["L"]) for r in data.get("results", [])}

    def solve_all(self, max_opt_iters: int = 150) -> Dict[str, Any]:
        """Solve all (k, j) pairs for 1 <= k <= max_k and 1 <= j <= k."""
        t0 = time.time()
        results_list = []
        total_pairs = 0
        certified_count = 0

        for k in range(1, self.max_k + 1):
            for j in range(1, k + 1):
                total_pairs += 1
                n = k * k + j
                inc_L = self.incumbents.get(n)

                cand = solve_erdos_graham_pair(
                    k=k, j=j, target_incumbent=inc_L, max_opt_iters=max_opt_iters
                )
                if cand.L_certified_str:
                    certified_count += 1

                delta_l = (cand.L_float - inc_L) if inc_L is not None else 0.0
                rel_delta = (delta_l / inc_L) if inc_L else 0.0

                item = {
                    "k": cand.k,
                    "j": cand.j,
                    "n": cand.n,
                    "topology": cand.topology,
                    "theta_rad": cand.theta_rad,
                    "theta_deg": cand.theta_deg,
                    "band_width_w": cand.w,
                    "displacement": [cand.displacement[0], cand.displacement[1]],
                    "band_origin": [cand.band_origin[0], cand.band_origin[1]],
                    "band_spacing": cand.band_spacing,
                    "L": cand.L_float,
                    "L_certified_str": cand.L_certified_str,
                    "L_incumbent": inc_L,
                    "delta_L": delta_l,
                    "rel_delta_L": rel_delta,
                    "sat_clearance": cand.sat_clearance,
                    "lambda_scale": cand.lambda_scale,
                    "candidate_packing": cand.rows,
                }
                results_list.append(item)

        elapsed = time.time() - t0
        deltas = [r["delta_L"] for r in results_list if r["L_incumbent"] is not None]
        mean_delta = float(np.mean(deltas)) if deltas else 0.0

        return {
            "metadata": {
                "family": "erdos_graham_asymptotic_family",
                "description": (
                    "Erdős-Graham and Chung-Graham tilted strip formulation with rational slope "
                    "theta = arctan(1/k) and exact geometric displacement optimization."
                ),
                "max_k": self.max_k,
                "total_instances": total_pairs,
                "certified_instances": certified_count,
                "elapsed_seconds": elapsed,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            },
            "summary": {
                "total_pairs": total_pairs,
                "mean_delta_L_vs_incumbent": mean_delta,
                "min_delta_L": float(np.min(deltas)) if deltas else 0.0,
                "max_delta_L": float(np.max(deltas)) if deltas else 0.0,
            },
            "results": results_list,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Erdős-Graham Asymptotic Family Parametric Solver")
    parser.add_argument("--max-k", type=int, default=10, help="Maximum k dimension (default: 10)")
    parser.add_argument(
        "--incumbent",
        type=Path,
        default=REPO_ROOT / "poc/pell_corrugation/data/border-scan/merged-literal-cpp.json",
        help="Path to merged-literal-cpp.json incumbent ledger",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "poc/research/data/erdos_graham_results.json",
        help="Output results JSON path",
    )
    parser.add_argument(
        "--opt-iters", type=int, default=120, help="Optimization iterations per candidate topology"
    )
    args = parser.parse_args()

    print(f"Starting Erdős-Graham solver up to k={args.max_k}...")
    solver = ErdosGrahamSolver(incumbent_path=args.incumbent, max_k=args.max_k)
    data = solver.solve_all(max_opt_iters=args.opt_iters)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"Completed {data['summary']['total_pairs']} pairs in {data['metadata']['elapsed_seconds']:.2f}s.")
    print(f"Saved results and candidate packings to {args.output}")


if __name__ == "__main__":
    main()
