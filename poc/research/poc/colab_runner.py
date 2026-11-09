"""Colab A100 Heavy Search Pipeline & Teardown (Idea 3 / Epigraph LP).

Executes parallel GPU-accelerated continuous angle and offset grid-screening using
PyTorch/NumPy on CUDA across the top 40 targets. Refines candidate packings with
4-sign-piece epigraph LP, includes automated cloud runtime self-termination to
eliminate lingering compute costs, and provides a local verification script to
ingest exported solution candidates and validate them against the official 50-digit
interval certifier.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import io
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Try importing PyTorch for CUDA acceleration; fallback gracefully to NumPy
try:
    import torch
except ImportError:
    torch = None

try:
    from scipy.optimize import linprog
    from scipy.spatial import cKDTree
except ImportError:
    linprog = None
    cKDTree = None

# Ensure baseline package is discoverable for official certification
REPO_ROOT = Path(__file__).resolve().parents[3]
BASELINE_SRC = REPO_ROOT / "poc/baseline/poc/src"
if str(BASELINE_SRC) not in sys.path and BASELINE_SRC.exists():
    sys.path.insert(0, str(BASELINE_SRC))

try:
    from squares.core import certify, evaluate, Packing, Puzzles
except ImportError:
    certify = None
    evaluate = None
    Packing = list  # type: ignore[misc,assignment]
    Puzzles = dict  # type: ignore[misc,assignment]

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
MERGED_LITERAL_PATH = REPO_ROOT / "poc/pell_corrugation/data/border-scan/merged-literal-cpp.json"
BASELINE_TARGETS_CSV = REPO_ROOT / "poc/boundary_zipper/data/baseline-targets.csv"

# ---------------------------------------------------------------------------
# Top 40 Target Ranking (Pell-17 corrugation, grid ceilings, & heuristic records)
# ---------------------------------------------------------------------------
TOP_40_TARGETS: tuple[int, ...] = (
    # Pell-17 family and top large-n offenders (unlocked via 4-sign-piece corrugation)
    972, 910, 971, 908, 909, 911, 973, 591, 592, 593, 543, 544, 545,
    # High-gain step discontinuities (grid ceilings)
    20, 30, 42, 56, 72,
    # 45-degree bands and corner pockets with high dilation waste
    26, 27, 38, 40, 51, 52, 65, 67,
    # Historical heuristic records / multi-angle clusters
    28, 29, 37, 39, 41, 50, 53, 54, 55, 66, 68, 69, 70, 71,
)

SIGNS: tuple[tuple[float, float], ...] = (
    (-1.0, -1.0),
    (-1.0, 1.0),
    (1.0, -1.0),
    (1.0, 1.0),
)
S2 = math.sqrt(2.0)


@dataclass
class CandidateResult:
    target_n: int
    incumbent_L: float
    unpolished_L: float
    polished_L: float
    delta_vs_incumbent: float
    within_1e3: bool
    family: str
    theta: float
    w: int
    v_off: float
    u_off: float
    stagger: float
    n_tilted: int
    n_upright: int
    layout: list[dict[str, float]]
    certified_L_50: Optional[str] = None
    score_delta: Optional[float] = None


# ---------------------------------------------------------------------------
# Exact Geometry & Classical SAT
# ---------------------------------------------------------------------------

def extent(q: np.ndarray) -> float:
    """Compute unconstrained square container side L for packing q."""
    if len(q) == 0:
        return 0.0
    c = np.abs(np.cos(q[:, 2]))
    s = np.abs(np.sin(q[:, 2]))
    rx = 0.5 * (c + s)
    ry = rx
    min_x = float(np.min(q[:, 0] - rx))
    max_x = float(np.max(q[:, 0] + rx))
    min_y = float(np.min(q[:, 1] - ry))
    max_y = float(np.max(q[:, 1] + ry))
    return max(max_x - min_x, max_y - min_y)


def normalize_box(q: np.ndarray, margin: float = 0.0) -> np.ndarray:
    """Translate centers so squares fit tightly in [margin, L - margin]^2."""
    c = np.abs(np.cos(q[:, 2]))
    s = np.abs(np.sin(q[:, 2]))
    rx = 0.5 * (c + s)
    ry = rx
    min_x = np.min(q[:, 0] - rx)
    min_y = np.min(q[:, 1] - ry)
    shifted = q.copy()
    shifted[:, 0] += margin - min_x
    shifted[:, 1] += margin - min_y
    return shifted


def sat_check(q: np.ndarray, clearance: float = 0.0) -> tuple[bool, float]:
    """Exhaustive all-pairs SAT separation check using NumPy."""
    n = len(q)
    if n <= 1:
        return True, float("inf")
    c = np.cos(q[:, 2])
    s = np.sin(q[:, 2])
    min_gap = float("inf")

    for i in range(n):
        dx = q[i + 1 :, 0] - q[i, 0]
        dy = q[i + 1 :, 1] - q[i, 1]
        dist2 = dx * dx + dy * dy
        # Potential collision filter: centers within 2*sqrt(2)
        cand = np.where(dist2 < 8.01)[0]
        if len(cand) == 0:
            continue
        cand_j = i + 1 + cand
        c_i, s_i = c[i], s[i]
        c_j, s_j = c[cand_j], s[cand_j]
        dx_c = dx[cand]
        dy_c = dy[cand]

        best = np.full(len(cand), -float("inf"))
        for phi in (q[i, 2], q[i, 2] + math.pi / 2.0):
            cp, sp = math.cos(phi), math.sin(phi)
            proj = np.abs(dx_c * cp + dy_c * sp)
            h_i = 0.5 * (abs(math.cos(q[i, 2] - phi)) + abs(math.sin(q[i, 2] - phi)))
            h_j = 0.5 * (np.abs(np.cos(q[cand_j, 2] - phi)) + np.abs(np.sin(q[cand_j, 2] - phi)))
            best = np.maximum(best, proj - (h_i + h_j))

        for offset in (0.0, math.pi / 2.0):
            phi_j = q[cand_j, 2] + offset
            cp = np.cos(phi_j)
            sp = np.sin(phi_j)
            proj = np.abs(dx_c * cp + dy_c * sp)
            h_i = 0.5 * (np.abs(np.cos(q[i, 2] - phi_j)) + np.abs(np.sin(q[i, 2] - phi_j)))
            h_j = 0.5 * (np.abs(np.cos(q[cand_j, 2] - phi_j)) + np.abs(np.sin(q[cand_j, 2] - phi_j)))
            best = np.maximum(best, proj - (h_i + h_j))

        cur_min = float(np.min(best))
        if cur_min < min_gap:
            min_gap = cur_min
        if np.any(best < clearance):
            return False, min_gap

    return True, min_gap


# ---------------------------------------------------------------------------
# PyTorch GPU-Accelerated Batch SAT Screening
# ---------------------------------------------------------------------------

class GPUScreener:
    """Vectorized PyTorch batch SAT evaluator running on CUDA or CPU."""

    def __init__(self, device: Optional[str] = None):
        if device is None:
            if torch is not None and torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu") if torch is not None else None
        else:
            self.device = torch.device(device) if torch is not None else None

    @property
    def is_cuda(self) -> bool:
        return self.device is not None and self.device.type == "cuda"

    def screen_batch(
        self,
        batch_candidates: Sequence[np.ndarray],
        clearance: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Screen batch of candidates of identical size N.

        Returns (L_arr, min_gap_arr) for each candidate in batch.
        """
        if len(batch_candidates) == 0:
            return np.array([]), np.array([])

        if torch is None or self.device is None:
            # Fallback to NumPy
            ls = []
            gaps = []
            for q in batch_candidates:
                ls.append(extent(q))
                _, g = sat_check(q, clearance=clearance)
                gaps.append(g)
            return np.array(ls), np.array(gaps)

        B = len(batch_candidates)
        N = len(batch_candidates[0])

        q_tensor = torch.tensor(
            np.stack(batch_candidates), dtype=torch.float32, device=self.device
        )  # [B, N, 3]

        # 1. Compute container extents in parallel on GPU
        c = torch.abs(torch.cos(q_tensor[:, :, 2]))
        s = torch.abs(torch.sin(q_tensor[:, :, 2]))
        r = 0.5 * (c + s)
        x_min = torch.min(q_tensor[:, :, 0] - r, dim=1).values
        x_max = torch.max(q_tensor[:, :, 0] + r, dim=1).values
        y_min = torch.min(q_tensor[:, :, 1] - r, dim=1).values
        y_max = torch.max(q_tensor[:, :, 1] + r, dim=1).values
        L_batch = torch.maximum(x_max - x_min, y_max - y_min)

        # 2. Pairwise SAT checks across all pairs (i, j)
        idx_i, idx_j = torch.triu_indices(N, N, offset=1, device=self.device)
        P = idx_i.shape[0]

        q_i = q_tensor[:, idx_i, :]  # [B, P, 3]
        q_j = q_tensor[:, idx_j, :]  # [B, P, 3]

        dx = q_j[:, :, 0] - q_i[:, :, 0]  # [B, P]
        dy = q_j[:, :, 1] - q_i[:, :, 1]
        dist2 = dx * dx + dy * dy

        # Only evaluate potential overlapping pairs (center dist^2 < 8.01)
        # 4 axes: phi_i0 = th_i, phi_i1 = th_i + pi/2, phi_j0 = th_j, phi_j1 = th_j + pi/2
        th_i = q_i[:, :, 2]
        th_j = q_j[:, :, 2]

        axes = torch.stack([th_i, th_i + 0.5 * math.pi, th_j, th_j + 0.5 * math.pi], dim=-1)  # [B, P, 4]
        cos_ax = torch.cos(axes)
        sin_ax = torch.sin(axes)

        proj = torch.abs(dx.unsqueeze(-1) * cos_ax + dy.unsqueeze(-1) * sin_ax)  # [B, P, 4]

        # Support functions
        h_i = 0.5 * (
            torch.abs(torch.cos(th_i.unsqueeze(-1) - axes))
            + torch.abs(torch.sin(th_i.unsqueeze(-1) - axes))
        )
        h_j = 0.5 * (
            torch.abs(torch.cos(th_j.unsqueeze(-1) - axes))
            + torch.abs(torch.sin(th_j.unsqueeze(-1) - axes))
        )

        pair_gaps = proj - (h_i + h_j)  # [B, P, 4]
        sat_gaps = torch.max(pair_gaps, dim=-1).values  # [B, P]

        # Far pairs (dist2 >= 8.0) are trivially non-overlapping
        sat_gaps = torch.where(dist2 < 8.01, sat_gaps, torch.tensor(1.0, device=self.device))

        min_gaps = torch.min(sat_gaps, dim=-1).values  # [B]

        return L_batch.cpu().numpy(), min_gaps.cpu().numpy()


# ---------------------------------------------------------------------------
# 4-Sign-Piece Epigraph LP Solver (Candidate Refinement)
# ---------------------------------------------------------------------------

def chosen_axis(q: np.ndarray, i: int, j: int, previous=None, hysteresis: float = 1e-9):
    """Find dominant separating axis between square i and j."""
    d = q[j, :2] - q[i, :2]
    choices = []
    for owner in (i, j):
        for k in (0, 1):
            phi = float(q[owner, 2]) + k * math.pi / 2.0
            a = np.array([math.cos(phi), math.sin(phi)])
            direction = 1 if float(d @ a) >= 0.0 else -1
            a *= direction
            h = sum(
                0.5 * (abs(math.cos(float(q[t, 2]) - phi)) + abs(math.sin(float(q[t, 2]) - phi)))
                for t in (i, j)
            )
            choices.append((float(d @ a - h), (owner, k, direction)))
    best = max(choices, key=lambda x: x[0])
    if previous is not None:
        for value, key in choices:
            if key == previous and value >= best[0] - hysteresis:
                return key
    return best[1]


def piece_gap_gradient(q: np.ndarray, i: int, j: int, axis, signs: tuple[float, float]):
    """Gradient of pairwise gap under 4-sign-piece epigraph linearization."""
    owner, k, direction = axis
    phi = float(q[owner, 2]) + k * math.pi / 2.0
    a = direction * np.array([math.cos(phi), math.sin(phi)])
    ap = np.array([-a[1], a[0]])
    phi += 0.0 if direction == 1 else math.pi
    other = j if owner == i else i
    r = float(q[other, 2]) - phi
    s1, s2 = signs
    hp = 0.5 * (-s1 * math.sin(r) + s2 * math.cos(r))
    d = q[j, :2] - q[i, :2]
    g = float(d @ a - 0.5 - 0.5 * (s1 * math.cos(r) + s2 * math.sin(r)))
    grad = np.zeros(6)
    grad[:2] = -a
    grad[3:5] = a
    grad[2 if owner == i else 5] = float(d @ ap) + hp
    grad[2 if other == i else 5] = -hp
    return g, grad


def _lp_step_4sign(
    q: np.ndarray,
    upper_L: float,
    position_trust: float,
    angle_trust: float,
    clearance: float,
    axes: dict,
) -> tuple[np.ndarray | None, str]:
    """Single 4-sign-piece epigraph LP step to shrink container side L."""
    if linprog is None or cKDTree is None:
        return None, "missing_scipy"

    from scipy.sparse import coo_matrix

    n = len(q)
    free = angle_trust > 0.0
    stride = 3 if free else 2
    nv = stride * n + 3  # offsets (dx, dy, [dtheta]) + xmin, ymin, L
    rr: list[int] = []
    cc: list[int] = []
    vv: list[float] = []
    rhs: list[float] = []

    def row(coeff: dict[int, float], b: float) -> None:
        r = len(rhs)
        rhs.append(float(b))
        for col, val in coeff.items():
            if val:
                rr.append(r)
                cc.append(col)
                vv.append(float(val))

    # Bounding box epigraph constraints for each square
    for i in range(n):
        t = float(q[i, 2])
        c, s = math.cos(t), math.sin(t)
        pieces = (
            [(0.5 * (a * c + b * s), 0.5 * (-a * s + b * c)) for a, b in SIGNS]
            if free
            else [(0.5 * (abs(c) + abs(s)), 0.0)]
        )
        for h, dh in pieces:
            for dim in (0, 1):
                lower = {stride * i + dim: -1.0, stride * n + dim: 1.0}
                upperrow = {stride * i + dim: 1.0, stride * n + dim: -1.0, nv - 1: -1.0}
                if free:
                    lower[stride * i + 2] = dh
                    upperrow[stride * i + 2] = dh
                row(lower, float(q[i, dim]) - h - clearance)
                row(upperrow, -float(q[i, dim]) - h - clearance)

    # Pairwise collision separation constraints
    radius = S2 + 2.0 * S2 * position_trust + 1e-6
    tree = cKDTree(q[:, :2])
    pairs = sorted(tree.query_pairs(radius))

    for i, j in pairs:
        key = (i, j)
        axis = chosen_axis(q, i, j, axes.get(key))
        axes[key] = axis
        if free:
            for signs in SIGNS:
                g, grad = piece_gap_gradient(q, i, j, axis, signs)
                coeff = {}
                for k_dim in range(3):
                    coeff[stride * i + k_dim] = -grad[k_dim]
                    coeff[stride * j + k_dim] = -grad[3 + k_dim]
                row(coeff, g - clearance)
        else:
            owner, k_ax, direction = axis
            phi = float(q[owner, 2]) + k_ax * math.pi / 2.0
            a = direction * np.array([math.cos(phi), math.sin(phi)])
            phi += 0.0 if direction == 1 else math.pi
            other = j if owner == i else i
            r = float(q[other, 2]) - phi
            h_j = 0.5 * (abs(math.cos(r)) + abs(math.sin(r)))
            d = q[j, :2] - q[i, :2]
            g = float(d @ a - 0.5 - h_j)
            row({stride * i: a[0], stride * i + 1: a[1], stride * j: -a[0], stride * j + 1: -a[1]}, g - clearance)

    # Container upper bound constraint
    row({nv - 1: 1.0}, upper_L)

    bounds = []
    for _ in range(n):
        bounds.extend([(-position_trust, position_trust), (-position_trust, position_trust)])
        if free:
            bounds.append((-angle_trust, angle_trust))
    bounds.extend([(0.0, None), (0.0, None), (0.0, None)])

    c_obj = np.zeros(nv)
    c_obj[nv - 1] = 1.0

    a_mat = coo_matrix((vv, (rr, cc)), shape=(len(rhs), nv)).tocsc()
    res = linprog(c_obj, A_ub=a_mat, b_ub=np.array(rhs), bounds=bounds, method="highs")
    if not res.success:
        return None, res.message

    step = np.zeros_like(q)
    for i in range(n):
        step[i, 0] = res.x[stride * i]
        step[i, 1] = res.x[stride * i + 1]
        if free:
            step[i, 2] = res.x[stride * i + 2]

    return step, "ok"


def polish_4sign_lp(
    q: np.ndarray,
    max_iters: int = 25,
    position_trust: float = 0.04,
    angle_trust: float = 0.005,
    clearance: float = 5e-9,
) -> tuple[np.ndarray, float]:
    """Iteratively refine candidate layout q using the 4-sign-piece epigraph LP."""
    if linprog is None:
        return q.copy(), extent(q)

    best_q = normalize_box(q.copy())
    best_L = extent(best_q)
    axes: dict = {}

    pos_t = position_trust
    ang_t = angle_trust

    for it in range(max_iters):
        step, status = _lp_step_4sign(
            best_q,
            upper_L=best_L,
            position_trust=pos_t,
            angle_trust=ang_t,
            clearance=clearance,
            axes=axes,
        )
        if step is None:
            pos_t *= 0.5
            ang_t *= 0.5
            if pos_t < 1e-6:
                break
            continue

        # Line search with strict positive clearance verification
        improved = False
        for alpha in (1.0, 0.75, 0.5, 0.25, 0.1):
            cand_q = best_q + alpha * step
            cand_q = normalize_box(cand_q)
            cand_L = extent(cand_q)
            if cand_L < best_L - 1e-12:
                valid, _ = sat_check(cand_q, clearance=clearance * 0.5)
                if valid:
                    best_q = cand_q
                    best_L = cand_L
                    improved = True
                    break

        if not improved:
            pos_t *= 0.5
            ang_t *= 0.5
            if pos_t < 1e-6:
                break

    return best_q, best_L


# ---------------------------------------------------------------------------
# Scratch Parametric Band & Layout Generator
# ---------------------------------------------------------------------------

def generate_scratch_band(
    L: float,
    theta: float,
    w: int,
    v_off: float = 0.0,
    u_off: float = 0.0,
    stagger: float = 0.0,
    cross_band: bool = False,
    grid_align: str = "corners",
) -> np.ndarray:
    """Generate tilted band layout of width w immersed inside container of side L."""
    tilted: list[list[float]] = []
    upright: list[list[float]] = []

    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    u_dir = np.array([cos_t, sin_t])
    v_dir = np.array([-sin_t, cos_t])

    h_w = 0.5 * (abs(cos_t) + abs(sin_t))
    ctr = np.array([L / 2.0, L / 2.0])

    def add_band(th, u_d, v_d, h_val, shift_sign=1.0):
        diag_len = S2 * L + 4.0
        n_long = int(math.ceil(diag_len)) + 6
        for col in range(w):
            v_val = (col - (w - 1) / 2.0 + v_off * shift_sign) * 1.0
            s_val = (stagger if col % 2 == 1 else 0.0) + u_off
            for row in range(-n_long // 2, n_long // 2 + 1):
                u_val = row * 1.0 + s_val
                p = ctr + u_val * u_d + v_val * v_d
                if (h_val <= p[0] <= L - h_val) and (h_val <= p[1] <= L - h_val):
                    tilted.append([float(p[0]), float(p[1]), float(th)])

    add_band(theta, u_dir, v_dir, h_w, 1.0)
    if cross_band:
        u_dir2 = np.array([-sin_t, cos_t])
        v_dir2 = np.array([-cos_t, -sin_t])
        add_band(theta + math.pi / 2.0, u_dir2, v_dir2, h_w, -1.0)

    # Fill empty triangular pockets with upright squares
    tilted_arr = np.array(tilted) if tilted else np.zeros((0, 3))
    k = int(math.ceil(L)) + 1
    xs = np.arange(k) + 0.5 if grid_align == "origin" else np.linspace(0.5, L - 0.5, k)
    ys = np.arange(k) + 0.5 if grid_align == "origin" else np.linspace(0.5, L - 0.5, k)

    for xu in xs:
        if xu < 0.5 or xu > L - 0.5:
            continue
        for yu in ys:
            if yu < 0.5 or yu > L - 0.5:
                continue
            if len(tilted_arr) == 0:
                upright.append([float(xu), float(yu), 0.0])
                continue

            # SAT test upright candidate against tilted array
            dx = tilted_arr[:, 0] - xu
            dy = tilted_arr[:, 1] - yu
            ct = np.cos(tilted_arr[:, 2])
            st = np.sin(tilted_arr[:, 2])
            h_tot = 0.5 + 0.5 * (np.abs(ct) + np.abs(st))
            c1 = np.abs(dx) < h_tot - 1e-9
            c2 = np.abs(dy) < h_tot - 1e-9
            c3 = np.abs(dx * ct + dy * st) < h_tot - 1e-9
            c4 = np.abs(-dx * st + dy * ct) < h_tot - 1e-9
            overlap = c1 & c2 & c3 & c4
            if not np.any(overlap):
                upright.append([float(xu), float(yu), 0.0])

    squares = tilted + upright
    return np.array(squares) if squares else np.zeros((0, 3))


def prune_to_n(q: np.ndarray, target_n: int) -> np.ndarray:
    """Select exactly target_n squares that minimize container extent."""
    if len(q) <= target_n:
        return q.copy()

    current = q.copy()
    center = np.mean(current[:, :2], axis=0)

    while len(current) > target_n:
        dists = np.sum((current[:, :2] - center) ** 2, axis=1)
        worst_idx = int(np.argmax(dists))
        current = np.delete(current, worst_idx, axis=0)

    return current


# ---------------------------------------------------------------------------
# Parallel Search Across Top 40 Targets
# ---------------------------------------------------------------------------

def load_incumbents(path: Path) -> dict[int, float]:
    """Load incumbent container side L for each n from merged-literal-cpp.json."""
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {int(r["n"]): float(r["L"]) for r in data.get("results", [])}


def load_baseline_target_packings(path: Path) -> dict[int, np.ndarray]:
    """Load raw candidate coordinates from baseline-targets.csv if available."""
    if not path.exists():
        return {}
    puzzles: dict[int, list[list[float]]] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n_val = int(row["n"])
            if n_val not in puzzles:
                puzzles[n_val] = []
            puzzles[n_val].append([float(row["x"]), float(row["y"]), float(row["angle"])])
    return {n: np.array(sqs) for n, sqs in puzzles.items()}


def search_single_target(
    target_n: int,
    incumbent_L: float,
    screener: GPUScreener,
    theta_grid: Sequence[float],
    batch_size: int = 128,
    existing_q: Optional[np.ndarray] = None,
) -> CandidateResult:
    """Search for target n using GPU screening followed by 4-sign LP polish."""
    best_candidate: Optional[np.ndarray] = None
    best_unpolished_L = float("inf")
    best_params: dict[str, Any] = {}

    # If an existing high-quality packing exists (e.g. incumbent n=972 or n=908), include it
    if existing_q is not None and len(existing_q) == target_n:
        valid, _ = sat_check(existing_q, clearance=-1e-9)
        if valid:
            ext = extent(existing_q)
            best_candidate = existing_q.copy()
            best_unpolished_L = ext
            best_params = {"family": "incumbent_seed", "theta": 0.785398, "w": 13}

    k = int(math.ceil(math.sqrt(target_n)))
    max_w = min(k, 4)
    widths = list(range(1, max_w + 1))
    offsets = [-0.5, -0.25, 0.0, 0.25, 0.5]
    trial_Ls = [incumbent_L, incumbent_L - 0.002, incumbent_L + 0.005, incumbent_L + 0.015]

    # Generate batch candidate pool for GPU screening
    candidate_pool: list[np.ndarray] = []
    meta_pool: list[dict[str, Any]] = []

    for L_trial in trial_Ls:
        for w in widths:
            for theta in theta_grid:
                for v_off in offsets:
                    for u_off in (-0.5, 0.0, 0.5):
                        for stagger in (0.0, 0.5):
                            for cross_band in (False, True) if target_n in (5, 10, 17, 26, 37, 50, 65) else (False,):
                                seed = generate_scratch_band(
                                    L=L_trial,
                                    theta=theta,
                                    w=w,
                                    v_off=v_off,
                                    u_off=u_off,
                                    stagger=stagger,
                                    cross_band=cross_band,
                                )
                                if len(seed) >= target_n:
                                    pruned = prune_to_n(seed, target_n)
                                    candidate_pool.append(pruned)
                                    meta_pool.append({
                                        "theta": theta,
                                        "w": w,
                                        "v_off": v_off,
                                        "u_off": u_off,
                                        "stagger": stagger,
                                        "cross_band": cross_band,
                                    })

    # Screen candidates in batches using GPU screener
    for i in range(0, len(candidate_pool), batch_size):
        chunk = candidate_pool[i : i + batch_size]
        chunk_meta = meta_pool[i : i + batch_size]
        ls, gaps = screener.screen_batch(chunk, clearance=-1e-9)
        for idx, (l_val, gap_val) in enumerate(zip(ls, gaps)):
            if gap_val >= -1e-8 and l_val < best_unpolished_L:
                best_unpolished_L = float(l_val)
                best_candidate = chunk[idx].copy()
                best_params = chunk_meta[idx]

    # Fallback to pure upright grid if no candidate was found
    if best_candidate is None:
        grid_sq = []
        for r_i in range(k):
            for c_i in range(k):
                if len(grid_sq) < target_n:
                    grid_sq.append([0.5 + r_i, 0.5 + c_i, 0.0])
        best_candidate = np.array(grid_sq)
        best_unpolished_L = float(k)
        best_params = {"family": "upright_grid", "theta": 0.0, "w": 0}

    # Polish top candidate using 4-sign-piece epigraph LP
    polished_q = best_candidate.copy()
    polished_L = best_unpolished_L
    if best_unpolished_L <= incumbent_L + 0.05:
        polished_q, polished_L = polish_4sign_lp(best_candidate, max_iters=25)

    delta = polished_L - incumbent_L
    score_delta = delta / math.sqrt(target_n)
    n_tilted = int(np.sum(np.abs(polished_q[:, 2]) > 1e-4))
    n_upright = len(polished_q) - n_tilted

    layout = [
        {
            "id": i,
            "x": float(polished_q[i, 0]),
            "y": float(polished_q[i, 1]),
            "angle": float(polished_q[i, 2]),
        }
        for i in range(len(polished_q))
    ]

    return CandidateResult(
        target_n=target_n,
        incumbent_L=incumbent_L,
        unpolished_L=best_unpolished_L,
        polished_L=polished_L,
        delta_vs_incumbent=delta,
        within_1e3=delta <= 1e-3,
        family=best_params.get("family", "cross_band" if best_params.get("cross_band") else "tilted_band"),
        theta=float(best_params.get("theta", 0.0)),
        w=int(best_params.get("w", 0)),
        v_off=float(best_params.get("v_off", 0.0)),
        u_off=float(best_params.get("u_off", 0.0)),
        stagger=float(best_params.get("stagger", 0.0)),
        n_tilted=n_tilted,
        n_upright=n_upright,
        layout=layout,
        score_delta=score_delta,
    )


def run_pipeline(
    targets: Sequence[int] = TOP_40_TARGETS,
    output_path: Optional[Path] = None,
    batch_size: int = 128,
    teardown_after: bool = False,
    device: Optional[str] = None,
) -> dict[str, Any]:
    """Execute complete A100 search pipeline across specified targets."""
    t0 = time.time()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_file = output_path or (DATA_DIR / "colab_a100_candidates.json")

    screener = GPUScreener(device=device)
    print(f"[A100 PIPELINE] Initializing Screener on device: {screener.device} (CUDA: {screener.is_cuda})")
    print(f"[A100 PIPELINE] Evaluating {len(targets)} targets with batch size {batch_size}")

    incumbents = load_incumbents(MERGED_LITERAL_PATH)
    baseline_packings = load_baseline_target_packings(BASELINE_TARGETS_CSV)

    # Key rational angles + continuous grid
    key_thetas = [
        math.pi / 4.0,           # 45 deg
        math.atan(0.5),          # 26.565 deg
        math.atan(2.0),          # 63.435 deg
        math.atan(1.0 / 3.0),    # 18.435 deg
        math.atan(3.0),          # 71.565 deg
        math.atan(2.0 / 3.0),    # 33.690 deg
        math.atan(1.5),          # 56.310 deg
        math.atan(3.0 / 4.0),    # 36.870 deg
        math.atan(4.0 / 3.0),    # 53.130 deg
        0.4240, 0.7013, 0.6435,  # Heuristic angles
    ]
    uniform_thetas = np.linspace(0.20, math.pi / 2.0 - 0.20, 16).tolist()
    theta_grid = sorted(set(round(th, 6) for th in key_thetas + uniform_thetas))

    results: dict[str, Any] = {}
    wins_or_matches = 0

    for idx, n in enumerate(targets):
        inc_L = incumbents.get(n, float(math.ceil(math.sqrt(n))))
        existing = baseline_packings.get(n)

        res = search_single_target(
            target_n=n,
            incumbent_L=inc_L,
            screener=screener,
            theta_grid=theta_grid,
            batch_size=batch_size,
            existing_q=existing,
        )
        if res.delta_vs_incumbent <= 1e-6:
            wins_or_matches += 1

        results[str(n)] = asdict(res)
        print(
            f"[{idx + 1:2d}/{len(targets):2d}] n={n:3d} | Inc: {inc_L:9.6f} | "
            f"Unpolished: {res.unpolished_L:9.6f} | Polished: {res.polished_L:9.6f} | "
            f"Delta: {res.delta_vs_incumbent:+10.6f} | Family: {res.family}"
        )

    duration = time.time() - t0
    payload = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "device": str(screener.device),
            "is_cuda": screener.is_cuda,
            "duration_seconds": duration,
            "total_targets": len(targets),
            "wins_or_matches": wins_or_matches,
        },
        "candidates": results,
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[A100 PIPELINE] Saved {len(results)} candidate results to {out_file}")

    if teardown_after:
        teardown_colab_runtime()

    return payload


# ---------------------------------------------------------------------------
# Automated Colab Runtime Teardown
# ---------------------------------------------------------------------------

def teardown_colab_runtime(dry_run: bool = False, force_vm_shutdown: bool = False) -> None:
    """Automated teardown to immediately terminate Colab instance and eliminate costs."""
    print("\n=======================================================")
    print("AUTOMATED TEARDOWN / SELF-TERMINATION ROUTINE INITIATED")
    print("=======================================================")

    if dry_run:
        print("[TEARDOWN] Dry run enabled; self-termination bypassed.")
        return

    # 1. Attempt Colab native runtime unassign
    try:
        from google.colab import runtime  # type: ignore[import-not-found]
        print("[TEARDOWN] Disconnecting Google Colab runtime via runtime.unassign()...")
        runtime.unassign()
        print("[TEARDOWN] Colab runtime unassign request sent successfully.")
        return
    except ImportError:
        print("[TEARDOWN] google.colab.runtime not found (not running in Colab).")
    except Exception as e:
        print(f"[TEARDOWN] Colab unassign raised: {e}")

    # 2. Dedicated Cloud VM fallback
    if force_vm_shutdown and sys.platform.startswith("linux"):
        try:
            print("[TEARDOWN] Issuing system shutdown (sudo shutdown -h now)...")
            os.system("sudo shutdown -h now")
        except Exception as e:
            print(f"[TEARDOWN] System shutdown command failed: {e}")


# ---------------------------------------------------------------------------
# Local Verification Script (Official 50-Digit Certifier)
# ---------------------------------------------------------------------------

def verify_candidates(
    candidates_file: Path | str,
    output_report: Optional[Path | str] = None,
    dilation: float = 1.0 + 5e-9,
) -> dict[str, Any]:
    """Ingest exported candidate solutions and certify them against the official 50-digit metric."""
    cand_path = Path(candidates_file)
    if not cand_path.exists():
        raise FileNotFoundError(f"Candidate file not found: {cand_path}")

    print("\n=======================================================")
    print("OFFICIAL 50-DIGIT CERTIFICATION VERIFIER")
    print(f"Candidate source: {cand_path}")
    print(f"Safety dilation:  {dilation:.12e}")
    print("=======================================================\n")

    incumbents_meta = {}
    if MERGED_LITERAL_PATH.exists():
        with open(MERGED_LITERAL_PATH, "r", encoding="utf-8") as f:
            incumbents_meta = {int(r["n"]): r for r in json.load(f).get("results", [])}

    # Ingest candidates (from JSON or CSV)
    raw_puzzles: dict[int, list[dict[str, float]]] = {}

    if cand_path.suffix.lower() == ".json":
        with open(cand_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        candidates_dict = data.get("candidates", data)
        for key, val in candidates_dict.items():
            try:
                n_val = int(key)
            except ValueError:
                continue
            if isinstance(val, dict) and "layout" in val:
                raw_puzzles[n_val] = val["layout"]
            elif isinstance(val, list):
                raw_puzzles[n_val] = val
    else:
        # Submission CSV format: id,n,square_id,x,y,angle
        with open(cand_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                n_val = int(row["n"])
                if n_val not in raw_puzzles:
                    raw_puzzles[n_val] = []
                raw_puzzles[n_val].append({
                    "id": int(row["square_id"]),
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                    "angle": float(row["angle"]),
                })

    verification_records: dict[str, Any] = {}
    verified_count = 0
    wins_count = 0

    print(f"{'n':4s} | {'Incumbent L (50-digit prefix)':30s} | {'Certified L (50-digit prefix)':30s} | {'Delta L':12s} | {'Status'}")
    print("-" * 115)

    for n_val in sorted(raw_puzzles.keys()):
        sqs = raw_puzzles[n_val]
        pts = np.array([[s["x"], s["y"], s["angle"]] for s in sqs], dtype=np.float64)

        # Apply safety dilation and shift to positive quadrant
        pts = normalize_box(pts, margin=1e-8)
        pts[:, :2] *= dilation

        rows: list[dict[str, str]] = []
        for i, (x_val, y_val, a_val) in enumerate(pts):
            rows.append({
                "square_id": str(i),
                "x": format(x_val, ".17g"),
                "y": format(y_val, ".17g"),
                "angle": format(a_val, ".17g"),
            })

        inc_info = incumbents_meta.get(n_val, {})
        inc_L_str = inc_info.get("L", "")
        inc_L_float = float(inc_L_str) if inc_L_str else float("nan")

        cert_str = ""
        reason = "certifier_missing"
        status = "UNVERIFIED"
        delta_L = float("nan")
        delta_score = float("nan")

        if certify is not None and evaluate is not None:
            _, _, cert_side, eval_reason = evaluate(n_val, rows)
            if cert_side is not None and eval_reason == "ok":
                cert_str = cert_side
                reason = "ok"
                verified_count += 1
                cert_float = float(cert_str)
                if not math.isnan(inc_L_float):
                    delta_L = cert_float - inc_L_float
                    delta_score = delta_L / math.sqrt(n_val)
                    if delta_L < -1e-9:
                        status = "STRICT_WIN"
                        wins_count += 1
                    elif abs(delta_L) <= 1e-9:
                        status = "MATCH"
                    else:
                        status = "VALID_LOSE"
                else:
                    status = "VALID"
            else:
                reason = eval_reason
                status = f"REJECTED: {eval_reason}"
        else:
            # Fallback extent computation when mpmath/certify is absent
            cert_float = extent(pts)
            cert_str = f"{cert_float:.50f}"
            status = "NUMERICAL_ONLY"

        verification_records[str(n_val)] = {
            "n": n_val,
            "status": status,
            "reason": reason,
            "certified_L": cert_str,
            "incumbent_L": inc_L_str,
            "delta_L": delta_L,
            "delta_score": delta_score,
        }

        inc_disp = (inc_L_str[:28] + "..") if len(inc_L_str) > 30 else inc_L_str
        cert_disp = (cert_str[:28] + "..") if len(cert_str) > 30 else cert_str
        print(f"{n_val:4d} | {inc_disp:30s} | {cert_disp:30s} | {delta_L:+12.6e} | {status}")

    print("-" * 115)
    print(f"Summary: Verified {verified_count}/{len(raw_puzzles)} candidates. Strict wins: {wins_count}")

    report_payload = {
        "verified_count": verified_count,
        "total_evaluated": len(raw_puzzles),
        "wins_count": wins_count,
        "records": verification_records,
    }

    if output_report is not None:
        rep_path = Path(output_report)
        with open(rep_path, "w", encoding="utf-8") as f:
            json.dump(report_payload, f, indent=2)
        print(f"Saved certification report to {rep_path}")

    return report_payload


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Colab A100 heavy search pipeline & certifier")
    parser.add_argument("--search", action="store_true", help="Run parallel GPU search pipeline")
    parser.add_argument("--targets", nargs="+", type=int, default=list(TOP_40_TARGETS), help="Target n values")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for GPU screening")
    parser.add_argument("--output", type=str, default=None, help="Output file for candidates or report")
    parser.add_argument("--teardown", action="store_true", help="Trigger automated runtime teardown after search")
    parser.add_argument("--verify", type=str, default=None, help="Verify candidates file using 50-digit certifier")
    parser.add_argument("--dilation", type=float, default=1.0 + 5e-9, help="Safety dilation for certification")

    args = parser.parse_args()

    if args.verify is not None:
        verify_candidates(
            candidates_file=args.verify,
            output_report=args.output,
            dilation=args.dilation,
        )
    elif args.search:
        out_p = Path(args.output) if args.output else None
        run_pipeline(
            targets=args.targets,
            output_path=out_p,
            batch_size=args.batch_size,
            teardown_after=args.teardown,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
