"""Colab A100 High-N Sweep Runner with Streaming and Verified Teardown.

Parallelizes the full high-n sweep across n in [600, 1000] using:
1. Batch screening on CUDA/GPU or CPU (vectorized PyTorch/NumPy SAT evaluator).
2. Parallel multi-core 4-sign-piece epigraph LP relaxation across CPU cores.
3. Hybrid generation combining Pell-17 core+border sweeping with vacancy deletion
   and staircase band defect relaxation.
4. Immediate append-only streaming to JSONL and CSV on disk (with fsync) so zero
   progress is lost on disconnect.
5. Explicit verified teardown calling google.colab.runtime.unassign().
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import io
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# PyTorch acceleration (graceful CPU fallback if not installed)
try:
    import torch
except ImportError:
    torch = None

try:
    from scipy.optimize import linprog
    from scipy.spatial import cKDTree
    from scipy.sparse import coo_matrix
except ImportError:
    linprog = None
    cKDTree = None
    coo_matrix = None

# Repo paths and baseline certification discovery
REPO_ROOT = Path(__file__).resolve().parents[3]
BASELINE_SRC = REPO_ROOT / "poc/baseline/poc/src"
if str(BASELINE_SRC) not in sys.path and BASELINE_SRC.exists():
    sys.path.insert(0, str(BASELINE_SRC))

try:
    from squares.core import certify, evaluate
except ImportError:
    certify = None
    evaluate = None

BORDER_SCAN_DIR = REPO_ROOT / "poc/pell_corrugation/data/border-scan"
MERGED_LITERAL_PATH = BORDER_SCAN_DIR / "merged-literal-cpp.json"
DEFAULT_STREAM_JSONL = REPO_ROOT / "outputs/high_n_stream.jsonl"
DEFAULT_STREAM_CSV = REPO_ROOT / "outputs/high_n_submission.csv"

SIGNS: tuple[tuple[float, float], ...] = (
    (-1.0, -1.0),
    (-1.0, 1.0),
    (1.0, -1.0),
    (1.0, 1.0),
)
S2 = math.sqrt(2.0)
INV_S2 = 1.0 / S2


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class CandidatePacking:
    n: int
    q: np.ndarray
    family: str
    unpolished_L: float
    meta: dict[str, Any]


@dataclass
class EvalResult:
    n: int
    family: str
    incumbent_L: float
    unpolished_L: float
    polished_L: float
    delta_L: float
    score_gain: float
    is_improvement: bool
    is_match: bool
    sat_ok: bool
    min_gap: float
    q: np.ndarray
    meta: dict[str, Any]
    timestamp: str
    runtime_sec: float = 0.0


# ---------------------------------------------------------------------------
# Geometry & Classical SAT
# ---------------------------------------------------------------------------

def extent(q: np.ndarray) -> float:
    """Compute container side L for packing q."""
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
    if len(q) == 0:
        return q.copy()
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
    """Exhaustive all-pairs SAT separation check."""
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
        cand = np.where(dist2 < 8.01)[0]
        if len(cand) == 0:
            continue
        cand_j = i + 1 + cand
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
# GPU-Accelerated Batch SAT Screening
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
        """Screen batch of candidates of identical square count N."""
        if len(batch_candidates) == 0:
            return np.array([]), np.array([])

        if torch is None or self.device is None:
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

        # 1. Extents
        c = torch.abs(torch.cos(q_tensor[:, :, 2]))
        s = torch.abs(torch.sin(q_tensor[:, :, 2]))
        r = 0.5 * (c + s)
        x_min = torch.min(q_tensor[:, :, 0] - r, dim=1).values
        x_max = torch.max(q_tensor[:, :, 0] + r, dim=1).values
        y_min = torch.min(q_tensor[:, :, 1] - r, dim=1).values
        y_max = torch.max(q_tensor[:, :, 1] + r, dim=1).values
        L_batch = torch.maximum(x_max - x_min, y_max - y_min)

        # 2. Pairwise SAT
        idx_i, idx_j = torch.triu_indices(N, N, offset=1, device=self.device)
        P = idx_i.shape[0]

        q_i = q_tensor[:, idx_i, :]  # [B, P, 3]
        q_j = q_tensor[:, idx_j, :]  # [B, P, 3]

        dx = q_j[:, :, 0] - q_i[:, :, 0]  # [B, P]
        dy = q_j[:, :, 1] - q_i[:, :, 1]
        dist2 = dx * dx + dy * dy

        th_i = q_i[:, :, 2]
        th_j = q_j[:, :, 2]
        axes = torch.stack([th_i, th_i + 0.5 * math.pi, th_j, th_j + 0.5 * math.pi], dim=-1)

        cos_ax = torch.cos(axes)
        sin_ax = torch.sin(axes)
        proj = torch.abs(dx.unsqueeze(-1) * cos_ax + dy.unsqueeze(-1) * sin_ax)

        h_i = 0.5 * (
            torch.abs(torch.cos(th_i.unsqueeze(-1) - axes))
            + torch.abs(torch.sin(th_i.unsqueeze(-1) - axes))
        )
        h_j = 0.5 * (
            torch.abs(torch.cos(th_j.unsqueeze(-1) - axes))
            + torch.abs(torch.sin(th_j.unsqueeze(-1) - axes))
        )

        pair_gaps = proj - (h_i + h_j)
        sat_gaps = torch.max(pair_gaps, dim=-1).values
        sat_gaps = torch.where(dist2 < 8.01, sat_gaps, torch.tensor(1.0, device=self.device))
        min_gaps = torch.min(sat_gaps, dim=-1).values

        return L_batch.cpu().numpy(), min_gaps.cpu().numpy()


# ---------------------------------------------------------------------------
# 4-Sign-Piece Epigraph LP Relaxation
# ---------------------------------------------------------------------------

def chosen_axis(q: np.ndarray, i: int, j: int, previous=None, hysteresis: float = 1e-9):
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
    free: bool = True,
) -> tuple[Optional[np.ndarray], str]:
    if linprog is None or cKDTree is None or coo_matrix is None:
        return None, "scipy_missing"

    n = len(q)
    stride = 3 if free else 2
    nv = stride * n + 3
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
    """Iteratively refine candidate layout q using 4-sign-piece epigraph LP."""
    if linprog is None:
        return q.copy(), extent(q)

    best_q = normalize_box(q.copy())
    best_L = extent(best_q)
    axes: dict = {}

    pos_t = position_trust
    ang_t = angle_trust

    for _ in range(max_iters):
        step, _ = _lp_step_4sign(
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

        improved = False
        for alpha in (1.0, 0.75, 0.5, 0.25, 0.1):
            cand_q = normalize_box(best_q + alpha * step)
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
# Core + Border and Vacancy Deletion Generators
# ---------------------------------------------------------------------------

def build_corrugated_band(
    w: int,
    l: int,
    a1: int,
    a2: int,
    b1: int,
    b2: int,
    delta: float = 0.002935,
) -> tuple[np.ndarray, float]:
    """Synthetic out-of-phase corrugated staircase-band core."""
    L0 = 0.5 * (a1 + a2) + 1.0 + w * INV_S2
    q: list[list[float]] = []
    gap_se_ne = max(0.0, (L0 - b2) - a1 - 0.05)
    gap_se_sw = max(0.0, (L0 - b1) - a1 - 0.05)

    for i in range(a1):
        for j in range(a1 - i):
            x, y = L0 - 0.5 - i, 0.5 + j * (1.0 + 1e-6)
            if i == a1 - 1 and j == 0 and gap_se_sw > 0:
                x -= min(0.19, 0.45 * gap_se_sw)
            if i == 0 and j == a1 - 1 and gap_se_ne > 0:
                x -= min(0.20, 0.45 * gap_se_ne)
            q.append([x, y, 0.0])

    for i in range(a2):
        for j in range(a2 - i):
            x, y = 0.5 + i * (1.0 + 1e-6), L0 - 0.5 - j
            q.append([x, y, 0.0])

    for i in range(b1):
        for j in range(b1 - i):
            q.append([0.5 + i * (1.0 + 1e-6), 0.5 + j, 0.0])

    for i in range(b2):
        for j in range(b2 - i):
            q.append([L0 - 0.5 - i * (1.0 + 1e-6), L0 - 0.5 - j, 0.0])

    u_teeth_se = np.array([(L0 + a1 - 1.0 - 2.0 * i) * INV_S2 for i in range(a1)])
    u_teeth_nw = np.array([(L0 - a2 + 1.0 + 2.0 * j) * INV_S2 for j in range(a2)])

    v_start = -0.5 * (w - 1.0) - 0.5 * (a1 - a2) * INV_S2
    u_lo = (b1 + 1.0) * INV_S2 + 0.5
    u_hi = S2 * L0 - (b2 + 1.0) * INV_S2 - 0.5
    u_cols = np.linspace(u_lo, u_hi, l)

    du_se_arr = np.array([uc - u_teeth_se[np.argmin(np.abs(uc - u_teeth_se))] for uc in u_cols])
    du_nw_arr = np.array([uc - u_teeth_nw[np.argmin(np.abs(uc - u_teeth_nw))] for uc in u_cols])

    pinch_c: list[int] = []
    pinch_dth: list[float] = []
    for c in range(l):
        if max(abs(du_se_arr[c]), abs(du_nw_arr[c])) < 0.50:
            pinch_c.append(c)
            pinch_dth.append(-abs(delta) if du_se_arr[c] > 0 else +abs(delta))

    if len(pinch_c) > 0:
        dth_cols = np.interp(np.arange(l), pinch_c, pinch_dth)
    else:
        dth_cols = np.zeros(l)

    for c in range(l):
        uc = u_cols[c]
        du_se = du_se_arr[c]
        du_nw = du_nw_arr[c]
        dth = dth_cols[c]

        v_shift = np.zeros(w)
        if c == 0 and a1 > a2:
            v_shift[:] = -0.15
        elif c == l - 1 and a1 > a2:
            v_shift[:] = -0.17
        elif abs(dth) < 0.5 * abs(delta):
            if abs(du_se) >= 0.50:
                v_shift[:] = +0.85 * (abs(du_se) - 0.50)
            elif abs(du_nw) >= 0.50:
                v_shift[:] = -0.85 * (abs(du_nw) - 0.50)
        else:
            if abs(du_nw) >= 0.50:
                v_shift[0] = -0.05 * (abs(du_nw) - 0.50)
            elif abs(du_se) >= 0.50:
                v_shift[-1] = +0.05 * (abs(du_se) - 0.50)

        for s in range(w):
            v = v_start + s * 1.0 + v_shift[s]
            u = uc + (s - 0.5 * (w - 1)) * math.sin(dth)
            x = (u + v) * INV_S2
            y = (u - v) * INV_S2
            q.append([x, y, math.pi / 4.0 + dth])

    return np.array(q, dtype=float), L0


def wrap_l_border(
    q_core: np.ndarray,
    t: int,
    k_total: int,
    margin: float = 1e-7,
) -> np.ndarray:
    """Wrap normalized core packing with t-layer upright L-border on North and East."""
    if t <= 0:
        return normalize_box(q_core, margin=0.0)

    q_norm = normalize_box(q_core, margin=0.0)
    L_core = extent(q_norm)
    L_total = L_core + t + 2.0 * margin
    k_core = k_total - t
    border: list[list[float]] = []

    # East border columns
    for i in range(t):
        for j in range(k_total):
            border.append([L_total - 0.5 - i * (1.0 + 1e-8), 0.5 + j * (1.0 + 1e-8), 0.0])

    # North border rows
    for i in range(k_core):
        for j in range(t):
            border.append([0.5 + i * (1.0 + 1e-8), L_total - 0.5 - j * (1.0 + 1e-8), 0.0])

    return np.vstack([q_norm, np.array(border, dtype=float)])


def apply_vacancy_deletion(
    q: np.ndarray,
    target_n: int,
    collapse_shift: float = 0.20,
) -> np.ndarray:
    """Selectively delete surplus squares and propagate slack into vacancies."""
    if len(q) <= target_n:
        return q.copy()

    k_del = len(q) - target_n
    current = q.copy()

    # Prioritize boundary/perimeter squares (farthest from center or flank extremities)
    ctr = np.mean(current[:, :2], axis=0)
    dists = np.sum((current[:, :2] - ctr) ** 2, axis=1)

    # Candidate deletion indices from the exterior boundary
    del_indices = set(np.argsort(dists)[-k_del:])

    # Apply defect propagation: shift remaining squares along adjacent columns toward vacancy
    remaining_indices = [idx for idx in range(len(current)) if idx not in del_indices]
    pruned = current[remaining_indices].copy()

    # Slight inward relaxation shift toward center to facilitate LP contraction
    vec_to_center = ctr - pruned[:, :2]
    norm_vec = np.linalg.norm(vec_to_center, axis=1, keepdims=True) + 1e-8
    pruned[:, :2] += (vec_to_center / norm_vec) * (collapse_shift * 0.05)

    return pruned


def build_staircase_band_45(
    w: int,
    l: int,
    a1: int,
    a2: int,
    b1: int,
    b2: int,
    drop_band: int = 0,
) -> tuple[np.ndarray, float]:
    """Exact wall-constrained 45-degree staircase-band packing."""
    L0 = 0.5 * (a1 + a2) + 1.0 + w * INV_S2
    q: list[list[float]] = []

    for i in range(a1):
        for j in range(a1 - i):
            if i == a1 - 1 and j == 0 and a1 + b1 > L0 + 1e-9:
                continue
            if i == 0 and j == a1 - 1 and a1 + b2 > L0 + 1e-9:
                continue
            q.append([L0 - 0.5 - i, 0.5 + j, 0.0])

    for i in range(a2):
        for j in range(a2 - i):
            if i == 0 and j == a2 - 1 and a2 + b1 > L0 + 1e-9:
                continue
            if i == a2 - 1 and j == 0 and a2 + b2 > L0 + 1e-9:
                continue
            q.append([0.5 + i, L0 - 0.5 - j, 0.0])

    for i in range(b1):
        for j in range(b1 - i):
            q.append([0.5 + i, 0.5 + j, 0.0])

    for i in range(b2):
        for j in range(b2 - i):
            q.append([L0 - 0.5 - i, L0 - 0.5 - j, 0.0])

    v_start = -0.5 * (w - 1.0) - 0.5 * (a1 - a2) * INV_S2
    u_wall = 0.5 * (w - 1.0) + 0.5 * abs(a1 - a2) * INV_S2 + INV_S2
    u_lo = max((b1 + 1.0) * INV_S2 + 0.5, u_wall)
    u_hi = S2 * L0 - max((b2 + 1.0) * INV_S2 + 0.5, u_wall)
    u_cols = np.linspace(u_lo, u_hi, l) if l > 1 else np.array([0.5 * (u_lo + u_hi)])

    band: list[list[float]] = []
    for c in range(l):
        uc = u_cols[c]
        for s in range(w):
            v = v_start + s * 1.0
            x = (uc + v) * INV_S2
            y = (uc - v) * INV_S2
            band.append([x, y, math.pi / 4.0])

    if drop_band > 0 and len(band) >= drop_band:
        band = band[:-drop_band]

    q.extend(band)
    return np.array(q, dtype=float), L0


def generate_scratch_band(
    L: float,
    theta: float,
    w: int,
    v_off: float = 0.0,
    u_off: float = 0.0,
    stagger: float = 0.0,
) -> np.ndarray:
    """Generate continuous-angle tilted band with corner pocket filling."""
    tilted: list[list[float]] = []
    upright: list[list[float]] = []

    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    u_dir = np.array([cos_t, sin_t])
    v_dir = np.array([-sin_t, cos_t])

    h_w = 0.5 * (abs(cos_t) + abs(sin_t))
    ctr = np.array([L / 2.0, L / 2.0])

    diag_len = S2 * L + 4.0
    n_long = int(math.ceil(diag_len)) + 6
    for col in range(w):
        v_val = (col - (w - 1) / 2.0 + v_off) * 1.0
        s_val = (stagger if col % 2 == 1 else 0.0) + u_off
        for row in range(-n_long // 2, n_long // 2 + 1):
            u_val = row * 1.0 + s_val
            p = ctr + u_val * u_dir + v_val * v_dir
            if (h_w <= p[0] <= L - h_w) and (h_w <= p[1] <= L - h_w):
                tilted.append([float(p[0]), float(p[1]), float(theta)])

    tilted_arr = np.array(tilted) if tilted else np.zeros((0, 3))
    k = int(math.ceil(L)) + 1
    coords = np.linspace(0.5, L - 0.5, k)

    for xu in coords:
        if xu < 0.5 or xu > L - 0.5:
            continue
        for yu in coords:
            if yu < 0.5 or yu > L - 0.5:
                continue
            if len(tilted_arr) == 0:
                upright.append([float(xu), float(yu), 0.0])
                continue

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


# ---------------------------------------------------------------------------
# Core Cache Loader
# ---------------------------------------------------------------------------

def load_or_generate_cores(widths: Sequence[int] = (6, 8, 9, 10, 12, 13, 15, 16)) -> dict[int, np.ndarray]:
    """Load cached Pell-17 cores from disk or synthesize them dynamically."""
    cores: dict[int, np.ndarray] = {}
    for w in widths:
        npz_path = BORDER_SCAN_DIR / f"core-w{w}.npz"
        if npz_path.exists():
            try:
                data = np.load(npz_path)
                cores[w] = np.asarray(data["q"], dtype=float)
                continue
            except Exception:
                pass

        # Dynamic fallback generation
        b = int(math.floor(w * INV_S2))
        a1, a2 = b + 9, b + 8
        l = w + 12
        q_init, _ = build_corrugated_band(w, l, a1, a2, b, b, delta=0.002935)
        cores[w] = q_init

    return cores


def load_incumbents(path: Path = MERGED_LITERAL_PATH) -> dict[int, float]:
    """Load incumbent container side L for each n."""
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {int(r["n"]): float(r["L"]) for r in data.get("results", [])}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Candidate Construction Strategy for n in [600, 1000]
# ---------------------------------------------------------------------------

def generate_candidates_for_n(
    target_n: int,
    cores: dict[int, np.ndarray],
    incumbent_L: float,
) -> list[CandidatePacking]:
    """Generate diverse candidate packings combining core+border and vacancy deletion."""
    candidates: list[CandidatePacking] = []
    k_tot = int(math.ceil(math.sqrt(target_n)))

    # 1. Core + Border Sweep with Vacancy Deletion across widths
    for w, q_core in cores.items():
        n_core = len(q_core)
        # Check border layer thickness t in 0..5
        for t in range(0, 6):
            n_wrapped = n_core + 2 * t * k_tot - t * t
            if n_wrapped >= target_n and (n_wrapped - target_n) <= 30:
                q_wrap = wrap_l_border(q_core, t=t, k_total=k_tot)
                q_del = apply_vacancy_deletion(q_wrap, target_n)
                if len(q_del) == target_n:
                    candidates.append(
                        CandidatePacking(
                            n=target_n,
                            q=q_del,
                            family=f"pell17_w{w}_t{t}_del{n_wrapped - target_n}",
                            unpolished_L=extent(q_del),
                            meta={"w": w, "t": t, "surplus_deleted": n_wrapped - target_n},
                        )
                    )

    # 2. Staircase Band 45-degree parameterized construction
    for w in (1, 2, 3, 4, 8, 12, 13):
        b = int(math.floor(w * INV_S2))
        a1 = k_tot - b - 1
        a2 = a1 - 1
        if a1 > 0 and a2 > 0:
            l = w + 12
            q_sc, L0 = build_staircase_band_45(w, l, a1, a2, b, b)
            if len(q_sc) >= target_n and len(q_sc) - target_n <= 35:
                q_del = apply_vacancy_deletion(q_sc, target_n)
                if len(q_del) == target_n:
                    candidates.append(
                        CandidatePacking(
                            n=target_n,
                            q=q_del,
                            family=f"staircase45_w{w}_del{len(q_sc) - target_n}",
                            unpolished_L=extent(q_del),
                            meta={"w": w, "L0": L0},
                        )
                    )

    # 3. Scratch Band at candidate thetas
    for th in (math.pi / 4.0, math.atan(2.0 / 3.0), math.atan(3.0 / 4.0)):
        for w in (2, 3, 4):
            q_scratch = generate_scratch_band(L=incumbent_L, theta=th, w=w)
            if len(q_scratch) >= target_n:
                q_pruned = apply_vacancy_deletion(q_scratch, target_n)
                if len(q_pruned) == target_n:
                    candidates.append(
                        CandidatePacking(
                            n=target_n,
                            q=q_pruned,
                            family=f"scratch_th{th:.3f}_w{w}",
                            unpolished_L=extent(q_pruned),
                            meta={"theta": th, "w": w},
                        )
                    )

    return candidates


# ---------------------------------------------------------------------------
# Streaming Logger (Append-Only JSONL and CSV)
# ---------------------------------------------------------------------------

class CandidateStreamer:
    """Manages append-only streaming to JSONL and CSV with verified fsync."""

    def __init__(self, jsonl_path: Path, csv_path: Path):
        self.jsonl_path = Path(jsonl_path)
        self.csv_path = Path(csv_path)

        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)

        self.jsonl_file = open(self.jsonl_path, "a", encoding="utf-8")

        # Initialize CSV header if file doesn't exist or is empty
        csv_exists = self.csv_path.exists() and self.csv_path.stat().st_size > 0
        self.csv_file = open(self.csv_path, "a", encoding="utf-8", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        if not csv_exists:
            self.csv_writer.writerow(["id", "n", "square_id", "x", "y", "angle"])
            self.csv_file.flush()
            os.fsync(self.csv_file.fileno())

        self.global_id = 0

    def record_candidate(self, res: EvalResult) -> None:
        """Stream an evaluated candidate immediately to disk."""
        # 1. JSONL log
        layout_dicts = [
            {"square_id": i, "x": float(x), "y": float(y), "angle": float(a)}
            for i, (x, y, a) in enumerate(res.q)
        ]

        payload = {
            "n": res.n,
            "family": res.family,
            "incumbent_L": res.incumbent_L,
            "unpolished_L": res.unpolished_L,
            "polished_L": res.polished_L,
            "delta_L": res.delta_L,
            "score_gain": res.score_gain,
            "is_improvement": res.is_improvement,
            "is_match": res.is_match,
            "sat_ok": res.sat_ok,
            "min_gap": res.min_gap,
            "runtime_sec": res.runtime_sec,
            "meta": res.meta,
            "timestamp": res.timestamp,
            "layout": layout_dicts,
        }

        self.jsonl_file.write(json.dumps(payload) + "\n")
        self.jsonl_file.flush()
        os.fsync(self.jsonl_file.fileno())

        # 2. CSV submission format for improvement/match candidates
        if res.is_improvement or res.is_match or res.delta_L < 0.05:
            for i, (x, y, a) in enumerate(res.q):
                self.csv_writer.writerow([
                    self.global_id,
                    res.n,
                    i,
                    format(x, ".17g"),
                    format(y, ".17g"),
                    format(a, ".17g"),
                ])
                self.global_id += 1
            self.csv_file.flush()
            os.fsync(self.csv_file.fileno())

    def close(self) -> None:
        try:
            self.jsonl_file.close()
            self.csv_file.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Worker Task for Parallel Relaxation
# ---------------------------------------------------------------------------

def _relax_candidate_worker(
    target_n: int,
    family: str,
    q_cand: np.ndarray,
    incumbent_L: float,
    meta: dict[str, Any],
    lp_iters: int = 25,
) -> EvalResult:
    """Worker task running 4-sign-piece LP relaxation on a CPU core."""
    t0 = time.perf_counter()
    unpolished_L = extent(q_cand)
    polished_q, polished_L = polish_4sign_lp(q_cand, max_iters=lp_iters)
    sat_ok, min_gap = sat_check(polished_q)
    runtime_sec = time.perf_counter() - t0

    delta_L = polished_L - incumbent_L
    score_gain = -delta_L / math.sqrt(target_n)
    is_improvement = (delta_L < -1e-9) and sat_ok
    is_match = (abs(delta_L) <= 1e-9) and sat_ok

    ts = datetime.now(timezone.utc).isoformat()

    return EvalResult(
        n=target_n,
        family=family,
        incumbent_L=incumbent_L,
        unpolished_L=unpolished_L,
        polished_L=polished_L,
        delta_L=delta_L,
        score_gain=score_gain,
        is_improvement=is_improvement,
        is_match=is_match,
        sat_ok=sat_ok,
        min_gap=min_gap,
        q=polished_q,
        meta=meta,
        timestamp=ts,
        runtime_sec=round(runtime_sec, 3),
    )


# ---------------------------------------------------------------------------
# Colab Runtime Teardown
# ---------------------------------------------------------------------------

def teardown_colab_runtime(dry_run: bool = False, force: bool = False) -> None:
    """Explicit teardown calling google.colab.runtime.unassign() with execution guards."""
    print("=" * 70)
    print("[TEARDOWN] Initiating verified Colab runtime self-termination guard...")

    if dry_run:
        print("[TEARDOWN] Dry run enabled: self-termination bypassed.")
        print("=" * 70)
        return

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

    if force and sys.platform.startswith("linux"):
        try:
            print("[TEARDOWN] Issuing cloud instance shutdown (sudo shutdown -h now)...")
            os.system("sudo shutdown -h now")
        except Exception as e:
            print(f"[TEARDOWN] Shutdown failed: {e}")

    print("=" * 70)


# ---------------------------------------------------------------------------
# High-Precision Certification Tool
# ---------------------------------------------------------------------------

def verify_candidates(
    candidates_file: str | Path,
    output_report: Optional[str | Path] = None,
    dilation: float = 1.0 + 5e-9,
) -> dict[str, Any]:
    """Ingest streamed candidates and verify using official 50-digit certifier."""
    cand_path = Path(candidates_file)
    if not cand_path.exists():
        raise FileNotFoundError(f"Candidate file not found: {cand_path}")

    incumbents = load_incumbents()
    raw_puzzles: dict[int, list[dict[str, float]]] = {}

    if cand_path.suffix.lower() == ".jsonl":
        with open(cand_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                n_val = int(row["n"])
                # Keep candidate with smallest polished_L
                if n_val not in raw_puzzles or row.get("polished_L", float("inf")) < extent(
                    np.array([[s["x"], s["y"], s["angle"]] for s in raw_puzzles[n_val]])
                ):
                    raw_puzzles[n_val] = row["layout"]
    elif cand_path.suffix.lower() == ".json":
        with open(cand_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for key, val in data.items():
            try:
                n_val = int(key)
                if isinstance(val, dict) and "layout" in val:
                    raw_puzzles[n_val] = val["layout"]
                elif isinstance(val, list):
                    raw_puzzles[n_val] = val
            except ValueError:
                pass
    else:
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

    print(f"{'n':4s} | {'Incumbent L':20s} | {'Certified L':20s} | {'Delta L':12s} | {'Status'}")
    print("-" * 80)

    for n_val in sorted(raw_puzzles.keys()):
        sqs = raw_puzzles[n_val]
        pts = np.array([[s["x"], s["y"], s["angle"]] for s in sqs], dtype=np.float64)
        pts = normalize_box(pts, margin=1e-8)
        pts[:, :2] *= dilation

        rows: list[dict[str, str]] = [
            {
                "square_id": str(i),
                "x": format(x_val, ".17g"),
                "y": format(y_val, ".17g"),
                "angle": format(a_val, ".17g"),
            }
            for i, (x_val, y_val, a_val) in enumerate(pts)
        ]

        inc_L = incumbents.get(n_val, float("nan"))
        cert_str = ""
        status = "UNVERIFIED"
        delta_L = float("nan")

        if certify is not None and evaluate is not None:
            _, _, cert_side, eval_reason = evaluate(n_val, rows)
            if cert_side is not None and eval_reason == "ok":
                cert_str = cert_side
                verified_count += 1
                cert_float = float(cert_str)
                if not math.isnan(inc_L):
                    delta_L = cert_float - inc_L
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
                status = f"REJECTED: {eval_reason}"
        else:
            cert_float = extent(pts)
            cert_str = f"{cert_float:.17g}"
            status = "NUMERICAL_ONLY"

        verification_records[str(n_val)] = {
            "n": n_val,
            "status": status,
            "certified_L": cert_str,
            "incumbent_L": inc_L,
            "delta_L": delta_L,
        }

        cert_display = f"{float(cert_str):20.10f}" if cert_str else f"{'N/A':>20s}"
        print(f"{n_val:4d} | {inc_L:20.10f} | {cert_display} | {delta_L:+12.6e} | {status}")

    print("-" * 80)
    print(f"Summary: Verified {verified_count}/{len(raw_puzzles)} candidates. Strict wins: {wins_count}")

    report_payload = {
        "verified_count": verified_count,
        "total_evaluated": len(raw_puzzles),
        "wins_count": wins_count,
        "records": verification_records,
    }

    if output_report is not None:
        with open(output_report, "w", encoding="utf-8") as f:
            json.dump(report_payload, f, indent=2)
        print(f"Saved certification report to {output_report}")

    return report_payload


# ---------------------------------------------------------------------------
# Full Pipeline Execution
# ---------------------------------------------------------------------------

def run_high_n_sweep(
    n_min: int = 600,
    n_max: int = 1000,
    targets: Optional[Sequence[int]] = None,
    workers: Optional[int] = None,
    batch_size: int = 128,
    lp_iters: int = 25,
    stream_jsonl: Path = DEFAULT_STREAM_JSONL,
    stream_csv: Path = DEFAULT_STREAM_CSV,
    teardown_after: bool = False,
    dry_run_teardown: bool = False,
) -> None:
    """Execute high-n sweep across n in [600, 1000] with parallel relaxation and streaming."""
    target_list = sorted(targets) if targets is not None else list(range(n_min, n_max + 1))
    cpu_cores = workers if workers is not None else max(1, os.cpu_count() or 1)

    print("=" * 80)
    print(f"Starting High-N Colab Sweep across {len(target_list)} targets: [{target_list[0]}..{target_list[-1]}]")
    print(f"Configuration: CPU Workers = {cpu_cores}, Batch Size = {batch_size}, LP Iters = {lp_iters}")
    print(f"Streaming JSONL Log: {stream_jsonl}")
    print(f"Streaming CSV Output: {stream_csv}")
    print("=" * 80)

    # 1. Setup GPU Screener
    screener = GPUScreener()
    device_name = "CUDA" if screener.is_cuda else "CPU"
    print(f"Batch Screener Active Device: {device_name}")

    # 2. Load Incumbents and Cores
    incumbents = load_incumbents()
    cores = load_or_generate_cores()
    print(f"Loaded {len(cores)} base Pell-17 cores (widths: {sorted(cores.keys())}).")

    streamer = CandidateStreamer(stream_jsonl, stream_csv)
    total_screened = 0
    total_improved = 0
    total_matched = 0

    # 3. Parallel Pool for Multi-Core LP Relaxation
    with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_cores) as executor:
        for n_val in target_list:
            inc_L = incumbents.get(n_val, math.ceil(math.sqrt(n_val)))
            raw_cands = generate_candidates_for_n(n_val, cores, inc_L)
            if not raw_cands:
                continue

            # Batch screening on GPU
            cand_arrays = [c.q for c in raw_cands]
            L_screen, gap_screen = screener.screen_batch(cand_arrays, clearance=-1e-4)
            total_screened += len(cand_arrays)

            # Filter candidates: prioritize small unpolished L and near-valid gap
            valid_or_close = np.where(gap_screen >= -1e-4)[0]
            if len(valid_or_close) == 0:
                best_idx = int(np.argmin(L_screen))
                selected_indices = [best_idx]
            else:
                sorted_valid = valid_or_close[np.argsort(L_screen[valid_or_close])]
                selected_indices = sorted_valid[:3].tolist()

            # Submit top candidate(s) to parallel multi-core relaxation pool
            futures = [
                executor.submit(
                    _relax_candidate_worker,
                    n_val,
                    raw_cands[idx].family,
                    raw_cands[idx].q,
                    inc_L,
                    raw_cands[idx].meta,
                    lp_iters,
                )
                for idx in selected_indices
            ]

            for fut in concurrent.futures.as_completed(futures):
                try:
                    result = fut.result()
                    streamer.record_candidate(result)

                    flag = ""
                    if result.is_improvement:
                        total_improved += 1
                        flag = " [IMPROVEMENT]"
                    elif result.is_match:
                        total_matched += 1
                        flag = " [MATCH]"

                    print(
                        f"[n={result.n:4d}] {result.family:30s} | "
                        f"Incumbent: {result.incumbent_L:9.5f} | "
                        f"Polished: {result.polished_L:9.5f} | "
                        f"Delta: {result.delta_L:+9.5f}{flag} | "
                        f"Runtime: {result.runtime_sec:5.2f}s"
                    )
                except Exception as ex:
                    print(f"[ERROR] Target n={n_val} relaxation failed: {ex}")

    streamer.close()
    print("=" * 80)
    print(f"Sweep Completed: Screened {total_screened} candidates. Improvements: {total_improved}, Matches: {total_matched}")
    print(f"All progress streamed to {stream_jsonl} and {stream_csv}")
    print("=" * 80)

    if teardown_after:
        teardown_colab_runtime(dry_run=dry_run_teardown)


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Colab A100 High-N Sweep Runner with Streaming & Teardown")
    parser.add_argument("--n-min", type=int, default=600, help="Minimum n in sweep")
    parser.add_argument("--n-max", type=int, default=1000, help="Maximum n in sweep")
    parser.add_argument("--targets", nargs="+", type=int, default=None, help="Explicit target n list")
    parser.add_argument("--workers", type=int, default=None, help="Number of CPU worker processes")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for GPU screening")
    parser.add_argument("--lp-iters", type=int, default=25, help="Max iterations for 4-sign LP polish")
    parser.add_argument("--stream-jsonl", type=str, default=str(DEFAULT_STREAM_JSONL), help="Stream output JSONL")
    parser.add_argument("--stream-csv", type=str, default=str(DEFAULT_STREAM_CSV), help="Stream output CSV")
    parser.add_argument("--teardown", action="store_true", help="Trigger verified Colab teardown on completion")
    parser.add_argument("--dry-run-teardown", action="store_true", help="Bypass actual unassign call")
    parser.add_argument("--verify", type=str, default=None, help="Verify candidates file using 50-digit certifier")
    parser.add_argument("--dilation", type=float, default=1.0 + 5e-9, help="Safety dilation for certification")

    args = parser.parse_args()

    if args.verify is not None:
        verify_candidates(
            candidates_file=args.verify,
            output_report=args.stream_jsonl if args.stream_jsonl != str(DEFAULT_STREAM_JSONL) else None,
            dilation=args.dilation,
        )
    else:
        run_high_n_sweep(
            n_min=args.n_min,
            n_max=args.n_max,
            targets=args.targets,
            workers=args.workers,
            batch_size=args.batch_size,
            lp_iters=args.lp_iters,
            stream_jsonl=Path(args.stream_jsonl),
            stream_csv=Path(args.stream_csv),
            teardown_after=args.teardown,
            dry_run_teardown=args.dry_run_teardown,
        )


if __name__ == "__main__":
    main()
