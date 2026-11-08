"""Vacancy Deletion POC 3: Defect propagation on public n=972 incumbent.

Relaxes the jammed (w=13, t=4) Pell-17 public record at n=972 (L = 31.692388155428)
by deleting m in {1, 2} squares from the 233-square tilted band, creating internal
vacancies that propagate along column contacts to target n=971 and n=970 with an active
container shrink LP polisher.
"""
from __future__ import annotations

import argparse
import csv
import io
from itertools import product
import json
import math
import sys
import time
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parents[1] / "data"
DOCS = Path(__file__).resolve().parents[1] / "docs"
REPO = Path(__file__).resolve().parents[4] if len(Path(__file__).resolve().parents) > 4 else Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "poc/baseline/poc/src"))
from squares import _official_metric as metric
from squares.core import certify

S2 = math.sqrt(2.0)
INV_S2 = 1.0 / S2
SIGNS = tuple(product((-1.0, 1.0), repeat=2))

INCUMBENT_972_L = "31.69238815542838555961566082107349804168366869287818"
PUBLIC_971_L = "31.67877987534328000000000000000000000000000000000000"
PORTFOLIO_971_L = "31.70710678118654800000000000000000000000000000000000"
PORTFOLIO_972_L = "31.70710678118654800000000000000000000000000000000000"


@dataclass
class PolishResult:
    q: np.ndarray
    L: float
    iterations: int = 0
    accepted: int = 0


def extent(q: np.ndarray) -> float:
    c = np.abs(np.cos(q[:, 2]))
    s = np.abs(np.sin(q[:, 2]))
    h = 0.5 * (c + s)
    lo = np.min(q[:, :2] - h[:, None], axis=0)
    hi = np.max(q[:, :2] + h[:, None], axis=0)
    return float(np.max(hi - lo))


def normalize_box(q: np.ndarray, margin: float = 0.0) -> np.ndarray:
    q = q.copy()
    h = 0.5 * (np.abs(np.cos(q[:, 2])) + np.abs(np.sin(q[:, 2])))
    lo = np.min(q[:, :2] - h[:, None], axis=0)
    q[:, :2] -= lo - margin
    return q


def sat_check(q: np.ndarray, clearance: float = 0.0) -> tuple[bool, float]:
    q = np.asarray(q, dtype=np.longdouble)
    c, s = np.cos(q[:, 2]), np.sin(q[:, 2])
    u = np.column_stack((c, s))
    v = np.column_stack((-s, c))
    minimum = np.longdouble(np.inf)
    for i in range(len(q) - 1):
        d = q[i + 1 :, :2] - q[i, :2]
        best = np.full(len(d), -np.inf, dtype=np.longdouble)
        for owner_axes in (False, True):
            for basis in (u, v):
                a = basis[i + 1 :] if owner_axes else np.broadcast_to(basis[i], d.shape)
                hi = 0.5 * (np.abs(a @ u[i]) + np.abs(a @ v[i]))
                hj = 0.5 * (np.abs(np.sum(a * u[i + 1 :], axis=1)) + np.abs(np.sum(a * v[i + 1 :], axis=1)))
                g = np.abs(np.sum(d * a, axis=1)) - hi - hj
                best = np.maximum(best, g)
        m = float(np.min(best))
        if m < minimum:
            minimum = m
        if m < clearance:
            return False, float(minimum)
    return True, float(minimum)


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


def _lp(
    q: np.ndarray,
    upper: float,
    position_trust: float,
    angle_trust: float,
    clearance: float,
    axes: dict,
) -> tuple[np.ndarray | None, str]:
    from scipy.sparse import coo_matrix

    n = len(q)
    free = angle_trust > 0.0
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

    radius = S2 + 2.0 * S2 * position_trust + 1e-10
    pairs = sorted(cKDTree(q[:, :2]).query_pairs(radius))
    for i, j in pairs:
        key = (i, j)
        axis = chosen_axis(q, i, j, axes.get(key))
        axes[key] = axis
        if free:
            for signs in SIGNS:
                g, grad = piece_gap_gradient(q, i, j, axis, signs)
                coeff = {}
                for k in range(3):
                    coeff[stride * i + k] = -grad[k]
                    coeff[stride * j + k] = -grad[3 + k]
                row(coeff, g - clearance)
        else:
            owner, k, direction = axis
            phi = float(q[owner, 2]) + k * math.pi / 2.0
            a = direction * np.array([math.cos(phi), math.sin(phi)])
            h = sum(
                0.5 * (abs(math.cos(float(q[t, 2]) - phi)) + abs(math.sin(float(q[t, 2]) - phi)))
                for t in (i, j)
            )
            row(
                {stride * i: a[0], stride * i + 1: a[1], stride * j: -a[0], stride * j + 1: -a[1]},
                float((q[j, :2] - q[i, :2]) @ a - h - clearance),
            )

    bounds = []
    for _ in range(n):
        bounds.extend([(-position_trust, position_trust)] * 2)
        if free:
            bounds.append((-angle_trust, angle_trust))
    bounds.extend([(None, None), (None, None), (1.0, upper)])
    objective = np.zeros(nv)
    objective[-1] = 1.0
    A = coo_matrix((vv, (rr, cc)), shape=(len(rhs), nv)).tocsr()
    res = linprog(
        objective,
        A_ub=A,
        b_ub=rhs,
        bounds=bounds,
        method="highs",
        options={"primal_feasibility_tolerance": 1e-9, "dual_feasibility_tolerance": 1e-9, "time_limit": 5.0},
    )
    if not res.success or not np.isfinite(res.x).all():
        return None, str(res.message)
    out = q.copy()
    out[:, :2] += res.x[: stride * n].reshape(n, stride)[:, :2]
    if free:
        out[:, 2] += res.x[: stride * n].reshape(n, stride)[:, 2]
    return out, str(res.message)


def polish_packing(
    q: np.ndarray,
    L_init: float,
    iterations: int = 25,
    position_trust: float = 0.05,
    angle_trust: float = 0.005,
    clearance: float = 5e-9,
) -> PolishResult:
    """Polishes candidate packing with active container shrink LP."""
    q_norm = normalize_box(q, margin=clearance)
    axes: dict[tuple[int, int], list[tuple[float, float, int]]] = {}
    repaired, _ = _lp(q_norm, L_init + 0.05, position_trust, 0.0, clearance, axes)
    if repaired is None:
        repaired = q_norm
    best = PolishResult(repaired.copy(), extent(repaired))

    for it in range(iterations):
        old_L = best.L
        # Step 1: Fixed-angle translation step
        fixed, _ = _lp(best.q, old_L + 1e-4, position_trust, 0.0, clearance, axes)
        if fixed is not None and sat_check(fixed)[0] and extent(fixed) < best.L:
            best.q = fixed
            best.L = extent(fixed)
            best.accepted += 1

        # Step 2: Joint position and angle perturbation proposal
        if angle_trust > 0:
            proposal, _ = _lp(best.q, best.L + 1e-4, position_trust, angle_trust, clearance, axes)
            if proposal is not None:
                origin = best.q.copy()
                for bt in range(8):
                    alpha = 2.0 ** (-bt)
                    seed = origin + alpha * (proposal - origin)
                    rep, _ = _lp(seed, best.L + 1e-3, position_trust, 0.0, clearance, {})
                    if rep is not None and sat_check(rep)[0] and extent(rep) < best.L:
                        best.q = rep
                        best.L = extent(rep)
                        best.accepted += 1
                        break

        best.iterations += 1
        if old_L - best.L < 1e-10 and it >= 5:
            break

    return best


def load_incumbent_972(path: Path | str | None = None) -> np.ndarray:
    """Loads the certified 972-square public incumbent from baseline targets."""
    if path is None:
        path = REPO / "poc/boundary_zipper/data/baseline-targets.csv"
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Incumbent file not found at {path}")

    squares = []
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["n"]) == 972:
                squares.append((int(row["square_id"]), float(row["x"]), float(row["y"]), float(row["angle"])))

    squares.sort(key=lambda x: x[0])
    if len(squares) != 972:
        raise ValueError(f"Expected 972 squares in incumbent, found {len(squares)}")

    q = np.array([[x, y, a] for _, x, y, a in squares], dtype=float)
    return q


def analyze_incumbent_structure(q: np.ndarray) -> dict:
    """Decomposes n=972 packing into upright components and 13 tilted columns."""
    n = len(q)
    # Tilted squares have |angle - pi/4| < 0.2 or |sin(2*angle)| > 0.5
    is_tilted = np.abs(np.sin(2.0 * q[:, 2])) > 0.5
    tilted_indices = np.where(is_tilted)[0]
    upright_indices = np.where(~is_tilted)[0]

    # Coordinate projections along diagonal band coordinate frame
    # In n=972, transverse coordinate across the 13 columns is u = (x+y)/sqrt(2),
    # and longitudinal coordinate along the column is v = (-x+y)/sqrt(2)
    u_all = (q[:, 0] + q[:, 1]) * INV_S2
    v_all = (-q[:, 0] + q[:, 1]) * INV_S2

    u_tilted = u_all[tilted_indices]
    v_tilted = v_all[tilted_indices]

    # Cluster tilted squares into columns using 1D transverse coordinate u
    sorted_order = np.argsort(u_tilted)
    sorted_indices = tilted_indices[sorted_order]
    sorted_u = u_tilted[sorted_order]

    columns: list[list[int]] = []
    current_col = [sorted_indices[0]]
    for idx, u_val in zip(sorted_indices[1:], sorted_u[1:]):
        if abs(u_val - u_all[current_col[-1]]) > 0.40:
            columns.append(current_col)
            current_col = [idx]
        else:
            current_col.append(idx)
    columns.append(current_col)

    # In each column, sort squares longitudinally by v
    for col in columns:
        col.sort(key=lambda i: v_all[i])

    column_lengths = [len(c) for c in columns]

    return dict(
        upright_count=len(upright_indices),
        tilted_count=len(tilted_indices),
        upright_indices=upright_indices,
        tilted_indices=tilted_indices,
        columns=columns,
        column_lengths=column_lengths,
        u=u_all,
        v=v_all,
    )


def apply_longitudinal_defect_shift(
    q: np.ndarray,
    column: list[int],
    deleted_u: float,
    u_all: np.ndarray,
    shift_amount: float = 0.25,
) -> np.ndarray:
    """Shifts squares along the column towards the vacancy defect to propagate slack."""
    q_out = q.copy()
    band_dir = np.array([INV_S2, INV_S2])
    for idx in column:
        u_val = u_all[idx]
        if u_val < deleted_u:
            # Shift squares below vacancy towards +u
            q_out[idx, :2] += shift_amount * band_dir
        else:
            # Shift squares above vacancy towards -u
            q_out[idx, :2] -= shift_amount * band_dir
    return q_out


def create_defect_candidate(
    q_972: np.ndarray,
    structure: dict,
    delete_count: int = 1,
    strategy: str = "end_column",
    col_idx: int = 0,
    end: str = "min",
    propagate_shift: bool = True,
) -> tuple[np.ndarray, list[int]]:
    """Creates a candidate packing for n=971 or n=970 by deleting squares and propagating defect."""
    columns = structure["columns"]
    u_all = structure["u"]
    col = columns[col_idx]

    deleted_indices: list[int] = []
    if delete_count == 1:
        del_idx = col[0] if end == "min" else col[-1]
        deleted_indices.append(del_idx)
    elif delete_count == 2:
        if strategy == "same_column_both_ends":
            deleted_indices.extend([col[0], col[-1]])
        else:
            # Delete one from each of two adjacent flank columns
            col2 = columns[col_idx + 1 if col_idx + 1 < len(columns) else col_idx - 1]
            deleted_indices.append(col[0] if end == "min" else col[-1])
            deleted_indices.append(col2[0] if end == "min" else col2[-1])

    # Propagate defect along column(s) before dropping squares
    q_mod = q_972.copy()
    if propagate_shift:
        for del_idx in deleted_indices:
            deleted_u = u_all[del_idx]
            # Identify which column this index belongs to
            for c in columns:
                if del_idx in c:
                    remaining_col = [i for i in c if i not in deleted_indices]
                    q_mod = apply_longitudinal_defect_shift(
                        q_mod, remaining_col, deleted_u, u_all, shift_amount=0.15
                    )
                    break

    # Drop the deleted squares
    keep_mask = np.ones(len(q_972), dtype=bool)
    keep_mask[deleted_indices] = False
    q_candidate = q_mod[keep_mask]

    return q_candidate, deleted_indices


def run_defect_propagation_scan(
    q_972: np.ndarray,
    target_n: int = 971,
    iterations: int = 25,
) -> tuple[PolishResult, dict]:
    """Scans candidate column end deletions on n=972 and polishes with container shrink LP."""
    structure = analyze_incumbent_structure(q_972)
    delete_count = 972 - target_n
    if delete_count not in (1, 2):
        raise ValueError(f"Target n must be 971 or 970 (delete 1 or 2 squares), got {target_n}")

    best_result: PolishResult | None = None
    best_meta: dict = {}
    candidates_tested = 0

    # Test flank columns (cols 0, 1, 11, 12) and interior columns
    cols_to_test = list(range(len(structure["columns"])))
    ends_to_test = ["min", "max"]

    for c_idx in cols_to_test:
        for end in ends_to_test:
            candidates_tested += 1
            q_cand, del_ids = create_defect_candidate(
                q_972,
                structure,
                delete_count=delete_count,
                strategy="end_column",
                col_idx=c_idx,
                end=end,
                propagate_shift=True,
            )

            # Quick repair and extent estimation
            q_norm = normalize_box(q_cand, margin=5e-9)
            init_L = extent(q_norm)

            # Run active container shrink LP
            res = polish_packing(
                q_norm,
                L_init=init_L,
                iterations=iterations,
                position_trust=0.05,
                angle_trust=0.005,
                clearance=5e-9,
            )

            if best_result is None or res.L < best_result.L:
                best_result = res
                best_meta = dict(
                    target_n=target_n,
                    deleted_count=delete_count,
                    column_index=c_idx,
                    column_length=len(structure["columns"][c_idx]),
                    end=end,
                    deleted_indices=[int(x) for x in del_ids],
                    initial_L=init_L,
                    polished_L=res.L,
                    iterations_run=res.iterations,
                    accepted_steps=res.accepted,
                )

    best_meta["candidates_tested"] = candidates_tested
    return best_result, best_meta


def rows(q: np.ndarray) -> list[dict[str, str]]:
    q = q.copy()
    q[:, :2] -= q[:, :2].mean(0)
    return [
        dict(
            square_id=str(i),
            x=format(x, ".17g"),
            y=format(y, ".17g"),
            angle=format(a, ".17g"),
        )
        for i, (x, y, a) in enumerate(q)
    ]


def csv_text(puzzles: dict[int, list[dict[str, str]]]) -> str:
    f = io.StringIO()
    w = csv.DictWriter(f, fieldnames=("id", "n", "square_id", "x", "y", "angle"))
    w.writeheader()
    idx = 0
    for n, rs in sorted(puzzles.items()):
        for r in rs:
            w.writerow(dict(id=idx, n=n, **{k: r[k] for k in ("square_id", "x", "y", "angle")}))
            idx += 1
    return f.getvalue()


def certify_transport(text: str, path: Path | None = None) -> dict[str, dict]:
    literal = metric._parse_submission_csv(text)
    transported_text = pd.read_csv(io.StringIO(text)).to_csv(index=False)
    transported = metric._parse_submission_csv(transported_text)
    if path is not None:
        path.write_text(transported_text)
    result: dict[str, dict] = {}
    for n in literal:
        entry: dict[str, dict] = {}
        for mode, puzzles in (("literal", literal), ("actual_pandas", transported)):
            try:
                entry[mode] = dict(ok=True, L=str(certify(n, puzzles[n])))
            except ValueError as e:
                entry[mode] = dict(ok=False, error=str(e))
        result[str(n)] = entry
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Vacancy deletion POC 3: Defect propagation on public n=972 incumbent.")
    ap.add_argument("--input", type=Path, default=REPO / "poc/boundary_zipper/data/baseline-targets.csv")
    ap.add_argument("--out", type=Path, default=HERE)
    ap.add_argument("--targets", nargs="+", type=int, default=[971, 970])
    ap.add_argument("--iterations", type=int, default=20)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)

    start_time = time.monotonic()
    q_972 = load_incumbent_972(args.input)
    struct = analyze_incumbent_structure(q_972)

    puzzles: dict[int, list[dict[str, str]]] = {}
    results_summary: dict = {}

    for target_n in args.targets:
        print(f"Running defect propagation on n={target_n} (deleting {972 - target_n} squares)...")
        best_res, meta = run_defect_propagation_scan(q_972, target_n=target_n, iterations=args.iterations)
        puzzles[target_n] = rows(best_res.q)
        results_summary[str(target_n)] = meta

    text = csv_text(puzzles)
    checks = certify_transport(text, args.out / "output-pandas.csv")
    (args.out / "output.csv").write_text(text)

    # Comparisons against incumbents and baselines
    report = {
        "elapsed_seconds": time.monotonic() - start_time,
        "incumbent_972_L": INCUMBENT_972_L,
        "public_971_L": PUBLIC_971_L,
        "portfolio_971_L": PORTFOLIO_971_L,
        "structure": {
            "upright_count": struct["upright_count"],
            "tilted_count": struct["tilted_count"],
            "column_lengths": struct["column_lengths"],
        },
        "targets": results_summary,
        "certificates": checks,
    }

    (args.out / "results.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
