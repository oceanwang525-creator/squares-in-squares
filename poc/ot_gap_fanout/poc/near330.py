#!/usr/bin/env python3
"""Targeted continuation of the n=330 optimal-transport near miss."""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linprog, minimize

ROOT = Path(__file__).resolve().parents[3]
BASE_PROBE = ROOT / "poc/ot_multidonor_barycenter/poc/probe.py"
DOCS = ROOT / "poc/ot_gap_fanout/docs"
FINAL = ROOT / "poc/ot_gap_fanout/final.csv"
N = 330


def load_base():
    spec = importlib.util.spec_from_file_location("near330_base", BASE_PROBE)
    if spec is None or spec.loader is None:
        raise ImportError(BASE_PROBE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def axis_sides(q: np.ndarray, base) -> tuple[float, float]:
    e = base.extents(q[:, 2])
    width = float((q[:, 0] + e).max() - (q[:, 0] - e).min())
    height = float((q[:, 1] + e).max() - (q[:, 1] - e).min())
    return width, height


def scale_axis(
    q: np.ndarray,
    base,
    *,
    target_side: float,
    axes: tuple[int, ...],
) -> np.ndarray:
    """Contract selected center axes about the occupied bounding-box midpoint."""
    e = base.extents(q[:, 2])
    out = q.copy()
    current = axis_sides(q, base)
    for axis in axes:
        lo = float((q[:, axis] - e).min())
        hi = float((q[:, axis] + e).max())
        ratio = min(1.0, float(target_side) / current[axis])
        midpoint = 0.5 * (lo + hi)
        out[:, axis] = midpoint + ratio * (q[:, axis] - midpoint)
    return out


def reproduce_seed(
    base,
    groups,
    *,
    epsilon: float,
    schedule: tuple[tuple[int, float], ...],
    delta: float,
    alpha: float,
    iterations: int,
):
    current = groups[N]
    incumbent = base.side(current)
    maps = {
        offset: base.transport_map(current, groups[N + offset], epsilon)
        for offset, _ in schedule
    }
    total = sum(weight for _, weight in schedule)
    mixed = np.zeros((N, 2))
    cos_sum = np.zeros(N)
    sin_sum = np.zeros(N)
    for offset, weight in schedule:
        mapped_xy, mapped_angles, _ = maps[offset]
        w = weight / total
        mixed += w * mapped_xy
        cos_sum += w * np.cos(2 * mapped_angles)
        sin_sum += w * np.sin(2 * mapped_angles)
    mixed_angles = 0.5 * np.arctan2(sin_sum, cos_sum)
    target = incumbent - delta
    q0 = base.place_centered(current, mixed, mixed_angles, target, alpha)
    q, penalty, violation = base.repair(q0, target, iterations)
    centers, dilation = base.polish.minimal_feasible_dilation(
        q[:, :2], q[:, 2], tau=1e-10
    )
    return (
        np.column_stack([centers, q[:, 2]]),
        {
            "incumbent": incumbent,
            "target_side": target,
            "initial_penalty": float(penalty),
            "initial_violation": float(violation),
            "dilation": float(dilation),
            "dilated_side": float(base.polish.exact_bbox_side(centers, q[:, 2])),
        },
    )


def tighten(base, q: np.ndarray):
    """Reach the previous five-SLP endpoint, then test zero-margin refinements."""
    stages = []
    centers = base.polish.shift_positive(q[:, :2], q[:, 2])
    theta = q[:, 2].copy()

    for margin in (1e-9, 1e-10, 1e-11, 1e-12, 0.0):
        centers, side = base.polish.fixed_angle_lp_polish(
            centers,
            theta,
            rounds=32,
            time_limit=120.0,
            margin=margin,
        )
        violation = base.overlap.max_violation(
            np.column_stack([centers, theta]), side
        )
        stages.append(
            {
                "kind": "fixed-LP",
                "margin": margin,
                "side": float(side),
                "violation": float(max(violation[0], violation[1])),
                "width": axis_sides(np.column_stack([centers, theta]), base)[0],
                "height": axis_sides(np.column_stack([centers, theta]), base)[1],
            }
        )

    for margin_set, rounds, tr_th, tr_xy in (
        ((1e-10, 2e-11, 5e-12), 24, 5e-4, 3e-3),
        ((1e-11, 1e-12, 0.0), 32, 2e-4, 1e-3),
        ((1e-12, 1e-13, 0.0), 40, 5e-5, 3e-4),
        ((0.0,), 48, 1e-5, 5e-5),
        ((0.0,), 64, 5e-6, 2e-5),
        ((0.0,), 80, 2e-6, 1e-5),
        ((0.0,), 96, 5e-7, 3e-6),
    ):
        centers, theta, side = base.polish.fast_joint_xytheta_slp(
            centers,
            theta,
            rounds=rounds,
            tr_th0=tr_th,
            tr_xy0=tr_xy,
            slp_margin=min(margin_set),
            lp_margins=margin_set,
        )
        q_out = np.column_stack([centers, theta])
        violation = base.overlap.max_violation(q_out, side)
        stages.append(
            {
                "kind": "joint-SLP",
                "margins": list(margin_set),
                "side": float(side),
                "violation": float(max(violation[0], violation[1])),
                "width": axis_sides(q_out, base)[0],
                "height": axis_sides(q_out, base)[1],
            }
        )
    return centers, theta, stages


def recover_after_contraction(
    base,
    q: np.ndarray,
    incumbent: float,
    target: float,
    iterations: int,
):
    """Re-jam a tiny contraction without admitting the original overlap."""
    def objective(flat: np.ndarray):
        candidate = flat.reshape(-1, 3)
        return base.overlap.penalty(
            candidate, target, grad=True, margin=1e-12
        )

    result = minimize(
        objective,
        q.ravel(),
        method="L-BFGS-B",
        jac=True,
        options={
            "maxiter": iterations,
            "maxls": 100,
            "maxfun": iterations * 20,
            "ftol": 0.0,
            "gtol": 1e-16,
        },
    )
    repaired = result.x.reshape(-1, 3)
    violation_pair = base.overlap.max_violation(repaired, target)
    penalty = float(result.fun)
    violation = float(max(violation_pair[0], violation_pair[1]))
    info = {
        "repair_penalty": float(penalty),
        "repair_violation": float(violation),
    }
    if violation > 1e-8:
        return None, info
    centers, dilation = base.polish.minimal_feasible_dilation(
        repaired[:, :2], repaired[:, 2], tau=1e-10
    )
    theta = repaired[:, 2].copy()
    info["dilation"] = float(dilation)
    info["dilated_side"] = float(base.polish.exact_bbox_side(centers, theta))
    centers, side = base.polish.fixed_angle_lp_polish(
        centers, theta, rounds=32, time_limit=120.0, margin=0.0
    )
    info["lp_side"] = float(side)
    centers, theta, side = base.polish.fast_joint_xytheta_slp(
        centers,
        theta,
        rounds=48,
        tr_th0=2e-4,
        tr_xy0=1e-3,
        slp_margin=0.0,
        lp_margins=(1e-10, 1e-12, 0.0),
    )
    candidate = np.column_stack([centers, theta])
    overlap = base.overlap.max_violation(candidate, side)
    info["final_side"] = float(side)
    info["final_violation"] = float(max(overlap[0], overlap[1]))
    if side < incumbent:
        try:
            cert = base.certify_candidate(candidate, N, incumbent)
        except Exception as exc:
            info["official_error"] = str(exc)
            cert = None
        if cert:
            info["certified"] = {
                "literal_side": cert["literal_side"],
                "pandas_side": cert["pandas_side"],
            }
        return cert, info
    return None, info


def resolve_at_target(
    base,
    q: np.ndarray,
    target: float,
    rounds: int = 40,
):
    """Exact-Jacobian feasibility pass that may start from a tiny overlap."""
    centers = base.polish.shift_positive(q[:, :2], q[:, 2])
    theta = q[:, 2].copy()
    current_linear2 = q.copy()
    current_linear2[:, :2] = centers
    current_overlap = base.overlap.max_violation(current_linear2, target)
    current_pen = max(current_overlap[0], current_overlap[1])
    best_q = current_linear2.copy()
    best_pen = float(current_pen)
    tr_xy, tr_th = 3e-3, 2e-4
    accepted = 0

    for _ in range(rounds):
        A, b = base.polish._build_joint_slp_vectorized(
            centers[:, 0],
            centers[:, 1],
            theta,
            tr_xy=tr_xy,
            tr_th=tr_th,
            margin=0.0,
        )
        bounds = (
            [(-tr_xy, tr_xy)] * (2 * len(theta))
            + [(-tr_th, tr_th)] * len(theta)
            + [(target, target)]
        )
        result = linprog(
            np.zeros(3 * len(theta) + 1),
            A_ub=A,
            b_ub=b,
            bounds=bounds,
            method="highs",
            options={
                "dual_feasibility_tolerance": 1e-9,
                "primal_feasibility_tolerance": 1e-9,
                "time_limit": 20.0,
            },
        )
        if not result.success or result.x is None:
            tr_xy *= 0.5
            tr_th *= 0.5
            if tr_xy < 1e-7:
                break
            continue
        dx = result.x[: len(theta)]
        dy = result.x[len(theta) : 2 * len(theta)]
        dth = result.x[2 * len(theta) : 3 * len(theta)]
        improved = False
        for factor in (1.0, 0.5, 0.25):
            candidate = np.column_stack([
                centers[:, 0] + factor * dx,
                centers[:, 1] + factor * dy,
                theta + factor * dth,
            ])
            candidate[:, :2] = base.polish.shift_positive(
                candidate[:, :2], candidate[:, 2]
            )
            ov = base.overlap.max_violation(candidate, target)
            pen = float(max(ov[0], ov[1]))
            if pen < current_pen - 1e-14:
                centers = candidate[:, :2].copy()
                theta = candidate[:, 2].copy()
                current_pen = pen
                if pen < best_pen:
                    best_q = candidate.copy()
                    best_pen = pen
                improved = True
                accepted += 1
                break
        if current_pen <= 1e-11:
            break
        if improved:
            tr_xy = max(1e-6, tr_xy * 0.9)
            tr_th = max(1e-7, tr_th * 0.9)
        else:
            tr_xy *= 0.5
            tr_th *= 0.5
            if tr_xy < 1e-7:
                break
    return best_q, {
        "accepted_steps": accepted,
        "max_violation": float(best_pen),
    }


def contraction_candidates(
    base,
    centers: np.ndarray,
    theta: np.ndarray,
    incumbent: float,
    recovery_iterations: int,
):
    q0 = np.column_stack([centers, theta])
    candidates = []
    for reduction in (1e-9, 1.25e-9, 1.5e-9, 2e-9):
        target = incumbent - reduction
        certified, info = recover_after_contraction(
            base,
            q0,
            incumbent,
            target,
            recovery_iterations,
        )
        candidates.append(
            {
                "reduction": reduction,
                "axes": ["direct-repair"],
                "side": info.get("final_side"),
                "max_violation": info.get("final_violation"),
                "official_attempted": True,
                "recovery": info,
                **({"certified": certified} if certified else {}),
            }
        )
    reductions = (
        8.5e-10,
        1e-9,
        1.25e-9,
        1.5e-9,
        2e-9,
        3e-9,
        5e-9,
        8e-9,
        1e-8,
    )
    for reduction in reductions:
        target = incumbent - reduction
        for axes in ((0,), (1,), (0, 1)):
            q = scale_axis(q0, base, target_side=target, axes=axes)
            q[:, :2] = base.polish.shift_positive(q[:, :2], q[:, 2])
            side = base.side(q)
            overlap = base.overlap.max_violation(q, target)
            item = {
                "reduction": reduction,
                "axes": list(axes),
                "side": float(side),
                "max_violation": float(max(overlap[0], overlap[1])),
                "official_attempted": False,
                "q": q,
            }
            if side < incumbent:
                item["official_attempted"] = True
                try:
                    certified = base.certify_candidate(q, N, incumbent)
                except Exception as exc:
                    item["official_error"] = str(exc)
                    certified = None
                if certified:
                    item["certified"] = certified
            if reduction <= 2e-9:
                candidates.append(item)
            if (
                not item.get("certified")
                and axes == (0, 1)
                and reduction <= 3e-9
            ):
                resolved, resolve_info = resolve_at_target(
                    base, q, target, rounds=40
                )
                item["resolve"] = resolve_info
                if max(
                    base.overlap.max_violation(resolved, target)[:2]
                ) <= 1e-11 and base.side(resolved) < incumbent:
                    try:
                        certified_resolved = base.certify_candidate(
                            resolved, N, incumbent
                        )
                    except Exception as exc:
                        item["resolve_official_error"] = str(exc)
                        certified_resolved = None
                    if certified_resolved:
                        item["certified"] = certified_resolved
                        if item not in candidates:
                            candidates.append(item)
                recovered, recovery = recover_after_contraction(
                    base,
                    q,
                    incumbent,
                    target,
                    recovery_iterations,
                )
                item["recovery"] = recovery
                if recovered:
                    item["certified"] = recovered
                    if item not in candidates:
                        candidates.append(item)
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--schedule", default="-1:2,1:1")
    parser.add_argument("--epsilon", type=float, default=0.018)
    parser.add_argument("--delta", type=float, default=1e-6)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--out", default="")
    parser.add_argument("--skip-contractions", action="store_true")
    args = parser.parse_args()
    started = time.monotonic()
    base = load_base()
    groups = base.load_groups()
    schedule = tuple(
        (int(part.split(":")[0]), float(part.split(":")[1]))
        for part in args.schedule.split(",")
    )
    q, seed = reproduce_seed(
        base,
        groups,
        epsilon=args.epsilon,
        schedule=schedule,
        delta=args.delta,
        alpha=args.alpha,
        iterations=args.iterations,
    )
    centers, theta, stages = tighten(base, q)
    resolved_cert = None
    final_q = np.column_stack([centers, theta])
    final_side = stages[-1]["side"]
    if final_side <= seed["incumbent"] - 1e-9:
        try:
            resolved_cert = base.certify_candidate(
                final_q, N, seed["incumbent"]
            )
        except Exception as exc:
            stages.append({"kind": "final-official-error", "error": str(exc)})
    contenders = [] if args.skip_contractions else contraction_candidates(
        base, centers, theta, seed["incumbent"], args.iterations
    )
    certified = resolved_cert or next(
        (item["certified"] for item in contenders if item.get("certified")), None
    )
    payload = {
        "n": N,
        "seed": {
            "epsilon": args.epsilon,
            "schedule": [list(item) for item in schedule],
            "delta": args.delta,
            "alpha": args.alpha,
            **seed,
        },
        "stages": stages,
        "tightened_side": stages[-1]["side"],
        "contractions": [
            {key: value for key, value in item.items() if key != "q"}
            for item in contenders
        ],
        "certified": (
            {
                "literal_side": certified["literal_side"],
                "pandas_side": certified["pandas_side"],
                "final_text": certified["final_text"],
            }
            if certified
            else None
        ),
        "elapsed_seconds": time.monotonic() - started,
    }
    DOCS.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else DOCS / "near330.json"
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if certified:
        parsed = base.union.parse_text(certified["final_text"])
        existing = base.union.parse_text(FINAL.read_text(encoding="utf-8"))
        existing[N] = parsed[N]
        FINAL.write_text(base.union.final_text(existing), encoding="utf-8")
        (DOCS / "near330-win.txt").write_text(
            certified["final_text"], encoding="utf-8"
        )
    print(json.dumps({
        "certified": payload["certified"] and {
            "literal_side": payload["certified"]["literal_side"],
            "pandas_side": payload["certified"]["pandas_side"],
        },
        "best_final_side": min(
            [item["side"] for item in contenders] or [float("inf")]
        ),
        "elapsed_seconds": payload["elapsed_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
