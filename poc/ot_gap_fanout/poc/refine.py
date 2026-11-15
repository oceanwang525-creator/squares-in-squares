#!/usr/bin/env python3
"""Run LP after near-feasible transport starts, including over-budget dialations."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
BASE_PROBE = ROOT / "poc/ot_multidonor_barycenter/poc/probe.py"
DOCS = ROOT / "poc/ot_gap_fanout/docs"
FINAL = ROOT / "poc/ot_gap_fanout/final.csv"


def load_base():
    spec = importlib.util.spec_from_file_location("ot_refine_base", BASE_PROBE)
    if spec is None or spec.loader is None:
        raise ImportError(BASE_PROBE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def scan(n: int, seconds: float, iterations: int, top_starts: int) -> dict:
    base = load_base()
    groups = base.load_groups()
    current = groups[n]
    incumbent = base.side(current)
    offsets = [k for k in (-4, -3, -2, -1, 1, 2, 3, 4) if n + k in groups]
    started = time.monotonic()
    maps = {}
    for offset in offsets:
        for epsilon in (0.018, 0.025, 0.035):
            maps[(offset, epsilon)] = base.transport_map(
                current, groups[n + offset], epsilon
            )

    schedules = base.schedules(offsets)
    candidates = []
    for epsilon in (0.018, 0.025, 0.035):
        for schedule in schedules:
            if not all((offset, epsilon) in maps for offset, _ in schedule):
                continue
            total = sum(weight for _, weight in schedule)
            mixed = np.zeros((n, 2))
            cos_sum = np.zeros(n)
            sin_sum = np.zeros(n)
            for offset, weight in schedule:
                mapped_xy, mapped_angles, _ = maps[(offset, epsilon)]
                w = weight / total
                mixed += w * mapped_xy
                cos_sum += w * np.cos(2 * mapped_angles)
                sin_sum += w * np.sin(2 * mapped_angles)
            mixed_angles = 0.5 * np.arctan2(sin_sum, cos_sum)
            for delta in (1e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4):
                side_target = incumbent - delta
                for alpha in (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5):
                    q0 = base.place_centered(
                        current, mixed, mixed_angles, side_target, alpha
                    )
                    penalty = float(
                        base.overlap.penalty(q0, side_target, grad=False)
                    )
                    candidates.append(
                        (penalty, epsilon, schedule, delta, alpha, side_target, q0)
                    )
    candidates.sort(key=lambda row: row[0])

    attempts = []
    for (
        initial_penalty,
        epsilon,
        schedule,
        delta,
        alpha,
        side_target,
        q0,
    ) in candidates[:top_starts]:
        if time.monotonic() - started > seconds * 2:
            break
        q, penalty, violation = base.repair(q0, side_target, iterations)
        attempts.append(
            {
                "epsilon": epsilon,
                "schedule": [list(item) for item in schedule],
                "delta": delta,
                "alpha": alpha,
                "initial_penalty": initial_penalty,
                "final_penalty": penalty,
                "max_violation": violation,
                "_q": q,
            }
        )
    attempts.sort(key=lambda row: (row["final_penalty"], row["max_violation"]))

    certified = None
    for entry in attempts[:6]:
        q = entry["_q"]
        centers, dilation = base.polish.minimal_feasible_dilation(
            q[:, :2], q[:, 2], tau=1e-10
        )
        q_dilated = np.column_stack([centers, q[:, 2]])
        dilation_side = base.polish.exact_bbox_side(
            q_dilated[:, :2], q_dilated[:, 2]
        )
        entry["dilation"] = dilation
        entry["dilated_side"] = dilation_side

        # Unlike the parent probe, run LP even when dilation is a few
        # micrometres over budget. The LP can recover more side than the
        # dilation cost; only the final exact side is allowed to qualify.
        try:
            centers_lp, lp_side = base.polish.fixed_angle_lp_polish(
                q_dilated[:, :2],
                q_dilated[:, 2],
                rounds=24,
                time_limit=90.0,
            )
            q_lp = np.column_stack([centers_lp, q_dilated[:, 2]])
            lp_overlap, lp_wall, _ = base.overlap.max_violation(q_lp, lp_side)
            entry["fixed_angle_lp_side"] = lp_side
            entry["fixed_angle_lp_violation"] = max(lp_overlap, lp_wall)
            if max(lp_overlap, lp_wall) <= 1e-8 and lp_side < incumbent:
                certified = base.certify_candidate(q_lp, n, incumbent)
                if certified:
                    entry["selected"] = "dilated-LP"
                    break
            if max(lp_overlap, lp_wall) <= 1e-8:
                centers_slp, theta_slp, slp_side = (
                    base.polish.fast_joint_xytheta_slp(
                        centers_lp,
                        q_dilated[:, 2],
                        rounds=16,
                        tr_th0=0.006,
                        tr_xy0=0.04,
                    )
                )
                q_slp = np.column_stack([centers_slp, theta_slp])
                slp_overlap, slp_wall, _ = base.overlap.max_violation(
                    q_slp, slp_side
                )
                entry["joint_slp_side"] = slp_side
                entry["joint_slp_violation"] = max(slp_overlap, slp_wall)
                if max(slp_overlap, slp_wall) <= 1e-8 and slp_side < incumbent:
                    certified = base.certify_candidate(q_slp, n, incumbent)
                    if certified:
                        entry["selected"] = "dilated-LP-joint-SLP"
                        break
                if max(slp_overlap, slp_wall) <= 1e-8:
                    centers_slp2, theta_slp2, slp_side2 = (
                        base.polish.fast_joint_xytheta_slp(
                            centers_slp,
                            theta_slp,
                            rounds=24,
                            tr_th0=0.002,
                            tr_xy0=0.012,
                            slp_margin=5e-10,
                            lp_margins=(1e-9, 5e-10, 1e-10),
                        )
                    )
                    q_slp2 = np.column_stack([centers_slp2, theta_slp2])
                    slp2_overlap, slp2_wall, _ = base.overlap.max_violation(
                        q_slp2, slp_side2
                    )
                    entry["joint_slp2_side"] = slp_side2
                    entry["joint_slp2_violation"] = max(slp2_overlap, slp2_wall)
                    if max(slp2_overlap, slp2_wall) <= 1e-8:
                        centers_lp2, lp_side2 = base.polish.fixed_angle_lp_polish(
                            centers_slp2,
                            theta_slp2,
                            rounds=16,
                            time_limit=90.0,
                            margin=1e-10,
                        )
                        q_lp2 = np.column_stack([centers_lp2, theta_slp2])
                        lp2_overlap, lp2_wall, _ = base.overlap.max_violation(
                            q_lp2, lp_side2
                        )
                        entry["fixed_angle_lp2_side"] = lp_side2
                        entry["fixed_angle_lp2_violation"] = max(
                            lp2_overlap, lp2_wall
                        )
                        if max(lp2_overlap, lp2_wall) <= 1e-8:
                            certified = base.certify_candidate(
                                q_lp2, n, incumbent
                            )
                            if certified:
                                entry["selected"] = "LP-joint-SLP-joint-SLP-LP"
                                break
                        if not certified:
                            centers_slp3, theta_slp3, slp_side3 = (
                                base.polish.fast_joint_xytheta_slp(
                                    centers_lp2,
                                    theta_slp2,
                                    rounds=24,
                                    tr_th0=5e-4,
                                    tr_xy0=2e-3,
                                    slp_margin=1e-10,
                                    lp_margins=(5e-10, 1e-10, 5e-11),
                                )
                            )
                            q_slp3 = np.column_stack([centers_slp3, theta_slp3])
                            slp3_overlap, slp3_wall, _ = (
                                base.overlap.max_violation(q_slp3, slp_side3)
                            )
                            entry["joint_slp3_side"] = slp_side3
                            entry["joint_slp3_violation"] = max(
                                slp3_overlap, slp3_wall
                            )
                            if max(slp3_overlap, slp3_wall) <= 1e-8:
                                centers_lp3, lp_side3 = (
                                    base.polish.fixed_angle_lp_polish(
                                        centers_slp3,
                                        theta_slp3,
                                        rounds=16,
                                        time_limit=90.0,
                                        margin=5e-11,
                                    )
                                )
                                q_lp3 = np.column_stack([centers_lp3, theta_slp3])
                                lp3_overlap, lp3_wall, _ = (
                                    base.overlap.max_violation(q_lp3, lp_side3)
                                )
                                entry["fixed_angle_lp3_side"] = lp_side3
                                entry["fixed_angle_lp3_violation"] = max(
                                    lp3_overlap, lp3_wall
                                )
                                if max(lp3_overlap, lp3_wall) <= 1e-8:
                                    certified = base.certify_candidate(
                                        q_lp3, n, incumbent
                                    )
                                    if certified:
                                        entry["selected"] = "three-SLP-LP"
                                        break
                                if not certified:
                                    centers_slp4, theta_slp4, slp_side4 = (
                                        base.polish.fast_joint_xytheta_slp(
                                            centers_lp3,
                                            theta_slp3,
                                            rounds=24,
                                            tr_th0=1e-4,
                                            tr_xy0=5e-4,
                                            slp_margin=5e-11,
                                            lp_margins=(1e-10, 5e-11, 1e-11),
                                        )
                                    )
                                    q_slp4 = np.column_stack(
                                        [centers_slp4, theta_slp4]
                                    )
                                    slp4_overlap, slp4_wall, _ = (
                                        base.overlap.max_violation(
                                            q_slp4, slp_side4
                                        )
                                    )
                                    entry["joint_slp4_side"] = slp_side4
                                    entry["joint_slp4_violation"] = max(
                                        slp4_overlap, slp4_wall
                                    )
                                    if max(slp4_overlap, slp4_wall) <= 1e-8:
                                        centers_lp4, lp_side4 = (
                                            base.polish.fixed_angle_lp_polish(
                                                centers_slp4,
                                                theta_slp4,
                                                rounds=16,
                                                time_limit=90.0,
                                                margin=1e-11,
                                            )
                                        )
                                        q_lp4 = np.column_stack(
                                            [centers_lp4, theta_slp4]
                                        )
                                        lp4_overlap, lp4_wall, _ = (
                                            base.overlap.max_violation(
                                                q_lp4, lp_side4
                                            )
                                        )
                                        entry["fixed_angle_lp4_side"] = lp_side4
                                        entry["fixed_angle_lp4_violation"] = max(
                                            lp4_overlap, lp4_wall
                                        )
                                        if max(lp4_overlap, lp4_wall) <= 1e-8:
                                            certified = base.certify_candidate(
                                                q_lp4, n, incumbent
                                            )
                                            if certified:
                                                entry["selected"] = "four-SLP-LP"
                                                break
                                        if not certified:
                                            centers_slp5, theta_slp5, slp_side5 = (
                                                base.polish.fast_joint_xytheta_slp(
                                                    centers_lp4,
                                                    theta_slp4,
                                                    rounds=16,
                                                    tr_th0=2e-5,
                                                    tr_xy0=1e-4,
                                                    slp_margin=1e-11,
                                                    lp_margins=(5e-11, 1e-11, 1e-12),
                                                )
                                            )
                                            q_slp5 = np.column_stack(
                                                [centers_slp5, theta_slp5]
                                            )
                                            slp5_overlap, slp5_wall, _ = (
                                                base.overlap.max_violation(
                                                    q_slp5, slp_side5
                                                )
                                            )
                                            entry["joint_slp5_side"] = slp_side5
                                            entry["joint_slp5_violation"] = max(
                                                slp5_overlap, slp5_wall
                                            )
                                            if max(slp5_overlap, slp5_wall) <= 1e-8:
                                                centers_lp5, lp_side5 = (
                                                    base.polish.fixed_angle_lp_polish(
                                                        centers_slp5,
                                                        theta_slp5,
                                                        rounds=12,
                                                        time_limit=90.0,
                                                        margin=5e-12,
                                                    )
                                                )
                                                q_lp5 = np.column_stack(
                                                    [centers_lp5, theta_slp5]
                                                )
                                                lp5_overlap, lp5_wall, _ = (
                                                    base.overlap.max_violation(
                                                        q_lp5, lp_side5
                                                    )
                                                )
                                                entry["fixed_angle_lp5_side"] = lp_side5
                                                entry["fixed_angle_lp5_violation"] = max(
                                                    lp5_overlap, lp5_wall
                                                )
                                                if max(lp5_overlap, lp5_wall) <= 1e-8:
                                                    certified = base.certify_candidate(
                                                        q_lp5, n, incumbent
                                                    )
                                                    if certified:
                                                        entry["selected"] = "five-SLP-LP"
                                                        break
        except Exception as exc:
            entry["fixed_angle_lp_error"] = str(exc)

    result = {
        "n": n,
        "incumbent_side": incumbent,
        "maps": len(maps),
        "screened_starts": len(candidates),
        "repaired_starts": len(attempts),
        "attempts": [
            {key: value for key, value in entry.items() if key != "_q"}
            for entry in attempts[:24]
        ],
        "certified": certified,
        "elapsed_seconds": time.monotonic() - started,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", required=True)
    parser.add_argument("--seconds-per-target", type=float, default=420.0)
    parser.add_argument("--repair-iterations", type=int, default=15000)
    parser.add_argument("--top-starts", type=int, default=40)
    args = parser.parse_args()
    targets = [int(item) for item in args.targets.split(",") if item]
    DOCS.mkdir(parents=True, exist_ok=True)
    base = load_base()
    existing = base.union.parse_text(FINAL.read_text(encoding="utf-8"))
    all_results = []
    wins = {}
    for n in targets:
        result = scan(
            n,
            args.seconds_per_target,
            args.repair_iterations,
            args.top_starts,
        )
        certified = result.pop("certified", None)
        all_results.append(result)
        (DOCS / f"refine-result-{n}.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        if certified:
            existing[n] = base.union.parse_text(certified["final_text"])[n]
            wins[str(n)] = {
                "literal_side": certified["literal_side"],
                "pandas_side": certified["pandas_side"],
            }
            (DOCS / f"refine-win-{n}.txt").write_text(
                certified["final_text"], encoding="utf-8"
            )
    FINAL.write_text(base.union.final_text(existing), encoding="utf-8")
    (DOCS / "refine-results.json").write_text(
        json.dumps({"results": all_results, "wins": wins}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"wins": wins}, indent=2))


if __name__ == "__main__":
    main()
