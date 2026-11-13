#!/usr/bin/env python3
"""H22 Falsification Gate & Scaling Runner: Iterated Contact-Mechanism Walk + C++ Masked Melt."""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from decimal import Decimal
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

POC_DIR = Path(__file__).resolve().parent
if str(POC_DIR) not in sys.path:
    sys.path.insert(0, str(POC_DIR))

import mechwalk_lib as ml  # noqa: E402
import utils as hu  # noqa: E402
from cpp_wrap import cpp_eval_sat_pen_bbox, cpp_expand_sym, full_to_free_sym, run_cpp_masked_melt  # noqa: E402
from joint_slp import joint_xytheta_slp  # noqa: E402

GATE_TARGETS = [463, 692, 701]
SCALE_TARGETS = [
    639, 691, 745, 858, 918, 973, 915, 542, 598, 459,
    978, 634, 695, 861, 920, 926, 499, 804, 793, 850,
    909, 686, 851, 852, 912,
]


def _polish_worker(args: tuple) -> dict:
    """Relax residual penetration, run 2-stage joint_xytheta_slp + fixed-angle LP + dual certification."""
    idx, label, n, L_ref, x, y, th, cpp_pen, cpp_L = args
    t0 = time.time()
    x, y, th, L_cur = ml.normalize(x, y, th)

    if 1e-7 < cpp_pen < 0.025:
        L_try = max(L_cur, cpp_L) + max(0.002, 1.8 * cpp_pen)
        xr, yr, thr, _ = hu.relax_lbfgs(x, y, th, L_try, maxiter=160, w_wall=4.0, w_pair=1.0, side=1.0)
        pen_r, L_r = cpp_eval_sat_pen_bbox(xr, yr, thr, side=1.0)
        if pen_r <= cpp_pen + 1e-6:
            x, y, th, cpp_pen, cpp_L = xr, yr, thr, pen_r, L_r

    pen_pre, L_pre = cpp_eval_sat_pen_bbox(x, y, th, side=1.0)
    ok_slp, L_slp = False, L_pre
    x_f, y_f, th_f = x.copy(), y.copy(), th.copy()

    if pen_pre < 0.015:
        xs, ys, ts, Ls, ok_s = joint_xytheta_slp(
            x, y, th, rounds=10, tr_th0=0.020, tr_xy0=0.09, margin=1e-8, tol=1e-8, patience=2
        )
        if ok_s:
            x_f, y_f, th_f, L_slp, ok_slp = xs, ys, ts, Ls, True
            if L_slp < L_ref + 8e-4:
                xs2, ys2, ts2, Ls2, ok_s2 = joint_xytheta_slp(
                    x_f, y_f, th_f, rounds=12, tr_th0=0.006, tr_xy0=0.025, margin=1e-8, tol=1e-11, patience=2
                )
                if ok_s2 and Ls2 < L_slp:
                    x_f, y_f, th_f, L_slp = xs2, ys2, ts2, Ls2

    ver: dict = {"ok": False}
    best_ver_dec: Decimal | None = None
    best_x, best_y = x_f.copy(), y_f.copy()
    for m_lp in (1e-7, 2e-8, 5e-9, 1e-6):
        xf2, yf2, L_lp2, ok_lp2 = hu.fixed_angle_lp_tighten(x_f, y_f, th_f, rounds=5, margin=m_lp)
        if ok_lp2 and L_lp2 <= L_slp + 5e-6:
            cand_ver = ml.certify_canonical(n, xf2, yf2, th_f)
            if cand_ver.get("ok"):
                c_dec = max(Decimal(cand_ver["L_lit"]), Decimal(cand_ver["L_pandas"]))
                if best_ver_dec is None or c_dec < best_ver_dec:
                    best_ver_dec = c_dec
                    best_x, best_y, L_slp, ok_slp, ver = xf2, yf2, L_lp2, True, cand_ver

    if not ver.get("ok"):
        cand_ver = ml.certify_canonical(n, x_f, y_f, th_f)
        if cand_ver.get("ok"):
            ver = cand_ver
            best_x, best_y = x_f.copy(), y_f.copy()

    pen_fin, L_fin = cpp_eval_sat_pen_bbox(best_x, best_y, th_f, side=1.0)
    return {
        "idx": idx,
        "label": label,
        "n": n,
        "cpp_pen": float(cpp_pen),
        "cpp_L": float(cpp_L),
        "ok_slp": bool(ok_slp),
        "pen_fin": float(pen_fin),
        "L_fin": float(L_fin),
        "x": best_x,
        "y": best_y,
        "th": th_f,
        "verify": {k: v for k, v in ver.items() if k != "rows"},
        "rows": ver.get("rows"),
        "dt": round(time.time() - t0, 2),
    }


def build_mode_walk_seeds(
    x: np.ndarray,
    y: np.ndarray,
    th: np.ndarray,
    L_cur: float,
    tols: tuple[float, ...],
    modes_per_tol: int,
    max_total_modes: int,
) -> tuple[list[dict], dict]:
    """Extract deduplicated soft modes across contact_tol thresholds and build stepped seeds."""
    n = len(x)
    all_modes: list[tuple[float, float, np.ndarray, np.ndarray, str]] = []
    diag_tols: dict[str, dict] = {}

    for tol in tols:
        jac, info = ml.contact_jacobian(x, y, th, contact_tol=tol)
        sv, raw_modes = ml.soft_modes(jac, k=max(14, modes_per_tol * 2))
        sel = ml.select_diverse_modes(raw_modes, sv, x, y, max_keep=modes_per_tol)
        diag_tols[f"{tol:.0e}"] = {
            "rows": info["constraints"],
            "pair": info["pair_contacts"],
            "wall": info["wall_contacts"],
            "sv": [round(float(s[1]), 6) for s in sel],
            "neff": [round(float(s[3]), 1) for s in sel],
        }
        for _, sval, mode_vec, n_eff in sel:
            if any(abs(float(np.dot(mode_vec, prev[2]))) > 0.90 for prev in all_modes):
                continue
            disp, _ = ml.mode_participation(mode_vec, n)
            all_modes.append((sval, n_eff, mode_vec, disp, f"t{tol:.0e}"))

    # Diversify: keep lowest singular values + highest collective n_eff
    if len(all_modes) > max_total_modes:
        by_sv = sorted(range(len(all_modes)), key=lambda i: all_modes[i][0])
        by_neff = sorted(range(len(all_modes)), key=lambda i: -all_modes[i][1])
        picked: list[int] = []
        used: set[int] = set()
        for src_list in (by_sv, by_neff):
            for idx in src_list:
                if len(picked) >= max_total_modes:
                    break
                if idx not in used and len(picked) < (max_total_modes // 2 if src_list is by_sv else max_total_modes):
                    picked.append(idx)
                    used.add(idx)
        for idx in by_sv:
            if len(picked) >= max_total_modes:
                break
            if idx not in used:
                picked.append(idx)
                used.add(idx)
        all_modes = [all_modes[i] for i in picked]

    seeds: list[dict] = []
    # Scale by L_inf (max single-square step) so collective modes also make real finite moves
    target_max_steps = (0.008, 0.025, 0.075, 0.18)
    for m_idx, (sval, n_eff, mode_vec, disp, t_lbl) in enumerate(all_modes):
        max_disp = float(np.max(disp))
        if max_disp < 1e-8:
            continue
        # Active pocket mask: top moving squares + spatial neighbors within R=2.4
        top_k_seeds = max(6, min(16, n // 30))
        top_idx = np.argsort(disp)[-top_k_seeds:]
        core = disp >= max(0.25 * max_disp, float(np.quantile(disp, 0.85)))
        core[top_idx] = True
        cx_c, cy_c = x[core], y[core]
        if len(cx_c) > 0:
            d2 = (x[:, None] - cx_c[None, :]) ** 2 + (y[:, None] - cy_c[None, :]) ** 2
            active_bool = np.min(d2, axis=1) <= 2.4 ** 2
        else:
            active_bool = core
        # Cap active fraction at 50% so outer frame stays rigid during masked melt
        if np.sum(active_bool) > 0.50 * n:
            thresh = float(np.quantile(disp, 0.65))
            active_bool = disp >= thresh
        active_u8 = active_bool.astype(np.uint8)
        mask_3n = np.concatenate([active_u8, active_u8, active_u8])

        used_amps: list[float] = []
        for step in target_max_steps:
            amp = min(step / max_disp, 2.5)
            if any(abs(amp - ua) / max(amp, ua, 1e-9) < 0.25 for ua in used_amps):
                continue
            used_amps.append(amp)
            for sgn, sgn_lbl in ((1.0, "p"), (-1.0, "m")):
                dx = sgn * amp * mode_vec[:n]
                dy = sgn * amp * mode_vec[n : 2 * n]
                dth = sgn * amp * mode_vec[2 * n :]
                wx, wy, wth, _ = ml.normalize(x + dx, y + dy, th + dth)
                seeds.append(
                    {
                        "x": wx,
                        "y": wy,
                        "th": wth,
                        "mask": mask_3n,
                        "label": f"m{m_idx}_{t_lbl}_{sgn_lbl}_{step:.3g}",
                        "amp": amp,
                        "n_eff": n_eff,
                        "sv": sval,
                    }
                )
    return seeds, diag_tols


def _run_round(
    n: int,
    round_idx: int,
    beam_states: list[dict],
    L_inc_float: float,
    n_threads: int,
    top_k_polish: int,
) -> tuple[list[dict], dict]:
    """Run one mechanism-walk round across beam states (both standard rejam + hybrid masked melt)."""
    all_seeds: list[dict] = []
    round_diags: list[dict] = []
    tols = (2e-5, 2e-4, 1e-3) if round_idx <= 2 else (5e-5, 4e-4, 2e-3)

    for b_idx, st in enumerate(beam_states):
        seeds, diag = build_mode_walk_seeds(
            st["x"], st["y"], st["th"], st["L"],
            tols=tols,
            modes_per_tol=4 if len(beam_states) > 1 else 6,
            max_total_modes=6 if len(beam_states) > 1 else 9,
        )
        for s in seeds:
            s["label"] = f"r{round_idx}_b{b_idx}_{s['label']}"
        all_seeds.extend(seeds)
        round_diags.append({"beam": b_idx, "L_in": st["L"], "tols": diag, "n_seeds": len(seeds)})

    if not all_seeds:
        return [], {"diags": round_diags, "rejam_feas": 0, "masked_feas": 0}

    # Path A: Pure mechanism-walk SAT re-jam (all-1s mask, no random ILS)
    q_rejam: list[np.ndarray] = []
    m_rejam: list[np.ndarray] = []
    l_rejam: list[list[float]] = []
    ones_mask = np.ones(3 * n, dtype=np.uint8)
    for idx, s in enumerate(all_seeds):
        if idx % 2 == 0:
            sched = [L_inc_float + 0.004, L_inc_float + 0.0005, L_inc_float - 1.5e-4]
        else:
            sched = [L_inc_float + 0.012, L_inc_float + 0.002, L_inc_float - 4e-4]
        L0 = sched[0]
        q_rejam.append(full_to_free_sym(s["x"], s["y"], s["th"], L0))
        m_rejam.append(ones_mask)
        l_rejam.append(sched)

    qb_a, qf_a, lf_a, _, po_a, lb_a = run_cpp_masked_melt(
        n, 1, q_rejam, m_rejam, l_rejam, [0, 0, 0],
        a_start=0.94, n_melt_stages=3, masked_iters=140, full_iters=260,
        n_threads=n_threads, base_seed=51000 + round_idx * 1000 + n,
    )

    # Path B: Hybrid Soft-Mode -> C++ Masked Melt (mode-activity mask + cavity ILS)
    q_mask: list[np.ndarray] = []
    m_mask: list[np.ndarray] = []
    l_mask: list[list[float]] = []
    for idx, s in enumerate(all_seeds):
        if idx % 2 == 0:
            sched = [L_inc_float + 0.003, L_inc_float - 2e-5, L_inc_float - 2e-4]
        else:
            sched = [L_inc_float + 0.009, L_inc_float + 0.0005, L_inc_float - 4e-4]
        L0 = sched[0]
        q_mask.append(full_to_free_sym(s["x"], s["y"], s["th"], L0))
        m_mask.append(s["mask"])
        l_mask.append(sched)

    qb_b, qf_b, lf_b, _, po_b, lb_b = run_cpp_masked_melt(
        n, 1, q_mask, m_mask, l_mask, [0, 5, 10],
        a_start=0.82, n_melt_stages=4, masked_iters=160, full_iters=240,
        n_threads=n_threads, base_seed=71000 + round_idx * 1000 + n,
    )

    # Gather top candidates from both Path A and Path B (diversify across modes and amplitudes)
    def pick_top(lf, po, lb, qb, qf, l_sched, tag, quota):
        order = sorted(
            range(len(all_seeds)),
            key=lambda i: (0 if lf[i] < 1e9 else 1, float(lf[i]) if lf[i] < 1e9 else float(po[i])),
        )
        chosen: list[tuple] = []
        seen_modes: set[str] = set()
        seen_amps: set[str] = set()
        # Pass 1: diversify by mode AND encourage distinct amplitudes
        for i in order:
            parts = all_seeds[i]["label"].split("_")
            mode_key = "_".join(parts[:4])
            amp_key = parts[-1]
            if mode_key in seen_modes and (len(seen_amps) >= 2 or amp_key in seen_amps):
                continue
            seen_modes.add(mode_key)
            seen_amps.add(amp_key)
            use_f = bool(lf[i] < 1e9)
            L_ref = float(lf[i]) if use_f else float(l_sched[i][-1])
            x, y, th = cpp_expand_sym(n, 1, L_ref, qf[i] if use_f else qb[i])
            chosen.append((len(chosen), f"{tag}_{all_seeds[i]['label']}", n, L_inc_float, x, y, th, 0.0 if use_f else float(po[i]), float(lf[i]) if use_f else float(lb[i])))
            if len(chosen) >= quota:
                return chosen
        # Pass 2: fill remaining quota by raw score
        used_lbls = {c[1] for c in chosen}
        for i in order:
            lbl = f"{tag}_{all_seeds[i]['label']}"
            if lbl in used_lbls:
                continue
            use_f = bool(lf[i] < 1e9)
            L_ref = float(lf[i]) if use_f else float(l_sched[i][-1])
            x, y, th = cpp_expand_sym(n, 1, L_ref, qf[i] if use_f else qb[i])
            chosen.append((len(chosen), lbl, n, L_inc_float, x, y, th, 0.0 if use_f else float(po[i]), float(lf[i]) if use_f else float(lb[i])))
            if len(chosen) >= quota:
                break
        return chosen

    half_k = max(2, top_k_polish // 2)
    tasks = pick_top(lf_a, po_a, lb_a, qb_a, qf_a, l_rejam, "rejam", half_k)
    tasks += pick_top(lf_b, po_b, lb_b, qb_b, qf_b, l_mask, "mmelt", top_k_polish - len(tasks))
    for k_i in range(len(tasks)):
        t = list(tasks[k_i])
        t[0] = k_i
        tasks[k_i] = tuple(t)

    n_workers = min(4, max(1, n_threads), len(tasks))
    with mp.Pool(processes=n_workers) as pool:
        polished = pool.map(_polish_worker, tasks)

    info = {
        "diags": round_diags,
        "n_seeds": len(all_seeds),
        "rejam_feas": int(np.sum(lf_a < 1e9)),
        "rejam_best_Lf": float(np.min(lf_a)) if np.any(lf_a < 1e9) else None,
        "rejam_min_pen": float(np.min(po_a)),
        "masked_feas": int(np.sum(lf_b < 1e9)),
        "masked_best_Lf": float(np.min(lf_b)) if np.any(lf_b < 1e9) else None,
        "masked_min_pen": float(np.min(po_b)),
    }
    return polished, info


def _state_distance(a: dict, b: dict) -> float:
    """Measure max per-square coordinate/angle difference between two normalized states."""
    dx = np.max(np.abs(a["x"] - b["x"]))
    dy = np.max(np.abs(a["y"] - b["y"]))
    dth = np.max(np.abs((a["th"] - b["th"] + np.pi) % (2.0 * np.pi) - np.pi))
    return float(max(dx, dy, dth))


def solve_target(
    n: int,
    incumbent: dict,
    n_rounds: int = 4,
    beam_width: int = 2,
    top_k_polish: int = 8,
    n_threads: int = 4,
) -> tuple[dict, dict | None]:
    """Run iterated mechanism_walk + masked_melt across n_rounds on target n."""
    t0 = time.time()
    L_inc_dec: Decimal = incumbent["L_dec"]
    L_inc_float = float(L_inc_dec)

    beam: list[dict] = [{
        "x": incumbent["x"].copy(),
        "y": incumbent["y"].copy(),
        "th": incumbent["th"].copy(),
        "L": incumbent["L"],
        "L_dec": L_inc_dec,
        "label": "incumbent",
    }]

    best_overall: dict | None = None
    best_overall_dec: Decimal | None = None
    round_logs: list[dict] = []

    print(f"\n=== H22 n={n} | L_inc={str(L_inc_dec)[:20]} ({incumbent['source']}) ===", flush=True)

    for r_idx in range(1, n_rounds + 1):
        tr = time.time()
        polished, r_info = _run_round(n, r_idx, beam, L_inc_float, n_threads, top_k_polish)

        valid = [p for p in polished if p["verify"].get("ok")]
        valid.sort(key=lambda p: max(Decimal(p["verify"]["L_lit"]), Decimal(p["verify"]["L_pandas"])))

        round_best_dL = None
        for p in valid:
            p_dec = max(Decimal(p["verify"]["L_lit"]), Decimal(p["verify"]["L_pandas"]))
            dL = float(p_dec - L_inc_dec)
            if round_best_dL is None:
                round_best_dL = dL
            if best_overall_dec is None or p_dec < best_overall_dec:
                best_overall_dec = p_dec
                best_overall = p

        # Update beam: keep distinct low-L valid states (and always keep best overall if valid)
        candidates = list(beam)
        for p in valid:
            p_dec = max(Decimal(p["verify"]["L_lit"]), Decimal(p["verify"]["L_pandas"]))
            candidates.append({
                "x": p["x"],
                "y": p["y"],
                "th": p["th"],
                "L": float(p_dec),
                "L_dec": p_dec,
                "label": p["label"],
            })
        candidates.sort(key=lambda c: c["L_dec"])
        new_beam: list[dict] = []
        for c in candidates:
            if all(_state_distance(c, nb) > 1e-3 for nb in new_beam):
                new_beam.append(c)
                if len(new_beam) >= beam_width:
                    break
        beam = new_beam if new_beam else beam

        print(
            f"  [round {r_idx}/{n_rounds}] seeds={r_info['n_seeds']} "
            f"rejam={r_info['rejam_feas']} (best={r_info['rejam_best_Lf']}) "
            f"mmelt={r_info['masked_feas']} (best={r_info['masked_best_Lf']}) "
            f"valid={len(valid)} round_dL={round_best_dL} best_dL={float(best_overall_dec - L_inc_dec) if best_overall_dec else None} "
            f"[{time.time() - tr:.1f}s]",
            flush=True,
        )
        round_logs.append({
            "round": r_idx,
            "dt_s": round(time.time() - tr, 2),
            "rejam_feas": r_info["rejam_feas"],
            "rejam_best_Lf": r_info["rejam_best_Lf"],
            "rejam_min_pen": r_info["rejam_min_pen"],
            "masked_feas": r_info["masked_feas"],
            "masked_best_Lf": r_info["masked_best_Lf"],
            "masked_min_pen": r_info["masked_min_pen"],
            "n_valid": len(valid),
            "round_best_dL": round_best_dL,
            "top_candidates": [
                {
                    "label": p["label"],
                    "cpp_pen": p["cpp_pen"],
                    "cpp_L": p["cpp_L"],
                    "pen_fin": p["pen_fin"],
                    "L_fin": p["L_fin"],
                    "verify": p["verify"],
                    "dL": float(max(Decimal(p["verify"]["L_lit"]), Decimal(p["verify"]["L_pandas"])) - L_inc_dec)
                    if p["verify"].get("ok") else None,
                }
                for p in polished[:6]
            ],
        })

    status = "MISS"
    best_dL = float(best_overall_dec - L_inc_dec) if best_overall_dec is not None else None
    win_entry: dict | None = None
    if best_dL is not None and best_dL <= -1e-9:
        status = "REAL_WIN"
        win_entry = {
            "n": n,
            "label": best_overall["label"],
            "L_lit": best_overall["verify"]["L_lit"],
            "L_pandas": best_overall["verify"]["L_pandas"],
            "L_inc": str(L_inc_dec),
            "dL": best_dL,
            "rows": best_overall["rows"],
        }
    elif best_dL is not None and best_dL < 0.0:
        status = "TIE"

    print(
        f"  => RESULT n={n}: {status} best_dL={best_dL} "
        f"({best_overall['label'] if best_overall else 'none'}) [{time.time() - t0:.1f}s]",
        flush=True,
    )
    summary = {
        "n": n,
        "status": status,
        "L_inc": str(L_inc_dec),
        "inc_source": incumbent["source"],
        "best_dL": best_dL,
        "best_label": best_overall["label"] if best_overall else None,
        "best_L_lit": best_overall["verify"]["L_lit"] if best_overall else None,
        "best_L_pandas": best_overall["verify"]["L_pandas"] if best_overall else None,
        "dt_s": round(time.time() - t0, 2),
        "rounds": round_logs,
    }
    return summary, win_entry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=int, nargs="+", default=GATE_TARGETS)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--beam", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--tag", type=str, default="gate")
    args = parser.parse_args()

    ml.DATA_DIR.mkdir(parents=True, exist_ok=True)
    ml.DOCS_DIR.mkdir(parents=True, exist_ok=True)

    portfolio = ml.load_origin_main_portfolio(args.targets)

    # Load any prior certified REAL WINs from data/win_n*.json
    wins_by_n: dict[int, dict] = {}
    for wp in sorted(ml.DATA_DIR.glob("win_n*.json")):
        w = json.loads(wp.read_text(encoding="utf-8"))
        wins_by_n[int(w["n"])] = w

    results: list[dict] = []
    for n in args.targets:
        summary, win_entry = solve_target(
            n,
            portfolio[n],
            n_rounds=args.rounds,
            beam_width=args.beam,
            top_k_polish=args.top_k,
            n_threads=args.threads,
        )
        results.append(summary)
        if win_entry is not None:
            wins_by_n[n] = win_entry
            ml.save_win_locked(win_entry, ml.DATA_DIR, ml.FINAL_CSV)
        elif not ml.FINAL_CSV.exists():
            ml.write_final_csv(wins_by_n, ml.FINAL_CSV)

        (ml.DOCS_DIR / f"{args.tag}_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    # Ensure final.csv exists if zero wins occurred across all shards
    if not ml.FINAL_CSV.exists():
        ml.write_final_csv(wins_by_n, ml.FINAL_CSV)
    n_wins = sum(1 for r in results if r["status"] == "REAL_WIN")
    print(f"\n=== H22 {args.tag.upper()} COMPLETE: {n_wins}/{len(results)} REAL WINs ===", flush=True)


if __name__ == "__main__":
    main()
