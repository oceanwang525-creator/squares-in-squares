#!/usr/bin/env python3
"""H117 + H119 (Round 3): Alternating Mechwalk + PSLP2 / Row980 Reduced-Arm Ratchet + LPSlack."""
from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import importlib.util
import json
import math
import sys
import time
from decimal import Decimal
from pathlib import Path

import numpy as np

POC_DIR = Path(__file__).resolve().parent
GSTEP_DIR = POC_DIR.parent
REPO_ROOT = GSTEP_DIR.parents[1]

for _p in (
    POC_DIR,
    REPO_ROOT / "poc" / "row_ratchet_980" / "poc",
    REPO_ROOT / "poc" / "swarm_mechwalk" / "poc",
    REPO_ROOT / "poc" / "swarm_highd" / "poc",
    REPO_ROOT / "poc" / "swarm_lpslack" / "poc",
    REPO_ROOT / "poc" / "auto_union" / "poc",
    REPO_ROOT / "poc" / "baseline" / "poc" / "src",
):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import cpp_wrap  # noqa: E402
import gstep_lib as gl  # noqa: E402
import lpslack  # noqa: E402
import mechwalk_lib as ml  # noqa: E402
import pslp2  # noqa: E402
import row980  # noqa: E402
import union as au  # noqa: E402

_mw_spec = importlib.util.spec_from_file_location(
    "mw_run_gate", REPO_ROOT / "poc" / "swarm_mechwalk" / "poc" / "run_gate.py"
)
assert _mw_spec and _mw_spec.loader
mw_gate = importlib.util.module_from_spec(_mw_spec)
sys.modules["mw_run_gate"] = mw_gate
_mw_spec.loader.exec_module(mw_gate)

GROUP_A = [974, 975, 912, 913, 852, 863, 924, 799, 748, 687, 758, 876, 910, 686]
GROUP_B = [979, 981, 978, 977, 976, 919, 918, 917, 916, 859, 858, 857, 856]
GROUP_C = [969, 908, 791, 847, 872, 813, 818, 761, 706, 650, 599, 595, 550, 459, 453]
ALL_R3_TARGETS = GROUP_A + GROUP_B + GROUP_C
DEFAULT_CACHE = Path("/tmp/gstep_r3_frontier.json")


def _normalize_box(X: np.ndarray, Y: np.ndarray, T: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    X = np.asarray(X, dtype=np.float64).copy()
    Y = np.asarray(Y, dtype=np.float64).copy()
    T = np.asarray(T, dtype=np.float64).copy()
    e = 0.5 * (np.abs(np.cos(T)) + np.abs(np.sin(T)))
    X -= float(np.min(X - e))
    Y -= float(np.min(Y - e))
    L = float(max(np.max(X + e), np.max(Y + e)))
    return X, Y, T, L


def _certify_with_microscale(n: int, X: np.ndarray, Y: np.ndarray, T: np.ndarray) -> dict | None:
    """Certify (X, Y, T) via dual literal/pandas check, trying minimal scale slack."""
    X, Y, T, _ = _normalize_box(X, Y, T)
    best: dict | None = None
    best_dec: Decimal | None = None
    for eps in (0.0, 1e-14, 1e-13, 1e-12, 5e-12, 2e-11, 1e-10):
        scale = 1.0 + eps
        ver = ml.certify_canonical(n, X * scale, Y * scale, T)
        if not ver.get("ok"):
            continue
        if ver["L_lit"] != ver["L_pandas"]:
            continue
        c_dec = max(Decimal(ver["L_lit"]), Decimal(ver["L_pandas"]))
        if best_dec is None or c_dec < best_dec:
            best_dec = c_dec
            best = ver
            if eps == 0.0:
                break
    return best


def _pslp_squeeze_state(
    X: np.ndarray,
    Y: np.ndarray,
    T: np.ndarray,
    try_reduced: bool = True,
    max_it_full: int = 160,
    max_it_red: int = 200,
    always_full: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Run row980 reduced-arm squeeze (when tilted arms exist) + full-row pslp2 squeeze."""
    X0, Y0, T0, L0 = _normalize_box(X, Y, T)
    n = len(X0)
    K = int(math.floor(L0))
    f0 = L0 - K
    best_X, best_Y, best_T, best_L, best_v = X0, Y0, T0, L0, pslp2.viol(X0, Y0, T0, K, K, f0)
    ran_reduced = False

    a = np.column_stack([X0, Y0, T0])
    # 1. Reduced active-arm model (H119) when tilted arm squares exist and form <78% of packing
    if try_reduced:
        t_deg = np.degrees(np.mod(T0, np.pi / 2.0))
        til = np.minimum(t_deg, 90.0 - t_deg) > 2.0
        n_til = int(np.sum(til))
        if 2 <= n_til <= int(0.65 * n):
            try:
                idx, act, stat, mov, mov_full = row980.build_model(a, L0, K)
                if 8 <= len(idx) <= int(0.80 * n) and int(np.sum(mov_full)) > 0 and int(np.sum(act)) >= 2:
                    full, Ln, vr, _ = row980.run_squeeze(
                        a, L0, K, idx, stat, mov, mov_full, trust0=0.02, max_it=max_it_red
                    )
                    fv = row980.full_viol(full, Ln)
                    if fv <= 1e-11:
                        ran_reduced = True
                        if Ln < best_L:
                            best_X, best_Y, best_T, best_L, best_v = (
                                full[:, 0], full[:, 1], full[:, 2], Ln, fv
                            )
            except Exception as exc:
                print(f"    [row980 skip] {exc}", flush=True)

    # 2. Full-row pslp2 squeeze (H117): polish reduced output (or start state) across all 4 corners
    if not ran_reduced or always_full:
        f_start = best_L - K
        f_full, Xf, Yf, Tf, vf, _ = pslp2.squeeze(
            best_X, best_Y, best_T, K, K, f_start, trust0=0.02, max_it=max_it_full
        )
        if vf <= 1e-12 and K + f_full < best_L:
            best_X, best_Y, best_T, best_L, best_v = Xf, Yf, Tf, K + f_full, vf
    elif ran_reduced:
        f2, X2, Y2, T2, v2, _ = pslp2.squeeze(
            best_X, best_Y, best_T, K, K, best_L - K, trust0=4e-3, max_it=min(80, max_it_full)
        )
        if v2 <= 1e-12 and K + f2 < best_L:
            best_X, best_Y, best_T, best_L, best_v = X2, Y2, T2, K + f2, v2

    best_X, best_Y, best_T, best_L = _normalize_box(best_X, best_Y, best_T)
    return best_X, best_Y, best_T, best_L, best_v


def _lpslack_polish(n: int, ver: dict, time_limit: float = 15.0) -> dict:
    """Optional lpslack polish on a certified state; returns better or original ver."""
    try:
        L_worst = max(Decimal(ver["L_lit"]), Decimal(ver["L_pandas"]))
        res = lpslack.optimize_group(
            n,
            ver["rows"],
            str(L_worst),
            outer_iters=3,
            fixed_rounds=4,
            slp_modes=("dual_non90", "non90"),
            tr_th_schedule=(1e-4, 1e-5),
            cycles=1,
            time_limit=time_limit,
        )
        if res and res.get("status") in ("WIN", "TINY_WIN"):
            lit = res.get("literal_L")
            pan = res.get("pandas_L")
            if lit is not None and pan is not None and lit == pan:
                new_dec = max(Decimal(lit), Decimal(pan))
                if new_dec < L_worst:
                    return {
                        "ok": True,
                        "L_lit": lit,
                        "L_pandas": pan,
                        "L_dec_str": str(new_dec),
                        "L_float": float(new_dec),
                        "rows": res["rows"],
                    }
    except Exception:
        pass
    return ver


def build_or_load_frontier(targets: list[int], cache_path: Path = DEFAULT_CACHE) -> dict[int, dict]:
    """Load frontier for targets from cache if complete, else scan repo and populate cache."""
    cached: dict[int, dict] = {}
    if cache_path.exists():
        raw = json.loads(cache_path.read_text(encoding="utf-8"))
        for k, v in raw.items():
            cached[int(k)] = {
                "x": np.array(v["x"], dtype=np.float64),
                "y": np.array(v["y"], dtype=np.float64),
                "th": np.array(v["th"], dtype=np.float64),
                "L": float(v["L"]),
                "L_dec": Decimal(v["L_dec"]),
                "source": str(v["source"]),
            }
    missing = [n for n in targets if n not in cached]
    if missing:
        print(f"Loading true origin/main frontier for {len(missing)} missing target(s)...", flush=True)
        loaded = gl.fast_load_true_frontier(missing, certify=True)
        for n, d in loaded.items():
            cached[int(n)] = d
        ser = {
            str(n): {
                "x": d["x"].tolist(),
                "y": d["y"].tolist(),
                "th": d["th"].tolist(),
                "L": float(d["L"]),
                "L_dec": str(d["L_dec"]),
                "source": d["source"],
            }
            for n, d in sorted(cached.items())
        }
        cache_path.write_text(json.dumps(ser), encoding="utf-8")
    return {n: cached[n] for n in targets}


def ratchet_target(
    n: int,
    incumbent: dict,
    max_cycles: int = 2,
    mw_rounds: int = 3,
    top_k_polish: int = 4,
    top_squeeze: int = 2,
    n_threads: int = 1,
    save_wins: bool = True,
) -> dict:
    """Run Stage-0 pslp2 squeeze (or resume existing Stage-0 win) + alternating mechwalk/pslp2 ratchet."""
    t0 = time.time()
    L_inc_dec: Decimal = incumbent["L_dec"]
    best_dec = L_inc_dec
    best_ver: dict | None = None
    best_label = "incumbent"

    cur_x = incumbent["x"].copy()
    cur_y = incumbent["y"].copy()
    cur_th = incumbent["th"].copy()
    cur_L = float(incumbent["L"])

    wp = gl.DATA_DIR / f"win_n{n}.json"
    resumed_win = False
    if wp.exists():
        try:
            prev = json.loads(wp.read_text(encoding="utf-8"))
            prev_dec = max(Decimal(prev["L_lit"]), Decimal(prev["L_pandas"]))
            if prev_dec < L_inc_dec and len(prev.get("rows", [])) == n:
                best_dec = prev_dec
                best_ver = {
                    "ok": True,
                    "L_lit": prev["L_lit"],
                    "L_pandas": prev["L_pandas"],
                    "L_dec_str": str(prev_dec),
                    "L_float": float(prev_dec),
                    "rows": prev["rows"],
                }
                best_label = str(prev.get("label", "stage0_pslp2"))
                cur_x, cur_y, cur_th, cur_L = _normalize_box(
                    np.array([float(r["x"]) for r in prev["rows"]]),
                    np.array([float(r["y"]) for r in prev["rows"]]),
                    np.array([float(r["angle"]) for r in prev["rows"]]),
                )
                resumed_win = True
        except Exception:
            pass

    print(
        f"\n=== R3 H117/H119 n={n} | L_inc={str(L_inc_dec)[:20]} ({incumbent['source']}) "
        f"{'[resumed ' + str(best_dec)[:18] + ']' if resumed_win else ''}===",
        flush=True,
    )

    def _checkpoint(ver_dict: dict, lbl: str) -> None:
        if not save_wins:
            return
        v_dec = max(Decimal(ver_dict["L_lit"]), Decimal(ver_dict["L_pandas"]))
        dL_c = float(v_dec - L_inc_dec)
        if dL_c <= -1e-9:
            gl.save_win_locked({
                "n": int(n),
                "label": lbl,
                "L_lit": ver_dict["L_lit"],
                "L_pandas": ver_dict["L_pandas"],
                "L_dec_str": ver_dict["L_dec_str"],
                "L_float": float(v_dec),
                "L_inc": str(L_inc_dec),
                "inc_source": incumbent["source"],
                "dL": dL_c,
                "is_real_win": True,
                "rows": ver_dict["rows"],
            })

    # Stage 0: Direct pslp2 + row980 squeeze on incumbent (skip if already resumed Stage-0 win)
    if not resumed_win:
        ts = time.time()
        sx, sy, st, sL, sv = _pslp_squeeze_state(cur_x, cur_y, cur_th, try_reduced=True, max_it_full=240, always_full=True)
        if sv <= 1e-11 and sL < cur_L:
            ver0 = _certify_with_microscale(n, sx, sy, st)
            if ver0 is not None:
                v0_dec = max(Decimal(ver0["L_lit"]), Decimal(ver0["L_pandas"]))
                dL0 = float(v0_dec - L_inc_dec)
                print(f"  [stage0 pslp2] dL={dL0:+.3e} L={ver0['L_dec_str'][:20]} [{time.time()-ts:.1f}s]", flush=True)
                if v0_dec < best_dec:
                    best_dec = v0_dec
                    best_ver = ver0
                    best_label = "stage0_pslp2"
                    cur_x, cur_y, cur_th, cur_L = sx, sy, st, float(best_dec)
                    _checkpoint(best_ver, best_label)
            else:
                print(f"  [stage0 pslp2] sL={sL:.9f} cert failed [{time.time()-ts:.1f}s]", flush=True)
        else:
            print(f"  [stage0 pslp2] no drop (sL={sL:.9f} vs {cur_L:.9f}, v={sv:.1e}) [{time.time()-ts:.1f}s]", flush=True)

    beam = [{
        "x": cur_x.copy(),
        "y": cur_y.copy(),
        "th": cur_th.copy(),
        "L": cur_L,
        "L_dec": best_dec,
        "label": best_label,
    }]

    # Alternating ratchet cycles: mechwalk -> pslp2 squeeze
    for c_idx in range(max_cycles):
        cycle_improved = False
        for r_idx in range(1, mw_rounds + 1):
            tr = time.time()
            polished, r_info = mw_gate._run_round(
                n,
                c_idx * mw_rounds + r_idx,
                beam,
                float(best_dec),
                n_threads=n_threads,
                top_k_polish=top_k_polish,
            )
            valid = [p for p in polished if p["verify"].get("ok")]
            valid.sort(key=lambda p: max(Decimal(p["verify"]["L_lit"]), Decimal(p["verify"]["L_pandas"])))

            # Record any direct mechwalk win first
            for p in valid:
                if p["verify"]["L_lit"] != p["verify"]["L_pandas"]:
                    continue
                p_dec = max(Decimal(p["verify"]["L_lit"]), Decimal(p["verify"]["L_pandas"]))
                if p_dec < best_dec:
                    best_dec = p_dec
                    best_ver = {
                        "ok": True,
                        "L_lit": p["verify"]["L_lit"],
                        "L_pandas": p["verify"]["L_pandas"],
                        "L_dec_str": str(p_dec),
                        "L_float": float(p_dec),
                        "rows": p["rows"],
                    }
                    best_label = f"c{c_idx}_{p['label']}"
                    cycle_improved = True
                    _checkpoint(best_ver, best_label)

            # Select top distinct candidates (valid OR near-feasible close to best_L) for pslp2 squeeze
            squeeze_cands: list[dict] = []
            seen_states: list[dict] = []
            pool = sorted(
                [p for p in polished if p["pen_fin"] < 1e-4 and p["L_fin"] < float(best_dec) + 4e-4],
                key=lambda p: (0 if p["verify"].get("ok") else 1, p["L_fin"]),
            )
            for p in pool:
                st_c = {"x": p["x"], "y": p["y"], "th": p["th"]}
                if all(mw_gate._state_distance(st_c, prev) > 1e-3 for prev in seen_states):
                    seen_states.append(st_c)
                    squeeze_cands.append(p)
                    if len(squeeze_cands) >= top_squeeze:
                        break

            for p in squeeze_cands:
                qx, qy, qt, qL, qv = _pslp_squeeze_state(p["x"], p["y"], p["th"], try_reduced=True, max_it_full=180)
                if qv <= 1e-11 and qL < float(best_dec) - 1e-10:
                    ver_q = _certify_with_microscale(n, qx, qy, qt)
                    if ver_q is not None:
                        q_dec = max(Decimal(ver_q["L_lit"]), Decimal(ver_q["L_pandas"]))
                        if q_dec < best_dec:
                            best_dec = q_dec
                            best_ver = ver_q
                            best_label = f"c{c_idx}_{p['label']}+pslp2"
                            cycle_improved = True
                            _checkpoint(best_ver, best_label)
                            print(
                                f"  [c{c_idx} r{r_idx} RATCHET WIN] {best_label} -> dL={float(best_dec - L_inc_dec):+.3e}",
                                flush=True,
                            )

            # Refresh beam (size <= 2) with best state + best distinct valid mechwalk state
            cand_beam = list(beam)
            if best_ver is not None:
                bx, by, bt, bL = _normalize_box(
                    np.array([float(r["x"]) for r in best_ver["rows"]]),
                    np.array([float(r["y"]) for r in best_ver["rows"]]),
                    np.array([float(r["angle"]) for r in best_ver["rows"]]),
                )
                cand_beam.append({"x": bx, "y": by, "th": bt, "L": bL, "L_dec": best_dec, "label": best_label})
            for p in valid[:4]:
                p_dec = max(Decimal(p["verify"]["L_lit"]), Decimal(p["verify"]["L_pandas"]))
                cand_beam.append({"x": p["x"], "y": p["y"], "th": p["th"], "L": float(p_dec), "L_dec": p_dec, "label": p["label"]})
            cand_beam.sort(key=lambda c: c["L_dec"])
            new_beam: list[dict] = []
            for cb in cand_beam:
                if all(mw_gate._state_distance(cb, nb) > 1e-3 for nb in new_beam):
                    new_beam.append(cb)
                    if len(new_beam) >= 2:
                        break
            if new_beam:
                beam = new_beam

            print(
                f"  [c{c_idx} r{r_idx}/{mw_rounds}] seeds={r_info.get('n_seeds', 0)} "
                f"rejam={r_info.get('rejam_feas', 0)} mmelt={r_info.get('masked_feas', 0)} "
                f"valid={len(valid)} best_dL={float(best_dec - L_inc_dec):+.3e} [{time.time()-tr:.1f}s]",
                flush=True,
            )

        if not cycle_improved:
            print(f"  [cycle {c_idx}] converged (no further drop)", flush=True)
            break

    dL_best = float(best_dec - L_inc_dec)
    is_win = bool(best_ver is not None and dL_best <= -1e-9)
    if is_win and best_ver is not None:
        best_ver = _lpslack_polish(n, best_ver, time_limit=20.0)
        best_dec = max(Decimal(best_ver["L_lit"]), Decimal(best_ver["L_pandas"]))
        dL_best = float(best_dec - L_inc_dec)
        if save_wins:
            win_entry = {
                "n": int(n),
                "label": best_label,
                "L_lit": best_ver["L_lit"],
                "L_pandas": best_ver["L_pandas"],
                "L_dec_str": best_ver["L_dec_str"],
                "L_float": float(best_dec),
                "L_inc": str(L_inc_dec),
                "inc_source": incumbent["source"],
                "dL": dL_best,
                "is_real_win": True,
                "rows": best_ver["rows"],
            }
            saved = gl.save_win_locked(win_entry)
            print(f"  => SAVED WIN n={n}: dL={dL_best:+.3e} L={str(best_dec)[:20]} saved={saved} [{time.time()-t0:.1f}s]", flush=True)
    else:
        print(f"  => NO WIN n={n}: best_dL={dL_best:+.3e} [{time.time()-t0:.1f}s]", flush=True)

    return {
        "n": int(n),
        "inc_source": incumbent["source"],
        "L_inc": str(L_inc_dec),
        "L_best": str(best_dec),
        "dL": dL_best,
        "is_real_win": is_win,
        "label": best_label,
        "dt_s": round(time.time() - t0, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--targets", type=int, nargs="+", default=GROUP_A)
    parser.add_argument("--tag", type=str, default="r3")
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--top-k-polish", type=int, default=4)
    parser.add_argument("--top-squeeze", type=int, default=2)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE)
    args = parser.parse_args()

    cpp_wrap.get_cpp_lib()
    gl.DATA_DIR.mkdir(parents=True, exist_ok=True)
    gl.DOCS_DIR.mkdir(parents=True, exist_ok=True)

    if args.cache_only:
        build_or_load_frontier(ALL_R3_TARGETS, args.cache_path)
        print(f"Frontier cache ready at {args.cache_path} ({len(ALL_R3_TARGETS)} targets)")
        return

    frontier = build_or_load_frontier(args.targets, args.cache_path)
    results: list[dict] = []
    out_doc = gl.DOCS_DIR / f"r3_{args.tag}.json"
    for n in args.targets:
        res = ratchet_target(
            n,
            frontier[n],
            max_cycles=args.cycles,
            mw_rounds=args.rounds,
            top_k_polish=args.top_k_polish,
            top_squeeze=args.top_squeeze,
            n_threads=args.threads,
        )
        results.append(res)
        out_doc.write_text(json.dumps(results, indent=2), encoding="utf-8")

    wins = [r for r in results if r["is_real_win"]]
    print(f"\n=== R3 {args.tag}: {len(wins)}/{len(results)} STRICT WINs ===", flush=True)
    for w in wins:
        print(f"  n={w['n']}: dL={w['dL']:+.3e} ({w['label']})", flush=True)


if __name__ == "__main__":
    main()
