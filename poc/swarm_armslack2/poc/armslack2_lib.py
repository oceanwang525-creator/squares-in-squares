"""H125 Core Library: Arm-Adjacent 0-deg Layer Liberation + Masked C++ Interface Melt + 4-Corner pslp + lpslack + mechwalk."""
from __future__ import annotations

import math
import sys
import time
from decimal import Decimal
from pathlib import Path

import numpy as np

POC_DIR = Path(__file__).resolve().parent
REPO_ROOT = POC_DIR.parents[2]
HIGHD_POC = REPO_ROOT / "poc" / "swarm_highd" / "poc"
MECHWALK_POC = REPO_ROOT / "poc" / "swarm_mechwalk" / "poc"
LPSLACK_POC = REPO_ROOT / "poc" / "swarm_lpslack" / "poc"

for _p in (POC_DIR, HIGHD_POC, MECHWALK_POC, LPSLACK_POC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import cpp_wrap  # noqa: E402

cpp_wrap.CPP_LIB = Path("/tmp/libhighd_ils_armslack2.so")

import arm_model as am  # noqa: E402
import lpslack  # noqa: E402
import mechwalk_lib as ml  # noqa: E402
import portfolio as pf  # noqa: E402
import utils as hu  # noqa: E402
from cpp_wrap import cpp_eval_sat_pen_bbox, cpp_expand_sym, full_to_free_sym, run_cpp_masked_melt  # noqa: E402
from joint_slp import joint_xytheta_slp  # noqa: E402


def _mask3(active: np.ndarray) -> np.ndarray:
    m = np.asarray(active, dtype=np.uint8)
    return np.concatenate([m, m, m])


def build_interface_melt_seeds(
    x_base: np.ndarray,
    y_base: np.ndarray,
    th_base: np.ndarray,
    L_base: float,
    rng: np.random.Generator,
) -> tuple[list[np.ndarray], list[np.ndarray], list[list[float]], list[str]]:
    """Build masked C++ interface-melt seeds liberating 1-2 layers of arm-adjacent 0-deg squares."""
    a0 = np.column_stack([x_base, y_base, th_base])
    info = am.identify_interface_layers(a0, layer1_r=1.65, layer2_r=2.75)
    clusters = am.cluster_tilted_arms(a0, tilt_deg=2.0, link_radius=1.9)

    tilted = info["tilted"] | info["slight"]
    layer1 = info["layer1"]
    layer2 = info["layer2"]

    q_list: list[np.ndarray] = []
    m_list: list[np.ndarray] = []
    ls_list: list[list[float]] = []
    labels: list[str] = []

    def add_seed(
        active_bool: np.ndarray,
        bdry_kick: np.ndarray,
        L_pad: float,
        L_end: float,
        amp_xy: float,
        amp_th_til: float,
        amp_th_bdry: float,
        label: str,
        bias_inward: bool = False,
    ) -> None:
        n = len(x_base)
        if int(active_bool.sum()) == 0:
            return
        L0 = L_base + L_pad
        sc = L0 / L_base
        xs = x_base * sc
        ys = y_base * sc
        ts = th_base.copy()

        act_idx = np.flatnonzero(active_bool)
        xs[act_idx] += rng.uniform(-amp_xy, amp_xy, len(act_idx))
        ys[act_idx] += rng.uniform(-amp_xy, amp_xy, len(act_idx))

        til_act = active_bool & (~bdry_kick)
        bd_act = active_bool & bdry_kick
        if np.any(til_act):
            k = int(til_act.sum())
            ts[til_act] += rng.uniform(-amp_th_til, amp_th_til, k)
        if np.any(bd_act):
            k = int(bd_act.sum())
            # Give liberated 0-deg boundary squares a 1..5 deg tilt kick to escape the theta=0 kink
            signs = rng.choice(np.array([-1.0, 1.0]), size=k)
            ts[bd_act] += signs * rng.uniform(0.018, amp_th_bdry, k)
            if bias_inward and np.any(tilted):
                # Nudge boundary 0-deg squares slightly toward the nearest tilted square
                tx = x_base[tilted]
                ty = y_base[tilted]
                for idx in np.flatnonzero(bd_act):
                    j = int(np.argmin((tx - x_base[idx]) ** 2 + (ty - y_base[idx]) ** 2))
                    dx, dy = float(tx[j] - x_base[idx]), float(ty[j] - y_base[idx])
                    norm = math.hypot(dx, dy) + 1e-9
                    step = float(rng.uniform(0.015, 0.045))
                    xs[idx] += step * (dx / norm)
                    ys[idx] += step * (dy / norm)

        q_list.append(full_to_free_sym(xs, ys, ts, L0))
        m_list.append(_mask3(active_bool))
        ls_list.append([L0, L_base + 0.25 * L_pad, L_base + 0.0002, L_base + L_end])
        labels.append(label)

    # 1. Full interface liberation (tilted + 1 or 2 layers of 0-deg boundary)
    if np.any(tilted):
        m_full1 = tilted | layer1
        m_full2 = tilted | layer2
        for s in range(4):
            add_seed(
                m_full1,
                layer1,
                L_pad=0.008 if s < 2 else 0.015,
                L_end=-0.0004 if s % 2 == 0 else -0.0018,
                amp_xy=0.020 if s < 2 else 0.040,
                amp_th_til=0.020,
                amp_th_bdry=0.055 if s < 2 else 0.090,
                label=f"iface_L1_{s}",
                bias_inward=(s % 2 == 1),
            )
        for s in range(4):
            add_seed(
                m_full2,
                layer2,
                L_pad=0.010 if s < 2 else 0.018,
                L_end=-0.0005 if s % 2 == 0 else -0.0022,
                amp_xy=0.020 if s < 2 else 0.045,
                amp_th_til=0.020,
                amp_th_bdry=0.055 if s < 2 else 0.085,
                label=f"iface_L2_{s}",
                bias_inward=(s % 2 == 1),
            )

    # 2. Per-brick / per-arm focused 2-layer interface liberation (keeps rest of packing rigid)
    for ci, cl in enumerate(clusters[:2]):
        d_cl = am._min_dist_to_pts(x_base, y_base, a0[cl, :2])
        cl_mask = np.zeros(len(x_base), dtype=bool)
        cl_mask[cl] = True
        bd1 = (~cl_mask) & (~tilted) & (d_cl <= 1.75)
        bd2 = (~cl_mask) & (~tilted) & (d_cl <= 2.85)
        for s in range(4):
            bd = bd1 if s < 2 else bd2
            act = cl_mask | bd | (tilted & (d_cl <= 1.75))
            add_seed(
                act,
                bd,
                L_pad=0.008 if s % 2 == 0 else 0.016,
                L_end=-0.0004 if s % 2 == 0 else -0.0020,
                amp_xy=0.025 if s % 2 == 0 else 0.050,
                amp_th_til=0.030,
                amp_th_bdry=0.065 if s % 2 == 0 else 0.10,
                label=f"arm{ci}_lib_{s}",
                bias_inward=(s % 2 == 1),
            )

    # 3. End-cap / corner-junction pocket melt (around tilted cluster endpoints)
    for ci, cl in enumerate(clusters[:2]):
        cx_pts = [
            (float(x_base[cl].min()), float(y_base[cl].max())),
            (float(x_base[cl].max()), float(y_base[cl].min())),
            (float(x_base[cl].min()), float(y_base[cl].min())),
            (float(x_base[cl].max()), float(y_base[cl].max())),
        ]
        # Pick the 2 corners closest to box walls or interior seam
        for pi, (px, py) in enumerate(cx_pts[:2]):
            rad = 5.5
            near = ((x_base - px) ** 2 + (y_base - py) ** 2) <= rad * rad
            bd_near = near & (~tilted)
            add_seed(
                near,
                bd_near,
                L_pad=0.012,
                L_end=-0.0012,
                amp_xy=0.040,
                amp_th_til=0.035,
                amp_th_bdry=0.085,
                label=f"junc_c{ci}_p{pi}",
                bias_inward=True,
            )

    return q_list, m_list, ls_list, labels


def run_pslp_cascade(
    a_start: np.ndarray,
    L_start: float,
    K: int,
    max_restarts: int = 6,
    reaches: tuple[float, ...] = (3.5, 2.6),
) -> tuple[np.ndarray, float]:
    """Run multi-restart 4-corner reduced pslp (liberating 1-2 layers of 0-deg boundary) + full pslp."""
    a_best, L_best = am.normalize_arr(a_start)
    if am.full_viol(a_best, L_best) > 1e-11:
        return a_best, L_best

    for reach in reaches:
        a_cur, L_cur = a_best.copy(), L_best
        was_reduced = False
        for rep in range(max_restarts):
            m = am.build_model(a_cur, L_cur, act_reach=reach, layer=1.6)
            was_reduced = was_reduced or m["reduced"]
            mi = 180 if m["reduced"] else 65
            a_s, L_s, _, _ = am.run_squeeze_reduced(a_cur, L_cur, K, model=m, trust0=0.012, max_it=mi)
            if am.full_viol(a_s, L_s) <= 1e-11 and L_s < L_cur - 1e-10:
                gain = L_cur - L_s
                a_cur, L_cur = a_s, L_s
                if gain < 1e-9:
                    break
            else:
                break
        if L_cur < L_best - 1e-11 and am.full_viol(a_cur, L_cur) <= 1e-11:
            a_best, L_best = a_cur, L_cur
        if was_reduced:
            break

    # Final full-row kink-safe pslp polish
    a_f, L_f, _, _ = am.run_squeeze_full(a_best, L_best, K, trust0=0.005, max_it=80)
    if am.full_viol(a_f, L_f) <= 1e-11 and L_f < L_best:
        a_best, L_best = a_f, L_f
    return a_best, L_best


def run_lpslack_polish(
    n: int,
    a_cur: np.ndarray,
    L_ref_str: str,
    time_limit: float = 15.0,
) -> tuple[np.ndarray | None, dict | None]:
    """Run lpslack with dual_all (so boundary 0-deg squares can tilt) + dual_non90 + non90."""
    rows = [
        {
            "square_id": str(i),
            "x": repr(float(a_cur[i, 0])),
            "y": repr(float(a_cur[i, 1])),
            "angle": repr(float(a_cur[i, 2])),
        }
        for i in range(n)
    ]
    res = lpslack.optimize_group(
        n,
        rows,
        L_ref_str,
        outer_iters=8,
        fixed_rounds=8,
        slp_modes=("dual_all", "dual_non90", "non90"),
        tr_th_schedule=(1e-3, 3e-4, 8e-5),
        cycles=1,
        time_limit=time_limit,
    )
    if res.get("status") in ("WIN", "TINY_WIN") and res.get("rows"):
        rs = sorted(res["rows"], key=lambda r: int(r["square_id"]))
        ax = np.array([float(r["x"]) for r in rs], dtype=np.float64)
        ay = np.array([float(r["y"]) for r in rs], dtype=np.float64)
        at = np.array([float(r["angle"]) for r in rs], dtype=np.float64)
        a_out, _ = am.normalize_arr(np.column_stack([ax, ay, at]))
        return a_out, res
    return None, None


def _relax_candidate(x: np.ndarray, y: np.ndarray, th: np.ndarray, L_ref: float, pen: float) -> tuple[np.ndarray, float, float]:
    """Relax a slightly infeasible C++ melt candidate into feasibility for pslp."""
    x, y, th, L_cur = ml.normalize(x, y, th)
    if 1e-7 < pen < 0.025:
        L_try = max(L_cur, L_ref) + max(0.002, 1.8 * pen)
        xr, yr, thr, _ = hu.relax_lbfgs(x, y, th, L_try, maxiter=160, w_wall=4.0, w_pair=1.0, side=1.0)
        pen_r, L_r = cpp_eval_sat_pen_bbox(xr, yr, thr, side=1.0)
        if pen_r <= pen + 1e-6:
            x, y, th, pen, L_cur = xr, yr, thr, pen_r, L_r
    if pen < 0.015:
        xs, ys, ts, Ls, ok_s = joint_xytheta_slp(
            x, y, th, rounds=10, tr_th0=0.018, tr_xy0=0.08, margin=1e-8, tol=1e-8, patience=2
        )
        if ok_s:
            x, y, th, L_cur = xs, ys, ts, Ls
            pen, L_cur = cpp_eval_sat_pen_bbox(x, y, th, side=1.0)
    a, L_out = am.normalize_arr(np.column_stack([x, y, th]))
    return a, L_out, am.full_viol(a, L_out)


def _state_dist(a1: np.ndarray, a2: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((a1[:, :2] - a2[:, :2]) ** 2, axis=1) + 0.25 * (a1[:, 2] - a2[:, 2]) ** 2)))


def _run_mechwalk_step(
    n: int,
    a_base: np.ndarray,
    L_base: float,
    K: int,
    n_threads: int,
    seed_offset: int,
) -> list[tuple[np.ndarray, float, str]]:
    """Run 1 round of SVD soft-mode mechanism walk + C++ masked melt -> return feasible states."""
    x, y, th = a_base[:, 0], a_base[:, 1], a_base[:, 2]
    info_l = am.identify_interface_layers(a_base)
    bdry = info_l["layer1"]
    jac, _ = ml.contact_jacobian(x, y, th, contact_tol=1e-5)
    sv, raw_modes = ml.soft_modes(jac, k=16)
    sel = ml.select_diverse_modes(raw_modes, sv, x, y, max_keep=6)
    if not sel:
        return []

    q_list, m_list, ls_list, labels = [], [], [], []
    for m_idx, _, mode_vec, _ in sel[:4]:
        disp, _ = ml.mode_participation(mode_vec, n)
        max_d = float(np.max(disp))
        if max_d < 1e-8:
            continue
        core = disp >= max(0.25 * max_d, float(np.quantile(disp, 0.85)))
        cx_c, cy_c = x[core], y[core]
        d2 = (x[:, None] - cx_c[None, :]) ** 2 + (y[:, None] - cy_c[None, :]) ** 2
        active_bool = (np.min(d2, axis=1) <= 2.4 ** 2) | (bdry & (np.min(d2, axis=1) <= 3.2 ** 2))
        if int(active_bool.sum()) > 0.55 * n:
            active_bool = disp >= float(np.quantile(disp, 0.65))
        mask3 = _mask3(active_bool)
        for step in (0.015, 0.05):
            amp = min(step / max_d, 2.5)
            for sgn, slbl in ((1.0, "p"), (-1.0, "m")):
                wx, wy, wth, Lw = ml.normalize(
                    x + sgn * amp * mode_vec[:n],
                    y + sgn * amp * mode_vec[n : 2 * n],
                    th + sgn * amp * mode_vec[2 * n :],
                )
                L0 = max(Lw, L_base) + 0.008
                q_list.append(full_to_free_sym(wx, wy, wth, L0))
                m_list.append(mask3)
                ls_list.append([L0, L_base + 0.002, L_base + 0.0001, L_base - 0.0015])
                labels.append(f"mw_m{m_idx}_{slbl}_{step:.3g}")

    if not q_list:
        return []

    qb, qf, lf, _, po, lb = run_cpp_masked_melt(
        n, 1, q_list, m_list, ls_list, [4, 6, 8, 10],
        a_start=0.92, n_melt_stages=3, masked_iters=130, full_iters=120,
        n_threads=n_threads, base_seed=seed_offset,
    )
    order = sorted(range(len(labels)), key=lambda i: (0 if lf[i] < 1e9 else 1, float(lf[i]) if lf[i] < 1e9 else float(po[i])))
    out = []
    for i in order[:3]:
        use_f = lf[i] < 1e9
        L_ref = float(lf[i]) if use_f else ls_list[i][-1]
        cx, cy, cth = cpp_expand_sym(n, 1, L_ref, qf[i] if use_f else qb[i])
        a_r, L_r, fv = _relax_candidate(cx, cy, cth, float(lb[i]) if not use_f else L_ref, 0.0 if use_f else float(po[i]))
        if fv <= 1e-11:
            out.append((a_r, L_r, labels[i]))
    return out


def solve_target(
    n: int,
    incumbent: dict,
    n_threads: int = 2,
    enable_mechwalk: bool = True,
) -> tuple[dict, dict | None]:
    """Run full H125 pipeline on target n against incumbent dict."""
    t0 = time.time()
    L_inc_dec: Decimal = incumbent["L_dec"]
    L_b = incumbent["L"]
    K = int(math.floor(L_b))

    a0 = np.column_stack([incumbent["x"], incumbent["y"], incumbent["th"]])
    a0, L0 = am.normalize_arr(a0)

    best_dec = L_inc_dec
    best_win: dict | None = None
    best_a = a0.copy()
    best_L = L0
    best_label = "incumbent"

    def try_update(a_cand: np.ndarray, lbl: str) -> bool:
        nonlocal best_dec, best_win, best_a, best_L, best_label
        ver = pf.certify_state(n, a_cand[:, 0], a_cand[:, 1], a_cand[:, 2])
        if not ver.get("ok"):
            return False
        L_lit = Decimal(ver["L_lit"])
        L_pd = Decimal(ver["L_pandas"])
        L_w = max(L_lit, L_pd)
        if L_w < best_dec:
            best_a = np.column_stack([ver["x"], ver["y"], ver["th"]])
            best_L = float(L_w)
            best_dec = L_w
            best_label = lbl
            if L_lit < incumbent["L_lit"] and L_pd < incumbent["L_pandas"]:
                best_win = {
                    "n": int(n),
                    "label": lbl,
                    "L_lit": ver["L_lit"],
                    "L_pandas": ver["L_pandas"],
                    "L_inc": str(L_inc_dec),
                    "inc_source": incumbent["source"],
                    "dL": float(L_w - L_inc_dec),
                    "rows": ver["rows"],
                }
                pf.save_win_locked(best_win)
                return True
        return False

    print(f"\n=== H125 n={n} (m={K}, r={n-K*K:+d}) L_inc={str(L_inc_dec)[:18]} ({incumbent['source']}) ===", flush=True)

    # Stage 1: Direct 4-corner reduced pslp (act_reach=3.5, 2.6) + full pslp + lpslack(dual_all)
    a_p0, L_p0 = run_pslp_cascade(a0, L0, K, max_restarts=6, reaches=(3.5, 2.6))
    if L_p0 < L0 - 1e-11:
        try_update(a_p0, "direct_pslp")
    a_lp0, _ = run_lpslack_polish(n, best_a, str(best_dec))
    if a_lp0 is not None:
        try_update(a_lp0, "direct_pslp+lpslack")

    # Stage 2: Arm-adjacent 0-deg layer liberation + Masked C++ Interface Melt (from best_a)
    rng = np.random.default_rng(12500 + n)
    q_list, m_list, ls_list, labels = build_interface_melt_seeds(best_a[:, 0], best_a[:, 1], best_a[:, 2], best_L, rng)
    n_feas = 0
    if q_list:
        qb, qf, lf, _, po, lb = run_cpp_masked_melt(
            n, 1, q_list, m_list, ls_list, [4, 6, 8, 10],
            a_start=0.94, n_melt_stages=3, masked_iters=140, full_iters=120,
            n_threads=n_threads, base_seed=55000 + n,
        )
        n_feas = int(np.sum(lf < 1e9))
        order = sorted(
            range(len(labels)),
            key=lambda i: (0 if lf[i] < 1e9 else 1, float(lf[i]) if lf[i] < 1e9 else float(po[i])),
        )
        picked: list[tuple[np.ndarray, float, str]] = []
        for i in order:
            if len(picked) >= 3:
                break
            use_f = lf[i] < 1e9
            if not use_f and float(po[i]) > 0.015:
                continue
            L_ref = float(lf[i]) if use_f else ls_list[i][-1]
            cx, cy, cth = cpp_expand_sym(n, 1, L_ref, qf[i] if use_f else qb[i])
            a_r, L_r, fv = _relax_candidate(cx, cy, cth, float(lb[i]) if not use_f else L_ref, 0.0 if use_f else float(po[i]))
            if fv <= 1e-11 and all(_state_dist(a_r, p[0]) > 3e-4 for p in picked):
                picked.append((a_r, L_r, labels[i]))

        for a_r, L_r, lbl in picked:
            a_ps, L_ps = run_pslp_cascade(a_r, L_r, K, max_restarts=5, reaches=(3.5, 2.6))
            if L_ps < float(best_dec) - 1e-10:
                try_update(a_ps, f"{lbl}+pslp")
                a_lp, _ = run_lpslack_polish(n, best_a, str(best_dec))
                if a_lp is not None:
                    try_update(a_lp, f"{lbl}+pslp+lpslack")

    # Stage 3: mechwalk + pslp + lpslack on the current best state
    if enable_mechwalk:
        mw_cands = _run_mechwalk_step(n, best_a, best_L, K, n_threads=n_threads, seed_offset=77000 + n)
        for a_m, L_m, lbl in mw_cands[:2]:
            a_ps, L_ps = run_pslp_cascade(a_m, L_m, K, max_restarts=5, reaches=(3.5, 2.6))
            if L_ps < float(best_dec) - 1e-10:
                try_update(a_ps, f"{lbl}+pslp")
                a_lp, _ = run_lpslack_polish(n, best_a, str(best_dec))
                if a_lp is not None:
                    try_update(a_lp, f"{lbl}+pslp+lpslack")

    dL_val = float(best_dec - L_inc_dec)
    is_real_win = best_win is not None and (L_inc_dec - best_dec >= Decimal("1e-9"))
    status = "REAL_WIN" if is_real_win else ("TINY_WIN" if best_win is not None else "MISS")
    print(
        f"  [{status}] n={n}: best={best_label} dL={dL_val:+.6e} L={str(best_dec)[:18]} "
        f"melt_feas={n_feas}/{len(q_list)} [{time.time()-t0:.1f}s]",
        flush=True,
    )
    summary = {
        "n": int(n),
        "m": int(K),
        "r": int(n - K * K),
        "L_inc": str(L_inc_dec),
        "inc_source": incumbent["source"],
        "L_best": str(best_dec),
        "dL": dL_val,
        "status": status,
        "best_label": best_label,
        "melt_seeds": len(q_list),
        "melt_feas": n_feas,
        "dt_s": round(time.time() - t0, 2),
    }
    return summary, best_win
