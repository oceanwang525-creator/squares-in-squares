#!/usr/bin/env python3
"""Hypothesis H77 (b7xplant): Standalone diagonal-mode brick_mechwalk on g=+6,+7,+8 bricks + zero-overlap pinwheels."""
from __future__ import annotations

import csv
import fcntl
import io
import json
import math
import os
import sys
import warnings
from decimal import Decimal
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
B7_DIR = REPO_ROOT / "poc" / "swarm_b7xplant"
DATA_DIR = B7_DIR / "data"
DOCS_DIR = B7_DIR / "docs"
FINAL_CSV = B7_DIR / "final.csv"
MERGED_CSV = REPO_ROOT / "poc" / "pell_corrugation" / "data" / "border-scan" / "merged.csv"

AUTO_UNION_POC = REPO_ROOT / "poc" / "auto_union" / "poc"
B8_POC = REPO_ROOT / "poc" / "swarm_b8xplant" / "poc"
B5_POC = REPO_ROOT / "poc" / "swarm_b5xplant" / "poc"
LPSLACK_POC = REPO_ROOT / "poc" / "swarm_lpslack" / "poc"
HIGHD_POC = REPO_ROOT / "poc" / "swarm_highd" / "poc"
NEW159_POC = REPO_ROOT / "poc" / "swarm_new159" / "poc"
CASCADES_POC = REPO_ROOT / "poc" / "swarm_cascades" / "poc"
GAD_POC = REPO_ROOT / "poc" / "rectangle_gadgets" / "poc"
BASELINE_SRC = REPO_ROOT / "poc" / "baseline" / "poc" / "src"

for _p in (
    AUTO_UNION_POC,
    B8_POC,
    LPSLACK_POC,
    HIGHD_POC,
    NEW159_POC,
    CASCADES_POC,
    GAD_POC,
    BASELINE_SRC,
):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import b8_lib as bl  # noqa: E402
import brick_rect_lib as bslp  # noqa: E402
import lpslack  # noqa: E402
import slp_utils as su  # noqa: E402
import union as auto_union  # noqa: E402
import utils as hu  # noqa: E402
from cpp_wrap import cpp_eval_sat_pen_bbox  # noqa: E402
from joint_slp import joint_xytheta_slp  # noqa: E402


def make_brick(
    w: int,
    h: int,
    g: int,
    x: np.ndarray,
    y: np.ndarray,
    th: np.ndarray,
    src_n: int = 0,
    tag: str = "",
) -> dict | None:
    """Validate and normalize a rectangular sub-brick."""
    if g <= 0 or w <= 0 or h <= 0 or len(x) != w * h + g:
        return None
    bx, by, W, H = bl.rect_bbox(x, y, th)
    pen, _ = cpp_eval_sat_pen_bbox(bx, by, th)
    if pen > 1e-7:
        return None
    fx, fy = W - float(w), H - float(h)
    f = max(fx, fy)
    if f >= 1.0 - 1e-7:
        return None
    return {
        "w": int(w),
        "h": int(h),
        "g": int(g),
        "x": bx,
        "y": by,
        "th": np.asarray(th, dtype=np.float64).copy(),
        "fx": float(fx),
        "fy": float(fy),
        "f": float(f),
        "src_n": int(src_n),
        "tag": str(tag),
    }


def load_portfolio_targets(target_ns: list[int] | set[int]) -> dict[int, dict]:
    """Fast fail-closed loader for the full origin/main portfolio (merged.csv + all poc/*/final.csv)."""
    need = {int(n) for n in target_ns}
    cands: dict[int, list[tuple[str, list[dict], list[dict] | None]]] = {n: [] for n in need}

    with MERGED_CSV.open(newline="", encoding="utf-8") as fh:
        tmp: dict[int, list[dict]] = {}
        for row in csv.DictReader(fh):
            n = int(row["n"])
            if n in need:
                tmp.setdefault(n, []).append(
                    {
                        "square_id": int(row["square_id"]),
                        "x": row["x"],
                        "y": row["y"],
                        "angle": row["angle"],
                    }
                )
        for n, rows in tmp.items():
            rows.sort(key=lambda r: int(r["square_id"]))
            cands[n].append(("merged.csv", rows, None))

    prefixes = tuple(f"{n:03}_" if n < 1000 else "1000_" for n in need)
    for p in sorted((REPO_ROOT / "poc").glob("*/final.csv")):
        if p.parent.name == "swarm_b7xplant":
            continue
        lines = p.read_text(encoding="utf-8").splitlines()
        if len(lines) <= 1:
            continue
        sub = [lines[0]] + [ln for ln in lines[1:] if ln.startswith(prefixes)]
        if len(sub) <= 1:
            continue
        txt = "\n".join(sub) + "\n"
        try:
            a = auto_union.parse_text(txt)
            b = auto_union.parse_text(pd.read_csv(io.StringIO(txt)).to_csv(index=False))
        except Exception:
            continue
        for n, rows in a.items():
            if n in need and len(rows) == n and len(b.get(n, [])) == n:
                rows.sort(key=lambda r: int(r["square_id"]))
                b[n].sort(key=lambda r: int(r["square_id"]))
                cands[n].append((str(p.relative_to(REPO_ROOT)), rows, b[n]))

    best: dict[int, dict] = {}
    for n in sorted(need):
        scored = []
        for src, r_lit, r_pd in cands[n]:
            x = np.array([float(r["x"]) for r in r_lit], dtype=np.float64)
            y = np.array([float(r["y"]) for r in r_lit], dtype=np.float64)
            th = np.array([float(r["angle"]) for r in r_lit], dtype=np.float64)
            x, y, th, Lf = bl.normalize(x, y, th)
            scored.append((Lf, src, r_lit, r_pd, x, y, th))
        scored.sort(key=lambda t: t[0])
        min_Lf = scored[0][0]
        best_dec: Decimal | None = None
        best_entry: dict | None = None
        for Lf, src, r_lit, r_pd, x, y, th in scored:
            if Lf > min_Lf + 1e-6 and best_dec is not None:
                break
            if r_pd is not None:
                res = auto_union.check((n, r_lit, r_pd))
                if res is None:
                    continue
                Ld = max(Decimal(res[0]), Decimal(res[1]))
            else:
                Ld = Decimal(auto_union.certify(n, r_lit))
            if best_dec is None or Ld < best_dec:
                best_dec = Ld
                best_entry = {
                    "x": x,
                    "y": y,
                    "th": th,
                    "L": Lf,
                    "L_dec": Ld,
                    "rows": r_lit,
                    "source": src,
                }
        if best_entry is None:
            raise RuntimeError(f"No valid incumbent found for n={n}")
        best[n] = best_entry
    return best


def extract_corner_cut(
    x: np.ndarray,
    y: np.ndarray,
    th: np.ndarray,
    L: float,
    corner: str,
    w: int,
    h: int,
    g_exp: int,
    src_n: int,
    tag: str,
    margin: float | None = None,
) -> dict | None:
    """Extract a (w, h, g_exp) sub-brick from a corner of a packing."""
    hx = bl.half_extent(th)
    m_int = int(math.floor(L + 1e-9))
    f_pack = L - m_int
    if margin is None:
        margin = f_pack + 0.045
    if corner == "TL":
        m = (x + hx <= w + margin) & (y - hx >= L - (h + margin))
    elif corner == "TR":
        m = (x - hx >= L - (w + margin)) & (y - hx >= L - (h + margin))
    elif corner == "BL":
        m = (x + hx <= w + margin) & (y + hx <= h + margin)
    elif corner == "BR":
        m = (x - hx >= L - (w + margin)) & (y + hx <= h + margin)
    else:
        raise ValueError(f"Unknown corner {corner}")
    if int(m.sum()) != w * h + g_exp:
        return None
    return make_brick(w, h, g_exp, x[m], y[m], th[m], src_n=src_n, tag=tag)


def compress_brick_slp(b: dict, n_trials: int = 10, seed: int = 7701) -> dict:
    """Gentle downward L-BFGS + joint_xytheta_slp_rect + fixed_angle_lp_tighten_rect."""
    w, h, g = b["w"], b["h"], b["g"]
    x0, y0, t0 = b["x"].copy(), b["y"].copy(), b["th"].copy()
    W0, H0 = w + b["fx"], h + b["fy"]
    f0 = b["f"]
    fmin = min(b["fx"], b["fy"])
    rng = np.random.default_rng(seed)
    best = b
    tilt = np.minimum(np.mod(t0, 0.5 * math.pi), 0.5 * math.pi - np.mod(t0, 0.5 * math.pi)) > math.radians(2.0)

    if f0 > 0.67:
        ft_sched = (f0 - 0.002, max(0.665, f0 - 0.012), max(0.650, f0 - 0.025))
    else:
        ft_sched = (f0, max(0.635, fmin + 0.002), max(0.634, fmin - 0.004))

    for trial in range(n_trials):
        cx, cy, ct = x0.copy(), y0.copy(), t0.copy()
        if trial > 0:
            if trial % 3 == 1:
                mask = ~tilt
            elif trial % 3 == 2:
                mask = (cy < 2.2) | (cy > H0 - 2.2) | (cx < 2.2) | (cx > W0 - 2.2)
            else:
                mask = np.ones(len(cx), dtype=bool)
            nm = int(mask.sum())
            if nm > 0:
                amp = 0.04 if trial % 3 != 0 else 0.02
                cx[mask] += rng.uniform(-amp, amp, nm)
                cy[mask] += rng.uniform(-amp, amp, nm)
                ct[mask] += rng.uniform(-1.2 * amp, 1.2 * amp, nm)
            for ft in ft_sched:
                cx, cy, ct, _ = bslp.relax_rect_lbfgs(cx, cy, ct, w + ft, h + ft, maxiter=220)
        sx, sy, st, _, ok = bslp.joint_xytheta_slp_rect(
            cx, cy, ct, w, h, rounds=14, tr_th0=0.022, tr_xy0=0.09, tol=1e-11
        )
        if ok:
            tx, ty, tf, tok = bslp.fixed_angle_lp_tighten_rect(sx, sy, st, w, h, rounds=5, margin=1e-8)
            if tok and tf < max(bslp.rect_bbox(sx, sy, st)[2] - w, bslp.rect_bbox(sx, sy, st)[3] - h):
                sx, sy = tx, ty
            cb = make_brick(w, h, g, sx, sy, st, src_n=b.get("src_n", 0), tag=b["tag"])
            if cb is not None and cb["f"] < best["f"] - 1e-10:
                best = cb
    return best


def _try_brick_direction(
    best_state: tuple[np.ndarray, np.ndarray, np.ndarray],
    v: np.ndarray,
    w: int,
    h: int,
    best_f: float,
    amps: tuple[float, ...],
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], float, float | None]:
    nf = len(best_state[0])
    for amp in amps:
        for sgn in (1.0, -1.0):
            cx = best_state[0] + sgn * amp * v[:nf]
            cy = best_state[1] + sgn * amp * v[nf : 2 * nf]
            ct = best_state[2] + sgn * amp * v[2 * nf :]
            rx, ry, rt = None, None, None
            for df_try in (-1.5e-3, -2e-4, 2e-4):
                tx, ty, tt, pen = bslp.relax_rect_lbfgs(
                    cx, cy, ct, w + best_f + df_try, h + best_f + df_try, maxiter=280
                )
                if pen <= 1e-6:
                    rx, ry, rt = tx, ty, tt
                    break
            if rx is None:
                continue
            sx, sy, st, sf, ok = bslp.joint_xytheta_slp_rect(rx, ry, rt, w, h, rounds=8)
            if ok and sf < best_f - 1e-8:
                return (sx, sy, st), sf, sgn * amp
    return best_state, best_f, None


def standalone_brick_walk_diag(b: dict, name: str, rounds: int = 3, do_slp_pre: bool = True) -> dict:
    """Run compress_brick_slp + standalone Contact-Jacobian SVD walk with diagonal mode pairs d(i+-j)."""
    w, h, g = b["w"], b["h"], b["g"]
    f0 = b["f"]
    cur = compress_brick_slp(b, n_trials=8, seed=7700 + len(b["x"])) if do_slp_pre else b
    best_f = cur["f"]
    best_state = (cur["x"].copy(), cur["y"].copy(), cur["th"].copy())
    free_all = np.ones(len(best_state[0]), dtype=bool)
    inv_s2 = 1.0 / math.sqrt(2.0)

    if best_f < f0 - 1e-8:
        print(f"  [BRICK {name}] pre-SLP: {f0:.9f} -> {best_f:.9f} (df={best_f-f0:+.6e})", flush=True)

    for rnd in range(rounds):
        improved = False
        for ctol in (1e-5, 4e-4, 2e-3):
            jac, _, _ = bl.localized_contact_jacobian(
                best_state[0], best_state[1], best_state[2], free_all, contact_tol=ctol
            )
            if jac.shape[0] == 0:
                continue
            _, raw_modes = bl.localized_soft_modes(jac, k=14)
            kept: list[np.ndarray] = []
            for m in raw_modes:
                v = m.astype(np.float64).copy()
                nrm = float(np.linalg.norm(v))
                if nrm < 1e-10:
                    continue
                v /= nrm
                if any(abs(float(np.dot(v, prev))) > 0.94 for prev in kept):
                    continue
                kept.append(v)
                if len(kept) >= 10:
                    break

            for mi, v in enumerate(kept):
                new_state, new_f, hit_amp = _try_brick_direction(
                    best_state, v, w, h, best_f, (0.004, 0.010, 0.025, 0.07)
                )
                if hit_amp is not None:
                    best_f = new_f
                    best_state = new_state
                    improved = True
                    print(
                        f"  [BRICK {name}] rnd={rnd} ctol={ctol:.0e} m{mi} amp={hit_amp} -> f={best_f:.9f} (df={best_f-f0:+.6e})",
                        flush=True,
                    )

            kcap = min(6, len(kept))
            for i in range(kcap):
                for j in range(i + 1, kcap):
                    for d_sgn, dtag in ((1.0, "+"), (-1.0, "-")):
                        vd = inv_s2 * (kept[i] + d_sgn * kept[j])
                        new_state, new_f, hit_amp = _try_brick_direction(
                            best_state, vd, w, h, best_f, (0.008, 0.025, 0.06)
                        )
                        if hit_amp is not None:
                            best_f = new_f
                            best_state = new_state
                            improved = True
                            print(
                                f"  [BRICK {name}] rnd={rnd} ctol={ctol:.0e} d({i}{dtag}{j}) amp={hit_amp} -> f={best_f:.9f} (df={best_f-f0:+.6e})",
                                flush=True,
                            )
        if not improved:
            break

    # Final tight rectangular SLP + fixed-angle LP tighten
    sx, sy, st, sf, ok = bslp.joint_xytheta_slp_rect(
        best_state[0], best_state[1], best_state[2], w, h, rounds=16, tr_th0=0.015, tr_xy0=0.06, tol=1e-12
    )
    if ok and sf < best_f:
        best_state = (sx, sy, st)
        best_f = sf

    bx, by, Wf, Hf = bl.rect_bbox(best_state[0], best_state[1], best_state[2])
    fx, fy = Wf - w, Hf - h
    best_f = max(fx, fy)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_p = DATA_DIR / f"brick_{name}.json"
    payload = {
        "name": name,
        "w": int(w),
        "h": int(h),
        "g": int(g),
        "src_n": int(b.get("src_n", 0)),
        "f0": float(f0),
        "fx": float(fx),
        "fy": float(fy),
        "f": float(best_f),
        "df": float(best_f - f0),
        "x": bx.tolist(),
        "y": by.tolist(),
        "th": best_state[2].tolist(),
    }
    out_p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {
        "name": name,
        "tag": name,
        "w": int(w),
        "h": int(h),
        "g": int(g),
        "src_n": int(b.get("src_n", 0)),
        "f0": float(f0),
        "fx": float(fx),
        "fy": float(fy),
        "f": float(best_f),
        "df": float(best_f - f0),
        "x": bx,
        "y": by,
        "th": best_state[2].copy(),
    }


def polish_and_certify_b7(
    n: int,
    x: np.ndarray,
    y: np.ndarray,
    th: np.ndarray,
    inc_dec: Decimal,
    slp_rounds: int = 20,
    run_lpslack: bool = True,
) -> dict | None:
    """Polish with joint_xytheta_slp_tight + fixed_angle_lp_tighten + optional lpslack, then dual-certify."""
    x, y, th, _ = bl.normalize(x, y, th)
    sx, sy, st, _, ok = bslp.joint_xytheta_slp_tight(x, y, th)
    if not ok:
        sx, sy, st, _, ok = joint_xytheta_slp(
            x, y, th, rounds=slp_rounds, tr_th0=0.025, tr_xy0=0.10, margin=1e-8, tol=1e-10, patience=3
        )
    if not ok:
        pen0, _ = cpp_eval_sat_pen_bbox(x, y, th)
        if pen0 > 1e-10:
            return None
        sx, sy, st = x, y, th

    cands = []
    for margin in (1e-7, 2e-8, 5e-9):
        fx, fy, _, fixed_ok = hu.fixed_angle_lp_tighten(sx, sy, st, rounds=6, margin=margin)
        if fixed_ok:
            cert = bl.certify_canonical(n, fx, fy, st)
            if cert["ok"]:
                cands.append(cert)
    cert_raw = bl.certify_canonical(n, sx, sy, st)
    if cert_raw["ok"]:
        cands.append(cert_raw)
    if not cands:
        for eps in (1e-10, 1e-9):
            cert_sc = bl.certify_canonical(n, sx * (1.0 + eps), sy * (1.0 + eps), st)
            if cert_sc["ok"]:
                cands.append(cert_sc)
                break
    if not cands:
        return None

    best = min(cands, key=lambda c: max(Decimal(c["L_lit"]), Decimal(c["L_pandas"])))

    if run_lpslack:
        L_best_now = max(Decimal(best["L_lit"]), Decimal(best["L_pandas"]))
        # Run lpslack alternating SLP + unbounded LP + uniform shrink from best feasible state
        if L_best_now < inc_dec + Decimal("2e-3"):
            rows_in = sorted(best["rows"], key=lambda r: int(r["square_id"]))
            lp_res = lpslack.optimize_group(
                n,
                rows_in,
                str(L_best_now),
                outer_iters=6,
                fixed_rounds=10,
                slp_modes=("dual_non90", "non90"),
                tr_th_schedule=(4e-4, 1e-4, 3e-5),
                time_limit=12.0,
            )
            if lp_res.get("rows"):
                # Re-certify via canonical final_text in literal + pandas modes
                rs_lp = sorted(lp_res["rows"], key=lambda r: int(r["square_id"]))
                lx = np.array([float(r["x"]) for r in rs_lp], dtype=np.float64)
                ly = np.array([float(r["y"]) for r in rs_lp], dtype=np.float64)
                lt = np.array([float(r["angle"]) for r in rs_lp], dtype=np.float64)
                # Also verify high-precision rows directly via auto_union.final_text
                text = auto_union.final_text({n: rs_lp})
                lit = auto_union.parse_text(text)
                pd_t = auto_union.parse_text(pd.read_csv(io.StringIO(text)).to_csv(index=False))
                res_chk = auto_union.check((n, lit[n], pd_t[n]))
                if res_chk is not None:
                    L_lp = max(Decimal(res_chk[0]), Decimal(res_chk[1]))
                    if L_lp < L_best_now:
                        best = {
                            "ok": True,
                            "L_lit": res_chk[0],
                            "L_pandas": res_chk[1],
                            "L_dec_str": str(L_lp),
                            "L_float": float(L_lp),
                            "rows": rs_lp,
                        }

    L_worst = max(Decimal(best["L_lit"]), Decimal(best["L_pandas"]))
    dL = L_worst - inc_dec
    best["n"] = int(n)
    best["dL"] = float(dL)
    best["is_real_win"] = bool(dL <= Decimal("-1e-9"))
    return best


def save_win_b7(win_entry: dict) -> bool:
    """Thread/process-safe save to poc/swarm_b7xplant/data/win_n{n}.json and final.csv."""
    return bl.save_win_locked(win_entry, data_dir=DATA_DIR, final_path=FINAL_CSV)


def write_header_only_final_csv() -> None:
    FINAL_CSV.parent.mkdir(parents=True, exist_ok=True)
    FINAL_CSV.write_text("id,x,y,deg\n", encoding="utf-8")
