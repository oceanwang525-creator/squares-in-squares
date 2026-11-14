#!/usr/bin/env python3
"""H131 third-generation masked-pocket melt gate and checkpointed runner."""
from __future__ import annotations

import argparse
from decimal import Decimal
import importlib.util
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import tempfile
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("BLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
HIGHD_POC = REPO_ROOT / "poc" / "swarm_highd" / "poc"
BASELINE_SRC = REPO_ROOT / "poc" / "baseline" / "poc" / "src"
for p in (Path(__file__).resolve().parent, HIGHD_POC, BASELINE_SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pocket_seeds as ps  # noqa: E402
import utils as u  # noqa: E402
from cpp_wrap import cpp_eval_sat_pen_bbox, cpp_expand_sym, run_cpp_masked_melt  # noqa: E402
from joint_slp import joint_xytheta_slp  # noqa: E402
from squares.core import certify, kaggle_roundtrip, read_submission  # noqa: E402

MERGED_CSV = REPO_ROOT / "poc" / "pell_corrugation" / "data" / "border-scan" / "merged.csv"
H128_POC = REPO_ROOT / "poc" / "swarm_masked_melt_bootstrap"
H128_DATA = H128_POC / "data"
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "data"
DOCS_DIR = HERE.parent / "docs"
CKPT_DIR = DOCS_DIR / "ckpt"
FINAL_CSV = HERE.parent / "final.csv"

GATE_TARGETS = [861, 926, 695]
EXTENSION_TARGETS = [634]


def _load_union_mod():
    spec = importlib.util.spec_from_file_location("auto_union_mod", REPO_ROOT / "poc/auto_union/poc/union.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def certify_canonical(n: int, x: np.ndarray, y: np.ndarray, th: np.ndarray) -> dict:
    """Serialize via canonical union.final_text and certify BOTH literal and pandas roundtrip."""
    union_mod = _load_union_mod()
    hx = 0.5 * (np.abs(np.cos(th)) + np.abs(np.sin(th)))
    sx = x - float((x - hx).min()) + 1e-12
    sy = y - float((y - hx).min()) + 1e-12
    rows = [
        {
            "square_id": str(i),
            "x": repr(float(sx[i])),
            "y": repr(float(sy[i])),
            "angle": repr(float(th[i])),
        }
        for i in range(n)
    ]
    text = union_mod.final_text({n: rows})
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "cand.csv"
        p.write_text(text, encoding="utf-8")
        lit_groups, pd_groups = union_mod.load(p)
        res = union_mod.check((n, lit_groups[n], pd_groups[n]))
    if res is None:
        return {"ok": False, "L_lit": None, "L_pandas": None}
    L_lit, L_pd = res[0], res[1]
    return {
        "ok": True,
        "L_lit": L_lit,
        "L_pandas": L_pd,
        "L_float": float(max(Decimal(L_lit), Decimal(L_pd))),
        "rows": rows,
    }


def _polish_worker(args: tuple) -> dict:
    idx, label, n, L_base, x, y, th, cpp_pen, cpp_L = args
    t0 = time.time()
    hx = 0.5 * (np.abs(np.cos(th)) + np.abs(np.sin(th)))
    x = x - float((x - hx).min())
    y = y - float((y - hx).min())

    if 1e-7 < cpp_pen < 0.025:
        L_try = cpp_L + max(0.002, 1.8 * cpp_pen)
        xr, yr, thr, _ = u.relax_lbfgs(x, y, th, L_try, maxiter=160, w_wall=4.0, w_pair=1.0, side=1.0)
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
            if L_slp < L_base + 5e-4:
                xs2, ys2, ts2, Ls2, ok_s2 = joint_xytheta_slp(
                    x_f, y_f, th_f, rounds=12, tr_th0=0.006, tr_xy0=0.025, margin=1e-8, tol=1e-11, patience=2
                )
                if ok_s2 and Ls2 < L_slp:
                    x_f, y_f, th_f, L_slp = xs2, ys2, ts2, Ls2

    ver: dict = {"ok": False}
    best_ver_dec: Decimal | None = None
    for m_lp in (1e-7, 2e-8, 5e-9, 1e-6):
        xf2, yf2, L_lp2, ok_lp2 = u.fixed_angle_lp_tighten(x_f, y_f, th_f, rounds=5, margin=m_lp)
        if ok_lp2 and L_lp2 <= L_slp + 5e-6:
            if L_lp2 < L_base - 1e-12 or not ver.get("ok"):
                cand_ver = certify_canonical(n, xf2, yf2, th_f)
                if cand_ver.get("ok"):
                    c_dec = max(Decimal(cand_ver["L_lit"]), Decimal(cand_ver["L_pandas"]))
                    if best_ver_dec is None or c_dec < best_ver_dec:
                        best_ver_dec = c_dec
                        x_f, y_f, L_slp, ok_slp, ver = xf2, yf2, L_lp2, True, cand_ver

    pen_fin, L_fin = cpp_eval_sat_pen_bbox(x_f, y_f, th_f, side=1.0)
    return {
        "idx": idx,
        "label": label,
        "n": n,
        "cpp_pen": float(cpp_pen),
        "cpp_L": float(cpp_L),
        "ok_slp": bool(ok_slp),
        "pen_fin": float(pen_fin),
        "L_fin": float(L_fin),
        "verify": {k: v for k, v in ver.items() if k != "rows"},
        "rows": ver.get("rows"),
        "dt": round(time.time() - t0, 2),
    }


def load_seed_geometry(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Load the complete H128 child group used as the generation-3 geometry."""
    path = H128_DATA / f"win_n{n}.csv"
    if not path.exists():
        raise FileNotFoundError(f"H131 requires H128 child geometry: {path}")
    frame = pd.read_csv(path)
    if list(frame.columns) != ["id", "x", "y", "deg"] or len(frame) != n:
        raise ValueError(f"invalid H128 source group: {path}")
    frame["index"] = frame["id"].astype(str).str.rsplit("_", n=1).str[-1].astype(int)
    frame = frame.sort_values("index")
    if frame["index"].tolist() != list(range(n)):
        raise ValueError(f"incomplete or duplicate H128 source group: {path}")
    strip = lambda values: values.astype(str).str.removeprefix("s").to_numpy(np.float64)
    x = strip(frame["x"])
    y = strip(frame["y"])
    th = np.radians(strip(frame["deg"]))
    hx = 0.5 * (np.abs(np.cos(th)) + np.abs(np.sin(th)))
    x -= float((x - hx).min())
    y -= float((y - hx).min())
    L = float(max((x + hx).max(), (y + hx).max()))
    return x, y, th, L


def load_bases(
    n: int, base_puzzles: dict
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, float,
    Decimal, Decimal, Decimal, Decimal,
]:
    """Return H128 geometry plus upstream and H128 literal/pandas limits."""
    x, y, th, seed_L = load_seed_geometry(n)
    upstream_lit = Decimal(certify(n, base_puzzles[n]))
    rt = kaggle_roundtrip({n: base_puzzles[n]})
    upstream_pd = Decimal(certify(n, rt[n]))

    union_mod = _load_union_mod()
    h128_path = H128_DATA / f"win_n{n}.csv"
    h128_lit_groups, h128_pd_groups = union_mod.load(h128_path)
    result = union_mod.check((n, h128_lit_groups[n], h128_pd_groups[n]))
    if result is None:
        raise ValueError(f"H128 source group does not certify: n={n}")
    h128_lit, h128_pd = map(Decimal, result)
    if h128_lit >= upstream_lit or h128_pd >= upstream_pd:
        raise ValueError(f"H128 source is not a strict upstream win: n={n}")
    return x, y, th, seed_L, upstream_lit, upstream_pd, h128_lit, h128_pd


def solve_target(
    n: int,
    x_b: np.ndarray,
    y_b: np.ndarray,
    th_b: np.ndarray,
    L_b: float,
    upstream_lit: Decimal,
    upstream_pd: Decimal,
    h128_lit: Decimal,
    h128_pd: Decimal,
    n_threads: int,
    masked_iters: int = 220,
    full_iters: int = 220,
    a_start: float = 0.81,
    top_k: int = 8,
    rng_offset: int = 131100,
    base_seed_offset: int = 131700,
    l_expand: float = 0.042,
    generation: str = "H131",
) -> dict:
    t0 = time.time()
    # H128 solved these child geometries. Generation 3 deliberately uses a new
    # deterministic stream, wider masked neighborhoods, and a five-stage retreat.
    rng = np.random.default_rng(rng_offset + n)
    q_list, m_list, ls_list, labels = ps.build_h131_seeds(
        n, x_b, y_b, th_b, L_b, rng, L_expand=l_expand
    )
    ils_iters = [6, 10, 15, 19, 22]

    qb, qf, lf, _, po, lb = run_cpp_masked_melt(
        n, 1, q_list, m_list, ls_list, ils_iters,
        a_start=a_start, n_melt_stages=4, masked_iters=masked_iters, full_iters=full_iters,
        n_threads=n_threads, base_seed=base_seed_offset + n,
    )
    n_feas = int(np.sum(lf < 1e9))
    best_feas = float(np.min(lf)) if n_feas else None
    print(
        f"\n=== {generation} n={n} | L_seed={L_b:.9f} | N={len(labels)} feas={n_feas} "
        f"best_Lf={best_feas} min_pen={float(np.min(po)):.2e} ===",
        flush=True,
    )

    score_fn = lambda i: (0 if lf[i] < 1e9 else 1, float(lf[i]) if lf[i] < 1e9 else float(po[i]))
    order = sorted(range(len(labels)), key=score_fn)
    keep, used = [], set()
    for pref, qta in (("mpk_void_", 4), ("mpk_hot_", 3), ("mpk_seam_", 2)):
        c = 0
        for i in order:
            if i not in used and labels[i].startswith(pref):
                keep.append(i)
                used.add(i)
                c += 1
                if c >= qta or len(keep) >= top_k:
                    break
    for i in order:
        if len(keep) >= top_k:
            break
        if i not in used:
            keep.append(i)
            used.add(i)

    tasks = []
    for i in keep:
        use_f = lf[i] < 1e9
        L_ref = float(lf[i]) if use_f else ls_list[i][-1]
        x, y, th = cpp_expand_sym(n, 1, L_ref, qf[i] if use_f else qb[i])
        tasks.append((i, labels[i], n, L_b, x, y, th, 0.0 if use_f else float(po[i]), float(lf[i]) if use_f else float(lb[i])))

    with mp.Pool(processes=min(4, max(1, n_threads), len(tasks))) as pool:
        polished = pool.map(_polish_worker, tasks)
    polished.sort(key=lambda r: (0 if r["verify"].get("ok") else 1, float(r["verify"].get("L_float", 1e9)) if r["verify"].get("ok") else r["pen_fin"]))

    best_valid, best_real = None, None
    best_key: tuple[Decimal, Decimal] | None = None
    for r in polished:
        v = r["verify"]
        if v.get("ok"):
            up_dl_lit = Decimal(v["L_lit"]) - upstream_lit
            up_dl_pd = Decimal(v["L_pandas"]) - upstream_pd
            h_dl_lit = Decimal(v["L_lit"]) - h128_lit
            h_dl_pd = Decimal(v["L_pandas"]) - h128_pd
            key = (max(up_dl_lit, up_dl_pd), max(h_dl_lit, h_dl_pd))
            if best_key is None or key < best_key:
                best_valid = r
                best_key = key
            upstream_real = key[0] <= Decimal("-1e-9")
            h128_real = h_dl_lit < 0 and h_dl_pd < 0
            if upstream_real and h128_real:
                if best_real is None:
                    best_real = r
                else:
                    current = best_real["verify"]
                    current_key = (
                        max(
                            Decimal(current["L_lit"]) - upstream_lit,
                            Decimal(current["L_pandas"]) - upstream_pd,
                        ),
                        max(
                            Decimal(current["L_lit"]) - h128_lit,
                            Decimal(current["L_pandas"]) - h128_pd,
                        ),
                    )
                    if key < current_key:
                        best_real = r

    status = "MISS"
    up_dL_val: float | None = None
    h128_dL_val: float | None = None
    reported = best_real if best_real is not None else best_valid
    if reported is not None:
        v = reported["verify"]
        up_dL_val = float(max(
            Decimal(v["L_lit"]) - upstream_lit,
            Decimal(v["L_pandas"]) - upstream_pd,
        ))
        h128_dL_val = float(max(
            Decimal(v["L_lit"]) - h128_lit,
            Decimal(v["L_pandas"]) - h128_pd,
        ))
        if best_real is not None:
            status = "REAL_WIN"
        elif up_dL_val < 0 and h128_dL_val < 0:
            status = "TIE"
        print(
            f"  [{status}] n={n}: {reported['label']} "
            f"dUp={up_dL_val:+.6e} dH128={h128_dL_val:+.6e} "
            f"L_lit={v['L_lit'][:22]} [{time.time()-t0:.1f}s]",
            flush=True,
        )
    else:
        print(
            f"  [NO VALID POLISHED] n={n} min_pen_fin={min(r['pen_fin'] for r in polished):.2e} [{time.time()-t0:.1f}s]",
            flush=True,
        )

    return {
        "n": n,
        "L_H128_seed": L_b,
        "upstream_L_lit": str(upstream_lit),
        "upstream_L_pd": str(upstream_pd),
        "H128_L_lit": str(h128_lit),
        "H128_L_pd": str(h128_pd),
        "dt_s": round(time.time() - t0, 2),
        "n_starts": len(labels),
        "n_feas": n_feas,
        "best_cpp_Lf": best_feas,
        "min_cpp_pen": float(np.min(po)),
        "status": status,
        "best_dL_upstream": up_dL_val,
        "best_dL_H128": h128_dL_val,
        "best_label": reported["label"] if reported is not None else None,
        "best_L_lit": reported["verify"]["L_lit"] if reported is not None else None,
        "best_L_pd": reported["verify"]["L_pandas"] if reported is not None else None,
        "top_polished": [{k: v for k, v in r.items() if k != "rows"} for r in polished],
        "win": best_real,
    }


def write_final_csv(wins_by_n: dict[int, dict], out_path: Path = FINAL_CSV) -> None:
    """Write ONLY certified REAL WINs (dL <= -1e-9); header-only if none."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not wins_by_n:
        out_path.write_text("id,x,y,deg\n", encoding="utf-8")
        return
    union_mod = _load_union_mod()
    groups = {n: wins_by_n[n]["rows"] for n in sorted(wins_by_n)}
    out_path.write_text(union_mod.final_text(groups), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", type=int, nargs="+", default=GATE_TARGETS)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--masked-iters", type=int, default=220)
    ap.add_argument("--full-iters", type=int, default=220)
    ap.add_argument("--a-start", type=float, default=0.81)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--stop-consec-nonreal", type=int, default=12)
    ap.add_argument("--out-json", type=Path, default=DOCS_DIR / "gate_results.json")
    ap.add_argument("--rerun", action="store_true")
    args = ap.parse_args()
    n_threads = min(4, max(1, args.threads))

    t_all = time.time()
    base_puzzles = read_submission(MERGED_CSV)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    wins_by_n = {}
    consec_nonreal = 0

    for n in args.targets:
        ckpt = CKPT_DIR / f"n_{n}.json"
        if ckpt.exists() and not args.rerun and json.loads(ckpt.read_text(encoding="utf-8")).get("runner") == "H131":
            rec = json.loads(ckpt.read_text(encoding="utf-8"))
            results.append(rec)
            st = rec.get("status", "MISS")
            if st == "REAL_WIN":
                consec_nonreal = 0
                win_path = DATA_DIR / f"win_n{n}.csv"
                if win_path.exists():
                    union_mod = _load_union_mod()
                    lit_g, _ = union_mod.load(win_path)
                    wins_by_n[n] = {"rows": lit_g[n]}
            else:
                consec_nonreal += 1
            print(
                f"[CACHED] n={n} status={st} "
                f"dUp={rec.get('best_dL_upstream')} dH128={rec.get('best_dL_H128')} "
                f"consec_nonreal={consec_nonreal}/{args.stop_consec_nonreal}",
                flush=True,
            )
            if consec_nonreal >= args.stop_consec_nonreal:
                print(f"Stopping early on cached streak: {consec_nonreal} >= {args.stop_consec_nonreal}", flush=True)
                break
            continue

        x_b, y_b, th_b, L_b, up_lit, up_pd, h_lit, h_pd = load_bases(n, base_puzzles)
        res = solve_target(
            n, x_b, y_b, th_b, L_b, up_lit, up_pd, h_lit, h_pd,
            n_threads=n_threads,
            masked_iters=args.masked_iters,
            full_iters=args.full_iters,
            a_start=args.a_start,
            top_k=args.top_k,
        )
        w = res.pop("win")
        res["runner"] = "H131"
        ckpt.write_text(json.dumps(res, indent=2), encoding="utf-8")
        results.append(res)

        if w is not None:
            consec_nonreal = 0
            wins_by_n[n] = w
            write_final_csv({n: w}, DATA_DIR / f"win_n{n}.csv")
            write_final_csv(wins_by_n, FINAL_CSV)
        else:
            consec_nonreal += 1

        args.out_json.write_text(json.dumps({"results": results, "dt_total_s": round(time.time() - t_all, 2)}, indent=2), encoding="utf-8")

        if consec_nonreal >= args.stop_consec_nonreal:
            print(f"\nStopping cleanly: {consec_nonreal} consecutive non-REAL-WIN targets reached --stop-consec-nonreal={args.stop_consec_nonreal}.", flush=True)
            break

    write_final_csv(wins_by_n, FINAL_CSV)
    real_wins = [r["n"] for r in results if r["status"] == "REAL_WIN"]
    ties = [r["n"] for r in results if r["status"] == "TIE"]
    print(f"\nH131 run done in {time.time()-t_all:.1f}s | REAL_WINS={real_wins} | TIES={ties}", flush=True)


if __name__ == "__main__":
    main()
