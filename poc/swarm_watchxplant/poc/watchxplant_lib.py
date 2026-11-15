#!/usr/bin/env python3
"""H122 (watchxplant): Cross-Donor public_watch_nov14 + New Champions Sub-Brick Extraction,
1-Row/1-Col Peeling, vstrip Splicing & Pinwheel + Masked C++ Corner Melt + pslp + lpslack."""
from __future__ import annotations

import io
import json
import math
import os
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
WX_DIR = HERE.parent
DATA_DIR = WX_DIR / "data"
BRICKS_DIR = DATA_DIR / "bricks"
DOCS_DIR = WX_DIR / "docs"
FINAL_CSV = WX_DIR / "final.csv"
REPO_ROOT = WX_DIR.parents[1]

B8_POC = REPO_ROOT / "poc" / "swarm_b8xplant" / "poc"
CP2_POC = REPO_ROOT / "poc" / "swarm_cleanpeel2" / "poc"
GSTEP_POC = REPO_ROOT / "poc" / "swarm_gstep" / "poc"
PEELSTEP_POC = REPO_ROOT / "poc" / "swarm_peelstep" / "poc"
AUTO_UNION_POC = REPO_ROOT / "poc" / "auto_union" / "poc"
LPSLACK_POC = REPO_ROOT / "poc" / "swarm_lpslack" / "poc"
PSLP_POC = REPO_ROOT / "poc" / "row_ratchet_980" / "poc"

for _p in (B8_POC, CP2_POC, GSTEP_POC, PEELSTEP_POC, AUTO_UNION_POC, LPSLACK_POC, PSLP_POC):
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import b8_lib as bl  # noqa: E402
import brick_rect_lib as bslp  # noqa: E402
import cleanpeel2_lib as cp2  # noqa: E402
import gstep_lib as gl  # noqa: E402
import peelstep_lib as pl  # noqa: E402
import pslp  # noqa: E402
import union as auto_union  # noqa: E402
from cpp_wrap import cpp_eval_sat_pen_bbox  # noqa: E402

WORKTREES_ROOT = Path("/home/lucas/.gemini/antigravity-cli/worktrees/a4fb0b93-b63b-426d-aa12-67454e99bc08")

EXTERNAL_BRICK_DIRS = [
    REPO_ROOT / "poc" / "swarm_peelstep" / "data",
    REPO_ROOT / "poc" / "swarm_pocketpeel" / "data" / "bricks",
    REPO_ROOT / "poc" / "swarm_gstep" / "data",
    REPO_ROOT / "poc" / "swarm_cleanpeel2" / "data",
    REPO_ROOT / "poc" / "swarm_stepdown" / "data",
    REPO_ROOT / "poc" / "swarm_b7xplant" / "data",
    REPO_ROOT / "poc" / "swarm_b8xplant" / "data",
    REPO_ROOT / "poc" / "swarm_thinband8" / "data",
    WORKTREES_ROOT / "swarm-vstrip492-96e978de" / "poc" / "swarm_vstrip492" / "data",
    WORKTREES_ROOT / "swarm-vstrip852-39f88515" / "poc" / "swarm_vstrip852" / "data",
    WORKTREES_ROOT / "swarm-b7xplant-a586260b" / "poc" / "swarm_b7xplant" / "data",
]

EXTRA_GIT_FINALS: list[tuple[str, str]] = []


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    BRICKS_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)


def make_brick(w: int, h: int, g: int, x: np.ndarray, y: np.ndarray, th: np.ndarray, tag: str = "") -> dict:
    return cp2.make_brick(w, h, g, x, y, th, tag=tag)


def save_brick_json(b: dict, name: str) -> Path:
    ensure_dirs()
    out_p = BRICKS_DIR / f"brick_{name}.json"
    payload = {
        "name": name,
        "w": int(b["w"]),
        "h": int(b["h"]),
        "g": int(b["g"]),
        "fx": float(b["fx"]),
        "fy": float(b["fy"]),
        "f": float(b["f"]),
        "x": b["x"].tolist(),
        "y": b["y"].tolist(),
        "th": b["th"].tolist(),
    }
    out_p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_p


def load_brick_json(path: Path, tag: str = "") -> dict:
    return cp2.load_brick_json(path, tag=tag)


def _rows_to_arrays(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert auto_union parsed rows (angle already in radians) to float arrays."""
    rs = sorted(rows, key=lambda r: int(r["square_id"]))
    x = np.array([float(r["x"]) for r in rs], dtype=np.float64)
    y = np.array([float(r["y"]) for r in rs], dtype=np.float64)
    th = np.array([float(r["angle"]) for r in rs], dtype=np.float64)
    return x, y, th


def fast_load_true_frontier(target_ns: list[int] | set[int], certify: bool = True) -> dict[int, dict]:
    """Load TRUE frontier (merged.csv + poc/*/final.csv + PR branches) fast by filtering float L first."""
    need = {int(n) for n in target_ns}
    best: dict[int, dict] = {}

    df = pd.read_csv(bl.MERGED_CSV, usecols=["n", "square_id", "x", "y", "angle"])
    for n_val, grp in df[df["n"].isin(need)].groupby("n"):
        n = int(n_val)
        grp_s = grp.sort_values("square_id")
        x, y, th, L_f = bl.normalize(
            grp_s["x"].to_numpy(float, copy=True),
            grp_s["y"].to_numpy(float, copy=True),
            grp_s["angle"].to_numpy(float, copy=True),
        )
        rows = [
            {
                "square_id": int(r.square_id),
                "x": repr(float(r.x)),
                "y": repr(float(r.y)),
                "angle": repr(float(r.angle)),
            }
            for r in grp_s.itertuples(index=False)
        ]
        best[n] = {"x": x, "y": y, "th": th, "L": L_f, "L_dec": Decimal(f"{L_f:.17g}"), "rows": rows, "source": "merged.csv"}

    need_prefixes = tuple(f"{n:03}_" for n in sorted(need))
    sources: list[tuple[str, str]] = []
    for p in sorted((REPO_ROOT / "poc").glob("*/final.csv")):
        if p == FINAL_CSV or not p.exists() or p.stat().st_size <= 20:
            continue
        sources.append((str(p.relative_to(REPO_ROOT)), p.read_text(encoding="utf-8")))
    for br, rel in EXTRA_GIT_FINALS:
        try:
            txt = subprocess.check_output(["git", "show", f"{br}:{rel}"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL, text=True)
            sources.append((f"{br}:{rel}", txt))
        except Exception:
            continue

    candidates: dict[int, list[tuple[float, str, list[dict], list[dict]]]] = {n: [] for n in need}
    for src_label, txt in sources:
        lines = txt.splitlines()
        if len(lines) <= 1:
            continue
        sub = [lines[0]] + [ln for ln in lines[1:] if ln.startswith(need_prefixes)]
        if len(sub) <= 1:
            continue
        sub_txt = "\n".join(sub) + "\n"
        try:
            lit = auto_union.parse_text(sub_txt)
            pd_t = auto_union.parse_text(pd.read_csv(io.StringIO(sub_txt)).to_csv(index=False))
        except Exception:
            continue
        for n in need:
            if n in lit and n in pd_t and len(lit[n]) == n:
                x0, y0, th0 = _rows_to_arrays(lit[n])
                _, _, _, L_f = bl.normalize(x0, y0, th0)
                if n not in best or L_f < best[n]["L"] + 1e-9:
                    candidates[n].append((L_f, src_label, lit[n], pd_t[n]))

    for n in sorted(need):
        if certify and n in best:
            best[n]["L_dec"] = Decimal(auto_union.certify(n, best[n]["rows"]))
        cands = sorted(candidates[n], key=lambda t: t[0])
        for L_f, src_label, rows_lit, rows_pd in cands:
            if n in best and L_f > float(best[n]["L_dec"]) + 1e-9:
                break
            if certify:
                res = auto_union.check((n, rows_lit, rows_pd))
                if res is None:
                    continue
                L_dec = max(Decimal(res[0]), Decimal(res[1]))
            else:
                L_dec = Decimal(f"{L_f:.17g}")
            if n not in best or L_dec < best[n]["L_dec"]:
                x0, y0, th0 = _rows_to_arrays(rows_lit)
                x, y, th, L_norm = bl.normalize(x0, y0, th0)
                best[n] = {"x": x, "y": y, "th": th, "L": L_norm, "L_dec": L_dec, "source": src_label}
        if n in best:
            best[n].pop("rows", None)
    return best


def load_all_pooled_bricks(top_per_shape: int = 2, include_external: bool = True) -> list[dict]:
    """Pool local watchxplant bricks + all external bricks + transposes."""
    raw: list[dict] = []
    dirs = [BRICKS_DIR]
    if include_external:
        dirs.extend(d for d in EXTERNAL_BRICK_DIRS if d.exists())
    for d in dirs:
        if not d.exists():
            continue
        for p in sorted(d.glob("*.json")):
            if not p.name.startswith("brick_"):
                continue
            try:
                raw.append(load_brick_json(p))
            except Exception:
                continue

    # Also pull pocketpeel bricks from git if sibling worktree wasn't present
    pp_wt = WORKTREES_ROOT / "swarm-pocketpeel-eafce120" / "poc" / "swarm_pocketpeel" / "data" / "bricks"
    if include_external and not pp_wt.exists():
        try:
            names = subprocess.check_output(
                ["git", "ls-tree", "-r", "--name-only", "origin/swarm-pocketpeel-pr1", "poc/swarm_pocketpeel/data/bricks/"],
                cwd=REPO_ROOT,
                stderr=subprocess.DEVNULL,
                text=True,
            ).splitlines()
            for rel in names:
                if not rel.endswith(".json"):
                    continue
                txt = subprocess.check_output(["git", "show", f"origin/swarm-pocketpeel-pr1:{rel}"], cwd=REPO_ROOT, text=True)
                d = json.loads(txt)
                raw.append(
                    make_brick(
                        int(d["w"]),
                        int(d["h"]),
                        int(d["g"]),
                        np.array(d["x"], dtype=np.float64),
                        np.array(d["y"], dtype=np.float64),
                        np.array(d["th"], dtype=np.float64),
                        tag=d.get("name", Path(rel).stem.replace("brick_", "")),
                    )
                )
        except Exception:
            pass

    by_shape: dict[tuple[int, int, int], list[dict]] = {}
    seen_sig: set[tuple[int, int, int, float]] = set()
    for b in raw:
        for cand in (b, bl.transform_brick(b, "T")):
            if cand is not b:
                cand["tag"] = b["tag"] + "_T"
            key = (cand["w"], cand["h"], cand["g"])
            sig = (*key, round(cand["f"], 6))
            if sig in seen_sig:
                continue
            seen_sig.add(sig)
            by_shape.setdefault(key, []).append(cand)

    kept: list[dict] = []
    for key in sorted(by_shape):
        lst = sorted(by_shape[key], key=lambda c: c["f"])
        kept.extend(lst[:top_per_shape])
    return kept


def compress_brick_pslp(b: dict, n_trials: int = 4, seed: int = 42, do_shake: bool = True) -> dict:
    """Compress brick with optional compress_brick_slp + kink-safe rectangular pslp.squeeze."""
    best = cp2.compress_brick_slp(b, n_trials=n_trials, seed=seed) if n_trials > 0 else b
    w, h, g = best["w"], best["h"], best["g"]
    nx, ny, W, H = bl.rect_bbox(best["x"], best["y"], best["th"])
    f0 = max(W - w, H - h)
    f1, X1, Y1, T1, v1, _ = pslp.squeeze(nx, ny, best["th"], w, h, f0, max_it=200)
    cand_x, cand_y, cand_t, cand_f = X1, Y1, T1, f1
    if v1 > 1e-12:
        cand_x, cand_y, cand_t, cand_f = nx, ny, best["th"], f0
    if do_shake:
        rng = np.random.default_rng(seed + 777)
        f2, X2, Y2, T2 = pslp.shake_squeeze(cand_x, cand_y, cand_t, w, h, cand_f, rounds=2, rng=rng)
        if f2 < cand_f - 1e-10 and pslp.viol(X2, Y2, T2, w, h, f2) <= 1e-12:
            cand_x, cand_y, cand_t, cand_f = X2, Y2, T2, f2
    pen, _ = cpp_eval_sat_pen_bbox(cand_x, cand_y, cand_t)
    if pen <= 1e-8:
        pb = make_brick(w, h, g, cand_x, cand_y, cand_t, tag=best["tag"])
        if pb["f"] < best["f"] - 1e-9:
            return pb
    return best


def pslp_and_certify(
    n: int,
    x: np.ndarray,
    y: np.ndarray,
    th: np.ndarray,
    inc_dec: Decimal,
    label: str,
    shake_rounds: int = 2,
    seed: int = 123,
) -> dict | None:
    """Apply kink-safe pslp squeeze (+ optional shake_squeeze) and dual-certify."""
    m = int(math.isqrt(n))
    nx, ny, nth, L0 = bl.normalize(x, y, th)
    f0 = L0 - m
    f1, X1, Y1, T1, v1, _ = pslp.squeeze(nx, ny, nth, m, m, f0, max_it=250)
    best_x, best_y, best_t, best_f = X1, Y1, T1, f1
    if v1 > 1e-12:
        best_x, best_y, best_t, best_f = nx, ny, nth, f0
    if shake_rounds > 0:
        rng = np.random.default_rng(seed)
        f2, X2, Y2, T2 = pslp.shake_squeeze(best_x, best_y, best_t, m, m, best_f, rounds=shake_rounds, rng=rng)
        if f2 < best_f - 1e-10 and pslp.viol(X2, Y2, T2, m, m, f2) <= 1e-12:
            best_x, best_y, best_t, best_f = X2, Y2, T2, f2
    cert = bl.polish_and_certify(n, best_x, best_y, best_t, inc_dec, slp_rounds=16)
    if cert is not None:
        cert["label"] = f"{label}+pslp"
    return cert


def _update_cert(best: dict | None, cand: dict | None) -> dict | None:
    if cand is None:
        return best
    if best is None or cand["L_float"] < best["L_float"]:
        return cand
    return best


def solve_pinwheel_pair(
    n: int,
    m: int,
    r: int,
    b1_base: dict,
    b2_base: dict,
    drop_mode: str,
    inc_dec: Decimal,
    tag: str,
    base_seed: int,
    n_threads: int = 2,
) -> tuple[dict | None, list]:
    """Direct SLP + masked C++ corner/seam melt + pslp + lpslack."""
    best_cert: dict | None = None
    seeds, masks, scheds, labels = [], [], [], []
    inc_f = float(inc_dec)
    kx = b1_base["w"] + b2_base["w"] - m
    ky = b1_base["h"] + b2_base["h"] - m
    is_zero_ov = (kx <= 0 and ky <= 0)

    drop_cands = ("b1_BR", "b2_TL") if drop_mode == "b1_BR" else (drop_mode,)
    for dm in drop_cands:
        for m1 in ("id", "flipx", "flipy", "rot180"):
            b1 = bl.transform_brick(b1_base, m1)
            for m2 in ("id", "flipx", "flipy", "rot180"):
                b2 = bl.transform_brick(b2_base, m2)
                f_hi = max(b1["fx"], b1["fy"], b2["fx"], b2["fy"])
                f_avg = 0.5 * (max(b1["fx"], b1["fy"]) + max(b2["fx"], b2["fy"]))
                f_opts = ((f_hi, "fhi"),) if abs(b1["f"] - b2["f"]) < 5e-4 else ((f_hi, "fhi"), (f_avg, "favg"))
                for f_use, ftag in f_opts:
                    built = pl.build_pinwheel_kxky(m, r, b1, b2, sf=0.0, f_override=f_use, drop_mode=dm)
                    if built is None:
                        continue
                    ax, ay, at, L_tgt, sm = built
                    if is_zero_ov and ftag == "fhi" and (m1, m2) in (("id", "id"), ("flipx", "flipy"), ("rot180", "rot180")):
                        pen, _ = cpp_eval_sat_pen_bbox(ax, ay, at)
                        if pen <= 1e-10:
                            sx, sy, st, _, ok = bslp.joint_xytheta_slp_tight(ax, ay, at)
                            use_x, use_y, use_t = (sx, sy, st) if ok else (ax, ay, at)
                            cert = bl.polish_and_certify(n, use_x, use_y, use_t, inc_dec, slp_rounds=16)
                            if cert is not None:
                                cert["label"] = f"{tag}_{m1}_{m2}_direct"
                                best_cert = _update_cert(best_cert, cert)

                    if ftag == "fhi" or len(seeds) < 24:
                        w1, h1, w2, h2 = b1["w"], b1["h"], b2["w"], b2["h"]
                        corner_sm = (
                            (sm > 0)
                            | ((np.abs(ax - w1) <= 4.0) & (np.abs(ay - (m - h1)) <= 4.0))
                            | ((np.abs(ax - (m - w2)) <= 4.0) & (np.abs(ay - h2) <= 4.0))
                        ).astype(np.uint8)
                        if inc_f - L_tgt > 0.005:
                            L_end = min(inc_f - 0.001, max(L_tgt + 0.001, inc_f - 0.004))
                            L_mid = max(L_tgt + 0.008, L_end + 0.006)
                            L_start = max(L_tgt + 0.022, L_end + 0.016)
                        else:
                            L_end = min(inc_f - 3e-4, L_tgt + 0.001) if L_tgt + 0.001 < inc_f else 0.5 * (L_tgt + inc_f)
                            L_mid = L_tgt + 0.005
                            L_start = L_tgt + 0.018
                        seeds.append((ax, ay, at))
                        masks.append(corner_sm)
                        scheds.append([L_start, L_mid, L_end])
                        labels.append(f"{tag}_{dm}_{m1}_{m2}_{ftag}")

    diag = []
    if seeds:
        if len(seeds) > 24:
            seeds, masks, scheds, labels = seeds[:24], masks[:24], scheds[:24], labels[:24]
        relaxed = bl.su.cpp_batch_melt(
            seeds,
            scheds,
            sym_mode=1,
            a_start=0.962,
            n_melt_stages=3,
            seam_masks=masks,
            seam_iters=220,
            full_iters=260,
            n_ruin_iters=0,
            n_threads=n_threads,
            base_seed=base_seed,
        )
        pool = []
        for bi, row in enumerate(relaxed):
            nx, ny, nth, Lm = bl.normalize(row["x"], row["y"], row["th"])
            lbl = f"{labels[bi]}_melt"
            if row["best_L_feas"] is not None:
                pool.append((0.0, Lm, float(row["pen"]), nx, ny, nth, lbl))
            else:
                pool.append((1.0 + float(row["pen"]), Lm, float(row["pen"]), nx, ny, nth, lbl))
        pool.sort(key=lambda t: (t[0], t[1]))
        diag = [(t[0] == 0.0, round(t[1], 6), round(t[2], 6), t[6]) for t in pool[:3]]

        for rank_i, (_, _, pen, cx, cy, cth, lbl) in enumerate(pool[:3]):
            if pen > 0.03:
                continue
            sx, sy, st, _, ok = bslp.joint_xytheta_slp_tight(cx, cy, cth)
            use_x, use_y, use_t = (sx, sy, st) if ok else (cx, cy, cth)
            cert = bl.polish_and_certify(n, use_x, use_y, use_t, inc_dec, slp_rounds=18)
            if cert is not None:
                cert["label"] = lbl
                best_cert = _update_cert(best_cert, cert)
            if rank_i == 0 and pen <= 1e-6:
                c_pslp = pslp_and_certify(
                    n, use_x, use_y, use_t, inc_dec, lbl, shake_rounds=2, seed=base_seed + 500
                )
                best_cert = _update_cert(best_cert, c_pslp)

    if best_cert is not None and best_cert["L_float"] < inc_f + 0.008:
        best_cert = pl.run_lpslack_polish(n, best_cert, inc_dec)
        # Final pslp squeeze on best certified result
        fx, fy, ft = _rows_to_arrays(best_cert["rows"])
        c_pslp2 = pslp_and_certify(
            n, fx, fy, ft, inc_dec, best_cert["label"], shake_rounds=2, seed=base_seed + 900
        )
        best_cert = _update_cert(best_cert, c_pslp2)

    return best_cert, diag


def save_win_locked(win_entry: dict) -> bool:
    return bl.save_win_locked(win_entry, data_dir=DATA_DIR, final_path=FINAL_CSV)
