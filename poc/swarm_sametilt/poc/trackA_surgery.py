#!/usr/bin/env python3
"""Track A: Direct neighbor surgery on same-tilt dual-crystal packings (n=905..915, plus 918..923, 792..797)."""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from decimal import Decimal
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sametilt_lib as st  # noqa: E402
from trackB_homology import run_seed_batch  # noqa: E402

DOCS = st.DOCS_DIR


def square_clearances(x: np.ndarray, y: np.ndarray, th: np.ndarray, L: float) -> np.ndarray:
    """Compute per-square minimum gap to walls and neighbors."""
    h = st.hx_of(th)
    wall = np.minimum(np.minimum(x - h, L - (x + h)), np.minimum(y - h, L - (y + h)))
    c, s = np.cos(th), np.sin(th)
    n = len(x)
    min_nb = np.full(n, 1e9)
    for i in range(n):
        xi, yi, ci, si = x[i], y[i], c[i], s[i]
        for j in range(i + 1, n):
            dx, dy = x[j] - xi, y[j] - yi
            if abs(dx) > 1.75 or abs(dy) > 1.75:
                continue
            rad = 0.5 * (1.0 + abs(ci * c[j] + si * s[j]) + abs(ci * s[j] - si * c[j]))
            gap = max(
                abs(dx * ci + dy * si) - rad,
                abs(-dx * si + dy * ci) - rad,
                abs(dx * c[j] + dy * s[j]) - rad,
                abs(-dx * s[j] + dy * c[j]) - rad,
            )
            if gap < min_nb[i]:
                min_nb[i] = gap
            if gap < min_nb[j]:
                min_nb[j] = gap
    return np.minimum(wall, min_nb)


def build_insertion_seeds(
    n_target: int,
    donors: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, float]],
    L_inc: float,
    rng: np.random.Generator,
) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]]:
    """Insert k = n_target - src_n squares into largest voids of donor."""
    candidates = []
    for src_n, (x, y, t, L) in donors.items():
        k = n_target - src_n
        if k < 1 or k > 6:
            continue
        if L >= L_inc - 2e-4:
            continue
        candidates.append((k, L, src_n, x, y, t))
    candidates.sort(key=lambda item: (item[0], item[1], -item[2]))

    seeds = []
    for k, L, src_n, x, y, t in candidates:
        vp = st.clearance_points(x, y, t, L, 14, step=0.08, min_sep=0.76)
        if len(vp) < k:
            continue
        # Generate diverse k-combinations from top voids
        pool_idx = list(range(min(len(vp), k + 5)))
        combos = [list(range(k))]
        all_c = list(itertools.combinations(pool_idx, k))
        if len(all_c) > 1:
            # Deterministic spaced picks
            step = max(1, len(all_c) // 5)
            for c in all_c[1::step][:4]:
                if list(c) not in combos:
                    combos.append(list(c))

        for ci, idxs in enumerate(combos[:4]):
            for ang_tag, ang_fn in (
                ("u0", lambda m: np.zeros(m)),
                ("t65", lambda m: np.full(m, np.radians(65.6))),
                ("mix", lambda m: rng.uniform(0.0, 0.5 * np.pi, m)),
            ):
                if ci > 1 and ang_tag != "u0":
                    continue
                xr, yr, tr = x.copy(), y.copy(), t.copy()
                vctrs = []
                for idx in idxs:
                    vx, vy, _ = vp[idx]
                    xr = np.append(xr, vx)
                    yr = np.append(yr, vy)
                    vctrs.append([vx, vy])
                tr = np.concatenate([tr, ang_fn(k)])
                # Also include BR slack region center (28.5, 3.5) if L > 25
                ctrs = np.array(vctrs)
                if L > 25.0:
                    ctrs = np.vstack([ctrs, [[L - 2.0, 3.5], [5.3, 9.4]]])
                # Target Lt between L + 0.005*k and L_inc - 5e-4
                Lt = min(L_inc - 5e-4, max(L + 0.005 * k, L_inc - 0.012))
                seeds.append((f"ins{k}_{src_n}_c{ci}_{ang_tag}", xr, yr, tr, Lt, ctrs))
    return seeds[:8]


def build_deletion_seeds(
    n_target: int,
    donors: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, float]],
    L_inc: float,
) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]]:
    """Remove k = src_n - n_target squares from high-slack / boundary positions and slide strips inward."""
    seeds = []
    for src_n, (x, y, t, L) in sorted(donors.items()):
        k = src_n - n_target
        if k < 1 or k > 6:
            continue
        clr = square_clearances(x, y, t, L)
        h = st.hx_of(t)
        # 1. Remove k loosest squares overall
        loose_idx = np.argsort(-clr)[:k]
        # 2. Remove k squares near right boundary (x > L - 2.5) with highest clearance
        right_mask = np.where(x + h > L - 2.5)[0]
        right_idx = right_mask[np.argsort(-clr[right_mask])[:k]] if len(right_mask) >= k else loose_idx
        # 3. Remove k squares near top boundary (y > L - 2.5) with highest clearance
        top_mask = np.where(y + h > L - 2.5)[0]
        top_idx = top_mask[np.argsort(-clr[top_mask])[:k]] if len(top_mask) >= k else loose_idx
        # 4. Remove k squares in BR region (x > L - 5, y < 9.5)
        br_mask = np.where((x > L - 5.0) & (y < 9.5))[0]
        br_idx = br_mask[np.argsort(-clr[br_mask])[:k]] if len(br_mask) >= k else loose_idx

        Lt = min(L_inc - 1e-3, L - 0.005 * k)
        for tag, ridx, slide_mode in (
            ("loose", loose_idx, "none"),
            ("right_slide", right_idx, "x_in"),
            ("top_slide", top_idx, "y_in"),
            ("br_slide", br_idx, "x_in"),
        ):
            keep = np.ones(len(x), bool)
            keep[ridx] = False
            xr, yr, tr = x[keep].copy(), y[keep].copy(), t[keep].copy()
            ctrs = np.c_[x[ridx], y[ridx]]
            if slide_mode == "x_in":
                # Gently slide squares to the right of deleted centroids inward
                xmin_del = float(x[ridx].min())
                sh = xr > xmin_del
                xr[sh] -= min(0.04, max(0.005, L - Lt))
                ctrs = np.vstack([ctrs, [[L - 1.5, 4.5], [L - 1.5, L - 1.5]]])
            elif slide_mode == "y_in":
                ymin_del = float(y[ridx].min())
                sh = yr > ymin_del
                yr[sh] -= min(0.04, max(0.005, L - Lt))
                ctrs = np.vstack([ctrs, [[4.5, L - 1.5], [L - 1.5, L - 1.5]]])
            seeds.append((f"del{k}_{src_n}_{tag}", xr, yr, tr, Lt, ctrs))
    return seeds[:8]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", type=int, nargs="*", default=None)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--iters", type=int, default=220)
    args = ap.parse_args()

    st.DOCS_DIR.mkdir(parents=True, exist_ok=True)
    st.DATA_DIR.mkdir(parents=True, exist_ok=True)
    packs, best_L = st.load_frontier()
    rng = np.random.default_rng(20261114)

    donors: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, float]] = {
        910: packs[910],
        909: packs[909],
        920: packs[920],
        793: packs[793],
    }
    for p in sorted(st.DATA_DIR.glob("win_n*.json")):
        d = json.loads(p.read_text())
        wn = int(d["n"])
        rows = d["rows"]
        donors[wn] = st.norm_pack(
            np.array([float(r["x"]) for r in rows]),
            np.array([float(r["y"]) for r in rows]),
            np.array([float(r["angle"]) for r in rows]),
        )

    default_targets = [
        911, 912, 913, 914, 915,
        851, 852, 853, 854, 855, 856,
        908, 907, 906, 905, 909,
        972, 973, 974, 975, 976,
        921, 922, 919,
        794, 795, 792,
    ]
    targets = args.targets if args.targets else default_targets

    summary = {}
    for n_t in targets:
        for p in sorted(st.DATA_DIR.glob("win_n*.json")):
            d = json.loads(p.read_text())
            wn = int(d["n"])
            rows = d["rows"]
            w_pack = st.norm_pack(
                np.array([float(r["x"]) for r in rows]),
                np.array([float(r["y"]) for r in rows]),
                np.array([float(r["angle"]) for r in rows]),
            )
            if wn not in donors or w_pack[3] < donors[wn][3]:
                donors[wn] = w_pack
            if Decimal(d["L_lit"]) < best_L[wn]:
                best_L[wn] = Decimal(d["L_lit"])

        L_inc_dec = best_L[n_t]
        L_inc = float(L_inc_dec)
        print(f"\n=== Track A Target n={n_t} (frontier L={L_inc:.12f}) ===", flush=True)
        seeds = build_insertion_seeds(n_t, donors, L_inc, rng)
        seeds += build_deletion_seeds(n_t, donors, L_inc)
        if not seeds:
            print(f"[{n_t}] No eligible seeds.", flush=True)
            continue

        cert = run_seed_batch(
            n_t,
            seeds[:6],
            L_inc_dec,
            args.threads,
            rng,
            n_seeds_per_spec=6,
            masked_iters=args.iters,
            full_iters=args.iters + 40,
            top_polish=3,
            R_range=(5.5, 11.5),
            allow_full=True,
        )
        if cert is not None and cert["is_real_win"]:
            best_L[n_t] = Decimal(cert["L_lit"])
            rows = cert["rows"]
            donors[n_t] = st.norm_pack(
                np.array([float(r["x"]) for r in rows]),
                np.array([float(r["y"]) for r in rows]),
                np.array([float(r["angle"]) for r in rows]),
            )
            summary[n_t] = {"L": cert["L_lit"], "dL": cert["dL"]}
        else:
            summary[n_t] = None
        (DOCS / "trackA_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
