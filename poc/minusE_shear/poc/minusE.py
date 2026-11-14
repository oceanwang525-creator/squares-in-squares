"""H5 -- "minus-E" constructor: (K+1)x(K+1) grid, delete an L along the top row
and right column (keeping exactly D = n - K^2 border cells), then shear /
rotate the resulting blocks so the staircase seam interlocks, and compress the
container with the repository's fixed-angle LP followed by the Lipschitz
free-angle SLP.

For n = K^2 + D the arithmetic is exact: a (K+1)x(K+1) grid has (K+1)^2 cells,
E = (K+1)^2 - n = 2K+1-D must be deleted, and the border L-band

    top row    (x, K)      x = 0..K
    right col  (K, y)      y = 0..K-1

has exactly 2K+1 cells, so *exactly D of them survive* and the inner KxK block
is untouched.  Every deletion pattern below is therefore a choice of which D
band cells to keep.

Stage A (fast):  base side -> fixed-angle LP (12 rounds, no trust region).
Stage B:         best candidates -> free-angle Lipschitz SLP -> official
                 certification in BOTH literal and pandas transport.

Usage:
  python poc/minusE_shear/poc/minusE.py --n-list 492,908,328 --jobs 32
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import sys
import time
from decimal import Decimal
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "poc/baseline/poc/src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lp_core as LP  # noqa: E402
from squares import core  # noqa: E402

DEG = math.pi / 180.0


# --------------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------------
def band_cells(G):
    """Border L of a GxG grid: top row left->right, then right column bottom->top."""
    top = [(x, G - 1) for x in range(G)]
    right = [(G - 1, y) for y in range(G - 1)]
    return top, right


def keep_patterns(G, D, rng, n_random=6):
    """Yield (name, list_of_kept_band_cells).  Exactly D cells each."""
    top, right = band_cells(G)
    band = top + right
    out = []
    if D <= 0:
        return out
    # 1. contiguous, anchored at the two ends of the band
    if D <= G:
        out.append(("topL", top[:D]))
        out.append(("topR", top[G - D:]))
    if D <= G - 1:
        out.append(("botR", right[:D]))
        out.append(("topcolR", right[G - 1 - D:]))
    # 2. corner splits a:b  (a on the top row right end, b on the column top end)
    for a in range(max(0, D - (G - 1)), min(D, G) + 1):
        b = D - a
        if 0 <= b <= G - 1:
            out.append((f"corner{a}_{b}", top[G - a:] + right[G - 1 - b:]))
    # 3. outer splits a:b (a on the top row LEFT end, b on the column BOTTOM end)
    for a in range(max(0, D - (G - 1)), min(D, G) + 1):
        b = D - a
        if 0 <= b <= G - 1:
            out.append((f"outer{a}_{b}", top[:a] + right[:b]))
    # 4. evenly spread along the band (staircase phase)
    if D >= 2 and len(band) > D:
        idx = [int(round(k * (len(band) - 1) / (D - 1))) for k in range(D)]
        out.append(("spread", [band[i] for i in idx]))
        idx2 = [int(round(k * (len(band) - 1) / (D - 1))) for k in range(D)]
        out.append(("spread_rev", [band[i] for i in reversed(idx2)]))
    # 5. random subsets
    for s in range(n_random):
        pick = rng.choice(len(band), size=D, replace=False)
        out.append((f"rand{s}", [band[i] for i in sorted(pick)]))
    # dedupe
    seen, uniq = set(), []
    for name, cells in out:
        key = tuple(sorted(cells))
        if key in seen:
            continue
        seen.add(key)
        uniq.append((name, cells))
    return uniq


def state_from_cells(cells, angles=None):
    """cells -> (n,2) centres, (n,) angles.  Cell (i,j) spans [i,i+1]x[j,j+1]."""
    c = np.array([[i + 0.5, j + 0.5] for i, j in cells], dtype=float)
    a = np.zeros(len(cells)) if angles is None else np.asarray(angles, float)
    return c, a


def overlaps_pairs(c, a, tol=1e-7):
    """True if any pair overlaps (exact SAT, nearest pairs only, vectorised)."""
    if len(c) < 2:
        return False
    pairs = np.array(sorted(cKDTree(c).query_pairs(1.55)), dtype=int)
    if pairs.size == 0:
        return False
    i, j = pairs[:, 0], pairs[:, 1]
    u = np.column_stack([np.cos(a), np.sin(a)])
    v = np.column_stack([-np.sin(a), np.cos(a)])
    axes = np.stack([u[i], v[i], u[j], v[j]], axis=1)          # (P,4,2)
    proj = np.abs(np.einsum("pkd,pd->pk", axes, c[j] - c[i]))
    rad = 0.5 * (np.abs(np.einsum("pkd,pd->pk", axes, u[i]))
                 + np.abs(np.einsum("pkd,pd->pk", axes, v[i]))
                 + np.abs(np.einsum("pkd,pd->pk", axes, u[j]))
                 + np.abs(np.einsum("pkd,pd->pk", axes, v[j])))
    return bool(np.any(np.all(proj < rad - tol, axis=1)))


def rotate_block(c, idx, theta, centre=None):
    """Rotate the given centres about their centroid (or a supplied point)."""
    c = c.copy()
    if len(idx) == 0:
        return c
    if centre is None:
        centre = c[idx].mean(axis=0)
    R = np.array([[math.cos(theta), -math.sin(theta)],
                  [math.sin(theta), math.cos(theta)]])
    c[idx] = (c[idx] - centre) @ R.T + centre
    return c


def translate_block(c, idx, dxy):
    c = c.copy()
    c[idx] += np.asarray(dxy, float)
    return c


# --------------------------------------------------------------------------
# candidate families
# --------------------------------------------------------------------------
SQ2 = math.sqrt(0.5)
OFFSETS = [(0, -0.25), (0, -0.5), (0, -SQ2), (0, -1.0),
           (-0.25, 0), (-0.5, 0), (-SQ2, 0), (-1.0, 0),
           (-0.25, -0.25), (-0.5, -0.5), (-SQ2, -SQ2), (-1.0, -1.0),
           (SQ2, -SQ2), (-SQ2, SQ2), (1.0, -1.0), (-1.0, 1.0),
           (-0.25, -0.5), (-0.5, -0.25), (-0.5, -SQ2), (-SQ2, -0.5)]
# dense diagonal slides: "shift the top-right block by (-d,-d)" (H5)
DIAG = [(d, -d) for d in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)]
DIAG += [(-d, -d) for d in (0.1, 0.2, 0.3, 0.4, 0.5, SQ2, 0.8)]
TILTS = [45.0, -45.0, 40.0, 50.0, 30.0, 60.0]
JITTER = [(-0.5, 0.0), (0.0, 0.0), (0.5, 0.0), (0.0, -0.5), (0.0, 0.5),
          (0.5, -0.5), (-0.5, 0.5), (SQ2, -SQ2), (-SQ2, SQ2)]


def candidates(n, rng, n_random=6):
    """Yield (name, centres, angles) for every minus-E assembly tried."""
    K = math.isqrt(n)
    D = n - K * K
    G = K + 1
    assert G * G - n == 2 * K + 1 - D and 0 <= D <= 2 * K + 1
    inner = [(i, j) for i in range(K) for j in range(K)]
    pats = keep_patterns(G, D, rng, n_random=n_random)

    for pname, keep in pats:
        cells = inner + list(keep)
        c0, a0 = state_from_cells(cells)
        nk = np.arange(len(inner), len(cells))          # band-cell indices
        yield f"{pname}/base", c0, a0
        if len(nk) == 0:
            continue

        # --- tilt each band cell about its own centre ------------------------
        for tilt in TILTS:
            aa = a0.copy()
            aa[nk] = tilt * DEG
            if not overlaps_pairs(c0, aa):
                yield f"{pname}/tiltcell{tilt:+.0f}", c0, aa

        # --- rotate the band cells as one block, anchor at the TR corner -----
        for tilt in TILTS:
            rot = rotate_block(c0, nk, tilt * DEG)
            aa = a0.copy()
            aa[nk] = tilt * DEG                       # the block really rotates
            ext = LP.extents(aa[nk])[:, None]
            lo = (rot[nk] - ext).min(axis=0)          # bounding-box lower edge
            hi = (rot[nk] + ext).max(axis=0)          # bounding-box upper edge
            for ax, ay in ((-1, 1), (0, 0), (-1, 0), (0, 1)):
                shift = np.array([K - hi[0] + 0.5 * ax, K - lo[1] + 0.5 * ay])
                for jx, jy in JITTER:
                    cc = translate_block(rot, nk, shift + np.array([jx, jy]))
                    if not overlaps_pairs(cc, aa):
                        yield (f"{pname}/tiltband{tilt:+.0f}"
                               f"({ax:+d},{ay:+d})J{jx:+.2f},{jy:+.2f}"), cc, aa

        # --- two-block shear: cut the assembly and move one part -------------
        for cutname, idx in block_splits(G, cells):
            if len(idx) == 0 or len(idx) == len(cells):
                continue
            moves = OFFSETS if cutname in ("halfTR", "stripUp") else \
                OFFSETS[:12]
            for dxy in moves:
                cc = translate_block(c0, idx, dxy)
                if not overlaps_pairs(cc, a0):
                    yield (f"{pname}/{cutname}"
                           f"D{dxy[0]:+.2f},{dxy[1]:+.2f}"), cc, a0
            if cutname in ("halfTR", "stripUp"):
                for dxy in DIAG:
                    cc = translate_block(c0, idx, dxy)
                    if not overlaps_pairs(cc, a0):
                        yield (f"{pname}/{cutname}"
                               f"G{dxy[0]:+.2f},{dxy[1]:+.2f}"), cc, a0
            for tilt in (45.0, -45.0, 90.0):
                rot = rotate_block(c0, idx, tilt * DEG)
                for dx, dy in ((0, 0), (-0.5, -0.5), (0.5, 0.5), (-1, 0), (0, -1)):
                    ccc = translate_block(rot, idx, (dx, dy))
                    if not overlaps_pairs(ccc, a0):
                        yield (f"{pname}/{cutname}R{tilt:+.0f}"
                               f"({dx:+.1f},{dy:+.1f})"), ccc, a0


def block_splits(G, cells):
    """Named index sets (into the assembly) for the 'two block' interlock moves."""
    ii = np.array([c[0] for c in cells])
    jj = np.array([c[1] for c in cells])
    return [
        ("band", np.flatnonzero((ii == G - 1) | (jj == G - 1))),
        ("toprow", np.flatnonzero(jj == G - 1)),
        ("rightcol", np.flatnonzero(ii == G - 1)),
        ("halfTR", np.flatnonzero(ii + jj >= G - 1)),
        ("stripUp", np.flatnonzero(ii + jj >= G)),
        ("diagHigh", np.flatnonzero(ii - jj >= 2)),
    ]


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------
def to_rows(c, a):
    return [dict(square_id=str(i), x=repr(float(c[i, 0])),
                 y=repr(float(c[i, 1])), angle=repr(float(a[i])))
            for i in range(len(a))]


def eval_candidate(task):
    n, name, c, a, rounds, time_limit, gate = task
    base = LP.exact_side(c, a)
    if base > gate:
        return n, name, base, None, None, a
    out = LP.fixed_lp_refine(a, c, rounds=rounds, time_limit=time_limit)
    if out is None:
        return n, name, base, None, None, a
    side, cen = out
    if side * side < n - 1e-9:
        # 492 unit squares cannot fit in a 22.0 box: the unconstrained LP walked
        # squares through each other.  Drop it before it wastes an SLP slot.
        return n, name, base, None, None, a
    return n, name, base, side, cen, a


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-list", default="492,908,328")
    ap.add_argument("--jobs", type=int, default=32)
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--time-limit", type=float, default=60.0)
    ap.add_argument("--top", type=int, default=8,
                    help="candidates per n sent to the free-angle SLP")
    ap.add_argument("--base-gate", type=float, default=0.8,
                    help="skip candidates whose raw base side exceeds "
                         "incumbent + this")
    ap.add_argument("--n-random", type=int, default=6)
    ap.add_argument("--max-cands", type=int, default=1200,
                    help="LP budget per n (best raw base sides first)")
    ap.add_argument("--no-gate-tilted", action="store_true",
                    help="LP every tilted candidate even if its raw base side "
                         "is far above the incumbent")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--outdir", default=str(Path(__file__).resolve().parents[1] / "docs"))
    args = ap.parse_args()

    gap = {int(r["n"]): r for r in csv.DictReader(
        open(REPO / "poc/koboshchan_coordinator/data/gap_vs_frontier.csv"))}
    targets = [int(x) for x in args.n_list.split(",") if x.strip()]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for n in targets:
        rng = np.random.default_rng(args.seed + n)
        t0 = time.time()
        cands = list(candidates(n, rng, n_random=args.n_random))
        seen, uniq = set(), []
        for nm, c, a in cands:
            key = tuple(np.round(np.concatenate([c.ravel(), a]), 4).tolist())
            if key in seen:
                continue
            seen.add(key)
            uniq.append((nm, c, a))
        print(f"n={n}: {len(uniq)} unique candidates "
              f"(K={gap[n]['K']} D={gap[n]['D']}) in {time.time()-t0:.1f}s",
              flush=True)
        # rank by raw base side and cap the number of LPs.  Half of the budget
        # is reserved for candidates with tilted squares: only they carry a new
        # contact topology, an all-upright minus-E shape is a rigid grid.
        sides = [LP.exact_side(c, a) for _, c, a in uniq]
        tilted = [bool(np.any(np.abs(a) > 1e-9)) for _, _, a in uniq]
        n_unique = len(uniq)
        min_i = int(np.argmin(sides))
        min_base_all, min_base_all_name = sides[min_i], uniq[min_i][0]
        t_idx = sorted((i for i in range(len(uniq)) if tilted[i]),
                       key=lambda i: sides[i])
        u_idx = sorted((i for i in range(len(uniq)) if not tilted[i]),
                       key=lambda i: sides[i])
        half = args.max_cands // 2
        keep_idx = t_idx[:half] + u_idx[:args.max_cands - len(t_idx[:half])]
        uniq = [uniq[i] for i in keep_idx]
        print(f"n={n}: {len(uniq)} candidates kept for the LP "
              f"({sum(1 for i in keep_idx if tilted[i])} tilted), "
              f"base sides {min(sides[i] for i in keep_idx):.6f} .. "
              f"{max(sides[i] for i in keep_idx):.6f}", flush=True)

        gate = float(Decimal(gap[n]["ours"])) + args.base_gate
        tasks = []
        for nm, c, a in uniq:
            g = float("inf") if (args.no_gate_tilted
                                 and np.any(np.abs(a) > 1e-9)) else gate
            tasks.append((n, nm, c, a, args.rounds, args.time_limit, g))
        results = []
        with mp.get_context("fork").Pool(max(1, min(args.jobs, os.cpu_count() or 1))) as pool:
            for res in pool.imap_unordered(eval_candidate, tasks, chunksize=4):
                results.append(res)
        results.sort(key=lambda r: (r[3] is None, r[3] if r[3] is not None else 9e9))
        ours = Decimal(gap[n]["ours"])
        front = Decimal(gap[n]["frontier"])
        best_lp = next((r[3] for r in results if r[3] is not None), None)
        print(f"n={n}: LP done in {time.time()-t0:.1f}s; "
              f"best LP side {best_lp if best_lp is None else round(best_lp, 9)} "
              f"({results[0][1]}) vs ours {float(ours):.9f} "
              f"frontier {float(front):.9f}", flush=True)

        # ---------------- stage B: free-angle SLP on the best candidates ----
        best_rows, best_val, best_name = None, None, None
        for (_, name, base, side, cen, aa0) in results[:args.top]:
            if cen is None:
                continue
            # the raw LP solution is a candidate too -- certify it as it stands
            # before any SLP step, otherwise a feasible LP state can be lost.
            try:
                rows0 = to_rows(cen, aa0)
                val0 = Decimal(core.certify(n, rows0))
                if best_val is None or val0 < best_val:
                    best_val, best_rows, best_name = val0, rows0, name + "[lp]"
            except Exception:  # noqa: BLE001
                pass
            th = aa0.copy()
            L = LP.exact_side(cen, th)
            ta, tm = 0.002, 0.008
            cur_c, cur_t = cen, th
            for _ in range(8):
                step = LP.slp_step(cur_t, cur_c, L, ta, tm)
                if step is None:
                    ta *= 0.5
                    tm *= 0.5
                    if ta < 1e-7:
                        break
                    continue
                nc, nt, moved = step
                rows = to_rows(nc, nt)
                try:
                    val = Decimal(core.certify(n, rows))
                except Exception:  # noqa: BLE001
                    ta *= 0.5
                    tm *= 0.5
                    if ta < 1e-7:
                        break
                    continue
                cur_c, cur_t = nc, nt
                L = LP.exact_side(cur_c, cur_t)
                if best_val is None or val < best_val:
                    best_val, best_rows, best_name = val, rows, name
                if moved < 1e-10:
                    break
                ta = min(0.002, max(ta * 1.3, 1e-6))
                tm = min(0.008, max(tm * 1.3, 1e-6))

        entry = dict(n=n, K=int(gap[n]["K"]), D=int(gap[n]["D"]),
                     ours=str(ours), frontier=str(front),
                     leader=str(gap[n]["leader"]),
                     n_unique_before_cap=n_unique,
                     min_base_all=min_base_all, min_base_all_name=min_base_all_name,
                     min_base_tilted=min((sides[i] for i in range(len(sides))
                                          if tilted[i]), default=None),
                     n_candidates=len(uniq),
                     lp_best=best_lp, lp_best_name=results[0][1],
                     lp_best_base=results[0][2],
                     top10=[dict(name=r[1], base=r[2], lp=r[3]) for r in results[:10]],
                     seconds=round(time.time() - t0, 1))
        if best_val is not None:
            entry["slp_best"] = str(best_val)
            entry["slp_best_name"] = best_name
        print(f"n={n}: slp best {best_val} ({best_name})", flush=True)
        summary[n] = entry

        # persist any candidate better than the incumbent (both transports)
        if best_rows is not None and best_val < ours:
            transported = core.kaggle_roundtrip({n: best_rows})
            pl = Decimal(core.certify(n, transported[n]))
            entry["pandas"] = str(pl)
            if pl < ours:
                path = outdir / f"win_{n}.json"
                json.dump(best_rows, open(path, "w"), indent=1)
                entry["win"] = str(path)

    json.dump(summary, open(outdir / "scan.json", "w"), indent=1)
    print(f"wrote {outdir/'scan.json'}", flush=True)


if __name__ == "__main__":
    mp.freeze_support()
    main()
