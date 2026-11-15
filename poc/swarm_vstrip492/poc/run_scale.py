#!/usr/bin/env python3
"""H106 (vstrip492) Scale-Up Sweep across K = 20..31, D = 7..16."""
from __future__ import annotations

import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from decimal import Decimal
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import vstrip492_lib as vl  # noqa: E402


def _select_zero_ct0_pairs(
    m: int, r: int, f_inc: float, bricks: list[dict], max_fm_margin: float = 0.0018
) -> list[tuple]:
    """Select exact zero-overlap (ov=0, ct=0) pinwheel brick pairs sorted by max(f1, f2)."""
    best_by_shape: dict[tuple, tuple] = {}
    for i, b1 in enumerate(bricks):
        if b1["g"] >= r or b1["w"] >= m or b1["h"] >= m:
            continue
        for j in range(i, len(bricks)):
            b2 = bricks[j]
            if b1["g"] + b2["g"] != r:
                continue
            if b2["w"] >= m or b2["h"] >= m:
                continue
            if b1["w"] + b2["w"] > m or b1["h"] + b2["h"] > m:
                continue
            if b1["w"] + b2["w"] + b1["h"] + b2["h"] > 2 * m - 1:
                continue
            fm = max(b1["f"], b2["f"])
            if fm >= f_inc + max_fm_margin:
                continue
            sk = tuple(sorted([(b1["w"], b1["h"], b1["g"]), (b2["w"], b2["h"], b2["g"])]))
            cand = (fm, b1, b2)
            if sk not in best_by_shape or fm < best_by_shape[sk][0]:
                best_by_shape[sk] = cand
    return sorted(best_by_shape.values(), key=lambda t: t[0])


def _solve_zero_target(args: tuple[int, str, list[dict], int]) -> dict:
    n, inc_str, bricks, max_pairs = args
    inc_dec = Decimal(inc_str)
    wn = vl.DATA_DIR / f"win_n{n}.json"
    if wn.exists():
        wd = json.loads(wn.read_text(encoding="utf-8"))
        inc_dec = min(inc_dec, max(Decimal(wd["L_lit"]), Decimal(wd["L_pandas"])))

    inc_f = float(inc_dec)
    m = int(math.floor(inc_f))
    r = n - m * m
    f_inc = inc_f - m
    pairs = _select_zero_ct0_pairs(m, r, f_inc, bricks)

    best_n: dict | None = None
    for pidx, (fm, b1, b2) in enumerate(pairs[:max_pairs]):
        cur_f_inc = float(inc_dec) - m
        if best_n is not None and best_n["is_real_win"] and fm >= cur_f_inc + 0.0005:
            break
        tag = f"n{n}_{b1['tag']}_{b2['tag']}"
        # Run direct zero-overlap SLP + lpslack + targeted mechwalk (no slow melt needed for ov=0,ct=0)
        cert = vl.solve_pair(
            n,
            m,
            b1,
            b2,
            inc_dec,
            tag,
            base_seed=n * 100 + pidx,
            max_ov=0,
            do_melt=False,
            do_lpslack=True,
            do_mechwalk=True,
            n_threads=1,
        )
        if cert is not None:
            if cert["is_real_win"]:
                saved = vl.save_win_locked(cert)
                print(
                    f"  *** WIN n={n:3d} ({m},{r:2d}) p{pidx}: {b1['w']}x{b1['h']}+{b2['w']}x{b2['h']} "
                    f"L={cert['L_dec_str'][:18]} dL={cert['dL']:+.6e} saved={saved} ***",
                    flush=True,
                )
                inc_dec = min(inc_dec, max(Decimal(cert["L_lit"]), Decimal(cert["L_pandas"])))
            if best_n is None or cert["L_float"] < best_n["L_float"]:
                best_n = cert

    return {
        "n": n,
        "m": m,
        "r": r,
        "inc": inc_str,
        "n_cands": len(pairs),
        "win": bool(best_n and best_n["is_real_win"]),
        "best_L": best_n["L_dec_str"] if best_n else None,
        "best_dL": best_n["dL"] if best_n else None,
        "label": best_n["label"] if best_n else None,
    }


def main() -> None:
    t0 = time.time()
    force = "--force" in sys.argv
    max_pairs = 4

    bricks = vl.load_all_bricks(include_external=True, max_per_shape=2)
    targets = [m * m + r for m in range(20, 32) for r in range(7, 17) if m * m + r <= 1000]
    frontier = vl.fast_load_true_frontier(targets)

    ckpt_path = vl.DOCS_DIR / "scale_results.json"
    vl.DOCS_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}
    if ckpt_path.exists() and not force:
        try:
            results = json.loads(ckpt_path.read_text(encoding="utf-8"))
        except Exception:
            results = {}

    ranked_jobs: list[tuple[float, int]] = []
    for n in targets:
        if str(n) in results:
            continue
        d = frontier[n]
        m = int(math.floor(d["L"]))
        r = n - m * m
        f_inc = d["L"] - m
        pairs = _select_zero_ct0_pairs(m, r, f_inc, bricks)
        if pairs:
            ranked_jobs.append((pairs[0][0] - f_inc, n))

    ranked_jobs.sort()
    print(f"Loaded {len(bricks)} bricks. Phase 1 zero_ct0 queue: {len(ranked_jobs)} targets (done={len(results)}).", flush=True)
    for df, n in ranked_jobs[:20]:
        d = frontier[n]
        m = int(math.floor(d["L"]))
        print(f"  q n={n:3d} ({m},{n-m*m:2d}) best_df={df:+.5f} inc={str(d['L_dec'])[:16]} ({d['source']})", flush=True)

    jobs = [(n, str(frontier[n]["L_dec"]), bricks, max_pairs) for _, n in ranked_jobs]
    with ProcessPoolExecutor(max_workers=2) as pool:
        futs = {pool.submit(_solve_zero_target, j): j[0] for j in jobs}
        for fut in as_completed(futs):
            res = fut.result()
            results[str(res["n"])] = res
            ckpt_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
            print(
                f"[done {len(results)}] n={res['n']:3d} win={res['win']} "
                f"dL={res['best_dL']} ({res['n_cands']} cands)",
                flush=True,
            )

    wins = [r for r in results.values() if r["win"]]
    print(f"\nPHASE 1 COMPLETE ({len(wins)}/{len(results)} wins) in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
