#!/usr/bin/env python3
"""Step 2: 30-min Gate Sweep on {537, 584, 635, 913, 585, 687, 740, 853, 854, 912} (and CLI targets)."""
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
    os.environ[_v] = "1"

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vstrip492_lib as vl  # noqa: E402

GATE_TARGETS = [585, 853, 912, 687, 740, 854, 913, 635, 584, 537]


def _select_candidate_pairs(m: int, r: int, f_inc: float, bricks: list[dict], max_fm_margin: float = 0.002) -> list[tuple]:
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
            if b1["w"] + b2["w"] > m + 1 or b1["h"] + b2["h"] > m + 1:
                continue
            if b1["w"] + b2["w"] > m and b1["h"] + b2["h"] > m:
                continue
            fm = max(b1["f"], b2["f"])
            if fm >= f_inc + max_fm_margin:
                continue
            ov = max(0, b1["w"] + b2["w"] - m) + max(0, b1["h"] + b2["h"] - m)
            ct = 1 if (b1["w"] + b2["w"] + b1["h"] + b2["h"] >= 2 * m and ov == 0) else 0
            sk = tuple(sorted([(b1["w"], b1["h"], b1["g"]), (b2["w"], b2["h"], b2["g"])]))
            cand = (ov, fm, ct, b1, b2)
            if sk not in best_by_shape or (ov, fm, ct) < best_by_shape[sk][:3]:
                best_by_shape[sk] = cand
    pairs = sorted(best_by_shape.values(), key=lambda t: (t[0], t[1], t[2]))
    return pairs


def _solve_target(args: tuple[int, str, list[dict], int]) -> dict:
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
    pairs = _select_candidate_pairs(m, r, f_inc, bricks)

    print(f"[n={n} m={m} r=+{r} inc={inc_f:.6f}]: {len(pairs)} candidate shape pairs", flush=True)
    best_n: dict | None = None
    for pidx, (ov, fm, ct, b1, b2) in enumerate(pairs[:max_pairs]):
        cur_f_inc = float(inc_dec) - m
        if best_n is not None and best_n["is_real_win"] and (best_n["dL"] <= -1e-3 or fm >= cur_f_inc + 0.001):
            break
        tag = f"n{n}_{b1['tag']}_{b2['tag']}"
        print(
            f"  [n={n} p{pidx}] ov={ov} ct={ct} fm={fm:.5f} (df={fm-cur_f_inc:+.5f}): "
            f"{b1['w']}x{b1['h']}(g={b1['g']},{b1['f']:.5f})+{b2['w']}x{b2['h']}(g={b2['g']},{b2['f']:.5f})",
            flush=True,
        )
        cert = vl.solve_pair(
            n, m, b1, b2, inc_dec, tag, base_seed=n * 100 + pidx,
            max_ov=1, do_melt=True, do_lpslack=True, do_mechwalk=True, n_threads=1,
        )
        if cert is not None:
            print(
                f"  [n={n} p{pidx}] -> {cert['label']}: L={cert['L_dec_str'][:18]} dL={cert['dL']:+.6e} win={cert['is_real_win']}",
                flush=True,
            )
            if cert["is_real_win"]:
                saved = vl.save_win_locked(cert)
                print(f"  *** SAVED WIN n={n}: saved={saved} dL={cert['dL']:+.6e} ***", flush=True)
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
        "dL": best_n["dL"] if best_n else None,
        "label": best_n["label"] if best_n else None,
    }


def main() -> None:
    t0 = time.time()
    args = [int(a) for a in sys.argv[1:] if not a.startswith("--")]
    targets = args or GATE_TARGETS
    max_pairs = 3

    bricks = vl.load_all_bricks(include_external=True, max_per_shape=2)
    print(f"Loaded {len(bricks)} brick shapes/orientations.", flush=True)
    frontier = vl.fast_load_true_frontier(targets)
    for n in targets:
        d = frontier[n]
        m = int(math.floor(d["L"]))
        print(f"  Frontier n={n:3d} (m={m}, r=+{n-m*m:2d}): {str(d['L_dec'])[:20]} ({d['source']})", flush=True)

    jobs = [(n, str(frontier[n]["L_dec"]), bricks, max_pairs) for n in targets]
    results = {}
    with ProcessPoolExecutor(max_workers=2) as pool:
        futs = {pool.submit(_solve_target, j): j[0] for j in jobs}
        for fut in as_completed(futs):
            res = fut.result()
            results[str(res["n"])] = res

    vl.DOCS_DIR.mkdir(parents=True, exist_ok=True)
    out_name = "gate_probe.json" if not args else "sweep_results.json"
    (vl.DOCS_DIR / out_name).write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    wins = [r for r in results.values() if r["win"]]
    print(f"\nDONE ({len(wins)}/{len(results)} wins) in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
