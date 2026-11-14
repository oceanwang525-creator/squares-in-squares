#!/usr/bin/env python3
"""30-min gate probe for H94 (band3) on {914, 975, 976} + harvested catalog scan."""
from __future__ import annotations

import json
import math
import os
import sys
import time
from decimal import Decimal
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import band3_lib as b3  # noqa: E402


def main() -> None:
    t0 = time.time()
    print("[1/3] Harvesting incumbent bricks from merged.csv...", flush=True)
    harvested = b3.harvest_incumbent_bricks(min_g=6, max_f=0.80, save=True)
    print(f"  Harvested {len(harvested)} unique (w,h,g) bricks.", flush=True)

    bricks = b3.load_all_bricks(include_external=True)
    print(f"[2/3] Loaded {len(bricks)} total brick shapes/orientations.", flush=True)

    targets = [914, 975, 976]
    frontier = b3.gl.fast_load_true_frontier(targets, certify=True)
    results = {}

    for n in targets:
        inc_dec = frontier[n]["L_dec"]
        inc_f = float(inc_dec)
        m = int(math.floor(inc_f))
        r = n - m * m
        f_inc = inc_f - m

        pairs = []
        seen = set()
        for i, b1 in enumerate(bricks):
            for j, b2 in enumerate(bricks):
                if i > j:
                    continue
                if b1["g"] + b2["g"] != r:
                    continue
                if b1["w"] + b2["w"] > m + 1 or b1["h"] + b2["h"] > m + 1:
                    continue
                if b1["w"] + b2["w"] > m and b1["h"] + b2["h"] > m:
                    continue
                fm = max(b1["f"], b2["f"])
                if fm >= f_inc - 5e-4:
                    continue
                ov = max(0, b1["w"] + b2["w"] - m) + max(0, b1["h"] + b2["h"] - m)
                corner_touch = 1 if (b1["w"] + b2["w"] + b1["h"] + b2["h"] >= 2 * m and ov == 0) else 0
                pk = tuple(sorted([(b1["w"], b1["h"], b1["g"], b1["tag"]), (b2["w"], b2["h"], b2["g"], b2["tag"])]))
                if pk in seen:
                    continue
                seen.add(pk)
                pairs.append((ov, corner_touch, fm, b1, b2))

        pairs.sort(key=lambda t: (t[0], t[2], t[1]))
        print(f"\n[n={n} m={m} r=+{r} inc={inc_f:.6f} f_inc={f_inc:.6f}]: {len(pairs)} improving pairs", flush=True)
        for ov, ct, fm, b1, b2 in pairs[:6]:
            print(
                f"  cand ov={ov} ct={ct} fm={fm:.6f} (d={fm-f_inc:+.5f}): "
                f"{b1['tag']}({b1['w']}x{b1['h']},g={b1['g']},f={b1['f']:.5f}) + "
                f"{b2['tag']}({b2['w']}x{b2['h']},g={b2['g']},f={b2['f']:.5f})",
                flush=True,
            )

        best_n = None
        for pidx, (ov, ct, fm, b1, b2) in enumerate(pairs[:4]):
            if fm >= float(inc_dec) - m - 1e-6:
                continue
            tag = f"n{n}_{b1['tag']}_{b2['tag']}"
            cert = b3.solve_pair(n, m, b1, b2, inc_dec, tag, base_seed=n * 100 + pidx, max_ov=1, do_melt=True)
            if cert is not None:
                print(
                    f"  -> {cert['label']}: L={cert['L_dec_str'][:18]} dL={cert['dL']:+.6e} win={cert['is_real_win']}",
                    flush=True,
                )
                if cert["is_real_win"]:
                    saved = b3.save_win_locked(cert)
                    print(f"  *** SAVED WIN n={n}: saved={saved} ***", flush=True)
                    inc_dec = min(inc_dec, max(Decimal(cert["L_lit"]), Decimal(cert["L_pandas"])))
                if best_n is None or cert["L_float"] < best_n["L_float"]:
                    best_n = cert

        results[str(n)] = {
            "inc": str(frontier[n]["L_dec"]),
            "win": bool(best_n and best_n["is_real_win"]),
            "best_L": best_n["L_dec_str"] if best_n else None,
            "dL": best_n["dL"] if best_n else None,
            "label": best_n["label"] if best_n else None,
        }

    b3.DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (b3.DOCS_DIR / "gate_probe.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nGATE PROBE DONE in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
