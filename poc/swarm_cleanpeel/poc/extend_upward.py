#!/usr/bin/env python3
"""Extend H16 win n=401 (20.533029303) upward via 1x1 L-borders + joint_xytheta_slp to n=442, 485, 530, 577."""
from __future__ import annotations

import json
import math
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import cleanpeel_lib as cp  # noqa: E402
import union as auto_union  # noqa: E402


def extend_one_layer(x: np.ndarray, y: np.ndarray, th: np.ndarray, L: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Extend normalized [0, L]^2 packing by one 1x1 upright L-border on top & right."""
    m = int(math.floor(L))
    b = np.arange(0.5, m, 1.0)
    off = L + 0.5 + 1e-6
    cx = np.concatenate([x, np.full(m, off), b, [off]])
    cy = np.concatenate([y, b, np.full(m, off), [off]])
    ct = np.concatenate([th, np.zeros(2 * m + 1)])
    return cp.norm_pack(cx, cy, ct)


def _sametilt_577_dec() -> Decimal | None:
    try:
        txt = subprocess.check_output(
            ["git", "-C", str(cp.REPO), "show", "origin/swarm-sametilt:poc/swarm_sametilt/final.csv"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        g = auto_union.parse_text(txt)
        if 577 in g:
            return Decimal(auto_union.certify(577, g[577]))
    except Exception:
        pass
    return None


def main():
    d401 = json.loads((cp.DATA_DIR / "win_n401.json").read_text())
    rows = d401["rows"]
    cur_x, cur_y, cur_t, cur_L = cp.norm_pack(
        np.array([float(r["x"]) for r in rows]),
        np.array([float(r["y"]) for r in rows]),
        np.array([float(r["angle"]) for r in rows]),
    )

    targets = [442, 485, 530, 577]
    _, best_L, _ = cp.load_frontier(target_ns=set(targets))
    st_577 = _sametilt_577_dec()
    if st_577 is not None and st_577 < best_L[577]:
        best_L[577] = st_577

    # Also compare against our existing saved wins for 442, 485
    for wn in (442, 485):
        wp = cp.DATA_DIR / f"win_n{wn}.json"
        if wp.exists():
            wd = json.loads(wp.read_text())
            w_dec = Decimal(max(wd["L_lit"], wd["L_pd"]))
            if w_dec < best_L[wn]:
                best_L[wn] = w_dec

    results = []
    for n_t in targets:
        cur_x, cur_y, cur_t, cur_L = extend_one_layer(cur_x, cur_y, cur_t, cur_L)
        assert len(cur_x) == n_t, f"Expected {n_t}, got {len(cur_x)}"
        L_inc_dec = best_L[n_t]
        print(f"\n=== Extending upward to n={n_t} (raw_L={cur_L:.12f}, inc={float(L_inc_dec):.12f}) ===", flush=True)

        # First check raw exact cert
        raw_cert = cp.certify_pack_against_frontier(n_t, cur_x, cur_y, cur_t, L_inc_dec)
        best_cert = raw_cert if (raw_cert and raw_cert["is_real_win"]) else None
        best_label = f"ext_{n_t}_from401_raw"
        if raw_cert:
            print(f"  raw: L={raw_cert['L_float']:.12f} dL={raw_cert['dL']:+.6e} win={raw_cert['is_real_win']}", flush=True)

        # Now polish with joint_xytheta_slp + fixed_angle_lp_tighten
        pol_cert = cp.polish_candidate(n_t, cur_x, cur_y, cur_t, L_inc_dec)
        if pol_cert:
            print(f"  polish: L={pol_cert['L_float']:.12f} dL={pol_cert['dL']:+.6e} win={pol_cert['is_real_win']}", flush=True)
            if pol_cert["is_real_win"] and (best_cert is None or Decimal(pol_cert["L_lit"]) < Decimal(best_cert["L_lit"])):
                best_cert = pol_cert
                best_label = f"ext_{n_t}_from401_slp"

        if best_cert and best_cert["is_real_win"]:
            saved = cp.save_win_and_rebuild_final(n_t, best_cert, best_label)
            print(f"  *** SAVED n={n_t} (updated={saved}) L={best_cert['L_lit'][:28]} dL={best_cert['dL']:+.6e} ***", flush=True)
            brows = best_cert["rows"]
            cur_x, cur_y, cur_t, cur_L = cp.norm_pack(
                np.array([float(r["x"]) for r in brows]),
                np.array([float(r["y"]) for r in brows]),
                np.array([float(r["angle"]) for r in brows]),
            )
            results.append({"n": n_t, "L": best_cert["L_lit"], "dL": best_cert["dL"], "label": best_label, "win": True})
        else:
            results.append({"n": n_t, "win": False})

    (cp.DOCS_DIR / "upward_extend_results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
