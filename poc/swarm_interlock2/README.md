# Hypothesis H101 — 2-D Overhang (`kx + ky <= 2, |I| = 0`) Pinwheel Composer (`swarm_interlock2`)

## Summary

Hypothesis **H101** composes two `(w_i, h_i, g_i)` sub-pockets (including 90-degree transposes) placed in diagonally opposite corners (`BL` and `TR`) with **1-axis or 2-axis overhang** (`kx = max(0, w_1 + w_2 - K)`, `ky = max(0, h_1 + h_2 - K)`, `kx + ky in {1, 2}`) and **zero bounding-box intersection** (`|I| = kx * ky == 0`), tiled with upright squares on the remaining `(K - w_1) x (K - h_2)` and `(K - w_2) x (K - h_1)` blocks:

- **Regime 1 (`zero_gap`)**: `D = g_1 + g_2 - (kx + ky)` with `1` unit of non-overhanging axis slack (`h_1 + h_2 <= K - 1` when `kx > 0`, or `w_1 + w_2 <= K - 1` when `ky > 0`). Exact 50-digit donor extraction + upright tiling (`assemble_zero_gap_50d`) + `lpslack` + soft-mode `mechwalk_compress` polish.
- **Regime 2 (`seam_touch`)**: `D = g_1 + g_2` with exact seam contact on the non-overhanging axis (`h_1 + h_2 == K` when `kx > 0`, or `w_1 + w_2 == K` when `ky > 0`). Assembled with C++ masked penalty melt (`run_cpp_masked_melt`) + `brick_slp` + `lpslack`.

## Sources & Data

- `poc/pocket_lift_scan3/data/pockets_pareto_e3_sane.json` (1,516 sanitized proper sub-pockets)
- Bricks from `poc/swarm_gstep/data/brick_*.json`, `poc/swarm_b7xplant/data/brick_*.json`, `poc/swarm_thinb4/data/brick_*.json`
- Scanned sub-pockets from `n in {492, 852, 911, 974, 975}` (`poc/swarm_interlock2/poc/scan_5rows.py`, donors cached in `data/donors_5rows.json`; `data/scanned_5rows.json` excluded from PR to keep all committed files under 1 MB).
- Combined index: `7,730` unique `(w, h, g)` keys (`12,797` items).

## Gate Results (`poc/swarm_interlock2/poc/run_gate.py`, 8433.6s)

### Certified Wins (`literal == pandas` via `poc/auto_union/poc/union.py`)

| `n` | `(K, D)` | `(kx, ky)` | Regime | Formula | `L_inc` | `L_lit` | `dL` |
|---:|:---:|:---:|:---|:---|---:|---:|---:|
| `748` | `(27, 19)` | `(1, 0)` | `zero_gap` | `e3_n804_7x18_g9 + e3_n748_21x8_g11` (`s00_mw`) | `27.8271519546314247` | `27.8268092691779018` | **`-3.426855e-04`** |
| `910` | `(30, 10)` | `(1, 0)` | `zero_gap` | `thin_B5_10x19_n793_10x19_g5 + brick_B6_10x21_n972_b8_21x10_g6` (`s00_mw`) | `30.6564209740910615` | `30.6564056705494223` | **`-1.530354e-05`** |

### Non-Winning Gate Evaluations

- **Regime 1 (`zero_gap`)**: `849` (`+1.9605e-4`), `791` (`+1.2340e-3`), `640` (`+1.4342e-3`), `454` (`+2.4192e-3`), `795` (`+2.4785e-3`), `912` (`+2.6225e-3`).
- **Regime 2 (`seam_touch`)**: All 6 tested gate targets stalled above `L_inc` (`n=738`: `+1.8242e-2`, `n=852`: `+3.3635e-2`, `n=414`: `+3.3844e-2`, `n=973`: `+3.4652e-2`, `n=336`: `+6.5565e-2`, `n=579`: `+1.5166e-1`), showing that exact-touch (`h_1 + h_2 = K`) overhang seams lack the 1-unit corridor slack required to relax `L` below `L_inc`.

## Checks

- Unit tests: `/tmp/highrpeel_venv/bin/pytest -v poc/swarm_interlock2/tests/test_interlock2.py`
- CSV validation: `.github/scripts/check_final_csv.py poc/swarm_interlock2/final.csv`
